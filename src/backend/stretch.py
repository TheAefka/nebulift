import numpy as np
from backend.classify import DIFFUSE, COMPACT, UNCLASSIFIED
from dataclasses import dataclass


def asinh_stretch(x, stretch_factor=100.0, black_point=0.001):
    x = np.asarray(x, dtype=np.float32)
    shifted = np.clip(x - black_point, 0.0, None)

    denom = np.arcsinh(stretch_factor * max(1.0 - black_point, np.finfo(np.float32).eps))
    if denom == 0: # Division by zero
        return np.zeros_like(x, dtype=np.float32)

    return np.arcsinh(stretch_factor * shifted) / denom


def compute_background_mode(bg_pixels: np.ndarray, n_bins: int = 2000) -> float:
    # Clean data
    bg_pixels = np.asarray(bg_pixels, dtype=np.float32)
    bg_pixels = bg_pixels[np.isfinite(bg_pixels)]
    
    # Clip extreme 1% values
    lo, hi = np.percentile(bg_pixels, [1.0, 99.0])
    clipped = bg_pixels[(bg_pixels >= lo) & (bg_pixels <= hi)]

    # Find mode
    counts, edges = np.histogram(clipped, bins=n_bins)
    mode_idx = int(np.argmax(counts))
    return float((edges[mode_idx] + edges[mode_idx + 1]) / 2.0)


def group_pixels_by_object(id_map):
    valid_mask = id_map >= 0
    valid_pos = np.flatnonzero(valid_mask)
    valid_ids = id_map[valid_mask]

    sort_order = np.argsort(valid_ids)
    sorted_pos = valid_pos[sort_order]
    sorted_ids = valid_ids[sort_order]

    object_ids, start_indices, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
    return object_ids, sorted_pos, start_indices, counts



@dataclass
class Object:
    id: int
    label: int
    value: float
    root: "Object" = None
    parent: "Object" = None
    is_stacked: bool = False
    root_floor_value: float = 0.0
    stretch_factor: float = 0.0
    stretched_floor_value: float = 0.0

    @property
    def is_diffuse(self):
        return self.label == DIFFUSE
    
    @property
    def is_compact(self):
        return self.label == COMPACT

    @property
    def is_unclassified(self):
        return self.label == UNCLASSIFIED

    @property
    def is_background(self):
        return self.id < 0
    
    @property
    def is_object(self):
        return not self.is_background

class Stretch:
    def __init__(
        self,
        nodes,
        img_data,
        id_map_flat,
        class_map,
        bg_stretch_factor,
        diffuse_stretch_factor,
        compact_stretch_factor,
        black_point=0.001,
        id_to_type_lut=None
    ):
        self.nodes = nodes
        self.img_data = img_data
        self.id_map_flat = id_map_flat
        self.class_map = class_map
        self.bg_stretch_factor = bg_stretch_factor
        self.diffuse_stretch_factor = diffuse_stretch_factor
        self.compact_stretch_factor = compact_stretch_factor
        self.black_point = black_point
        self.id_to_type_lut = id_to_type_lut
        self.object_cache = {}

    
    def get_label(self, obj_id):
        if (self.id_to_type_lut is not None and 0 <= obj_id < len(self.id_to_type_lut)):
            return int(self.id_to_type_lut[obj_id])
        return int(self.class_map.ravel()[obj_id])


    def get_stretch_factor(self, label):
        if label == UNCLASSIFIED:
            return self.bg_stretch_factor
        elif label == DIFFUSE:
            return self.diffuse_stretch_factor
        elif label == COMPACT:
            return self.compact_stretch_factor
        else:
            return self.bg_stretch_factor  # Default to background stretch for unknown labels


    def build_object(self, obj_id):
        parent_idx = self.nodes[obj_id].parent
        parent_val = float(self.img_data[parent_idx])
        label = self.get_label(obj_id)
        return Object(
            id=obj_id,
            label=label,
            parent=None,
            value=parent_val,
            stretch_factor=self.get_stretch_factor(label)
        )


    def compute_object_info(self, object_id):
        if object_id in self.object_cache:
            return self.object_cache[object_id]

        parent_idx = self.nodes[object_id].parent
        parent_val = float(self.img_data[parent_idx])
        parent_obj_id = int(self.id_map_flat[parent_idx])

        label = self.get_label(object_id)

        object = Object(
            id=object_id,
            label=label,
            value=parent_val,
            stretch_factor=self.get_stretch_factor(label)
        )

        if parent_obj_id < 0:
            # Parent is background
            object.stretched_floor_value = asinh_stretch(parent_val, self.bg_stretch_factor, self.black_point)
        else:
            parent = self.compute_object_info(parent_obj_id)
            object.parent = parent
            object.is_stacked = parent.is_diffuse and object.is_diffuse

            if object.is_unclassified:
                object.stretch_factor = parent.stretch_factor
                if parent.is_diffuse:
                    object.root = parent.root if parent.is_stacked else parent
                    object.root_floor_value = object.root.value
                    object.stretched_floor_value = asinh_stretch(
                        parent_val - object.root_floor_value,
                        object.stretch_factor,
                        self.black_point,
                    )
                else:
                    object.stretched_floor_value = parent.stretched_floor_value
                    object.root_floor_value = parent.root_floor_value
            elif object.is_stacked:
                # Nested/stacked diffuse objects. Stretch relative to root parent value.
                object.root = parent.root if parent.is_stacked else parent
                object.root_floor_value = object.root.value
                object.stretched_floor_value = asinh_stretch(parent_val - object.root_floor_value, object.stretch_factor, self.black_point)
            else:
                # Other objects. Stretch relative to parent.
                object.stretched_floor_value = parent.stretched_floor_value + asinh_stretch(parent_val - parent.value, parent.stretch_factor, self.black_point)

        # Add to cache
        self.object_cache[object_id] = object

        return object



def apply_adaptive_stretch(
    image: np.ndarray,
    id_map: np.ndarray,
    class_map: np.ndarray,
    bg_stretch_factor: float,
    diffuse_stretch_factor: float,
    compact_stretch_factor: float,
    black_point: float = 0.001,
    compact_label: int = COMPACT,
    diffuse_label: int = DIFFUSE,
    mto_struct=None,
    id_to_type_lut=None,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    channels = image.shape[2] if image.ndim == 3 else 1
    image_flat = image.reshape(-1, channels) if channels > 1 else image.ravel()[:, np.newaxis]

    nodes       = mto_struct.mt.contents.nodes
    img_data    = mto_struct.mt.contents.img.data
    id_map_flat = id_map.ravel()
    bg_mask     = id_map_flat < 0

    result_flat = np.zeros_like(image_flat)
    
    result_flat[bg_mask] = asinh_stretch(image_flat[bg_mask], bg_stretch_factor, black_point)

    stretch = Stretch(
        nodes=nodes,
        img_data=img_data,
        id_map_flat=id_map_flat,
        class_map=class_map,
        bg_stretch_factor=bg_stretch_factor,
        diffuse_stretch_factor=diffuse_stretch_factor,
        compact_stretch_factor=compact_stretch_factor,
        black_point=black_point,
        id_to_type_lut=id_to_type_lut
    )

    object_ids, sorted_positions, start_indices, counts = group_pixels_by_object(id_map_flat)


    def get_object_depth(obj_id, depth_cache):
        if obj_id in depth_cache:
            return depth_cache[obj_id]
        
        parent_idx = nodes[obj_id].parent
        parent_obj = id_map_flat[parent_idx]

        if parent_obj < 0:
            depth = 0
        else:
            depth = 1 + get_object_depth(parent_obj, depth_cache)

        depth_cache[obj_id] = depth
        return depth

    depth_cache = {}
    ordered_object_ids = np.argsort(
        [get_object_depth(obj_id, depth_cache) for obj_id in object_ids]
    )
    object_ids = object_ids[ordered_object_ids]
    start_indices = start_indices[ordered_object_ids]
    counts = counts[ordered_object_ids]


    for obj_id, start_idx, count in zip(object_ids, start_indices, counts):
        obj = stretch.compute_object_info(obj_id)
        positions = sorted_positions[start_idx : start_idx + count]
        pixels = image_flat[positions]

        parent_idx = nodes[obj_id].parent
        parent_obj_id = int(id_map_flat[parent_idx])
        parent_obj = stretch.compute_object_info(parent_obj_id) if parent_obj_id >= 0 else None
        floor_value = result_flat[parent_idx]

        if obj.is_unclassified:
            if parent_obj is not None and parent_obj.is_diffuse:
                root = parent_obj.root if parent_obj.is_stacked else parent_obj
                result_flat[positions] = asinh_stretch(
                    pixels - root.value,
                    obj.stretch_factor,
                    black_point,
                )
            elif parent_obj is not None and parent_obj.is_unclassified and parent_obj.root is not None:
                result_flat[positions] = asinh_stretch(
                    pixels - parent_obj.root.value,
                    obj.stretch_factor,
                    black_point,
                )
            else:
                result_flat[positions] = floor_value + asinh_stretch(
                    pixels - obj.value,
                    obj.stretch_factor,
                black_point,
                )
        elif obj.is_compact and parent_obj is not None and parent_obj.is_diffuse:
            result_flat[positions] = floor_value + asinh_stretch(
                pixels - parent_obj.value,
                obj.stretch_factor,
                black_point,
            )
        elif obj.is_stacked:
            result_flat[positions] = asinh_stretch(
                pixels - obj.root_floor_value,
                obj.stretch_factor,
                black_point,
            )
        elif obj.stretch_factor == 0:
            result_flat[positions] = floor_value
        else:
            result_flat[positions] = floor_value + asinh_stretch(
                pixels - obj.value,
                obj.stretch_factor,
                black_point,
            )

    if channels > 1:
        ratio = result_flat / (image_flat + 1e-6)
        stretched = image_flat * ratio
        return stretched.reshape(image.shape)

    return result_flat.reshape(image.shape)