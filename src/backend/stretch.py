import numpy as np
from backend.classify import DIFFUSE, COMPACT, UNCLASSIFIED
from dataclasses import dataclass


def asinh_transformation(x, stretch=1.0, blackpoint=0.0):
    """
    Apply an arcsinh stretch to the input pixel values.

    Args:
        x (np.ndarray): Input pixel values
        stretch (int): Stretch factor
        blackpoint (float): Threshold to treat as black 
    
    https://siril.readthedocs.io/en/latest/processing/stretching.html#asinh-transformation
    """
    x = np.asarray(x, dtype=np.float32)
    shifted = np.clip(x - blackpoint, 0.0, None)

    denom = np.arcsinh(stretch * max(1.0 - blackpoint, np.finfo(np.float32).eps))
    if denom == 0:
        return np.zeros_like(x, dtype=np.float32)

    return np.arcsinh(stretch * shifted) / denom



# def compute_background_mode(bg_pixels: np.ndarray, n_bins: int = 2000) -> float:
#     # Clean data
#     bg_pixels = np.asarray(bg_pixels, dtype=np.float32)
#     bg_pixels = bg_pixels[np.isfinite(bg_pixels)]
    
#     # Clip extreme 1% values
#     lo, hi = np.percentile(bg_pixels, [1.0, 99.0])
#     clipped = bg_pixels[(bg_pixels >= lo) & (bg_pixels <= hi)]

#     # Find mode
#     counts, edges = np.histogram(clipped, bins=n_bins)
#     mode_idx = int(np.argmax(counts))
#     return float((edges[mode_idx] + edges[mode_idx + 1]) / 2.0)


def group_pixels_by_object(id_map):
    """
    Group pixels by their object IDs

    Args:
        id_map: 1D array of object IDs for each pixel

    Returns:
        object_ids: Array of unique object IDs
        sorted_positions: Array of pixel indices sorted by object ID
        start_indices: Array of starting indices in sorted_positions for each object
        counts: Array of pixel counts for each object
    """
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
    saddle_raw: float
    base_level: float
    stretch: float
    blackpoint: float

    @property
    def is_diffuse(self):
        return self.label == DIFFUSE

    @property
    def is_compact(self):
        return self.label == COMPACT

    @property
    def is_unclassified(self):
        return self.label == UNCLASSIFIED

    def evaluate(self, pixels: np.ndarray) -> np.ndarray:
        result = np.full(len(pixels), self.base_level, dtype=np.float32)
        return result + asinh_transformation(pixels - self.saddle_raw, self.stretch, self.blackpoint)

    def evaluate_at(self, raw_value: float) -> float:
        return float(self.base_level + asinh_transformation(raw_value - self.saddle_raw, self.stretch, self.blackpoint))


class Stretch:
    def __init__(
        self,
        nodes,
        img_data,
        id_map_flat,
        class_map,
        result_lum,
        bg_stretch,
        diffuse_stretch,
        compact_stretch,
        blackpoint_background,
        blackpoint_compact,
        blackpoint_diffuse,
        bg_offset,
        # bg_mode=None,
        id_to_type_lut=None,
    ):
        self.nodes = nodes
        self.img_data = img_data
        self.id_map_flat = id_map_flat
        self.class_map = class_map
        self.result_lum = result_lum
        self.bg_stretch = bg_stretch
        self.diffuse_stretch = diffuse_stretch
        self.compact_stretch = compact_stretch
        self.blackpoint_background = blackpoint_background
        self.blackpoint_compact = blackpoint_compact
        self.blackpoint_diffuse = blackpoint_diffuse
        # self.bg_mode = bg_mode
        self.bg_offset = bg_offset
        self.id_to_type_lut = id_to_type_lut
        self._cache: dict[int, Object] = {}

    def get_label(self, obj_id: int) -> int:
        """
        Return the classification label for a given object ID
        """
        if self.id_to_type_lut is not None and 0 <= obj_id < len(self.id_to_type_lut):
            return int(self.id_to_type_lut[obj_id])
        return int(self.class_map.ravel()[obj_id])

    def get_stretch(self, label: int) -> float:
        """
        Return the stretch factor for a given classification label
        """
        if label == DIFFUSE:
            return self.diffuse_stretch
        if label == COMPACT:
            return self.compact_stretch
        return self.bg_stretch

    def get_blackpoint(self, label: int) -> float:
        """
        Return the blackpoint for a given classification label
        """
        if label == DIFFUSE:
            return self.blackpoint_diffuse
        if label == COMPACT:
            return self.blackpoint_compact
        return self.blackpoint_background

    def compute_object(self, obj_id: int) -> Object:
        """
        Create an Object instance for a given object ID, computing its
        properties based on its parent and classification label.
        """
        if obj_id in self._cache:
            return self._cache[obj_id]

        # Get parent details
        parent_idx = self.nodes[obj_id].parent
        saddle_raw = float(self.img_data[parent_idx])
        parent_obj_id = int(self.id_map_flat[parent_idx])

        # Get object details
        label = self.get_label(obj_id)
        stretch = self.get_stretch(label)

        # If unclassified but stacked on a parent, inherit parent's label and stretch
        if label == UNCLASSIFIED and parent_obj_id >= 0:
            parent_label = self.get_label(parent_obj_id)
            stretch = self.get_stretch(parent_label)
            label = parent_label
        
        # Process parent object
        if parent_obj_id < 0:
            # Parent is background.
            parent_base = float(self.result_lum[parent_idx])
            parent_label = None
        else:
            parent_obj = self.compute_object(parent_obj_id)
            
            parent_base = float(self.result_lum[parent_idx])
            parent_label = parent_obj.label
            
            # If objects of the same classification are stacked,
            # child should inherit saddle and base level of parent
            # to avoid double stretching discontinuities.
            if label == parent_label:
                saddle_raw = parent_obj.saddle_raw
                parent_base = parent_obj.base_level
        
        effective_stretch = stretch

        # Create Object instance
        obj = Object(
            id=obj_id,
            label=label,
            saddle_raw=saddle_raw,
            base_level=parent_base,
            stretch=effective_stretch,
            blackpoint=self.get_blackpoint(label),
        )
        
        self._cache[obj_id] = obj
        return obj



def apply_adaptive_stretch(
    image: np.ndarray,
    id_map: np.ndarray,
    class_map: np.ndarray,
    bg_stretch: float,
    diffuse_stretch: float,
    compact_stretch: float,
    blackpoint_background: float = 0.001,
    blackpoint_compact: float = 0.001,
    blackpoint_diffuse: float = 0.001,
    bg_offset_fraction: float = 0.5,
    compact_label: int = COMPACT,
    diffuse_label: int = DIFFUSE,
    mto_struct=None,
    id_to_type_lut=None,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    channels = image.shape[2] if image.ndim == 3 else 1

    nodes = mto_struct.mt.contents.nodes
    img_data = mto_struct.mt.contents.img.data
    id_map_flat = id_map.ravel()
    object_ids, sorted_positions, start_indices, counts = group_pixels_by_object(id_map_flat)

    depth_cache: dict[int, int] = {} 

    def get_object_depth(obj_id):
        """
        Recursively compute depth of object in the tree. 0 = background
        """
        if obj_id in depth_cache:
            return depth_cache[obj_id]
        parent_idx = nodes[obj_id].parent
        parent_obj = id_map_flat[parent_idx]
        depth = 0 if parent_obj < 0 else 1 + get_object_depth(int(parent_obj))
        depth_cache[obj_id] = depth
        return depth

    # Sort objects by depth such that parents are processed before children
    order = np.argsort([get_object_depth(int(oid)) for oid in object_ids])
    object_ids = object_ids[order]
    start_indices = start_indices[order]
    counts = counts[order]

    def stretch_scalar_plane(plane_flat: np.ndarray) -> np.ndarray:
        """
        Apply adaptive stretch to a single-channel, using object tree
        to deduce stretch parameters for each pixel.
        """
        bg_mask = id_map_flat < 0

        # bg_mode = compute_background_mode(plane_flat[bg_mask]) if bg_mask.any() else 0.0

        result_plane = np.empty(len(plane_flat), dtype=np.float32)

        # Background pixels
        result_plane[bg_mask] = asinh_transformation(
            plane_flat[bg_mask],
            bg_stretch,
            blackpoint_background,
        )

        # Create stretch instance
        stretch = Stretch(
            nodes=nodes,
            img_data=img_data,
            id_map_flat=id_map_flat,
            class_map=class_map,
            result_lum=result_plane,
            bg_stretch=bg_stretch,
            diffuse_stretch=diffuse_stretch,
            compact_stretch=compact_stretch,
            blackpoint_background=blackpoint_background,
            blackpoint_compact=blackpoint_compact,
            blackpoint_diffuse=blackpoint_diffuse,
            # bg_mode=bg_mode,
            bg_offset=0.0,
            id_to_type_lut=id_to_type_lut,
        )

        for obj_id, start_idx, count in zip(object_ids, start_indices, counts):
            obj = stretch.compute_object(int(obj_id))
            positions = sorted_positions[start_idx:start_idx+count]
            pixels = plane_flat[positions]
            result_plane[positions] = obj.evaluate(pixels)

        return result_plane

    image_flat = image.reshape(-1, channels) if channels > 1 else image.ravel()[:, np.newaxis]

    if channels > 1:
        # Coloured images
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        lum_flat = np.dot(image_flat, weights)

        stretched_lum_flat = stretch_scalar_plane(lum_flat)

        ratio = np.zeros_like(lum_flat)
        valid_mask = lum_flat > 1e-8
        ratio[valid_mask] = stretched_lum_flat[valid_mask] / lum_flat[valid_mask]

        result_flat = image_flat * ratio[:, np.newaxis]

        # max_vals = np.max(result_flat, axis=1)
        # clip_mask = max_vals > 1.0
        # if clip_mask.any():
        #     result_flat[clip_mask] = result_flat[clip_mask] / max_vals[clip_mask, np.newaxis]

    else:
        # Greyscale images
        stretched = stretch_scalar_plane(image_flat[:, 0])
        result_flat = stretched[:, np.newaxis]

    return np.clip(result_flat.reshape(image.shape), 0.0, 1.0)