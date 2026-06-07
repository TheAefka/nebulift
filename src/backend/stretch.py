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
    saddle_raw: float
    base_level: float
    stretch_factor: float
    black_point: float

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
        return result + asinh_stretch(pixels - self.saddle_raw, self.stretch_factor, self.black_point)

    def evaluate_at(self, raw_value: float) -> float:
        return float(self.base_level + asinh_stretch(raw_value - self.saddle_raw, self.stretch_factor, self.black_point))


class Stretch:
    def __init__(
        self,
        nodes,
        img_data,
        id_map_flat,
        class_map,
        result_lum,
        bg_stretch_factor,
        diffuse_stretch_factor,
        compact_stretch_factor,
        black_point,
        bg_mode,
        bg_offset,
        id_to_type_lut=None,
    ):
        self.nodes = nodes
        self.img_data = img_data
        self.id_map_flat = id_map_flat
        self.class_map = class_map
        self.result_lum = result_lum
        self.bg_stretch_factor = bg_stretch_factor
        self.diffuse_stretch_factor = diffuse_stretch_factor
        self.compact_stretch_factor = compact_stretch_factor
        self.black_point = black_point
        self.bg_mode = bg_mode
        self.bg_offset = bg_offset
        self.id_to_type_lut = id_to_type_lut
        self._cache: dict[int, Object] = {}

    def get_label(self, obj_id: int) -> int:
        if self.id_to_type_lut is not None and 0 <= obj_id < len(self.id_to_type_lut):
            return int(self.id_to_type_lut[obj_id])
        return int(self.class_map.ravel()[obj_id])

    def get_stretch_factor(self, label: int) -> float:
        if label == DIFFUSE:
            return self.diffuse_stretch_factor
        if label == COMPACT:
            return self.compact_stretch_factor
        return self.bg_stretch_factor

    def compute_object(self, obj_id: int) -> Object:
        if obj_id in self._cache:
            return self._cache[obj_id]

        # Get parent details
        parent_idx = self.nodes[obj_id].parent
        saddle_raw = float(self.img_data[parent_idx])
        parent_obj_id = int(self.id_map_flat[parent_idx])

        label = self.get_label(obj_id)
        sf = self.get_stretch_factor(label)

        
        # Process parent object
        if parent_obj_id < 0:
            # Parent is background.
            parent_base = float(self.result_lum[parent_idx])
            parent_label = None
        else:
            parent_obj = self.compute_object(parent_obj_id)
            parent_base = float(self.result_lum[parent_idx])
            parent_label = parent_obj.label
            if label == parent_label:
                saddle_raw = parent_obj.saddle_raw
                parent_base = parent_obj.base_level
        effective_sf = sf

        obj = Object(
            id=obj_id,
            label=label,
            saddle_raw=saddle_raw,
            base_level=parent_base,
            stretch_factor=effective_sf,
            black_point=self.black_point,
        )
        self._cache[obj_id] = obj
        return obj


def apply_adaptive_stretch(
    image: np.ndarray,
    id_map: np.ndarray,
    class_map: np.ndarray,
    bg_stretch_factor: float,
    diffuse_stretch_factor: float,
    compact_stretch_factor: float,
    black_point: float = 0.001,
    bg_offset_fraction: float = 0.5,
    compact_label: int = COMPACT,
    diffuse_label: int = DIFFUSE,
    mto_struct=None,
    id_to_type_lut=None,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    channels = image.shape[2] if image.ndim == 3 else 1
    image_flat = image.reshape(-1, channels) if channels > 1 else image.ravel()[:, np.newaxis]

    nodes = mto_struct.mt.contents.nodes
    img_data = mto_struct.mt.contents.img.data
    id_map_flat = id_map.ravel()
    object_ids, sorted_positions, start_indices, counts = group_pixels_by_object(id_map_flat)

    depth_cache: dict[int, int] = {} # dict[obj_id, depth]

    def get_object_depth(obj_id):
        if obj_id in depth_cache:
            return depth_cache[obj_id]
        parent_idx = nodes[obj_id].parent
        parent_obj = id_map_flat[parent_idx]
        depth = 0 if parent_obj < 0 else 1 + get_object_depth(int(parent_obj))
        depth_cache[obj_id] = depth
        return depth

    order = np.argsort([get_object_depth(int(oid)) for oid in object_ids])
    object_ids = object_ids[order]
    start_indices = start_indices[order]
    counts = counts[order]

    def stretch_scalar_plane(plane_flat: np.ndarray) -> np.ndarray:
        bg_mask = id_map_flat < 0

        # Background: offset + asinh with bp=0 to keep noise texture intact.
        bg_mode = compute_background_mode(plane_flat[bg_mask]) if bg_mask.any() else 0.0
        bg_offset = bg_mode * bg_offset_fraction

        result_plane = np.empty(len(plane_flat), dtype=np.float32)
        result_plane[bg_mask] = bg_offset + asinh_stretch(
            plane_flat[bg_mask] - bg_mode, bg_stretch_factor, black_point=0.0
        )

        stretch = Stretch(
            nodes=nodes,
            img_data=img_data,
            id_map_flat=id_map_flat,
            class_map=class_map,
            result_lum=result_plane,
            bg_stretch_factor=bg_stretch_factor,
            diffuse_stretch_factor=diffuse_stretch_factor,
            compact_stretch_factor=compact_stretch_factor,
            black_point=black_point,
            bg_mode=bg_mode,
            bg_offset=bg_offset,
            id_to_type_lut=id_to_type_lut,
        )

        for obj_id, start_idx, count in zip(object_ids, start_indices, counts):
            obj = stretch.compute_object(int(obj_id))
            positions = sorted_positions[start_idx: start_idx + count]
            pixels = plane_flat[positions]
            result_plane[positions] = obj.evaluate(pixels)

        return result_plane

    if channels > 1:
        result_channels = []
        for channel_index in range(channels):
            result_channels.append(stretch_scalar_plane(image_flat[:, channel_index]))
        result_flat = np.stack(result_channels, axis=1)
    else:
        result_flat = stretch_scalar_plane(image_flat[:, 0])[:, np.newaxis]

    return np.clip(result_flat.reshape(image.shape), 0.0, 1.0)