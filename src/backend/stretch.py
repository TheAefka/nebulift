import numpy as np
from scipy.ndimage import minimum as _nd_minimum


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



def compute_object_floor(image: np.ndarray, id_map: np.ndarray) -> np.ndarray:
    image_f = np.asarray(image, dtype=np.float32)
    valid_mask = id_map >= 0

    unique_ids = np.unique(id_map[valid_mask]) # Unique IDs list

    # Shift background (id=-1) to _nd_minimum ignored value (0)
    shifted_labels = (id_map + 1).astype(np.int32)
    # Find the min pixel val of each object
    obj_mins = _nd_minimum(image_f, labels=shifted_labels, index=(unique_ids + 1))

    # Lookup table: find min pixel per object
    max_id = int(unique_ids.max())
    lut = np.zeros(max_id + 2, dtype=np.float32)
    lut[unique_ids] = obj_mins

    # Build image where each object's pixels is replaced by its floor value
    safe_ids = np.where(valid_mask, id_map, 0).astype(np.int32)
    floors = np.where(valid_mask, lut[safe_ids], image_f)
    return floors.astype(np.float32)




def apply_adaptive_stretch(image: np.ndarray,
                        id_map: np.ndarray,
                        class_map: np.ndarray,
                        bg_stretch_factor: float,
                        diffuse_stretch_factor: float,
                        compact_stretch_factor: float,
                        black_point: float = 0.001,
                        compact_label: int = 1,
                        diffuse_label: int = 0,
                        ) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)

    # Calculate reference levels
    bg_pixels = image[id_map < 0]
    bg_level = compute_background_mode(bg_pixels)

    img_max = float(np.nanmax(image))
    img_range = max(img_max - bg_level, float(np.finfo(np.float32).eps))

    def _norm(v):
        return np.clip((np.asarray(v, dtype=np.float32) - bg_level) / img_range, 0.0, None)
    
    floor_map_raw = compute_object_floor(image, id_map)

    result = np.empty_like(image, dtype=np.float32)

    # Background stretch
    bg_mask = id_map < 0
    result[bg_mask] = asinh_stretch(_norm(image[bg_mask]), bg_stretch_factor, black_point)

    # Unclassified objects stretch
    unclassified_mask = (id_map >= 0) & ~((class_map == diffuse_label) | (class_map == compact_label))
    result[unclassified_mask] = asinh_stretch(_norm(image[unclassified_mask]), bg_stretch_factor, black_point)

    # Diffuse stretch
    diffuse_mask = class_map == diffuse_label
    if diffuse_mask.any():
        d_floor_norm = _norm(floor_map_raw[diffuse_mask])
        d_delta_norm = np.maximum(_norm(image[diffuse_mask]) - d_floor_norm, 0.0)
        result[diffuse_mask] = (
            asinh_stretch(d_floor_norm, bg_stretch_factor, black_point)
            + asinh_stretch(d_delta_norm, diffuse_stretch_factor, 0)
        )
    
    # Compact stretch
    compact_mask = class_map == compact_label
    if compact_mask.any():
        n_compact = int(compact_mask.sum())
        c_floor_norm = _norm(floor_map_raw[compact_mask])
        c_delta_norm = np.maximum(_norm(image[compact_mask]) - c_floor_norm, 0.0)
        result[compact_mask] = (
            asinh_stretch(np.zeros(n_compact, dtype=np.float32), bg_stretch_factor, black_point)
            + asinh_stretch(c_floor_norm, diffuse_stretch_factor, 0)
            + asinh_stretch(c_delta_norm, compact_stretch_factor, 0)
        ) # TODO: (issue) this still creates halo/border due to (?) flat flooring

    return np.clip(result, 0.0, 1.0)