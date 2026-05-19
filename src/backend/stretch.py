import numpy as np
import tqdm
from backend.classify import DIFFUSE, COMPACT, UNCLASSIFIED


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

def get_direct_parent(obj_id: int, id_map_flat: np.ndarray, nodes, img_data) -> tuple:
    parent_idx = nodes[obj_id].parent
    parent_val = img_data[parent_idx]
    parent_obj = id_map_flat[parent_idx]   # -1 == background
    return parent_val, parent_obj, parent_idx


def process_pixel_value(image, obj_id, id_map_flat, nodes, img_data, image_flat, class_map, idx, bg_factor, diff_factor, compact_factor, black_point=0.001):
    # Background pixel
    if obj_id < 0:
        return asinh_stretch(image_flat[idx], bg_factor, black_point)
    
    parent_val, parent_obj, parent_idx = get_direct_parent(obj_id, id_map_flat, nodes, img_data)
    child_label = class_map.ravel()[obj_id]
    child_stretch_factor = bg_factor if child_label == UNCLASSIFIED else (diff_factor if child_label == DIFFUSE else compact_factor)

    if parent_obj >= 0:
        # Stacked objects
        parent_value_stretched = process_pixel_value(image, parent_obj, id_map_flat, nodes, img_data, image_flat, class_map, parent_idx, bg_factor, diff_factor, compact_factor, black_point)
    else:
        # Standalone object, parent is background
        parent_value_stretched = asinh_stretch(parent_val, bg_factor, black_point)

    child_contribution = image_flat[idx] - parent_val
    child_contribution_stretched = asinh_stretch(child_contribution, child_stretch_factor, black_point)

    return parent_value_stretched + child_contribution_stretched



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

    nodes       = mto_struct.mt.contents.nodes
    img_data    = mto_struct.mt.contents.img.data
    id_map_flat = id_map.ravel()
    image_flat  = image.ravel()
    bg_mask     = id_map < 0

    result = np.zeros_like(image, dtype=np.float32)
    result[bg_mask] = asinh_stretch(image[bg_mask], bg_stretch_factor, black_point)

    object_indices = np.argwhere(~bg_mask)
    for idx in tqdm.tqdm(object_indices, total=len(object_indices), desc="Applying adaptive stretch"):
        idx = tuple(idx)
        obj_id = id_map[idx]
        idx_flat = np.ravel_multi_index(idx, id_map.shape)
        result[idx] = process_pixel_value(image, obj_id, id_map_flat, nodes, img_data, image_flat, class_map, idx_flat, bg_stretch_factor, diffuse_stretch_factor, compact_stretch_factor, black_point)
    return result