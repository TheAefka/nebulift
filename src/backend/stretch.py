import numpy as np
import tqdm
from backend.classify import DIFFUSE, COMPACT


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
        return asinh_stretch(image[idx], bg_factor, black_point)
    
    parent_val, parent_obj, parent_idx = get_direct_parent(obj_id, id_map_flat, nodes, img_data)
    child_label = class_map.ravel()[obj_id]
    child_stretch_factor = bg_factor if child_label < 0 else (diff_factor if child_label == DIFFUSE else compact_factor)
    if parent_obj >= 0:
        # Stacked objects
        parent_value = image_flat[parent_idx]
        parent_value_stretched = process_pixel_value(image, parent_obj, id_map_flat, nodes, img_data, image_flat, class_map, idx, bg_factor, diff_factor, compact_factor, black_point)

        child_value = image[idx]
        child_contribution = child_value - parent_value
        child_contribution_stretched = asinh_stretch(child_contribution, child_stretch_factor, black_point)

        result = parent_value_stretched + child_contribution_stretched
    else:
        # Standalone object, parent is background
        bg_contribution = asinh_stretch(parent_val, bg_factor, black_point)
        
        child_value = image[idx]
        child_contribution = child_value - parent_val
        child_contribution_stretched = asinh_stretch(child_contribution, child_stretch_factor, black_point)
        
        result = bg_contribution + child_contribution_stretched

    return result



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

    result = np.zeros_like(image, dtype=np.float32)
    for idx in tqdm.tqdm(np.ndindex(image.shape), total=np.prod(image.shape), desc="Applying adaptive stretch"):
        obj_id = id_map[idx]
        result[idx] = process_pixel_value(image, obj_id, id_map_flat, nodes, img_data, image_flat, class_map, idx, bg_stretch_factor, diffuse_stretch_factor, compact_stretch_factor, black_point)
    return result