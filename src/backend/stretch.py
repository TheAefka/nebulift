import numpy as np


def asinh_stretch(x, stretch_factor=100.0, black_point=0.001):
    x = np.asarray(x, dtype=np.float32)
    shifted = np.clip(x - black_point, 0.0, None)

    denom = np.arcsinh(stretch_factor * max(1.0 - black_point, np.finfo(np.float32).eps))
    if denom == 0: # Division by zero
        return np.zeros_like(x, dtype=np.float32)

    return np.arcsinh(stretch_factor * shifted) / denom


def apply_adaptive_stretch(image: np.ndarray, 
                           label_map: np.ndarray,
                           compact_stretch_fn,
                           diffuse_stretch_fn,
                           compact_label: int = 1,
                           diffuse_label: int = 0,
                           black_point: float = 0.001
                           ) -> np.ndarray:
    # Start with a background stretch everywhere.
    result = asinh_stretch(image, black_point=black_point).astype(np.float32, copy=False)

    # Apply diffuse stretch on diffuse-class pixels.
    diffuse_mask = label_map == diffuse_label
    result[diffuse_mask] = diffuse_stretch_fn(image[diffuse_mask])

    # Apply compact stretch on top of the already stretched parent values.
    compact_mask = label_map == compact_label
    result[compact_mask] = compact_stretch_fn(result[compact_mask])

    return np.clip(result, 0, 1)