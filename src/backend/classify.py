import numpy as np
import pandas as pd

DIFFUSE = 0
COMPACT = 1
UNCLASSIFIED = -1

def classify_objects(parameters_df: pd.DataFrame,
                     r_fwhm_threshold: float = None,
                     a_b_threshold: float = None) -> pd.DataFrame:
    # Classify objects based on R_fwhm (full width at half maximum)
    r = parameters_df['R_fwhm']
    if r_fwhm_threshold is None:
        r_fwhm_threshold = np.nanpercentile(r, 99)
    # Classify objects based on a_b (axis ratio)
    if a_b_threshold is not None:
        a_b = parameters_df['A/B']
        compact_mask = (r < r_fwhm_threshold) & (a_b <= a_b_threshold)
        diffuse_mask = (r >= r_fwhm_threshold) & (a_b > a_b_threshold)
        labels = np.full(r.shape, UNCLASSIFIED, dtype=np.int8)
        labels[compact_mask] = COMPACT
        labels[diffuse_mask] = DIFFUSE
    else:
        labels = np.where(r < r_fwhm_threshold, COMPACT, DIFFUSE)
        labels = np.where(r.isna(), UNCLASSIFIED, labels)

    parameters_df['source_type'] = labels
    return parameters_df
