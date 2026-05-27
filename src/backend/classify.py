import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

DIFFUSE = 0
COMPACT = 1
UNCLASSIFIED = -1

EPS = 1e-12
FEATURES = ["R_fwhm", "A/B", "R_e"]


def _safe_log1p(values: pd.Series) -> pd.Series:
    return np.log1p(np.clip(values.astype(float), a_min=0.0, a_max=None))


def _roundness_penalty(values: pd.Series) -> pd.Series:
    return np.abs(np.log(np.clip(values.astype(float), a_min=EPS, a_max=None)))


def classify_objects(
    parameters_df: pd.DataFrame,
    r_fwhm_threshold: float = None,
    a_b_threshold: float = None,
    min_confidence: float = 0.7,
) -> pd.DataFrame:
    df = parameters_df.copy()
    df["source_type"] = UNCLASSIFIED

    transformed = pd.DataFrame(index=df.index)

    transformed["log_R_fwhm"] = _safe_log1p(df["R_fwhm"])
    transformed["log_R_e"] = _safe_log1p(df["R_e"])
    transformed["roundness"] = _roundness_penalty(df["A/B"])

    feats = transformed.replace([np.inf, -np.inf], np.nan)
    valid_rows = feats.dropna()

    # Scale features for GMM
    X = StandardScaler().fit_transform(valid_rows.values)

    # Fit GMM to classify into 2 clusters
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=10,
        random_state=0,
    )
    labels = gmm.fit_predict(X)
    probas = gmm.predict_proba(X)
    max_probas = probas.max(axis=1) # Best confidence score for each object

    # Assuming compact objects are smaller and rounder, determine which cluster
    # is compact based on mean feature values
    compact_scores = []
    for cluster_id in range(2):
        cluster_mask = labels == cluster_id
        cluster_values = valid_rows.loc[cluster_mask]

        compact_score = 0.0
        for column in cluster_values.columns:
            mean_value = cluster_values[column].mean()
            if column in ["log_R_fwhm", "log_R_e", "roundness"]:
                compact_score -= mean_value

        compact_scores.append(compact_score)

    compact_cluster = int(np.argmax(compact_scores))

    # If classification of an object above confidence threshold, assign compact
    # or diffuse, otherwise leave as unclassified
    classified = np.where(
        max_probas >= min_confidence,
        np.where(labels == compact_cluster, COMPACT, DIFFUSE),
        UNCLASSIFIED,
    )

    df.loc[valid_rows.index, "source_type"] = classified
    return df