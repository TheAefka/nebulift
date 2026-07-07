import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

DIFFUSE = 0
COMPACT = 1
NOISE   = 2
UNCLASSIFIED = -1

EPS = 1e-12
FEATURES = ["R_fwhm", "A/B", "R_e"]

def label_to_name(label: int) -> str:
    return ["Diffuse", "Compact", "Noise", "Background"][label]

def _safe_log1p(values: pd.Series) -> pd.Series:
    return np.log1p(np.clip(values.astype(float), a_min=0.0, a_max=None))

def _roundness_penalty(values: pd.Series) -> pd.Series:
    return np.abs(np.log(np.clip(values.astype(float), a_min=EPS, a_max=None)))



def classify_objects_thresholds(
    parameters_df: pd.DataFrame,
    thresholds: dict[str, float] = None,
) -> pd.DataFrame:
    """
    Simple per-feature threshold classifier.
    """
    df = parameters_df.copy()
    df['source_type'] = DIFFUSE  # default everything to diffuse

    mask_compact = pd.Series(True, index=df.index)

    for key, value in thresholds.items():
        if key in df.columns and value is not None:
            mask_compact &= df[key] <= value

    df.loc[mask_compact, 'source_type'] = COMPACT
    return df



def classify_objects_gmm1(
    parameters_df: pd.DataFrame,
    min_confidence: float = 0.7,
) -> pd.DataFrame:
    """
    GMM classifier using log(R_fwhm), log(R_e), and a roundness penalty
    based on A/B.
    This is under the assumption that compact objects are smaller (lower
    R_fwhm and R_e) and rounder (A/B closer to 1).

    A compact score is calculated for each cluster based on the average of these features,
    and the cluster with the higher compact score is assigned as COMPACT.
    """
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

def classify_objects_gmm2(
    parameters_df: pd.DataFrame,
    min_confidence: float = 0.7,
) -> pd.DataFrame:
    """
    GMM classifier clustering with mu_median, mu_mean, A/B, and concentration (R90/R10).

    Under the assumption that compact objects have a larger difference between
    mu_mean and mu_median due to their sharp central peak, the cluster with 
    highest average profile gap (|mu_mean - mu_median|) is assigned as COMPACT.
    """
    df = parameters_df.copy()
    has_id = 'ID' in df.columns
    if has_id:
        df = df.set_index('ID')
        
    df["source_type"] = UNCLASSIFIED

    transformed = pd.DataFrame(index=df.index)
    transformed["mu_median"] = df["mu_median"]
    transformed["mu_mean"] = df["mu_mean"]
    transformed["A/B"] = df["A"] / df["B"]
    transformed["concentration"] = df["R90"] / df["R10"]

    transformed["concentration"] = _safe_log1p(transformed["concentration"])

    feats = transformed.replace([np.inf, -np.inf], np.nan).dropna()

    # Scale and Fit GMM
    X = StandardScaler().fit_transform(feats.values)
    gmm = GaussianMixture(n_components=2, covariance_type="full", n_init=10, random_state=0)
    
    labels = gmm.fit_predict(X)
    probas = gmm.predict_proba(X)
    max_probas = probas.max(axis=1)

    feats['profile_gap'] = np.abs(feats['mu_mean'] - feats['mu_median'])
    
    gap_0 = feats.loc[labels == 0, 'profile_gap'].mean()
    gap_1 = feats.loc[labels == 1, 'profile_gap'].mean()
    
    # Cluster with highest average profile gap is compact
    compact_cluster = 0 if gap_0 > gap_1 else 1

    # If classification of object above confidence threshold, assign label,
    # otherwise leave as unclassified
    classified = np.where(
        max_probas >= min_confidence,
        np.where(labels == compact_cluster, COMPACT, DIFFUSE),
        UNCLASSIFIED,
    )

    df.loc[feats.index, "source_type"] = classified
    
    return df.reset_index() if has_id else df

