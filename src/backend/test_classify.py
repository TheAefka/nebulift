"""
Script to test combinations of features for GMM classification
and evaluate against known labels. Not part of the main Nebulift application.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import itertools

DIFFUSE = 0
COMPACT = 1
UNCLASSIFIED = -1

def classify_objects_gmm(
    df: pd.DataFrame, 
    features: list, 
    known_labels: dict,
    min_confidence: float = 0.7
):
    """
    Runs GMM on a given set of features and evaluates against known labels.
    """
    valid_rows = df[features].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(valid_rows) < 2:
        return 0.0  # Failsafe for empty data
        
    # Scale features
    X = StandardScaler().fit_transform(valid_rows.values)

    # Fit GMM
    gmm = GaussianMixture(n_components=2, covariance_type="full", n_init=10, random_state=0)
    labels = gmm.fit_predict(X)
    probas = gmm.predict_proba(X)
    max_probas = probas.max(axis=1)

    # Map labels to the dataframe
    results = pd.DataFrame({'ID': valid_rows.index, 'cluster': labels, 'confidence': max_probas})
    results = results.set_index('ID')
    
    # Filter only objects we have ground truth for, and passed confidence threshold
    eval_df = results.loc[results.index.intersection(known_labels.keys())].copy()
    eval_df = eval_df[eval_df['confidence'] >= min_confidence]
    
    if eval_df.empty:
        return 0.0

    # Add ground truth
    eval_df['truth'] = eval_df.index.map(known_labels)

    # We don't know if GMM Cluster 0 is Compact or Diffuse. 
    
    # Mapping A: Cluster 0 = Diffuse, Cluster 1 = Compact
    preds_A = np.where(eval_df['cluster'] == 1, COMPACT, DIFFUSE)
    score_A = f1_score(eval_df['truth'], preds_A, pos_label=COMPACT, average='macro')
    
    # Mapping B: Cluster 0 = Compact, Cluster 1 = Diffuse
    preds_B = np.where(eval_df['cluster'] == 0, COMPACT, DIFFUSE)
    score_B = f1_score(eval_df['truth'], preds_B, pos_label=COMPACT, average='macro')
    
    # Return best F1 score
    return max(score_A, score_B)


def find_best_parameters(df: pd.DataFrame, candidate_features: list, known_labels: dict, min_features=2, max_features=4):
    """
    Iterates through combinations of candidate features to find the best performing set.
    """
    df = df.set_index('ID')
        
    # Calculate derived features
    if 'A' in df.columns and 'B' in df.columns:
        df['A/B'] = df['A'] / df['B']
        if 'A/B' not in candidate_features:
            candidate_features.append('A/B')
            
    if 'R90' in df.columns and 'R10' in df.columns:
        df['concentration'] = df['R90'] / df['R10']
        if 'concentration' not in candidate_features:
            candidate_features.append('concentration')

    # Remove columns not in dataframe
    candidate_features = [f for f in candidate_features if f in df.columns]

    results = []
    
    print(f"Testing feature combinations on {len(known_labels)} known objects...")
    
    for r in range(min_features, max_features + 1):
        for combo in itertools.combinations(candidate_features, r):
            features_list = list(combo)
            
            # Apply log1p to highly skewed features to normalize distributions for GMM
            temp_df = df.copy()
            for feat in features_list:
                if feat in ['R_fwhm', 'R_e', 'R10', 'R90', 'total_flux']:
                    temp_df[feat] = np.log1p(np.clip(temp_df[feat].astype(float), 0, None))
            
            score = classify_objects_gmm(temp_df, features_list, known_labels)
            results.append({'features': features_list, 'f1_score': score})

    # Sort results by best score
    results_df = pd.DataFrame(results).sort_values(by='f1_score', ascending=False).reset_index(drop=True)
    
    print("\nTop 5 Feature Combinations:")
    for i, row in results_df.head(5).iterrows():
        print(f"Score: {row['f1_score']:.4f} | Features: {row['features']}")
        
    return results_df



# {object id: class label, ...} used as ground truth
known_labels = {
13268262: 0,
16174824: 1,
15660031: 1,
14961839: 1,
16889754: 1,
16764032: 1,
16031975: 1,
16397818: 0,
16168930: 0,
16054525: 0,
16632596: 0,
16695612: 0,
16604041: 0,
16706999: 0,
15934752: 0,
16770035: 0,
19928326: 1,
18097012: 1,
19080943: 1,
19693423: 1,
19544408: 1,
18193927: 1,
19589312: 1,
19692361: 1,
19017334: 1,
21065485: 1,
21683715: 1,
12512172: 1,
12826784: 1,
13868487: 1,
10156208: 1,
12421562: 1,
12781676: 1,
11408620: 1,
9538482: 1,
6929452: 0,
7209703: 1,
6397101: 0,
5756547: 0,
5567679: 0,
6231345: 0,
5847890: 0,
5670543: 0,
6053679: 0,
6110934: 0,
6602959: 1,
6425624: 1,
9593335: 0,
7247137: 0,
6108601: 1,
6715210: 1,
7046989: 1,
7184545: 1,
8626278: 1,
8592267: 1,
8632077: 1,
8923809: 1,
13334737: 1,
13077272: 1,
12768223: 1,
13563632: 1,
14725459: 1,
15629468: 1,
15753237: 1,
8133366: 1,
9740070: 1,
}

candidates = [
    'mu_max', 'mu_median', 'mu_mean', 
    'A/B', 'concentration', 'total_flux',
    'R_fwhm', 'R_e', 'R10', 'R90',
]


# Load df_parameters from a CSV. Assuming it was generated by MTOlib/Nebulift beforehand
df_parameters = pd.read_csv('parameters.csv', index_col='ID')

best_combos = find_best_parameters(df_parameters, candidates, known_labels)

print("\nBest Feature Combination:")
if not best_combos.empty:
    best_row = best_combos.iloc[0]
    print(f"Score: {best_row['f1_score']:.4f} | Features: {best_row['features']}")