"""
Cluster analysis of Danish municipalities.

Applies K-means and hierarchical clustering to a feature matrix of
socioeconomic indicators. Includes elbow method and silhouette analysis
to select the number of clusters, and produces interpretable cluster
profiles for the write-up.

Design decisions
----------------
- StandardScaler before clustering: all features are on different scales
  (percentages, kr., rates). Scaling ensures no single variable dominates
  by magnitude.
- K-means for the primary clustering: interpretable, fast, and easy to
  explain in an interview or on a project page.
- Hierarchical (Ward linkage) as a cross-check: does not require k to be
  specified in advance, useful for validating the k-means choice.
- k=5 as default: enough to reveal structural variation without
  over-fragmenting. Verified by elbow and silhouette.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer


# Features used for clustering.
# Each is a meaningful socioeconomic dimension; together they capture
# demographic structure, labour market outcomes, income, education, and housing.
FEATURE_COLS = [
    "pct_elderly",        # Demographic: share aged 65+
    "pct_youth",          # Demographic: share aged 0-17
    "unemployment_rate",  # Labour market
    "median_income",      # Prosperity
    "pct_higher_edu",     # Human capital
    "pct_social_housing", # Housing structure
]

# Human-readable names for the write-up and chart tooltips
FEATURE_LABELS = {
    "pct_elderly":        "Elderly (65+) %",
    "pct_youth":          "Youth (0-17) %",
    "unemployment_rate":  "Unemployment rate %",
    "median_income":      "Median income (kr.)",
    "pct_higher_edu":     "Higher education %",
    "pct_social_housing": "Social housing %",
}


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Extracts, imputes, and scales the feature matrix from the raw dataset.
    Only uses features that are actually present in the DataFrame, so the
    pipeline works correctly when some DST tables fail to load.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]

    if missing:
        print(f"[cluster] Missing features (table fetch failed): {missing}")
    print(f"[cluster] Clustering on {len(available)} features: {available}")

    X_raw = df[available].copy()

    # Median imputation for suppressed/missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    names = df["municipality_name"].tolist() if "municipality_name" in df.columns else df["OMRÅDE"].tolist()
    return X_scaled, names, scaler, imputer


def elbow_and_silhouette(X: np.ndarray, k_range: range = range(2, 11)) -> pd.DataFrame:
    """
    Computes inertia (elbow method) and silhouette score for a range of k values.

    Returns a DataFrame with columns: k, inertia, silhouette.
    Used to make an informed choice of k before running the final model.
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        results.append({
            "k":          k,
            "inertia":    km.inertia_,
            "silhouette": silhouette_score(X, labels),
        })
    return pd.DataFrame(results)


def run_kmeans(X: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Runs K-means clustering with k clusters.

    n_init=20 reduces sensitivity to random initialisation. random_state=42
    ensures reproducibility across runs.

    Returns an array of integer cluster labels (0-indexed).
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    return km.fit_predict(X)


def run_hierarchical(X: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Runs agglomerative hierarchical clustering with Ward linkage.

    Ward linkage minimises the total within-cluster variance at each step,
    which tends to produce compact, roughly equal-sized clusters.
    Used as a cross-validation of the K-means result.
    """
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return hc.fit_predict(X)


def cluster_profiles(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Computes the mean of each feature per cluster.

    Returns a DataFrame with one row per cluster and one column per feature,
    plus a 'size' column showing how many municipalities are in each cluster.
    This is the primary input for the written interpretation of the clusters.
    """
    df = df.copy()
    df["cluster"] = labels

    agg = {}
    for col in FEATURE_COLS:
        if col in df.columns:
            agg[col] = "mean"
    agg["OMRÅDE"] = "count"

    profiles = (
        df.groupby("cluster")
        .agg(agg)
        .rename(columns={"OMRÅDE": "size"})
        .round(2)
    )
    return profiles


def name_clusters(profiles: pd.DataFrame) -> dict[int, str]:
    """
    Assigns a short descriptive name to each cluster based on its profile.

    Names are derived from the dominant characteristics visible in the profile
    DataFrame. These are used as legend labels on the choropleth map.

    The naming logic is heuristic -- it flags the most extreme deviations
    from the national average profile.
    """
    names = {}
    cols = [c for c in FEATURE_COLS if c in profiles.columns]
    national_mean = profiles[cols].mean()

    for cluster_id, row in profiles.iterrows():
        deviations = (row[cols] - national_mean) / (national_mean.abs() + 1e-9)

        # Find the two most positive and most negative deviations
        top_pos = deviations.nlargest(2)
        top_neg = deviations.nsmallest(2)

        # Use the single strongest signal to name the cluster
        strongest = abs(deviations).idxmax()
        direction = "high" if deviations[strongest] > 0 else "low"
        label_map = {
            "pct_elderly":        f"Ageing municipalities",
            "pct_youth":          f"Young families",
            "unemployment_rate":  f"High unemployment" if direction == "high" else "Low unemployment",
            "median_income":      f"High income" if direction == "high" else "Lower income",
            "pct_higher_edu":     f"Highly educated" if direction == "high" else "Lower education",
            "pct_social_housing": f"High social housing" if direction == "high" else "Low social housing",
        }
        names[cluster_id] = label_map.get(strongest, f"Cluster {cluster_id + 1}")

    # Ensure names are unique by appending the cluster id for any duplicates
    name_counts = {}
    for name in names.values():
        name_counts[name] = name_counts.get(name, 0) + 1

    seen_names = {}
    for cid in sorted(names.keys()):
        name = names[cid]
        if name_counts[name] > 1:
            seen_names[name] = seen_names.get(name, 0) + 1
            names[cid] = f"{name} {seen_names[name]}"

    return names
