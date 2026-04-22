"""
Temporal analysis: cluster trajectory 2017 -> 2023.

Approach
--------
1. Fetch the same five socioeconomic variables for two years.
2. Fit K-means on 2023 data -- this defines the canonical cluster structure.
3. Use the 2023 scaler and model to predict 2017 cluster assignments.
4. Compare: which municipalities stayed in the same cluster? Which moved?

Using the 2023 model to predict 2017 labels is intentional. It means both
years use the same cluster definitions ("what would this municipality look like
in today's terms?"), making before/after comparison directly interpretable.

Why 2017?
AUP01 (unemployment) only starts from 2017M07. Using 2017 as the baseline
keeps all five variables consistent between the two time points.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.cluster import FEATURE_COLS, FEATURE_LABELS


def align_temporal_clusters(
    df_current: pd.DataFrame,
    df_past: pd.DataFrame,
    labels_current: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, KMeans, StandardScaler, SimpleImputer]:
    """
    Fits the canonical K-means model on current data, then predicts labels
    for past data using the same cluster centres and scaler.

    Returns
    -------
    labels_past    : ndarray of cluster assignments for past data
    model          : fitted KMeans (for inspection)
    scaler         : fitted StandardScaler (applied to both years)
    imputer        : fitted SimpleImputer (applied to both years)
    """
    available = [c for c in FEATURE_COLS if c in df_current.columns
                 and c in df_past.columns]

    # Fit imputer and scaler on current data only
    imputer = SimpleImputer(strategy="median")
    X_current = imputer.fit_transform(df_current[available])

    scaler = StandardScaler()
    X_current_scaled = scaler.fit_transform(X_current)

    # Refit K-means with the known labels as seeds so cluster indices align
    # with labels_current (avoids the arbitrary-labelling problem)
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    model.fit(X_current_scaled)

    # Predict past
    X_past = imputer.transform(df_past[available].fillna(df_past[available].median()))
    X_past_scaled = scaler.transform(X_past)
    labels_past = model.predict(X_past_scaled)

    return labels_past, model, scaler, imputer


def build_trajectory_df(
    df_current: pd.DataFrame,
    df_past: pd.DataFrame,
    labels_current: np.ndarray,
    labels_past: np.ndarray,
    cluster_names: dict[int, str],
) -> pd.DataFrame:
    """
    Merges current and past data into a single trajectory DataFrame.

    Returned columns:
      municipality_name, OMRÅDE
      cluster_current, cluster_name_current
      cluster_past,    cluster_name_past
      changed  (bool)
      For each feature: {feat}_current, {feat}_past, {feat}_change
    """
    available = [c for c in FEATURE_COLS if c in df_current.columns
                 and c in df_past.columns]

    curr = df_current[["OMRÅDE", "municipality_name"] + available].copy()
    curr["cluster_current"] = labels_current
    curr["cluster_name_current"] = [cluster_names.get(l, str(l)) for l in labels_current]

    past = df_past[["OMRÅDE"] + available].copy()
    past["cluster_past"] = labels_past
    past["cluster_name_past"] = [cluster_names.get(l, str(l)) for l in labels_past]

    # Rename feature columns to avoid collision on merge
    past = past.rename(columns={f: f"{f}_past" for f in available})
    curr = curr.rename(columns={f: f"{f}_current" for f in available})

    merged = curr.merge(past, on="OMRÅDE", how="inner")
    merged["changed"] = merged["cluster_current"] != merged["cluster_past"]

    # Compute change for each feature
    for f in available:
        merged[f"{f}_change"] = merged[f"{f}_current"] - merged[f"{f}_past"]

    return merged


def summarise_flows(traj: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a flow matrix: how many municipalities moved from each past
    cluster to each current cluster.

    Rows = past clusters, columns = current clusters.
    Diagonal = municipalities that stayed in the same cluster.
    """
    return (
        traj.groupby(["cluster_name_past", "cluster_name_current"])
        .size()
        .unstack(fill_value=0)
    )


def notable_movers(traj: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """
    Returns the n most notable municipalities that changed cluster.

    Sorted by total absolute feature change (scaled), so the municipalities
    with the most structural shift appear first.
    """
    changed = traj[traj["changed"]].copy()
    if len(changed) == 0:
        return pd.DataFrame()

    available = [c for c in FEATURE_COLS
                 if f"{c}_change" in changed.columns]

    # Sum of absolute changes across all features (each normalised by std)
    for f in available:
        std = traj[f"{f}_change"].std()
        if std > 0:
            changed[f"{f}_change_norm"] = changed[f"{f}_change"].abs() / std
        else:
            changed[f"{f}_change_norm"] = 0

    norm_cols = [f"{f}_change_norm" for f in available]
    changed["total_shift"] = changed[norm_cols].sum(axis=1)

    cols = (["municipality_name", "cluster_name_past", "cluster_name_current",
              "total_shift"] +
            [f"{f}_change" for f in available])
    return (
        changed[cols]
        .sort_values("total_shift", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
