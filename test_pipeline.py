"""
test_pipeline.py -- Pipeline smoke test using synthetic data.

Runs the full cluster -> visualise pipeline without hitting any external API.
Produces a real output HTML file in outputs/test_map.html.

Run with:
    python test_pipeline.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.cluster   import (
    prepare_features, elbow_and_silhouette,
    run_kmeans, run_hierarchical,
    cluster_profiles, name_clusters, FEATURE_COLS,
)
from src.visualise import build_choropleth


def make_synthetic_data(n: int = 98, seed: int = 42) -> pd.DataFrame:
    """
    Generates a plausible synthetic dataset for 98 municipalities.

    Values are drawn from distributions centred on real Danish national averages:
      pct_elderly:        ~20%  (range 14-30)
      pct_youth:          ~20%  (range 15-27)
      unemployment_rate:  ~5%   (range 2-14)
      median_income:      ~310k (range 220k-450k)
      pct_higher_edu:     ~32%  (range 15-60)
      pct_social_housing: ~18%  (range 2-45)

    Five latent cluster types are embedded to give meaningful cluster structure.
    """
    rng = np.random.default_rng(seed)

    # Five cluster archetypes (approximate real patterns)
    archetypes = [
        # [elderly, youth, unemployment, income, higher_edu, social_housing]
        [28, 17, 6.5, 260_000, 22, 25],   # Ageing, rural, lower income
        [18, 23, 4.0, 340_000, 38, 12],   # Young families, suburban
        [16, 18, 8.0, 220_000, 20, 38],   # High unemployment, social housing
        [14, 19, 3.5, 420_000, 55, 8],    # Highly educated, urban, high income
        [22, 20, 5.0, 300_000, 28, 18],   # Average Danish municipality
    ]

    rows = []
    municipality_names = [f"Kommune {i:03d}" for i in range(1, n + 1)]

    for i in range(n):
        archetype = archetypes[i % len(archetypes)]
        noise_scale = [1.5, 1.2, 0.8, 15_000, 3.0, 3.0]
        vals = [
            max(0, archetype[j] + rng.normal(0, noise_scale[j]))
            for j in range(6)
        ]
        rows.append({
            "OMRÅDE":              f"{i+1:03d}",
            "municipality_code":   f"{i+1:04d}",
            "municipality_name":   municipality_names[i],
            "pop_total":           rng.integers(5_000, 600_000),
            "pct_elderly":         vals[0],
            "pct_youth":           vals[1],
            "unemployment_rate":   vals[2],
            "median_income":       vals[3],
            "pct_higher_edu":      vals[4],
            "pct_social_housing":  vals[5],
        })

    return pd.DataFrame(rows)


def make_synthetic_geojson(df: pd.DataFrame) -> dict:
    """
    Generates a minimal synthetic GeoJSON FeatureCollection.

    Each municipality gets a small square polygon laid out in a grid,
    purely to exercise the Plotly choropleth rendering.
    """
    features = []
    n = len(df)
    cols = int(np.ceil(np.sqrt(n)))

    for idx, row in df.iterrows():
        col_pos = idx % cols
        row_pos = idx // cols

        # Small square in lon/lat space (not geographically accurate)
        lon0 = 8.0 + col_pos * 0.35
        lat0 = 57.5 - row_pos * 0.25
        lon1, lat1 = lon0 + 0.3, lat0 + 0.2

        features.append({
            "type": "Feature",
            "properties": {
                "kode": row["municipality_code"],
                "navn": row["municipality_name"],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon0, lat0], [lon1, lat0],
                    [lon1, lat1], [lon0, lat1], [lon0, lat0],
                ]],
            },
        })

    return {"type": "FeatureCollection", "features": features}


def run():
    print("=== Synthetic pipeline test ===\n")

    # 1. Synthetic data
    df = make_synthetic_data(n=98)
    print(f"Synthetic dataset: {len(df)} municipalities, columns: {list(df.columns)}")

    # 2. Features
    X, names, scaler, imputer = prepare_features(df)
    print(f"Feature matrix: {X.shape}")

    # 3. Cluster selection
    print("\nElbow / silhouette analysis:")
    scores = elbow_and_silhouette(X, k_range=range(2, 8))
    print(scores.to_string(index=False))

    # 4. Clustering
    k = 5
    km_labels = run_kmeans(X, k=k)
    hc_labels = run_hierarchical(X, k=k)

    df["cluster"]    = km_labels
    df["cluster_hc"] = hc_labels

    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(km_labels, hc_labels)
    print(f"\nK-means vs hierarchical ARI: {ari:.3f}")

    profiles = cluster_profiles(df, km_labels)
    cluster_names = name_clusters(profiles)
    print("\nCluster names:")
    for cid, name in cluster_names.items():
        size = (km_labels == cid).sum()
        print(f"  {cid}: {name} (n={size})")

    # 5. GeoJSON
    geojson = make_synthetic_geojson(df)

    # 6. Visualisation
    Path("outputs").mkdir(exist_ok=True)
    output = build_choropleth(
        df, geojson, cluster_names,
        output_path="outputs/test_map.html"
    )
    size_kb = Path(output).stat().st_size // 1024
    print(f"\nOutput: {output} ({size_kb} KB)")
    print("PASS -- open outputs/test_map.html in a browser to inspect.")


if __name__ == "__main__":
    run()
