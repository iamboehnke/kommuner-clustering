"""
main.py -- Full analysis pipeline.

Runs: fetch -> feature engineering -> cluster selection -> clustering
      -> visualisation -> HTML output.

Usage:
    python main.py              # full run, uses cached data if available
    python main.py --no-cache   # force re-fetch from DST API
    python main.py --k 5        # override number of clusters

The output file is outputs/map.html. Commit this file to the repo and
GitHub Pages will serve it at:
    https://mikkelbohnke.github.io/kommuner-clustering/outputs/map.html
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.fetch     import build_dataset, fetch_geodata
from src.cluster   import (
    prepare_features, elbow_and_silhouette,
    run_kmeans, run_hierarchical,
    cluster_profiles, name_clusters,
)
from src.visualise import build_choropleth


def main():
    parser = argparse.ArgumentParser(description="Danish municipality clustering")
    parser.add_argument("--no-cache", action="store_true", help="Force re-fetch from DST")
    parser.add_argument("--k",        type=int, default=5,  help="Number of clusters")
    parser.add_argument("--output",   default="outputs/map.html", help="Output HTML path")
    args = parser.parse_args()

    # 1. Data
    print("=== Step 1: Fetching data ===")
    df = build_dataset(cache=not args.no_cache)
    print(f"  {len(df)} municipalities loaded")
    print(f"  Columns: {list(df.columns)}")

    # 2. Features
    print("\n=== Step 2: Feature preparation ===")
    X, names, scaler, imputer = prepare_features(df)
    print(f"  Feature matrix: {X.shape}")

    # 3. Cluster selection (informational -- doesn't change the run)
    print("\n=== Step 3: Cluster selection ===")
    scores = elbow_and_silhouette(X)
    print(scores.to_string(index=False))
    best_sil = scores.loc[scores["silhouette"].idxmax()]
    print(f"  Best silhouette score: k={int(best_sil['k'])}, score={best_sil['silhouette']:.3f}")
    print(f"  Using k={args.k} (override with --k)")

    # 4. Clustering
    print("\n=== Step 4: Clustering ===")
    km_labels = run_kmeans(X, k=args.k)
    hc_labels = run_hierarchical(X, k=args.k)

    df["cluster"]    = km_labels
    df["cluster_hc"] = hc_labels

    # Agreement between K-means and hierarchical (rough check)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(km_labels, hc_labels)
    print(f"  K-means vs hierarchical ARI: {ari:.3f} (1.0 = identical)")

    profiles = cluster_profiles(df, km_labels)
    print("\n  Cluster profiles (K-means):")
    print(profiles.to_string())

    cluster_names = name_clusters(profiles)
    print("\n  Cluster names:")
    for cid, name in cluster_names.items():
        size = (km_labels == cid).sum()
        print(f"    {cid}: {name} (n={size})")

    # 5. GeoJSON
    print("\n=== Step 5: Fetching GeoJSON ===")
    geojson = fetch_geodata()
    print(f"  {len(geojson['features'])} municipality boundaries")

    # 6. Visualisation
    print("\n=== Step 6: Building visualisation ===")
    output = build_choropleth(df, geojson, cluster_names, output_path=args.output)
    print(f"  Output: {output}")
    print("\nDone. Open the HTML file in a browser to view the map.")


if __name__ == "__main__":
    main()
