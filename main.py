"""
main.py -- Full analysis pipeline including temporal comparison.

Usage:
    python main.py                    # uses cached data, temporal on
    python main.py --no-cache         # re-fetch from DST API
    python main.py --no-temporal      # skip temporal (faster, single year)
    python main.py --k 5              # override cluster count
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.fetch     import build_dataset, build_dataset_for_year, fetch_geodata
from src.cluster   import (
    prepare_features, elbow_and_silhouette,
    run_kmeans, run_hierarchical,
    cluster_profiles, name_clusters,
)
from src.temporal  import (
    align_temporal_clusters, build_trajectory_df,
    summarise_flows, notable_movers,
)
from src.visualise import build_choropleth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache",    action="store_true")
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--k",           type=int, default=5)
    parser.add_argument("--output",      default="outputs/map.html")
    args = parser.parse_args()

    cache = not args.no_cache

    # -----------------------------------------------------------------
    # Step 1: Current data (2023)
    # -----------------------------------------------------------------
    print("=== Step 1: Current data (2023) ===")
    df_current = build_dataset_for_year("2023", cache=cache)
    print(f"  {len(df_current)} municipalities, {len(df_current.columns)} columns")

    # -----------------------------------------------------------------
    # Step 2: Features and clustering
    # -----------------------------------------------------------------
    print("\n=== Step 2: Clustering ===")
    X, names, scaler, imputer = prepare_features(df_current)
    print(f"  Feature matrix: {X.shape}")

    scores = elbow_and_silhouette(X)
    print(scores.to_string(index=False))

    k = args.k
    labels_current = run_kmeans(X, k=k)
    labels_hc      = run_hierarchical(X, k=k)

    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels_current, labels_hc)
    print(f"  K-means vs hierarchical ARI: {ari:.3f}")

    df_current["cluster"] = labels_current
    profiles      = cluster_profiles(df_current, labels_current)
    cluster_names = name_clusters(profiles)

    print("\n  Cluster names:")
    for cid, name in cluster_names.items():
        n = (labels_current == cid).sum()
        print(f"    {cid}: {name} (n={n})")

    # -----------------------------------------------------------------
    # Step 3: Temporal analysis (2017 -> 2023)
    # -----------------------------------------------------------------
    traj = None
    if not args.no_temporal:
        print("\n=== Step 3: Temporal analysis (2017 -> 2023) ===")
        try:
            df_past = build_dataset_for_year("2017", cache=cache)

            # Align: predict 2017 assignments using the 2023 cluster model
            labels_past, model, _, _ = align_temporal_clusters(
                df_current, df_past, labels_current, k=k
            )
            df_past["cluster"] = labels_past

            # Build trajectory
            traj = build_trajectory_df(
                df_current, df_past,
                labels_current, labels_past,
                cluster_names,
            )

            flows   = summarise_flows(traj)
            movers  = notable_movers(traj, n=20)
            changed = traj["changed"].sum()

            print(f"  {changed} of {len(traj)} municipalities changed cluster")
            print(f"\n  Flow matrix (past -> current):")
            print(flows.to_string())

            if len(movers) > 0:
                print(f"\n  Top movers:")
                print(movers[["municipality_name",
                               "cluster_name_past",
                               "cluster_name_current"]].to_string(index=False))

        except Exception as e:
            print(f"  WARNING: Temporal analysis failed: {e}")
            traj = None

    # -----------------------------------------------------------------
    # Step 4: GeoJSON
    # -----------------------------------------------------------------
    print("\n=== Step 4: GeoJSON ===")
    geojson = fetch_geodata()
    print(f"  {len(geojson['features'])} features")

    # -----------------------------------------------------------------
    # Step 5: Visualisation
    # -----------------------------------------------------------------
    print("\n=== Step 5: Visualisation ===")
    output = build_choropleth(
        df_current, geojson, cluster_names,
        trajectory=traj,
        output_path=args.output,
    )
    print(f"  Done: {output}")


if __name__ == "__main__":
    main()
