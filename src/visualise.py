"""
Generates the interactive choropleth map and supporting charts.

All outputs are self-contained HTML files that can be served as static
files on GitHub Pages. No server or JavaScript framework required.

The choropleth uses Plotly's Choroplethmapbox with a Mapbox light
background accessed via the public Stamen Toner Lite tiles (no API key).

Layout philosophy
-----------------
- The map is the primary output. It embeds everything needed to display
  and explore all 98 municipalities in a single HTML file (~2-3 MB).
- Hovering a municipality shows its name, cluster, and all feature values.
- A legend panel shows the cluster name and count.
- A profile chart below the map shows the mean feature values per cluster
  as a radar/spider chart for quick comparison.
"""

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

from src.cluster import FEATURE_COLS, FEATURE_LABELS


# Colour palette: 5 clusters, colourblind-friendly
CLUSTER_COLOURS = [
    "#2c7bb6",  # blue
    "#d7191c",  # red
    "#1a9641",  # green
    "#fdae61",  # orange
    "#762a83",  # purple
]


def build_choropleth(
    df: pd.DataFrame,
    geojson: dict,
    cluster_names: dict[int, str],
    output_path: str = "outputs/map.html",
) -> str:
    """
    Builds a self-contained interactive choropleth HTML file.

    Parameters
    ----------
    df : DataFrame with columns municipality_code, municipality_name,
         cluster (int), and all FEATURE_COLS
    geojson : GeoJSON FeatureCollection from DAWA
    cluster_names : dict mapping cluster int -> descriptive name
    output_path : where to write the HTML file

    Returns the path to the written file.
    """
    Path(output_path).parent.mkdir(exist_ok=True)

    # Map feature columns to GeoJSON 'kode' property
    # DAWA uses zero-padded 4-digit codes (e.g. "0101")
    df = df.copy()
    df["municipality_code"] = df["municipality_code"].astype(str).str.zfill(4)
    df["cluster_name"] = df["cluster"].map(cluster_names)
    df["cluster_str"]  = df["cluster"].astype(str)

    # Build hover text
    def hover_text(row):
        lines = [f"<b>{row['municipality_name']}</b>",
                 f"Cluster: {row['cluster_name']}",
                 ""]
        for col in FEATURE_COLS:
            if col in row and pd.notna(row[col]):
                label = FEATURE_LABELS.get(col, col)
                val   = row[col]
                if "income" in col:
                    lines.append(f"{label}: {val:,.0f} kr.")
                else:
                    lines.append(f"{label}: {val:.1f}%")
        return "<br>".join(lines)

    df["hover"] = df.apply(hover_text, axis=1)

    # One trace per cluster so the legend works correctly
    fig = go.Figure()

    n_clusters = df["cluster"].nunique()
    for i in sorted(df["cluster"].unique()):
        subset    = df[df["cluster"] == i]
        colour    = CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)]
        name      = cluster_names.get(i, f"Cluster {i+1}")
        count     = len(subset)

        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson,
            locations=subset["municipality_code"],
            featureidkey="properties.kode",
            z=[i] * len(subset),          # constant z so all in cluster share one colour
            zmin=0,
            zmax=n_clusters - 1,
            colorscale=[[0, colour], [1, colour]],
            showscale=False,
            name=f"{name} (n={count})",
            text=subset["hover"],
            hovertemplate="%{text}<extra></extra>",
            marker_opacity=0.85,
            marker_line_width=0.5,
            marker_line_color="white",
        ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=5.8,
        mapbox_center={"lat": 56.0, "lon": 10.5},
        margin={"r": 0, "t": 60, "l": 0, "b": 0},
        height=650,
        title={
            "text": "Socioeconomic Clustering of Danish Municipalities",
            "font": {"size": 18, "family": "Arial"},
            "x": 0.5,
        },
        legend={
            "title": {"text": "Cluster"},
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "#ccc",
            "borderwidth": 1,
            "x": 0.01,
            "y": 0.99,
        },
    )

    # --- Radar / spider chart for cluster profiles ---
    profile_fig = _build_radar(df, cluster_names)

    # --- Combine into one HTML file ---
    map_html     = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="choropleth")
    radar_html   = profile_fig.to_html(full_html=False, include_plotlyjs=False, div_id="radar")

    html = _wrap_html(map_html, radar_html)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[visualise] Map written to {output_path}")
    return output_path


def _hex_to_rgba(hex_colour: str, alpha: float = 0.15) -> str:
    """Converts a #rrggbb hex string to an rgba(...) string."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _build_radar(df: pd.DataFrame, cluster_names: dict[int, str]) -> go.Figure:
    """
    Builds a radar chart comparing mean feature values across clusters.

    Values are normalised to 0-100 within each feature so all axes share
    the same scale. This makes cluster shapes visually comparable.
    """
    # Compute mean per cluster
    profile = df.groupby("cluster")[FEATURE_COLS].mean()

    # Normalise each feature to 0-100
    for col in FEATURE_COLS:
        col_min = profile[col].min()
        col_max = profile[col].max()
        rng = col_max - col_min
        if rng > 0:
            profile[col] = 100 * (profile[col] - col_min) / rng
        else:
            profile[col] = 50.0

    categories = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]
    # Close the polygon by repeating the first category
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    for i, row in profile.iterrows():
        vals = row[FEATURE_COLS].tolist()
        vals_closed = vals + [vals[0]]
        colour = CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=categories_closed,
            fill="toself",
            name=cluster_names.get(i, f"Cluster {i+1}"),
            line_color=colour,
            fillcolor=_hex_to_rgba(colour, alpha=0.15),
            opacity=0.8,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title={
            "text": "Cluster Profiles (normalised)",
            "x": 0.5,
            "font": {"size": 15},
        },
        height=450,
        margin={"t": 60, "b": 20},
    )
    return fig


def _wrap_html(map_html: str, radar_html: str) -> str:
    """
    Wraps the two Plotly figures into a styled full-page HTML document.

    The page is designed to be embedded as an iframe on mikkelbohnke.com
    or opened standalone. It matches the site's dark/light aesthetic.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Socioeconomic Clustering of Danish Municipalities</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f6f8fa;
      color: #24292f;
      padding: 24px 16px;
    }}
    .container {{ max-width: 960px; margin: 0 auto; }}
    h1 {{
      font-size: 22px;
      margin-bottom: 6px;
    }}
    .meta {{
      font-size: 13px;
      color: #57606a;
      margin-bottom: 20px;
    }}
    .card {{
      background: white;
      border: 1px solid #d0d7de;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 20px;
    }}
    .section-title {{
      font-size: 14px;
      font-weight: 600;
      color: #57606a;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 12px;
    }}
    .source {{
      font-size: 11px;
      color: #8c959f;
      margin-top: 16px;
      text-align: center;
    }}
    a {{ color: #0969da; }}
  </style>
</head>
<body>
<div class="container">

  <div class="card">
    <p class="section-title">Interactive Map</p>
    {map_html}
    <p class="source">
      Data: Statistics Denmark (dst.dk), CC 4.0 BY &bull;
      Boundaries: Dataforsyningen (DAWA) &bull;
      Analysis: K-means clustering, k=5
    </p>
  </div>

  <div class="card">
    <p class="section-title">Cluster Profiles</p>
    {radar_html}
  </div>

</div>
</body>
</html>"""
