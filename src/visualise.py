"""
Generates the self-contained interactive HTML dashboard.

Layout
------
1. Choropleth map          -- 98 municipalities coloured by cluster
2. Cluster profile bars    -- mean values per cluster per feature (clearer than radar)
3. Cluster summary cards   -- what each cluster means in plain language
4. Municipality table      -- all 98 rows, sortable, searchable

All output is a single HTML file served via GitHub Pages.
"""

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from src.cluster import FEATURE_COLS, FEATURE_LABELS


CLUSTER_COLOURS = [
    "#2c7bb6",   # blue
    "#d7191c",   # red
    "#1a9641",   # green
    "#f4a700",   # amber
    "#762a83",   # purple
]


def build_choropleth(
    df: pd.DataFrame,
    geojson: dict,
    cluster_names: dict[int, str],
    trajectory=None,
    output_path: str = "outputs/map.html",
) -> str:
    Path(output_path).parent.mkdir(exist_ok=True)

    df = df.copy()
    df["municipality_code"] = df["municipality_code"].astype(str).str.zfill(4)
    df["cluster_name"] = df["cluster"].map(cluster_names)

    # Build rich hover text
    def hover(row):
        lines = [f"<b>{row['municipality_name']}</b>",
                 f"<i>{row['cluster_name']}</i>", ""]
        for col in FEATURE_COLS:
            if col in row and pd.notna(row[col]):
                label = FEATURE_LABELS.get(col, col)
                val   = row[col]
                if "income" in col:
                    lines.append(f"{label}: {val:,.0f} kr.")
                else:
                    lines.append(f"{label}: {val:.1f}%")
        return "<br>".join(lines)

    df["hover"] = df.apply(hover, axis=1)

    # --- Choropleth ---
    map_fig = go.Figure()
    n_clusters = df["cluster"].nunique()
    for i in sorted(df["cluster"].unique()):
        sub    = df[df["cluster"] == i]
        colour = CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)]
        name   = cluster_names.get(i, f"Cluster {i+1}")
        map_fig.add_trace(go.Choroplethmapbox(
            geojson=geojson,
            locations=sub["municipality_code"],
            featureidkey="properties.kode",
            z=[i] * len(sub),
            zmin=0, zmax=n_clusters - 1,
            colorscale=[[0, colour], [1, colour]],
            showscale=False,
            name=f"{name} (n={len(sub)})",
            text=sub["hover"],
            hovertemplate="%{text}<extra></extra>",
            marker_opacity=0.85,
            marker_line_width=0.8,
            marker_line_color="white",
        ))

    map_fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=5.8,
        mapbox_center={"lat": 56.0, "lon": 10.5},
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=600,
        title={"text": "Socioeconomic Clustering of Danish Municipalities",
               "font": {"size": 17}, "x": 0.5},
        legend={"title": {"text": "Cluster"},
                "bgcolor": "rgba(255,255,255,0.9)",
                "bordercolor": "#ccc", "borderwidth": 1,
                "x": 0.01, "y": 0.99},
    )

    # --- Bar charts: cluster profiles ---
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    profiles = df.groupby("cluster")[available_features].mean()

    bar_fig = go.Figure()
    for i, row in profiles.iterrows():
        colour = CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)]
        name   = cluster_names.get(i, f"Cluster {i+1}")
        labels = [FEATURE_LABELS.get(c, c) for c in available_features]
        vals   = row[available_features].tolist()
        bar_fig.add_trace(go.Bar(
            name=name,
            x=labels,
            y=vals,
            marker_color=colour,
            opacity=0.85,
        ))

    bar_fig.update_layout(
        barmode="group",
        title={"text": "Mean Feature Values per Cluster", "x": 0.5,
               "font": {"size": 15}},
        height=380,
        margin={"t": 50, "b": 80, "l": 60, "r": 20},
        legend={"orientation": "h", "y": -0.25},
        yaxis_title="Value",
        plot_bgcolor="white",
        yaxis={"gridcolor": "#eee"},
    )

    # --- Cluster summary cards data ---
    # For each cluster compute the most distinctive features (highest z-score)
    grand_mean = df[available_features].mean()
    grand_std  = df[available_features].std().replace(0, 1)

    cluster_summaries = {}
    for i in sorted(df["cluster"].unique()):
        sub     = df[df["cluster"] == i]
        profile = sub[available_features].mean()
        z       = (profile - grand_mean) / grand_std
        top     = z.abs().nlargest(3)
        traits  = []
        for feat in top.index:
            direction = "høj" if z[feat] > 0 else "lav"
            label     = FEATURE_LABELS.get(feat, feat)
            val       = profile[feat]
            if "income" in feat:
                traits.append(f"{direction} {label.lower()} ({val:,.0f} kr.)")
            else:
                traits.append(f"{direction} {label.lower()} ({val:.1f}%)")
        examples = sub.nlargest(3, "pop_total")["municipality_name"].tolist() \
            if "pop_total" in sub.columns else sub["municipality_name"].head(3).tolist()
        cluster_summaries[i] = {
            "name":     cluster_names.get(i, f"Cluster {i+1}"),
            "colour":   CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)],
            "n":        len(sub),
            "traits":   traits,
            "examples": examples,
        }

    # --- Municipality table data ---
    table_cols = ["municipality_name", "cluster_name"] + available_features
    table_df   = df[table_cols].copy()
    table_df   = table_df.sort_values("municipality_name")
    # Round for display
    for col in available_features:
        if "income" in col:
            table_df[col] = table_df[col].round(0).astype(int)
        else:
            table_df[col] = table_df[col].round(1)
    table_rows_json = table_df.to_json(orient="records", force_ascii=False)

    # --- Render ---
    map_html = map_fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="map")
    bar_html = bar_fig.to_html(full_html=False, include_plotlyjs=False, div_id="bars")

    html = _wrap_html(map_html, bar_html, cluster_summaries,
                      available_features, table_rows_json,
                      trajectory=trajectory,
                      cluster_names=cluster_names)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = Path(output_path).stat().st_size // 1024
    print(f"[visualise] Written to {output_path} ({size_kb} KB)")
    return output_path


# ---------------------------------------------------------------------------

def _wrap_html(map_html, bar_html, summaries, features, table_rows_json,
               trajectory=None, cluster_names=None) -> str:
    feature_labels_js = json.dumps(
        {f: FEATURE_LABELS.get(f, f) for f in features}, ensure_ascii=False
    )

    cards_html = ""
    for i, s in summaries.items():
        traits_li  = "".join(f"<li>{t}</li>" for t in s["traits"])
        examples_t = ", ".join(s["examples"])
        cards_html += f"""
        <div class="cluster-card" style="border-top:4px solid {s['colour']}">
          <div class="cluster-card-header">
            <span class="cluster-dot" style="background:{s['colour']}"></span>
            <strong>{s['name']}</strong>
            <span class="cluster-n">n={s['n']}</span>
          </div>
          <ul class="traits">{traits_li}</ul>
          <div class="examples">Eksempler: {examples_t}</div>
        </div>"""

    # Build temporal sections if trajectory data is available
    temporal_section = ""
    if trajectory is not None and cluster_names is not None and len(trajectory) > 0:
        sankey_html  = _build_sankey(trajectory, cluster_names)
        movers_html  = _build_movers_table_html(trajectory)
        n_changed    = trajectory["changed"].sum()
        n_total      = len(trajectory)
        temporal_section = f"""
  <div class="card">
    <h2>Cluster trajectories 2017 → 2023</h2>
    <p style="font-size:13px;color:#57606a;margin-bottom:16px">
      {n_changed} ud af {n_total} kommuner skiftede klynge i perioden 2017–2023.
      Brede strømme = mange kommuner i den overgang. Diagonale strømme (samme farve)
      = kommuner der forblev i samme klynge.
    </p>
    {sankey_html}
  </div>

  <div class="card">
    <h2>Kommuner der skiftede klynge ({n_changed} stk.)</h2>
    <p style="font-size:13px;color:#57606a;margin-bottom:12px">
      Δ-kolonnerne viser ændringen i den socioøkonomiske indikator fra 2017 til 2023.
      Grøn = forbedring relativt til klyngedefinitionen, rød = forværring.
    </p>
    {movers_html}
  </div>"""

    return f"""<!DOCTYPE html>
<html lang="da">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kommuner Clustering</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f0f2f5; color: #24292f; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px; }}
    h2 {{ font-size: 13px; font-weight: 700; letter-spacing: .08em;
          text-transform: uppercase; color: #57606a; margin-bottom: 14px; }}
    .card {{ background: white; border: 1px solid #d0d7de; border-radius: 8px;
             padding: 20px; margin-bottom: 20px; }}

    /* Cluster summary cards */
    .cards-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px,1fr));
                   gap: 14px; }}
    .cluster-card {{ background: white; border: 1px solid #d0d7de; border-radius: 6px;
                     padding: 14px; }}
    .cluster-card-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }}
    .cluster-dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
    .cluster-n {{ margin-left: auto; font-size: 12px; color: #57606a; }}
    .traits {{ padding-left: 16px; font-size: 13px; color: #444; line-height: 1.6; }}
    .traits li {{ margin-bottom: 3px; }}
    .examples {{ margin-top: 10px; font-size: 11px; color: #57606a; font-style: italic; }}

    /* Municipality table */
    .search-bar {{ width: 100%; padding: 8px 12px; border: 1px solid #d0d7de;
                   border-radius: 6px; font-size: 14px; margin-bottom: 12px; outline: none; }}
    .search-bar:focus {{ border-color: #0969da; }}
    .tbl-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ background: #f6f8fa; padding: 8px 12px; text-align: left;
          border-bottom: 2px solid #d0d7de; white-space: nowrap;
          cursor: pointer; user-select: none; }}
    th:hover {{ background: #eaeef2; }}
    th .sort-icon {{ color: #aaa; margin-left: 4px; font-size: 11px; }}
    td {{ padding: 7px 12px; border-bottom: 1px solid #eaecef; white-space: nowrap; }}
    tr:hover td {{ background: #f6f8fa; }}
    .cluster-pill {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
                     font-size: 11px; font-weight: 600; color: white; }}
    .no-results {{ text-align: center; padding: 24px; color: #57606a; font-size: 14px; }}

    /* Responsive */
    @media (max-width: 600px) {{
      .cards-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
<div class="container">

  <div class="card">
    <h2>Interactive Map</h2>
    {map_html}
    <p style="font-size:11px;color:#8c959f;margin-top:10px;text-align:center">
      Data: Statistics Denmark (dst.dk), CC 4.0 BY &bull;
      Boundaries: Dataforsyningen (DAWA) &bull; Analysis: K-means clustering, k=5
    </p>
  </div>

  <div class="card">
    <h2>Cluster Profiles</h2>
    {bar_html}
  </div>

  <div class="card">
    <h2>Cluster Summary</h2>
    <div class="cards-grid">{cards_html}</div>
  </div>

  {temporal_section}

  <div class="card">
    <h2>All Municipalities</h2>
    <input class="search-bar" id="search" type="text"
           placeholder="Search municipality or cluster..." oninput="filterTable()">
    <div class="tbl-wrap">
      <table id="tbl">
        <thead id="thead"></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>

</div>

<script>
const ROWS = {table_rows_json};
const FEAT_LABELS = {feature_labels_js};

const CLUSTER_COLOURS = {json.dumps(list(CLUSTER_COLOURS))};

const COL_ORDER = ["municipality_name", "cluster_name",
  ...Object.keys(FEAT_LABELS)];
const COL_LABELS = {{
  municipality_name: "Kommune",
  cluster_name: "Cluster",
  ...FEAT_LABELS,
}};

let sortCol = "municipality_name";
let sortAsc = true;

function clusterColor(name) {{
  const colours = {{}}; 
  {_cluster_colour_js(summaries)}
  return colours[name] || "#999";
}}

function buildHeader() {{
  const tr = document.createElement("tr");
  COL_ORDER.forEach(col => {{
    if (!(col in COL_LABELS)) return;
    const th = document.createElement("th");
    th.innerHTML = COL_LABELS[col] + '<span class="sort-icon">⇅</span>';
    th.onclick = () => {{
      if (sortCol === col) sortAsc = !sortAsc;
      else {{ sortCol = col; sortAsc = true; }}
      renderTable(currentRows());
    }};
    tr.appendChild(th);
  }});
  document.getElementById("thead").appendChild(tr);
}}

function currentRows() {{
  const q = document.getElementById("search").value.toLowerCase();
  return ROWS.filter(r =>
    (r.municipality_name || "").toLowerCase().includes(q) ||
    (r.cluster_name || "").toLowerCase().includes(q)
  );
}}

function renderTable(rows) {{
  const sorted = [...rows].sort((a, b) => {{
    let av = a[sortCol], bv = b[sortCol];
    if (typeof av === "string") av = av.toLowerCase();
    if (typeof bv === "string") bv = bv.toLowerCase();
    if (av < bv) return sortAsc ? -1 : 1;
    if (av > bv) return sortAsc ? 1 : -1;
    return 0;
  }});
  const tbody = document.getElementById("tbody");
  tbody.innerHTML = "";
  if (sorted.length === 0) {{
    tbody.innerHTML = '<tr><td colspan="99" class="no-results">No municipalities match your search.</td></tr>';
    return;
  }}
  sorted.forEach(r => {{
    const tr = document.createElement("tr");
    COL_ORDER.forEach(col => {{
      if (!(col in COL_LABELS)) return;
      const td = document.createElement("td");
      if (col === "cluster_name") {{
        const colour = clusterColor(r.cluster_name);
        td.innerHTML = `<span class="cluster-pill" style="background:${{colour}}">${{r.cluster_name}}</span>`;
      }} else if (col === "median_income") {{
        td.textContent = r[col] != null ? Number(r[col]).toLocaleString("da-DK") + " kr." : "—";
      }} else if (col in FEAT_LABELS) {{
        td.textContent = r[col] != null ? r[col] + "%" : "—";
      }} else {{
        td.textContent = r[col] != null ? r[col] : "—";
      }}
      tr.appendChild(td);
    }});
    tbody.appendChild(tr);
  }});
}}

function filterTable() {{ renderTable(currentRows()); }}

buildHeader();
renderTable(ROWS);
</script>
</body>
</html>"""


def _cluster_colour_js(summaries: dict) -> str:
    """Generates JS object literal mapping cluster name -> colour."""
    lines = []
    for s in summaries.values():
        name   = s["name"].replace("'", "\\'")
        colour = s["colour"]
        lines.append(f"  colours['{name}'] = '{colour}';")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Temporal visualisation additions
# ---------------------------------------------------------------------------

def _build_sankey(traj: "pd.DataFrame", cluster_names: dict) -> str:
    """
    Builds a Plotly Sankey diagram showing how municipalities flowed between
    clusters from 2017 to 2023.

    Left nodes  = 2017 clusters
    Right nodes = 2023 clusters
    Flow width  = number of municipalities in that transition
    """
    import plotly.graph_objects as go

    n = len(cluster_names)
    # Node labels: past clusters on left, current on right
    node_labels = (
        [f"{cluster_names[i]}\n(2017)" for i in range(n)] +
        [f"{cluster_names[i]}\n(2023)" for i in range(n)]
    )
    node_colours = (
        [CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)] for i in range(n)] +
        [CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)] for i in range(n)]
    )

    # Build flows
    source_list, target_list, value_list, link_colours = [], [], [], []
    for past_c in range(n):
        for curr_c in range(n):
            count = len(traj[
                (traj["cluster_past"]    == past_c) &
                (traj["cluster_current"] == curr_c)
            ])
            if count == 0:
                continue
            source_list.append(past_c)
            target_list.append(curr_c + n)
            value_list.append(count)
            # Same cluster = solid colour, changed = light grey
            if past_c == curr_c:
                base = CLUSTER_COLOURS[curr_c % len(CLUSTER_COLOURS)].lstrip("#")
                r, g, b = int(base[0:2],16), int(base[2:4],16), int(base[4:6],16)
                link_colours.append(f"rgba({r},{g},{b},0.45)")
            else:
                link_colours.append("rgba(150,150,150,0.3)")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=22,
            line=dict(color="white", width=0.5),
            label=node_labels,
            color=node_colours,
        ),
        link=dict(
            source=source_list,
            target=target_list,
            value=value_list,
            color=link_colours,
        ),
    ))
    fig.update_layout(
        title={"text": "Cluster trajectories 2017 → 2023",
               "x": 0.5, "font": {"size": 15}},
        height=400,
        margin={"t": 50, "b": 20, "l": 20, "r": 20},
        font_size=11,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="sankey")


def _build_movers_table_html(traj: "pd.DataFrame") -> str:
    """Builds an HTML table of municipalities that changed cluster."""
    changed = traj[traj["changed"]].copy()
    if len(changed) == 0:
        return "<p style='color:#57606a;font-size:14px'>No municipalities changed cluster.</p>"

    from src.cluster import FEATURE_COLS, FEATURE_LABELS
    available_changes = [f"{c}_change" for c in FEATURE_COLS if f"{c}_change" in changed.columns]

    rows_html = ""
    for _, row in changed.iterrows():
        direction = "▲" if _is_improvement(row) else "▼"
        dir_class = "improved" if _is_improvement(row) else "declined"
        change_cells = ""
        for col in available_changes:
            feat  = col.replace("_change", "")
            label = FEATURE_LABELS.get(feat, feat)
            val   = row[col]
            sign  = "+" if val > 0 else ""
            if "income" in feat:
                formatted = f"{sign}{val:,.0f} kr."
            else:
                formatted = f"{sign}{val:.1f}%"
            colour = "#1a7f37" if val > 0 else "#cf222e"
            change_cells += f'<td style="color:{colour}">{formatted}</td>'

        rows_html += f"""
        <tr>
          <td><strong>{row['municipality_name']}</strong></td>
          <td>{row['cluster_name_past']}</td>
          <td>{row['cluster_name_current']}</td>
          <td class="{dir_class}">{direction}</td>
          {change_cells}
        </tr>"""

    feat_headers = "".join(
        f"<th>Δ {FEATURE_LABELS.get(col.replace('_change',''), col.replace('_change',''))}</th>"
        for col in available_changes
    )

    return f"""
    <input class="search-bar" id="movers-search" type="text"
           placeholder="Search municipality..."
           oninput="filterMovers()">
    <div class="tbl-wrap">
    <table id="movers-tbl">
      <thead>
        <tr>
          <th>Kommune</th>
          <th>Cluster 2017</th>
          <th>Cluster 2023</th>
          <th>Retning</th>
          {feat_headers}
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    <style>
      .improved {{ color: #1a7f37; font-weight:700; }}
      .declined {{ color: #cf222e; font-weight:700; }}
    </style>
    <script>
    function filterMovers() {{
      const q = document.getElementById('movers-search').value.toLowerCase();
      document.querySelectorAll('#movers-tbl tbody tr').forEach(tr => {{
        tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
      }});
    }}
    </script>"""


def _is_improvement(row: "pd.Series") -> bool:
    """
    Heuristic: a municipality 'improved' if income or education increased
    OR unemployment decreased.
    """
    improved = 0
    if "median_income_change" in row and row["median_income_change"] > 0:
        improved += 1
    if "pct_higher_edu_change" in row and row["pct_higher_edu_change"] > 0:
        improved += 1
    if "unemployment_rate_change" in row and row["unemployment_rate_change"] < 0:
        improved += 1
    return improved >= 2
