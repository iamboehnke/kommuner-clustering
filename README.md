# Kommuner Clustering

Unsupervised machine learning on all 98 Danish municipalities using official register data from Statistics Denmark.

**Live map:** [mikkelbohnke.github.io/kommuner-clustering/outputs/map.html](https://mikkelbohnke.github.io/kommuner-clustering/outputs/map.html)

---

## What it does

Fetches six socioeconomic indicators for every Danish municipality from the DST StatBank API, standardises and clusters them with K-means (k=5), validates against hierarchical clustering, and generates a self-contained interactive choropleth. A GitHub Actions workflow rebuilds the map quarterly as DST publishes new data.

## Features used

| Feature | DST table | What it measures |
|---|---|---|
| Elderly share (65+) | `FOLK1A` | Ageing pressure |
| Youth share (0–17) | `FOLK1A` | Demand for schools/childcare |
| Unemployment rate | `AUP01` | Labour market health |
| Median disposable income | `INDKP101` | Household prosperity |
| Higher education share | `HFUDD11` | Human capital |
| Social housing share | `BOL101` | Housing structure |

## Repository structure

```
kommuner-clustering/
├── .github/
│   └── workflows/
│       └── refresh.yml          Quarterly cron: fetch, cluster, commit map
├── data/
│   └── municipalities.parquet   Cached DST data (committed, skip re-fetch)
├── outputs/
│   └── map.html                 Self-contained interactive choropleth
├── src/
│   ├── fetch.py                 DST API fetcher + DAWA GeoJSON
│   ├── cluster.py               K-means, hierarchical, silhouette analysis
│   ├── visualise.py             Plotly choropleth + radar chart
│   └── municipalities.py        Static lookup: DST code -> municipality name
├── main.py                      Full pipeline runner
├── test_pipeline.py             Smoke test with synthetic data (no API calls)
├── requirements.txt
└── README.md
```

## Running locally

```bash
pip install -r requirements.txt

# Full run (uses cached data/municipalities.parquet if present)
python main.py

# Force re-fetch from DST API
python main.py --no-cache

# Override cluster count
python main.py --k 6

# Smoke test without hitting any external API
python test_pipeline.py
```

## Deployment

GitHub Pages serves `outputs/map.html` as a static file. The quarterly GitHub Actions workflow re-fetches data, re-clusters, and commits the updated HTML automatically.

Enable GitHub Pages in repo Settings -> Pages -> Source: Deploy from branch `main`, folder `/outputs`.

## Data licence

Statistics Denmark data: CC 4.0 BY — source reference required.
Municipality boundaries: Dataforsyningen (DAWA), free for use with attribution.
