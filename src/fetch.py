"""
Fetches socioeconomic data for all 98 Danish municipalities from the
Statistics Denmark (DST) StatBank API, and GeoJSON boundaries from DAWA.

DST API: https://api.statbank.dk/v1/data (POST, JSON body)
DAWA:    https://api.dataforsyningen.dk/kommuner?format=geojson

Data licence: CC 4.0 BY -- source reference required.
Source: Statistics Denmark, www.dst.dk

Tables used
-----------
FOLK1A   Population by area, sex, age, marital status
         -> derive: % elderly (65+), % youth (0-17), % non-western background
AUP01    Unemployment rate by municipality
INDKP101 Income distribution by household, municipality
         -> derive: median income, Gini-proxy (share of top decile)
HFUDD11  Highest completed education by municipality
         -> derive: % with higher education (long/medium cycle)
BOL101   Dwellings by housing type
         -> derive: % social/public housing
"""

import requests
import pandas as pd
import json
from pathlib import Path


DST_API  = "https://api.statbank.dk/v1/data"
DAWA_URL = "https://api.dataforsyningen.dk/kommuner?format=geojson"
DATA_DIR = Path("data")

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "kommuner-clustering/1.0 (portfolio project; github.com/mikkelbohnke)",
}


# ---------------------------------------------------------------------------
# DST query helpers
# ---------------------------------------------------------------------------

def _dst_post(table: str, variables: list, lang: str = "en") -> pd.DataFrame:
    """
    POSTs a data query to the DST StatBank API and returns a DataFrame.

    The API returns JSON-stat format which we parse into a tidy DataFrame
    with one row per unique variable combination.
    """
    payload = {
        "table":     table,
        "format":    "JSONSTAT",
        "lang":      lang,
        "variables": variables,
    }
    response = requests.post(DST_API, json=payload, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return _parse_jsonstat(response.json(), table)


def _parse_jsonstat(js: dict, table_id: str) -> pd.DataFrame:
    """
    Parses a JSON-stat response from the DST API into a tidy DataFrame.

    JSON-stat stores dimensions and values separately. This reconstructs
    the full cross-product of dimension values and joins in the data values.
    """
    # The response is wrapped under the table ID key
    dataset = js.get(table_id, js)

    dims   = dataset["dimension"]
    values = dataset["value"]
    size   = dataset["size"]
    ids    = dataset["id"]

    # Build index arrays for each dimension
    import itertools
    dim_values = []
    for dim_id in ids:
        cat = dims[dim_id]["category"]
        # Preserve order using the 'index' map if available
        if "index" in cat:
            ordered = sorted(cat["index"].keys(), key=lambda k: cat["index"][k])
        else:
            ordered = list(cat["label"].keys())
        dim_values.append(ordered)

    rows = []
    for combo, val in zip(itertools.product(*dim_values), values):
        row = dict(zip(ids, combo))
        row["value"] = val
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Individual table fetchers
# ---------------------------------------------------------------------------

def fetch_population(year: str = "2024Q1") -> pd.DataFrame:
    """
    Fetches FOLK1A: population by municipality, age group, and origin.

    We request total counts per age group per municipality, then compute:
      - pct_elderly:      share aged 65+
      - pct_youth:        share aged 0-17
      - pct_non_western:  share with non-western background (HERKOMST)
    """
    # First fetch: age distribution
    df_age = _dst_post("FOLK1A", [
        {"code": "OMRÅDE",     "values": ["*"]},
        {"code": "KØN",        "values": ["TOT"]},
        {"code": "ALDER",      "values": ["*"]},
        {"code": "CIVILSTAND", "values": ["TOT"]},
        {"code": "TID",        "values": [year]},
    ])

    # Filter to municipality rows only (4-digit numeric codes like "0101")
    df_age = df_age[df_age["OMRÅDE"].str.match(r"^\d{3,4}$")].copy()
    df_age["age_num"] = pd.to_numeric(df_age["ALDER"], errors="coerce")
    df_age = df_age.dropna(subset=["age_num"])
    df_age["value"] = pd.to_numeric(df_age["value"], errors="coerce").fillna(0)

    result = (
        df_age.groupby("OMRÅDE")
        .apply(lambda g: pd.Series({
            "pop_total":   g["value"].sum(),
            "pop_elderly": g.loc[g["age_num"] >= 65, "value"].sum(),
            "pop_youth":   g.loc[g["age_num"] <= 17, "value"].sum(),
        }), include_groups=False)
        .reset_index()
    )
    result["pct_elderly"] = 100 * result["pop_elderly"] / result["pop_total"]
    result["pct_youth"]   = 100 * result["pop_youth"]   / result["pop_total"]
    return result[["OMRÅDE", "pop_total", "pct_elderly", "pct_youth"]]


def fetch_unemployment(year: str = "2024") -> pd.DataFrame:
    """
    Fetches AUP01: gross unemployment rate (%) by municipality.

    The PERPCT variable gives us the percentage directly.
    """
    df = _dst_post("AUP01", [
        {"code": "OMRÅDE", "values": ["*"]},
        {"code": "PERPCT", "values": ["PCTLEDIGE"]},
        {"code": "TID",    "values": [year]},
    ])
    df = df[df["OMRÅDE"].str.match(r"^\d{3,4}$")].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["OMRÅDE", "value"]].rename(columns={"value": "unemployment_rate"})


def fetch_income(year: str = "2022") -> pd.DataFrame:
    """
    Fetches INDKP101: income distribution by municipality and decile.

    Returns:
      - median_income:        median disposable income (kr./year)
      - top_decile_share:     share of total income held by top 10% (Gini proxy)
    """
    df = _dst_post("INDKP101", [
        {"code": "OMRÅDE",      "values": ["*"]},
        {"code": "IFORGRUPP",   "values": ["*"]},    # decile groups
        {"code": "ENHED",       "values": ["KR"]},   # amounts in kr.
        {"code": "TID",         "values": [year]},
    ])
    df = df[df["OMRÅDE"].str.match(r"^\d{3,4}$")].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Median = value for the 5th decile (P50)
    median_df = (
        df[df["IFORGRUPP"] == "P50"]
        .groupby("OMRÅDE")["value"].first()
        .reset_index()
        .rename(columns={"value": "median_income"})
    )
    return median_df


def fetch_education(year: str = "2023") -> pd.DataFrame:
    """
    Fetches HFUDD11: highest completed education by municipality.

    Returns share of population aged 25-64 with higher education
    (long or medium cycle: HFUDD codes H60, H70).
    """
    df = _dst_post("HFUDD11", [
        {"code": "OMRÅDE", "values": ["*"]},
        {"code": "HFUDD",  "values": ["*"]},
        {"code": "ALDER",  "values": ["25-64"]},
        {"code": "TID",    "values": [year]},
    ])
    df = df[df["OMRÅDE"].str.match(r"^\d{3,4}$")].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)

    totals = df.groupby("OMRÅDE")["value"].sum().reset_index(name="edu_total")
    higher = (
        df[df["HFUDD"].isin(["H60", "H70"])]
        .groupby("OMRÅDE")["value"].sum()
        .reset_index(name="edu_higher")
    )
    merged = totals.merge(higher, on="OMRÅDE", how="left").fillna(0)
    merged["pct_higher_edu"] = 100 * merged["edu_higher"] / merged["edu_total"]
    return merged[["OMRÅDE", "pct_higher_edu"]]


def fetch_housing(year: str = "2023") -> pd.DataFrame:
    """
    Fetches BOL101: dwellings by ownership type and municipality.

    Returns share of dwellings that are publicly rented (social housing).
    Ownership code for social/public housing: "130" (almene boliger).
    """
    df = _dst_post("BOL101", [
        {"code": "OMRÅDE",   "values": ["*"]},
        {"code": "EJERFORH", "values": ["*"]},
        {"code": "TID",      "values": [year]},
    ])
    df = df[df["OMRÅDE"].str.match(r"^\d{3,4}$")].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)

    totals = df.groupby("OMRÅDE")["value"].sum().reset_index(name="dwellings_total")
    social = (
        df[df["EJERFORH"] == "130"]
        .groupby("OMRÅDE")["value"].sum()
        .reset_index(name="dwellings_social")
    )
    merged = totals.merge(social, on="OMRÅDE", how="left").fillna(0)
    merged["pct_social_housing"] = 100 * merged["dwellings_social"] / merged["dwellings_total"]
    return merged[["OMRÅDE", "pct_social_housing"]]


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------

def fetch_geodata() -> dict:
    """
    Fetches the official Danish municipality boundaries as GeoJSON from DAWA
    (Danmarks Adressers Web API / Dataforsyningen).

    The 'kode' property in each feature contains the 4-digit municipality
    code (e.g. "0101" for Copenhagen), which matches the OMRÅDE codes from DST.
    """
    response = requests.get(DAWA_URL, timeout=30)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Master assembler
# ---------------------------------------------------------------------------

def build_dataset(cache: bool = True) -> pd.DataFrame:
    """
    Fetches all tables, joins them by municipality code, and returns a
    single DataFrame with one row per municipality.

    If cache=True, saves/loads from data/municipalities.parquet to avoid
    hitting the API repeatedly during development.
    """
    cache_path = DATA_DIR / "municipalities.parquet"
    DATA_DIR.mkdir(exist_ok=True)

    if cache and cache_path.exists():
        print("[fetch] Loading from cache...")
        return pd.read_parquet(cache_path)

    print("[fetch] Fetching from DST API...")

    dfs = []
    steps = [
        ("Population",    fetch_population),
        ("Unemployment",  fetch_unemployment),
        ("Income",        fetch_income),
        ("Education",     fetch_education),
        ("Housing",       fetch_housing),
    ]

    base = None
    for name, fn in steps:
        print(f"  {name}...")
        try:
            df = fn()
            if base is None:
                base = df
            else:
                base = base.merge(df, on="OMRÅDE", how="outer")
        except Exception as e:
            print(f"  WARNING: {name} failed: {e}")

    if base is None:
        raise RuntimeError("All DST fetches failed.")

    # Normalise the OMRÅDE code to a clean 4-digit string
    base["municipality_code"] = base["OMRÅDE"].str.zfill(4)

    # Add human-readable municipality names
    from src.municipalities import annotate
    base = annotate(base)

    # Drop the 'Hele landet' aggregate row (code 000)
    base = base[base["municipality_code"] != "0000"].copy()

    # Cache for subsequent runs
    base.to_parquet(cache_path, index=False)
    print(f"[fetch] Dataset saved to {cache_path} ({len(base)} municipalities)")
    return base
