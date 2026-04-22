"""
Fetches socioeconomic data for all 98 Danish municipalities from the
Statistics Denmark (DST) StatBank API, and GeoJSON boundaries from DAWA.

DST API: https://api.statbank.dk/v1/data (POST, JSON body, CSV format)
DAWA:    https://api.dataforsyningen.dk/kommuner?format=geojson

Data licence: CC 4.0 BY -- source reference required.
Source: Statistics Denmark, www.dst.dk

Tables used
-----------
FOLK1A   Population by area, age -- derive elderly/youth share
AUP01    Gross unemployment rate by municipality
INDKP101 Income distribution -- derive median income
HFUDD11  Education by municipality -- derive higher education share
BOL101   Dwellings by ownership -- derive social housing share
"""

import json
import io
import requests
import pandas as pd
from pathlib import Path


DST_API  = "https://api.statbank.dk/v1/data"
DAWA_URL = "https://api.dataforsyningen.dk/kommuner?format=geojson"
DATA_DIR = Path("data")

HEADERS = {
    "Content-Type": "application/json; charset=utf-8",
    "User-Agent": "kommuner-clustering/1.0 (portfolio; github.com/iamboehnke)",
}

# Municipality OMRÅDE codes are 3-digit numeric strings in the DST API
# (e.g. "101" for Copenhagen, "461" for Odense). Codes starting with
# "0" or above "900" are regions/national aggregates -- filter these out.
REGION_PREFIXES = ("0", "08", "09", "10", "11", "12", "81", "82", "83", "84", "85")


# ---------------------------------------------------------------------------
# Core request function
# ---------------------------------------------------------------------------

def _dst_post(table: str, variables: list, lang: str = "en") -> pd.DataFrame:
    """
    POSTs a CSV data query to the DST StatBank API.

    Uses CSV format (semicolon-delimited) rather than JSON-stat because
    it is simpler to parse and less likely to break on edge cases.

    Logs the full request payload to stdout so failures are diagnosable
    in the GitHub Actions log.
    """
    payload = {
        "table":     table,
        "format":    "CSV",
        "lang":      lang,
        "variables": variables,
    }

    # Log the payload so we can debug 400 errors from the Actions log
    print(f"  [dst] POST {table}: {json.dumps(payload, ensure_ascii=False)[:300]}")

    try:
        response = requests.post(
            DST_API,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=HEADERS,
            timeout=30,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Network error fetching {table}: {e}") from e

    if not response.ok:
        raise RuntimeError(
            f"DST API returned {response.status_code} for {table}. "
            f"Body: {response.text[:300]}"
        )

    # Parse semicolon-delimited CSV response
    try:
        df = pd.read_csv(
            io.StringIO(response.text),
            sep=";",
            decimal=",",      # Danish decimal separator
            thousands=".",    # Danish thousands separator
            encoding="utf-8",
        )
    except Exception as e:
        raise RuntimeError(f"CSV parse failed for {table}: {e}") from e

    print(f"  [dst] {table}: {len(df)} rows, columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Municipality code helpers
# ---------------------------------------------------------------------------

def _is_municipality(code: str) -> bool:
    """
    Returns True if a OMRÅDE code string is a municipality (3-digit, 101-860).
    Filters out national total (000), region codes (0810-0860 etc.), and
    any other aggregates.
    """
    try:
        n = int(code.strip())
        return 100 <= n <= 900
    except (ValueError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Individual table fetchers
# ---------------------------------------------------------------------------

def fetch_population(year: str = "2024K1") -> pd.DataFrame:
    """
    Fetches FOLK1A: population by municipality and age group.

    Returns pct_elderly (65+) and pct_youth (0-17) per municipality.
    We fetch all ages (*) and compute shares in Python.
    """
    df = _dst_post("FOLK1A", [
        {"code": "OMRÅDE",     "values": ["*"]},
        {"code": "KØN",        "values": ["TOT"]},
        {"code": "ALDER",      "values": ["*"]},
        {"code": "CIVILSTAND", "values": ["TOT"]},
        {"code": "TID",        "values": [year]},
    ], lang="da")

    # Filter to municipalities only
    # OMRÅDE column contains text like "101 København" -- extract numeric code
    df = df.copy()
    df["kode"] = df.iloc[:, 0].astype(str).str.extract(r"^(\d+)").iloc[:, 0]
    df = df[df["kode"].apply(_is_municipality)].copy()

    # ALDER column: extract numeric age
    df["age_num"] = pd.to_numeric(
        df.iloc[:, 2].astype(str).str.extract(r"^(\d+)").iloc[:, 0],
        errors="coerce"
    )

    # Value column is the last column
    val_col = df.columns[-1]
    df["n"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0)

    result = (
        df.dropna(subset=["age_num"])
        .groupby("kode")
        .apply(lambda g: pd.Series({
            "pop_total":   g["n"].sum(),
            "pop_elderly": g.loc[g["age_num"] >= 65, "n"].sum(),
            "pop_youth":   g.loc[g["age_num"] <= 17, "n"].sum(),
        }), include_groups=False)
        .reset_index()
    )
    result = result[result["pop_total"] > 0].copy()
    result["pct_elderly"] = 100 * result["pop_elderly"] / result["pop_total"]
    result["pct_youth"]   = 100 * result["pop_youth"]   / result["pop_total"]
    return result.rename(columns={"kode": "OMRÅDE"})[
        ["OMRÅDE", "pop_total", "pct_elderly", "pct_youth"]
    ]


def fetch_unemployment(year: str = "2024") -> pd.DataFrame:
    """
    Fetches AUP01: gross unemployment rate by municipality (%).

    Fetches all PERPCT values and filters to the percentage rows.
    """
    df = _dst_post("AUP01", [
        {"code": "OMRÅDE", "values": ["*"]},
        {"code": "PERPCT", "values": ["*"]},
        {"code": "TID",    "values": [year]},
    ], lang="da")

    df = df.copy()
    df["kode"] = df.iloc[:, 0].astype(str).str.extract(r"^(\d+)").iloc[:, 0]
    df = df[df["kode"].apply(_is_municipality)].copy()

    # PERPCT column: keep rows labelled as percentage (pct / procent)
    perpct_col = df.columns[1]
    pct_mask = df[perpct_col].astype(str).str.lower().str.contains("pct|procent|ledighed")
    if pct_mask.sum() == 0:
        # Fallback: just take whatever the second variable value is
        pct_mask = df[perpct_col] == df[perpct_col].iloc[0]

    df_pct = df[pct_mask].copy()
    val_col = df_pct.columns[-1]
    df_pct["unemployment_rate"] = pd.to_numeric(df_pct[val_col], errors="coerce")

    return (
        df_pct.groupby("kode")["unemployment_rate"].mean()
        .reset_index()
        .rename(columns={"kode": "OMRÅDE"})
    )


def fetch_income(year: str = "2022") -> pd.DataFrame:
    """
    Fetches INDKP101: income distribution by municipality.

    Fetches all IFORGRUPP values and extracts the median (P50 decile).
    """
    df = _dst_post("INDKP101", [
        {"code": "OMRÅDE",    "values": ["*"]},
        {"code": "IFORGRUPP", "values": ["*"]},
        {"code": "ENHED",     "values": ["*"]},
        {"code": "TID",       "values": [year]},
    ], lang="da")

    df = df.copy()
    df["kode"] = df.iloc[:, 0].astype(str).str.extract(r"^(\d+)").iloc[:, 0]
    df = df[df["kode"].apply(_is_municipality)].copy()

    # Find the median / P50 row
    ifor_col = df.columns[1]
    enhed_col = df.columns[2]

    median_mask = (
        df[ifor_col].astype(str).str.upper().str.contains("P50|MEDIAN|50")
    )
    # Filter to kr. (amounts), not percentages
    kr_mask = df[enhed_col].astype(str).str.upper().str.contains("KR|KRONER")

    df_median = df[median_mask & kr_mask].copy()
    if len(df_median) == 0:
        # Fallback: try just the median rows without unit filter
        df_median = df[median_mask].copy()

    val_col = df_median.columns[-1]
    df_median["median_income"] = pd.to_numeric(df_median[val_col], errors="coerce")

    return (
        df_median.groupby("kode")["median_income"].first()
        .reset_index()
        .rename(columns={"kode": "OMRÅDE"})
    )


def fetch_education(year: str = "2023") -> pd.DataFrame:
    """
    Fetches HFUDD11: highest completed education by municipality.

    Returns share with higher education (long/medium cycle).
    Fetches all HFUDD values, identifies the higher-education codes, computes share.
    """
    df = _dst_post("HFUDD11", [
        {"code": "OMRÅDE", "values": ["*"]},
        {"code": "HFUDD",  "values": ["*"]},
        {"code": "ALDER",  "values": ["*"]},
        {"code": "TID",    "values": [year]},
    ], lang="da")

    df = df.copy()
    df["kode"] = df.iloc[:, 0].astype(str).str.extract(r"^(\d+)").iloc[:, 0]
    df = df[df["kode"].apply(_is_municipality)].copy()

    # HFUDD column: codes beginning with H6 or H7 are higher education
    hfudd_col = df.columns[1]
    higher_mask = df[hfudd_col].astype(str).str.upper().str.match(r"H[67]")

    val_col = df.columns[-1]
    df["n"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0)

    totals = df.groupby("kode")["n"].sum().reset_index(name="edu_total")
    higher = (
        df[higher_mask].groupby("kode")["n"].sum()
        .reset_index(name="edu_higher")
    )

    merged = totals.merge(higher, on="kode", how="left").fillna(0)
    merged = merged[merged["edu_total"] > 0].copy()
    merged["pct_higher_edu"] = 100 * merged["edu_higher"] / merged["edu_total"]
    return merged[["kode", "pct_higher_edu"]].rename(columns={"kode": "OMRÅDE"})


def fetch_housing(year: str = "2023") -> pd.DataFrame:
    """
    Fetches BOL101: dwellings by ownership type and municipality.

    Returns share of dwellings that are publicly rented (almene boliger).
    Fetches all EJERFORH values and identifies social housing rows.
    """
    df = _dst_post("BOL101", [
        {"code": "OMRÅDE",   "values": ["*"]},
        {"code": "EJERFORH", "values": ["*"]},
        {"code": "TID",      "values": [year]},
    ], lang="da")

    df = df.copy()
    df["kode"] = df.iloc[:, 0].astype(str).str.extract(r"^(\d+)").iloc[:, 0]
    df = df[df["kode"].apply(_is_municipality)].copy()

    ejer_col = df.columns[1]
    # Social/public housing: coded as "almene" or type 130 in DST
    social_mask = df[ejer_col].astype(str).str.lower().str.contains(
        "almen|alm\\.| 130|sociale|offentlig"
    )

    val_col = df.columns[-1]
    df["n"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0)

    totals = df.groupby("kode")["n"].sum().reset_index(name="dwellings_total")
    social = (
        df[social_mask].groupby("kode")["n"].sum()
        .reset_index(name="dwellings_social")
    )

    merged = totals.merge(social, on="kode", how="left").fillna(0)
    merged = merged[merged["dwellings_total"] > 0].copy()
    merged["pct_social_housing"] = 100 * merged["dwellings_social"] / merged["dwellings_total"]
    return merged[["kode", "pct_social_housing"]].rename(columns={"kode": "OMRÅDE"})


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------

def fetch_geodata() -> dict:
    """
    Fetches official Danish municipality boundaries from DAWA.
    The 'kode' property matches the numeric part of the DST OMRÅDE codes.
    """
    response = requests.get(DAWA_URL, timeout=30)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Master assembler
# ---------------------------------------------------------------------------

def build_dataset(cache: bool = True) -> pd.DataFrame:
    """
    Fetches all tables, joins them, and returns one row per municipality.
    """
    cache_path = DATA_DIR / "municipalities.parquet"
    DATA_DIR.mkdir(exist_ok=True)

    if cache and cache_path.exists():
        print("[fetch] Loading from cache...")
        return pd.read_parquet(cache_path)

    print("[fetch] Fetching from DST API...")

    steps = [
        ("Population",   fetch_population),
        ("Unemployment", fetch_unemployment),
        ("Income",       fetch_income),
        ("Education",    fetch_education),
        ("Housing",      fetch_housing),
    ]

    base = None
    for name, fn in steps:
        print(f"\n[fetch] {name}...")
        try:
            df = fn()
            print(f"  -> {len(df)} municipalities")
            base = df if base is None else base.merge(df, on="OMRÅDE", how="outer")
        except Exception as e:
            print(f"  WARNING: {name} failed: {e}")

    if base is None:
        raise RuntimeError("All DST fetches failed.")

    base["municipality_code"] = base["OMRÅDE"].astype(str).str.zfill(4)

    from src.municipalities import annotate
    base = annotate(base)

    # Drop aggregate rows (national total, regions)
    base = base[base["OMRÅDE"].apply(_is_municipality)].copy()
    base = base[base["municipality_name"] != "Unknown"].copy()

    base.to_parquet(cache_path, index=False)
    print(f"\n[fetch] Dataset saved: {len(base)} municipalities, {len(base.columns)} columns")
    return base
