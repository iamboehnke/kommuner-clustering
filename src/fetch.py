"""
Fetches socioeconomic data for all 98 Danish municipalities from the
Statistics Denmark (DST) StatBank API, and GeoJSON boundaries from DAWA.

DST API: https://api.statbank.dk/v1/data (POST, JSON body, CSV format)
DAWA:    https://api.dataforsyningen.dk/kommuner?format=geojson

Data licence: CC 4.0 BY -- source reference required.
Source: Statistics Denmark, www.dst.dk

Tables and confirmed variable codes (verified from live API responses)
-----------------------------------------------------------------------
FOLK1A      Population: OMRÅDE, KØN, ALDER, CIVILSTAND, TID -> INDHOLD
AULAAR      Net unemployment: OMRÅDE, KØN, PERPCT, TID -> INDHOLD
            (AUP01 does NOT have PERPCT -- use AULAAR instead)
INDKP101    Income distribution: OMRÅDE, IFORGRUPP, INDKOMSTTYPE, ENHED, TID -> INDHOLD
            (INDKOMSTTYPE is a required non-eliminatable variable)
HFUDD11     Education: BOPKOMMUNEKODE, HFUDD, ALDER, TID -> INDHOLD
            (uses BOPKOMMUNEKODE not OMRÅDE for municipality)
BOL101      Housing: OMRÅDE, EJERFORH, BEBO, TID -> INDHOLD
            (BEBO is a required non-eliminatable variable)
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


def _is_municipality(code: str) -> bool:
    """
    Returns True for 3-digit numeric municipality codes (101-860).
    Filters out national totals, regions, and other aggregates.
    """
    try:
        n = int(str(code).strip())
        return 100 <= n <= 900
    except (ValueError, TypeError):
        return False


def _extract_kode(series: pd.Series) -> pd.Series:
    """
    Extracts the numeric code from DST OMRÅDE text like '101 København' -> '101'.
    Also handles plain numeric strings.
    """
    return series.astype(str).str.extract(r"^(\d+)", expand=False).str.strip()


# ---------------------------------------------------------------------------
# Core request function
# ---------------------------------------------------------------------------

def _dst_post(table: str, variables: list, lang: str = "da") -> pd.DataFrame:
    """
    POSTs a CSV data query to the DST StatBank API.

    Uses CSV format (semicolon-delimited). Logs the full payload so that
    any future 400 errors are immediately diagnosable in the Actions log.
    """
    payload = {
        "table":     table,
        "format":    "CSV",
        "lang":      lang,
        "variables": variables,
    }

    payload_str = json.dumps(payload, ensure_ascii=False)
    print(f"  [dst] POST {table}: {payload_str[:400]}")

    try:
        response = requests.post(
            DST_API,
            data=payload_str.encode("utf-8"),
            headers=HEADERS,
            timeout=30,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Network error fetching {table}: {e}") from e

    if not response.ok:
        raise RuntimeError(
            f"DST API returned {response.status_code} for {table}. "
            f"Body: {response.text[:400]}"
        )

    try:
        df = pd.read_csv(
            io.StringIO(response.text),
            sep=";",
            decimal=",",
            thousands=".",
            encoding="utf-8",
        )
    except Exception as e:
        raise RuntimeError(f"CSV parse failed for {table}: {e}") from e

    print(f"  [dst] {table}: {len(df)} rows, columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Individual table fetchers
# ---------------------------------------------------------------------------

def fetch_population(year: str = "2024K1") -> pd.DataFrame:
    """
    FOLK1A: Population by municipality and age.
    Variables confirmed: OMRÅDE, KØN, ALDER, CIVILSTAND, TID -> INDHOLD

    Returns pct_elderly (65+) and pct_youth (0-17) per municipality.
    """
    df = _dst_post("FOLK1A", [
        {"code": "OMRÅDE",     "values": ["*"]},
        {"code": "KØN",        "values": ["TOT"]},
        {"code": "ALDER",      "values": ["*"]},
        {"code": "CIVILSTAND", "values": ["TOT"]},
        {"code": "TID",        "values": [year]},
    ])

    df = df.copy()
    df["kode"] = _extract_kode(df["OMRÅDE"])
    df = df[df["kode"].apply(_is_municipality)].copy()

    # ALDER values look like "0 0 år", "65 65-69 år", "100 100+ år"
    # Extract the leading integer
    df["age_num"] = pd.to_numeric(
        df["ALDER"].astype(str).str.extract(r"^(\d+)", expand=False),
        errors="coerce"
    )
    df["n"] = pd.to_numeric(df["INDHOLD"], errors="coerce").fillna(0)
    df = df.dropna(subset=["age_num"]).copy()

    # Build totals using explicit filtering (avoids groupby apply issues)
    totals  = df.groupby("kode")["n"].sum().rename("pop_total")
    elderly = df[df["age_num"] >= 65].groupby("kode")["n"].sum().rename("pop_elderly")
    youth   = df[df["age_num"] <= 17].groupby("kode")["n"].sum().rename("pop_youth")

    result = pd.concat([totals, elderly, youth], axis=1).fillna(0).reset_index()
    result = result[result["pop_total"] > 0].copy()
    result["pct_elderly"] = 100 * result["pop_elderly"] / result["pop_total"]
    result["pct_youth"]   = 100 * result["pop_youth"]   / result["pop_total"]
    return result.rename(columns={"kode": "OMRÅDE"})[
        ["OMRÅDE", "pop_total", "pct_elderly", "pct_youth"]
    ]


def fetch_unemployment(year: str = "2023") -> pd.DataFrame:
    """
    AULAAR: Net full-time unemployment by municipality.
    Variables: OMRÅDE, KØN, PERPCT, TID -> INDHOLD

    Note: AUP01 does NOT have the PERPCT variable -- AULAAR does.
    We fetch all PERPCT values and keep the percentage rows.
    """
    df = _dst_post("AULAAR", [
        {"code": "OMRÅDE", "values": ["*"]},
        {"code": "KØN",    "values": ["TOT"]},
        {"code": "PERPCT", "values": ["*"]},
        {"code": "TID",    "values": [year]},
    ])

    df = df.copy()
    df["kode"] = _extract_kode(df["OMRÅDE"])
    df = df[df["kode"].apply(_is_municipality)].copy()

    # Keep rows where PERPCT column indicates a rate/percentage
    perpct_col = "PERPCT"
    pct_mask = df[perpct_col].astype(str).str.lower().str.contains(
        "pct|procent|ledighed|rate", na=False
    )
    df_pct = df[pct_mask].copy() if pct_mask.any() else df.copy()

    df_pct["unemployment_rate"] = pd.to_numeric(df_pct["INDHOLD"], errors="coerce")
    return (
        df_pct.groupby("kode")["unemployment_rate"].mean()
        .reset_index()
        .rename(columns={"kode": "OMRÅDE"})
    )


def fetch_income(year: str = "2022") -> pd.DataFrame:
    """
    INDKP101: Income distribution by municipality.
    Variables: OMRÅDE, IFORGRUPP, INDKOMSTTYPE, ENHED, TID -> INDHOLD
    (INDKOMSTTYPE is required and non-eliminatable)

    Returns median disposable income per municipality.
    """
    df = _dst_post("INDKP101", [
        {"code": "OMRÅDE",       "values": ["*"]},
        {"code": "IFORGRUPP",    "values": ["*"]},
        {"code": "INDKOMSTTYPE", "values": ["*"]},
        {"code": "ENHED",        "values": ["*"]},
        {"code": "TID",          "values": [year]},
    ])

    df = df.copy()
    df["kode"] = _extract_kode(df["OMRÅDE"])
    df = df[df["kode"].apply(_is_municipality)].copy()

    # Keep rows with median / P50 income group
    ifor_col = "IFORGRUPP"
    median_mask = df[ifor_col].astype(str).str.upper().str.contains(
        "P50|MEDIAN|50", na=False
    )

    # Keep rows with kr. amounts (not percentages)
    enhed_col = "ENHED"
    kr_mask = df[enhed_col].astype(str).str.upper().str.contains(
        "KR|KRONER", na=False
    )

    df_m = df[median_mask & kr_mask].copy()
    if len(df_m) == 0:
        df_m = df[median_mask].copy()

    df_m["median_income"] = pd.to_numeric(df_m["INDHOLD"], errors="coerce")
    return (
        df_m.groupby("kode")["median_income"].mean()
        .reset_index()
        .rename(columns={"kode": "OMRÅDE"})
    )


def fetch_education(year: str = "2023") -> pd.DataFrame:
    """
    HFUDD11: Highest completed education by municipality of residence.
    Variables: BOPKOMMUNEKODE, HFUDD, ALDER, TID -> INDHOLD
    (uses BOPKOMMUNEKODE, NOT OMRÅDE -- education tables differ)

    Returns share with higher education (H6x/H7x codes).
    """
    df = _dst_post("HFUDD11", [
        {"code": "BOPKOMMUNEKODE", "values": ["*"]},
        {"code": "HFUDD",          "values": ["*"]},
        {"code": "ALDER",          "values": ["*"]},
        {"code": "TID",            "values": [year]},
    ])

    df = df.copy()
    # Extract municipality code from BOPKOMMUNEKODE column
    kode_col = "BOPKOMMUNEKODE"
    df["kode"] = _extract_kode(df[kode_col])
    df = df[df["kode"].apply(_is_municipality)].copy()

    # Higher education: HFUDD codes starting with H6 or H7
    hfudd_col = "HFUDD"
    higher_mask = df[hfudd_col].astype(str).str.upper().str.match(r"H[67]")

    df["n"] = pd.to_numeric(df["INDHOLD"], errors="coerce").fillna(0)

    totals = df.groupby("kode")["n"].sum().rename("edu_total")
    higher = df[higher_mask].groupby("kode")["n"].sum().rename("edu_higher")

    result = pd.concat([totals, higher], axis=1).fillna(0).reset_index()
    result = result[result["edu_total"] > 0].copy()
    result["pct_higher_edu"] = 100 * result["edu_higher"] / result["edu_total"]
    return result[["kode", "pct_higher_edu"]].rename(columns={"kode": "OMRÅDE"})


def fetch_housing(year: str = "2023") -> pd.DataFrame:
    """
    BOL101: Dwellings by ownership type and municipality.
    Variables: OMRÅDE, EJERFORH, BEBO, TID -> INDHOLD
    (BEBO is required and non-eliminatable)

    Returns share of dwellings that are social/public housing.
    """
    df = _dst_post("BOL101", [
        {"code": "OMRÅDE",   "values": ["*"]},
        {"code": "EJERFORH", "values": ["*"]},
        {"code": "BEBO",     "values": ["*"]},
        {"code": "TID",      "values": [year]},
    ])

    df = df.copy()
    df["kode"] = _extract_kode(df["OMRÅDE"])
    df = df[df["kode"].apply(_is_municipality)].copy()

    ejer_col = "EJERFORH"
    # Social/public housing (almene boliger) -- code typically contains "alm"
    social_mask = df[ejer_col].astype(str).str.lower().str.contains(
        "almen|alm\\.|130|sociale|offentlig", na=False
    )

    df["n"] = pd.to_numeric(df["INDHOLD"], errors="coerce").fillna(0)

    totals = df.groupby("kode")["n"].sum().rename("dwellings_total")
    social = df[social_mask].groupby("kode")["n"].sum().rename("dwellings_social")

    result = pd.concat([totals, social], axis=1).fillna(0).reset_index()
    result = result[result["dwellings_total"] > 0].copy()
    result["pct_social_housing"] = 100 * result["dwellings_social"] / result["dwellings_total"]
    return result[["kode", "pct_social_housing"]].rename(columns={"kode": "OMRÅDE"})


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------

def fetch_geodata() -> dict:
    """
    Fetches official Danish municipality boundaries from DAWA.
    The 'kode' property in each feature matches the DST municipality codes.
    """
    response = requests.get(DAWA_URL, timeout=30)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Master assembler
# ---------------------------------------------------------------------------

def build_dataset(cache: bool = True) -> pd.DataFrame:
    """
    Fetches all tables, joins them by OMRÅDE code, and returns one row
    per municipality. Caches to data/municipalities.parquet.
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
    failed = []
    for name, fn in steps:
        print(f"\n[fetch] {name}...")
        try:
            df = fn()
            print(f"  -> {len(df)} municipalities")
            base = df if base is None else base.merge(df, on="OMRÅDE", how="outer")
        except Exception as e:
            print(f"  WARNING: {name} failed: {e}")
            failed.append(name)

    if base is None:
        raise RuntimeError("All DST fetches failed.")

    if failed:
        print(f"\n[fetch] Partial success. Failed: {failed}")

    base["municipality_code"] = base["OMRÅDE"].astype(str).str.zfill(4)

    from src.municipalities import annotate
    base = annotate(base)

    base = base[base["OMRÅDE"].apply(_is_municipality)].copy()
    base = base[base["municipality_name"] != "Unknown"].copy()

    base.to_parquet(cache_path, index=False)
    print(f"\n[fetch] Saved: {len(base)} municipalities, {len(base.columns)} columns")
    return base
