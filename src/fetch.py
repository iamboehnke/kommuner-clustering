"""
Fetches socioeconomic data for all 98 Danish municipalities from the
Statistics Denmark (DST) StatBank API, and GeoJSON boundaries from DAWA.

Variable codes confirmed from live API metadata (2026-04-22):

  FOLK1A:   OMRÅDE, KØN (values: TOT/1/2), ALDER, CIVILSTAND, Tid  -> INDHOLD
  INDKP101: OMRÅDE, INDKOMSTTYPE (required), ENHED (required),
            KOEN (values: MOK/M/K), Tid                            -> INDHOLD
  HFUDD11:  BOPOMR (municipality), HERKOMST, HFUDD, ALDER, KØN, Tid -> INDHOLD
  BOL101:   OMRÅDE, BEBO (required), ANVENDELSE, UDLFORH, EJER,
            OPFØRELSESÅR, Tid                                       -> INDHOLD

Unemployment: AULAAR has no municipality dimension. Use AUP01 metadata
to find the correct table. Clustering runs on 4 variables if all else fails.
"""

import json
import io
import requests
import pandas as pd
from pathlib import Path
from src.municipalities import MUNICIPALITY_NAMES


DST_API      = "https://api.statbank.dk/v1/data"
DST_INFO_API = "https://api.statbank.dk/v1/tableinfo"
DAWA_URL     = "https://api.dataforsyningen.dk/kommuner?format=geojson"
DATA_DIR     = Path("data")

HEADERS = {
    "Content-Type": "application/json; charset=utf-8",
    "User-Agent": "kommuner-clustering/1.0 (portfolio; github.com/iamboehnke)",
}

VALID_CODES = set(MUNICIPALITY_NAMES.keys())  # {"101", "147", ...}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_codes() -> list[str]:
    return list(MUNICIPALITY_NAMES.keys())


def print_table_info(table: str) -> None:
    """Prints live variable codes for a table to the Actions log."""
    try:
        r = requests.get(
            f"{DST_INFO_API}/{table}",
            params={"lang": "da", "format": "JSON"},
            timeout=15,
        )
        if not r.ok:
            print(f"  [meta] {table}: HTTP {r.status_code}")
            return
        info = r.json()
        print(f"  [meta] {table}: {info.get('text', '')}")
        for var in info.get("variables", []):
            vals = [v["id"] for v in var.get("values", [])[:5]]
            req  = "(REQUIRED)" if not var.get("elimination") else "(optional)"
            print(f"    '{var['id']}' {req}: e.g. {vals}")
    except Exception as e:
        print(f"  [meta] {table}: {e}")


def _dst_post(table: str, variables: list) -> pd.DataFrame:
    """POST to DST CSV API. Logs payload for diagnostics."""
    payload = {"table": table, "format": "CSV", "lang": "da", "variables": variables}
    payload_str = json.dumps(payload, ensure_ascii=False)
    print(f"  [dst] POST {table}: {payload_str[:400]}")

    try:
        r = requests.post(
            DST_API,
            data=payload_str.encode("utf-8"),
            headers=HEADERS,
            timeout=45,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Network: {e}") from e

    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")

    # Read as all-string first, then convert INDHOLD manually.
    # This avoids any ambiguity with Danish thousands/decimal separators.
    df = pd.read_csv(io.StringIO(r.text), sep=";", dtype=str, encoding="utf-8")
    print(f"  [dst] {table}: {len(df)} rows, cols: {list(df.columns)}")
    return df


def _to_numeric(series: pd.Series) -> pd.Series:
    """
    Converts a DST INDHOLD column to float.
    DST uses '.' as thousands separator and ',' as decimal separator.
    Suppressed cells contain '..' which coerces to NaN then 0.
    """
    return (
        pd.to_numeric(
            series.astype(str)
                  .str.strip()
                  .str.replace(".", "", regex=False)   # strip thousands sep
                  .str.replace(",", ".", regex=False),  # decimal sep -> dot
            errors="coerce"
        ).fillna(0)
    )


def _kode(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Extracts the numeric municipality code from a DST area column.
    Values may be bare codes ('101') or labelled ('101 København').
    Returns a Series of stripped code strings.
    """
    return df[col].astype(str).str.strip().str.extract(r"(\d+)", expand=False)


# ---------------------------------------------------------------------------
# Table fetchers
# ---------------------------------------------------------------------------

def fetch_population(year: str = "2024K1") -> pd.DataFrame:
    """
    FOLK1A -- population by municipality and single-year age.
    Confirmed variables: OMRÅDE, KØN, ALDER, CIVILSTAND, Tid -> INDHOLD
    """
    codes = _get_codes()
    df = _dst_post("FOLK1A", [
        {"code": "OMRÅDE",     "values": codes},
        {"code": "KØN",        "values": ["TOT"]},
        {"code": "ALDER",      "values": ["*"]},
        {"code": "CIVILSTAND", "values": ["TOT"]},
        {"code": "TID",        "values": [year]},
    ])

    df["kode"]    = _kode(df, "OMRÅDE")
    df["age_num"] = pd.to_numeric(
        df["ALDER"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce"
    )
    df["n"] = _to_numeric(df["INDHOLD"])

    # Debug: show a sample so any remaining issues are visible in the log
    print(f"  DEBUG FOLK1A sample -- OMRÅDE: {df['OMRÅDE'].head(3).tolist()}")
    print(f"  DEBUG FOLK1A kode: {df['kode'].head(3).tolist()}")
    print(f"  DEBUG FOLK1A n: {df['n'].head(3).tolist()}")

    df = df.dropna(subset=["age_num"]).copy()
    df = df[df["kode"].isin(VALID_CODES)].copy()
    print(f"  After filtering: {len(df)} rows across {df['kode'].nunique()} municipalities")

    if df.empty:
        raise RuntimeError(
            "FOLK1A: 0 rows after filtering. "
            f"OMRÅDE sample: {df['OMRÅDE'].head(5).tolist()}, "
            f"kode sample: {df['kode'].head(5).tolist()}"
        )

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


def fetch_unemployment() -> pd.DataFrame:
    """
    Municipality-level unemployment.
    AULAAR has no municipality dimension -- we try AUP01 and read its metadata.
    If that fails, we skip unemployment gracefully.
    """
    codes = _get_codes()
    print_table_info("AUP01")

    # Try with the first variable code listed in the AUP01 metadata
    # The actual codes will be visible in the log above
    df = _dst_post("AUP01", [
        {"code": "OMRÅDE", "values": codes},
        {"code": "TID",    "values": ["2023"]},
    ])

    df["kode"] = _kode(df, "OMRÅDE")
    df = df[df["kode"].isin(VALID_CODES)].copy()
    val_col = "INDHOLD" if "INDHOLD" in df.columns else df.columns[-1]
    df["n"] = _to_numeric(df[val_col])

    print(f"  AUP01 col sample: {list(df.columns)}")
    print(f"  AUP01 first rows:\n{df.head(3).to_string()}")

    # Take the mean across all non-OMRÅDE/TID columns per municipality
    return (
        df.groupby("kode")["n"].mean()
        .reset_index()
        .rename(columns={"kode": "OMRÅDE", "n": "unemployment_rate"})
    )


def fetch_income(year: str = "2022") -> pd.DataFrame:
    """
    INDKP101 -- income by municipality.
    Confirmed variables: OMRÅDE, INDKOMSTTYPE (req), ENHED (req), KOEN, Tid -> INDHOLD
    ENHED values: 101=antal, 110=gns, 116=median kr.
    KOEN values: MOK=total, M=mænd, K=kvinder
    """
    codes = _get_codes()
    df = _dst_post("INDKP101", [
        {"code": "OMRÅDE",       "values": codes},
        {"code": "INDKOMSTTYPE", "values": ["*"]},
        {"code": "ENHED",        "values": ["*"]},
        {"code": "KOEN",         "values": ["MOK"]},
        {"code": "TID",          "values": [year]},
    ])

    print(f"  INDKP101 ENHED sample: {df['ENHED'].unique()[:6].tolist()}")
    print(f"  INDKP101 INDKOMSTTYPE sample: {df['INDKOMSTTYPE'].unique()[:6].tolist()}")

    df["kode"] = _kode(df, "OMRÅDE")
    df = df[df["kode"].isin(VALID_CODES)].copy()
    df["n"] = _to_numeric(df["INDHOLD"])

    # ENHED 116 = median income in kr.; fall back to 110 (mean) if not present
    enhed_col = "ENHED"
    median_mask = df[enhed_col].astype(str).str.contains("116", na=False)
    if not median_mask.any():
        median_mask = df[enhed_col].astype(str).str.contains("110", na=False)
    if not median_mask.any():
        median_mask = pd.Series([True] * len(df), index=df.index)

    df_m = df[median_mask].copy()
    return (
        df_m.groupby("kode")["n"].mean()
        .reset_index()
        .rename(columns={"kode": "OMRÅDE", "n": "median_income"})
    )


def fetch_education(year: str = "2023") -> pd.DataFrame:
    """
    HFUDD11 -- education by municipality of residence.
    Confirmed municipality variable: BOPOMR (not KOMKODE, not BOPKOMMUNEKODE)
    HFUDD codes H6x/H7x = higher education (medium/long cycle)
    """
    codes = _get_codes()
    df = _dst_post("HFUDD11", [
        {"code": "BOPOMR",    "values": codes},
        {"code": "HFUDD",     "values": ["*"]},
        {"code": "HERKOMST",  "values": ["TOT"]},
        {"code": "ALDER",     "values": ["*"]},
        {"code": "KØN",       "values": ["TOT"]},
        {"code": "TID",       "values": [year]},
    ])

    # HFUDD11 uses BOPOMR for municipality
    omr_col = "BOPOMR" if "BOPOMR" in df.columns else df.columns[0]
    df["kode"] = _kode(df, omr_col)
    df = df[df["kode"].isin(VALID_CODES)].copy()
    df["n"] = _to_numeric(df["INDHOLD"])

    hfudd_col = "HFUDD" if "HFUDD" in df.columns else df.columns[1]
    higher_mask = df[hfudd_col].astype(str).str.match(r"H[67]")

    totals = df.groupby("kode")["n"].sum().rename("edu_total")
    higher = df[higher_mask].groupby("kode")["n"].sum().rename("edu_higher")

    result = pd.concat([totals, higher], axis=1).fillna(0).reset_index()
    result = result[result["edu_total"] > 0].copy()
    result["pct_higher_edu"] = 100 * result["edu_higher"] / result["edu_total"]
    return result[["kode", "pct_higher_edu"]].rename(columns={"kode": "OMRÅDE"})


def fetch_housing(year: str = "2023") -> pd.DataFrame:
    """
    BOL101 -- dwellings by ownership and municipality.
    Confirmed variables: OMRÅDE, BEBO (req), ANVENDELSE, UDLFORH, EJER, OPFØRELSESÅR, Tid
    EJER values: 10/20/30/41/SK/UOP2 -- '30' is typically almene boliger
    """
    codes = _get_codes()
    df = _dst_post("BOL101", [
        {"code": "OMRÅDE",      "values": codes},
        {"code": "BEBO",        "values": ["*"]},
        {"code": "EJER",        "values": ["*"]},
        {"code": "TID",         "values": [year]},
    ])

    print(f"  BOL101 EJER sample: {df['EJER'].unique()[:8].tolist()}")

    df["kode"] = _kode(df, "OMRÅDE")
    df = df[df["kode"].isin(VALID_CODES)].copy()
    df["n"] = _to_numeric(df["INDHOLD"])

    ejer_col = "EJER" if "EJER" in df.columns else df.columns[1]
    # Social/almene boliger: EJER='30' or any label containing 'alm'
    social_mask = (
        df[ejer_col].astype(str).str.strip().eq("30") |
        df[ejer_col].astype(str).str.lower().str.contains("almen|alm\\.", na=False)
    )
    print(f"  BOL101 social rows: {social_mask.sum()} of {len(df)}")

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
    """Fetches official Danish municipality boundaries from DAWA."""
    r = requests.get(DAWA_URL, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Master assembler
# ---------------------------------------------------------------------------

def build_dataset(cache: bool = True) -> pd.DataFrame:
    """
    Fetches all tables, joins by OMRÅDE, returns one row per municipality.
    Partial data (some tables failing) is handled gracefully -- clustering
    runs on whichever variables successfully loaded.
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

    if base is None or len(base) == 0:
        raise RuntimeError(
            f"No data loaded at all. Failed: {failed}. "
            "Check [meta] and DEBUG lines above."
        )
    if failed:
        print(f"\n[fetch] Partial data. Failed: {failed}. Clustering on available variables.")

    base["municipality_code"] = base["OMRÅDE"].astype(str).str.zfill(4)

    from src.municipalities import annotate
    base = annotate(base)
    base = base[base["OMRÅDE"].isin(VALID_CODES)].copy()

    base.to_parquet(cache_path, index=False)
    print(f"\n[fetch] Saved: {len(base)} municipalities, {len(base.columns)} columns")
    return base
