"""
Fetches socioeconomic data for all 98 Danish municipalities from the
Statistics Denmark (DST) StatBank API, and GeoJSON boundaries from DAWA.

DST API: https://api.statbank.dk/v1/data  (POST, CSV)
         https://api.statbank.dk/v1/tableinfo/{id}  (GET, metadata)

Strategy
--------
Rather than requesting all OMRÅDE values and filtering, we request only
the 98 known municipality codes directly. This avoids every filtering
ambiguity and guarantees we get exactly the municipalities we want.

On the first run this module also fetches table metadata (variable codes
and their allowed values) for every table, and prints them to the Actions
log. This makes any future 400 error immediately diagnosable without
needing another debug cycle.
"""

import json
import io
import requests
import pandas as pd
from pathlib import Path


DST_API      = "https://api.statbank.dk/v1/data"
DST_INFO_API = "https://api.statbank.dk/v1/tableinfo"
DAWA_URL     = "https://api.dataforsyningen.dk/kommuner?format=geojson"
DATA_DIR     = Path("data")

HEADERS = {
    "Content-Type": "application/json; charset=utf-8",
    "User-Agent": "kommuner-clustering/1.0 (portfolio; github.com/iamboehnke)",
}


# ---------------------------------------------------------------------------
# Municipality codes -- loaded once, used in every API request
# ---------------------------------------------------------------------------

def _get_municipality_codes() -> list[str]:
    """Returns all 98 municipality codes as strings, e.g. ['101', '147', ...]."""
    from src.municipalities import MUNICIPALITY_NAMES
    return list(MUNICIPALITY_NAMES.keys())


# ---------------------------------------------------------------------------
# Table metadata diagnostic
# ---------------------------------------------------------------------------

def print_table_info(table: str) -> None:
    """
    Fetches table metadata from the DST tableinfo endpoint and prints:
      - Table description
      - Every variable code and its first 5 allowed values

    Called before every fetch so that the Actions log always shows the
    exact variable codes available, making 400 errors immediately fixable.
    """
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
        print(f"  [meta] {table}: {info.get('text', '(no description)')}")
        for var in info.get("variables", []):
            sample_vals = [v["id"] for v in var.get("values", [])[:6]]
            elim = "(eliminatable)" if var.get("elimination") else "(REQUIRED)"
            print(f"    var '{var['id']}' {elim}: e.g. {sample_vals}")
    except Exception as e:
        print(f"  [meta] {table}: {e}")


# ---------------------------------------------------------------------------
# Core POST helper
# ---------------------------------------------------------------------------

def _dst_post(table: str, variables: list) -> pd.DataFrame:
    """
    POSTs a CSV query to the DST StatBank API and returns a DataFrame.
    Logs the payload before every request so failures are diagnosable.
    """
    payload = {
        "table":     table,
        "format":    "CSV",
        "lang":      "da",
        "variables": variables,
    }
    payload_str = json.dumps(payload, ensure_ascii=False)
    print(f"  [dst] POST {table}: {payload_str[:500]}")

    try:
        response = requests.post(
            DST_API,
            data=payload_str.encode("utf-8"),
            headers=HEADERS,
            timeout=45,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Network error {table}: {e}") from e

    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code} for {table}: {response.text[:400]}"
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
        raise RuntimeError(f"CSV parse failed {table}: {e}") from e

    print(f"  [dst] {table}: {len(df)} rows, columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Individual table fetchers
# ---------------------------------------------------------------------------

def fetch_population(year: str = "2024K1") -> pd.DataFrame:
    """
    FOLK1A: Population by municipality, age, sex, marital status.
    Request only the 98 municipality codes directly -- no filtering needed.

    Returns: OMRÅDE, pop_total, pct_elderly, pct_youth
    """
    codes = _get_municipality_codes()
    print_table_info("FOLK1A")

    df = _dst_post("FOLK1A", [
        {"code": "OMRÅDE",     "values": codes},
        {"code": "KØN",        "values": ["TOT"]},
        {"code": "ALDER",      "values": ["*"]},
        {"code": "CIVILSTAND", "values": ["TOT"]},
        {"code": "TID",        "values": [year]},
    ])

    # OMRÅDE values now only contain the 98 municipalities we requested
    # Identify the column for municipality and for the count
    omr_col = df.columns[0]   # OMRÅDE
    alder_col = "ALDER"
    val_col = "INDHOLD"

    df["age_num"] = pd.to_numeric(
        df[alder_col].astype(str).str.extract(r"^(\d+)", expand=False),
        errors="coerce"
    )
    df["n"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0)
    df = df.dropna(subset=["age_num"]).copy()

    # Extract just the numeric code from OMRÅDE (e.g. "101 København" -> "101")
    df["kode"] = df[omr_col].astype(str).str.extract(r"^(\d+)", expand=False)

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
    Unemployment by municipality.
    Fetches table metadata first so the log shows the exact variable codes.
    Then requests with those codes.
    """
    codes = _get_municipality_codes()
    print_table_info("AULAAR")

    # AULAAR variable codes are printed above -- adjust if they differ
    # Common alternatives for the municipality variable: BOPKOM, KOMKODE, REGION
    # Common alternatives for percent/persons: PERPCT -> might be just two rows
    df = _dst_post("AULAAR", [
        {"code": "BOPKOMMUNEDK", "values": codes},
        {"code": "KØN",          "values": ["TOT"]},
        {"code": "PERPCT",       "values": ["*"]},
        {"code": "TID",          "values": [year]},
    ])

    omr_col  = df.columns[0]
    df["kode"] = df[omr_col].astype(str).str.extract(r"^(\d+)", expand=False)

    # Keep the percentage rows (not person counts)
    perpct_col = "PERPCT" if "PERPCT" in df.columns else df.columns[1]
    pct_mask = df[perpct_col].astype(str).str.lower().str.contains(
        "pct|procent|ledighed", na=False
    )
    df_pct = df[pct_mask] if pct_mask.any() else df

    val_col = "INDHOLD" if "INDHOLD" in df.columns else df.columns[-1]
    df_pct = df_pct.copy()
    df_pct["unemployment_rate"] = pd.to_numeric(df_pct[val_col], errors="coerce")

    return (
        df_pct.groupby("kode")["unemployment_rate"].mean()
        .reset_index()
        .rename(columns={"kode": "OMRÅDE"})
    )


def fetch_income(year: str = "2022") -> pd.DataFrame:
    """
    Income distribution by municipality.
    Metadata printed first -- adjust variable codes based on log output.
    """
    codes = _get_municipality_codes()
    print_table_info("INDKP101")

    df = _dst_post("INDKP101", [
        {"code": "OMRÅDE",       "values": codes},
        {"code": "INDKOMSTTYPE", "values": ["*"]},
        {"code": "ENHED",        "values": ["*"]},
        {"code": "TID",          "values": [year]},
    ])

    omr_col = df.columns[0]
    df["kode"] = df[omr_col].astype(str).str.extract(r"^(\d+)", expand=False)
    val_col = "INDHOLD" if "INDHOLD" in df.columns else df.columns[-1]

    # Try to find a median / representative income value
    # If INDKOMSTTYPE has a "mediandisponibel" or similar, use that
    # Otherwise take the mean across all income types
    indk_col = "INDKOMSTTYPE" if "INDKOMSTTYPE" in df.columns else df.columns[1]
    kr_mask = ~df[val_col].astype(str).str.contains(r"\.\.", na=True)

    df_kr = df[kr_mask].copy()
    df_kr["income_val"] = pd.to_numeric(df_kr[val_col], errors="coerce")

    return (
        df_kr.groupby("kode")["income_val"].median()
        .reset_index()
        .rename(columns={"kode": "OMRÅDE", "income_val": "median_income"})
    )


def fetch_education(year: str = "2023") -> pd.DataFrame:
    """
    Highest completed education by municipality.
    Metadata printed first -- HFUDD11 uses a different municipality variable.
    """
    codes = _get_municipality_codes()
    print_table_info("HFUDD11")

    # Try KOMKODE as the municipality variable (common in education tables)
    df = _dst_post("HFUDD11", [
        {"code": "KOMKODE", "values": codes},
        {"code": "HFUDD",   "values": ["*"]},
        {"code": "ALDER",   "values": ["*"]},
        {"code": "TID",     "values": [year]},
    ])

    omr_col = df.columns[0]
    df["kode"] = df[omr_col].astype(str).str.extract(r"^(\d+)", expand=False)
    val_col = "INDHOLD" if "INDHOLD" in df.columns else df.columns[-1]

    hfudd_col = "HFUDD" if "HFUDD" in df.columns else df.columns[1]
    higher_mask = df[hfudd_col].astype(str).str.upper().str.match(r"H[67]")

    df["n"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0)
    totals = df.groupby("kode")["n"].sum().rename("edu_total")
    higher = df[higher_mask].groupby("kode")["n"].sum().rename("edu_higher")

    result = pd.concat([totals, higher], axis=1).fillna(0).reset_index()
    result = result[result["edu_total"] > 0].copy()
    result["pct_higher_edu"] = 100 * result["edu_higher"] / result["edu_total"]
    return result[["kode", "pct_higher_edu"]].rename(columns={"kode": "OMRÅDE"})


def fetch_housing(year: str = "2023") -> pd.DataFrame:
    """
    Dwellings by ownership type and municipality.
    Metadata printed first -- BOL101 variable names verified from log.
    """
    codes = _get_municipality_codes()
    print_table_info("BOL101")

    df = _dst_post("BOL101", [
        {"code": "OMRÅDE", "values": codes},
        {"code": "EJFTYP", "values": ["*"]},
        {"code": "BEBO",   "values": ["*"]},
        {"code": "TID",    "values": [year]},
    ])

    omr_col = df.columns[0]
    df["kode"] = df[omr_col].astype(str).str.extract(r"^(\d+)", expand=False)
    val_col = "INDHOLD" if "INDHOLD" in df.columns else df.columns[-1]

    # Social housing: look for "almen" in the ownership type column
    ejf_col = "EJFTYP" if "EJFTYP" in df.columns else df.columns[1]
    social_mask = df[ejf_col].astype(str).str.lower().str.contains(
        "almen|social|offentlig|almene", na=False
    )

    df["n"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0)
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
    response = requests.get(DAWA_URL, timeout=30)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Master assembler
# ---------------------------------------------------------------------------

def build_dataset(cache: bool = True) -> pd.DataFrame:
    """
    Fetches all tables, joins by OMRÅDE code, returns one row per municipality.
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
            print(f"  -> {len(df)} rows")
            base = df if base is None else base.merge(df, on="OMRÅDE", how="outer")
        except Exception as e:
            print(f"  WARNING: {name} failed: {e}")
            failed.append(name)

    if base is None or len(base) == 0:
        raise RuntimeError(
            f"No municipality data loaded. Failed tables: {failed}. "
            "Check the [meta] lines above for the correct variable codes."
        )

    if failed:
        print(f"\n[fetch] Partial success. Failed: {failed}")

    base["municipality_code"] = base["OMRÅDE"].astype(str).str.zfill(4)

    from src.municipalities import annotate, MUNICIPALITY_NAMES
    base = annotate(base)

    # Keep only the 98 recognised municipalities
    valid_codes = set(MUNICIPALITY_NAMES.keys())
    base = base[base["OMRÅDE"].isin(valid_codes)].copy()

    base.to_parquet(cache_path, index=False)
    print(f"\n[fetch] Saved: {len(base)} municipalities, {len(base.columns)} columns")
    return base
