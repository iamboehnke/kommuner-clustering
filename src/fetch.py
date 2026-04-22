"""
Fetches socioeconomic data for all 98 Danish municipalities from DST.

KEY INSIGHT (confirmed from live API responses):
The DST CSV API returns TEXT LABELS in area columns, not numeric codes.
Requesting OMRÅDE=["101"] returns "København" in the CSV, not "101".
Solution: reverse-lookup from name -> code using municipalities.py.

Confirmed variable names and time formats:
  FOLK1A:   OMRÅDE (text), KØN, ALDER, CIVILSTAND, TID -> INDHOLD
  AUP01:    OMRÅDE (text), ALDER, KØN, Tid (monthly: "2024M01") -> INDHOLD
  INDKP101: OMRÅDE (text), INDKOMSTTYPE (text), ENHED (text), KOEN, Tid -> INDHOLD
  HFUDD11:  BOPOMR (text), HFUDD, HERKOMST, ALDER, KØN, Tid -> INDHOLD
  BOL101:   OMRÅDE (text), BEBO, EJER (text), Tid -> INDHOLD
              social housing label = "Almene boligselskaber"
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

# Forward: code -> name  e.g. {"101": "København"}
# Reverse: name -> code  e.g. {"København": "101"}
CODE_TO_NAME = MUNICIPALITY_NAMES
NAME_TO_CODE = {v: k for k, v in MUNICIPALITY_NAMES.items()}
VALID_CODES  = set(MUNICIPALITY_NAMES.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_codes() -> list[str]:
    return list(MUNICIPALITY_NAMES.keys())


def _dst_post(table: str, variables: list) -> pd.DataFrame:
    """POST to DST CSV API. Reads all columns as strings for reliable parsing."""
    payload = {"table": table, "format": "CSV", "lang": "da", "variables": variables}
    payload_str = json.dumps(payload, ensure_ascii=False)
    print(f"  [dst] POST {table}")

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

    df = pd.read_csv(io.StringIO(r.text), sep=";", dtype=str, encoding="utf-8")
    print(f"  [dst] {table}: {len(df)} rows, cols: {list(df.columns)}")
    return df


def _to_numeric(series: pd.Series) -> pd.Series:
    """
    Converts a DST INDHOLD column to float.
    DST uses '.' as thousands sep and ',' as decimal sep in Danish locale.
    Suppressed cells contain '..' -> NaN -> 0.
    """
    return (
        pd.to_numeric(
            series.astype(str)
                  .str.strip()
                  .str.replace(".", "", regex=False)
                  .str.replace(",", ".", regex=False),
            errors="coerce"
        ).fillna(0)
    )


def _resolve_kode(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Resolves municipality codes from a DST text-label area column.

    DST returns text labels like 'København', 'Frederiksberg' in the CSV.
    We reverse-lookup against our known municipality name->code mapping.
    Unrecognised names get NaN (subsequently filtered out).
    """
    return df[col].astype(str).str.strip().map(NAME_TO_CODE)


# ---------------------------------------------------------------------------
# Table fetchers
# ---------------------------------------------------------------------------

def fetch_population(year: str = "2024K1") -> pd.DataFrame:
    """
    FOLK1A: Population by municipality, age, sex, marital status.
    OMRÅDE column returns text labels -> reverse-lookup to codes.
    """
    codes = _get_codes()
    df = _dst_post("FOLK1A", [
        {"code": "OMRÅDE",     "values": codes},
        {"code": "KØN",        "values": ["TOT"]},
        {"code": "ALDER",      "values": ["*"]},
        {"code": "CIVILSTAND", "values": ["TOT"]},
        {"code": "TID",        "values": [year]},
    ])

    df["kode"] = _resolve_kode(df, "OMRÅDE")
    df = df[df["kode"].notna() & df["kode"].isin(VALID_CODES)].copy()

    df["age_num"] = pd.to_numeric(
        df["ALDER"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce"
    )
    df["n"] = _to_numeric(df["INDHOLD"])
    df = df.dropna(subset=["age_num"]).copy()

    print(f"  FOLK1A: {df['kode'].nunique()} municipalities, "
          f"{df['age_num'].nunique()} age groups")

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


def fetch_unemployment(year_month: str = "2024M01") -> pd.DataFrame:
    """
    AUP01: Unemployment rate by municipality.
    Time format is monthly: "2024M01". OMRÅDE returns text labels.
    """
    codes = _get_codes()
    df = _dst_post("AUP01", [
        {"code": "OMRÅDE", "values": codes},
        {"code": "ALDER",  "values": ["TOT"]},
        {"code": "KØN",    "values": ["TOT"]},
        {"code": "TID",    "values": [year_month]},
    ])

    df["kode"] = _resolve_kode(df, "OMRÅDE")
    df = df[df["kode"].notna() & df["kode"].isin(VALID_CODES)].copy()
    df["unemployment_rate"] = _to_numeric(df["INDHOLD"])

    print(f"  AUP01: {df['kode'].nunique()} municipalities")
    return (
        df.groupby("kode")["unemployment_rate"].mean()
        .reset_index()
        .rename(columns={"kode": "OMRÅDE"})
    )


def fetch_income(year: str = "2022") -> pd.DataFrame:
    """
    INDKP101: Income by municipality.
    ENHED label 'Gennemsnit for alle personer (kr.)' = mean income for all persons.
    INDKOMSTTYPE label '1 Disponibel indkomst...' = disposable income.
    """
    codes = _get_codes()
    df = _dst_post("INDKP101", [
        {"code": "OMRÅDE",       "values": codes},
        {"code": "INDKOMSTTYPE", "values": ["*"]},
        {"code": "ENHED",        "values": ["*"]},
        {"code": "KOEN",         "values": ["MOK"]},
        {"code": "TID",          "values": [year]},
    ])

    df["kode"] = _resolve_kode(df, "OMRÅDE")
    df = df[df["kode"].notna() & df["kode"].isin(VALID_CODES)].copy()
    df["n"] = _to_numeric(df["INDHOLD"])

    # Filter to: disposable income + mean income for all persons
    indk_mask = df["INDKOMSTTYPE"].astype(str).str.startswith("1 ")
    enhed_mask = df["ENHED"].astype(str).str.contains("Gennemsnit for alle", na=False)

    df_m = df[indk_mask & enhed_mask].copy()
    if len(df_m) == 0:
        # Fallback: any mean income row
        df_m = df[enhed_mask].copy()
    if len(df_m) == 0:
        df_m = df.copy()

    print(f"  INDKP101: {df_m['kode'].nunique()} municipalities")
    return (
        df_m.groupby("kode")["n"].mean()
        .reset_index()
        .rename(columns={"kode": "OMRÅDE", "n": "median_income"})
    )


def fetch_education(year: str = "2023") -> pd.DataFrame:
    """
    HFUDD11: Education by municipality of residence.
    Municipality variable is BOPOMR (returns text labels -> reverse-lookup).
    Higher education = HFUDD codes starting with H6 or H7.
    """
    codes = _get_codes()
    df = _dst_post("HFUDD11", [
        {"code": "BOPOMR",   "values": codes},
        {"code": "HFUDD",    "values": ["*"]},
        {"code": "HERKOMST", "values": ["TOT"]},
        {"code": "ALDER",    "values": ["*"]},
        {"code": "KØN",      "values": ["TOT"]},
        {"code": "TID",      "values": [year]},
    ])

    df["kode"] = _resolve_kode(df, "BOPOMR")
    df = df[df["kode"].notna() & df["kode"].isin(VALID_CODES)].copy()
    df["n"] = _to_numeric(df["INDHOLD"])

    # HFUDD codes H6x = medium cycle higher education, H7x = long cycle
    hfudd_col = "HFUDD"
    higher_mask = df[hfudd_col].astype(str).str.match(r"H[67]")

    totals = df.groupby("kode")["n"].sum().rename("edu_total")
    higher = df[higher_mask].groupby("kode")["n"].sum().rename("edu_higher")

    result = pd.concat([totals, higher], axis=1).fillna(0).reset_index()
    result = result[result["edu_total"] > 0].copy()
    result["pct_higher_edu"] = 100 * result["edu_higher"] / result["edu_total"]

    print(f"  HFUDD11: {len(result)} municipalities")
    return result[["kode", "pct_higher_edu"]].rename(columns={"kode": "OMRÅDE"})


def fetch_housing(year: str = "2023") -> pd.DataFrame:
    """
    BOL101: Dwellings by ownership and municipality.
    Social housing label confirmed: 'Almene boligselskaber'.
    """
    codes = _get_codes()
    df = _dst_post("BOL101", [
        {"code": "OMRÅDE", "values": codes},
        {"code": "BEBO",   "values": ["*"]},
        {"code": "EJER",   "values": ["*"]},
        {"code": "TID",    "values": [year]},
    ])

    df["kode"] = _resolve_kode(df, "OMRÅDE")
    df = df[df["kode"].notna() & df["kode"].isin(VALID_CODES)].copy()
    df["n"] = _to_numeric(df["INDHOLD"])

    social_mask = df["EJER"].astype(str).str.strip() == "Almene boligselskaber"

    totals = df.groupby("kode")["n"].sum().rename("dwellings_total")
    social = df[social_mask].groupby("kode")["n"].sum().rename("dwellings_social")

    result = pd.concat([totals, social], axis=1).fillna(0).reset_index()
    result = result[result["dwellings_total"] > 0].copy()
    result["pct_social_housing"] = 100 * result["dwellings_social"] / result["dwellings_total"]

    print(f"  BOL101: {len(result)} municipalities")
    return result[["kode", "pct_social_housing"]].rename(columns={"kode": "OMRÅDE"})


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------

def fetch_geodata() -> dict:
    """
    Fetches official Danish municipality boundaries from DAWA and simplifies
    the geometry to reduce file size.

    DAWA provides 1:1000-scale precision (suitable for printed maps).
    For a national-level web choropleth we only need ~1:100000 precision.
    Simplifying to 0.005 degree tolerance (~500m) reduces the file from
    ~150 MB to ~2 MB while looking identical at national zoom level.
    """
    import geopandas as gpd
    import json as _json

    r = requests.get(DAWA_URL, timeout=60)
    r.raise_for_status()

    gdf = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.005, preserve_topology=True)

    simplified = _json.loads(gdf.to_json())
    print(f"[fetch] GeoJSON: {len(simplified['features'])} features after simplification")
    return simplified


# ---------------------------------------------------------------------------
# Master assembler
# ---------------------------------------------------------------------------

def build_dataset(cache: bool = True) -> pd.DataFrame:
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
            f"No data loaded. Failed: {failed}. "
            "Check the log above for details."
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
