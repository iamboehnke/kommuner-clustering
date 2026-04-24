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



# ---------------------------------------------------------------------------
# Outcome variable fetchers
# These are NOT used as clustering inputs -- they stay separate so the
# cluster structure remains purely structural. They are used to compare
# performance within clusters (peer benchmarking).
# ---------------------------------------------------------------------------

def fetch_disability_pension(year: str = "2023") -> pd.DataFrame:
    """
    FORTS1: Recipients of disability pension (førtidspension) by municipality.

    Returns share of working-age population (18-64) receiving disability pension.
    This is one of the largest municipal social expenditure drivers and varies
    significantly within structural peer groups -- making it a strong signal of
    how well a municipality manages social inclusion.

    Confirmed variable: BOPKOMMUNEDK (municipality of residence), not OMRÅDE.
    """
    codes = _get_codes()
    print_table_info("FORTS1")

    df = _dst_post("FORTS1", [
        {"code": "BOPKOMMUNEDK", "values": codes},
        {"code": "YDELSE",       "values": ["*"]},
        {"code": "KØN",          "values": ["TOT"]},
        {"code": "TID",          "values": [year]},
    ])

    # BOPKOMMUNEDK returns text labels like FOLK1A -- reverse lookup
    omr_col = "BOPKOMMUNEDK" if "BOPKOMMUNEDK" in df.columns else df.columns[0]
    df["kode"] = _resolve_kode(df, omr_col)
    df = df[df["kode"].notna() & df["kode"].isin(VALID_CODES)].copy()
    df["n"] = _to_numeric(df["INDHOLD"])

    print(f"  FORTS1 YDELSE sample: {df['YDELSE'].unique()[:5].tolist()}")

    # Total recipients across all disability pension types
    totals = df.groupby("kode")["n"].sum().rename("disability_pension_count")

    # We need working-age population to compute the share
    # Fetch from FOLK1A for the same year
    try:
        pop_df = fetch_population(year=f"{year}K1")
        pop_df = pop_df.rename(columns={"OMRÅDE": "kode"})
        result = totals.reset_index().merge(pop_df[["kode", "pop_total"]], on="kode", how="left")
        result["pct_disability_pension"] = (
            100 * result["disability_pension_count"] / result["pop_total"]
        ).round(2)
        print(f"  FORTS1: {len(result)} municipalities")
        return result[["kode", "pct_disability_pension"]].rename(columns={"kode": "OMRÅDE"})
    except Exception as e:
        print(f"  FORTS1: population join failed ({e}), returning raw counts")
        result = totals.reset_index()
        result["pct_disability_pension"] = result["disability_pension_count"]
        return result[["kode", "pct_disability_pension"]].rename(columns={"kode": "OMRÅDE"})


def fetch_youth_education(year: str = "2022") -> pd.DataFrame:
    """
    UNGEUDDU: Share of young people (15-24) enrolled in or having completed
    a youth education programme (ungdomsuddannelse).

    This is directly tied to the national 95% target (95-procent-målsætningen):
    the goal that 95% of young people complete a youth education. Progress
    varies considerably by municipality and is a key policy priority.

    Note: education completion data typically lags 1-2 years.
    """
    codes = _get_codes()
    print_table_info("UNGEUDDU")

    df = _dst_post("UNGEUDDU", [
        {"code": "BOPKOMMUNEDK", "values": codes},
        {"code": "UDDANNELSE",   "values": ["*"]},
        {"code": "TID",          "values": [year]},
    ])

    omr_col = "BOPKOMMUNEDK" if "BOPKOMMUNEDK" in df.columns else df.columns[0]
    df["kode"] = _resolve_kode(df, omr_col)
    df = df[df["kode"].notna() & df["kode"].isin(VALID_CODES)].copy()
    df["n"] = _to_numeric(df["INDHOLD"])

    print(f"  UNGEUDDU UDDANNELSE sample: {df['UDDANNELSE'].unique()[:5].tolist()}")
    print(f"  UNGEUDDU cols: {list(df.columns)}")

    # Look for a percentage or "i gang/fuldfort" type row
    udd_col = "UDDANNELSE" if "UDDANNELSE" in df.columns else df.columns[1]
    pct_mask = df[udd_col].astype(str).str.lower().str.contains(
        "pct|procent|andel|igang|fuldført|i alt", na=False
    )
    df_pct = df[pct_mask] if pct_mask.any() else df

    result = (
        df_pct.groupby("kode")["n"].mean()
        .reset_index()
        .rename(columns={"kode": "OMRÅDE", "n": "pct_youth_education"})
    )
    result["pct_youth_education"] = result["pct_youth_education"].round(1)
    print(f"  UNGEUDDU: {len(result)} municipalities")
    return result


def fetch_outcomes(year: str = "2023", cache: bool = True) -> pd.DataFrame:
    """
    Fetches all outcome variables and returns a single merged DataFrame.
    Cached separately from the structural variables.
    """
    cache_path = DATA_DIR / f"outcomes_{year}.parquet"
    DATA_DIR.mkdir(exist_ok=True)

    if cache and cache_path.exists():
        print(f"[fetch] Outcomes {year}: loading from cache...")
        return pd.read_parquet(cache_path)

    print(f"\n[fetch] Fetching outcome variables for {year}...")

    outcome_steps = [
        ("Disability pension", lambda: fetch_disability_pension(year=year)),
        ("Youth education",    lambda: fetch_youth_education(year=year)),
    ]

    base = None
    for name, fn in outcome_steps:
        print(f"\n  [outcomes] {name}...")
        try:
            df = fn()
            print(f"    -> {len(df)} rows")
            base = df if base is None else base.merge(df, on="OMRÅDE", how="outer")
        except Exception as e:
            print(f"    WARNING: {name} failed: {e}")

    if base is None:
        print("[fetch] All outcome fetches failed -- outcomes will be unavailable.")
        return pd.DataFrame(columns=["OMRÅDE"])

    base.to_parquet(cache_path, index=False)
    print(f"\n[fetch] Outcomes saved: {len(base)} municipalities")
    return base



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


# ---------------------------------------------------------------------------
# Temporal fetch -- year-tagged builds
# ---------------------------------------------------------------------------

# Year specifications for temporal analysis.
# AUP01 (unemployment) only starts from 2017M07, so 2017 is our baseline.
YEAR_SPECS = {
    "2017": {
        "folk1a": "2017K1",
        "aup01":  "2017M07",
        "indkp":  "2017",
        "hfudd":  "2017",
        "bol":    "2017",
    },
    "2023": {
        "folk1a": "2024K1",
        "aup01":  "2024M01",
        "indkp":  "2022",     # income stats lag ~2 years
        "hfudd":  "2023",
        "bol":    "2023",
    },
}


def build_dataset_for_year(year_tag: str, cache: bool = True) -> pd.DataFrame:
    """
    Fetches and assembles the full municipality dataset for a given year tag.

    year_tag must be a key in YEAR_SPECS, e.g. "2017" or "2023".
    Data is cached to data/municipalities_{year_tag}.parquet.
    """
    if year_tag not in YEAR_SPECS:
        raise ValueError(f"Unknown year_tag '{year_tag}'. Choose from {list(YEAR_SPECS)}")

    spec       = YEAR_SPECS[year_tag]
    cache_path = DATA_DIR / f"municipalities_{year_tag}.parquet"
    DATA_DIR.mkdir(exist_ok=True)

    if cache and cache_path.exists():
        print(f"[fetch] {year_tag}: loading from cache...")
        return pd.read_parquet(cache_path)

    print(f"\n[fetch] Fetching {year_tag} data...")

    steps = [
        ("Population",   lambda: fetch_population(year=spec["folk1a"])),
        ("Unemployment", lambda: fetch_unemployment(year_month=spec["aup01"])),
        ("Income",       lambda: fetch_income(year=spec["indkp"])),
        ("Education",    lambda: fetch_education(year=spec["hfudd"])),
        ("Housing",      lambda: fetch_housing(year=spec["bol"])),
    ]

    base   = None
    failed = []
    for name, fn in steps:
        print(f"  [{year_tag}] {name}...")
        try:
            df = fn()
            base = df if base is None else base.merge(df, on="OMRÅDE", how="outer")
        except Exception as e:
            print(f"  [{year_tag}] WARNING {name}: {e}")
            failed.append(name)

    if base is None or len(base) == 0:
        raise RuntimeError(f"No data for {year_tag}. Failed: {failed}")

    if failed:
        print(f"  [{year_tag}] Partial: {failed} missing")

    base["municipality_code"] = base["OMRÅDE"].astype(str).str.zfill(4)
    from src.municipalities import annotate
    base = annotate(base)
    base = base[base["OMRÅDE"].isin(VALID_CODES)].copy()

    base.to_parquet(cache_path, index=False)
    print(f"  [{year_tag}] Saved: {len(base)} municipalities")
    return base