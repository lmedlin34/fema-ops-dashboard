# app.py
import io
import zipfile
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import pydeck as pdk
from urllib.parse import quote
from typing import Optional, Iterable, Union

# ----------------------------
# PAGE CONFIG & THEME CSS
# ----------------------------
st.set_page_config(page_title="FEMA Operations Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
/* Banner */
.banner {
  padding: 28px 22px;
  border-radius: 22px;
  background: linear-gradient(135deg, #0ea5e9, #6366f1);
  color: white; border: 1px solid rgba(255,255,255,0.25);
  box-shadow: 0 12px 30px rgba(2,6,23,0.15);
  margin-bottom: 14px;
}
.banner h1 { margin: 0 0 6px 0; font-size: 2.1rem; }
.banner p { margin: 0; opacity: 0.95; }

/* KPI cards */
.kpi-row { display: flex; gap: 20px; justify-content: space-between; margin: 10px 0 20px; }
.kpi-card {
  flex: 1; padding: 26px 22px; border-radius: 20px;
  color: white; text-align: left; box-shadow: 0 10px 24px rgba(2,6,23,0.10);
}
.kpi-title { font-size: 0.95rem; opacity: 0.95; margin-top: 10px; }
.kpi-value { font-size: 2.6rem; font-weight: 700; margin-top: 6px; }
.kpi-icon { font-size: 28px; opacity: 0.95; }

/* three-stop vertical gradients */
.kpi-blue {
  background: linear-gradient(180deg,#60a5fa 0%, #2563eb 50%, #1e3a8a 100%);
}
.kpi-green {
  background: linear-gradient(180deg,#6ee7b7 0%, #10b981 50%, #064e3b 100%);
}
.kpi-orange {
  background: linear-gradient(180deg,#fcd34d 0%, #f59e0b 50%, #b45309 100%);
}

/* Tab titles spacing */
.block-title { margin: 10px 0 -6px; font-weight: 700; }
</style>
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# ----------------------------
# DATA FETCHERS
# ----------------------------
BASE = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"

def _to_iso_z(x, end_of_day=False) -> str:
    ts = pd.to_datetime(x)
    if end_of_day:
        ts = ts.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59, milliseconds=999)
    else:
        ts = ts.normalize()
    return ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")

def fetch_fema(
    state: Optional[str],
    start_date: Union[str, pd.Timestamp],
    *,
    limit: int = 5000,
    max_pages: int = 100,
    select: Optional[Iterable[str]] = (
        "disasterNumber,state,incidentType,designatedArea,declarationDate,fyDeclared,"
        "declarationType,placeCode,fipsStateCode,fipsCountyCode,ihProgramDeclared,"
        "paProgramDeclared,hmProgramDeclared,incidentBeginDate,incidentEndDate"
    ).split(","),
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch FEMA declarations from start_date → now (EOD UTC). Adds 5-digit FIPS column."""
    start_iso = _to_iso_z(start_date, end_of_day=False)
    end_iso   = _to_iso_z(pd.Timestamp.utcnow(), end_of_day=True)

    filters = [f"declarationDate ge '{start_iso}'", f"declarationDate le '{end_iso}'"]
    if state:
        filters.append(f"state eq '{state.upper()}'")
    filter_str = " and ".join(filters)

    parts = [
        "$filter=" + quote(filter_str, safe=" ()'=<>&"),
        "$orderby=" + quote("declarationDate desc"),
        f"$top={limit}",
        "$skip=0",
    ]
    if select:
        parts.append("$select=" + ",".join(select))
    qs = "&".join(parts)

    frames, skip = [], 0
    for _ in range(max_pages):
        url = f"{BASE}?{qs.replace('$skip=0', f'$skip={skip}')}"
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        rows = payload.get("DisasterDeclarationsSummaries", [])
        if not rows:
            break
        frames.append(pd.json_normalize(rows))
        if len(rows) < limit:
            break
        skip += limit

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Dates → tz-naive
    for c in ["declarationDate", "incidentBeginDate", "incidentEndDate", "lastIAFLOPDate", "lastRefresh"]:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            if getattr(s.dtype, "tz", None) is not None:
                s = s.dt.tz_convert("UTC").dt.tz_localize(None)
            df[c] = s

    # 5-digit county FIPS for joins (fipsStateCode 2 + fipsCountyCode 3)
    if {"fipsStateCode", "fipsCountyCode"}.issubset(df.columns):
        df["fipsCountyCodeFull"] = df["fipsStateCode"].astype(str).str.zfill(2) + df["fipsCountyCode"].astype(str).str.zfill(3)
    else:
        df["fipsCountyCodeFull"] = pd.NA

    return df

# ---------- Robust FIPS loader: ZIP/TXT, counties/cousubs, with fallbacks ----------
import io, zipfile, requests

@st.cache_data(show_spinner=True)
def load_fips_lookup_from_gazetteer_urls(urls: list[str]) -> pd.DataFrame:
    """
    Try multiple Census Gazetteer URLs until one works.
    Supports .zip (with embedded .txt) and .txt directly.
    Handles 'counties' (5-digit GEOID) and 'cousubs' (10-digit GEOID; collapsed to county).
    Returns columns: fipsCountyCode (5), latitude, longitude
    """
    last_err = None

    def _read_txt(file_like) -> pd.DataFrame:
        df = pd.read_csv(file_like, sep="|", dtype={"GEOID": str}, engine="python")
        req = {"GEOID", "INTPTLAT", "INTPTLONG"}
        if not req.issubset(df.columns):
            raise ValueError("Gazetteer missing GEOID / INTPTLAT / INTPTLONG")
        df = df[["GEOID", "INTPTLAT", "INTPTLONG"]].rename(
            columns={"INTPTLAT": "latitude", "INTPTLONG": "longitude"}
        )
        # normalize numerics early
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])
        # counties vs cousubs
        if df["GEOID"].str.len().median() >= 10:
            # cousubs -> collapse to 5-digit county by GEOID[:5]
            df["fipsCountyCode"] = df["GEOID"].str[:5]
            df = (df.groupby("fipsCountyCode", as_index=False)
                    .agg(latitude=("latitude", "mean"),
                         longitude=("longitude", "mean")))
        else:
            df["fipsCountyCode"] = df["GEOID"].str.zfill(5)
            df = df[["fipsCountyCode", "latitude", "longitude"]]
        df["fipsCountyCode"] = df["fipsCountyCode"].astype(str).str.zfill(5)
        return df[["fipsCountyCode", "latitude", "longitude"]]

    for url in urls:
        try:
            resp = requests.get(url, timeout=90)
            resp.raise_for_status()
            buf = io.BytesIO(resp.content)

            if url.lower().endswith(".zip"):
                # First: try to open as a ZIP
                try:
                    with zipfile.ZipFile(buf) as zf:
                        # pick a .txt; prefer filenames containing 'counties'
                        candidates = [n for n in zf.namelist() if n.lower().endswith(".txt")]
                        if not candidates:
                            raise ValueError("ZIP does not contain a .txt Gazetteer file.")
                        candidates.sort(key=lambda n: (("counties" not in n.lower()), n))
                        with zf.open(candidates[0]) as f:
                            df = _read_txt(io.TextIOWrapper(f, encoding="utf-8"))
                            return df
                except zipfile.BadZipFile:
                    # Some servers return HTML with .zip extension — fall back to reading as TXT bytes
                    try:
                        df = _read_txt(io.BytesIO(resp.content))
                        return df
                    except Exception as e2:
                        last_err = e2
                        continue
            else:
                # Plain TXT
                df = _read_txt(buf)
                return df

        except Exception as e:
            last_err = e
            continue

    # If all candidates failed:
    raise last_err if last_err else RuntimeError("All Gazetteer sources failed")


# ----------------------------
# HELPERS
# ----------------------------
def compute_kpis(df: pd.DataFrame, date_col: str = "declarationDate") -> dict:
    """Compute top-level KPIs for FEMA dashboard"""
    total = len(df)

    # --- Average incident length (by unique disasterNumber) ---
    avg_len_text, avg_len_val = "N/A", "N/A"
    if {"incidentBeginDate", "incidentEndDate"}.issubset(df.columns):
        tmp = (
            df[["disasterNumber", "incidentBeginDate", "incidentEndDate"]]
            .drop_duplicates(subset=["disasterNumber"])
            .assign(
                incidentBeginDate=lambda d: pd.to_datetime(d["incidentBeginDate"], errors="coerce"),
                incidentEndDate=lambda d: pd.to_datetime(d["incidentEndDate"], errors="coerce"),
            )
        )
        tmp = tmp.dropna(subset=["incidentBeginDate", "incidentEndDate"])
        if not tmp.empty:
            tmp["length_days"] = (tmp["incidentEndDate"] - tmp["incidentBeginDate"]).dt.days
            avg_len_val = round(tmp["length_days"].mean(), 1)
            avg_len_text = f"{avg_len_val} days"

    return {
        "Total": (total, "declarations"),
        "AvgLength": (avg_len_val, avg_len_text),
    }

def kpi_deltas(df, date_col="declarationDate"):
    s = pd.to_datetime(df[date_col], errors="coerce").dropna()
    today = pd.Timestamp.today().normalize()
    w = pd.Timedelta(days=30)
    curr = s[(s >= today - w)]
    prev = s[(s < today - w) & (s >= today - 2*w)]
    return {"new_curr": int(curr.size), "new_prev": int(prev.size), "delta": int(curr.size - prev.size)}

def pick_fips_source(df: pd.DataFrame) -> str | None:
    if "fipsCountyCodeFull" in df.columns:
        return "fipsCountyCodeFull"
    if "fipsCountyCode" in df.columns:
        return "fipsCountyCode"
    return None

# ----------------------------
# SIDEBAR (filters + refresh + optional upload)
# ----------------------------
qp = dict(st.query_params)  # immutable snapshot (dict)

# Pull values (they come as strings or lists of strings)
# Safely coerce types and set defaults if missing
url_state = (qp.get("state") or [""])[0] if isinstance(qp.get("state"), list) else (qp.get("state") or "")
url_start = (qp.get("start") or [""])[0] if isinstance(qp.get("start"), list) else (qp.get("start") or "")

# Optional FY bounds from URL
def _to_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

url_fy_min = _to_int((qp.get("fy_min") or [""])[0] if isinstance(qp.get("fy_min"), list) else qp.get("fy_min"))
url_fy_max = _to_int((qp.get("fy_max") or [""])[0] if isinstance(qp.get("fy_max"), list) else qp.get("fy_max"))

# Use these as *initial* defaults for your controls
default_state = url_state or "SC"
default_start_date = pd.to_datetime(url_start, errors="coerce").date() if url_start else (pd.Timestamp.today() - pd.Timedelta(days=365)).date()

st.sidebar.header("Filters")
state_in = st.sidebar.text_input("State (2-letter, optional)", default_state).strip() or None
start_input = st.sidebar.date_input("Start date", value=default_start_date)

# ... when you compute FY slider bounds later, you can apply URL defaults if present:
# if fy_min == fy_max -> show caption (handled already)
# else if url_fy_min/url_fy_max present, use them as the initial slider value

if "last_refresh_ts" not in st.session_state:
    st.session_state.last_refresh_ts = None
if st.sidebar.button("🔄 Refresh data", use_container_width=True):
    st.session_state.last_refresh_ts = pd.Timestamp.utcnow().isoformat()
    st.toast("Refreshed from FEMA.", icon="✅")
    st.rerun()

with st.sidebar.expander("Geo lookup (optional)"):
    st.caption("Provide a FIPS→lat/lon CSV if web lookup fails. Columns: fipsCountyCode, latitude, longitude.")
    fips_upload = st.file_uploader("Upload FIPS lookup CSV", type=["csv"])

@st.cache_data(show_spinner=True)
def get_fema_data(state, start_date, _bust):
    return fetch_fema(state, start_date)

@st.cache_data(show_spinner=True)
def get_fips_lookup(upload):
    # If user uploaded a CSV, use it
    if upload is not None:
        df = pd.read_csv(upload, dtype={"GEOID": str})
        need = {"fipsCountyCode", "latitude", "longitude"}
        if not need.issubset(df.columns):
            raise ValueError("Uploaded CSV must have columns: fipsCountyCode, latitude, longitude")
        df["fipsCountyCode"] = df["fipsCountyCode"].astype(str).str.zfill(5)
        return df[list(need)]

    # Otherwise try robust set of Gazetteer sources (zip + txt, counties + cousubs)
    GAZ_URLS = [
        # Your 2025 cousubs ZIP (will auto-collapse to county)
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2025_Gazetteer/2025_Gaz_counties_national.zip",
        # Known-stable counties TXT from 2023 (fast & reliable)
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_counties_national.zip",
        # Another backup year
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.zip",
    ]
    return load_fips_lookup_from_gazetteer_urls(GAZ_URLS)

# ----------------------------
# DATA LOAD
# ----------------------------
df = get_fema_data(state_in, pd.Timestamp(start_input), st.session_state.last_refresh_ts)
try:
    fips_lookup = get_fips_lookup(fips_upload)
except Exception as e:
    fips_lookup = None
    st.warning(f"FIPS lookup unavailable ({e}). Heatmap will be disabled until fixed or file uploaded.")

# ----------------------------
# BANNER + STORY
# ----------------------------
st.markdown(f"""
<div class="banner">
  <h1>FEMA Disaster Declarations Dashboard</h1>
  <p>{state_in or "All States"} | {pd.to_datetime(start_input).date()} → {pd.Timestamp.today().date()}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**What this shows:** real-time FEMA disaster declarations, filtered by state, with KPIs, a county heatmap, and trends.  
**Who uses it:** emergency ops leads & analysts who need at-a-glance incident load, recency, and geography.  
**Why it matters:** fast situational awareness drives staffing, funding requests, and interagency coordination.
""")

# ----------------------------
# FILTERS (in-page refiners)
# ----------------------------
if df.empty:
    st.info("No FEMA data for the selected filters yet. Try widening the start date.")
    st.stop()

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    types = sorted(df["incidentType"].dropna().unique().tolist()) if "incidentType" in df.columns else []
    type_sel = st.multiselect("Incident type", types, default=types[:3] if types else [])
with colB:
    dtypes = sorted(df["declarationType"].dropna().unique().tolist()) if "declarationType" in df.columns else []
    dtype_sel = st.multiselect("Declaration type", dtypes, default=dtypes[:2] if dtypes else [])
with colC:
    # --- Robust FY filter (handles single-year + empty cases)
    fy = None
    if "fyDeclared" in df.columns:
        fy_series = pd.to_numeric(df["fyDeclared"], errors="coerce").dropna().astype(int)
        if not fy_series.empty:
            fy_min, fy_max = int(fy_series.min()), int(fy_series.max())
            if fy_min == fy_max:
                # Single FY only — display note instead of slider
                st.caption(f"Fiscal year filter: only **FY {fy_max}** present in current dataset.")
                fy = (fy_max, fy_max)
            else:
                fy = st.slider(
                    "Fiscal year filter",
                    min_value=fy_min,
                    max_value=fy_max,
                    value=(max(fy_min, fy_max - 1), fy_max),
                    step=1,
                )

# Apply all filters
mask = pd.Series(True, index=df.index)
if types:
    mask &= df["incidentType"].isin(type_sel) if type_sel else mask
if dtypes:
    mask &= df["declarationType"].isin(dtype_sel) if dtype_sel else mask
if fy and "fyDeclared" in df.columns:
    mask &= pd.to_numeric(df["fyDeclared"], errors="coerce").between(fy[0], fy[1])

df_f = df[mask].copy()
if df_f.empty:
    st.info("No records match the current in-page filters.")
    st.stop()

# ----------------------------
# TABS
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Map", "Trends", "Data"])

# ---- Overview
with tab1:
    kpis = compute_kpis(df_f, "declarationDate")
    delta = kpi_deltas(df_f, "declarationDate")

    # ---- KPI ROW ----
    st.markdown(
        """
        <style>
        .kpi-row {
            display: flex;
            justify-content: space-between;
            align-items: stretch;
            flex-wrap: nowrap;
            gap: 1rem;
            margin-top: 1rem;
            margin-bottom: 1.5rem;
        }
        .kpi-card {
            flex: 1;
            text-align: center;
            color: white;
            padding: 1.25rem;
            border-radius: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
        }
        .kpi-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }
        .kpi-icon {
            font-size: 2rem;
            margin-bottom: 0.25rem;
            opacity: 0.9;
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .kpi-title {
            font-size: 0.95rem;
            margin-top: 0.3rem;
            opacity: 0.9;
        }
        /* Three-color gradients */
        .kpi-blue {
            background: linear-gradient(180deg, #60a5fa 0%, #2563eb 50%, #1e3a8a 100%);
        }
        .kpi-green {
            background: linear-gradient(180deg, #6ee7b7 0%, #10b981 50%, #064e3b 100%);
        }
        .kpi-orange {
            background: linear-gradient(180deg, #fcd34d 0%, #f59e0b 50%, #b45309 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- KPI ROW ----
    st.markdown(
        f"""
        <div class="kpi-row">
            <div class="kpi-card kpi-blue">
                <div class="kpi-icon"><i class="fa-solid fa-burst"></i></div>
                <div class="kpi-value">{kpis['Total'][0]:,}</div>
                <div class="kpi-title">Total Declarations</div>
            </div>
            <div class="kpi-card kpi-green">
                <div class="kpi-icon"><i class="fa-solid fa-plus"></i></div>
                <div class="kpi-value">{delta['new_curr']:,}</div>
                <div class="kpi-title">New in Last 30 Days (Δ {delta['delta']:+,} vs prior 30)</div>
            </div>
            <div class="kpi-card kpi-orange">
                <div class="kpi-icon"><i class="fa-solid fa-clock-rotate-left"></i></div>
                <div class="kpi-value">{kpis['AvgLength'][1]}</div>
                <div class="kpi-title">Average Length of Incident</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Incident types bar chart (unique incidents) ----
    st.markdown("### Incidents by type (unique disasters)")
    if {"incidentType", "disasterNumber"}.issubset(df_f.columns):
        tmp = (
            df_f[["incidentType", "disasterNumber"]]
            .drop_duplicates(subset=["disasterNumber"])
            .assign(incidentType=lambda d: d["incidentType"].fillna("Unknown"))
            .groupby("incidentType", as_index=False)
            .size()
            .rename(columns={"size": "unique_incidents"})
            .sort_values("unique_incidents", ascending=False)
        )
        # Pretty labels
        tmp["incidentType"] = tmp["incidentType"].str.title()

        fig_bar = px.bar(
            tmp,
            x="incidentType",
            y="unique_incidents",
            text="unique_incidents",
            labels={"incidentType": "Incident Type", "unique_incidents": "Unique Disasters"},
            title=None,
        )

        # layout polish
        fig_bar.update_traces(textposition="outside", cliponaxis=False)
        fig_bar.update_layout(
            xaxis_title=None,
            yaxis_title="Unique Disasters",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_tickangle=-25,
            height=450,
        )

        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.caption("No suitable columns (`incidentType`, `disasterNumber`) available for charting.")

    # ---- Preview data section ----
    with st.expander("Preview data"):
        st.dataframe(df_f.head(75), use_container_width=True)

# ---- Map
with tab2:
    st.markdown("### Disaster location heatmap")
    src_col = pick_fips_source(df_f)
    if fips_lookup is None:
        st.caption("Heatmap disabled (FIPS→lat/lon lookup not available).")
    elif src_col is None:
        st.caption("Heatmap disabled (missing FIPS in FEMA data).")
    else:
        geo_df = df_f.copy()
        # dedicated normalized key
        geo_df["fips_join"] = (
            geo_df[src_col].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(5)
        )
        # merge
        merged = geo_df.merge(
            fips_lookup[["fipsCountyCode", "latitude", "longitude"]],
            left_on="fips_join",
            right_on="fipsCountyCode",
            how="left",
        ).dropna(subset=["latitude", "longitude"])

        # --- Pick a stable FIPS column to group by, regardless of suffixing ---
    candidates = ["fipsCountyCode", "fipsCountyCode_y", "fipsCountyCode_x", "fips_join", "fipsCountyCodeFull"]
    fips_col = next((c for c in candidates if c in merged.columns), None)

    if fips_col is None:
        st.error("FIPS column not found after merge. Available columns: " + ", ".join(map(str, merged.columns)))
    else:
        # normalize to 5-digit numeric string
        merged["fips_for_group"] = (
            merged[fips_col].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(5)
        )

        # Aggregate once per county (weight by count)
        agg = (
            merged.groupby(["fips_for_group", "latitude", "longitude"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )

        center_lat = float(agg["latitude"].mean())
        center_lon = float(agg["longitude"].mean())

        tooltip = {"html": "<b>FIPS:</b> {fips_for_group}<br/><b>Count:</b> {count}",
                "style": {"color": "white"}}

        layer = pdk.Layer(
            "HeatmapLayer",
            data=agg,
            get_position='[longitude, latitude]',
            get_weight="count",
            radiusPixels=40,
            intensity=1.0,
            threshold=0.02,
        )

        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6 if state_in else 4, pitch=0)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip),
                        use_container_width=True)

        with st.expander("Heatmap data (county aggregates)"):
            st.dataframe(agg.sort_values("count", ascending=False).head(100), use_container_width=True)

# ---- Trends
with tab3:
    st.markdown("### Trends")
    if "declarationDate" in df_f.columns:
        df_f["month"] = pd.to_datetime(df_f["declarationDate"]).dt.to_period("M").dt.to_timestamp()
        ts = df_f.groupby("month", as_index=False).size()
        fig = px.line(ts, x="month", y="size", markers=True,
                      labels={"size":"Declarations","month":"Month"},
                      title="Declarations per month")
        st.plotly_chart(fig, use_container_width=True)

        if "incidentType" in df_f.columns:
            st.markdown("#### By incident type")
            ts2 = (df_f.groupby(["month","incidentType"], as_index=False)
                        .size().rename(columns={"size":"count"}))
            fig2 = px.area(ts2, x="month", y="count", color="incidentType",
                           title="Monthly declarations by incident type", groupnorm=None)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("No declarationDate column detected.")

# ---- Data
with tab4:
    st.markdown("### Data & export")
    st.download_button(
        "Download filtered data (CSV)",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name=f"fema_filtered_{(state_in or 'ALL').upper()}_{pd.Timestamp.today().date()}.csv",
        mime="text/csv"
    )

    with st.expander("Methods & Notes"):
        st.markdown("""
- **Source:** FEMA OpenFEMA `DisasterDeclarationsSummaries` API  
- **Window:** Start date → today (EOD UTC)  
- **KPIs:**  
  - *Total Declarations* = record count after filters  
  - *New in 30 Days* = records with `declarationDate ≥ today−30` (Δ vs. prior-30)  
  - *Avg Days Since* = mean(`today − declarationDate`)  
- **Geography:** county centroids from Census Gazetteer; points aggregated by 5-digit FIPS.  
- **Refresh:** cache-busted manual refresh with graceful error handling.
""")
    st.dataframe(df_f, use_container_width=True)


# ---- WRITE URL PARAMS (new API) ----
# Only strings or list[str] are supported in query params.
st.query_params["state"] = state_in or ""
st.query_params["start"] = str(pd.to_datetime(start_input).date())

# Optional FY params
if fy:
    st.query_params["fy_min"] = str(fy[0])
    st.query_params["fy_max"] = str(fy[1])
else:
    # Clean up if not filtering by FY
    if "fy_min" in st.query_params: del st.query_params["fy_min"]
    if "fy_max" in st.query_params: del st.query_params["fy_max"]