"""
app.py  —  Energy Management Dashboard (Phase 3)
=================================================
Tabs: HVAC Control | LED Lighting | Renewables (PV/Solar HW/Wind) |
      Fabric & Envelope | Solar Gain & Overheating | Full Summary
Run: python -m streamlit run app.py
"""
from __future__ import annotations

import io
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config as C
import data_generator as dg
import weather as wx
from building_profiles import PROFILES, get_profile
from energy_model import train, predict_annual_kwh
from hvac_optimizer import SCENARIOS, PRETTY, _npv
from report import generate_pdf
import lighting as lt
import meter_regression as mr
import epc_lookup as epc
import solar_pv as spv
import solar_thermal as sth
import fabric as fab
import wind as wd
import solar_gain as sgain
import building_3d as b3d
import planning_check as pc

st.set_page_config(
    page_title="Michelle's Project – Energy Management Dashboard",
    layout="wide",
)

PALETTE = ["#1f4e79", "#2e75b6", "#5b9bd5", "#9dc3e6", "#c55a11", "#70ad47", "#ffc000"]


# ============================================================================
# Financial helpers (Phase 2 upgrades)
# ============================================================================
def _irr_pct(
    annual_saving: float,
    capex_net: float,
    lifetime: int,
    degradation: float,
    maintenance: float,
    inflation: float,
) -> float:
    """IRR (%) using numpy-financial if available; NaN otherwise."""
    try:
        import numpy_financial as npf
        flows = [-capex_net]
        for y in range(1, lifetime + 1):
            net = annual_saving * ((1 + inflation) * (1 - degradation)) ** y - maintenance
            flows.append(max(net, 0))
        val = npf.irr(flows)
        return float(val) * 100 if (val is not None and not math.isnan(float(val))) else float("nan")
    except ImportError:
        return float("nan")


def _npv_inflated(
    annual_saving: float,
    capex_net: float,
    lifetime: int,
    rate: float,
    degradation: float,
    maintenance: float,
    inflation: float,
) -> float:
    """NPV with annual price inflation and equipment degradation."""
    total = -capex_net
    for y in range(1, lifetime + 1):
        net_y = annual_saving * ((1 + inflation) * (1 - degradation)) ** y - maintenance
        total += net_y / (1 + rate) ** y
    return total


# ============================================================================
# Cached data loaders
# ============================================================================
@st.cache_data(show_spinner="Fetching weather data…", ttl=86_400)
def _get_weather(location: str, sim_years: int):
    try:
        lat, lon, name = wx.geocode(location)
        df = wx.fetch_weather(lat, lon, sim_years)
        return df, name, None
    except Exception as exc:
        return None, location, str(exc)


@st.cache_data(show_spinner=False, ttl=86_400)
def _get_latlon(location: str) -> tuple[float, float]:
    """Latitude/longitude for solar position calculations (falls back to London)."""
    try:
        lat, lon, _ = wx.geocode(location)
        return lat, lon
    except Exception:
        return 51.5, -0.12


@st.cache_data(
    show_spinner="Generating data and training surrogate model – first run only…"
)
def _load_synthetic(building_type: str, floor_area: int,
                    location: str, sim_years: int,
                    hvac_target_override: float | None = None):
    weather_df, location_name, wx_error = (None, location, None)
    if location.strip():
        weather_df, location_name, wx_error = _get_weather(location.strip(), sim_years)

    data = dg.generate(years=sim_years, weather_df=weather_df,
                       profile_name=building_type, floor_area=floor_area)

    # If a calibrated HVAC target comes from meter regression, rescale
    if hvac_target_override and hvac_target_override > 0:
        raw_annual = data["hvac_kwh"].sum() / sim_years
        scale_adj  = hvac_target_override / raw_annual if raw_annual > 0 else 1.0
        data["hvac_kwh"] *= scale_adj
        data.attrs["scale"] *= scale_adj

    train_df = dg.make_training_data(data)
    model, metrics, _ = train(train_df)
    energy = _run_scenarios(model, data, sim_years)
    return energy, metrics, data, location_name, wx_error, weather_df


@st.cache_data(show_spinner="Parsing upload and training model on your data…")
def _load_uploaded(file_bytes: bytes, building_type: str, floor_area: int):
    df, warnings = _parse_upload(file_bytes, building_type, floor_area)
    sim_years    = max(1, round(len(df) / 8760))
    train_df     = dg.make_training_data(df)
    model, metrics, _ = train(train_df)
    energy = _run_scenarios(model, df, sim_years)
    return energy, metrics, df, warnings, None


# ============================================================================
# Shared helpers
# ============================================================================
def _run_scenarios(model, df: pd.DataFrame, sim_years: int) -> dict:
    baseline_kwh = predict_annual_kwh(model, SCENARIOS["baseline"](df), sim_years)
    energy = {}
    for key, builder in SCENARIOS.items():
        kwh   = predict_annual_kwh(model, builder(df), sim_years)
        saved = max(baseline_kwh - kwh, 0)
        energy[key] = {
            "scenario":   PRETTY[key],
            "annual_kwh": kwh,
            "saved_kwh":  saved,
            "saving_pct": max(saved / baseline_kwh * 100, 0),
        }
    return energy


def _parse_upload(file_bytes: bytes, building_type: str, floor_area: int):
    df = pd.read_csv(io.BytesIO(file_bytes))
    warnings: list[str] = []
    time_cols = {"hour", "dayofweek", "month", "is_workday"}
    has_ts        = "timestamp" in df.columns
    has_time_cols = time_cols.issubset(df.columns)
    if not has_ts and not has_time_cols:
        raise ValueError(
            "CSV must contain a 'timestamp' column, "
            "or all four of: hour, dayofweek, month, is_workday."
        )
    if has_ts:
        ts = pd.to_datetime(df["timestamp"])
        df["timestamp"] = ts
        if "hour"       not in df.columns: df["hour"]      = ts.dt.hour
        if "dayofweek"  not in df.columns: df["dayofweek"] = ts.dt.dayofweek
        if "month"      not in df.columns: df["month"]     = ts.dt.month
        if "is_workday" not in df.columns:
            df["is_workday"] = ts.dt.dayofweek.isin({0,1,2,3,4}).astype(int)
        ts_idx = pd.DatetimeIndex(ts)
    else:
        ts_idx = None
    if "hvac_kwh" not in df.columns:
        raise ValueError("CSV must contain a 'hvac_kwh' column.")
    if "t_out"   not in df.columns: df["t_out"]  = 10.0
    if "solar"   not in df.columns:
        df["solar"] = dg._solar_gain(ts_idx) if ts_idx is not None else 0.3
    if "occ_frac" not in df.columns:
        df["occ_frac"] = np.where(
            (df["hour"].between(8, 17)) & (df["is_workday"] == 1), 0.7, 0.05
        )
        warnings.append("'occ_frac' not found – using default 08:00–18:00 schedule.")
    if "system_on" not in df.columns:
        df["system_on"] = np.where(
            (df["hour"].between(6, 18)) & (df["is_workday"] == 1), 1.0, 0.0
        )
    for col, default in [
        ("cool_set", C.COOLING_SETPOINT), ("heat_set", C.HEATING_SETPOINT),
        ("deadband", C.DEADBAND), ("fan_factor", 1.0), ("vent_frac", 1.0),
    ]:
        if col not in df.columns: df[col] = default
    raw   = dg.hvac_energy_kw(df.t_out, df.solar, df.occ_frac, df.system_on,
                               df.cool_set, df.heat_set, df.deadband,
                               df.fan_factor, df.vent_frac)
    raw_t = float(raw.sum()); real_t = float(df["hvac_kwh"].sum())
    df.attrs["scale"]      = (real_t / raw_t) if raw_t > 0 else 1.0
    df.attrs["floor_area"] = floor_area
    df.attrs["profile"]    = building_type
    return df, warnings


@st.cache_data(show_spinner=False)
def _sample_xlsx_bytes() -> bytes:
    data   = dg.generate(years=1)
    sample = data.head(48)[
        ["timestamp", "t_out", "solar", "occ_frac", "system_on",
         "cool_set", "heat_set", "deadband", "fan_factor", "vent_frac", "hvac_kwh"]
    ].copy()
    sample["timestamp"] = sample["timestamp"].astype(str)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        sample.to_excel(writer, sheet_name="Sample data", index=False)
    return buf.getvalue()


def _confidence_badge(data_source: str, location: str) -> None:
    """Limitation 1 — show data quality / screening disclaimer."""
    if data_source == "Upload my own CSV":
        quality, icon = "Your uploaded data", "✅"
        detail = "Results are based on your uploaded hourly CSV. This improves accuracy but does not constitute a formal energy certificate."
    elif data_source == "Monthly electricity bills":
        quality, icon = "Meter regression (IPMVP Option B)", "📊"
        detail = "HVAC baseline derived by degree-day regression on your monthly bills. Directionally accurate; formal measurement (Option A) is required for M&V compliance."
    else:
        quality, icon = "Synthetic / modelled data", "⚠️"
        detail = "No real meter data supplied. Results use a machine-learning surrogate calibrated to building-type benchmarks."
    loc_note = f" Location: {location}." if location.strip() else " No location set — using default weather."
    st.info(
        f"{icon} **Screening tool** — {quality}.{loc_note} "
        f"{detail} "
        "Results are indicative and should not be used as formal compliance evidence "
        "without review by a qualified engineer. "
        "[CIBSE guidance](https://www.cibse.org) | "
        "[Non-domestic EPC register](https://epc.opendatacommunities.org)",
        icon="ℹ️",
    )


@st.cache_data(show_spinner="Checking planning constraints…", ttl=3_600)
def _check_planning(lat: float, lon: float) -> dict:
    """Cached planning constraint check — re-runs at most once per hour."""
    return pc.check(lat, lon)


def _compute_results(energy, elec_price, carbon_factor, discount_rate, lifetime,
                     capex_map, degradation_rate, maintenance_map,
                     inflation_rate=0.0, grant_pct=0.0, thermal_mass_factor=1.0):
    rows = []
    for key, e in energy.items():
        saved       = e["saved_kwh"] * thermal_mass_factor
        cost        = saved * elec_price
        carbon      = saved * carbon_factor / 1000
        capex       = capex_map.get(key, 0)
        capex_net   = capex * (1 - grant_pct / 100)
        maintenance = maintenance_map.get(key, 0)
        payback = capex_net / cost if (capex_net > 0 and cost > 0) else np.nan
        npv     = _npv_inflated(cost, capex_net, lifetime, discount_rate,
                                degradation_rate, maintenance, inflation_rate) \
                  if capex > 0 else np.nan
        irr     = _irr_pct(cost, capex_net, lifetime,
                            degradation_rate, maintenance, inflation_rate) \
                  if capex > 0 else np.nan
        rows.append({
            "key":                      key,
            "Strategy":                 e["scenario"],
            "Annual HVAC (kWh/yr)":     e["annual_kwh"] * thermal_mass_factor,
            "Saved (kWh/yr)":           saved,
            "Saving (%)":               e["saving_pct"],
            "Cost saving (GBP/yr)":     cost,
            "Carbon saving (tCO2e/yr)": carbon,
            "CAPEX (GBP)":              capex,
            "Payback (yrs)":            payback,
            "NPV (GBP)":                npv,
            "IRR (%)":                  irr,
        })
    return pd.DataFrame(rows)


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("Building settings")

    building_type = st.selectbox(
        "Building type",
        options=list(PROFILES.keys()),
        format_func=lambda k: PROFILES[k]["label"],
        index=0,
    )
    profile = get_profile(building_type)
    st.caption(f"{profile['description']}")
    if profile["is_247"]:
        st.info("24/7 building — occupancy scheduling saving is minimal.", icon="ℹ️")

    floor_area = st.slider("Floor area (m²)", 200, 100_000, C.FLOOR_AREA_M2, 200)

    location = st.text_input(
        "Location (UK postcode or city)", value="London",
        placeholder="e.g. EC1A 1BB or Manchester",
    )

    with st.expander("EPC information"):
        epc_band_input = st.selectbox(
            "Current EPC band",
            ["A+", "A", "B", "C", "D", "E", "F", "G"],
            index=3,
        )
        heating_fuel = st.selectbox(
            "Primary heating fuel",
            list(epc.FUEL_CO2_FACTORS.keys()),
            index=1,  # gas
            format_func=lambda k: {
                "electricity": "Electricity", "gas": "Natural gas",
                "oil": "Oil", "lpg": "LPG",
                "heat_pump": "Heat pump (electricity)",
                "biomass": "Biomass", "district_heat": "District heating",
            }.get(k, k.title()),
            help="Used for the SBEM-like EPC score calculation in the Full Summary tab.",
        )
        epc_api_key = st.text_input("EPC API key (optional)", type="password",
                                    help="Free from epc.opendatacommunities.org — "
                                         "leave blank to use manual band above.")
        epc_email   = st.text_input("EPC API email (optional)",
                                    help="Email registered with the EPC portal.")

    # ── Construction type (Limitation 3 — thermal mass fix) ─────────────────
    with st.expander("Construction type (thermal mass)"):
        construction_type = st.selectbox(
            "Construction type",
            list(C.THERMAL_MASS.keys()),
            format_func=lambda k: C.THERMAL_MASS[k]["label"],
            help="Affects the HVAC baseline — heavier construction naturally reduces "
                 "HVAC demand through thermal buffering.",
        )
        st.caption(C.THERMAL_MASS[construction_type]["description"])
        if construction_type != "lightweight":
            factor = C.THERMAL_MASS[construction_type]["hvac_factor"]
            st.caption(
                f"HVAC baseline correction: ×{factor:.2f} "
                f"({(1-factor)*100:.0f}% lower than lightweight equivalent)."
            )

    st.divider()

    # ── Data source ─────────────────────────────────────────────────────────
    st.header("Data source")
    data_source = st.radio(
        "data_source",
        ["Synthetic (built-in)", "Upload my own CSV", "Monthly electricity bills"],
        label_visibility="collapsed",
    )
    uploaded_file   = None
    monthly_kwh_in  = None

    if data_source == "Upload my own CSV":
        uploaded_file = st.file_uploader("Upload hourly CSV", type="csv")
        st.download_button(
            "Download sample file (.xlsx)",
            data=_sample_xlsx_bytes(),
            file_name="hvac_sample.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    elif data_source == "Monthly electricity bills":
        with st.expander("Enter 12 monthly totals (kWh)", expanded=True):
            st.caption("Total site electricity — the regression separates HVAC from base load.")
            month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
            monthly_kwh_in = []
            cols = st.columns(2)
            for i, m in enumerate(month_names):
                with cols[i % 2]:
                    val = st.number_input(m, min_value=0, max_value=10_000_000,
                                         value=0, step=1_000, key=f"bill_{i}")
                    monthly_kwh_in.append(float(val))

    st.divider()

    # ── Economic assumptions ─────────────────────────────────────────────────
    st.header("Economic assumptions")
    elec_price    = st.slider("Electricity (GBP/kWh)",   0.10, 0.60, C.ELECTRICITY_PRICE,    0.01)
    gas_price     = st.slider("Gas (GBP/kWh)",           0.02, 0.20, C.GAS_PRICE,            0.005)
    carbon_factor = st.slider("Carbon factor (kgCO2e/kWh)", 0.05, 0.50, C.CARBON_FACTOR,     0.001)
    discount_rate = st.slider("Discount rate (%)",        1,   15,   int(C.DISCOUNT_RATE*100), 1) / 100
    lifetime      = st.slider("HVAC measure lifetime (yrs)", 5, 30, C.MEASURE_LIFETIME_YEARS,  1)
    inflation_rate= st.slider("Energy price inflation (%/yr)", 0.0, 8.0,
                               float(C.INFLATION_RATE * 100), 0.1) / 100
    grant_pct     = st.slider("Grant / subsidy (%)",     0, 50, int(C.GRANT_PCT),  5)

    st.divider()

    # ── Degradation & Maintenance ────────────────────────────────────────────
    st.header("Degradation & maintenance")
    degradation_rate = st.slider(
        "Equipment degradation (%/yr)",
        0.0, 3.0, float(C.DEGRADATION_RATE * 100), 0.1,
    ) / 100
    with st.expander("Annual maintenance costs (GBP/yr)"):
        maint_occ   = st.number_input("Occupancy scheduling",       0, 50_000, C.MAINTENANCE_COSTS["occupancy_scheduling"], 500)
        maint_therm = st.number_input("Smart thermostats",          0, 50_000, C.MAINTENANCE_COSTS["smart_thermostats"],    500)
        maint_bas   = st.number_input("Building Automation System", 0, 50_000, C.MAINTENANCE_COSTS["bas"],                  500)
        maint_comb  = st.number_input("Combined strategy",          0, 50_000, C.MAINTENANCE_COSTS["combined"],             500)

    st.divider()

    # ── HVAC Capital costs ───────────────────────────────────────────────────
    st.header("HVAC capital costs (GBP)")
    capex_occ   = st.slider("Occupancy scheduling",        10_000, 100_000, C.CAPEX["occupancy_scheduling"], 1_000)
    capex_therm = st.slider("Smart thermostats",            5_000,  80_000, C.CAPEX["smart_thermostats"],    1_000)
    capex_bas   = st.slider("Building Automation System",  50_000, 300_000, C.CAPEX["bas"],                  5_000)
    capex_comb  = st.slider("Combined strategy",           50_000, 400_000, C.CAPEX["combined"],             5_000)


# ============================================================================
# LOAD DATA
# ============================================================================
st.title("Michelle's Project – Energy Management Dashboard")

upload_warnings: list[str] = []
wx_warning: str | None = None
weather_df_for_solar: pd.DataFrame | None = None
regression_result: dict | None = None

# Handle monthly bills regression
hvac_target_override: float | None = None
if data_source == "Monthly electricity bills" and monthly_kwh_in:
    non_zero = [v for v in monthly_kwh_in if v > 0]
    if len(non_zero) >= 6:
        # Get monthly HDD/CDD from weather if available
        wx_df, wx_name, wx_err = _get_weather(location.strip() or "London", C.SIM_YEARS) \
            if location.strip() else (None, "London", None)
        if wx_df is not None:
            monthly_dd = mr.degree_days_from_weather(wx_df)
            # Average across years if multi-year weather
            dd_avg = monthly_dd.groupby("month")[["hdd", "cdd"]].mean()
            hdds   = [dd_avg.loc[m, "hdd"] if m in dd_avg.index else 0 for m in range(1, 13)]
            cdds   = [dd_avg.loc[m, "cdd"] if m in dd_avg.index else 0 for m in range(1, 13)]
        else:
            hdds, cdds = mr.degree_days_from_monthly_temps(C.MONTHLY_MEAN_TEMP)
        try:
            regression_result = mr.regress(non_zero, hdds[:len(non_zero)], cdds[:len(non_zero)])
            hvac_target_override = regression_result.get("annual_hvac_kwh")
        except Exception as e:
            st.warning(f"Regression error: {e}")

if data_source == "Upload my own CSV":
    if not uploaded_file:
        st.info("Upload a CSV file in the sidebar to get started.")
        st.stop()
    try:
        energy, metrics, raw_data, upload_warnings, weather_df_for_solar = _load_uploaded(
            uploaded_file.getvalue(), building_type, floor_area
        )
    except ValueError as exc:
        st.error(f"Could not read your file: {exc}")
        st.stop()
    data_label = (
        f"Uploaded: {uploaded_file.name}  |  {len(raw_data):,} rows  |  "
        f"{PROFILES[building_type]['label']}  |  {floor_area:,} m²"
    )
else:
    sim_years = C.SIM_YEARS
    energy, metrics, raw_data, location_name, wx_error, weather_df_for_solar = _load_synthetic(
        building_type, floor_area, location, sim_years, hvac_target_override
    )
    if wx_error:
        wx_warning = f"Weather fetch failed ({wx_error}) — using synthetic weather."
        loc_str = "Synthetic weather"
    else:
        loc_str = location_name or location
    target = C.HVAC_ENERGY_INTENSITY_KWH_M2 * floor_area * (
        PROFILES[building_type].get("hvac_kwh_m2", 100) / 100
    )
    if hvac_target_override:
        target = hvac_target_override
    data_label = (
        f"{PROFILES[building_type]['label']}  |  {floor_area:,} m²  |  "
        f"{loc_str}  |  baseline HVAC ~{target:,.0f} kWh/yr"
    )

for w in upload_warnings:
    st.warning(w)
if wx_warning:
    st.warning(wx_warning)
if profile["is_247"]:
    st.info(
        f"**24/7 building ({profile['label']})** — occupancy scheduling saving is minimal. "
        "BAS variable-speed drives and smart thermostats remain fully effective.",
        icon="ℹ️",
    )

st.caption(f"{data_label}  |  Surrogate model R² = {metrics['r2']:.4f}")

# Limitation 1 — screening tool disclaimer / data confidence badge
_confidence_badge(data_source, location)

# Limitation 3 — thermal mass correction factor
thermal_mass_factor: float = C.THERMAL_MASS[construction_type]["hvac_factor"]

# Prepare shared financial inputs
capex_map = {
    "occupancy_scheduling": capex_occ, "smart_thermostats": capex_therm,
    "bas": capex_bas, "combined": capex_comb,
}
maintenance_map = {
    "occupancy_scheduling": maint_occ, "smart_thermostats": maint_therm,
    "bas": maint_bas, "combined": maint_comb,
}

df_results = _compute_results(
    energy, elec_price, carbon_factor, discount_rate, lifetime,
    capex_map, degradation_rate, maintenance_map, inflation_rate, grant_pct,
    thermal_mass_factor=thermal_mass_factor,
)
non_bl = df_results[df_results["key"] != "baseline"].copy()
best   = non_bl.sort_values("Cost saving (GBP/yr)", ascending=False).iloc[0]

# Baseline HVAC kWh — scaled by thermal mass factor (Limitation 3)
baseline_hvac_kwh = energy["baseline"]["annual_kwh"] * thermal_mass_factor
total_building_kwh = baseline_hvac_kwh / max(PROFILES[building_type].get("hvac_kwh_m2", 100) / 100 * 0.45, 0.1)


# ============================================================================
# TABS
# ============================================================================
tab_hvac, tab_light, tab_solar, tab_fabric, tab_gain, tab_summary = st.tabs(
    ["HVAC Control", "LED Lighting", "Renewables", "Fabric & Envelope",
     "Solar Gain & Overheating", "Full Summary"]
)

# Initialise variables that are set inside tab blocks but referenced in later tabs,
# so they are always defined regardless of execution order or early errors.
zone_results: list = []


# ============================================================================
# TAB 1 — HVAC CONTROL
# ============================================================================
with tab_hvac:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best annual cost saving",  f"GBP {best['Cost saving (GBP/yr)']:,.0f}", best["Strategy"])
    c2.metric("Building type",            profile["label"])
    c3.metric("Floor area",               f"{floor_area:,} m²")
    c4.metric("Electricity price",        f"GBP {elec_price:.2f}/kWh")
    c5.metric("Degradation rate",         f"{degradation_rate * 100:.1f}%/yr")

    # Monthly bills regression info
    if regression_result:
        st.info("📊 **Degree-day regression from your monthly bills**")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Total electricity (yr)", f"{regression_result['annual_total_kwh']:,.0f} kWh")
        rc2.metric("Estimated HVAC", f"{regression_result['annual_hvac_kwh']:,.0f} kWh")
        rc3.metric("HVAC share", f"{regression_result['hvac_share_pct']:.0f}%")
        rc4.metric("Regression R²", f"{regression_result['r2']:.2f}")
        if regression_result.get("warning"):
            st.warning(regression_result["warning"])

    st.divider()
    st.subheader("Results by strategy")
    styled = (
        df_results.drop(columns=["key"])
        .style
        .format({
            "Annual HVAC (kWh/yr)":     "{:,.0f}",
            "Saved (kWh/yr)":           "{:,.0f}",
            "Saving (%)":               "{:.1f}",
            "Cost saving (GBP/yr)":     "{:,.0f}",
            "Carbon saving (tCO2e/yr)": "{:.1f}",
            "CAPEX (GBP)":              "{:,.0f}",
            "Payback (yrs)":            "{:.1f}",
            "NPV (GBP)":                "{:,.0f}",
            "IRR (%)":                  "{:.1f}",
        }, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    with st.expander("Download data"):
        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button("Hourly dataset",
            raw_data.to_csv(index=False).encode(), "hvac_hourly_data.csv", "text/csv")
        dl2.download_button("Scenario results",
            df_results.drop(columns=["key"]).to_csv(index=False).encode(),
            "hvac_scenario_results.csv", "text/csv")
        with dl3:
            if st.button("Generate PDF report", key="gen_pdf"):
                pdf_assumptions = {
                    "elec_price": elec_price, "carbon_factor": carbon_factor,
                    "discount_rate": discount_rate, "lifetime": lifetime,
                    "capex_occ": capex_occ, "capex_therm": capex_therm,
                    "capex_bas": capex_bas, "capex_comb": capex_comb,
                }
                st.session_state["pdf_bytes"] = generate_pdf(
                    df_results, pdf_assumptions, metrics, data_label
                )
            if "pdf_bytes" in st.session_state:
                st.download_button(
                    "Download PDF report",
                    data=st.session_state["pdf_bytes"],
                    file_name="hvac_optimisation_report.pdf",
                    mime="application/pdf",
                    key="dl_pdf",
                )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Energy saving by strategy")
        fig, ax = plt.subplots(figsize=(5, 3.8))
        bars = ax.bar(range(len(non_bl)), non_bl["Saving (%)"], color=PALETTE[1:1+len(non_bl)])
        ax.set_ylabel("HVAC energy saving (%)")
        ax.set_xticks(range(len(non_bl)))
        ax.set_xticklabels([s.replace(" ", "\n") for s in non_bl["Strategy"]], fontsize=8)
        for bar, val in zip(bars, non_bl["Saving (%)"]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                    f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with col2:
        st.subheader("Simple payback period")
        pb = non_bl.dropna(subset=["Payback (yrs)"])
        if pb.empty:
            st.info("No payback data to display.")
        else:
            fig2, ax2 = plt.subplots(figsize=(5, 3.8))
            ax2.barh(pb["Strategy"], pb["Payback (yrs)"], color=PALETTE[2])
            ax2.set_xlabel("Simple payback period (years)")
            ax2.invert_yaxis()
            for patch, val in zip(ax2.patches, pb["Payback (yrs)"]):
                ax2.text(val + 0.05, patch.get_y() + patch.get_height()/2,
                         f"{val:.1f} yr", va="center", fontsize=9)
            ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True); plt.close(fig2)

    with st.expander("Surrogate model performance"):
        m1, m2, m3 = st.columns(3)
        m1.metric("R²",   f"{metrics['r2']:.4f}")
        m2.metric("MAE",  f"{metrics['mae']:.2f} kWh/h")
        m3.metric("RMSE", f"{metrics['rmse']:.2f} kWh/h")
        st.caption(
            f"Building type: {profile['label']}  |  Floor area: {floor_area:,} m²  |  "
            f"Trained on {metrics['n_train']:,} rows, tested on {metrics['n_test']:,} rows.  "
            f"Mean target: {metrics['mean_target']:.2f} kWh/h."
        )


# ============================================================================
# TAB 2 — LED LIGHTING
# ============================================================================
with tab_light:
    st.subheader("LED Lighting Upgrade Analysis")
    st.markdown(
        "Configure the existing lighting in your building to calculate the "
        "saving from a full LED retrofit with optional controls."
    )

    # ── Inputs ──────────────────────────────────────────────────────────────
    li_c1, li_c2 = st.columns([2, 1])
    with li_c1:
        light_hours = st.slider(
            "Annual operating hours", 500, 8_760, C.LIGHTING_HOURS_PER_YEAR, 50,
            help="Hours per year the lights are on. Typical office: 2,250 h/yr.",
        )
        light_lifetime = st.slider("LED system lifetime (years)", 5, 30, C.LIGHTING_LIFETIME_YEARS, 1)
    with li_c2:
        st.info(
            f"**Operating hours guide**\n"
            f"- Office (9–17h, 5 days): ~1,800 h/yr\n"
            f"- Retail (7 days): ~4,200 h/yr\n"
            f"- Hospital (24/7): ~8,760 h/yr"
        )

    st.markdown("---")
    st.markdown("**Configure lighting zones** (up to 3 zones)")

    zone_results = []
    total_area_pct = 0.0

    for z_idx in range(1, 4):
        with st.expander(f"Zone {z_idx}" + (" (required)" if z_idx == 1 else " (optional)"),
                         expanded=(z_idx == 1)):
            enable_zone = True if z_idx == 1 else st.checkbox(f"Enable zone {z_idx}", key=f"en_{z_idx}")
            if enable_zone:
                zc1, zc2, zc3, zc4 = st.columns(4)
                with zc1:
                    fitting  = st.selectbox(
                        "Fitting type",
                        options=list(lt.FITTINGS.keys()),
                        format_func=lambda k: lt.FITTINGS[k]["label"],
                        key=f"fit_{z_idx}",
                    )
                with zc2:
                    area_pct = st.slider(
                        "Area covered (%)", 5, 100,
                        100 if z_idx == 1 else 30,
                        5, key=f"apct_{z_idx}",
                    )
                with zc3:
                    control  = st.selectbox(
                        "Control upgrade",
                        options=list(lt.CONTROLS.keys()),
                        format_func=lambda k: lt.CONTROLS[k]["label"],
                        key=f"ctrl_{z_idx}",
                    )
                with zc4:
                    capex_ov = st.number_input(
                        "CAPEX override (GBP, 0 = auto)",
                        min_value=0, max_value=5_000_000, value=0, step=1_000,
                        key=f"capx_{z_idx}",
                        help="Leave 0 to use the standard installed cost rate."
                    )

                r = lt.calculate(
                    floor_area    = floor_area,
                    fitting_type  = fitting,
                    control_type  = control,
                    area_fraction = area_pct / 100,
                    operating_hours = light_hours,
                    elec_price    = elec_price,
                    capex_override= capex_ov if capex_ov > 0 else None,
                )
                r["zone"] = f"Zone {z_idx}"
                zone_results.append(r)
                total_area_pct += area_pct

    if zone_results:
        # ── Results ─────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Lighting results")
        totals = lt.total_saving(zone_results)

        km1, km2, km3, km4 = st.columns(4)
        km1.metric("Total energy saving", f"{totals['total_saving_kwh']:,.0f} kWh/yr",
                   f"{totals['total_saving_pct']:.0f}% reduction")
        km2.metric("Annual cost saving",  f"GBP {totals['total_cost_saving']:,.0f}")
        km3.metric("Total CAPEX",         f"GBP {totals['total_capex']:,.0f}")
        km4.metric("Simple payback",
                   f"{totals['payback_years']:.1f} yrs" if not math.isnan(totals['payback_years']) else "—")

        # NPV for lighting
        capex_net_light = totals["total_capex"] * (1 - grant_pct / 100)
        npv_light = _npv_inflated(
            totals["total_cost_saving"], capex_net_light,
            light_lifetime, discount_rate, 0.01, 0, inflation_rate
        )
        irr_light = _irr_pct(
            totals["total_cost_saving"], capex_net_light,
            light_lifetime, 0.01, 0, inflation_rate
        )

        nk1, nk2, nk3 = st.columns(3)
        nk1.metric(f"NPV ({light_lifetime} yr)", f"GBP {npv_light:,.0f}")
        nk2.metric("IRR", f"{irr_light:.1f}%" if not math.isnan(irr_light) else "—")
        nk3.metric("Carbon saving",
                   f"{totals['total_saving_kwh'] * C.CARBON_FACTOR / 1000:.1f} tCO2e/yr")

        # Zone breakdown table
        light_table = pd.DataFrame([{
            "Zone":              r["zone"],
            "Fitting type":      r["fitting_label"],
            "Controls":          r["control_label"],
            "Lit area (m²)":     r["lit_area_m2"],
            "Baseline (kWh/yr)": r["baseline_kwh"],
            "After LED (kWh/yr)":r["led_kwh"],
            "Saving (kWh/yr)":   r["saving_kwh"],
            "Saving (%)":        r["saving_pct"],
            "Cost saving (GBP/yr)": r["cost_saving_gbp"],
            "CAPEX (GBP)":       r["capex_gbp"],
            "Payback (yrs)":     r["payback_years"],
        } for r in zone_results])
        st.dataframe(
            light_table.style.format({
                "Lit area (m²)": "{:,.0f}", "Baseline (kWh/yr)": "{:,.0f}",
                "After LED (kWh/yr)": "{:,.0f}", "Saving (kWh/yr)": "{:,.0f}",
                "Saving (%)": "{:.1f}", "Cost saving (GBP/yr)": "{:,.0f}",
                "CAPEX (GBP)": "{:,.0f}", "Payback (yrs)": "{:.1f}",
            }, na_rep="—"),
            use_container_width=True, hide_index=True
        )

        # Bar chart
        fig, ax = plt.subplots(figsize=(7, 3.5))
        categories = [r["zone"] for r in zone_results]
        baseline_vals = [r["baseline_kwh"] for r in zone_results]
        led_vals      = [r["led_kwh"]      for r in zone_results]
        x = np.arange(len(categories))
        w = 0.35
        ax.bar(x - w/2, baseline_vals, w, label="Before LED", color=PALETTE[4])
        ax.bar(x + w/2, led_vals,      w, label="After LED",  color=PALETTE[2])
        ax.set_ylabel("Annual energy (kWh/yr)")
        ax.set_xticks(x); ax.set_xticklabels(categories)
        ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)


# ============================================================================
# TAB 3 — SOLAR ENERGY
# ============================================================================
with tab_solar:
    st.subheader("Renewable Energy Analysis")

    # ── Limitation 4 fix — planning constraint check ─────────────────────────
    with st.expander("Planning & grid connection check for this location", expanded=True):
        if not location.strip():
            st.info("Enter a postcode or city in the sidebar to run the planning check.")
            _plan = None
        else:
            _lat_p, _lon_p = _get_latlon(location.strip())
            _plan = _check_planning(_lat_p, _lon_p)
            if _plan["api_error"]:
                st.warning(
                    "Planning Data API could not be reached — results below are based "
                    "on general UK Permitted Development rules only."
                )
            if _plan["constraints"]:
                st.error(
                    f"Planning constraints detected near **{location.strip()}**: "
                    f"{', '.join(_plan['constraints'])}. "
                    "Additional consents are likely required — see details below."
                )
            else:
                st.success(
                    "No statutory planning designations detected within 200 m of "
                    f"**{location.strip()}**. Standard Permitted Development rules apply."
                )
            if _plan["restrictions"]:
                st.markdown("**Technology-specific planning notes:**")
                for note in _plan["restrictions"]:
                    st.markdown(f"- {note}")
            st.caption(
                "Source: Planning Data API (planning.data.gov.uk). "
                "This is a screening check only — always verify with your Local Planning Authority "
                "before committing to an installation."
            )

    sol_tab_pv, sol_tab_shw, sol_tab_wind = st.tabs(
        ["Solar Photovoltaic (PV)", "Solar Hot Water", "Wind Turbine"]
    )

    # ── SOLAR PV ─────────────────────────────────────────────────────────────
    with sol_tab_pv:
        st.markdown("Estimate annual PV generation, self-consumption, and export income.")
        pv_c1, pv_c2, pv_c3 = st.columns(3)
        with pv_c1:
            pv_capacity   = st.number_input("System capacity (kWp)", 1.0, 5000.0, 100.0, 10.0)
            pv_orientation= st.selectbox("Orientation",
                list(spv.ORIENTATION_FACTORS.keys()), index=0)
            pv_tilt       = st.select_slider("Tilt angle (°)", [0,10,15,20,30,35,40,45,50], value=30)
        with pv_c2:
            pv_export_tariff = st.slider("SEG export tariff (GBP/kWh)",
                                         0.04, 0.30, C.SOLAR_PV_EXPORT_TARIFF, 0.01)
            pv_capex_kwp  = st.slider("CAPEX (GBP/kWp)", 800, 2_000, 1_200, 50)
            pv_lifetime   = st.slider("System lifetime (years)", 10, 30, C.SOLAR_PV_LIFETIME, 1)
        with pv_c3:
            st.info(
                "**Typical UK PV outputs**\n"
                "- 100 kWp south 30°: ~90,000 kWh/yr\n"
                "- 250 kWp south 30°: ~225,000 kWh/yr\n"
                "- Performance ratio default: 80%"
            )

        if st.button("Calculate solar PV", type="primary", key="calc_pv"):
            pv_r = spv.financials(
                capacity_kwp          = pv_capacity,
                orientation           = pv_orientation,
                tilt_deg              = pv_tilt,
                building_type         = building_type,
                building_annual_load_kwh = baseline_hvac_kwh,
                elec_price            = elec_price,
                export_tariff         = pv_export_tariff,
                discount_rate         = discount_rate,
                inflation_rate        = inflation_rate,
                grant_pct             = float(grant_pct),
                weather_df            = weather_df_for_solar,
                lifetime_years        = pv_lifetime,
                capex_per_kwp         = float(pv_capex_kwp),
            )
            st.session_state["pv_result"] = pv_r

        if "pv_result" in st.session_state:
            pv_r = st.session_state["pv_result"]
            st.divider()
            pv_m1, pv_m2, pv_m3, pv_m4 = st.columns(4)
            pv_m1.metric("Annual generation",  f"{pv_r['annual_gen_kwh']:,.0f} kWh")
            pv_m2.metric("Self-consumed",       f"{pv_r['self_consumed_kwh']:,.0f} kWh")
            pv_m3.metric("Exported",            f"{pv_r['exported_kwh']:,.0f} kWh")
            pv_m4.metric("Annual income",       f"GBP {pv_r['total_annual_income_gbp']:,.0f}")

            pv_m5, pv_m6, pv_m7, pv_m8 = st.columns(4)
            pv_m5.metric("Total CAPEX",         f"GBP {pv_r['total_capex_gbp']:,.0f}")
            pv_m6.metric("Payback",
                         f"{pv_r['payback_years']:.1f} yrs" if not math.isnan(pv_r['payback_years']) else "—")
            pv_m7.metric(f"NPV ({pv_lifetime} yr)", f"GBP {pv_r['npv_gbp']:,.0f}")
            pv_m8.metric("IRR",
                         f"{pv_r['irr_pct']:.1f}%" if not math.isnan(pv_r['irr_pct']) else "—")

            st.metric("Carbon avoided", f"{pv_r['carbon_saved_tco2e']:.1f} tCO2e/yr")

            # Income breakdown donut
            fig, ax = plt.subplots(figsize=(4, 3))
            vals   = [pv_r["import_saved_gbp"], pv_r["export_income_gbp"]]
            labels = [f"Import saving\nGBP {vals[0]:,.0f}",
                      f"Export income\nGBP {vals[1]:,.0f}"]
            ax.pie(vals, labels=labels, colors=[PALETTE[1], PALETTE[5]],
                   autopct="%1.0f%%", startangle=90)
            ax.set_title("Annual income breakdown")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

            # Grid connection notes (Limitation 4)
            grid_notes = pc.grid_connection_notes(capacity_kwp=pv_capacity)
            if grid_notes:
                with st.expander("Grid connection guidance (DNO)", expanded=False):
                    for note in grid_notes:
                        st.markdown(f"- {note}")

    # ── SOLAR HOT WATER ───────────────────────────────────────────────────────
    with sol_tab_shw:
        st.markdown("Estimate solar thermal yield and displacement of gas or electricity for DHW.")
        shw_c1, shw_c2 = st.columns(2)
        with shw_c1:
            shw_area       = st.number_input("Collector area (m²)", 1.0, 2000.0, 50.0, 5.0)
            shw_orientation= st.selectbox("Orientation",
                list(sth.ORIENTATION_FACTORS.keys()), index=0, key="shw_ori")
            shw_fuel       = st.selectbox("Existing fuel for hot water",
                ["gas", "electricity"], format_func=str.capitalize)
            shw_capex_m2   = st.slider("CAPEX (GBP/m² collector)", 300, 1_000, 650, 25)
            shw_lifetime   = st.slider("System lifetime (years)", 10, 25, C.SOLAR_THERMAL_LIFETIME, 1)
        with shw_c2:
            dhw_intensity = C.DHW_INTENSITY_KWH_M2.get(building_type, 5)
            dhw_demand    = dhw_intensity * floor_area
            st.info(
                f"**Building DHW demand**\n\n"
                f"- Intensity ({profile['label']}): **{dhw_intensity} kWh/m²/yr**\n"
                f"- Total demand: **{dhw_demand:,.0f} kWh/yr**\n\n"
                f"Practical solar fraction maximum: **70%**"
            )

        if st.button("Calculate solar hot water", type="primary", key="calc_shw"):
            shw_r = sth.financials(
                collector_area_m2 = shw_area,
                floor_area        = floor_area,
                building_type     = building_type,
                orientation       = shw_orientation,
                fuel_type         = shw_fuel,
                elec_price        = elec_price,
                gas_price         = gas_price,
                discount_rate     = discount_rate,
                inflation_rate    = inflation_rate,
                grant_pct         = float(grant_pct),
                weather_df        = weather_df_for_solar,
                lifetime_years    = shw_lifetime,
                capex_per_m2      = float(shw_capex_m2),
            )
            st.session_state["shw_result"] = shw_r

        if "shw_result" in st.session_state:
            shw_r = st.session_state["shw_result"]
            st.divider()
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Annual solar yield",  f"{shw_r['net_yield_kwh']:,.0f} kWh")
            sm2.metric("Solar fraction",      f"{shw_r['solar_fraction_pct']:.0f}%")
            sm3.metric("Annual cost saving",  f"GBP {shw_r['cost_saving_gbp']:,.0f}")
            sm4.metric("Carbon saving",       f"{shw_r['carbon_saved_tco2e']:.1f} tCO2e/yr")

            sm5, sm6, sm7 = st.columns(3)
            sm5.metric("Total CAPEX",  f"GBP {shw_r['total_capex_gbp']:,.0f}")
            sm6.metric("Payback",
                       f"{shw_r['payback_years']:.1f} yrs" if not math.isnan(shw_r['payback_years']) else "—")
            sm7.metric(f"NPV ({shw_lifetime} yr)", f"GBP {shw_r['npv_gbp']:,.0f}")

    # ── WIND TURBINE ───────────────────────────────────────────────────────────
    with sol_tab_wind:
        st.markdown(
            "Estimate generation from a small wind turbine using hourly wind speed "
            "at the building's location, height-corrected to hub height."
        )
        w_c1, w_c2, w_c3 = st.columns(3)
        with w_c1:
            wind_capacity = st.number_input("Turbine rated capacity (kW)", 1.0, 500.0, 10.0, 1.0)
            wind_type = st.selectbox(
                "Turbine type",
                options=list(wd.TURBINE_TYPES.keys()),
                format_func=lambda k: wd.TURBINE_TYPES[k]["label"],
            )
            wind_hub_height = st.slider("Hub height (m)", 5, 60, 15, 1)
        with w_c2:
            wind_export_tariff = st.slider(
                "SEG export tariff (GBP/kWh)", 0.04, 0.30, C.SOLAR_PV_EXPORT_TARIFF, 0.01,
                key="wind_seg",
            )
            wind_capex_kw = st.slider(
                "CAPEX (GBP/kW)", 1_000, 5_000, wd.TURBINE_TYPES["small"]["capex_per_kw"], 100,
            )
            wind_lifetime = st.slider(
                "System lifetime (years)", 10, 25, C.WIND_LIFETIME_YEARS, 1, key="wind_life",
            )
        with w_c3:
            st.info(
                "**Site suitability**\n"
                "Mean wind speed below 5 m/s at hub height is rarely cost-effective "
                "for small wind.\n\n"
                "Planning permission is normally required for building-mounted and "
                "freestanding turbines."
            )

        if st.button("Calculate wind generation", type="primary", key="calc_wind"):
            wind_r = wd.financials(
                rated_capacity_kw        = wind_capacity,
                hub_height_m             = wind_hub_height,
                turbine_type             = wind_type,
                building_type            = building_type,
                building_annual_load_kwh = baseline_hvac_kwh,
                elec_price               = elec_price,
                export_tariff            = wind_export_tariff,
                discount_rate            = discount_rate,
                inflation_rate           = inflation_rate,
                grant_pct                = float(grant_pct),
                weather_df               = weather_df_for_solar,
                lifetime_years           = wind_lifetime,
                capex_per_kw             = float(wind_capex_kw),
            )
            st.session_state["wind_result"] = wind_r

        if "wind_result" in st.session_state:
            wind_r = st.session_state["wind_result"]
            st.divider()
            if not wind_r["site_suitable"]:
                st.warning(
                    f"⚠️ Site suitability warning: mean wind speed at hub height is only "
                    f"{wind_r['mean_wind_speed_ms']:.1f} m/s (below the 5 m/s practical "
                    f"viability threshold). Payback is likely to be very long."
                )
            wm1, wm2, wm3, wm4 = st.columns(4)
            wm1.metric("Mean wind speed (hub)", f"{wind_r['mean_wind_speed_ms']:.1f} m/s")
            wm2.metric("Annual generation",     f"{wind_r['annual_gen_kwh']:,.0f} kWh")
            wm3.metric("Self-consumed",          f"{wind_r['self_consumed_kwh']:,.0f} kWh")
            wm4.metric("Annual income",          f"GBP {wind_r['total_annual_income_gbp']:,.0f}")

            wm5, wm6, wm7, wm8 = st.columns(4)
            wm5.metric("Total CAPEX", f"GBP {wind_r['total_capex_gbp']:,.0f}")
            wm6.metric("Payback",
                       f"{wind_r['payback_years']:.1f} yrs" if not math.isnan(wind_r['payback_years']) else "—")
            wm7.metric(f"NPV ({wind_lifetime} yr)", f"GBP {wind_r['npv_gbp']:,.0f}")
            wm8.metric("IRR",
                       f"{wind_r['irr_pct']:.1f}%" if not math.isnan(wind_r['irr_pct']) else "—")

            st.metric("Carbon avoided", f"{wind_r['carbon_saved_tco2e']:.1f} tCO2e/yr")
            # Grid connection notes for wind (Limitation 4)
            grid_notes_w = pc.grid_connection_notes(turbine_kw=wind_capacity)
            if grid_notes_w:
                with st.expander("Grid connection guidance (DNO)", expanded=False):
                    for note in grid_notes_w:
                        st.markdown(f"- {note}")


# ============================================================================
# TAB 4 — FABRIC & ENVELOPE
# ============================================================================
with tab_fabric:
    st.subheader("Fabric & Envelope Improvement Analysis")

    fb_c1, fb_c2 = st.columns([2, 1])
    with fb_c2:
        fab_fuel     = st.selectbox("Existing heating fuel",
                                    ["gas", "electricity"], format_func=str.capitalize)
        fab_hdd      = st.slider("Annual heating degree-days", 500, 4_000, C.HDD_ANNUAL, 50,
                                 help="Base 15.5°C. UK typical: 2,000–2,800.")
        fab_lifetime = st.slider("Measure lifetime (years)", 15, 50, C.FABRIC_LIFETIME, 1)

    with fb_c1:
        st.markdown("**Select elements to improve and enter areas:**")
        with st.expander("Pre-set retrofit packages", expanded=False):
            pkg = st.radio("Apply a package", ["None"] + list(fab.PACKAGES.keys()),
                           format_func=lambda k: fab.PACKAGES[k]["label"] if k != "None" else "None",
                           horizontal=True)
            if pkg != "None":
                st.caption(fab.PACKAGES[pkg]["description"])

    # Element configuration
    package_elements = fab.PACKAGES.get(pkg if pkg != "None" else "", {}).get("elements", [])
    element_data = []
    for key, defn in fab.ELEMENTS.items():
        is_in_package = key in package_elements
        with st.expander(
            f"{'✅ ' if is_in_package else ''}{defn['label']}",
            expanded=is_in_package
        ):
            enable = st.checkbox("Include this element", value=is_in_package, key=f"fab_en_{key}")
            if enable:
                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    area = st.number_input("Area (m²)", 0.0, 50_000.0, 500.0, 50.0, key=f"fab_a_{key}")
                with fc2:
                    u_old = st.number_input("Current U-value (W/m²K)",
                                            0.10, 8.0, float(defn["u_typical"]), 0.05,
                                            key=f"fab_uo_{key}")
                with fc3:
                    u_new = st.number_input("Target U-value (W/m²K)",
                                            0.05, 8.0, float(defn["u_target"]), 0.01,
                                            key=f"fab_un_{key}")
                st.caption(f"Typical CAPEX: £{defn['capex_m2']}/m²  |  {defn['description']}")
                element_data.append({"key": key, "area_m2": area, "u_old": u_old, "u_new": u_new})

    if element_data and st.button("Calculate fabric savings", type="primary", key="calc_fab"):
        fab_r = fab.calculate(
            elements       = element_data,
            hdd_annual     = fab_hdd,
            fuel_type      = fab_fuel,
            elec_price     = elec_price,
            gas_price      = gas_price,
            discount_rate  = discount_rate,
            inflation_rate = inflation_rate,
            grant_pct      = float(grant_pct),
            lifetime_years = fab_lifetime,
        )
        st.session_state["fab_result"] = fab_r

    if "fab_result" in st.session_state:
        fab_r = st.session_state["fab_result"]
        st.divider()
        st.subheader("Fabric results")
        fm1, fm2, fm3, fm4 = st.columns(4)
        fm1.metric("Total heating saving", f"{fab_r['total_saving_kwh']:,.0f} kWh/yr")
        fm2.metric("Annual cost saving",   f"GBP {fab_r['total_cost_saving']:,.0f}")
        fm3.metric("Total CAPEX",          f"GBP {fab_r['total_capex']:,.0f}")
        fm4.metric("Payback",
                   f"{fab_r['payback_years']:.1f} yrs" if not math.isnan(fab_r['payback_years']) else "—")

        fm5, fm6 = st.columns(2)
        fm5.metric(f"NPV ({fab_lifetime} yr)", f"GBP {fab_r['npv_gbp']:,.0f}")
        fm6.metric("Carbon saving", f"{fab_r['total_carbon_tco2e']:.1f} tCO2e/yr")

        fab_table = pd.DataFrame([{
            "Element":          row["element"],
            "Area (m²)":        row["area_m2"],
            "U-old (W/m²K)":   row["u_old"],
            "U-new (W/m²K)":   row["u_new"],
            "Saving (kWh/yr)":  row["saving_kwh"],
            "Cost saving (GBP/yr)": row["cost_saving_gbp"],
            "CAPEX (GBP)":      row["capex_gbp"],
            "Payback (yrs)":    row["payback_years"],
        } for row in fab_r["element_rows"]])
        st.dataframe(
            fab_table.style.format({
                "Area (m²)": "{:,.0f}", "U-old (W/m²K)": "{:.2f}",
                "U-new (W/m²K)": "{:.2f}", "Saving (kWh/yr)": "{:,.0f}",
                "Cost saving (GBP/yr)": "{:,.0f}", "CAPEX (GBP)": "{:,.0f}",
                "Payback (yrs)": "{:.1f}",
            }, na_rep="—"),
            use_container_width=True, hide_index=True
        )

        # Bar chart by element
        if fab_r["element_rows"]:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            el_labels = [r["element"].split("—")[0].strip() for r in fab_r["element_rows"]]
            el_save   = [r["saving_kwh"] for r in fab_r["element_rows"]]
            ax.barh(el_labels, el_save, color=PALETTE[1])
            ax.set_xlabel("Annual heating saving (kWh/yr)")
            ax.invert_yaxis()
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)


# ============================================================================
# TAB — SOLAR GAIN & OVERHEATING (CIBSE TM52)
# ============================================================================
with tab_gain:
    st.subheader("Solar Gain & Overheating Risk (CIBSE TM52)")
    st.markdown(
        "Physically-based hourly solar gain through glazing, by facade, using the "
        "building's real solar position and weather data — replacing the simplified "
        "sine-wave solar proxy used elsewhere in the model."
    )

    sg_c1, sg_c2 = st.columns([2, 1])
    with sg_c2:
        sg_orientation = st.slider(
            "Building orientation (° from North)", 0, 359, 0, 5,
            help="Rotates all facades together. 0° = front facade faces North.",
        )
        sg_shading = st.slider(
            "External shading factor", 0.3, 1.0, 1.0, 0.05,
            help="1.0 = no shading. Lower values represent overhangs, fins or blinds.",
        )
        sg_height = st.slider(
            "Building height (m)", 3, 100, int(C.DEFAULT_BUILDING_HEIGHT_M), 1,
        )
    with sg_c1:
        st.markdown("**Glazing per facade**")
        facade_defs = [
            {"label": "North", "base_azimuth": 0},
            {"label": "East",  "base_azimuth": 90},
            {"label": "South", "base_azimuth": 180},
            {"label": "West",  "base_azimuth": 270},
        ]
        facade_inputs = []
        fcols = st.columns(4)
        for i, fd in enumerate(facade_defs):
            with fcols[i]:
                st.caption(fd["label"])
                f_area = st.number_input(
                    "Glazing area (m²)", 0.0, 2_000.0, 100.0, 10.0, key=f"sg_area_{fd['label']}"
                )
                f_gval = st.number_input(
                    "g-value", 0.10, 0.90, 0.40, 0.05, key=f"sg_g_{fd['label']}"
                )
                facade_inputs.append({
                    "label":           fd["label"],
                    "azimuth_deg":     (fd["base_azimuth"] + sg_orientation) % 360,
                    "glazing_area_m2": f_area,
                    "g_value":         f_gval,
                    "shading_factor":  sg_shading,
                })

    if st.button("Calculate solar gain", type="primary", key="calc_gain"):
        if weather_df_for_solar is None:
            st.error(
                "Solar gain modelling needs real hourly weather data — switch Data "
                "source to Synthetic or Upload, with a valid location."
            )
        else:
            lat_g, lon_g = _get_latlon(location.strip() or "London")
            st.session_state["gain_result"] = sgain.calculate(
                weather_df    = weather_df_for_solar,
                lat           = lat_g,
                facades       = facade_inputs,
                floor_area_m2 = floor_area,
            )

    if "gain_result" in st.session_state:
        gain_r = st.session_state["gain_result"]
        st.divider()
        gm1, gm2, gm3 = st.columns(3)
        gm1.metric("Total annual solar gain", f"{gain_r['total_annual_gain_kwh']:,.0f} kWh/yr")
        gm2.metric(
            "TM52 exceedance hours", f"{gain_r['exceedance_hours']:,} h/yr",
            help=f"Occupied hours above the {gain_r['adaptive_limit_c']:.0f}°C adaptive comfort limit",
        )
        gm3.metric("Overheating risk", "⚠️ Flagged" if gain_r["overheating_flag"] else "✅ OK")

        if gain_r["overheating_flag"]:
            st.warning(
                f"Overheating risk flagged: {gain_r['exceedance_hours']} occupied hours/yr "
                f"exceed the CIBSE TM52 adaptive comfort threshold "
                f"(>{C.TM52_EXCEEDANCE_HOURS_THRESHOLD} hours). Consider increased shading, "
                f"reduced glazing g-value, or added thermal mass."
            )

        gain_table = pd.DataFrame([{
            "Facade":                 r["label"],
            "Azimuth (°)":            r["azimuth_deg"],
            "Glazing area (m²)":      r["glazing_area_m2"],
            "g-value":                r["g_value"],
            "Shading factor":         r["shading_factor"],
            "Annual gain (kWh/yr)":   r["annual_gain_kwh"],
            "Peak irradiance (W/m²)": r["peak_irradiance_wm2"],
        } for r in gain_r["facade_rows"]])
        st.dataframe(
            gain_table.style.format({
                "Azimuth (°)": "{:.0f}", "Glazing area (m²)": "{:,.0f}",
                "g-value": "{:.2f}", "Shading factor": "{:.2f}",
                "Annual gain (kWh/yr)": "{:,.0f}", "Peak irradiance (W/m²)": "{:,.0f}",
            }, na_rep="—"),
            use_container_width=True, hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(
            [r["label"] for r in gain_r["facade_rows"]],
            [r["annual_gain_kwh"] for r in gain_r["facade_rows"]],
            color=PALETTE[:len(gain_r["facade_rows"])],
        )
        ax.set_ylabel("Annual solar gain (kWh/yr)")
        ax.set_title("Solar gain by facade")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        with st.expander("3D building visualisation", expanded=False):
            facade_gains_map = {r["label"]: r["annual_gain_kwh"] for r in gain_r["facade_rows"]}
            fig3d = b3d.render(floor_area, sg_height, facade_gains_map)
            st.pyplot(fig3d, use_container_width=True); plt.close(fig3d)
            st.caption(
                "Illustrative block model — facades coloured by relative annual solar "
                "gain. Not to architectural scale; cosmetic only."
            )


# ============================================================================
# TAB 5 — FULL SUMMARY
# ============================================================================
with tab_summary:
    st.subheader("Full Energy Management Summary")
    st.markdown("Combined view of all modelled improvements.")

    # Gather available results
    hvac_saving_kwh  = float(best["Saved (kWh/yr)"])
    hvac_cost        = float(best["Cost saving (GBP/yr)"])
    hvac_capex       = float(best["CAPEX (GBP)"])
    hvac_carbon      = float(best["Carbon saving (tCO2e/yr)"])

    light_saving_kwh = 0.0; light_cost = 0.0; light_capex = 0.0; light_carbon = 0.0
    if zone_results:
        totals_l     = lt.total_saving(zone_results)
        light_saving_kwh = totals_l["total_saving_kwh"]
        light_cost       = totals_l["total_cost_saving"]
        light_capex      = totals_l["total_capex"]
        light_carbon     = light_saving_kwh * C.CARBON_FACTOR / 1000

    pv_saving_kwh = 0.0; pv_cost = 0.0; pv_capex = 0.0; pv_carbon = 0.0
    if "pv_result" in st.session_state:
        pv_r = st.session_state["pv_result"]
        pv_saving_kwh = pv_r["self_consumed_kwh"]
        pv_cost       = pv_r["total_annual_income_gbp"]
        pv_capex      = pv_r["total_capex_gbp"]
        pv_carbon     = pv_r["carbon_saved_tco2e"]

    shw_saving_kwh = 0.0; shw_cost = 0.0; shw_capex = 0.0; shw_carbon = 0.0
    if "shw_result" in st.session_state:
        shw_r = st.session_state["shw_result"]
        shw_saving_kwh = shw_r["fuel_saved_kwh"]
        shw_cost       = shw_r["cost_saving_gbp"]
        shw_capex      = shw_r["total_capex_gbp"]
        shw_carbon     = shw_r["carbon_saved_tco2e"]

    fab_saving_kwh = 0.0; fab_cost = 0.0; fab_capex = 0.0; fab_carbon = 0.0
    if "fab_result" in st.session_state:
        fab_r = st.session_state["fab_result"]
        fab_saving_kwh = fab_r["total_saving_kwh"]
        fab_cost       = fab_r["total_cost_saving"]
        fab_capex      = fab_r["total_capex"]
        fab_carbon     = fab_r["total_carbon_tco2e"]

    wind_saving_kwh = 0.0; wind_cost = 0.0; wind_capex_tot = 0.0; wind_carbon = 0.0
    if "wind_result" in st.session_state:
        wind_r = st.session_state["wind_result"]
        wind_saving_kwh = wind_r["self_consumed_kwh"]
        wind_cost       = wind_r["total_annual_income_gbp"]
        wind_capex_tot  = wind_r["total_capex_gbp"]
        wind_carbon     = wind_r["carbon_saved_tco2e"]

    total_saving_kwh = (hvac_saving_kwh + light_saving_kwh + pv_saving_kwh
                        + shw_saving_kwh + fab_saving_kwh + wind_saving_kwh)
    total_cost       = (hvac_cost + light_cost + pv_cost + shw_cost
                        + fab_cost + wind_cost)
    total_capex      = (hvac_capex + light_capex + pv_capex + shw_capex
                        + fab_capex + wind_capex_tot)
    total_carbon     = (hvac_carbon + light_carbon + pv_carbon + shw_carbon
                        + fab_carbon + wind_carbon)
    overall_payback  = total_capex / total_cost if total_cost > 0 else float("nan")

    # KPI row
    sk1, sk2, sk3, sk4, sk5 = st.columns(5)
    sk1.metric("Total annual cost saving", f"GBP {total_cost:,.0f}")
    sk2.metric("Total energy saving",      f"{total_saving_kwh:,.0f} kWh/yr")
    sk3.metric("Total carbon saving",      f"{total_carbon:.1f} tCO2e/yr")
    sk4.metric("Total CAPEX",              f"GBP {total_capex:,.0f}")
    sk5.metric("Blended payback",
               f"{overall_payback:.1f} yrs" if not math.isnan(overall_payback) else "—")

    st.divider()

    # EPC projection — improved SBEM-like scoring (Limitation 2 fix)
    with st.expander("EPC band projection (SBEM-like carbon-weighted scoring)", expanded=True):
        # Derive intensities (kWh/m²/yr) for each end-use
        _hvac_intensity    = baseline_hvac_kwh / max(floor_area, 1)
        _light_intensity   = light_saving_kwh  / max(floor_area, 1) if light_saving_kwh else 0.0
        _pv_offset_m2      = (pv_saving_kwh  / max(floor_area, 1)) if pv_saving_kwh  else 0.0
        _wind_offset_m2    = (wind_saving_kwh / max(floor_area, 1)) if wind_saving_kwh else 0.0
        _dhw_intensity     = float(C.DHW_INTENSITY_KWH_M2.get(building_type, 5))
        _other_intensity   = _hvac_intensity * 0.30  # approx small power share

        # Current state — unimproved
        _score_now, _band_now = epc.sbem_asset_rating(
            hvac_kwh_m2    = _hvac_intensity,
            lighting_kwh_m2 = _light_intensity * 1.5,  # pre-LED estimate
            dhw_kwh_m2     = _dhw_intensity,
            other_kwh_m2   = _other_intensity,
            building_type  = building_type,
            heating_fuel   = heating_fuel,
        )

        # After all improvements
        _hvac_saving_pct   = float(best["Saving (%)"]) if not math.isnan(float(best.get("Saving (%)", 0))) else 0.0
        _light_saving_pct  = (light_saving_kwh / max(light_saving_kwh / 0.5, 1)) * 100 if light_saving_kwh else 0.0
        _new_band, _new_score, _narrative = epc.project_band_sbem(
            current_score      = _score_now,
            hvac_saving_pct    = _hvac_saving_pct,
            lighting_saving_pct= _light_saving_pct,
            pv_offset_kwh_m2   = _pv_offset_m2,
            wind_offset_kwh_m2 = _wind_offset_m2,
            floor_area_m2      = floor_area,
            building_type      = building_type,
            heating_fuel       = heating_fuel,
        )

        ep1, ep2, ep3, ep4 = st.columns(4)
        ep1.metric("Manual EPC band (entered)", epc_band_input)
        ep2.metric("Model-estimated current score", f"{_score_now:.0f}", _band_now)
        ep3.metric("Projected score after improvements", f"{_new_score:.0f}")
        ep4.metric("Projected EPC band", _new_band,
                   delta=f"from {epc_band_input}")
        st.caption(
            f"Methodology: simplified SBEM carbon-weighted BEP. Heating fuel: {heating_fuel}. "
            f"{_narrative} "
            "A registered EPC assessor is required for formal MEES compliance evidence."
        )
        if epc_api_key and epc_email and location:
            with st.spinner("Checking EPC register…"):
                try:
                    records = epc.lookup_by_postcode(location.replace(" ", ""),
                                                     epc_email, epc_api_key)
                    if records:
                        r0 = epc.parse_record(records[0])
                        st.success(f"EPC found: **{r0['address']}** — Band **{r0['band']}** "
                                   f"(score {r0['score']}, lodged {r0['lodgement_date']})")
                    else:
                        st.info("No EPC records found for this postcode in the register.")
                except Exception as e:
                    st.warning(f"EPC API lookup failed: {e}")

    # Stacked bar chart — savings by category
    categories  = ["HVAC control", "LED lighting", "Solar PV", "Solar thermal", "Fabric", "Wind"]
    cost_values = [hvac_cost, light_cost, pv_cost, shw_cost, fab_cost, wind_cost]
    kwh_values  = [hvac_saving_kwh, light_saving_kwh, pv_saving_kwh, shw_saving_kwh, fab_saving_kwh, wind_saving_kwh]

    active_cats  = [(c, cv, kv) for c, cv, kv in zip(categories, cost_values, kwh_values) if cv > 0]
    if active_cats:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        cats = [a[0] for a in active_cats]
        cvs  = [a[1] for a in active_cats]
        kvs  = [a[2] for a in active_cats]

        bars1 = axes[0].bar(cats, cvs, color=PALETTE[:len(cats)])
        axes[0].set_ylabel("Annual cost saving (GBP/yr)")
        axes[0].set_title("Cost saving by technology")
        for bar, v in zip(bars1, cvs):
            axes[0].text(bar.get_x() + bar.get_width()/2, v + 50,
                         f"£{v:,.0f}", ha="center", fontsize=8, fontweight="bold")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

        bars2 = axes[1].bar(cats, kvs, color=PALETTE[:len(cats)])
        axes[1].set_ylabel("Energy saving (kWh/yr)")
        axes[1].set_title("Energy saving by technology")
        for bar, v in zip(bars2, kvs):
            axes[1].text(bar.get_x() + bar.get_width()/2, v + 50,
                         f"{v:,.0f}", ha="center", fontsize=8, fontweight="bold")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    else:
        st.info("Run analyses in the HVAC, Lighting, Solar and Fabric tabs to populate this chart.")

    # Summary table
    sum_table = pd.DataFrame({
        "Technology":        categories,
        "Annual saving (kWh/yr)": kwh_values,
        "Annual saving (GBP/yr)": cost_values,
        "Carbon (tCO2e/yr)": [hvac_carbon, light_carbon, pv_carbon, shw_carbon, fab_carbon, wind_carbon],
        "CAPEX (GBP)":       [hvac_capex, light_capex, pv_capex, shw_capex, fab_capex, wind_capex_tot],
    })
    sum_table["Payback (yrs)"] = sum_table.apply(
        lambda row: row["CAPEX (GBP)"] / row["Annual saving (GBP/yr)"]
        if row["Annual saving (GBP/yr)"] > 0 and row["CAPEX (GBP)"] > 0 else float("nan"),
        axis=1
    )
    # Add totals row
    totals_row = pd.DataFrame([{
        "Technology": "TOTAL",
        "Annual saving (kWh/yr)": total_saving_kwh,
        "Annual saving (GBP/yr)": total_cost,
        "Carbon (tCO2e/yr)":     total_carbon,
        "CAPEX (GBP)":            total_capex,
        "Payback (yrs)":          overall_payback,
    }])
    sum_table = pd.concat([sum_table, totals_row], ignore_index=True)
    st.dataframe(
        sum_table.style.format({
            "Annual saving (kWh/yr)": "{:,.0f}", "Annual saving (GBP/yr)": "{:,.0f}",
            "Carbon (tCO2e/yr)": "{:.1f}", "CAPEX (GBP)": "{:,.0f}",
            "Payback (yrs)": "{:.1f}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True
    )

    st.download_button(
        "Download summary (CSV)",
        sum_table.to_csv(index=False).encode(),
        "energy_management_summary.csv", "text/csv",
    )
