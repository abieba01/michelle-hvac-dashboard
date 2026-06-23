"""
app.py  -  Streamlit dashboard for the HVAC Optimisation AI Model.
Run with:  streamlit run app.py
"""
from __future__ import annotations

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config as C
import data_generator as dg
from energy_model import train, predict_annual_kwh
from hvac_optimizer import SCENARIOS, PRETTY

st.set_page_config(page_title="Michelle's Project – HVAC Optimisation Dashboard", layout="wide")

PALETTE = ["#1f4e79", "#2e75b6", "#5b9bd5", "#9dc3e6", "#c55a11"]


# ---------------------------------------------------------------------------
# Synthetic data path – cached once for the lifetime of the server process
# ---------------------------------------------------------------------------
@st.cache_resource(
    show_spinner="Generating synthetic data and training the surrogate model – first run only…"
)
def _load_synthetic():
    data     = dg.generate()
    train_df = dg.make_training_data(data)
    model, metrics, _ = train(train_df)
    return _run_scenarios(model, data, C.SIM_YEARS), metrics, data


# ---------------------------------------------------------------------------
# Uploaded data path – re-runs whenever the file content changes
# ---------------------------------------------------------------------------
def _parse_upload(file_bytes: bytes) -> tuple[pd.DataFrame, list[str]]:
    """
    Read uploaded CSV, derive time columns from timestamp, fill any missing
    control-lever columns with defaults, and calibrate the physics scale factor
    so the augmentation step stays aligned with the real kWh totals.
    Returns (df, list_of_warning_strings).
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    warnings: list[str] = []

    # ── Time columns ──────────────────────────────────────────────────────
    time_cols = {"hour", "dayofweek", "month", "is_workday"}
    has_ts       = "timestamp" in df.columns
    has_time_cols = time_cols.issubset(df.columns)
    if not has_ts and not has_time_cols:
        raise ValueError(
            "CSV must contain a 'timestamp' column, "
            "or all four of: hour, dayofweek, month, is_workday."
        )
    if has_ts:
        ts = pd.to_datetime(df["timestamp"])
        df["timestamp"] = ts
        if "hour"        not in df.columns: df["hour"]       = ts.dt.hour
        if "dayofweek"   not in df.columns: df["dayofweek"]  = ts.dt.dayofweek
        if "month"       not in df.columns: df["month"]       = ts.dt.month
        if "is_workday"  not in df.columns:
            df["is_workday"] = ts.dt.dayofweek.isin({0, 1, 2, 3, 4}).astype(int)
        ts_idx = pd.DatetimeIndex(ts)
    else:
        ts_idx = None

    # ── Energy target ─────────────────────────────────────────────────────
    if "hvac_kwh" not in df.columns:
        raise ValueError(
            "CSV must contain a 'hvac_kwh' column "
            "(hourly HVAC electricity consumption in kWh)."
        )

    # ── Optional columns – fill with defaults ─────────────────────────────
    if "t_out" not in df.columns:
        df["t_out"] = 10.0
        warnings.append(
            "'t_out' (outdoor temperature) not found – defaulted to 10 °C. "
            "Model accuracy will be reduced."
        )

    if "solar" not in df.columns:
        df["solar"] = (
            dg._solar_gain(ts_idx) if ts_idx is not None else 0.3
        )

    if "occ_frac" not in df.columns:
        df["occ_frac"] = np.where(
            (df["hour"].between(8, 17)) & (df["is_workday"] == 1), 0.7, 0.05
        )
        warnings.append(
            "'occ_frac' (occupancy fraction) not found – "
            "using a basic 08:00–18:00 weekday schedule."
        )

    if "system_on" not in df.columns:
        df["system_on"] = np.where(
            (df["hour"].between(6, 18)) & (df["is_workday"] == 1), 1.0, 0.0
        )

    for col, default in [
        ("cool_set",   C.COOLING_SETPOINT),
        ("heat_set",   C.HEATING_SETPOINT),
        ("deadband",   C.DEADBAND),
        ("fan_factor", 1.0),
        ("vent_frac",  1.0),
    ]:
        if col not in df.columns:
            df[col] = default

    # ── Calibrate scale so augmented physics aligns with real totals ──────
    raw = dg.hvac_energy_kw(
        df.t_out, df.solar, df.occ_frac, df.system_on,
        df.cool_set, df.heat_set, df.deadband, df.fan_factor, df.vent_frac,
    )
    raw_total  = float(raw.sum())
    real_total = float(df["hvac_kwh"].sum())
    df.attrs["scale"] = (real_total / raw_total) if raw_total > 0 else 1.0

    return df, warnings


@st.cache_data(show_spinner="Parsing upload and training model on your data…")
def _load_uploaded(file_bytes: bytes):
    df, warnings = _parse_upload(file_bytes)
    sim_years    = max(1, round(len(df) / 8760))
    train_df     = dg.make_training_data(df)
    model, metrics, _ = train(train_df)
    return _run_scenarios(model, df, sim_years), metrics, df, warnings


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
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


@st.cache_data(show_spinner=False)
def _sample_xlsx_bytes() -> bytes:
    """Return a two-sheet Excel workbook: 48 rows of sample data + a README sheet."""
    data = dg.generate(years=1)
    sample = data.head(48)[
        ["timestamp", "t_out", "solar", "occ_frac", "system_on",
         "cool_set", "heat_set", "deadband", "fan_factor", "vent_frac", "hvac_kwh"]
    ].copy()
    sample["timestamp"] = sample["timestamp"].astype(str)

    readme = pd.DataFrame(
        [
            ("timestamp",  "Yes *",  "datetime",
             "Date and time of the reading, one row per hour. "
             "Format: YYYY-MM-DD HH:MM:SS. "
             "* If omitted, the four columns hour / dayofweek / month / is_workday must all be present instead.",
             "—"),
            ("hvac_kwh",   "Yes",    "kWh",
             "Hourly HVAC electricity consumption. "
             "This is the value the model learns to predict. "
             "Obtain from a dedicated sub-meter or smart meter.",
             "—"),
            ("t_out",      "No",     "°C",
             "Outdoor dry-bulb temperature. "
             "From a weather station, BMS sensor, or a free weather API. "
             "Including this column significantly improves model accuracy.",
             "10 °C (constant) — accuracy will be reduced"),
            ("solar",      "No",     "0 – 1",
             "Normalised solar gain. "
             "0 = no sun (night or fully overcast), 1 = peak midsummer midday. "
             "If omitted and a timestamp column is present, the app estimates it "
             "from the hour of day and day of year.",
             "Estimated from timestamp; 0.3 constant if no timestamp"),
            ("occ_frac",   "No",     "0 – 1",
             "Fraction of peak design occupancy present in the building. "
             "Sources: people-counter system, CO₂-sensor inference, door-access logs. "
             "Including this column significantly improves model accuracy.",
             "0.7 during 08:00–18:00 on weekdays, 0.05 otherwise"),
            ("system_on",  "No",     "0 or 1",
             "Whether the HVAC plant is running: 1 = on, 0 = off. "
             "From the BMS plant on/off status point.",
             "1 during 06:00–18:00 on weekdays, 0 otherwise"),
            ("cool_set",   "No",     "°C",
             "Active cooling setpoint. "
             "From the BMS or zone thermostat controller.",
             f"{C.COOLING_SETPOINT}"),
            ("heat_set",   "No",     "°C",
             "Active heating setpoint. "
             "From the BMS or zone thermostat controller.",
             f"{C.HEATING_SETPOINT}"),
            ("deadband",   "No",     "°C",
             "Thermostat dead-band — the temperature margin either side of the setpoint "
             "before heating or cooling activates. "
             "From the BMS controller configuration.",
             f"{C.DEADBAND}"),
            ("fan_factor",  "No",    "0 – 1",
             "AHU supply-fan speed as a fraction of full speed. "
             "From the variable-speed drive (VSD) output signal. "
             "Use 1.0 if the fan runs at a single fixed speed.",
             "1.0 (full speed)"),
            ("vent_frac",  "No",     "0 – 1",
             "Ventilation rate as a fraction of the design (maximum) rate. "
             "From the AHU damper position or CO₂-controlled DCV controller. "
             "Use 1.0 if ventilation is not demand-controlled.",
             "1.0 (full design rate)"),
        ],
        columns=["Column", "Required", "Unit", "Description", "Default if missing"],
    )

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        sample.to_excel(writer, sheet_name="Sample data", index=False)
        readme.to_excel(writer, sheet_name="README", index=False)
    return buf.getvalue()


def _compute_results(energy, elec_price, carbon_factor, discount_rate, lifetime, capex_map):
    rows = []
    for key, e in energy.items():
        saved  = e["saved_kwh"]
        cost   = saved * elec_price
        carbon = saved * carbon_factor / 1000
        capex  = capex_map.get(key, 0)
        payback = capex / cost if (capex > 0 and cost > 0) else np.nan
        npv = (
            sum(cost / (1 + discount_rate) ** y for y in range(1, lifetime + 1)) - capex
        ) if capex > 0 else np.nan
        rows.append({
            "key":                       key,
            "Strategy":                  e["scenario"],
            "Annual HVAC (kWh/yr)":      e["annual_kwh"],
            "Saved (kWh/yr)":            saved,
            "Saving (%)":                e["saving_pct"],
            "Cost saving (GBP/yr)":      cost,
            "Carbon saving (tCO2e/yr)":  carbon,
            "CAPEX (GBP)":               capex,
            "Payback (yrs)":             payback,
            "NPV (GBP)":                 npv,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Data source")
    data_source = st.radio(
        "data_source",
        ["Synthetic (built-in)", "Upload my own CSV"],
        label_visibility="collapsed",
    )

    if data_source == "Upload my own CSV":
        with st.expander("Column requirements"):
            st.markdown(
                "**Must be present**\n"
                "- `hvac_kwh` – hourly HVAC energy (kWh)\n"
                "- `timestamp` *or* the four columns `hour`, `dayofweek`, `month`, `is_workday`\n\n"
                "**Auto-filled if missing** (model less accurate without them)\n"
                "- `t_out` – outdoor temp (°C)\n"
                "- `occ_frac` – occupancy 0–1\n\n"
                "**Defaulted to fixed values if missing**\n"
                "- `solar`, `system_on`, `cool_set`, `heat_set`, "
                "`deadband`, `fan_factor`, `vent_frac`"
            )
        uploaded_file = st.file_uploader("Upload hourly CSV", type="csv")
        st.download_button(
            label="Download sample file (.xlsx)",
            data=_sample_xlsx_bytes(),
            file_name="hvac_sample.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Excel workbook with two sheets: 48 rows of sample data and a README explaining every column.",
        )

    st.divider()
    st.header("Economic assumptions")
    elec_price    = st.slider("Electricity price (GBP/kWh)",  0.10, 0.60, C.ELECTRICITY_PRICE,        0.01)
    carbon_factor = st.slider("Carbon factor (kgCO2e/kWh)",   0.05, 0.50, C.CARBON_FACTOR,            0.001)
    discount_rate = st.slider("Discount rate (%)",              1,   15,   int(C.DISCOUNT_RATE * 100), 1) / 100
    lifetime      = st.slider("Measure lifetime (years)",       5,   30,   C.MEASURE_LIFETIME_YEARS,   1)

    st.divider()
    st.header("Capital costs (GBP)")
    capex_occ   = st.slider("Occupancy scheduling",         10_000, 100_000, C.CAPEX["occupancy_scheduling"], 1_000)
    capex_therm = st.slider("Smart thermostats",             5_000,  80_000, C.CAPEX["smart_thermostats"],    1_000)
    capex_bas   = st.slider("Building Automation System",   50_000, 300_000, C.CAPEX["bas"],                  5_000)
    capex_comb  = st.slider("Combined strategy",            50_000, 400_000, C.CAPEX["combined"],              5_000)


# ---------------------------------------------------------------------------
# Load data (one branch per data source)
# ---------------------------------------------------------------------------
st.title("Michelle's Project – HVAC Optimisation Dashboard")

upload_warnings: list[str] = []

if data_source == "Upload my own CSV":
    if not uploaded_file:
        st.info("Upload a CSV file in the sidebar to get started. "
                "Expand 'Column requirements' above the uploader to see what's needed.")
        st.stop()
    try:
        energy, metrics, raw_data, upload_warnings = _load_uploaded(uploaded_file.getvalue())
    except ValueError as exc:
        st.error(f"Could not read your file: {exc}")
        st.stop()
    data_label = f"Uploaded: {uploaded_file.name}  ({len(raw_data):,} rows)"
else:
    energy, metrics, raw_data = _load_synthetic()
    data_label = (
        f"Synthetic data  |  {C.SIM_YEARS} years  |  "
        f"{C.FLOOR_AREA_M2:,} m²  |  "
        f"baseline HVAC ~{C.TARGET_ANNUAL_HVAC_KWH:,.0f} kWh/yr"
    )

for w in upload_warnings:
    st.warning(w)

st.caption(f"{data_label}  |  Surrogate model R² = {metrics['r2']:.4f}")

# ---------------------------------------------------------------------------
# Financial results
# ---------------------------------------------------------------------------
capex_map = {
    "occupancy_scheduling": capex_occ,
    "smart_thermostats":    capex_therm,
    "bas":                  capex_bas,
    "combined":             capex_comb,
}
df     = _compute_results(energy, elec_price, carbon_factor, discount_rate, lifetime, capex_map)
non_bl = df[df["key"] != "baseline"].copy()

best = non_bl.sort_values("Cost saving (GBP/yr)", ascending=False).iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best annual cost saving", f"GBP {best['Cost saving (GBP/yr)']:,.0f}", best["Strategy"])
c2.metric("Electricity price",        f"GBP {elec_price:.2f}/kWh")
c3.metric("Discount rate",            f"{discount_rate * 100:.0f}%")
c4.metric("Measure lifetime",         f"{lifetime} yrs")

st.divider()

st.subheader("Results by strategy")
styled = (
    df.drop(columns=["key"])
    .style
    .format(
        {
            "Annual HVAC (kWh/yr)":     "{:,.0f}",
            "Saved (kWh/yr)":           "{:,.0f}",
            "Saving (%)":               "{:.1f}",
            "Cost saving (GBP/yr)":     "{:,.0f}",
            "Carbon saving (tCO2e/yr)": "{:.1f}",
            "CAPEX (GBP)":              "{:,.0f}",
            "Payback (yrs)":            "{:.1f}",
            "NPV (GBP)":                "{:,.0f}",
        },
        na_rep="—",
    )
)
st.dataframe(styled, use_container_width=True, hide_index=True)

with st.expander("Download data"):
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        label=f"Hourly dataset ({len(raw_data):,} rows)",
        data=raw_data.to_csv(index=False).encode(),
        file_name="hvac_hourly_data.csv",
        mime="text/csv",
        help="Weather, occupancy, control levers and HVAC demand for every row in the dataset.",
    )
    dl2.download_button(
        label="Scenario results table",
        data=df.drop(columns=["key"]).to_csv(index=False).encode(),
        file_name="hvac_scenario_results.csv",
        mime="text/csv",
        help="The five-row summary with current slider values applied.",
    )

st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Energy saving by strategy")
    fig, ax = plt.subplots(figsize=(5, 3.8))
    bars = ax.bar(range(len(non_bl)), non_bl["Saving (%)"], color=PALETTE[1:1 + len(non_bl)])
    ax.set_ylabel("HVAC energy saving (%)")
    ax.set_xticks(range(len(non_bl)))
    ax.set_xticklabels([s.replace(" ", "\n") for s in non_bl["Strategy"]], fontsize=8)
    for bar, val in zip(bars, non_bl["Saving (%)"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col2:
    st.subheader("Simple payback period by strategy")
    pb = non_bl.dropna(subset=["Payback (yrs)"])
    if pb.empty:
        st.info("No payback data to display.")
    else:
        fig2, ax2 = plt.subplots(figsize=(5, 3.8))
        ax2.barh(pb["Strategy"], pb["Payback (yrs)"], color=PALETTE[2])
        ax2.set_xlabel("Simple payback period (years)")
        ax2.invert_yaxis()
        for patch, val in zip(ax2.patches, pb["Payback (yrs)"]):
            ax2.text(val + 0.05, patch.get_y() + patch.get_height() / 2,
                     f"{val:.1f} yr", va="center", fontsize=9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

# ---------------------------------------------------------------------------
# Model details footer
# ---------------------------------------------------------------------------
with st.expander("Surrogate model performance"):
    m1, m2, m3 = st.columns(3)
    m1.metric("R2",   f"{metrics['r2']:.4f}")
    m2.metric("MAE",  f"{metrics['mae']:.2f} kWh/h")
    m3.metric("RMSE", f"{metrics['rmse']:.2f} kWh/h")
    st.caption(
        f"Trained on {metrics['n_train']:,} rows, "
        f"tested on {metrics['n_test']:,} rows. "
        f"Mean target: {metrics['mean_target']:.2f} kWh/h."
    )
