"""
hvac_optimizer.py
=================
Turns the trained surrogate model into a decision tool.

For each optimisation strategy the optimiser constructs a counterfactual operating
schedule (it rewrites the control-lever columns of the baseline log) and asks the
surrogate model to predict the resulting annual HVAC energy. Comparing each
scenario against the baseline yields the energy, cost and carbon savings that feed
the project's financial and environmental analysis.

Strategies
----------
baseline              Conventional fixed 07:00-19:00 weekday schedule.
occupancy_scheduling  Plant follows measured occupancy (+1 h pre-conditioning).
smart_thermostats     Adaptive set-points, wider dead-band, occupancy setback.
bas                   Variable-speed fans/pumps, optimal start/stop, set-point reset.
combined              All three measures applied together.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import config as C
from energy_model import predict_annual_kwh


# ----------------------------------------------------------------------------
# Counterfactual schedule builders (each returns a modified copy of the log)
# ----------------------------------------------------------------------------
def _occupancy_schedule(df: pd.DataFrame) -> pd.DataFrame:
    p = df.copy()
    # plant runs only when the floor is genuinely occupied (sensor-driven)
    p["system_on"] = (p["occ_frac"] > 0.10).astype(float)
    # demand-controlled ventilation: rate tracks actual occupancy (min 20%)
    p["vent_frac"] = np.clip(p["occ_frac"], 0.20, 1.0)
    return p


def _smart_thermostat(df: pd.DataFrame) -> pd.DataFrame:
    p = df.copy()
    # thermostats modulate temperature, not the central plant schedule, so the
    # dumb baseline run-times are retained; savings come from a wider comfort
    # dead-band, set-point reset and a learned deep-setback when zones are empty
    low_occ = p["occ_frac"] < 0.08
    p["deadband"] = 2.0
    p["cool_set"] = np.where(low_occ, C.SETBACK_COOLING, 23.5)
    p["heat_set"] = np.where(low_occ, C.SETBACK_HEATING, 19.5)
    return p


def _bas(df: pd.DataFrame) -> pd.DataFrame:
    p = df.copy()
    # variable-speed drives cut base fan/pump load
    p["fan_factor"] = 0.80
    # optimal start/stop tightens the dumb schedule to 07:00-19:00 weekday only
    hour = p["hour"].to_numpy()
    weekday = np.isin(p["dayofweek"].to_numpy(), [0, 1, 2, 3, 4])
    p["system_on"] = ((hour >= 7) & (hour < 19) & weekday).astype(float)
    # supply-temperature / set-point reset and a modest ventilation reset
    p["cool_set"] = 23.0
    p["vent_frac"] = 0.85
    return p


def _combined(df: pd.DataFrame) -> pd.DataFrame:
    p = _occupancy_schedule(df)          # occupancy runtime + DCV
    # adaptive set-points / dead-band
    low_occ = p["occ_frac"] < 0.08
    p["deadband"] = 2.0
    p["cool_set"] = np.where(low_occ, C.SETBACK_COOLING, 23.0)
    p["heat_set"] = np.where(low_occ, C.SETBACK_HEATING, 19.5)
    p["fan_factor"] = 0.80               # BAS variable-speed drives
    return p


SCENARIOS = {
    "baseline":             lambda df: df.copy(),
    "occupancy_scheduling": _occupancy_schedule,
    "smart_thermostats":    _smart_thermostat,
    "bas":                  _bas,
    "combined":             _combined,
}

PRETTY = {
    "baseline":             "Baseline (fixed schedule)",
    "occupancy_scheduling": "Occupancy-based scheduling",
    "smart_thermostats":    "Smart thermostats",
    "bas":                  "Building Automation System",
    "combined":             "Combined optimisation",
}


# ----------------------------------------------------------------------------
# Financial helpers
# ----------------------------------------------------------------------------
def _npv(annual_saving: float, capex: float,
         years: int = C.MEASURE_LIFETIME_YEARS, rate: float = C.DISCOUNT_RATE) -> float:
    discounted = sum(annual_saving / (1 + rate) ** y for y in range(1, years + 1))
    return discounted - capex


def evaluate(model, df: pd.DataFrame, sim_years: int = C.SIM_YEARS) -> pd.DataFrame:
    """Run every scenario through the surrogate and assemble the results table."""
    baseline_kwh = predict_annual_kwh(model, SCENARIOS["baseline"](df), sim_years)
    rows = []
    for key, builder in SCENARIOS.items():
        scen = builder(df)
        annual_kwh = predict_annual_kwh(model, scen, sim_years)
        saved_kwh = baseline_kwh - annual_kwh
        pct = saved_kwh / baseline_kwh * 100
        cost_saving = saved_kwh * C.ELECTRICITY_PRICE
        carbon_saving = saved_kwh * C.CARBON_FACTOR / 1000  # tCO2e
        capex = C.CAPEX.get(key, 0)
        payback = capex / cost_saving if cost_saving > 0 else np.nan
        roi = cost_saving / capex * 100 if capex > 0 else np.nan
        npv = _npv(cost_saving, capex) if capex > 0 else np.nan
        rows.append({
            "scenario": PRETTY[key],
            "annual_hvac_kwh": annual_kwh,
            "energy_saved_kwh": max(saved_kwh, 0),
            "saving_pct": max(pct, 0),
            "cost_saving_gbp": max(cost_saving, 0),
            "carbon_saving_tco2e": max(carbon_saving, 0),
            "capex_gbp": capex,
            "payback_years": payback,
            "roi_pct": roi,
            "npv_15yr_gbp": npv,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import data_generator as dg
    from energy_model import train
    data = dg.generate()
    model, _, _ = train(dg.make_training_data(data))
    results = evaluate(model, data)
    pd.set_option("display.width", 140, "display.max_columns", 20)
    print(results.round(2).to_string(index=False))
