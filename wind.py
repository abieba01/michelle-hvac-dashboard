"""
wind.py
=======
Small-scale wind turbine generation and financial analysis.

Methodology:
  - Hourly wind speed at 10 m from Open-Meteo (weather.py), height-corrected
    to hub height using the wind power law: v_hub = v_10m * (h_hub/10)^alpha
  - A simplified cubic power curve between cut-in and rated speed, flat
    output at rated power up to cut-out speed, zero outside that range
  - Building-mounted turbines suffer extra turbulence losses vs freestanding
    turbines, captured by a flat efficiency_factor per turbine type
  - Site suitability warning if the mean annual hub-height wind speed is
    below the practical viability threshold (default 5 m/s)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import config as C

# ---------------------------------------------------------------------------
# Turbine type definitions
# ---------------------------------------------------------------------------
TURBINE_TYPES: dict[str, dict] = {
    "micro": {
        "label":             "Building-mounted micro wind",
        "cut_in":            3.0,    # m/s
        "rated_speed":       14.0,   # m/s
        "cut_out":           25.0,   # m/s
        "efficiency_factor": 0.70,   # turbulence losses at building-mounted sites
        "capex_per_kw":      3_500,  # GBP/kW installed
        "typical_kw_range":  (1, 15),
    },
    "small": {
        "label":             "Freestanding small wind turbine",
        "cut_in":            2.5,
        "rated_speed":       11.0,
        "cut_out":           25.0,
        "efficiency_factor": 0.92,
        "capex_per_kw":      2_500,
        "typical_kw_range":  (5, 100),
    },
}

# Wind generation is less correlated with daytime occupancy than solar
WIND_SELF_CONSUMPTION: dict[str, float] = {
    "office": 0.45, "hotel": 0.50, "hospital": 0.55,
    "school": 0.35, "retail": 0.45, "industrial": 0.55,
}


# ---------------------------------------------------------------------------
# Hub height correction (wind power law)
# ---------------------------------------------------------------------------
def hub_height_correction(
    v_10m: np.ndarray | float, hub_height_m: float, alpha: float = 0.143
) -> np.ndarray | float:
    """Correct measured 10 m wind speed to hub height using the power law."""
    return v_10m * (hub_height_m / 10.0) ** alpha


# ---------------------------------------------------------------------------
# Power curve
# ---------------------------------------------------------------------------
def power_curve_kw(
    wind_speed: np.ndarray, rated_capacity_kw: float, turbine_type: str
) -> np.ndarray:
    """
    Simplified turbine power curve (kW output per hour).

    Cubic ramp-up between cut-in and rated speed (power ~ v^3), flat at
    rated power between rated speed and cut-out, zero outside that range.
    """
    spec = TURBINE_TYPES[turbine_type]
    v = np.asarray(wind_speed, dtype=float)
    cut_in, rated, cut_out = spec["cut_in"], spec["rated_speed"], spec["cut_out"]
    eff = spec["efficiency_factor"]

    power = np.zeros_like(v)
    ramp = (v >= cut_in) & (v < rated)
    power[ramp] = rated_capacity_kw * ((v[ramp] - cut_in) / (rated - cut_in)) ** 3

    flat = (v >= rated) & (v <= cut_out)
    power[flat] = rated_capacity_kw

    return power * eff


# ---------------------------------------------------------------------------
# Annual generation
# ---------------------------------------------------------------------------
def annual_generation_kwh(
    rated_capacity_kw: float,
    hub_height_m: float,
    turbine_type: str,
    weather_df: pd.DataFrame | None,
) -> tuple[float, float, np.ndarray | None]:
    """
    Return (annual_generation_kwh, mean_hub_wind_speed_ms, hourly_generation | None).

    Requires weather_df with a 'wind_speed' column (m/s at 10 m). Falls back
    to a UK-average 5.5 m/s mean wind speed (Rayleigh-ish flat profile) if
    real wind data is unavailable.
    """
    if weather_df is not None and "wind_speed" in weather_df.columns and len(weather_df) > 0:
        v_10m = weather_df["wind_speed"].to_numpy(dtype=float)
        v_hub = hub_height_correction(v_10m, hub_height_m)
        hourly_kw = power_curve_kw(v_hub, rated_capacity_kw, turbine_type)
        n_years = max(1, round(len(weather_df) / 8760))
        annual = float(hourly_kw.sum() / n_years)
        return annual, float(v_hub.mean()), hourly_kw
    else:
        # Fallback: assume a flat UK-average 5.5 m/s at 10 m, height-corrected
        v_hub = hub_height_correction(5.5, hub_height_m)
        spec = TURBINE_TYPES[turbine_type]
        # crude capacity-factor approximation for a flat wind speed
        cf = min(max((v_hub - spec["cut_in"]) / (spec["rated_speed"] - spec["cut_in"]), 0), 1) ** 3
        annual = rated_capacity_kw * spec["efficiency_factor"] * cf * 8760
        return annual, v_hub, None


# ---------------------------------------------------------------------------
# Self-consumption
# ---------------------------------------------------------------------------
def self_consumption_kwh(
    annual_generation: float,
    building_type: str,
    building_annual_load_kwh: float,
) -> tuple[float, float]:
    """Return (self_consumed_kwh, exported_kwh)."""
    sc_ratio = WIND_SELF_CONSUMPTION.get(building_type, 0.45)
    self_consumed = min(annual_generation * sc_ratio, building_annual_load_kwh)
    exported = max(annual_generation - self_consumed, 0)
    return self_consumed, exported


# ---------------------------------------------------------------------------
# Financial calculation
# ---------------------------------------------------------------------------
def financials(
    rated_capacity_kw: float,
    hub_height_m: float,
    turbine_type: str,
    building_type: str,
    building_annual_load_kwh: float,
    elec_price: float,
    export_tariff: float,
    discount_rate: float,
    inflation_rate: float,
    grant_pct: float,
    weather_df: pd.DataFrame | None = None,
    lifetime_years: int = C.WIND_LIFETIME_YEARS,
    degradation: float = C.WIND_DEGRADATION,
    capex_per_kw: float | None = None,
) -> dict:
    """
    Full financial model for a small wind turbine installation.

    Returns a dict with:
        annual_gen_kwh, mean_wind_speed_ms, site_suitable, self_consumed_kwh,
        exported_kwh, import_saved_gbp, export_income_gbp,
        total_annual_income_gbp, carbon_saved_tco2e, total_capex_gbp,
        capex_after_grant_gbp, payback_years, npv_gbp, irr_pct
    """
    spec = TURBINE_TYPES[turbine_type]
    if capex_per_kw is None:
        capex_per_kw = spec["capex_per_kw"]

    annual_gen, mean_v, _ = annual_generation_kwh(
        rated_capacity_kw, hub_height_m, turbine_type, weather_df
    )
    site_suitable = mean_v >= C.WIND_MIN_VIABLE_MS

    self_con, exported = self_consumption_kwh(
        annual_gen, building_type, building_annual_load_kwh
    )

    import_saved = self_con * elec_price
    export_income = exported * export_tariff
    total_income_0 = import_saved + export_income

    capex = rated_capacity_kw * capex_per_kw
    capex_net = capex * (1 - grant_pct / 100)

    payback = capex_net / total_income_0 if total_income_0 > 0 else float("nan")

    npv = -capex_net
    irr_flows = [-capex_net]
    for y in range(1, lifetime_years + 1):
        degr_factor = (1 - degradation) ** y
        price_factor = (1 + inflation_rate) ** y
        net_y = total_income_0 * degr_factor * price_factor
        npv += net_y / (1 + discount_rate) ** y
        irr_flows.append(net_y)

    irr_pct = _irr(irr_flows)
    carbon_saved = self_con * C.CARBON_FACTOR / 1000  # tCO2e/yr

    return {
        "annual_gen_kwh":          annual_gen,
        "mean_wind_speed_ms":      mean_v,
        "site_suitable":           site_suitable,
        "self_consumed_kwh":       self_con,
        "exported_kwh":            exported,
        "import_saved_gbp":        import_saved,
        "export_income_gbp":       export_income,
        "total_annual_income_gbp": total_income_0,
        "carbon_saved_tco2e":      carbon_saved,
        "total_capex_gbp":         capex,
        "capex_after_grant_gbp":   capex_net,
        "payback_years":           payback,
        "npv_gbp":                 npv,
        "irr_pct":                 irr_pct,
    }


def _irr(cash_flows: list[float]) -> float:
    """Internal rate of return via numpy-financial (optional dependency)."""
    try:
        import numpy_financial as npf
        val = npf.irr(cash_flows)
        return float(val) * 100 if not np.isnan(val) else float("nan")
    except ImportError:
        return float("nan")
