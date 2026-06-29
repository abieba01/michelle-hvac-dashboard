"""
solar_pv.py
===========
Solar PV generation and financial analysis.

Uses real hourly irradiance from Open-Meteo (weather.py) when available,
otherwise falls back to UK monthly average peak-sun-hour data.

Generation model:
  P_ac (kW) = capacity_kwp × (irradiance_W_m2 / 1000) × orientation_factor
               × tilt_factor × performance_ratio

Self-consumption is estimated from building-type profiles.
Export income uses the Smart Export Guarantee (SEG) tariff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import config as C

# ---------------------------------------------------------------------------
# Correction factors (multiply against south-facing 30° tilt value)
# ---------------------------------------------------------------------------
ORIENTATION_FACTORS: dict[str, float] = {
    "South":      1.00,
    "South-West": 0.97,
    "South-East": 0.97,
    "East":       0.85,
    "West":       0.85,
    "North":      0.60,
}

TILT_FACTORS: dict[int, float] = {
    0:  0.85,   # flat roof
    10: 0.93,
    15: 0.96,
    20: 0.98,
    30: 1.00,   # optimal UK tilt
    35: 0.99,
    40: 0.97,
    45: 0.94,
    50: 0.90,
}

# UK monthly average peak sun hours (south-facing, 30° tilt)
# Source: Photovoltaic Geographical Information System (PVGIS) EU Joint Research Centre
UK_MONTHLY_PSH: dict[int, float] = {
    1: 1.0, 2: 2.0, 3: 3.5, 4: 4.5, 5: 5.3,
    6: 5.8, 7: 5.6, 8: 5.1, 9: 4.0, 10: 2.6,
    11: 1.4, 12: 0.9,
}  # peak sun hours per day (≈ kWh/m²/day at 1000 W/m² STC)


def _nearest_tilt(tilt_deg: int) -> float:
    keys = sorted(TILT_FACTORS.keys())
    closest = min(keys, key=lambda k: abs(k - tilt_deg))
    return TILT_FACTORS[closest]


# ---------------------------------------------------------------------------
# Generation calculation
# ---------------------------------------------------------------------------
def annual_generation_kwh(
    capacity_kwp: float,
    orientation: str,
    tilt_deg: int,
    weather_df: pd.DataFrame | None,
    performance_ratio: float = C.SOLAR_PV_PERF_RATIO,
) -> tuple[float, np.ndarray | None]:
    """
    Return (annual_generation_kwh, hourly_generation_array | None).

    If weather_df is provided (columns: timestamp, solar where solar is
    normalised 0-1 irradiance from Open-Meteo shortwave_radiation / 900),
    hourly generation is computed from real irradiance. Otherwise uses UK
    monthly average peak-sun-hours.
    """
    o_factor = ORIENTATION_FACTORS.get(orientation, 1.0)
    t_factor = _nearest_tilt(tilt_deg)
    combined = o_factor * t_factor * performance_ratio

    if weather_df is not None and "solar" in weather_df.columns and len(weather_df) > 0:
        # solar column is already normalised (0–1); un-normalise to kW/m² by ×0.9
        # then: P_ac = kWp × (irradiance_kW_m2) × factor = kWp × (solar × 0.9) × factor
        irr_kw_m2 = weather_df["solar"].to_numpy(dtype=float) * 0.9
        hourly_gen = capacity_kwp * irr_kw_m2 * combined  # kWh per hour
        annual     = float(hourly_gen.sum() / max(1, round(len(weather_df) / 8760)))
        return annual, hourly_gen
    else:
        # UK average fallback: PSH × days_in_month summed across 12 months
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        annual = sum(
            UK_MONTHLY_PSH[m] * days_in_month[m - 1] * capacity_kwp * combined
            for m in range(1, 13)
        )
        return annual, None


# ---------------------------------------------------------------------------
# Self-consumption
# ---------------------------------------------------------------------------
def self_consumption_kwh(
    annual_generation: float,
    building_type: str,
    building_annual_load_kwh: float,
) -> tuple[float, float]:
    """
    Return (self_consumed_kwh, exported_kwh).

    Uses a building-type self-consumption ratio, capped so we never export more
    than is generated and never self-consume more than the building load.
    """
    sc_ratio = C.SOLAR_SELF_CONSUMPTION.get(building_type, 0.60)
    self_consumed = min(annual_generation * sc_ratio, building_annual_load_kwh)
    exported      = max(annual_generation - self_consumed, 0)
    return self_consumed, exported


# ---------------------------------------------------------------------------
# Financial calculation
# ---------------------------------------------------------------------------
def financials(
    capacity_kwp: float,
    orientation: str,
    tilt_deg: int,
    building_type: str,
    building_annual_load_kwh: float,
    elec_price: float,
    export_tariff: float,
    discount_rate: float,
    inflation_rate: float,
    grant_pct: float,
    weather_df: pd.DataFrame | None = None,
    performance_ratio: float = C.SOLAR_PV_PERF_RATIO,
    lifetime_years: int = C.SOLAR_PV_LIFETIME,
    panel_degradation: float = C.SOLAR_PV_DEGRADATION,
    capex_per_kwp: float = 1_200,
) -> dict:
    """
    Full financial model for a solar PV installation.

    Returns a dict with:
        annual_gen_kwh, self_consumed_kwh, exported_kwh,
        import_saved_gbp, export_income_gbp, total_annual_income_gbp,
        total_capex_gbp, capex_after_grant_gbp,
        payback_years, npv_gbp, irr_pct
    """
    annual_gen, _ = annual_generation_kwh(
        capacity_kwp, orientation, tilt_deg, weather_df, performance_ratio
    )
    self_con, exported = self_consumption_kwh(
        annual_gen, building_type, building_annual_load_kwh
    )

    import_saved   = self_con  * elec_price
    export_income  = exported  * export_tariff
    total_income_0 = import_saved + export_income

    capex      = capacity_kwp * capex_per_kwp
    capex_net  = capex * (1 - grant_pct / 100)

    payback = capex_net / total_income_0 if total_income_0 > 0 else float("nan")

    # Inflation-adjusted NPV with panel degradation
    npv = -capex_net
    irr_flows = [-capex_net]
    for y in range(1, lifetime_years + 1):
        panel_factor = (1 - panel_degradation) ** y
        price_factor = (1 + inflation_rate)   ** y
        net_y = total_income_0 * panel_factor * price_factor
        npv  += net_y / (1 + discount_rate) ** y
        irr_flows.append(net_y)

    irr_pct = _irr(irr_flows)

    carbon_saved = self_con * C.CARBON_FACTOR / 1000  # tCO2e/yr

    return {
        "annual_gen_kwh":          annual_gen,
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
