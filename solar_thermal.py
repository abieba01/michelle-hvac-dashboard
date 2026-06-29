"""
solar_thermal.py
================
Solar hot water (SHW) thermal yield and financial analysis.

Methodology: simplified CIBSE AM12 / BS EN 15316-4-3 approach.
  - Collector yield from irradiance and collector efficiency
  - DHW demand from building type profile
  - Solar fraction capped at practical maximum (70%)
  - Gas or electricity displacement depending on existing fuel type
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import config as C

# ---------------------------------------------------------------------------
# Monthly irradiance on a tilted collector surface (UK, south-facing 45° tilt)
# kWh/m²/month — CIBSE AM12 Table B1 representative UK mid-latitude values
# ---------------------------------------------------------------------------
UK_MONTHLY_IRRADIANCE_KWH_M2 = {
    1: 19, 2: 32, 3: 56, 4: 83, 5: 111, 6: 120,
    7: 118, 8: 102, 9: 72, 10: 42, 11: 22, 12: 15,
}  # total: ~792 kWh/m²/yr

# Orientation multiplier relative to south-facing 45° tilt
ORIENTATION_FACTORS = {
    "South":      1.00,
    "South-West": 0.95,
    "South-East": 0.95,
    "East":       0.80,
    "West":       0.80,
    "North":      0.55,
}

# DHW demand by building type (kWh/m²/yr)
DHW_INTENSITY = C.DHW_INTENSITY_KWH_M2


# ---------------------------------------------------------------------------
# Core yield calculation
# ---------------------------------------------------------------------------
def annual_yield_kwh(
    collector_area_m2: float,
    orientation: str = "South",
    collector_efficiency: float = C.SOLAR_THERMAL_EFFICIENCY,
    pipe_loss_fraction: float = 0.15,
    weather_df: pd.DataFrame | None = None,
) -> float:
    """
    Estimate annual useful heat delivered by the solar thermal system (kWh/yr).

    If weather_df is provided (columns: timestamp, solar), uses real irradiance.
    Solar column is normalised (0–1) where 1 ≈ 900 W/m²; convert to kWh/m²/hr by ×0.9.
    """
    o_factor = ORIENTATION_FACTORS.get(orientation, 1.0)

    if weather_df is not None and "solar" in weather_df.columns and len(weather_df) > 0:
        # Un-normalise and convert W/m² → kWh/m²/hr (already hourly)
        irr_kwh_m2_hr = weather_df["solar"].to_numpy(dtype=float) * 0.9
        annual_irr    = float(irr_kwh_m2_hr.sum()) / max(1, round(len(weather_df) / 8760))
    else:
        annual_irr = sum(UK_MONTHLY_IRRADIANCE_KWH_M2.values()) * o_factor

    annual_irr *= o_factor
    gross_yield  = collector_area_m2 * annual_irr * collector_efficiency
    net_yield    = gross_yield * (1 - pipe_loss_fraction)
    return max(net_yield, 0)


def solar_fraction(net_yield_kwh: float, dhw_demand_kwh: float) -> float:
    """
    Fraction of annual DHW demand met by solar (practical max 70% due to summer
    stagnation risk and overnight demand that solar cannot cover).
    """
    if dhw_demand_kwh <= 0:
        return 0.0
    raw = net_yield_kwh / dhw_demand_kwh
    return min(raw, 0.70)


# ---------------------------------------------------------------------------
# Financial calculation
# ---------------------------------------------------------------------------
def financials(
    collector_area_m2: float,
    floor_area: float,
    building_type: str,
    orientation: str,
    fuel_type: str,                # "gas" or "electricity"
    elec_price: float,
    gas_price: float,
    discount_rate: float,
    inflation_rate: float,
    grant_pct: float,
    weather_df: pd.DataFrame | None = None,
    collector_efficiency: float = C.SOLAR_THERMAL_EFFICIENCY,
    lifetime_years: int = C.SOLAR_THERMAL_LIFETIME,
    capex_per_m2: float = 650,      # installed cost £/m² collector
) -> dict:
    """
    Financial analysis for a solar thermal installation.

    Returns a dict with:
        dhw_demand_kwh, net_yield_kwh, solar_fraction_pct,
        fuel_saved_kwh, cost_saving_gbp, carbon_saved_tco2e,
        total_capex_gbp, capex_after_grant_gbp, payback_years, npv_gbp
    """
    dhw_intensity = DHW_INTENSITY.get(building_type, 5)
    dhw_demand    = dhw_intensity * floor_area

    net_yield = annual_yield_kwh(
        collector_area_m2, orientation, collector_efficiency, weather_df=weather_df
    )
    sf        = solar_fraction(net_yield, dhw_demand)
    fuel_saved = dhw_demand * sf  # kWh of fuel displaced

    fuel_price   = gas_price if fuel_type == "gas" else elec_price
    carbon_factor= C.GAS_CARBON if fuel_type == "gas" else C.CARBON_FACTOR
    cost_saving_0 = fuel_saved * fuel_price
    carbon_saved  = fuel_saved * carbon_factor / 1000  # tCO2e

    capex     = collector_area_m2 * capex_per_m2
    capex_net = capex * (1 - grant_pct / 100)
    payback   = capex_net / cost_saving_0 if cost_saving_0 > 0 else float("nan")

    npv = -capex_net
    for y in range(1, lifetime_years + 1):
        net_y = cost_saving_0 * (1 + inflation_rate) ** y
        npv  += net_y / (1 + discount_rate) ** y

    return {
        "dhw_demand_kwh":       dhw_demand,
        "net_yield_kwh":        net_yield,
        "solar_fraction_pct":   sf * 100,
        "fuel_saved_kwh":       fuel_saved,
        "cost_saving_gbp":      cost_saving_0,
        "carbon_saved_tco2e":   carbon_saved,
        "total_capex_gbp":      capex,
        "capex_after_grant_gbp": capex_net,
        "payback_years":        payback,
        "npv_gbp":              npv,
    }
