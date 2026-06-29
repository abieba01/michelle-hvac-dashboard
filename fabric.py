"""
fabric.py
=========
Building fabric improvement energy and financial analysis.

Heat loss model (CIBSE Guide A / BS EN ISO 13790):
  Q_saving = (U_old − U_new) × area × HDD × 24 / (system_efficiency × 1000)  [kWh/yr]

Fabric savings reduce the HVAC heating load; this module calculates each
element independently and sums the results.
"""
from __future__ import annotations

import numpy as np
import config as C

# ---------------------------------------------------------------------------
# Element definitions
# ---------------------------------------------------------------------------
ELEMENTS: dict[str, dict] = {
    "wall_cavity": {
        "label":       "External wall — cavity fill",
        "u_typical":   1.6,     # existing uninsulated cavity (W/m²K)
        "u_target":    0.30,    # after insulation (Part L notional)
        "capex_m2":    15,      # installed £/m²
        "description": "Blown mineral fibre or EPS bead injection into existing cavity.",
    },
    "wall_ext": {
        "label":       "External wall — external wall insulation (EWI)",
        "u_typical":   1.6,
        "u_target":    0.18,
        "capex_m2":    80,
        "description": "EPS or mineral wool with render or cladding finish.",
    },
    "wall_int": {
        "label":       "External wall — internal wall insulation (IWI)",
        "u_typical":   1.6,
        "u_target":    0.22,
        "capex_m2":    50,
        "description": "PIR board or mineral wool with plasterboard facing.",
    },
    "roof_flat": {
        "label":       "Flat roof — additional insulation",
        "u_typical":   0.70,
        "u_target":    0.15,
        "capex_m2":    25,
        "description": "PIR or mineral wool in warm-roof construction.",
    },
    "roof_pitched": {
        "label":       "Pitched roof — loft / rafter insulation",
        "u_typical":   0.40,
        "u_target":    0.13,
        "capex_m2":    12,
        "description": "Blown or batt mineral wool between and below rafters.",
    },
    "glazing_double": {
        "label":       "Glazing — double-to-triple glazing upgrade",
        "u_typical":   2.8,
        "u_target":    0.80,
        "capex_m2":    400,
        "description": "Argon-filled triple-glazed units with low-e coating.",
    },
    "glazing_single": {
        "label":       "Glazing — single-to-double glazing upgrade",
        "u_typical":   5.6,
        "u_target":    1.60,
        "capex_m2":    300,
        "description": "Standard argon-filled double-glazed units.",
    },
    "floor": {
        "label":       "Ground floor — underfloor insulation",
        "u_typical":   0.70,
        "u_target":    0.22,
        "capex_m2":    20,
        "description": "PIR or mineral wool between/beneath floor structure.",
    },
}

# Pre-set retrofit packages: (elements to include, target U-values)
PACKAGES: dict[str, dict] = {
    "basic": {
        "label":       "Basic (quick wins)",
        "description": "Cavity wall fill + loft insulation. Minimal disruption.",
        "elements":    ["wall_cavity", "roof_pitched"],
    },
    "standard": {
        "label":       "Standard retrofit",
        "description": "Cavity/flat roof insulation + double-to-triple glazing upgrade.",
        "elements":    ["wall_cavity", "roof_flat", "glazing_double"],
    },
    "deep": {
        "label":       "Deep retrofit",
        "description": "All elements improved to Part L 2021 target U-values.",
        "elements":    list(ELEMENTS.keys()),
    },
}


# ---------------------------------------------------------------------------
# Calculation functions
# ---------------------------------------------------------------------------
def element_saving(
    u_old: float,
    u_new: float,
    area_m2: float,
    hdd_annual: float = C.HDD_ANNUAL,
    system_efficiency: float = C.HEATING_EFF,
) -> float:
    """
    Annual heating energy saved by improving one fabric element (kWh/yr).

    Q_saving = (U_old − U_new) × area × HDD × 24 / (efficiency × 1000)
    """
    delta_u = max(u_old - u_new, 0)
    return delta_u * area_m2 * hdd_annual * 24 / (system_efficiency * 1000)


def calculate(
    elements: list[dict],       # list of {key, area_m2, u_old, u_new}
    hdd_annual: float,
    fuel_type: str,             # "gas" or "electricity"
    elec_price: float,
    gas_price: float,
    discount_rate: float,
    inflation_rate: float,
    grant_pct: float,
    system_efficiency: float = C.HEATING_EFF,
    lifetime_years: int = C.FABRIC_LIFETIME,
) -> dict:
    """
    Full fabric improvement financial analysis across all specified elements.

    Each element dict must have: key, area_m2, u_old, u_new.
    Returns aggregated saving, CAPEX, payback and NPV.
    """
    fuel_price    = gas_price if fuel_type == "gas" else elec_price
    carbon_factor = C.GAS_CARBON if fuel_type == "gas" else C.CARBON_FACTOR

    rows = []
    for el in elements:
        key     = el.get("key", "wall_cavity")
        defn    = ELEMENTS.get(key, ELEMENTS["wall_cavity"])
        u_old   = el.get("u_old", defn["u_typical"])
        u_new   = el.get("u_new", defn["u_target"])
        area    = el.get("area_m2", 0)
        capex   = area * defn["capex_m2"]

        saving_kwh   = element_saving(u_old, u_new, area, hdd_annual, system_efficiency)
        cost_saving  = saving_kwh * fuel_price
        carbon_saved = saving_kwh * carbon_factor / 1000

        rows.append({
            "element":          defn["label"],
            "area_m2":          area,
            "u_old":            u_old,
            "u_new":            u_new,
            "saving_kwh":       saving_kwh,
            "cost_saving_gbp":  cost_saving,
            "carbon_tco2e":     carbon_saved,
            "capex_gbp":        capex,
            "payback_years":    capex / cost_saving if cost_saving > 0 else float("nan"),
        })

    total_saving_kwh  = sum(r["saving_kwh"]      for r in rows)
    total_cost        = sum(r["cost_saving_gbp"]  for r in rows)
    total_carbon      = sum(r["carbon_tco2e"]     for r in rows)
    total_capex       = sum(r["capex_gbp"]        for r in rows)
    capex_net         = total_capex * (1 - grant_pct / 100)
    payback           = capex_net / total_cost if total_cost > 0 else float("nan")

    npv = -capex_net
    for y in range(1, lifetime_years + 1):
        net_y = total_cost * (1 + inflation_rate) ** y
        npv  += net_y / (1 + discount_rate) ** y

    return {
        "element_rows":         rows,
        "total_saving_kwh":     total_saving_kwh,
        "total_cost_saving":    total_cost,
        "total_carbon_tco2e":   total_carbon,
        "total_capex":          total_capex,
        "capex_after_grant":    capex_net,
        "payback_years":        payback,
        "npv_gbp":              npv,
    }
