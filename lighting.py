"""
lighting.py
===========
LED lighting upgrade savings calculation for commercial buildings.
Covers all major fitting types with optional occupancy and daylight controls.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Fitting types: existing_w_m2, LED replacement w_m2, installed CAPEX/m²
# ---------------------------------------------------------------------------
FITTINGS: dict[str, dict] = {
    "t8_fluorescent": {
        "label":        "Fluorescent T8 (standard tube)",
        "existing_w_m2": 15,
        "led_w_m2":      6,
        "capex_m2":      18,
    },
    "t5_fluorescent": {
        "label":        "Fluorescent T5 (slim tube)",
        "existing_w_m2": 10,
        "led_w_m2":      5,
        "capex_m2":      15,
    },
    "halogen": {
        "label":        "Halogen (downlights / spotlights)",
        "existing_w_m2": 25,
        "led_w_m2":      4,
        "capex_m2":      22,
    },
    "metal_halide": {
        "label":        "Metal halide (high-bay)",
        "existing_w_m2": 20,
        "led_w_m2":      7,
        "capex_m2":      25,
    },
    "hps": {
        "label":        "High-pressure sodium",
        "existing_w_m2": 15,
        "led_w_m2":      6,
        "capex_m2":      20,
    },
    "led_existing": {
        "label":        "LED (already installed — controls only)",
        "existing_w_m2": 6,
        "led_w_m2":      6,
        "capex_m2":      0,
    },
}

CONTROLS: dict[str, dict] = {
    "none":     {"label": "No additional controls",      "extra_saving": 0.00, "capex_m2": 0},
    "presence": {"label": "Presence detection (PIR)",    "extra_saving": 0.25, "capex_m2": 5},
    "daylight": {"label": "Daylight dimming",            "extra_saving": 0.15, "capex_m2": 8},
    "both":     {"label": "Presence + daylight dimming", "extra_saving": 0.35, "capex_m2": 12},
}


def calculate(
    floor_area: float,
    fitting_type: str,
    control_type: str,
    area_fraction: float,       # fraction of floor area with this fitting type (0–1)
    operating_hours: float,     # hours per year lights are on
    elec_price: float,
    capex_override: float | None = None,
) -> dict:
    """
    Energy and financial saving for one fitting-type + control combination.

    Returns a dict with saving_kwh, cost_saving_gbp, capex_gbp, payback_years,
    saving_pct, and labels for display.
    """
    fitting  = FITTINGS.get(fitting_type, FITTINGS["t8_fluorescent"])
    control  = CONTROLS.get(control_type, CONTROLS["none"])
    lit_area = floor_area * area_fraction

    baseline_kw = fitting["existing_w_m2"] * lit_area / 1000
    led_base_kw = fitting["led_w_m2"] * lit_area / 1000
    control_cut = led_base_kw * control["extra_saving"]
    final_kw    = led_base_kw - control_cut

    baseline_kwh = baseline_kw * operating_hours
    led_kwh      = final_kw   * operating_hours
    saving_kwh   = max(baseline_kwh - led_kwh, 0)
    cost_saving  = saving_kwh * elec_price

    if capex_override is not None:
        capex = capex_override
    else:
        capex = (fitting["capex_m2"] + control["capex_m2"]) * lit_area

    saving_pct   = saving_kwh / baseline_kwh * 100 if baseline_kwh > 0 else 0.0
    payback      = capex / cost_saving if cost_saving > 0 else float("nan")

    return {
        "fitting_label":   fitting["label"],
        "control_label":   control["label"],
        "lit_area_m2":     lit_area,
        "baseline_kwh":    baseline_kwh,
        "led_kwh":         led_kwh,
        "saving_kwh":      saving_kwh,
        "saving_pct":      saving_pct,
        "cost_saving_gbp": cost_saving,
        "capex_gbp":       capex,
        "payback_years":   payback,
    }


def total_saving(results: list[dict]) -> dict:
    """Aggregate multiple fitting-type results into one summary dict."""
    total_base = sum(r["baseline_kwh"] for r in results)
    total_led  = sum(r["led_kwh"]      for r in results)
    total_save = sum(r["saving_kwh"]   for r in results)
    total_cost = sum(r["cost_saving_gbp"] for r in results)
    total_capex= sum(r["capex_gbp"]    for r in results)
    return {
        "total_baseline_kwh": total_base,
        "total_led_kwh":      total_led,
        "total_saving_kwh":   total_save,
        "total_saving_pct":   total_save / total_base * 100 if total_base > 0 else 0.0,
        "total_cost_saving":  total_cost,
        "total_capex":        total_capex,
        "payback_years":      total_capex / total_cost if total_cost > 0 else float("nan"),
    }
