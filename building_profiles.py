"""
building_profiles.py — occupancy curves and operating parameters for each building type.
Each profile drives the synthetic data generator and controls which optimisation
strategies are applicable (e.g. occupancy scheduling is N/A for 24/7 buildings).
"""
from __future__ import annotations

PROFILES: dict[str, dict] = {
    "office": {
        "label":          "Commercial Office",
        "description":    "Mon–Fri, 08:00–18:00",
        "is_247":         False,
        "min_occ":        0.00,
        "workdays":       {0, 1, 2, 3, 4},
        "holiday_months": set(),
        "occ_curve": {
            7: 0.15, 8: 0.55, 9: 0.85, 10: 0.95, 11: 0.97,
            12: 0.80, 13: 0.82, 14: 0.95, 15: 0.93,
            16: 0.85, 17: 0.55, 18: 0.20, 19: 0.05,
        },
        "hvac_kwh_m2":    100,
        "m2_per_person":  20,
    },
    "hotel": {
        "label":          "Hotel",
        "description":    "24/7 — high evening/night occupancy",
        "is_247":         True,
        "min_occ":        0.25,
        "workdays":       {0, 1, 2, 3, 4, 5, 6},
        "holiday_months": set(),
        "occ_curve": {
            0: 0.75, 1: 0.80, 2: 0.82, 3: 0.80, 4: 0.72, 5: 0.58,
            6: 0.52, 7: 0.48, 8: 0.42, 9: 0.38, 10: 0.33, 11: 0.38,
            12: 0.48, 13: 0.52, 14: 0.50, 15: 0.55, 16: 0.62,
            17: 0.72, 18: 0.80, 19: 0.88, 20: 0.92, 21: 0.90,
            22: 0.86, 23: 0.82,
        },
        "hvac_kwh_m2":    120,
        "m2_per_person":  30,
    },
    "hospital": {
        "label":          "Hospital / Care Home",
        "description":    "24/7 — always occupied, strict ventilation",
        "is_247":         True,
        "min_occ":        0.55,
        "workdays":       {0, 1, 2, 3, 4, 5, 6},
        "holiday_months": set(),
        "occ_curve":      {h: 0.65 + 0.20 * int(8 <= h <= 17) for h in range(24)},
        "hvac_kwh_m2":    160,
        "m2_per_person":  15,
    },
    "school": {
        "label":          "School / University",
        "description":    "Term-time Mon–Fri, 08:00–17:00. Long summer shutdown.",
        "is_247":         False,
        "min_occ":        0.00,
        "workdays":       {0, 1, 2, 3, 4},
        "holiday_months": {7, 8},   # July & August — near-empty
        "occ_curve": {
            7: 0.10, 8: 0.65, 9: 0.92, 10: 0.96, 11: 0.96,
            12: 0.70, 13: 0.78, 14: 0.92, 15: 0.85,
            16: 0.38, 17: 0.14, 18: 0.05,
        },
        "hvac_kwh_m2":    75,
        "m2_per_person":  5,
    },
    "retail": {
        "label":          "Retail",
        "description":    "Mon–Sat 09:00–21:00, Sun 10:00–17:00",
        "is_247":         False,
        "min_occ":        0.00,
        "workdays":       {0, 1, 2, 3, 4, 5, 6},
        "holiday_months": set(),
        "occ_curve": {
            9: 0.30, 10: 0.52, 11: 0.68, 12: 0.82, 13: 0.88,
            14: 0.82, 15: 0.78, 16: 0.72, 17: 0.82, 18: 0.78,
            19: 0.62, 20: 0.42, 21: 0.15,
        },
        "hvac_kwh_m2":    85,
        "m2_per_person":  10,
    },
    "industrial": {
        "label":          "Industrial / Warehouse",
        "description":    "Mon–Fri, two-shift pattern 06:00–20:00",
        "is_247":         False,
        "min_occ":        0.00,
        "workdays":       {0, 1, 2, 3, 4},
        "holiday_months": set(),
        "occ_curve": {
            6: 0.50, 7: 0.90, 8: 0.96, 9: 0.96, 10: 0.96, 11: 0.96,
            12: 0.68, 13: 0.92, 14: 0.96, 15: 0.96, 16: 0.96,
            17: 0.70, 18: 0.50, 19: 0.28, 20: 0.08,
        },
        "hvac_kwh_m2":    60,
        "m2_per_person":  50,
    },
}

DEFAULT = "office"


def get_profile(name: str) -> dict:
    return PROFILES.get(name, PROFILES[DEFAULT])
