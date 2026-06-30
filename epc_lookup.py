"""
epc_lookup.py
=============
UK Energy Performance Certificate (EPC) utilities.

Supports:
  1. Optional live lookup from the UK EPC open data API (requires free account).
  2. Manual band entry + score estimation.
  3. Projecting a new EPC band after energy improvements.
  4. Simplified SBEM-like asset rating using carbon-weighted BEP (Limitation 2 fix).

Non-domestic EPC asset rating scale:
  A+ ≤ 25 | A 26–50 | B 51–75 | C 76–100 | D 101–125 | E 126–150 | F 151–175 | G ≥ 176
"""
from __future__ import annotations

import requests
import config as C

# ---------------------------------------------------------------------------
# Carbon emission factors by fuel type (kgCO2e/kWh)  — BEIS 2023 values
# ---------------------------------------------------------------------------
FUEL_CO2_FACTORS: dict[str, float] = {
    "electricity":  0.207,
    "gas":          0.203,
    "oil":          0.266,
    "lpg":          0.214,
    "heat_pump":    0.069,   # electricity ÷ COP 3.0 (typical ASHP)
    "biomass":      0.031,
    "district_heat": 0.175,
}

# Reference BEP (kgCO2/m2/yr) calibrated so that a CIBSE TM46 "typical" building
# (not good practice, not poor) scores approximately 100-115 (Band D).
# A good-practice building scores ~75 (Band B/C), and a poor building ~140 (Band E/F).
# Calibrated against CIBSE TM46 benchmarks and typical UK carbon intensities.
_REF_BEP: dict[str, float] = {
    "office":      48.0,
    "hotel":       62.0,
    "hospital":    95.0,
    "school":      30.0,
    "retail":      40.0,
    "industrial":  26.0,
    "residential": 28.0,
}

# ---------------------------------------------------------------------------
# Band lookup tables
# ---------------------------------------------------------------------------
# Each entry: (band_label, min_score, max_score)
BANDS = C.EPC_BANDS   # defined in config.py

BAND_COLOURS = {
    "A+": "#00a651", "A": "#50b848", "B": "#b2d235",
    "C": "#fff200", "D": "#f7941d", "E": "#f15a29",
    "F": "#ed1c24", "G": "#991b1e",
}


def score_to_band(score: float) -> str:
    """Return the EPC band label for a given asset rating score."""
    for label, lo, hi in BANDS:
        if lo <= score <= hi:
            return label
    return "G"


def band_to_midpoint(band: str) -> float:
    """Return the midpoint score for a band (used for display only)."""
    for label, lo, hi in BANDS:
        if label == band:
            return (lo + hi) / 2
    return 150.0


def project_new_band(current_score: float, saving_pct: float) -> tuple[str, float]:
    """
    Estimate the new EPC band after reducing total site energy by saving_pct %.
    The asset rating scales roughly linearly with energy intensity.

    Returns (new_band_label, new_score).
    """
    new_score = current_score * (1 - saving_pct / 100)
    return score_to_band(new_score), new_score


def kwh_m2_to_score(kwh_m2: float, building_type: str = "office") -> float:
    """
    Approximate non-domestic asset rating score from total kWh/m2/yr.
    Legacy helper — use sbem_asset_rating() for fuel-weighted accuracy.
    """
    benchmarks = {
        "office": 250, "hotel": 300, "hospital": 450,
        "school": 150, "retail": 200, "industrial": 130,
    }
    ref = benchmarks.get(building_type, 250)
    return max(1.0, kwh_m2 / ref * 100)


def sbem_asset_rating(
    hvac_kwh_m2: float,
    lighting_kwh_m2: float = 0.0,
    dhw_kwh_m2: float = 0.0,
    other_kwh_m2: float = 0.0,
    building_type: str = "office",
    heating_fuel: str = "gas",
    pv_offset_kwh_m2: float = 0.0,
    wind_offset_kwh_m2: float = 0.0,
) -> tuple[float, str]:
    """
    Simplified SBEM-like non-domestic asset rating.

    Converts energy use by fuel to a carbon-weighted Building Energy Performance
    (BEP, kgCO2/m2/yr) and divides by the reference BEP for the building type.
    This is directionally equivalent to the SBEM NCM approach and correctly
    reflects fuel-switching and on-site generation in a way the kWh-only
    heuristic cannot.

    Parameters
    ----------
    hvac_kwh_m2       : HVAC energy intensity (kWh/m2/yr, post-improvement)
    lighting_kwh_m2   : Lighting energy intensity (kWh/m2/yr)
    dhw_kwh_m2        : Domestic hot water energy intensity (kWh/m2/yr)
    other_kwh_m2      : Small power / equipment (kWh/m2/yr)
    building_type     : building type key — determines reference BEP
    heating_fuel      : fuel used for heating ('gas', 'electricity', 'heat_pump', etc.)
    pv_offset_kwh_m2  : on-site PV generation consumed on site (kWh/m2/yr) — reduces CO2
    wind_offset_kwh_m2: on-site wind generation consumed on site (kWh/m2/yr)

    Returns
    -------
    (score, band) — score is the asset rating (100 = typical reference building)
    """
    elec_co2  = FUEL_CO2_FACTORS["electricity"]
    heat_co2  = FUEL_CO2_FACTORS.get(heating_fuel, FUEL_CO2_FACTORS["gas"])

    # Each end-use billed at its dominant fuel type
    hvac_co2      = hvac_kwh_m2   * heat_co2   # heating + cooling (cooling = electricity in practice)
    lighting_co2  = lighting_kwh_m2 * elec_co2
    dhw_co2       = dhw_kwh_m2    * heat_co2
    other_co2     = other_kwh_m2  * elec_co2
    # On-site renewable generation offsets grid electricity carbon
    pv_credit     = pv_offset_kwh_m2   * elec_co2
    wind_credit   = wind_offset_kwh_m2 * elec_co2

    bep = max(0.0, hvac_co2 + lighting_co2 + dhw_co2 + other_co2 - pv_credit - wind_credit)
    ref = _REF_BEP.get(building_type, 100.0)
    score = max(1.0, bep / ref * 100.0)
    return score, score_to_band(score)


def project_band_sbem(
    current_score: float,
    hvac_saving_pct: float = 0.0,
    lighting_saving_pct: float = 0.0,
    pv_offset_kwh_m2: float = 0.0,
    wind_offset_kwh_m2: float = 0.0,
    floor_area_m2: float = 1.0,   # kept for API compatibility; not used in calculation
    building_type: str = "office",
    heating_fuel: str = "gas",
) -> tuple[str, float, str]:
    """
    Project a new EPC band after improvements, with fuel-weighted carbon accounting.

    Parameters
    ----------
    current_score        : asset rating score before improvements
    hvac_saving_pct      : HVAC energy saving as a % of HVAC baseline
    lighting_saving_pct  : lighting energy saving as a % of lighting baseline
    pv_offset_kwh_m2     : on-site PV generation self-consumed (kWh/m2/yr)
    wind_offset_kwh_m2   : on-site wind generation self-consumed (kWh/m2/yr)
    building_type        : determines reference BEP
    heating_fuel         : fuel used for heating — affects carbon weight of HVAC saving

    Returns (new_band, new_score, narrative).
    """
    ref_bep  = _REF_BEP.get(building_type, 48.0)
    heat_co2 = FUEL_CO2_FACTORS.get(heating_fuel, FUEL_CO2_FACTORS["gas"])
    elec_co2 = FUEL_CO2_FACTORS["electricity"]

    # Reconstruct current BEP from score
    current_bep = current_score * ref_bep / 100.0  # kgCO2/m2/yr

    # HVAC saving: HVAC carbon is ~55% of total BEP for a gas-heated building.
    # Adjust the share by the relative carbon factor vs electricity to
    # account for fuel-switching correctly.
    hvac_share = 0.55 * (heat_co2 / 0.203)   # gas-normalised share
    hvac_co2_saved = current_bep * min(hvac_share, 0.80) * (hvac_saving_pct / 100)

    # Lighting saving: electricity, ~15% of total BEP
    light_co2_saved = current_bep * 0.15 * (lighting_saving_pct / 100)

    # On-site renewables: direct kgCO2/m2/yr credit at grid electricity factor
    pv_co2_credit   = pv_offset_kwh_m2   * elec_co2
    wind_co2_credit = wind_offset_kwh_m2 * elec_co2

    new_bep   = max(0.5, current_bep - hvac_co2_saved - light_co2_saved
                    - pv_co2_credit - wind_co2_credit)
    new_score = max(1.0, new_bep / ref_bep * 100.0)
    new_band  = score_to_band(new_score)

    drivers: list[str] = []
    if hvac_saving_pct > 0:
        drivers.append(f"HVAC control -{hvac_saving_pct:.0f}%")
    if lighting_saving_pct > 0:
        drivers.append(f"LED lighting -{lighting_saving_pct:.0f}%")
    if pv_offset_kwh_m2 > 0 or wind_offset_kwh_m2 > 0:
        total_re = pv_offset_kwh_m2 + wind_offset_kwh_m2
        drivers.append(f"renewables -{total_re:.1f} kWh/m2/yr")
    narrative = (
        f"Projected score {new_score:.0f} ({new_band}) from {current_score:.0f} "
        f"({score_to_band(current_score)}) after: {', '.join(drivers)}. "
        f"Fuel: {heating_fuel}." if drivers else "No improvements modelled yet."
    )
    return new_band, new_score, narrative


# ---------------------------------------------------------------------------
# API lookup (optional — requires free registration)
# ---------------------------------------------------------------------------
_EPC_API_BASE = "https://epc.opendatacommunities.org/api/v1"


def lookup_by_postcode(
    postcode: str,
    email: str,
    api_key: str,
    building_class: str = "non-domestic",
) -> list[dict]:
    """
    Fetch EPC records for a postcode from the UK EPC Open Data API.

    Parameters
    ----------
    postcode       : UK postcode string (spaces are stripped)
    email          : email registered at epc.opendatacommunities.org
    api_key        : API key from the same portal
    building_class : 'domestic' or 'non-domestic'

    Returns list of dicts; each dict represents one EPC record.
    Empty list if no records found.
    Raises requests.HTTPError on auth failures.
    """
    postcode = postcode.upper().replace(" ", "")
    endpoint = f"{_EPC_API_BASE}/{building_class}/search"
    params   = {"postcode": postcode, "size": 10}
    auth     = (email, api_key)
    headers  = {"Accept": "application/json"}

    resp = requests.get(endpoint, params=params, auth=auth,
                        headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("rows", [])


def parse_record(record: dict) -> dict:
    """
    Extract the key fields we display from a raw EPC API response row.
    Field names differ between domestic and non-domestic schemas.
    """
    return {
        "address":       record.get("address", record.get("address1", "—")),
        "postcode":      record.get("postcode", "—"),
        "band":          record.get("asset-rating-band",
                         record.get("current-energy-rating", "—")),
        "score":         record.get("asset-rating",
                         record.get("current-energy-efficiency", None)),
        "floor_area_m2": record.get("floor-area", record.get("total-floor-area", None)),
        "main_fuel":     record.get("main-fuel", record.get("main-heating-fuel", "—")),
        "lodgement_date": record.get("lodgement-date", "—"),
    }
