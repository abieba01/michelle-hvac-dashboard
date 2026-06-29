"""
epc_lookup.py
=============
UK Energy Performance Certificate (EPC) utilities.

Supports:
  1. Optional live lookup from the UK EPC open data API (requires free account).
  2. Manual band entry + score estimation.
  3. Projecting a new EPC band after energy improvements.

Non-domestic EPC asset rating scale:
  A+ ≤ 25 | A 26–50 | B 51–75 | C 76–100 | D 101–125 | E 126–150 | F 151–175 | G ≥ 176
"""
from __future__ import annotations

import requests
import config as C

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
    Approximate non-domestic asset rating score from kWh/m²/yr.
    Reference benchmarks from CIBSE TM46.
    """
    benchmarks = {
        "office":     250,
        "hotel":      300,
        "hospital":   450,
        "school":     150,
        "retail":     200,
        "industrial": 130,
    }
    ref = benchmarks.get(building_type, 250)
    # Score 100 = reference building; score 0 = zero energy
    return max(1.0, kwh_m2 / ref * 100)


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
