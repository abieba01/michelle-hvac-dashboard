"""
planning_check.py
=================
UK planning constraint screening tool.

Uses the Planning Data API (planning.data.gov.uk — free, no key required) to
check for statutory designations near a building's location.

Datasets checked:
  - listed-building         (Grade I, II*, II)
  - conservation-area
  - national-park
  - area-of-outstanding-natural-beauty
  - world-heritage-site

For each technology type, returns whether full planning permission is likely
required beyond normal Permitted Development rights.
"""
from __future__ import annotations

import requests

_API = "https://www.planning.data.gov.uk/entity.json"
_TIMEOUT = 6  # seconds per request

_DATASETS = [
    ("listed-building",                    "listed_building",     "Listed building"),
    ("conservation-area",                  "conservation_area",   "Conservation area"),
    ("national-park",                      "national_park",       "National Park"),
    ("area-of-outstanding-natural-beauty", "aonb",                "Area of Outstanding Natural Beauty"),
    ("world-heritage-site",                "world_heritage_site", "World Heritage Site"),
]

# Technologies that automatically need full planning regardless of designation
_ALWAYS_FULL_PLANNING = [
    "Wind turbine — always requires full planning permission on commercial buildings "
    "(not permitted development). A planning application is needed before any installation.",
]

# Extra restrictions triggered by each designation
_DESIGNATION_RESTRICTIONS: dict[str, list[str]] = {
    "listed_building": [
        "Solar PV / solar thermal — any roof installation on a listed building needs "
        "Listed Building Consent from the local planning authority.",
        "External wall insulation — alterations to the external appearance of a listed "
        "building require Listed Building Consent.",
        "Ground source heat pump — excavation near a listed building may require consent.",
    ],
    "conservation_area": [
        "Solar PV / solar thermal — panels that are visible from a public highway in a "
        "conservation area are not permitted development and require planning permission.",
        "External wall insulation — changes to the external appearance may need consent "
        "depending on the local authority's conservation area appraisal.",
    ],
    "national_park": [
        "Solar PV ground-mount — stricter assessment applies in National Parks. "
        "Any system covering more than 9 m² requires full planning permission.",
        "Wind turbine — not permitted development within a National Park.",
    ],
    "aonb": [
        "Solar PV ground-mount — stricter assessment applies in AONBs.",
        "Wind turbine — not permitted development within an AONB.",
    ],
    "world_heritage_site": [
        "All visible external renewable technologies require planning permission "
        "and impact assessment in World Heritage Sites.",
    ],
}


def check(lat: float, lon: float, radius_m: int = 200) -> dict:
    """
    Check planning constraints within radius_m metres of lat/lon.

    Returns:
        listed_building       : bool
        listed_building_grade : str | None
        conservation_area     : bool
        national_park         : bool
        aonb                  : bool
        world_heritage_site   : bool
        constraints           : list[str]  — human-readable triggered constraints
        restrictions          : list[str]  — tech-specific planning notes
        api_error             : bool       — True if any API call failed
    """
    result: dict = {
        "listed_building":       False,
        "listed_building_grade": None,
        "conservation_area":     False,
        "national_park":         False,
        "aonb":                  False,
        "world_heritage_site":   False,
        "constraints":           [],
        "restrictions":          list(_ALWAYS_FULL_PLANNING),
        "api_error":             False,
    }

    for dataset, key, label in _DATASETS:
        try:
            resp = requests.get(
                _API,
                params={
                    "dataset":    dataset,
                    "latitude":   round(lat, 6),
                    "longitude":  round(lon, 6),
                    "radius":     radius_m,
                },
                timeout=_TIMEOUT,
            )
            if resp.status_code == 200:
                entities = resp.json().get("entities", [])
                if entities:
                    result[key] = True
                    result["constraints"].append(label)
                    result["restrictions"].extend(_DESIGNATION_RESTRICTIONS.get(key, []))
                    if dataset == "listed-building":
                        e0 = entities[0]
                        grade = (
                            e0.get("listed-building-grade")
                            or e0.get("name", "")
                        )
                        result["listed_building_grade"] = grade or "Unknown"
            else:
                result["api_error"] = True
        except Exception:
            result["api_error"] = True

    return result


def grid_connection_notes(capacity_kwp: float = 0.0,
                          turbine_kw: float = 0.0) -> list[str]:
    """
    Return DNO grid connection guidance based on system size.
    """
    notes = []
    total_kw = capacity_kwp + turbine_kw

    if total_kw == 0:
        return notes

    if total_kw <= 3.68:
        notes.append(
            "G98 self-notification: systems up to 3.68 kW single-phase can be "
            "connected under G98 — notify your DNO within 28 days of commissioning. "
            "No prior approval needed."
        )
    elif total_kw <= 50:
        notes.append(
            "G98/G99 threshold: systems between 3.68 kW and 50 kW require a G99 "
            "application to your Distribution Network Operator (DNO) before installation. "
            "Allow 45 working days for approval."
        )
    else:
        notes.append(
            f"Large system ({total_kw:.0f} kW): full G99 application required. "
            "DNO approval typically takes 3-12 months. A connection cost quote "
            "(typically GBP 2,000-50,000+) will be provided by the DNO."
        )

    if total_kw > 16:
        notes.append(
            "DNO reinforcement may be required if the local network has limited "
            "export headroom. Check your DNO's interactive capacity map before "
            "committing to system size."
        )
    return notes
