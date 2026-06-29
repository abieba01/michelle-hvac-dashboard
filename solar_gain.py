"""
solar_gain.py
=============
Physically-based hourly solar gain through glazing, by facade, replacing the
simplified sine-wave solar proxy used elsewhere in the model.

Methodology:
  - Hourly solar altitude/azimuth from a simplified NOAA solar position
    algorithm (declination, equation of time, hour angle) using the
    building's latitude from weather.py's geocoding step.
  - Incident irradiance on each vertical facade from global horizontal
    irradiance (GHI), split into a beam component (cosine-projected onto
    the facade) and isotropic sky-diffuse + ground-reflected components
    (simplified Liu-Jordan style model).
  - Transmitted solar gain: Q = irradiance x glazing_area x g_value x shading
  - CIBSE TM52 overheating risk: occupied hours where a simplified operative
    temperature (outdoor temp + solar-gain-driven uplift) exceeds the
    adaptive comfort limit are counted and flagged if the count is high.

Note: timestamps from weather.py are already in local civil time (Open-Meteo
timezone="auto"), so the equation-of-time correction below captures the
dominant solar-time offset; the smaller longitude/timezone-rounding
component (typically <30 min) is neglected, which is acceptable for the
hourly-resolution gain estimates used here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import config as C

GROUND_REFLECTANCE = 0.20   # typical urban/grass ground albedo
DIFFUSE_FRACTION    = 0.30  # fraction of GHI assumed diffuse (UK average climate)


# ---------------------------------------------------------------------------
# Solar position
# ---------------------------------------------------------------------------
def solar_position(timestamps: pd.Series, lat: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (altitude_deg, azimuth_deg) arrays for each timestamp.

    Azimuth convention: degrees clockwise from North (North=0, East=90,
    South=180, West=270). Based on the simplified NOAA solar position
    algorithm.
    """
    ts = pd.to_datetime(timestamps)
    doy = ts.dt.dayofyear.to_numpy(dtype=float)
    hour_decimal = ts.dt.hour.to_numpy(dtype=float) + ts.dt.minute.to_numpy(dtype=float) / 60.0

    gamma = 2 * np.pi / 365.0 * (doy - 1 + (hour_decimal - 12) / 24.0)

    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )  # minutes
    decl = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )  # radians

    tst = hour_decimal * 60 + eqtime  # true solar time, minutes
    hour_angle = tst / 4 - 180        # degrees

    lat_rad = np.radians(lat)
    ha_rad = np.radians(hour_angle)

    cos_zenith = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(ha_rad)
    cos_zenith = np.clip(cos_zenith, -1, 1)
    zenith = np.arccos(cos_zenith)
    altitude = 90 - np.degrees(zenith)

    sin_zenith = np.sin(zenith)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_az = (np.sin(decl) - np.sin(lat_rad) * cos_zenith) / (np.cos(lat_rad) * sin_zenith)
    cos_az = np.clip(np.nan_to_num(cos_az, nan=1.0), -1, 1)
    azimuth = np.degrees(np.arccos(cos_az))
    azimuth = np.where(hour_angle > 0, 360 - azimuth, azimuth)

    return altitude, azimuth


# ---------------------------------------------------------------------------
# Facade irradiance
# ---------------------------------------------------------------------------
def facade_irradiance_wm2(
    ghi_wm2: np.ndarray,
    altitude_deg: np.ndarray,
    sun_azimuth_deg: np.ndarray,
    facade_azimuth_deg: float,
    ground_reflectance: float = GROUND_REFLECTANCE,
    diffuse_fraction: float = DIFFUSE_FRACTION,
) -> np.ndarray:
    """
    Incident irradiance (W/m²) on a vertical facade of given azimuth.

    Splits GHI into beam + diffuse, projects beam onto the facade via the
    angle of incidence, and adds isotropic sky-diffuse and ground-reflected
    components for a vertical (90° tilt) surface.
    """
    ghi = np.asarray(ghi_wm2, dtype=float)
    alt = np.asarray(altitude_deg, dtype=float)
    sun_az = np.asarray(sun_azimuth_deg, dtype=float)

    alt_rad = np.radians(np.clip(alt, 0.1, 90))
    sun_up = alt > 5.0  # ignore near-horizon angles where beam/sin(alt) is unstable

    beam_frac = 1 - diffuse_fraction
    diffuse_ghi = ghi * diffuse_fraction
    beam_ghi = ghi * beam_frac

    # direct-normal irradiance estimated from GHI beam component, capped at a
    # physically plausible clear-sky maximum (~1100 W/m2) to avoid blow-up
    # from dividing by a small sin(altitude) near sunrise/sunset
    dni = np.where(sun_up, np.minimum(beam_ghi / np.sin(alt_rad), 1100.0), 0.0)

    az_diff = np.radians(facade_azimuth_deg - sun_az)
    cos_incidence = np.cos(alt_rad) * np.cos(az_diff)
    cos_incidence = np.clip(cos_incidence, 0, None)  # facade self-shaded when sun behind it

    beam_facade = np.where(sun_up, dni * cos_incidence, 0.0)
    sky_diffuse_facade = diffuse_ghi * 0.5            # vertical surface sees half the sky dome
    ground_facade = ghi * ground_reflectance * 0.5    # sees half the ground

    total = beam_facade + sky_diffuse_facade + ground_facade
    return np.clip(total, 0, None)


# ---------------------------------------------------------------------------
# Transmitted solar gain through glazing
# ---------------------------------------------------------------------------
def transmitted_gain_kwh(
    facade_irr_wm2: np.ndarray,
    glazing_area_m2: float,
    g_value: float,
    shading_factor: float = 1.0,
) -> np.ndarray:
    """Hourly transmitted solar gain (kWh) through one facade's glazing."""
    return facade_irr_wm2 * glazing_area_m2 * g_value * shading_factor / 1000.0


# ---------------------------------------------------------------------------
# Full multi-facade calculation
# ---------------------------------------------------------------------------
def calculate(
    weather_df: pd.DataFrame,
    lat: float,
    facades: list[dict],   # each: {label, azimuth_deg, glazing_area_m2, g_value, shading_factor}
    floor_area_m2: float,
    occupied_mask: np.ndarray | None = None,
    adaptive_limit: float = 28.0,
    exceedance_threshold: int = 50,
) -> dict:
    """
    Compute hourly and annual solar gain across all facades, plus a simplified
    TM52 overheating risk assessment.

    weather_df must have columns: timestamp, t_out, solar (normalised 0-1,
    where 1 corresponds to ~900 W/m² GHI).

    Returns a dict with: facade_rows (per-facade kWh/yr breakdown),
    hourly_gain_kwh (total across facades), total_annual_gain_kwh,
    exceedance_hours, overheating_flag, peak_facade_irradiance (W/m² by facade).
    """
    ghi = weather_df["solar"].to_numpy(dtype=float) * 900.0  # un-normalise to W/m²
    altitude, sun_az = solar_position(weather_df["timestamp"], lat)

    facade_rows = []
    hourly_total = np.zeros(len(weather_df))
    for f in facades:
        irr = facade_irradiance_wm2(ghi, altitude, sun_az, f["azimuth_deg"])
        gain_hourly = transmitted_gain_kwh(
            irr, f["glazing_area_m2"], f["g_value"], f.get("shading_factor", 1.0)
        )
        hourly_total += gain_hourly
        n_years = max(1, round(len(weather_df) / 8760))
        facade_rows.append({
            "label":            f.get("label", f"Facade {f['azimuth_deg']}°"),
            "azimuth_deg":      f["azimuth_deg"],
            "glazing_area_m2":  f["glazing_area_m2"],
            "g_value":          f["g_value"],
            "shading_factor":   f.get("shading_factor", 1.0),
            "annual_gain_kwh":  float(gain_hourly.sum() / n_years),
            "peak_irradiance_wm2": float(irr.max()),
        })

    n_years = max(1, round(len(weather_df) / 8760))
    total_annual_gain = float(hourly_total.sum() / n_years)

    # Simplified operative-temperature uplift from solar gain
    t_out = weather_df["t_out"].to_numpy(dtype=float)
    solar_gain_wm2_floor = hourly_total * 1000.0 / max(floor_area_m2, 1)
    operative_temp = t_out + solar_gain_wm2_floor * 0.025  # heuristic thermal responsiveness

    if occupied_mask is None:
        # default: assume standard 08:00-18:00 weekday occupancy
        hours = pd.to_datetime(weather_df["timestamp"]).dt.hour.to_numpy()
        dow = pd.to_datetime(weather_df["timestamp"]).dt.dayofweek.to_numpy()
        occupied_mask = (hours >= C.OPEN_HOUR) & (hours < C.CLOSE_HOUR) & (dow < 5)

    exceedance_hours = int(np.sum((operative_temp > adaptive_limit) & occupied_mask) / n_years)
    overheating_flag = exceedance_hours > exceedance_threshold

    return {
        "facade_rows":           facade_rows,
        "total_annual_gain_kwh": total_annual_gain,
        "exceedance_hours":      exceedance_hours,
        "overheating_flag":      overheating_flag,
        "adaptive_limit_c":      adaptive_limit,
    }
