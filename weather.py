"""
weather.py — real hourly weather from Open-Meteo (free, no API key needed).
Falls back gracefully when offline or location not found.
"""
from __future__ import annotations
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests


_ARCHIVE  = "https://archive-api.open-meteo.com/v1/archive"
_GEOCODE  = "https://geocoding-api.open-meteo.com/v1/search"
_POSTCODE = "https://api.postcodes.io/postcodes/{}"


def _uk_postcode(postcode: str):
    try:
        r = requests.get(
            _POSTCODE.format(postcode.replace(" ", "").upper()), timeout=8
        )
        if r.status_code == 200:
            d = r.json()["result"]
            return (
                float(d["latitude"]),
                float(d["longitude"]),
                f"{d['postcode']}, {d.get('admin_district', 'UK')}",
            )
    except Exception:
        pass
    return None, None, None


def geocode(location: str) -> tuple[float, float, str]:
    """Convert a UK postcode or place name to (lat, lon, display_name).
    Raises ValueError if nothing is found."""
    loc = location.strip()
    if re.match(r"^[A-Za-z]{1,2}\d{1,2}[A-Za-z]?\s*\d[A-Za-z]{2}$", loc):
        lat, lon, name = _uk_postcode(loc)
        if lat is not None:
            return lat, lon, name
    r = requests.get(
        _GEOCODE,
        params={"name": loc, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        raise ValueError(f"Location '{location}' not found.")
    d = results[0]
    return float(d["latitude"]), float(d["longitude"]), f"{d['name']}, {d.get('country', '')}"


def fetch_weather(lat: float, lon: float, years: int = 3) -> pd.DataFrame:
    """
    Fetch `years` years of hourly temperature and solar irradiance.
    Returns DataFrame with columns: timestamp, t_out (°C), solar (0–1).
    Open-Meteo archive needs a few days lag, so end date is today minus 5 days.
    """
    end   = date.today() - timedelta(days=5)
    start = end.replace(year=end.year - years)

    r = requests.get(
        _ARCHIVE,
        params={
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start.isoformat(),
            "end_date":   end.isoformat(),
            "hourly":     "temperature_2m,shortwave_radiation,wind_speed_10m",
            "timezone":   "auto",
        },
        timeout=30,
    )
    r.raise_for_status()
    h = r.json()["hourly"]

    df = pd.DataFrame({
        "timestamp":  pd.to_datetime(h["time"]),
        "t_out":      pd.array(h["temperature_2m"],     dtype="Float64"),
        "solar_wm2":  pd.array(h["shortwave_radiation"], dtype="Float64"),
        "wind_speed": pd.array(h.get("wind_speed_10m", [None] * len(h["time"])), dtype="Float64"),
    })
    df["solar"] = (df["solar_wm2"] / 900.0).clip(0, 1)
    df = df.drop(columns=["solar_wm2"]).dropna(subset=["t_out", "solar"]).reset_index(drop=True)

    # Trim/warn to keep exactly years × 8760 rows
    target = years * 8760
    if len(df) >= target:
        df = df.tail(target).reset_index(drop=True)
    return df
