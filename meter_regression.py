"""
meter_regression.py
===================
IPMVP Option B degree-day regression.

Separates the weather-driven HVAC load from the weather-independent base load
using monthly electricity bill data and monthly degree days. Requires at least
6 months of data; 12+ months gives reliable results.

Model: total_kWh = base + a × HDD + b × CDD
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Degree-day helpers
# ---------------------------------------------------------------------------
HDD_BASE = 15.5  # heating degree-day base temperature (°C, UK standard)
CDD_BASE = 22.0  # cooling degree-day base temperature (°C)


def degree_days_from_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly HDD and CDD from an hourly Open-Meteo weather DataFrame
    (columns: timestamp, t_out). Returns a DataFrame with columns:
    year, month, hdd, cdd, mean_temp.
    """
    df = weather_df[["timestamp", "t_out"]].copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    daily = df.groupby("date")["t_out"].mean().reset_index()
    daily.columns = ["date", "daily_mean"]
    daily["date"]    = pd.to_datetime(daily["date"])
    daily["year"]    = daily["date"].dt.year
    daily["month"]   = daily["date"].dt.month
    daily["hdd_day"] = (HDD_BASE - daily["daily_mean"]).clip(lower=0)
    daily["cdd_day"] = (daily["daily_mean"] - CDD_BASE).clip(lower=0)

    monthly = (
        daily.groupby(["year", "month"])
        .agg(hdd=("hdd_day", "sum"), cdd=("cdd_day", "sum"),
             mean_temp=("daily_mean", "mean"))
        .reset_index()
    )
    return monthly


def degree_days_from_monthly_temps(monthly_mean_temps: list[float]) -> tuple[list, list]:
    """
    Approximate monthly HDD and CDD from 12 monthly mean temperatures.
    Uses a ±5 °C daily swing assumption (suitable when no hourly data available).
    """
    hdds, cdds = [], []
    for t in monthly_mean_temps:
        hdd = max(0.0, HDD_BASE - t) * 30
        cdd = max(0.0, t - CDD_BASE) * 30
        hdds.append(hdd)
        cdds.append(cdd)
    return hdds, cdds


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def regress(
    monthly_kwh:  list[float],
    monthly_hdd:  list[float],
    monthly_cdd:  list[float],
) -> dict:
    """
    Fit total_kWh = base + a × HDD + b × CDD by OLS.

    Returns a dict with:
        base_monthly_kwh   – weather-independent monthly base load
        hdd_coefficient    – heating sensitivity (kWh / HDD)
        cdd_coefficient    – cooling sensitivity (kWh / CDD)
        annual_total_kwh
        annual_base_kwh    – non-HVAC annual consumption
        annual_hvac_kwh    – HVAC-attributed annual consumption
        hvac_share_pct
        r2                 – model R²
        n_months
        warning            – string if data quality is low, else None
    """
    n = len(monthly_kwh)
    if n < 6:
        raise ValueError("At least 6 months of data are required.")

    y = np.array(monthly_kwh, dtype=float)
    X = np.column_stack([
        np.ones(n),
        np.array(monthly_hdd, dtype=float),
        np.array(monthly_cdd, dtype=float),
    ])

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    base_per_month, hdd_coeff, cdd_coeff = coeffs

    # Non-physical negative coefficients → clip to zero
    hdd_coeff = max(float(hdd_coeff), 0)
    cdd_coeff = max(float(cdd_coeff), 0)

    y_pred = X @ np.array([base_per_month, hdd_coeff, cdd_coeff])
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = max(1 - ss_res / ss_tot, 0.0) if ss_tot > 0 else 0.0

    annual_total = float(sum(monthly_kwh))
    annual_base  = max(float(base_per_month) * 12, 0)
    annual_hvac  = max(annual_total - annual_base, 0)
    hvac_share   = annual_hvac / annual_total if annual_total > 0 else 0.5

    warning = None
    if r2 < 0.70:
        warning = (
            f"R² = {r2:.2f} — low confidence. "
            "Consider providing more months of data or checking for data anomalies."
        )
    elif hdd_coeff == 0 and cdd_coeff == 0:
        warning = (
            "Both HDD and CDD coefficients are zero — degree days may lack variation. "
            "Try a location with more seasonal temperature swing."
        )

    return {
        "base_monthly_kwh":  float(base_per_month),
        "hdd_coefficient":   hdd_coeff,
        "cdd_coefficient":   cdd_coeff,
        "annual_total_kwh":  annual_total,
        "annual_base_kwh":   annual_base,
        "annual_hvac_kwh":   annual_hvac,
        "hvac_share_pct":    hvac_share * 100,
        "r2":                r2,
        "n_months":          n,
        "warning":           warning,
    }
