"""
data_generator.py
=================
Generates a physically-plausible hourly operating dataset for the case-study
building. Supports real weather data (from weather.py) or synthetic fallback,
and multiple building type profiles (from building_profiles.py).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import config as C
from building_profiles import get_profile


def _diurnal_temperature(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Synthetic outdoor temperature: monthly mean + diurnal sine + AR1 weather noise."""
    month_mean = np.array([C.MONTHLY_MEAN_TEMP[m - 1] for m in timestamps.month])
    hour = timestamps.hour.to_numpy()
    diurnal = -np.cos((hour - 5) / 24 * 2 * np.pi) * (C.DAILY_TEMP_SWING / 2)
    n = len(timestamps)
    phi, sigma = 0.96, 1.1
    eps = rng.normal(0, sigma, n)
    noise = np.empty(n)
    noise[0] = eps[0]
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + eps[i]
    noise -= noise.mean()
    return month_mean + diurnal + noise


def _solar_gain(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Normalised solar gain proxy (0–1), daylight only, stronger in summer."""
    hour = timestamps.hour.to_numpy()
    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    seasonal = 0.5 + 0.5 * np.sin((timestamps.dayofyear.to_numpy() - 80) / 365 * 2 * np.pi)
    return daylight * seasonal


def _occupancy(timestamps: pd.DatetimeIndex, rng: np.random.Generator,
               profile: dict | None = None) -> np.ndarray:
    """
    Fraction of design occupancy present (0–1).
    Uses the building-type profile's occupancy curve, workdays and holiday months.
    """
    if profile is None:
        profile = get_profile("office")

    hour      = timestamps.hour.to_numpy()
    dow       = timestamps.dayofweek.to_numpy()
    month     = timestamps.month.to_numpy()
    is_work   = np.isin(dow, list(profile["workdays"]))
    is_hol    = np.isin(month, list(profile.get("holiday_months", set())))

    base = np.zeros(len(timestamps))
    for h, frac in profile["occ_curve"].items():
        base[hour == h] = frac
    base[~is_work] = 0.0

    # School/university: skeleton staff during summer holidays
    if is_hol.any():
        base[is_hol] *= 0.05

    # Day-to-day variability and small out-of-hours presence
    base = base * rng.normal(1.0, 0.08, len(timestamps)).clip(0.6, 1.2)
    base[~is_work] += rng.uniform(0, 0.04, (~is_work).sum())

    # 24/7 buildings: enforce minimum occupancy floor
    min_occ = profile.get("min_occ", 0.0)
    return np.clip(base, min_occ, 1.0)


def _baseline_schedule(timestamps: pd.DatetimeIndex,
                       profile: dict | None = None) -> np.ndarray:
    """
    Conventional 'dumb' fixed schedule for the given building type.
    Office default: 06:00–19:00 weekday + Saturday 09:00–13:00.
    24/7 buildings: always on.
    """
    if profile is None:
        profile = get_profile("office")

    if profile["is_247"]:
        return np.ones(len(timestamps))

    hour    = timestamps.hour.to_numpy()
    dow     = timestamps.dayofweek.to_numpy()
    workdays = list(profile["workdays"])

    # Derive run hours from profile: 1 hour before open to 1 hour after close
    open_h  = min(profile["occ_curve"].keys()) - 1
    close_h = max(profile["occ_curve"].keys()) + 1
    weekday = np.isin(dow, workdays)
    run     = (hour >= max(open_h, 6)) & (hour < min(close_h, 22)) & weekday

    # Office: add Saturday half-day (original behaviour)
    if "office" in profile.get("label", "").lower():
        saturday = dow == 5
        run = run | ((hour >= 9) & (hour < 13) & saturday)

    return run.astype(float)


def hvac_energy_kw(t_out, solar, occ_frac, system_on,
                   cool_set=C.COOLING_SETPOINT, heat_set=C.HEATING_SETPOINT,
                   deadband=C.DEADBAND, fan_factor=1.0, vent_frac=1.0, scale=1.0):
    """
    Physics-informed instantaneous HVAC electrical demand (kW).

    Fan/pump base load now uses the cubic fan law (power ∝ speed³), which is
    the physically correct relationship for variable-speed drives.
    All other terms remain linear in their respective drivers.
    """
    t_out     = np.asarray(t_out,     dtype=float)
    solar     = np.asarray(solar,     dtype=float)
    occ_frac  = np.asarray(occ_frac,  dtype=float)
    system_on = np.asarray(system_on, dtype=float)

    effective_temp = t_out + 4.0 * solar + 3.0 * occ_frac
    cool_demand    = np.clip(effective_temp - (cool_set + deadband), 0, None)
    heat_demand    = np.clip((heat_set - deadband) - t_out,          0, None)

    # Cubic fan law: fan power ∝ (speed ratio)³  (was linear — Feature 1.6 fix)
    base_fan_pump = 28.0 * (fan_factor ** 3)
    cooling       = 7.5  * cool_demand
    heating       = 6.0  * heat_demand
    ventilation   = 30.0 * vent_frac * fan_factor   # volume linear with speed

    demand = system_on * (base_fan_pump + cooling + heating + ventilation)
    return demand * scale


def calibrate_scale(df: pd.DataFrame,
                    target_kwh: float | None = None,
                    sim_years: int = C.SIM_YEARS) -> float:
    """
    Find the scale factor that makes the baseline annual HVAC match target_kwh.
    target_kwh defaults to the building-intensity × floor-area product from config.
    """
    if target_kwh is None:
        target_kwh = C.TARGET_ANNUAL_HVAC_KWH
    raw        = hvac_energy_kw(df.t_out, df.solar, df.occ_frac, df.system_on,
                                vent_frac=df.vent_frac)
    annual_raw = raw.sum() / sim_years
    return target_kwh / annual_raw if annual_raw > 0 else 1.0


def generate(years: int = C.SIM_YEARS, seed: int = C.RANDOM_SEED,
             weather_df: pd.DataFrame | None = None,
             profile_name: str = "office",
             floor_area: int | None = None) -> pd.DataFrame:
    """
    Build and return the full hourly dataset as a DataFrame.

    Parameters
    ----------
    years        : simulation years (used only when weather_df is None)
    seed         : random seed
    weather_df   : optional real weather from weather.fetch_weather()
                   must contain columns: timestamp, t_out, solar
    profile_name : building type key from building_profiles.PROFILES
    floor_area   : override for FLOOR_AREA_M2 (scales target kWh)
    """
    profile  = get_profile(profile_name)
    rng      = np.random.default_rng(seed)

    if floor_area is None:
        floor_area = C.FLOOR_AREA_M2

    # ── Timestamps and weather ───────────────────────────────────────────────
    if weather_df is not None and len(weather_df) > 0:
        n_rows    = len(weather_df)
        years_act = max(1, round(n_rows / 8760))
        ts        = pd.DatetimeIndex(weather_df["timestamp"])
        t_out_arr = weather_df["t_out"].to_numpy(dtype=float)
        solar_arr = weather_df["solar"].to_numpy(dtype=float)
    else:
        years_act = years
        periods   = years * 8760
        ts        = pd.date_range("2022-01-01", periods=periods, freq="h")
        t_out_arr = _diurnal_temperature(ts, rng)
        solar_arr = _solar_gain(ts)

    df = pd.DataFrame({"timestamp": ts})
    df["hour"]       = ts.hour
    df["dayofweek"]  = ts.dayofweek
    df["month"]      = ts.month
    peak_occ = floor_area // profile.get("m2_per_person", C.OCCUPANCY_DENSITY_M2)
    df["is_workday"] = pd.Series(ts.dayofweek).isin(profile["workdays"]).astype(int).values
    df["t_out"]      = t_out_arr
    df["solar"]      = solar_arr
    df["occ_frac"]   = _occupancy(ts, rng, profile)
    df["occupancy"]  = (df["occ_frac"] * peak_occ).round().astype(int)

    df["system_on"]  = _baseline_schedule(ts, profile)
    df["cool_set"]   = C.COOLING_SETPOINT
    df["heat_set"]   = C.HEATING_SETPOINT
    df["deadband"]   = C.DEADBAND
    df["fan_factor"] = 1.0
    df["vent_frac"]  = 1.0

    # Target scales with floor area: energy intensity × m²
    target_kwh = C.HVAC_ENERGY_INTENSITY_KWH_M2 * floor_area
    # Adjust by building-type intensity ratio (hospital uses more than office)
    profile_intensity_ratio = profile.get("hvac_kwh_m2", 100) / 100
    target_kwh *= profile_intensity_ratio

    scale = calibrate_scale(df, target_kwh=target_kwh, sim_years=years_act)
    df.attrs["scale"]      = scale
    df.attrs["floor_area"] = floor_area
    df.attrs["profile"]    = profile_name

    df["hvac_kwh"] = hvac_energy_kw(
        df.t_out, df.solar, df.occ_frac, df.system_on,
        df.cool_set, df.heat_set, df.deadband, df.fan_factor, df.vent_frac, scale,
    )
    df["hvac_kwh"] = (df["hvac_kwh"] * rng.normal(1.0, 0.03, len(df))).clip(lower=0)
    return df


def make_training_data(df: pd.DataFrame, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """
    Build an augmented dataset for fitting the surrogate model.
    Preserves weather/occupancy conditions; randomises control levers.
    """
    rng   = np.random.default_rng(seed + 1)
    scale = df.attrs.get("scale", 1.0)
    blocks = [df.copy()]

    for _ in range(3):
        p = df.copy()
        n = len(p)
        p["system_on"] = ((p["occ_frac"] > rng.uniform(0.02, 0.3, n)) |
                          (p["system_on"] > 0)).astype(float)
        p["system_on"] *= rng.binomial(1, 0.95, n)
        p["cool_set"]   = rng.uniform(22.0, 26.0, n)
        p["heat_set"]   = rng.uniform(17.0, 21.0, n)
        p["deadband"]   = rng.uniform(0.5,   3.0, n)
        p["fan_factor"] = rng.uniform(0.6,   1.05, n)
        p["vent_frac"]  = rng.uniform(0.2,   1.0, n)
        p["hvac_kwh"]   = hvac_energy_kw(
            p.t_out, p.solar, p.occ_frac, p.system_on,
            p.cool_set, p.heat_set, p.deadband, p.fan_factor, p.vent_frac, scale,
        )
        p["hvac_kwh"] = (p["hvac_kwh"] * rng.normal(1.0, 0.03, n)).clip(lower=0)
        blocks.append(p)

    out = pd.concat(blocks, ignore_index=True)
    out.attrs["scale"]      = scale
    out.attrs["floor_area"] = df.attrs.get("floor_area", C.FLOOR_AREA_M2)
    out.attrs["profile"]    = df.attrs.get("profile", "office")
    return out


if __name__ == "__main__":
    data = generate()
    annual = data["hvac_kwh"].sum() / C.SIM_YEARS
    print(f"Rows: {len(data):,}  ({C.SIM_YEARS} years hourly)")
    print(f"Calibration scale factor: {data.attrs['scale']:.4f}")
    print(f"Baseline annual HVAC energy: {annual:,.0f} kWh  (target {C.TARGET_ANNUAL_HVAC_KWH:,.0f})")
    print(data.head())
