"""
data_generator.py
=================
Generates a physically-plausible hourly operating dataset for the case-study
building. Real Building Management System (BMS) logs are rarely available for a
student project, so this module synthesises an equivalent dataset that:

  * spans several years at hourly resolution (8,760 h/yr),
  * contains the weather, occupancy and control variables a real BMS would log,
  * produces an HVAC electricity demand from a transparent physics-informed
    model, and
  * is automatically calibrated so the annual HVAC total matches the
    case-study baseline (~1,000,000 kWh/yr).

The generated table is the single source of truth used to train the
machine-learning model and to drive the control optimiser.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import config as C


def _diurnal_temperature(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Outdoor dry-bulb temperature: monthly mean + diurnal sine + weather noise."""
    month_mean = np.array([C.MONTHLY_MEAN_TEMP[m - 1] for m in timestamps.month])
    # coldest ~05:00, warmest ~15:00
    hour = timestamps.hour.to_numpy()
    diurnal = -np.cos((hour - 5) / 24 * 2 * np.pi) * (C.DAILY_TEMP_SWING / 2)
    # mean-reverting (AR1/Ornstein-Uhlenbeck) synoptic weather noise: persistent
    # day-to-day swings that stay bounded around zero (unlike a random walk)
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
    """Normalised solar gain proxy (0-1), daylight only, stronger in summer."""
    hour = timestamps.hour.to_numpy()
    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)  # 0 at night
    seasonal = 0.5 + 0.5 * np.sin((timestamps.dayofyear.to_numpy() - 80) / 365 * 2 * np.pi)
    return daylight * seasonal


def _occupancy(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Fraction of design occupancy present (0-1) following a working-day curve."""
    hour = timestamps.hour.to_numpy()
    dow = timestamps.dayofweek.to_numpy()
    is_workday = np.isin(dow, list(C.WORKDAYS))

    # smooth ramp-up / lunch dip / ramp-down profile across the working day
    base = np.zeros(len(timestamps))
    occ_curve = {
        7: 0.15, 8: 0.55, 9: 0.85, 10: 0.95, 11: 0.97, 12: 0.80,
        13: 0.82, 14: 0.95, 15: 0.93, 16: 0.85, 17: 0.55, 18: 0.20, 19: 0.05,
    }
    for h, frac in occ_curve.items():
        base[hour == h] = frac
    base[~is_workday] = 0.0

    # day-to-day variability + a little out-of-hours presence
    base = base * rng.normal(1.0, 0.08, len(timestamps)).clip(0.6, 1.2)
    base[(~is_workday)] += rng.uniform(0, 0.05, (~is_workday).sum())
    return np.clip(base, 0, 1)


def _baseline_schedule(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Conventional 'dumb' fixed schedule representative of an unoptimised building:
    plant runs 06:00-20:00 every weekday (well beyond actual occupancy at the
    shoulders) plus a half-day Saturday facilities run 08:00-17:00. Combined with
    fixed design-rate ventilation, this is what leaves room for optimisation.
    """
    hour = timestamps.hour.to_numpy()
    dow = timestamps.dayofweek.to_numpy()
    weekday = np.isin(dow, [0, 1, 2, 3, 4])
    saturday = dow == 5
    weekday_run = (hour >= 6) & (hour < 19) & weekday
    saturday_run = (hour >= 9) & (hour < 13) & saturday
    return (weekday_run | saturday_run).astype(float)


def hvac_energy_kw(t_out, solar, occ_frac, system_on,
                   cool_set=C.COOLING_SETPOINT, heat_set=C.HEATING_SETPOINT,
                   deadband=C.DEADBAND, fan_factor=1.0, vent_frac=1.0, scale=1.0):
    """
    Physics-informed instantaneous HVAC electrical demand (kW).

    The response surface is intentionally transparent: a base fan/pump load that
    exists whenever plant runs, plus cooling and heating coil loads driven by the
    gap between outdoor temperature (lifted by solar and occupant gains) and the
    comfort set-points, plus a ventilation term. `vent_frac` is the ventilation
    rate as a fraction of the design rate: a conventional system fixes it at 1.0
    (it over-ventilates a half-empty floor), whereas demand-controlled ventilation
    lets it track actual occupancy. `scale` is fixed once by calibrate_scale().
    All control levers feed in here, which is what lets the optimiser change them
    and observe the effect.
    """
    t_out = np.asarray(t_out, dtype=float)
    solar = np.asarray(solar, dtype=float)
    occ_frac = np.asarray(occ_frac, dtype=float)
    system_on = np.asarray(system_on, dtype=float)

    # effective indoor heat driver: outdoor temp + solar gain + occupant gain
    effective_temp = t_out + 4.0 * solar + 3.0 * occ_frac

    cool_demand = np.clip(effective_temp - (cool_set + deadband), 0, None)
    heat_demand = np.clip((heat_set - deadband) - t_out, 0, None)

    base_fan_pump = 28.0 * fan_factor          # kW when plant runs
    cooling = 7.5 * cool_demand
    heating = 6.0 * heat_demand
    ventilation = 30.0 * vent_frac * fan_factor

    demand = system_on * (base_fan_pump + cooling + heating + ventilation)
    return demand * scale


def calibrate_scale(df: pd.DataFrame) -> float:
    """Find the scale factor that makes baseline annual HVAC == target kWh."""
    raw = hvac_energy_kw(df.t_out, df.solar, df.occ_frac, df.system_on,
                         vent_frac=df.vent_frac)
    annual_raw = raw.sum() / C.SIM_YEARS          # kWh/yr at scale=1
    return C.TARGET_ANNUAL_HVAC_KWH / annual_raw


def generate(years: int = C.SIM_YEARS, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Build and return the full hourly dataset as a DataFrame."""
    rng = np.random.default_rng(seed)
    periods = years * 8760
    ts = pd.date_range("2022-01-01", periods=periods, freq="h")

    df = pd.DataFrame({"timestamp": ts})
    df["hour"] = ts.hour
    df["dayofweek"] = ts.dayofweek
    df["month"] = ts.month
    df["is_workday"] = df["dayofweek"].isin(C.WORKDAYS).astype(int)
    df["t_out"] = _diurnal_temperature(ts, rng)
    df["solar"] = _solar_gain(ts)
    df["occ_frac"] = _occupancy(ts, rng)
    df["occupancy"] = (df["occ_frac"] * C.PEAK_OCCUPANCY).round().astype(int)

    # conventional control (the levers the optimiser will later change)
    df["system_on"] = _baseline_schedule(ts)
    df["cool_set"] = C.COOLING_SETPOINT
    df["heat_set"] = C.HEATING_SETPOINT
    df["deadband"] = C.DEADBAND
    df["fan_factor"] = 1.0
    df["vent_frac"] = 1.0          # fixed design-rate ventilation (no DCV)

    scale = calibrate_scale(df)
    df.attrs["scale"] = scale

    df["hvac_kwh"] = hvac_energy_kw(
        df.t_out, df.solar, df.occ_frac, df.system_on,
        df.cool_set, df.heat_set, df.deadband, df.fan_factor, df.vent_frac, scale,
    )
    # small metering noise so the ML model has something realistic to fit
    df["hvac_kwh"] = (df["hvac_kwh"] * rng.normal(1.0, 0.03, len(df))).clip(lower=0)
    return df


def make_training_data(df: pd.DataFrame, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """
    Build an AUGMENTED dataset for fitting the surrogate model.

    The baseline operating log holds the control levers constant, so a model
    trained on it alone cannot learn how energy responds to set-point, dead-band
    or fan changes. Here we take the real weather/occupancy conditions and append
    perturbed copies in which the control levers are randomised across their
    plausible operating ranges, recomputing the physics-based energy each time.
    The model therefore learns the full control response surface, which is what
    makes its counterfactual predictions trustworthy for the optimiser.
    """
    rng = np.random.default_rng(seed + 1)
    scale = df.attrs["scale"]
    blocks = [df.copy()]

    for _ in range(3):  # three perturbed replicas
        p = df.copy()
        n = len(p)
        # randomise occupancy-aware scheduling, set-points, dead-band, fan speed
        p["system_on"] = ((p["occ_frac"] > rng.uniform(0.02, 0.3, n)) |
                          (p["system_on"] > 0)).astype(float)
        p["system_on"] *= rng.binomial(1, 0.95, n)
        p["cool_set"] = rng.uniform(22.0, 26.0, n)
        p["heat_set"] = rng.uniform(17.0, 21.0, n)
        p["deadband"] = rng.uniform(0.5, 3.0, n)
        p["fan_factor"] = rng.uniform(0.6, 1.05, n)
        p["vent_frac"] = rng.uniform(0.2, 1.0, n)
        p["hvac_kwh"] = hvac_energy_kw(
            p.t_out, p.solar, p.occ_frac, p.system_on,
            p.cool_set, p.heat_set, p.deadband, p.fan_factor, p.vent_frac, scale,
        )
        p["hvac_kwh"] = (p["hvac_kwh"] * rng.normal(1.0, 0.03, n)).clip(lower=0)
        blocks.append(p)

    out = pd.concat(blocks, ignore_index=True)
    out.attrs["scale"] = scale
    return out


if __name__ == "__main__":
    data = generate()
    annual = data["hvac_kwh"].sum() / C.SIM_YEARS
    print(f"Rows: {len(data):,}  ({C.SIM_YEARS} years hourly)")
    print(f"Calibration scale factor: {data.attrs['scale']:.4f}")
    print(f"Baseline annual HVAC energy: {annual:,.0f} kWh "
          f"(target {C.TARGET_ANNUAL_HVAC_KWH:,.0f})")
    print(data.head())
