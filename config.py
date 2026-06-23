"""
config.py
=========
Central assumptions for the HVAC optimisation AI model.

Every constant used by the data generator, the machine-learning model and the
control optimiser lives here so the whole study can be re-calibrated from a
single place. The defaults reproduce the case-study office building described in
the project report (10,000 m2, ~2,000,000 kWh/yr, HVAC ~= 50% of load).
"""

# ----------------------------------------------------------------------------
# Building physical description
# ----------------------------------------------------------------------------
FLOOR_AREA_M2 = 10_000          # gross internal floor area
PEAK_OCCUPANCY = 500            # design occupancy (people)
OPEN_HOUR = 8                   # nominal building open time (local)
CLOSE_HOUR = 18                 # nominal building close time (local)
WORKDAYS = {0, 1, 2, 3, 4}     # Monday=0 ... Sunday=6

# ----------------------------------------------------------------------------
# Energy baseline (used to calibrate the synthetic generator)
# ----------------------------------------------------------------------------
ANNUAL_ELECTRICITY_KWH = 2_000_000      # whole-building electricity
HVAC_SHARE_OF_LOAD = 0.50               # HVAC fraction of total electricity
TARGET_ANNUAL_HVAC_KWH = ANNUAL_ELECTRICITY_KWH * HVAC_SHARE_OF_LOAD  # 1,000,000

# ----------------------------------------------------------------------------
# Economic and environmental factors
# ----------------------------------------------------------------------------
ELECTRICITY_PRICE = 0.28        # GBP per kWh (commercial tariff)
CARBON_FACTOR = 0.207           # kgCO2e per kWh (grid electricity)
CURRENCY = "GBP"
DISCOUNT_RATE = 0.06            # for net-present-value calculations
MEASURE_LIFETIME_YEARS = 15     # assumed equipment life

# Indicative installed capital cost of each measure (GBP)
CAPEX = {
    "occupancy_scheduling": 45_000,
    "smart_thermostats":    30_000,
    "bas":                  150_000,
    "combined":             200_000,
}

# ----------------------------------------------------------------------------
# Thermal comfort set-points (deg C)
# ----------------------------------------------------------------------------
COOLING_SETPOINT = 22.0         # cooling target during occupied hours
HEATING_SETPOINT = 20.0         # heating target during occupied hours
DEADBAND = 1.0                  # default dead-band (+/- around set-point)
SETBACK_COOLING = 27.0          # unoccupied relaxed cooling target
SETBACK_HEATING = 15.0          # unoccupied relaxed heating target

# ----------------------------------------------------------------------------
# Simulation period
# ----------------------------------------------------------------------------
SIM_YEARS = 3                   # years of synthetic hourly data to generate
RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# Climate (gentle maritime / temperate profile, deg C monthly means)
# ----------------------------------------------------------------------------
MONTHLY_MEAN_TEMP = [5, 5.5, 7.5, 9.5, 12.5, 15.5,
                     17.5, 17, 14.5, 11, 7.5, 5.5]
DAILY_TEMP_SWING = 6.0          # peak-to-trough diurnal swing
