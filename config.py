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
FLOOR_AREA_M2 = 10_000              # gross internal floor area (m²)
OCCUPANCY_DENSITY_M2 = 20           # m² per person (CIBSE Guide A office default)
PEAK_OCCUPANCY = FLOOR_AREA_M2 // OCCUPANCY_DENSITY_M2   # 500 for default building
OPEN_HOUR = 8                       # nominal building open time (local)
CLOSE_HOUR = 18                     # nominal building close time (local)
WORKDAYS = {0, 1, 2, 3, 4}         # Monday=0 ... Sunday=6

# Default building type (must match a key in building_profiles.PROFILES)
DEFAULT_BUILDING_TYPE = "office"

# ----------------------------------------------------------------------------
# Energy baseline (used to calibrate the synthetic generator)
# ----------------------------------------------------------------------------
HVAC_ENERGY_INTENSITY_KWH_M2 = 100      # kWh/m²/yr HVAC (CIBSE TM46 typical office)
TARGET_ANNUAL_HVAC_KWH = HVAC_ENERGY_INTENSITY_KWH_M2 * FLOOR_AREA_M2  # 1,000,000
ANNUAL_ELECTRICITY_KWH = 2_000_000      # whole-building electricity (kept for reference)
HVAC_SHARE_OF_LOAD = 0.50               # HVAC fraction of total electricity

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
# Equipment degradation and maintenance (Phase 1 additions)
# ----------------------------------------------------------------------------
DEGRADATION_RATE = 0.015        # annual efficiency decline (1.5 %/yr) applied in NPV
MAINTENANCE_COSTS = {           # annual maintenance cost per strategy (GBP/yr)
    "occupancy_scheduling": 2_000,
    "smart_thermostats":    1_000,
    "bas":                  8_000,
    "combined":             10_000,
}

# ----------------------------------------------------------------------------
# Simulation period
# ----------------------------------------------------------------------------
SIM_YEARS = 3                   # years of synthetic hourly data to generate
RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# Phase 2 — energy prices and carbon
# ----------------------------------------------------------------------------
GAS_PRICE        = 0.07         # GBP/kWh natural gas (commercial tariff)
GAS_CARBON       = 0.203        # kgCO2e/kWh natural gas
INFLATION_RATE   = 0.035        # annual energy price inflation for NPV
GRANT_PCT        = 0.0          # default grant / subsidy (% of CAPEX)

# ----------------------------------------------------------------------------
# Phase 2 — LED lighting defaults
# ----------------------------------------------------------------------------
LIGHTING_HOURS_PER_YEAR = 2_250     # operating hours/year (typical office)
LIGHTING_LIFETIME_YEARS = 20        # LED lamp life before replacement

# ----------------------------------------------------------------------------
# Phase 2 — solar PV defaults
# ----------------------------------------------------------------------------
SOLAR_PV_PERF_RATIO  = 0.80         # system performance ratio (losses)
SOLAR_PV_EXPORT_TARIFF = 0.15       # GBP/kWh Smart Export Guarantee rate
SOLAR_PV_LIFETIME    = 25           # years (panel warranty period)
SOLAR_PV_DEGRADATION = 0.005        # 0.5 %/yr panel output degradation

# UK self-consumption fraction by building type (solar peaks during occupancy)
SOLAR_SELF_CONSUMPTION = {
    "office": 0.65, "hotel": 0.30, "hospital": 0.70,
    "school": 0.55, "retail": 0.60, "industrial": 0.65,
}

# ----------------------------------------------------------------------------
# Phase 2 — solar thermal defaults
# ----------------------------------------------------------------------------
SOLAR_THERMAL_EFFICIENCY = 0.50     # net collector efficiency
SOLAR_THERMAL_LIFETIME   = 20       # years
DHW_INTENSITY_KWH_M2 = {           # annual domestic hot water demand (kWh/m²/yr)
    "office": 5, "hotel": 30, "hospital": 40,
    "school": 8, "retail": 2,  "industrial": 5,
}

# ----------------------------------------------------------------------------
# Phase 2 — fabric defaults
# ----------------------------------------------------------------------------
HDD_ANNUAL     = 2_400      # annual heating degree-days (UK average, base 15.5 °C)
HEATING_EFF    = 0.90       # boiler/heat-pump system efficiency
FABRIC_LIFETIME = 30        # years

# EPC non-domestic asset rating bands (lower score = better)
EPC_BANDS = [
    ("A+", 0,  25),
    ("A",  26, 50),
    ("B",  51, 75),
    ("C",  76, 100),
    ("D",  101, 125),
    ("E",  126, 150),
    ("F",  151, 175),
    ("G",  176, 999),
]

# ----------------------------------------------------------------------------
# Climate (gentle maritime / temperate profile, deg C monthly means)
# ----------------------------------------------------------------------------
MONTHLY_MEAN_TEMP = [5, 5.5, 7.5, 9.5, 12.5, 15.5,
                     17.5, 17, 14.5, 11, 7.5, 5.5]
DAILY_TEMP_SWING = 6.0          # peak-to-trough diurnal swing

# ----------------------------------------------------------------------------
# Phase 3 — wind turbine defaults
# ----------------------------------------------------------------------------
WIND_LIFETIME_YEARS = 20            # turbine design life
WIND_DEGRADATION    = 0.005         # 0.5 %/yr output degradation
WIND_MIN_VIABLE_MS  = 5.0           # mean annual wind speed below which siting is flagged

# ----------------------------------------------------------------------------
# Phase 3 — solar gain / overheating defaults
# ----------------------------------------------------------------------------
TM52_ADAPTIVE_LIMIT_K = 28.0        # simplified CIBSE TM52 upper comfort temperature (deg C)
TM52_EXCEEDANCE_HOURS_THRESHOLD = 50  # occupied hours above limit before overheating is flagged

# ----------------------------------------------------------------------------
# Phase 3 — 3D visualisation defaults
# ----------------------------------------------------------------------------
DEFAULT_BUILDING_HEIGHT_M = 12.0    # assumed building height for the block model

# ----------------------------------------------------------------------------
# Limitation 3 fix — thermal mass / construction type correction factors
# hvac_factor: multiplier applied to the surrogate-model HVAC baseline and savings.
# Heavy-mass buildings have lower HVAC demand than the lightweight default assumed
# by the synthetic data generator (CIBSE AM11 / CIBSE KS6 informed estimates).
# ----------------------------------------------------------------------------
THERMAL_MASS: dict = {
    "lightweight": {
        "label":       "Lightweight (steel / timber / curtain-wall glazing)",
        "hvac_factor": 1.00,
        "description": "Fast thermal response. No diurnal buffering. "
                       "Cooling peaks sharply in the afternoon.",
    },
    "medium": {
        "label":       "Medium mass (brick / block / mixed construction)",
        "hvac_factor": 0.94,
        "description": "Moderate buffering. Typical UK brick-clad commercial building. "
                       "~6% lower HVAC consumption than lightweight equivalent.",
    },
    "heavy": {
        "label":       "Heavy mass (in-situ concrete frame / stone / pre-1980 stock)",
        "hvac_factor": 0.87,
        "description": "Strong diurnal buffering. ~13% lower HVAC consumption than "
                       "lightweight. Night pre-cooling strategies particularly effective.",
    },
}

# Default heating fuel (used for SBEM-like EPC scoring)
DEFAULT_HEATING_FUEL = "gas"
