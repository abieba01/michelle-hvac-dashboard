# HVAC Optimisation AI Model

A self-contained Python project that builds a machine-learning model to quantify
the energy, cost and carbon benefits of three HVAC optimisation strategies in a
representative commercial office building. It is the computational engine behind
the dissertation *Evaluating HVAC Optimisation Strategies in Commercial Office
Buildings*.

## What it does

1. **Synthesises** a realistic three-year hourly operating dataset for the
   case-study building (weather, occupancy, control settings, HVAC energy),
   automatically calibrated so the baseline matches ~1,000,000 kWh/yr.
2. **Trains** a Gradient Boosting surrogate model that learns HVAC electricity
   demand as a function of operating conditions *and* the controllable levers
   (schedule, set-points, dead-band, fan speed, ventilation rate).
3. **Optimises** by rewriting the control levers for each strategy and using the
   surrogate to predict the resulting annual energy.
4. **Evaluates** energy, cost (payback, ROI, 15-year NPV) and carbon outcomes,
   and exports a results table plus four charts.

## Files

| File | Purpose |
|------|---------|
| `config.py` | All building, economic and climate assumptions (edit here to re-calibrate). |
| `data_generator.py` | Physics-informed synthetic dataset + augmented training set. |
| `energy_model.py` | Trains and evaluates the Gradient Boosting surrogate. |
| `hvac_optimizer.py` | Defines the five control scenarios and the financial maths. |
| `main.py` | Runs the whole pipeline and writes `outputs/`. |

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Outputs are written to `./outputs/`:
`scenario_results.csv`, `model_metrics.json`, and four PNG charts.

## Headline results (default assumptions)

| Strategy | Energy saving | Annual £ saving | Payback | CO2 saving |
|----------|--------------:|----------------:|--------:|-----------:|
| Occupancy-based scheduling | 22.7% | £63,500 | 0.7 yr | 47 tCO2e |
| Smart thermostats | 9.1% | £25,400 | 1.2 yr | 19 tCO2e |
| Building Automation System | 28.5% | £80,000 | 1.9 yr | 59 tCO2e |
| Combined optimisation | 36.0% | £101,000 | 2.0 yr | 75 tCO2e |

## Extending the model

* Swap the synthetic generator for real BMS exports by producing a DataFrame with
  the same columns (`FEATURES` in `energy_model.py`) and skipping `data_generator`.
* Re-tariff or re-grid the study by editing `ELECTRICITY_PRICE` and `CARBON_FACTOR`
  in `config.py`.
* Add new strategies by writing a builder function in `hvac_optimizer.py` and
  registering it in the `SCENARIOS` dictionary.

> The synthetic dataset is a transparent stand-in for metered data so the project
> is fully reproducible without a proprietary BMS feed. All energy figures should
> be revalidated against measured data before use in a real investment decision.
