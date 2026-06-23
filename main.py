"""
main.py
=======
End-to-end pipeline for the HVAC optimisation study.

Run this single file to reproduce every quantitative result in the report:

    python main.py

It will (1) generate the synthetic dataset, (2) train and evaluate the surrogate
energy model, (3) run all five control scenarios through the optimiser, and
(4) write a results table, a metrics summary and four publication-quality charts
into ./outputs/.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import data_generator as dg
from energy_model import train, FEATURES
from hvac_optimizer import evaluate, SCENARIOS, PRETTY

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

PALETTE = ["#1f4e79", "#2e75b6", "#5b9bd5", "#9dc3e6", "#c55a11", "#548235"]
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})


def main():
    print("=" * 64)
    print("HVAC OPTIMISATION AI MODEL  -  full pipeline")
    print("=" * 64)

    # 1. Data ---------------------------------------------------------------
    print("\n[1/4] Generating synthetic hourly dataset ...")
    data = dg.generate()
    baseline_annual = data["hvac_kwh"].sum() / C.SIM_YEARS
    print(f"      {len(data):,} hourly records over {C.SIM_YEARS} years")
    print(f"      baseline HVAC energy: {baseline_annual:,.0f} kWh/yr")

    # 2. Model --------------------------------------------------------------
    print("\n[2/4] Training surrogate energy model ...")
    train_df = dg.make_training_data(data)
    model, metrics, importances = train(train_df)
    print(f"      test R2 = {metrics['r2']:.4f} | "
          f"MAE = {metrics['mae']:.2f} kWh/h | RMSE = {metrics['rmse']:.2f} kWh/h")

    # 3. Scenarios ----------------------------------------------------------
    print("\n[3/4] Evaluating optimisation scenarios ...")
    results = evaluate(model, data)
    print(results[["scenario", "saving_pct", "cost_saving_gbp",
                   "carbon_saving_tco2e", "payback_years"]].round(2).to_string(index=False))

    # 4. Outputs ------------------------------------------------------------
    print("\n[4/4] Writing outputs to ./outputs ...")
    results.to_csv(OUT / "scenario_results.csv", index=False)
    with open(OUT / "model_metrics.json", "w") as f:
        json.dump({"metrics": metrics,
                   "feature_importance": importances.round(4).to_dict(),
                   "baseline_annual_kwh": baseline_annual}, f, indent=2)

    _chart_savings(results)
    _chart_payback(results)
    _chart_importance(importances)
    _chart_load_profile(data)
    print("      saved: scenario_results.csv, model_metrics.json, 4 charts")
    print("\nDone.")
    return results, metrics, importances


# ----------------------------------------------------------------------------
# Charts (each saved as PNG for embedding in the report)
# ----------------------------------------------------------------------------
def _chart_savings(results: pd.DataFrame):
    d = results[results.saving_pct > 0]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(d.scenario, d.saving_pct, color=PALETTE[1:1 + len(d)])
    ax.set_ylabel("HVAC energy saving (%)")
    ax.set_title("Annual HVAC energy saving by optimisation strategy")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([s.replace(" ", "\n") for s in d.scenario], fontsize=9)
    for b, v in zip(bars, d.saving_pct):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}%",
                ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "chart_savings.png", dpi=150); plt.close(fig)


def _chart_payback(results: pd.DataFrame):
    d = results[results.capex_gbp > 0]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.barh(d.scenario, d.payback_years, color=PALETTE[2])
    ax.set_xlabel("Simple payback period (years)")
    ax.set_title("Investment payback by strategy")
    ax.invert_yaxis()
    for b, v in zip(bars, d.payback_years):
        ax.text(v + 0.03, b.get_y() + b.get_height() / 2, f"{v:.2f} yr",
                va="center", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "chart_payback.png", dpi=150); plt.close(fig)


def _chart_importance(importances: pd.Series):
    d = importances[importances > 0.001].sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(d.index, d.values, color=PALETTE[0])
    ax.set_xlabel("Relative feature importance")
    ax.set_title("Drivers of HVAC energy in the surrogate model")
    fig.tight_layout(); fig.savefig(OUT / "chart_importance.png", dpi=150); plt.close(fig)


def _chart_load_profile(data: pd.DataFrame):
    """Mean 24-hour HVAC profile, weekday vs weekend."""
    wd = data[data.is_workday == 1].groupby("hour").hvac_kwh.mean()
    we = data[data.is_workday == 0].groupby("hour").hvac_kwh.mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(wd.index, wd.values, color=PALETTE[0], lw=2.2, label="Weekday")
    ax.plot(we.index, we.values, color=PALETTE[4], lw=2.2, label="Weekend")
    ax.fill_between(wd.index, wd.values, alpha=0.12, color=PALETTE[0])
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Mean HVAC demand (kWh)")
    ax.set_title("Average daily HVAC load profile (baseline)")
    ax.set_xticks(range(0, 24, 2)); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "chart_load_profile.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
