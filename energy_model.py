"""
energy_model.py
===============
Trains the machine-learning model at the heart of the study.

A Gradient Boosting Regressor learns HVAC electricity demand as a function of
the building's operating conditions (weather, occupancy) AND its control levers
(set-points, dead-band, fan factor, on/off state). Because the control levers are
input features, the trained model becomes a fast surrogate that the optimiser can
query thousands of times to evaluate "what-if" control strategies without re-running
a physics simulation.

The module reports standard regression metrics (R2, MAE, RMSE) on a held-out test
set and the model's feature importances, both of which are reported in the project.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

FEATURES = [
    "hour", "dayofweek", "month", "is_workday",
    "t_out", "solar", "occ_frac",
    "system_on", "cool_set", "heat_set", "deadband", "fan_factor", "vent_frac",
]
TARGET = "hvac_kwh"


def train(df: pd.DataFrame, seed: int = 42):
    """Train the surrogate energy model; return (model, metrics, importances)."""
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, shuffle=True
    )

    model = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "mean_target": float(y.mean()),
    }
    importances = (
        pd.Series(model.feature_importances_, index=FEATURES)
        .sort_values(ascending=False)
    )
    return model, metrics, importances


def predict_annual_kwh(model, scenario_df: pd.DataFrame, sim_years: int) -> float:
    """Use the trained surrogate to predict an annual HVAC total (kWh/yr)."""
    pred = model.predict(scenario_df[FEATURES].to_numpy())
    return float(np.clip(pred, 0, None).sum() / sim_years)


if __name__ == "__main__":
    import data_generator as dg
    data = dg.generate()
    train_df = dg.make_training_data(data)
    mdl, m, imp = train(train_df)
    print("Model performance on held-out test set")
    print(f"  R2   : {m['r2']:.4f}")
    print(f"  MAE  : {m['mae']:.2f} kWh/h")
    print(f"  RMSE : {m['rmse']:.2f} kWh/h")
    print(f"  mean target: {m['mean_target']:.2f} kWh/h")
    print("\nFeature importances")
    for k, v in imp.items():
        print(f"  {k:12s} {v:.3f}")
