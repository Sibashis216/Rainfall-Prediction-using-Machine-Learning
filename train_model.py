"""
train_model.py
────────────────────────────────────────────────────────
Trains the Random Forest Regressor on fall.csv and saves
the fitted model + scaler to /model using joblib (NOT
pickle) so the artifacts stay compatible across sklearn
versions when loaded by the Flask backend.

Usage:
    python train_model.py
Requires:
    fall.csv in the same folder as this script
    (columns: year, month, day, tempavg, DPavg, humidity avg,
     SLPavg, visibilityavg, windavg, Rainfall)
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "fall.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_NAMES = ["year", "tempavg", "DPavg", "humidity avg",
                  "SLPavg", "visibilityavg", "windavg"]


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Place your fall.csv (2362 records) "
            f"next to train_model.py before running this script."
        )
    ds = pd.read_csv(path)
    ds = ds.drop(["month", "day"], axis=1, errors="ignore")
    return ds


def train():
    print("Loading dataset...")
    ds = load_dataset(DATA_PATH)
    print(f"Dataset shape: {ds.shape}")

    x = ds.iloc[:, :7].values
    y = ds.iloc[:, 7].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=0
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    print("Training Random Forest Regressor (500 trees)...")
    regressor = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        random_state=0,
        n_jobs=-1,
    )
    regressor.fit(x_train_scaled, y_train)

    ypred_train = regressor.predict(x_train_scaled)
    ypred_test = regressor.predict(x_test_scaled)

    metrics = {
        "train_r2": round(float(r2_score(y_train, ypred_train)), 4),
        "test_r2": round(float(r2_score(y_test, ypred_test)), 4),
        "test_mae": round(float(mean_absolute_error(y_test, ypred_test)), 4),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_estimators": 500,
    }

    importances = dict(zip(FEATURE_NAMES,
                            [round(float(v), 4) for v in regressor.feature_importances_]))

    print("\nMetrics:", metrics)
    print("Feature importances:", importances)

    joblib.dump(regressor, os.path.join(MODEL_DIR, "rainfall_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(
        {"metrics": metrics, "importances": importances, "features": FEATURE_NAMES},
        os.path.join(MODEL_DIR, "meta.joblib"),
    )
    print(f"\n✅ Saved model, scaler, and metadata to {MODEL_DIR}")


if __name__ == "__main__":
    train()
