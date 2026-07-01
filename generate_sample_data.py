"""
generate_sample_data.py
────────────────────────────────────────────────────────
Creates a SYNTHETIC fall.csv so the app can be trained and
demoed end-to-end even without your original 2362-record
dataset. Replace this file's output with your real fall.csv
before training for actual accuracy / interview presentation.

Usage:
    python generate_sample_data.py
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "fall.csv")

rng = np.random.default_rng(42)
N = 2362

year = rng.integers(2000, 2021, N)
month = rng.integers(1, 13, N)
day = rng.integers(1, 29, N)

tempavg = rng.normal(22, 6, N).clip(-5, 42)
DPavg = tempavg - rng.normal(6, 3, N)
humidity = (60 + (DPavg - tempavg) * -1.5 + rng.normal(0, 8, N)).clip(20, 100)
SLPavg = rng.normal(1013, 6, N)
visibility = (10 - humidity / 20 + rng.normal(0, 1.5, N)).clip(0.5, 12)
windavg = rng.normal(9, 4, N).clip(0, 35)

# Rainfall driven mostly by humidity + inverse pressure + inverse visibility,
# with noise, floored at 0 to mimic real rainfall distributions (many dry days).
rain_signal = (
    (humidity - 50) * 0.6
    + (1013 - SLPavg) * 1.1
    + (8 - visibility) * 2.0
    + rng.normal(0, 8, N)
)
rainfall = np.clip(rain_signal, 0, None) ** 1.15
rainfall = np.round(rainfall, 1)

df = pd.DataFrame({
    "year": year,
    "month": month,
    "day": day,
    "tempavg": np.round(tempavg, 1),
    "DPavg": np.round(DPavg, 1),
    "humidity avg": np.round(humidity, 1),
    "SLPavg": np.round(SLPavg, 1),
    "visibilityavg": np.round(visibility, 1),
    "windavg": np.round(windavg, 1),
    "Rainfall": rainfall,
})

df.to_csv(OUT_PATH, index=False)
print(f"✅ Synthetic fall.csv written to {OUT_PATH} ({len(df)} rows)")
print("⚠️  This is DEMO data only — swap in your real fall.csv for the actual project.")
