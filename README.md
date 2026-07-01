# 🌧️ Rainfall Prediction Station

A full-stack web app around your notebook's Random Forest Regressor: a Flask
API serves predictions from the trained model, and an instrument-panel style
frontend lets you dial in atmospheric readings and see the predicted rainfall
on an animated gauge.

## Architecture

```
rainfall-prediction-app/
└── backend/
    ├── app.py                   # Flask app: serves UI + /api/predict, /api/meta
    ├── train_model.py           # Trains RandomForestRegressor, saves via joblib
    ├── generate_sample_data.py  # Optional: synthetic fall.csv for a quick demo
    ├── requirements.txt
    ├── fall.csv                 # Your dataset goes here (not included)
    ├── model/                   # rainfall_model.joblib, scaler.joblib, meta.joblib
    ├── templates/
    │   └── index.html
    └── static/
        ├── css/style.css
        └── js/script.js
```

**Why joblib instead of pickle:** your original student-performance project hit
version-mismatch errors loading `.pkl` files across sklearn versions. `joblib`
is what scikit-learn itself recommends for persisting estimators and is more
robust for large numpy-array-heavy objects like a 500-tree forest — same idea
we landed on for that earlier project.

## How prediction works

1. `train_model.py` reproduces your notebook's pipeline exactly: drop
   `month`/`day`, 80/20 train-test split (`random_state=0`), `StandardScaler`
   fit only on training data, `RandomForestRegressor(n_estimators=500,
   max_features='sqrt', random_state=0)`.
2. It saves three artifacts: the fitted model, the fitted scaler, and a
   `meta.joblib` with R² / MAE and feature importances (pre-computed so the
   frontend doesn't recompute stats on every request).
3. `app.py` loads those once at startup. `/api/predict` scales the incoming
   feature vector with the *same* fitted scaler before calling
   `model.predict()` — matching the notebook's leakage-free approach.
4. As a bonus not in the notebook, `/api/predict` also reports a 10th–90th
   percentile band across the forest's individual trees, giving the UI a
   sense of the model's uncertainty rather than just a single point value.

## Setup

```bash
cd rainfall-prediction-app/backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Option A — use your real dataset:
#   copy your fall.csv into this backend/ folder

# Option B — no dataset handy, just want to see it run:
python generate_sample_data.py   # writes a synthetic fall.csv (demo only)

python train_model.py            # trains + saves model/scaler/meta
python app.py                    # starts the server on http://localhost:5000
```

Open `http://localhost:5000` — set the seven sliders (year, temperature, dew
point, humidity, sea-level pressure, visibility, wind) and click **Run
Prediction**. The gauge animates to the predicted rainfall in mm, colored by
category (No Rain → Extreme Rain), with the diagnostics panel below showing
the model's R² / MAE and feature importances pulled straight from training.

## API reference

`GET /api/meta`
```json
{
  "metrics": {"train_r2": 0.97, "test_r2": 0.81, "test_mae": 3.42, "n_train": 1889, "n_test": 473, "n_estimators": 500},
  "importances": {"year": 0.02, "tempavg": 0.11, "...": "..."},
  "bounds": { "...slider min/max/step per feature..." }
}
```

`POST /api/predict`
```json
// request
{"year": 2020, "tempavg": 18, "DPavg": 16, "humidity avg": 65, "SLPavg": 1013, "visibilityavg": 6, "windavg": 8}

// response
{"rainfall_mm": 12.4, "range_low": 6.1, "range_high": 21.8, "category": "Moderate Rain"}
```

## For your interview prep

This gives you a live demo to pair with the README talking points you were
already polishing: the leakage-avoidance in the scaler fit, why `n_estimators`
and `max_features='sqrt'` were chosen, and now also a real answer to "how
would you deploy this" — a Flask inference layer decoupled from training,
versioned model artifacts, and a simple uncertainty signal on top of a point
estimate.
