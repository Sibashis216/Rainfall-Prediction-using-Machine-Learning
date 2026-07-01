"""
app.py — Rainfall Prediction backend (Flask)
────────────────────────────────────────────────────────
Serves:
  GET  /                    → frontend (instrument-panel UI)
  GET  /api/meta            → model metrics + feature importances
  POST /api/predict         → { rainfall_mm, confidence_band, category }

Model artifacts are loaded once at startup with joblib from /model.
Run `python train_model.py` first (after `python generate_sample_data.py`
if you don't have your own fall.csv yet).
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# static_folder points at /public so `python app.py` still serves CSS/JS locally.
# On Vercel, /public/** is served directly by the CDN and this route is unused.
app = Flask(__name__, static_folder="public", static_url_path="")

FEATURE_NAMES = ["year", "tempavg", "DPavg", "humidity avg",
                  "SLPavg", "visibilityavg", "windavg"]

# Reasonable input bounds used for both frontend sliders and backend validation
FEATURE_BOUNDS = {
    "year":          {"min": 1990, "max": 2035, "step": 1,   "unit": ""},
    "tempavg":       {"min": -10,  "max": 45,   "step": 0.1, "unit": "°C"},
    "DPavg":         {"min": -15,  "max": 35,   "step": 0.1, "unit": "°C"},
    "humidity avg":  {"min": 0,    "max": 100,  "step": 0.1, "unit": "%"},
    "SLPavg":        {"min": 970,  "max": 1050, "step": 0.1, "unit": "hPa"},
    "visibilityavg": {"min": 0,    "max": 15,   "step": 0.1, "unit": "km"},
    "windavg":       {"min": 0,    "max": 60,   "step": 0.1, "unit": "km/h"},
}

_model = None
_scaler = None
_meta = None


def load_artifacts():
    """Load model/scaler/meta once. Raises a clear error if missing."""
    global _model, _scaler, _meta
    model_path = os.path.join(MODEL_DIR, "rainfall_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    meta_path = os.path.join(MODEL_DIR, "meta.joblib")

    for p in (model_path, scaler_path, meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing '{os.path.basename(p)}'. Run `python train_model.py` "
                f"(after `python generate_sample_data.py` if you don't have your "
                f"own fall.csv) before starting the server."
            )

    _model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    _meta = joblib.load(meta_path)


def categorize(mm: float) -> str:
    if mm <= 0.5:
        return "No Rain"
    if mm <= 5:
        return "Light Rain"
    if mm <= 20:
        return "Moderate Rain"
    if mm <= 50:
        return "Heavy Rain"
    return "Extreme Rain"


@app.route("/")
def index():
    return render_template("index.html", bounds=FEATURE_BOUNDS, features=FEATURE_NAMES)


@app.route("/api/meta")
def meta():
    if _meta is None:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify({
        "metrics": _meta["metrics"],
        "importances": _meta["importances"],
        "bounds": FEATURE_BOUNDS,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if _model is None or _scaler is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    payload = request.get_json(silent=True) or {}
    missing = [f for f in FEATURE_NAMES if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        row = [float(payload[f]) for f in FEATURE_NAMES]
    except (TypeError, ValueError):
        return jsonify({"error": "All feature values must be numeric."}), 400

    x = np.array([row])
    x_scaled = _scaler.transform(x)

    # Point estimate from the forest's average
    pred = float(_model.predict(x_scaled)[0])
    pred = max(pred, 0.0)

    # Spread across individual trees → simple uncertainty band for the UI
    tree_preds = np.array([t.predict(x_scaled)[0] for t in _model.estimators_])
    lo = float(max(np.percentile(tree_preds, 10), 0.0))
    hi = float(max(np.percentile(tree_preds, 90), 0.0))

    return jsonify({
        "rainfall_mm": round(pred, 2),
        "range_low": round(lo, 2),
        "range_high": round(hi, 2),
        "category": categorize(pred),
    })


try:
    load_artifacts()
except FileNotFoundError as e:
    print(f"⚠️  {e}")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
