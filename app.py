"""
app.py — FastAPI backend for Cover Type Explorer
Serves the trained model, map generation, and static frontend files.
"""

import os
import random
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from xgboost import XGBClassifier

from map_generator import generate_map

# ── Load model at startup ───────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.json")
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠ No model found at {MODEL_PATH} — predictions will be random.")
        print(f"  Run 'python train_model.py' first to train the model.")

load_model()

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Cover Type Explorer", version="1.0.0")


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/api/generate_map")
async def api_generate_map(seed: int = None):
    """Generate a new procedural map."""
    if seed is None:
        seed = random.randint(0, 999999)
    result = generate_map(seed)
    return result


@app.post("/api/predict")
async def api_predict(req: PredictRequest):
    """Predict cover type for a single feature vector."""
    if model is None:
        return {"error": "Model not loaded. Run train_model.py first."}

    X = np.array([req.features], dtype=np.float32)
    pred = int(model.predict(X)[0])
    probas = model.predict_proba(X)[0]

    from map_generator import CLASS_NAMES
    return {
        "cover_type": pred,
        "cover_name": CLASS_NAMES[pred],
        "probabilities": {CLASS_NAMES[j]: round(float(probas[j]), 4) for j in range(7)},
    }


class BatchPredictRequest(BaseModel):
    vectors: list[list[float]]


@app.post("/api/batch_predict")
async def api_batch_predict(req: BatchPredictRequest):
    """Batch predict cover types for multiple feature vectors (lazy legend highlighting)."""
    if model is None:
        return {"error": "Model not loaded."}

    X = np.array(req.vectors, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    preds = model.predict(X)
    probas = model.predict_proba(X)

    from map_generator import CLASS_NAMES
    predictions = []
    for i in range(len(preds)):
        cls = int(preds[i])
        predictions.append({
            "cover_type": cls,
            "cover_name": CLASS_NAMES[cls],
            "probabilities": {CLASS_NAMES[j]: round(float(probas[i][j]), 4) for j in range(7)},
        })
    return {"predictions": predictions}


@app.get("/api/metrics")
async def api_metrics():
    """Return model performance metrics."""
    metrics_file = os.path.join(os.path.dirname(__file__), "metrics.json")
    if os.path.exists(metrics_file):
        import json
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {"error": "Metrics not found"}


# ── Serve frontend ──────────────────────────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))
