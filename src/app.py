# src/app.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel
from joblib import load

# relative Import aus demselben Paket "src"
from .features import featurize

# Pfade zu Artefakten (per ENV überschreibbar)
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH     = Path(os.getenv("PHISH_MODEL_PATH"    , ROOT/"artifacts/run_meine_full/best_model.joblib"))
FEATURES_PATH  = Path(os.getenv("PHISH_FEATURES_PATH" , ROOT/"artifacts/run_meine_full/feature_names.json"))
THRESHOLD_PATH = Path(os.getenv("PHISH_THRESHOLD_PATH", ROOT/"artifacts/run_meine_full/threshold.json"))

# >>> WICHTIG: die ASGI-App heisst genau 'app'
app = FastAPI(title="Phishing URL Detector", version="1.0")

def _vectorize_urls(urls: List[str]) -> np.ndarray:
    df = pd.DataFrame({"url": urls})
    X_new, feat_names = featurize(df)
    Xdf = pd.DataFrame(X_new, columns=feat_names)
    train_names = json.load(open(FEATURES_PATH, "r"))
    # fehlende Spalten ergänzen und Reihenfolge wie im Training
    for f in train_names:
        if f not in Xdf.columns:
            Xdf[f] = 0
    Xdf = Xdf[train_names]
    return Xdf.to_numpy()

def _proba(model, X: np.ndarray) -> Optional[np.ndarray]:
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        try:
            s = model.decision_function(X)
            return (s - s.min()) / (s.max() - s.min() + 1e-9)
        except Exception:
            return None

# Modell + Threshold laden (einmalig beim Start)
model = load(MODEL_PATH)
try:
    threshold = float(json.load(open(THRESHOLD_PATH, "r"))["threshold"])
except Exception:
    threshold = 0.5

class BatchRequest(BaseModel):
    urls: List[str]

def _score_urls(urls: List[str]):
    X = _vectorize_urls(urls)
    prob = _proba(model, X)
    if prob is None:
        pred = model.predict(X).astype(int).tolist()
        return [{"url": u, "prediction": int(p), "phish_score": None, "threshold": threshold}
                for u, p in zip(urls, pred)]
    pred = (prob >= threshold).astype(int)
    return [{"url": u, "prediction": int(p), "phish_score": float(s), "threshold": threshold}
            for u, p, s in zip(urls, pred, prob)]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "features": str(FEATURES_PATH),
        "threshold": threshold
    }

@app.get("/score")
def score(url: str = Query(..., description="Zu prüfende URL")):
    return _score_urls([url])[0]

@app.post("/score")
def score_batch(req: BatchRequest):
    return _score_urls(req.urls)
