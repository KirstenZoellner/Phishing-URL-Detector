from __future__ import annotations
from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# XGBoost: separate Bibliothek mit sklearn-kompatibler API
try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        "xgboost ist nicht installiert. Bitte mit `pip install xgboost` nachinstallieren."
    ) from e


def get_model_pipelines() -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    # Lineares Baseline-Modell (mit Skalierung, sparse-sicher)
    models["logreg"] = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=None,
            random_state=42,
        )),
    ])

    # Random Forest (keine Skalierung nötig)
    models["rf"] = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        )),
    ])

    # XGBoost (sklearn-kompatibel, kein Scaler nötig)
    models["xgboost"] = Pipeline([
        ("clf", XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,        # L2
            reg_alpha=0.0,         # L1
            n_jobs=-1,
            random_state=42,
            eval_metric="logloss", # wichtig für neuere xgboost-Versionen
            tree_method="auto",    # lässt XGB selbst wählen (hist/approx)
        )),
    ])

    return models
