from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve
)
from sklearn.base import clone
from joblib import dump
from collections import Counter

from .utils import set_seed
from .data import load_dataset
from .features import featurize
from .models import get_model_pipelines


def _get_probabilities(model, X):
    """Return class-1 probability or a scaled decision function in [0,1]."""
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        try:
            s = model.decision_function(X)
            smin, smax = s.min(), s.max()
            return (s - smin) / (smax - smin + 1e-9)
        except Exception:
            return None


def evaluate_holdout(model, X_test, y_test, outdir: Path, plots: bool):
    y_prob = _get_probabilities(model, X_test)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None,
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()

    if plots:
        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title('Confusion Matrix (Holdout)')
        plt.colorbar()
        tick_marks = np.arange(2)
        classes = ['Legit (0)', 'Phish (1)']
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2. if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        fig.tight_layout()
        fig.savefig(outdir / "confusion_matrix.png", dpi=160)
        plt.close(fig)

    # Classification report text (als Dict)
    metrics["classification_report"] = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    return metrics


def tune_threshold_cv(model, X_train: pd.DataFrame, y_train: np.ndarray, folds: int = 5):
    """
    Ermittelt per Stratified K-Fold auf dem Training einen Threshold, der den F1-Score maximiert.
    Gibt (best_threshold, best_f1_cv, oof_probs) zurück.
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y_train), dtype=float)

    for tr_idx, val_idx in skf.split(X_train, y_train):
        m = clone(model)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        p = _get_probabilities(m, X_train.iloc[val_idx])
        if p is None:
            # falls weder proba noch decision_function existiert – breche ab
            raise RuntimeError("Model does not provide probabilities or decision scores.")
        oof_probs[val_idx] = p

    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_train, oof_probs >= t) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    best_t = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])

    return best_t, best_f1, oof_probs


def main():
    parser = argparse.ArgumentParser(description="Train phishing URL detector")
    parser.add_argument("--data", required=True, help="Path to CSV with columns url,label")
    parser.add_argument("--outdir", default="artifacts", help="Output directory for model & metrics")
    parser.add_argument("--plots", action="store_true", help="Save matplotlib plots")
    parser.add_argument("--test_size", type=float, default=0.2, help="Holdout test size")
    parser.add_argument("--cv_folds", type=int, default=5, help="Stratified K-Folds")
    parser.add_argument("--auto_threshold", action="store_true",
                        help="Finde optimalen Probability-Threshold via CV und speichere ihn in artifacts/threshold.json")
    args = parser.parse_args()

    set_seed(42)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    X, feature_names = featurize(df)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # CV-Folds robust an kleinste Klasse im Training anpassen
    min_class_n = int(min(Counter(y_train).values()))
    folds = int(min(args.cv_folds, max(2, min_class_n)))  # min. 2 Folds
    print(f"Using {folds} CV folds (limited by smallest class size {min_class_n}).")

    models = get_model_pipelines()

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    cv_results = {}
    best_model_name = None
    best_f1 = -1.0
    best_fitted = None

    for name, pipe in models.items():
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring,
            n_jobs=-1, return_train_score=False
        )
        mean_scores = {k: float(np.nanmean(v)) for k, v in scores.items() if k.startswith("test_")}
        cv_results[name] = mean_scores

        # Fit auf gesamten Trainingssplit
        pipe.fit(X_train, y_train)
        holdout_metrics = evaluate_holdout(pipe, X_test, y_test, outdir, plots=args.plots)
        cv_results[name]["holdout"] = holdout_metrics

        # Auswahl per CV-F1
        f1_mean = mean_scores.get("test_f1", -1.0)
        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_model_name = name
            best_fitted = pipe

    # Artefakte speichern
    metrics_payload = {
        "cv_results": cv_results,
        "best_model": best_model_name,
        "feature_count": len(feature_names)
    }

    if best_fitted is not None:
        dump(best_fitted, outdir / "best_model.joblib")
        with open(outdir / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(feature_names, f, ensure_ascii=False, indent=2)

        # Optional: Threshold-Tuning + PR-Kurve
        if args.auto_threshold:
            # X_train als DataFrame sicherstellen (falls es ein NumPy-Array wäre)
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train, columns=feature_names)

            best_t, best_f1_cv, oof_probs = tune_threshold_cv(best_fitted, X_train.reset_index(drop=True), y_train, folds=folds)
            with open(outdir / "threshold.json", "w", encoding="utf-8") as f:
                json.dump({"threshold": best_t, "f1_cv": best_f1_cv}, f, ensure_ascii=False, indent=2)

            # PR-Kurve speichern
            precision, recall, thr = precision_recall_curve(y_train, oof_probs)
            fig = plt.figure()
            plt.plot(recall, precision)
            plt.title(f'Precision-Recall (CV); best thr={best_t:.2f}, F1={best_f1_cv:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            fig.tight_layout()
            fig.savefig(outdir / "precision_recall_cv.png", dpi=160)
            plt.close(fig)

            # Optional: Holdout-Metriken am getunten Threshold
            y_prob_holdout = _get_probabilities(best_fitted, X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feature_names))
            if y_prob_holdout is not None:
                y_pred_thr = (y_prob_holdout >= best_t).astype(int)
                holdout_thr = {
                    "threshold": best_t,
                    "accuracy": float(accuracy_score(y_test, y_pred_thr)),
                    "precision": float(precision_score(y_test, y_pred_thr, zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred_thr, zero_division=0)),
                    "f1": float(f1_score(y_test, y_pred_thr, zero_division=0)),
                    "roc_auc": float(roc_auc_score(y_test, y_prob_holdout))
                }
                metrics_payload["holdout_thresholded"] = holdout_thr

    #  Holdout-Predictions (für spätere Vergleiche/ROC) 
    if best_fitted is not None:
        # sicherstellen, dass es Spaltennamen gibt
        
        X_test_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feature_names)
        y_prob_holdout = _get_probabilities(best_fitted, X_test_df)
        y_pred_holdout = best_fitted.predict(X_test_df)
        # CSV schreiben: y_true, y_prob, y_pred

        np.savetxt(outdir / "holdout_predictions.csv",
                np.c_[y_test, (y_prob_holdout if y_prob_holdout is not None else np.full_like(y_test, np.nan)), y_pred_holdout],
                delimiter=",", header="y_true,y_prob,y_pred", comments="", fmt="%.10f")
    # --- ENDE NEU ---
        

    # Metriken speichern
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print(f"Done. Best model: {best_model_name}")
    print(f"Artifacts saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
