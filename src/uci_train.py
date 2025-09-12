from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def _get_prob(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        try:
            s = model.decision_function(X)
            return (s - s.min()) / (s.max() - s.min() + 1e-9)
        except Exception:
            return None

def evaluate_holdout(model, X_test, y_test, outdir: Path, plots: bool):
    y_prob = _get_prob(model, X_test)
    y_pred = model.predict(X_test)
    mets = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0,1]).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }
    if plots:
        cm = np.array(mets["confusion_matrix"])
        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest'); plt.title('Confusion Matrix (Holdout)'); plt.colorbar()
        ticks = np.arange(2); classes = ['Legit (0)','Phish (1)']
        plt.xticks(ticks, classes, rotation=45); plt.yticks(ticks, classes)
        thr = cm.max()/2 if cm.max()>0 else .5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, int(cm[i, j]), ha="center", color="white" if cm[i, j] > thr else "black")
        plt.ylabel('True'); plt.xlabel('Pred'); fig.tight_layout()
        fig.savefig(outdir / "confusion_matrix.png", dpi=160); plt.close(fig)
    return mets, y_prob, y_pred

def main():
    ap = argparse.ArgumentParser(description="Train on UCI phishing features (tabular)")
    ap.add_argument("--data", required=True, help="CSV mit UCI-Features + label (0/1)")
    ap.add_argument("--outdir", default="artifacts/uci_full", help="Output-Ordner")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--cv_folds", type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    assert "label" in df.columns, "Erwarte Spalte 'label'."

    X = df.drop(columns=["label"])
    y = df["label"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        "rf": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, n_jobs=-1, class_weight="balanced_subsample", random_state=42
            ))
        ]),
    }

    scoring = {"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","roc_auc":"roc_auc"}
    best_name, best_f1, best_fitted, best_prob, best_pred = None, -1.0, None, None, None
    cv_results = {}

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    for name, pipe in models.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        mean_scores = {k: float(np.nanmean(v)) for k, v in scores.items() if k.startswith("test_")}
        cv_results[name] = mean_scores

        pipe.fit(X_train, y_train)
        hold_mets, y_prob, y_pred = evaluate_holdout(pipe, X_test, y_test, outdir, plots=args.plots)
        cv_results[name]["holdout"] = hold_mets

        if mean_scores.get("test_f1", -1.0) > best_f1:
            best_f1, best_name, best_fitted = mean_scores["test_f1"], name, pipe
            best_prob, best_pred = y_prob, y_pred

    if best_fitted is not None:
        dump(best_fitted, outdir / "best_model.joblib")
        # Holdout-Predictions (für ROC/Compare)
        y_true = y_test
        np.savetxt(outdir / "holdout_predictions.csv",
                   np.c_[y_true,
                         (best_prob if best_prob is not None else np.full_like(y_true, np.nan)),
                         best_pred],
                   delimiter=",", header="y_true,y_prob,y_pred", comments="", fmt="%.10f")

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"cv_results": cv_results, "best_model": best_name, "feature_count": X.shape[1]},
                  f, ensure_ascii=False, indent=2)

    if args.plots and best_prob is not None:
        prec, rec, _ = precision_recall_curve(y_test, best_prob)
        fig = plt.figure(); plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Precision-Recall (Holdout) – {best_name}")
        fig.tight_layout(); fig.savefig(outdir / "precision_recall_holdout.png", dpi=160); plt.close(fig)

    print(f"Done. Best model: {best_name}")
    print(f"Artifacts: {outdir.resolve()}")

if __name__ == "__main__":
    main()
