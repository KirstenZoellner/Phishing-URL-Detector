import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

def load_run(run_dir: Path):
    run_dir = Path(run_dir)
    preds = pd.read_csv(run_dir / "holdout_predictions.csv")
    preds.columns = preds.columns.str.lower()
    y_true = preds["y_true"].astype(int).to_numpy()
    y_prob = preds["y_prob"].to_numpy()
    y_pred = preds["y_pred"].astype(int).to_numpy()
    with open(run_dir / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    best_model = metrics.get("best_model")
    holdout = metrics.get("cv_results", {}).get(best_model, {}).get("holdout", {})
    return dict(dir=run_dir, y_true=y_true, y_prob=y_prob, y_pred=y_pred,
                metrics=metrics, holdout=holdout, best_model=best_model)

def plot_roc(a, b, out_png):
    import matplotlib.pyplot as plt
    plt.figure()
    for run in (a, b):
        label = run["dir"].name  # <- Ordnername als Label
        mask = ~np.isnan(run["y_prob"])
        fpr, tpr, _ = roc_curve(run["y_true"][mask], run["y_prob"][mask])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],"--", linewidth=1)
    ...
def plot_conf(a, b, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    for ax, run in zip(axes, (a, b)):
        label = run["dir"].name
        cm = confusion_matrix(run["y_true"], run["y_pred"], labels=[0,1])
        ConfusionMatrixDisplay(cm, display_labels=["Legit (0)","Phish (1)"]).plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(f"Confusion (Holdout) – {label}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_a", required=True)  # artifacts\run_sample
    p.add_argument("--run_b", required=True)  # artifacts\run_meine_bal
    p.add_argument("--outdir", default="reports/compare")
    args = p.parse_args()

    a = load_run(Path(args.run_a))
    b = load_run(Path(args.run_b))
    outdir = Path(args.outdir)

    plot_roc(a, b, outdir / "roc_compare.png")
    plot_conf(a, b, outdir / "confusion_compare.png")

    def brief(run, name):
        h = run.get("holdout") or {}
        keys = ["accuracy","precision","recall","f1","roc_auc"]
        summ = {k: round(h[k], 4) for k in keys if k in h}
        print(f"[{name}] best_model={run['best_model']} | {summ}")

    brief(a, "sample")
    brief(b, "meine_bal50k")
    print(f"Saved: {outdir.resolve()}")

if __name__ == "__main__":
    main()
