from __future__ import annotations
import argparse
import json
import pandas as pd
from joblib import load
from .features import featurize


def _get_probabilities(model, X):
    """Return class-1 probability or scaled decision scores [0,1], else None."""
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        try:
            s = model.decision_function(X)
            smin, smax = s.min(), s.max()
            return (s - smin) / (smax - smin + 1e-9)
        except Exception:
            return None


def _load_threshold(threshold_file: str | None, cli_threshold: float | None) -> float | None:
    """
    Priorität: threshold_file > cli_threshold > None
    """
    if threshold_file:
        try:
            with open(threshold_file, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return float(obj.get("threshold", 0.5))
        except Exception:
            # Wenn Datei fehlt/defekt ist: auf CLI-Threshold zurückfallen
            pass
    return cli_threshold


def predict_urls(model_path: str, urls: list[str], threshold: float | None = None) -> pd.DataFrame:
    model = load(model_path)
    df = pd.DataFrame({"url": urls})
    X, _ = featurize(df)

    proba = _get_probabilities(model, X)

    if threshold is not None and proba is not None:
        preds = (proba >= threshold).astype(int)
    else:
        # Fallback auf Model-Predict, wenn kein Threshold/keine Probas vorhanden
        preds = model.predict(X)

    out = pd.DataFrame({"url": urls, "prediction": preds})
    if proba is not None:
        out["phish_score"] = proba
    return out


def main():
    parser = argparse.ArgumentParser(description="Predict phishing for URLs")
    parser.add_argument("--model", required=True, help="Path to joblib model")
    parser.add_argument("--urls", nargs="*", help="One or more URLs")
    parser.add_argument("--input_csv", help="CSV file with column 'url'")
    parser.add_argument("--out_csv", help="Optional output CSV path")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Probability threshold for class 1 (phishing). If omitted, model.predict is used.")
    parser.add_argument("--threshold_file",
                        help="Path to JSON with {'threshold': float}. Overrides --threshold if present.")
    args = parser.parse_args()

    # URLs einsammeln
    urls: list[str] = []
    if args.urls:
        urls.extend(args.urls)
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        df.columns = df.columns.str.lower()
        if "url" not in df.columns:
            raise ValueError("input_csv must have a 'url' column")
        urls.extend(df["url"].astype(str).tolist())

    if not urls:
        raise SystemExit("Provide --urls or --input_csv")

    thr = _load_threshold(args.threshold_file, args.threshold)

    res = predict_urls(args.model, urls, threshold=thr)
    print(res.to_string(index=False))

    if args.out_csv:
        res.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
