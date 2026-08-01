"""
Fits the FINAL strike-probability model for production use (paper trading /
the dashboard), on ALL available historical data -- separate from
walk_forward.py, which exists purely to VALIDATE the approach on held-out
folds. Now that validation is done and the approach cleared its baseline,
this refits on the full dataset for the best real-world fit.

Deliberately uses ONLY the "distance+time+vol" feature set (model C from the
walk-forward report), not the 7-indicator confluence features -- those were
tested independently and shown to add no significant improvement (p=0.054,
losing to the simpler model in 2 of 6 folds). Shipping the simpler model is
not a compromise; it's what the validation actually supports.

Run after research/strike_probability/results/features.csv exists:
    python fit_final_model.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[3]  # kalshi-15min-bot/
MODEL_DIR = ROOT / "model"

FEATURES = ["distance_pct", "minutes_remaining", "realized_vol",
            "dist_over_reachable", "current_side_leading"]


def main():
    features_path = Path(__file__).resolve().parent.parent / "results" / "features.csv"
    df = pd.read_csv(features_path, parse_dates=["close_time"])
    df = df.dropna(subset=FEATURES + ["y"])

    clf = LogisticRegression(max_iter=2000)
    clf.fit(df[FEATURES], df["y"])

    # In-sample fit quality (NOT a validation number -- see the walk-forward
    # report in this same folder for the honest out-of-fold numbers this
    # model is actually justified by).
    p_in_sample = clf.predict_proba(df[FEATURES])[:, 1]
    brier_in_sample = brier_score_loss(df["y"], p_in_sample)
    ll_in_sample = log_loss(df["y"], p_in_sample)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "strike_prob_model.pkl")

    meta = {
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "n_training_rows": int(len(df)),
        "n_training_markets": int(df["ticker"].nunique()),
        "training_date_range": [str(df["close_time"].min()), str(df["close_time"].max())],
        "in_sample_brier": float(brier_in_sample),
        "in_sample_log_loss": float(ll_in_sample),
        "coefficients": dict(zip(FEATURES, clf.coef_[0].tolist())),
        "intercept": float(clf.intercept_[0]),
        "validation_note": (
            "In-sample metrics above are NOT the validation result -- they will look "
            "better than reality because the model is scored on data it was fit on. "
            "The real, honest, out-of-fold numbers are in "
            "research/strike_probability/README.md (walk-forward Brier ~0.150 vs. "
            "naive baseline ~0.160, market-level bootstrap p<0.0001)."
        ),
    }
    (MODEL_DIR / "strike_prob_model_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Fitted on {len(df)} rows from {df['ticker'].nunique()} markets")
    print(f"In-sample Brier={brier_in_sample:.4f}  LogLoss={ll_in_sample:.4f}")
    print(f"Coefficients: {meta['coefficients']}")
    print(f"Intercept: {meta['intercept']:.4f}")
    print(f"Saved to {MODEL_DIR / 'strike_prob_model.pkl'}")


if __name__ == "__main__":
    main()
