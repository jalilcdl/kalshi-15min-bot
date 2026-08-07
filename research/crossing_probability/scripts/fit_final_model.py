"""Fit the SHIPPING crossing model and persist it.

FEATURE SET DECISION (revised from the first draft of the README -- read this
before "simplifying" it back):

The walk-forward compared a 10-feature logistic against a 2-feature one:

    10 features        Brier 0.1706   AUC 0.8116   max decile calib gap 0.032
    dor + mins_left    Brier 0.1716   AUC 0.8098   max decile calib gap 0.054

The extra eight buy 0.001 Brier and *cost* 0.002 AUC. Against that, shipping
them means plumbing rsi_val / mom / accel / vol_ratio / net_ticks into live
code: indicators.Signals does not expose any of them (only realized_vol_pct,
window_delta_pct, ema_sep_pct), so they would have to be newly surfaced and
their live-vs-research equivalence separately established.

That equivalence is exactly where this project has been bitten before -- the
dashboard/Telegram contradiction and the stale-price incident were both
train/serve or implementation skew. Measured directly:

    realized_vol   live vs research   diff 5e-16   (machine precision)
    window_delta   live vs research   diff 0       (exact)
    ema_sep        live vs research   diff 1e-5    (EMA seeding, negligible)

realized_vol is already proven equivalent and already used live by the
settlement model; minutes_remaining is exact arithmetic. So dist_over_reachable
+ minutes_remaining introduces ZERO new train/serve surface, while the 10-feature
version introduces five unverified ones for a 0.001 Brier gain.

Ship the two-feature model. The calibration cost (5.4 vs 3.2 pts max gap in a
single decile; 1.8 vs 1.7 pts mean) is real and is stated in the README rather
than buried.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RESULTS = Path(__file__).resolve().parent.parent / "results"

from walk_forward_crossing import LABEL, make_folds  # noqa: E402

SHIP_FEATURES = ["dist_over_reachable", "minutes_remaining"]


def main():
    df = pd.read_csv(RESULTS / "crossing_features.csv", parse_dates=["close_time"])
    df = df.sort_values(["close_time", "checkpoint_min"]).reset_index(drop=True)

    # Re-verify the shipped feature set standalone, out-of-sample, so the
    # numbers written into the pkl metadata describe THIS model.
    ys, ps = [], []
    for tr_t, te_t in make_folds(df):
        tr, te = df[df["ticker"].isin(tr_t)], df[df["ticker"].isin(te_t)]
        pipe = Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000))]).fit(tr[SHIP_FEATURES], tr[LABEL])
        ps.append(pipe.predict_proba(te[SHIP_FEATURES])[:, 1])
        ys.append(te[LABEL].to_numpy())
    y, p = np.concatenate(ys), np.clip(np.concatenate(ps), 1e-6, 1 - 1e-6)

    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    calib = pd.DataFrame({"b": q, "pred": p, "actual": y}).groupby("b").agg(
        n=("actual", "size"), mean_pred=("pred", "mean"), actual_rate=("actual", "mean"))
    calib["gap"] = calib["actual_rate"] - calib["mean_pred"]

    oos = dict(brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p)),
               auc=float(roc_auc_score(y, p)),
               max_calib_gap=float(calib["gap"].abs().max()),
               mean_calib_gap=float(calib["gap"].abs().mean()))
    print("=== shipped (2-feature) model, pooled out-of-sample ===")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in oos.items()))
    print("\ncalibration deciles:")
    print(calib.to_string(float_format=lambda x: f"{x:.4f}"))
    calib.reset_index().to_csv(RESULTS / "calibration_shipped.csv", index=False)

    # Final fit on everything.
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000))]).fit(df[SHIP_FEATURES], df[LABEL])
    out = ROOT / "model" / "crossing_prob_model.pkl"
    joblib.dump({"pipeline": pipe, "features": SHIP_FEATURES, "label": LABEL,
                 "label_definition": "1 if any later 1-min CLOSE lands on the opposite "
                                     "side of the strike before window close",
                 "n_train_rows": int(len(df)),
                 "n_train_markets": int(df["ticker"].nunique()),
                 "trained_through": str(df["close_time"].max()),
                 "oos": oos}, out)
    print(f"\nsaved -> {out}")
    print("coefficients (standardized):")
    for f, c in zip(SHIP_FEATURES, pipe.named_steps["clf"].coef_[0]):
        print(f"  {f:22s} {c:+.4f}")

    print("\n=== badge reference grid (median vol) ===")
    grid = []
    for mins in (2, 5, 9, 13):
        for dor in (0.15, 0.25, 0.5, 1.0, 2.0):
            X = pd.DataFrame([{"dist_over_reachable": dor, "minutes_remaining": mins}])[SHIP_FEATURES]
            grid.append(dict(mins_left=mins, dor=dor,
                             flip=round(float(pipe.predict_proba(X)[0, 1]), 3)))
    print(pd.DataFrame(grid).pivot(index="mins_left", columns="dor", values="flip").to_string())


if __name__ == "__main__":
    main()
