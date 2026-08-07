"""
Walk-forward validation of the strike-crossing probability, against baselines
that actually have a chance of winning.

Candidates:
  A. base_rate    -- one global number from the training fold. The "is any of
                     this worth it" floor.
  B. time_only    -- training-fold flip rate bucketed by minutes remaining.
                     Time decay alone is a strong effect here, so a distance/vol
                     model must beat THIS, not just the global rate.
  C. reflection   -- parameter-free closed form. For driftless Brownian motion
                     the probability of touching a barrier d away within time T
                     is 2*Phi(-d/(sigma*sqrt(T))). The existing feature
                     dist_over_reachable IS d/(sigma*sqrt(T)), so this is
                     2*Phi(-dist_over_reachable) with NOTHING fitted. If the
                     fitted model can't beat a textbook formula, the fitting is
                     not earning its place.
  D. logistic     -- fitted on the same feature family as the settlement model.

Folds are chronological and expanding, split on market close_time so every
checkpoint of a given market lands in exactly one fold -- the 13 checkpoints of
one window share a price path and are NOT independent. Same discipline as
../../strike_probability/scripts/walk_forward.py.

Reported: Brier, log loss, AUC, and a calibration table (predicted vs actual),
because a crossing probability that is discriminative but miscalibrated is
useless for the intended use -- the number gets DISPLAYED, so it has to mean
what it says.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

RESULTS = Path(__file__).resolve().parent.parent / "results"

FEATURES = [
    "abs_distance_pct", "minutes_remaining", "realized_vol",
    "dist_over_reachable", "reachable_move_pct",
    "vol_ratio", "rsi_val", "mom", "accel", "net_ticks",
]
LABEL = "flip_close"


def make_folds(df, n_folds=6, min_train_frac=0.30):
    by_time = df.drop_duplicates("ticker")[["ticker", "close_time"]].sort_values("close_time")
    n = len(by_time)
    start = int(n * min_train_frac)
    bounds = np.linspace(start, n, n_folds + 1).astype(int)
    folds = []
    for i in range(n_folds):
        tr = set(by_time["ticker"].iloc[:bounds[i]])
        te = set(by_time["ticker"].iloc[bounds[i]:bounds[i + 1]])
        if te:
            folds.append((tr, te))
    return folds


def reflection_p(dist_over_reachable):
    """P(touch barrier before T) for driftless BM. Clipped to [0,1]; the formula
    can exceed 1 for very small standardized distances."""
    return np.clip(2.0 * norm.cdf(-np.asarray(dist_over_reachable, dtype=float)), 0.0, 1.0)


def evaluate(name, y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(model=name, n=len(y), brier=brier_score_loss(y, p),
                logloss=log_loss(y, p), auc=roc_auc_score(y, p),
                mean_pred=float(np.mean(p)), actual=float(np.mean(y)))


def main():
    df = pd.read_csv(RESULTS / "crossing_features.csv", parse_dates=["close_time"])
    df = df.sort_values(["close_time", "checkpoint_min"]).reset_index(drop=True)
    print(f"{len(df):,} rows / {df['ticker'].nunique():,} markets · base rate {df[LABEL].mean():.4f}\n")

    folds = make_folds(df)
    preds = {k: [] for k in ("base_rate", "time_only", "reflection", "logistic")}
    ys, metas = [], []

    for i, (tr_t, te_t) in enumerate(folds, 1):
        tr = df[df["ticker"].isin(tr_t)]
        te = df[df["ticker"].isin(te_t)]
        y_te = te[LABEL].to_numpy()

        # A. base rate
        preds["base_rate"].append(np.full(len(te), tr[LABEL].mean()))

        # B. time-only lookup (fallback to global mean for unseen buckets)
        tbl = tr.groupby("checkpoint_min")[LABEL].mean()
        preds["time_only"].append(te["checkpoint_min"].map(tbl).fillna(tr[LABEL].mean()).to_numpy())

        # C. reflection principle -- nothing fitted
        preds["reflection"].append(reflection_p(te["dist_over_reachable"]))

        # D. logistic
        sc = StandardScaler().fit(tr[FEATURES])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(tr[FEATURES]), tr[LABEL])
        preds["logistic"].append(clf.predict_proba(sc.transform(te[FEATURES]))[:, 1])

        ys.append(y_te)
        metas.append(te[["ticker", "checkpoint_min", "minutes_remaining",
                         "dist_over_reachable", "abs_distance_pct"]])
        print(f"fold {i}: train {len(tr):,} rows / test {len(te):,} rows "
              f"({te['close_time'].min():%Y-%m-%d} -> {te['close_time'].max():%Y-%m-%d})")

    y = np.concatenate(ys)
    out = {k: np.concatenate(v) for k, v in preds.items()}
    meta = pd.concat(metas, ignore_index=True)

    print("\n=== pooled out-of-sample (all folds) ===")
    res = pd.DataFrame([evaluate(k, y, v) for k, v in out.items()])
    res = res.sort_values("brier")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Significance of the best model vs each baseline: market-level bootstrap,
    # resampling whole markets so correlated checkpoints move together.
    best = res.iloc[0]["model"]
    print(f"\n=== market-level bootstrap: does '{best}' really beat each baseline on Brier? ===")
    rng = np.random.default_rng(11)
    tickers = meta["ticker"].to_numpy()
    uniq = np.unique(tickers)
    idx_by_ticker = {t: np.where(tickers == t)[0] for t in uniq}
    for other in [k for k in out if k != best]:
        diffs = []
        for _ in range(1000):
            samp = rng.choice(uniq, size=len(uniq), replace=True)
            rows = np.concatenate([idx_by_ticker[t] for t in samp])
            yy = y[rows]
            if yy.min() == yy.max():
                continue
            d = brier_score_loss(yy, np.clip(out[other][rows], 1e-6, 1 - 1e-6)) - \
                brier_score_loss(yy, np.clip(out[best][rows], 1e-6, 1 - 1e-6))
            diffs.append(d)
        diffs = np.array(diffs)
        p = float(np.mean(diffs <= 0))  # fraction where best did NOT beat other
        print(f"  {best} vs {other:12s}: mean Brier improvement {diffs.mean():+.5f}  "
              f"95% CI [{np.percentile(diffs,2.5):+.5f}, {np.percentile(diffs,97.5):+.5f}]  p={p:.4f}")

    # Calibration for every candidate that could plausibly be shipped.
    print("\n=== calibration (predicted vs actual), decile buckets ===")
    calib_frames = []
    for name in ("logistic", "reflection", "time_only"):
        p = out[name]
        q = pd.qcut(p, 10, labels=False, duplicates="drop")
        c = pd.DataFrame({"bucket": q, "pred": p, "actual": y}).groupby("bucket").agg(
            n=("actual", "size"), mean_pred=("pred", "mean"), actual_rate=("actual", "mean"))
        c["gap"] = c["actual_rate"] - c["mean_pred"]
        c.insert(0, "model", name)
        calib_frames.append(c.reset_index())
        print(f"\n-- {name} --")
        print(c.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"   max |gap| = {c['gap'].abs().max():.4f}   mean |gap| = {c['gap'].abs().mean():.4f}")

    pd.concat(calib_frames).to_csv(RESULTS / "calibration.csv", index=False)
    res.to_csv(RESULTS / "model_comparison.csv", index=False)

    # Where it matters most: the sit-out state (read displayed, can't act yet).
    print("\n=== actual flip rate by (minutes remaining x standardized distance) ===")
    meta = meta.assign(actual=y, pred=out[best])
    meta["dor_bucket"] = pd.cut(meta["dist_over_reachable"],
                                [0, 0.25, 0.5, 1.0, 2.0, np.inf],
                                labels=["<0.25", "0.25-0.5", "0.5-1", "1-2", ">2"])
    meta["mins_bucket"] = pd.cut(meta["minutes_remaining"], [0, 3, 6, 9, 15],
                                 labels=["0-3m", "3-6m", "6-9m", "9-15m"])
    grid = meta.groupby(["mins_bucket", "dor_bucket"], observed=True).agg(
        n=("actual", "size"), actual=("actual", "mean"), pred=("pred", "mean"))
    grid["gap"] = grid["actual"] - grid["pred"]
    print(grid.to_string(float_format=lambda x: f"{x:.4f}"))
    grid.reset_index().to_csv(RESULTS / "flip_rate_grid.csv", index=False)


if __name__ == "__main__":
    main()
