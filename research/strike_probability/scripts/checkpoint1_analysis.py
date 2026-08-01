"""
Isolates model performance specifically at checkpoint_min==1 -- the earliest
point evaluated in this project, roughly 60-120s into each 15-min window
(the first fully-closed 1-min Coinbase bar after open; 1-min bar granularity
is the finest this data supports, so "the first 30 seconds" isn't literally
reachable -- this is the closest real proxy for it).

Reuses the EXACT walk-forward out-of-fold predictions already computed and
validated in walk_forward.py (results/walk_forward_pooled_predictions.csv) --
these are predictions from the same pooled models (trained on all 13
checkpoints, exactly what's deployed in model/strike_prob_model.pkl) evaluated
out-of-fold. No retraining here, just re-slicing already-validated results by
checkpoint_min, so this is directly the "how does the live model do this
early" question -- and it costs nothing to re-run since it's the same fold
data as before.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

RESULTS = Path(__file__).resolve().parent.parent / "results"
RNG = np.random.default_rng(23)
N_BOOT = 2000


def scored(df, col):
    p = df[col].clip(1e-6, 1 - 1e-6)
    return brier_score_loss(df["y"], p), log_loss(df["y"], p)


def market_bootstrap_vs_const(df, col, const_p, n_boot=N_BOOT):
    """Bootstrap (over markets) the Brier-score gap between a model's
    predictions and a constant-probability baseline (the honest 'no info,
    just the base rate' null)."""
    tmp = df.copy()
    tmp["sq_model"] = (tmp[col] - tmp["y"]) ** 2
    tmp["sq_const"] = (const_p - tmp["y"]) ** 2
    by_market = tmp.groupby("ticker")[["sq_model", "sq_const"]].mean()
    tickers = by_market.index.to_numpy()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sample = RNG.choice(tickers, size=len(tickers), replace=True)
        s = by_market.loc[sample]
        diffs[i] = s["sq_model"].mean() - s["sq_const"].mean()
    actual = by_market["sq_model"].mean() - by_market["sq_const"].mean()
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return actual, p_value


def main():
    pooled = pd.read_csv(RESULTS / "walk_forward_pooled_predictions.csv")

    ck1 = pooled[pooled["checkpoint_min"] == 1]
    later = pooled[pooled["checkpoint_min"] >= 11]  # the "late window" comparison, mirrors backtest_strike.py's checkpoints
    overall_base_rate = pooled["y"].mean()

    print(f"Checkpoint 1 (~60-120s into window): {len(ck1)} rows, {ck1['ticker'].nunique()} markets")
    print(f"Checkpoint >=11 (late window, comparison): {len(later)} rows, {later['ticker'].nunique()} markets")
    print(f"Base rate (all checkpoints): {overall_base_rate:.4f}  |  "
          f"base rate at checkpoint 1: {ck1['y'].mean():.4f}\n")

    print(f"{'Model':38s} {'Brier(ck1)':>11s} {'LogLoss(ck1)':>13s} {'Brier(late)':>12s}")
    for col, name in [("pA", "A: Base rate (time-bucketed)"),
                       ("pB", "B: Current-side-holds"),
                       ("pC", "C: Distance+time+vol only"),
                       ("pD", "D: Full model (+confluence)")]:
        b1, ll1 = scored(ck1, col)
        b2, _ = scored(later, col)
        print(f"{name:38s} {b1:11.4f} {ll1:13.4f} {b2:12.4f}")

    print("\n=== Is checkpoint-1 model D distinguishable from 'just guess the base rate'? ===")
    const_p = ck1["y"].mean()  # the honest no-information null for this exact slice
    gap, p_value = market_bootstrap_vs_const(ck1, "pD", const_p)
    verdict = "clears the null" if p_value < 0.05 and gap < 0 else "NOT distinguishable from a coin flip"
    print(f"Full model (D) vs. constant base-rate guess ({const_p:.3f}): "
          f"Brier gap={gap:+.5f}, bootstrap p={p_value:.4f} -> {verdict}")

    gapC, pC = market_bootstrap_vs_const(ck1, "pC", const_p)
    verdictC = "clears the null" if pC < 0.05 and gapC < 0 else "NOT distinguishable from a coin flip"
    print(f"Distance+time+vol (C) vs. constant base-rate guess: "
          f"Brier gap={gapC:+.5f}, bootstrap p={pC:.4f} -> {verdictC}")

    print("\n=== Calibration at checkpoint 1 (full model D) ===")
    ck1c = ck1.copy()
    ck1c["p_bucket"] = pd.cut(ck1c["pD"], bins=[0, .3, .4, .45, .5, .55, .6, .7, 1.0])
    calib = ck1c.groupby("p_bucket", observed=True).agg(n=("y", "size"), mean_pred=("pD", "mean"), actual_rate=("y", "mean"))
    print(calib.to_string())

    calib.to_csv(RESULTS / "checkpoint1_calibration.csv")

    print("\n=== Directional accuracy at checkpoint 1 (current_side_leading vs. actual) ===")
    features = pd.read_csv(RESULTS / "features.csv")
    ck1_full = features[features["checkpoint_min"] == 1]
    acc_current_side = (ck1_full["current_side_leading"] == ck1_full["y"]).mean()
    print(f"'Whichever side is barely ahead 60-120s in' accuracy: {acc_current_side:.4f} "
          f"(n={len(ck1_full)}, base rate {ck1_full['y'].mean():.4f})")


if __name__ == "__main__":
    main()
