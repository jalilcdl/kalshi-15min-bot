"""
Compares the fitted models (B/C/D from walk_forward.py) against the ACTUAL
currently-shipped strike_distance.py heuristic (the 1.5-sigma rule), on the
same pooled out-of-fold test rows. strike_distance.py outputs a label
(Favors YES / Favors NO / Too close to call), not a probability, so this
compares directional accuracy on calls made, excluding abstentions -- the
same methodology backtest_strike.py already uses, applied here at a much
larger sample size (thousands of markets instead of dozens).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def heuristic_label(row):
    distance_pct = row["distance_pct"]
    mins = row["minutes_remaining"]
    reachable = row["reachable_move_pct"]
    realized_vol = row["realized_vol"]
    confidence = row["confidence"]
    direction = row["direction"]

    winning_side = "yes" if distance_pct >= 0 else "no"
    losing_side = "no" if winning_side == "yes" else "yes"
    required_per_min = abs(distance_pct) / mins if mins > 0 else float("inf")

    if reachable > 0 and abs(distance_pct) >= config.STRIKE_SAFE_SIGMAS * reachable:
        return winning_side
    if confidence >= config.CONFIDENCE_HIGH and direction != 0:
        trend_side = "yes" if direction > 0 else "no"
        if trend_side == winning_side:
            return winning_side
        if required_per_min <= realized_vol:
            return losing_side
        return None
    return None


def prob_call(p, threshold=0.5):
    return np.where(p >= threshold, "yes", "no")


def main():
    features = pd.read_csv(RESULTS / "features.csv")
    pooled = pd.read_csv(RESULTS / "walk_forward_pooled_predictions.csv")
    df = pooled.merge(features, on=["ticker", "close_time", "checkpoint_min", "y"], how="left")

    df["heuristic_call"] = df.apply(heuristic_label, axis=1)
    made_call = df["heuristic_call"].notna()
    actual = np.where(df["y"] == 1, "yes", "no")

    print(f"Rows evaluated: {len(df)}")
    print(f"\n=== Currently-shipped strike_distance.py heuristic ===")
    n_calls = made_call.sum()
    n_correct = (df.loc[made_call, "heuristic_call"] == actual[made_call]).sum()
    print(f"  Made a call on {n_calls}/{len(df)} rows ({n_calls/len(df):.1%}), "
          f"abstained ('too close to call') on the rest")
    print(f"  Accuracy on calls made: {n_correct}/{n_calls} = {n_correct/n_calls:.1%}")

    print(f"\n=== Fitted models, thresholded at 50%, SAME rows the heuristic called on ===")
    for label, name in [("B", "Current-side-holds (time-bucketed)"),
                         ("C", "Distance+time+vol only"),
                         ("D", "Full model (+confluence)")]:
        call = prob_call(df[f"p{label}"])
        acc_same_rows = (call[made_call] == actual[made_call]).mean()
        acc_all_rows = (call == actual).mean()
        print(f"  {name:38s}  on heuristic's {n_calls} calls: {acc_same_rows:.1%}   "
              f"on all {len(df)} rows: {acc_all_rows:.1%}")

    print(f"\n=== Fitted models on rows the heuristic ABSTAINED on (the hard cases) ===")
    abstained = ~made_call
    print(f"  {abstained.sum()} rows")
    for label, name in [("B", "Current-side-holds"), ("C", "Distance+time+vol"), ("D", "Full model")]:
        call = prob_call(df[f"p{label}"])
        acc = (call[abstained] == actual[abstained]).mean()
        print(f"  {name:38s}  accuracy: {acc:.1%}")


if __name__ == "__main__":
    main()
