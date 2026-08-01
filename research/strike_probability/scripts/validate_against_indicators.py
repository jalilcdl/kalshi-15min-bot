"""
Validates signals_fast.compute_all() (vectorized, continuous EMA/RSI) against
indicators.compute_signals() + classifier.classify() (the literal, real
production functions the bot and dashboard actually call) -- on REAL BTC
data, not synthetic. This is the ground truth check for the strike-
probability feature pipeline: if this doesn't match, every feature built on
top of it is wrong.

Run after btc_1min.csv exists:
    python validate_against_indicators.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from classifier import classify  # noqa: E402
from coinbase_feed import Candle  # noqa: E402
from indicators import compute_signals  # noqa: E402
from signals_fast import compute_all  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "btc_1min.csv"


def main():
    df = pd.read_csv(DATA)
    df = df.sort_values("timestamp").reset_index(drop=True)
    fast = compute_all(df)

    lb = config.LOOKBACK_BARS
    n = len(df)
    # Sample 40 evenly-spaced ready indices across the real series.
    sample_idx = np.linspace(lb, n - 1, 40).astype(int)
    sample_idx = sorted(set(sample_idx.tolist()))

    rows = []
    for t in sample_idx:
        window = df.iloc[t - lb + 1: t + 1]
        candles = [Candle(int(r.timestamp), r.low, r.high, r.open, r.close, r.volume)
                   for r in window.itertuples()]
        sig = compute_signals(candles)
        state = classify(sig)

        exact_votes = {r.name: r.vote for r in sig.results}
        fast_row = fast.iloc[t]
        fast_votes = {
            "EMA cross": fast_row["vote_ema"], "RSI": fast_row["vote_rsi"],
            "Volume surge": fast_row["vote_vol"], "Window delta": fast_row["vote_delta"],
            "Micro momentum": fast_row["vote_mom"], "Acceleration": fast_row["vote_accel"],
            "Tick trend": fast_row["vote_tick"],
        }
        vote_match = all(exact_votes[k] == fast_votes[k] for k in exact_votes)
        state_match = state.state == fast_row["state"]
        conf_match = state.confidence == fast_row["confidence"]
        rows.append(dict(t=t, vote_match=vote_match, state_match=state_match,
                          conf_match=conf_match, exact_state=state.state,
                          fast_state=fast_row["state"], exact_conf=state.confidence,
                          fast_conf=fast_row["confidence"]))

    res = pd.DataFrame(rows)
    print(f"Sampled {len(res)} real ready bars from {DATA.name}")
    print(f"Vote agreement (all 7):  {res['vote_match'].mean():.1%}")
    print(f"State agreement:         {res['state_match'].mean():.1%}")
    print(f"Confidence agreement:    {res['conf_match'].mean():.1%}")
    mismatches = res[~(res.vote_match & res.state_match & res.conf_match)]
    if len(mismatches):
        print(f"\n{len(mismatches)} mismatch(es):")
        print(mismatches.to_string(index=False))
    else:
        print("\nNo mismatches -- signals_fast.py is a faithful, validated stand-in "
              "for indicators.py + classifier.py on real data.")


if __name__ == "__main__":
    main()
