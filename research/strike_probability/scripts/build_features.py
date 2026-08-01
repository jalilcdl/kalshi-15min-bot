"""
Build the strike-probability training table: for every settled KXBTC15M
market, at several no-lookahead checkpoints inside its 15-min window,
compute the same features a live evaluation would have seen at that exact
minute, plus the actual settlement label.

Checkpoints: minutes 1..13 into each window (13 per market). These are
correlated within a market (same price path) -- that's accounted for in the
walk-forward harness (whole markets, never individual checkpoints, are
assigned to a fold) and flagged in the results writeup, not hidden.

No lookahead: every feature at checkpoint `ck` uses only 1-min bars with
timestamp <= open_time + ck minutes.
"""
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from signals_fast import compute_all  # noqa: E402

CHECKPOINT_MINUTES = list(range(1, 14))  # 1..13


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    btc = pd.read_csv(data_dir / "btc_1min.csv").sort_values("timestamp").reset_index(drop=True)
    markets = pd.read_csv(data_dir / "settled_markets.csv", parse_dates=["open_time", "close_time"])

    print(f"Computing indicators over {len(btc)} 1-min bars ...")
    sig = compute_all(btc)
    ts_to_idx = {int(ts): i for i, ts in enumerate(btc["timestamp"])}
    lb = config.LOOKBACK_BARS

    rows = []
    skipped_not_ready = 0
    skipped_missing_bar = 0
    for m in markets.itertuples():
        open_ts = int(m.open_time.timestamp())
        close_ts = int(m.close_time.timestamp())
        strike = m.strike
        if pd.isna(strike):
            continue
        for ck in CHECKPOINT_MINUTES:
            eval_ts = (open_ts + ck * 60) // 60 * 60
            idx = ts_to_idx.get(eval_ts)
            if idx is None:
                skipped_missing_bar += 1
                continue
            if idx < lb:
                skipped_not_ready += 1
                continue

            price = btc["close"].iloc[idx]
            row = sig.iloc[idx]
            mins_remaining = (close_ts - eval_ts) / 60.0
            if mins_remaining <= 0:
                continue

            distance_pct = (price - strike) / strike * 100.0
            realized_vol = row["realized_vol"]
            reachable = realized_vol * sqrt(mins_remaining)
            dist_over_reachable = abs(distance_pct) / reachable if reachable > 1e-9 else np.nan

            rows.append(dict(
                ticker=m.ticker, close_time=m.close_time, checkpoint_min=ck,
                minutes_remaining=mins_remaining, strike=strike, price=price,
                distance_pct=distance_pct, current_side_leading=int(distance_pct >= 0),
                realized_vol=realized_vol, reachable_move_pct=reachable,
                dist_over_reachable=dist_over_reachable,
                window_delta_pct=row["delta_pct"], ema_sep_pct=row["sep_pct"],
                rsi_val=row["rsi_val"], vol_ratio=row["vol_ratio"],
                mom=row["mom"], accel=row["accel"], net_ticks=row["net_ticks"],
                direction=row["direction"], confidence=row["confidence"],
                vote_ema=row["vote_ema"], vote_rsi=row["vote_rsi"], vote_vol=row["vote_vol"],
                vote_delta=row["vote_delta"], vote_mom=row["vote_mom"],
                vote_accel=row["vote_accel"], vote_tick=row["vote_tick"],
                trend_strong=int(row["trend_lbl"] == "Strong"),
                trend_weak=int(row["trend_lbl"] == "Weak"),
                momentum_bullish=int(row["momentum_lbl"] == "Bullish"),
                momentum_bearish=int(row["momentum_lbl"] == "Bearish"),
                y=1 if m.result == "yes" else 0,
            ))

    df = pd.DataFrame(rows)
    out_path = data_dir.parent / "results" / "features.csv"
    df.to_csv(out_path, index=False)
    print(f"Built {len(df)} checkpoint-rows from {markets['ticker'].nunique()} markets "
          f"({df['ticker'].nunique()} markets represented)")
    print(f"Skipped: {skipped_not_ready} (warmup not ready), {skipped_missing_bar} (missing 1-min bar)")
    print(f"Label base rate: {df['y'].mean():.1%} YES")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
