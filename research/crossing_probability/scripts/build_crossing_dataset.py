"""
Build the strike-crossing training table.

QUESTION (different from the settlement model's): the settlement model answers
"which side wins at close." This answers "how stable is the read I'm looking at
right now" -- given where price sits relative to the strike, how likely is it to
cross back over the strike at least once before the window closes.

Formally, at checkpoint minute `ck` of a window, with price P, strike K:
    current_side = sign(P - K)
    LABEL = 1 if ANY later observation in (ck, close] is on the opposite side.

Two label variants are built, because "crossed" is genuinely ambiguous on 1-min
bars and the choice changes the number materially:

  flip_close    -- a later 1-min CLOSE is on the opposite side. This is what the
                   dashboard would actually SHOW flipping, since it renders one
                   spot price per refresh. The honest match to the displayed read.
  flip_touch    -- a later bar's [low, high] range reaches the strike at all.
                   A superset: counts wicks that touch and retreat without any
                   close landing on the other side.

flip_close is the primary. flip_touch is carried so the writeup can be explicit
about how much of the difference is wick noise rather than a real regime change.

NO LOOKAHEAD: features at checkpoint `ck` use only bars with timestamp <=
open_time + ck minutes; the label uses only bars strictly after that. Feature
definitions are lifted verbatim from ../../strike_probability/scripts/
build_features.py so the two models are directly comparable and the live code
can reuse the exact same inputs.
"""
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "strike_probability" / "scripts"))

import config  # noqa: E402
from signals_fast import compute_all  # noqa: E402

CHECKPOINT_MINUTES = list(range(1, 14))  # 1..13, same as the settlement model


def main():
    sp_data = ROOT / "research" / "strike_probability" / "data"
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    btc = pd.read_csv(sp_data / "btc_1min.csv").sort_values("timestamp").reset_index(drop=True)
    markets = pd.read_csv(sp_data / "settled_markets.csv",
                          parse_dates=["open_time", "close_time"])

    print(f"Computing indicators over {len(btc)} 1-min bars ...")
    sig = compute_all(btc)
    ts_to_idx = {int(ts): i for i, ts in enumerate(btc["timestamp"])}
    lb = config.LOOKBACK_BARS

    closes = btc["close"].to_numpy()
    highs = btc["high"].to_numpy()
    lows = btc["low"].to_numpy()
    stamps = btc["timestamp"].to_numpy()

    rows = []
    skipped_missing_bar = skipped_not_ready = skipped_no_future = skipped_at_strike = 0

    for m in markets.itertuples():
        strike = m.strike
        if pd.isna(strike):
            continue
        open_ts = int(m.open_time.timestamp())
        close_ts = int(m.close_time.timestamp())

        for ck in CHECKPOINT_MINUTES:
            eval_ts = (open_ts + ck * 60) // 60 * 60
            idx = ts_to_idx.get(eval_ts)
            if idx is None:
                skipped_missing_bar += 1
                continue
            if idx < lb:
                skipped_not_ready += 1
                continue

            price = closes[idx]
            distance_pct = (price - strike) / strike * 100.0
            if distance_pct == 0.0:
                skipped_at_strike += 1   # side undefined; rare but must not be guessed
                continue
            side = 1 if distance_pct > 0 else -1

            # Future bars strictly after the checkpoint, through close.
            fut = np.where((stamps > eval_ts) & (stamps <= close_ts))[0]
            if fut.size == 0:
                skipped_no_future += 1
                continue

            fut_close, fut_high, fut_low = closes[fut], highs[fut], lows[fut]
            if side > 0:
                flip_close = bool(np.any(fut_close < strike))
                flip_touch = bool(np.any(fut_low <= strike))
            else:
                flip_close = bool(np.any(fut_close > strike))
                flip_touch = bool(np.any(fut_high >= strike))

            row = sig.iloc[idx]
            mins_remaining = (close_ts - eval_ts) / 60.0
            realized_vol = row["realized_vol"]                    # percent units
            reachable = realized_vol * sqrt(mins_remaining)
            dist_over_reachable = abs(distance_pct) / reachable if reachable > 1e-9 else np.nan

            rows.append(dict(
                ticker=m.ticker, close_time=m.close_time, checkpoint_min=ck,
                minutes_remaining=mins_remaining, strike=strike, price=price,
                distance_pct=distance_pct, abs_distance_pct=abs(distance_pct),
                current_side=side, realized_vol=realized_vol,
                reachable_move_pct=reachable, dist_over_reachable=dist_over_reachable,
                n_future_bars=int(fut.size),
                window_delta_pct=row["delta_pct"], ema_sep_pct=row["sep_pct"],
                rsi_val=row["rsi_val"], vol_ratio=row["vol_ratio"],
                mom=row["mom"], accel=row["accel"], net_ticks=row["net_ticks"],
                flip_close=int(flip_close), flip_touch=int(flip_touch),
            ))

    df = pd.DataFrame(rows).sort_values(["close_time", "checkpoint_min"]).reset_index(drop=True)
    df = df.dropna(subset=["dist_over_reachable"])
    out = out_dir / "crossing_features.csv"
    df.to_csv(out, index=False)

    print(f"\nWrote {len(df):,} checkpoint rows across {df['ticker'].nunique():,} markets -> {out}")
    print(f"skipped: missing_bar={skipped_missing_bar} not_ready={skipped_not_ready} "
          f"no_future={skipped_no_future} exactly_at_strike={skipped_at_strike}")
    print(f"\nbase rate flip_close: {df['flip_close'].mean():.4f}")
    print(f"base rate flip_touch: {df['flip_touch'].mean():.4f}")
    print(f"\nflip_close rate by minutes remaining:")
    print(df.groupby("checkpoint_min")[["flip_close", "flip_touch"]].agg(["mean", "size"]).to_string())


if __name__ == "__main__":
    main()
