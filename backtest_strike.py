"""Backtest the EXPERIMENTAL strike-distance module against settled Kalshi
KXBTC15M markets.

For each settled 15-min market it replays the strike-distance evaluation at
several checkpoints inside the window (using only Coinbase 1-min data known
at that moment) and scores the "Favors YES/NO" label against the actual
Kalshi settlement.

Usage:
  python backtest_strike.py [--hours 24]

Caveats:
  - Coinbase spot is a proxy for CF Benchmarks' BRTI (the actual settlement
    index); small basis differences will add noise near the strike.
  - Checkpoint evaluation uses bar closes; live evaluation sees mid-bar
    prices. Treat results as directional, not exact.
"""

import sys
import time
from collections import defaultdict

from classifier import classify
from coinbase_feed import fetch_range_1min
from indicators import compute_signals
from kalshi_feed import get_settled_markets
from strike_distance import evaluate as evaluate_strike

CHECKPOINT_MINUTES = [3, 7, 11, 13]  # minutes into the 15-min window
WARMUP_BARS = 300                    # 1-min bars of history each checkpoint needs


def run(hours: float) -> None:
    now = int(time.time())
    start = now - int(hours * 3600)
    print(f"Fetching settled markets for the last {hours:g}h ...")
    markets = get_settled_markets(min_close_ts=start, max_close_ts=now)
    print(f"  {len(markets)} settled markets with strike + result")
    if not markets:
        print("No settled markets in range — widen --hours.")
        return

    lo = min(int(m.open_time.timestamp()) for m in markets) - WARMUP_BARS * 60
    hi = max(int(m.close_time.timestamp()) for m in markets) + 60
    print(f"Fetching Coinbase 1-min candles ({(hi - lo) // 3600}h span, paged) ...")
    candles = fetch_range_1min(lo, hi)
    print(f"  {len(candles)} bars")
    by_ts = {c.ts: i for i, c in enumerate(candles)}

    # stats[(checkpoint, band)] = [hits, calls];  "Too close to call" tracked separately
    stats: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    tctc = defaultdict(int)
    skipped = 0

    for m in markets:
        open_ts = int(m.open_time.timestamp())
        for ck in CHECKPOINT_MINUTES:
            eval_ts = open_ts + ck * 60
            idx = by_ts.get((eval_ts // 60) * 60)
            if idx is None or idx < WARMUP_BARS:
                skipped += 1
                continue
            window = candles[idx - WARMUP_BARS + 1: idx + 1]
            price = window[-1].close
            sig = compute_signals(window)
            state = classify(sig)
            mins_left = (int(m.close_time.timestamp()) - eval_ts) / 60.0
            read = evaluate_strike(price, m, sig, state, minutes_remaining=mins_left)

            if read.label == "Favors YES":
                call = "yes"
            elif read.label == "Favors NO":
                call = "no"
            else:
                tctc[ck] += 1
                continue
            key = (ck, state.confidence_band)
            stats[key][1] += 1
            if call == m.result:
                stats[key][0] += 1

    print(f"\n=== Strike-distance backtest ({len(markets)} markets, "
          f"checkpoints at {CHECKPOINT_MINUTES} min into window) ===")
    if skipped:
        print(f"(skipped {skipped} checkpoint evals — missing candle data)\n")
    print(f"{'checkpoint':>10} {'conf band':>10} {'calls':>6} {'hit rate':>9}")
    for ck in CHECKPOINT_MINUTES:
        for band in ("high", "moderate", "low"):
            hits, calls = stats.get((ck, band), (0, 0))
            if calls:
                print(f"{ck:>9}m {band:>10} {calls:>6} {hits / calls:>8.1%}")
        if tctc[ck]:
            print(f"{ck:>9}m {'(no call)':>10} {tctc[ck]:>6}       n/a")

    total_hits = sum(v[0] for v in stats.values())
    total_calls = sum(v[1] for v in stats.values())
    if total_calls:
        print(f"\nOverall: {total_hits}/{total_calls} = {total_hits / total_calls:.1%} "
              f"on directional calls")
        print("Reminder: a naive 'call whichever side is currently winning' baseline "
              "is already well above 50% late in the window — compare against that, "
              "not coin-flip, before trusting this module.")


if __name__ == "__main__":
    hours = 24.0
    if "--hours" in sys.argv:
        hours = float(sys.argv[sys.argv.index("--hours") + 1])
    run(hours)
