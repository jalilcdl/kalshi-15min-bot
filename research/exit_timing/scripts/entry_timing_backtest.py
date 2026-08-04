"""
Tests a specific, testable claim: does waiting until ~2-5 minutes into the
window (instead of the ~60-120s mark everything else in this project uses)
give a more reliable read on whether the 30% favorable-move target will hit?

Reuses decide_entry() from exit_timing_backtest.py UNCHANGED -- the exact
same production entry rule (settlement model + min-distance gate + min-edge
gate) -- just replayed at different checkpoint minutes using the real
candlestick price paths already collected. No new model, no new logic;
this is purely "does the SAME decision rule work better if made later."

Two known confounds, checked explicitly rather than glossed over:
  1. Less time remaining at later checkpoints mechanically leaves less room
     for a 30% move to happen before close. The reported hit rate already
     reflects this honestly (it scans the REAL remaining candles to the
     REAL close, no synthetic time budget) -- but it means a later
     checkpoint's hit rate needs to be read as "hit rate AND still enough
     runway," not accuracy in a vacuum. Median time-to-hit vs. minutes
     remaining is reported at every checkpoint for exactly this reason.
  2. The naive "whichever side is currently leading" baseline is recomputed
     at every checkpoint too, so any improvement can be attributed to the
     entry rule getting genuinely better information, not just to the
     mechanical effect of less time (which would move the naive baseline
     the same way).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exit_timing_backtest import decide_entry, SIZE  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(__file__).resolve().parent.parent / "results"
DATA = Path(__file__).resolve().parent.parent / "data"
STRIKE_PROB_RESULTS = ROOT / "research" / "strike_probability" / "results"
STRIKE_PROB_DATA = ROOT / "research" / "strike_probability" / "data"

CHECKPOINTS = [1, 2, 3, 4, 5, 7, 10]
TARGET = 0.30
RNG = np.random.default_rng(31)
N_BOOT = 2000


def build_at_checkpoint(features, markets, candles_by_ticker, checkpoint_min):
    ck = features[features["checkpoint_min"] == checkpoint_min].drop(columns=["close_time"]).copy()
    ck = ck.merge(markets[["ticker", "open_time", "close_time", "result"]], on="ticker", how="left")

    rows = []
    for row in ck.itertuples():
        cs = candles_by_ticker.get(row.ticker)
        if cs is None or cs.empty:
            continue
        entry_ts = int(row.open_time.timestamp()) + checkpoint_min * 60
        entry_bar = cs[cs["ts"] >= entry_ts]
        if entry_bar.empty:
            continue
        entry_bar = entry_bar.iloc[0]
        yes_ask, yes_bid = entry_bar["yes_ask_close"], entry_bar["yes_bid_close"]
        if pd.isna(yes_ask) or pd.isna(yes_bid):
            continue
        forward = cs[cs["ts"] > entry_bar["ts"]]
        if forward.empty:
            continue

        # Naive, model-free baseline: whichever side is currently ahead of the
        # strike -- recomputed at THIS checkpoint, so it moves with the same
        # "less time left" confound the model-gated entries are subject to.
        naive_side = "yes" if row.current_side_leading == 1 else "no"
        naive_entry_price = yes_ask if naive_side == "yes" else 1.0 - yes_bid
        naive_exit = forward["yes_bid_close"] if naive_side == "yes" else 1.0 - forward["yes_ask_close"]
        naive_gain = (naive_exit - naive_entry_price) / naive_entry_price
        naive_hit = bool((naive_gain >= TARGET).any())

        rec = dict(ticker=row.ticker, checkpoint_min=checkpoint_min,
                   minutes_remaining=row.minutes_remaining, close_time=row.close_time,
                   naive_side=naive_side, naive_hit_30=naive_hit)

        decision = decide_entry(row._asdict(), yes_ask, yes_bid)
        if decision is not None:
            side, entry_price = decision["side"], decision["entry_price"]
            exit_series = forward["yes_bid_close"] if side == "yes" else 1.0 - forward["yes_ask_close"]
            gain = (exit_series - entry_price) / entry_price
            hit = gain >= TARGET
            rec.update(entered=True, side=side, entry_price=entry_price, edge=decision["edge"],
                      hit_30=bool(hit.any()),
                      mins_to_hit_30=((forward.loc[hit[hit].index[0], "ts"] - entry_bar["ts"]) / 60.0) if hit.any() else np.nan)
        else:
            rec.update(entered=False, side=None, entry_price=np.nan, edge=np.nan, hit_30=np.nan, mins_to_hit_30=np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def market_bootstrap_paired(df1, df2, col, n_boot=N_BOOT):
    """Paired bootstrap on the intersection of tickers present (with a
    non-null outcome) in both checkpoints -- controls for market-specific
    effects instead of treating the two checkpoint samples as independent."""
    m = df1[["ticker", col]].merge(df2[["ticker", col]], on="ticker", suffixes=("_1", "_2")).dropna()
    if len(m) < 20:
        return np.nan, np.nan, len(m)
    tickers = m["ticker"].to_numpy()
    v1, v2 = m[f"{col}_1"].to_numpy(dtype=float), m[f"{col}_2"].to_numpy(dtype=float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, len(m), size=len(m))
        diffs[i] = v2[idx].mean() - v1[idx].mean()
    actual = v2.mean() - v1.mean()
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return actual, p, len(m)


def main():
    features = pd.read_csv(STRIKE_PROB_RESULTS / "features.csv")
    markets = pd.read_csv(STRIKE_PROB_DATA / "settled_markets.csv", parse_dates=["open_time", "close_time"])
    candles = pd.read_csv(DATA / "candlesticks.csv")
    candles_by_ticker = {t: g.sort_values("ts").reset_index(drop=True) for t, g in candles.groupby("ticker")}
    print(f"Loaded candlesticks for {len(candles_by_ticker)} markets\n")

    results = {}
    for ck in CHECKPOINTS:
        df = build_at_checkpoint(features, markets, candles_by_ticker, ck)
        results[ck] = df
        entered = df[df["entered"]]
        naive_hit_rate = df["naive_hit_30"].mean()
        n_entered = len(entered)
        hit_rate = entered["hit_30"].mean() if n_entered else float("nan")
        med_mins_remaining = df["minutes_remaining"].median()
        med_time_to_hit = entered.loc[entered["hit_30"] == True, "mins_to_hit_30"].median() if n_entered else float("nan")  # noqa: E712
        print(f"checkpoint={ck:2d}min  n_entries={n_entered:4d}/{len(df):4d}  "
              f"entry_hit_rate={hit_rate*100:5.1f}%  naive_hit_rate={naive_hit_rate*100:5.1f}%  "
              f"median_mins_remaining={med_mins_remaining:5.1f}  median_time_to_hit={med_time_to_hit:4.1f}m")
        df.to_csv(RESULTS / f"entry_timing_checkpoint_{ck}.csv", index=False)

    print("\n=== Statistical comparison: each checkpoint vs. checkpoint 1 (paired, market-level bootstrap) ===")
    base = results[1]
    for ck in CHECKPOINTS[1:]:
        cur = results[ck]
        # Entry-gated hit rate (only markets that entered at BOTH checkpoints -- a fair, paired comparison)
        actual, p, n = market_bootstrap_paired(base[base["entered"]], cur[cur["entered"]], "hit_30")
        # Naive baseline hit rate (all markets, since naive always "enters")
        actual_n, p_n, n_n = market_bootstrap_paired(base, cur, "naive_hit_30")
        sig = "SIGNIFICANT" if not np.isnan(p) and p < 0.05 else "not significant"
        sig_n = "SIGNIFICANT" if not np.isnan(p_n) and p_n < 0.05 else "not significant"
        print(f"  1min -> {ck}min:  entry-gated gap={actual:+.4f} (n={n}, p={p:.4f}, {sig})   "
              f"naive gap={actual_n:+.4f} (n={n_n}, p={p_n:.4f}, {sig_n})")


if __name__ == "__main__":
    main()
