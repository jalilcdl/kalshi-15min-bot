"""
Backtests the user's ACTUAL intended strategy: buy early (~60-120s into the
window, matching the existing checkpoint-1 entry), wait for the contract
price to move in your favor, sell early if it does -- rather than holding
every position to settlement (which is what paper_trader.py currently does,
and what the whole strike-probability validation was built around).

This is a genuinely different question from "will this settle YES/NO," and
needs real historical CONTRACT price paths to answer -- fetch_candlesticks.py
pulls those from Kalshi's own candlestick API (real bid/ask history per
market, not a model-derived proxy).

Entry: reuses the exact same decision rule as paper_trader.decide_trade()
(distance/time/vol model + minimum-distance gate + minimum-edge gate),
applied to checkpoint_min==1 rows from the strike-probability feature table,
using the REAL yes_ask/no_ask read off the candlestick at that same minute
(not the single settlement-time snapshot in settled_markets.csv, which is
useless for this -- it only reflects price near/at close).

Exit: for every entered trade, scans the market's own remaining candlesticks
for the first minute the position's exit value (bid side -- what you could
actually sell into, not the ask) reaches each favorable-move threshold.
Reports both a CONSERVATIVE read (minute-close prices only) and an
OPTIMISTIC read (minute high/low, i.e. best price touched intra-minute --
requires watching every tick to actually capture, so treat as an upper bound,
not a promise).

METHODOLOGY CAVEAT (read before trusting this): the entry MODEL used here is
model/strike_prob_model.pkl, fit on ALL 45 days of this same dataset (see
research/strike_probability/fit_final_model.py). The settlement-prediction
side of that model was already validated on a proper walk-forward, held-out
basis (see ../strike_probability/README.md). But THIS analysis -- entry
decisions from the all-data model, replayed against real historical price
paths on the same 45 days -- is a historical REPLAY of what the live
strategy would have done, not a fresh nested walk-forward test of entry+exit
together. The price-path data itself was never used to fit anything, so
there's no direct leakage, but it's not the same rigor as a strict train/test
split. Flagged plainly, not hidden.
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
from fees import kalshi_fee  # noqa: E402
from model.strike_probability import compute_features, predict_p_yes  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATA = Path(__file__).resolve().parent.parent / "data"
STRIKE_PROB_RESULTS = ROOT / "research" / "strike_probability" / "results"

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]
SIZE = config.PAPER_TRADE_SIZE


def decide_entry(row, yes_ask, yes_bid):
    """Exact same rule as paper_trader.decide_trade(), replayed on historical data."""
    if pd.isna(yes_ask) or pd.isna(yes_bid):
        return None
    price, strike, mins_remaining, vol = row["price"], row["strike"], row["minutes_remaining"], row["realized_vol"]
    feats = compute_features(price, strike, mins_remaining, vol)
    if feats["dist_over_reachable"] < config.PAPER_TRADE_MIN_DIST_OVER_REACHABLE:
        return None
    p_yes = predict_p_yes(price, strike, mins_remaining, vol)
    no_ask = 1.0 - yes_bid
    fee_yes = kalshi_fee(SIZE, yes_ask)
    fee_no = kalshi_fee(SIZE, no_ask)
    edge_yes = p_yes - yes_ask - fee_yes / SIZE
    edge_no = (1.0 - p_yes) - no_ask - fee_no / SIZE
    if edge_yes >= config.PAPER_TRADE_MIN_EDGE and edge_yes >= edge_no:
        return dict(side="yes", entry_price=yes_ask, fee=fee_yes, p_model=p_yes, edge=edge_yes)
    if edge_no >= config.PAPER_TRADE_MIN_EDGE:
        return dict(side="no", entry_price=no_ask, fee=fee_no, p_model=1.0 - p_yes, edge=edge_no)
    return None


def main():
    features = pd.read_csv(STRIKE_PROB_RESULTS / "features.csv")
    ck1 = features[features["checkpoint_min"] == 1].copy()
    markets = pd.read_csv(ROOT / "research" / "strike_probability" / "data" / "settled_markets.csv",
                          parse_dates=["open_time", "close_time"])
    ck1 = ck1.merge(markets[["ticker", "open_time", "close_time", "result"]], on="ticker", how="left")

    candles = pd.read_csv(DATA / "candlesticks.csv")
    candles_by_ticker = {t: g.sort_values("ts").reset_index(drop=True) for t, g in candles.groupby("ticker")}
    print(f"Loaded candlesticks for {len(candles_by_ticker)} markets "
          f"({len(candles)} total 1-min bars)")

    trades = []
    n_no_candles, n_no_entry_bar = 0, 0
    for row in ck1.itertuples():
        cs = candles_by_ticker.get(row.ticker)
        if cs is None or cs.empty:
            n_no_candles += 1
            continue

        entry_ts = int(row.open_time.timestamp()) + 60  # checkpoint_min==1 -> ~60-120s in
        entry_bar = cs[cs["ts"] >= entry_ts]
        if entry_bar.empty:
            n_no_entry_bar += 1
            continue
        entry_bar = entry_bar.iloc[0]
        yes_ask, yes_bid = entry_bar["yes_ask_close"], entry_bar["yes_bid_close"]

        decision = decide_entry(row._asdict(), yes_ask, yes_bid)
        if decision is None:
            continue

        side = decision["side"]
        entry_price = decision["entry_price"]
        forward = cs[cs["ts"] > entry_bar["ts"]]
        if forward.empty:
            continue

        # Conservative exit value (minute close) and optimistic (best intra-minute touch)
        if side == "yes":
            exit_close = forward["yes_bid_close"]
            exit_optimistic = forward["yes_bid_high"]
        else:
            exit_close = 1.0 - forward["yes_ask_close"]
            exit_optimistic = 1.0 - forward["yes_ask_low"]
        gain_close = (exit_close - entry_price) / entry_price
        gain_optimistic = (exit_optimistic - entry_price) / entry_price

        rec = dict(ticker=row.ticker, side=side, entry_price=entry_price, entry_fee=decision["fee"],
                   p_model=decision["p_model"], edge=decision["edge"], result=row.result,
                   max_gain_close=float(gain_close.max()) if len(gain_close) else np.nan,
                   max_gain_optimistic=float(gain_optimistic.max()) if len(gain_optimistic) else np.nan)
        for thr in THRESHOLDS:
            hit_close = gain_close[gain_close >= thr]
            hit_opt = gain_optimistic[gain_optimistic >= thr]
            pct = int(thr * 100)
            rec[f"hit_close_{pct}"] = bool(len(hit_close))
            rec[f"mins_to_hit_close_{pct}"] = (
                (forward.loc[hit_close.index[0], "ts"] - entry_bar["ts"]) / 60.0 if len(hit_close) else np.nan)
            rec[f"exit_price_close_{pct}"] = (
                float(exit_close.loc[hit_close.index[0]]) if len(hit_close) else np.nan)
            rec[f"hit_opt_{pct}"] = bool(len(hit_opt))
            rec[f"mins_to_hit_opt_{pct}"] = (
                (forward.loc[hit_opt.index[0], "ts"] - entry_bar["ts"]) / 60.0 if len(hit_opt) else np.nan)
        trades.append(rec)

    df = pd.DataFrame(trades)
    df.to_csv(RESULTS / "exit_timing_trades.csv", index=False)
    print(f"\n{len(df)} trades entered (of {len(ck1)} checkpoint-1 rows; "
          f"{n_no_candles} markets missing candlestick data, {n_no_entry_bar} missing an entry-minute bar)")
    if df.empty:
        print("No trades entered -- cannot evaluate exit timing on zero trades.")
        return

    print(f"\n{'Target':>8s} {'Hit rate (close)':>18s} {'Median min-to-hit':>19s} "
          f"{'Hit rate (optimistic)':>23s} {'Median min-to-hit':>19s}")
    for thr in THRESHOLDS:
        pct = int(thr * 100)
        hit_c = df[f"hit_close_{pct}"].mean()
        med_c = df.loc[df[f"hit_close_{pct}"], f"mins_to_hit_close_{pct}"].median()
        hit_o = df[f"hit_opt_{pct}"].mean()
        med_o = df.loc[df[f"hit_opt_{pct}"], f"mins_to_hit_opt_{pct}"].median()
        print(f"{pct:>7d}% {hit_c*100:>17.1f}% {med_c:>18.1f}m {hit_o*100:>22.1f}% {med_o:>18.1f}m")

    print(f"\nMax favorable excursion (conservative, close basis): "
          f"mean={df['max_gain_close'].mean()*100:.1f}%  median={df['max_gain_close'].median()*100:.1f}%")
    print(f"Fraction of trades that NEVER moved favorably at all (max gain <= 0): "
          f"{(df['max_gain_close'] <= 0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
