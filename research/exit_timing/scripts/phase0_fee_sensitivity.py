"""
Phase 0: does the edge survive REAL fees?

Reruns the full entry gate + exit backtest under each candidate fee model, at
both the paper size (10) and the size real execution is scoped for (4).

Why rerun rather than patch exit_timing_trades.csv: the fee feeds the ENTRY
GATE (edge = p - price - fee/size >= 3c), so changing the fee changes WHICH
trades qualify, not just their cost. Patching the existing CSV would silently
hold the trade set fixed at whatever the old (over-charging) model admitted.

Three fee models, because Kalshi's own sources disagree on rounding and the
difference is material at a 3c gate:

  cent_total    ceil(rate*C*P*(1-P) to $0.01)   <- what this repo shipped
  deci_milli    ceil(rate*C*P*(1-P) to $0.0001) <- docs.kalshi.com/getting_started/
                                                   fee_rounding, now in fees.py
  cent_per_ctr  ceil(rate*P*(1-P) to $0.01) * C <- third-party summaries' reading;
                                                   the most expensive candidate

If the edge survives the MOST expensive model, the residual ambiguity doesn't
matter and Phase 1 can proceed. If it only survives the cheapest, the formula
must be pinned down before any real money moves.

Fee rate/multiplier are the VERIFIED values (0.07, multiplier 1) -- see fees.py.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from model.strike_probability import compute_features, predict_p_yes  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATA = Path(__file__).resolve().parent.parent / "data"
SP = ROOT / "research" / "strike_probability"
THRESHOLDS = [0.35]           # the deployed exit target
RATE = config.KALSHI_TAKER_FEE_RATE * config.KALSHI_FEE_MULTIPLIER


def fee_cent_total(c, p):
    return math.ceil(RATE * c * p * (1 - p) * 100) / 100


def fee_deci_milli(c, p):
    return math.ceil(RATE * c * p * (1 - p) / 0.0001) * 0.0001


def fee_cent_per_ctr(c, p):
    return (math.ceil(RATE * p * (1 - p) * 100) / 100) * c


FEE_MODELS = {"cent_total": fee_cent_total,
              "deci_milli": fee_deci_milli,
              "cent_per_ctr": fee_cent_per_ctr}


def decide_entry(row, yes_ask, yes_bid, fee_fn, size):
    """paper_trader.decide_trade()'s rule, with the fee model swapped in."""
    if pd.isna(yes_ask) or pd.isna(yes_bid):
        return None
    price, strike = row["price"], row["strike"]
    mins, vol = row["minutes_remaining"], row["realized_vol"]
    feats = compute_features(price, strike, mins, vol)
    if feats["dist_over_reachable"] < config.PAPER_TRADE_MIN_DIST_OVER_REACHABLE:
        return None
    p_yes = predict_p_yes(price, strike, mins, vol)
    no_ask = 1.0 - yes_bid
    f_yes, f_no = fee_fn(size, yes_ask), fee_fn(size, no_ask)
    e_yes = p_yes - yes_ask - f_yes / size
    e_no = (1.0 - p_yes) - no_ask - f_no / size
    if e_yes >= config.PAPER_TRADE_MIN_EDGE and e_yes >= e_no:
        return dict(side="yes", entry_price=yes_ask, fee=f_yes)
    if e_no >= config.PAPER_TRADE_MIN_EDGE:
        return dict(side="no", entry_price=no_ask, fee=f_no)
    return None


def main():
    feats = pd.read_csv(SP / "results" / "features.csv")
    ck1 = feats[feats["checkpoint_min"] == 1].copy()
    # features.csv already carries close_time; drop it so the merge doesn't
    # produce close_time_x/_y and silently break the attribute lookup below.
    ck1 = ck1.drop(columns=[c for c in ("close_time", "open_time", "result") if c in ck1.columns])
    mkts = pd.read_csv(SP / "data" / "settled_markets.csv", parse_dates=["open_time", "close_time"])
    ck1 = ck1.merge(mkts[["ticker", "open_time", "close_time", "result"]], on="ticker", how="left")

    candles = pd.read_csv(DATA / "candlesticks.csv")
    by_ticker = {t: g.sort_values("ts").reset_index(drop=True) for t, g in candles.groupby("ticker")}
    print(f"candlesticks for {len(by_ticker):,} markets | {len(ck1):,} checkpoint-1 rows\n")

    rows = []
    for size in (10, 4):
        for name, fee_fn in FEE_MODELS.items():
            trades = []
            for r in ck1.itertuples():
                cs = by_ticker.get(r.ticker)
                if cs is None or cs.empty:
                    continue
                open_ts = int(pd.Timestamp(r.open_time).timestamp())
                close_ts = int(pd.Timestamp(r.close_time).timestamp())
                entry_ts = open_ts + 60
                at = cs[cs["ts"] >= entry_ts]
                if at.empty:
                    continue
                bar = at.iloc[0]
                d = decide_entry(r._asdict(), bar.get("yes_ask_close"), bar.get("yes_bid_close"),
                                 fee_fn, size)
                if d is None:
                    continue

                # exit path: first bar where the position is up >= threshold on the
                # price you could actually sell into
                after = cs[(cs["ts"] > bar["ts"]) & (cs["ts"] <= close_ts)]
                ep = d["entry_price"]
                exit_price = np.nan
                for b in after.itertuples():
                    val = b.yes_bid_close if d["side"] == "yes" else (
                        1.0 - b.yes_ask_close if pd.notna(b.yes_ask_close) else np.nan)
                    if pd.isna(val) or ep <= 0:
                        continue
                    if (val - ep) / ep >= THRESHOLDS[0]:
                        exit_price = val
                        break
                trades.append(dict(side=d["side"], entry_price=ep, entry_fee=d["fee"],
                                   result=r.result, exit_price=exit_price))

            t = pd.DataFrame(trades)
            if t.empty:
                continue
            win_hold = (t["side"] == t["result"]).astype(int)
            cost = size * t["entry_price"] + t["entry_fee"]
            payout_hold = size * win_hold

            hit = t["exit_price"].notna()
            exit_fee = t["exit_price"].apply(lambda p: fee_fn(size, p) if pd.notna(p) else 0.0)
            payout = np.where(hit, size * t["exit_price"].fillna(0), payout_hold)
            profit = payout - cost - np.where(hit, exit_fee, 0.0)

            profit_hold = (payout_hold - cost).sum()
            rows.append(dict(
                size=size, fee_model=name, n_trades=len(t),
                mean_entry_fee=t["entry_fee"].mean(),
                fee_pct_of_stake=t["entry_fee"].mean() / (size * t["entry_price"].mean()) * 100,
                turnover=cost.sum(),
                roi_hold=profit_hold / cost.sum() * 100,
                roi_exit35=profit.sum() / cost.sum() * 100,
                hit_rate=hit.mean() * 100,
                net_hold=profit_hold, net_exit35=profit.sum()))

    out = pd.DataFrame(rows)
    print("=== edge under each fee model (checkpoint-1 entries, +35% exit target) ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out.to_csv(RESULTS / "phase0_fee_sensitivity.csv", index=False)

    print("\n=== summary: does the edge survive? ===")
    for size in (10, 4):
        sub = out[out["size"] == size]
        if sub.empty:
            continue
        worst = sub.loc[sub["roi_exit35"].idxmin()]
        best = sub.loc[sub["roi_exit35"].idxmax()]
        print(f"  size {size:>2}: ROI(+35% exit) ranges {worst['roi_exit35']:+.2f}% "
              f"({worst['fee_model']}) to {best['roi_exit35']:+.2f}% ({best['fee_model']}) | "
              f"hold-to-settle {sub['roi_hold'].min():+.2f}%..{sub['roi_hold'].max():+.2f}% | "
              f"trades {sub['n_trades'].min():,}-{sub['n_trades'].max():,}")


if __name__ == "__main__":
    main()
