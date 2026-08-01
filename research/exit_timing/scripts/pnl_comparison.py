"""
Full round-trip P&L comparison: "sell early at the first favorable-move
threshold, else hold to settlement" vs. "always hold to settlement"
(paper_trader.py's current behavior) -- on the SAME entered trades, using
real historical prices and REAL DOUBLE-SIDED FEES (a round-trip scalp pays
the Kalshi taker fee on both the entry AND the exit; holding to settlement
only ever pays it once, on entry -- settlement itself isn't a fee-charged
trade). This is the number that actually answers "does exiting early beat
just holding," not just "does the price move favorably."

Uses the conservative (minute-close) exit prices from exit_timing_backtest.py
-- the optimistic (intra-minute high/low) read requires catching a tick you
may not actually see, so it's excluded from the money math on purpose.

Run after exit_timing_backtest.py has produced results/exit_timing_trades.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from fees import kalshi_fee  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]
SIZE = config.PAPER_TRADE_SIZE


def main():
    df = pd.read_csv(RESULTS / "exit_timing_trades.csv")
    if df.empty:
        print("No trades to evaluate.")
        return

    df["win_hold"] = (df["side"] == df["result"]).astype(int)
    df["cost_entry"] = SIZE * df["entry_price"] + df["entry_fee"]
    df["payout_hold"] = SIZE * df["win_hold"]
    df["profit_hold"] = df["payout_hold"] - df["cost_entry"]

    print(f"n = {len(df)} trades (checkpoint-1 entries that cleared the model's edge threshold)\n")
    print(f"{'Strategy':45s} {'Turnover':>10s} {'Net profit':>12s} {'ROI':>8s} {'Win rate':>9s}")

    turnover_hold = df["cost_entry"].sum()
    profit_hold = df["profit_hold"].sum()
    print(f"{'Always hold to settlement (baseline)':45s} ${turnover_hold:>9.2f} "
          f"${profit_hold:>+11.2f} {profit_hold/turnover_hold*100:>7.1f}% {df['win_hold'].mean()*100:>8.1f}%")

    for thr in THRESHOLDS:
        pct = int(thr * 100)
        hit = df[f"hit_close_{pct}"]
        exit_price = df[f"exit_price_close_{pct}"]
        exit_fee = exit_price.apply(lambda p: kalshi_fee(SIZE, p) if pd.notna(p) else 0.0)

        payout = np.where(hit, SIZE * exit_price, df["payout_hold"])
        extra_fee = np.where(hit, exit_fee, 0.0)
        profit = payout - df["cost_entry"] - extra_fee
        win = np.where(hit, True, df["win_hold"].astype(bool))  # "sold for a gain" counts as a win

        turnover = df["cost_entry"].sum()
        net_profit = profit.sum()
        roi = net_profit / turnover * 100
        win_rate = win.mean() * 100
        hit_rate = hit.mean() * 100
        print(f"{'Sell at +' + str(pct) + '% (else hold)':45s} ${turnover:>9.2f} "
              f"${net_profit:>+11.2f} {roi:>7.1f}% {win_rate:>8.1f}%   (hit {hit_rate:.0f}% of trades)")

    print("\nNote: 'win rate' here means 'exited for a gain OR settled in your favor' -- not the same")
    print("thing as the settlement-only win rate reported elsewhere in this project. ROI is net of BOTH")
    print("the entry fee and (when the target is hit) a second exit-trade fee, unlike the hold-only")
    print("baseline which only ever pays the entry fee once.")


if __name__ == "__main__":
    main()
