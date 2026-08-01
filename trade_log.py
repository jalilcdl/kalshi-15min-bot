"""
Forward-test trade log: a dead-simple record of the Kalshi KXBTC15M trades
you actually placed, so the bot earns (or loses) a real track record instead
of being judged on vibes. Mirrors the ethos of the sports models' bet
trackers, simplified for Kalshi's mechanics.

The log is a plain CSV (DATA_DIR/trade_log.csv) you edit by hand -- add a row
when you place a trade, fill in `result` once the 15-min window settles.
This is a READ-ONLY log for the dashboard: nothing here places orders.

Columns:
    date, entry_time, market_ticker, side, entry_price_cents, size,
    model_read, btc_price_at_entry, strike, result, cost, payout, notes,
    mode, p_model, fee, exit_time, exit_price, exit_fee, exit_reason

Kalshi mechanics, unlike American-odds sports bets:
  - `entry_price_cents` IS the market's implied probability (a 40c YES
    contract prices ~40% chance of YES). No odds conversion needed.
  - Each contract costs `entry_price_cents / 100` dollars and pays exactly
    $1.00 if it settles on your side, $0 otherwise. No pushes on this
    market type.
  - `side`: "yes" or "no" -- which side of the strike you bought.
  - `result`: "yes" or "no" (the settlement outcome), or blank while the
    window is still open (pending). win = (side == result).
  - `cost` / `payout`: OPTIONAL real dollar amounts actually paid/received,
    capturing Kalshi's per-contract trading fee. If left blank, cost/payout
    fall back to the quoted entry price at `size` contracts, fee-blind.
  - `fee`: the trading fee charged on entry (dollars), if known -- kept
    separate from `cost` so it's visible on its own, not just buried inside it.
  - `model_read`: free-text snapshot of what the dashboard/bot said at entry.
  - `p_model`: the strike-probability model's stated probability of the side
    actually bought, at entry time (e.g. 0.68 for a YES buy the model rated
    68% likely). This is what calibration is checked against -- see
    research/strike_probability/README.md for the model's validated
    (non-live) accuracy; this column is what lets the LIVE record be
    checked the same way once it accumulates.
  - `mode`: "paper" (simulated fill against a real live quote, no real money)
    or "live" (an actual trade you placed). Missing/blank is treated as
    "live" -- the safer default, so a real trade is never silently
    miscounted as a paper one. paper_trader.py always writes "paper"
    explicitly. NEVER combine paper and live totals in a headline number --
    summarize()/format_summary() take a `mode` filter for exactly this reason.
  - `exit_reason`: "target_hit" (sold early because the position moved the
    configured favorable amount -- see config.PAPER_TRADE_EXIT_TARGET and
    research/exit_timing/README.md for why that threshold was chosen) or
    "settlement" (held to expiry) or blank while still pending. A row with
    exit_reason="target_hit" is treated as settled/won for reporting even
    though the real settlement result may still be unknown -- the position
    was already closed.
  - `exit_time` / `exit_price`: when and at what real quoted price a
    target-hit sale happened. `exit_fee`: the taker fee paid on THAT closing
    trade -- a round-trip early exit pays a fee on entry AND exit, unlike
    holding to settlement which only ever pays the entry fee once.

No Closing Line Value here (unlike the sports trackers): a 15-min Kalshi
window settles minutes after most entries, so there typically isn't a
separate, sharper "closing price" to compare against the way a sportsbook's
final pre-game line works. Omitted rather than faked.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRADE_LOG_FILE = DATA_DIR / "trade_log.csv"

_SETTLED = {"yes", "no"}
COLUMNS = [
    "date", "entry_time", "market_ticker", "side", "entry_price_cents", "size",
    "model_read", "btc_price_at_entry", "strike", "result", "cost", "payout", "notes",
    "mode", "p_model", "fee", "exit_time", "exit_price", "exit_fee", "exit_reason",
]


def load_log(path: Path | None = None) -> pd.DataFrame:
    """Load the trade log, normalize types, and compute derived columns."""
    path = path or TRADE_LOG_FILE
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    df["entry_price_cents"] = pd.to_numeric(df["entry_price_cents"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(1.0)
    df["side"] = df["side"].astype("string").str.strip().str.lower()
    df["result"] = df["result"].astype("string").str.strip().str.lower().replace({"nan": pd.NA, "": pd.NA})
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df["payout"] = pd.to_numeric(df["payout"], errors="coerce")
    df["p_model"] = pd.to_numeric(df["p_model"], errors="coerce")
    df["fee"] = pd.to_numeric(df["fee"], errors="coerce")
    df["mode"] = df["mode"].astype("string").str.strip().str.lower().replace({"nan": pd.NA, "": pd.NA}).fillna("live")
    df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
    df["exit_fee"] = pd.to_numeric(df["exit_fee"], errors="coerce")
    df["exit_reason"] = df["exit_reason"].astype("string").str.strip().str.lower().replace({"nan": pd.NA, "": pd.NA})

    # .fillna(False): exit_reason is nullable-string dtype, so == produces a nullable
    # "boolean" Series with pd.NA for blank rows -- plain bool ops/np.where downstream
    # can't handle that NA, so pin it to a concrete bool right away.
    target_hit = (df["exit_reason"] == "target_hit").fillna(False)
    df["settled"] = df["result"].isin(_SETTLED) | target_hit
    # Nested np.where with pd.NA-bearing object arrays raises "boolean value of NA
    # is ambiguous" -- pandas' own nullable-aware .mask() sidesteps that entirely.
    win = pd.Series(pd.NA, index=df.index, dtype="boolean")
    win = win.mask(df["result"].isin(_SETTLED), df["side"] == df["result"])
    win = win.mask(target_hit, True)
    df["win"] = win
    df["basis"] = np.where(target_hit | (df["cost"].notna() & df["payout"].notna()), "realized", "quoted")
    df["profit"] = df.apply(_row_profit, axis=1)
    return df


def _row_profit(r):
    if not r["settled"]:
        return None
    if r["basis"] == "realized":
        return r["payout"] - r["cost"]
    cost = r["size"] * r["entry_price_cents"] / 100.0
    payout = r["size"] * 1.0 if r["win"] else 0.0
    return payout - cost


def _row_cost(r):
    if r["basis"] == "realized" and pd.notna(r["cost"]):
        return r["cost"]
    return r["size"] * r["entry_price_cents"] / 100.0 if pd.notna(r["entry_price_cents"]) else np.nan


def summarize(df: pd.DataFrame | None = None, mode: str | None = None) -> dict:
    """mode: filter to "paper" or "live" only; None = both (rarely what you want --
    see the mode docstring above on why paper/live should stay separate)."""
    df = load_log() if df is None else df
    if mode is not None:
        df = df[df["mode"] == mode]
    if df.empty:
        return {"bets_total": 0, "wins": 0, "losses": 0, "pending": 0,
                "win_rate": np.nan, "turnover": 0.0, "profit": 0.0, "roi": np.nan}

    settled = df[df["settled"]]
    wins = int((settled["win"] == True).sum())  # noqa: E712
    losses = int((settled["win"] == False).sum())  # noqa: E712
    pending = int(len(df) - len(settled))
    decided = wins + losses
    win_rate = wins / decided if decided else np.nan

    turnover = settled.apply(_row_cost, axis=1).sum() if len(settled) else 0.0
    profit = settled["profit"].sum() if len(settled) else 0.0
    roi = profit / turnover if turnover else np.nan

    realized = settled[settled["basis"] == "realized"]
    fees_paid = settled["fee"].fillna(0.0).sum() if len(settled) else 0.0

    # Calibration: does the model's stated probability match the actual settled
    # win rate? Only meaningful with p_model logged (paper_trader.py always logs it).
    graded = settled.dropna(subset=["p_model"])
    calibration_gap = float(graded["p_model"].mean() - (graded["win"] == True).mean()) if len(graded) else np.nan  # noqa: E712

    return {
        "bets_total": int(len(df)),
        "wins": wins, "losses": losses, "pending": pending,
        "win_rate": win_rate,
        "turnover": float(turnover),
        "profit": float(profit),
        "roi": roi,
        "n_realized": int(len(realized)),
        "n_quoted": int(len(settled) - len(realized)),
        "fees_paid": float(fees_paid),
        "mean_p_model": float(graded["p_model"].mean()) if len(graded) else np.nan,
        "calibration_gap": calibration_gap,
        "n_with_p_model": int(len(graded)),
    }


def format_summary(df: pd.DataFrame | None = None, mode: str | None = None) -> str:
    df = load_log() if df is None else df
    if mode is not None:
        df = df[df["mode"] == mode]
    if df.empty:
        return "No trades logged yet. Add rows to " + str(TRADE_LOG_FILE)
    s = summarize(df)
    win_rate_s = "n/a" if pd.isna(s["win_rate"]) else f"{s['win_rate'] * 100:.1f}%"
    roi_s = "n/a" if pd.isna(s["roi"]) else f"{s['roi'] * 100:+.1f}%"
    lines = [
        "=" * 60,
        "KALSHI TRADE LOG - forward-test record",
        "=" * 60,
        f"Record: {s['wins']}-{s['losses']} (W-L)   |   {s['pending']} pending",
        f"Win rate (decided): {win_rate_s}",
        f"Turnover: ${s['turnover']:.2f}   Net profit: ${s['profit']:+.2f}   ROI: {roi_s}",
        f"Accounting basis: {s['n_realized']} realized (real $, fees included) / "
        f"{s['n_quoted']} quoted (fee-blind)",
    ]
    n_settled = s["wins"] + s["losses"]
    if n_settled < 30:
        lines.append(
            f"\n[!] Only {n_settled} settled trade(s). Far too small to mean anything -- "
            "a positive ROI here is noise, not proven edge. A real read needs 50-100+ trades."
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Show the Kalshi trade-log forward-test summary.")
    parser.add_argument("--table", action="store_true", help="also print the full trade log")
    args = parser.parse_args()
    df = load_log()
    if args.table and not df.empty:
        print(df[COLUMNS].to_string(index=False))
        print()
    print(format_summary(df))


if __name__ == "__main__":
    main()
