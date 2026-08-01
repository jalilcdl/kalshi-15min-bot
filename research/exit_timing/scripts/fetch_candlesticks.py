"""
Fetches real historical 1-min price/bid/ask candlesticks for every settled
market in the strike-probability dataset (research/strike_probability/data/
settled_markets.csv). This is what lets the exit-timing question ("does the
contract move 20-40% in the buyer's favor before close, and how fast") be
answered with real historical prices instead of a model-derived proxy.

~4,250 markets x 1 request each -- this takes a while (rate-limited), so it
writes incrementally (one row group per market, appended to CSV) and can be
safely re-run: already-fetched tickers are skipped.

Usage: python fetch_candlesticks.py [out_csv]
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # kalshi-15min-bot/
sys.path.insert(0, str(ROOT))

from kalshi_feed import get_candlesticks  # noqa: E402

SETTLED_MARKETS = ROOT / "research" / "strike_probability" / "data" / "settled_markets.csv"


def main(out_path: str):
    out_path = Path(out_path)
    markets = pd.read_csv(SETTLED_MARKETS, parse_dates=["open_time", "close_time"])
    markets = markets.sort_values("close_time").reset_index(drop=True)

    already = set()
    if out_path.exists():
        existing = pd.read_csv(out_path, usecols=["ticker"])
        already = set(existing["ticker"].unique())
        print(f"Resuming: {len(already)} markets already fetched", file=sys.stderr)

    write_header = not out_path.exists()
    n_done, n_skipped, n_empty = 0, 0, 0
    t0 = time.time()

    for i, m in enumerate(markets.itertuples()):
        if m.ticker in already:
            n_skipped += 1
            continue
        start_ts = int(m.open_time.timestamp()) - 60
        end_ts = int(m.close_time.timestamp()) + 60
        try:
            candles = get_candlesticks(m.ticker, start_ts, end_ts)
        except Exception as exc:
            print(f"  [{i}] {m.ticker} FAILED: {exc}", file=sys.stderr)
            time.sleep(1.0)
            continue

        if not candles:
            n_empty += 1
        else:
            rows = [dict(ticker=m.ticker, ts=c.ts, price_close=c.price_close,
                         yes_bid_close=c.yes_bid_close, yes_bid_high=c.yes_bid_high, yes_bid_low=c.yes_bid_low,
                         yes_ask_close=c.yes_ask_close, yes_ask_high=c.yes_ask_high, yes_ask_low=c.yes_ask_low,
                         volume=c.volume) for c in candles]
            pd.DataFrame(rows).to_csv(out_path, mode="a", header=write_header, index=False)
            write_header = False

        n_done += 1
        if n_done % 100 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            remaining = (len(markets) - n_skipped - n_done) / rate if rate > 0 else float("nan")
            print(f"  {n_done} fetched ({n_empty} empty), {n_skipped} skipped, "
                  f"~{remaining/60:.1f} min remaining", file=sys.stderr)
        time.sleep(0.3)

    print(f"Done. {n_done} fetched this run ({n_empty} empty), {n_skipped} already had data.")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "../data/candlesticks.csv"
    main(out)
