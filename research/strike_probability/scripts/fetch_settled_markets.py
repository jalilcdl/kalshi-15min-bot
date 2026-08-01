"""
Fetch settled KXBTC15M market history from Kalshi's public API and write to
CSV. Ground truth for the strike-probability model: real strikes and real
settlement outcomes, not a proxy.

Usage: python fetch_settled_markets.py [days] [out_csv]
"""
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # kalshi-15min-bot/
sys.path.insert(0, str(ROOT))

from kalshi_feed import get_settled_markets  # noqa: E402


def main(days: float, out_path: str):
    now = int(time.time())
    start = now - int(days * 86400)
    # Kalshi's cursor pagination inside get_settled_markets is capped at
    # max_pages*200 rows per call; chunk the pull into multi-day windows so
    # we never hit that cap even for a long history, and so a mid-pull
    # failure only costs one chunk instead of the whole range.
    chunk_days = 3
    all_markets = {}
    cur = start
    n_chunks = 0
    while cur < now:
        chunk_end = min(cur + int(chunk_days * 86400), now)
        markets = get_settled_markets(min_close_ts=cur, max_close_ts=chunk_end)
        for m in markets:
            all_markets[m.ticker] = m
        n_chunks += 1
        print(f"  chunk {n_chunks}: {len(markets)} markets "
              f"({len(all_markets)} unique so far)", file=sys.stderr)
        cur = chunk_end

    rows = sorted(all_markets.values(), key=lambda m: m.close_time)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "event_ticker", "strike", "open_time", "close_time",
                    "yes_bid", "yes_ask", "result"])
        for m in rows:
            w.writerow([m.ticker, m.event_ticker, m.strike,
                        m.open_time.isoformat(), m.close_time.isoformat(),
                        m.yes_bid, m.yes_ask, m.result])

    print(f"Done. {len(rows)} unique settled markets written to {out_path}")
    if rows:
        print(f"Range: {rows[0].close_time} -> {rows[-1].close_time}")
        n_yes = sum(1 for m in rows if m.result == "yes")
        print(f"Base rate: {n_yes}/{len(rows)} = {n_yes/len(rows):.1%} settled YES")


if __name__ == "__main__":
    days = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
    out = sys.argv[2] if len(sys.argv) > 2 else "../data/settled_markets.csv"
    main(days, out)
