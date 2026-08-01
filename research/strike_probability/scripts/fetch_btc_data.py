"""
Fetch Coinbase BTC-USD 1-min OHLCV covering the settled-markets range plus a
300-bar (5h) warmup buffer before the earliest window open.

Usage: python fetch_btc_data.py [settled_csv] [out_csv]
"""
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # kalshi-15min-bot/
sys.path.insert(0, str(ROOT))

from coinbase_feed import fetch_range_1min  # noqa: E402


def main(settled_csv: str, out_path: str):
    with open(settled_csv) as f:
        rows = list(csv.DictReader(f))
    open_times = [datetime.fromisoformat(r["open_time"]) for r in rows]
    close_times = [datetime.fromisoformat(r["close_time"]) for r in rows]
    lo = int(min(open_times).timestamp()) - 300 * 60
    hi = int(max(close_times).timestamp()) + 60

    print(f"Fetching Coinbase 1-min candles from "
          f"{datetime.fromtimestamp(lo, tz=timezone.utc)} to "
          f"{datetime.fromtimestamp(hi, tz=timezone.utc)} ...", file=sys.stderr)
    candles = fetch_range_1min(lo, hi)
    print(f"  {len(candles)} candles fetched", file=sys.stderr)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([c.ts, c.open, c.high, c.low, c.close, c.volume])
    print(f"Done. Written to {out_path}")


if __name__ == "__main__":
    settled = sys.argv[1] if len(sys.argv) > 1 else "../data/settled_markets.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "../data/btc_1min.csv"
    main(settled, out)
