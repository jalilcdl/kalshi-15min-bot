"""BTC/USD 1-min OHLCV feed from the Coinbase Exchange public REST API.

No auth required. Returns candles oldest-first as Candle namedtuples.
"""

from collections import namedtuple

import requests

import config

Candle = namedtuple("Candle", "ts low high open close volume")

_session = requests.Session()
_session.headers["User-Agent"] = "kalshi-15min-intel-bot/1.0 (personal use)"


def fetch_1min_candles(limit: int = config.LOOKBACK_BARS,
                       start: int | None = None,
                       end: int | None = None) -> list[Candle]:
    """Fetch 1-min candles, oldest first.

    Without start/end, Coinbase returns the most recent ~300 bars.
    With start/end (unix seconds), returns that range (max 300 bars per call).
    """
    params: dict = {"granularity": 60}
    if start is not None and end is not None:
        from datetime import datetime, timezone
        params["start"] = datetime.fromtimestamp(start, tz=timezone.utc).isoformat()
        params["end"] = datetime.fromtimestamp(end, tz=timezone.utc).isoformat()

    resp = _session.get(
        f"{config.COINBASE_BASE}/products/{config.COINBASE_PRODUCT}/candles",
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    rows = resp.json()  # [[time, low, high, open, close, volume], ...] newest first
    candles = [Candle(int(r[0]), float(r[1]), float(r[2]), float(r[3]),
                      float(r[4]), float(r[5])) for r in rows]
    candles.sort(key=lambda c: c.ts)
    return candles[-limit:]


def fetch_range_1min(start: int, end: int, chunk_delay: float = 0.3) -> list[Candle]:
    """Fetch an arbitrary 1-min candle range, paging in 300-bar chunks.

    Used by the backtester; live bot only needs fetch_1min_candles(). A
    multi-week pull is hundreds of chunks, so a small inter-chunk delay (plus
    429 backoff) keeps this well under Coinbase's public rate limit.
    """
    import time as _time
    out: list[Candle] = []
    chunk = 300 * 60
    cur = start
    while cur < end:
        chunk_end = min(cur + chunk, end)
        for attempt in range(5):
            try:
                out.extend(fetch_1min_candles(limit=300, start=cur, end=chunk_end))
                break
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429 and attempt < 4:
                    _time.sleep(chunk_delay * (2 ** attempt) + 1.0)
                    continue
                raise
        cur = chunk_end
        if chunk_delay:
            _time.sleep(chunk_delay)
    # De-dupe on timestamp, keep sorted
    seen: dict[int, Candle] = {c.ts: c for c in out}
    return [seen[ts] for ts in sorted(seen)]


def latest_price(candles: list[Candle]) -> float:
    return candles[-1].close
