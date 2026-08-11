"""BTC/USD price feeds from the Coinbase Exchange public REST API.

No auth required.

TWO feeds, and the split matters -- read this before changing a call site:

  fetch_spot_price()   -> real-time last trade. Use this for ANYTHING that
                          compares price to a strike (the model's distance
                          feature, entry decisions, alerts, the dashboard's
                          live read).
  fetch_1min_candles() -> 1-min OHLCV history. Use this ONLY for volatility
                          and indicator math over a lookback window, and for
                          backtests. Do NOT derive a live price from it.

Why (incident 2026-08-06): the /candles endpoint lags, and the lag VARIES --
measured live at 56s, 76s, 313s and 336s within one 60-second span, while
/ticker stayed under 1.2s. latest_price(candles) returns the close of the
newest AVAILABLE bar, so it was feeding the model a price 1-6 minutes old.
Two dashboard instances polling seconds apart landed on different "newest
bar" values and rendered OPPOSITE trade recommendations for the same market
at the same moment (P(YES) 72.2% "buy YES" vs 41.7% "buy NO"). Near an
at-the-money strike this is not a rounding difference: BTC sat $10-30 from
the strike while individual minutes had $15-32 ranges, so a stale price
routinely lands on the wrong SIDE of the strike and flips the call.

It was also a train/serve skew. research/strike_probability/scripts/
build_features.py takes price = the close of the bar AT the evaluation
minute -- zero lag by construction. Live inference was handing the model a
price from minutes earlier, a regime it was never validated on. Using the
ticker restores the convention the model was actually fitted under.

Volatility deliberately still comes from candles: it's an aggregate over a
lookback window, where a minute of lag barely moves the estimate, unlike a
spot-vs-strike comparison.
"""

import time
from collections import namedtuple
from datetime import datetime, timezone

import requests

import config

Candle = namedtuple("Candle", "ts low high open close volume")
# ts: when the price was actually observed (UTC). source: "ticker" (real-time)
# or "candle_fallback" (stale -- ticker was unreachable). Callers should SHOW
# the age rather than implying freshness; see the dashboard's "Last updated".
Spot = namedtuple("Spot", "price ts source")

_session = requests.Session()
_session.headers["User-Agent"] = "kalshi-15min-intel-bot/1.0 (personal use)"


def fetch_spot_price(candles: list["Candle"] | None = None) -> Spot:
    """Real-time BTC last-trade price (sub-second lag in practice).

    Falls back to the newest candle close if /ticker is unreachable, rather
    than raising -- a stale price flagged as stale is more useful to the
    caller than a hard failure mid-window. Pass `candles` if you already have
    them so the fallback costs no extra request. The fallback is genuinely
    stale (that is the whole point of this module's docstring), so anything
    user-facing must surface Spot.ts / Spot.source instead of implying the
    number is current.
    """
    try:
        resp = _session.get(
            f"{config.COINBASE_BASE}/products/{config.COINBASE_PRODUCT}/ticker",
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        ts = datetime.fromisoformat(payload["time"].replace("Z", "+00:00"))
        return Spot(float(payload["price"]), ts, "ticker")
    except Exception:
        if candles is None:
            candles = fetch_1min_candles()
        newest = candles[-1]
        return Spot(newest.close, datetime.fromtimestamp(newest.ts, timezone.utc),
                    "candle_fallback")


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

    # Retry transient transport failures. Kalshi's request paths were hardened
    # for exactly this on 2026-08-11, but THIS call was missed, and a single
    # RemoteDisconnected from Coinbase then aborted a whole live_trader cycle
    # (18:00:26 UTC). It sits in the entry section, after exits, so an exit in
    # progress is never affected -- but a lost cycle is still a window the model
    # never got to evaluate, over a failure that clears on the next attempt.
    rows = None
    for attempt in range(4):
        try:
            resp = _session.get(
                f"{config.COINBASE_BASE}/products/{config.COINBASE_PRODUCT}/candles",
                params=params,
                timeout=15,
            )
            if (resp.status_code == 429 or resp.status_code >= 500) and attempt < 3:
                time.sleep(0.4 * (2 ** attempt) + 0.5)
                continue
            resp.raise_for_status()
            rows = resp.json()  # [[time, low, high, open, close, volume], ...]
            break
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 3:
                raise
            time.sleep(0.4 * (2 ** attempt) + 0.5)
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
    """Close of the newest candle. HISTORICAL/BACKTEST USE ONLY.

    Do not use this for a live price -- see the module docstring. In a
    backtest the newest bar IS the evaluation moment, so this is correct
    there; live it can be 1-6 minutes behind. Live callers want
    fetch_spot_price().
    """
    return candles[-1].close
