"""Kalshi public market-data feed for the 15-min BTC up/down series (KXBTC15M).

Market mechanics (verified against the live API):
  - Each 15-min window is one event; the market's floor_strike is the BTC
    index price (CF Benchmarks BRTI 60s average) captured at window open.
  - strike_type is greater_or_equal: YES settles if the BRTI 60s average
    just before close_time is >= floor_strike.
  - No auth needed for market data.

Note: early in a window floor_strike can be unset ("Target price: TBD")
until the open-print average is published; we return None for strike then.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

import config


@dataclass
class Candlestick:
    ts: int              # end_period_ts (unix seconds)
    price_close: float | None
    yes_bid_close: float | None
    yes_bid_high: float | None
    yes_bid_low: float | None
    yes_ask_close: float | None
    yes_ask_high: float | None
    yes_ask_low: float | None
    volume: float

_session = requests.Session()
_session.headers["User-Agent"] = "kalshi-15min-intel-bot/1.0 (personal use)"


@dataclass
class KalshiMarket:
    ticker: str
    event_ticker: str
    strike: float | None      # floor_strike; None until Kalshi publishes it
    open_time: datetime
    close_time: datetime
    yes_bid: float | None     # dollars, 0..1
    yes_ask: float | None
    result: str               # "" while open; "yes"/"no" once settled

    def minutes_remaining(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return max(0.0, (self.close_time - now).total_seconds() / 60.0)


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _to_market(m: dict) -> KalshiMarket:
    def _f(key: str) -> float | None:
        v = m.get(key)
        try:
            return float(v) if v not in (None, "") else None
        except (TypeError, ValueError):
            return None

    return KalshiMarket(
        ticker=m["ticker"],
        event_ticker=m.get("event_ticker", ""),
        strike=_f("floor_strike"),
        open_time=_parse_ts(m["open_time"]),
        close_time=_parse_ts(m["close_time"]),
        yes_bid=_f("yes_bid_dollars"),
        yes_ask=_f("yes_ask_dollars"),
        result=m.get("result", ""),
    )


def _get(url: str, params: dict | None = None, timeout: int = 15,
         retries: int = 4):
    """GET with retry on transient failures. Use this for every call in this
    module rather than _session.get directly.

    Written after a SWEEP rather than after an incident. Four separate retry
    fixes shipped on 2026-08-11 -- kalshi_auth._request, live_trader.
    market_quote, live_executor._post and then _delete -- each one added only
    after that specific call failed in production. Auditing every outbound call
    at once then found get_active_market() still bare, which is the most-called
    network function in the live loop: live_trader hits it twice per cycle, and
    an unhandled ConnectionError there aborts the whole cycle.
    """
    for attempt in range(retries):
        try:
            resp = _session.get(url, params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries - 1:
                raise
            time.sleep(0.4 * (2 ** attempt) + 0.5)
            continue
        if (resp.status_code == 429 or resp.status_code >= 500) and attempt < retries - 1:
            time.sleep(0.4 * (2 ** attempt) + 0.5)
            continue
        return resp
    return resp


def get_market_by_ticker(ticker: str) -> KalshiMarket | None:
    """Look up a single market by ticker -- used to resolve individual pending
    paper trades without re-pulling a whole settled-markets range each time."""
    resp = _get(f"{config.KALSHI_BASE}/markets/{ticker}", timeout=15)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json().get("market")
    return _to_market(data) if data else None


def get_candlesticks(ticker: str, start_ts: int, end_ts: int,
                     series_ticker: str = config.KALSHI_SERIES_TICKER,
                     period_interval: int = 1, retries: int = 5,
                     base_delay: float = 0.3) -> list[Candlestick]:
    """Real historical per-minute price/bid/ask candlesticks for one market
    (confirmed retained for at least the 45-day depth this project uses).
    Used to backtest actual intra-window price-path behavior (e.g. "does the
    contract move 20-40% in the buyer's favor before close"), which the
    settlement-only data (KalshiMarket.yes_bid/yes_ask, a single snapshot at
    fetch time) cannot answer.
    """
    import time as _time
    url = f"{config.KALSHI_BASE}/series/{series_ticker}/markets/{ticker}/candlesticks"
    params = {"period_interval": period_interval, "start_ts": start_ts, "end_ts": end_ts}
    for attempt in range(retries):
        resp = _session.get(url, params=params, timeout=20)
        if resp.status_code == 429:
            _time.sleep(base_delay * (2 ** attempt) + 1.0)
            continue
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        break
    else:
        resp.raise_for_status()

    out = []
    for c in resp.json().get("candlesticks", []):
        price = c.get("price", {})
        yb = c.get("yes_bid", {})
        ya = c.get("yes_ask", {})

        def _f(d, key):
            v = d.get(key)
            try:
                return float(v) if v not in (None, "") else None
            except (TypeError, ValueError):
                return None

        out.append(Candlestick(
            ts=int(c["end_period_ts"]),
            price_close=_f(price, "close_dollars"),
            yes_bid_close=_f(yb, "close_dollars"), yes_bid_high=_f(yb, "high_dollars"), yes_bid_low=_f(yb, "low_dollars"),
            yes_ask_close=_f(ya, "close_dollars"), yes_ask_high=_f(ya, "high_dollars"), yes_ask_low=_f(ya, "low_dollars"),
            volume=float(c.get("volume_fp", 0.0) or 0.0),
        ))
    return sorted(out, key=lambda c: c.ts)


def get_active_market() -> KalshiMarket | None:
    """Return the currently-trading 15-min BTC market (nearest close), or None."""
    resp = _get(
        f"{config.KALSHI_BASE}/markets",
        params={"series_ticker": config.KALSHI_SERIES_TICKER,
                "status": "open", "limit": 20},
        timeout=15,
    )
    resp.raise_for_status()
    markets = [_to_market(m) for m in resp.json().get("markets", [])]
    if not markets:
        return None
    now = datetime.now(timezone.utc)
    live = [m for m in markets if m.close_time > now]
    if not live:
        return None
    return min(live, key=lambda m: m.close_time)


def get_settled_markets(min_close_ts: int, max_close_ts: int,
                        max_pages: int = 50, page_delay: float = 0.25) -> list[KalshiMarket]:
    """Fetch settled 15-min BTC markets in a close-time range (for backtesting).

    page_delay: seconds to sleep between pages, and the base for 429 backoff.
    A multi-week pull is dozens of pages; a small delay avoids tripping
    Kalshi's rate limit, which a burst of back-to-back requests will hit.
    """
    import time as _time
    out: list[KalshiMarket] = []
    cursor = None
    for page in range(max_pages):
        params: dict = {"series_ticker": config.KALSHI_SERIES_TICKER,
                        "status": "settled", "limit": 200,
                        "min_close_ts": min_close_ts,
                        "max_close_ts": max_close_ts}
        if cursor:
            params["cursor"] = cursor

        for attempt in range(5):
            resp = _session.get(f"{config.KALSHI_BASE}/markets", params=params, timeout=20)
            if resp.status_code == 429:
                _time.sleep(page_delay * (2 ** attempt) + 1.0)
                continue
            resp.raise_for_status()
            break
        else:
            resp.raise_for_status()  # exhausted retries -- surface the last error

        data = resp.json()
        out.extend(_to_market(m) for m in data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor:
            break
        if page_delay:
            _time.sleep(page_delay)
    return [m for m in out if m.strike is not None and m.result in ("yes", "no")]
