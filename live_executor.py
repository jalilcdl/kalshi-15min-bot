"""
REAL order execution against Kalshi. Phase 2: DEMO ENVIRONMENT ONLY.

=== HOW THIS DIFFERS FROM paper_trader.py ===
paper_trader.py simulates fills against quotes and writes data/trade_log.csv.
It never contacts an exchange. THIS module sends real orders to a real matching
engine. The two share NO state, NO log file, and NO code path:

    paper_trader.py  -> data/trade_log.csv        mode="paper"   simulated
    live_executor.py -> data/live_orders.csv      real orders    live_executor.log

That separation is deliberate and load-bearing. This project has already shipped
a bug where simulated activity was described in words ("Trade confirmed") that
read as real execution. Keeping the files, the schemas and the vocabulary
disjoint means a real order can never be mistaken for a simulated one by
looking at the wrong file.

=== SAFETY PROPERTIES ===
1. DEMO-ONLY, ENFORCED IN CODE. Every write call checks KALSHI_ENV=="demo" and
   raises otherwise. Phase 2 has no business touching production, and honouring
   that by convention alone is exactly how accidents happen.
2. SIZE CAP. LIVE_MAX_CONTRACTS (4) is enforced on every order.
3. WRITE-AHEAD IDEMPOTENCY. The client_order_id is generated and written to disk
   BEFORE the request is sent. If the process dies mid-flight, or the response
   is lost to a timeout, the retry reuses the same id and the exchange dedupes
   instead of double-filling. A uuid generated after a failure is a new order.
4. EXCHANGE IS THE TRUTH. reconcile() compares the local log against
   /portfolio/positions and reports drift. Local state is never trusted alone.

kalshi_auth.py stays GET-only; the signed POST lives here, so the read module
keeps its "cannot write" property.
"""
import csv
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

import config
import kalshi_auth

ORDERS_CSV = Path(__file__).parent / "data" / "live_orders.csv"
ORDER_PATH = "/portfolio/events/orders"

COLUMNS = ["ts_utc", "client_order_id", "order_id", "ticker", "side", "count",
           "price", "time_in_force", "status", "fill_count", "remaining_count",
           "avg_fill_price", "fee_paid", "env", "note"]

_session = requests.Session()
_session.headers["User-Agent"] = "kalshi-15min-intel-bot/1.0 (personal use)"


class LiveExecError(RuntimeError):
    pass


class EnvironmentGuardError(LiveExecError):
    """Raised when a write is attempted outside the demo environment."""


def _require_demo():
    if config.KALSHI_ENV != "demo":
        raise EnvironmentGuardError(
            f"live_executor is DEMO-ONLY in Phase 2, but KALSHI_ENV={config.KALSHI_ENV!r}. "
            "Refusing to send an order. Production execution is a later phase with "
            "loss caps, a kill switch and supervised rollout -- none of which exist yet."
        )


def _stamp():
    return datetime.now(timezone.utc).isoformat()


# --- local order log (separate from the paper trade log) ---------------------

def _ensure_csv():
    ORDERS_CSV.parent.mkdir(exist_ok=True)
    if not ORDERS_CSV.exists():
        with ORDERS_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLUMNS)


def _append(row: dict):
    _ensure_csv()
    with ORDERS_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(c, "") for c in COLUMNS])


def _update(client_order_id: str, **fields):
    """Rewrite the row for this client_order_id. Small file, whole-file rewrite
    is fine and keeps the write atomic enough for a single-process caller."""
    _ensure_csv()
    rows = list(csv.DictReader(ORDERS_CSV.open(encoding="utf-8")))
    for r in rows:
        if r["client_order_id"] == client_order_id:
            r.update({k: ("" if v is None else v) for k, v in fields.items()})
    tmp = ORDERS_CSV.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    tmp.replace(ORDERS_CSV)


def load_orders() -> list[dict]:
    if not ORDERS_CSV.exists():
        return []
    return list(csv.DictReader(ORDERS_CSV.open(encoding="utf-8")))


def find_by_client_id(client_order_id: str) -> dict | None:
    for r in load_orders():
        if r["client_order_id"] == client_order_id:
            return r
    return None


# --- signed POST (kalshi_auth stays GET-only) --------------------------------

def _post(path: str, payload: dict, timeout: int = 20) -> tuple[int, dict]:
    base = config.KALSHI_TRADE_BASE
    full_path = f"{urlparse(base).path}{path}"
    ts = str(int(time.time() * 1000))
    headers = {
        "KALSHI-ACCESS-KEY": config.KALSHI_API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": kalshi_auth._sign(ts, "POST", full_path),
        "Content-Type": "application/json",
    }
    resp = _session.post(base + path, headers=headers, json=payload, timeout=timeout)
    try:
        body = resp.json()
    except ValueError:
        body = {"raw": resp.text[:400]}
    return resp.status_code, body


def _delete(path: str, timeout: int = 20) -> tuple[int, dict]:
    base = config.KALSHI_TRADE_BASE
    full_path = f"{urlparse(base).path}{path}"
    ts = str(int(time.time() * 1000))
    headers = {
        "KALSHI-ACCESS-KEY": config.KALSHI_API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": kalshi_auth._sign(ts, "DELETE", full_path),
    }
    resp = _session.delete(base + path, headers=headers, timeout=timeout)
    try:
        body = resp.json()
    except ValueError:
        body = {"raw": resp.text[:400]}
    return resp.status_code, body


# --- orders ------------------------------------------------------------------

def place_order(ticker: str, side: str, count: int, price: float,
                time_in_force: str = "immediate_or_cancel",
                client_order_id: str | None = None,
                post_only: bool = False, note: str = "") -> dict:
    """Send one order. side is 'bid' (buy YES) or 'ask' (buy NO / sell YES).

    Returns a dict with status in {filled, partial, no_fill, resting, rejected,
    unknown}. "unknown" means the request left but the outcome was not
    observed -- treat it as possibly-live and reconcile, never as failed.
    """
    _require_demo()
    if side not in ("bid", "ask"):
        raise LiveExecError(f"side must be 'bid' or 'ask', got {side!r}")
    if not (0 < count <= config.LIVE_MAX_CONTRACTS):
        raise LiveExecError(
            f"count {count} outside 1..{config.LIVE_MAX_CONTRACTS} "
            f"(config.LIVE_MAX_CONTRACTS)")
    if not (0.0 < price < 1.0):
        raise LiveExecError(f"price {price} must be strictly between 0 and 1")

    # WRITE-AHEAD: persist the id before the request exists on the wire, so a
    # crash or timeout can be retried with the SAME id and deduped by Kalshi.
    cid = client_order_id or str(uuid.uuid4())
    if find_by_client_id(cid) is None:
        _append({"ts_utc": _stamp(), "client_order_id": cid, "ticker": ticker,
                 "side": side, "count": count, "price": f"{price:.4f}",
                 "time_in_force": time_in_force, "status": "sending",
                 "env": config.KALSHI_ENV, "note": note})

    payload = {
        "ticker": ticker, "side": side,
        "count": f"{count:.2f}", "price": f"{price:.4f}",
        "time_in_force": time_in_force,
        "self_trade_prevention_type": "taker_at_cross",
        "client_order_id": cid,
        "post_only": post_only,
    }

    try:
        sc, body = _post(ORDER_PATH, payload)
    except Exception as exc:
        # Request may or may not have reached the matching engine. This is the
        # dangerous case: do NOT mark it failed.
        _update(cid, status="unknown", note=f"{note} transport error: {exc}".strip())
        return {"status": "unknown", "client_order_id": cid, "error": str(exc)}

    if sc not in (200, 201):
        err = body.get("error", body)
        _update(cid, status="rejected", note=f"{note} HTTP {sc}: {json.dumps(err)[:180]}".strip())
        return {"status": "rejected", "http": sc, "client_order_id": cid, "error": err}

    fill = float(body.get("fill_count", 0) or 0)
    remaining = float(body.get("remaining_count", 0) or 0)
    if fill >= count:
        status = "filled"
    elif fill > 0:
        status = "partial"
    elif remaining > 0:
        status = "resting"
    else:
        status = "no_fill"        # IOC/FOK that matched nothing and was cancelled

    _update(cid, order_id=body.get("order_id", ""), status=status,
            fill_count=body.get("fill_count", ""),
            remaining_count=body.get("remaining_count", ""),
            avg_fill_price=body.get("average_fill_price", ""),
            fee_paid=body.get("average_fee_paid", ""))
    return {"status": status, "http": sc, "client_order_id": cid,
            "order_id": body.get("order_id"), "fill_count": fill,
            "remaining_count": remaining,
            "avg_fill_price": body.get("average_fill_price"),
            "fee_paid": body.get("average_fee_paid"), "raw": body}


def cancel_order(order_id: str) -> dict:
    """Cancel a resting order.

    Path is the V2 one. DELETE /portfolio/orders/{id} is the v1 endpoint and
    returns 410 deprecated_v1_order_endpoint -- found the hard way in Phase 2b,
    where a resting order could not be cancelled and was left on the book.
    """
    _require_demo()
    sc, body = _delete(f"{ORDER_PATH}/{order_id}")
    return {"http": sc, "ok": sc in (200, 201, 204), "raw": body}


def get_resting_orders() -> list[dict]:
    """Open orders per the exchange (not the local log)."""
    data = kalshi_auth._request("/portfolio/orders", params={"status": "resting"})
    return data.get("orders", []) or []


def refresh_order_status(client_order_id: str) -> dict | None:
    """Re-read a resting order from the exchange and update the local row.

    place_order() classifies from the IMMEDIATE response, which is terminal for
    IOC/FOK but NOT for good_till_canceled: a resting order can fill later and
    the local status then goes stale. Observed directly in Phase 2b -- a resting
    sell of 4 sat at fill=0 for 24s, then the market maker lifted 1, leaving the
    local log still saying "resting" while the exchange said 1 of 4 filled.

    Anything long-lived that rests orders must call this (or reconcile()) rather
    than trusting what place_order returned at submit time.
    """
    row = find_by_client_id(client_order_id)
    if not row or not row.get("order_id"):
        return None
    for o in get_resting_orders():
        if o.get("order_id") != row["order_id"]:
            continue
        filled = float(o.get("fill_count_fp") or 0)
        remaining = float(o.get("remaining_count_fp") or 0)
        want = float(row.get("count") or 0)
        status = ("filled" if filled >= want > 0 else
                  "partial" if filled > 0 else "resting")
        _update(client_order_id, status=status, fill_count=f"{filled:.2f}",
                remaining_count=f"{remaining:.2f}")
        return {"status": status, "fill_count": filled, "remaining_count": remaining}
    # Not resting any more: it either filled completely or was cancelled. The
    # fills endpoint is authoritative for which.
    fills = kalshi_auth.get_fills(limit=50, ticker=row["ticker"])
    matched = sum(float(f.get("count_fp") or 0)
                  for f in fills if f.get("order_id") == row["order_id"])
    if matched > 0:
        want = float(row.get("count") or 0)
        _update(client_order_id, status="filled" if matched >= want else "partial",
                fill_count=f"{matched:.2f}", remaining_count="0.00")
        return {"status": "filled" if matched >= want else "partial",
                "fill_count": matched, "remaining_count": 0.0}
    _update(client_order_id, status="cancelled", remaining_count="0.00")
    return {"status": "cancelled", "fill_count": 0.0, "remaining_count": 0.0}


# --- reconciliation ----------------------------------------------------------

def reconcile() -> dict:
    """Compare the local order log against the exchange's own view.

    The exchange is authoritative. Any position it reports that the local log
    does not explain is drift -- most likely an order that filled after the
    local process lost track of it, which is precisely the failure that makes
    naive retry logic double up.
    """
    positions = kalshi_auth.get_positions()
    resting = get_resting_orders()
    local = load_orders()

    local_filled = [r for r in local if r["status"] in ("filled", "partial")]
    local_unknown = [r for r in local if r["status"] in ("unknown", "sending")]
    # Field names are the API's real ones, confirmed against a live payload in
    # Phase 2b: position_fp / market_exposure_dollars / fees_paid_dollars.
    # An earlier guess at "position"/"market_exposure" silently read None for
    # every field, which made a real 4-contract position look like no position
    # at all -- exactly the drift reconciliation exists to catch.
    exch_tickers = {p.get("ticker") for p in positions
                    if float(p.get("position_fp") or 0) != 0}
    local_tickers = {r["ticker"] for r in local_filled}

    # Drift in EITHER direction is drift. An exchange position the log cannot
    # explain is the dangerous one (an order filled that we lost track of), but
    # a logged fill with no position is also wrong unless it settled, so both
    # are surfaced and both clear in_sync.
    orphan_positions = sorted(exch_tickers - local_tickers)
    # A logged fill with no open position is EXPECTED once that market settles:
    # get_positions() asks for unsettled only, so settled markets drop out. Only
    # count it as drift if the market is still open. Without this the reconciler
    # cries drift on every trade that reaches expiry, which would train whoever
    # reads it to ignore the one signal that matters.
    orphan_fills = []
    for tkr in sorted(local_tickers - exch_tickers):
        # A logged fill with no open position is EXPECTED in two ordinary cases:
        #   1. the market settled (get_positions asks for unsettled only), and
        #   2. we entered AND closed it -- a completed round trip.
        # Case 2 was missing and made reconcile() report DRIFT on every
        # successful round trip while the market was still open. A reconciler
        # that cries wolf on normal success trains whoever reads it to ignore
        # the one signal that matters, which is worse than no reconciler.
        try:
            net = sum(float(f.get("count_fp") or 0) *
                      (1 if f.get("action") == "buy" else -1)
                      for f in kalshi_auth.get_fills(limit=100, ticker=tkr))
            if abs(net) < 1e-9:
                continue                      # flat by our own fills: not drift
        except Exception:
            pass
        try:
            m = requests.get(f"{config.KALSHI_TRADE_BASE}/markets/{tkr}",
                             timeout=15).json().get("market", {})
            if m.get("status") in ("active", "initialized"):
                orphan_fills.append(tkr)
        except Exception:
            orphan_fills.append(tkr)   # can't prove it settled -> treat as drift
    return {
        "exchange_positions": positions,
        "exchange_resting_orders": len(resting),
        "local_rows": len(local),
        "local_filled_or_partial": len(local_filled),
        "local_unresolved": len(local_unknown),
        "positions_not_in_local_log": orphan_positions,
        "local_fills_without_position": orphan_fills,
        "in_sync": not orphan_positions and not orphan_fills and not local_unknown,
    }


def position_summary() -> list[dict]:
    """Positions with the API's real field names decoded to plain numbers."""
    out = []
    for p in kalshi_auth.get_positions():
        out.append({
            "ticker": p.get("ticker"),
            "contracts": float(p.get("position_fp") or 0),
            "exposure": float(p.get("market_exposure_dollars") or 0),
            "fees_paid": float(p.get("fees_paid_dollars") or 0),
            "realized_pnl": float(p.get("realized_pnl_dollars") or 0),
        })
    return out
