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
    _require_demo()
    sc, body = _delete(f"/portfolio/orders/{order_id}")
    return {"http": sc, "ok": sc in (200, 201, 204), "raw": body}


def get_resting_orders() -> list[dict]:
    """Open orders per the exchange (not the local log)."""
    data = kalshi_auth._request("/portfolio/orders", params={"status": "resting"})
    return data.get("orders", []) or []


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
    exch_tickers = {p.get("ticker") for p in positions if p.get("position")}
    local_tickers = {r["ticker"] for r in local_filled}

    return {
        "exchange_positions": positions,
        "exchange_resting_orders": len(resting),
        "local_rows": len(local),
        "local_filled_or_partial": len(local_filled),
        "local_unresolved": len(local_unknown),
        "positions_not_in_local_log": sorted(exch_tickers - local_tickers),
        "local_fills_without_position": sorted(local_tickers - exch_tickers),
        "in_sync": not (exch_tickers - local_tickers) and not local_unknown,
    }
