"""
Authenticated Kalshi access -- READ ONLY.

Phase 1 of the auto-trader scoping (see the Phase 0/1 plan). This module signs
requests and reads account state. It deliberately contains NO order placement,
cancellation, or any other state-changing call. `_request()` hard-rejects any
method other than GET, so adding a write path requires editing this file on
purpose rather than by accident.

=== AUTH MECHANISM ===
Kalshi uses an RSA key pair, not a shared secret -- the private key never
crosses the wire, so a leaked request log cannot be replayed into orders.

Every request carries three headers:
    KALSHI-ACCESS-KEY        the Key ID
    KALSHI-ACCESS-TIMESTAMP  milliseconds since epoch
    KALSHI-ACCESS-SIGNATURE  base64(RSA-PSS-SHA256(timestamp + METHOD + path))

The signed string concatenates the SAME timestamp sent in the header, the
uppercase HTTP method, and the path INCLUDING the /trade-api/v2 prefix but
EXCLUDING the query string, e.g.:
    1703123456789GET/trade-api/v2/portfolio/balance

=== CREDENTIALS -- Jalil supplies these himself ===
Nothing here creates, requests, transmits or stores credentials. Two env vars,
same pattern as TELEGRAM_BOT_TOKEN (see config.py):
    KALSHI_API_KEY_ID        Key ID string from Kalshi's UI
    KALSHI_PRIVATE_KEY_PATH  path to the downloaded .pem (default
                             kalshi_private_key.pem in the repo root, gitignored)
    KALSHI_ENV               "demo" (default) or "prod"

KALSHI_ENV defaults to demo on purpose: production must be an explicit opt-in,
never the consequence of an unset variable. Demo and production are separate
hosts with separate credentials -- a demo key will not authenticate against
production.
"""
import base64
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

import config

_session = requests.Session()
_session.headers["User-Agent"] = "kalshi-15min-intel-bot/1.0 (personal use)"

_private_key = None


class KalshiTransportError(RuntimeError):
    """Network failed repeatedly. Distinct from KalshiAuthError so a caller can
    tell 'the exchange is unreachable' from 'your credentials are wrong' -- the
    first is worth retrying, the second never is."""


class KalshiAuthError(RuntimeError):
    """Credential or signing problem -- distinct from an HTTP/API failure so
    callers can tell 'you are not set up' from 'the request failed'."""


def configured() -> bool:
    """True if both a key id and a readable key file are present. Mirrors
    telegram_client.configured() so callers can degrade gracefully."""
    if not config.KALSHI_API_KEY_ID:
        return False
    return Path(config.KALSHI_PRIVATE_KEY_PATH).expanduser().is_file()


def _load_private_key():
    global _private_key
    if _private_key is not None:
        return _private_key
    if not config.KALSHI_API_KEY_ID:
        raise KalshiAuthError(
            "KALSHI_API_KEY_ID is not set. Generate an API key in Kalshi's UI "
            "(demo: demo.kalshi.co) and put the Key ID in your .env."
        )
    path = Path(config.KALSHI_PRIVATE_KEY_PATH).expanduser()
    if not path.is_file():
        raise KalshiAuthError(
            f"private key not found at {path}. Save the RSA private key Kalshi "
            "showed you at generation time to that path (it is gitignored), or "
            "point KALSHI_PRIVATE_KEY_PATH somewhere else."
        )
    data = path.read_bytes()
    try:
        _private_key = serialization.load_pem_private_key(data, password=None)
    except Exception as exc:
        raise KalshiAuthError(
            f"could not parse {path} as an unencrypted PEM private key ({exc}). "
            "Kalshi issues an RSA key in PEM format -- paste it whole, including "
            "the -----BEGIN...----- and -----END...----- lines."
        ) from exc
    return _private_key


def _sign(timestamp_ms: str, method: str, path: str) -> str:
    """base64 RSA-PSS/SHA256 over timestamp+METHOD+path (no query string)."""
    key = _load_private_key()
    message = f"{timestamp_ms}{method.upper()}{path}".encode()
    signature = key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode()


def _request(path: str, params: dict | None = None, timeout: int = 20,
             retries: int = 4) -> dict:
    """Signed GET against the configured environment.

    GET ONLY -- this module is read-only by construction. Retries 429 and 5xx
    with backoff; surfaces 401 immediately, since a signing/credential problem
    will not fix itself by retrying.
    """
    base = config.KALSHI_TRADE_BASE
    # The signature covers the path WITH the /trade-api/v2 prefix and WITHOUT
    # the query string. Derive it with urlparse rather than string-splitting on
    # the host: splitting on "kalshi.co" also matches inside "kalshi.com" and
    # silently yields "m/trade-api/v2" for production, which signs correctly
    # against the wrong path and 401s with no obvious cause.
    full_path = f"{urlparse(base).path}{path}"

    for attempt in range(retries):
        ts = str(int(time.time() * 1000))
        headers = {
            "KALSHI-ACCESS-KEY": config.KALSHI_API_KEY_ID,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": _sign(ts, "GET", full_path),
        }
        # Transport errors must be retried too. Previously only HTTP status codes
        # (429/5xx) were retried, so a ConnectionError raised by the socket layer
        # propagated straight out and killed the whole cycle. Measured 10
        # RemoteDisconnected errors in one session -- and a cycle lost mid-exit is
        # precisely when losing a cycle costs the most, because the position stays
        # open a further 60s while the price moves.
        try:
            resp = _session.get(base + path, params=params, headers=headers,
                                timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt < retries - 1:
                time.sleep(0.4 * (2 ** attempt) + 0.5)
                continue
            raise KalshiTransportError(
                f"GET {path} failed after {retries} attempts: {exc}") from exc

        if resp.status_code == 401:
            raise KalshiAuthError(
                f"401 from {config.KALSHI_ENV} for GET {path}: {resp.text[:200]}\n"
                "Usual causes: Key ID and private key are from different keys; "
                "the key belongs to the other environment (demo keys do not work "
                "against production or vice versa); or local clock skew is large "
                "enough that the signed timestamp is rejected."
            )
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt < retries - 1:
                time.sleep(0.4 * (2 ** attempt) + 0.5)
                continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return {}


# --- read-only endpoints -----------------------------------------------------

def get_balance() -> dict:
    """Account balance. Amounts come back in cents on most Kalshi fields."""
    return _request("/portfolio/balance")


def get_positions(settlement_status: str = "unsettled") -> list[dict]:
    """Current positions. Defaults to unsettled -- the ones that still matter.

    THIS is the call a real trader would reconcile against on startup. Local
    state must never be the only record of whether a position exists: if the
    process dies between placing an order and writing it down, the exchange
    knows and the local file does not.
    """
    data = _request("/portfolio/positions", params={"settlement_status": settlement_status})
    return data.get("market_positions", []) or []


def get_fills(limit: int = 100, ticker: str | None = None) -> list[dict]:
    """Recent fills -- the exchange's record of what actually executed, at what
    price, and what fee was actually charged. The fee field here is the ground
    truth to check fees.py against once any real order exists."""
    params: dict = {"limit": limit}
    if ticker:
        params["ticker"] = ticker
    return _request("/portfolio/fills", params=params).get("fills", []) or []


def whoami() -> dict:
    """Cheap end-to-end auth check: signs a request and confirms the exchange
    accepts it. Returns the balance payload plus the environment used."""
    bal = get_balance()
    return {"env": config.KALSHI_ENV, "base": config.KALSHI_TRADE_BASE,
            "key_id": (config.KALSHI_API_KEY_ID[:6] + "...") if config.KALSHI_API_KEY_ID else "",
            "balance": bal}
