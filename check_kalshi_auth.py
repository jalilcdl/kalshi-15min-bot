"""
Phase 1 verification: does authenticated, READ-ONLY Kalshi access work?

Run this after dropping in a demo API key. It places no orders and cannot --
kalshi_auth is GET-only by construction.

    python check_kalshi_auth.py

Checks, in order, so a failure points at the actual cause:
  1. config -- which environment, are the credentials even present
  2. key file -- parses as an unencrypted RSA PEM
  3. signing -- produces a verifiable RSA-PSS signature offline (no network)
  4. /portfolio/balance   -- the real end-to-end auth test
  5. /portfolio/positions -- what reconciliation would read on startup
  6. /portfolio/fills     -- and what the exchange says you actually paid
"""
import sys
from pathlib import Path

import config
import kalshi_auth
from kalshi_auth import KalshiAuthError


def main():
    ok = True
    print("=" * 68)
    print(f"1. CONFIG")
    print(f"   KALSHI_ENV               {config.KALSHI_ENV}")
    print(f"   base URL                 {config.KALSHI_TRADE_BASE}")
    if config.KALSHI_ENV != "demo":
        print("   *** NOT the demo environment. Phase 1 is read-only, but this "
              "is a real account. ***")
    key_id = config.KALSHI_API_KEY_ID
    print(f"   KALSHI_API_KEY_ID        {(key_id[:6] + '...' + key_id[-4:]) if key_id else '(NOT SET)'}")
    print(f"   KALSHI_PRIVATE_KEY_PATH  {config.KALSHI_PRIVATE_KEY_PATH}")
    if not kalshi_auth.configured():
        print("\n   NOT CONFIGURED. Set KALSHI_API_KEY_ID and save the private key.")
        print("   See the setup steps in the Phase 1 section of README/SETUP.")
        return 1

    print("\n2. PRIVATE KEY")
    try:
        key = kalshi_auth._load_private_key()
        print(f"   parsed OK -- RSA {key.key_size}-bit")
    except KalshiAuthError as exc:
        print(f"   FAILED: {exc}")
        return 1

    print("\n3. SIGNING (offline, verified against the public key)")
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        import base64 as _b64
        ts, method, path = "1700000000000", "GET", "/trade-api/v2/portfolio/balance"
        sig = kalshi_auth._sign(ts, method, path)
        key.public_key().verify(
            _b64.b64decode(sig), f"{ts}{method}{path}".encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256())
        print(f"   signature verifies against its own public key ({len(sig)} b64 chars)")
    except Exception as exc:
        print(f"   FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("\n4. GET /portfolio/balance  (the real auth test)")
    try:
        bal = kalshi_auth.get_balance()
        print(f"   OK -- {bal}")
    except KalshiAuthError as exc:
        print(f"   AUTH FAILED:\n   {exc}")
        return 1
    except Exception as exc:
        print(f"   REQUEST FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("\n5. GET /portfolio/positions  (what startup reconciliation reads)")
    try:
        pos = kalshi_auth.get_positions()
        print(f"   OK -- {len(pos)} unsettled position(s)")
        for p in pos[:5]:
            print(f"      {p.get('ticker')}  position={p.get('position')}  "
                  f"exposure={p.get('market_exposure')}")
        if not pos:
            print("      (none -- expected on a fresh demo account)")
    except Exception as exc:
        print(f"   FAILED: {type(exc).__name__}: {exc}")
        ok = False

    print("\n6. GET /portfolio/fills  (exchange's record of real fees paid)")
    try:
        fills = kalshi_auth.get_fills(limit=10)
        print(f"   OK -- {len(fills)} recent fill(s)")
        for f in fills[:5]:
            print(f"      {f.get('ticker')}  {f.get('side')}  count={f.get('count')}  "
                  f"price={f.get('yes_price')}  fee={f.get('fee_paid')}")
        if not fills:
            print("      (none -- expected until an order is actually placed)")
    except Exception as exc:
        print(f"   FAILED: {type(exc).__name__}: {exc}")
        ok = False

    print("\n" + "=" * 68)
    print("PHASE 1 PASS -- signed read-only access works." if ok else
          "PHASE 1 PARTIAL -- auth works, some reads failed (see above).")
    print("No order capability exists in this codebase yet.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
