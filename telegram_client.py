"""Telegram delivery. Falls back to console printing when no token is set,
so the bot can be dry-run before wiring up credentials."""

import requests

import config

_session = requests.Session()


def configured() -> bool:
    return bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID)


def send(text: str) -> bool:
    """Send a message; returns True on success. Prints to console in dry-run."""
    if not configured():
        print("\n--- DRY RUN (no TELEGRAM_BOT_TOKEN/CHAT_ID set) ---")
        print(text)
        print("--- END ---\n", flush=True)
        return True
    try:
        resp = _session.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text,
                  "disable_web_page_preview": True},
            timeout=15,
        )
        ok = resp.status_code == 200 and resp.json().get("ok", False)
        if not ok:
            print(f"[telegram] send failed: {resp.status_code} {resp.text[:200]}",
                  flush=True)
        return ok
    except requests.RequestException as e:
        print(f"[telegram] send error: {e}", flush=True)
        return False
