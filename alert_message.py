"""Builds the Telegram alert message in the reference-bot format."""

import config
from classifier import MarketState
from indicators import Signals
from kalshi_feed import KalshiMarket
from strike_distance import StrikeRead


def _analysis_sentence(state: MarketState, sig: Signals, strike: StrikeRead) -> str:
    """Rule-based 1-2 sentence market summary (no LLM needed)."""
    move = sig.window_delta_pct
    if state.state == "UPTREND":
        s1 = (f"BTC is pushing higher with a {move:+.2f}% move over the last 15 minutes "
              f"and {sig.confidence}/7 indicators aligned bullish.")
    elif state.state == "DOWNTREND":
        s1 = (f"BTC is selling off with a {move:+.2f}% move over the last 15 minutes "
              f"and {sig.confidence}/7 indicators aligned bearish.")
    elif state.state == "STABLE":
        s1 = (f"BTC is drifting with low volatility "
              f"({sig.realized_vol_pct:.3f}%/min) and no directional edge.")
    else:
        s1 = (f"BTC is choppy — {state.momentum.lower()} momentum with "
              f"{state.volatility.lower()} volatility and no clean trend.")

    if strike.available and strike.label != "n/a":
        s2 = (f" Price sits {strike.distance_pct:+.2f}% from the strike with "
              f"{strike.minutes_remaining:.0f}m left ({strike.label.lower()}).")
    else:
        s2 = ""
    return s1 + s2


def _recommendation(state: MarketState, strike: StrikeRead) -> tuple[str, str]:
    """Returns (recommendation, avoid) lines."""
    high_conf = state.confidence_band == "high"
    trending = state.state in ("UPTREND", "DOWNTREND")
    trend_side = "YES" if state.direction > 0 else "NO"

    if strike.final_minutes:
        rec = "Stand down — inside the final minutes of the window."
        avoid = (f"Avoid all entries in the final {config.FINAL_MINUTES_NOISY} minutes "
                 f"of a window; settlement-print noise dominates.")
        return rec, avoid

    if high_conf and trending and strike.available and strike.label.startswith("Favors"):
        favored = strike.label.split()[-1]
        if favored == trend_side:
            rec = (f"Favor {favored} continuation entries — trend, confluence, and "
                   f"strike distance all point the same way.")
        else:
            rec = (f"Strike math favors {favored} but trend points {trend_side} — "
                   f"reduced conviction, treat as a fade setup only.")
        avoid = "Avoid entries against confluence."
    elif high_conf and trending:
        rec = (f"Favor {trend_side} continuation entries on trend + confluence "
               f"(no strike read available).")
        avoid = "Avoid entries against confluence."
    else:
        rec = "Avoid directional entries, wait for confirmation."
        avoid = ("Avoid forcing trades in ranging/stable conditions and avoid "
                 f"entries in the final {config.FINAL_MINUTES_NOISY} minutes of a window.")
    return rec, avoid


def build_message(price: float, market: KalshiMarket | None, sig: Signals,
                  state: MarketState, strike: StrikeRead) -> str:
    if market is not None and market.strike is not None:
        strike_line = (f"\U0001F3AF Active Strike: ${market.strike:,.2f} "
                       f"({strike.minutes_remaining:.0f}m remaining)")
        dist_line = (f"\U0001F4CF Distance to Strike: {strike.distance_pct:+.2f}% "
                     f"({strike.label.lower()})")
    elif market is not None:
        strike_line = (f"\U0001F3AF Active Strike: TBD "
                       f"({market.minutes_remaining():.0f}m remaining)")
        dist_line = "\U0001F4CF Distance to Strike: n/a (strike not yet published)"
    else:
        strike_line = "\U0001F3AF Active Strike: no open 15-min market found"
        dist_line = "\U0001F4CF Distance to Strike: n/a"

    rec, avoid = _recommendation(state, strike)
    analysis = _analysis_sentence(state, sig, strike)
    indicator_lines = "\n".join(
        f"  {'\U0001F7E2' if r.vote > 0 else ('\U0001F534' if r.vote < 0 else '⚪')} "
        f"{r.name}: {r.detail}" for r in sig.results)

    return (
        f"\U0001F4C8 BTC 15M Kalshi Intel \U0001F514\n"
        f"{state.emoji} MARKET ALERT — {state.state}\n"
        f"\U0001F4B0 Current BTC Price: ${price:,.2f}\n"
        f"{strike_line}\n"
        f"{dist_line}\n"
        f"\U0001F4CA Market Conditions:\n"
        f"• Trend Strength: {state.trend_strength}\n"
        f"• Momentum: {state.momentum}\n"
        f"• Volatility: {state.volatility}\n"
        f"• Confidence: {state.confidence}/7 indicators aligned "
        f"({state.confidence_band})\n"
        f"{indicator_lines}\n"
        f"\U0001F9E0 AI Market Analysis:\n{analysis}\n"
        f"✅ Recommendation:\n{rec}\n"
        f"❌ AVOID:\n{avoid}"
    )
