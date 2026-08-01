"""Four-state alert classifier: DOWNTREND / UPTREND / RANGING / STABLE.

Mirrors the reference bot's model: each state carries Trend Strength,
Momentum, and Volatility labels plus the confluence confidence score.
"""

from dataclasses import dataclass

import config
from indicators import Signals


@dataclass
class MarketState:
    state: str            # UPTREND / DOWNTREND / RANGING / STABLE
    emoji: str
    trend_strength: str   # Strong / Weak / Unclear
    momentum: str         # Bullish / Bearish / Neutral / Mixed
    volatility: str       # High / Moderate / Low
    confidence: int       # 0..7
    confidence_band: str  # high / moderate / low
    direction: int        # +1 / -1 / 0


_MOMENTUM_INDICATORS = {"RSI", "Micro momentum", "Acceleration", "Tick trend"}


def classify(sig: Signals) -> MarketState:
    # Volatility label
    if sig.realized_vol_pct >= config.VOL_HIGH_PCT:
        volatility = "High"
    elif sig.realized_vol_pct >= config.VOL_MODERATE_PCT:
        volatility = "Moderate"
    else:
        volatility = "Low"

    # Trend strength: size of the 15-min move, confirmed by EMA alignment
    move = abs(sig.window_delta_pct)
    ema_aligned = (sig.ema_sep_pct * sig.window_delta_pct) > 0
    if move >= config.TREND_STRONG_PCT and ema_aligned:
        trend_strength = "Strong"
    elif move >= config.TREND_WEAK_PCT:
        trend_strength = "Weak"
    else:
        trend_strength = "Unclear"

    # Momentum label from the momentum-family indicators
    votes = [r.vote for r in sig.results if r.name in _MOMENTUM_INDICATORS]
    bull, bear = sum(v > 0 for v in votes), sum(v < 0 for v in votes)
    if bull and not bear:
        momentum = "Bullish"
    elif bear and not bull:
        momentum = "Bearish"
    elif not bull and not bear:
        momentum = "Neutral"
    else:
        momentum = "Mixed"

    # Confidence band
    if sig.confidence >= config.CONFIDENCE_HIGH:
        band = "high"
    elif sig.confidence >= config.CONFIDENCE_MODERATE:
        band = "moderate"
    else:
        band = "low"

    # Four-state model
    if trend_strength == "Strong" and momentum == "Bullish" and sig.direction > 0:
        state, emoji = "UPTREND", "\U0001F7E2"      # green circle
    elif trend_strength == "Strong" and momentum == "Bearish" and sig.direction < 0:
        state, emoji = "DOWNTREND", "\U0001F534"    # red circle
    elif volatility == "Low" and momentum == "Neutral" and trend_strength != "Strong":
        state, emoji = "STABLE", "\U0001F535"       # blue circle
    else:
        state, emoji = "RANGING", "\U0001F7E1"      # yellow circle

    return MarketState(
        state=state,
        emoji=emoji,
        trend_strength=trend_strength,
        momentum=momentum,
        volatility=volatility,
        confidence=sig.confidence,
        confidence_band=band,
        direction=sig.direction,
    )
