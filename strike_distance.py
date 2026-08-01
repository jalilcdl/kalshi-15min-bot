"""Strike-distance module (EXPERIMENTAL — backtest before trusting).

Compares how far BTC sits from the active Kalshi 15-min strike against
what realized volatility says is reachable in the time remaining, then
blends in the confluence direction to emit:
    "Favors YES" / "Favors NO" / "Too close to call"

For KXBTC15M, YES settles if the settlement index at close >= strike
(the strike is the BTC index price at window open). So:
    price above strike  -> YES currently winning
    price below strike  -> NO currently winning
"""

from dataclasses import dataclass
from math import sqrt

import config
from classifier import MarketState
from indicators import Signals
from kalshi_feed import KalshiMarket


@dataclass
class StrikeRead:
    available: bool
    label: str                 # "Favors YES" / "Favors NO" / "Too close to call" / "n/a"
    distance_pct: float
    required_move_per_min: float
    reachable_move_pct: float  # vol-implied reachable move before expiry
    minutes_remaining: float
    final_minutes: bool        # inside the noisy last-N-minutes zone
    note: str


def evaluate(price: float, market: KalshiMarket | None, sig: Signals,
             state: MarketState, minutes_remaining: float | None = None) -> StrikeRead:
    if market is None or market.strike is None:
        return StrikeRead(False, "n/a", 0.0, 0.0, 0.0, 0.0, False,
                          "no active market / strike not yet published")

    mins = market.minutes_remaining() if minutes_remaining is None else minutes_remaining
    distance_pct = (price - market.strike) / market.strike * 100.0
    required_per_min = abs(distance_pct) / mins if mins > 0 else float("inf")

    # Vol-implied reachable move: sigma(1-min returns) scales with sqrt(t)
    reachable = sig.realized_vol_pct * sqrt(max(mins, 0.0))
    winning_side = "YES" if distance_pct >= 0 else "NO"
    losing_side = "NO" if winning_side == "YES" else "YES"

    # 1) Distance is beyond what vol can plausibly close -> current side holds
    if reachable > 0 and abs(distance_pct) >= config.STRIKE_SAFE_SIGMAS * reachable:
        label = f"Favors {winning_side}"
        note = (f"distance is {abs(distance_pct) / reachable:.1f}x the vol-implied "
                f"reachable move ({reachable:.2f}%)")
    # 2) Inside the noise band: lean on momentum + confidence
    elif state.confidence_band == "high" and sig.direction != 0:
        trend_side = "YES" if sig.direction > 0 else "NO"
        if trend_side == winning_side:
            label = f"Favors {winning_side}"
            note = "trend and confluence aligned with the winning side"
        elif required_per_min <= sig.realized_vol_pct:
            # Trend points at the losing side and can plausibly cross in time
            label = f"Favors {losing_side}"
            note = (f"high-confidence trend against current side; needs "
                    f"{required_per_min:.3f}%/min vs {sig.realized_vol_pct:.3f}%/min vol")
        else:
            label = "Too close to call"
            note = "trend opposes current side but likely lacks time to cross"
    else:
        label = "Too close to call"
        note = "distance within vol noise and confluence not decisive"

    return StrikeRead(
        available=True,
        label=label,
        distance_pct=distance_pct,
        required_move_per_min=required_per_min,
        reachable_move_pct=reachable,
        minutes_remaining=mins,
        final_minutes=mins <= config.FINAL_MINUTES_NOISY,
        note=note,
    )
