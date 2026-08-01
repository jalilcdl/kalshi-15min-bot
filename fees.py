"""
Kalshi's publicly documented standard taker fee formula. See the caveat in
config.py (KALSHI_TAKER_FEE_RATE) -- this is a general published rate, not
verified against KXBTC15M specifically.
"""
import math

import config


def kalshi_fee(contracts: int, price_dollars: float) -> float:
    """fee = ceil(rate * C * P * (1-P) * 100) / 100, in dollars.
    Peaks at P=0.50 (max uncertainty), goes to 0 as P -> 0 or 1.
    """
    price_dollars = min(max(price_dollars, 0.0), 1.0)
    raw_cents = config.KALSHI_TAKER_FEE_RATE * contracts * price_dollars * (1.0 - price_dollars) * 100.0
    return math.ceil(raw_cents) / 100.0
