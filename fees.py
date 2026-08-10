"""
Kalshi trading fees for KXBTC15M.

=== VERIFIED 2026-08-10 against Kalshi's own API (Phase 0) ===
Previously this file carried a caveat that the 0.07 rate was "a general
published rate, NOT verified against KXBTC15M specifically." That is now
resolved, from the authoritative source rather than help-centre prose:

  GET /series/KXBTC15M          -> fee_type = "quadratic", fee_multiplier = 1
  GET /series/fee_changes?...   -> {"series_fee_change_arr": []}  (none scheduled)

Two things follow, both settled rather than assumed:

1. NO SERIES MULTIPLIER. fee_multiplier = 1, so KXBTC15M pays the standard
   rate -- no crypto or 15-minute surcharge. (KXETH15M, KXBTCD and KXETH all
   report 1 as well.)

2. NO MAKER FEES ON THIS SERIES. The API's fee_type enum is
   {quadratic, quadratic_with_maker_fees, flat}. KXBTC15M is plain
   "quadratic" -- NOT the maker-fee variant. Kalshi's help centre and various
   third-party summaries contradict each other on whether makers pay (one
   says exempt, another says makers pay ~25% of taker); the schema settles it
   for THIS series: resting orders are fee-exempt here.

   This matters for a future limit-order design: crossing the spread (what
   paper_trader.py models today) pays the taker fee below, but posting and
   resting would pay ZERO. That is a real, quantified argument for a
   maker-style execution redesign later -- not acted on here.

=== ROUNDING (the bug this file previously had) ===
The old implementation rounded UP TO THE NEAREST CENT. Kalshi documents
rounding to the nearest $0.0001 (docs.kalshi.com/getting_started/fee_rounding:
"Fee from the fee model, rounded up to the nearest $0.0001"). Cent-rounding
systematically OVER-charged, which was conservative but wrong, and the error
is worst at small order sizes -- precisely the 4-contract size real execution
is scoped for, where a whole-cent ceiling is a large fraction of the raw fee.

Kalshi also documents a separate "rounding fee" component
(net = trade fee + rounding fee - rebate, floored at $0). That component is
not modelled here: it is a sub-$0.0001 balance-precision adjustment and is
not publicly specified. Treat this as the trade fee only.
"""
import math

import config


def kalshi_fee(contracts: float, price_dollars: float,
               fee_multiplier: float = config.KALSHI_FEE_MULTIPLIER) -> float:
    """Taker fee in dollars for `contracts` at `price_dollars` (0..1).

    fee = ceil(rate * multiplier * C * P * (1-P) / 0.0001) * 0.0001

    Quadratic in price: peaks at P=0.50 (maximum uncertainty), goes to zero as
    P approaches 0 or 1. Matches the series' declared fee_type="quadratic".

    Returns the TAKER fee. Resting/maker orders on KXBTC15M are fee-exempt
    (see module docstring) -- this function does not model that case, because
    nothing in this project posts resting orders yet.
    """
    price_dollars = min(max(price_dollars, 0.0), 1.0)
    raw = config.KALSHI_TAKER_FEE_RATE * fee_multiplier * contracts * price_dollars * (1.0 - price_dollars)
    # Round UP to the nearest $0.0001, per Kalshi's documented granularity.
    return math.ceil(raw / 0.0001) * 0.0001
