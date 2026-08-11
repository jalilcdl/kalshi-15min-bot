"""Central config for the Kalshi 15-min BTC intel bot.

Values can be overridden via environment variables or a .env file in the
project root. Everything here is tuned for 15-minute Kalshi windows
(~3x the lookbacks of the original 5-min Polymarket-style config).
"""

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Tiny .env loader (KEY=VALUE lines, # comments). No dependency needed."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --- Delivery ---------------------------------------------------------------
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID", "")

# --- Data feeds -------------------------------------------------------------
COINBASE_BASE = "https://api.exchange.coinbase.com"
COINBASE_PRODUCT = "BTC-USD"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES_TICKER = _env("KALSHI_SERIES_TICKER", "KXBTC15M")

# --- Authenticated Kalshi access (kalshi_auth.py) ----------------------------
# Read-only for now: balance / positions / fills. NOTHING here places orders.
#
# KALSHI_ENV defaults to "demo" ON PURPOSE. Production access must be an
# explicit, deliberate opt-in -- never something you get by forgetting to set a
# variable. The demo exchange is a separate host with separate credentials; a
# demo key will NOT authenticate against production and vice versa.
KALSHI_ENV = _env("KALSHI_ENV", "demo").strip().lower()
KALSHI_API_BASES = {
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
    "prod": "https://api.elections.kalshi.com/trade-api/v2",
}
KALSHI_TRADE_BASE = KALSHI_API_BASES.get(KALSHI_ENV, KALSHI_API_BASES["demo"])

# Credentials, same env-var pattern as TELEGRAM_BOT_TOKEN. Jalil generates these
# himself in Kalshi's UI; nothing in this repo creates, requests, or stores them.
#   KALSHI_API_KEY_ID        the Key ID string shown next to the generated key
#   KALSHI_PRIVATE_KEY_PATH  path to the downloaded RSA private key (.pem)
# The .pem is gitignored (*.pem). Kalshi displays the private key exactly once
# and cannot re-issue it -- if it is lost, revoke the key and generate a new one.
KALSHI_API_KEY_ID = _env("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = _env("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")

# --- Live execution (live_executor.py) ---------------------------------------
# Hard cap on contracts per real order, enforced in live_executor.place_order()
# on EVERY order. Raised 4 -> 25 on 2026-08-11 at Jalil's request (demo only).
# Phase 0 showed size is economically neutral once fees round correctly, so the
# number itself costs no ROI -- but see LIVE_DAILY_LOSS_CAP: at 25 contracts a
# single losing trade can exceed the whole daily loss cap, which changes what
# that cap means.
LIVE_MAX_CONTRACTS = 25

# Contracts the trader ASKS for on each entry. Kept separate from the cap on
# purpose: the cap is a safety gate that must hold whatever the sizing logic
# does, and a gate that is defined as "whatever we intended to send" cannot
# catch a sizing bug. Entry size was previously min(cap, PAPER_TRADE_SIZE),
# which silently pinned live orders to the PAPER trader's size -- raising the
# cap alone would have sent 10, not 25.
LIVE_TRADE_SIZE = 25

# Daily realized-loss cap for live_trader.py. On breach: NO NEW ENTRIES for the
# rest of the UTC day; open positions still ride and still exit (halting exits
# would strand capital in exactly the scenario the cap exists to protect
# against). Resets at 00:00 UTC.
#
# $20 chosen deliberately, not arbitrarily. Worst case per trade is
# 4 contracts x $1.00 = $4 lost if a position settles worthless, so $20 is ~5
# maximum-loss trades. At the strategy's validated ~75% target-hit rate, five
# max-loss trades in one day is far outside ordinary variance and reads as
# "something is wrong" (model drift, a bad feed, a market regime the model was
# never fitted on) rather than a bad run. It is also ~20% of the funded demo
# balance, so it trips well before the account is meaningfully damaged. Revisit
# once real per-day P&L variance has been observed rather than assumed.
LIVE_DAILY_LOSS_CAP = 20.00

# Kill switch. If this file EXISTS, no new entries are opened -- checked every
# cycle, before anything else. A filesystem check on purpose: it must work when
# the process is wedged, unreachable, or mid-failure, so it depends on nothing
# the program itself controls. Create it to halt, delete it to resume:
#     type nul > STOP_TRADING        (Windows)
#     touch STOP_TRADING             (POSIX)
# Exits are deliberately NOT blocked -- a kill switch that traps you in an open
# position is worse than no kill switch.
LIVE_KILL_SWITCH_FILE = Path(__file__).parent / "STOP_TRADING"



# How many 1-min bars to keep in memory / request per refresh.
# 300 bars = 5 hours: enough for EMA63 warmup plus volume baselines.
LOOKBACK_BARS = 300

# --- Polling / alert cadence ------------------------------------------------
POLL_SECONDS = 60           # evaluate once per 1-min bar
MIN_ALERT_GAP_SECONDS = 300  # never push more often than every 5 min...
# ...except on a state change or a new 15-min window, which always alert.

# --- Indicator tuning (15-min retune: ~3x the 5-min lookbacks) ---------------
EMA_FAST = 27               # 9 on 3-min bars == 27 on 1-min bars
EMA_SLOW = 63               # 21 on 3-min bars == 63 on 1-min bars
EMA_MIN_SEP_PCT = 0.02      # fast/slow separation below this = no vote

RSI_PERIOD = 14             # computed on resampled 3-min bars
RSI_BULL = 55.0
RSI_BEAR = 45.0

VOLUME_SURGE_RATIO = 1.5    # last 15-min vol vs avg of prior 15-min blocks
VOLUME_BASELINE_BLOCKS = 16  # 16 x 15min = 4h baseline

WINDOW_DELTA_MINUTES = 15   # matches Kalshi expiry window
WINDOW_DELTA_DEADBAND_PCT = 0.05

MOMENTUM_SPAN = 9           # EMA span over 1-min returns (3x the 5-min config)
MOMENTUM_DEADBAND_PCT = 0.01  # per-min smoothed return deadband

ACCEL_LAG = 5               # momentum-now vs momentum-N-bars-ago
ACCEL_DEADBAND_PCT = 0.005

TICK_TREND_BARS = 5         # unchanged from 5-min config (tick proxy)

# --- Classifier thresholds ----------------------------------------------------
# Realized volatility = stdev of 1-min returns over the last 15 min, in %.
VOL_HIGH_PCT = 0.08
VOL_MODERATE_PCT = 0.04

TREND_STRONG_PCT = 0.15     # |15-min move| for a Strong trend
TREND_WEAK_PCT = 0.05       # below this the trend is Unclear/none

CONFIDENCE_HIGH = 6         # 6/7 or 7/7 aligned
CONFIDENCE_MODERATE = 4     # 4-5/7 aligned

# --- Strike-distance module (EXPERIMENTAL — backtest before trusting) --------
STRIKE_SAFE_SIGMAS = 1.5    # |distance| beyond this many expected sigmas =
                            # current side very likely holds
FINAL_MINUTES_NOISY = 2     # avoid entries in the last N minutes of a window

# --- Paper trading (paper_trader.py) -----------------------------------------
# Kalshi taker fee base rate. VERIFIED against KXBTC15M 2026-08-10 (Phase 0) --
# the previous "not verified against KXBTC15M specifically" caveat is resolved.
KALSHI_TAKER_FEE_RATE = 0.07

# Series-specific multiplier, read straight from the API rather than inferred:
#   GET /series/KXBTC15M                            -> fee_type="quadratic", fee_multiplier=1
#   GET /series/fee_changes?series_ticker=KXBTC15M   -> [] (nothing scheduled)
# So no crypto/15-min surcharge. The fee_type enum is
# {quadratic, quadratic_with_maker_fees, flat} and this series is plain
# "quadratic", i.e. resting/maker orders are fee-exempt here. Re-check those two
# endpoints if anything changes; full reasoning in fees.py.
KALSHI_FEE_MULTIPLIER = 1.0

PAPER_TRADE_SIZE = 10        # contracts per simulated fill (flat, not confidence-scaled --
                            # same "flat stake" ethos as the sports models' bet trackers)
PAPER_TRADE_MIN_EDGE = 0.03  # minimum model-implied edge net of fees (in price-dollars,
                            # e.g. 0.03 = 3 cents) required to paper-enter a trade. A
                            # starting judgment call, not derived from data (the walk-forward
                            # validation didn't have historical bid/ask to backtest an entry
                            # rule against) -- revisit once paper-trading data accumulates.
PAPER_TRADE_POLL_SECONDS = 60  # how often paper_trader.py evaluates the active market

# Stop opportunistically entering once this many minutes have elapsed since the
# window opened, even if edge later clears PAPER_TRADE_MIN_EDGE. Added 2026-08-04
# after a user-flagged live trade entered ~5min in on a thin (4.4c) edge -- before
# this, the continuous poll loop would keep retrying and take ANY edge that cleared,
# however late, up to the FINAL_MINUTES_NOISY cutoff. research/exit_timing/README.md
# SS5c found the 30%-target hit rate declines monotonically with entry lateness
# (78.6% at 1min down to 49.8% at 10min) -- that finding was about NOT delaying the
# first look, but the same data applies here: entries the early ~60-120s check missed
# and the continuous loop caught later are, by construction, always the late kind.
# Originally set to 10 (widest option offered); revised down to 5 same-day per
# explicit user decision -- 5 is where SS5c's data starts showing meaningfully worse
# hit rates (checkpoint 5: 60.1% vs checkpoint 1: 78.6%), so the safety net now stops
# right at the point the data flags as the meaningful cliff, not the loosest bound
# tested. FINAL_MINUTES_NOISY (last 2 min of the window) still applies underneath
# this as a separate, stricter guard -- unchanged.
PAPER_TRADE_MAX_ENTRY_MINUTES_ELAPSED = 5

# Minimum distance-over-reachable-move required before the model is trusted at all.
# The model's smoke test caught a real failure mode on its very first live evaluation:
# right at distance_pct ~ 0 (a genuine coin-flip price, indistinguishable from Coinbase
# tick noise), the `current_side_leading` feature flips discretely 0->1 and the model
# output a 71% probability against a real 37c market price -- a much bigger gap than the
# ~6-7 point overconfidence the walk-forward calibration table already showed in the
# 0.6-0.7 predicted bucket. Rather than hand-patch the fitted coefficients, entries are
# gated out below this threshold, same spirit as strike_distance.py's own "too close to
# call" treatment of near-ties.
PAPER_TRADE_MIN_DIST_OVER_REACHABLE = 0.15

# Favorable-move exit target: sell early once the position's value has risen this
# much (e.g. 0.35 = 35%) rather than holding every trade to settlement -- this is
# the user's actual intended strategy (buy early, exit on a quick favorable move),
# which is a different question from "will this settle YES/NO" and was validated
# separately. See research/exit_timing/README.md for the real, walk-forward-style
# backtest this comes from, on 2,790 real historical trades: 35% was the
# best-performing point within the user's stated 20-40% range, but the HONEST
# headline finding there is that simply holding to settlement outperformed EVERY
# tested early-exit threshold in aggregate ROI (12.4% vs 10.9% at best) --
# consistently, though not at full statistical significance (p=0.127). Deployed
# anyway because the point of paper trading is to test the strategy actually
# intended to be run, not the theoretical backtest optimum -- read the full
# writeup before trusting this number with real money.
PAPER_TRADE_EXIT_TARGET = 0.35
