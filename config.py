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
# Kalshi's publicly documented standard taker fee: fee = ceil(0.07 * C * P * (1-P))
# in dollars, C=contracts, P=price in dollars (0..1). This is a general-purpose
# published rate, NOT verified against KXBTC15M specifically -- Kalshi's fee
# schedule can vary by series and change over time. Treat as a documented
# estimate, and cross-check against Kalshi's current fee schedule before
# trusting the paper P&L as a precise real-dollar forecast.
KALSHI_TAKER_FEE_RATE = 0.07

PAPER_TRADE_SIZE = 10        # contracts per simulated fill (flat, not confidence-scaled --
                            # same "flat stake" ethos as the sports models' bet trackers)
PAPER_TRADE_MIN_EDGE = 0.03  # minimum model-implied edge net of fees (in price-dollars,
                            # e.g. 0.03 = 3 cents) required to paper-enter a trade. A
                            # starting judgment call, not derived from data (the walk-forward
                            # validation didn't have historical bid/ask to backtest an entry
                            # rule against) -- revisit once paper-trading data accumulates.
PAPER_TRADE_POLL_SECONDS = 60  # how often paper_trader.py evaluates the active market

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
