"""7-indicator confluence engine, retuned for 15-minute Kalshi windows.

Each indicator emits a vote: +1 (bullish), -1 (bearish), or 0 (no signal),
plus a human-readable detail string. The confidence score is how many of
the 7 agree with the majority direction.

Retune vs the 5-min config (~3x lookbacks):
  EMA 9/21 -> 27/63 on 1-min bars
  RSI 14 -> 14 on resampled 3-min bars
  momentum/acceleration smoothing widened 3x
  window delta / volume baseline -> 15-min windows
  tick trend kept as-is (proxied by last-5 1-min closes; swap in a
  WebSocket tick feed later if you want true tick data)
"""

from dataclasses import dataclass
from statistics import pstdev

import config
from coinbase_feed import Candle


@dataclass
class IndicatorResult:
    name: str
    vote: int          # +1 bullish, -1 bearish, 0 neutral/no-signal
    detail: str


@dataclass
class Signals:
    results: list[IndicatorResult]
    direction: int         # +1 / -1 / 0 (majority direction)
    confidence: int        # how many of the 7 agree with the majority
    window_delta_pct: float
    realized_vol_pct: float  # stdev of 1-min returns over last 15 min, in %
    ema_sep_pct: float


def ema(values: list[float], span: int) -> list[float]:
    k = 2.0 / (span + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1.0 - k))
    return out


def _rsi(closes: list[float], period: int) -> float | None:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for a, b in zip(closes[:-1], closes[1:]):
        d = b - a
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    # Wilder smoothing
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _sign_with_deadband(value: float, deadband: float) -> int:
    if value > deadband:
        return 1
    if value < -deadband:
        return -1
    return 0


def compute_signals(candles: list[Candle]) -> Signals:
    """Run all 7 indicators over 1-min candles (oldest first)."""
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]
    results: list[IndicatorResult] = []

    # 1. EMA fast/slow cross (27/63 on 1-min)
    fast = ema(closes, config.EMA_FAST)[-1]
    slow = ema(closes, config.EMA_SLOW)[-1]
    sep_pct = (fast - slow) / slow * 100.0
    vote = _sign_with_deadband(sep_pct, config.EMA_MIN_SEP_PCT)
    results.append(IndicatorResult(
        "EMA cross", vote, f"EMA{config.EMA_FAST}/{config.EMA_SLOW} sep {sep_pct:+.3f}%"))

    # 2. RSI(14) on resampled 3-min bars
    closes_3m = closes[::-1][::3][::-1]  # every 3rd close, anchored to newest
    rsi = _rsi(closes_3m, config.RSI_PERIOD)
    if rsi is None:
        vote = 0
        detail = "RSI: insufficient data"
    else:
        vote = 1 if rsi >= config.RSI_BULL else (-1 if rsi <= config.RSI_BEAR else 0)
        detail = f"RSI(14, 3m) {rsi:.1f}"
    results.append(IndicatorResult("RSI", vote, detail))

    # 3. Volume surge vs 15-min baseline (direction from concurrent price move)
    w = config.WINDOW_DELTA_MINUTES
    vol_now = sum(volumes[-w:])
    blocks = []
    for i in range(config.VOLUME_BASELINE_BLOCKS):
        seg = volumes[-(i + 2) * w: -(i + 1) * w]
        if len(seg) == w:
            blocks.append(sum(seg))
    baseline = sum(blocks) / len(blocks) if blocks else 0.0
    delta_pct = (closes[-1] - closes[-w]) / closes[-w] * 100.0 if len(closes) > w else 0.0
    if baseline > 0 and vol_now / baseline >= config.VOLUME_SURGE_RATIO:
        vote = _sign_with_deadband(delta_pct, 0.0)
        detail = f"volume surge x{vol_now / baseline:.2f} (price {delta_pct:+.2f}%)"
    else:
        vote = 0
        ratio = vol_now / baseline if baseline > 0 else 0.0
        detail = f"volume normal x{ratio:.2f}"
    results.append(IndicatorResult("Volume surge", vote, detail))

    # 4. Window delta over the 15-min Kalshi horizon
    vote = _sign_with_deadband(delta_pct, config.WINDOW_DELTA_DEADBAND_PCT)
    results.append(IndicatorResult("Window delta", vote, f"15m move {delta_pct:+.2f}%"))

    # 5. Micro momentum: EMA-smoothed 1-min returns (smoothing widened 3x)
    rets = [(b - a) / a * 100.0 for a, b in zip(closes[:-1], closes[1:])]
    mom_series = ema(rets, config.MOMENTUM_SPAN)
    mom = mom_series[-1]
    vote = _sign_with_deadband(mom, config.MOMENTUM_DEADBAND_PCT)
    results.append(IndicatorResult("Micro momentum", vote, f"smoothed ret {mom:+.3f}%/min"))

    # 6. Acceleration: momentum now vs N bars ago
    lag = config.ACCEL_LAG
    accel = mom - mom_series[-1 - lag] if len(mom_series) > lag else 0.0
    vote = _sign_with_deadband(accel, config.ACCEL_DEADBAND_PCT)
    results.append(IndicatorResult("Acceleration", vote, f"d(momentum) {accel:+.3f}"))

    # 7. Tick trend: net direction of last N 1-min closes (tick proxy)
    n = config.TICK_TREND_BARS
    moves = [1 if b > a else (-1 if b < a else 0)
             for a, b in zip(closes[-n - 1:-1], closes[-n:])]
    net = sum(moves)
    vote = 1 if net >= 2 else (-1 if net <= -2 else 0)
    results.append(IndicatorResult("Tick trend", vote, f"last {n} bars net {net:+d}"))

    # Confidence: majority agreement among the 7
    bull = sum(1 for r in results if r.vote > 0)
    bear = sum(1 for r in results if r.vote < 0)
    if bull > bear:
        direction, confidence = 1, bull
    elif bear > bull:
        direction, confidence = -1, bear
    else:
        direction, confidence = 0, bull  # tied (incl. all-neutral)

    realized_vol = pstdev(rets[-w:]) if len(rets) >= w else 0.0

    return Signals(
        results=results,
        direction=direction,
        confidence=confidence,
        window_delta_pct=delta_pct,
        realized_vol_pct=realized_vol,
        ema_sep_pct=sep_pct,
    )
