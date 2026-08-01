"""
Vectorized reimplementation of indicators.py + classifier.py, for computing
the 7-indicator confluence + 4-state classifier over an entire BTC 1-min
series at once instead of recomputing from scratch (Python loops over a
300-bar window) at every one of thousands of checkpoints.

Imports every threshold directly from config.py -- no duplicated constants,
so this can never silently drift from the values indicators.py/classifier.py
actually use.

Uses continuously-running (incremental) EMA/RSI instead of indicators.py's
literal "recompute from scratch over the trailing 300-bar window every call"
behavior. This is a numerical approximation, justified because the reseed
effect decays to negligible within the window (EMA span 27/63, RSI Wilder
alpha=1/14 all converge well within ~250 bars) -- validated in
validate_against_indicators.py by comparing this module's output against
indicators.compute_signals() called directly on real 300-bar windows.
"""
import numpy as np
import pandas as pd

import config


def _continuous_wilder_rsi(closes_dec: np.ndarray, period: int) -> np.ndarray:
    n = len(closes_dec)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    diffs = np.diff(closes_dec)
    gains = np.maximum(diffs, 0.0)
    losses = np.maximum(-diffs, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    out[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for j in range(period, len(diffs)):
        avg_gain = (avg_gain * (period - 1) + gains[j]) / period
        avg_loss = (avg_loss * (period - 1) + losses[j]) / period
        out[j + 1] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns ['close','volume'] indexed 0..n-1, time order.
    Returns a DataFrame aligned to df.index with every vote/state/label used
    downstream, plus the raw ingredients (sep_pct, rsi_val, etc.) needed for
    the strike-probability features.
    """
    closes = df["close"].to_numpy()
    vols = df["volume"].to_numpy()
    n = len(df)
    lb = config.LOOKBACK_BARS

    ema_fast = pd.Series(closes).ewm(span=config.EMA_FAST, adjust=False).mean().to_numpy()
    ema_slow = pd.Series(closes).ewm(span=config.EMA_SLOW, adjust=False).mean().to_numpy()
    sep_pct = (ema_fast - ema_slow) / ema_slow * 100.0
    vote_ema = np.where(np.abs(sep_pct) > config.EMA_MIN_SEP_PCT, np.sign(sep_pct), 0).astype(int)

    rsi_val = np.full(n, np.nan)
    for phase in range(3):
        idx = np.arange(phase, n, 3)
        dec = closes[idx]
        rsi_val[idx] = _continuous_wilder_rsi(dec, config.RSI_PERIOD)
    vote_rsi = np.zeros(n, dtype=int)
    vote_rsi[rsi_val >= config.RSI_BULL] = 1
    vote_rsi[rsi_val <= config.RSI_BEAR] = -1
    vote_rsi[np.isnan(rsi_val)] = 0

    w = config.WINDOW_DELTA_MINUTES
    vol_series = pd.Series(vols)
    roll_w_sum = vol_series.rolling(w).sum().to_numpy()
    vol_now = roll_w_sum
    nblk = config.VOLUME_BASELINE_BLOCKS
    block_sums = np.zeros(n)
    block_counts = np.zeros(n)
    for b in range(nblk):
        shift = (b + 1) * w
        shifted = np.roll(roll_w_sum, shift)
        shifted[:shift] = np.nan
        valid = ~np.isnan(shifted)
        block_sums[valid] += shifted[valid]
        block_counts[valid] += 1
    baseline = np.divide(block_sums, block_counts, out=np.zeros(n), where=block_counts > 0)
    close_series = pd.Series(closes)
    close_shift_w = close_series.shift(w - 1).to_numpy()  # matches closes[-w] (a w-1-bar-back ref)
    delta_pct = np.where(close_shift_w > 0, (closes - close_shift_w) / close_shift_w * 100.0, 0.0)
    vol_ratio = np.divide(vol_now, baseline, out=np.zeros(n), where=baseline > 0)
    vote_vol = np.where((baseline > 0) & (vol_ratio >= config.VOLUME_SURGE_RATIO), np.sign(delta_pct), 0).astype(int)

    vote_delta = np.where(delta_pct > config.WINDOW_DELTA_DEADBAND_PCT, 1,
                           np.where(delta_pct < -config.WINDOW_DELTA_DEADBAND_PCT, -1, 0))

    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = (closes[1:] - closes[:-1]) / closes[:-1] * 100.0
    mom = pd.Series(rets).ewm(span=config.MOMENTUM_SPAN, adjust=False).mean().to_numpy()
    vote_mom = np.where(mom > config.MOMENTUM_DEADBAND_PCT, 1, np.where(mom < -config.MOMENTUM_DEADBAND_PCT, -1, 0))
    accel = np.full(n, 0.0)
    lag = config.ACCEL_LAG
    accel[lag + 1:] = mom[lag + 1:] - mom[1:n - lag]
    vote_accel = np.where(accel > config.ACCEL_DEADBAND_PCT, 1, np.where(accel < -config.ACCEL_DEADBAND_PCT, -1, 0))

    ttb = config.TICK_TREND_BARS
    tick_sign = np.sign(closes[1:] - closes[:-1])
    tick_sign = np.concatenate([[0], tick_sign])
    net_ticks = pd.Series(tick_sign).rolling(ttb).sum().to_numpy()
    vote_tick = np.where(net_ticks >= 2, 1, np.where(net_ticks <= -2, -1, 0))

    realized_vol = pd.Series(rets).rolling(w).std(ddof=0).to_numpy()
    realized_vol = np.nan_to_num(realized_vol, nan=0.0)

    out = pd.DataFrame(dict(
        sep_pct=sep_pct, vote_ema=vote_ema, rsi_val=rsi_val, vote_rsi=vote_rsi,
        vol_ratio=vol_ratio, vote_vol=vote_vol, delta_pct=delta_pct, vote_delta=vote_delta,
        mom=mom, vote_mom=vote_mom, accel=accel, vote_accel=vote_accel,
        net_ticks=net_ticks, vote_tick=vote_tick, realized_vol=realized_vol,
    ))
    out["ready"] = np.arange(n) >= lb

    votes = out[["vote_ema", "vote_rsi", "vote_vol", "vote_delta", "vote_mom", "vote_accel", "vote_tick"]]
    bull = (votes > 0).sum(axis=1)
    bear = (votes < 0).sum(axis=1)
    out["bull_count"] = bull
    out["bear_count"] = bear
    out["direction"] = np.where(bull > bear, 1, np.where(bear > bull, -1, 0))
    out["confidence"] = np.where(bull >= bear, bull, bear)

    volatility_lbl = np.where(out["realized_vol"] >= config.VOL_HIGH_PCT, "High",
                        np.where(out["realized_vol"] >= config.VOL_MODERATE_PCT, "Moderate", "Low"))
    ema_aligned = (~out["sep_pct"].isna()) & (out["sep_pct"] * out["delta_pct"] > 0)
    abs_move = out["delta_pct"].abs()
    trend_lbl = np.where((abs_move >= config.TREND_STRONG_PCT) & ema_aligned, "Strong",
                  np.where(abs_move >= config.TREND_WEAK_PCT, "Weak", "Unclear"))
    mom_bull = (out["vote_rsi"] > 0).astype(int) + (out["vote_mom"] > 0).astype(int) + (out["vote_accel"] > 0).astype(int) + (out["vote_tick"] > 0).astype(int)
    mom_bear = (out["vote_rsi"] < 0).astype(int) + (out["vote_mom"] < 0).astype(int) + (out["vote_accel"] < 0).astype(int) + (out["vote_tick"] < 0).astype(int)
    momentum_lbl = np.where((mom_bull > 0) & (mom_bear == 0), "Bullish",
                     np.where((mom_bear > 0) & (mom_bull == 0), "Bearish",
                       np.where((mom_bull == 0) & (mom_bear == 0), "Neutral", "Mixed")))
    out["volatility_lbl"] = volatility_lbl
    out["trend_lbl"] = trend_lbl
    out["momentum_lbl"] = momentum_lbl

    ready_mask = out["ready"].to_numpy()
    state = np.full(n, "WARMUP", dtype=object)
    state[ready_mask] = "RANGING"
    up_mask = ready_mask & (trend_lbl == "Strong") & (momentum_lbl == "Bullish") & (out["direction"].to_numpy() > 0)
    down_mask = ready_mask & (trend_lbl == "Strong") & (momentum_lbl == "Bearish") & (out["direction"].to_numpy() < 0)
    stable_mask = ready_mask & (volatility_lbl == "Low") & (momentum_lbl == "Neutral") & (trend_lbl != "Strong") & ~up_mask & ~down_mask
    state[up_mask] = "UPTREND"
    state[down_mask] = "DOWNTREND"
    state[stable_mask] = "STABLE"
    out["state"] = state

    return out
