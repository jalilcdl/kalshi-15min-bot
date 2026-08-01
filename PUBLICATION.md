# TradingView publication — Kalshi 15m BTC Intel

**Title:** Kalshi 15m BTC Intel — 7-Indicator Confluence State Classifier

## What it does

This indicator classifies the Bitcoin market into one of four states — **UPTREND, DOWNTREND, RANGING, or STABLE** — over a rolling 15-minute horizon, using a confluence vote across 7 independent indicators. It was originally built as an alerting bot for 15-minute BTC event contracts (Kalshi's KXBTC15M series), where the only question that matters is: *which way is price leaning over the next 15 minutes, and how confident can I be?*

It is designed for **COINBASE:BTCUSD** but works on any liquid BTC pair. It is **timeframe-independent**: the script requests 1-minute data internally and computes everything on its own rolling 300-bar (5-hour) 1-minute buffer, so you can view it on any chart timeframe from 1 minute up.

## How it works

Seven indicators each cast a vote: bullish (+1), bearish (−1), or neutral (0):

1. **EMA cross** — EMA 27/63 on 1-minute closes (equivalent to 9/21 on 3-minute bars); votes on the sign of the separation, with a deadband to filter chop.
2. **RSI(14)** — computed on a 3-minute resample; bullish above 55, bearish below 45.
3. **Volume surge** — last 15 minutes of volume vs. a 4-hour baseline of 15-minute blocks; a surge ≥1.5× votes in the direction of the concurrent price move.
4. **Window delta** — the % move over the trailing 15 minutes, with a deadband.
5. **Micro momentum** — EMA-smoothed 1-minute returns.
6. **Acceleration** — momentum now vs. 5 bars ago (is the move speeding up or fading?).
7. **Tick trend** — net direction of the last 5 one-minute closes.

**Confidence** = how many of the 7 agree with the majority direction (x/7).

A classifier then combines trend strength (size of the 15-minute move, confirmed by EMA alignment), a momentum label (from the momentum-family indicators), and realized volatility (population stdev of 1-minute returns) into the four states. UPTREND/DOWNTREND require a strong, EMA-confirmed move with aligned momentum; STABLE requires low volatility and neutral momentum; everything else is RANGING.

## What's on the chart

- **Dashboard panel** — current state, confidence dot-meter (●●●●●○○), trend/momentum/volatility labels, 15-minute move, realized vol, and each indicator's live vote with its reading. Position, size, theme (dark/light), and a compact mode are configurable.
- **Confluence-colored candles** — gradient from gray (no agreement) to full trend color (7/7), so conviction is visible bar by bar.
- **Glow EMAs + trend cloud** — the cloud deepens as EMA separation grows.
- **BUY/SELL labels** at the bar where the state flips into UPTREND/DOWNTREND (switchable to plain triangles).
- **Background tint** for directional and stable states (RANGING is untinted by default).

## Alerts

Create one alert with condition **"Any alert() function call"** to receive every state transition with detail, e.g. `RANGING → UPTREND (6/7, +0.21% 15m move)`. Separate alert conditions exist for high-confidence (≥6/7) UPTREND and DOWNTREND only.

## Notes and limitations

- All thresholds (RSI levels, deadbands, surge ratio, confidence bands) are exposed as inputs; defaults are the values the original bot ran with.
- The state updates intrabar in real time; on closed history it reflects one evaluation per chart bar. Signals print when a state *begins*, which is after a move has partially developed — this is a regime classifier, not a prediction.
- 1-minute intrabar data depth is limited by TradingView, so older chart history may show a warm-up state; the live edge is unaffected.
- RANGING means *no edge — stand aside*. Most of the time the market is RANGING; that is the indicator working as intended, not a defect.

This is a decision-support tool, not financial advice. No indicator wins consistently on 15-minute binaries; always size positions accordingly.
