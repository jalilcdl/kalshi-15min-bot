# Kalshi 15-Min BTC Intel Bot

Personal-use Telegram alert bot for Kalshi's 15-minute BTC up/down markets
(`KXBTC15M`), built on the validated distance+time+volatility settlement
model (`model/strike_probability.py`) — the same model and fee/edge logic
`paper_trader.py` actually trades on, not the 7-indicator confluence engine
(shown to have no validated edge — see `research/kalshi-btc-validation/`).
**Information/alert layer only — no trading or order execution.**

## How the Kalshi market works (verified against the live API)

Each 15-minute window is one event. The market's `floor_strike` is the BTC
index price (CF Benchmarks BRTI, 60-second average) captured at window open.
**YES settles if the BRTI average just before close is ≥ the strike** — i.e.
"did BTC go up over these 15 minutes." Early in a window the strike can show
as TBD until Kalshi publishes the open print; the bot handles that.

## Setup

```
pip install -r requirements.txt
copy .env.example .env    # then fill in Telegram credentials
```

Without a `.env` the bot runs in **dry-run mode** — alerts print to the
console, which is the easiest way to sanity-check before wiring Telegram.

## Run

```
python bot.py --once      # single evaluation, print/send one alert, exit
python bot.py             # main loop: evaluate every ~60s
```

Two alert types, both Telegram, both built from real live data:

- **Entry "game plan"** — once per window, fired ~60-240s after it opens (the
  same early read the model and `paper_trader.py` use for their own first
  evaluation). Says the model's live P(YES), which side (if any) clears the
  real fee-adjusted edge gate, and — if there's a trade — the exit-target
  price, including the case where the target is mathematically unreachable
  from that entry price (already above where +35% would cross 100¢). If a
  window first comes into view more than 4 minutes old (e.g. the bot just
  restarted), it's skipped rather than sent late.
- **Target-hit nudge** — fired the moment a live paper position (from
  `paper_trader.py`'s own `data/trade_log.csv`) crosses the favorable-move
  exit target, so the alert and the actual paper trade can never disagree.
  On first startup, all *existing* target-hit trades are marked as already
  notified so days of paper-trading history don't get replayed as alerts.

## Dashboard (read-only)

```
pip install -r requirements.txt   # includes streamlit, plotly, pandas
run_dashboard.bat                 # or: python -m streamlit run dashboard/app.py
```

A live, mobile-friendly Streamlit dashboard — **never places orders**, view-only
on top of the same public Coinbase/Kalshi feeds and `indicators.py` /
`classifier.py` / `strike_distance.py` the bot uses. Pages:

- **Scoreboard** — live BTC price, active `KXBTC15M` market card (strike,
  bid/ask, time left, strike-distance read), the 7-indicator vote table, and
  a clickable list of recently-settled windows.
- **Window detail** — click a settled window to replay the model at fixed
  checkpoints (3/7/11/13 min in) using only data known at that moment, next
  to the actual settlement — a no-lookahead sanity check on any single window.
- **Trade log** — your actual Kalshi fills, read from `data/trade_log.csv`
  (you edit the CSV by hand; see `trade_log.py`). Record/W-L/P&L/ROI, with an
  honest small-sample warning under 30 settled trades.
- **Model performance** — the direction-signal validation result (**no
  validated edge**) and the strike-probability model's walk-forward result
  (**partially validated** — beats the naive baseline significantly; the
  7-indicator confluence engine adds nothing on top of it), both sourced from
  `data/validation/` and the `research/` folders, kept clearly separated.
- **About & limitations** — what's validated, what isn't, and known caveats
  (Coinbase-vs-BRTI basis, signal firing frequency).

Optional password gate for a public URL: set `KALSHI_DASHBOARD_PASSWORD` (env
var locally, or Streamlit Cloud secret — see `.streamlit/secrets.toml.example`).
No password needed for local/private-network use (e.g. behind Tailscale).

## Paper trading

```
research/strike_probability/scripts/fit_final_model.py   # (re)fit the deployed model, if needed
install_paper_trader_task.ps1                             # registers + starts the Windows task
```

`paper_trader.py` runs continuously (Windows Scheduled Task, same pattern as
the alert bot — see below) and simulates fills **against real live Kalshi
quotes**, not theoretical prices: every ~60s it checks the active market,
computes the fitted strike-probability model's P(YES), and compares it to
the real YES/NO ask (including Kalshi's real per-contract taker fee) using
`model/strike_probability.py` + `fees.py`. If either side clears
`config.PAPER_TRADE_MIN_EDGE` net of fees, it logs one simulated trade
(`mode=paper`) to `data/trade_log.csv` — one entry per market, gated out
within `config.PAPER_TRADE_MIN_DIST_OVER_REACHABLE` of a noise-level
distance to strike (a real overconfidence failure mode the smoke test caught
on its first live evaluation — see `research/strike_probability/README.md`).
**Exit logic simulates the actual intended strategy — buy early, sell on a
favorable move, not hold to settlement.** Each cycle, every pending
position's real live bid is checked against `config.PAPER_TRADE_EXIT_TARGET`
(35%, real fee charged on the exit too); if hit, the trade closes right
there (`exit_reason=target_hit`). If not, it resolves against the real
settlement once the window closes (`exit_reason=settlement`), same as
before. This target was chosen from a real historical backtest — see
`research/exit_timing/README.md` — which found hitting 20-40% happens often
and fast, but **holding to settlement actually outperformed every early-exit
threshold tested in aggregate ROI**, consistently though not at full
statistical significance. Deployed at 35% (the best-performing point in the
user's stated range) anyway, because paper trading exists to test the
strategy actually intended to be run, not the backtest optimum — read that
README before trusting this with real money.

Results accumulate in the dashboard's Trade log page, viewable from a
phone, kept in a clearly separate section from any real (`mode=live`)
trades you log by hand — this is designed to run for at least a couple of
weeks before its numbers mean anything.

Monitor it:
```
Get-Content paper_trader.log -Tail 40 -Wait
python trade_log.py --table
```

## Architecture

| File | Role |
|---|---|
| `config.py` | All tuning knobs + `.env` loading |
| `coinbase_feed.py` | BTC/USD 1-min OHLCV (Coinbase Exchange public REST) |
| `kalshi_feed.py` | Active + settled `KXBTC15M` markets (Kalshi public API) |
| `indicators.py` | 7-indicator confluence engine + confidence score |
| `classifier.py` | Four-state model: 🔴 DOWNTREND / 🟢 UPTREND / 🟡 RANGING / 🔵 STABLE |
| `strike_distance.py` | **Experimental** favors-YES/NO read vs strike + time left |
| `alert_message.py` | Telegram message template + rule-based analysis text |
| `telegram_client.py` | Delivery (console fallback in dry-run) |
| `bot.py` | Main loop, alert triggers, throttling |
| `backtest_strike.py` | Backtests the strike module vs settled Kalshi markets |
| `dashboard/app.py` | Read-only Streamlit dashboard (see Dashboard section below) |
| `trade_log.py` | Forward-test trade tracker (paper + live) backing the dashboard's Trade log page |
| `model/strike_probability.py` | Loads the fitted distance+time+vol logistic model, exposes `predict_p_yes()` |
| `fees.py` | Kalshi's documented taker fee formula |
| `paper_trader.py` | Continuous loop: simulates fills vs. real live quotes, logs to `data/trade_log.csv` |
| `install_paper_trader_task.ps1` | Registers the paper trader as a Windows Scheduled Task |
| `research/strike_probability/` | Walk-forward validation of the strike-probability model (see its own README) |
| `research/exit_timing/` | Real-price backtest of the buy-early/sell-on-favorable-move strategy (see its own README) |

## The 7 indicators (15-min retune: ~3x the 5-min lookbacks)

1. **EMA cross** — 27/63 on 1-min bars (≡ 9/21 on 3-min)
2. **RSI** — 14 on resampled 3-min bars (bull ≥55 / bear ≤45)
3. **Volume surge** — last 15-min volume vs 4h baseline of 15-min blocks; direction from concurrent price move
4. **Window delta** — % move over the 15-min Kalshi horizon
5. **Micro momentum** — EMA-smoothed 1-min returns (smoothing widened 3x)
6. **Acceleration** — momentum now vs 5 bars ago
7. **Tick trend** — net direction of last 5 bars (kept as-is; currently a
   close-to-close proxy — swap in a WebSocket tick feed for true tick data)

**Confidence** = how many of the 7 agree with the majority direction:
6–7 high, 4–5 moderate, ≤3 low. High confidence is the sizing gate.

## Strike-distance module — EXPERIMENTAL

Computes distance to strike, required move per minute, and the vol-implied
reachable move (`σ₁ₘᵢₙ · √minutes_left`), then labels the window
**Favors YES / Favors NO / Too close to call**. Distance beyond
1.5 expected sigmas → current side holds; inside the noise band it defers
to high-confidence confluence direction.

Backtest before trusting it:

```
python backtest_strike.py --hours 24
```

Replays every settled market at minutes 3/7/11/13 of its window using only
data known at that moment, and scores calls against actual Kalshi
settlements. First run (12h, 48 markets, 62 calls) hit 96.8% — **but** most
calls fire late in the window where "current side holds" is already a strong
baseline, and it abstains on most early checkpoints. Compare against the
current-side baseline, not coin-flip. Coinbase spot is also a proxy for the
BRTI settlement index, so near-strike results carry basis noise.

Kalshi-specific guardrail: the bot always advises standing down in the
final 2 minutes of a window (`FINAL_MINUTES_NOISY`) — settlement-print
noise dominates there.
