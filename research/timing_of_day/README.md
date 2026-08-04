# Is there a real hour-of-day or day-of-week edge?

User's question, verbatim: "based on the backtesting from Kalshi, when would
be the best times of the day to trade? Which days of the week?" Flagged
upfront as prone to false positives — 24 hours × 7 days is a lot of chances
for a random-looking pattern to appear significant by chance alone.

## Methodology (designed to survive that concern, not just note it)

- **Data**: `exit_timing_trades.csv` — 2,790 real checkpoint-1 (~60-120s
  entry, matching production) trades, real candlestick price paths, joined to
  `settled_markets.csv` for each market's real UTC open time. No synthetic
  data. 46 unique days, 2026-06-17 to 2026-08-01.
- **Metric**: hit rate on the deployed +35% exit target
  (`config.PAPER_TRADE_EXIT_TARGET`), using the "close" (conservative) hit
  definition already established as this project's headline metric in
  `research/exit_timing/README.md`.
- **Multiple-comparison control**: an **omnibus chi-square test** first, one
  per family (hour-of-day, day-of-week) — "does hit rate vary across this
  grouping at all?" This is 2 total tests, not 31, and is what actually
  controls the false-positive rate here. Only if an omnibus test rejects does
  it make sense to look bucket-by-bucket. Per-bucket comparisons are also
  reported (two-proportion z-test, each bucket vs. everyone else pooled)
  against a **Bonferroni-corrected threshold of 0.05 / 31 ≈ 0.00161** (31 =
  24 hour buckets + 7 day buckets), not the raw 0.05, in case anyone wants to
  eyeball individual buckets anyway.
- **Sample size floor**: buckets under 30 trades are marked insufficient and
  excluded from the significance claims (none hit this in practice — hour
  buckets ranged 87–135 trades, day buckets 340–461).
- **Plausible mechanism check**: real BTC 1-min realized volatility
  (Coinbase) and real Kalshi contract bid/ask spread, both by UTC hour — a
  pattern needs a plausible cause behind it, not just a number that happened
  to look different.

## Result: no real hour-of-day or day-of-week edge

| Test | Result |
|---|---|
| Hour-of-day omnibus (chi², dof=23) | chi2=20.40, **p=0.617** |
| Day-of-week omnibus (chi², dof=6) | chi2=6.38, **p=0.382** |
| Individual buckets clearing Bonferroni (p<0.00161) | **0 of 24 hours, 0 of 7 days** |

Both omnibus tests are nowhere near significant, so per the methodology above
this stops there — the per-bucket tables below are provided for transparency,
not as a finding. **No hour or day should be read as better or worse to
trade; the honest answer is that time-of-day and day-of-week don't move the
needle on this strategy's hit rate, at current sample size.**

Full per-bucket numbers: `results/hour_of_day_hit_rate.csv`,
`results/day_of_week_hit_rate.csv`. Hit rates range 67.7%–82.5% across hours
and 71.5%–78.2% across days — normal sampling noise around the 74.8% overall
rate given ~90–460 trades per bucket, not a real signal (every single bucket
p-value is well above even the *uncorrected* 0.05 threshold except two
hours that sit around p≈0.06–0.09, which don't survive Bonferroni and
shouldn't survive common sense either — a coin flipped 168 times, in 31
different groupings, will produce a few groups that look unusual).

## Mechanism check: a real cycle exists, but it doesn't reach the strategy

BTC's own realized volatility **does** have a real daily cycle — lowest
(~0.035–0.04%/min) around 09:00–11:00 UTC (Asia/early-Europe overnight for
the US), highest (~0.079–0.092%/min) around 13:00–15:00 UTC (US market
open / US-Europe overlap). That's the expected, well-known crypto volatility
pattern and is a real, plausible mechanism *in principle*.

But it doesn't show up anywhere downstream:
- Kalshi's own contract spread is essentially flat across all 24 hours
  (0.067–0.072, no visible cycle) — correlation with BTC volatility across
  hours is **-0.09**, near zero, not the positive relationship you'd expect
  if Kalshi's market-making tightened/widened with BTC's own activity cycle.
- The strategy's hit rate is flat too (see above).

**Plausible explanation, not a proven one**: the settlement-probability
model (`model/strike_probability.py`) already takes realized volatility as
an input feature — it's part of "distance+time+volatility." If the model is
already adjusting its probability estimate for how volatile things are right
now, the *entry gate* (which requires edge over that already-volatility-aware
probability) would naturally end up hit-rate-flat across the volatility
cycle even though the raw volatility cycle is real. That's consistent with
what's observed, not independently confirmed here.

Full numbers: `results/mechanism_check_by_hour.csv`.

## Reproduce

```
cd scripts
python hour_day_backtest.py
```

## Files

```
scripts/hour_day_backtest.py           Omnibus + per-bucket tests, mechanism check
results/hour_of_day_hit_rate.csv       Per-hour n, hit rate, p-value vs rest
results/day_of_week_hit_rate.csv       Per-day n, hit rate, p-value vs rest
results/mechanism_check_by_hour.csv    Real BTC volatility & Kalshi spread by hour
```
