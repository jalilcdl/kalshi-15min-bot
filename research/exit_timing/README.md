# Exit-timing validation — buy early, sell on a favorable move

**This answers a genuinely different question from everything else in this
project.** The strike-probability model and its walk-forward validation
(`../strike_probability/`) answer "will this settle YES or NO." The user's
actual intended strategy is different: **buy early in the window, wait for
the contract's own price to move 20-40% in your favor, and sell before
expiry** — a price-timing / scalp strategy, not a settlement bet. Nothing
built before this answered that. This does, using real historical Kalshi
contract prices (not a model-derived proxy).

**Bottom line: hitting a 20-40% favorable move happens often and fast (69-86%
of entries, in 2-6 minutes median) — but in aggregate dollar terms, simply
holding every position to settlement outperformed every early-exit threshold
tested, consistently, though not at full statistical significance. The exit
logic is deployed anyway, at the best-performing point in the stated range
(35%), because paper trading exists to test the strategy actually intended
to be run — but go in knowing the honest backtest doesn't clearly favor it
over just holding.**

## 1. Data: real historical contract prices, not a proxy

Kalshi's API exposes real per-minute price/bid/ask candlesticks for every
market (`/series/{s}/markets/{t}/candlesticks`), confirmed to retain the
full 45-day history this project already had settled-market data for.
Pulled all 4,250 markets (`scripts/fetch_candlesticks.py`, rate-limited,
resumable) — 67,897 1-min bars, 4,244 markets with usable data (6 came back
empty). This is what makes the analysis below real rather than a
model-implied approximation.

## 2. Method

**Entry:** replays the exact same rule `paper_trader.decide_trade()` uses —
distance+time+volatility model, minimum-distance gate, minimum-edge gate —
at checkpoint 1 (~60-120s into the window, same timing already established
in `../strike_probability/README.md` §3b), using the REAL yes_ask/no_ask
read off the candlestick at that minute (not the single settlement-time
snapshot in `settled_markets.csv`, which only reflects price near/at close
and is useless for this). 2,790 of 4,250 checkpoint-1 rows cleared the entry
threshold.

**Exit:** for every entered trade, scans the market's own remaining
candlesticks for the first minute the position's **sellable value** (the
bid on your side — you sell into the bid, not the ask; for NO, that's
`1 - yes_ask`) reaches each threshold in {20, 25, 30, 35, 40}%. Reports both
a **conservative** read (minute-close prices only) and an **optimistic**
one (best price touched intra-minute, via the high/low — this requires
watching every tick to actually capture, so treat it as an upper bound, not
a promise). Money math (§4) uses the conservative read only.

**METHODOLOGY CAVEAT — read this before trusting the numbers:** the entry
model is `model/strike_prob_model.pkl`, fit on ALL 45 days of this same
dataset (its settlement-prediction accuracy was already validated properly
on a held-out walk-forward basis elsewhere — see
`../strike_probability/README.md`). This analysis reuses that all-data model
to decide entries, then examines real historical price paths afterward on
the same 45 days. The price-path data itself was never used to fit
anything, so there's no direct leakage — but this is a **historical replay**
of what the live strategy would have done, not a fresh nested walk-forward
test of entry+exit together. Flagged plainly, not hidden.

## 3. Does the target actually get hit, and how fast?

| Target | Hit rate (close) | Median time | Hit rate (optimistic) | Median time |
|---|---|---|---|---|
| +20% | 86.0% | 2.0 min | 90.6% | 2.0 min |
| +25% | 82.5% | 3.0 min | 87.0% | 2.0 min |
| +30% | 78.6% | 4.0 min | 83.3% | 3.0 min |
| +35% | 74.8% | 5.0 min | 78.7% | 4.0 min |
| +40% | 69.4% | 6.0 min | 73.2% | 4.0 min |

n=2,790 entered trades. Mean/median max favorable excursion (even among
trades that didn't hit a given target): 60.6% / 56.1%. Only 3.0% of entered
trades never moved favorably at all. **On the narrow question "does the
price move 20-40% in my favor before expiry" — yes, usually, and within a
few minutes.** This part of the intended strategy is real.

## 4. Does selling early actually make more money than holding?

This is the number that matters, and it's the honest surprise: **no, not in
this data.** Full round-trip P&L, real prices, **real double-sided fees**
(a round-trip early exit pays Kalshi's taker fee on entry AND exit; holding
to settlement only ever pays it once):

| Strategy | Turnover | Net profit | ROI | "Win" rate |
|---|---|---|---|---|
| **Always hold to settlement** | $16,793.93 | **+$2,076.07** | **12.4%** | 67.6% |
| Sell at +20% (else hold) | $16,793.93 | +$1,483.39 | 8.8% | 87.0% |
| Sell at +25% (else hold) | $16,793.93 | +$1,598.39 | 9.5% | 84.7% |
| Sell at +30% (else hold) | $16,793.93 | +$1,716.25 | 10.2% | 82.5% |
| **Sell at +35% (else hold)** | $16,793.93 | +$1,825.81 | **10.9%** | 80.6% |
| Sell at +40% (else hold) | $16,793.93 | +$1,802.64 | 10.7% | 78.3% |

Every early-exit variant "wins" (exits for a gain or settles favorably) far
more *often* than holding (78-87% vs. 67.6%) — but holding still comes out
**ahead in total dollars**, and by a consistent margin across every
threshold tested. 35% was the best of the tested early-exit points, still
~1.5 percentage points of ROI behind just holding.

**Why:** decomposed the trades that DID hit each target — of those, roughly
**77-85% would have gone on to settle in the buyer's favor anyway** (the
full $1.00 payout, not just the capped 20-40% gain — upside foregone by
selling early), and only **15-23% would have reversed to a total loss**
(the case where selling early genuinely saved the trade). The insurance
value of locking in a small gain doesn't outweigh the much larger payouts
given up on the majority of trades that were headed for a win regardless.

**Statistical significance:** market-level bootstrap (2,000 resamples),
hold vs. the best-performing scalp variant (+35%): profit gap -$250.26,
**p=0.127**. The direction is consistent across every threshold and both a
partial (n=404) and full (n=2,790) sample — but it does not clear the
conventional p<0.05 bar. Read this as "hold looks better and the pattern is
consistent, not as "definitively proven."

## 5. What was actually deployed, and why

`config.PAPER_TRADE_EXIT_TARGET = 0.35` — `paper_trader.py` now checks every
pending position's real live bid each cycle and sells early if it's up 35%,
else holds to settlement as before (`trade_log.py` tracks this distinctly:
`exit_reason` = "target_hit" vs "settlement", with its own fee column since
an early exit pays fees twice).

**This was deployed despite §4's finding, on purpose.** The point of paper
trading here is to test the strategy actually intended to be run for real,
not the theoretical backtest optimum — and 35% is the best-supported point
within the user's own stated 20-40% range. But go in with eyes open: the
honest backtest says simply holding may do better in aggregate, even though
selling early "works" (exits positively) much more often. Watch the live
paper-trading numbers in the dashboard's Trade log page against this
expectation as real data accumulates.

## 5b. Direct-regression side-selection model — a validated null result

Follow-up: could a model built specifically to predict "which side hits
+30%" pick sides better than what `paper_trader.py` already uses? Tested
properly, in two parts, and the honest answer is **no** — with a genuinely
useful confirmatory finding underneath it.

**Part 1 — confidence gate (`scripts/fit_hit_target_model.py`).** Fit
`P(hit +30% | side, entry_price, minutes_remaining, realized_vol)` on the
2,790 trades in `exit_timing_trades.csv`, same walk-forward rigor as
everywhere else. Clears the base rate (Brier 0.1547 vs. 0.1665, p<0.0001) —
but is statistically **indistinguishable from just using entry price alone**
(p=0.923). Nothing new here: side, time remaining, and volatility add zero
information once entry price is known, and entry price's importance was
already obvious (cheap contracts have more room to travel 30% before hitting
the 0-100¢ ceiling).

**Important scope problem with Part 1**: `exit_timing_trades.csv` only
contains the side the settlement-probability model **already chose** to
enter. A model fit on it can only ever answer "given this side was picked,
how likely is it to hit +30%" — using it to validate *side selection* would
be circular, since the sample never saw what the other side would have done.

**Part 2 — genuine side selection (`scripts/side_selection_backtest.py`).**
Fixed that by rebuilding a side-symmetric dataset directly from the real
candlestick data: both the YES and the NO hypothetical outcome, computed for
every checkpoint-1 row across all 4,244 markets, unconditional on any
model's prior choice (4,244 × 2 = 8,488 rows, cached in
`results/side_symmetric_dataset.csv`). Fit the same walk-forward model as a
standalone side-picker (pick whichever side it rates higher) and compared
against real alternatives, not just a coin flip:

| Side-selection strategy | Hit rate on chosen side |
|---|---|
| Coin flip | 65.6% |
| Always take the cheaper side (mechanical, direction-agnostic) | 64.1-64.4% |
| **Fitted hit-target side-selection model** | 65.8% |
| **Existing settlement-probability model's pick** | **75.4%** |

The new model, purpose-built for exactly this question, ties with a coin
flip and the naive cheaper-side rule (p=0.292 vs. cheaper-side — not
significant). The **existing** settlement-probability model — built to
answer a completely different question (will this settle YES/NO) — beats
all of them by a wide, highly significant margin (gap +0.113 vs.
cheaper-side, p<0.0001).

**Why:** a "likely to settle YES" read means that side's price is expected
to trend toward 100¢ as the window plays out — a real directional lean that
naturally produces the large relative gains needed to clear a 30% target
along the way. Entry price alone or a coin flip carries no directional
information, so neither reliably rides a trend toward the target the way an
actual settlement forecast does.

**Decision: did not wire either new model into `paper_trader.py`.** Per the
original bar ("if it clears validation") — it didn't. The existing
settlement-probability model's side selection is confirmed, from a fully
independent angle, to already be the best available choice for this
strategy too. That's a real, useful result: it's a second, unrelated
validation of a decision already in production, not a wasted exercise.
Full numbers: `results/hit_target_side_selection_summary.json`.

## 5c. Does waiting until 2-5 minutes into the window give a better read?

A specific, testable claim from real trading experience: does entering later
(2-5 min in, instead of the ~60-120s mark everything here uses) give a more
reliable read on whether the 30% target hits? Tested directly, real data,
same production entry rule (`decide_entry()`, unchanged) replayed at
checkpoints 1/2/3/4/5/7/10 minutes using the real candlestick price paths.

**Short answer: no — the data says the opposite, clearly.**

| Checkpoint | Entries (of 4,244) | Hit rate | Naive baseline hit rate | Median mins left | Median time-to-hit | Mean edge |
|---|---|---|---|---|---|---|
| **1 min** | 2,790 (65.7%) | **78.6%** | 75.4% | 14.0 | 4.0m | 16.9¢ |
| 2 min | 2,766 (65.2%) | 72.7% | 68.6% | 13.0 | 4.0m | 15.3¢ |
| 3 min | 2,750 (64.8%) | 66.2% | 61.7% | 12.0 | 4.0m | 14.4¢ |
| 4 min | 2,737 (64.5%) | 62.9% | 56.7% | 11.0 | 3.0m | 13.7¢ |
| 5 min | 2,658 (62.6%) | 60.1% | 51.2% | 10.0 | 3.0m | 13.4¢ |
| 7 min | 2,471 (58.2%) | 57.1% | 38.5% | 8.0 | 2.0m | 12.0¢ |
| 10 min | 2,521 (59.4%) | 49.8% | 25.7% | 5.0 | 1.0m | 12.4¢ |

Every later checkpoint is significantly worse than checkpoint 1 (paired
market-level bootstrap, same markets compared at both timestamps): p<0.0001
at every single checkpoint, monotonically declining from 78.6% to 49.8%.

**Both confounds you flagged, checked directly, not glossed over:**

- **Less time remaining is real and mechanical** (median 14 min left at
  checkpoint 1 down to 5 min at checkpoint 10) — but the reported hit rate
  already reflects this honestly; it scans the real remaining candles to
  the real close, no synthetic time budget. So this isn't an artifact to
  correct for, it's the actual, honest answer to "if you enter this late,
  what's your real chance." Read the decline as "this is genuinely worse,"
  not "the data needs adjusting."
- **The naive, model-free baseline (whichever side is currently ahead)
  declines even faster** than the entry-gated rate (75.4% → 25.7% vs. 78.6%
  → 49.8%). That means the entry rule's edge *over the naive baseline*
  actually widens with later entry (+3.2 points at checkpoint 1 growing to
  +24.1 points at checkpoint 10) — waiting does make the model's selection
  relatively more valuable. But the absolute odds of success are still
  clearly worse the later you wait, because there's mechanically less
  runway. Relative improvement does not overcome absolute decline here.

**Checked the "sharper confidence" alternative too**, in case "more
accurate" meant "the read feels more certain" rather than "the target hits
more often": mean edge magnitude also *decreases* with later checkpoints
(16.9¢ at 1 min down to ~12-14¢ from 5-10 min). No support for that reading
either — the model isn't becoming more confident later, it's becoming less.

**A plausible, honest explanation for the experience, not a data-backed
one:** this project already found (§3b of `../strike_probability/README.md`)
that *settlement* prediction (will it settle YES/NO) gets dramatically more
accurate with elapsed time — Brier 0.2304 at checkpoint 1 improving to
~0.0757 by checkpoint 11+. That's real, just for a different question than
"will it swing 30% before close." It's plausible the two are getting
conflated — a genuinely sharper settlement read later in the window doesn't
mean a better 30%-swing read, because the swing question is dominated by
how much runway is left, not by how sure the model is.

**Implication for alert timing and the paper trader: no change.** The
current ~60-120s entry point isn't just unchanged by this test — it's
confirmed as the best-performing point among everything tested here, on
both hit rate and edge magnitude. Moving entries to 2-5 minutes would make
things worse on the numbers, not better. (Earlier than ~60-120s isn't
testable with 1-min candle data — that's the real floor, same limit noted
in `../strike_probability/README.md` §3b.)

Reproduce: `python scripts/entry_timing_backtest.py`. Full per-checkpoint
data: `results/entry_timing_checkpoint_{1,2,3,4,5,7,10}.csv`.

## 6. Reproducing this

```
cd scripts
python fetch_candlesticks.py ../data/candlesticks.csv   # resumable, ~25 min for the full history
python exit_timing_backtest.py
python pnl_comparison.py
python fit_hit_target_model.py           # confidence-gate model (§5b, part 1)
python side_selection_backtest.py        # side-symmetric dataset + selection test (§5b, part 2)
python entry_timing_backtest.py          # 2-5min entry-timing test (§5c)
```

## 7. Files

```
data/candlesticks.csv                Real 1-min price/bid/ask history, 4,244 markets
scripts/fetch_candlesticks.py        Rate-limited, resumable candlestick fetcher
scripts/exit_timing_backtest.py      Entry replay + hit-rate/time-to-target analysis
scripts/pnl_comparison.py            Full round-trip P&L: sell-early vs. hold, real fees
scripts/fit_hit_target_model.py       Confidence-gate model on already-entered trades (§5b pt.1)
scripts/side_selection_backtest.py    Side-symmetric dataset + genuine side-selection test (§5b pt.2)
results/exit_timing_trades.csv       Every entered trade: entry, exit prices/times per threshold
results/side_symmetric_dataset.csv    Both YES/NO hypothetical outcomes, all 4,244 markets
results/hit_target_side_selection_summary.json  Full numbers backing §5b
scripts/entry_timing_backtest.py     Tests whether entering at 2-5min beats ~60-120s (§5c)
results/entry_timing_checkpoint_*.csv  Per-checkpoint entry/hit data backing §5c
```
