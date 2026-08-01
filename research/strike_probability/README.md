# Strike-probability model — validation report

**Follow-up (paper trading):** this model has since been fit on the full
dataset and deployed for live paper trading — see `../../paper_trader.py`
and the "Paper trading" section of the project README. Worth knowing before
trusting it further: the paper trader's very first live evaluation caught a
real failure mode this walk-forward report didn't surface — right at
distance-to-strike ≈ 0 (a genuine coin-flip price), the `current_side_leading`
feature flips discretely and the model output 71% for a side the real market
priced at 37c, a much larger gap than the ~6-7 point overconfidence already
visible in §3's calibration table for the 0.6-0.7 bucket. A minimum-distance
entry gate (`config.PAPER_TRADE_MIN_DIST_OVER_REACHABLE`) was added to keep
the paper trader from acting on this specific blind spot; the underlying
model coefficients were deliberately left untouched rather than hand-patched.
Treat this as a reminder that a backtest, however careful, can still miss
edge cases live data finds on the first try.


**Bottom line: unlike the direction/confluence signal (no edge, see the
sibling `tradingview-mcp/research/kalshi-btc-validation` report), the
distance-to-strike + time + volatility framing shows a real, validated,
walk-forward improvement over the naive baseline — but almost all of that
improvement comes from distance/time/vol alone. The 7-indicator confluence
engine adds essentially nothing on top of it. That's a genuine, if partial,
positive result, and it comes with real caveats below — read them before
treating this as tradeable.**

## 0. The question this asks (and why it's different from the direction signal)

The earlier validation asked "does BTC's next-15-minute direction have
predictable structure?" — no. This asks a different question: "**given
where price currently sits relative to the Kalshi strike, how much time is
left, and how volatile things have been, what's the actual probability this
settles YES?**" That's a first-passage / barrier problem, not a direction
call — distance and time do real, provable work here even when raw
direction doesn't. `strike_distance.py` already gestured at this with a
fixed "1.5 sigma" heuristic; this work replaces the guess with a model
**fit and validated against 4,250 real settled Kalshi markets**.

## 1. Data

- **Settled markets:** 4,250 real `KXBTC15M` markets, 2026-06-17 to
  2026-08-01 (45 days), pulled from Kalshi's public API (`fetch_settled_markets.py`).
  Base rate: 49.9% settled YES — confirms the earlier finding that BTC's
  unconditional 15-min direction is a coin flip.
- **BTC price:** 65,096 Coinbase 1-min bars covering the full range plus a
  300-bar warmup buffer (`fetch_btc_data.py`).
- **Checkpoints:** for every settled market, minutes 1 through 13 into the
  window (13 checkpoints/market, 55,245 rows total before dropping rows
  missing a 1-min bar). Each checkpoint uses only data available at that
  exact minute — no lookahead.

## 2. Method

`signals_fast.py` is a vectorized reimplementation of the bot's actual
`indicators.py` + `classifier.py`, validated by running the *real* production
functions directly on 40 real 300-bar windows and comparing every one of the
7 votes, the classifier state, and the confidence score:
**100% agreement, 0 mismatches** (`validate_against_indicators.py`).

`build_features.py` computes, per checkpoint: distance to strike,
minutes remaining, realized volatility, the vol-implied "reachable move"
(`realized_vol * sqrt(minutes_remaining)`), distance-over-reachable (the
existing heuristic's core ratio), which side currently leads, plus all 7
confluence-engine outputs.

`walk_forward.py` evaluates four approaches, **chronological folds only**
(6 expanding-window folds, ~1 week each, first 30% of history as pure
training warmup), with **whole markets — never individual checkpoints —
assigned to a fold**, since a market's 13 checkpoints share one price path
and splitting them would both leak information and double-count:

| | Model | What it uses |
|---|---|---|
| A | Base rate | Nothing but "what fraction settle YES this far into a window" — the floor |
| B | Current-side-holds (time-bucketed) | Which side currently leads + how far into the window — the naive baseline the bot's own docs already flag as a strong bar to clear |
| C | Distance+time+vol only | Signed distance to strike, minutes remaining, realized vol, distance/reachable-move ratio, current side — logistic regression, no confluence indicators |
| D | Full model | Everything in C plus all 7 confluence-engine outputs (EMA separation, RSI, volume surge, momentum, acceleration, tick trend, direction, confidence, trend/momentum labels) |

Scored with **Brier score** and **log loss** (proper scoring rules that
reward calibration, not just hit rate — the right metric for something
you'd size a real probability against). Significance checked with a
**market-level paired bootstrap** (2,000 resamples of whole markets, not
individual rows, respecting the within-market correlation).

## 3. Results

Pooled out-of-fold (38,674 rows, 2,975 markets, across all 6 folds):

| Model | Brier score | Log loss |
|---|---|---|
| A — Base rate | 0.2502 | 0.6935 |
| B — Current-side-holds (naive) | 0.1600 | 0.4921 |
| C — Distance+time+vol only | 0.1500 | 0.4631 |
| D — Full model (+confluence) | 0.1495 | 0.4610 |

Market-level bootstrap, full model (D) vs. each baseline:

| Comparison | Brier gap | p-value |
|---|---|---|
| D vs. A (base rate) | -0.1006 (better) | <0.0001 |
| D vs. B (current-side-holds) | -0.0105 (better) | <0.0001 |
| D vs. C (distance+time+vol only) | -0.0005 (better) | 0.054 (not significant at conventional 0.05) |

**Reading this honestly: the real win is B → C** (adding actual distance
magnitude, time, and realized volatility to a logistic model, instead of a
coarse "which side leads, bucketed by minute" table) — a large, highly
significant improvement, consistent across **all 6 folds** without
exception. **C → D (adding the 7-indicator confluence engine) is not
statistically significant**, and in the per-fold breakdown D actually loses
to C in 2 of the 6 folds. Same lesson as the direction-signal work: this
particular indicator engine doesn't earn its complexity here either.

### Against the actual shipped `strike_distance.py` heuristic

The current heuristic (1.5-sigma rule) makes a call on only 26.3% of
checkpoints (10,176/38,674) — it abstains ("too close to call") the rest of
the time. On the calls it DOES make, it's accurate: **94.8%**. All three
fitted models match it almost exactly on those same easy rows (~95.0%) —
no real difference, because those are the obvious cases.

**The actual value is in the 73.7% of rows (28,498) the heuristic abstains
on** — the genuinely hard, "too close to call" cases:

| Model | Accuracy on the heuristic's abstained rows |
|---|---|
| Current-side-holds | 72.5% |
| Distance+time+vol only | 72.6% |
| Full model (+confluence) | 72.7% |

A fitted model turns most of that 73.7% "no signal" zone into a usable,
meaningfully-better-than-coin-flip probability (~73% vs. 50%) — and again,
the confluence indicators add nothing (72.6% → 72.7%, noise-level).

### Calibration (full model D)

| Predicted range | n | mean predicted | actual rate |
|---|---|---|---|
| 0.0–0.4 | 17,145 | 21.5% | 18.9% |
| 0.4–0.45 | 2,305 | 42.4% | 45.0% |
| 0.45–0.5 | 1,605 | 47.2% | 51.2% |
| 0.5–0.55 | 465 | 51.9% | 55.1% |
| 0.55–0.6 | 65 | 56.9% | 66.2% |
| 0.6–0.7 | 1,286 | 67.8% | 61.2% |
| 0.7–0.8 | 7,452 | 75.0% | 75.2% |
| 0.8–1.0 | 8,351 | 88.8% | 92.7% |

Broadly well-calibrated at the extremes (where most of the mass sits); the
0.5–0.6 range is noisier (small samples, 65-465 rows) and shows some drift —
worth more data before trusting predictions in that narrow band specifically.

## 3b. How good is this at ~30 seconds into the window?

Direct question: is there anything usable this early, or is it genuinely a
coin flip like intuition says? **Short answer: it is measurably not a coin
flip, but the edge is small compared to later in the window, and it doesn't
get any real boost from knowing what the previous window did.**

**Data resolution caveat first:** Coinbase's public feed is 1-minute bars,
so "30 seconds in" isn't literally reachable — the earliest evaluable point
is the first fully-closed 1-min bar, i.e. roughly **60–120 seconds** into
the window. That's the closest real proxy to what was asked for, and every
number below is that slice specifically (`checkpoint_min == 1` in
`results/features.csv`), not a new data pull — it reuses the exact
walk-forward folds and out-of-fold predictions from §2/§3.
(`scripts/checkpoint1_analysis.py`, `scripts/checkpoint1_specialized_model.py`)

**Is it distinguishable from a coin flip?** Yes, clearly, in the statistical
sense — but the honest word for the size of it is "small."

| | Brier score |
|---|---|
| Base rate / no information (constant guess) | 0.2502 |
| Full model (D), general fit, evaluated at checkpoint 1 | 0.2304 |
| Full model (D), evaluated at checkpoint ≥11 (late window, for scale) | 0.0757 |

Full model vs. a constant base-rate guess, market-level bootstrap: Brier gap
**-0.0196, p<0.0001** — real, not noise, at n=2,975 markets. But set next to
the late-window number (0.0757), checkpoint-1's 0.2304 makes clear almost
all of this model's power comes from elapsed time and distance, not from
whatever's knowable in the first couple minutes. One more way to see the
same thing: **"whichever side is barely ahead 60–120s in" is right 64.0% of
the time** (n=4,250, base rate 49.9%) — a real, useful-sounding number, but
nowhere close to the ~90%+ hit rate the same read gets late in the window.

**Does a model built specifically for this early checkpoint do any better
than just using the general one this early?** Yes, by a small but real
margin. A model trained (walk-forward, same folds) on checkpoint-1 rows only
scores **Brier 0.2263** vs. the general pooled model's **0.2304** at the same
checkpoint — gap **-0.0042, p=0.007**. Real, but 4,250 rows is a much
thinner training set than the full 55,245-row pool (1/13th the data), so
treat this as "worth using a dedicated early-checkpoint model if this
matters to you," not "free lunch."

**Does knowing what the PREVIOUS window did help?** No. Added two explicit
features — whether the immediately preceding window settled YES/NO, and its
own realized 15-min return — on top of the checkpoint-1-specialized model:
**Brier 0.2264 vs. 0.2263 without them, p=0.705 (no significant
improvement).** This isn't for lack of trying to capture "carryover" —
realized volatility, momentum, and EMA separation in the standard feature
set are already continuous, non-resetting calculations that span the window
boundary, so they already reflect anything from before the window opened
that the market's own price action would show. Explicitly telling the model
"the last one settled YES" or "the last one moved +0.3%" adds nothing on
top of that. Direct answer to the specific hypothesis: **no evidence of a
prior-window carryover effect**, at least not one these features can see.

**Practical implication:** the live paper trader (`paper_trader.py`) already
evaluates every 60 seconds regardless of how far into a window it is, so it
can and does fire this early sometimes — currently using the general model,
not a checkpoint-1-specialized one, with the existing minimum-distance gate
as the main protection against the noisiest early reads. Swapping in a
dedicated early-checkpoint model for the first couple minutes of a window is
a reasonable next step if this margin is worth capturing, but it's a small
edge on a thin training set — not implemented here without being asked.

## 4. Honest caveats — read before treating this as tradeable

- **Correlated checkpoints.** 13 checkpoints per market share one price
  path. Fold assignment and the bootstrap both operate at the market level
  to avoid pseudo-replication, but the raw row count (38,674) overstates
  independent information — think of this as ~2,975 independent trials with
  13 correlated looks each, not 38,674 independent ones.
- **No fee/spread model.** A 73% probability estimate is not automatically
  a profitable trade — Kalshi charges a per-contract trading fee and there's
  a real bid-ask spread. Neither is modeled here. Before sizing anything
  against these probabilities, the edge needs to be checked against actual
  achievable entry prices, not the theoretical fair probability.
  `trade_log.py`'s realized-vs-quoted accounting (already in the dashboard)
  is exactly the mechanism to measure this once you're trading — the
  quoted-only numbers there run ~6-7% optimistic on Kalshi historically.
- **Single 45-day window, one volatility regime.** All of this is from one
  recent stretch of BTC behavior. It hasn't been tested across a genuinely
  different regime (e.g., a much calmer or much more volatile period).
  Re-run `walk_forward.py` periodically as more history accumulates.
- **Coinbase spot vs. Kalshi's BRTI settlement index** — same caveat as
  the direction-signal work; they can diverge slightly, mattering most
  exactly in the near-strike cases this model is meant to help with.
- **Logistic regression only.** Deliberately kept simple — this project's
  own MLB model already tested and rejected gradient-boosted trees for a
  similar-sized problem. If a real edge exists in a nonlinear interaction
  logistic regression can't capture, a tree-based model might find more —
  but only worth trying once there's a specific reason to believe linear
  isn't enough, not by default.

## 5. What to do with this

- **Don't retire the direction/confluence signal claims based on this** —
  they're a separate question with a separate (negative) answer.
- **Do treat the distance+time+vol probability model as the more credible
  half of the bot** — it clears real baselines with a real, statistically
  significant margin, consistently across every fold, on a large real
  sample. It's the closest thing in this project to an actually validated
  edge so far.
- **Don't keep the 7 confluence indicators for this purpose** — they're not
  earning their complexity in this framing, same conclusion as before, from
  an independent test.
- **Next real step, if this direction is worth pursuing further:** paper-
  trade using the fitted probabilities against real quoted Kalshi prices
  (not the theoretical fair probability) for a few weeks, log it in
  `trade_log.py`, and see whether the edge survives real fees and spread.
  That's the only way to answer the question this report can't: is it
  profitable, not just accurate.

## 6. Reproducing this

```
cd scripts
python fetch_settled_markets.py 45 ../data/settled_markets.csv
python fetch_btc_data.py ../data/settled_markets.csv ../data/btc_1min.csv
python validate_against_indicators.py     # confirms signals_fast.py == real indicators.py
python build_features.py
python walk_forward.py
python compare_to_heuristic.py
python checkpoint1_analysis.py            # isolates the ~60-120s-into-window slice (see §3b)
python checkpoint1_specialized_model.py   # checkpoint-1-specific fit + prior-window feature test
```

Requires `numpy`, `pandas`, `scipy`, `scikit-learn`, `requests` — the last
two (`scipy`, `scikit-learn`) are research-only and deliberately NOT added
to the dashboard's `requirements.txt`, same precedent as the MLB project.

## 7. Files

```
data/settled_markets.csv          4,250 real settled KXBTC15M markets
data/btc_1min.csv                 65,096 Coinbase BTC-USD 1-min bars
scripts/fetch_settled_markets.py  Kalshi settled-market history fetcher (rate-limit hardened)
scripts/fetch_btc_data.py         Coinbase 1-min data fetcher for the matching range
scripts/signals_fast.py           Vectorized indicators.py/classifier.py, validated against the real modules
scripts/validate_against_indicators.py   Proves signals_fast.py matches real indicators.py on real data
scripts/build_features.py         Builds the 55,245-row no-lookahead checkpoint feature table
scripts/walk_forward.py           Chronological walk-forward harness, baselines, logistic models, bootstrap, calibration
scripts/compare_to_heuristic.py   Head-to-head vs. the actual shipped strike_distance.py heuristic
results/features.csv              Full feature table
results/walk_forward_fold_metrics.csv     Per-fold Brier/log-loss for all 4 approaches
results/walk_forward_pooled_predictions.csv  Every out-of-fold prediction, all 4 approaches
results/calibration_table.csv     Calibration bucket table
scripts/checkpoint1_analysis.py   Isolates checkpoint-1 (~60-120s-in) performance from existing results (§3b)
scripts/checkpoint1_specialized_model.py  Checkpoint-1-specific fit + prior-window carryover feature test (§3b)
results/checkpoint1_calibration.csv       Calibration bucket table, checkpoint-1 slice only
results/checkpoint1_specialized_predictions.csv  Out-of-fold predictions from the checkpoint-1-specialized models
```
