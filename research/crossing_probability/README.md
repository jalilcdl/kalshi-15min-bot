# Strike-crossing ("will this read flip?") probability

**Status: SHIPPED** — "Flip risk" badge on the Scoreboard market card.

Validated and calibrated, with one framing caveat (§6) and one honest deflation
of what the model actually is (§4). **The shipped model is the TWO-feature
variant**, not the 10-feature one an earlier draft of this file leaned toward;
§4 explains why the extra eight were dropped.

## 1. The question, and how it differs from the settlement model

`model/strike_probability.py` answers **"which side wins at close"** — P(YES).
This answers a different question: **"how stable is the read I'm looking at
right now."**

Formally, at a checkpoint minute inside a live window, with price `P` and
strike `K`:

```
current_side = sign(P - K)
LABEL = 1 if ANY later 1-min close in (now, window close] is on the opposite side
```

That is a first-passage / barrier-crossing question, not a terminal-value
question. It is most useful exactly where the settlement model is least
actionable — the sit-out state, where a read is displayed but no trade clears
the entry gate, and the only real question is "is this number going to hold?"

Two label variants were built, because "crossed" is genuinely ambiguous on
1-min bars and the choice moves the base rate by 7 points:

| label | definition | base rate |
|---|---|---|
| **`flip_close`** (shipped) | a later 1-min **close** lands on the other side | **37.0%** |
| `flip_touch` | a later bar's high/low **range reaches** the strike | 44.1% |

`flip_close` is shipped because it is the one that was actually calibrated and
tested end to end, and because it matches what the card is seen to do: it
renders one spot price per refresh, so the displayed side flips when a price
prints on the other side, not when a wick grazes the strike.

## 2. Data

- 55,241 checkpoint rows across **4,250 settled markets** (minutes 1–13 of each
  window), from `research/strike_probability/data/`.
- **No lookahead**: features at minute `ck` use only bars `<= open + ck`; the
  label uses only bars strictly after. Feature definitions are lifted verbatim
  from the settlement model's `build_features.py`.
- The 13 checkpoints of one window share a price path and are **not
  independent**. Handled explicitly: whole markets are assigned to folds, and
  significance is bootstrapped by resampling whole markets.

Raw time decay, before any modelling:

| minutes into window | 1 | 3 | 5 | 7 | 9 | 11 | 13 |
|---|---|---|---|---|---|---|---|
| flip_close rate | 62.4% | 53.5% | 44.3% | 36.0% | 28.8% | 21.2% | 14.0% |

## 3. Walk-forward results

6 chronological expanding folds, 38,670 out-of-sample rows. Baselines chosen so
the fitted model has to actually earn its place. **The shipped row is
`logistic (2-feature)`** — the 10-feature row is kept for comparison only:

| model | Brier ↓ | log loss ↓ | AUC ↑ | max decile calib gap |
|---|---|---|---|---|
| **logistic, 2-feature (SHIPPED)** | **0.1716** | **0.5122** | **0.8098** | **0.054** |
| logistic, 10-feature (not shipped, §4) | 0.1706 | 0.5082 | 0.8116 | 0.032 |
| reflection (closed form, nothing fitted) | 0.1870 | 0.5692 | 0.8065 | 0.183 |
| time-only lookup | 0.2117 | 0.6114 | 0.6865 | 0.018 |
| global base rate | 0.2349 | 0.6627 | 0.4942 | — |

Market-level bootstrap (1,000 resamples of whole markets), Brier improvement of
the fitted logistic over each baseline — all decisive:

| vs | mean improvement | 95% CI | p |
|---|---|---|---|
| base rate | +0.0642 | [+0.0607, +0.0676] | <0.001 |
| time-only | +0.0411 | [+0.0379, +0.0444] | <0.001 |
| reflection | +0.0165 | [+0.0139, +0.0190] | <0.001 |

Calibration of the **shipped** model (deciles, out-of-sample): **max gap 5.4
pts, mean gap 1.8 pts.** The worst decile predicts 72.4% where the actual rate
is 77.8%; the best-behaved middle deciles are within 0.2 pts. Good enough to
display a number and mean it, and the residual is a mild *under*-statement at
the high end rather than overconfidence.

## 4. The honest deflation: this is barrier physics, and why 8 features were dropped

`reflection` above is the textbook driftless-Brownian result
`P(touch) = 2·Φ(−d/(σ√T))`, with **nothing fitted**. The existing feature
`dist_over_reachable` *is* `d/(σ√T)`, so it drops straight in.

It scores **AUC 0.8065 vs the fitted model's 0.8098–0.8116** — statistically the
same ranking power — but Brier 0.1870, because it is badly miscalibrated: it
over-predicts crossing by up to **18 points** in the upper deciles (real BTC is
not driftless BM, and 1-min close sampling misses intra-bar paths).

| model | Brier | AUC | max calib gap |
|---|---|---|---|
| logistic, all 10 features | 0.1706 | 0.8116 | 0.032 |
| **`dist_over_reachable` + `minutes_remaining` (shipped)** | **0.1716** | **0.8098** | **0.054** |
| reflection + Platt scaling | 0.1733 | 0.8061 | 0.055 |
| reflection, raw | 0.1870 | 0.8065 | 0.183 |

**The signal is almost entirely `distance / (vol · √time)`.** The other eight
features buy 0.001 Brier and *cost* 0.002 AUC.

They were dropped for a concrete engineering reason, not taste. `indicators.Signals`
exposes only `realized_vol_pct`, `window_delta_pct`, `ema_sep_pct` — **none** of
`rsi_val`, `mom`, `accel`, `vol_ratio`, `net_ticks`. Shipping the 10-feature model
meant newly surfacing five fields and separately proving each equivalent to its
research counterpart. That equivalence is precisely where this project has been
bitten twice (the dashboard/Telegram contradiction and the stale-price incident
were both implementation/train-serve skew). Measured directly:

| feature | live vs research | diff |
|---|---|---|
| `realized_vol` | identical | 5e-16 |
| `window_delta` | identical | 0 |
| `ema_sep` | negligible (EMA seeding) | 1e-5 |

`realized_vol` is already proven equivalent and already consumed live by the
settlement model; `minutes_remaining` is exact arithmetic. So the shipped pair
adds **zero new train/serve surface**, where the 10-feature version would have
added five unverified ones for 0.001 Brier. The calibration cost (5.4 vs 3.2 pts
max gap, 1.8 vs 1.7 mean) is real and is stated here rather than buried.

Shipped coefficients (standardized): `dist_over_reachable` **−2.039**,
`minutes_remaining` **+0.221**.

## 5. Does "crossing" actually mean "the read flips"?

Checked, not assumed. Over 8,000 sampled checkpoints, `sign(P(YES) − 0.5)`
matched `sign(price − strike)` **99.88%** of the time. The 10 disagreements sit
at a median distance of 0.0007% from the strike — the settlement model hedging
essentially *at* the strike, not a different notion of "side". The label and the
UI claim are the same event.

## 6. Caveat that matters for the live number

The label is built from **1-min closes**, but the live dashboard reads a
**real-time ticker** (see the 2026-08-06 feed fix). A live read can therefore
visibly flip on a sub-minute move that never produced a 1-min close on the other
side — an event this label does not count.

Direction of the bias is knowable: `flip_touch`, which counts intra-bar reaches,
runs **7 points higher** than `flip_close` (44.1% vs 37.0%). So the displayed
number is, if anything, a mild **under**-estimate of how often the read will
appear to flip. Better that way round than the reverse — and it is why the badge
reads "chance price closes back across the strike before this window ends"
rather than "your read wobbles."

## 7. What the number looks like

Shipped model, flip probability by minutes left × distance in reachable-move
units:

| mins left | 0.15 | 0.25 | 0.50 | 1.00 | 2.00 |
|---|---|---|---|---|---|
| 2 | 56% | 51% | 39% | 19% | 3% |
| 5 | 61% | 56% | 43% | 22% | 4% |
| 9 | 66% | 62% | 49% | 26% | 5% |
| 13 | 71% | 67% | 55% | 31% | 6% |

Applied to the card that prompted this (strike $64,931, 9.2 min left, read
P(NO) 66%), the price/vol combinations consistent with that read give a flip
probability of **49–61%** — i.e. that particular read was a coin flip, which is
precisely the thing the card was not saying.

Badge bands are descriptive labels on a continuous number, not decision rules:
≥55% "high" (red), ≥30% "moderate" (orange), else "low" (green). The percentage
is the content.

The badge returns **nothing at all** — rather than a confident "0%" — when the
strike is unpublished, the window is over, or realized vol is ~0. A flat tape
genuinely carries no crossing information and the honest output is silence.

## 8. Reproduce

```
cd scripts
python build_crossing_dataset.py     # -> results/crossing_features.csv
python walk_forward_crossing.py      # -> model_comparison.csv, calibration.csv, flip_rate_grid.csv
python sanity_checks.py              # -> check_recalibration.csv, check_read_flip_alignment.csv
python fit_final_model.py            # -> model/crossing_prob_model.pkl (2-feature, shipped)
```
