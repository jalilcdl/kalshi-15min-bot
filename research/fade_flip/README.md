# Fade-the-leader on high flip risk — validation

**Verdict: the data does not support this strategy. Every threshold/target
combination tested loses money, and the reason is structural rather than a
matter of tuning.**

## The rule tested

At a read, if the flip-risk model says the current read is unstable
(P(price closes back across the strike) > X), buy the side that is currently
*losing*. Exit when up Y cents; otherwise hold to settlement.

## Method

- **4,244 markets** with complete data; 2,971 usable at the first read after
  filtering degenerate quotes. 241,962 simulated trades
  (13 checkpoints × 7 targets).
- **Walk-forward**: the flip model is refit inside each of 6 chronological
  expanding folds on training markets only; the strategy is scored only on
  test-fold rows. Same fold convention as `walk_forward_crossing.py`.
- **Prices**: buy YES at `yes_ask`, buy NO at `1 − yes_bid`; exit by selling at
  the opposite side of the book. Both legs cross the spread, so both pay the
  taker fee from `fees.py` (~2.5¢ round trip, 5.8% of the average stake).
- **Exits use 1-minute closes**, never intrabar highs — filling at a spike
  assumes liquidity that may not have existed.

## Results

Every cell in the grid is negative, net of fees, at the first read:

| flip > | target | n | hit% | avg net |
|---|---|---|---|---|
| 0.00 (no filter) | 0.10 | 2971 | 57.7% | **−$0.0926** |
| 0.60 | 0.10 | 1892 | 64.0% | **−$0.0759** |
| 0.70 | 0.175 | 1003 | 60.9% | **−$0.0598** |
| 0.75 | 0.175 | 389 | 65.3% | **−$0.0337** (best cell) |

The best cell's 95% CI (market-level bootstrap) is [−0.0731, +0.0020] — it
includes zero, on 389 trades.

**Out-of-sample parameter selection** (pick the best cell on folds 1..k, score
on fold k+1): pooled **−$0.0402 per trade over 649 trades**. The apparent
improvement at high thresholds does not survive being chosen honestly.

## Why it loses — two compounding reasons

**1. The payoff is capped on the upside and full on the downside.**
A winner is capped at +Y. A loser gives back the entire stake, because if the
faded side had recovered, its price would have risen toward $1.00 and passed +Y
on the way. This is confirmed exactly in the data: of trades that missed the
target, **0 of 1,256 went on to win at settlement**. Required vs actual hit rate:

| target | need | actual | gap |
|---|---|---|---|
| 0.05 | 75.1% | 64.1% | −11.0 pts |
| 0.10 | 68.5% | 57.7% | −10.7 pts |
| 0.15 | 63.4% | 53.8% | −9.7 pts |
| 0.20 | 58.6% | 49.2% | −9.4 pts |

**2. The faded side is systematically overpriced.** At the first read the
trailing side costs 43.5% but wins only **36.5%** of the time — a −7.0 point
edge *against* the fade before any friction. A stricter flip filter narrows that
gap monotonically but never closes it:

| flip > | implied | actual | edge |
|---|---|---|---|
| 0.00 | 43.5% | 36.5% | −7.0 |
| 0.60 | 46.8% | 40.9% | −5.9 |
| 0.70 | 48.7% | 45.2% | −3.5 |
| 0.75 | 50.2% | 47.3% | −2.9 |

There is no data above flip ≈ 0.80, so no threshold exists that reaches parity.

## Time remaining

Tested at every checkpoint. The strategy is negative at all of them, with 95%
CIs excluding zero for minutes 14 down to 5 (threshold 0.60, target 0.10).
Entering later is *less* bad — −$0.076 at 14 minutes vs −$0.036 at 4 minutes —
but the trend is toward zero, never through it, and the late-window cells have
small samples. Time-remaining is not a filter that rescues the rule.

## The flip model itself is fine

The signal does carry real information: it orders trades correctly, and every
metric improves monotonically with the threshold. The problem is not the model —
it is that the model is being used to pick a side whose price already more than
reflects the instability.

## Incidental observation, NOT a validated finding

The mirror of this result is that the *leading* side wins 63.5% while costing
56.5% — a +7.0 point edge for following rather than fading. That number is
gross: it ignores the spread paid to buy the leader and the fees on that trade,
both of which would erode it. It is a hypothesis worth its own test, not a
recommendation.

## Reproduce

```
python research/fade_flip/scripts/build_fade_dataset.py   # ~4 min
python research/fade_flip/scripts/analyse_fade.py
```
