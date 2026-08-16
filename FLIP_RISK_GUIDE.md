# Flip risk: how to actually use it

Short version: **flip risk is a "can I trust this read?" gauge, not a trade
signal.** Use it to decide whether to trade the window at all — not which side
to take.

---

## What the number means

> P(the price closes back across the strike at least once before this window ends)

High flip risk = the read on screen is unstable and likely to change.
Low flip risk = the current leader is probably still the leader at the close.

It says **nothing** about which side is mispriced. That is the whole reason the
fade idea failed.

---

## What it is NOT good for: fading the leader

Tested properly on 4,244 historical markets, walk-forward, with real spread and
fees. **Every threshold and every exit target lost money.**

| flip > | target | hit rate | avg P&L |
|---|---|---|---|
| none | 10¢ | 57.7% | **−$0.093** |
| 0.60 | 10¢ | 64.0% | **−$0.076** |
| 0.75 | 17.5¢ | 65.3% | **−$0.034** (best; CI still includes zero) |

Two reasons it can't work:

1. **The trailing side is already overpriced.** At the first read it costs 43.5%
   and wins 36.5%. You're paying 7 points over fair value before costs.
2. **The payoff is upside-capped, downside-full.** Of trades that missed the
   target, **0 of 1,256** went on to win at settlement — if your side recovers,
   its price passes your target on the way, so you always take the small win and
   never the big one, while losers cost the whole stake. You'd need a 68% hit
   rate at a 10¢ target; you get 58%.

**Why it felt like it was working:** the hit rate really is 65–77%. Most of
these trades do win. They just don't win enough to pay for the ones that don't.

---

## What it IS good for: a veto on following the leader

Same data, same rigour, costs included. Take the **leading** side, and use flip
risk to decide whether to bother:

| flip < | hit rate | avg P&L | n |
|---|---|---|---|
| no filter | 78.8% | +$0.025 | 2971 |
| **0.70** | 81.9% | **+$0.042** | 1968 |
| **0.60** | 83.5% | **+$0.050** | 1079 |
| 0.50 | 85.2% | +$0.068 | 533 |
| 0.40 | 83.8% | +$0.069 | 222 |

Positive in all six folds. Stricter veto = better per trade, fewer trades.
Below 0.40 the samples get thin — don't read much into those rows.

**Holding to settlement beat taking profit early** (+$0.064 vs +$0.050 at
flip < 0.60). On the favoured side, an exit target caps your winner while a
loser still costs the full stake. The 10–15¢ exit isn't what makes this work.

---

## How to read it in practice

| Flip risk | What it means | Reasonable action |
|---|---|---|
| **< 0.40** | Read is solid | Best setups. Follow the leader. |
| **0.40–0.60** | Read is fairly stable | Still positive. Follow the leader. |
| **0.60–0.70** | Getting shaky | Marginal. Smaller or skip. |
| **> 0.70** | Read is unreliable | **Sit out.** Do not fade — just don't trade. |

The single biggest change from what you've been doing: when flip risk is **high**,
that is a signal to **stay out**, not to bet against the leader.

---

## Before you put money on this

Three honest caveats:

1. **The best-looking edge sits on the least trustworthy data.** It's largest at
   2 minutes left (+13 points). That's exactly where the live book has been
   observed one-sided and quoting 0.005 on contracts that settled at $1.00. The
   backtest assumes you get filled at the quoted price. Late in a window, you
   often can't. Treat early-window entries as the trustworthy ones.

2. **This is historical, not live.** Nothing here has been traded. A ~7 point
   mispricing in a niche 15-minute market can be a real inefficiency or a
   liquidity artifact, and it can disappear once someone trades it.

3. **It's a different bet from the automated bot.** The bot trades a validated
   settlement-probability edge with a +35% exit. This is a separate idea and
   should be paper-traded on its own before it's trusted or combined.

---

*Full methodology and numbers: `research/fade_flip/README.md`.
Reproduce: `python research/fade_flip/scripts/build_follow_dataset.py`.*
