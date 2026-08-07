"""
Live inference for the strike-crossing ("flip risk") probability.

Answers a DIFFERENT question from strike_probability.predict_p_yes(). That one
is "which side wins at close." This one is "how stable is the read on screen":

    P(at least one later 1-min CLOSE lands on the opposite side of the strike
       before this window closes | distance, time left, realized volatility)

Useful precisely in the sit-out state, where a read is displayed but no trade
clears the entry gate and the only real question is whether the number will
hold. See research/crossing_probability/README.md for the full validation.

Out-of-sample (6 walk-forward folds, 4,250 markets, 38,670 test rows):
    Brier 0.1716 | AUC 0.8098 | max decile calibration gap 0.054 (mean 0.018)
Beats a global base rate, a time-only lookup, and the closed-form driftless-BM
reflection probability, all at p<0.001 on a market-level bootstrap.

Two features only -- dist_over_reachable and minutes_remaining. That is not
laziness: an 8-feature-richer variant scored Brier 0.1706 / AUC 0.8116, i.e.
0.001 better and 0.002 worse respectively, while requiring five indicator
fields to be newly plumbed into live code and separately proven equivalent to
their research counterparts. realized_vol (which feeds dist_over_reachable) is
already verified equivalent live-vs-research to 5e-16 and already consumed by
the settlement model, so this adds zero new train/serve surface. Reasoning in
research/crossing_probability/scripts/fit_final_model.py.

CAVEAT worth repeating wherever this number is shown: the label counts 1-min
CLOSES on the far side. The live dashboard reads a real-time ticker, so a read
can visibly wobble on a sub-minute move that never produced such a close. The
intra-bar variant of the label runs ~7 points higher (44.1% vs 37.0% base
rate), so this number is a mild UNDER-estimate of apparent flipping. Describe
it as "price closes back across the strike," not "your read wobbles."
"""
from math import sqrt
from pathlib import Path

import joblib
import pandas as pd

_MODEL_PATH = Path(__file__).resolve().parent / "crossing_prob_model.pkl"

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{_MODEL_PATH} not found -- run research/crossing_probability/"
                "scripts/fit_final_model.py first."
            )
        _bundle = joblib.load(_MODEL_PATH)
    return _bundle


def model_metadata() -> dict:
    b = _load()
    return {k: v for k, v in b.items() if k != "pipeline"}


def predict_flip_prob(price: float, strike: float, minutes_remaining: float,
                      realized_vol_pct: float) -> float | None:
    """Probability the price closes back across `strike` before the window ends.

    Returns None when the inputs can't support a meaningful answer rather than
    returning a confident-looking number:
      - no strike published yet,
      - window already over (nothing left to cross in),
      - realized vol ~0, which makes dist_over_reachable explode or divide by
        zero. A flat tape genuinely carries no crossing information and the
        honest output is "unknown", not "0%".
    """
    if strike is None or price is None or minutes_remaining is None:
        return None
    if minutes_remaining <= 0 or realized_vol_pct is None or realized_vol_pct <= 1e-9:
        return None

    distance_pct = (price - strike) / strike * 100.0
    reachable = realized_vol_pct * sqrt(minutes_remaining)
    if reachable <= 1e-9:
        return None
    dist_over_reachable = abs(distance_pct) / reachable

    b = _load()
    X = pd.DataFrame([{"dist_over_reachable": dist_over_reachable,
                       "minutes_remaining": minutes_remaining}])[b["features"]]
    return float(b["pipeline"].predict_proba(X)[0, 1])
