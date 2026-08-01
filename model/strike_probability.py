"""
Loads the fitted strike-probability model (distance + time + realized
volatility only -- see research/strike_probability/README.md for why the
7-indicator confluence engine was deliberately left out) and exposes a
single predict_p_yes() function for live use.

Refit periodically as more settled-market history accumulates:
    cd research/strike_probability/scripts
    python fetch_settled_markets.py 45 ../data/settled_markets.csv
    python fetch_btc_data.py ../data/settled_markets.csv ../data/btc_1min.csv
    python build_features.py
    python fit_final_model.py
"""
import json
from math import sqrt
from pathlib import Path

import joblib
import pandas as pd

_MODEL_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _MODEL_DIR / "strike_prob_model.pkl"
_META_PATH = _MODEL_DIR / "strike_prob_model_meta.json"

_model = None
_meta = None


def _load():
    global _model, _meta
    if _model is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{_MODEL_PATH} not found -- run research/strike_probability/scripts/"
                "fit_final_model.py first."
            )
        _model = joblib.load(_MODEL_PATH)
        _meta = json.loads(_META_PATH.read_text())
    return _model, _meta


def model_metadata() -> dict:
    _, meta = _load()
    return meta


def compute_features(price: float, strike: float, minutes_remaining: float,
                     realized_vol_pct: float) -> dict:
    """Same feature definitions as research/strike_probability/scripts/build_features.py."""
    distance_pct = (price - strike) / strike * 100.0
    reachable = realized_vol_pct * sqrt(max(minutes_remaining, 0.0))
    dist_over_reachable = abs(distance_pct) / reachable if reachable > 1e-9 else 0.0
    return dict(
        distance_pct=distance_pct,
        minutes_remaining=minutes_remaining,
        realized_vol=realized_vol_pct,
        dist_over_reachable=dist_over_reachable,
        current_side_leading=1 if distance_pct >= 0 else 0,
    )


def predict_p_yes(price: float, strike: float, minutes_remaining: float,
                  realized_vol_pct: float) -> float:
    """Probability the market settles YES (price >= strike at close)."""
    model, meta = _load()
    feats = compute_features(price, strike, minutes_remaining, realized_vol_pct)
    row = pd.DataFrame([{f: feats[f] for f in meta["features"]}])
    return float(model.predict_proba(row)[0][1])
