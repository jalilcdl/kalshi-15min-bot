"""
Two follow-up questions for the checkpoint-1 (~60-120s into window) slice:

1. Does a model TRAINED specifically on checkpoint-1 data (rather than the
   general pooled model from walk_forward.py, which is trained on all 13
   checkpoints and evaluated at checkpoint 1 in checkpoint1_analysis.py) do
   any better? Only ~4,250 rows here (1/13th of the full dataset), so this
   is a real small-sample test, not a free upgrade.

2. Is there any usable signal from the PRECEDING window carrying over --
   its own realized return, and whether it settled YES or NO? The
   continuous (non-window-resetting) indicator features already carry some
   of this implicitly (realized_vol, momentum, EMA separation are all
   rolling calculations that span the window boundary) -- this tests
   whether explicit prior-window features add anything on top of that.

Same walk-forward discipline as before: chronological folds, whole markets
never split, market-level bootstrap for significance.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from walk_forward import make_folds, market_level_bootstrap  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATA = Path(__file__).resolve().parent.parent / "data"

FEATURES_C1 = ["distance_pct", "minutes_remaining", "realized_vol",
               "dist_over_reachable", "current_side_leading"]
FEATURES_D1 = FEATURES_C1 + [
    "ema_sep_pct", "rsi_val", "vol_ratio", "mom", "accel", "net_ticks",
    "direction", "confidence", "window_delta_pct",
    "trend_strong", "trend_weak", "momentum_bullish", "momentum_bearish",
]
FEATURES_E1 = FEATURES_D1 + ["prior_result", "prior_window_return_pct"]


def build_prior_window_features(markets: pd.DataFrame) -> pd.DataFrame:
    """For each market, find the immediately preceding window (its close_time
    == this market's open_time, since KXBTC15M windows are back-to-back) and
    attach its result and realized return."""
    markets = markets.sort_values("close_time").reset_index(drop=True)
    by_close = markets.set_index("close_time")

    prior_result, prior_return = [], []
    for row in markets.itertuples():
        prior = by_close.index.get_loc(row.open_time) if row.open_time in by_close.index else None
        if prior is None:
            prior_result.append(np.nan)
            prior_return.append(np.nan)
            continue
        p = markets.iloc[prior]
        prior_result.append(1 if p["result"] == "yes" else 0)
        prior_return.append((row.strike - p["strike"]) / p["strike"] * 100.0)
    markets = markets.copy()
    markets["prior_result"] = prior_result
    markets["prior_window_return_pct"] = prior_return
    return markets


def fit_eval_fold(train_df, test_df, feature_cols):
    train = train_df.dropna(subset=feature_cols + ["y"])
    test = test_df.dropna(subset=feature_cols)
    if len(train) < 50 or len(test) == 0:
        return None
    clf = LogisticRegression(max_iter=2000)
    clf.fit(train[feature_cols], train["y"])
    return pd.Series(clf.predict_proba(test[feature_cols])[:, 1], index=test.index)


def main():
    markets = pd.read_csv(DATA / "settled_markets.csv", parse_dates=["open_time", "close_time"])
    markets = build_prior_window_features(markets)
    n_missing_prior = markets["prior_result"].isna().sum()
    print(f"{len(markets)} markets, {n_missing_prior} with no matched prior window "
          f"(gaps in the settled-market chain, e.g. Kalshi downtime)")

    features = pd.read_csv(RESULTS / "features.csv", parse_dates=["close_time"])
    ck1 = features[features["checkpoint_min"] == 1].copy()
    ck1 = ck1.merge(markets[["ticker", "prior_result", "prior_window_return_pct"]], on="ticker", how="left")
    print(f"Checkpoint-1 rows with prior-window features: {ck1['prior_result'].notna().sum()}/{len(ck1)}\n")

    folds = make_folds(ck1, n_folds=6)
    all_rows = []
    for i, (train_tickers, test_tickers) in enumerate(folds):
        train_df = ck1[ck1["ticker"].isin(train_tickers)]
        test_df = ck1[ck1["ticker"].isin(test_tickers)].copy()

        pC1 = fit_eval_fold(train_df, test_df, FEATURES_C1)
        pD1 = fit_eval_fold(train_df, test_df, FEATURES_D1)
        pE1 = fit_eval_fold(train_df, test_df, FEATURES_E1)
        if pC1 is None or pD1 is None or pE1 is None:
            continue
        common = test_df.index.intersection(pC1.index).intersection(pD1.index).intersection(pE1.index)
        sub = test_df.loc[common].copy()
        sub["pC1"], sub["pD1"], sub["pE1"] = pC1.loc[common], pD1.loc[common], pE1.loc[common]
        all_rows.append(sub)

    pooled = pd.concat(all_rows, ignore_index=False)
    print(f"Pooled out-of-fold: {len(pooled)} rows, {pooled['ticker'].nunique()} markets\n")

    print("=== Checkpoint-1-SPECIALIZED models (trained only on checkpoint-1 data) ===")
    for col, name in [("pC1", "C1: Distance+time+vol only (checkpoint-1-specific fit)"),
                       ("pD1", "D1: + confluence indicators (checkpoint-1-specific fit)"),
                       ("pE1", "E1: + explicit prior-window result/return")]:
        p = pooled[col].clip(1e-6, 1 - 1e-6)
        print(f"  {name:56s}  Brier={brier_score_loss(pooled['y'], p):.4f}  "
              f"LogLoss={log_loss(pooled['y'], p):.4f}")

    print("\n=== Does adding explicit prior-window info (E1) beat the general confluence set (D1)? ===")
    actual, p_value, _ = market_level_bootstrap(pooled, pooled["pE1"], pooled["pD1"], "E1", "D1")
    verdict = "helps" if (p_value < 0.05 and actual < 0) else "no significant improvement"
    print(f"  Brier gap={actual:+.5f}  bootstrap p={p_value:.4f}  -> {verdict}")

    print("\n=== Does the checkpoint-1-specialized fit (D1) beat the general pooled model at checkpoint 1? ===")
    general = pd.read_csv(RESULTS / "walk_forward_pooled_predictions.csv")
    general_ck1 = general[general["checkpoint_min"] == 1][["ticker", "pD"]]
    merged = pooled[["ticker", "y", "pD1"]].merge(general_ck1, on="ticker", how="inner")
    if len(merged):
        actual2, p2, _ = market_level_bootstrap(merged, merged["pD1"], merged["pD"], "D1_specialized", "D_general")
        verdict2 = "specialized fit helps" if (p2 < 0.05 and actual2 < 0) else "no significant difference"
        print(f"  n={len(merged)} common markets. Brier gap (specialized - general)={actual2:+.5f}  "
              f"bootstrap p={p2:.4f}  -> {verdict2}")

    pooled.to_csv(RESULTS / "checkpoint1_specialized_predictions.csv", index=False)


if __name__ == "__main__":
    main()
