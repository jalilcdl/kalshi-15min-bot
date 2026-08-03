"""
Walk-forward-validated model: P(the entered side's contract moves 30%+ in
the buyer's favor before the window closes | side, entry price, minutes
remaining, realized volatility) -- fit on the 2,790 real historical trades
in exit_timing_trades.csv, same rigor as the settlement-probability model
(chronological folds, market-level bootstrap, calibration table).

IMPORTANT SCOPE NOTE: this dataset only contains the side the settlement-
probability model ALREADY chose to enter (via decide_entry()'s edge gate).
That means a model fit on it answers "given this side was already picked,
how likely is it to hit +30%" -- a confidence gate on top of the existing
side selection, NOT an unbiased test of "which side should I have picked."
For that second question (needed to actually replace side-selection, not
just add a filter on top of it), see side_selection_backtest.py, which
rebuilds a side-symmetric dataset (both YES and NO hypothetical outcomes
for every checkpoint) from the same underlying candlestick data.

Base rate here is ~78.6% (2,193/2,790) hit +30% -- NOT 50%. A constant
prediction near 0.786 already has a deceptively low Brier score with zero
skill, so "beats a coin flip" is not the bar; "beats the true base rate,
and beats an entry-price-only baseline" is.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "strike_probability" / "scripts"))
from walk_forward import make_folds, market_level_bootstrap  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
STRIKE_PROB = ROOT / "research" / "strike_probability"

FEATURES_PRICE_ONLY = ["entry_price"]
FEATURES_FULL = ["is_yes", "entry_price", "minutes_remaining", "realized_vol"]


def load_data():
    trades = pd.read_csv(RESULTS / "exit_timing_trades.csv")
    features = pd.read_csv(STRIKE_PROB / "results" / "features.csv")
    ck1 = features[features["checkpoint_min"] == 1][["ticker", "minutes_remaining", "realized_vol"]]
    markets = pd.read_csv(STRIKE_PROB / "data" / "settled_markets.csv", parse_dates=["close_time"])[["ticker", "close_time"]]

    df = trades.merge(ck1, on="ticker", how="left").merge(markets, on="ticker", how="left")
    df["is_yes"] = (df["side"] == "yes").astype(int)
    df["y"] = df["hit_close_30"].astype(int)
    df = df.dropna(subset=FEATURES_FULL + ["y", "close_time"])
    return df.sort_values("close_time").reset_index(drop=True)


def fit_eval_fold(train_df, test_df, feature_cols):
    if len(train_df) < 50 or len(test_df) == 0:
        return None
    clf = LogisticRegression(max_iter=2000)
    clf.fit(train_df[feature_cols], train_df["y"])
    return pd.Series(clf.predict_proba(test_df[feature_cols])[:, 1], index=test_df.index)


def main():
    df = load_data()
    base_rate = df["y"].mean()
    print(f"n={len(df)} trades, base rate (hit +30%) = {base_rate:.4f}\n")

    folds = make_folds(df, n_folds=6)
    all_rows = []
    for i, (train_tickers, test_tickers) in enumerate(folds):
        train_df = df[df["ticker"].isin(train_tickers)]
        test_df = df[df["ticker"].isin(test_tickers)].copy()

        train_rate = train_df["y"].mean()
        p_base = pd.Series(train_rate, index=test_df.index)
        p_price = fit_eval_fold(train_df, test_df, FEATURES_PRICE_ONLY)
        p_full = fit_eval_fold(train_df, test_df, FEATURES_FULL)
        if p_price is None or p_full is None:
            continue

        test_df["p_base"] = p_base
        test_df["p_price"] = p_price
        test_df["p_full"] = p_full
        all_rows.append(test_df)
        print(f"Fold {i+1}/{len(folds)}  n_test={len(test_df):4d}  train_rate={train_rate:.3f}  "
              f"Brier base={brier_score_loss(test_df['y'], p_base):.4f}  "
              f"price-only={brier_score_loss(test_df['y'], p_price):.4f}  "
              f"full={brier_score_loss(test_df['y'], p_full):.4f}")

    pooled = pd.concat(all_rows, ignore_index=False)
    print(f"\n=== Pooled out-of-fold ({len(pooled)} rows, {pooled['ticker'].nunique()} markets) ===")
    for col, name in [("p_base", "Base rate (train mean, no features)"),
                       ("p_price", "Entry price only"),
                       ("p_full", "Full (side + price + time + vol)")]:
        p = pooled[col].clip(1e-6, 1 - 1e-6)
        print(f"  {name:38s}  Brier={brier_score_loss(pooled['y'], p):.4f}  LogLoss={log_loss(pooled['y'], p):.4f}")

    print("\n=== Significance (market-level bootstrap) ===")
    actual, pval, _ = market_level_bootstrap(pooled, pooled["p_full"], pooled["p_base"], "full", "base")
    print(f"  Full model vs. base rate:        Brier gap={actual:+.5f}  p={pval:.4f}  "
          f"{'SIGNIFICANT' if pval < 0.05 and actual < 0 else 'not significant'}")
    actual2, pval2, _ = market_level_bootstrap(pooled, pooled["p_full"], pooled["p_price"], "full", "price_only")
    print(f"  Full model vs. entry-price-only:  Brier gap={actual2:+.5f}  p={pval2:.4f}  "
          f"{'SIGNIFICANT' if pval2 < 0.05 and actual2 < 0 else 'not significant'}")

    print("\n=== Calibration (full model) ===")
    pooled["p_bucket"] = pd.cut(pooled["p_full"], bins=[0, .6, .7, .75, .8, .85, .9, 1.0])
    calib = pooled.groupby("p_bucket", observed=True).agg(n=("y", "size"), mean_pred=("p_full", "mean"), actual_rate=("y", "mean"))
    print(calib.to_string())

    pooled.to_csv(RESULTS / "hit_target_model_predictions.csv", index=False)
    calib.to_csv(RESULTS / "hit_target_model_calibration.csv")
    print(f"\nSaved to {RESULTS}/hit_target_model_predictions.csv, hit_target_model_calibration.csv")


if __name__ == "__main__":
    main()
