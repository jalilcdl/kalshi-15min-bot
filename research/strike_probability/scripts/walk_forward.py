"""
Walk-forward evaluation of the strike-probability model against three
baselines, on real settled KXBTC15M markets. Chronological folds only --
never shuffled -- and whole MARKETS (all ~13 of their checkpoint rows) are
assigned to a fold together, never split, since those rows are correlated
(same price path) and splitting them would both leak and double-count.

Baselines, weakest to strongest prior:
  A. Base rate      -- P(YES) = train-set frequency of YES, bucketed by
                        checkpoint minute. Ignores price entirely -- the floor.
  B. Current side, time-bucketed -- P(current side wins) = train-set
                        empirical accuracy of "current side holds," bucketed
                        by checkpoint minute. This is the naive baseline
                        backtest_strike.py's own README says any real model
                        must beat, not a coin flip.
  C. Distance+time+vol only -- logistic regression on the "options-pricing"
                        features alone (distance to strike, time remaining,
                        realized vol, distance-over-reachable-move, current
                        side). No confluence indicators at all.
  D. Full model      -- C's features plus the 7-indicator confluence
                        engine's outputs (EMA separation, RSI, volume,
                        momentum, acceleration, tick trend, direction,
                        confidence, trend/momentum labels).

D vs C isolates whether the indicator engine adds anything beyond pure
distance/time/vol. D and C vs B isolates whether either beats the naive
baseline the bot's own docs already warn about.

Metrics: Brier score (mean squared error of the probability, lower=better)
and log loss, both proper scoring rules that reward calibration, not just
accuracy. A market-level (not row-level) paired bootstrap gives a fair
significance check that respects the correlation between a market's own
checkpoints.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

RNG = np.random.default_rng(11)
N_BOOT = 2000

FEATURES_C = ["distance_pct", "minutes_remaining", "realized_vol",
              "dist_over_reachable", "current_side_leading"]
FEATURES_D = FEATURES_C + [
    "ema_sep_pct", "rsi_val", "vol_ratio", "mom", "accel", "net_ticks",
    "direction", "confidence", "window_delta_pct",
    "trend_strong", "trend_weak", "momentum_bullish", "momentum_bearish",
]


def load_features():
    path = Path(__file__).resolve().parent.parent / "results" / "features.csv"
    df = pd.read_csv(path, parse_dates=["close_time"])
    df = df.sort_values(["close_time", "checkpoint_min"]).reset_index(drop=True)
    return df


def make_folds(df, n_folds=6, min_train_frac=0.30):
    """Chronological expanding-window folds, split on market close_time so
    every checkpoint of a given market lands in exactly one fold."""
    tickers_by_time = df.drop_duplicates("ticker")[["ticker", "close_time"]].sort_values("close_time")
    n_markets = len(tickers_by_time)
    start_at = int(n_markets * min_train_frac)
    test_boundaries = np.linspace(start_at, n_markets, n_folds + 1).astype(int)

    folds = []
    for i in range(n_folds):
        train_tickers = set(tickers_by_time["ticker"].iloc[:test_boundaries[i]])
        test_tickers = set(tickers_by_time["ticker"].iloc[test_boundaries[i]:test_boundaries[i + 1]])
        if not test_tickers:
            continue
        folds.append((train_tickers, test_tickers))
    return folds


def baseline_A(train_df, test_df):
    rate_by_ck = train_df.groupby("checkpoint_min")["y"].mean()
    overall = train_df["y"].mean()
    return test_df["checkpoint_min"].map(rate_by_ck).fillna(overall).to_numpy()


def baseline_B(train_df, test_df):
    correct = (train_df["current_side_leading"] == train_df["y"]).astype(int)
    acc_by_ck = correct.groupby(train_df["checkpoint_min"]).mean()
    overall_acc = correct.mean()
    p_current_side = test_df["checkpoint_min"].map(acc_by_ck).fillna(overall_acc)
    # Convert "P(current side wins)" to "P(YES)" depending on which side is current.
    p_yes = np.where(test_df["current_side_leading"] == 1, p_current_side, 1 - p_current_side)
    return np.clip(p_yes, 1e-6, 1 - 1e-6)


def fit_logistic(train_df, test_df, feature_cols):
    train = train_df.dropna(subset=feature_cols + ["y"])
    test = test_df.dropna(subset=feature_cols)
    if len(train) < 50 or len(test) == 0:
        return None, test.index
    clf = LogisticRegression(max_iter=2000)
    clf.fit(train[feature_cols], train["y"])
    p = clf.predict_proba(test[feature_cols])[:, 1]
    return pd.Series(p, index=test.index), test.index


def market_level_bootstrap(test_df, p1, p2, name1, name2, n_boot=N_BOOT):
    """Paired bootstrap over MARKETS (not rows) on the Brier-score gap
    between two prediction sets, respecting within-market correlation."""
    tmp = test_df.copy()
    tmp["p1"], tmp["p2"] = p1, p2
    tmp["sq1"] = (tmp["p1"] - tmp["y"]) ** 2
    tmp["sq2"] = (tmp["p2"] - tmp["y"]) ** 2
    by_market = tmp.groupby("ticker")[["sq1", "sq2"]].mean()
    tickers = by_market.index.to_numpy()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sample = RNG.choice(tickers, size=len(tickers), replace=True)
        s = by_market.loc[sample]
        diffs[i] = s["sq1"].mean() - s["sq2"].mean()
    actual = by_market["sq1"].mean() - by_market["sq2"].mean()
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return actual, p_value, diffs


def main():
    df = load_features()
    folds = make_folds(df)
    print(f"{len(df)} checkpoint-rows, {df['ticker'].nunique()} markets, {len(folds)} walk-forward folds\n")

    all_test_rows = []
    fold_metrics = []
    for i, (train_tickers, test_tickers) in enumerate(folds):
        train_df = df[df["ticker"].isin(train_tickers)]
        test_df = df[df["ticker"].isin(test_tickers)].copy()

        pA = baseline_A(train_df, test_df)
        pB = baseline_B(train_df, test_df)
        pC, idx_c = fit_logistic(train_df, test_df, FEATURES_C)
        pD, idx_d = fit_logistic(train_df, test_df, FEATURES_D)
        if pC is None or pD is None:
            continue

        common_idx = test_df.index.intersection(idx_c).intersection(idx_d)
        sub = test_df.loc[common_idx].copy()
        sub["pA"] = pd.Series(pA, index=test_df.index).loc[common_idx]
        sub["pB"] = pd.Series(pB, index=test_df.index).loc[common_idx]
        sub["pC"] = pC.loc[common_idx]
        sub["pD"] = pD.loc[common_idx]

        row = dict(
            fold=i, n_train_markets=len(train_tickers), n_test_markets=len(test_tickers),
            n_test_rows=len(sub), test_start=test_df["close_time"].min(), test_end=test_df["close_time"].max(),
        )
        for label in ["A", "B", "C", "D"]:
            p = sub[f"p{label}"].clip(1e-6, 1 - 1e-6)
            row[f"brier_{label}"] = brier_score_loss(sub["y"], p)
            row[f"logloss_{label}"] = log_loss(sub["y"], p)
        fold_metrics.append(row)
        all_test_rows.append(sub)
        print(f"Fold {i+1}/{len(folds)}  test={row['test_start'].date()}..{row['test_end'].date()}  "
              f"n_markets={row['n_test_markets']:4d}  "
              f"Brier A={row['brier_A']:.4f} B={row['brier_B']:.4f} "
              f"C={row['brier_C']:.4f} D={row['brier_D']:.4f}")

    fm = pd.DataFrame(fold_metrics)
    pooled = pd.concat(all_test_rows, ignore_index=False)

    print(f"\n=== Pooled out-of-fold results ({len(pooled)} rows, {pooled['ticker'].nunique()} markets) ===")
    for label, name in [("A", "Base rate (time-bucketed)"), ("B", "Current-side-holds (time-bucketed)"),
                         ("C", "Distance+time+vol only"), ("D", "Full model (+ 7-indicator confluence)")]:
        p = pooled[f"p{label}"].clip(1e-6, 1 - 1e-6)
        brier = brier_score_loss(pooled["y"], p)
        ll = log_loss(pooled["y"], p)
        print(f"  {name:38s}  Brier={brier:.4f}  LogLoss={ll:.4f}")

    print("\n=== Market-level paired bootstrap (does the full model beat each baseline?) ===")
    for label, name in [("A", "vs Base rate"), ("B", "vs Current-side-holds"), ("C", "vs Distance+time+vol only")]:
        actual, p_value, _ = market_level_bootstrap(pooled, pooled["pD"], pooled[f"p{label}"], "D", label)
        direction = "BETTER" if actual < 0 else "WORSE"
        print(f"  Full model {name:28s}  Brier gap={actual:+.5f} ({direction})  bootstrap p={p_value:.4f}")

    # Calibration table: does a stated 60-70% actually land ~60-70% of the time?
    print("\n=== Calibration (full model D, pooled) ===")
    pooled["p_bucket"] = pd.cut(pooled["pD"], bins=[0, .4, .45, .5, .55, .6, .7, .8, 1.0])
    calib = pooled.groupby("p_bucket", observed=True).agg(n=("y", "size"), mean_pred=("pD", "mean"), actual_rate=("y", "mean"))
    print(calib.to_string())

    out_dir = Path(__file__).resolve().parent.parent / "results"
    fm.to_csv(out_dir / "walk_forward_fold_metrics.csv", index=False)
    pooled[["ticker", "close_time", "checkpoint_min", "y", "pA", "pB", "pC", "pD"]].to_csv(
        out_dir / "walk_forward_pooled_predictions.csv", index=False)
    calib.to_csv(out_dir / "calibration_table.csv")
    print(f"\nSaved: walk_forward_fold_metrics.csv, walk_forward_pooled_predictions.csv, calibration_table.csv")


if __name__ == "__main__":
    main()
