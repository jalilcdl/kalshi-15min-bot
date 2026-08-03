"""
fit_hit_target_model.py answers "given the side the settlement model already
picked, how likely is it to hit +30%" -- exit_timing_trades.csv only contains
that already-chosen side, so it CANNOT validly test "which side should I
pick," only "should I trust the side I already picked." Using it to replace
side-selection would be circular.

This script builds the side-symmetric dataset that question actually needs:
for EVERY checkpoint-1 row across all 4,250 markets (no entry-model gating
at all), computes BOTH the YES-side and the NO-side hypothetical outcome --
entry price, forward exit path, and whether each would have hit +30% -- from
the same real candlestick data already collected. This is unconditional on
any existing model's side choice, so a model fit on it can be honestly
evaluated as a standalone side-selector.

Also tests the obvious mechanical baseline: since Kalshi's YES+NO ask prices
sum to ~1 (plus the spread), whichever side is cheaper mechanically has more
room to travel 30% before hitting the 0-100c ceiling/floor -- entirely
independent of any real view on which way BTC will move. fit_hit_target_model
already showed entry price is THE dominant driver of hit-rate; this checks
whether a fitted model actually beats just "always take the cheaper side,"
or whether that simple, direction-agnostic rule already captures the edge.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "strike_probability" / "scripts"))
sys.path.insert(0, str(ROOT))
from walk_forward import make_folds  # noqa: E402
import config  # noqa: E402
from model.strike_probability import predict_p_yes  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = Path(__file__).resolve().parent.parent / "results"
STRIKE_PROB = ROOT / "research" / "strike_probability"

FEATURES = ["is_yes", "entry_price", "minutes_remaining", "realized_vol"]
TARGET_PCT = 0.30
RNG = np.random.default_rng(42)
N_BOOT = 2000


def build_side_symmetric_dataset():
    features = pd.read_csv(STRIKE_PROB / "results" / "features.csv")
    ck1 = features[features["checkpoint_min"] == 1].drop(columns=["close_time"]).copy()
    markets = pd.read_csv(STRIKE_PROB / "data" / "settled_markets.csv", parse_dates=["open_time", "close_time"])
    ck1 = ck1.merge(markets[["ticker", "open_time", "close_time", "result"]], on="ticker", how="left")

    candles = pd.read_csv(DATA / "candlesticks.csv")
    candles_by_ticker = {t: g.sort_values("ts").reset_index(drop=True) for t, g in candles.groupby("ticker")}

    rows = []
    for row in ck1.itertuples():
        cs = candles_by_ticker.get(row.ticker)
        if cs is None or cs.empty:
            continue
        entry_ts = int(row.open_time.timestamp()) + 60
        entry_bar = cs[cs["ts"] >= entry_ts]
        if entry_bar.empty:
            continue
        entry_bar = entry_bar.iloc[0]
        yes_ask, yes_bid = entry_bar["yes_ask_close"], entry_bar["yes_bid_close"]
        if pd.isna(yes_ask) or pd.isna(yes_bid):
            continue
        forward = cs[cs["ts"] > entry_bar["ts"]]
        if forward.empty:
            continue

        for side, entry_price, exit_series in [
            ("yes", yes_ask, forward["yes_bid_close"]),
            ("no", 1.0 - yes_bid, 1.0 - forward["yes_ask_close"]),
        ]:
            if entry_price <= 0:
                continue
            gain = (exit_series - entry_price) / entry_price
            hit = bool((gain >= TARGET_PCT).any())
            rows.append(dict(
                ticker=row.ticker, close_time=row.close_time, side=side, is_yes=int(side == "yes"),
                entry_price=entry_price, minutes_remaining=row.minutes_remaining,
                realized_vol=row.realized_vol, result=row.result, y=int(hit),
            ))

    return pd.DataFrame(rows)


def fit_eval_fold(train_df, test_df, feature_cols):
    if len(train_df) < 50 or len(test_df) == 0:
        return None
    clf = LogisticRegression(max_iter=2000)
    clf.fit(train_df[feature_cols], train_df["y"])
    return pd.Series(clf.predict_proba(test_df[feature_cols])[:, 1], index=test_df.index)


def market_level_bootstrap_2col(df, sq1_col, sq2_col, n_boot=N_BOOT):
    by_market = df.groupby("ticker")[[sq1_col, sq2_col]].mean()
    tickers = by_market.index.to_numpy()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sample = RNG.choice(tickers, size=len(tickers), replace=True)
        s = by_market.loc[sample]
        diffs[i] = s[sq1_col].mean() - s[sq2_col].mean()
    actual = by_market[sq1_col].mean() - by_market[sq2_col].mean()
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return actual, p


def main():
    cache = RESULTS / "side_symmetric_dataset.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["close_time"])
        print(f"Loaded cached side-symmetric dataset: {len(df)} rows")
    else:
        print("Building side-symmetric dataset from candlesticks (both YES and NO per market)...")
        df = build_side_symmetric_dataset()
        df.to_csv(cache, index=False)
        print(f"Built and cached: {len(df)} rows")

    n_markets = df["ticker"].nunique()
    print(f"{n_markets} markets x 2 sides = {len(df)} rows. Base rate (hit +30%): {df['y'].mean():.4f}\n")

    # --- Walk-forward fit of the side-selection model ---------------------------
    folds = make_folds(df, n_folds=6)
    all_rows = []
    for i, (train_tickers, test_tickers) in enumerate(folds):
        train_df = df[df["ticker"].isin(train_tickers)]
        test_df = df[df["ticker"].isin(test_tickers)].copy()
        p = fit_eval_fold(train_df, test_df, FEATURES)
        if p is None:
            continue
        test_df["p_model"] = p
        all_rows.append(test_df)
    pooled = pd.concat(all_rows, ignore_index=False)
    print(f"Pooled out-of-fold: {len(pooled)} rows, {pooled['ticker'].nunique()} markets")
    print(f"Brier (side-selection model): {brier_score_loss(pooled['y'], pooled['p_model']):.4f}  "
          f"LogLoss: {log_loss(pooled['y'], pooled['p_model']):.4f}\n")

    # --- Reshape to one row per market with both sides' predictions -------------
    wide = pooled.pivot_table(index="ticker", columns="side", values=["p_model", "entry_price", "y", "close_time", "result"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.dropna(subset=["p_model_yes", "p_model_no"])
    print(f"{len(wide)} markets with both sides scored\n")

    # --- Side-selection strategies ------------------------------------------------
    print("=== Side-selection strategies: hit rate on the CHOSEN side ===")

    # 1. Coin flip (average of both sides' base rate, as an unbiased reference)
    coinflip_hit = (wide["y_yes"].mean() + wide["y_no"].mean()) / 2
    print(f"  Coin flip (random side):                 {coinflip_hit*100:.1f}%")

    # 2. Always take the cheaper side (mechanical, direction-agnostic)
    cheaper_is_yes = wide["entry_price_yes"] < wide["entry_price_no"]
    cheap_hit = np.where(cheaper_is_yes, wide["y_yes"], wide["y_no"]).mean()
    print(f"  Always take cheaper side (mechanical):   {cheap_hit*100:.1f}%")

    # 3. Fitted side-selection model (higher predicted P(hit 30%))
    model_pick_yes = wide["p_model_yes"] >= wide["p_model_no"]
    model_hit = np.where(model_pick_yes, wide["y_yes"], wide["y_no"]).mean()
    print(f"  Fitted side-selection model:              {model_hit*100:.1f}%")

    # 4. Existing settlement-probability model's side choice (predict_p_yes > 0.5)
    #    Recompute using the same features already available (no live call needed --
    #    reuse distance/vol features already in strike_probability results).
    sp_features = pd.read_csv(STRIKE_PROB / "results" / "features.csv")
    sp_ck1 = sp_features[sp_features["checkpoint_min"] == 1][["ticker", "price", "strike", "minutes_remaining", "realized_vol"]]
    settle_p = {}
    for r in sp_ck1.itertuples():
        settle_p[r.ticker] = predict_p_yes(r.price, r.strike, r.minutes_remaining, r.realized_vol)
    wide["settlement_p_yes"] = wide.index.map(settle_p)
    valid = wide.dropna(subset=["settlement_p_yes"])
    settle_pick_yes = valid["settlement_p_yes"] >= 0.5
    settle_hit = np.where(settle_pick_yes, valid["y_yes"], valid["y_no"]).mean()
    print(f"  Existing settlement-probability model's pick: {settle_hit*100:.1f}%  (n={len(valid)})")

    # --- Does the fitted model beat the cheaper-side mechanical baseline? -------
    print("\n=== Does the fitted model beat 'always take the cheaper side'? ===")
    chosen_p_model = np.where(model_pick_yes, wide["p_model_yes"], wide["p_model_no"])
    chosen_y_model = np.where(model_pick_yes, wide["y_yes"], wide["y_no"])
    chosen_y_cheap = np.where(cheaper_is_yes, wide["y_yes"], wide["y_no"])
    cmp_df = pd.DataFrame({"ticker": wide.index, "sq_model": (1 - chosen_y_model) ** 1.0,
                           "sq_cheap": (1 - chosen_y_cheap) ** 1.0})
    # Using (1 - hit) as the "loss" -- lower is better, so this compares miss rates directly.
    actual, pval = market_level_bootstrap_2col(cmp_df, "sq_model", "sq_cheap")
    print(f"  Miss-rate gap (model - cheaper-side): {actual:+.4f}  bootstrap p={pval:.4f}  "
          f"{'model meaningfully different' if pval < 0.05 else 'NOT significantly different'}")
    agreement = (model_pick_yes == cheaper_is_yes).mean()
    print(f"  Agreement between model's pick and cheaper-side pick: {agreement*100:.1f}% of markets")


if __name__ == "__main__":
    main()
