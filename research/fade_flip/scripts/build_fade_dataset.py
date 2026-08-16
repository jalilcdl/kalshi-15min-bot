"""
Backtest: fade the leading side when flip risk is high, exit on a favourable move.

THE RULE UNDER TEST (Jalil's discretionary strategy)
  At a read, if the flip-risk model says the current read is unstable
  (P(price closes back across the strike) > X), BUY THE SIDE THAT IS CURRENTLY
  LOSING. Exit when the position is up Y cents, otherwise hold to settlement.

WALK-FORWARD DISCIPLINE
The flip model is REFIT INSIDE EACH FOLD on training markets only, and the
strategy is evaluated only on test-fold rows. Folds are chronological and
expanding, split on market close_time so every checkpoint of a market lands in
exactly one fold -- the same convention as
research/crossing_probability/scripts/walk_forward_crossing.py.

Using the shipped crossing_prob_model.pkl for this instead would be in-sample:
it was fitted on these very markets, so its "predictions" here already know the
answers. That variant is computed too, separately labelled, because it is what
the badge actually shows Jalil -- but the walk-forward numbers are the ones that
say whether the edge is real.

PRICE CONVENTIONS (Kalshi)
  buy YES -> pay yes_ask          exit by selling at yes_bid
  buy NO  -> pay (1 - yes_bid)    exit by selling at (1 - yes_ask)
Entry and exit both cross the spread, so both are TAKER fills and both pay the
taker fee from fees.py. Bars are 1-minute CLOSES: the exit test uses closing
quotes, never intrabar highs, because filling at an intrabar spike assumes a
fill that may never have been available. An optimistic intrabar variant is
reported alongside so the gap is visible rather than hidden.

    python research/fade_flip/scripts/build_fade_dataset.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from fees import kalshi_fee  # noqa: E402

CF = ROOT / "research/crossing_probability/results/crossing_features.csv"
CANDLES = ROOT / "research/exit_timing/data/candlesticks.csv"
SETTLED = ROOT / "research/strike_probability/data/settled_markets.csv"
OUT = ROOT / "research/fade_flip/results"
OUT.mkdir(parents=True, exist_ok=True)


def make_folds(df, n_folds=6, min_train_frac=0.30):
    """Chronological expanding folds, split on market close_time.

    Copied deliberately from walk_forward_crossing.make_folds so results are
    comparable with the flip model's own validation.
    """
    by_time = (df.drop_duplicates("ticker")[["ticker", "close_time"]]
               .sort_values("close_time"))
    n = len(by_time)
    start = int(n * min_train_frac)
    bounds = np.linspace(start, n, n_folds + 1).astype(int)
    folds = []
    for i in range(n_folds):
        tr = set(by_time.ticker.iloc[:bounds[i]])
        te = set(by_time.ticker.iloc[bounds[i]:bounds[i + 1]])
        if tr and te:
            folds.append((tr, te))
    return folds


def fit_flip_model(train_df):
    """The shipped 2-feature specification, refit on this fold's training data."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    feats = ["dist_over_reachable", "minutes_remaining"]
    pipe = Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=1000))])
    pipe.fit(train_df[feats], train_df["flip_close"])
    return pipe, feats


def main():
    print("loading...")
    cf = pd.read_csv(CF, parse_dates=["close_time"])
    cs = pd.read_csv(CANDLES)
    sm = pd.read_csv(SETTLED)[["ticker", "result"]]

    cf = cf.merge(sm, on="ticker", how="inner")
    cf = cf[cf.ticker.isin(set(cs.ticker))]
    # Unit-agnostic on purpose. pandas parsed this column as datetime64[us],
    # not [ns], so the usual astype("int64") // 10**9 was off by 1000x and
    # produced timestamps near zero -- every bar then looked ~56 years away and
    # the whole simulation silently returned no trades.
    cf["close_ts"] = cf.close_time.map(lambda x: int(x.timestamp()))

    # bar index: ticker -> {minutes_to_close: (yes_bid_close, yes_ask_close,
    #                                          yes_bid_high, yes_ask_low)}
    cs = cs.merge(cf.drop_duplicates("ticker")[["ticker", "close_ts"]], on="ticker")
    cs["mtc"] = ((cs.close_ts - cs.ts) / 60).round().astype(int)
    cs = cs[(cs.mtc >= 0) & (cs.mtc <= 15)]
    bars = {}
    for tkr, g in cs.groupby("ticker"):
        bars[tkr] = {int(r.mtc): (r.yes_bid_close, r.yes_ask_close,
                                  r.yes_bid_high, r.yes_ask_low)
                     for r in g.itertuples()}

    print(f"markets: {cf.ticker.nunique():,}   checkpoint rows: {len(cf):,}")

    # ---- walk-forward flip probabilities, test folds only -------------------
    cf = cf.sort_values(["close_time", "checkpoint_min"]).reset_index(drop=True)
    folds = make_folds(cf)
    parts = []
    for i, (tr_t, te_t) in enumerate(folds, 1):
        tr = cf[cf.ticker.isin(tr_t)]
        te = cf[cf.ticker.isin(te_t)].copy()
        pipe, feats = fit_flip_model(tr)
        te["flip_prob_wf"] = pipe.predict_proba(te[feats])[:, 1]
        te["fold"] = i
        parts.append(te)
        print(f"  fold {i}: train {tr.ticker.nunique():,} mkts -> "
              f"test {te.ticker.nunique():,} mkts ({len(te):,} rows)")
    d = pd.concat(parts, ignore_index=True)

    # the shipped model, for comparison only (in-sample on these markets)
    from model.crossing_probability import _load
    b = _load()
    d["flip_prob_shipped"] = b["pipeline"].predict_proba(d[b["features"]])[:, 1]

    # ---- simulate every (checkpoint, target) trade once ---------------------
    # Side is fixed by the rule, so entry price and the whole forward path do
    # not depend on the THRESHOLD -- only on the checkpoint. Simulate once per
    # (row, target) and let the threshold be a filter afterwards. That keeps the
    # grid honest: identical trades, different subsets.
    rows = []
    targets = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]
    for r in d.itertuples():
        bt = bars.get(r.ticker)
        if not bt:
            continue
        entry_mtc = int(round(r.minutes_remaining))
        if entry_mtc not in bt:
            continue
        ybid, yask, _, _ = bt[entry_mtc]
        # FADE: current_side 1 = price above strike (YES leading) -> buy NO
        fade_yes = (r.current_side != 1)
        cost = yask if fade_yes else (1.0 - ybid)
        if not (0.02 <= cost <= 0.98):
            continue                      # unusable/degenerate quote
        won = (r.result == "yes") if fade_yes else (r.result == "no")

        # forward path, strictly after the entry bar
        path = [(m, bt[m]) for m in sorted(bt, reverse=True) if m < entry_mtc]
        for tgt in targets:
            exit_val = exit_mtc = None
            exit_val_hi = None            # optimistic intrabar variant
            for m, (bb, aa, bhi, alo) in path:
                val = bb if fade_yes else (1.0 - aa)
                val_hi = bhi if fade_yes else (1.0 - alo)
                if exit_val_hi is None and val_hi - cost >= tgt:
                    exit_val_hi = cost + tgt
                if val - cost >= tgt:
                    exit_val, exit_mtc = val, m
                    break
            hit = exit_val is not None
            gross = (exit_val - cost) if hit else ((1.0 if won else 0.0) - cost)
            fee = kalshi_fee(1, cost) + (kalshi_fee(1, exit_val) if hit else 0.0)
            gross_hi = (exit_val_hi - cost) if exit_val_hi is not None else \
                       ((1.0 if won else 0.0) - cost)
            rows.append({
                "ticker": r.ticker, "fold": r.fold, "close_time": r.close_time,
                "checkpoint_min": r.checkpoint_min,
                "minutes_remaining": r.minutes_remaining,
                "flip_wf": r.flip_prob_wf, "flip_shipped": r.flip_prob_shipped,
                "target": tgt, "fade_side": "yes" if fade_yes else "no",
                "entry_cost": cost, "hit": hit, "exit_mtc": exit_mtc,
                "won_settle": won, "gross": gross, "fee": fee,
                "net": gross - fee, "gross_intrabar": gross_hi,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "fade_trades.csv", index=False)
    print(f"\nsimulated {len(out):,} candidate trades "
          f"({out.ticker.nunique():,} markets x {out.checkpoint_min.nunique()} "
          f"checkpoints x {len(targets)} targets)")
    print(f"written to {OUT / 'fade_trades.csv'}")


if __name__ == "__main__":
    main()
