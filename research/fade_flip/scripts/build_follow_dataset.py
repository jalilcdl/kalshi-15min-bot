"""
Backtest: FOLLOW the leading side (the mirror of the fade test), full costs.

The fade study left one loose end: its mirror said the LEADING side wins 63.5%
while costing 56.5%, a +7.0 point gross edge. That number was explicitly flagged
as not-a-finding, because it ignored the two things that decide whether a 7
point edge survives:

  SPREAD -- buying the leader means paying its ASK, not the mid. Both the fade
            and the follow cross the spread; the naive mirror pretended one of
            them didn't.
  FEES   -- ~2.5c round trip from fees.py, on a stake that is now LARGER
            (~0.57 rather than ~0.44), because you are buying the favourite.

This script pays both, honestly.

It also tests something the fade study had no reason to: HOLD TO SETTLEMENT with
no exit target at all. On the favoured side the exit-at-+Y rule is suspect --
it caps the winner at +Y while a loser still costs the whole stake, and the
whole stake is now bigger. Whether taking profit early helps or hurts is exactly
what the target sweep plus the settle-only baseline answers.

Method is otherwise identical to build_fade_dataset.py -- same folds, same
per-fold refit of the flip model, same 1-minute-close exits, same taker fees --
so the two results are directly comparable.

    python research/fade_flip/scripts/build_follow_dataset.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fees import kalshi_fee                                    # noqa: E402
from build_fade_dataset import make_folds, fit_flip_model      # noqa: E402

CF = ROOT / "research/crossing_probability/results/crossing_features.csv"
CANDLES = ROOT / "research/exit_timing/data/candlesticks.csv"
SETTLED = ROOT / "research/strike_probability/data/settled_markets.csv"
OUT = ROOT / "research/fade_flip/results"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]


def main():
    print("loading...")
    cf = pd.read_csv(CF, parse_dates=["close_time"])
    cs = pd.read_csv(CANDLES)
    sm = pd.read_csv(SETTLED)[["ticker", "result"]]

    cf = cf.merge(sm, on="ticker", how="inner")
    cf = cf[cf.ticker.isin(set(cs.ticker))]
    cf["close_ts"] = cf.close_time.map(lambda x: int(x.timestamp()))

    cs = cs.merge(cf.drop_duplicates("ticker")[["ticker", "close_ts"]], on="ticker")
    cs["mtc"] = ((cs.close_ts - cs.ts) / 60).round().astype(int)
    cs = cs[(cs.mtc >= 0) & (cs.mtc <= 15)]
    bars = {}
    for tkr, g in cs.groupby("ticker"):
        bars[tkr] = {int(r.mtc): (r.yes_bid_close, r.yes_ask_close)
                     for r in g.itertuples()}

    cf = cf.sort_values(["close_time", "checkpoint_min"]).reset_index(drop=True)
    parts = []
    for i, (tr_t, te_t) in enumerate(make_folds(cf), 1):
        tr = cf[cf.ticker.isin(tr_t)]
        te = cf[cf.ticker.isin(te_t)].copy()
        pipe, feats = fit_flip_model(tr)
        te["flip_prob_wf"] = pipe.predict_proba(te[feats])[:, 1]
        te["fold"] = i
        parts.append(te)
        print(f"  fold {i}: train {tr.ticker.nunique():,} -> test {te.ticker.nunique():,}")
    d = pd.concat(parts, ignore_index=True)

    rows = []
    for r in d.itertuples():
        bt = bars.get(r.ticker)
        if not bt:
            continue
        entry_mtc = int(round(r.minutes_remaining))
        if entry_mtc not in bt:
            continue
        ybid, yask = bt[entry_mtc]
        # FOLLOW: current_side 1 = price above strike -> YES is leading -> buy YES.
        # Either way we pay the ASK on the side we buy; that is the spread the
        # mirror calculation quietly skipped.
        long_yes = (r.current_side == 1)
        cost = yask if long_yes else (1.0 - ybid)
        if not (0.02 <= cost <= 0.98):
            continue
        won = (r.result == "yes") if long_yes else (r.result == "no")

        path = [(m, bt[m]) for m in sorted(bt, reverse=True) if m < entry_mtc]

        # baseline with no exit rule at all
        settle_gross = (1.0 if won else 0.0) - cost
        settle_net = settle_gross - kalshi_fee(1, cost)

        for tgt in TARGETS:
            exit_val = exit_mtc = None
            for m, (bb, aa) in path:
                val = bb if long_yes else (1.0 - aa)
                if val - cost >= tgt:
                    exit_val, exit_mtc = val, m
                    break
            hit = exit_val is not None
            gross = (exit_val - cost) if hit else settle_gross
            fee = kalshi_fee(1, cost) + (kalshi_fee(1, exit_val) if hit else 0.0)
            rows.append({
                "ticker": r.ticker, "fold": r.fold,
                "checkpoint_min": r.checkpoint_min,
                "minutes_remaining": r.minutes_remaining,
                "flip_wf": r.flip_prob_wf, "target": tgt,
                "long_side": "yes" if long_yes else "no",
                "entry_cost": cost, "hit": hit, "exit_mtc": exit_mtc,
                "won_settle": won, "gross": gross, "fee": fee,
                "net": gross - fee,
                "settle_only_net": settle_net,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "follow_trades.csv", index=False)
    print(f"\nsimulated {len(out):,} trades over {out.ticker.nunique():,} markets")
    print(f"written to {OUT / 'follow_trades.csv'}")


if __name__ == "__main__":
    main()
