"""
Does "fade the leader when flip risk is high" actually make money?

Reads research/fade_flip/results/fade_trades.csv (built by build_fade_dataset.py,
walk-forward: flip probabilities come from a model refit on prior markets only).

WHAT THIS ASKS, IN ORDER
  1. the grid Jalil asked for: threshold X x exit target Y
  2. the question the grid cannot answer on its own -- does the FLIP FILTER add
     anything, or is any edge just structural from fading? Compared against
     fading with no filter at all, and against doing the opposite (following
     the leader).
  3. does time remaining change the answer (minute 1 vs minute 12)?
  4. does the best cell survive being CHOSEN out of sample? Picking the best of
     ~40 cells on the same data that scores it is the classic way to publish
     noise. Parameters are selected on early folds and scored on later ones.

    python research/fade_flip/scripts/analyse_fade.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "research/fade_flip/results"

MIN_N = 200          # below this a cell is noise; flagged, not reported as real


def summarise(g):
    return pd.Series({
        "n": len(g),
        "markets": g.ticker.nunique(),
        "hit%": g.hit.mean() * 100,
        "avg_net": g.net.mean(),
        "avg_gross": g.gross.mean(),
        "total_net": g.net.sum(),
    })


def market_bootstrap(g, n_boot=2000, seed=0):
    """CI on mean net P&L, resampling MARKETS not trades.

    Trades from one market share a price path, so treating them as independent
    would overstate precision -- the same convention as the other studies here.
    """
    if g.empty:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    by_mkt = g.groupby("ticker").net.mean().values
    n = len(by_mkt)
    means = [rng.choice(by_mkt, n, replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    d = pd.read_csv(RES / "fade_trades.csv")
    first = d[d.checkpoint_min == 1]          # "first read of the session"
    print(f"loaded {len(d):,} simulated trades over {d.ticker.nunique():,} markets")
    print(f"first-read rows: {len(first):,} ({first.ticker.nunique():,} markets)\n")

    thresholds = [0.00, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    targets = sorted(d.target.unique())

    print("=" * 96)
    print("1. GRID -- first read (14 min left), walk-forward flip prob, net of fees")
    print("=" * 96)
    print(f"{'thresh':>7} {'target':>7} {'n':>6} {'mkts':>6} {'hit%':>7} "
          f"{'avg net':>9} {'total':>10}  flag")
    grid = []
    for x in thresholds:
        for y in targets:
            g = first[(first.flip_wf > x) & (first.target == y)]
            if g.empty:
                continue
            s = summarise(g)
            flag = "SMALL" if s.n < MIN_N else ""
            grid.append({"thresh": x, "target": y, **s.to_dict()})
            print(f"{x:7.2f} {y:7.3f} {int(s.n):6d} {int(s.markets):6d} "
                  f"{s['hit%']:6.1f}% {s.avg_net:+9.4f} {s.total_net:+10.2f}  {flag}")
    gdf = pd.DataFrame(grid)
    gdf.to_csv(RES / "grid_first_read.csv", index=False)

    print()
    print("=" * 96)
    print("2. IS THE FLIP FILTER DOING ANY WORK?")
    print("=" * 96)
    best = gdf[gdf.n >= MIN_N].sort_values("avg_net", ascending=False).head(1)
    if not best.empty:
        bx, by = float(best.thresh.iloc[0]), float(best.target.iloc[0])
        print(f"best cell with n>={MIN_N}: threshold {bx:.2f}, target {by:.3f}, "
              f"avg net {float(best.avg_net.iloc[0]):+.4f}")
    else:
        bx, by = 0.60, 0.10
        print("no cell reached the sample-size floor; using 0.60/0.10 for comparison")

    for label, sel in (
        ("FADE, flip filter",      first[(first.flip_wf > bx) & (first.target == by)]),
        ("FADE, NO filter",        first[first.target == by]),
        ("FADE, LOW flip only",    first[(first.flip_wf <= bx) & (first.target == by)]),
    ):
        if sel.empty:
            continue
        s = summarise(sel)
        lo, hi = market_bootstrap(sel)
        print(f"  {label:22s} n={int(s.n):5d} hit {s['hit%']:5.1f}%  "
              f"avg net {s.avg_net:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"{'  <- includes 0' if lo <= 0 <= hi else ''}")

    print()
    print("=" * 96)
    print("3. DOES TIME REMAINING MATTER? (threshold %.2f, target %.3f)" % (bx, by))
    print("=" * 96)
    print(f"{'minutes left':>12} {'n':>6} {'hit%':>7} {'avg net':>9} {'95% CI':>26}")
    tr = []
    for cp in sorted(d.checkpoint_min.unique()):
        g = d[(d.checkpoint_min == cp) & (d.flip_wf > bx) & (d.target == by)]
        if len(g) < 30:
            continue
        s = summarise(g)
        lo, hi = market_bootstrap(g)
        mins = g.minutes_remaining.iloc[0]
        tr.append({"minutes_remaining": mins, **s.to_dict(), "lo": lo, "hi": hi})
        star = " *" if (lo > 0 or hi < 0) else ""
        print(f"{mins:12.0f} {int(s.n):6d} {s['hit%']:6.1f}% {s.avg_net:+9.4f} "
              f"  [{lo:+.4f}, {hi:+.4f}]{star}")
    pd.DataFrame(tr).to_csv(RES / "by_time_remaining.csv", index=False)
    print("  * = 95% CI excludes zero")

    print()
    print("=" * 96)
    print("4. OUT-OF-SAMPLE PARAMETER SELECTION")
    print("=" * 96)
    print("pick the best (threshold, target) on folds 1..k, score it on fold k+1")
    rows = []
    for k in sorted(d.fold.unique())[:-1]:
        tr_d = first[first.fold <= k]
        te_d = first[first.fold == k + 1]
        cells = []
        for x in thresholds:
            for y in targets:
                g = tr_d[(tr_d.flip_wf > x) & (tr_d.target == y)]
                if len(g) >= 100:
                    cells.append((g.net.mean(), x, y, len(g)))
        if not cells or te_d.empty:
            continue
        cells.sort(reverse=True)
        _, x, y, ntr = cells[0]
        te = te_d[(te_d.flip_wf > x) & (te_d.target == y)]
        if te.empty:
            continue
        rows.append({"train_folds": f"1-{k}", "picked_thresh": x, "picked_target": y,
                     "train_n": ntr, "test_n": len(te),
                     "test_avg_net": te.net.mean(), "test_hit%": te.hit.mean() * 100})
        print(f"  train 1-{k}: picked X={x:.2f} Y={y:.3f}  ->  test fold {k+1}: "
              f"n={len(te):4d} hit {te.hit.mean()*100:5.1f}% "
              f"avg net {te.net.mean():+.4f}")
    oos = pd.DataFrame(rows)
    if not oos.empty:
        oos.to_csv(RES / "oos_selection.csv", index=False)
        print(f"\n  pooled out-of-sample avg net: {oos.test_avg_net.mean():+.4f} "
              f"per trade over {int(oos.test_n.sum()):,} trades")
        print(f"  picked parameters stable? thresholds "
              f"{sorted(oos.picked_thresh.unique())}, targets "
              f"{sorted(oos.picked_target.unique())}")

    print()
    print("=" * 96)
    print("5. SANITY: the same grid using the SHIPPED (in-sample) model")
    print("=" * 96)
    for x in (0.50, 0.60, 0.70):
        gw = first[(first.flip_wf > x) & (first.target == by)]
        gs = first[(first.flip_shipped > x) & (first.target == by)]
        if gw.empty or gs.empty:
            continue
        print(f"  X={x:.2f}: walk-forward n={len(gw):5d} avg {gw.net.mean():+.4f}   |   "
              f"shipped n={len(gs):5d} avg {gs.net.mean():+.4f}")


if __name__ == "__main__":
    main()
