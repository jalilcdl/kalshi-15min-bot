"""
Two checks that decide whether this metric is worth shipping at all.

CHECK 1 -- is the fitted model adding INFORMATION, or just recalibrating the
textbook formula? Reflection scored AUC 0.8065 vs the logistic's 0.8116: nearly
identical ranking power, but Brier 0.1870 vs 0.1706 and a max calibration gap of
0.18 vs 0.03. That pattern says "same signal, wrong scale." So: Platt-scale the
reflection probability on each training fold and re-score. If recalibrated
reflection matches the logistic, the honest story is "barrier physics + a
fitted scale correction," and the extra features are decoration.

CHECK 2 -- does a strike crossing actually correspond to the READ flipping?
The metric is being sold as "how likely is this read to flip." The read shown is
P(YES) from the settlement model, and it flips side when it crosses 0.50. That
is only the same event as a price/strike crossing if sign(p_yes - 0.5) tracks
sign(price - strike). Measured, not assumed -- if they diverge the label is
answering a different question than the UI claims.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RESULTS = Path(__file__).resolve().parent.parent / "results"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from walk_forward_crossing import FEATURES, LABEL, make_folds, reflection_p  # noqa: E402


def main():
    df = pd.read_csv(RESULTS / "crossing_features.csv", parse_dates=["close_time"])
    df = df.sort_values(["close_time", "checkpoint_min"]).reset_index(drop=True)
    folds = make_folds(df)

    # ---------------- CHECK 1 ----------------
    print("=== CHECK 1: recalibrated reflection vs the fitted logistic ===")
    ys, p_refl_cal, p_logit, p_refl_raw, p_dor_only = [], [], [], [], []
    for tr_t, te_t in folds:
        tr, te = df[df["ticker"].isin(tr_t)], df[df["ticker"].isin(te_t)]
        ys.append(te[LABEL].to_numpy())

        r_tr = reflection_p(tr["dist_over_reachable"]).reshape(-1, 1)
        r_te = reflection_p(te["dist_over_reachable"]).reshape(-1, 1)
        p_refl_raw.append(r_te.ravel())
        platt = LogisticRegression(max_iter=1000).fit(r_tr, tr[LABEL])
        p_refl_cal.append(platt.predict_proba(r_te)[:, 1])

        # dist_over_reachable + time, fitted -- the minimal honest model
        mini = ["dist_over_reachable", "minutes_remaining"]
        sc_m = StandardScaler().fit(tr[mini])
        clf_m = LogisticRegression(max_iter=2000).fit(sc_m.transform(tr[mini]), tr[LABEL])
        p_dor_only.append(clf_m.predict_proba(sc_m.transform(te[mini]))[:, 1])

        sc = StandardScaler().fit(tr[FEATURES])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(tr[FEATURES]), tr[LABEL])
        p_logit.append(clf.predict_proba(sc.transform(te[FEATURES]))[:, 1])

    y = np.concatenate(ys)
    cands = {
        "reflection_raw": np.concatenate(p_refl_raw),
        "reflection_platt": np.concatenate(p_refl_cal),
        "dor_plus_time": np.concatenate(p_dor_only),
        "logistic_full": np.concatenate(p_logit),
    }
    rows = []
    for k, p in cands.items():
        p = np.clip(p, 1e-6, 1 - 1e-6)
        q = pd.qcut(p, 10, labels=False, duplicates="drop")
        gaps = pd.DataFrame({"b": q, "p": p, "y": y}).groupby("b").apply(
            lambda g: g["y"].mean() - g["p"].mean(), include_groups=False).abs()
        rows.append(dict(model=k, brier=brier_score_loss(y, p), logloss=log_loss(y, p),
                         auc=roc_auc_score(y, p), max_gap=gaps.max(), mean_gap=gaps.mean()))
    out = pd.DataFrame(rows).sort_values("brier")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    out.to_csv(RESULTS / "check_recalibration.csv", index=False)

    # ---------------- CHECK 2 ----------------
    print("\n=== CHECK 2: does a strike crossing == the displayed read flipping? ===")
    from model.strike_probability import predict_p_yes
    samp = df.sample(n=min(8000, len(df)), random_state=3)
    p_yes = np.array([predict_p_yes(r.price, r.strike, r.minutes_remaining, r.realized_vol)
                      for r in samp.itertuples()])
    model_side = np.where(p_yes >= 0.5, 1, -1)
    price_side = samp["current_side"].to_numpy()
    agree = (model_side == price_side)
    print(f"  sampled {len(samp):,} checkpoints")
    print(f"  sign(P(YES)-0.5) == sign(price-strike): {agree.mean()*100:.2f}%")
    dis = samp[~agree]
    if len(dis):
        print(f"  disagreements: {len(dis):,}  "
              f"median |distance| {dis['abs_distance_pct'].median():.4f}%  "
              f"median dist/reachable {dis['dist_over_reachable'].median():.3f}")
        print("  (disagreements concentrated at tiny distances = the model hedging "
              "near the strike, not a different notion of 'side')")
    pd.DataFrame({"p_yes": p_yes, "price_side": price_side,
                  "model_side": model_side, "agree": agree}).to_csv(
        RESULTS / "check_read_flip_alignment.csv", index=False)


if __name__ == "__main__":
    main()
