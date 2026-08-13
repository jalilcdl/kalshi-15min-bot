"""
Tests for edge-proportional sizing, the risk-aware cap, and the UTC date roll.

    python test_sizing_and_cap.py

Nothing here touches the exit path or the exit breakers -- those are covered by
test_exit_completion.py and are deliberately not modified.
"""
import sys
from datetime import datetime, timezone

import config
import live_trader as lt

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


MIN_EDGE = config.PAPER_TRADE_MIN_EDGE
FULL = config.SIZING_FULL_EDGE

print("=" * 76)
print("SIZING CURVE AT THE EXTREMES")
print("=" * 76)

# --- right at the 3c minimum -------------------------------------------------
n_min, b_min = lt.size_for_edge(MIN_EDGE, 0.50)
check("a barely-qualifying edge gets the MINIMUM budget",
      abs(b_min - config.SIZING_MIN_RISK) < 1e-9,
      f"edge {MIN_EDGE:.3f} -> budget ${b_min:.2f}")
check("...but still trades (no dead zone above the threshold)", n_min > 0,
      f"{n_min} contracts")

# --- mid range ---------------------------------------------------------------
mid_edge = (MIN_EDGE + FULL) / 2
n_mid, b_mid = lt.size_for_edge(mid_edge, 0.50)
expect_mid = (config.SIZING_MIN_RISK + config.SIZING_MAX_RISK) / 2
check("the midpoint edge gets the midpoint budget",
      abs(b_mid - expect_mid) < 1e-9,
      f"edge {mid_edge:.3f} -> ${b_mid:.2f} (expected ${expect_mid:.2f})")

# --- strong edge -------------------------------------------------------------
n_full, b_full = lt.size_for_edge(FULL, 0.50)
check("edge at SIZING_FULL_EDGE gets the full budget",
      abs(b_full - config.SIZING_MAX_RISK) < 1e-9, f"${b_full:.2f}")
n_huge, b_huge = lt.size_for_edge(0.40, 0.50)
check("an enormous edge does NOT exceed the full budget",
      abs(b_huge - config.SIZING_MAX_RISK) < 1e-9,
      f"edge 0.40 -> ${b_huge:.2f}, not more")

# --- monotonic ---------------------------------------------------------------
edges = [0.03, 0.04, 0.06, 0.08, 0.10, 0.13, 0.15, 0.25]
sizes = [lt.size_for_edge(e, 0.50)[0] for e in edges]
check("size is non-decreasing in edge", all(b >= a for a, b in zip(sizes, sizes[1:])),
      f"{list(zip(edges, sizes))}")

# --- below threshold ---------------------------------------------------------
n_low, b_low = lt.size_for_edge(0.01, 0.50)
check("a sub-threshold edge is clamped to the minimum, never negative",
      n_low > 0 and abs(b_low - config.SIZING_MIN_RISK) < 1e-9,
      f"edge 0.01 -> ${b_low:.2f}, {n_low} contracts "
      f"(the 3c gate upstream is what actually rejects these)")

print()
print("=" * 76)
print("THE ACTUAL BUG: thin edge on an expensive contract")
print("=" * 76)
# 2026-08-13: a 3.44c edge took 25 contracts at 0.63 and risked $15.75, the same
# stake a 27c edge received.
n_thin, _ = lt.size_for_edge(0.0344, 0.63)
n_strong, _ = lt.size_for_edge(0.2709, 0.46)
check("thin edge no longer gets a full-size stake", n_thin < 25,
      f"3.4c edge @ 0.63 -> {n_thin} contracts = ${n_thin*0.63:.2f} (was 25 = $15.75)")
check("strong edge still gets a large stake", n_strong > n_thin * 2,
      f"27c edge @ 0.46 -> {n_strong} contracts = ${n_strong*0.46:.2f}")
check("dollar risk is bounded by SIZING_MAX_RISK at every price",
      all(lt.size_for_edge(0.30, c)[0] * c <= config.SIZING_MAX_RISK + 1e-9
          for c in (0.05, 0.2, 0.5, 0.7, 0.84, 0.95)),
      "checked across the price range at maximum edge")
check("an expensive contract gets FEWER contracts than a cheap one, same edge",
      lt.size_for_edge(0.09, 0.84)[0] < lt.size_for_edge(0.09, 0.30)[0],
      f"0.84 -> {lt.size_for_edge(0.09, 0.84)[0]}, "
      f"0.30 -> {lt.size_for_edge(0.09, 0.30)[0]}")
check("never exceeds the hard contract cap",
      all(lt.size_for_edge(e, c)[0] <= config.LIVE_MAX_CONTRACTS
          for e in (0.03, 0.1, 0.5) for c in (0.01, 0.05, 0.5, 0.99)))

print()
print("=" * 76)
print("RISK-AWARE LOSS CAP")
print("=" * 76)
real_kill = lt.kill_switch_active
lt.kill_switch_active = lambda: False
try:
    cap = config.effective_loss_cap()
    # Plenty of headroom: any sane trade is fine.
    ok, _ = lt.entries_allowed({"realized_pnl": 0.0}, trade_risk=15.0)
    check("a full-size trade is allowed with a full cap", ok)

    # Headroom smaller than the stake -- the old code allowed this.
    s = {"realized_pnl": -(cap - 10.0)}     # $10 of headroom left
    ok_old, _ = lt.entries_allowed(s)                       # old question
    ok_new, why = lt.entries_allowed(s, trade_risk=15.0)    # new question
    check("old check (P&L only) would have allowed a cap-breaching trade", ok_old)
    check("new check REFUSES it", not ok_new, why[:96])

    ok_fit, _ = lt.entries_allowed(s, trade_risk=5.0)
    check("a trade that FITS the remaining headroom is still allowed", ok_fit,
          "$5 risk into $10 headroom")

    # Exactly at the boundary.
    ok_edge, _ = lt.entries_allowed({"realized_pnl": -(cap - 10.0)}, trade_risk=10.0)
    check("a trade that exactly consumes the headroom is refused", not ok_edge,
          "$10 risk into $10 headroom -> would land exactly on the cap")

    # Be precise about what this actually bounds. An APPROVED trade can no
    # longer cross the cap by itself at all: the check refuses anything whose
    # worst case lands on or past -cap, so the deepest an approved trade can
    # take the day is just short of the cap.
    worst_after_approved = 0.0
    for headroom in (1.0, 5.0, 14.99, 30.0):
        s2 = {"realized_pnl": -(cap - headroom)}
        for risk in (0.5, 3.0, 8.0, 15.0):
            allowed, _ = lt.entries_allowed(s2, trade_risk=risk)
            if allowed:
                worst_after_approved = min(worst_after_approved,
                                           s2["realized_pnl"] - risk)
    check("no approved trade can cross the cap on its own",
          worst_after_approved > -cap,
          f"deepest reachable via an approved trade: ${worst_after_approved:.2f} "
          f"vs cap ${-cap:.2f}")
    # RESIDUAL, stated honestly rather than papered over: the check sees only
    # REALIZED P&L, so risk already sitting in open positions is not counted.
    # With one-entry-per-window that is usually at most one other position, so
    # the real bound is cap + open exposure, not cap alone.
    check("known residual: open-position risk is still uncounted",
          True,
          f"bound is cap + open exposure (<= ${config.SIZING_MAX_RISK:.2f} per "
          f"position); previously ONE trade alone could exceed the cap by "
          f"${config.LIVE_MAX_CONTRACTS * 0.75 - 0:.2f}")
finally:
    lt.kill_switch_active = real_kill

print()
print("=" * 76)
print("UTC DATE ROLL")
print("=" * 76)
today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
stale = {"utc_date": "1999-01-01", "realized_pnl": 72.41, "entries": 11,
         "exits": 16, "exit_fills": 44, "halted_reason": "x",
         "exiting": {"SOME-TICKER": {"attempts": 2}}}
rolled = lt.roll_session_if_new_day(dict(stale))
check("date advances to today", rolled["utc_date"] == today, rolled["utc_date"])
check("daily counters reset",
      rolled["entries"] == 0 and rolled["exits"] == 0 and rolled["exit_fills"] == 0)
check("realized P&L resets (recomputed from the exchange next)",
      rolled["realized_pnl"] == 0.0)
check("halt reason clears", rolled["halted_reason"] == "")
check("IN-PROGRESS EXIT STATE SURVIVES the roll",
      rolled.get("exiting") == {"SOME-TICKER": {"attempts": 2}},
      "a position mid-exit at midnight is still mid-exit at 00:01")
same = {"utc_date": today, "realized_pnl": 5.0, "entries": 3, "exits": 2}
check("a same-day session is left completely untouched",
      lt.roll_session_if_new_day(dict(same)) == same)

print()
print("=" * 76)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results) - n_fail}/{len(results)} checks passed")
for name, ok, detail in results:
    if not ok:
        print(f"  FAILED: {name} -- {detail}")
print("RESULT:", "ALL PASS" if not n_fail else f"{n_fail} FAILED")
sys.exit(1 if n_fail else 0)
