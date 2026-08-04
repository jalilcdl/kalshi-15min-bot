"""
Builds the Telegram "game plan" alert for each 15-min Kalshi window, using
the validated distance+time+volatility settlement model (model/strike_
probability.py) and the same fee/edge/exit-target logic paper_trader.py
actually trades on -- NOT the 7-indicator confluence engine or the old
strike_distance.py heuristic, which this project has since shown have no
validated edge (see research/kalshi-btc-validation/ and research/
strike_probability/README.md).

Three message types. ALL THREE describe paper_trader.py's SIMULATED fills --
this bot places no real orders. Never use "confirmed"/"executed"/similar
language that could read as a real-order confirmation to someone trading
real cash off these alerts (see the 2026-08-03 incident notes in bot.py:
build_paper_entry_followup_alert() used to say "Trade confirmed" and a user
screenshot flagged it as dangerously ambiguous -- fixed, keep it that way).

  build_entry_alert()         -- once per window, ~60-120s in: what the model
                                 sees, which side (if any) clears the entry
                                 gates, and what to expect if you take it.
  build_paper_entry_followup_alert() -- safety-net follow-up: paper_trader.py
                                 polls continuously and can find a simulated
                                 edge later in the window even if the early
                                 check said "sit out" -- this fires whenever a
                                 paper entry appears in trade_log.csv that the
                                 early check didn't already announce, so a
                                 paper trade can never go un-alerted (see
                                 bot.py's check_new_entries()). Explicitly
                                 labeled PAPER/simulated, and flags how late
                                 into the window it landed since research/
                                 exit_timing/README.md §5c found later entries
                                 have a materially lower historical target-hit
                                 rate -- entries this alert catches are, by
                                 construction, always the late kind.
  build_target_hit_alert()    -- fired when a live paper position crosses the
                                 favorable-move exit target, pulled straight
                                 from paper_trader.py's own trade_log.csv row
                                 so it can never disagree with the real paper
                                 trade.
"""
import config
from paper_trader import TradeDecision

_REASON_TEXT = {
    "final_minutes": (
        f"Final {config.FINAL_MINUTES_NOISY} minutes of the window — "
        "settlement-print noise dominates, no call made."
    ),
    "no_quote": "No live bid/ask available for this market yet.",
    "no_strike": "Strike not published yet (TBD) — no read computable until Kalshi sets it.",
    "too_close_to_strike": (
        "Distance to strike is inside the noise band for current volatility "
        "— the model isn't trusted this close (see config."
        "PAPER_TRADE_MIN_DIST_OVER_REACHABLE)."
    ),
    "edge_too_small": (
        f"Neither side's edge clears the {config.PAPER_TRADE_MIN_EDGE*100:.0f}¢ "
        "minimum after fees."
    ),
    "entry_window_closed": (
        f"More than {config.PAPER_TRADE_MAX_ENTRY_MINUTES_ELAPSED} minutes into the "
        "window already — too late for a fresh entry, later entries have a "
        "validated lower target-hit rate (see research/exit_timing/README.md §5c)."
    ),
}


def _target_price_line(entry_price: float) -> str:
    """The exit-target price, or a note when it's mathematically unreachable
    (entry already close enough to the 100c ceiling that +TARGET% would need
    to price above $1.00 -- a real edge case the smoke test surfaced)."""
    target_price = entry_price * (1.0 + config.PAPER_TRADE_EXIT_TARGET)
    if target_price > 0.99:
        return (f"🎯 Exit target: unreachable from {entry_price*100:.0f}¢ "
                f"(+{config.PAPER_TRADE_EXIT_TARGET*100:.0f}% would need >100¢) "
                "— this one rides to settlement either way.")
    return (f"🎯 Exit target: sell at ~{target_price*100:.0f}¢ "
            f"(+{config.PAPER_TRADE_EXIT_TARGET*100:.0f}%) if it gets there\n"
            "   Historically hits ~75% of entries, median ~5 min\n"
            "   (heads up: holding to settlement scored slightly better on\n"
            "   average in backtesting — real tradeoff, not resolved either way)")


def build_entry_alert(price: float, market, decision: TradeDecision) -> str:
    """Layout: model read / recommendation leads, ticker-price-strike info block
    moved to the bottom (2026-08-04, user request -- the call is what matters
    at a glance, the identifying details can come after)."""
    mins_left = decision.mins_remaining if decision.mins_remaining is not None else market.minutes_remaining()
    strike_txt = f"${market.strike:,.2f}" if market.strike is not None else "TBD"
    if market.strike:
        dist_pct = (price - market.strike) / market.strike * 100.0
        dist_txt = f"({dist_pct:+.2f}%)"
    else:
        dist_txt = ""

    title = "📈 BTC 15M Kalshi — Game Plan 🔔\n"
    info_block = (
        f"🎯 {market.ticker} · {mins_left:.1f}m left\n"
        f"💰 BTC ${price:,.2f} · Strike {strike_txt} {dist_txt}"
    )

    if decision.action == "enter":
        model_line = f"🧠 Model: {decision.p_yes*100:.0f}% YES (validated settlement model)\n"
        take_line = f"👉 Take: {decision.side.upper()} @ {decision.entry_price*100:.0f}¢ · edge after fees: {decision.edge*100:+.0f}¢\n"
        verdict = f"✅ TRADE THIS — edge clears the {config.PAPER_TRADE_MIN_EDGE*100:.0f}¢ minimum\n"
        target_line = _target_price_line(decision.entry_price)
        body = model_line + take_line + verdict + target_line
    else:
        model_line = f"🧠 Model: {decision.p_yes*100:.0f}% YES\n" if decision.p_yes is not None else "🧠 Model: n/a\n"
        reason_txt = _REASON_TEXT.get(decision.reason, decision.reason)
        body = model_line + f"🚫 SIT OUT — {reason_txt}"

    footer = f"\n\n⛔ Skip the last {config.FINAL_MINUTES_NOISY} minutes of any window"
    return title + "\n" + body + "\n\n" + info_block + footer


def build_paper_entry_followup_alert(ticker: str, side: str, entry_price_cents: float,
                                     p_model: float, mins_into_window: float | None = None) -> str:
    timing_note = ""
    if mins_into_window is not None:
        timing_note = (
            f"\n(landed ~{mins_into_window:.0f} min into the window -- later entries have "
            "historically had a lower target-hit rate, see research/exit_timing/README.md §5c)"
        )
    return (
        f"\U0001F4DD PAPER trade (simulated, NOT a real order): {ticker}\n"
        f"{side.upper()} @ {entry_price_cents:.0f}¢ (model: {p_model*100:.0f}%)\n"
        "The ~60-120s check said sit out; paper_trader.py's continuous check "
        f"found edge later and logged a simulated fill.{timing_note}"
    )


def build_target_hit_alert(ticker: str, side: str, entry_price_cents: float,
                           exit_price: float, gain: float) -> str:
    target_pct = config.PAPER_TRADE_EXIT_TARGET * 100
    return (
        f"🔔 Target hit: {ticker}\n"
        f"{side.upper()} now {exit_price*100:.0f}¢ (from {entry_price_cents:.0f}¢, "
        f"{gain*100:+.0f}%, target was +{target_pct:.0f}%) — consider selling"
    )
