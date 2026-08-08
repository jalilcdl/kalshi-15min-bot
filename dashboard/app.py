"""
Streamlit dashboard for the Kalshi 15-min BTC Intel bot.

Run with:  streamlit run dashboard/app.py
(or use run_dashboard.bat in the project root)

READ-ONLY. This dashboard displays live market data and the bot's signal
reads; it never places orders. Trades are logged manually to
data/trade_log.csv (see trade_log.py) after you act on Kalshi's own site/app.

Navigation is a session-state "website" flow (same pattern as the mlb-model /
cfb-model / nfl-model dashboards): a Scoreboard homepage -> a settled-window
detail view -> back, plus top-nav sections for Trade log, Model performance,
and About. The live scoreboard section is an auto-refreshing st.fragment so
the rest of the page doesn't rerun every tick.
"""
import hmac
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import trade_log
from classifier import classify
from coinbase_feed import fetch_1min_candles, fetch_range_1min, fetch_spot_price
from indicators import compute_signals
from kalshi_feed import get_active_market, get_settled_markets
from model.crossing_probability import predict_flip_prob
# THE shared entry-decision function -- also what bot.py's Telegram alerts use.
# Deliberately not reimplemented here; see _model_read_card().
from paper_trader import evaluate_trade
from strike_distance import evaluate as evaluate_strike

st.set_page_config(page_title="Kalshi BTC Intel", layout="wide", page_icon=":material/currency_bitcoin:")


# ---------------------------------------------------------------------------
# Optional password gate (opt-in via KALSHI_DASHBOARD_PASSWORD)
# ---------------------------------------------------------------------------
def _check_password():
    # Streamlit Cloud exposes secrets as env vars too, so os.environ.get() alone
    # covers both local (.env-style) and Cloud use -- st.secrets.get() would raise
    # StreamlitSecretNotFoundError when no secrets.toml exists at all, unlike a
    # normal dict, so it's deliberately not used here.
    expected = os.environ.get("KALSHI_DASHBOARD_PASSWORD")
    if not expected:
        return  # no password configured -> no gate
    if st.session_state.get("_pw_ok"):
        return

    entered = st.text_input("Password", type="password", key="_pw_input")
    if entered == "":
        st.caption("This dashboard is password-protected. Enter the password to continue.")
        st.stop()
    if hmac.compare_digest(entered, expected):
        st.session_state["_pw_ok"] = True
        st.rerun()
    else:
        st.error("Incorrect password.")
        st.stop()


_check_password()


# ---------------------------------------------------------------------------
# Cached data fetchers -- short TTLs since this is a real-time dashboard
# ---------------------------------------------------------------------------
@st.cache_data(ttl="20s")
def _get_candles():
    """Volatility / indicator / chart history ONLY -- never the live price.
    See coinbase_feed's module docstring (2026-08-06 incident)."""
    return fetch_1min_candles()


@st.cache_data(ttl="5s")
def _get_spot(_candles):
    """Real-time price for the strike comparison. Short TTL because this is
    the input the whole model read hinges on. Leading underscore on the arg
    keeps Streamlit from trying to hash the candle list."""
    return fetch_spot_price(_candles)


@st.cache_data(ttl="10s")
def _get_active_market():
    try:
        return get_active_market(), None
    except Exception as exc:
        return None, str(exc)


@st.cache_data(ttl="5m")
def _get_recent_settled(hours=6.0):
    now = int(time.time())
    try:
        return get_settled_markets(min_close_ts=now - int(hours * 3600), max_close_ts=now), None
    except Exception as exc:
        return [], str(exc)


@st.cache_data(ttl="1h")
def _load_validation_summary():
    import json
    path = ROOT / "data" / "validation" / "direction_signal_backtest_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(ttl="1h")
def _load_strike_prob_summary():
    import json
    path = ROOT / "data" / "validation" / "strike_probability_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(ttl="1h")
def _load_hit_target_summary():
    import json
    path = ROOT / "data" / "validation" / "hit_target_side_selection_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(ttl="1h")
def _load_exit_timing_summary():
    import json
    path = ROOT / "data" / "validation" / "exit_timing_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Navigation state
# ---------------------------------------------------------------------------
st.session_state.setdefault("view", "home")
st.session_state.setdefault("window_ticker", None)


def _go(view):
    st.session_state.view = view
    if view != "window":
        st.session_state.window_ticker = None


def _open_window(ticker):
    st.session_state.view = "window"
    st.session_state.window_ticker = ticker


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")
st.sidebar.caption(f"Series: `{config.KALSHI_SERIES_TICKER}`  ·  Source: Coinbase BTC-USD 1-min")
st.sidebar.info(
    "Read-only. This dashboard never places orders — it shows what the bot "
    "sees so you can decide and trade manually on Kalshi."
)
if st.sidebar.button("Force refresh now", icon=":material/refresh:"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(
    "Scoreboard auto-refreshes every 15s. Everything else is cached briefly "
    "(active market 10s, settled-window list 5min) to stay easy on the public APIs."
)


# ---------------------------------------------------------------------------
# Top navigation bar
# ---------------------------------------------------------------------------
st.title("Kalshi BTC Intel")

_NAV = [
    ("home", "Scoreboard", ":material/scoreboard:"),
    ("trades", "Trade log", ":material/receipt_long:"),
    ("performance", "Model performance", ":material/verified:"),
    ("about", "About & limitations", ":material/info:"),
]
_active = st.session_state.view
with st.container(horizontal=True):
    for key, label, icon in _NAV:
        is_active = (key == _active) or (key == "home" and _active == "window")
        st.button(
            label, icon=icon, key=f"nav_{key}",
            type="primary" if is_active else "secondary",
            on_click=_go, args=(key,), width="stretch",
        )
st.divider()


# ---------------------------------------------------------------------------
# Shared render helpers
# ---------------------------------------------------------------------------
_STATE_COLOR = {"UPTREND": "green", "DOWNTREND": "red", "STABLE": "blue", "RANGING": "orange", "WARMUP": "gray"}


def _state_badge(state_str):
    color = _STATE_COLOR.get(state_str, "gray")
    return f":{color}[**{state_str}**]"


def _votes_table(sig):
    rows = [{"indicator": r.name, "vote": {1: "▲", -1: "▼", 0: "–"}[r.vote], "detail": r.detail}
            for r in sig.results]
    return pd.DataFrame(rows)


# Same skip-reason wording the Telegram alerts use (alert_message._REASON_TEXT),
# phrased for the dashboard. Kept parallel on purpose -- if the two surfaces
# describe the same state in different words, that reads as a disagreement.
_SKIP_TEXT = {
    "final_minutes": f"Final {config.FINAL_MINUTES_NOISY} minutes — settlement-print noise dominates.",
    "no_quote": "No live bid/ask available for this market yet.",
    "no_strike": "Strike not published yet (TBD) — no read computable until Kalshi sets it.",
    "too_close_to_strike": ("Too close to the strike relative to current volatility — "
                            "the model isn't trusted here."),
    "edge_too_small": f"No side's edge clears the {config.PAPER_TRADE_MIN_EDGE*100:.0f}c minimum after fees.",
    "entry_window_closed": (f"More than {config.PAPER_TRADE_MAX_ENTRY_MINUTES_ELAPSED} min into the window — "
                            "too late for a fresh entry (later entries have a validated lower "
                            "target-hit rate; see research/exit_timing/README.md §5c)."),
}


def _flip_risk_badge(price, market, decision):
    """How likely the read directly above this is to reverse before close.

    Deliberately worded as "price closes back across strike", not "your read
    flips": the model's label counts 1-min CLOSES on the far side, while this
    page renders a real-time ticker, so a read can visibly wobble on a
    sub-minute move the label never counted. The intra-bar variant of the label
    runs ~7 points higher, so this number under-states apparent flipping and
    the wording has to match what was actually validated.
    See research/crossing_probability/README.md."""
    p = predict_flip_prob(price, market.strike, decision.mins_remaining,
                          decision.realized_vol)
    if p is None:
        return  # flat tape / no strike / window over -- say nothing rather than "0%"

    # Three bands, so it reads at a glance. Thresholds are descriptive labels on
    # a continuous number, not decision rules -- the percentage is the content.
    if p >= 0.55:
        colour, word = "red", "high"
    elif p >= 0.30:
        colour, word = "orange", "moderate"
    else:
        colour, word = "green", "low"

    st.markdown(
        f"Flip risk: :{colour}[**{p*100:.0f}%** ({word})]  \n"
        f"<span style='opacity:0.7;font-size:0.85em'>chance price closes back across "
        f"the strike before this window ends</span>",
        unsafe_allow_html=True,
        help="Validated separately from the settlement read above: 4,250 markets, "
             "6 walk-forward folds, Brier 0.1716 / AUC 0.810, max decile calibration "
             "gap 5.4 pts. Beats base-rate, time-only and the closed-form "
             "reflection baselines at p<0.001. Counts 1-min closes on the far side, "
             "so it slightly under-states brief intra-minute wobbles.",
    )


def _model_read_card(price, market, sig, mins_left):
    """Live model read, rendered straight off paper_trader.evaluate_trade() --
    the exact same call bot.py's Telegram alerts use.

    This used to reimplement the entry math inline and had drifted badly from
    the real thing: it was missing the final_minutes, entry_window_closed and
    no_quote gates, so it displayed "clears the entry threshold right now" for
    roughly 10 of every window's 15 minutes while the bot was correctly sitting
    out. That is the dashboard-vs-Telegram disagreement the user reported
    (2026-08-04). Never reimplement these gates here -- call evaluate_trade().

    sig.realized_vol_pct is passed through so this shares the dashboard's
    already-cached candles instead of triggering a second Coinbase fetch on
    every 15s refresh tick."""
    if market.strike is None:
        st.caption("Strike not published yet (TBD) — model needs it to compute a read.")
        return

    decision = evaluate_trade(price, market, realized_vol_pct=sig.realized_vol_pct)

    # The live model read stays on screen and keeps updating for the whole
    # window, including during sit-out stretches -- it's the number the user
    # watches evolve. Only the ACTION below it is gated. (evaluate_trade()
    # attaches the read to its early-skip returns too, precisely so this can
    # never go blank mid-window.)
    if decision.p_yes is not None:
        side = "YES" if decision.p_yes >= 0.5 else "NO"
        side_prob = decision.p_yes if side == "YES" else 1.0 - decision.p_yes
        st.metric(f"Model read — P({side})", f"{side_prob*100:.1f}%",
                  help="Live settlement probability from the validated distance+time+volatility "
                       "model. Updates every 15s for the full window, independent of whether "
                       "the entry gates currently allow a trade.")
        _flip_risk_badge(price, market, decision)
        if decision.dist_over_reachable is not None:
            st.caption(f"P(YES) {decision.p_yes*100:.1f}%  ·  realized vol {decision.realized_vol:.3f}%  "
                       f"·  dist/reachable {decision.dist_over_reachable:.2f}  "
                       f"·  {decision.mins_remaining:.1f}m left")
    else:
        st.metric("Model read", "n/a")

    if decision.action == "enter":
        st.success(f"**TRADE THIS** — {decision.side.upper()} @ {decision.entry_price*100:.0f}c · "
                   f"edge after fees **{decision.edge*100:+.1f}c**")
    else:
        st.caption(f"**Sit out** — {_SKIP_TEXT.get(decision.reason, decision.reason)}")


def _manual_position_tracker(market):
    """Read-only helper for someone trading by hand: log what you actually
    bought and watch live progress toward the validated 35% exit target,
    updating on the same 15s refresh as the rest of the Scoreboard. This
    dashboard never places orders -- it only tracks what you tell it."""
    pos = st.session_state.get("manual_position")
    st.markdown("**My position** (manual, tracked locally — not a real order)")

    if pos is None:
        with st.form("manual_position_form", border=False):
            c1, c2 = st.columns(2)
            side = c1.selectbox("Side", ["yes", "no"], key="mp_side")
            entry_cents = c2.number_input("Bought at (cents)", min_value=1, max_value=99,
                                          value=50, step=1, key="mp_entry_cents")
            if st.form_submit_button("Track this position"):
                st.session_state["manual_position"] = {
                    "side": side, "entry_price": entry_cents / 100.0,
                    "ticker": market.ticker, "entered_at": datetime.now(timezone.utc).isoformat(),
                }
                st.rerun()
        return

    if pos["ticker"] != market.ticker:
        st.warning("This position's window has closed. Clear it and log a new one for the "
                   f"current market (`{market.ticker}`).")
        if st.button("Clear position", key="mp_clear_stale"):
            st.session_state["manual_position"] = None
            st.rerun()
        return

    exit_value = market.yes_bid if pos["side"] == "yes" else 1.0 - market.yes_ask
    gain = (exit_value - pos["entry_price"]) / pos["entry_price"]
    target = config.PAPER_TRADE_EXIT_TARGET

    c1, c2, c3 = st.columns(3)
    c1.metric("Entry", f"{pos['entry_price']*100:.0f}c ({pos['side'].upper()})")
    c2.metric("Current value", f"{exit_value*100:.0f}c", f"{gain*100:+.1f}%")
    c3.metric("Target", f"{target*100:.0f}%")
    st.progress(min(max(gain / target, 0.0), 1.0),
               text=f"{gain*100:.1f}% of the way to a {target*100:.0f}% exit" if gain >= 0
               else f"{gain*100:.1f}% — currently below entry")
    if gain >= target:
        st.success(f"Target reached — this is where the validated exit rule would sell.")
    if st.button("Clear position", key="mp_clear"):
        st.session_state["manual_position"] = None
        st.rerun()


def _price_chart(candles, strike=None, title="BTC-USD, last 5h (1-min)"):
    df = pd.DataFrame({"time": [c.ts for c in candles], "close": [c.close for c in candles]})
    df["time"] = pd.to_datetime(df["time"], unit="s")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["close"], mode="lines", name="BTC-USD",
                              line=dict(color="#2ca02c" if df["close"].iloc[-1] >= df["close"].iloc[0] else "#d62728")))
    if strike is not None:
        fig.add_hline(y=strike, line_dash="dash", line_color="gray", annotation_text=f"strike {strike:,.0f}")
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10),
                       showlegend=False, template="plotly_dark")
    return fig


# ---------------------------------------------------------------------------
# View: Scoreboard (home) -- live section is a fragment so the rest of the
# page (nav, sidebar) doesn't rerun on every 15s tick.
# ---------------------------------------------------------------------------
@st.fragment(run_every="15s")
def _live_scoreboard():
    try:
        candles = _get_candles()
    except Exception as exc:
        st.error(f"Couldn't fetch BTC price data: {exc}")
        return
    if len(candles) < config.LOOKBACK_BARS:
        st.warning(f"Only {len(candles)} of {config.LOOKBACK_BARS} 1-min bars available — "
                   "indicator warming up.")
        return

    spot = _get_spot(candles)
    price = spot.price
    sig = compute_signals(candles)
    state = classify(sig)
    market, market_err = _get_active_market()

    # Report the age of the PRICE THE MODEL ACTUALLY USED, not the render time.
    # This caption used to print datetime.now() regardless, so a 5-minute-stale
    # price was presented under a fresh timestamp -- exactly how two views ended
    # up showing confident, opposite recommendations on 2026-08-06.
    price_age = (datetime.now(timezone.utc) - spot.ts).total_seconds()
    if spot.source == "ticker":
        st.caption(
            f"BTC price is **{price_age:.0f}s old** (Coinbase last trade, "
            f"{spot.ts:%H:%M:%S} UTC) · refreshes every 15s. "
            "**The Telegram alert is a one-time snapshot ~1-2 min after the window opens; "
            "this read keeps updating. Same decision function, so any difference is the "
            "market moving, not the two disagreeing.**"
        )
    else:
        st.error(
            f"⚠️ Live ticker unavailable — falling back to a **{price_age:.0f}s-old** 1-min "
            f"candle close from {spot.ts:%H:%M:%S} UTC. Near an at-the-money strike a stale "
            "price can land on the wrong side and flip the recommendation. Treat this read "
            "as unreliable until the ticker recovers."
        )

    with st.container(horizontal=True):
        st.metric("BTC-USD", f"${price:,.2f}", border=True)
        # These two come from the SAME 7-indicator confluence engine as the votes
        # panel below, so they carry the same caveat -- marking the panel but
        # leaving these unmarked would half-defeat the point.
        _CONFLUENCE_HELP = ("Chart context, not a prediction — from the 7-indicator "
                            "confluence engine, which measured 44–48% directional accuracy "
                            "over 15 min (worse than a coin flip). See the votes panel below.")
        st.metric("Market state", state.state, border=True,
                   help=f"{state.trend_strength} trend · {state.momentum} momentum · "
                        f"{state.volatility} volatility. {_CONFLUENCE_HELP}")
        st.metric("Confidence", f"{state.confidence}/7 ({state.confidence_band})", border=True,
                  help="Agreement among the 7 chart indicators — NOT a confidence in any "
                       f"prediction. {_CONFLUENCE_HELP} The ≥6/7 filter tested *worse* than "
                       "the unfiltered signal.")
        st.metric("15-min window move", f"{sig.window_delta_pct:+.2f}%", border=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        with st.container(border=True):
            st.markdown(f"**State: {_state_badge(state.state)} {state.emoji}**")
            st.plotly_chart(_price_chart(candles, strike=market.strike if market else None),
                             width="stretch", key="scoreboard_chart")

    with col_r:
        with st.container(border=True):
            st.markdown("**Active Kalshi market**")
            if market_err:
                st.warning(f"Kalshi feed unavailable: {market_err}")
            elif market is None:
                st.info("No active KXBTC15M market found right now.")
            else:
                mins_left = market.minutes_remaining()
                st.write(f"`{market.ticker}`")
                strike_txt = f"${market.strike:,.0f}" if market.strike is not None else "TBD"
                st.write(f"Strike: **{strike_txt}**  ·  Closes in **{mins_left:.1f} min**")
                if market.yes_bid is not None and market.yes_ask is not None:
                    st.write(f"YES bid/ask: **{market.yes_bid*100:.0f}c / {market.yes_ask*100:.0f}c**")
                _model_read_card(price, market, sig, mins_left)

        with st.container(border=True):
            # CONTEXT PANEL, NOT A PREDICTION. This is the same 7-indicator
            # confluence engine the TradingView Pine script draws (that script
            # is a port OF this Python, not the other way round -- see
            # kalshi_15m_indicator.pine's header). It is deliberately NOT given
            # the visual weight of the model read or the flip-risk badge above,
            # because unlike those two it is not merely unvalidated -- it was
            # measured as actively worse than a coin flip. Do not promote it to
            # a peer signal without new evidence that clears the same bar.
            st.markdown("**7-indicator votes** — chart context, not a prediction")
            st.caption(
                ":orange[**Not a signal.**] Validated on 4,765 instances: entering on these "
                "states won 44–48% of the time over the next 15 min — *worse* than a coin "
                "flip (p<0.0001 to p=0.02), and the high-confidence filter (≥6/7) made it "
                "**worse**, not better. Shown as a discretionary read of what the chart "
                "indicators are doing. The model read and flip risk above are the validated "
                "numbers."
            )
            st.dataframe(_votes_table(sig), width="stretch", hide_index=True)

        if market is not None and market.yes_bid is not None and market.yes_ask is not None:
            with st.container(border=True):
                _manual_position_tracker(market)


def render_scoreboard():
    _live_scoreboard()

    st.divider()
    st.subheader("Recent settled windows")
    st.caption("Click a window to see exactly what the model said at the time vs. what actually happened.")
    markets, err = _get_recent_settled(hours=6.0)
    if err:
        st.warning(f"Couldn't load settled markets: {err}")
        return
    if not markets:
        st.info("No settled KXBTC15M windows in the last 6 hours.")
        return
    markets = sorted(markets, key=lambda m: m.close_time, reverse=True)[:20]
    for m in markets:
        with st.container(horizontal=True, border=True):
            st.write(f"`{m.ticker}`")
            st.write(f"strike ${m.strike:,.0f}" if m.strike else "strike n/a")
            st.write(f"result: **{m.result.upper()}**")
            st.write(m.close_time.strftime("%H:%M UTC"))
            st.button("Open", key=f"open_{m.ticker}", on_click=_open_window, args=(m.ticker,))


# ---------------------------------------------------------------------------
# View: settled-window detail -- replays the model at fixed checkpoints
# using only data known at that moment (mirrors backtest_strike.py).
# ---------------------------------------------------------------------------
_CHECKPOINT_MINUTES = [3, 7, 11, 13]


def render_window_detail(ticker):
    if st.button(":material/arrow_back: Back to scoreboard", key="back_from_window"):
        _go("home")
        st.rerun()

    markets, err = _get_recent_settled(hours=24.0)
    if err:
        st.error(f"Couldn't load settled markets: {err}")
        return
    market = next((m for m in markets if m.ticker == ticker), None)
    if market is None:
        st.warning("Window not found in the last 24h of settled markets (it may have aged out of cache).")
        return

    st.subheader(f"`{market.ticker}`")
    st.write(f"Strike **${market.strike:,.0f}**  ·  Result: **{market.result.upper()}**  ·  "
             f"{market.open_time:%H:%M} → {market.close_time:%H:%M} UTC")

    open_ts = int(market.open_time.timestamp())
    close_ts = int(market.close_time.timestamp())
    lo = open_ts - config.LOOKBACK_BARS * 60
    hi = close_ts + 60
    with st.spinner("Replaying model checkpoints..."):
        try:
            candles = fetch_range_1min(lo, hi)
        except Exception as exc:
            st.error(f"Couldn't fetch candle history for this window: {exc}")
            return
        by_ts = {c.ts: i for i, c in enumerate(candles)}

        rows = []
        for ck in _CHECKPOINT_MINUTES:
            eval_ts = (open_ts + ck * 60) // 60 * 60
            idx = by_ts.get(eval_ts)
            if idx is None or idx < config.LOOKBACK_BARS:
                continue
            window = candles[idx - config.LOOKBACK_BARS + 1: idx + 1]
            price = window[-1].close
            sig = compute_signals(window)
            state = classify(sig)
            mins_left = (close_ts - eval_ts) / 60.0
            read = evaluate_strike(price, market, sig, state, minutes_remaining=mins_left)
            call = "YES" if read.label == "Favors YES" else ("NO" if read.label == "Favors NO" else "no call")
            correct = (call.lower() == market.result) if call != "no call" else None
            rows.append({
                "minute": ck, "price": f"${price:,.0f}", "state": state.state,
                "confidence": f"{state.confidence}/7", "call": read.label,
                "correct": "n/a" if correct is None else ("✅" if correct else "❌"),
            })

    if not rows:
        st.info("No checkpoints had enough warmup history to evaluate (window too close to the start of fetched data).")
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "Checkpoints replay the model using only data available at that minute of the window — "
        "no lookahead. `correct` compares the call to the actual settlement. A handful of rows "
        "from one window proves nothing on its own; see Model performance for the aggregate."
    )


# ---------------------------------------------------------------------------
# View: Trade log
# ---------------------------------------------------------------------------
def _trade_block(df, mode, title, caption, empty_msg):
    st.markdown(f"### {title}")
    st.caption(caption)
    sub = df[df["mode"] == mode]
    if sub.empty:
        st.info(empty_msg)
        return

    s = trade_log.summarize(sub)
    with st.container(horizontal=True):
        st.metric("Record (W-L)", f"{s['wins']}-{s['losses']}", help=f"{s['pending']} pending", border=True)
        st.metric("Win rate", "n/a" if pd.isna(s["win_rate"]) else f"{s['win_rate']*100:.1f}%", border=True)
        st.metric("Net profit", f"${s['profit']:+.2f}", help=f"Fees paid: ${s['fees_paid']:.2f}", border=True)
        st.metric("ROI", "n/a" if pd.isna(s["roi"]) else f"{s['roi']*100:+.1f}%", border=True)

    if s["n_with_p_model"] > 0:
        gap_txt = "n/a" if pd.isna(s["calibration_gap"]) else f"{s['calibration_gap']*100:+.1f} pts"
        mean_p_txt = "n/a" if pd.isna(s["mean_p_model"]) else f"{s['mean_p_model']*100:.1f}%"
        win_rate_txt = "n/a" if pd.isna(s["win_rate"]) else f"{s['win_rate']*100:.1f}%"
        st.caption(
            f"Calibration check: model's mean stated probability was {mean_p_txt} vs. an actual "
            f"settled win rate of {win_rate_txt} (gap: {gap_txt}, on {s['n_with_p_model']} settled "
            "trades with a logged probability). A large positive gap means the model has been "
            "overconfident live, not just in backtest."
        )

    n_settled = s["wins"] + s["losses"]
    if n_settled < 30:
        st.warning(
            f"Only {n_settled} settled trade(s). Far too small to mean anything — a positive ROI "
            "here is noise, not proven edge. Let the sample grow (this is meant to run for weeks) "
            "before trusting it."
        )

    show = sub[["date", "entry_time", "market_ticker", "side", "entry_price_cents", "size",
                "p_model", "fee", "result", "profit", "notes"]].copy()
    show["result"] = show["result"].fillna("pending")
    st.dataframe(show.sort_values("entry_time", ascending=False), width="stretch", hide_index=True)


def render_trades():
    st.subheader("Trade log")
    df = trade_log.load_log()
    if df.empty:
        st.info(f"No trades logged yet. `paper_trader.py` fills this in automatically as it runs; "
                f"edit `{trade_log.TRADE_LOG_FILE.relative_to(ROOT)}` directly to add real trades.")
        st.dataframe(pd.DataFrame(columns=trade_log.COLUMNS), width="stretch", hide_index=True)
        return

    st.error(
        "**Paper and live trades are shown in separate sections on purpose — never sum them "
        "together.** Paper trades are simulated (no real money); live trades are real fills you "
        "logged by hand. A row with no `mode` set is treated as live, the safer default."
    )

    _trade_block(
        df, "paper", "Paper trading (simulated)",
        "Automated by `paper_trader.py`, running continuously against real live Kalshi quotes "
        "using the validated distance+time+volatility model. Cost includes the real Kalshi taker "
        "fee for every fill — this is testing whether the backtested edge survives real execution "
        "costs, not just settlement accuracy. No real money is at risk.",
        "No paper trades yet — paper_trader.py should be running (check paper_trader.log). "
        "It logs a row automatically whenever the model's edge clears the threshold.",
    )
    st.divider()
    _trade_block(
        df, "live", "Live trades (real money)",
        "Trades you actually placed on Kalshi yourself. Edit "
        f"`{trade_log.TRADE_LOG_FILE.relative_to(ROOT)}` by hand to add one, or fill in `result` "
        "once a window settles.",
        f"No live trades logged yet. Add rows with `mode=live` to "
        f"`{trade_log.TRADE_LOG_FILE.relative_to(ROOT)}`.",
    )


# ---------------------------------------------------------------------------
# View: Model performance -- honest, sourced from the validation writeup
# ---------------------------------------------------------------------------
def render_performance():
    st.subheader("Model performance")
    st.caption(
        "Two separate, independently-tested layers: the 7-indicator direction/confluence "
        "signal, and the experimental strike-distance module. They are NOT equally validated — "
        "read both sections before trusting either."
    )

    st.markdown("### Direction / confluence signal (UPTREND / DOWNTREND state)")
    summary = _load_validation_summary()
    if summary is None:
        st.warning("Validation summary not found at data/validation/direction_signal_backtest_summary.json.")
    else:
        buy15 = summary.get("FULL_SYSTEM_buy (UPTREND state entry)", {}).get("h15", {})
        sell15 = summary.get("FULL_SYSTEM_sell (DOWNTREND state entry)", {}).get("h15", {})
        st.error(
            "**NO VALIDATED EDGE** — backtested on 60 days of real Coinbase BTC-USD 1-min data "
            "(walk-forward, no lookahead, no parameter fitting). At the 15-minute horizon that "
            "matters for Kalshi:"
        )
        if buy15:
            st.write(
                f"- BUY (state → UPTREND): directional win rate **{buy15['directional_win_rate']*100:.1f}%** "
                f"vs. a 50/50 coin flip (p={buy15['p_binom_vs_coinflip']:.4f}) — "
                f"{'significantly worse than chance' if buy15['p_binom_vs_coinflip'] < 0.05 and buy15['directional_win_rate'] < 0.5 else 'not distinguishable from chance'}."
            )
        if sell15:
            st.write(
                f"- SELL (state → DOWNTREND): directional win rate **{sell15['directional_win_rate']*100:.1f}%** "
                f"vs. a 50/50 coin flip (p={sell15['p_binom_vs_coinflip']:.4f})."
            )
        st.caption(
            "Full methodology, all horizons, and the sub-period trend-riding check: see "
            "`tradingview-mcp/research/kalshi-btc-validation/README.md` in the sibling project "
            "(this dashboard ships a static copy of the summary numbers only, so it stays "
            "self-contained for cloud deployment)."
        )

    st.markdown("### Strike-distance / probability model")
    sp = _load_strike_prob_summary()
    if sp is None:
        st.warning("Strike-probability validation summary not found at data/validation/strike_probability_summary.json.")
    else:
        wf = sp["walk_forward_pooled"]["brier"]
        boot = sp["bootstrap_vs_full_model"]
        heur = sp["vs_shipped_heuristic"]
        st.success(
            f"**PARTIALLY VALIDATED** — walk-forward tested on {sp['data']['n_settled_markets']:,} real "
            f"settled Kalshi markets ({sp['data']['date_range']}), chronological folds, no lookahead. "
            "The distance+time+volatility framing shows a real, significant improvement over the naive "
            "\"current side holds\" baseline, consistent across every fold."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Base rate (Brier)", f"{wf['A_base_rate']:.4f}", help="Lower is better. No price info at all.")
        c2.metric("Current-side-holds (Brier)", f"{wf['B_current_side_holds']:.4f}", help="Naive baseline")
        c3.metric("Distance+time+vol (Brier)", f"{wf['C_distance_time_vol']:.4f}",
                  help=f"vs. current-side-holds: p={boot['D_vs_B']['p_value']:.4f} (significant)")
        c4.metric("Full model +confluence (Brier)", f"{wf['D_full_model']:.4f}",
                  help=f"vs. distance+time+vol only: p={boot['D_vs_C']['p_value']:.3f} (NOT significant)")
        st.warning(
            f"**But the 7-indicator confluence engine adds nothing on top of pure distance/time/vol** "
            f"(p={boot['D_vs_C']['p_value']:.3f}, not significant; loses to the simpler model in 2 of 6 "
            "folds) — same conclusion as the direction signal above, from an independent test."
        )
        st.markdown(
            f"**Against the actual shipped heuristic:** it calls {heur['heuristic_call_rate']*100:.0f}% of "
            f"checkpoints at {heur['heuristic_accuracy_on_calls']*100:.1f}% accuracy, but abstains "
            f"(\"too close to call\") on the other {heur['heuristic_abstain_rate']*100:.0f}%. On exactly "
            f"those abstained cases, the fitted model gives "
            f"**{heur['accuracy_on_heuristic_abstained_rows']['full_model']*100:.1f}% accuracy** — "
            "meaningfully better than a coin flip, on the cases where the current bot has nothing to say."
        )
        st.error(
            "**Not yet validated against real trading costs.** These are calibrated probabilities, not "
            "a profitability check — Kalshi's per-contract fee and the real bid-ask spread aren't modeled "
            "here. A ~73% accurate probability is not automatically a profitable trade. Log real fills in "
            "the Trade log page to find out."
        )
        st.caption("Full methodology, per-fold breakdown, and calibration table: "
                   "`research/strike_probability/README.md` in this project.")

    st.markdown("### Exit-timing strategy (buy early, sell on a favorable move)")
    et = _load_exit_timing_summary()
    if et is None:
        st.warning("Exit-timing summary not found at data/validation/exit_timing_summary.json.")
    else:
        hold = et["pnl_comparison"]["hold_to_settlement"]
        best = et["pnl_comparison"]["sell_35"]
        sig = et["hold_vs_best_scalp_significance"]
        st.info(
            f"This is the strategy actually running live in `paper_trader.py` — buy early "
            f"(~60-120s into the window), sell once the position is up "
            f"**{et['deployed_target']*100:.0f}%**, else hold to settlement. Backtested on "
            f"{et['data']['n_entered_trades']:,} real historical trades using real Kalshi "
            "contract prices (not a model-derived proxy)."
        )
        c1, c2 = st.columns(2)
        c1.metric("Hold-to-settlement ROI (baseline)", f"{hold['roi']*100:.1f}%",
                  help=f"Win rate {hold['win_rate']*100:.1f}%")
        c2.metric("Best early-exit ROI (+35%, deployed)", f"{best['roi']*100:.1f}%",
                  help=f"Win rate {best['win_rate']*100:.1f}%")
        st.warning(
            "**Honest finding: holding to settlement outperformed every early-exit threshold "
            "tested (20-40%), consistently.** Hitting a favorable move happens often (69-86% of "
            f"entries) and fast (2-6 min median) — but of trades that hit early, ~77-85% would "
            "have settled in the buyer's favor anyway (bigger payout given up), and only ~15-23% "
            "were genuinely saved from a loss. The gap is consistent but not fully proven "
            f"(bootstrap p={sig['bootstrap_p']:.3f}, not significant at conventional p<0.05)."
        )
        st.caption(
            "Deployed at 35% anyway — paper trading exists to test the strategy actually "
            "intended to be run, not the backtest optimum. Full methodology and the trade-by-"
            "trade decomposition: `research/exit_timing/README.md` in this project."
        )

    st.markdown("### Side-selection model — tested, and rejected")
    ht = _load_hit_target_summary()
    if ht is None:
        st.warning("Summary not found at data/validation/hit_target_side_selection_summary.json.")
    else:
        b = ht["dataset_b_side_selection"]
        rates = b["hit_rate_by_strategy"]
        st.info(
            "Built and walk-forward validated a model specifically targeting \"which side hits "
            "the 30% favorable move\" — a fair question, since nothing before this tested side "
            "selection directly for the scalp strategy. **Result: it doesn't beat a coin flip, "
            "and the model already in production beats it decisively. Not deployed.**"
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Coin flip", f"{rates['coin_flip']*100:.1f}%", help="Hit rate on chosen side")
        c2.metric("Always cheaper side", f"{rates['always_cheaper_side']*100:.1f}%",
                  help="Mechanical, direction-agnostic rule")
        c3.metric("New fitted side-selector", f"{rates['fitted_hit_target_side_selector']*100:.1f}%",
                  help="Purpose-built for this exact question")
        c4.metric("Existing settlement model", f"{rates['existing_settlement_probability_model']*100:.1f}%",
                  help="Already running in production — built for a different question entirely")
        st.warning(
            f"The new model, built specifically to answer this question, ties with a coin flip and "
            f"the naive \"cheaper side\" rule (p={b['significance']['fitted_model_vs_cheaper_side']['p_value']:.3f}, "
            "not significant). The **existing** settlement-probability model — built to predict "
            "settlement, not this — wins by a wide, highly significant margin "
            f"(p={b['significance']['settlement_model_vs_cheaper_side']['p_value']:.4f}) because a "
            "\"likely to settle YES\" read implies a real drift toward 100¢ that a hit-rate-only "
            "model has no way to see."
        )
        st.caption(
            "A validated negative result, not a wasted one — it confirms the side-selection logic "
            "already in `paper_trader.py` from a fully independent angle. Both models were tested "
            "properly (side-symmetric dataset, no circularity) before being set aside — see "
            "`research/exit_timing/README.md` section 5b."
        )


# ---------------------------------------------------------------------------
# View: About & limitations
# ---------------------------------------------------------------------------
def render_about():
    st.subheader("How this works")
    st.markdown(
        """
This dashboard is a **read-only viewer** on top of the Kalshi 15-min BTC Intel bot
(`bot.py`). It never places orders — every number here is informational.

**Data sources:**
- BTC-USD 1-min OHLCV: Coinbase Exchange public REST API (no key needed).
- Active/settled `KXBTC15M` markets: Kalshi public market-data API (no key needed).

**Two prediction layers, tested to very different standards:**
1. **7-indicator confluence + 4-state classifier** (EMA cross, RSI, volume surge, window
   delta, micro-momentum, acceleration, tick trend) — rigorously backtested on real data.
   **Result: no validated directional edge at the 15-minute horizon**, and it doesn't add
   anything to the probability model below either, tested independently. See Model performance.
2. **Distance-to-strike probability model** — logistic regression on distance to strike,
   time remaining, and realized volatility, walk-forward validated against 4,250 real settled
   Kalshi markets. **Result: a real, statistically significant improvement over the naive
   "current side holds" baseline**, and it gives usable ~73% accurate probabilities on the
   ~74% of cases the currently-shipped heuristic just abstains on. The most credible validated
   piece of this bot so far — but not yet checked against real trading fees/spread, so accuracy
   is not the same as proven profitability. See Model performance for the full breakdown.

**Known limitations:**
- Coinbase spot BTC-USD is a proxy for Kalshi's actual settlement index (CF Benchmarks BRTI);
  they can diverge slightly, which matters most near the strike.
- The direction signal fires very frequently (roughly every 18 minutes on average across the
  validation sample) — it behaves more like a running market-state readout than a rare,
  high-conviction turning-point call.
- Nothing here is investment advice. Treat every read as context for your own judgment, not
  a recommendation to take a specific side or size.
        """
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
view = st.session_state.view
if view == "home":
    render_scoreboard()
elif view == "window":
    render_window_detail(st.session_state.window_ticker)
elif view == "trades":
    render_trades()
elif view == "performance":
    render_performance()
elif view == "about":
    render_about()
