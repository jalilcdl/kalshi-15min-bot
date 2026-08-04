"""
Is there a real hour-of-day or day-of-week edge in this strategy?

User's question, verbatim: "based on the backtesting from Kalshi, when would
be the best times of the day to trade? Which days of the week?" Explicitly
flagged as prone to false positives (24 hours x 7 days = lots of chances for
a random pattern to look significant by chance alone) -- so this uses an
omnibus test FIRST (one test per family: "is there any real variation across
hours at all", "is there any real variation across days at all") before ever
looking at individual buckets, which sidesteps the 24/7-way multiple-comparison
problem almost entirely (2 omnibus tests, not 31 pairwise ones). Only if an
omnibus test rejects does this go bucket-by-bucket, and even then every
individual bucket p-value is reported against a Bonferroni-corrected threshold
(0.05 / 31 lookups = 0.00161), not the raw 0.05.

Primary metric: hit rate on the deployed +35% exit target (config.
PAPER_TRADE_EXIT_TARGET), using the "close" (conservative) hit definition
already established as this project's headline metric in
research/exit_timing/README.md. Data: exit_timing_trades.csv, the real
checkpoint-1 (~60-120s entry, matching production) dataset -- 2,790 real
would-enter trades, real candlestick price paths, no synthetic data.

Secondary/mechanism check: BTC realized volatility (real Coinbase 1-min
data) and Kalshi contract bid/ask spread (real candlestick data) by hour --
a plausible pattern needs a plausible cause, not just a number that happened
to be different.

Usage: python hour_day_backtest.py
"""
import numpy as np
import pandas as pd
from scipy import stats

TRADES = "../../exit_timing/results/exit_timing_trades.csv"
SETTLED = "../../strike_probability/data/settled_markets.csv"
BTC_1MIN = "../../strike_probability/data/btc_1min.csv"
CANDLES = "../../exit_timing/data/candlesticks.csv"
OUT_DIR = "../results"

N_LOOKUPS = 24 + 7  # hour buckets + day buckets, for Bonferroni correction
BONFERRONI_ALPHA = 0.05 / N_LOOKUPS
MIN_BUCKET_N = 30  # below this, don't trust a per-bucket rate at all


def load_trades_with_time():
    trades = pd.read_csv(TRADES)
    settled = pd.read_csv(SETTLED)[["ticker", "open_time"]]
    settled["open_time"] = pd.to_datetime(settled["open_time"], utc=True)
    df = trades.merge(settled, on="ticker", how="inner")
    assert len(df) == len(trades), f"lost rows on join: {len(trades)} -> {len(df)}"
    df["hour_utc"] = df["open_time"].dt.hour
    df["hour_et"] = (df["open_time"] - pd.Timedelta(hours=4)).dt.hour  # UTC-4, matches user's local TZ (confirmed elsewhere in this project)
    df["dow"] = df["open_time"].dt.dayofweek  # 0=Mon .. 6=Sun
    df["hit35"] = df["hit_close_35"].astype(bool)
    return df


def omnibus_test(df, group_col, label):
    """Chi-square test of independence: does hit rate vary across this
    grouping at all? One test for the whole family -- this is what actually
    controls the false-positive rate here, not the per-bucket p-values."""
    table = pd.crosstab(df[group_col], df["hit35"])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    print(f"\n=== Omnibus test: does hit rate vary by {label}? ===")
    print(f"chi2={chi2:.2f}, dof={dof}, p={p:.4f}")
    if p < 0.05:
        print(f"p < 0.05 -- there IS real variation across {label}. "
              f"Proceeding to per-bucket comparison (Bonferroni alpha={BONFERRONI_ALPHA:.5f}).")
    else:
        print(f"p >= 0.05 -- NO evidence of real variation across {label}. "
              f"Per-bucket numbers below are provided for transparency only -- "
              f"do not read any single bucket as a finding.")
    return p


def per_bucket_table(df, group_col, label, order=None):
    overall_rate = df["hit35"].mean()
    overall_n = len(df)
    rows = []
    groups = df.groupby(group_col)
    for key, g in groups:
        n = len(g)
        hits = g["hit35"].sum()
        rate = hits / n if n else float("nan")
        if n < MIN_BUCKET_N:
            rows.append((key, n, rate, None, "insufficient N (<30)"))
            continue
        # two-proportion z-test: this bucket vs. everyone else pooled
        rest_n = overall_n - n
        rest_hits = df["hit35"].sum() - hits
        rest_rate = rest_hits / rest_n
        pooled_rate = (hits + rest_hits) / (n + rest_n)
        se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1 / n + 1 / rest_n))
        z = (rate - rest_rate) / se if se > 0 else 0.0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        flag = "CLEARS Bonferroni" if p < BONFERRONI_ALPHA else ""
        rows.append((key, n, rate, p, flag))
    out = pd.DataFrame(rows, columns=[group_col, "n", "hit_rate", "p_vs_rest", "flag"])
    if order is not None:
        out = out.set_index(group_col).reindex(order).reset_index()
    print(f"\n--- Per-bucket hit rate on +35% target by {label} (overall: {overall_rate:.1%}, n={overall_n}) ---")
    print(out.to_string(index=False))
    return out


def mechanism_check(df):
    """Plausible-mechanism check: real BTC 1-min realized volatility and real
    Kalshi contract spread, by UTC hour. A pattern with no plausible mechanism
    behind it is much more likely to be noise, even if it clears significance."""
    btc = pd.read_csv(BTC_1MIN)
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], unit="s", utc=True)
    btc["hour_utc"] = btc["timestamp"].dt.hour
    btc = btc.sort_values("timestamp")
    btc["logret"] = np.log(btc["close"] / btc["close"].shift(1))
    vol_by_hour = btc.groupby("hour_utc")["logret"].std() * 100  # in %, 1-min realized vol
    vol_by_hour.name = "btc_1min_realized_vol_pct"

    candles = pd.read_csv(CANDLES)
    candles["ts"] = pd.to_datetime(candles["ts"], unit="s", utc=True)
    candles["hour_utc"] = candles["ts"].dt.hour
    candles["spread"] = candles["yes_ask_close"] - candles["yes_bid_close"]
    spread_by_hour = candles.groupby("hour_utc")["spread"].mean()
    spread_by_hour.name = "kalshi_mean_spread"

    mech = pd.concat([vol_by_hour, spread_by_hour], axis=1)
    print("\n--- Mechanism check: real BTC volatility & real Kalshi spread by UTC hour ---")
    print(mech.to_string())
    corr = mech["btc_1min_realized_vol_pct"].corr(mech["kalshi_mean_spread"])
    print(f"\ncorrelation(volatility, spread) across hours: {corr:.2f} "
          "(sanity check -- these should move together if both reflect the same liquidity/activity cycle)")
    return mech


def main():
    df = load_trades_with_time()
    print(f"Loaded {len(df)} real checkpoint-1 (~60-120s entry) trades, "
          f"{df['open_time'].dt.date.nunique()} unique days, "
          f"{df['open_time'].min().date()} to {df['open_time'].max().date()}")

    p_hour = omnibus_test(df, "hour_utc", "hour of day (UTC)")
    p_dow = omnibus_test(df, "dow", "day of week")

    hour_table = per_bucket_table(df, "hour_utc", "hour of day (UTC)", order=list(range(24)))
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_table = per_bucket_table(df, "dow", "day of week", order=list(range(7)))
    dow_table["dow_name"] = dow_table["dow"].map(dict(enumerate(dow_names)))

    mech = mechanism_check(df)

    hour_table.to_csv(f"{OUT_DIR}/hour_of_day_hit_rate.csv", index=False)
    dow_table.to_csv(f"{OUT_DIR}/day_of_week_hit_rate.csv", index=False)
    mech.to_csv(f"{OUT_DIR}/mechanism_check_by_hour.csv")

    print("\n=== SUMMARY ===")
    print(f"Hour-of-day omnibus p={p_hour:.4f} -> {'REAL VARIATION' if p_hour < 0.05 else 'no evidence of real variation'}")
    print(f"Day-of-week omnibus p={p_dow:.4f} -> {'REAL VARIATION' if p_dow < 0.05 else 'no evidence of real variation'}")
    n_hour_sig = (hour_table["p_vs_rest"].notna() & (hour_table["p_vs_rest"] < BONFERRONI_ALPHA)).sum()
    n_dow_sig = (dow_table["p_vs_rest"].notna() & (dow_table["p_vs_rest"] < BONFERRONI_ALPHA)).sum()
    print(f"Hours clearing Bonferroni (alpha={BONFERRONI_ALPHA:.5f}): {n_hour_sig} of 24")
    print(f"Days clearing Bonferroni (alpha={BONFERRONI_ALPHA:.5f}): {n_dow_sig} of 7")


if __name__ == "__main__":
    main()
