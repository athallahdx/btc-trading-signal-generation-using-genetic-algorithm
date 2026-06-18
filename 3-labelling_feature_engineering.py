import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import os
import glob

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — change these
# ─────────────────────────────────────────────────────────────────────────────
TIMEFRAME  = "5m"   # "5m" or "10m"

MONTHS = [
    "BTCUSDT-2024-7-July",
    "BTCUSDT-2024-8-August",
    "BTCUSDT-2024-9-September",
    "BTCUSDT-2024-10-October",
    "BTCUSDT-2024-11-November",
    "BTCUSDT-2024-12-December",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def resolve_paths(timeframe):
    if timeframe == "5m":
        subdir   = "Processed_5m"
        filename = "ALL_5min.csv"
        suffix   = "5min"
    elif timeframe == "10m":
        subdir   = "Processed_10m"
        filename = "ALL_10min.csv"
        suffix   = "10min"
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}. Use '5m' or '10m'.")
    return subdir, filename, suffix


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD ALL MONTHS
# ─────────────────────────────────────────────────────────────────────────────

def load_all_months(months, timeframe):
    subdir, filename, suffix = resolve_paths(timeframe)
    dfs = []

    for month in months:
        path = os.path.join(month, subdir, filename)
        if not os.path.exists(path):
            print(f"  [WARN] Not found, skipping: {path}")
            continue
        df = pd.read_csv(path, parse_dates=["time"], index_col="time")
        df = df.sort_index()
        dfs.append(df)
        print(f"  Loaded {month:<35} {len(df):>7,} bars")

    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    print(f"\n  Total: {len(combined):,} bars "
          f"({combined.index[0].date()} -> {combined.index[-1].date()})")
    return combined


# -----------------------------------------------------------------------------
# 2. LABEL MARKET REGIME
# ─────────────────────────────────────────────────────────────────────────────

def label_regime(df, window=20, vol_threshold=0.6, trend_threshold=0.001):
    df = df.copy()

    df["returns"]      = df["close"].pct_change()
    df["volatility"]   = df["returns"].rolling(window).std()
    df["vol_quantile"] = df["volatility"].rolling(window * 5).rank(pct=True)

    df["ema"]          = df["close"].ewm(span=window).mean()
    df["ema_slope"]    = df["ema"].pct_change(5)

    adx                = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=window)
    df["adx"]          = adx.adx()
    df["adx_pos"]      = adx.adx_pos()
    df["adx_neg"]      = adx.adx_neg()

    high_vol     = df["vol_quantile"] >= vol_threshold
    strong_trend = df["adx"] >= 25

    conditions = [
        strong_trend & (df["adx_pos"] > df["adx_neg"]) & (df["ema_slope"] >  trend_threshold),
        strong_trend & (df["adx_pos"] < df["adx_neg"]) & (df["ema_slope"] < -trend_threshold),
        high_vol & ~strong_trend,
        ~strong_trend & ~high_vol,
    ]
    labels = ["trending_up", "trending_down", "high_volatility", "ranging"]

    df["regime"] = np.select(conditions, labels, default="ranging")
    df["regime"] = df["regime"].astype("category")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df):
    df = df.copy()

    # ── order flow ──────────────────────────────────────────────────────────
    df["delta_ratio"]           = df["delta"]        / (df["volume"]       + 1e-9)
    df["buy_sell_ratio"]        = df["buy_vol"]       / (df["sell_vol"]     + 1e-9)
    df["cvd_slope_5"]           = df["cvd"].diff(5)
    df["cvd_slope_10"]          = df["cvd"].diff(10)
    df["cvd_zscore"]            = (df["cvd"] - df["cvd"].rolling(20).mean()) \
                                  / (df["cvd"].rolling(20).std() + 1e-9)
    df["notional_buy_ratio"]    = df["buy_notional"]  / (df["notional"]     + 1e-9)
    df["notional_sell_ratio"]   = df["sell_notional"] / (df["notional"]     + 1e-9)
    df["large_trade_imbalance"] = df["large_buy_count"] - df["large_sell_count"]
    df["large_trade_ratio"]     = df["large_buy_count"] / (df["large_sell_count"] + 1e-9)
    df["trade_intensity"]       = df["trades"]        / (df["volume"]       + 1e-9)

    # ── price action ────────────────────────────────────────────────────────
    df["returns"]               = df["close"].pct_change()
    df["returns_5"]             = df["close"].pct_change(5)
    df["returns_10"]            = df["close"].pct_change(10)
    df["hl_range"]              = (df["high"] - df["low"])                    / (df["close"] + 1e-9)
    df["bar_body"]              = (df["close"] - df["open"])                  / (df["close"] + 1e-9)
    df["upper_wick"]            = (df["high"] - df[["open","close"]].max(axis=1)) / (df["close"] + 1e-9)
    df["lower_wick"]            = (df[["open","close"]].min(axis=1) - df["low"]) / (df["close"] + 1e-9)

    # ── moving averages ─────────────────────────────────────────────────────
    for span in [9, 21, 50]:
        df[f"ema_{span}"]       = df["close"].ewm(span=span).mean()
    df["ema_cross_9_21"]        = (df["ema_9"]  - df["ema_21"]) / (df["close"] + 1e-9)
    df["ema_cross_21_50"]       = (df["ema_21"] - df["ema_50"]) / (df["close"] + 1e-9)

    # ── momentum ────────────────────────────────────────────────────────────
    df["rsi_14"]                = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["rsi_7"]                 = ta.momentum.RSIIndicator(df["close"], window=7).rsi()
    stoch                       = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"]               = stoch.stoch()
    df["stoch_d"]               = stoch.stoch_signal()

    # ── trend ───────────────────────────────────────────────────────────────
    adx                         = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"]                   = adx.adx()
    df["adx_pos"]               = adx.adx_pos()
    df["adx_neg"]               = adx.adx_neg()
    df["adx_diff"]              = df["adx_pos"] - df["adx_neg"]
    macd                        = ta.trend.MACD(df["close"])
    df["macd"]                  = macd.macd()
    df["macd_signal"]           = macd.macd_signal()
    df["macd_diff"]             = macd.macd_diff()

    # ── volatility ──────────────────────────────────────────────────────────
    df["atr"]                   = ta.volatility.AverageTrueRange(
                                      df["high"], df["low"], df["close"]).average_true_range()
    df["atr_ratio"]             = df["atr"] / (df["close"] + 1e-9)
    bb                          = ta.volatility.BollingerBands(df["close"])
    df["bb_width"]              = bb.bollinger_wband()
    df["bb_pct"]                = bb.bollinger_pband()

    # ── volume ──────────────────────────────────────────────────────────────
    df["vol_zscore"]            = (df["volume"] - df["volume"].rolling(20).mean()) \
                                  / (df["volume"].rolling(20).std() + 1e-9)
    df["vol_ratio"]             = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)
    df["notional_zscore"]       = (df["notional"] - df["notional"].rolling(20).mean()) \
                                  / (df["notional"].rolling(20).std() + 1e-9)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. SUMMARY & PLOT
# ─────────────────────────────────────────────────────────────────────────────

def regime_summary(df):
    counts  = df["regime"].value_counts()
    pct     = df["regime"].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"count": counts, "pct": pct.round(2)})
    print("\n-- Regime Distribution --------------------------")
    print(summary)
    return summary

def plot_regime(df, n_bars=1000, suffix="5min"):
    colors = {
        "trending_up":     "lime",
        "trending_down":   "red",
        "high_volatility": "orange",
        "ranging":         "gray",
    }
    sample = df.iloc[-n_bars:]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#0f0f0f")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#333333")

    axes[0].plot(sample.index, sample["close"], color="white", linewidth=0.7)
    for regime, color in colors.items():
        mask = sample["regime"] == regime
        axes[0].fill_between(sample.index,
                             sample["close"].min(), sample["close"].max(),
                             where=mask, alpha=0.15, color=color, label=regime)
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="upper left", fontsize=8, facecolor="#1a1a1a", labelcolor="white")
    axes[0].set_title(f"BTCUSDT {suffix} — Price with Market Regime", color="white")

    bar_width = 0.003 if suffix == "5min" else 0.006
    delta_colors = ["lime" if d >= 0 else "red" for d in sample["delta"]]
    axes[1].bar(sample.index, sample["delta"], color=delta_colors, width=bar_width)
    axes[1].axhline(0, color="#555555", linewidth=0.5)
    axes[1].set_ylabel("Delta")

    axes[2].plot(sample.index, sample["cvd"], color="#4da6ff", linewidth=0.8)
    axes[2].axhline(0, color="#555555", linewidth=0.5)
    axes[2].set_ylabel("CVD")

    plt.tight_layout()
    fname = f"regime_plot_{suffix}.png"
    plt.savefig(fname, dpi=150, facecolor="#0f0f0f")
    plt.show()
    print(f"Saved -> {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save(df, suffix, months):
    try:
        # Extract year from first month string like "BTCUSDT-2025-1-January" -> "2025"
        year = months[0].split("-")[1]
    except Exception:
        year = "2024"
    fname = f"BTCUSDT_{year}_6m_end_features_{suffix}.csv"
    df.to_csv(fname, index=True)
    print(f"\nSaved  -> {os.path.abspath(fname)}")
    print(f"Shape  : {df.shape}")
    print(f"Columns: {len(df.columns)}")
    return fname


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _, _, suffix = resolve_paths(TIMEFRAME)

    print(f"\nTimeframe : {TIMEFRAME}  ({suffix})")
    print(f"Months    : {len(MONTHS)}")
    print(f"\nLoading all months...")
    df = load_all_months(MONTHS, TIMEFRAME)

    print("\nLabeling regimes...")
    df = label_regime(df)

    print("Building features...")
    df = build_features(df)

    df = df.dropna()
    print(f"After dropna: {len(df):,} bars")

    regime_summary(df)
    plot_regime(df, n_bars=1000, suffix=suffix)
    save(df, suffix, MONTHS)