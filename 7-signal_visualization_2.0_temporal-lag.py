"""
GA Evolution Replay Animation
──────────────────────────────
Replays the GA evolution history, reconstructs best individual per generation,
runs the full backtest, and produces a dashboard-style MP4 animation.

Supports BOTH JSON formats:
  • Legacy (v<3.0): flat 31-feature space, no lags in feature names
  • v3.0+ Temporal-Feature-Expansion: 155-feature space with _lagN suffixes
    The CSV is automatically expanded in-memory with all required lag columns.

Usage:
    # v3.0 temporal-feature-expansion
    python tmp.py \
        --json ga_evolution_history_2025_3.0_temporal-feature-expansion.json \
        --csv  BTCUSDT_2025_6m_features_5min.csv \
        --out  ga_replay_2025_temporal-feature-expansion.mp4

    # shortcut (from lab)
        python signal_visualization_3.0_temporal-lag.py --json ga_evolution_history_3.0_2024_temporal-feature-expansion.json --csv BTCUSDT_2024_6m_features_5min.csv --out ga_replay_3.0_2024_temporal-feature-expansion.mp4

        python signal_visualization_3.0_temporal-lag.py --json ga_evolution_history_3.0_2024_temporal-feature-expansion.json --csv BTCUSDT_2024_6m_end_features_5min.csv --out ga_replay_3.0_2024_end_temporal-feature-expansion.mp4

    # legacy (31-feature)
    python tmp.py \
        --json ga_evolution_history_2024.json \
        --csv  BTCUSDT_2024_6m_features_5min.csv \
        --out  ga_replay_2024.mp4
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# COLOURS / STYLE
# ──────────────────────────────────────────────────────────────────────────────
BG           = "#0d0d12"
PANEL_BG     = "#12121a"
BORDER       = "#1e2030"
BULL         = "#00e676"
BEAR         = "#ff3d57"
NEUTRAL_BLUE = "#4a90e2"
GOLD         = "#ffd740"
CYAN         = "#40c4ff"
TEXT_PRI     = "#e0e0f0"
TEXT_SEC     = "#7a8094"
TP_COL       = "#00b0ff"
SL_COL       = "#ff6b6b"
TIMEOUT      = "#ffb300"
GRID_COL     = "#1a1a26"

matplotlib.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT_SEC,
    "xtick.color":       TEXT_SEC,
    "ytick.color":       TEXT_SEC,
    "text.color":        TEXT_PRI,
    "grid.color":        GRID_COL,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.4,
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  BORDER,
    "legend.labelcolor": TEXT_SEC,
    "font.size":         9,
})


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS  (mirrors ga_trade_signal_feature_selection*.py)
# ──────────────────────────────────────────────────────────────────────────────

BASE_FEATURE_COLS = [
    "delta_ratio", "buy_sell_ratio", "cvd_slope_5", "cvd_slope_10",
    "cvd_zscore", "notional_buy_ratio", "notional_sell_ratio",
    "large_trade_imbalance", "large_trade_ratio", "trade_intensity",
    "hl_range", "bar_body", "upper_wick", "lower_wick",
    "ema_cross_9_21", "ema_cross_21_50",
    "rsi_14", "rsi_7", "stoch_k", "stoch_d",
    "adx", "adx_diff", "macd", "macd_signal", "macd_diff",
    "atr_ratio", "bb_width", "bb_pct",
    "vol_zscore", "vol_ratio", "notional_zscore",
]

# Fibonacci-spaced lags (matches GA v3.0 Temporal-Feature-Expansion)
DEFAULT_LOOKBACK_LAGS = [1, 2, 3, 5, 8]

# Legacy flat feature list (v<3.0)
LEGACY_FEATURE_COLS = BASE_FEATURE_COLS


def build_lagged_feature_cols(base_cols: list, lags: list) -> list:
    """Reconstruct the expanded feature list in lag-first order."""
    return [f"{col}_lag{lag}" for lag in lags for col in base_cols]


def detect_mode(evo: dict) -> str:
    """
    Return 'temporal' if the JSON was produced by v3.0+, else 'legacy'.
    Detection heuristic: presence of 'lookback_lags' key OR any feature_col
    containing '_lag'.
    """
    if "lookback_lags" in evo:
        return "temporal"
    feature_cols = evo.get("feature_cols", [])
    if feature_cols and any("_lag" in c for c in feature_cols):
        return "temporal"
    # Check first generation's best_features
    gens = evo.get("generations", [])
    if gens:
        bf = gens[0].get("best_features", [])
        if any("_lag" in f for f in bf):
            return "temporal"
    return "legacy"


# ──────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ──────────────────────────────────────────────────────────────────────────────

def _add_lag_columns(df: pd.DataFrame, base_cols: list, lags: list) -> pd.DataFrame:
    """
    Generate lag columns in-memory from the raw CSV columns.
    Each lagged column `{col}_lag{k}` = value of `col` k bars ago (i.e. shift(k)).
    This mirrors exactly what the GA training code does so signals are consistent.
    """
    frames = [df]
    for lag in lags:
        shifted = df[base_cols].shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in base_cols]
        frames.append(shifted)
    return pd.concat(frames, axis=1)


def prepare_df(csv_path: str, feature_cols: list, mode: str,
               base_cols: list = None, lags: list = None) -> pd.DataFrame:
    """
    Load the raw CSV and return a DataFrame ready for signal generation.

    • legacy mode  – behaviour identical to the original tmp.py:
      apply a single shift(1) to the 31 flat features, then scale.

    • temporal mode – generate all lag columns from the raw base features
      (shift(lag) for each lag in `lags`), then scale the expanded set.
      The raw atr_ratio column is preserved for the volatility filter; the
      atr_raw column used by the backtest is taken from atr_ratio_lag1 * close.
    """
    df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time").sort_index()
    df["returns"] = df["close"].pct_change()

    if mode == "legacy":
        df["atr_raw"] = df["atr_ratio"] * df["close"]
        needed = feature_cols + ["open", "high", "low", "close", "returns", "atr_raw"]
        df = df[[c for c in needed if c in df.columns]]

        df[feature_cols] = df[feature_cols].shift(1)
        df["atr_raw"]    = df["atr_raw"].shift(1)
        df = df.dropna()

    else:  # temporal
        # Confirm all base feature columns exist in the CSV
        missing = [c for c in base_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Base feature columns missing from CSV: {missing}\n"
                f"CSV columns: {list(df.columns)}"
            )

        # Generate all lag columns in-memory
        df = _add_lag_columns(df, base_cols, lags)

        # atr_raw for backtest TP/SL: use lag-1 ATR so it's known before entry
        df["atr_raw"] = df["atr_ratio_lag1"] * df["close"]

        # atr_ratio (unlagged) for ATR-ratio filter inside backtest
        # already present from the original CSV

        ohlcr = ["open", "high", "low", "close", "returns", "atr_raw", "atr_ratio"]
        keep  = feature_cols + [c for c in ohlcr if c in df.columns]
        df = df[keep].dropna()

    # ── Scale feature columns with RobustScaler, clip to [-1, 1] ────────────
    scaler = RobustScaler()
    scaler.fit(df[feature_cols])
    df[feature_cols] = scaler.transform(df[feature_cols])
    df[feature_cols] = df[feature_cols].clip(-3, 3) / 3
    return df


# ──────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL RECONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def reconstruct_individual(gen_data: dict, feature_cols: list) -> dict:
    """Build a standard individual dict from a generation's best_* fields."""
    n = len(feature_cols)
    feat_index = {f: i for i, f in enumerate(feature_cols)}

    mask    = [0]   * n
    weights = [0.0] * n

    feat_names   = gen_data["best_features"]
    feat_weights = gen_data["best_weights"]

    for fname, w in zip(feat_names, feat_weights.values()):
        if fname in feat_index:
            idx          = feat_index[fname]
            mask[idx]    = 1
            weights[idx] = float(w)

    return {
        "mask":    mask,
        "weights": weights,
        "buy_th":  float(gen_data["best_buy_th"]),
        "sell_th": float(gen_data["best_sell_th"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, ind: dict, feature_cols: list) -> pd.Series:
    feat_matrix    = df[feature_cols].values
    mask           = np.array(ind["mask"],    dtype=float)
    weights        = np.array(ind["weights"], dtype=float)
    active_weights = mask * weights

    if np.sum(mask) == 0:
        return pd.Series(0, index=df.index)

    norm   = np.sum(np.abs(active_weights)) + 1e-9
    scores = (feat_matrix @ active_weights) / norm
    scores = np.clip(scores, -1, 1)

    signals = np.where(
        scores >  ind["buy_th"],  1,
        np.where(scores < ind["sell_th"], -1, 0)
    )
    return pd.Series(signals, index=df.index)


# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST  (verbatim from original)
# ──────────────────────────────────────────────────────────────────────────────

def backtest_tpsl(df: pd.DataFrame, signals: pd.Series, cfg: dict):
    fee           = cfg.get("fee",           0.0004)
    slippage      = cfg.get("slippage",      0.0002)
    tp_mult       = cfg.get("tp_atr_mult",   3.0)
    sl_mult       = cfg.get("sl_atr_mult",   1.0)
    max_bars_held = cfg.get("max_bars_held", None)
    min_atr_ratio = cfg.get("min_atr_ratio", None)

    closes    = df["close"].values
    highs     = df["high"].values
    lows      = df["low"].values
    atr_vals  = df["atr_raw"].values
    atr_ratio = df["atr_ratio"].values if "atr_ratio" in df.columns else None
    sigs      = signals.values
    n         = len(df)

    pnl_arr  = np.zeros(n)
    pos_arr  = np.zeros(n)
    position = 0
    entry_price = tp_price = sl_price = 0.0
    entry_bar = 0
    trade_log = []

    for i in range(1, n):
        if position == 0:
            sig = sigs[i]
            if sig != 0:
                atr = atr_vals[i]
                if atr <= 0 or np.isnan(atr):
                    continue
                if min_atr_ratio is not None and atr_ratio is not None:
                    if atr_ratio[i] < min_atr_ratio:
                        continue

                entry_price = closes[i] * (1.0 + slippage * sig)
                position    = sig
                entry_bar   = i

                if position == 1:
                    tp_price = entry_price + atr * tp_mult
                    sl_price = entry_price - atr * sl_mult
                else:
                    tp_price = entry_price - atr * tp_mult
                    sl_price = entry_price + atr * sl_mult

                pnl_arr[i] -= (fee + slippage)
                pos_arr[i]   = position
                trade_log.append({
                    "entry_bar":    i,
                    "entry_time":   df.index[i],
                    "direction":    "long" if position == 1 else "short",
                    "entry_price":  entry_price,
                    "tp_price":     tp_price,
                    "sl_price":     sl_price,
                    "atr_at_entry": atr,
                })
        else:
            high = highs[i]
            low  = lows[i]

            hit_tp = (position ==  1 and high >= tp_price) or \
                     (position == -1 and low  <= tp_price)
            hit_sl = (position ==  1 and low  <= sl_price) or \
                     (position == -1 and high >= sl_price)

            bars_in_trade = i - entry_bar
            timed_out     = (max_bars_held is not None) and \
                            (bars_in_trade >= max_bars_held)

            if hit_tp or hit_sl or timed_out:
                if hit_sl:
                    exit_price = sl_price;  exit_type = "SL"
                elif hit_tp:
                    exit_price = tp_price;  exit_type = "TP"
                else:
                    exit_price = closes[i] * (1.0 - slippage * position)
                    exit_type  = "TIMEOUT"

                trade_pnl   = position * (exit_price - entry_price) / entry_price
                trade_pnl  -= (fee + slippage)
                pnl_arr[i] += trade_pnl
                pos_arr[i]   = position

                if trade_log:
                    trade_log[-1].update({
                        "exit_bar":   i,
                        "exit_time":  df.index[i],
                        "exit_price": exit_price,
                        "exit_type":  exit_type,
                        "pnl":        trade_pnl,
                        "bars_held":  bars_in_trade,
                    })
                position = 0
            else:
                pnl_arr[i] = position * (closes[i] - closes[i-1]) / closes[i-1]
                pos_arr[i]  = position

    return (pd.Series(pnl_arr, index=df.index),
            pd.Series(pos_arr, index=df.index),
            trade_log)


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(strat_ret: pd.Series, positions: pd.Series, bpy: int = 72576):
    sr     = strat_ret.fillna(0)
    eq     = (1 + sr).cumprod()
    tot    = float(eq.iloc[-1] - 1)
    mu, sigma = sr.mean(), sr.std()
    sharpe = float((mu / sigma) * np.sqrt(bpy)) if sigma >= 1e-9 else -999.0
    dd     = float((eq / eq.cummax() - 1).min())
    active = sr[sr != 0]
    wr     = float((active > 0).mean()) if len(active) > 0 else 0.0
    return dict(sharpe=sharpe, total_return=tot, max_dd=dd, win_rate=wr)


def compute_trade_stats(trade_log):
    closed = [t for t in trade_log if "exit_type" in t]
    if not closed:
        return dict(n=0, tp=0, sl=0, timeout=0,
                    tp_pct=0, sl_pct=0, to_pct=0, avg_hold=0, pf=0)
    n       = len(closed)
    tp_list = [t for t in closed if t["exit_type"] == "TP"]
    sl_list = [t for t in closed if t["exit_type"] == "SL"]
    to_list = [t for t in closed if t["exit_type"] == "TIMEOUT"]
    pnls    = [t["pnl"] for t in closed]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p <= 0]
    pf      = abs(sum(wins) / sum(losses)) if losses else float("inf")
    holds   = [t["bars_held"] for t in closed]
    return dict(n=n,
                tp=len(tp_list), sl=len(sl_list), timeout=len(to_list),
                tp_pct=len(tp_list)/n, sl_pct=len(sl_list)/n,
                to_pct=len(to_list)/n,
                avg_hold=float(np.mean(holds)),
                pf=pf)


# ──────────────────────────────────────────────────────────────────────────────
# ANIMATION BUILD
# ──────────────────────────────────────────────────────────────────────────────

def build_animation(df: pd.DataFrame,
                    history: list,
                    global_cfg: dict,
                    feature_cols: list,
                    mode: str,
                    lags: list,
                    output_path: str,
                    fps: int = 8,
                    interval_ms: int = 125):

    N_FEATURES = len(feature_cols)
    N_GEN      = len(history)
    n_bars     = len(df)
    xs         = np.arange(n_bars)

    version_tag = global_cfg.get("version", "legacy")
    lag_tag     = (f"lags [{','.join(str(l) for l in lags)}]"
                   if mode == "temporal" else "lag 1 only")

    # ── Pre-compute all generations ─────────────────────────────────────────
    print(f"Pre-computing {N_GEN} generations …", flush=True)
    gen_results = []
    for gi, gd in enumerate(history):
        ind              = reconstruct_individual(gd, feature_cols)
        sigs             = generate_signals(df, ind, feature_cols)
        sr, pos, tlog    = backtest_tpsl(df, sigs, global_cfg)
        eq               = (1 + sr.fillna(0)).cumprod()
        ts               = compute_trade_stats(tlog)
        metrics          = compute_metrics(sr, pos)
        closed           = [t for t in tlog if "exit_type" in t]
        gen_results.append(dict(
            ind=ind, signals=sigs, strat_ret=sr,
            equity=eq, trade_log=tlog, closed=closed,
            ts=ts, metrics=metrics, gd=gd,
        ))
        if (gi + 1) % 10 == 0 or gi == N_GEN - 1:
            print(f"  gen {gi+1:>4d}/{N_GEN}  "
                  f"best_fit={gd['best_fitness']:.4f}  "
                  f"trades={len(closed)}", flush=True)

    price_lo = df["low"].min()  * 0.998
    price_hi = df["high"].max() * 1.002

    # ── Layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(19, 9), dpi=110, facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1],
                            left=0.04, right=0.98,
                            top=0.88, bottom=0.06, wspace=0.03)

    ax_candle = fig.add_subplot(gs[0, 0])
    ax_stats  = fig.add_subplot(gs[0, 1])

    for ax in (ax_candle, ax_stats):
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
    ax_stats.set_xticks([])
    ax_stats.set_yticks([])

    # ─ Static price line ────────────────────────────────────────────────────
    ax_candle.set_xlim(-1, n_bars + 1)
    ax_candle.set_ylim(price_lo, price_hi)
    ax_candle.set_ylabel("Price (USDT)", color=TEXT_SEC, fontsize=15)
    ax_candle.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_candle.set_xticks([])
    ax_candle.grid(True, axis="y", alpha=0.3)
    ax_candle.plot(xs, df["close"].values,
                   color=NEUTRAL_BLUE, lw=1.5, alpha=0.9, zorder=2, label="Price")

    # ─ Dynamic layers ───────────────────────────────────────────────────────
    scat_buy,  = ax_candle.plot([], [], "^", color=BULL,   ms=12, zorder=6,
                                mec="#00ff4c", mew=0.5, label="BUY")
    scat_sell, = ax_candle.plot([], [], "v", color=BEAR,   ms=12, zorder=6,
                                mec="#ff6b6b", mew=0.5, label="SELL")
    trade_lines_col = LineCollection([], colors=[], linewidths=1.2,
                                     linestyles="--", alpha=0.6, zorder=4)
    ax_candle.add_collection(trade_lines_col)
    tp_scat,   = ax_candle.plot([], [], "D", color=TP_COL, ms=5, zorder=7,
                                label="TP hit", mec="white", mew=0.3)
    sl_scat,   = ax_candle.plot([], [], "X", color=SL_COL, ms=5, zorder=7,
                                label="SL hit", mec="white", mew=0.3)
    to_scat,   = ax_candle.plot([], [], "s", color=TIMEOUT, ms=4, zorder=7,
                                label="Timeout", mec="white", mew=0.3)

    ax_candle.legend(loc="upper left", fontsize=10, ncol=6,
                     framealpha=0.6, handletextpad=0.3, columnspacing=0.8)

    gen_label = ax_candle.text(
        0.99, 0.97, "", transform=ax_candle.transAxes,
        ha="right", va="top", fontsize=15, color=GOLD, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor=PANEL_BG, edgecolor=BORDER, alpha=0.8))

    stats_text = ax_stats.text(
        0.08, 0.95, "", transform=ax_stats.transAxes,
        ha="left", va="top", fontsize=15, color=TEXT_PRI,
        fontfamily="monospace", fontweight="semibold", linespacing=1.6)

    # ─ Titles ───────────────────────────────────────────────────────────────
    fig.text(0.50, 0.965,
             "Genetic Algorithm Trade Signal Evolution — BTC/USDT 5-min",
             ha="center", va="top", fontsize=22,
             color=TEXT_PRI, fontweight="bold")

    fig.text(0.50, 0.9250,
             f"TP={global_cfg.get('tp_atr_mult', 3.0)}×ATR  "
             f"SL={global_cfg.get('sl_atr_mult', 1.0)}×ATR  "
             f"v{version_tag}  |  {N_FEATURES} features  {lag_tag}  |  "
             f"{n_bars:,} bars",
             ha="center", va="top", fontsize=14, color=TEXT_SEC)

    # ── Update function ──────────────────────────────────────────────────────
    def update(frame_idx):
        r      = gen_results[frame_idx]
        gd     = r["gd"]
        ind    = r["ind"]
        sigs   = r["signals"]
        closed = r["closed"]
        ts     = r["ts"]
        m      = r["metrics"]

        # Signals
        buy_xs  = xs[sigs.values ==  1]
        sell_xs = xs[sigs.values == -1]
        scat_buy.set_data(buy_xs,  df["close"].values[buy_xs]  * 0.996)
        scat_sell.set_data(sell_xs, df["close"].values[sell_xs] * 1.004)

        # Trade outcome markers
        tp_xs, tp_ys = [], []
        sl_xs, sl_ys = [], []
        to_xs, to_ys = [], []
        trade_segs, trade_colors = [], []

        for t in closed:
            eb, xb = t["entry_bar"], t["exit_bar"]
            et     = t["exit_type"]
            col    = TP_COL if et == "TP" else (SL_COL if et == "SL" else TIMEOUT)
            trade_segs.append([(eb, t["entry_price"]), (xb, t["exit_price"])])
            trade_colors.append(col)
            if et == "TP":
                tp_xs.append(xb); tp_ys.append(t["tp_price"])
            elif et == "SL":
                sl_xs.append(xb); sl_ys.append(t["sl_price"])
            else:
                to_xs.append(xb); to_ys.append(t["exit_price"])

        trade_lines_col.set_segments(trade_segs)
        trade_lines_col.set_colors(trade_colors)
        tp_scat.set_data(tp_xs, tp_ys)
        sl_scat.set_data(sl_xs, sl_ys)
        to_scat.set_data(to_xs, to_ys)

        # Generation label
        gen_label.set_text(f"Gen {gd['generation']:>3d}/{N_GEN}")

        # Stats panel
        pf_str = f"{ts['pf']:.2f}" if ts["pf"] < 100 else "inf"

        # Active-lag breakdown (temporal mode only)
        if mode == "temporal":
            active_feats = [feature_cols[i]
                            for i, m_val in enumerate(ind["mask"]) if m_val]
            lag_counts = {}
            for lag in lags:
                cnt = sum(1 for f in active_feats if f.endswith(f"_lag{lag}"))
                if cnt:
                    lag_counts[lag] = cnt
            lag_line = "  ".join(
                f"lag{k}:{v}" for k, v in sorted(lag_counts.items()))
        else:
            lag_line = ""

        n_active = int(sum(ind["mask"]))
        lines = [
            "--- PERFORMANCE ---",
            "",
            f"Total Return  {m['total_return']:>+8.2%}",
            f"Sharpe Ratio  {m['sharpe']:>8.2f}",
            f"Max Drawdown  {m['max_dd']:>8.2%}",
            f"Win Rate      {m['win_rate']:>8.2%}",
            "",
            "--- TRADE STATS ---",
            "",
            f"Closed Trades {ts['n']:>8d}",
            f"TP Hits       {ts['tp']:>5d}  ({ts['tp_pct']:>5.1%})",
            f"SL Hits       {ts['sl']:>5d}  ({ts['sl_pct']:>5.1%})",
            f"Timeouts      {ts['timeout']:>5d}  ({ts['to_pct']:>5.1%})",
            "",
            f"Profit Factor {pf_str:>8s}",
            f"Avg Bars Held {ts['avg_hold']:>8.1f}",
            "",
            "--- GA FITNESS ---",
            "",
            f"Best Fitness  {gd['best_fitness']:>8.4f}",
            f"Mean Fitness  {gd['mean_fitness']:>8.4f}",
            f"Features Used {n_active:>5d}/{N_FEATURES}",
        ]
        if mode == "temporal" and lag_line:
            lines += ["", "--- LAG BREAKDOWN ---", "", lag_line]

        stats_text.set_text("\n".join(lines))

        return (scat_buy, scat_sell, trade_lines_col,
                tp_scat, sl_scat, to_scat, gen_label, stats_text)

    # ── Export ───────────────────────────────────────────────────────────────
    print(f"\nRendering {N_GEN} frames → {output_path} …", flush=True)
    anim = FuncAnimation(fig, update, frames=N_GEN,
                         interval=interval_ms, blit=False)
    writer = FFMpegWriter(fps=fps,
                          metadata=dict(title="GA Evolution Replay",
                                        artist="GA Visualiser"),
                          extra_args=["-crf", "18", "-preset", "slow",
                                      "-pix_fmt", "yuv420p"])
    anim.save(output_path, writer=writer,
              savefig_kwargs=dict(facecolor=BG))
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="GA Evolution Replay Animator")
    ap.add_argument("--json", required=True, help="ga_evolution_history.json")
    ap.add_argument("--csv",  required=True, help="BTCUSDT features CSV")
    ap.add_argument("--out",  default="ga_replay.mp4", help="output MP4")
    ap.add_argument("--fps",  type=int, default=8)
    args = ap.parse_args()

    # ── Load JSON ─────────────────────────────────────────────────────────
    print(f"Loading evolution history: {args.json}")
    with open(args.json) as f:
        evo = json.load(f)

    history    = evo["generations"]
    global_cfg = evo.get("config", {})
    global_cfg.setdefault("fee",           0.0004)
    global_cfg.setdefault("slippage",      0.0002)
    global_cfg.setdefault("tp_atr_mult",   3.0)
    global_cfg.setdefault("sl_atr_mult",   1.0)
    global_cfg.setdefault("max_bars_held", None)
    global_cfg.setdefault("min_atr_ratio", None)

    # ── Detect legacy vs temporal-feature-expansion mode ──────────────────
    mode = detect_mode(evo)
    print(f"  Mode        : {mode}")

    if mode == "temporal":
        # Prefer lags/features recorded in the JSON; fall back to defaults
        lags         = evo.get("lookback_lags",    DEFAULT_LOOKBACK_LAGS)
        base_cols    = evo.get("base_feature_cols", BASE_FEATURE_COLS)
        feature_cols = evo.get("feature_cols",
                               build_lagged_feature_cols(base_cols, lags))
        print(f"  Lags        : {lags}")
        print(f"  Base feats  : {len(base_cols)}")
        print(f"  Total feats : {len(feature_cols)}")
    else:
        lags         = [1]
        base_cols    = LEGACY_FEATURE_COLS
        feature_cols = LEGACY_FEATURE_COLS
        print(f"  Total feats : {len(feature_cols)}  (legacy flat)")

    print(f"  Generations : {len(history)}")
    print(f"  Config      : {global_cfg}")

    # ── Load and prepare CSV ──────────────────────────────────────────────
    print(f"\nLoading market data: {args.csv}")
    df = prepare_df(args.csv,
                    feature_cols=feature_cols,
                    mode=mode,
                    base_cols=base_cols,
                    lags=lags)
    print(f"  Bars loaded : {len(df):,}  "
          f"({df.index[0].date()} → {df.index[-1].date()})")

    # ── Animate ──────────────────────────────────────────────────────────
    build_animation(df, history, global_cfg,
                    feature_cols=feature_cols,
                    mode=mode,
                    lags=lags,
                    output_path=args.out,
                    fps=args.fps)


if __name__ == "__main__":
    main()