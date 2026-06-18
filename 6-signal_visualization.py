"""
GA Evolution Replay Animation
──────────────────────────────
Replays the GA evolution history, reconstructs best individual per generation,
runs the full backtest, and produces a dashboard-style MP4 animation.

Usage:
    python ga_replay_animation.py \
        --json  /path/to/ga_evolution_history.json \
        --
           /path/to/BTCUSDT_features.csv  \
        --out   ga_replay.mp4
        python signal_visualization.py --json ga_evolution_history_3.0_2024_temporal-feature-expansion.json --csv BTCUSDT_2024_6m_features_5min.csv --out ga_replay_2.0_2024.mp4
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# COLOURS / STYLE
# ──────────────────────────────────────────────────────────────────────────────
BG        = "#0d0d12"
PANEL_BG  = "#12121a"
BORDER    = "#1e2030"
BULL      = "#00e676"
BEAR      = "#ff3d57"
NEUTRAL   = "#607d8b"
NEUTRAL_BLUE = "#4a90e2"
GOLD      = "#ffd740"
CYAN      = "#40c4ff"
PURPLE    = "#ce93d8"
TEXT_PRI  = "#e0e0f0"
TEXT_SEC  = "#7a8094"
TP_COL    = "#00b0ff"
SL_COL    = "#ff6b6b"
TIMEOUT   = "#ffb300"
GRID_COL  = "#1a1a26"

FONT_MONO = {"fontfamily": "monospace"}
FONT_SANS = {}

matplotlib.rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     PANEL_BG,
    "axes.edgecolor":     BORDER,
    "axes.labelcolor":    TEXT_SEC,
    "xtick.color":        TEXT_SEC,
    "ytick.color":        TEXT_SEC,
    "text.color":         TEXT_PRI,
    "grid.color":         GRID_COL,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.4,
    "legend.facecolor":   PANEL_BG,
    "legend.edgecolor":   BORDER,
    "legend.labelcolor":  TEXT_SEC,
    "font.size":          9,
})


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS – mirrors ga_trade_signal_feature_selection.py
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "delta_ratio","buy_sell_ratio","cvd_slope_5","cvd_slope_10",
    "cvd_zscore","notional_buy_ratio","notional_sell_ratio",
    "large_trade_imbalance","large_trade_ratio","trade_intensity",
    "hl_range","bar_body","upper_wick","lower_wick",
    "ema_cross_9_21","ema_cross_21_50",
    "rsi_14","rsi_7","stoch_k","stoch_d",
    "adx","adx_diff","macd","macd_signal","macd_diff",
    "atr_ratio","bb_width","bb_pct",
    "vol_zscore","vol_ratio","notional_zscore",
]
N_FEATURES = len(FEATURE_COLS)
YEAR = "2024"


# ──────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ──────────────────────────────────────────────────────────────────────────────

def prepare_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time").sort_index()
    df["returns"] = df["close"].pct_change()
    df["atr_raw"] = df["atr_ratio"] * df["close"]
    needed = FEATURE_COLS + ["open","high","low","close","returns","atr_raw"]
    df = df[[c for c in needed if c in df.columns]]

    df[FEATURE_COLS] = df[FEATURE_COLS].shift(1)
    df["atr_raw"]    = df["atr_raw"].shift(1)
    df = df.dropna()

    scaler = RobustScaler()
    scaler.fit(df[FEATURE_COLS])
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    df[FEATURE_COLS] = df[FEATURE_COLS].clip(-3, 3) / 3
    return df


# ──────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL RECONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def reconstruct_individual(gen_data: dict) -> dict:
    """Build a standard individual dict from a generation's best_* fields."""
    feat_names = gen_data["best_features"]
    feat_weights = gen_data["best_weights"]

    mask    = [0] * N_FEATURES
    weights = [0.0] * N_FEATURES
    for fname, w in zip(feat_names, feat_weights.values()):
        if fname in FEATURE_COLS:
            idx = FEATURE_COLS.index(fname)
            mask[idx]    = 1
            weights[idx] = float(w)

    return {
        "mask":    mask,
        "weights": weights,
        "buy_th":  float(gen_data["best_buy_th"]),
        "sell_th": float(gen_data["best_sell_th"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION  (verbatim from original)
# ──────────────────────────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, ind: dict) -> pd.Series:
    feat_matrix    = df[FEATURE_COLS].values
    mask           = np.array(ind["mask"], dtype=float)
    weights        = np.array(ind["weights"], dtype=float)
    active_weights = mask * weights

    if np.sum(mask) == 0:
        return pd.Series(0, index=df.index)

    norm   = np.sum(np.abs(active_weights)) + 1e-9
    scores = (feat_matrix @ active_weights) / norm
    scores = np.clip(scores, -1, 1)

    signals = np.where(
        scores > ind["buy_th"],  1,
        np.where(scores < ind["sell_th"], -1, 0)
    )
    return pd.Series(signals, index=df.index)


# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST  (verbatim from original)
# ──────────────────────────────────────────────────────────────────────────────

def backtest_tpsl(df: pd.DataFrame, signals: pd.Series, cfg: dict):
    fee           = cfg.get("fee",         0.0004)
    slippage      = cfg.get("slippage",    0.0002)
    tp_mult       = cfg.get("tp_atr_mult", 3.0)
    sl_mult       = cfg.get("sl_atr_mult", 1.0)
    max_bars_held = cfg.get("max_bars_held", None)
    min_atr_ratio = cfg.get("min_atr_ratio", None)

    closes    = df["close"].values
    highs     = df["high"].values
    lows      = df["low"].values
    atr_vals  = df["atr_raw"].values
    atr_ratio = df["atr_ratio"].values if "atr_ratio" in df.columns else None
    sigs      = signals.values
    n         = len(df)

    pnl_arr = np.zeros(n)
    pos_arr = np.zeros(n)

    position    = 0
    entry_price = 0.0
    tp_price    = 0.0
    sl_price    = 0.0
    entry_bar   = 0
    trade_log   = []

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
                pos_arr[i]  = position
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
                pos_arr[i]  = position

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
                pos_arr[i] = position

    strat_ret = pd.Series(pnl_arr, index=df.index)
    positions = pd.Series(pos_arr, index=df.index)
    return strat_ret, positions, trade_log


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(strat_ret: pd.Series, positions: pd.Series, bpy: int = 72576):
    sr  = strat_ret.fillna(0)
    eq  = (1 + sr).cumprod()
    tot = float(eq.iloc[-1] - 1)
    mu, sigma = sr.mean(), sr.std()
    sharpe = float((mu / sigma) * np.sqrt(bpy)) if sigma >= 1e-9 else -999.0
    dd = float((eq / eq.cummax() - 1).min())

    active = sr[sr != 0]
    wr = float((active > 0).mean()) if len(active) > 0 else 0.0

    closed_trades = [t for t in []]  # placeholder
    return dict(sharpe=sharpe, total_return=tot, max_dd=dd, win_rate=wr)


def compute_trade_stats(trade_log):
    closed = [t for t in trade_log if "exit_type" in t]
    if not closed:
        return dict(n=0, tp=0, sl=0, timeout=0, tp_pct=0, sl_pct=0, to_pct=0,
                    avg_hold=0, pf=0)
    n = len(closed)
    tp_list = [t for t in closed if t["exit_type"] == "TP"]
    sl_list = [t for t in closed if t["exit_type"] == "SL"]
    to_list = [t for t in closed if t["exit_type"] == "TIMEOUT"]
    pnls    = [t["pnl"] for t in closed]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p <= 0]
    pf = abs(sum(wins) / sum(losses)) if losses else float("inf")
    holds = [t["bars_held"] for t in closed]
    return dict(n=n,
                tp=len(tp_list), sl=len(sl_list), timeout=len(to_list),
                tp_pct=len(tp_list)/n, sl_pct=len(sl_list)/n,
                to_pct=len(to_list)/n,
                avg_hold=float(np.mean(holds)),
                pf=pf)


# ──────────────────────────────────────────────────────────────────────────────
# CANDLESTICK HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def build_candle_segments(df, xs):
    """Return (body_segs, wick_segs, colors) for candlestick drawing."""
    body_segs, wick_segs, colors = [], [], []
    for x, (_, row) in zip(xs, df.iterrows()):
        o, h, l, c = row.open, row.high, row.low, row.close
        col = BULL if c >= o else BEAR
        colors.append(col)
        body_segs.append([(x, min(o,c)), (x, max(o,c))])
        wick_segs.append([(x, l),        (x, h)])
    return body_segs, wick_segs, colors


# ──────────────────────────────────────────────────────────────────────────────
# ANIMATION BUILD
# ──────────────────────────────────────────────────────────────────────────────

def build_animation(df: pd.DataFrame,
                    history: list,
                    global_cfg: dict,
                    output_path: str,
                    fps: int = 8,
                    interval_ms: int = 125):

    N_GEN   = len(history)
    n_bars  = len(df)
    xs      = np.arange(n_bars)

    # Pre-compute results for every generation
    print(f"Pre-computing {N_GEN} generations …", flush=True)
    gen_results = []
    for gi, gd in enumerate(history):
        ind      = reconstruct_individual(gd)
        sigs     = generate_signals(df, ind)
        sr, pos, tlog = backtest_tpsl(df, sigs, global_cfg)
        eq       = (1 + sr.fillna(0)).cumprod()
        ts       = compute_trade_stats(tlog)
        metrics  = compute_metrics(sr, pos)
        closed   = [t for t in tlog if "exit_type" in t]
        gen_results.append(dict(
            ind=ind,
            signals=sigs,
            strat_ret=sr,
            equity=eq,
            trade_log=tlog,
            closed=closed,
            ts=ts,
            metrics=metrics,
            gd=gd,
        ))
        if (gi+1) % 10 == 0 or gi == N_GEN-1:
            print(f"  gen {gi+1}/{N_GEN}  best_fit={gd['best_fitness']:.4f}  "
                  f"trades={len(closed)}", flush=True)

    best_fitness_series = [r["gd"]["best_fitness"]    for r in gen_results]
    mean_fitness_series = [r["gd"]["mean_fitness"]    for r in gen_results]
    best_ever_series    = [r["gd"]["best_ever_score"] for r in gen_results]
    feat_freq_series    = [r["gd"]["avg_features"]    for r in gen_results]
    gen_nums            = [r["gd"]["generation"]       for r in gen_results]

    # Price limits
    price_lo = df["low"].min()  * 0.998
    price_hi = df["high"].max() * 1.002

    # ── Layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(19, 9), dpi=110, facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1],
                            left=0.04, right=0.98, top=0.88, bottom=0.06,
                            wspace=0.03)

    # Left – price line chart
    ax_candle = fig.add_subplot(gs[0, 0])
    # Right – stats panel (invisible axes, text only)
    ax_stats  = fig.add_subplot(gs[0, 1])
    ax_stats.set_facecolor(PANEL_BG)
    ax_stats.set_xticks([])
    ax_stats.set_yticks([])
    for spine in ax_stats.spines.values():
        spine.set_edgecolor(BORDER)

    ax_candle.set_facecolor(PANEL_BG)
    for spine in ax_candle.spines.values():
        spine.set_edgecolor(BORDER)

    # ─ Price line chart (static base) ───────────────────────────────────────
    ax_candle.set_xlim(-1, n_bars + 1)
    ax_candle.set_ylim(price_lo, price_hi)
    ax_candle.set_ylabel("Price (USDT)", color=TEXT_SEC, fontsize=15)
    ax_candle.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}"))
    ax_candle.set_xticks([])
    ax_candle.grid(True, axis="y", alpha=0.3)
    ax_candle.set_title("", color=TEXT_PRI, fontsize=16, pad=4)

    # Draw neutral blue line chart
    ax_candle.plot(xs, df["close"].values, color=NEUTRAL_BLUE, lw=1.5, alpha=0.9, zorder=2, label="Price")

    # ─ Dynamic signal / trade layers ────────────────────────────────────────
    scat_buy,  = ax_candle.plot([], [], "^", color=BULL,  ms=12, zorder=6,
                                 mec="#00ff4c", mew=0.5, label="BUY")
    scat_sell, = ax_candle.plot([], [], "v", color=BEAR,  ms=12, zorder=6,
                                 mec="#ff6b6b", mew=0.5, label="SELL")
    trade_lines_col = LineCollection([], colors=[], linewidths=1.2,
                                     linestyles="--", alpha=0.6, zorder=4)
    ax_candle.add_collection(trade_lines_col)

    tp_scat,  = ax_candle.plot([], [], "D", color=TP_COL,  ms=5, zorder=7,
                                label="TP hit", mec="white", mew=0.3)
    sl_scat,  = ax_candle.plot([], [], "X", color=SL_COL,  ms=5, zorder=7,
                                label="SL hit", mec="white", mew=0.3)
    to_scat,  = ax_candle.plot([], [], "s", color=TIMEOUT, ms=4, zorder=7,
                                label="Timeout", mec="white", mew=0.3)

    ax_candle.legend(loc="upper left", fontsize=10, ncol=6,
                     framealpha=0.6, handletextpad=0.3, columnspacing=0.8)

    # Generation label on candle chart
    gen_label = ax_candle.text(0.99, 0.97, "", transform=ax_candle.transAxes,
                               ha="right", va="top", fontsize=15,
                               color=GOLD, fontweight="bold",
                               bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor=PANEL_BG, edgecolor=BORDER,
                                          alpha=0.8))

    # ─ Stats text (right panel) ─────────────────────────────────────────────
    stats_text = ax_stats.text(
        0.08, 0.95, "", transform=ax_stats.transAxes,
        ha="left", va="top", fontsize=15, color=TEXT_PRI,
        fontfamily="monospace", fontweight="semibold", linespacing=1.6,
    )

    # ─ Super title ──────────────────────────────────────────────────────────
    title_txt = fig.text(
        0.50, 0.965,
        "Genetic Algorithm Trade Signal Evolution -- BTC/USDT 5-min",
        ha="center", va="top",
        fontsize=22, color=TEXT_PRI,
        fontweight="bold"
    )

    subtitle_txt = fig.text(
        0.50, 0.9250,
        f"TP={global_cfg.get('tp_atr_mult',3.0)}xATR  "
        f"SL={global_cfg.get('sl_atr_mult',1.0)}xATR  "
        f"RR=2:1  |  {n_bars:,} bars displayed",
        ha="center", va="top",
        fontsize=16, color=TEXT_SEC
    )

    # ──────────────────────────────────────────────────────────────────────────
    # UPDATE FUNCTION
    # ──────────────────────────────────────────────────────────────────────────

    def update(frame_idx):
        r   = gen_results[frame_idx]
        gd  = r["gd"]
        ind = r["ind"]
        sigs     = r["signals"]
        closed   = r["closed"]
        eq       = r["equity"]
        ts       = r["ts"]
        m        = r["metrics"]

        gi = frame_idx

        # ── Candle: signals ──────────────────────────────────────────────────
        buy_xs  = xs[sigs.values ==  1]
        sell_xs = xs[sigs.values == -1]
        buy_ys  = df["close"].values[buy_xs]  * 0.996
        sell_ys = df["close"].values[sell_xs] * 1.004
        scat_buy.set_data(buy_xs,  buy_ys)
        scat_sell.set_data(sell_xs, sell_ys)

        # ── Candle: trade outcome markers ───────────────────────────────────
        tp_xs, tp_ys = [], []
        sl_xs, sl_ys = [], []
        to_xs, to_ys = [], []
        trade_segs   = []
        trade_colors = []

        for t in closed:
            eb = t["entry_bar"]; xb = t["exit_bar"]
            ep = t["entry_price"]
            ex = t["exit_price"]
            et = t["exit_type"]
            col = TP_COL if et == "TP" else (SL_COL if et == "SL" else TIMEOUT)

            trade_segs.append([(eb, ep), (xb, ex)])
            trade_colors.append(col)

            if et == "TP":
                tp_xs.append(xb); tp_ys.append(t["tp_price"])
            elif et == "SL":
                sl_xs.append(xb); sl_ys.append(t["sl_price"])
            else:
                to_xs.append(xb); to_ys.append(ex)

        trade_lines_col.set_segments(trade_segs)
        trade_lines_col.set_colors(trade_colors)
        tp_scat.set_data(tp_xs,  tp_ys)
        sl_scat.set_data(sl_xs,  sl_ys)
        to_scat.set_data(to_xs,  to_ys)

        # ── Generation label ─────────────────────────────────────────────────
        gen_label.set_text(f"Gen {gd['generation']:>3d}/{N_GEN}")

        # ── Stats panel ──────────────────────────────────────────────────────
        ret_col  = BULL if m["total_return"] >= 0 else BEAR
        dd_col   = BEAR if m["max_dd"] < -0.10 else GOLD
        sh_col   = BULL if m["sharpe"] > 1.0 else (GOLD if m["sharpe"] > 0 else BEAR)
        wr_col   = BULL if m["win_rate"] > 0.5 else (GOLD if m["win_rate"] > 0.4 else BEAR)

        pf_str   = f"{ts['pf']:.2f}" if ts['pf'] < 100 else "inf"
        pf_col   = BULL if ts['pf'] > 1.5 else (GOLD if ts['pf'] > 1.0 else BEAR)

        lines = [
            f"--- PERFORMANCE ---",
            f"",
            f"Total Return  {m['total_return']:>+8.2%}",
            f"Sharpe Ratio  {m['sharpe']:>8.2f}",
            f"Max Drawdown  {m['max_dd']:>8.2%}",
            f"Win Rate      {m['win_rate']:>8.2%}",
            f"",
            f"--- TRADE STATS ---",
            f"",
            f"Closed Trades {ts['n']:>8d}",
            f"TP Hits       {ts['tp']:>5d}  ({ts['tp_pct']:>5.1%})",
            f"SL Hits       {ts['sl']:>5d}  ({ts['sl_pct']:>5.1%})",
            f"Timeouts      {ts['timeout']:>5d}  ({ts['to_pct']:>5.1%})",
            f"",
            f"Profit Factor {pf_str:>8s}",
            f"Avg Bars Held {ts['avg_hold']:>8.1f}",
            f"",
            f"--- GA FITNESS ---",
            f"",
            f"Best Fitness  {gd['best_fitness']:>8.4f}",
            f"Mean Fitness  {gd['mean_fitness']:>8.4f}",
            f"Features Used {int(sum(ind['mask'])):>5d}/{N_FEATURES}",
        ]
        stats_text.set_text("\n".join(lines))

        return (scat_buy, scat_sell, trade_lines_col, tp_scat, sl_scat,
                to_scat, gen_label, stats_text)

    # ──────────────────────────────────────────────────────────────────────────
    # EXPORT
    # ──────────────────────────────────────────────────────────────────────────
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
    # fill in defaults that may not be in the stored config
    global_cfg.setdefault("fee",          0.0004)
    global_cfg.setdefault("slippage",     0.0002)
    global_cfg.setdefault("tp_atr_mult",  3.0)
    global_cfg.setdefault("sl_atr_mult",  1.0)
    global_cfg.setdefault("max_bars_held", None)
    global_cfg.setdefault("min_atr_ratio", None)

    print(f"  Generations : {len(history)}")
    print(f"  Config      : {global_cfg}")

    # ── Load CSV ──────────────────────────────────────────────────────────
    print(f"\nLoading market data: {args.csv}")
    df = prepare_df(args.csv)
    print(f"  Bars loaded : {len(df):,}  "
          f"({df.index[0].date()} → {df.index[-1].date()})")

    # ── Animate ───────────────────────────────────────────────────────────
    build_animation(df, history, global_cfg,
                    output_path=args.out,
                    fps=args.fps)


if __name__ == "__main__":
    main()