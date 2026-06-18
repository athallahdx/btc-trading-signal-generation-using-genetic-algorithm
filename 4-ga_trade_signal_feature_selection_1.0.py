import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import RobustScaler
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "year": 2024,
    "input":          "BTCUSDT_2024_6m_features_5min.csv",
    "train_end":      "2024-04-15",
    "val_end":        "2024-05-15",
    # GA
    "pop_size":       100,
    "generations":    200,
    "crossover_rate": 0.8,
    "mutation_rate":  0.05,
    "tournament_k":   3,
    "elite_n":        2,
    # fitness
    "min_trades":     30,
    "max_sharpe":     5.0,
    "bars_per_year":  252 * 288,      # 5-min bars, crypto 24/7
    # TP/SL — in atr unit
    "tp_atr_mult":    2.0,
    "sl_atr_mult":    1.0,
    # cost
    "fee":            0.0004,
    "slippage":       0.0002,
    # ── FIX 2: max holding period (bars) ─────────────────────────────────────
    # 100 bars × 5 min = ~8 hours; prevents trades getting stuck for days
    "max_bars_held":  100,
    # ── FIX 2b: minimum ATR ratio to enter (volatility filter) ───────────────
    # Blocks entries when market is compressed / ranging
    # Set to None to disable; or e.g. 0.003 (0.3% of price)
    "min_atr_ratio":  0.003,
    # ── FIX 1: Walk-forward windows ──────────────────────────────────────────
    # Each tuple: (train_start, train_end, val_start, val_end)
    # Using 3 rolling windows across the available data
    "wf_windows": [
        ("2024-01-01", "2024-02-29", "2024-03-01", "2024-03-31"),
        ("2024-01-01", "2024-03-31", "2024-04-01", "2024-04-30"),
        ("2024-01-01", "2024-04-30", "2024-05-01", "2024-05-31"),
    ],
}

FEATURE_COLS = [
    # order flow
    "delta_ratio", "buy_sell_ratio", "cvd_slope_5", "cvd_slope_10",
    "cvd_zscore", "notional_buy_ratio", "notional_sell_ratio",
    "large_trade_imbalance", "large_trade_ratio", "trade_intensity",
    # price action
    "hl_range", "bar_body", "upper_wick", "lower_wick",
    # moving averages
    "ema_cross_9_21", "ema_cross_21_50",
    # momentum
    "rsi_14", "rsi_7", "stoch_k", "stoch_d",
    # trend
    "adx", "adx_diff", "macd", "macd_signal", "macd_diff",
    # volatility
    "atr_ratio", "bb_width", "bb_pct",
    # volume
    "vol_zscore", "vol_ratio", "notional_zscore",
]
N_FEATURES = len(FEATURE_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(cfg):
    df = pd.read_csv(
        cfg["input"],
        parse_dates=["time"],
        index_col="time"
    ).sort_index()

    df["returns"]  = df["close"].pct_change()
    df["atr_raw"]  = df["atr_ratio"] * df["close"]

    needed = FEATURE_COLS + ["open", "high", "low", "close", "returns", "atr_raw"]
    df = df[needed]

    df[FEATURE_COLS] = df[FEATURE_COLS].shift(1)
    df["atr_raw"]    = df["atr_raw"].shift(1)
    df = df.dropna()

    train = df[df.index < cfg["train_end"]].copy()
    val   = df[(df.index >= cfg["train_end"]) & (df.index < cfg["val_end"])].copy()
    test  = df[df.index >= cfg["val_end"]].copy()

    scaler = RobustScaler()
    scaler.fit(train[FEATURE_COLS])

    for split in [train, val, test]:
        split[FEATURE_COLS] = scaler.transform(split[FEATURE_COLS])
        split[FEATURE_COLS] = split[FEATURE_COLS].clip(-3, 3) / 3

    print(f"Features      : {N_FEATURES}")
    print(f"Train         : {len(train):>6,} bars  ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"Validation    : {len(val):>6,} bars  ({val.index[0].date()} → {val.index[-1].date()})")
    print(f"Test          : {len(test):>6,} bars  ({test.index[0].date()} → {test.index[-1].date()})")

    return train, val, test, scaler


def prepare_window(df_full, train_start, train_end, val_start, val_end):
    """
    Prepare a single walk-forward window: fit scaler on that window's train slice,
    transform both train and val slices independently.
    """
    train_w = df_full[(df_full.index >= train_start) & (df_full.index < train_end)].copy()
    val_w   = df_full[(df_full.index >= val_start)   & (df_full.index < val_end)].copy()

    scaler_w = RobustScaler()
    scaler_w.fit(train_w[FEATURE_COLS])

    train_w[FEATURE_COLS] = scaler_w.transform(train_w[FEATURE_COLS])
    train_w[FEATURE_COLS] = train_w[FEATURE_COLS].clip(-3, 3) / 3
    val_w[FEATURE_COLS]   = scaler_w.transform(val_w[FEATURE_COLS])
    val_w[FEATURE_COLS]   = val_w[FEATURE_COLS].clip(-3, 3) / 3

    return train_w, val_w


# ─────────────────────────────────────────────────────────────────────────────
# 2. INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────

def random_individual():
    n_active   = np.random.randint(3, max(4, N_FEATURES // 2))
    mask       = [0] * N_FEATURES
    active_idx = np.random.choice(N_FEATURES, size=n_active, replace=False)
    for i in active_idx:
        mask[i] = 1

    weights = []
    for i in range(N_FEATURES):
        weights.append(np.random.uniform(-1.0, 1.0) if mask[i] == 1 else 0.0)

    # ── FIX 3: Symmetric thresholds ─────────────────────────────────────────
    # Both buy and sell are equidistant from 0, preventing long/short bias.
    # threshold sampled from [0.1, 0.7]; buy = +th, sell = -th
    half_th  = np.random.uniform(0.1, 0.7)
    buy_th   = float(half_th)
    sell_th  = float(-half_th)

    return {
        "mask":    mask,
        "weights": weights,
        "buy_th":  buy_th,
        "sell_th": sell_th,
    }


def clone(ind):
    return {
        "mask":    ind["mask"].copy(),
        "weights": ind["weights"].copy(),
        "buy_th":  float(ind["buy_th"]),
        "sell_th": float(ind["sell_th"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(df, ind):
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


# ─────────────────────────────────────────────────────────────────────────────
# 4. BACKTEST WITH FIXED TP/SL ATR + MAX HOLD + VOLATILITY FILTER
# ─────────────────────────────────────────────────────────────────────────────

def backtest_tpsl(df, signals, cfg):
    fee           = cfg["fee"]
    slippage      = cfg["slippage"]
    tp_mult       = cfg["tp_atr_mult"]
    sl_mult       = cfg["sl_atr_mult"]
    max_bars_held = cfg.get("max_bars_held", None)      # FIX 2a
    min_atr_ratio = cfg.get("min_atr_ratio", None)      # FIX 2b

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

    trade_log = []

    for i in range(1, n):
        # ── FLAT ─────────────────────────────────────────────────────────────
        if position == 0:
            sig = sigs[i]
            if sig != 0:
                atr = atr_vals[i]
                if atr <= 0 or np.isnan(atr):
                    continue

                # ── FIX 2b: volatility filter — skip if market too quiet ────
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

        # ── IN POSITION: check TP / SL / max hold ────────────────────────────
        else:
            high = highs[i]
            low  = lows[i]

            hit_tp = (position ==  1 and high >= tp_price) or \
                     (position == -1 and low  <= tp_price)
            hit_sl = (position ==  1 and low  <= sl_price) or \
                     (position == -1 and high >= sl_price)

            # ── FIX 2a: max holding period timeout ───────────────────────────
            bars_in_trade = i - entry_bar
            timed_out     = (max_bars_held is not None) and (bars_in_trade >= max_bars_held)

            if hit_tp or hit_sl or timed_out:
                if hit_sl:
                    exit_price = sl_price
                    exit_type  = "SL"
                elif hit_tp:
                    exit_price = tp_price
                    exit_type  = "TP"
                else:
                    # timeout exit at current close (with slippage)
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
                pnl_arr[i] = position * (closes[i] - closes[i - 1]) / closes[i - 1]
                pos_arr[i] = position

    strat_ret = pd.Series(pnl_arr, index=df.index)
    positions = pd.Series(pos_arr, index=df.index)
    return strat_ret, positions, trade_log


# ─────────────────────────────────────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(strat_ret, positions, bars_per_year):
    strat_ret = strat_ret.fillna(0)
    equity    = (1 + strat_ret).cumprod()

    total_return = equity.iloc[-1] - 1

    mean = strat_ret.mean()
    std  = strat_ret.std()
    sharpe = (mean / std) * np.sqrt(bars_per_year) if std >= 1e-9 else -999.0

    rolling_max = equity.cummax()
    drawdown    = equity / rolling_max - 1
    max_dd      = drawdown.min()

    trade_changes = positions.diff().fillna(0)
    entries       = (trade_changes != 0).sum()
    total_trades  = int(entries / 2)

    active_returns = strat_ret[strat_ret != 0]
    win_rate       = float((active_returns > 0).mean()) if len(active_returns) > 0 else 0.0

    exposure = float((positions != 0).mean())

    return {
        "sharpe":       float(sharpe),
        "total_return": float(total_return),
        "max_dd":       float(max_dd),
        "win_rate":     float(win_rate),
        "trades":       int(total_trades),
        "exposure":     float(exposure),
    }


def compute_trade_stats(trade_log):
    closed = [t for t in trade_log if "exit_type" in t]
    if not closed:
        return {}

    pnls      = [t["pnl"] for t in closed]
    wins      = [p for p in pnls if p > 0]
    losses    = [p for p in pnls if p <= 0]
    tp_hits   = [t for t in closed if t["exit_type"] == "TP"]
    sl_hits   = [t for t in closed if t["exit_type"] == "SL"]
    to_hits   = [t for t in closed if t["exit_type"] == "TIMEOUT"]
    bars_held = [t["bars_held"] for t in closed]

    avg_win  = float(np.mean(wins))   if wins   else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    return {
        "closed_trades":  len(closed),
        "tp_hits":        len(tp_hits),
        "sl_hits":        len(sl_hits),
        "timeout_hits":   len(to_hits),
        "tp_rate":        len(tp_hits) / len(closed),
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "profit_factor":  abs(sum(wins) / sum(losses)) if losses else float("inf"),
        "avg_bars_held":  float(np.mean(bars_held)),
        "max_bars_held":  int(np.max(bars_held)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. FITNESS  — Walk-forward + Consistency Penalty
# ─────────────────────────────────────────────────────────────────────────────

def _single_window_sharpe(ind, val_df, cfg):
    """Compute Sharpe on one val slice. Returns -999 if not enough trades."""
    n_active = sum(ind["mask"])
    if n_active < 3:
        return -999.0

    signals              = generate_signals(val_df, ind)
    strat_ret, positions, _ = backtest_tpsl(val_df, signals, cfg)
    metrics              = compute_metrics(strat_ret, positions, cfg["bars_per_year"])

    if metrics["trades"] < max(5, cfg["min_trades"] // len(cfg["wf_windows"])):
        return -999.0

    sharpe = np.nan_to_num(metrics["sharpe"], nan=-999, posinf=cfg["max_sharpe"])
    return float(np.clip(sharpe, -5, cfg["max_sharpe"]))


def fitness(ind, wf_val_slices, cfg):
    """
    FIX 1 + FIX 4: Walk-forward fitness with consistency penalty.

    wf_val_slices : list of pre-sliced & scaled val DataFrames,
                    one per walk-forward window.

    Score = mean(window_sharpes) × trade_factor
            + return_bonus
            - dd_penalty
            - complexity_penalty
            - exposure_penalty
            - consistency_penalty   ← NEW: penalises variance across windows
    """
    n_active = sum(ind["mask"])
    if n_active < 3:
        return -999.0

    # ── Evaluate across all WF windows ───────────────────────────────────────
    window_sharpes = []
    all_metrics    = []

    for val_df in wf_val_slices:
        signals                 = generate_signals(val_df, ind)
        strat_ret, positions, _ = backtest_tpsl(val_df, signals, cfg)
        m                       = compute_metrics(strat_ret, positions, cfg["bars_per_year"])
        sh = np.nan_to_num(m["sharpe"], nan=-999, posinf=cfg["max_sharpe"])
        sh = float(np.clip(sh, -5, cfg["max_sharpe"]))
        window_sharpes.append(sh)
        all_metrics.append(m)

    # If any window has -999 it means the individual is degenerate
    valid = [s for s in window_sharpes if s > -999]
    if len(valid) < len(wf_val_slices) // 2 + 1:
        return -999.0

    # Aggregate metrics
    mean_sharpe  = float(np.mean(valid))
    total_trades = sum(m["trades"] for m in all_metrics)
    mean_dd      = float(np.mean([abs(m["max_dd"]) for m in all_metrics]))
    mean_return  = float(np.mean([m["total_return"] for m in all_metrics]))
    mean_exposure= float(np.mean([m["exposure"]     for m in all_metrics]))

    if total_trades < cfg["min_trades"]:
        return -999.0

    # ── FIX 4: Consistency penalty ───────────────────────────────────────────
    # Penalise high variance across windows: a strategy that scores 3,3,3
    # is far better than one that scores 9,0,0 with the same mean.
    sharpe_std         = float(np.std(valid))
    consistency_penalty = sharpe_std * 0.5   # half a Sharpe point per unit of std

    # ── Standard fitness terms ────────────────────────────────────────────────
    trade_factor        = min(np.sqrt(total_trades / 50), 1.0)
    dd_penalty          = (mean_dd ** 0.5) * 2.0
    complexity_penalty  = (n_active / N_FEATURES) * 0.15
    exposure_penalty    = 0.5 if mean_exposure < 0.05 else 0.0
    return_bonus        = np.tanh(mean_return * 5)

    score = (
        (mean_sharpe * trade_factor)
        + return_bonus
        - dd_penalty
        - complexity_penalty
        - exposure_penalty
        - consistency_penalty          # FIX 4
    )

    return float(score) if np.isfinite(score) else -999.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. GA OPERATORS
# ─────────────────────────────────────────────────────────────────────────────

def tournament_select(population, scores, k):
    idx  = np.random.choice(len(population), k, replace=False)
    best = idx[np.argmax([scores[i] for i in idx])]
    return clone(population[best])


def crossover(p1, p2):
    point = np.random.randint(1, N_FEATURES)

    def make_child(a, b):
        mask    = a["mask"][:point]    + b["mask"][point:]
        weights = a["weights"][:point] + b["weights"][point:]
        for i in range(N_FEATURES):
            if mask[i] == 0:
                weights[i] = 0.0

        # ── FIX 3: inherit threshold symmetrically ───────────────────────────
        # Pick one parent's half_th; child threshold is symmetric ±half_th
        parent = a if np.random.rand() > 0.5 else b
        half_th = parent["buy_th"]                     # buy_th == |sell_th|

        return {
            "mask":    mask,
            "weights": weights,
            "buy_th":  float(np.clip(half_th,  0.05, 0.95)),
            "sell_th": float(np.clip(-half_th, -0.95, -0.05)),
        }

    return make_child(p1, p2), make_child(p2, p1)


def mutate(ind, rate):
    ind = clone(ind)

    for i in range(N_FEATURES):
        if np.random.rand() < rate:
            ind["mask"][i] ^= 1
        if ind["mask"][i] == 1:
            if np.random.rand() < rate:
                ind["weights"][i] += np.random.uniform(-0.2, 0.2)
                ind["weights"][i]  = float(np.clip(ind["weights"][i], -1, 1))
        else:
            ind["weights"][i] = 0.0

    # ── FIX 3: mutate threshold symmetrically ────────────────────────────────
    if np.random.rand() < rate:
        delta          = np.random.uniform(-0.05, 0.05)
        new_half_th    = float(np.clip(ind["buy_th"] + delta, 0.05, 0.95))
        ind["buy_th"]  =  new_half_th
        ind["sell_th"] = -new_half_th

    active = sum(ind["mask"])
    if active < 3:
        inactive_idx = [i for i in range(N_FEATURES) if ind["mask"][i] == 0]
        chosen = np.random.choice(inactive_idx, size=(3 - active), replace=False)
        for i in chosen:
            ind["mask"][i]    = 1
            ind["weights"][i] = np.random.uniform(-1, 1)

    return ind


# ─────────────────────────────────────────────────────────────────────────────
# 8. GA MAIN LOOP  (FIX 1: receives wf_val_slices instead of single val_df)
# ─────────────────────────────────────────────────────────────────────────────

def run_ga(wf_val_slices, cfg):
    """
    wf_val_slices: list of val DataFrames, one per walk-forward window.
    The fitness function evaluates every individual on ALL windows.
    """
    pop = [random_individual() for _ in range(cfg["pop_size"])]

    history         = []
    best_ever       = None
    best_ever_score = -np.inf
    stagnation      = 0
    total_gens_run  = 0

    print(f"\n{'─'*72}")
    print(f" GA | pop={cfg['pop_size']} gen={cfg['generations']} features={N_FEATURES}")
    print(f" TP mult={cfg['tp_atr_mult']}×ATR  SL mult={cfg['sl_atr_mult']}×ATR  "
          f"RR={cfg['tp_atr_mult']/cfg['sl_atr_mult']:.1f}:1")
    print(f" Walk-forward windows : {len(wf_val_slices)}")
    print(f" Max bars held        : {cfg.get('max_bars_held','disabled')}")
    print(f" Min ATR ratio filter : {cfg.get('min_atr_ratio','disabled')}")
    print(f"{'─'*72}")

    for gen in range(cfg["generations"]):
        mutation_rate = cfg["mutation_rate"] * (0.995 ** gen)

        scores     = [fitness(ind, wf_val_slices, cfg) for ind in pop]
        best_idx   = int(np.argmax(scores))
        best_score = scores[best_idx]
        best_ind   = pop[best_idx]

        improved = best_score > best_ever_score
        if improved:
            best_ever_score = best_score
            best_ever       = clone(best_ind)
            stagnation      = 0
        else:
            stagnation += 1

        total_gens_run = gen + 1

        if stagnation >= 30:
            print("\nEarly stopping: no improvement.\n")
            break

        # ── Collect population-level statistics ───────────────────────────────
        valid_scores  = [s for s in scores if s > -999]
        n_valid       = len(valid_scores)
        mean_score    = float(np.mean(valid_scores))    if valid_scores else -999.0
        median_score  = float(np.median(valid_scores))  if valid_scores else -999.0
        worst_score   = float(np.min(valid_scores))     if valid_scores else -999.0
        std_score     = float(np.std(valid_scores))     if valid_scores else 0.0

        active_counts = [sum(ind["mask"]) for ind in pop]
        avg_features  = float(np.mean(active_counts))
        min_features  = int(np.min(active_counts))
        max_features  = int(np.max(active_counts))
        std_features  = float(np.std(active_counts))

        # Feature frequency across population
        feat_freq = []
        for fi in range(N_FEATURES):
            freq = float(np.mean([ind["mask"][fi] for ind in pop]))
            feat_freq.append(round(freq, 2))

        # Threshold statistics
        buy_ths  = [ind["buy_th"]  for ind in pop]
        sell_ths = [ind["sell_th"] for ind in pop]

        # Best individual of this generation
        best_active_idx = [i for i in range(N_FEATURES) if best_ind["mask"][i] == 1]
        best_features   = [FEATURE_COLS[i] for i in best_active_idx]
        best_weights    = {FEATURE_COLS[i]: round(float(best_ind["weights"][i]), 6)
                           for i in best_active_idx}

        gen_record = {
            "generation":       gen + 1,
            "improved":         improved,
            "stagnation":       stagnation,
            "mutation_rate":    round(mutation_rate, 6),
            # fitness stats
            "best_fitness":     round(float(best_score), 6),
            "mean_fitness":     round(mean_score, 6),
            "median_fitness":   round(median_score, 6),
            "worst_fitness":    round(worst_score, 6),
            "std_fitness":      round(std_score, 6),
            "n_valid":          n_valid,
            # feature stats
            "avg_features":     round(avg_features, 1),
            "min_features":     min_features,
            "max_features":     max_features,
            "std_features":     round(std_features, 4),
            "feature_freq":     feat_freq,
            # threshold stats
            "avg_buy_th":       round(float(np.mean(buy_ths)),  4),
            "avg_sell_th":      round(float(np.mean(sell_ths)), 4),
            "std_buy_th":       round(float(np.std(buy_ths)),   4),
            "std_sell_th":      round(float(np.std(sell_ths)),  4),
            # best individual of this generation
            "best_n_features":  len(best_features),
            "best_features":    best_features,
            "best_weights":     best_weights,
            "best_buy_th":      round(float(best_ind["buy_th"]),  4),
            "best_sell_th":     round(float(best_ind["sell_th"]), 4),
            # running best-ever
            "best_ever_score":  round(float(best_ever_score), 6),
        }
        history.append(gen_record)

        print(
            f" Gen {gen+1:>3d}/{cfg['generations']} "
            f"best={best_score:>7.4f} "
            f"mean={mean_score:>7.4f} "
            f"avg_feat={avg_features:>5.2f} "
            f"mut={mutation_rate:.4f}"
        )

        elite_idx   = np.argsort(scores)[::-1][:cfg["elite_n"]]
        elites      = [clone(pop[i]) for i in elite_idx]

        offspring   = []
        target_size = cfg["pop_size"] - cfg["elite_n"]

        while len(offspring) < target_size:
            p1 = tournament_select(pop, scores, cfg["tournament_k"])
            p2 = tournament_select(pop, scores, cfg["tournament_k"])

            if np.random.rand() < cfg["crossover_rate"]:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = clone(p1), clone(p2)

            offspring.append(mutate(c1, mutation_rate))
            if len(offspring) < target_size:
                offspring.append(mutate(c2, mutation_rate))

        n_immigrants = max(1, cfg["pop_size"] // 20)
        for _ in range(n_immigrants):
            replace_idx            = np.random.randint(len(offspring))
            offspring[replace_idx] = random_individual()

        pop = elites + offspring

    print(f"{'─'*72}")
    print(f" DONE | best fitness = {best_ever_score:.4f}")
    print(f"{'─'*72}\n")

    # ── Build and save ga_evolution_history.json ──────────────────────────────
    evolution_json = {
        "config": {
            "pop_size":       cfg["pop_size"],
            "generations":    cfg["generations"],
            "crossover_rate": cfg["crossover_rate"],
            "mutation_rate":  cfg["mutation_rate"],
            "tournament_k":   cfg["tournament_k"],
            "elite_n":        cfg["elite_n"],
            "tp_atr_mult":    cfg["tp_atr_mult"],
            "sl_atr_mult":    cfg["sl_atr_mult"],
            "min_trades":     cfg["min_trades"],
        },
        "feature_cols":      FEATURE_COLS,
        "n_features":        N_FEATURES,
        "total_generations": total_gens_run,
        "generations":       history,
    }
    with open(f"ga_evolution_history_1.0_{CONFIG['year']}.json", "w") as f:
        json.dump(evolution_json, f, indent=2)
    print(f"Saved → ga_evolution_history_1.0_{CONFIG['year']}.json")

    return best_ever, pd.DataFrame({
        "generation": [h["generation"]   for h in history],
        "best":       [h["best_fitness"]  for h in history],
        "mean":       [h["mean_fitness"]  for h in history],
        "avg_feat":   [h["avg_features"]  for h in history],
    })


# ─────────────────────────────────────────────────────────────────────────────
# 9. REPORT
# ─────────────────────────────────────────────────────────────────────────────

def interpret_sharpe(sharpe):
    if sharpe < 0:     return "❌ Losing"
    elif sharpe < 0.5: return "⚠️  Poor"
    elif sharpe < 1.0: return "🟡 Acceptable"
    elif sharpe < 2.0: return "✅ Good"
    elif sharpe < 3.0: return "✅ Very good"
    else:              return "🔍 Excellent — check overfit"


def report(label, df, ind, cfg):
    signals                      = generate_signals(df, ind)
    strat_ret, positions, t_log  = backtest_tpsl(df, signals, cfg)
    m                            = compute_metrics(strat_ret, positions, cfg["bars_per_year"])
    ts                           = compute_trade_stats(t_log)

    bh_pos = pd.Series(1.0, index=df.index)
    bh     = compute_metrics(df["returns"], bh_pos, cfg["bars_per_year"])

    rr = cfg["tp_atr_mult"] / cfg["sl_atr_mult"]

    print(f"\n{'═'*62}")
    print(f"  {label}")
    print(f"  TP={cfg['tp_atr_mult']}×ATR  SL={cfg['sl_atr_mult']}×ATR  RR={rr:.1f}:1")
    print(f"{'═'*62}")
    print(f"  {'Metric':<28} {'GA Strategy':>12}  {'Buy & Hold':>10}")
    print(f"  {'─'*54}")
    print(f"  {'Sharpe ratio':<28} {m['sharpe']:>12.4f}  {bh['sharpe']:>10.4f}")
    print(f"  {'Total return':<28} {m['total_return']:>11.2%}  {bh['total_return']:>9.2%}")
    print(f"  {'Max drawdown':<28} {m['max_dd']:>11.2%}  {bh['max_dd']:>9.2%}")
    print(f"  {'Win rate':<28} {m['win_rate']:>11.2%}  {'—':>10}")
    print(f"  {'Total trades':<28} {m['trades']:>12,}  {'—':>10}")
    print(f"  {'Sharpe verdict':<28} {interpret_sharpe(m['sharpe'])}")

    if ts:
        print(f"\n  ── Trade Detail (TP/SL) ─────────────────────────────")
        print(f"  {'Closed trades':<28} {ts['closed_trades']:>12,}")
        print(f"  {'TP hits':<28} {ts['tp_hits']:>12,}  ({ts['tp_rate']:>6.1%})")
        print(f"  {'SL hits':<28} {ts['sl_hits']:>12,}  ({1-ts['tp_rate']-ts['timeout_hits']/max(ts['closed_trades'],1):>6.1%})")
        print(f"  {'Timeout exits':<28} {ts['timeout_hits']:>12,}  ({ts['timeout_hits']/max(ts['closed_trades'],1):>6.1%})")
        print(f"  {'Avg win (per trade)':<28} {ts['avg_win']:>11.4%}")
        print(f"  {'Avg loss (per trade)':<28} {ts['avg_loss']:>11.4%}")
        print(f"  {'Profit factor':<28} {ts['profit_factor']:>12.3f}")
        print(f"  {'Avg bars held':<28} {ts['avg_bars_held']:>12.1f}")
        print(f"  {'Max bars held':<28} {ts['max_bars_held']:>12,}")

    print(f"{'═'*62}")
    return m, strat_ret, positions, signals, t_log


def print_best_individual(ind, cfg):
    selected = [
        (FEATURE_COLS[i], ind["weights"][i])
        for i in range(N_FEATURES) if ind["mask"][i] == 1
    ]
    selected.sort(key=lambda x: abs(x[1]), reverse=True)

    rr = cfg["tp_atr_mult"] / cfg["sl_atr_mult"]

    print(f"\n── Best Individual ──────────────────────────────────────")
    print(f"  Features : {len(selected)} / {N_FEATURES} selected")
    print(f"  Buy  th  : +{ind['buy_th']:.4f}  (symmetric)")
    print(f"  Sell th  :  {ind['sell_th']:.4f}  (symmetric)")
    print(f"  TP mult  : {cfg['tp_atr_mult']}×ATR")
    print(f"  SL mult  : {cfg['sl_atr_mult']}×ATR")
    print(f"  RR ratio : {rr:.1f}:1")
    print(f"  Max hold : {cfg.get('max_bars_held','disabled')} bars")
    print(f"\n  {'Feature':<30} {'Weight':>8}  {'Bar'}")
    print(f"  {'─'*55}")
    for feat, w in selected:
        bar = ("+" if w >= 0 else "-") * int(abs(w) * 12)
        print(f"  {feat:<30} {w:>+8.4f}  {bar}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 10. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_ga_history(history):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#0f0f0f")
        ax.spines[:].set_color("#333")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")

    gens = history["generation"]
    axes[0].plot(gens, history["best"], color="lime",    linewidth=1.5, label="Best")
    axes[0].plot(gens, history["mean"], color="#4da6ff", linewidth=1.0,
                 label="Mean", linestyle="--")
    axes[0].axhline(0, color="#555", linewidth=0.5)
    axes[0].set_ylabel("Fitness")
    axes[0].set_title("GA Evolution — Walk-forward + Consistency Fitness", color="white")
    axes[0].legend(facecolor="#1a1a1a", labelcolor="white")

    axes[1].plot(gens, history["avg_feat"], color="orange", linewidth=1.2)
    axes[1].set_ylabel("Active features")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylim(0, N_FEATURES + 1)
    axes[1].axhline(N_FEATURES // 2, color="#555", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(f"ga_history_1.0_{CONFIG['year']}.png", dpi=150, facecolor="#0f0f0f")
    plt.show()
    print(f"Saved → ga_history_1.0_{CONFIG['year']}.png")


def plot_equity_curve(df, strat_ret, positions, signals, title, cfg):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#0f0f0f")
        ax.spines[:].set_color("#333")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")

    ga_eq = (1 + strat_ret).cumprod()
    bh_eq = (1 + df["returns"]).cumprod()

    axes[0].plot(df.index, ga_eq, color="lime",    linewidth=1.2, label="GA strategy")
    axes[0].plot(df.index, bh_eq, color="#4da6ff", linewidth=1.0,
                 label="Buy & hold", linestyle="--")
    axes[0].set_ylabel("Equity")
    rr = cfg["tp_atr_mult"] / cfg["sl_atr_mult"]
    axes[0].set_title(
        f"{title}  |  TP={cfg['tp_atr_mult']}×ATR  SL={cfg['sl_atr_mult']}×ATR  "
        f"RR={rr:.1f}:1  MaxHold={cfg.get('max_bars_held','∞')}bars",
        color="white"
    )
    axes[0].legend(facecolor="#1a1a1a", labelcolor="white")

    axes[1].plot(df.index, df["close"], color="white", linewidth=0.6)
    buy_idx  = df.index[signals ==  1]
    sell_idx = df.index[signals == -1]
    axes[1].scatter(buy_idx,  df.loc[buy_idx,  "close"],
                    marker="^", color="lime", s=12, zorder=5, label="Buy signal")
    axes[1].scatter(sell_idx, df.loc[sell_idx, "close"],
                    marker="v", color="red",  s=12, zorder=5, label="Sell signal")
    axes[1].set_ylabel("Price")
    axes[1].legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)

    drawdown = ga_eq / ga_eq.cummax() - 1
    axes[2].fill_between(df.index, drawdown, 0, color="red", alpha=0.4)
    axes[2].set_ylabel("Drawdown")

    plt.tight_layout()
    fname = "equity_" + title.lower().replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150, facecolor="#0f0f0f")
    plt.show()
    print(f"Saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def random_baseline(wf_val_slices, cfg, n_trials=30):
    best_score, best_ind = -np.inf, None
    for _ in range(n_trials):
        ind = random_individual()
        s   = fitness(ind, wf_val_slices, cfg)
        if s > best_score:
            best_score, best_ind = s, ind
    return best_ind


def equal_weight_baseline():
    w = 1 / N_FEATURES
    return {
        "mask":    [1] * N_FEATURES,
        "weights": [w] * N_FEATURES,
        "buy_th":  0.5,
        "sell_th": -0.5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    print("\nLoading data...")

    # ── Load raw (unscaled) full dataset ─────────────────────────────────────
    df_raw = pd.read_csv(
        CONFIG["input"],
        parse_dates=["time"],
        index_col="time"
    ).sort_index()

    df_raw["returns"] = df_raw["close"].pct_change()
    df_raw["atr_raw"] = df_raw["atr_ratio"] * df_raw["close"]

    needed = FEATURE_COLS + ["open", "high", "low", "close", "returns", "atr_raw"]
    df_raw = df_raw[needed]
    df_raw[FEATURE_COLS] = df_raw[FEATURE_COLS].shift(1)
    df_raw["atr_raw"]    = df_raw["atr_raw"].shift(1)
    df_raw = df_raw.dropna()

    # ── FIX 1: Build walk-forward val slices (each scaled to its own window) ─
    print(f"\nBuilding {len(CONFIG['wf_windows'])} walk-forward windows...")
    wf_val_slices = []
    for (tr_s, tr_e, va_s, va_e) in CONFIG["wf_windows"]:
        _, val_w = prepare_window(df_raw, tr_s, tr_e, va_s, va_e)
        wf_val_slices.append(val_w)
        print(f"  Train {tr_s}→{tr_e}  |  Val {va_s}→{va_e}  ({len(val_w):,} bars)")

    # ── Standard single-split for final evaluation ───────────────────────────
    train_df, val_df, test_df, scaler = load_and_split(CONFIG)

    # ── GA ───────────────────────────────────────────────────────────────────
    best_ind, history = run_ga(wf_val_slices, CONFIG)
    print_best_individual(best_ind, CONFIG)
    plot_ga_history(history)

    # ── Evaluasi Validation ──────────────────────────────────────────────────
    val_m, val_strat, val_pos, val_sig, val_log = report(
        "VALIDATION SET", val_df, best_ind, CONFIG
    )
    plot_equity_curve(val_df, val_strat, val_pos, val_sig, "Validation Set", CONFIG)

    # ── Evaluasi Test ────────────────────────────────────────────────────────
    test_m, test_strat, test_pos, test_sig, test_log = report(
        "TEST SET", test_df, best_ind, CONFIG
    )
    plot_equity_curve(test_df, test_strat, test_pos, test_sig, "Test Set", CONFIG)

    # ── Overfit Check ────────────────────────────────────────────────────────
    ratio = val_m["sharpe"] / (abs(test_m["sharpe"]) + 1e-9)
    print(f"\n── Overfit Check ────────────────────────────────────────")
    print(f"  Val  Sharpe : {val_m['sharpe']:.4f}  {interpret_sharpe(val_m['sharpe'])}")
    print(f"  Test Sharpe : {test_m['sharpe']:.4f}  {interpret_sharpe(test_m['sharpe'])}")
    print(f"  Val/Test    : {ratio:.2f}x  "
          f"{'⚠️  possible overfit' if ratio > 2.0 else '✅ generalizes ok'}")

    # ── Baselines ────────────────────────────────────────────────────────────
    print("\nRunning baselines...")
    rand_ind  = random_baseline(wf_val_slices, CONFIG)
    equal_ind = equal_weight_baseline()

    rand_m,  _, _, _, _ = report("BASELINE: Random",        test_df, rand_ind,  CONFIG)
    equal_m, _, _, _, _ = report("BASELINE: Equal Weights", test_df, equal_ind, CONFIG)

    bh_pos = pd.Series(1.0, index=test_df.index)
    bh_m   = compute_metrics(test_df["returns"], bh_pos, CONFIG["bars_per_year"])

    # ── Final Comparison ─────────────────────────────────────────────────────
    print(f"\n{'═'*66}")
    print(f"  FINAL COMPARISON — Test Set")
    print(f"  (TP={CONFIG['tp_atr_mult']}×ATR  SL={CONFIG['sl_atr_mult']}×ATR  "
          f"RR={CONFIG['tp_atr_mult']/CONFIG['sl_atr_mult']:.1f}:1  "
          f"MaxHold={CONFIG.get('max_bars_held','∞')}bars)")
    print(f"{'═'*66}")
    print(f"  {'Method':<32} {'Sharpe':>7}  {'Return':>8}  {'MaxDD':>8}")
    print(f"  {'─'*60}")
    rows = [
        ("GA — WF + consistency fitness", test_m),
        ("Baseline: random selection",    rand_m),
        ("Baseline: equal weights",       equal_m),
        ("Buy & hold",                    bh_m),
    ]
    for name, m in rows:
        print(f"  {name:<32} {m['sharpe']:>7.4f}  "
              f"{m['total_return']:>7.2%}  {m['max_dd']:>7.2%}")
    print(f"{'═'*66}\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    test_ts = compute_trade_stats(test_log)
    result  = {
        "selected_features":   [FEATURE_COLS[i] for i in range(N_FEATURES)
                                 if best_ind["mask"][i] == 1],
        "n_features_selected": int(sum(best_ind["mask"])),
        "feature_mask":        best_ind["mask"],
        "weights":             best_ind["weights"],
        "buy_threshold":       best_ind["buy_th"],
        "sell_threshold":      best_ind["sell_th"],
        "symmetric_threshold": True,
        "tp_atr_mult":         CONFIG["tp_atr_mult"],
        "sl_atr_mult":         CONFIG["sl_atr_mult"],
        "rr_ratio":            CONFIG["tp_atr_mult"] / CONFIG["sl_atr_mult"],
        "max_bars_held":       CONFIG.get("max_bars_held"),
        "min_atr_ratio":       CONFIG.get("min_atr_ratio"),
        "wf_windows":          CONFIG["wf_windows"],
        # validation
        "val_sharpe":          val_m["sharpe"],
        "val_return":          val_m["total_return"],
        "val_max_dd":          val_m["max_dd"],
        # test
        "test_sharpe":         test_m["sharpe"],
        "test_return":         test_m["total_return"],
        "test_max_dd":         test_m["max_dd"],
        "test_win_rate":       test_m["win_rate"],
        "test_trades":         test_m["trades"],
        "test_tp_rate":        test_ts.get("tp_rate"),
        "test_timeout_rate":   test_ts.get("timeout_hits", 0) / max(test_ts.get("closed_trades", 1), 1),
        "test_profit_factor":  test_ts.get("profit_factor"),
        "test_avg_bars_held":  test_ts.get("avg_bars_held"),
        "test_max_bars_held":  test_ts.get("max_bars_held"),
        "overfit_ratio":       float(ratio),
    }
    with open(f"best_individual_1.0_{CONFIG['year']}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → best_individual_1.0_{CONFIG['year']}.json")