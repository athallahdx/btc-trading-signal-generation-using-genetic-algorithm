import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# =============================================================================
# CONFIG
# =============================================================================
JSON_PATH = "ga_evolution_history_3.0_2024_temporal-feature-expansion.json"
OUTPUT_MP4 = "ga_evolution_history_3.0_2024_temporal-feature-expansion.mp4"
FPS = 8
DPI = 180

# =============================================================================
# LOAD JSON
# =============================================================================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

feature_cols = data["feature_cols"]
generations = data["generations"]

# =============================================================================
# PREPARE DATAFRAME
# =============================================================================
rows = []

for gen in generations:
    rows.append({
        "generation":     gen["generation"],
        "best_fitness":   gen["best_fitness"],
        "mean_fitness":   gen["mean_fitness"],
        "median_fitness": gen["median_fitness"],
        "worst_fitness":  gen["worst_fitness"],
        "std_fitness":    gen["std_fitness"],
        "avg_features":   gen["avg_features"],
        "mutation_rate":  gen["mutation_rate"],
        "avg_buy_th":     gen["avg_buy_th"],
        "avg_sell_th":    gen["avg_sell_th"],
        "best_n_features":gen["best_n_features"],
        "best_ever_score":gen["best_ever_score"],
        "feature_freq":   gen["feature_freq"],
        "best_features":  gen["best_features"],
    })

stats_df = pd.DataFrame(rows)

# =============================================================================
# TYPOGRAPHY & STYLE
# =============================================================================
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     13,
    "axes.titlepad":      10,
    "axes.labelsize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "figure.titlesize":   20,
    "axes.facecolor":     "#111111",
    "figure.facecolor":   "#0A0A0A",
    "savefig.facecolor":  "#0A0A0A",
    "text.color":         "white",
    "axes.labelcolor":    "white",
    "axes.edgecolor":     "#333333",
    "xtick.color":        "#BBBBBB",
    "ytick.color":        "#BBBBBB",
    "grid.color":         "#333333",
    "legend.fontsize":    8,
    "legend.frameon":     False,
})

# =============================================================================
# FIGURE LAYOUT
# =============================================================================
fig = plt.figure(figsize=(20, 11))

# Top: title area.  Bottom: info panel (~18% of figure height).
# Middle: plots occupy the space in between.
INFO_PANEL_H = 0.20   # fraction of figure height reserved for info text
TITLE_H      = 0.04   # fraction reserved above plots for suptitle
TOP_MARGIN   = 1.0 - TITLE_H        # plots stop here (top)
BOTTOM_MARGIN = INFO_PANEL_H        # plots start here (bottom)

fig.subplots_adjust(
    left=0.06,
    right=0.97,
    top=TOP_MARGIN,
    bottom=BOTTOM_MARGIN,
    hspace=0.42,   # vertical space between row 0 and row 1
    wspace=0.38,   # horizontal space between columns
)

# Grid: 2 rows × 3 cols
#   Row 0 col 0-1 : Fitness (wide)
#   Row 0 col 2   : Feature Heatmap (tall, spans both rows)
#   Row 1 col 0   : Feature Complexity
#   Row 1 col 1   : Threshold Evolution
gs = GridSpec(
    2, 3,
    figure=fig,
    width_ratios=[1.0, 1.0, 1.6],   # heatmap gets more width
    height_ratios=[1.1, 1.0],        # fitness row slightly taller
)

ax_fitness    = fig.add_subplot(gs[0, :2])   # top-left wide panel
ax_heatmap    = fig.add_subplot(gs[:, 2])    # right column, both rows
ax_features   = fig.add_subplot(gs[1, 0])   # bottom-left
ax_thresholds = fig.add_subplot(gs[1, 1])   # bottom-centre

# =============================================================================
# PERSISTENT INFO PANEL (below plots)
# =============================================================================
info_box = fig.text(
    0.02,                # x  (left-aligned with a small margin)
    0.01,                # y  (sits inside the reserved bottom strip)
    "",
    fontsize=15,
    family="DejaVu Sans Mono",
    verticalalignment="bottom",
    bbox=dict(
        facecolor="#111111",
        edgecolor="#444444",
        boxstyle="round,pad=0.6",
        alpha=0.9,
    ),
)

# =============================================================================
# ANIMATION UPDATE
# =============================================================================
def update(frame_idx):
    ax_fitness.clear()
    ax_features.clear()
    ax_thresholds.clear()
    ax_heatmap.clear()

    current = stats_df.iloc[: frame_idx + 1]
    gens    = current["generation"]

    # -------------------------------------------------------------------------
    # FITNESS EVOLUTION
    # -------------------------------------------------------------------------
    ax_fitness.plot(gens, current["best_fitness"],
                    linewidth=2.5, label="Best Fitness")
    ax_fitness.plot(gens, current["mean_fitness"],
                    linewidth=1.8, alpha=0.85, label="Mean Fitness")
    ax_fitness.plot(gens, current["median_fitness"],
                    linewidth=1.8, linestyle="--", alpha=0.9, label="Median Fitness")
    ax_fitness.fill_between(
        gens,
        current["mean_fitness"] - current["std_fitness"],
        current["mean_fitness"] + current["std_fitness"],
        alpha=0.18, label="±1 Std",
    )
    ax_fitness.scatter(gens.iloc[-1], current["best_fitness"].iloc[-1],
                       s=100, zorder=5)

    ax_fitness.set_title("Fitness Evolution", fontweight="bold")
    ax_fitness.set_xlabel("Generation", fontsize=15)
    ax_fitness.tick_params(axis='x', labelsize=12)
    ax_fitness.set_ylabel("Fitness", fontsize=15)
    ax_fitness.tick_params(axis='y', labelsize=12)
    ax_fitness.grid(alpha=0.25)
    ax_fitness.legend(loc="lower right")

    # -------------------------------------------------------------------------
    # FEATURE COMPLEXITY
    # -------------------------------------------------------------------------
    ax_features.plot(gens, current["avg_features"],
                     linewidth=2.2, label="Avg Features")
    ax_features.plot(gens, current["best_n_features"],
                     linewidth=1.8, linestyle="--", alpha=0.9, label="Best Individual")

    ax_features.set_title("Feature Complexity", fontweight="bold")
    ax_features.set_xlabel("Generation", fontsize=15)
    ax_features.tick_params(axis='x', labelsize=12)
    ax_features.set_ylabel("# Features", fontsize=15)
    ax_features.tick_params(axis='y', labelsize=12)
    ax_features.grid(alpha=0.25)
    ax_features.legend(loc="upper right")

    # -------------------------------------------------------------------------
    # THRESHOLD EVOLUTION
    # -------------------------------------------------------------------------
    ax_thresholds.plot(gens, current["avg_buy_th"],
                       linewidth=2.2, label="Buy Threshold")
    ax_thresholds.plot(gens, current["avg_sell_th"],
                       linewidth=2.2, label="Sell Threshold")
    ax_thresholds.axhline(0, linestyle="--", linewidth=0.9, color="#888888")

    ax_thresholds.set_title("Threshold Evolution", fontweight="bold")
    ax_thresholds.set_xlabel("Generation", fontsize=15)
    ax_thresholds.tick_params(axis='x', labelsize=12)
    ax_thresholds.set_ylabel("Threshold", fontsize=15)
    ax_thresholds.tick_params(axis='y', labelsize=12)
    ax_thresholds.grid(alpha=0.25)
    ax_thresholds.legend(loc="upper right")

    # -------------------------------------------------------------------------
    # FEATURE IMPORTANCE HEATMAP  (top 10 features, spans both rows)
    # -------------------------------------------------------------------------
    all_feature_freq    = np.array(current["feature_freq"].tolist())
    overall_feature_freq = np.mean(all_feature_freq, axis=0)

    top10_idx      = np.argsort(overall_feature_freq)[-10:][::-1]
    top10_features = [feature_cols[i] for i in top10_idx]
    heatmap_data   = all_feature_freq[:, top10_idx].T   # (10 features × N gens)

    ax_heatmap.imshow(
        heatmap_data,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    ax_heatmap.set_title("Top 10 Selected Features", fontweight="bold")
    ax_heatmap.set_xlabel("Generation", fontsize=15)
    ax_heatmap.set_ylabel("Feature", fontsize=15)

    ax_heatmap.set_yticks(np.arange(len(top10_features)))
    ax_heatmap.set_yticklabels(top10_features, fontsize=12)

    n_gen_ticks = min(8, len(gens))
    gen_ticks   = np.linspace(0, len(gens) - 1, n_gen_ticks, dtype=int)
    ax_heatmap.set_xticks(gen_ticks)
    ax_heatmap.set_xticklabels(gens.iloc[gen_ticks].values, fontsize=12, rotation=30, ha="right")

    # -------------------------------------------------------------------------
    # INFO PANEL
    # -------------------------------------------------------------------------
    latest    = current.iloc[-1]
    best_feat = ", ".join(latest["best_features"])

    info_text = (
        f"Generation : {int(latest['generation']):>4d}   |   "
        f"Best Fitness : {latest['best_fitness']:.4f}   |   "
        f"Mean Fitness : {latest['mean_fitness']:.4f}   |   "
        f"Mutation Rate : {latest['mutation_rate']:.4f}   |   "
        f"Avg Features : {latest['avg_features']:.2f}\n"
        f"Best Features : {best_feat}"
    )

    spacer = " " * 100
    info_box.set_text(info_text + f"\n{spacer}")

    return []

# =============================================================================
# CREATE & SAVE ANIMATION
# =============================================================================
anim = FuncAnimation(
    fig,
    update,
    frames=len(stats_df),
    interval=160,
    blit=False,
    repeat=False,
)

print("[INFO] Rendering animation …")
anim.save(OUTPUT_MP4, writer="ffmpeg", fps=FPS, dpi=DPI)
print(f"[DONE] Saved → {OUTPUT_MP4}")

plt.close(fig)