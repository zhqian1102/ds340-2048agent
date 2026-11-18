# AI use statement:
# This plotting script was written by Claude (Anthropic, via Claude Code) to produce the
# figures for the Nov 17 TA check-in.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dqn import DQNAgent
from evaluate_baselines import run_game

os.makedirs("figures", exist_ok=True)

# Fixed colors per agent, used in every figure
COLORS = {"Random": "#2a78d6", "DQN": "#eb6834", "Rule-based": "#1baf7a"}
INK = "#0b0b0b"
MUTED = "#6b6a63"
GRID = "#e4e3dc"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
})

# ---------- evaluate DQN on the same seeds as the baselines (0-199, epsilon = 0) ----------
agent = DQNAgent("dqn.pt")
dqn_rows = [{"agent": "DQNAgent", "seed": s, **run_game(agent, s)} for s in range(200)]
dqn = pd.DataFrame(dqn_rows)
dqn.to_csv("dqn_eval_results.csv", index=False)

base = pd.read_csv("baseline_results.csv")
results = pd.concat([base, dqn], ignore_index=True)
results["agent"] = results["agent"].map({"RandomAgent": "Random", "DQNAgent": "DQN", "RuleBasedAgent": "Rule-based"})
order = ["Random", "DQN", "Rule-based"]

# ---------- Figure 1: learning curve ----------
log = pd.read_csv("training_log.csv")
log["ma"] = log["score"].rolling(50).mean()
means = results.groupby("agent")["score"].mean()

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(log["episode"], log["score"], color=COLORS["DQN"], alpha=0.18, linewidth=1)
ax.plot(log["episode"], log["ma"], color=COLORS["DQN"], linewidth=2, label="DQN training score (50-episode moving average)")
for name, offset, va in [("Random", -80, "top"), ("Rule-based", 60, "bottom")]:
    ax.axhline(means[name], color=COLORS[name], linewidth=2, linestyle="--")
    ax.text(log["episode"].max(), means[name] + offset, f"{name} (eval mean {means[name]:.0f})",
            color=INK, ha="right", va=va)
ax.axhline(means["DQN"], color=COLORS["DQN"], linewidth=2, linestyle=":")
ax.text(log["episode"].max(), means["DQN"] + 60, f"DQN, final model, ε=0 (eval mean {means['DQN']:.0f})",
        color=INK, ha="right", va="bottom")
ax.axvline(1000, color=MUTED, linewidth=1)
ax.text(1015, 4700, "ε reaches 0.05 →", color=MUTED, va="center")
ax.set_xlabel("Training episode")
ax.set_ylabel("Score")
ax.set_title("DQN learning curve (light line = individual episodes)", loc="left", color=INK)
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig("figures/1_learning_curve.png", dpi=200)

# ---------- Figure 2: score distribution per agent ----------
fig, ax = plt.subplots(figsize=(8, 5))
data = [results.loc[results["agent"] == a, "score"] for a in order]
bp = ax.boxplot(data, tick_labels=order, widths=0.5, patch_artist=True, showfliers=True,
                medianprops={"color": INK, "linewidth": 2},
                flierprops={"marker": "o", "markersize": 4, "markerfacecolor": MUTED, "markeredgecolor": "none"})
for patch, a in zip(bp["boxes"], order):
    patch.set_facecolor(COLORS[a])
    patch.set_alpha(0.85)
    patch.set_edgecolor(COLORS[a])
for i, a in enumerate(order, start=1):
    s = results.loc[results["agent"] == a, "score"]
    ax.text(i, s.max() + 250, f"mean {s.mean():.0f}\nmedian {s.median():.0f}", ha="center", va="bottom", color=INK)
ax.set_ylim(0, results["score"].max() * 1.2)
ax.set_ylabel("Final score")
ax.set_title("Final score over 200 games (same seeds 0–199)", loc="left", color=INK)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig("figures/2_score_distribution.png", dpi=200)

# ---------- Figure 3: % of games reaching each tile ----------
tiles = [128, 256, 512, 1024]
fig, ax = plt.subplots(figsize=(9, 5))
width = 0.26
x = np.arange(len(tiles))
for i, a in enumerate(order):
    mt = results.loc[results["agent"] == a, "max_tile"]
    pct = [(mt >= t).mean() * 100 for t in tiles]
    bars = ax.bar(x + (i - 1) * width, pct, width - 0.02, color=COLORS[a], label=a)
    for b, p in zip(bars, pct):
        ax.text(b.get_x() + b.get_width() / 2, p + 1, f"{p:.1f}%" if 0 < p < 1 else f"{p:.0f}%", ha="center", va="bottom", color=INK, fontsize=9)
ax.set_xticks(x, [f"≥ {t}" for t in tiles])
ax.set_ylim(0, 110)
ax.set_ylabel("% of games")
ax.set_xlabel("Highest tile reached")
ax.set_title("How often each agent reaches a given tile (200 games)", loc="left", color=INK)
ax.legend(frameon=False)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig("figures/3_max_tile_rates.png", dpi=200)

# ---------- summary table ----------
summary = results.groupby("agent").agg(
    mean_score=("score", "mean"), median_score=("score", "median"),
    mean_moves=("n_moves", "mean"),
).loc[order].round(1)
for t in [256, 512, 1024, 2048]:
    summary[f"reach_{t}_%"] = results.groupby("agent")["max_tile"].apply(lambda m: (m >= t).mean() * 100).loc[order].round(1)
summary.to_csv("figures/summary_table.csv")
print(summary.to_string())
