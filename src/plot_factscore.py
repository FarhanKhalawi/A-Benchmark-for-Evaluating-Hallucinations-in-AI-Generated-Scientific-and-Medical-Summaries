"""
plot_factscore.py — Beautiful FactScore Visualization
======================================================
Reads results_with_factscore.csv and generates a publication-quality
hallucination analysis figure.

Usage:
    python plot_factscore.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from models_config import OUTPUT_BASE_DIR, ACTIVE_MODEL

# ── Make ACTIVE_MODEL always a list ──────────────────────────
if isinstance(ACTIVE_MODEL, str):
    ACTIVE_MODEL = [ACTIVE_MODEL]

# ── Color palette (one per model) ────────────────────────────
PALETTE = [
    "#00C9FF",   # cyan-blue
    "#FF6B6B",   # coral
    "#A8FF78",   # lime green
    "#FFD93D",   # gold
    "#C77DFF",   # violet
]

# ── Load data ─────────────────────────────────────────────────
model_data = {}
for model_name in ACTIVE_MODEL:
    out_dir = OUTPUT_BASE_DIR.format(model=model_name)
    csv_path = f"{out_dir}/results_with_factscore.csv"
    try:
        df = pd.read_csv(csv_path)
        model_data[model_name] = df
        print(f"  Loaded {len(df)} rows from {csv_path}")
    except FileNotFoundError:
        print(f"  WARNING: {csv_path} not found — skipping {model_name}")

if not model_data:
    raise RuntimeError("No data loaded. Run factscore.py first.")

# ── Figure layout ─────────────────────────────────────────────
#   Row 0 : scatter  (hallucination score per sample)
#   Row 1 : bar      (mean supported / contradicted / not-verifiable)
n_models = len(model_data)
fig = plt.figure(figsize=(14, 13), facecolor="#0D1117")
fig.patch.set_facecolor("#0D1117")

gs = gridspec.GridSpec(
    3, 1,
    figure=fig,
    hspace=0.52,
    top=0.91, bottom=0.07,
    left=0.08, right=0.97,
)

GRID_KW  = dict(color="#30363D", linewidth=0.6, linestyle="--", alpha=0.8)
TITLE_KW = dict(color="#E6EDF3", fontsize=11, fontweight="bold", pad=10,
                fontfamily="monospace")
LABEL_KW = dict(color="#8B949E", fontsize=9, fontfamily="monospace")
TICK_KW  = dict(colors="#8B949E", labelsize=8)

# ─────────────────────────────────────────────────────────────
# Panel 0 — Scatter: hallucination score per sample
# ─────────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.set_facecolor("#161B22")

for spine in ax0.spines.values():
    spine.set_edgecolor("#30363D")

for idx, (model_name, df) in enumerate(model_data.items()):
    scores = df["factscore"].fillna(np.nan).values
    # hallucination = 1 - factscore
    halluc = 1 - scores
    x      = np.arange(1, len(halluc) + 1)
    color  = PALETTE[idx % len(PALETTE)]

    # jitter x slightly when multiple models
    jitter = (idx - (n_models - 1) / 2) * 0.25
    ax0.scatter(
        x + jitter, halluc,
        color=color, s=38, alpha=0.85, linewidths=0,
        zorder=3, label=model_name,
    )
    # rolling mean line
    window = max(1, len(halluc) // 8)
    roll   = pd.Series(halluc).rolling(window, center=True, min_periods=1).mean()
    ax0.plot(x + jitter, roll, color=color, linewidth=1.6, alpha=0.55, zorder=2)

# threshold lines
for thresh, label in [(0.2, "low"), (0.4, "medium"), (0.6, "high")]:
    ax0.axhline(thresh, color="#FF6B6B" if thresh == 0.6 else "#FFD93D" if thresh == 0.4 else "#8B949E",
                linewidth=0.8, linestyle=":", alpha=0.5, zorder=1)
    ax0.text(len(scores) + 0.5, thresh, label,
             color="#8B949E", fontsize=7, va="center", fontfamily="monospace")

ax0.set_ylim(-0.05, 1.05)
ax0.set_xlim(0, len(scores) + 2)
ax0.set_xlabel("Sample index", **LABEL_KW)
ax0.set_title("Hallucination score per sample  (1 − FactScore)", **TITLE_KW)
ax0.set_ylabel("Hallucination score", **LABEL_KW)
ax0.tick_params(axis="both", **TICK_KW)
ax0.yaxis.set_minor_locator(MultipleLocator(0.1))
ax0.grid(axis="y", **GRID_KW)
ax0.legend(
    frameon=True, facecolor="#161B22", edgecolor="#30363D",
    labelcolor="#E6EDF3", fontsize=8, loc="upper right",
    prop={"family": "monospace", "size": 8},
)

# ─────────────────────────────────────────────────────────────
# Panel 1 — Stacked bar: mean supported / not-verifiable / contradicted
# ─────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.set_facecolor("#161B22")
for spine in ax1.spines.values():
    spine.set_edgecolor("#30363D")

model_names  = list(model_data.keys())
x_pos        = np.arange(len(model_names))
bar_width    = 0.55

mean_supp  = []
mean_nv    = []
mean_contra= []
mean_total = []

for model_name, df in model_data.items():
    t = df["total_facts"].replace(0, np.nan)
    mean_supp.append(  (df["supported"]      / t).mean() * 100)
    mean_nv.append(    (df["not_verifiable"] / t).mean() * 100)
    mean_contra.append((df["contradicted"]   / t).mean() * 100)
    mean_total.append( df["total_facts"].mean())

mean_supp   = np.array(mean_supp)
mean_nv     = np.array(mean_nv)
mean_contra = np.array(mean_contra)

b1 = ax1.bar(x_pos, mean_supp,   bar_width, label="Supported",        color="#00C9FF", alpha=0.88, zorder=3)
b2 = ax1.bar(x_pos, mean_nv,     bar_width, bottom=mean_supp,         label="Not verifiable", color="#FFD93D", alpha=0.88, zorder=3)
b3 = ax1.bar(x_pos, mean_contra, bar_width, bottom=mean_supp+mean_nv, label="Contradicted",   color="#FF6B6B", alpha=0.88, zorder=3)

# value labels on bars
for i, (s, nv, c) in enumerate(zip(mean_supp, mean_nv, mean_contra)):
    ax1.text(i, s / 2,          f"{s:.0f}%",  ha="center", va="center", color="#0D1117", fontsize=8, fontweight="bold", fontfamily="monospace")
    ax1.text(i, s + nv / 2,     f"{nv:.0f}%", ha="center", va="center", color="#0D1117", fontsize=8, fontweight="bold", fontfamily="monospace")
    if c > 3:
        ax1.text(i, s + nv + c / 2, f"{c:.0f}%", ha="center", va="center", color="#0D1117", fontsize=8, fontweight="bold", fontfamily="monospace")
    # avg total facts annotation
    ax1.text(i, 103, f"avg {mean_total[i]:.1f} facts", ha="center", va="bottom",
             color="#8B949E", fontsize=7, fontfamily="monospace")

ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_names, color="#E6EDF3", fontsize=9, fontfamily="monospace")
ax1.set_ylim(0, 115)
ax1.set_ylabel("% of atomic facts", **LABEL_KW)
ax1.set_title("Fact verification breakdown  (mean across samples)", **TITLE_KW)
ax1.tick_params(axis="both", **TICK_KW)
ax1.grid(axis="y", **GRID_KW)
ax1.legend(
    frameon=True, facecolor="#161B22", edgecolor="#30363D",
    labelcolor="#E6EDF3", fontsize=8, loc="upper right",
    prop={"family": "monospace", "size": 8},
)


# ─────────────────────────────────────────────────────────────
# Super-title
# ─────────────────────────────────────────────────────────────
models_str = " · ".join(ACTIVE_MODEL)
fig.suptitle(
    f"FactScore Hallucination Analysis\n{models_str}",
    color="#E6EDF3", fontsize=13, fontweight="bold",
    fontfamily="monospace", y=0.975,
)

# ─────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────
out_path = "factscore_analysis.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#0D1117")
print(f"\nSaved → {out_path}")
plt.show()