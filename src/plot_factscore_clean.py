"""
plot_factscore.py — FactScore / Hallucination Boxplot Visualization
===================================================================
Thesis-friendly version. Produces two separate figures on a white
background, using the same cyan / yellow / red / purple / orange /
teal / blue palette as the earlier dark version. Figure and axes
titles are removed — the LaTeX \\caption{} block supplies the title
in the thesis.

Figure 1 — Hallucination score distribution per model
             hallucination = 1 - factscore

Figure 2 — Fact verification breakdown per model
             supported / not verifiable  (in %)

Both figures keep:
- mean annotation above each box
- filled diamond marker for the mean
- model order taken from ACTIVE_MODEL in models_config.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from models_config import OUTPUT_BASE_DIR, ACTIVE_MODEL


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
if isinstance(ACTIVE_MODEL, str):
    ACTIVE_MODEL = [ACTIVE_MODEL]

# Same palette you liked, kept as-is
PALETTE = [
    "#00C9FF",   # cyan
    "#FF6B6B",   # coral
    "#A8FF78",   # light green
    "#FFD93D",   # yellow
    "#C77DFF",   # purple
    "#FF9F43",   # orange
    "#1DD1A1",   # teal
    "#54A0FF",   # blue
    "#FF6EB4",   # pink
    "#01CBC6",   # cyan-2
]

# Light-theme styling for thesis use
TEXT_COLOR  = "#1F2937"   # near-black
MUTED_TEXT  = "#6B7280"   # medium gray
SPINE_COLOR = "#9CA3AF"   # light gray spines
GRID_COLOR  = "#E5E7EB"   # very light grid

# Standard sans-serif font (matplotlib default) — no monospace
plt.rcParams["font.family"]       = "DejaVu Sans"
plt.rcParams["axes.titleweight"]  = "normal"

GRID_KW  = dict(color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=1.0)
LABEL_KW = dict(color=TEXT_COLOR, fontsize=11)
TICK_KW  = dict(colors=TEXT_COLOR, labelsize=10)


# ─────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────
def ensure_required_columns(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    df = df.copy()

    required_defaults = {
        "factscore":       np.nan,
        "supported":       0,
        "not_verifiable":  0,
        "total_facts":     np.nan,
    }

    for col, default in required_defaults.items():
        if col not in df.columns:
            print(f"  WARNING: column '{col}' missing for {model_name} — filling with {default}")
            df[col] = default

    numeric_cols = ["factscore", "supported", "not_verifiable", "total_facts"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # reconstruct total_facts from the two categories we actually plot
    missing_total_mask = df["total_facts"].isna() | (df["total_facts"] <= 0)
    reconstructed_total = (
        df["supported"].fillna(0)
        + df["not_verifiable"].fillna(0)
    )
    df.loc[missing_total_mask, "total_facts"] = reconstructed_total.loc[missing_total_mask]

    return df


def safe_pct(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    denominator = denominator.replace(0, np.nan).astype(float)
    pct = (numerator / denominator) * 100.0
    pct = pct.replace([np.inf, -np.inf], np.nan).dropna()
    return pct.values


def _style_bp(bp, color: str, n: int) -> None:
    """Style a boxplot dict for the light theme."""
    for i in range(n):
        bp["boxes"][i].set(
            facecolor=color,
            alpha=0.45,
            linewidth=1.4,
            edgecolor=color,
        )
        for j in range(2):
            bp["whiskers"][i * 2 + j].set(
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.9,
            )
            bp["caps"][i * 2 + j].set(
                color=color,
                linewidth=1.4,
                alpha=0.9,
            )
        # dark median line — stands out against the light box fill on white
        bp["medians"][i].set(color=TEXT_COLOR, linewidth=2.0)

        if len(bp["fliers"]) > i:
            bp["fliers"][i].set(
                marker="o",
                markerfacecolor=color,
                markeredgecolor=color,
                alpha=0.55,
                markersize=3.5,
                markeredgewidth=0,
            )


# ─────────────────────────────────────────────────────────────
# Load data in the ACTIVE_MODEL order
# ─────────────────────────────────────────────────────────────
model_data = {}
for model_name in ACTIVE_MODEL:
    out_dir  = OUTPUT_BASE_DIR.format(model=model_name)
    csv_path = os.path.join(out_dir, "results_with_factscore1000s.csv")

    try:
        df = pd.read_csv(csv_path)
        df = ensure_required_columns(df, model_name)
        model_data[model_name] = df
        print(f"  Loaded {len(df):>5} rows from {csv_path}")
    except FileNotFoundError:
        print(f"  WARNING: {csv_path} not found — skipping {model_name}")

if not model_data:
    raise RuntimeError("No data loaded. Run factscore.py first.")

model_names = list(model_data.keys())     # preserves ACTIVE_MODEL order
n_models    = len(model_names)
colors      = [PALETTE[i % len(PALETTE)] for i in range(n_models)]
x_pos       = np.arange(1, n_models + 1)


# =============================================================
# FIGURE 1 — Hallucination score distribution
# =============================================================
fig1, ax0 = plt.subplots(figsize=(12, 5), facecolor="white",
                         layout="constrained")
ax0.set_facecolor("white")
for spine in ax0.spines.values():
    spine.set_edgecolor(SPINE_COLOR)
    spine.set_linewidth(0.8)
# cleaner look for thesis — hide top/right spines
ax0.spines["top"].set_visible(False)
ax0.spines["right"].set_visible(False)

halluc_data = []
for df in model_data.values():
    factscore = pd.to_numeric(df["factscore"], errors="coerce").dropna()
    halluc    = 1.0 - factscore
    halluc    = halluc.replace([np.inf, -np.inf], np.nan).dropna().values
    halluc_data.append(halluc)

for i, (halluc, color) in enumerate(zip(halluc_data, colors)):
    if len(halluc) == 0:
        ax0.text(
            x_pos[i], 0.02, "No data",
            ha="center", va="bottom",
            color=MUTED_TEXT, fontsize=9,
        )
        continue

    bp = ax0.boxplot(
        halluc,
        patch_artist=True,
        notch=False,
        widths=0.55,
        positions=[x_pos[i]],
        manage_ticks=False,
    )
    _style_bp(bp, color, 1)

    mean_val = float(np.mean(halluc))

    # mean annotation
    ax0.text(
        x_pos[i], mean_val + 0.025, f"{mean_val:.4f}",
        ha="center", va="bottom",
        color=TEXT_COLOR, fontsize=9, fontweight="bold",
    )
    # mean diamond: dark fill + white edge so it pops on any box color
    ax0.scatter(
        x_pos[i], mean_val,
        marker="D", color=TEXT_COLOR,
        edgecolor="white", linewidth=1.0,
        s=38, zorder=5,
    )

ax0.set_xlim(0.4, n_models + 0.6)
ax0.set_ylim(-0.03, 1.0)
ax0.set_xticks(x_pos)
ax0.set_xticklabels(model_names, color=TEXT_COLOR, fontsize=10,
                    rotation=15, ha="right")
ax0.set_ylabel("Hallucination score  (1 − FactScore)", **LABEL_KW)
ax0.tick_params(axis="both", **TICK_KW)
ax0.yaxis.set_minor_locator(MultipleLocator(0.05))
ax0.grid(axis="y", **GRID_KW)
ax0.set_axisbelow(True)

legend_handles = [
    Line2D([0], [0], color=TEXT_COLOR, linewidth=2.0, label="Median"),
    Line2D([0], [0], marker="D", color=TEXT_COLOR,
           markeredgecolor="white", markeredgewidth=1.0,
           linewidth=0, markersize=7, label="Mean"),
]
ax0.legend(
    handles=legend_handles,
    frameon=True, facecolor="white", edgecolor=SPINE_COLOR,
    labelcolor=TEXT_COLOR, fontsize=9, loc="upper right",
)

out_path1 = "Figure1_hallucination1000s.png"
fig1.savefig(out_path1, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {out_path1}")


# =============================================================
# FIGURE 2 — Fact verification: supported / not verifiable
# =============================================================
fig2, ax1 = plt.subplots(figsize=(12, 5), facecolor="white",
                         layout="constrained")
ax1.set_facecolor("white")
for spine in ax1.spines.values():
    spine.set_edgecolor(SPINE_COLOR)
    spine.set_linewidth(0.8)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Two categories, same cyan / yellow feel as before
CAT_COLORS = {
    "Supported":      "#00C9FF",   # cyan
    "Not verifiable": "#FFD93D",   # yellow
}

group_width = 0.60
n_cats      = len(CAT_COLORS)
box_width   = group_width / n_cats
offsets     = np.linspace(
    -(group_width - box_width) / 2,
    (group_width - box_width) / 2,
    n_cats,
)

for cat_idx, (cat_label, cat_color) in enumerate(CAT_COLORS.items()):
    pct_data = []

    for df in model_data.values():
        t = df["total_facts"].replace(0, np.nan).astype(float)

        if cat_label == "Supported":
            pct = safe_pct(df["supported"], t)
        else:   # Not verifiable
            pct = safe_pct(df["not_verifiable"], t)

        pct_data.append(pct)

    positions = x_pos + offsets[cat_idx]

    for i, (pct, pos) in enumerate(zip(pct_data, positions)):
        if len(pct) == 0:
            ax1.text(
                pos, 1.0, "N/A",
                ha="center", va="bottom",
                color=MUTED_TEXT, fontsize=8,
            )
            continue

        bp = ax1.boxplot(
            pct,
            patch_artist=True,
            notch=False,
            widths=box_width * 0.85,
            positions=[pos],
            manage_ticks=False,
        )
        _style_bp(bp, cat_color, 1)

        mean_pct = float(np.mean(pct))
        ax1.text(
            pos, mean_pct + 1.8, f"{mean_pct:.1f}%",
            ha="center", va="bottom",
            color=TEXT_COLOR, fontsize=8, fontweight="bold",
        )
        ax1.scatter(
            pos, mean_pct,
            marker="D", color=TEXT_COLOR,
            edgecolor="white", linewidth=1.0,
            s=26, zorder=5,
        )

ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_names, color=TEXT_COLOR, fontsize=10,
                    rotation=15, ha="right")
ax1.set_xlim(0.4, n_models + 0.6)
ax1.set_ylim(-5, 110)
ax1.set_ylabel("% of atomic facts", **LABEL_KW)
ax1.tick_params(axis="both", **TICK_KW)
ax1.grid(axis="y", **GRID_KW)
ax1.set_axisbelow(True)

legend_patches = [
    mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.55, label=lbl)
    for lbl, c in CAT_COLORS.items()
]
legend_extra = Line2D(
    [0], [0], marker="D", color=TEXT_COLOR,
    markeredgecolor="white", markeredgewidth=1.0,
    linewidth=0, markersize=6, label="Mean",
)
ax1.legend(
    handles=legend_patches + [legend_extra],
    frameon=True, facecolor="white", edgecolor=SPINE_COLOR,
    labelcolor=TEXT_COLOR, fontsize=9, loc="center right",
)

out_path2 = "Figure2_fact_verification1000s.png"
fig2.savefig(out_path2, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {out_path2}")

plt.show()