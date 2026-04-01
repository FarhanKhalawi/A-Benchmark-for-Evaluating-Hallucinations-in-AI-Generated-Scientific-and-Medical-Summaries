"""
plot_factscore.py — FactScore / Hallucination Boxplot Visualization
===================================================================
Reads results_with_factscorenew.csv for one or more models and generates
a publication-quality figure with:

1) Hallucination score distribution per model
   hallucination = 1 - factscore

2) Fact verification breakdown per model
   supported / not verifiable / contradicted  (in %)

This version fixes:
- top annotation shows MEAN hallucination
- bottom annotation shows MEAN percentage
- bottom plot also shows a white diamond for the mean
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from models_config import OUTPUT_BASE_DIR, ACTIVE_MODEL


if isinstance(ACTIVE_MODEL, str):
    ACTIVE_MODEL = [ACTIVE_MODEL]

PALETTE = [
    "#00C9FF",
    "#FF6B6B",
    "#A8FF78",
    "#FFD93D",
    "#C77DFF",
    "#FF9F43",
    "#1DD1A1",
    "#54A0FF",
    "#FF6EB4",
    "#01CBC6",
]

GRID_KW = dict(color="#30363D", linewidth=0.6, linestyle="--", alpha=0.8)
TITLE_KW = dict(
    color="#E6EDF3",
    fontsize=11,
    fontweight="bold",
    pad=10,
    fontfamily="monospace",
)
LABEL_KW = dict(color="#8B949E", fontsize=9, fontfamily="monospace")
TICK_KW = dict(colors="#8B949E", labelsize=8)


def ensure_required_columns(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    df = df.copy()

    required_defaults = {
        "factscore": np.nan,
        "supported": 0,
        "not_verifiable": 0,
        "contradicted": 0,
        "total_facts": np.nan,
    }

    for col, default in required_defaults.items():
        if col not in df.columns:
            print(f"  WARNING: column '{col}' missing for {model_name} — filling with {default}")
            df[col] = default

    numeric_cols = ["factscore", "supported", "not_verifiable", "contradicted", "total_facts"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_total_mask = df["total_facts"].isna() | (df["total_facts"] <= 0)
    reconstructed_total = (
        df["supported"].fillna(0)
        + df["not_verifiable"].fillna(0)
        + df["contradicted"].fillna(0)
    )
    df.loc[missing_total_mask, "total_facts"] = reconstructed_total.loc[missing_total_mask]

    return df


def safe_pct(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    denominator = denominator.replace(0, np.nan).astype(float)
    pct = (numerator / denominator) * 100.0
    pct = pct.replace([np.inf, -np.inf], np.nan).dropna()
    return pct.values


def _style_bp(bp, color: str, n: int) -> None:
    for i in range(n):
        bp["boxes"][i].set(
            facecolor=color,
            alpha=0.50,
            linewidth=1.4,
            edgecolor=color,
        )
        for j in range(2):
            bp["whiskers"][i * 2 + j].set(
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.7,
            )
            bp["caps"][i * 2 + j].set(
                color=color,
                linewidth=1.4,
                alpha=0.7,
            )
        bp["medians"][i].set(color="#FFFFFF", linewidth=2.2)

        if len(bp["fliers"]) > i:
            bp["fliers"][i].set(
                marker="o",
                color=color,
                alpha=0.55,
                markersize=4,
                markeredgewidth=0,
            )


model_data = {}

for model_name in ACTIVE_MODEL:
    out_dir = OUTPUT_BASE_DIR.format(model=model_name)
    csv_path = os.path.join(out_dir, "results_with_factscore100s.csv")

    try:
        df = pd.read_csv(csv_path)
        df = ensure_required_columns(df, model_name)
        model_data[model_name] = df
        print(f"  Loaded {len(df)} rows from {csv_path}")
    except FileNotFoundError:
        print(f"  WARNING: {csv_path} not found — skipping {model_name}")

if not model_data:
    raise RuntimeError("No data loaded. Run factscore.py first.")

model_names = list(model_data.keys())
n_models = len(model_names)
colors = [PALETTE[i % len(PALETTE)] for i in range(n_models)]

fig = plt.figure(figsize=(14, 12), facecolor="#0D1117")
fig.patch.set_facecolor("#0D1117")

gs = gridspec.GridSpec(
    2, 1,
    figure=fig,
    hspace=0.48,
    top=0.91,
    bottom=0.07,
    left=0.08,
    right=0.97,
)

x_pos = np.arange(1, n_models + 1)

# ─────────────────────────────────────────────────────────────
# Panel 0 — Hallucination = 1 - FactScore
# ─────────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.set_facecolor("#161B22")
for spine in ax0.spines.values():
    spine.set_edgecolor("#30363D")

halluc_data = []
for df in model_data.values():
    factscore = pd.to_numeric(df["factscore"], errors="coerce").dropna()
    halluc = 1.0 - factscore
    halluc = halluc.replace([np.inf, -np.inf], np.nan).dropna().values
    halluc_data.append(halluc)

for i, (halluc, color) in enumerate(zip(halluc_data, colors)):
    if len(halluc) == 0:
        ax0.text(
            x_pos[i], 0.02, "No data",
            ha="center", va="bottom",
            color="#8B949E", fontsize=8, fontfamily="monospace"
        )
        continue

    bp = ax0.boxplot(
        halluc,
        patch_artist=True,
        notch=False,
        widths=0.50,
        positions=[x_pos[i]],
        manage_ticks=False,
    )
    _style_bp(bp, color, 1)

    mean_val = np.mean(halluc)

    ax0.text(
        x_pos[i], mean_val + 0.025, f"{mean_val:.4f}",
        ha="center", va="bottom",
        color="#FFFFFF", fontsize=8,
        fontfamily="monospace", fontweight="bold",
    )

    ax0.scatter(
        x_pos[i], mean_val,
        marker="D", color="#FFFFFF", s=32, zorder=5, alpha=0.9,
    )

for thresh, label, c in [
    (0.2, "low", "#8B949E"),
    (0.4, "medium", "#FFD93D"),
    (0.6, "high", "#FF6B6B"),
]:
    ax0.axhline(thresh, color=c, linewidth=0.8, linestyle=":", alpha=0.5)
    ax0.text(
        n_models + 0.65, thresh, label,
        color=c, fontsize=7, va="center", fontfamily="monospace",
    )

ax0.set_xlim(0.3, n_models + 1.3)
ax0.set_ylim(-0.05, 1.12)
ax0.set_xticks(x_pos)
ax0.set_xticklabels(model_names, color="#E6EDF3", fontsize=9, fontfamily="monospace")
ax0.set_ylabel("Hallucination score  (1 − FactScore)", **LABEL_KW)
ax0.set_title("Hallucination score distribution per model", **TITLE_KW)
ax0.tick_params(axis="both", **TICK_KW)
ax0.yaxis.set_minor_locator(MultipleLocator(0.1))
ax0.grid(axis="y", **GRID_KW)

legend_handles = [
    Line2D([0], [0], color="#FFFFFF", linewidth=2.2, label="Median (boxplot line)"),
    Line2D([0], [0], marker="D", color="#FFFFFF", linewidth=0, markersize=7, label="Mean"),
]
ax0.legend(
    handles=legend_handles,
    frameon=True, facecolor="#161B22", edgecolor="#30363D",
    labelcolor="#E6EDF3", fontsize=8, loc="upper right",
    prop={"family": "monospace", "size": 8},
)

# ─────────────────────────────────────────────────────────────
# Panel 1 — supported / not verifiable / contradicted
# ─────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.set_facecolor("#161B22")
for spine in ax1.spines.values():
    spine.set_edgecolor("#30363D")

CAT_COLORS = {
    "Supported": "#00C9FF",
    "Not verifiable": "#FFD93D",
    "Contradicted": "#FF6B6B",
}

group_width = 0.78
n_cats = len(CAT_COLORS)
box_width = group_width / n_cats
offsets = np.linspace(
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
        elif cat_label == "Not verifiable":
            pct = safe_pct(df["not_verifiable"], t)
        else:
            pct = safe_pct(df["contradicted"], t)

        pct_data.append(pct)

    positions = x_pos + offsets[cat_idx]

    for i, (pct, pos) in enumerate(zip(pct_data, positions)):
        if len(pct) == 0:
            ax1.text(
                pos, 1.0, "N/A",
                ha="center", va="bottom",
                color="#8B949E", fontsize=7, fontfamily="monospace",
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

        mean_pct = np.mean(pct)
        ax1.text(
            pos, mean_pct + 1.5, f"{mean_pct:.1f}%",
            ha="center", va="bottom",
            color="#FFFFFF", fontsize=7, fontfamily="monospace",
        )

        ax1.scatter(
            pos, mean_pct,
            marker="D", color="#FFFFFF", s=20, zorder=5, alpha=0.9,
        )

ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_names, color="#E6EDF3", fontsize=9, fontfamily="monospace")
ax1.set_xlim(0.3, n_models + 1.3)
ax1.set_ylim(-5, 115)
ax1.set_ylabel("% of atomic facts", **LABEL_KW)
ax1.set_title(
    "Fact verification breakdown per model  (supported / not-verifiable / contradicted)",
    **TITLE_KW,
)
ax1.tick_params(axis="both", **TICK_KW)
ax1.grid(axis="y", **GRID_KW)

legend_patches = [
    mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.7, label=lbl)
    for lbl, c in CAT_COLORS.items()
]
legend_extra = Line2D([0], [0], marker="D", color="#FFFFFF", linewidth=0, markersize=6, label="Mean")
ax1.legend(
    handles=legend_patches + [legend_extra],
    frameon=True, facecolor="#161B22", edgecolor="#30363D",
    labelcolor="#E6EDF3", fontsize=8, loc="upper right",
    prop={"family": "monospace", "size": 8},
)

models_str = " · ".join(ACTIVE_MODEL)
fig.suptitle(
    f"FactScore Hallucination Analysis\n{models_str}",
    color="#E6EDF3", fontsize=13, fontweight="bold",
    fontfamily="monospace", y=0.975,
)

out_path = "Boxplot_factscore_analysis.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#0D1117")
print(f"\nSaved → {out_path}")

plt.show()