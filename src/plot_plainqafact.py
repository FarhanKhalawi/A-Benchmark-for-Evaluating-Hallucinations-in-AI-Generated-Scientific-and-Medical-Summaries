"""
plot_plainqafact.py — PlainQAFact-API Hallucination Visualization

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from models_config import OUTPUT_BASE_DIR, ACTIVE_MODEL


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
if isinstance(ACTIVE_MODEL, str):
    ACTIVE_MODEL = [ACTIVE_MODEL]

CSV_NAME = "plainqafact_api_results1000s.csv"

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

TEXT_COLOR  = "#1F2937"
MUTED_TEXT  = "#6B7280"
SPINE_COLOR = "#9CA3AF"
GRID_COLOR  = "#E5E7EB"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "normal"

GRID_KW  = dict(color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=1.0)
LABEL_KW = dict(color=TEXT_COLOR, fontsize=11)
TICK_KW  = dict(colors=TEXT_COLOR, labelsize=10)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def ensure_required_columns(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    df = df.copy()

    if "plainqafact_api_score" not in df.columns:
        raise ValueError(
            f"Column 'plainqafact_api_score' is missing for {model_name}"
        )

    df["plainqafact_api_score"] = pd.to_numeric(
        df["plainqafact_api_score"],
        errors="coerce"
    )

    df = df.dropna(subset=["plainqafact_api_score"])

   
    df = df[
        (df["plainqafact_api_score"] >= 0.0)
        & (df["plainqafact_api_score"] <= 1.0)
    ]

    
    df["hallucination_score"] = 1.0 - df["plainqafact_api_score"]

    return df


def _style_bp(bp, color: str, n: int) -> None:
    """Style a boxplot dict for the light thesis theme."""
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

        bp["medians"][i].set(
            color=TEXT_COLOR,
            linewidth=2.0
        )

        if len(bp["fliers"]) > i:
            bp["fliers"][i].set(
                marker="o",
                markerfacecolor=color,
                markeredgecolor=color,
                alpha=0.55,
                markersize=3.5,
                markeredgewidth=0,
            )


def compute_ecdf(values: np.ndarray):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


# ─────────────────────────────────────────────────────────────
# Load data in ACTIVE_MODEL order
# ─────────────────────────────────────────────────────────────
model_data = {}

for model_name in ACTIVE_MODEL:
    out_dir  = OUTPUT_BASE_DIR.format(model=model_name)
    csv_path = os.path.join(out_dir, CSV_NAME)

    try:
        df = pd.read_csv(csv_path)
        df = ensure_required_columns(df, model_name)

        if df.empty:
            print(f"  WARNING: no valid scores for {model_name} — skipping")
            continue

        model_data[model_name] = df
        print(f"  Loaded {len(df):>5} rows from {csv_path}")

    except FileNotFoundError:
        print(f"  WARNING: {csv_path} not found — skipping {model_name}")

    except Exception as e:
        print(f"  WARNING: failed to load {csv_path} for {model_name}: {e}")

if not model_data:
    raise RuntimeError("No data loaded. Run PlainQAFact evaluation first.")

model_names = list(model_data.keys())
n_models    = len(model_names)
colors      = [PALETTE[i % len(PALETTE)] for i in range(n_models)]
x_pos       = np.arange(1, n_models + 1)


# ─────────────────────────────────────────────────────────────
# Print summary
# ─────────────────────────────────────────────────────────────
print("\nSummary statistics (hallucination = 1 - PlainQAFact)")
print("-" * 80)

for model_name in model_names:
    scores = model_data[model_name]["hallucination_score"].values

    print(
        f"{model_name:30s} "
        f"N={len(scores):4d} | "
        f"mean={np.mean(scores):.4f} | "
        f"median={np.median(scores):.4f} | "
        f"min={np.min(scores):.4f} | "
        f"max={np.max(scores):.4f}"
    )


# =============================================================
# FIGURE 1 — PlainQAFact hallucination score boxplot
# =============================================================
fig1, ax0 = plt.subplots(
    figsize=(12, 5),
    facecolor="white",
    layout="constrained"
)

ax0.set_facecolor("white")

for spine in ax0.spines.values():
    spine.set_edgecolor(SPINE_COLOR)
    spine.set_linewidth(0.8)

ax0.spines["top"].set_visible(False)
ax0.spines["right"].set_visible(False)

score_data = []

for df in model_data.values():
    scores = pd.to_numeric(
        df["hallucination_score"],
        errors="coerce"
    ).dropna()

    scores = scores.replace([np.inf, -np.inf], np.nan).dropna().values
    score_data.append(scores)

for i, (scores, color) in enumerate(zip(score_data, colors)):
    if len(scores) == 0:
        ax0.text(
            x_pos[i], 0.02, "No data",
            ha="center", va="bottom",
            color=MUTED_TEXT, fontsize=9,
        )
        continue

    bp = ax0.boxplot(
        scores,
        patch_artist=True,
        notch=False,
        widths=0.55,
        positions=[x_pos[i]],
        manage_ticks=False,
    )

    _style_bp(bp, color, 1)

    mean_val = float(np.mean(scores))

    ax0.text(
        x_pos[i],
        mean_val + 0.025,
        f"{mean_val:.4f}",
        ha="center",
        va="bottom",
        color=TEXT_COLOR,
        fontsize=9,
        fontweight="bold",
    )

    ax0.scatter(
        x_pos[i],
        mean_val,
        marker="D",
        color=TEXT_COLOR,
        edgecolor="white",
        linewidth=1.0,
        s=38,
        zorder=5,
    )

ax0.set_xlim(0.4, n_models + 0.6)
ax0.set_ylim(-0.03, 1.03)

ax0.set_xticks(x_pos)
ax0.set_xticklabels(
    model_names,
    color=TEXT_COLOR,
    fontsize=10,
    rotation=15,
    ha="right"
)

ax0.set_ylabel(
    "PlainQAFact-API hallucination score (1 − PlainQAFact)",
    **LABEL_KW
)

ax0.tick_params(axis="both", **TICK_KW)
ax0.yaxis.set_minor_locator(MultipleLocator(0.05))
ax0.grid(axis="y", **GRID_KW)
ax0.set_axisbelow(True)

legend_handles = [
    Line2D(
        [0], [0],
        color=TEXT_COLOR,
        linewidth=2.0,
        label="Median"
    ),
    Line2D(
        [0], [0],
        marker="D",
        color=TEXT_COLOR,
        markeredgecolor="white",
        markeredgewidth=1.0,
        linewidth=0,
        markersize=7,
        label="Mean"
    ),
]

ax0.legend(
    handles=legend_handles,
    frameon=True,
    facecolor="white",
    edgecolor=SPINE_COLOR,
    labelcolor=TEXT_COLOR,
    fontsize=9,
    loc="upper right",
)

out_path1 = "Figure_PlainQAFact_score_boxplot1000s.png"
fig1.savefig(out_path1, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {out_path1}")


# =============================================================
# FIGURE 2 — ECDF of PlainQAFact hallucination scores
# =============================================================
fig2, ax1 = plt.subplots(
    figsize=(12, 5),
    facecolor="white",
    layout="constrained"
)

ax1.set_facecolor("white")

for spine in ax1.spines.values():
    spine.set_edgecolor(SPINE_COLOR)
    spine.set_linewidth(0.8)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

for model_name, color in zip(model_names, colors):
    scores = model_data[model_name]["hallucination_score"].values
    x, y   = compute_ecdf(scores)

    ax1.step(
        x, y,
        where="post",
        linewidth=2.0,
        color=color,
        label=model_name
    )

ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0.0, 1.03)

ax1.set_xlabel(
    "PlainQAFact-API hallucination score (1 − PlainQAFact)",
    **LABEL_KW
)

ax1.set_ylabel(
    "Cumulative proportion of summaries",
    **LABEL_KW
)

ax1.tick_params(axis="both", **TICK_KW)
ax1.grid(axis="both", **GRID_KW)
ax1.set_axisbelow(True)

ax1.legend(
    frameon=True,
    facecolor="white",
    edgecolor=SPINE_COLOR,
    labelcolor=TEXT_COLOR,
    fontsize=9,
    loc="lower right",
)

out_path2 = "Figure_PlainQAFact_score_ecdf1000s.png"
fig2.savefig(out_path2, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {out_path2}")

# =============================================================
# FIGURE 3 — Supported questions ratio
# =============================================================
fig3, ax2 = plt.subplots(
    figsize=(12, 5),
    facecolor="white",
    layout="constrained"
)

ax2.set_facecolor("white")

for spine in ax2.spines.values():
    spine.set_edgecolor(SPINE_COLOR)
    spine.set_linewidth(0.8)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ratios       = []
total_qs     = []
supported_qs = []

for model_name in model_names:
    df = model_data[model_name]

    if "n_supported_questions" in df.columns and "n_questions" in df.columns:
        total_q     = df["n_questions"].sum()
        supported_q = df["n_supported_questions"].sum()
        ratio       = supported_q / total_q if total_q > 0 else 0.0
    else:
        total_q     = 0
        supported_q = 0
        ratio       = 0.0

    ratios.append(ratio)
    total_qs.append(total_q)
    supported_qs.append(supported_q)

bars = ax2.bar(
    x_pos,
    ratios,
    width=0.6,
    color=colors,
    alpha=0.7,
    edgecolor=colors,
    linewidth=1.2,
    zorder=3,
)

for i, (bar, ratio, supported, total) in enumerate(
    zip(bars, ratios, supported_qs, total_qs)
):
   
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        ratio + 0.008,
        f"{ratio:.1%}",
        ha="center",
        va="bottom",
        color=TEXT_COLOR,
        fontsize=9,
        fontweight="bold",
    )

    
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        ratio / 2,
        f"{supported}/{total}",
        ha="center",
        va="center",
        color="black",
        fontsize=8.5,
        fontweight="bold",
    )

ax2.set_xlim(0.4, n_models + 0.6)
ax2.set_ylim(0.0, 1.1)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(
    model_names,
    color=TEXT_COLOR,
    fontsize=10,
    rotation=15,
    ha="right"
)
ax2.set_ylabel(
    "Proportion of questions answered from source article",
    **LABEL_KW
)
ax2.tick_params(axis="both", **TICK_KW)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax2.grid(axis="y", **GRID_KW)
ax2.set_axisbelow(True)

# Mean reference line
mean_ratio = float(np.mean(ratios))
ax2.axhline(
    mean_ratio,
    color=TEXT_COLOR,
    linewidth=1.2,
    linestyle=":",
    alpha=0.6,
    label=f"Mean across models: {mean_ratio:.1%}",
    zorder=2,
)

ax2.legend(
    frameon=True,
    facecolor="white",
    edgecolor=SPINE_COLOR,
    labelcolor=TEXT_COLOR,
    fontsize=9,
    loc="lower right",
)

out_path3 = "Figure_PlainQAFact_supported_questions1000s.png"
fig3.savefig(out_path3, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {out_path3}")

plt.show()