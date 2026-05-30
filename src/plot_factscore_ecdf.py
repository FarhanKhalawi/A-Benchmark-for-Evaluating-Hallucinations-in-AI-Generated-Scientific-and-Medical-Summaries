"""
plot_factscore_ecdf.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models_config import OUTPUT_BASE_DIR, ACTIVE_MODEL


# ============================================================
# Config
# ============================================================
if isinstance(ACTIVE_MODEL, str):
    ACTIVE_MODEL = [ACTIVE_MODEL]

FIG_BG = "#FFFFFF"
AX_BG = "#FFFFFF"
SPINE_COLOR = "#B8C0CC"
TEXT_COLOR = "#111827"
SUBTEXT_COLOR = "#4B5563"
GRID_COLOR = "#D9DEE5"


X_MAX = 0.50


SORT_MODELS_BY_MEAN = True


# ============================================================
# Helpers
# ============================================================
def ensure_required_columns(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    df = df.copy()

    if "factscore" not in df.columns:
        raise ValueError(f"'factscore' column is missing for {model_name}")

    df["factscore"] = pd.to_numeric(df["factscore"], errors="coerce")
    df = df.dropna(subset=["factscore"])

    # Keep only valid FactScore values
    df = df[(df["factscore"] >= 0.0) & (df["factscore"] <= 1.0)]

    if df.empty:
        raise ValueError(f"No valid factscore values found for {model_name}")

    return df


def load_hallucination_vector(csv_path: str, model_name: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df = ensure_required_columns(df, model_name)

    halluc = 1.0 - df["factscore"].values
    halluc = np.clip(halluc.astype(float), 0.0, 1.0)

    return halluc


def compute_ecdf(values: np.ndarray):
    """
    Returns sorted x-values and cumulative probability y-values.
    Example:
    y = 0.80 means 80% of summaries have hallucination score <= x.
    """
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


# ============================================================
# Load data
# ============================================================
model_vectors = {}

for model_name in ACTIVE_MODEL:
    out_dir = OUTPUT_BASE_DIR.format(model=model_name)
    csv_path = os.path.join(out_dir, "results_with_factscore100s.csv")

    try:
        halluc = load_hallucination_vector(csv_path, model_name)
        model_vectors[model_name] = halluc
        print(f"Loaded {len(halluc)} samples from {csv_path}")

    except FileNotFoundError:
        print(f"WARNING: {csv_path} not found — skipping {model_name}")

    except Exception as e:
        print(f"WARNING: failed to load {csv_path} for {model_name}: {e}")

if not model_vectors:
    raise RuntimeError("No data loaded. Check your CSV files first.")


# ============================================================
# Sort models by mean hallucination score
# ============================================================
model_names = list(model_vectors.keys())

if SORT_MODELS_BY_MEAN:
    model_names = sorted(
        model_names,
        key=lambda m: np.mean(model_vectors[m])
    )


# ============================================================
# Print summary
# ============================================================
print("\nSummary statistics")
print("-" * 70)

for model_name in model_names:
    values = model_vectors[model_name]
    print(
        f"{model_name:30s} "
        f"N={len(values):4d} | "
        f"mean={np.mean(values):.4f} | "
        f"median={np.median(values):.4f} | "
        f"max={np.max(values):.4f}"
    )


# ============================================================
# Plot ECDF
# ============================================================
fig, ax = plt.subplots(figsize=(12.5, 6.4), facecolor="white")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for model_name in model_names:
    values = model_vectors[model_name]
    x, y = compute_ecdf(values)

    ax.step(
        x,
        y,
        where="post",
        linewidth=2.0,
        label=model_name
    )

# Axis styling
for spine in ax.spines.values():
    spine.set_edgecolor(SPINE_COLOR)
    spine.set_linewidth(0.9)

ax.grid(
    True,
    color=GRID_COLOR,
    linewidth=0.8,
    alpha=0.60
)

# X-axis and Y-axis limits
ax.set_xlim(0.0, 0.90)
ax.set_ylim(0.0, 1.03)

# Axis ticks
ax.set_xticks(np.arange(0.0, 0.91, 0.1))
ax.set_yticks(np.arange(0.0, 1.01, 0.2))

# Axis labels
ax.set_xlabel(
    "Hallucination score (1 − FactScore)",
    fontsize=13,
    color=TEXT_COLOR,
    labelpad=8
)

ax.set_ylabel(
    "Cumulative proportion of summaries",
    fontsize=13,
    color=TEXT_COLOR,
    labelpad=8
)



# Tick styling
ax.tick_params(axis="x", labelsize=11, colors=SUBTEXT_COLOR)
ax.tick_params(axis="y", labelsize=11, colors=SUBTEXT_COLOR)

# Legend inside the plot
legend = ax.legend(
    title="Model",
    fontsize=10,
    title_fontsize=11,
    loc="lower right",
    frameon=True
)

legend.get_frame().set_edgecolor(SPINE_COLOR)
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_alpha(0.92)

# Layout
plt.tight_layout(rect=[0.03, 0.04, 0.98, 0.94])

# Save figure
out_path = "factscore_ecdf100s.png"

fig.savefig(
    out_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none"
)

print(f"\nSaved → {out_path}")

plt.show()