"""
plot_factscore_heatmap.py
============================================================
Option F — Per-sample hallucination heatmap

Creates a compact thesis/report-friendly heatmap:
- Y = model
- X = sample index
- Color = hallucination score = 1 - factscore

Uses ACTIVE_MODEL from models_config.py
Reads: results_with_factscore1000s.csv
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

# Optional: set to True if you want samples sorted by average difficulty
SORT_COLUMNS_BY_MEAN = False


# ============================================================
# Helpers
# ============================================================
def ensure_required_columns(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    df = df.copy()

    if "factscore" not in df.columns:
        raise ValueError(f"'factscore' column is missing for {model_name}")

    df["factscore"] = pd.to_numeric(df["factscore"], errors="coerce")
    return df


def load_hallucination_vector(csv_path: str, model_name: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df = ensure_required_columns(df, model_name)

    halluc = 1.0 - df["factscore"].values
    halluc = np.clip(halluc.astype(float), 0, 1)

    return halluc


# ============================================================
# Load data
# ============================================================
model_vectors = {}
max_len = 0

for model_name in ACTIVE_MODEL:
    out_dir = OUTPUT_BASE_DIR.format(model=model_name)
    csv_path = os.path.join(out_dir, "results_with_factscore1000s.csv")

    try:
        halluc = load_hallucination_vector(csv_path, model_name)
        model_vectors[model_name] = halluc
        max_len = max(max_len, len(halluc))
        print(f"Loaded {len(halluc)} samples from {csv_path}")
    except FileNotFoundError:
        print(f"WARNING: {csv_path} not found — skipping {model_name}")
    except Exception as e:
        print(f"WARNING: failed to load {csv_path} for {model_name}: {e}")

if not model_vectors:
    raise RuntimeError("No data loaded. Check your CSV files first.")

model_names = [m for m in ACTIVE_MODEL if m in model_vectors]
n_models = len(model_names)

if n_models == 0:
    raise RuntimeError("No matching models were loaded.")


# ============================================================
# Build matrix [models x samples]
# ============================================================
# Pad missing values with NaN so all rows have same width
heatmap_data = np.full((n_models, max_len), np.nan, dtype=float)

for i, model_name in enumerate(model_names):
    values = model_vectors[model_name]
    heatmap_data[i, :len(values)] = values

# Optional: sort columns by average hallucination across models
if SORT_COLUMNS_BY_MEAN:
    col_mean = np.nanmean(heatmap_data, axis=0)
    order = np.argsort(col_mean)
    heatmap_data = heatmap_data[:, order]
    x_label = "Sample rank (sorted by average hallucination)"
else:
    x_label = "Sample index"


# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(16, 4.8), facecolor=FIG_BG)
fig.patch.set_facecolor(FIG_BG)
ax.set_facecolor(AX_BG)

# Use a copy of viridis and show NaN as white
cmap = plt.cm.viridis.copy()
cmap.set_bad(color="white")

im = ax.imshow(
    heatmap_data,
    aspect="auto",
    interpolation="nearest",
    cmap=cmap,
    vmin=0.0,
    vmax=0.6,   # cap at 0.6 to make differences clearer
)

# Axis styling
for spine in ax.spines.values():
    spine.set_edgecolor(SPINE_COLOR)
    spine.set_linewidth(0.9)

ax.set_yticks(np.arange(n_models))
ax.set_yticklabels(model_names, fontsize=10, color=TEXT_COLOR)

# x ticks
if max_len >= 1000:
    xticks = [0, 200, 400, 600, 800, 999]
    xlabels = ["1", "200", "400", "600", "800", "1000"]
else:
    step = max(1, max_len // 5)
    xticks = list(range(0, max_len, step))
    if xticks[-1] != max_len - 1:
        xticks.append(max_len - 1)
    xlabels = [str(x + 1) for x in xticks]

ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, fontsize=9, color=SUBTEXT_COLOR)

ax.set_xlabel(x_label, fontsize=11, color=TEXT_COLOR)
ax.set_ylabel("Model", fontsize=11, color=TEXT_COLOR)

ax.set_title(
    "Per-sample hallucination heatmap across evaluated models",
    fontsize=16,
    fontweight="bold",
    color=TEXT_COLOR,
    pad=14,
)



# Optional faint separators between rows
for y in np.arange(-0.5, n_models, 1):
    ax.axhline(y, color=GRID_COLOR, linewidth=0.6, alpha=0.5)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
cbar.set_label("Hallucination score", fontsize=10, color=TEXT_COLOR)
cbar.ax.tick_params(labelsize=9, colors=SUBTEXT_COLOR)
cbar.outline.set_edgecolor(SPINE_COLOR)

plt.tight_layout(rect=[0.03, 0.06, 0.98, 0.90])

# Save
out_path = "factscore_heatmap1000s.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=FIG_BG)
print(f"Saved → {out_path}")

plt.show()