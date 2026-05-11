# ============================================================
# pearson_cross_metric.py
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

N_SAMPLES = 1000

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = PROJECT_ROOT
SAVE_DIR     = os.path.join(PROJECT_ROOT, "src", "plots")
os.makedirs(SAVE_DIR, exist_ok=True)

MODELS = [
    "Qwen3-0.6B", "Qwen3.5-9B", "GPT-4.1-mini",
    "DeepSeek-V3.1-Thinking", "DeepSeek-V3.1",
    "GPT-5.4-nano", "GPT-5.4-nano-reasoning",
    "GPT-5-mini", "Human-Written",
]

SUFFIX  = f"{N_SAMPLES}s"
METRICS = ["FactScore", "PlainQAFact", "LLM-Judge"]

def get_paths(model):
    base = os.path.join(OUTPUT_DIR, f"outputs({model})_pubmed_abstract")
    return {
        "factscore": os.path.join(base, f"results_with_factscore{SUFFIX}.csv"),
        "plain":     os.path.join(base, f"plainqafact_api_results{SUFFIX}.csv"),
        "judge":     os.path.join(base, f"results_llm_judge{SUFFIX}.csv"),
    }

def load_model(model):
    paths = get_paths(model)
    try:
        fs = pd.read_csv(paths["factscore"])
        pl = pd.read_csv(paths["plain"])
        jd = pd.read_csv(paths["judge"])
    except FileNotFoundError as e:
        print(f"  [SKIP] {model}: {e}")
        return None
    return pd.DataFrame({
        "model":       model,
        "FactScore":   1 - fs["factscore"],
        "PlainQAFact": 1 - pl["plainqafact_api_score"],
        "LLM-Judge":   jd["hallucination_score"],
    })

def pearson_pair(s1, s2):
    tmp = pd.concat([s1, s2], axis=1).dropna()
    if len(tmp) < 5:
        return np.nan
    r, _ = stats.pearsonr(tmp.iloc[:, 0], tmp.iloc[:, 1])
    return r

def build_corr_matrix(df):
    n = len(METRICS)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i][j] = pearson_pair(df[METRICS[i]], df[METRICS[j]])
    return mat

def draw_heatmap(ax, mat, title, fontsize_val=11):
    n = len(METRICS)
    cmap = plt.cm.YlGnBu
    vmin, vmax = -0.4, 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            brightness = norm(val)
            txt_color = "white" if brightness > 0.75 or brightness < 0.25 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=fontsize_val, fontweight="bold", color=txt_color)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(METRICS, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(METRICS, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    return im

# Load
print(f"\nLoading {N_SAMPLES}-sample files...")
frames = [load_model(m) for m in MODELS]
frames = [f for f in frames if f is not None]
pooled = pd.concat(frames, ignore_index=True)
print(f"Pooled shape: {pooled.shape}")

# Figure 1: Pooled
fig1, ax1 = plt.subplots(figsize=(4.5, 3.8))
fig1.patch.set_facecolor("white")
mat_pooled = build_corr_matrix(pooled)
im = draw_heatmap(ax1, mat_pooled, "", fontsize_val=13)
cbar = fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)
#fig1.suptitle("Pearson correlations between hallucination metrics",
 #             fontsize=11, fontweight="bold", y=1.02)
fig1.tight_layout()
out1 = os.path.join(SAVE_DIR, f"pearson_pooled_{N_SAMPLES}s.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out1}")
plt.close(fig1)

# Figure 2: Per-model
loaded_models = pooled["model"].unique().tolist()
n_models = len(loaded_models)
ncols = 3
nrows = (n_models + ncols - 1) // ncols
fig2, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.6))
fig2.patch.set_facecolor("white")
axes_flat = axes.flatten()
for ax_idx, (model, grp) in enumerate(pooled.groupby("model", sort=False)):
    ax = axes_flat[ax_idx]
    mat = build_corr_matrix(grp)
    draw_heatmap(ax, mat, model, fontsize_val=10)
for idx in range(n_models, len(axes_flat)):
    axes_flat[idx].set_visible(False)
fig2.subplots_adjust(right=0.88, hspace=0.55, wspace=0.4)
cbar_ax = fig2.add_axes([0.91, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap=plt.cm.YlGnBu,
                            norm=mcolors.Normalize(vmin=-0.4, vmax=1.0))
sm.set_array([])
cbar2 = fig2.colorbar(sm, cax=cbar_ax)
cbar2.ax.tick_params(labelsize=8)
#fig2.suptitle(
#    f"Per-model Pearson correlations between hallucination metrics ({N_SAMPLES} samples each)",
#    fontsize=11, fontweight="bold", y=1.01
#)
out2 = os.path.join(SAVE_DIR, f"pearson_per_model_{N_SAMPLES}s.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out2}")
plt.close(fig2)

print("\nDone.")