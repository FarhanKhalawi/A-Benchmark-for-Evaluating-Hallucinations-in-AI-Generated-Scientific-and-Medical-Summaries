"""
plot_invalid.py — Plot for invalid outputs per model
=======================================================
Shows the proportion of invalid summaries per model based on:
  - Missing Background/Methods/Results/Conclusion
  - More than 15 sentences
  - Empty or ERROR
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from models_config import OUTPUT_BASE_DIR, ACTIVE_MODEL, N_SAMPLES



# ============================================================
# VALIDATION
# ============================================================

def count_sentences(summary):
    clean = re.sub(r'\*\*', '', summary)
    clean = re.sub(r'\n+', ' ', clean)
    clean = re.sub(r'(background|methods|results|conclusion)\s*:', '', clean, flags=re.IGNORECASE)
    clean = clean.strip()
    return len(sent_tokenize(clean))

def is_valid_summary(summary: str) -> dict:
    if not summary or summary.startswith("ERROR"):
        return {"valid": False, "reason": "error"}

    summary_lower = re.sub(r'\*\*', '', summary).lower()
    required = ["background", "methods", "results", "conclusion"]
    missing = [s for s in required if s not in summary_lower]
    if missing:
        return {"valid": False, "reason": "missing_sections"}

    n_sents = count_sentences(summary)
    if n_sents > 15:
        return {"valid": False, "reason": "too_long"}

    return {"valid": True, "reason": None}


# ============================================================
# LOAD DATA
# ============================================================

if isinstance(ACTIVE_MODEL, str):
    ACTIVE_MODEL = [ACTIVE_MODEL]

model_stats = {}

for model_name in ACTIVE_MODEL:
    csv_path = os.path.join(
        OUTPUT_BASE_DIR.format(model=model_name),
        "results1000.csv"
    )
    try:
        df = pd.read_csv(csv_path).head(N_SAMPLES)
        summaries = df["generated_summary"].astype(str).tolist()

        total       = len(summaries)
        valid       = 0
        error       = 0
        missing_sec = 0
        too_long    = 0

        for s in summaries:
            result = is_valid_summary(s)
            if result["valid"]:
                valid += 1
            elif result["reason"] == "error":
                error += 1
            elif result["reason"] == "missing_sections":
                missing_sec += 1
            elif result["reason"] == "too_long":
                too_long += 1

        model_stats[model_name] = {
            "total"           : total,
            "valid_pct"       : valid       / total * 100,
            "error_pct"       : error       / total * 100,
            "missing_sec_pct" : missing_sec / total * 100,
            "too_long_pct"    : too_long    / total * 100,
        }
        print(f"  {model_name}: {valid}/{total} valid")

    except FileNotFoundError:
        print(f"  WARNING: {csv_path} not found — skipping {model_name}")

if not model_stats:
    raise RuntimeError("No data loaded. Run generate.py first.")


# ============================================================
# PLOT
# ============================================================

model_names   = list(model_stats.keys())
n_models      = len(model_names)
x_pos         = np.arange(n_models)

valid_pcts    = [model_stats[m]["valid_pct"]       for m in model_names]
missing_pcts  = [model_stats[m]["missing_sec_pct"] for m in model_names]
too_long_pcts = [model_stats[m]["too_long_pct"]    for m in model_names]
error_pcts    = [model_stats[m]["error_pct"]       for m in model_names]

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor('white')
fig.patch.set_edgecolor('white')
ax.set_facecolor('white')

bar_width = 0.65

# Stacked bars
ax.bar(x_pos, valid_pcts,    bar_width, label="Valid",
       color="#00C9FF", alpha=0.85)
ax.bar(x_pos, missing_pcts,  bar_width, label="Missing sections",
       color="#FFD93D", alpha=0.85,
       bottom=valid_pcts)
ax.bar(x_pos, too_long_pcts, bar_width, label="Too long (>14 sentences)",
       color="#FF6B6B", alpha=0.85,
       bottom=[v + m for v, m in zip(valid_pcts, missing_pcts)])
ax.bar(x_pos, error_pcts,    bar_width, label="ERROR",
       color="#8B949E", alpha=0.85,
       bottom=[v + m + t for v, m, t in zip(valid_pcts, missing_pcts, too_long_pcts)])

# Labels on valid bar
for pos, pct in zip(x_pos, valid_pcts):
    ax.text(
        pos, pct / 2, f"{pct:.1f}%",
        ha="center", va="center",
        color="black", fontsize=14,
        fontweight="bold", fontfamily="sans-serif"
    )

# Labels on invalid bars (only if segment is big enough to read)
for i, (pos, m_pct, t_pct, e_pct) in enumerate(
        zip(x_pos, missing_pcts, too_long_pcts, error_pcts)):
    bottom = valid_pcts[i]
    if m_pct > 3:
        ax.text(pos, bottom + m_pct / 2, f"{m_pct:.0f}%",
                ha="center", va="center", color="black",
                fontsize=15, fontweight="bold", fontfamily="monospace")
    bottom += m_pct
    if t_pct > 3:
        ax.text(pos, bottom + t_pct / 2, f"{t_pct:.0f}%",
                ha="center", va="center", color="black",
                fontsize=15, fontweight="bold", fontfamily="monospace")
    bottom += t_pct
    if e_pct > 3:
        ax.text(pos, bottom + e_pct / 2, f"{e_pct:.0f}%",
                ha="center", va="center", color="black",
                fontsize=15, fontweight="bold", fontfamily="monospace")

ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, fontsize=13, rotation=15, ha="right",
                   fontfamily="sans-serif")
ax.margins(x=0.05)
ax.set_ylabel("% of summaries", fontsize=13, fontfamily="monospace")

ax.set_ylim(0, 105)
ax.tick_params(axis='y', labelsize=13)

ax.grid(axis="y", linewidth=0.6, linestyle="--", alpha=0.8)
ax.axhline(100, linewidth=0.6, linestyle="--", color="#cccccc", alpha=0.8)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")

ax.legend(
    frameon=True, fontsize=12,
    bbox_to_anchor=(1.01, 1), loc='upper left',
    prop={"family": "monospace", "size": 12}
)

out_path = f"plot_invalid_outputs_{N_SAMPLES}s.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight",
            facecolor='white', edgecolor='white')
print(f"\nSaved → {out_path}")
plt.show()