"""
plot_invalid.py — Plot for invalid outputs per model
=======================================================
Shows the proportion of invalid summaries per model based on:
  - Missing Background/Methods/Results/Conclusion
  - More than 10 sentences
  - Empty or ERROR
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from models_config import OUTPUT_BASE_DIR, ACTIVE_MODEL

# ============================================================
# VALIDATION
# ============================================================

def is_valid_summary(summary: str) -> dict:
    """
    Check if the summary is valid.
    Returns dict with reason for invalidity.
    """
    if not summary or summary.startswith("ERROR"):
        return {"valid": False, "reason": "error"}

    summary_lower = summary.lower()
    required = ["background:", "methods:", "results:", "conclusion:"]
    missing = [s for s in required if s not in summary_lower]
    if missing:
        return {"valid": False, "reason": "missing_sections"}

    n_sents = len(sent_tokenize(summary))
    if n_sents > 10:
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
        "resultstest.csv"
    )
    try:
        df = pd.read_csv(csv_path)
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

model_names  = list(model_stats.keys())
n_models     = len(model_names)
x_pos        = np.arange(n_models)

valid_pcts    = [model_stats[m]["valid_pct"]       for m in model_names]
missing_pcts  = [model_stats[m]["missing_sec_pct"] for m in model_names]
too_long_pcts = [model_stats[m]["too_long_pct"]    for m in model_names]
error_pcts    = [model_stats[m]["error_pct"]       for m in model_names]

fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0D1117")
ax.set_facecolor("#161B22")
for spine in ax.spines.values():
    spine.set_edgecolor("#30363D")

bar_width = 0.6

# Stacked bars
ax.bar(x_pos, valid_pcts,    bar_width, label="Valid",
       color="#00C9FF", alpha=0.85)
ax.bar(x_pos, missing_pcts,  bar_width, label="Missing sections",
       color="#FFD93D", alpha=0.85,
       bottom=valid_pcts)
ax.bar(x_pos, too_long_pcts, bar_width, label="Too long (>10 sentences)",
       color="#FF6B6B", alpha=0.85,
       bottom=[v + m for v, m in zip(valid_pcts, missing_pcts)])
ax.bar(x_pos, error_pcts,    bar_width, label="ERROR",
       color="#8B949E", alpha=0.85,
       bottom=[v + m + t for v, m, t in zip(valid_pcts, missing_pcts, too_long_pcts)])

# Labels on valid bar
for i, (pos, pct) in enumerate(zip(x_pos, valid_pcts)):
    ax.text(
        pos, pct / 2, f"{pct:.0f}%",
        ha="center", va="center",
        color="#0D1117", fontsize=10,
        fontweight="bold", fontfamily="monospace"
    )

# Labels on invalid bars
for i, (pos, m_pct, t_pct, e_pct) in enumerate(
        zip(x_pos, missing_pcts, too_long_pcts, error_pcts)):
    bottom = valid_pcts[i]
    if m_pct > 3:
        ax.text(pos, bottom + m_pct / 2, f"{m_pct:.0f}%",
                ha="center", va="center", color="#0D1117",
                fontsize=9, fontweight="bold", fontfamily="monospace")
    bottom += m_pct
    if t_pct > 3:
        ax.text(pos, bottom + t_pct / 2, f"{t_pct:.0f}%",
                ha="center", va="center", color="#0D1117",
                fontsize=9, fontweight="bold", fontfamily="monospace")
    bottom += t_pct
    if e_pct > 3:
        ax.text(pos, bottom + e_pct / 2, f"{e_pct:.0f}%",
                ha="center", va="center", color="#0D1117",
                fontsize=9, fontweight="bold", fontfamily="monospace")

ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, color="#E6EDF3", fontsize=9,
                   fontfamily="monospace", rotation=15, ha="right")
ax.set_ylabel("% of summaries", color="#8B949E", fontsize=10,
              fontfamily="monospace")
ax.set_ylim(0, 110)
ax.set_title(
    "Proportion of valid and invalid outputs per model",
    color="#E6EDF3", fontsize=13, fontweight="bold",
    fontfamily="monospace", pad=15
)
ax.tick_params(colors="#8B949E", labelsize=9)
ax.grid(axis="y", color="#30363D", linewidth=0.6,
        linestyle="--", alpha=0.8)
ax.axhline(100, color="#30363D", linewidth=0.8)

ax.legend(
    frameon=True, facecolor="#161B22", edgecolor="#30363D",
    labelcolor="#E6EDF3", fontsize=9, loc="upper right",
    prop={"family": "monospace", "size": 9}
)

out_path = "plot_invalid_outputs.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#0D1117")
print(f"\nSaved → {out_path}")
plt.show()