"""
SUMMAC — Summary Consistency Scoring
================================================================
Scores each generated summary against its source article using
SummaCConv (local NLI model, no API calls needed).

Install once before running:
    pip install summac torch

Inputs:
  data/processed/pubmed_train_clean.csv         → article column
  outputs(Model)_pubmed_abstract/results1000.csv → generated_summary column

Output:
  outputs(Model)_pubmed_abstract/results_with_summac1000.csv
"""

import time
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from summac.model_summac import SummaCConv
from models_config import N_SAMPLES, OUTPUT_BASE_DIR, DATA_CSV, ACTIVE_MODEL


# ============================================================
# CONFIGURATION
# ============================================================

N_SAMPLE_THREADS = 4


# ── Thread-safe print ────────────────────────────────────────
_print_lock = threading.Lock()
def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# ============================================================
# SUMMAC MODEL  (loaded once, shared across all threads)
# ============================================================

print("\nLoading SummaCConv model (vitc)...")
summac_model = SummaCConv(
    models      = ["vitc"],
    bins        = "percentile",
    granularity = "sentence",
    nli_labels  = "e",           # entailment scores only
    device      = "cpu",         # switch to "cuda" if GPU available
    start_file  = "default",     # loads pre-trained weights
    agg         = "mean"
)
print("  SummaCConv ready.\n")

# SummaCConv is not thread-safe — protect with a lock
_summac_lock = threading.Lock()


# ============================================================
# SCORING
# ============================================================

def summac_pipeline(article: str, summary: str) -> float | None:
    """Returns a consistency score (0–1), or None on error."""
    try:
        with _summac_lock:
            result = summac_model.score([article], [summary])
        return round(float(result["scores"][0]), 4)
    except Exception as e:
        tprint(f"    [SUMMAC] ERROR: {e}")
        return None


def interpret(score) -> str:
    if score is None or pd.isna(score):
        return "n/a"
    if score >= 0.8: return "highly consistent"
    if score >= 0.6: return "mostly consistent"
    if score >= 0.4: return "partially consistent"
    return "likely inconsistent"


# ============================================================
# LOAD ARTICLE DATA
# ============================================================

print(f"Loading {DATA_CSV} ...")
data_df = pd.read_csv(DATA_CSV)
print(f"  {len(data_df)} rows | columns: {data_df.columns.tolist()}")

active_models = ACTIVE_MODEL if isinstance(ACTIVE_MODEL, list) else [ACTIVE_MODEL]


# ============================================================
# PER-SAMPLE WORKER
# ============================================================

def _process_one_sample(idx: int, article: str, summary: str, n: int) -> dict:
    tprint(f"\n{'='*55}  Sample {idx+1}/{n}  {'='*55}")

    if not summary or summary.startswith("ERROR"):
        tprint(f"  [Sample {idx+1}] SKIPPED — invalid summary")
        return {"idx": idx, "summac_score": None}

    score = summac_pipeline(article, summary)
    tprint(f"  [Sample {idx+1}] SUMMAC score : {score}  ({interpret(score)})")
    return {"idx": idx, "summac_score": score}


# ============================================================
# MAIN LOOP
# ============================================================

for CURRENT_MODEL in active_models:

    RESULTS_CSV = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/results1000.csv"
    OUT_CSV     = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/results_with_summac100.csv"

    print(f"\n\n{'#'*72}")
    print(f"  Processing model : {CURRENT_MODEL}")
    print(f"  Metric           : SummaCConv (vitc, sentence-level, local)")
    print(f"  Sample threads   : {N_SAMPLE_THREADS}")
    print(f"  Input            : {RESULTS_CSV}")
    print(f"  Output           : {OUT_CSV}")
    print(f"{'#'*72}")

    print("\nLoading results.csv ...")
    try:
        results_df = pd.read_csv(RESULTS_CSV)
    except FileNotFoundError:
        print(f"  [SKIP] File not found: {RESULTS_CSV}")
        continue

    print(f"  {len(results_df)} rows | columns: {results_df.columns.tolist()}")

    n         = min(N_SAMPLES, len(results_df))
    summaries = results_df["generated_summary"].astype(str).tolist()[:n]
    articles  = data_df["article"].astype(str).tolist()[:n]

    # ── Parallel processing ──────────────────────────────────
    results     = [None] * n
    completed   = 0
    total_start = time.time()

    with ThreadPoolExecutor(max_workers=N_SAMPLE_THREADS) as executor:
        futures = {
            executor.submit(
                _process_one_sample, i, articles[i], summaries[i], n
            ): i
            for i in range(n)
        }
        for future in as_completed(futures):
            result                 = future.result()
            results[result["idx"]] = result
            completed             += 1
            tprint(f"\n  ✓ Completed {completed}/{n} samples")

    total_elapsed = time.time() - total_start

    # ── Summary table ────────────────────────────────────────
    scores_df = pd.DataFrame([{"summac_score": r["summac_score"]} for r in results])
    valid     = scores_df["summac_score"].dropna()

    print("\n\n" + "="*55)
    print(f"  SUMMAC SUMMARY  —  {CURRENT_MODEL}")
    print(f"  Valid samples : {len(valid)}/{n}")
    print("="*55)
    print(f"{'Row':<6} {'Score':<10} {'Interpretation'}")
    print("-"*55)

    for i, row in scores_df.iterrows():
        s     = row["summac_score"]
        s_str = "N/A" if (s is None or pd.isna(s)) else f"{s:.4f}"
        print(f"{i+1:<6} {s_str:<10} {interpret(s)}")

    if len(valid) > 0:
        print("-"*55)
        print(f"  Mean : {valid.mean():.4f}")
        print(f"  Min  : {valid.min():.4f}")
        print(f"  Max  : {valid.max():.4f}")

    print(f"\n  ⏱ Total time: {total_elapsed/60:.1f} min ({total_elapsed:.1f}s)")

    # ── Save ─────────────────────────────────────────────────
    out_df = pd.concat(
        [results_df.iloc[:n].reset_index(drop=True), scores_df],
        axis=1
    )
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved → {OUT_CSV}")

print("\n\nAll models processed.")