"""
Stability Analysis — LLM-as-Judge (temperature=1.0)
=====================================================
Repeats the LLM-as-Judge evaluation N_RUNS times for selected models
using temperature=1.0 to measure run-to-run score variation.

Models evaluated:
  - DeepSeek-V3.1
  - DeepSeek-V3.1-Thinking

Output per model:
  outputs({model})_pubmed_abstract/stability_judge/
      run_1.csv  ...  run_5.csv         ← per-run hallucination scores
      stability_summary.csv             ← mean, std, min, max per sample
      stability_stats.txt               ← overall summary printed + saved

This experiment does NOT replace main results.
It only measures metric stability under temperature=1.0.
"""

import os
import time
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models_config import N_SAMPLES, OUTPUT_BASE_DIR, DATA_CSV


# ============================================================
# STABILITY CONFIGURATION
# ============================================================

STABILITY_MODELS = ["DeepSeek-V3.1", "DeepSeek-V3.1-Thinking"]

JUDGE_MODEL    = "gpt-4o-mini"
N_RUNS         = 5
TEMPERATURE    = 1.0          
N_THREADS      = 8
MAX_RETRIES    = 3
RETRY_DELAY    = 2.0

ALLOWED_SCORES = [
    0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
    0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
    0.8, 0.9, 1.0
]

SCORE_LABELS = {
    0.0 : "No hallucination",
    0.1 : "Very tiny hallucination",
    0.15: "Slight hallucination",
    0.2 : "Small hallucination",
    0.25: "Small-moderate hallucination",
    0.3 : "Noticeable hallucination",
    0.35: "Noticeable-moderate hallucination",
    0.4 : "Moderate hallucination",
    0.45: "Moderate-strong hallucination",
    0.5 : "Half correct / half hallucinated",
    0.55: "Strong-ish hallucination",
    0.6 : "Strong hallucination",
    0.65: "Strong-very strong hallucination",
    0.7 : "Very strong hallucination",
    0.8 : "Severe hallucination",
    0.9 : "Extremely strong hallucination",
    1.0 : "Completely hallucinated",
}

_print_lock = threading.Lock()
def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# ============================================================
# OPENAI CLIENT
# ============================================================

_api_key = os.environ.get("OPENAI_API_KEY", "")
if not _api_key:
    raise ValueError("OPENAI_API_KEY not set in environment / .env file")

openai_client = OpenAI(api_key=_api_key)

print("=" * 72)
print("  STABILITY ANALYSIS — LLM-as-Judge")
print("=" * 72)
print(f"  Judge model  : {JUDGE_MODEL}")
print(f"  Temperature  : {TEMPERATURE}  (main eval used 0.0)")
print(f"  Runs         : {N_RUNS}")
print(f"  Models       : {STABILITY_MODELS}")
print(f"  Purpose      : measure run-to-run score variation only")
print("=" * 72)


# ============================================================
# PROMPT  (identical to main eval)
# ============================================================

JUDGE_SYSTEM = (
    "You will evaluate how hallucinated the SUMMARY is compared to the SOURCE.\n"
    "Definition: A hallucination is any factual claim in the SUMMARY that is not supported by the SOURCE.\n\n"
    "Task:\n"
    "- Read the SOURCE carefully.\n"
    "- Read the SUMMARY carefully.\n"
    "- Decide how hallucinated the SUMMARY is.\n\n"
    "Give ONE number between 0 and 1 ONLY, where:\n"
    "0.0 = No hallucination at all (perfectly faithful)\n"
    "0.1 = Very tiny hallucination\n"
    "0.15 = Slight hallucination\n"
    "0.2 = Small hallucination\n"
    "0.25 = Small-moderate hallucination\n"
    "0.3 = Noticeable hallucination\n"
    "0.35 = Noticeable-moderate hallucination\n"
    "0.4 = Moderate hallucination\n"
    "0.45 = Moderate-strong hallucination\n"
    "0.5 = Half correct / half hallucinated\n"
    "0.55 = Strong-ish hallucination\n"
    "0.6 = Strong hallucination\n"
    "0.65 = Strong-very strong hallucination\n"
    "0.7 = Very strong hallucination\n"
    "0.8 = Severe hallucination\n"
    "0.9 = Extremely strong hallucination\n"
    "1.0 = Completely hallucinated\n\n"
    "IMPORTANT RULES:\n"
    "- Output ONLY the number.\n"
    "- Do NOT output any explanation or text.\n"
    "- Only use one of these exact values: "
    "0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, "
    "0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0"
)

JUDGE_USER = (
    "SOURCE:\n{article}\n\n"
    "SUMMARY:\n{summary}"
)


# ============================================================
# JUDGE FUNCTION  (temperature=1.0 here)
# ============================================================

def judge_sample(article: str, summary: str) -> dict:
    user_prompt = JUDGE_USER.format(
        article=article.strip(),
        summary=summary.strip()
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = openai_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=10,
                temperature=TEMPERATURE,   
            )

            raw   = response.choices[0].message.content.strip()
            score = float(raw)
            score = max(0.0, min(1.0, score))
            score = min(ALLOWED_SCORES, key=lambda x: abs(x - score))
            faithfulness = round(1.0 - score, 4)

            return {
                "hallucination_score": score,
                "faithfulness_score" : faithfulness,
                "raw_response"       : raw,
            }

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                return {
                    "hallucination_score": None,
                    "faithfulness_score" : None,
                    "raw_response"       : f"Error: {str(e)}",
                }

    return {
        "hallucination_score": None,
        "faithfulness_score" : None,
        "raw_response"       : "Max retries",
    }


def evaluate_sample(idx: int, article: str, summary: str) -> dict:
    if not summary or summary.startswith("ERROR"):
        return {
            "idx"                : idx,
            "hallucination_score": None,
            "faithfulness_score" : None,
            "raw_response"       : "Invalid summary",
        }
    result        = judge_sample(article, summary)
    result["idx"] = idx
    return result


# ============================================================
# ONE FULL RUN
# ============================================================

def run_once(articles, summaries, run_id, n, model_name):
    print(f"\n  ── Run {run_id}/{N_RUNS} ──────────────────────────────────")
    results   = [None] * n
    completed = 0
    errors    = 0
    t_start   = time.time()

    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = {
            executor.submit(evaluate_sample, i, articles[i], summaries[i]): i
            for i in range(n)
        }
        for future in as_completed(futures):
            result                   = future.result()
            results[result["idx"]]   = result
            completed               += 1
            if result["hallucination_score"] is None:
                errors += 1

            percent = (completed / n) * 100
            filled  = int(40 * completed / n)
            bar     = '█' * filled + '░' * (40 - filled)
            elapsed = time.time() - t_start
            rate    = completed / elapsed if elapsed > 0 else 0
            eta     = (n - completed) / rate if rate > 0 else 0
            print(
                f"\r    [{bar}] {completed}/{n} ({percent:.1f}%) "
                f"| Rate: {rate:.1f}/s | ETA: {eta:.0f}s | Errors: {errors}",
                end="", flush=True,
            )

    print()
    rows = [{k: v for k, v in r.items() if k != "idx"} for r in results]
    return pd.DataFrame(rows)


# ============================================================
# MAIN LOOP
# ============================================================

print(f"\nLoading article data from {DATA_CSV} ...")
data_df = pd.read_csv(DATA_CSV)
print(f"  {len(data_df)} rows loaded.")

for CURRENT_MODEL in STABILITY_MODELS:

    RESULTS_CSV = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/results1000.csv"
    OUT_DIR     = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/stability_judge"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n\n{'#'*72}")
    print(f"  Model : {CURRENT_MODEL}")
    print(f"  Runs  : {N_RUNS}  |  Temperature : {TEMPERATURE}")
    print(f"  Input : {RESULTS_CSV}")
    print(f"  OutDir: {OUT_DIR}")
    print(f"{'#'*72}")

    try:
        results_df = pd.read_csv(RESULTS_CSV)
    except FileNotFoundError:
        print(f"  [SKIP] File not found: {RESULTS_CSV}")
        continue

    n         = min(N_SAMPLES, len(results_df))
    summaries = results_df["generated_summary"].astype(str).tolist()[:n]
    articles  = data_df["article"].astype(str).tolist()[:n]

    print(f"  Samples : {n}")

    # ── Run N_RUNS times ─────────────────────────────────────
    all_hall_scores = []   

    for run_id in range(1, N_RUNS + 1):
        run_df = run_once(articles, summaries, run_id, n, CURRENT_MODEL)

        # Save individual run
        run_path = os.path.join(OUT_DIR, f"run_{run_id}.csv")
        run_df.to_csv(run_path, index=False, encoding="utf-8")
        print(f"    Saved → {run_path}")

        hall_series = run_df["hallucination_score"].rename(f"run_{run_id}")
        all_hall_scores.append(hall_series)

    # ── Stability summary ─────────────────────────────────────
    runs_df = pd.concat(all_hall_scores, axis=1)   

    summary_df = pd.DataFrame({
        "mean_hallucination" : runs_df.mean(axis=1),
        "std_hallucination"  : runs_df.std(axis=1),
        "min_hallucination"  : runs_df.min(axis=1),
        "max_hallucination"  : runs_df.max(axis=1),
        "range_hallucination": runs_df.max(axis=1) - runs_df.min(axis=1),
    })
    for col in runs_df.columns:
        summary_df[col] = runs_df[col]

    summary_path = os.path.join(OUT_DIR, "stability_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\n  Stability summary saved → {summary_path}")

    # ── Print statistics ──────────────────────────────────────
    valid_mask   = runs_df.notna().all(axis=1)
    valid_runs   = runs_df[valid_mask]
    mean_per_run = valid_runs.mean()
    std_per_run  = valid_runs.std()
    overall_std  = summary_df["std_hallucination"].dropna()

    stats_lines = []
    stats_lines.append("=" * 72)
    stats_lines.append(f"  STABILITY RESULTS — {CURRENT_MODEL}")
    stats_lines.append(f"  Judge: {JUDGE_MODEL} | Temperature: {TEMPERATURE} | Runs: {N_RUNS}")
    stats_lines.append("=" * 72)
    stats_lines.append(f"\n  Per-run mean hallucination score:")
    for run_col in runs_df.columns:
        m = valid_runs[run_col].mean()
        stats_lines.append(f"    {run_col}: {m:.4f}")

    stats_lines.append(f"\n  Cross-run statistics (over {len(valid_runs)} valid samples):")
    stats_lines.append(f"    Mean of per-sample std  : {overall_std.mean():.4f}")
    stats_lines.append(f"    Max  of per-sample std  : {overall_std.max():.4f}")
    stats_lines.append(f"    Mean of per-sample range: {summary_df['range_hallucination'].dropna().mean():.4f}")

    stats_lines.append(f"\n  Interpretation:")
    mean_std = overall_std.mean()
    if mean_std < 0.05:
        interp = "Very stable — metric is reliable even at temperature=1.0"
    elif mean_std < 0.10:
        interp = "Mostly stable — minor variation, metric is reasonably reliable"
    elif mean_std < 0.20:
        interp = "Moderate variation — temperature affects scores noticeably"
    else:
        interp = "High variation — metric is sensitive to temperature, use temperature=0.0"
    stats_lines.append(f"    {interp}")

    stats_lines.append(f"\n  Note: Main benchmark scores used temperature=0.0 and are unaffected.")
    stats_lines.append("=" * 72)

    stats_text = "\n".join(stats_lines)
    print(stats_text)

    stats_path = os.path.join(OUT_DIR, "stability_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(stats_text + "\n")
    print(f"\n  Stats saved → {stats_path}")


print("\n\n🎉 Stability analysis complete.")
print("   Main benchmark results are unchanged.")
print("   Use stability_summary.csv and stability_stats.txt for your report.")