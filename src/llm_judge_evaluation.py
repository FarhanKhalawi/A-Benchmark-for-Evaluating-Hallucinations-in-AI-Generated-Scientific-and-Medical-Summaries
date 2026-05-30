"""
LLM-as-Judge — Hallucination Score (0.0-1.0)
=============================================
0.0 = No hallucination (perfectly faithful)
1.0 = Completely hallucinated

Install:
    pip install openai python-dotenv pandas

Inputs:
  data/processed/pubmed_train_clean.csv          → article column
  outputs(Model)_pubmed_abstract/results1000.csv → generated_summary column

Output:
  outputs(Model)_pubmed_abstract/llm_judge_hallucination.csv
"""

import os
import time
import json
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models_config import N_SAMPLES, OUTPUT_BASE_DIR, DATA_CSV, ACTIVE_MODEL


# ============================================================
# CONFIGURATION
# ============================================================

JUDGE_MODEL    = "gpt-4o-mini"
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

_api_key = os.environ.get("OPENAI_API_KEY", "")
if not _api_key:
    raise ValueError("OPENAI_API_KEY not set in environment / .env file")

openai_client = OpenAI(api_key=_api_key)
print(f"OpenAI client ready.")
print(f"  Judge model : {JUDGE_MODEL}")
print(f"  Score range : 0.0 (no hallucination) → 1.0 (completely hallucinated)")


# ============================================================
# PROMPT — your exact prompt
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
# JUDGE FUNCTION
# ============================================================

def judge_sample(article: str, summary: str) -> dict:
    """Judge hallucination level — returns 0.0-1.0 score"""
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
                temperature=0.0      
            )
            
            raw = response.choices[0].message.content.strip()
            
            
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
        "raw_response"       : "Max retries"
    }


# ============================================================
# EVALUATE SINGLE SAMPLE
# ============================================================

def evaluate_sample(idx: int, article: str, summary: str) -> dict:
    """Evaluate a single summary"""
    
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
# MAIN LOOP
# ============================================================

print(f"\nLoading {DATA_CSV} ...")
data_df = pd.read_csv(DATA_CSV)
print(f"  {len(data_df)} rows | columns: {data_df.columns.tolist()}")

active_models = ACTIVE_MODEL if isinstance(ACTIVE_MODEL, list) else [ACTIVE_MODEL]

for CURRENT_MODEL in active_models:
    
    RESULTS_CSV = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/results1000.csv"
    OUT_CSV     = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/results_llm_judge1000s.csv"
    
    print(f"\n\n{'#'*72}")
    print(f"  Processing model : {CURRENT_MODEL}")
    print(f"  Judge model      : {JUDGE_MODEL}")
    print(f"  Threads          : {N_THREADS}")
    print(f"  Score            : 0.0 (faithful) → 1.0 (hallucinated)")
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
    
    results     = [None] * n
    completed   = 0
    errors      = 0
    total_start = time.time()
    
    print(f"\nEvaluating {n} summaries...")
    print()
    
    try:
        with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            futures = {
                executor.submit(evaluate_sample, i, articles[i], summaries[i]): i
                for i in range(n)
            }
            
            for future in as_completed(futures):
                result = future.result()
                results[result["idx"]] = result
                completed += 1
                
                if result["hallucination_score"] is None:
                    errors += 1
                
                percent = (completed / n) * 100
                filled  = int(50 * completed / n)
                bar     = '█' * filled + '░' * (50 - filled)
                elapsed = time.time() - total_start
                rate    = completed / elapsed if elapsed > 0 else 0
                eta     = (n - completed) / rate if rate > 0 else 0
                
                print(f"\r  Progress: [{bar}] {completed}/{n} ({percent:.1f}%) | "
                      f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s | Errors: {errors}",
                      end="", flush=True)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Saving partial results...")
        for i in range(n):
            if results[i] is None:
                results[i] = {
                    "idx"                : i,
                    "hallucination_score": None,
                    "faithfulness_score" : None,
                    "raw_response"       : "Interrupted",
                }
    
    print()
    
    rows      = [{k: v for k, v in r.items() if k != "idx"} for r in results]
    scores_df = pd.DataFrame(rows)
    
    valid_hall   = scores_df["hallucination_score"].dropna()
    valid_faith  = scores_df["faithfulness_score"].dropna()
    total_elapsed = time.time() - total_start
    
    # ── Summary ───────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"  HALLUCINATION SCORE SUMMARY  —  {CURRENT_MODEL}")
    print("="*72)
    print(f"  Total samples      : {n}")
    print(f"  Valid evaluations  : {len(valid_hall)}")
    print(f"  Errors/Skipped     : {errors}")
    
    if len(valid_hall) > 0:
        print(f"\n  Score Distribution (hallucination level):")
        print(f"  {'Score':<8} {'Label':<35} {'Count':<8} {'%':<8} Bar")
        print(f"  {'-'*72}")
        
        for s in ALLOWED_SCORES:
            count = int((valid_hall == s).sum())
            if count == 0:
                continue
            pct   = (count / len(valid_hall)) * 100
            bar   = '█' * int(pct / 2)
            label = SCORE_LABELS[s]
            print(f"  {s:<8} {label:<35} {count:<8} {pct:<7.1f}% {bar}")
        
        print(f"\n  {'─'*55}")
        print(f"  ── Hallucination Score (0=faithful, 1=hallucinated) ──")
        print(f"  Mean  : {valid_hall.mean():.4f}")
        print(f"  Median: {valid_hall.median():.4f}")
        print(f"  Std   : {valid_hall.std():.4f}")
        print(f"  Min   : {valid_hall.min():.4f}")
        print(f"  Max   : {valid_hall.max():.4f}")
        print(f"\n  ── Faithfulness Score (1=faithful, 0=hallucinated) ──")
        print(f"  Mean  : {valid_faith.mean():.4f}")
        print(f"  Median: {valid_faith.median():.4f}")
        print(f"  Std   : {valid_faith.std():.4f}")
        print(f"  Min   : {valid_faith.min():.4f}")
        print(f"  Max   : {valid_faith.max():.4f}")
    
    print(f"\n  ⏱ Total time : {total_elapsed/60:.1f} min ({total_elapsed:.1f}s)")
    print(f"  ⚡ Rate       : {len(valid_hall)/total_elapsed:.2f} samples/sec")
    
    # Detailed table
    print("\n" + "="*72)
    print(f"  DETAILED RESULTS (first 20)")
    print("="*72)
    print(f"{'#':<5} {'Hall.':<8} {'Faith.':<8} {'Label'}")
    print("-"*72)
    
    for i, row in scores_df.head(20).iterrows():
        hall  = f"{row['hallucination_score']:.2f}" if pd.notna(row["hallucination_score"]) else "—"
        faith = f"{row['faithfulness_score']:.2f}"  if pd.notna(row["faithfulness_score"])  else "—"
        label = SCORE_LABELS.get(row["hallucination_score"], "—") \
                if pd.notna(row["hallucination_score"]) else "—"
        print(f"{i+1:<5} {hall:<8} {faith:<8} {label}")
    
    if len(scores_df) > 20:
        print(f"\n  ... and {len(scores_df)-20} more (see CSV)")
    
    # Show high hallucination examples
    print("\n" + "="*72)
    print(f"  HIGH HALLUCINATION EXAMPLES (score >= 0.3)")
    print("="*72)
    
    high_hall = scores_df[scores_df["hallucination_score"] >= 0.3].head(3)
    
    if len(high_hall) == 0:
        print("  No samples with hallucination score >= 0.3 ✅")
    else:
        for idx, row in high_hall.iterrows():
            print(f"\n--- Sample {idx+1} | Hallucination: {row['hallucination_score']} "
                  f"| Faithfulness: {row['faithfulness_score']} ---")
            print(f"Label   : {SCORE_LABELS.get(row['hallucination_score'], '—')}")
            print(f"Summary : {summaries[idx][:150]}...")
            print("-"*72)
    
    # Save
    out_df = pd.concat(
        [results_df.iloc[:n].reset_index(drop=True), scores_df],
        axis=1
    )
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Saved → {OUT_CSV}")

print("\n\n🎉 All models processed.")