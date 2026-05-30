"""
Stability Analysis — FactScore (same config, 5 runs)
=====================================================
Repeats the FactScore evaluation N_RUNS times for selected models.



Models evaluated:
  - DeepSeek-V3.1
  - DeepSeek-V3.1-Thinking

Output per model:
  outputs({model})_pubmed_abstract/stability_factscore/
      run_1.csv  ...  run_5.csv         ← per-run factscore values
      stability_summary.csv             ← mean, std, min, max per sample
      stability_stats.txt               ← overall summary printed + saved

This experiment does NOT replace main results.
"""

import os
import re
import time
import threading
import pandas as pd
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from models_config import N_SAMPLES, OUTPUT_BASE_DIR, DATA_CSV

# ── NLTK ─────────────────────────────────────────────────────
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


# ============================================================
# STABILITY CONFIGURATION
# ============================================================

STABILITY_MODELS = ["DeepSeek-V3.1", "DeepSeek-V3.1-Thinking"]

AFG_MODEL        = "gpt-5-nano"
AFV_MODEL        = "gpt-5-nano"
N_RUNS           = 5
N_THREADS        = 16
N_SAMPLE_THREADS = 4
MAX_RETRIES      = 3
RETRY_DELAY      = 2.0
AFG_MAX_TOKENS   = 2000
AFV_MAX_TOKENS   = 50
MIN_FACTS        = 3

_print_lock = threading.Lock()
def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)

_api_key = os.environ.get("OPENAI_API_KEY", "")
if not _api_key:
    raise ValueError("OPENAI_API_KEY not set in environment / .env file")

openai_client = OpenAI(api_key=_api_key)

print("=" * 72)
print("  STABILITY ANALYSIS — FactScore")
print("=" * 72)
print(f"  AFG model    : {AFG_MODEL}  [reasoning_effort=minimal]")
print(f"  AFV model    : {AFV_MODEL}  [reasoning_effort=minimal]")
print(f"  Runs         : {N_RUNS}")
print(f"  Models       : {STABILITY_MODELS}")
print(f"  Temperature  : NOT configurable for gpt-5-nano + reasoning_effort")
print(f"  Purpose      : measure natural run-to-run variation")
print("=" * 72)


# ============================================================
# VALIDATION
# ============================================================

def count_sentences(summary: str) -> int:
    clean = re.sub(r'\*\*', '', summary)
    clean = re.sub(r'\n+', ' ', clean)
    clean = re.sub(r'(background|methods|results|conclusion)\s*:', '',
                   clean, flags=re.IGNORECASE)
    return len(nltk.sent_tokenize(clean.strip()))


def is_valid_summary(summary: str) -> dict:
    if not summary or summary.startswith("ERROR"):
        return {"valid": False, "reason": "error"}
    summary_lower = re.sub(r'\*\*', '', summary).lower()
    required = ["background", "methods", "results", "conclusion"]
    if any(s not in summary_lower for s in required):
        return {"valid": False, "reason": "missing_sections"}
    if count_sentences(summary) > 15:
        return {"valid": False, "reason": "too_long"}
    return {"valid": True, "reason": None}


# ============================================================
# API CALL  (identical to main eval — no temperature param)
# ============================================================

def _call_with_retry(model: str, system: str, user: str, max_tokens: int = 5) -> str:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_completion_tokens=max_tokens,
                reasoning_effort="minimal",
                # NOTE: temperature is not passed here — not supported
                # alongside reasoning_effort for gpt-5-nano.
                # This matches the main evaluation configuration exactly.
            )
            content       = resp.choices[0].message.content
            finish_reason = resp.choices[0].finish_reason
            if finish_reason == "length" and (not content or content.strip() == ""):
                tprint(f"    [API] WARNING: empty content, token budget too small.")
                return ""
            return (content or "").strip()
        except Exception as e:
            last_exc = e
            tprint(f"    [API] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(RETRY_DELAY * attempt)
    raise last_exc


# ============================================================
# STEP 1 — AFG
# ============================================================

ATOMIZE_SYSTEM = (
    "You are a precise fact extractor for scientific text. "
    "Decompose the summary into atomic facts. "
    "Output ONLY one fact per line. "
    "No bullets, no numbers, no headers. "
    "Each line must be one complete sentence. "
    "Do not group facts into paragraphs. "
    "Each fact must contain only ONE piece of information. "
    "If a sentence contains multiple facts, split them into separate lines. "
    "Minimum 3 facts per summary, even for short summaries."
)

ATOMIZE_USER_TMPL = (
    "Decompose the following scientific summary into atomic facts. "
    "One fact per line.\n\n"
    "SUMMARY:\n{summary}"
)

ATOMIZE_SYSTEM_RETRY = (
    "You are a precise fact extractor for scientific text. "
    "Decompose the summary into atomic facts. "
    "Output ONLY one fact per line. "
    "No bullets, no numbers, no headers. "
    "Each line must be one complete sentence. "
    "Do not group facts into paragraphs. "
    "Each fact must contain only ONE piece of information. "
    "If a sentence contains multiple facts, split them into separate lines. "
    "You MUST extract at least one fact per sentence. "
    "Every clause, finding, or detail is a separate fact. "
    "Do NOT combine multiple facts into one line."
)

ATOMIZE_USER_RETRY_TMPL = (
    "This summary contains multiple facts. "
    "You MUST extract at least one fact per sentence. "
    "Decompose into atomic facts, one per line.\n\n"
    "SUMMARY:\n{summary}"
)


def _parse_facts(raw: str) -> list:
    lines     = raw.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) <= 2 and len(raw) > 100:
        lines = nltk.sent_tokenize(raw)
    facts = []
    for line in lines:
        line = re.sub(r"^[\s\-\*\•\·\d\.]+", "", line).strip()
        if len(line) > 10:
            facts.append(line)
    return facts


def extract_atomic_facts(summary: str) -> list:
    if not summary or summary.startswith("ERROR"):
        return []
    user = ATOMIZE_USER_TMPL.format(summary=summary.strip())
    raw  = _call_with_retry(AFG_MODEL, ATOMIZE_SYSTEM, user, max_tokens=AFG_MAX_TOKENS)
    if not raw:
        return []
    facts = _parse_facts(raw)
    if len(facts) < MIN_FACTS:
        user_retry = ATOMIZE_USER_RETRY_TMPL.format(summary=summary.strip())
        raw_retry  = _call_with_retry(AFG_MODEL, ATOMIZE_SYSTEM_RETRY,
                                      user_retry, max_tokens=AFG_MAX_TOKENS)
        if raw_retry:
            facts_retry = _parse_facts(raw_retry)
            if len(facts_retry) > len(facts):
                facts = facts_retry
    return facts


# ============================================================
# STEP 2 — AFV
# ============================================================

VERIFY_SYSTEM = (
    "You are a strict scientific fact-checker. "
    "You are given a SOURCE TEXT and an ATOMIC FACT. "
    "Decide whether the atomic fact is supported by the source text.\n\n"
    "- Answer only 'true' if the fact is explicitly stated or directly implied.\n"
    "- Answer only 'false' if the fact contradicts or cannot be verified from the source.\n"
    "- Output exactly one word: true or false."
)

VERIFY_USER_TMPL = (
    "SOURCE TEXT:\n{source}\n\n"
    "ATOMIC FACT:\n{fact}\n\n"
    "Is this fact supported by the source text? Answer true or false."
)


def validate_fact_api(article: str, fact: str) -> bool:
    user   = VERIFY_USER_TMPL.format(source=article, fact=fact)
    answer = _call_with_retry(AFV_MODEL, VERIFY_SYSTEM, user,
                              max_tokens=AFV_MAX_TOKENS).lower()
    return answer.startswith("true")


# ============================================================
# FULL PIPELINE
# ============================================================

def factscore_pipeline(article: str, summary: str) -> dict:
    all_facts = extract_atomic_facts(summary)
    if not all_facts:
        return {"score": None, "supported": 0,
                "not_verifiable": 0, "total_facts": 0}

    labels = [None] * len(all_facts)

    def _verify(args):
        idx, fact = args
        try:
            return idx, validate_fact_api(article, fact)
        except Exception as e:
            return idx, False

    with ThreadPoolExecutor(max_workers=min(N_THREADS, len(all_facts))) as ex:
        for idx, is_supported in ex.map(_verify, enumerate(all_facts)):
            labels[idx] = is_supported

    supported      = sum(1 for v in labels if v)
    not_verifiable = sum(1 for v in labels if not v)
    total          = len(all_facts)

    return {
        "score"         : round(supported / total, 4) if total > 0 else None,
        "supported"     : supported,
        "not_verifiable": not_verifiable,
        "total_facts"   : total,
    }


# ============================================================
# PER-SAMPLE WORKER
# ============================================================

def _process_one_sample(idx: int, article: str, summary: str, n: int) -> dict:
    validity = is_valid_summary(summary)
    if not validity["valid"]:
        return {
            "idx"           : idx,
            "factscore"     : None,
            "supported"     : None,
            "not_verifiable": None,
            "total_facts"   : None,
            "invalid_reason": validity["reason"],
        }

    result = factscore_pipeline(article, summary)
    return {
        "idx"           : idx,
        "factscore"     : result["score"],
        "supported"     : result["supported"],
        "not_verifiable": result["not_verifiable"],
        "total_facts"   : result["total_facts"],
        "invalid_reason": None,
    }


# ============================================================
# ONE FULL RUN
# ============================================================

def run_once(articles, summaries, run_id, n):
    print(f"\n  ── Run {run_id}/{N_RUNS} ──────────────────────────────────")
    results   = [None] * n
    completed = 0
    t_start   = time.time()

    with ThreadPoolExecutor(max_workers=N_SAMPLE_THREADS) as executor:
        futures = {
            executor.submit(_process_one_sample, i, articles[i], summaries[i], n): i
            for i in range(n)
        }
        for future in as_completed(futures):
            result                 = future.result()
            results[result["idx"]] = result
            completed             += 1
            percent = (completed / n) * 100
            filled  = int(40 * completed / n)
            bar     = '█' * filled + '░' * (40 - filled)
            elapsed = time.time() - t_start
            rate    = completed / elapsed if elapsed > 0 else 0
            eta     = (n - completed) / rate if rate > 0 else 0
            print(
                f"\r    [{bar}] {completed}/{n} ({percent:.1f}%) "
                f"| Rate: {rate:.1f}/s | ETA: {eta:.0f}s",
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
    OUT_DIR     = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/stability_factscore"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n\n{'#'*72}")
    print(f"  Model  : {CURRENT_MODEL}")
    print(f"  Runs   : {N_RUNS}  |  Config: same as main eval (no temperature control)")
    print(f"  Input  : {RESULTS_CSV}")
    print(f"  OutDir : {OUT_DIR}")
    print(f"{'#'*72}")

    try:
        results_df = pd.read_csv(RESULTS_CSV)
    except FileNotFoundError:
        print(f"  [SKIP] File not found: {RESULTS_CSV}")
        continue

    n         = min(N_SAMPLES, len(results_df))
    summaries = results_df["generated_summary"].astype(str).tolist()[:n]
    articles  = data_df["article"].astype(str).tolist()[:n]

    pre_valid = sum(1 for s in summaries if is_valid_summary(s)["valid"])
    print(f"  Samples : {n}  |  Valid : {pre_valid}")

    # ── Run N_RUNS times ─────────────────────────────────────
    all_scores = []

    for run_id in range(1, N_RUNS + 1):
        run_df   = run_once(articles, summaries, run_id, n)
        run_path = os.path.join(OUT_DIR, f"run_{run_id}.csv")
        run_df.to_csv(run_path, index=False, encoding="utf-8")
        print(f"    Saved → {run_path}")
        all_scores.append(run_df["factscore"].rename(f"run_{run_id}"))

    # ── Stability summary ─────────────────────────────────────
    runs_df = pd.concat(all_scores, axis=1)

    summary_df = pd.DataFrame({
        "mean_factscore" : runs_df.mean(axis=1),
        "std_factscore"  : runs_df.std(axis=1),
        "min_factscore"  : runs_df.min(axis=1),
        "max_factscore"  : runs_df.max(axis=1),
        "range_factscore": runs_df.max(axis=1) - runs_df.min(axis=1),
    })
    for col in runs_df.columns:
        summary_df[col] = runs_df[col]

    summary_path = os.path.join(OUT_DIR, "stability_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\n  Stability summary saved → {summary_path}")

    # ── Print statistics ──────────────────────────────────────
    valid_mask  = runs_df.notna().all(axis=1)
    valid_runs  = runs_df[valid_mask]
    overall_std = summary_df["std_factscore"].dropna()

    stats_lines = []
    stats_lines.append("=" * 72)
    stats_lines.append(f"  STABILITY RESULTS — {CURRENT_MODEL}")
    stats_lines.append(f"  AFG/AFV: {AFG_MODEL} | reasoning_effort=minimal | Runs: {N_RUNS}")
    stats_lines.append(f"  NOTE: temperature was NOT configurable for this metric.")
    stats_lines.append(f"        Variation reflects natural sampling randomness only.")
    stats_lines.append("=" * 72)
    stats_lines.append(f"\n  Per-run mean FactScore:")
    for run_col in runs_df.columns:
        m = valid_runs[run_col].mean()
        stats_lines.append(f"    {run_col}: {m:.4f}")

    stats_lines.append(f"\n  Cross-run statistics (over {len(valid_runs)} valid samples):")
    stats_lines.append(f"    Mean of per-sample std  : {overall_std.mean():.4f}")
    stats_lines.append(f"    Max  of per-sample std  : {overall_std.max():.4f}")
    stats_lines.append(f"    Mean of per-sample range: {summary_df['range_factscore'].dropna().mean():.4f}")

    stats_lines.append(f"\n  Interpretation:")
    mean_std = overall_std.mean()
    if mean_std < 0.05:
        interp = "Very stable — metric is reliable across runs"
    elif mean_std < 0.10:
        interp = "Mostly stable — minor natural variation"
    elif mean_std < 0.20:
        interp = "Moderate variation — sampling randomness is noticeable"
    else:
        interp = "High variation — results should be interpreted with caution"
    stats_lines.append(f"    {interp}")

    stats_lines.append(f"\n  Note: Main benchmark scores are unaffected by this test.")
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