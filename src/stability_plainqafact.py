"""
Stability Analysis — PlainQAFact (temperature=1.0)
===================================================
Repeats the PlainQAFact evaluation N_RUNS times for selected models
using temperature=1.0 to measure run-to-run score variation.

Models evaluated:
  - DeepSeek-V3.1
  - DeepSeek-V3.1-Thinking

Output per model:
  outputs({model})_pubmed_abstract/stability_plainqafact/
      run_1.csv  ...  run_5.csv         ← per-run plainqafact scores
      stability_summary.csv             ← mean, std, min, max per sample
      stability_stats.txt               ← overall summary printed + saved

This experiment does NOT replace main results.
It only measures metric stability under temperature=1.0.
"""

import os
import re
import time
import json
import random
import threading
import warnings
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models_config import N_SAMPLES, OUTPUT_BASE_DIR, DATA_CSV, MAX_ARTICLE_CHARS


# ============================================================
# STABILITY CONFIGURATION
# ============================================================

STABILITY_MODELS  = ["DeepSeek-V3.1", "DeepSeek-V3.1-Thinking"]

JUDGE_MODEL       = "gpt-4o-mini"
N_RUNS            = 5
TEMPERATURE       = 1.0          
N_THREADS         = 8
MAX_RETRIES       = 5
RETRY_BASE_DELAY  = 2.0
MAX_QUESTIONS_TOTAL = 20
BERTSCORE_MODEL   = "distilbert-base-uncased"
BERTSCORE_BATCH_SIZE = 32
USE_TOKEN_OVERLAP_FALLBACK = True
SAMPLE_MAX_RETRIES = 3

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

_print_lock = threading.Lock()
def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)

_api_key = os.environ.get("OPENAI_API_KEY", "")
if not _api_key:
    raise ValueError("OPENAI_API_KEY not set in environment / .env file")

client = OpenAI(api_key=_api_key)

print("=" * 72)
print("  STABILITY ANALYSIS — PlainQAFact")
print("=" * 72)
print(f"  Judge model  : {JUDGE_MODEL}")
print(f"  Temperature  : {TEMPERATURE}  (main eval used 0.0)")
print(f"  Runs         : {N_RUNS}")
print(f"  Models       : {STABILITY_MODELS}")
print(f"  Purpose      : measure run-to-run score variation only")
print("=" * 72)


# ============================================================
# HELPERS
# ============================================================

def safe_text(x) -> str:
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    return str(x).strip()


def count_sentences(summary: str) -> int:
    summary = safe_text(summary)
    return len([
        s for s in re.split(r"(?<=[.!?])\s+", summary)
        if len(s.strip()) > 20
    ])


def call_gpt(system: str, user: str, max_tokens: int = 500) -> str:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=TEMPERATURE,   
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                sleep_time = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                sleep_time += random.uniform(0, 1.5)
                time.sleep(sleep_time)
            else:
                tprint(f"\n[GPT ERROR] Failed after {MAX_RETRIES} attempts: {e}")
                return ""
    tprint(f"\n[GPT ERROR] {last_error}")
    return ""


# ============================================================
# CALL 1 — QUESTION GENERATION
# ============================================================

QG_SYSTEM = """You are a biomedical question generator.

Given a full medical summary, generate factual questions that can be
answered from the summary and verified against the source article.

Rules:
- Cover facts from across the ENTIRE summary, not just the first sentence.
- Focus on specific facts: numbers, findings, treatments, diagnoses,
  outcomes, study groups, diseases, and measurements.
- Each question must target only one fact.
- Each question must be answerable with a short factual answer (1-10 words).
- Avoid vague questions such as "What did the study show?"
- Avoid questions that require explanation or interpretation.
- Output ONLY questions, one per line.
- No numbering. No bullets.
"""


def split_sentences(summary: str) -> list:
    summary = safe_text(summary)
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", summary)
        if len(s.strip()) > 20
    ]


def generate_questions(summary: str) -> tuple:
    summary = safe_text(summary)
    if not summary:
        return [], []

    sentences = split_sentences(summary)
    if not sentences:
        return [], []

    numbered = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sentences))

    user = f"""SUMMARY SENTENCES:
{numbered}

Generate up to {MAX_QUESTIONS_TOTAL} factual questions that cover the full summary.
For each question, prefix it with the sentence number it came from, like:
[1] What treatment was evaluated?
[3] How many patients were enrolled?

Rules:
- Each question must be answerable from its tagged sentence alone.
- Focus on specific facts: treatments, diagnoses, outcomes, study groups,
  diseases, findings, and measurements.
- Each question must be answerable with a short factual answer (1-10 words).
- Avoid vague questions such as "What did the study show?"
- No bullets.
"""

    raw = call_gpt(QG_SYSTEM, user, max_tokens=900)

    questions        = []
    question_sources = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\[(\d+)\]\s*", line)
        if match:
            sent_idx = int(match.group(1)) - 1
            q = line[match.end():].strip()
            q = re.sub(r"^\s*[-*•\d.)]+\s*", "", q).strip()
            if len(q) > 10 and q.endswith("?"):
                sent_idx = max(0, min(sent_idx, len(sentences) - 1))
                questions.append(q)
                question_sources.append(sentences[sent_idx])
        else:
            q = re.sub(r"^\s*[-*•\d.)]+\s*", "", line).strip()
            if len(q) > 10 and q.endswith("?"):
                questions.append(q)
                question_sources.append(summary)

    return questions[:MAX_QUESTIONS_TOTAL], question_sources[:MAX_QUESTIONS_TOTAL]


# ============================================================
# CALLS 2 & 3 — BATCH QA
# ============================================================

QA_BATCH_SYSTEM = """You are a biomedical question answering system.

Answer each question using ONLY information from the provided text.

Rules:
- If an answer is not explicitly in the text, respond with exactly: unanswerable
- Give SHORT answers, preferably 1-10 words.
- Do not explain.
- Do not add information from outside the text.
- Output ONLY valid JSON as a list of strings.
- The number of answers must equal the number of questions.

Example output:
["answer one", "unanswerable", "answer three"]
"""


def extract_json_list(raw: str, expected_len: int) -> list:
    raw = safe_text(raw)
    if not raw:
        return ["unanswerable"] * expected_len
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(raw)
    except Exception:
        match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        if not match:
            return ["unanswerable"] * expected_len
        try:
            obj = json.loads(match.group(0))
        except Exception:
            return ["unanswerable"] * expected_len
    if not isinstance(obj, list):
        return ["unanswerable"] * expected_len
    answers = []
    for ans in obj[:expected_len]:
        ans = safe_text(ans).lower()
        if not ans:
            ans = "unanswerable"
        if "unanswerable" in ans:
            ans = "unanswerable"
        ans = ans.split("\n")[0].strip()
        ans = re.sub(r"^answer:\s*", "", ans).strip()
        answers.append(ans if ans else "unanswerable")
    while len(answers) < expected_len:
        answers.append("unanswerable")
    return answers


def answer_questions_batch(context: str, questions: list, is_article: bool = False) -> list:
    context = safe_text(context)
    if not context or not questions:
        return ["unanswerable"] * len(questions)
    if is_article:
        context = context[:MAX_ARTICLE_CHARS]
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    user = f"""TEXT:
{context}

QUESTIONS:
{questions_text}
"""
    max_tokens = max(300, 80 * len(questions))
    raw = call_gpt(QA_BATCH_SYSTEM, user, max_tokens=max_tokens)
    return extract_json_list(raw, expected_len=len(questions))


# ============================================================
# BERTSCORE
# ============================================================

def token_overlap_score(pred: str, ref: str) -> float:
    pred_tokens = set(pred.lower().split())
    ref_tokens  = set(ref.lower().split())
    if not ref_tokens:
        return 0.0
    return len(pred_tokens & ref_tokens) / len(ref_tokens)


def compare_answers_bertscore(pred_answers: list, ref_answers: list):
    if not pred_answers or not ref_answers:
        return None

    valid_pairs = []
    zero_scores = []

    for pred, ref in zip(pred_answers, ref_answers):
        pred = pred if isinstance(pred, str) else str(pred)
        ref  = ref  if isinstance(ref,  str) else str(ref)
        pred = pred.strip().lower()
        ref  = ref.strip().lower()

        if ref == "unanswerable":
            continue
        if pred == "unanswerable":
            zero_scores.append(0.0)
            continue
        valid_pairs.append((pred, ref))

    if not valid_pairs and not zero_scores:
        return None

    bert_scores = []
    if valid_pairs:
        p_list = [p for p, r in valid_pairs]
        r_list = [r for p, r in valid_pairs]
        try:
            from bert_score import score as bert_score_fn
            _, _, F1 = bert_score_fn(
                p_list, r_list,
                model_type=BERTSCORE_MODEL,
                lang="en", verbose=False,
                batch_size=BERTSCORE_BATCH_SIZE,
            )
            bert_scores = [float(x) for x in F1.tolist()]
        except Exception as e:
            tprint(f"\n[BERTScore ERROR] {e}")
            if USE_TOKEN_OVERLAP_FALLBACK:
                tprint("[BERTScore] Using token-overlap fallback.")
                bert_scores = [token_overlap_score(p, r) for p, r in valid_pairs]
            else:
                bert_scores = [0.0] * len(valid_pairs)

    all_scores = bert_scores + zero_scores
    if not all_scores:
        return None
    return round(sum(all_scores) / len(all_scores), 4)


# ============================================================
# PIPELINE
# ============================================================

def _run_pipeline(article: str, summary: str):
    questions, question_sources = generate_questions(summary)
    if not questions:
        return None

    ref_answers  = answer_questions_batch(summary, questions, is_article=False)
    pred_answers = answer_questions_batch(article, questions, is_article=True)

    supported = sum(
        1 for p, r in zip(pred_answers, ref_answers)
        if r != "unanswerable" and p != "unanswerable"
    )

    score = compare_answers_bertscore(pred_answers, ref_answers)
    if score is None:
        return None

    return {
        "n_questions"          : len(questions),
        "n_supported_questions": supported,
        "plainqafact_api_score": score,
    }


def evaluate_sample(idx: int, article: str, summary: str) -> dict:
    article = safe_text(article)
    summary = safe_text(summary)

    empty = {
        "idx"                  : idx,
        "plainqafact_api_score": None,
        "n_questions"          : 0,
        "n_supported_questions": 0,
    }

    if not summary or summary.startswith("ERROR"):
        return empty

    for attempt in range(1, SAMPLE_MAX_RETRIES + 1):
        if attempt > 1:
            time.sleep(1.0)
        result = _run_pipeline(article, summary)
        if result is not None:
            return {
                "idx"                  : idx,
                "plainqafact_api_score": result["plainqafact_api_score"],
                "n_questions"          : result["n_questions"],
                "n_supported_questions": result["n_supported_questions"],
            }

    return empty


# ============================================================
# ONE FULL RUN
# ============================================================

def run_once(articles, summaries, run_id, n):
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
            result                 = future.result()
            results[result["idx"]] = result
            completed             += 1
            if result["plainqafact_api_score"] is None:
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
    OUT_DIR     = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/stability_plainqafact"
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
    summaries = results_df["generated_summary"].apply(safe_text).tolist()[:n]
    articles  = data_df["article"].apply(safe_text).tolist()[:n]

    print(f"  Samples : {n}")

    # ── Run N_RUNS times ─────────────────────────────────────
    all_scores = []

    for run_id in range(1, N_RUNS + 1):
        run_df   = run_once(articles, summaries, run_id, n)
        run_path = os.path.join(OUT_DIR, f"run_{run_id}.csv")
        run_df.to_csv(run_path, index=False, encoding="utf-8")
        print(f"    Saved → {run_path}")
        all_scores.append(run_df["plainqafact_api_score"].rename(f"run_{run_id}"))

    # ── Stability summary ─────────────────────────────────────
    runs_df = pd.concat(all_scores, axis=1)

    summary_df = pd.DataFrame({
        "mean_plainqafact" : runs_df.mean(axis=1),
        "std_plainqafact"  : runs_df.std(axis=1),
        "min_plainqafact"  : runs_df.min(axis=1),
        "max_plainqafact"  : runs_df.max(axis=1),
        "range_plainqafact": runs_df.max(axis=1) - runs_df.min(axis=1),
    })
    for col in runs_df.columns:
        summary_df[col] = runs_df[col]

    summary_path = os.path.join(OUT_DIR, "stability_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\n  Stability summary saved → {summary_path}")

    # ── Print statistics ──────────────────────────────────────
    valid_mask  = runs_df.notna().all(axis=1)
    valid_runs  = runs_df[valid_mask]
    overall_std = summary_df["std_plainqafact"].dropna()

    stats_lines = []
    stats_lines.append("=" * 72)
    stats_lines.append(f"  STABILITY RESULTS — {CURRENT_MODEL}")
    stats_lines.append(f"  Judge: {JUDGE_MODEL} | Temperature: {TEMPERATURE} | Runs: {N_RUNS}")
    stats_lines.append("=" * 72)
    stats_lines.append(f"\n  Per-run mean PlainQAFact score:")
    for run_col in runs_df.columns:
        m = valid_runs[run_col].mean()
        stats_lines.append(f"    {run_col}: {m:.4f}")

    stats_lines.append(f"\n  Cross-run statistics (over {len(valid_runs)} valid samples):")
    stats_lines.append(f"    Mean of per-sample std  : {overall_std.mean():.4f}")
    stats_lines.append(f"    Max  of per-sample std  : {overall_std.max():.4f}")
    stats_lines.append(f"    Mean of per-sample range: {summary_df['range_plainqafact'].dropna().mean():.4f}")

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