"""
PlainQAFact-API Evaluation — Whole-Summary 3-Call Version
==========================================================

Pipeline (3 GPT calls per sample, regardless of summary length):
  Call 1 — QG  : Generate ALL questions from the WHOLE summary
  Call 2 — QA  : Answer ALL questions from the WHOLE summary
  Call 3 — QA  : Answer ALL questions from the full source article
  Step 4 — BERTScore compares answers → final score

Improvement over previous version:
  OLD: 3 × N_sentences GPT calls per sample  (e.g. 15 calls for 5 sentences)
  NEW: exactly 3 GPT calls per sample        (flat, regardless of length)

Install:
    pip install openai python-dotenv pandas bert-score

Inputs:
  data/processed/pubmed_train_clean_tokens(1000).csv → article column
  outputs(Model)_pubmed_abstract/results1000.csv     → generated_summary column

Output:
  outputs(Model)_pubmed_abstract/plainqafact_api_results1000s.csv
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

from models_config import (
    N_SAMPLES,
    OUTPUT_BASE_DIR,
    DATA_CSV,
    ACTIVE_MODEL,
    MAX_ARTICLE_CHARS,
)


# ============================================================
# CONFIGURATION
# ============================================================

JUDGE_MODEL = "gpt-4o-mini"

# Do not use 32 unless your API limit is high.
# 4–8 is safer for long evaluations.
N_THREADS = 8

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0

# Max questions generated from the whole summary.
# Since we cover the entire summary at once, this should be larger
# than the old per-sentence MAX_QUESTIONS.
MAX_QUESTIONS_TOTAL = 20

# BERTScore model.
BERTSCORE_MODEL = "distilbert-base-uncased"
BERTSCORE_BATCH_SIZE = 32

USE_TOKEN_OVERLAP_FALLBACK = True

_print_lock = threading.Lock()

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="transformers",
)


def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


_api_key = os.environ.get("OPENAI_API_KEY", "")
if not _api_key:
    raise ValueError("OPENAI_API_KEY not set in environment / .env file")

client = OpenAI(api_key=_api_key)

print("OpenAI client ready.")
print(f"  Model             : {JUDGE_MODEL}")
print(f"  Pipeline          : QG (whole summary) → Batch QA → BERTScore")
print(f"  GPT calls/sample  : 3 (flat)")
print(f"  Threads           : {N_THREADS}")
print(f"  Max article chars : {MAX_ARTICLE_CHARS}")
print(f"  Max questions     : {MAX_QUESTIONS_TOTAL} per sample")


# ============================================================
# HELPER: Text cleaning
# ============================================================

def safe_text(x) -> str:
    """
    Convert any value to a clean string.
    Handles scalars, numpy arrays, and PyTorch tensors safely.
    pd.isna() raises ValueError when passed an array with >1 element,
    so we catch that and fall through to str().
    """
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        # x is an array/tensor — not a scalar NA, just convert it
        pass
    return str(x).strip()


def count_sentences(summary: str) -> int:
    """Count sentences in a summary (used only for reporting)."""
    summary = safe_text(summary)
    return len([
        s for s in re.split(r"(?<=[.!?])\s+", summary)
        if len(s.strip()) > 20
    ])


# ============================================================
# HELPER: GPT call with retry
# ============================================================

def call_gpt(system: str, user: str, max_tokens: int = 500) -> str:
    """
    Call GPT with exponential backoff and jitter.
    """
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
                temperature=0.0,
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
# CALL 1 — QUESTION GENERATION (whole summary)
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
    """Split summary into individual sentences."""
    summary = safe_text(summary)
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", summary)
        if len(s.strip()) > 20
    ]


def generate_questions(summary: str) -> tuple:
    """
    Generate up to MAX_QUESTIONS_TOTAL factual questions from the whole
    summary in a single GPT call.

    Returns:
        questions      : flat list of question strings
        question_sources: parallel list of the sentence each question came from
    """
    summary = safe_text(summary)
    if not summary:
        return [], []

    sentences = split_sentences(summary)
    if not sentences:
        return [], []

    # Build numbered sentence list so GPT can tag each question
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

    questions = []
    question_sources = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        # Extract sentence index prefix like [2]
        match = re.match(r"^\[(\d+)\]\s*", line)
        if match:
            sent_idx = int(match.group(1)) - 1  # convert to 0-based
            q = line[match.end():].strip()
            q = re.sub(r"^\s*[-*•\d.)]+\s*", "", q).strip()
            if len(q) > 10 and q.endswith("?"):
                # Clamp to valid sentence index
                sent_idx = max(0, min(sent_idx, len(sentences) - 1))
                questions.append(q)
                question_sources.append(sentences[sent_idx])
        else:
            # No prefix — fall back to whole summary as source
            q = re.sub(r"^\s*[-*•\d.)]+\s*", "", line).strip()
            if len(q) > 10 and q.endswith("?"):
                questions.append(q)
                question_sources.append(summary)

    return questions[:MAX_QUESTIONS_TOTAL], question_sources[:MAX_QUESTIONS_TOTAL]


# ============================================================
# CALLS 2 & 3 — BATCH QUESTION ANSWERING
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
    """
    Robustly extract a JSON list from model output.
    Falls back to unanswerable if parsing fails.
    """
    raw = safe_text(raw)
    if not raw:
        return ["unanswerable"] * expected_len

    # Strip markdown fences if the model accidentally adds them.
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
        # Keep only the first line if model adds extra text.
        ans = ans.split("\n")[0].strip()
        ans = re.sub(r"^answer:\s*", "", ans).strip()
        answers.append(ans if ans else "unanswerable")

    # Pad to expected length if model returned fewer answers.
    while len(answers) < expected_len:
        answers.append("unanswerable")

    return answers


def answer_questions_batch(context: str, questions: list, is_article: bool = False) -> list:
    """
    Answer all questions from a single context in one GPT call.

    For summary (is_article=False): uses the full summary text.
    For article (is_article=True):  uses article up to MAX_ARTICLE_CHARS.
    """
    context = safe_text(context)
    if not context or not questions:
        return ["unanswerable"] * len(questions)

    if is_article:
        context = context[:MAX_ARTICLE_CHARS]

    questions_text = "\n".join(
        f"{i + 1}. {q}" for i, q in enumerate(questions)
    )

    user = f"""TEXT:
{context}

QUESTIONS:
{questions_text}
"""

    # Scale tokens with the number of questions; raise the floor for large batches.
    max_tokens = max(300, 80 * len(questions))

    raw = call_gpt(QA_BATCH_SYSTEM, user, max_tokens=max_tokens)
    return extract_json_list(raw, expected_len=len(questions))


# ============================================================
# BERTSCORE COMPARISON
# ============================================================

def token_overlap_score(pred: str, ref: str) -> float:
    """Simple token-overlap fallback if BERTScore is unavailable."""
    pred_tokens = set(pred.lower().split())
    ref_tokens  = set(ref.lower().split())
    if not ref_tokens:
        return 0.0
    return len(pred_tokens & ref_tokens) / len(ref_tokens)


def compare_answers_bertscore(pred_answers: list, ref_answers: list):
    """
    Compare article answers (pred) vs summary answers (ref).

    Returns a float score, or None if the QA step produced no usable pairs
    (meaning the summary QA call failed entirely — not a real zero score).

    Scoring logic:
    - ref == unanswerable  → skip (bad question; summary couldn't answer it)
    - ref has answer, pred == unanswerable → score 0 (article hallucination)
    - both have answers    → BERTScore F1
    """
    if not pred_answers or not ref_answers:
        return None

    valid_pairs = []
    zero_scores = []

    for pred, ref in zip(pred_answers, ref_answers):
        # Force to plain Python str before any comparison —
        # guards against numpy arrays or tensors leaking from BERTScore.
        pred = pred if isinstance(pred, str) else str(pred)
        ref  = ref  if isinstance(ref,  str) else str(ref)
        pred = pred.strip().lower()
        ref  = ref.strip().lower()

        if ref == "unanswerable":
            # Question not answerable from summary → skip entirely.
            continue

        if pred == "unanswerable":
            # Summary has an answer but article does not → penalise.
            zero_scores.append(0.0)
            continue

        valid_pairs.append((pred, ref))

    if not valid_pairs and not zero_scores:
        # All summary answers were "unanswerable" — QA call failed entirely.
        # Return None so the sample is skipped rather than penalised with 0.
        return None

    bert_scores = []

    if valid_pairs:
        p_list = [p for p, r in valid_pairs]
        r_list = [r for p, r in valid_pairs]

        try:
            from bert_score import score as bert_score_fn

            _, _, F1 = bert_score_fn(
                p_list,
                r_list,
                model_type=BERTSCORE_MODEL,
                lang="en",
                verbose=False,
                batch_size=BERTSCORE_BATCH_SIZE,
            )
            bert_scores = [float(x) for x in F1.tolist()]

        except Exception as e:
            tprint(f"\n[BERTScore ERROR] {e}")
            if USE_TOKEN_OVERLAP_FALLBACK:
                tprint("[BERTScore] Using token-overlap fallback for this batch.")
                bert_scores = [
                    token_overlap_score(pred, ref)
                    for pred, ref in valid_pairs
                ]
            else:
                bert_scores = [0.0] * len(valid_pairs)

    all_scores = bert_scores + zero_scores
    if not all_scores:
        # No scoreable pairs survived — treat as skipped, not a real zero.
        return None

    return round(sum(all_scores) / len(all_scores), 4)


# ============================================================

# Number of times to retry the full 3-call pipeline if it returns None.
SAMPLE_MAX_RETRIES = 3


def _run_pipeline(article: str, summary: str):
    """
    Run one attempt of the 3-call pipeline.
    Returns a result dict on success, or None if the pipeline failed
    (empty questions or all summary QA answers were unanswerable).
    """
    # CALL 1: generate all questions from the whole summary
    questions, question_sources = generate_questions(summary)
    if not questions:
        tprint("  [PIPELINE] No questions generated — will retry if attempts remain.")
        return None

    # CALL 2: answer all questions using the whole summary
    ref_answers = answer_questions_batch(
        context=summary,
        questions=questions,
        is_article=False,
    )

    # CALL 3: answer all questions using the source article
    pred_answers = answer_questions_batch(
        context=article,
        questions=questions,
        is_article=True,
    )

    supported = sum(
        1
        for p, r in zip(pred_answers, ref_answers)
        if r != "unanswerable" and p != "unanswerable"
    )

    score = compare_answers_bertscore(pred_answers, ref_answers)

    if score is None:
        tprint("  [PIPELINE] All summary QA answers unanswerable — will retry if attempts remain.")
        return None

    return {
        "n_questions": len(questions),
        "n_supported_questions": supported,
        "plainqafact_api_score": score,
    }


def evaluate_sample(idx: int, article: str, summary: str) -> dict:
    """
    Run the 3-call pipeline on one sample with up to SAMPLE_MAX_RETRIES
    retries if question generation or summary QA fails.

      Call 1 -> generate all questions from the whole summary
      Call 2 -> answer all questions from the whole summary
      Call 3 -> answer all questions from the source article
    """
    article = safe_text(article)
    summary = safe_text(summary)

    n_sentences = count_sentences(summary)

    empty_result = {
        "idx": idx,
        "plainqafact_api_score": None,
        "n_sentences": n_sentences,
        "n_questions": 0,
        "n_supported_questions": 0,
    }

    if not summary or summary.startswith("ERROR"):
        return empty_result

    # Retry the full pipeline up to SAMPLE_MAX_RETRIES times.
    for attempt in range(1, SAMPLE_MAX_RETRIES + 1):
        if attempt > 1:
            tprint(f"  [RETRY] idx={idx} attempt {attempt}/{SAMPLE_MAX_RETRIES}")
            time.sleep(1.0)  # brief pause before retry

        result = _run_pipeline(article, summary)

        if result is not None:
            return {
                "idx": idx,
                "plainqafact_api_score": result["plainqafact_api_score"],
                "n_sentences": n_sentences,
                "n_questions": result["n_questions"],
                "n_supported_questions": result["n_supported_questions"],
            }

    # All attempts exhausted — record as skipped.
    tprint(f"  [SKIP] idx={idx} failed after {SAMPLE_MAX_RETRIES} attempts.")
    return empty_result

# ============================================================
# MAIN LOOP
# ============================================================

print(f"\nLoading {DATA_CSV} ...")
data_df = pd.read_csv(DATA_CSV)
print(f"  {len(data_df)} rows | columns: {data_df.columns.tolist()}")

if "article" not in data_df.columns:
    raise ValueError("DATA_CSV must contain an 'article' column.")

active_models = ACTIVE_MODEL if isinstance(ACTIVE_MODEL, list) else [ACTIVE_MODEL]

for CURRENT_MODEL in active_models:

    RESULTS_CSV = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/results1000.csv"
    OUT_CSV     = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/plainqafact_api_results1000ss.csv"

    print(f"\n\n{'#' * 72}")
    print(f"  Processing model  : {CURRENT_MODEL}")
    print(f"  Judge model       : {JUDGE_MODEL}")
    print(f"  Pipeline          : QG (whole summary) → Batch QA → BERTScore")
    print(f"  GPT calls/sample  : 3 (flat)")
    print(f"  Max article chars : {MAX_ARTICLE_CHARS}")
    print(f"  Max questions     : {MAX_QUESTIONS_TOTAL} per sample")
    print(f"  Threads           : {N_THREADS}")
    print(f"  Input             : {RESULTS_CSV}")
    print(f"  Output            : {OUT_CSV}")
    print(f"{'#' * 72}")

    print("\nLoading results.csv ...")
    try:
        results_df = pd.read_csv(RESULTS_CSV)
    except FileNotFoundError:
        print(f"  [SKIP] File not found: {RESULTS_CSV}")
        continue

    print(f"  {len(results_df)} rows | columns: {results_df.columns.tolist()}")

    if "generated_summary" not in results_df.columns:
        raise ValueError("results1000.csv must contain a 'generated_summary' column.")

    n = min(N_SAMPLES, len(results_df), len(data_df))

    summaries = results_df["generated_summary"].apply(safe_text).tolist()[:n]
    articles  = data_df["article"].apply(safe_text).tolist()[:n]

    results   = [None] * n
    completed = 0
    errors    = 0
    total_start = time.time()

    print(f"\nEvaluating {n} summaries (3 GPT calls each)...\n")

    try:
        with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            futures = {
                executor.submit(evaluate_sample, i, articles[i], summaries[i]): i
                for i in range(n)
            }

            for future in as_completed(futures):
                i = futures[future]

                try:
                    result = future.result()
                except Exception as e:
                    tprint(f"\n[SAMPLE ERROR] idx={i}: {e}")
                    result = {
                        "idx": i,
                        "plainqafact_api_score": None,
                        "n_sentences": 0,
                        "n_questions": 0,
                        "n_supported_questions": 0,
                    }

                results[result["idx"]] = result
                completed += 1

                if result["plainqafact_api_score"] is None:
                    errors += 1

                percent = (completed / n) * 100
                filled  = int(50 * completed / n)
                bar     = "█" * filled + "░" * (50 - filled)
                elapsed = time.time() - total_start
                rate    = completed / elapsed if elapsed > 0 else 0
                eta     = (n - completed) / rate if rate > 0 else 0

                print(
                    f"\r  Progress: [{bar}] {completed}/{n} ({percent:.1f}%) | "
                    f"Rate: {rate:.2f}/s | ETA: {eta:.0f}s | Errors: {errors}",
                    end="",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Saving partial results...")

    # Fill any samples that did not complete (interrupted / crashed).
    for i in range(n):
        if results[i] is None:
            results[i] = {
                "idx": i,
                "plainqafact_api_score": None,
                "n_sentences": 0,
                "n_questions": 0,
                "n_supported_questions": 0,
            }

    print()

    rows      = [{k: v for k, v in r.items() if k != "idx"} for r in results]
    scores_df = pd.DataFrame(rows)

    valid_scores  = scores_df["plainqafact_api_score"].dropna()
    total_elapsed = time.time() - total_start

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\n" + "=" * 72)
    print(f"  PLAINQAFACT-API SUMMARY — {CURRENT_MODEL}")
    print("=" * 72)
    print(f"  Total samples      : {n}")
    print(f"  Valid evaluations  : {len(valid_scores)}")
    print(f"  Errors/Skipped     : {errors}")
    print(f"  Total GPT calls    : {n * 3} (3 × {n})")

    if len(valid_scores) > 0:
        print("\n  Score Distribution:")

        bins = [
            (0.0,  0.2),
            (0.2,  0.4),
            (0.4,  0.6),
            (0.6,  0.8),
            (0.8,  1.01),
        ]
        labels = [
            "0.0-0.2 Very low",
            "0.2-0.4 Low",
            "0.4-0.6 Moderate",
            "0.6-0.8 Good",
            "0.8-1.0 Excellent",
        ]

        for (lo, hi), label in zip(bins, labels):
            count = int(((valid_scores >= lo) & (valid_scores < hi)).sum())
            pct   = (count / len(valid_scores)) * 100
            bar   = "█" * int(pct / 2)
            print(f"    {label:<25} : {count:3d} ({pct:5.1f}%) {bar}")

        print(f"\n  Mean Score  : {valid_scores.mean():.4f}")
        print(f"  Median      : {valid_scores.median():.4f}")
        print(f"  Std         : {valid_scores.std():.4f}")
        print(f"  Min         : {valid_scores.min():.4f}")
        print(f"  Max         : {valid_scores.max():.4f}")

    if total_elapsed > 0:
        print(f"\n  Total time : {total_elapsed / 60:.1f} min ({total_elapsed:.1f}s)")
        print(f"  Rate       : {len(valid_scores) / total_elapsed:.2f} samples/sec")

    # ============================================================
    # DETAILED TABLE
    # ============================================================

    print("\n" + "=" * 72)
    print("  DETAILED RESULTS — first 20")
    print("=" * 72)
    print(f"{'#':<5} {'Score':<10} {'Sentences':<12} {'Questions':<12} {'Supported':<12} {'Summary'}")
    print("-" * 90)

    for i, row in scores_df.head(20).iterrows():
        score = (
            f"{row['plainqafact_api_score']:.4f}"
            if pd.notna(row["plainqafact_api_score"])
            else "—"
        )
        nsent = str(int(row["n_sentences"]))         if pd.notna(row["n_sentences"])          else "—"
        nq    = str(int(row["n_questions"]))          if pd.notna(row["n_questions"])           else "—"
        nsup  = str(int(row["n_supported_questions"])) if pd.notna(row["n_supported_questions"]) else "—"
        summ  = summaries[i][:30] + "..." if len(summaries[i]) > 30 else summaries[i]

        print(f"{i + 1:<5} {score:<10} {nsent:<12} {nq:<12} {nsup:<12} {summ}")

    if len(scores_df) > 20:
        print(f"\n  ... and {len(scores_df) - 20} more — see CSV")

    # ============================================================
    # POST-PROCESS: remove suspected QA failures
    # Samples where n_supported=0 and n_questions=MAX are QA failures
    # (article QA too strict on technical content), not real hallucinations.
    # ============================================================

    qa_failure_mask = (
        (scores_df["plainqafact_api_score"] == 0.0) &
        (scores_df["n_supported_questions"] == 0) &
        (scores_df["n_questions"] == MAX_QUESTIONS_TOTAL)
    )
    n_corrected = int(qa_failure_mask.sum())
    if n_corrected > 0:
        scores_df.loc[qa_failure_mask, "plainqafact_api_score"] = None
        tprint(f"  [POST-PROCESS] {n_corrected} suspected QA failures set to None.")

    # Recompute valid_scores after post-processing
    valid_scores = scores_df["plainqafact_api_score"].dropna()

    # ============================================================
    # SAVE
    # ============================================================

    out_df = pd.concat(
        [results_df.iloc[:n].reset_index(drop=True), scores_df],
        axis=1,
    )
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Saved → {OUT_CSV}")

print("\n\n🎉 All models processed.")