"""
FactScore — Hallucination Detection (valid-samples only)
================================================================
Two-step pipeline:

  Step 1 — AFG: gpt-5-nano breaks each generated summary into atomic facts.
  Step 2 — AFV: gpt-5-nano validates each fact directly against the original
               article text via OpenAI API — parallel threads, fast & cheap.

Invalid summaries (ERROR, missing sections, too long) are SKIPPED:
  - They keep their row in the output CSV for traceability.
  - Their factscore is NaN, so they are excluded from the mean.

Install once before running:
    pip install openai python-dotenv

Inputs:
  data/processed/pubmed_train_clean.csv         → article column
  outputs(Model)_pubmed_abstract/results1000.csv → generated_summary column

Output:
  outputs(Model)_pubmed_abstract/results_with_factscore100s.csv
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
from models_config import N_SAMPLES, OUTPUT_BASE_DIR, DATA_CSV, ACTIVE_MODEL

# ── NLTK ─────────────────────────────────────────────────────
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


# ============================================================
# VALIDATION  (same rules as plot_invalid.py)
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
# CONFIGURATION
# ============================================================

AFG_MODEL        = "gpt-5-nano"
AFV_MODEL        = "gpt-5-nano"
N_THREADS        = 16
N_SAMPLE_THREADS = 4
MAX_RETRIES      = 3
RETRY_DELAY      = 2.0
AFG_MAX_TOKENS   = 2000
AFV_MAX_TOKENS   = 50
MIN_FACTS        = 3

# ── Thread-safe print ────────────────────────────────────────
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
print(f"OpenAI client ready.")
print(f"  AFG model: {AFG_MODEL}  [reasoning_effort=minimal]")
print(f"  AFV model: {AFV_MODEL}  [reasoning_effort=minimal]")


# ── Retry wrapper ────────────────────────────────────────────
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
# STEP 1 — AFG: atomic fact generation
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
    lines = raw.splitlines()
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
    tprint(f"    [AFG] Raw response ({len(raw)} chars): {raw[:100]!r}")

    if not raw:
        tprint("    [AFG] WARNING: empty response — skipping sample")
        return []

    facts = _parse_facts(raw)
    tprint(f"    [AFG] Extracted {len(facts)} atomic facts")

    if len(facts) < MIN_FACTS:
        tprint(f"    [AFG] Too few facts ({len(facts)} < {MIN_FACTS}) — retrying")
        user_retry = ATOMIZE_USER_RETRY_TMPL.format(summary=summary.strip())
        raw_retry  = _call_with_retry(AFG_MODEL, ATOMIZE_SYSTEM_RETRY,
                                      user_retry, max_tokens=AFG_MAX_TOKENS)
        tprint(f"    [AFG] Retry response ({len(raw_retry)} chars): {raw_retry[:100]!r}")

        if raw_retry:
            facts_retry = _parse_facts(raw_retry)
            tprint(f"    [AFG] Retry extracted {len(facts_retry)} atomic facts")
            if len(facts_retry) > len(facts):
                facts = facts_retry
                tprint(f"    [AFG] Using retry result ({len(facts)} facts)")
            else:
                tprint(f"    [AFG] Keeping original result ({len(facts)} facts)")

    return facts


# ============================================================
# STEP 2 — AFV: fact verification (parallel)
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
        tprint("    [pipeline] No atomic facts extracted — skipping AFV")
        return {
            "score": None, "supported": 0,
            "not_verifiable": 0, "total_facts": 0, "facts_detail": []
        }

    labels = [None] * len(all_facts)

    def _verify(args):
        idx, fact = args
        try:
            return idx, validate_fact_api(article, fact)
        except Exception as e:
            tprint(f"    [verify] fact {idx+1} error: {e}")
            return idx, False

    with ThreadPoolExecutor(max_workers=min(N_THREADS, len(all_facts))) as ex:
        for idx, is_supported in ex.map(_verify, enumerate(all_facts)):
            labels[idx] = is_supported

    supported      = 0
    not_verifiable = 0
    facts_detail   = []

    for fact, is_supported in zip(all_facts, labels):
        if is_supported:
            supported += 1
            facts_detail.append({"fact": fact, "label": "ENTAILMENT"})
        else:
            not_verifiable += 1
            facts_detail.append({"fact": fact, "label": "NEUTRAL"})

    total = len(all_facts)
    return {
        "score"         : round(supported / total, 4) if total > 0 else None,
        "supported"     : supported,
        "not_verifiable": not_verifiable,
        "total_facts"   : total,
        "facts_detail"  : facts_detail,
    }


# ============================================================
# LOAD ARTICLE DATA
# ============================================================

print(f"\nLoading {DATA_CSV} ...")
data_df = pd.read_csv(DATA_CSV)
print(f"  {len(data_df)} rows | columns: {data_df.columns.tolist()}")

active_models = ACTIVE_MODEL if isinstance(ACTIVE_MODEL, list) else [ACTIVE_MODEL]


# ============================================================
# PER-SAMPLE WORKER
# ============================================================

def _process_one_sample(idx: int, article: str, summary: str, n: int) -> dict:
    tprint(f"\n{'='*55}  Sample {idx+1}/{n}  {'='*55}")

    # ── Validity check — skip invalid summaries entirely ─────
    validity = is_valid_summary(summary)
    if not validity["valid"]:
        tprint(f"  [Sample {idx+1}] SKIPPED — invalid ({validity['reason']})")
        return {
            "idx":            idx,
            "factscore":      None,          # NaN → excluded from mean
            "supported":      None,
            "not_verifiable": None,
            "total_facts":    None,
            "invalid_reason": validity["reason"],
        }

    result = factscore_pipeline(article, summary)

    tprint(f"  [Sample {idx+1}] FactScore      : {result['score']}")
    tprint(f"  [Sample {idx+1}] Supported      : {result['supported']} / {result['total_facts']}")
    tprint(f"  [Sample {idx+1}] Not verifiable : {result['not_verifiable']} / {result['total_facts']}")

    lines = []
    for item in result["facts_detail"]:
        icon = "✓" if item["label"] == "ENTAILMENT" else "?"
        lines.append(f"    {icon} [{item['label']:<13}] {item['fact'][:90]}")
    if lines:
        tprint("\n".join(lines))

    return {
        "idx":            idx,
        "factscore":      result["score"],
        "supported":      result["supported"],
        "not_verifiable": result["not_verifiable"],
        "total_facts":    result["total_facts"],
        "invalid_reason": None,
    }


# ============================================================
# MAIN LOOP
# ============================================================

for CURRENT_MODEL in active_models:

    RESULTS_CSV = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/results1000.csv"
    OUT_CSV     = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/results_with_factscore1000s.csv"

    print(f"\n\n{'#'*72}")
    print(f"  Processing model : {CURRENT_MODEL}")
    print(f"  AFG              : {AFG_MODEL}  [reasoning_effort=minimal]")
    print(f"  AFV              : {AFV_MODEL}  [reasoning_effort=minimal, {N_THREADS} threads]")
    print(f"  Sample threads   : {N_SAMPLE_THREADS}")
    print(f"  Min facts        : {MIN_FACTS}  [retry if below threshold]")
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
    assert len(results_df) <= len(data_df), \
        "results.csv has more rows than pubmed_train_clean.csv"

    n         = min(N_SAMPLES, len(results_df))
    summaries = results_df["generated_summary"].astype(str).tolist()[:n]
    articles  = data_df["article"].astype(str).tolist()[:n]

    # ── Pre-count valid samples for logging ──────────────────
    pre_valid = sum(1 for s in summaries if is_valid_summary(s)["valid"])
    print(f"\n  Validity check: {pre_valid}/{n} valid summaries — only these will be scored.")

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

    rows = [{k: v for k, v in r.items() if k != "idx"} for r in results]
    total_elapsed = time.time() - total_start

    # ── Summary table ────────────────────────────────────────
    scores_df = pd.DataFrame(rows)
    valid_s   = scores_df["factscore"].dropna()
    n_valid   = len(valid_s)
    n_total   = len(scores_df)

    print("\n\n" + "="*72)
    print(f"  FACTSCORE SUMMARY  —  {CURRENT_MODEL}")
    print(f"  Valid samples used : {n_valid}/{n_total}")
    print("="*72)
    print(f"{'Row':<5} {'Score':<8} {'Supp.':<8} {'N/V':<8} {'Interpretation'}")
    print("-"*72)

    for i, row in scores_df.iterrows():
        s = row["factscore"]
        if s is None or pd.isna(s):
            reason = row.get("invalid_reason") or "n/a"
            interp, s_str = f"skipped ({reason})", "N/A"
        elif s >= 0.9:
            interp, s_str = "excellent", f"{s:.4f}"
        elif s >= 0.7:
            interp, s_str = "mostly faithful", f"{s:.4f}"
        elif s >= 0.5:
            interp, s_str = "some hallucination", f"{s:.4f}"
        else:
            interp, s_str = "significant hallucination", f"{s:.4f}"

        supp = "—" if pd.isna(row["supported"])      else int(row["supported"])
        nv   = "—" if pd.isna(row["not_verifiable"]) else int(row["not_verifiable"])
        print(f"{i+1:<5} {s_str:<8} {str(supp):<8} {str(nv):<8} {interp}")

    if n_valid > 0:
        print("-"*72)
        print(f"  Mean FactScore (over {n_valid} valid samples): {valid_s.mean():.4f}")
        print(f"  Min : {valid_s.min():.4f}")
        print(f"  Max : {valid_s.max():.4f}")
    else:
        print("\n  No valid samples to score.")

    print(f"\n  ⏱ Total time: {total_elapsed/60:.1f} min ({total_elapsed:.1f}s)")

    # ── Save ─────────────────────────────────────────────────
    out_df = pd.concat(
        [results_df.iloc[:n].reset_index(drop=True), scores_df],
        axis=1
    )
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved → {OUT_CSV}")

print("\n\nAll models processed.")