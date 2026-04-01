"""
FactScore — Hallucination Detection (Voyage AI Retrieval edition)
=================================================================
Three-step pipeline — follows FActScore methodology with modern retrieval:

  Step 1 — AFG     : OLMo breaks each summary into atomic facts
  Step 2 — Retrieval: Voyage AI (voyage-4-large) finds the most relevant
                      article passages for each atomic fact semantically
  Step 3 — AFV     : Gemma validates each fact against retrieved passages

Why Voyage AI instead of BM25:
  - BM25 uses keyword matching → fails on medical text where the same
    concept is expressed many ways ("heart attack" vs "myocardial infarction")
  - Voyage AI uses semantic embeddings → understands meaning, not just words
  - voyage-4-large is the best quality model from Voyage AI
  - Free tier is generous (your supervisor has used it without paying)

Why retrieval instead of full article:
  - Faster than passing the full article to Gemma every time
  - Still accurate because Voyage finds the right passages semantically
  - Follows the original FActScore methodology (retrieval + validation)

Install before running:
    pip install --upgrade git+https://github.com/lflage/OpenFActScore
    pip install voyageai
    python -m spacy download en_core_web_sm

    Add to .env:
        VOYAGE_API_KEY=your_key_here

Inputs:
  data/processed/pubmed_train_clean.csv         → article column
  outputs(Model)_pubmed_abstract/results.csv    → generated_summary column

Output:
  outputs(Model)_pubmed_abstract/results_with_openfactscore_voy.csv
"""

import os
import gc
import json
import time
import torch
import pandas as pd
import nltk

from models_config import N_SAMPLES, OUTPUT_BASE_DIR, DATA_CSV, ACTIVE_MODEL

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Stability flags ───────────────────────────────────────────────────────────
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCH_COMPILE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ── NLTK ─────────────────────────────────────────────────────────────────────
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize

# ── OpenFActScore AFG only ────────────────────────────────────────────────────
try:
    from factscore.atomic_facts import AtomicFactGenerator
except ImportError:
    raise ImportError(
        "\nOpenFActScore is not installed. Run:\n"
        "  pip install --upgrade git+https://github.com/lflage/OpenFActScore\n"
        "  python -m spacy download en_core_web_sm\n"
    )

# ── Fix: demons.json ──────────────────────────────────────────────────────────
import shutil, glob, sys, urllib.request

_KNOWLEDGE_DIR = ".cache/factscore_pubmed"
_demons_dir    = os.path.join(_KNOWLEDGE_DIR, "demos")
_demons_path   = os.path.join(_demons_dir, "demons.json")
if not os.path.exists(_demons_path):
    os.makedirs(_demons_dir, exist_ok=True)
    _demons = {
        "Beyoncé further expanded her acting career, starring in the 2006 musical film Dreamgirls with Jamie Foxx, and winning two Golden Globe nominations.": [
            "Beyoncé expanded her acting career.",
            "Beyoncé starred in the 2006 musical film Dreamgirls.",
            "Dreamgirls featured Jamie Foxx.",
            "Beyoncé received two Golden Globe nominations for Dreamgirls."
        ],
        "He was born in Los Angeles, California, and raised in the San Fernando Valley.": [
            "He was born in Los Angeles.",
            "Los Angeles is in California.",
            "He was raised in the San Fernando Valley."
        ],
        "The movie was directed by Steven Spielberg and starred Tom Hanks in the lead role.": [
            "The movie was directed by Steven Spielberg.",
            "Tom Hanks starred in the movie.",
            "Tom Hanks played the lead role."
        ],
        "She graduated from Harvard University in 1995 with a degree in computer science.": [
            "She graduated from Harvard University.",
            "She graduated in 1995.",
            "Her degree was in computer science."
        ],
        "The treatment group received 10mg of metformin daily for 12 weeks and showed significant improvement in blood glucose levels.": [
            "The treatment group received metformin.",
            "The dose was 10mg of metformin daily.",
            "The treatment lasted 12 weeks.",
            "The treatment group showed significant improvement in blood glucose levels."
        ]
    }
    with open(_demons_path, "w", encoding="utf-8") as _f:
        json.dump(_demons, _f, indent=2)
    print(f"  Created demons.json → {_demons_path}")

# ── Fix: roberta_stopwords.txt ────────────────────────────────────────────────
_stopwords_cwd = os.path.join(os.getcwd(), "roberta_stopwords.txt")
if not os.path.exists(_stopwords_cwd):
    _search_root = os.path.join(sys.prefix, "lib")
    _candidates  = glob.glob(
        os.path.join(_search_root, "**", "roberta_stopwords.txt"), recursive=True
    )
    if _candidates:
        shutil.copy(_candidates[0], _stopwords_cwd)
    else:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/lflage/OpenFActScore/main/roberta_stopwords.txt",
            _stopwords_cwd
        )

# ============================================================
# CONFIGURATION
# ============================================================

AFG_MODEL    = "allenai/OLMo-2-1124-7B-SFT"
AFV_MODEL    = "google/gemma-3-4b-it"
VOYAGE_MODEL = "voyage-4-large"

TOP_K_PASSAGES    = 3
PASSAGE_SENTENCES = 5


# ============================================================
# STEP 1 — AFG
# ============================================================

print(f"\nLoading AFG model (OLMo) for atomic fact generation...")
afg = AtomicFactGenerator(
    model_name = AFG_MODEL,
    demon_dir  = _demons_dir,
)
for attr, val in [("cache_dict", {}), ("model", None), ("tokenizer", None), ("add_n", 0)]:
    if not hasattr(afg.lm, attr):
        setattr(afg.lm, attr, val)
print("  AFG ready.")


# ============================================================
# STEP 2 — Voyage AI
# ============================================================

print(f"\nInitialising Voyage AI retrieval ({VOYAGE_MODEL})...")
try:
    import voyageai
except ImportError:
    raise ImportError("Run: pip install voyageai")

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY not set in .env")

voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
print(f"  Voyage AI ready — model: {VOYAGE_MODEL}, top_k: {TOP_K_PASSAGES}")


def split_article_into_passages(article: str, sentences_per_passage: int = PASSAGE_SENTENCES) -> list:
    sentences = sent_tokenize(article)
    passages  = []
    for i in range(0, len(sentences), sentences_per_passage - 1):
        passage = " ".join(sentences[i : i + sentences_per_passage])
        if passage.strip():
            passages.append(passage)
    return passages if passages else [article]


def retrieve_relevant_passages(fact: str, passages: list, top_k: int = TOP_K_PASSAGES) -> str:
    import numpy as np
    doc_result      = voyage_client.embed(passages, model=VOYAGE_MODEL, input_type="document")
    doc_embeddings  = np.array(doc_result.embeddings)
    query_result    = voyage_client.embed([fact], model=VOYAGE_MODEL, input_type="query")
    query_embedding = np.array(query_result.embeddings[0])
    similarities    = doc_embeddings @ query_embedding
    top_indices     = np.argsort(similarities)[::-1][:top_k]
    top_passages    = [passages[i] for i in sorted(top_indices)]
    return "\n\n".join(top_passages)


# ============================================================
# STEP 3 — AFV (Gemma)
# ============================================================

print(f"\nLoading AFV model (Gemma) for fact validation...")
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

afv_tokenizer = AutoTokenizer.from_pretrained(AFV_MODEL)
afv_model     = AutoModelForCausalLM.from_pretrained(
    AFV_MODEL,
    dtype      = torch.bfloat16,
    device_map = "auto",
)
afv_model.eval()
print(f"  AFV ready on {DEVICE}.")


def validate_fact_with_context(context: str, fact: str) -> bool:
    prompt = (
        "Answer the question based on the given context.\n\n"
        f"Context:\n{context}\n\n"
        f"Statement: {fact}\n"
        "Is the statement supported by the context? Answer only 'true' or 'false'.\n"
        "Answer:"
    )
    inputs = afv_tokenizer(
        prompt,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 4096,
    ).to(DEVICE)
    with torch.no_grad():
        output = afv_model.generate(
            **inputs,
            max_new_tokens = 5,
            do_sample      = False,
            pad_token_id   = afv_tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    answer     = afv_tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
    return answer.startswith("true")


# ============================================================
# FULL PIPELINE
# ============================================================

def factscore_voyage(article: str, summary: str) -> dict:
    atomic_facts_result, _ = afg.run(summary)
    all_facts = []
    for _, facts in atomic_facts_result:
        all_facts.extend(facts)

    if not all_facts:
        return {"score": None, "supported": 0,
                "not_verifiable": 0, "total_facts": 0, "facts_detail": []}

    passages       = split_article_into_passages(article)
    supported      = 0
    not_verifiable = 0
    facts_detail   = []

    for fact in all_facts:
        context      = retrieve_relevant_passages(fact, passages)
        is_supported = validate_fact_with_context(context, fact)
        if is_supported:
            supported += 1
            facts_detail.append({"fact": fact, "label": "ENTAILMENT"})
        else:
            not_verifiable += 1
            facts_detail.append({"fact": fact, "label": "NEUTRAL"})

    total = len(all_facts)
    score = round(supported / total, 4) if total > 0 else None
    return {
        "score"         : score,
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
# MAIN LOOP
# ============================================================

grand_start     = time.time()
all_model_times = {}

for CURRENT_MODEL in active_models:

    RESULTS_CSV = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/results.csv"
    OUT_CSV     = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/results_with_openfactscore_voy.csv"

    print(f"\n\n{'#'*72}")
    print(f"  Processing model : {CURRENT_MODEL}")
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

    rows        = []
    model_start = time.time()   # ← model timer start

    for i in range(n):
        sample_start = time.time()   # ← sample timer start
        print(f"\n{'='*55}  Sample {i+1}/{n}  {'='*55}")

        summary = summaries[i]
        article = articles[i]

        if summary.startswith("ERROR"):
            print("  Skipping ERROR row.")
            rows.append({"factscore": None, "supported": 0,
                         "not_verifiable": 0, "total_facts": 0,
                         "eval_time_sec": 0})
            continue

        result         = factscore_voyage(article, summary)
        sample_elapsed = time.time() - sample_start

        # ETA calculation
        samples_done      = i + 1
        avg_so_far        = (time.time() - model_start) / samples_done
        eta_seconds       = avg_so_far * (n - samples_done)

        print(f"  FactScore      : {result['score']}")
        print(f"  Supported      : {result['supported']} / {result['total_facts']}")
        print(f"  Not verifiable : {result['not_verifiable']} / {result['total_facts']}")
        print(f"  ⏱ Sample time  : {sample_elapsed:.1f}s  |  "
              f"Avg: {avg_so_far:.1f}s  |  "
              f"ETA: {eta_seconds/60:.1f} min remaining")
        print()

        for item in result["facts_detail"]:
            icon = "✓" if item["label"] == "ENTAILMENT" else "?"
            print(f"    {icon} [{item['label']:<13}] {item['fact'][:90]}")

        rows.append({
            "factscore"     : result["score"],
            "supported"     : result["supported"],
            "not_verifiable": result["not_verifiable"],
            "total_facts"   : result["total_facts"],
            "eval_time_sec" : round(sample_elapsed, 2),
        })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_elapsed                  = time.time() - model_start
    all_model_times[CURRENT_MODEL] = model_elapsed

    # ── Summary table ────────────────────────────────────────
    scores_df = pd.DataFrame(rows)
    valid_s   = scores_df["factscore"].dropna()

    print("\n\n" + "="*72)
    print(f"  FACTSCORE SUMMARY  —  {CURRENT_MODEL}")
    print("="*72)
    print(f"{'Row':<5} {'Score':<8} {'Supp.':<8} {'N/V':<8} {'Time(s)':<10} {'Interpretation'}")
    print("-"*72)

    for i, row in scores_df.iterrows():
        s = row["factscore"]
        t = row.get("eval_time_sec", 0)
        if   s is None : interp, s_str = "skipped",                   "N/A"
        elif s >= 0.9  : interp, s_str = "excellent",                 f"{s:.4f}"
        elif s >= 0.7  : interp, s_str = "mostly faithful",           f"{s:.4f}"
        elif s >= 0.5  : interp, s_str = "some hallucination",        f"{s:.4f}"
        else           : interp, s_str = "significant hallucination", f"{s:.4f}"
        print(f"{i+1:<5} {s_str:<8} {int(row['supported']):<8} "
              f"{int(row['not_verifiable']):<8} {t:<10.1f} {interp}")

    if len(valid_s) > 0:
        print("-"*72)
        print(f"{'Mean':<5} {valid_s.mean():.4f}")
        print(f"{'Min':<5} {valid_s.min():.4f}")
        print(f"{'Max':<5} {valid_s.max():.4f}")

    print(f"\n  ⏱ Model total    : {model_elapsed/60:.1f} min")
    print(f"  ⏱ Avg per sample : {model_elapsed/n:.1f}s")
    avg = model_elapsed / n
    for target in [5, 10, 50, 100, 200, 500, 1000]:
        est = target * avg / 60
        marker = "  ← current" if target == n else ""
        print(f"  ⏱ Est. for {target:<5} : {est:.1f} min{marker}")

    # ── Save ─────────────────────────────────────────────────
    out_df = pd.concat([results_df.iloc[:n].reset_index(drop=True), scores_df], axis=1)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved → {OUT_CSV}")

# ── Grand total timing summary ────────────────────────────────
grand_elapsed = time.time() - grand_start
print("\n\n" + "="*72)
print("  TIMING SUMMARY — ALL MODELS")
print("="*72)
print(f"  {'Model':<40} {'Total (min)':<14} {'Avg/sample (s)'}")
print(f"  {'-'*40} {'-'*14} {'-'*15}")
for model_name, elapsed in all_model_times.items():
    n_done = min(N_SAMPLES, len(pd.read_csv(
        OUTPUT_BASE_DIR.format(model=model_name) + "/results.csv")))
    print(f"  {model_name:<40} {elapsed/60:<14.1f} {elapsed/n_done:.1f}")
print(f"\n  Grand total : {grand_elapsed/60:.1f} min")
print("="*72)

print("\n\nAll models processed.")