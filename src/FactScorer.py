"""
FactScore — Hallucination Detection
=====================================
Verifies each atomic fact in the generated summary
against the original article from pubmed_train_clean.csv

Inputs:
  data/processed/pubmed_train_clean.csv                          → article column
  outputs(Qwen).../results.csv                                   → generated_summary column

Output:
  results_with_factscore.csv
"""

import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
import nltk

from models_config import OUTPUT_BASE_DIR, DATA_CSV, ACTIVE_MODEL

# ============================================================
# NLTK
# ============================================================
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

# ============================================================
# NLI MODEL  (loaded once, reused for all models)
# ============================================================

print("Loading NLI model...")
DEVICE = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
nli_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-small")
nli_model.eval()
if DEVICE == 0:
    nli_model = nli_model.cuda()

LABELS = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}
print(f"NLI model ready on {'GPU' if DEVICE == 0 else 'CPU'}.")

# ============================================================
# ATOMIC FACT EXTRACTOR
# ============================================================

def extract_atomic_facts(sentence: str) -> list:
    sentence = sentence.strip()
    if not sentence:
        return []

    parts = re.split(r"\s+(?:and|but|while|whereas|although)\s+",
                     sentence, flags=re.IGNORECASE)

    facts = []
    for part in parts:
        part = part.strip()
        facts.append(part if len(part.split()) >= 4 else sentence)

    seen, unique = set(), []
    for f in facts:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return unique or [sentence]


# ============================================================
# NLI VERIFIER — SLIDING WINDOW
# ============================================================

def verify_fact(article: str, fact: str) -> str:
    words      = article.split()
    chunk_size = 80
    overlap    = 20
    best_label = "NEUTRAL"
    best_prob  = 0.0

    for start in range(0, len(words), chunk_size - overlap):
        chunk  = " ".join(words[start : start + chunk_size])
        inputs = tokenizer(chunk, fact, return_tensors="pt",
                           truncation=True, max_length=512, padding=True)

        if DEVICE == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            probs = torch.softmax(nli_model(**inputs).logits, dim=-1)[0]

        entail_p = probs[1].item()
        contra_p = probs[0].item()

        if entail_p > 0.75:
            return "ENTAILMENT"

        if entail_p > best_prob:
            best_prob  = entail_p
            best_label = LABELS[probs.argmax().item()]

        if contra_p > 0.75:
            best_label = "CONTRADICTION"

        if start + chunk_size >= len(words):
            break

    return best_label


# ============================================================
# FACTSCORE
# ============================================================

def factscore(article: str, summary: str) -> dict:
    sentences = sent_tokenize(summary)
    if not sentences:
        return {"score": None, "supported": 0, "contradicted": 0,
                "not_verifiable": 0, "total_facts": 0, "facts_detail": []}

    all_facts = []
    for sent in sentences:
        all_facts.extend(extract_atomic_facts(sent))

    supported = contradicted = not_verifiable = 0
    facts_detail = []

    for fact in all_facts:
        label = verify_fact(article, fact)
        facts_detail.append({"fact": fact, "label": label})
        if   label == "ENTAILMENT"   : supported      += 1
        elif label == "CONTRADICTION": contradicted    += 1
        else                         : not_verifiable += 1

    total = len(all_facts)
    return {
        "score"         : round(supported / total, 4) if total else None,
        "supported"     : supported,
        "contradicted"  : contradicted,
        "not_verifiable": not_verifiable,
        "total_facts"   : total,
        "facts_detail"  : facts_detail,
    }


# ============================================================
# LOAD ARTICLE DATA ONCE
# ============================================================

print("\nLoading pubmed_train_clean.csv ...")
data_df = pd.read_csv(DATA_CSV)
print(f"  {len(data_df)} rows | columns: {data_df.columns.tolist()}")


# ============================================================
# SUPPORT BOTH STRING AND LIST FOR ACTIVE_MODEL
# ============================================================

active_models = ACTIVE_MODEL if isinstance(ACTIVE_MODEL, list) else [ACTIVE_MODEL]


# ============================================================
# MAIN LOOP — iterate over each model
# ============================================================

for CURRENT_MODEL in active_models:

    RESULTS_CSV = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/results.csv"
    OUT_CSV     = OUTPUT_BASE_DIR.format(model=CURRENT_MODEL) + "/results_with_factscore.csv"

    print(f"\n\n{'#'*72}")
    print(f"  Processing model: {CURRENT_MODEL}")
    print(f"  Input : {RESULTS_CSV}")
    print(f"  Output: {OUT_CSV}")
    print(f"{'#'*72}")

    # ── Load model results ───────────────────────────────────
    print("\nLoading results.csv ...")
    try:
        results_df = pd.read_csv(RESULTS_CSV)
    except FileNotFoundError:
        print(f"  [SKIP] File not found: {RESULTS_CSV}")
        continue

    print(f"  {len(results_df)} rows | columns: {results_df.columns.tolist()}")

    assert len(results_df) <= len(data_df), \
        "results.csv has more rows than pubmed_train_clean.csv"

    n         = len(results_df)
    articles  = data_df["article"].astype(str).tolist()[:n]
    summaries = results_df["generated_summary"].astype(str).tolist()

    # ── Run FactScore ────────────────────────────────────────
    rows = []

    for i in range(n):
        print(f"\n{'='*55}  Sample {i+1}/{n}  {'='*55}")

        summary = summaries[i]
        article = articles[i]

        if summary.startswith("ERROR"):
            print("  Skipping ERROR row.")
            rows.append({"factscore": None, "supported": 0, "contradicted": 0,
                         "not_verifiable": 0, "total_facts": 0})
            continue

        result = factscore(article, summary)

        print(f"  FactScore      : {result['score']}")
        print(f"  Supported      : {result['supported']} / {result['total_facts']}")
        print(f"  Contradicted   : {result['contradicted']} / {result['total_facts']}")
        print(f"  Not verifiable : {result['not_verifiable']} / {result['total_facts']}")
        print()
        for item in result["facts_detail"]:
            icon = "✓" if item["label"] == "ENTAILMENT"    else \
                   "✗" if item["label"] == "CONTRADICTION" else "?"
            print(f"    {icon} [{item['label']:<13}] {item['fact'][:90]}")

        rows.append({
            "factscore"     : result["score"],
            "supported"     : result["supported"],
            "contradicted"  : result["contradicted"],
            "not_verifiable": result["not_verifiable"],
            "total_facts"   : result["total_facts"],
        })

    # ── Summary table ────────────────────────────────────────
    scores_df = pd.DataFrame(rows)
    valid     = scores_df["factscore"].dropna()

    print("\n\n" + "="*72)
    print(f"  FACTSCORE SUMMARY  —  {CURRENT_MODEL}")
    print("="*72)
    print(f"{'Row':<5} {'Score':<8} {'Supp.':<8} {'Contra.':<10} {'N/V':<8} {'Interpretation'}")
    print("-"*72)

    for i, row in scores_df.iterrows():
        s = row["factscore"]
        if   s is None : interp, s_str = "skipped",                   "N/A"
        elif s >= 0.9  : interp, s_str = "excellent",                 f"{s:.4f}"
        elif s >= 0.7  : interp, s_str = "mostly faithful",           f"{s:.4f}"
        elif s >= 0.5  : interp, s_str = "some hallucination",        f"{s:.4f}"
        else           : interp, s_str = "significant hallucination", f"{s:.4f}"
        print(f"{i+1:<5} {s_str:<8} {int(row['supported']):<8} "
              f"{int(row['contradicted']):<10} {int(row['not_verifiable']):<8} {interp}")

    if len(valid) > 0:
        print("-"*72)
        print(f"{'Mean':<5} {valid.mean():.4f}")
        print(f"{'Min':<5} {valid.min():.4f}")
        print(f"{'Max':<5} {valid.max():.4f}")

    # ── Save ─────────────────────────────────────────────────
    out_df = pd.concat([results_df, scores_df], axis=1)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved → {OUT_CSV}")

print("\n\nAll models processed.")