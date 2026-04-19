"""
filter_valid.py — Filter valid summaries from results1000.csv → results.csv
============================================================================
Keeps only summaries that:
  - Are not ERROR
  - Contain Background, Methods, Results, Conclusion
  - Have ≤14 sentences (after cleaning)
"""

import os
import re
import pandas as pd
from nltk.tokenize import sent_tokenize
from models_config import OUTPUT_BASE_DIR, ACTIVE_MODEL

def count_sentences(summary):
    clean = re.sub(r'\*\*', '', summary)
    clean = re.sub(r'\n+', ' ', clean)
    clean = re.sub(r'(background|methods|results|conclusion)\s*:', '', clean, flags=re.IGNORECASE)
    clean = clean.strip()
    return len(sent_tokenize(clean))

def is_valid_summary(summary: str) -> bool:
    if not summary or summary.startswith("ERROR"):
        return False

    summary_lower = re.sub(r'\*\*', '', summary).lower()
    required = ["background", "methods", "results", "conclusion"]
    if any(s not in summary_lower for s in required):
        return False

    if count_sentences(summary) > 14:
        return False

    return True

if isinstance(ACTIVE_MODEL, str):
    ACTIVE_MODEL = [ACTIVE_MODEL]

for model_name in ACTIVE_MODEL:
    in_csv  = os.path.join(OUTPUT_BASE_DIR.format(model=model_name), "results1000.csv")
    out_csv = os.path.join(OUTPUT_BASE_DIR.format(model=model_name), "results.csv")

    if not os.path.exists(in_csv):
        print(f"  WARNING: {in_csv} not found — skipping {model_name}")
        continue

    df = pd.read_csv(in_csv)
    total = len(df)

    mask = df["generated_summary"].astype(str).apply(is_valid_summary)
    df_valid = df[mask].reset_index(drop=True)

    df_valid.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"  {model_name}: {len(df_valid)}/{total} valid → {out_csv}")

print("\nDone!")