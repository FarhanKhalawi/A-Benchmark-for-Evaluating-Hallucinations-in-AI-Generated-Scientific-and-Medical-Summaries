"""
Baseline Setup — Human-Written Abstracts as Reference
======================================================
This script takes the original human-written abstracts from the dataset
and formats them as if they were model outputs. This creates a BASELINE
for comparison against AI-generated abstracts.

After running this, you can use the SAME factscore.py and humanness scripts
to evaluate the human abstracts and see:
  - What FactScore do humans get? (should be ~1.0)
  - What humanness score do humans get? (should be 70-100)
  - How do AI models compare to this baseline?

Usage:
    python setup_baseline.py

Creates:
    outputs(Abstract-Baseline)_pubmed_abstract/results1000.csv
"""

import os
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Input: your dataset with human-written abstracts
INPUT_CSV = "data/processed/pubmed_train_clean_tokens(1000).csv"

# Output: formatted like model outputs for comparison
BASELINE_DIR = "outputs(Human-Written)_pubmed_abstract"
OUTPUT_CSV = f"{BASELINE_DIR}/results1000.csv"

# Number of samples to include (match your N_SAMPLES)
N_SAMPLES = 1000


# ============================================================
# MAIN
# ============================================================

print("="*72)
print("BASELINE SETUP - Human-Written Abstracts")
print("="*72)

# Load the dataset
print(f"\n1. Loading {INPUT_CSV} ...")
try:
    df = pd.read_csv(INPUT_CSV)
    print(f"   ✓ Loaded {len(df)} rows")
    print(f"   Columns: {df.columns.tolist()}")
except FileNotFoundError:
    print(f"   ✗ ERROR: File not found: {INPUT_CSV}")
    print(f"\n   Make sure the file exists at this location.")
    exit(1)

# Check for required columns
if 'abstract' not in df.columns:
    print(f"\n   ✗ ERROR: 'abstract' column not found in dataset")
    print(f"   Available columns: {df.columns.tolist()}")
    print(f"\n   This script expects a column named 'abstract' with human-written abstracts.")
    exit(1)

# Limit to N_SAMPLES
n = min(N_SAMPLES, len(df))
df_subset = df.head(n).copy()
print(f"\n2. Using first {n} samples")

# Create output directory
print(f"\n3. Creating directory: {BASELINE_DIR}")
Path(BASELINE_DIR).mkdir(parents=True, exist_ok=True)
print(f"   ✓ Directory ready")

# Create the results CSV in the same format as model outputs
print(f"\n4. Creating {OUTPUT_CSV} ...")

# Format: same as your model output files
# The key column is "generated_summary" - we'll put the human abstract there
output_df = pd.DataFrame({
    'generated_summary': df_subset['abstract'].astype(str).tolist(),
})


# Save
output_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"   ✓ Saved {len(output_df)} abstracts")

# Verify
print(f"\n5. Verification:")
verify_df = pd.read_csv(OUTPUT_CSV)
print(f"   Rows: {len(verify_df)}")
print(f"   Columns: {verify_df.columns.tolist()}")
print(f"\n   First abstract (truncated):")
print(f"   {verify_df['generated_summary'].iloc[0][:200]}...")

