import pandas as pd
import os
import tiktoken
from openai import OpenAI

client = OpenAI()

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(enc.encode(str(text)))


def is_biomedical(article_text):
    prompt = f"""
Answer ONLY with YES or NO.

Is the following text from a biomedical research article?

Count as YES if the article is about:
- Human or animal diseases, disorders, or conditions
- Clinical studies, trials, or medical treatments
- Drugs, medications, or therapies
- Human biology, anatomy, or physiology
- Public health, epidemiology, or mental health
- Medical imaging, diagnosis, or surgery

Count as NO if the article is about:
- Agriculture, farming, or food science
- Engineering, chemistry, or physics
- Education, psychology, or social science
- Technology, computing, or economics

Text:
{article_text[:1000]}
"""
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    answer = response.choices[0].message.content.strip().lower()
    return answer.startswith("yes")


print("Loading dataset...")
df = pd.read_csv("data/raw/pubmed_train.csv")
total_original = len(df)
print(f"Original dataset: {total_original} rows")


print("\nCleaning data...")
df_clean = df.dropna(subset=['article', 'abstract'])
after_dropna = len(df_clean)

df_clean = df_clean[
    (df_clean['article'].str.strip() != '') &
    (df_clean['abstract'].str.strip() != '')
]
after_empty = len(df_clean)

df_clean = df_clean[df_clean['article'].apply(count_tokens) >= 1000]
df_clean = df_clean[df_clean['abstract'].apply(count_tokens) >= 100]
df_clean = df_clean[df_clean['article'].apply(count_tokens) <= 6000]
after_tokens = len(df_clean)

# ✅ removed head(1000) — let it process as many rows as needed
df_clean = df_clean.reset_index(drop=True)


print("\nChecking biomedical articles with model...")
biomedical_rows = []
non_biomedical_rows = []
TARGET = 1000  # ✅ goal is exactly 1000 biomedical articles

for i, row in df_clean.iterrows():
    if len(biomedical_rows) >= TARGET:
        break  # ✅ stop as soon as we hit 1000

    if is_biomedical(row["article"]):
        biomedical_rows.append(row)
    else:
        non_biomedical_rows.append(row)

    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1} | YES: {len(biomedical_rows)} | NO: {len(non_biomedical_rows)}")

df_bio = pd.DataFrame(biomedical_rows)
df_bio = df_bio.reset_index(drop=True)
df_non_bio = pd.DataFrame(non_biomedical_rows)

total_checked = len(biomedical_rows) + len(non_biomedical_rows)
no_example = df_non_bio.iloc[0]["article"][:500] + "............." if len(df_non_bio) > 0 else "No rejected articles found"

os.makedirs("data/processed", exist_ok=True)
output_path = "data/processed/pubmed_train_clean_tokens(1000).csv"
df_bio.to_csv(output_path, index=False, encoding='utf-8')

report = f"""
====================================================
           DATASET FILTERING 
====================================================

STEP 1 — Original dataset
  Total rows loaded:                {total_original}

STEP 2 — Rule-based filtering
  After removing nulls:             {after_dropna}   (removed {total_original - after_dropna})
  After removing empty fields:      {after_empty}    (removed {after_dropna - after_empty})
  After token length filter:        {after_tokens}   (removed {after_empty - after_tokens})

STEP 3 — AI filtering (gpt-4.1-nano)
  Total checked by AI:              {total_checked}
  ✅ YES (biomedical):              {len(biomedical_rows)}
  ❌ NO  (not biomedical):          {len(non_biomedical_rows)}

FINAL DATASET
  Rows saved to file:               {len(df_bio)}
  Output path:                      {output_path}

====================================================
  EXAMPLE OF A REJECTED ARTICLE (NO):
====================================================
{no_example}
====================================================
"""

print(report)

report_path = "data/processed/filtering_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"✓ Report saved to: {report_path}")