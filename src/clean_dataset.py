import pandas as pd
import os


print("Loading dataset...")
df = pd.read_csv("data/raw/pubmed_train.csv")
print(f"Original dataset: {len(df)} rows")


print("\nCleaning data...")
df_clean = df.dropna(subset=['article', 'abstract'])
print(f"After removing NaN values: {len(df_clean)} rows")


df_clean = df_clean[
    (df_clean['article'].str.strip() != '') & 
    (df_clean['abstract'].str.strip() != '')
]
print(f"After removing empty strings: {len(df_clean)} rows")


df_clean = df_clean[df_clean['article'].str.len() >= 100]
print(f"After removing very short articles (<100 chars): {len(df_clean)} rows")

df_clean = df_clean[df_clean['abstract'].str.len() >= 50]
print(f"After removing very short abstracts (<50 chars): {len(df_clean)} rows")


df_clean = df_clean[df_clean['article'].str.len() <= 30000]
print(f"After filtering articles >30000 chars: {len(df_clean)} rows")


df_clean = df_clean.head(1000)
print(f"After taking first 1000 articles: {len(df_clean)} rows")


df_clean = df_clean.reset_index(drop=True)


os.makedirs("data/processed", exist_ok=True)


output_path = "data/processed/pubmed_train_clean.csv"
df_clean.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n✓ Cleaned dataset saved to: {output_path}")
print(f"  Final dataset: {len(df_clean)} rows")