import pandas as pd
import os
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(enc.encode(str(text)))


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


df_clean = df_clean[df_clean['article'].apply(count_tokens) >= 25]
print(f"After removing very short articles (<25 tokens): {len(df_clean)} rows")

df_clean = df_clean[df_clean['abstract'].apply(count_tokens) >= 12]
print(f"After removing very short abstracts (<12 tokens): {len(df_clean)} rows")


df_clean = df_clean[df_clean['article'].apply(count_tokens) <= 6000]
print(f"After filtering articles >6000 tokens: {len(df_clean)} rows")


df_clean = df_clean.head(10000)
print(f"After taking first 10000 articles: {len(df_clean)} rows")


df_clean = df_clean.reset_index(drop=True)


os.makedirs("data/processed", exist_ok=True)


output_path = "data/processed/pubmed_train_clean_tokens(10000).csv"
df_clean.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n✓ Cleaned dataset saved to: {output_path}")
print(f"  Final dataset: {len(df_clean)} rows")