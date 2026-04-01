import pandas as pd
import tiktoken

data_df = pd.read_csv("data/processed/pubmed_train_clean_tokens(10000).csv")

# Load tokenizer (cl100k_base is used by GPT-4/Claude-equivalent models)
enc = tiktoken.get_encoding("cl100k_base")

TOKEN_LIMIT = 1200  # adjust as needed
over_limit = 0

for i, row in data_df.head(10).iterrows():
    article = str(row["article"])
    token_count = len(enc.encode(article))
    if token_count > TOKEN_LIMIT:
        over_limit += 1
        print(f"Row {i}: {token_count} tokens")

print(f"\nTotal articles over {TOKEN_LIMIT} tokens (first 10000 rows): {over_limit} / 10000")
###########################################################################

under_limit = 0
for i, row in data_df.head(100).iterrows():
    article = str(row["article"])
    token_count = len(enc.encode(article))
    
    if token_count <= TOKEN_LIMIT:
        under_limit += 1
        print(f"Row {i}: {token_count} tokens")

print(f"\nTotal articles under {TOKEN_LIMIT} tokens (first 1000 rows): {under_limit} / 1000")