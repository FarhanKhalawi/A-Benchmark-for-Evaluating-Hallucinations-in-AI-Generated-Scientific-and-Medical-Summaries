import pandas as pd
import tiktoken

# Load dataset
data_df = pd.read_csv("data/processed/pubmed_train_clean_tokens(10000).csv")

# Load tokenizer
enc = tiktoken.get_encoding("cl100k_base")

TARGET_TOKENS = 4886

# Loop through the first 1000 rows
for i, row in data_df.head(1000).iterrows():
    
    article = str(row["article"])  # get article text
    
    token_count = len(enc.encode(article))  # count tokens
    
    # Check if the article has exactly 91 tokens
    if token_count == TARGET_TOKENS:
        
        print(f"\nRow {i}: {token_count} tokens")
        print("Article text:")
        print(article)
        print("-" * 80)