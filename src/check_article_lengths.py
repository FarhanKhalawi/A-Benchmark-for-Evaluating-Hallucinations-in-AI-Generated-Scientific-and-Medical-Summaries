import pandas as pd

data_df = pd.read_csv("data/processed/pubmed_train_clean_tokens(10000).csv")
#data_df = pd.read_csv("data/processed/pubmed_train_clean.csv")


over_limit = 0


for i, row in data_df.head(1000).iterrows():
    article = str(row["article"])
    if len(article) > 30000:
        over_limit += 1
        print(f"Row {i}: {len(article)} characters")

print(f"\nTotal articles over 30000 characters (first 1000 rows): {over_limit} / 1000")
