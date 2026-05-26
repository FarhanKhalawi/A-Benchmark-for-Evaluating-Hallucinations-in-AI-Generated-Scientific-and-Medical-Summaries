import pandas as pd

df_train = pd.read_csv("data/raw/pubmed_train.csv")
#df_valid = pd.read_csv("data/raw/pubmed_validation.csv")
#df_test  = pd.read_csv("data/raw/pubmed_test.csv")

#df_train = pd.read_csv("data/processed/pubmed_train_clean.csv")
print(df_train.head())

###############################################################
import pandas as pd

df = pd.read_csv("data/processed/pubmed_train_clean_tokens(1000).csv")
row = df.iloc[551]
print(row["article"][:1000])