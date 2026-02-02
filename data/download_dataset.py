#-------Install Python (Ubuntu / Debian)-----------
#sudo apt update
#sudo apt install -y python3 python3-venv python3-pip


#-------Create the virtual environment-----------
#python3 -m venv venv


#-------Activate the venv-----------
#source venv/bin/activate

#-------Upgrade pip & install required packages-----------
#pip install --upgrade pip
#pip install datasets pandas pyarrow


from datasets import load_dataset

# load the dataset
dataset = load_dataset("ccdv/pubmed-summarization")

# save splits to CSV
dataset["train"].to_csv("data/raw/pubmed_train.csv", index=False)
#dataset["validation"].to_csv("data/raw/pubmed_validation.csv", index=False)
#dataset["test"].to_csv("data/raw/pubmed_test.csv", index=False)

print("CSV files saved!")
