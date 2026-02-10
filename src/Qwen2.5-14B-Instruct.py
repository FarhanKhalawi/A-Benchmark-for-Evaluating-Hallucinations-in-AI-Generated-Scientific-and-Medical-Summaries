import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import gc
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

# -------------------------------
# NLTK: tokenization
# -------------------------------
import nltk
from nltk.tokenize import sent_tokenize


def _ensure_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


_ensure_nltk_data()

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM: {total_vram:.2f} GB")

# -------------------------------
# Config
# -------------------------------
model_name = "Qwen/Qwen2.5-14B-Instruct"
DATA_CSV = "data/raw/pubmed_train.csv"
OUT_DIR = "outputs(Qwen2.5-14B)_pubmed_abstract"

os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "results.csv")

N_SAMPLES = 3
SEED = 42


MAX_INPUT_TOKENS = 4096   
MAX_ARTICLE_CHARS = 12000  

torch.manual_seed(SEED)
np.random.seed(SEED)


# -------------------------------
# Load data
# -------------------------------
print("\nLoading dataset...")
data_df = pd.read_csv(DATA_CSV)
print(f"Dataset loaded: {len(data_df)} samples")
print(f"Columns: {data_df.columns.tolist()}")

# -------------------------------
# Configuring
# -------------------------------
print("\nConfiguring 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           
    bnb_4bit_compute_dtype=torch.bfloat16,  
    bnb_4bit_use_double_quant=True,      
)

print("Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Clear CUDA cache before loading model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,      
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)
print(f"Model loaded successfully!")
print(f"Model dtype: {model.dtype}")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"VRAM after model load — Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")


base = model.generation_config
gen_cfg = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=20,
    max_new_tokens=512,    
    min_new_tokens=80,     
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=getattr(base, "bos_token_id", None),
    eos_token_id=getattr(base, "eos_token_id", tokenizer.eos_token_id),
    use_cache=True,
)


# -------------------------------
# Main loop: summarise
# -------------------------------
rows_out = []
limit = min(N_SAMPLES, len(data_df))


for i in range(limit):
    print(f"\n====================== SAMPLE {i+1} / {limit} ======================\n")
    row = data_df.iloc[i]

    article_text = str(row["article"])
    abstract_text = str(row["abstract"])

    print(f"Article length: {len(article_text)} characters")
    print(f"Abstract length: {len(abstract_text)} characters")

    # Truncate article if too long
    if len(article_text) > MAX_ARTICLE_CHARS:
        article_text = article_text[:MAX_ARTICLE_CHARS]
        print(f"Article truncated to {MAX_ARTICLE_CHARS} characters")
    else:
        print("Article fits within context window — no truncation needed")

    prompt = (
        "Write a structured scientific abstract for the following article. "
        "ONLY use facts explicitly present in the text. "
        "If a number or unit is not present in the text, do NOT invent it. "
        "Format the abstract with 4 short parts in ONE paragraph:\n"
        "Background: ... Methods: ... Results: ... Conclusion: ...\n\n"
        f"{article_text}\n\nSummary:"
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        padding=False,
    ).to(model.device)

    print(f"Input tokens: {model_inputs.input_ids.shape[1]}")

   
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                generation_config=gen_cfg,
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        generated_summary = tokenizer.decode(
            output_ids, skip_special_tokens=True
        ).strip()

 
        generated_summary = re.sub(r"(?is)<think>.*?</think>\s*", "", generated_summary)
        generated_summary = re.sub(r"(?is)^.*?</think>\s*", "", generated_summary)

    
        sents = sent_tokenize(generated_summary)
        num_sentences = len(sents)

        print("\n--- SUMMARY (MODEL) ---\n")
        print(generated_summary)
        print(f"\nNumber of sentences generated: {num_sentences}")

        print("\n--- ORIGINAL ABSTRACT ---\n")
        print(abstract_text)

        rows_out.append({
            "generated_summary": generated_summary
        })

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n!!! CUDA OOM Error: {e}")
        print("Skipping this sample and freeing memory...")
        rows_out.append({"generated_summary": f"ERROR: CUDA OOM - {str(e)}"})

    except Exception as e:
        print(f"\n!!! ERROR generating summary: {e}")
        print("Skipping this sample...")
        rows_out.append({"generated_summary": f"ERROR: {str(e)}"})

    finally:
      
        if "model_inputs" in locals():
            del model_inputs
        if "generated_ids" in locals():
            del generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"VRAM after cleanup: {allocated:.2f} GB allocated")


# Save CSV
out_df = pd.DataFrame(rows_out)
if os.path.exists(OUT_CSV):
    out_df.to_csv(
        OUT_CSV,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8",
    )
else:
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")

print(f"\nSaved results to: {OUT_CSV}")