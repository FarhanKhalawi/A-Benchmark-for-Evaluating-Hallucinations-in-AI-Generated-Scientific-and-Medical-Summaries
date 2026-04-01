"""
generate.py  —  Generate summaries for all models in models_config.py
======================================================================
Run a single model:
    python generate.py --model Qwen3-4B-Thinking-2507

Run all models sequentially:
    python generate.py --all

Results are saved to:
    outputs(<short_name>)_pubmed_abstract/results.csv
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import gc
import argparse
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
import nltk
from nltk.tokenize import sent_tokenize

from models_config import MODELS, GENERATION, DATA_CSV, N_SAMPLES, SEED, \
                          MAX_INPUT_TOKENS, MAX_ARTICLE_CHARS, OUTPUT_BASE_DIR


# ── NLTK ────────────────────────────────────────────────────
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

_ensure_nltk()


# ── Helpers ─────────────────────────────────────────────────
def load_model(model_name: str):
    """Load tokenizer + 4-bit quantised model."""
    print(f"\nLoading tokenizer & model: {model_name}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"  dtype: {model.dtype}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved()  / 1e9
        print(f"  VRAM — Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

    return tokenizer, model


def unload_model(model):
    """Free GPU memory after a model is done."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  Model unloaded.")


# ── CHANGED: accepts optional max_new_tokens override ───────
def build_gen_config(base_cfg, tokenizer, max_new_tokens=None) -> GenerationConfig:
    gen = {**GENERATION}
    if max_new_tokens is not None:
        gen["max_new_tokens"] = max_new_tokens  # per-model override
    return GenerationConfig(
        **gen,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=getattr(base_cfg, "bos_token_id", None),
        eos_token_id=getattr(base_cfg, "eos_token_id", tokenizer.eos_token_id),
        use_cache=True,
    )


def strip_thinking(text: str):
    """
    Separate <think>…</think> block from the final answer.
    Returns (thinking_content, clean_answer).
    """
    thinking = ""
    match = re.search(r"(?is)<think>(.*?)</think>", text)
    if match:
        thinking = match.group(1).strip()
    clean = re.sub(r"(?is)<think>.*?</think>\s*", "", text)
    clean = re.sub(r"(?is)^.*?</think>\s*", "", clean)
    return thinking, clean.strip()


def summarise_one(article_text: str, tokenizer, model, gen_cfg, thinking: bool) -> tuple:
    """Run one article through the model. Returns (full_output, input_token_count)."""
    if len(article_text) > MAX_ARTICLE_CHARS:
        article_text = article_text[:MAX_ARTICLE_CHARS]
        print(f"  Article truncated to {MAX_ARTICLE_CHARS} characters")
    else:
        print(f"  Article fits within context window — no truncation needed")

    prompt = (
        "Write a structured scientific abstract for the following article. "
        "ONLY use facts explicitly present in the text. "
        "If a number or unit is not present in the text, do NOT invent it. "
        "Do NOT use any markdown formatting such as bold, italics, or headers. "
        "Format the abstract with 4 short parts in ONE paragraph:\n"
        "Background: ... Methods: ... Results: ... Conclusion: ...\n\n"
        f"{article_text}\n\nSummary:"
    )

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,       # True for thinking models, False for others
    )

    inputs = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        padding=False,
    ).to(model.device)

    n_input = inputs.input_ids.shape[1]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    with torch.no_grad():
        generated = model.generate(**inputs, generation_config=gen_cfg)

    output_ids = generated[0][n_input:].tolist()
    full_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    del inputs, generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return full_output, n_input


# ── Core: run one model config ───────────────────────────────
def run_model(cfg: dict, data_df: pd.DataFrame):
    short = cfg["short_name"]
    out_dir = OUTPUT_BASE_DIR.format(model=short)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "results.csv")

    print(f"\n{'='*60}")
    print(f"  MODEL  : {short}")
    print(f"  Output : {out_csv}")
    print(f"  Thinking: {cfg['thinking']}")
    print(f"{'='*60}")

    tokenizer, model = load_model(cfg["name"])

    # ── CHANGED: pass per-model max_new_tokens if defined ───
    gen_cfg = build_gen_config(
        model.generation_config,
        tokenizer,
        max_new_tokens=cfg.get("max_new_tokens"),  # None for non-thinking → uses shared default
    )
    print(f"  max_new_tokens: {gen_cfg.max_new_tokens}")

    rows  = []
    limit = min(N_SAMPLES, len(data_df))

    for i in range(limit):
        print(f"\n====================== SAMPLE {i+1}/{limit} ======================")
        row          = data_df.iloc[i]
        article_text = str(row["article"])
        abstract_text = str(row["abstract"])

        print(f"  Article length : {len(article_text)} characters")
        print(f"  Abstract length: {len(abstract_text)} characters")

        try:
            full_output, n_input = summarise_one(
                article_text, tokenizer, model, gen_cfg, cfg["thinking"]
            )
            print(f"  Input tokens: {n_input}")

            if cfg["thinking"]:
                thinking_content, summary = strip_thinking(full_output)
                thinking_len = len(thinking_content)
            else:
                thinking_content, summary = "", full_output
                thinking_len = 0

            n_sents = len(sent_tokenize(summary))
            print(f"  Summary sentences: {n_sents}")
            print(f"\n--- SUMMARY (MODEL) ---\n{summary}")
            print(f"\n--- ORIGINAL ABSTRACT ---\n{abstract_text}")

            rows.append({
                "generated_summary":     summary,
                "thinking_length_chars": thinking_len,
            })

        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {e}")
            rows.append({"generated_summary": f"ERROR: CUDA OOM - {e}"})

        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({"generated_summary": f"ERROR: {e}", "thinking_length_chars": 0})

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"  VRAM after cleanup: {allocated:.2f} GB allocated")

    unload_model(model)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nSaved → {out_csv}")


# ── CLI ──────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Generate summaries")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",   action="store_true",
                       help="Run all enabled models in models_config.py")
    group.add_argument("--model", type=str,
                       help="short_name of the model to run (e.g. Qwen2.5-14B-Instruct)")
    return p.parse_args()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading data: {DATA_CSV}")
    data_df = pd.read_csv(DATA_CSV)
    print(f"  {len(data_df)} rows | columns: {data_df.columns.tolist()}")

    args = parse_args()

    if args.all:
        targets = [m for m in MODELS if m.get("enabled", True)]
    else:
        targets = [m for m in MODELS if m["short_name"] == args.model]
        if not targets:
            raise ValueError(
                f"Model '{args.model}' not found in models_config.py. "
                f"Available: {[m['short_name'] for m in MODELS]}"
            )

    for cfg in targets:
        run_model(cfg, data_df)

    print("\nAll done!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:   # no arguments → run all enabled models
        sys.argv.append("--all")
    main()