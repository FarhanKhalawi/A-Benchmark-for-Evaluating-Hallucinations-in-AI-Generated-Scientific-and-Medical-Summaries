"""
generate.py  —  Generate summaries for all models in models_config.py
======================================================================
Run a single model:
    python generate.py --model GPT-4.1-mini

Run all models sequentially:
    python generate.py --all

Results are saved to:
    outputs(<short_name>)_pubmed_abstract/results.csv

Together.ai API models:
    Set TOGETHER_API_KEY in your environment or .env file:
        export TOGETHER_API_KEY=your_key_here
    In models_config.py, set "api": "together" and "together_model": "<model_id>"

OpenAI API models:
    Set OPENAI_API_KEY in your environment or .env file:
        export OPENAI_API_KEY=your_key_here
    In models_config.py, set "api": "openai" and "openai_model": "<model_id>"
    e.g. "openai_model": "gpt-4.1-mini"

Local GPU models:
    No API key needed. Set "api" to nothing (omit the key).
    Heavy imports (torch, transformers) are loaded lazily — only when a local
    model is actually being run, so API-only runs start in under 2 seconds.

Thinking models (e.g. GPT-OSS-20B, Apriel-15B-Thinker):
    Set "thinking": True and "max_new_tokens": 4096 in models_config.py.
    The <think> / <reasoning> block is stripped automatically.

Reasoning models (o1, o3, o4-*):
    These models use max_completion_tokens instead of max_tokens and do not
    support temperature / top_p / top_k sampling parameters. This is handled automatically.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load .env file if present (TOGETHER_API_KEY, OPENAI_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables

import re
import gc
import time
import argparse
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import nltk
from nltk.tokenize import sent_tokenize

from models_config import (
    MODELS, GENERATION, DATA_CSV, N_SAMPLES, SEED,
    MAX_INPUT_TOKENS, MAX_ARTICLE_CHARS, OUTPUT_BASE_DIR,
)


# ── NLTK ─────────────────────────────────────────────────────
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


# ── Thread-safe print ────────────────────────────────────────
_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# ── Lazy imports for heavy GPU libs ─────────────────────────
# Only loaded when a local GPU model is actually used.
# API-only runs (Together.ai / OpenAI) skip these entirely.
_torch = None
_transformers_loaded = False
AutoModelForCausalLM = None
AutoTokenizer = None
BitsAndBytesConfig = None
GenerationConfig = None

def _lazy_load_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch

def _lazy_load_transformers():
    global _transformers_loaded
    global AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
    if not _transformers_loaded:
        from transformers import (
            AutoModelForCausalLM as AMCL,
            AutoTokenizer as AT,
            BitsAndBytesConfig as BnB,
            GenerationConfig as GC,
        )
        AutoModelForCausalLM = AMCL
        AutoTokenizer = AT
        BitsAndBytesConfig = BnB
        GenerationConfig = GC
        _transformers_loaded = True


# ── Shared prompt builder ────────────────────────────────────
def build_prompt(article_text: str) -> str:
    if len(article_text) > MAX_ARTICLE_CHARS:
        article_text = article_text[:MAX_ARTICLE_CHARS]
    return (
        "Write a structured scientific abstract for the following article. "
        "ONLY use facts explicitly present in the text. "
        "If a number or unit is not present in the text, do NOT invent it. "
        "Do NOT use any markdown formatting such as bold, italics, or headers. "
        "STRICT LIMIT: Write maximum 7 sentences total, keeping each sentence concise. "
        "Format the abstract with 4 short parts in ONE plain text paragraph:\n"
        "Background: [1-2 sentences]. Methods: [1-2 sentences]. Results: [1-2 sentences]. Conclusion: [1 sentence].\n\n"
        f"{article_text}\n\nSummary:"
    )

# ── Preamble stripper ────────────────────────────────────────
def strip_preamble(text: str) -> str:
    """Remove lines like 'Here is the structured scientific abstract:'."""
    lines = text.strip().splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("background:"):
            return "\n".join(lines[i:]).strip()
    return text.strip()


# ── Thinking block stripper (shared by local + API paths) ────
def strip_thinking(text: str):
    """Strip <think>...</think> or <reasoning>...</reasoning> blocks."""
    thinking = ""
    match = re.search(r"(?is)(<think>.*?</think>|<reasoning>.*?</reasoning>)", text)
    if match:
        thinking = match.group(1).strip()
    clean = re.sub(r"(?is)<think>.*?</think>\s*", "", text)
    clean = re.sub(r"(?is)<reasoning>.*?</reasoning>\s*", "", clean)
    clean = re.sub(r"(?is)^.*?</(?:think|reasoning)>\s*", "", clean)
    return thinking, clean.strip()


# ── OpenAI reasoning-model detection ────────────────────────
# o1-*, o3-*, o4-*, gpt-5.* require max_completion_tokens and don't support
# temperature / top_p / top_k sampling parameters.
_OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")

def _is_openai_reasoning_model(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _OPENAI_REASONING_PREFIXES)


# ════════════════════════════════════════════════════════════
# LOCAL GPU PATH
# ════════════════════════════════════════════════════════════

def load_model(model_name: str):
    """Load tokenizer + 4-bit quantised model."""
    torch = _lazy_load_torch()
    _lazy_load_transformers()

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
    torch = _lazy_load_torch()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  Model unloaded.")


def build_gen_config(base_cfg, tokenizer, max_new_tokens=None):
    _lazy_load_transformers()
    gen = {**GENERATION}
    if max_new_tokens is not None:
        gen["max_new_tokens"] = max_new_tokens
    return GenerationConfig(
        **gen,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=getattr(base_cfg, "bos_token_id", None),
        eos_token_id=getattr(base_cfg, "eos_token_id", tokenizer.eos_token_id),
        use_cache=True,
    )


def summarise_one_local(article_text: str, tokenizer, model, gen_cfg, thinking: bool) -> tuple:
    """Run one article through a local GPU model. Returns (full_output, input_token_count)."""
    torch = _lazy_load_torch()

    if len(article_text) > MAX_ARTICLE_CHARS:
        article_text = article_text[:MAX_ARTICLE_CHARS]
        print(f"  Article truncated to {MAX_ARTICLE_CHARS} characters")
    else:
        print(f"  Article fits within context window — no truncation needed")

    prompt = build_prompt(article_text)

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
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


def run_model_local(cfg: dict, data_df: pd.DataFrame):
    """Run a local GPU model sequentially."""
    torch = _lazy_load_torch()

    short   = cfg["short_name"]
    out_dir = OUTPUT_BASE_DIR.format(model=short)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "results.csv")

    print(f"\n{'='*60}")
    print(f"  MODEL   : {short}  [LOCAL GPU]")
    print(f"  Output  : {out_csv}")
    print(f"  Thinking: {cfg['thinking']}")
    print(f"{'='*60}")

    tokenizer, model = load_model(cfg["name"])
    gen_cfg = build_gen_config(
        model.generation_config,
        tokenizer,
        max_new_tokens=cfg.get("max_new_tokens"),
    )
    print(f"  max_new_tokens: {gen_cfg.max_new_tokens}")

    rows        = []
    limit       = min(N_SAMPLES, len(data_df))
    total_start = time.time()

    for i in range(limit):
        print(f"\n====================== SAMPLE {i+1}/{limit} ======================")
        row           = data_df.iloc[i]
        article_text  = str(row["article"])
        abstract_text = str(row["abstract"])

        print(f"  Article length : {len(article_text)} characters")
        print(f"  Abstract length: {len(abstract_text)} characters")

        sample_start = time.time()

        try:
            full_output, n_input = summarise_one_local(
                article_text, tokenizer, model, gen_cfg, cfg["thinking"]
            )
            print(f"  Input tokens: {n_input}")

            if cfg["thinking"]:
                thinking_content, summary = strip_thinking(full_output)
                thinking_len = len(thinking_content)
            else:
                thinking_content, summary = "", full_output
                thinking_len = 0

            summary = strip_preamble(summary)
            n_sents = len(sent_tokenize(summary))
            print(f"  Summary sentences: {n_sents}")
            print(f"\n--- SUMMARY (MODEL) ---\n{summary}")
            print(f"\n--- ORIGINAL ABSTRACT ---\n{abstract_text}")

            sample_elapsed = time.time() - sample_start
            print(f"\n  ⏱ Sample time: {sample_elapsed:.1f}s")

            rows.append({
                "generated_summary":     summary,
                "thinking_length_chars": thinking_len,
                "time_seconds":          round(sample_elapsed, 2),
            })

        except Exception as e:
            sample_elapsed = time.time() - sample_start
            is_oom = "OutOfMemoryError" in type(e).__name__ or "CUDA out of memory" in str(e)
            label  = "OOM" if is_oom else "ERROR"
            print(f"  {label}: {e}")
            print(f"  ⏱ Sample time: {sample_elapsed:.1f}s")
            rows.append({
                "generated_summary":     f"ERROR: {e}",
                "thinking_length_chars": 0,
                "time_seconds":          round(sample_elapsed, 2),
            })

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"  VRAM after cleanup: {allocated:.2f} GB allocated")

    total_elapsed = time.time() - total_start
    avg_elapsed   = total_elapsed / limit if limit > 0 else 0

    print(f"\n{'='*60}")
    print(f"  ⏱ Total time    : {total_elapsed/60:.1f} min ({total_elapsed:.1f}s)")
    print(f"  ⏱ Avg per sample: {avg_elapsed:.1f}s")
    print(f"{'='*60}")

    unload_model(model)

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nSaved → {out_csv}")


# ════════════════════════════════════════════════════════════
# TOGETHER.AI API PATH
# ════════════════════════════════════════════════════════════

def _get_together_client():
    """Initialise Together client (reads TOGETHER_API_KEY from environment)."""
    try:
        from together import Together
    except ImportError:
        raise ImportError("Run:  pip install together")

    api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "TOGETHER_API_KEY not set.\n"
            "  export TOGETHER_API_KEY=your_key_here"
        )
    return Together(api_key=api_key)


def _summarise_one_api(
    idx: int,
    article_text: str,
    abstract_text: str,
    together_model: str,
    thinking: bool,
    client,
    max_new_tokens: int,
) -> dict:
    """
    Call Together.ai API for one sample.
    Runs inside a thread — uses tprint for safe output.
    Returns a result dict with 'idx' for ordering.
    """
    sample_start = time.time()

    truncated = len(article_text) > MAX_ARTICLE_CHARS
    if truncated:
        article_text = article_text[:MAX_ARTICLE_CHARS]

    prompt = build_prompt(article_text)

    try:
        # Pass enable_thinking=False for non-thinking models
        # (some Together.ai models like Qwen3.5-9B default to thinking mode)
        extra = {} if thinking else {"chat_template_kwargs": {"enable_thinking": False}}

        response = client.chat.completions.create(
            model=together_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=GENERATION.get("temperature", 0.7),
            top_p=GENERATION.get("top_p", 0.9),
            top_k=GENERATION.get("top_k", 20),
            **extra,
        )

        full_output = response.choices[0].message.content.strip()

        if thinking:
            thinking_content, summary = strip_thinking(full_output)
            thinking_len = len(thinking_content)
        else:
            thinking_content, summary = "", full_output
            thinking_len = 0

        summary = strip_preamble(summary)

        if not summary:
            raise ValueError(
                "Summary is empty after stripping thinking block. "
                f"Full output was: {full_output[:200]!r}"
            )

        usage   = response.usage
        n_input = usage.prompt_tokens if usage else 0

        sample_elapsed = time.time() - sample_start
        n_sents        = len(sent_tokenize(summary))

        tprint(f"\n====================== SAMPLE {idx+1} ======================")
        tprint(f"  Article length   : {len(article_text)} characters"
               + (" [truncated]" if truncated else " — no truncation needed"))
        tprint(f"  Abstract length  : {len(abstract_text)} characters")
        tprint(f"  Input tokens     : {n_input}")
        tprint(f"  Thinking chars   : {thinking_len}")
        tprint(f"  Summary sentences: {n_sents}")
        tprint(f"\n--- SUMMARY (MODEL) ---\n{summary}")
        tprint(f"\n--- ORIGINAL ABSTRACT ---\n{abstract_text}")
        tprint(f"\n  ⏱ Sample time: {sample_elapsed:.1f}s")

        return {
            "idx":                   idx,
            "generated_summary":     summary,
            "thinking_length_chars": thinking_len,
            "time_seconds":          round(sample_elapsed, 2),
        }

    except Exception as e:
        sample_elapsed = time.time() - sample_start
        tprint(f"\n  [Sample {idx+1}] ERROR: {e}  ({sample_elapsed:.1f}s)")
        return {
            "idx":                   idx,
            "generated_summary":     f"ERROR: {e}",
            "thinking_length_chars": 0,
            "time_seconds":          round(sample_elapsed, 2),
        }


def run_model_api(cfg: dict, data_df: pd.DataFrame):
    """Run a Together.ai API model with ThreadPoolExecutor (parallel requests)."""
    short          = cfg["short_name"]
    together_model = cfg["together_model"]
    thinking       = cfg.get("thinking", False)
    n_threads      = cfg.get("n_threads", 4)
    max_new_tokens = cfg.get("max_new_tokens", GENERATION.get("max_new_tokens", 512))
    out_dir        = OUTPUT_BASE_DIR.format(model=short)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "results.csv")

    print(f"\n{'='*60}")
    print(f"  MODEL      : {short}  [TOGETHER.AI API]")
    print(f"  API model  : {together_model}")
    print(f"  Thinking   : {thinking}")
    print(f"  Threads    : {n_threads}  ← parallel requests")
    print(f"  max_tokens : {max_new_tokens}")
    print(f"  Output     : {out_csv}")
    print(f"{'='*60}")

    client = _get_together_client()
    limit  = min(N_SAMPLES, len(data_df))

    samples = [
        (i, str(data_df.iloc[i]["article"]), str(data_df.iloc[i]["abstract"]))
        for i in range(limit)
    ]

    results     = [None] * limit
    completed   = 0
    total_start = time.time()

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {
            executor.submit(
                _summarise_one_api,
                idx, article, abstract, together_model, thinking, client, max_new_tokens
            ): idx
            for idx, article, abstract in samples
        }
        for future in as_completed(futures):
            result               = future.result()
            results[result["idx"]] = result
            completed           += 1
            tprint(f"\n  ✓ Completed {completed}/{limit} samples")

    total_elapsed = time.time() - total_start
    avg_elapsed   = total_elapsed / limit if limit > 0 else 0

    print(f"\n{'='*60}")
    print(f"  ⏱ Total time    : {total_elapsed/60:.1f} min ({total_elapsed:.1f}s)")
    print(f"  ⏱ Avg per sample: {avg_elapsed:.1f}s  (wall-clock / {n_threads} threads)")
    print(f"{'='*60}")

    rows = [{k: v for k, v in r.items() if k != "idx"} for r in results]
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nSaved → {out_csv}")


# ════════════════════════════════════════════════════════════
# OPENAI API PATH
# ════════════════════════════════════════════════════════════

def _get_openai_client():
    """Initialise OpenAI client (reads OPENAI_API_KEY from environment)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run:  pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set.\n"
            "  export OPENAI_API_KEY=your_key_here"
        )
    return OpenAI(api_key=api_key)


def _summarise_one_openai(
    idx: int,
    article_text: str,
    abstract_text: str,
    openai_model: str,
    client,
    max_new_tokens: int,
    reasoning: bool = None,
    reasoning_effort: str = None,        # ← NEW
) -> dict:
    """
    Call OpenAI API for one sample.
    Runs inside a thread — uses tprint for safe output.
    Returns a result dict with 'idx' for ordering.

    Reasoning models (o1-*, o3-*, o4-*, gpt-5.*):
      - Use max_completion_tokens instead of max_tokens
      - Do NOT support temperature / top_p sampling parameters
      - Support optional reasoning_effort: "none", "low", "medium", "high", "xhigh"
    """
    sample_start = time.time()

    truncated = len(article_text) > MAX_ARTICLE_CHARS
    if truncated:
        article_text = article_text[:MAX_ARTICLE_CHARS]

    prompt = build_prompt(article_text)

    # Detect reasoning model and build kwargs accordingly
    is_reasoning = reasoning if reasoning is not None else _is_openai_reasoning_model(openai_model)

    try:
        if is_reasoning:
            # Reasoning models: max_completion_tokens, no sampling params
            kwargs = dict(
                model=openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_new_tokens,
            )
            # Apply reasoning_effort if provided (supported by gpt-5.* and o-series)
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(**kwargs)
        else:
            # Standard models: max_tokens + sampling params
            response = client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=GENERATION.get("temperature", 0.7),
                top_p=GENERATION.get("top_p", 0.9),
            )

        full_output = (response.choices[0].message.content or "").strip()
        summary     = strip_preamble(full_output)

        if not summary:
            raise ValueError(
                "Empty response from OpenAI API. "
                f"Full output was: {full_output[:200]!r}"
            )

        n_input        = response.usage.prompt_tokens if response.usage else 0
        n_sents        = len(sent_tokenize(summary))
        sample_elapsed = time.time() - sample_start

        tprint(f"\n====================== SAMPLE {idx+1} ======================")
        tprint(f"  Article length   : {len(article_text)} characters"
               + (" [truncated]" if truncated else " — no truncation needed"))
        tprint(f"  Abstract length  : {len(abstract_text)} characters")
        tprint(f"  Input tokens     : {n_input}")
        tprint(f"  Uses max_completion_tokens : {is_reasoning}")
        tprint(f"  Reasoning effort : {reasoning_effort}")        # ← NEW
        tprint(f"  Summary sentences: {n_sents}")
        tprint(f"\n--- SUMMARY (MODEL) ---\n{summary}")
        tprint(f"\n--- ORIGINAL ABSTRACT ---\n{abstract_text}")
        tprint(f"\n  ⏱ Sample time: {sample_elapsed:.1f}s")

        return {
            "idx":                   idx,
            "generated_summary":     summary,
            "thinking_length_chars": 0,
            "time_seconds":          round(sample_elapsed, 2),
        }

    except Exception as e:
        sample_elapsed = time.time() - sample_start
        tprint(f"\n  [Sample {idx+1}] ERROR: {e}  ({sample_elapsed:.1f}s)")
        return {
            "idx":                   idx,
            "generated_summary":     f"ERROR: {e}",
            "thinking_length_chars": 0,
            "time_seconds":          round(sample_elapsed, 2),
        }


def run_model_openai(cfg: dict, data_df: pd.DataFrame):
    """Run an OpenAI API model with ThreadPoolExecutor (parallel requests)."""
    short            = cfg["short_name"]
    openai_model     = cfg["openai_model"]
    n_threads        = cfg.get("n_threads", 4)
    max_new_tokens   = cfg.get("max_new_tokens", GENERATION.get("max_new_tokens", 512))
    reasoning_effort = cfg.get("reasoning_effort", None)         # ← NEW
    out_dir          = OUTPUT_BASE_DIR.format(model=short)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "results.csv")

    is_reasoning = _is_openai_reasoning_model(openai_model)

    print(f"\n{'='*60}")
    print(f"  MODEL            : {short}  [OPENAI API]")
    print(f"  API model        : {openai_model}")
    print(f"  Uses max_completion_tokens : {is_reasoning}")
    print(f"  Reasoning effort : {reasoning_effort}")             # ← NEW
    print(f"  Threads          : {n_threads}  ← parallel requests")
    print(f"  max_tokens       : {max_new_tokens}")
    print(f"  Output           : {out_csv}")
    print(f"{'='*60}")

    client = _get_openai_client()
    limit  = min(N_SAMPLES, len(data_df))

    samples = [
        (i, str(data_df.iloc[i]["article"]), str(data_df.iloc[i]["abstract"]))
        for i in range(limit)
    ]

    results     = [None] * limit
    completed   = 0
    total_start = time.time()

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {
            executor.submit(
                _summarise_one_openai,
                idx, article, abstract,
                openai_model, client, max_new_tokens,
                cfg.get("reasoning", None),
                reasoning_effort,                                 # ← NEW
            ): idx
            for idx, article, abstract in samples
        }
        for future in as_completed(futures):
            result               = future.result()
            results[result["idx"]] = result
            completed           += 1
            tprint(f"\n  ✓ Completed {completed}/{limit} samples")

    total_elapsed = time.time() - total_start
    avg_elapsed   = total_elapsed / limit if limit > 0 else 0

    print(f"\n{'='*60}")
    print(f"  ⏱ Total time    : {total_elapsed/60:.1f} min ({total_elapsed:.1f}s)")
    print(f"  ⏱ Avg per sample: {avg_elapsed:.1f}s  (wall-clock / {n_threads} threads)")
    print(f"{'='*60}")

    rows = [{k: v for k, v in r.items() if k != "idx"} for r in results]
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nSaved → {out_csv}")


# ── Dispatcher ───────────────────────────────────────────────
def run_model(cfg: dict, data_df: pd.DataFrame):
    api = cfg.get("api", "local")
    if api == "together":
        run_model_api(cfg, data_df)
    elif api == "openai":
        run_model_openai(cfg, data_df)
    else:
        run_model_local(cfg, data_df)


# ── CLI ──────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Generate summaries")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",   action="store_true",
                       help="Run all enabled models in models_config.py")
    group.add_argument("--model", type=str,
                       help="short_name of the model to run")
    return p.parse_args()


def main():
    np.random.seed(SEED)

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

    # Only print GPU info if at least one local model will run
    has_local = any(t.get("api", "local") == "local" for t in targets)
    if has_local:
        torch = _lazy_load_torch()
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print(f"  GPU : {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    for cfg in targets:
        run_model(cfg, data_df)

    print("\nAll done!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv.append("--all")
    main()