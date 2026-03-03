# ============================================================
# models_config.py
# ============================================================

MODELS = [
    {
        "name":       "Qwen/Qwen3-4B-Thinking-2507",
        "short_name": "Qwen3-4B-Thinking-2507",
        "thinking":   True,
        "enabled":    True,   
        "max_new_tokens": 4096, 
    },
    {
        "name":       "Qwen/Qwen2.5-14B-Instruct",
        "short_name": "Qwen2.5-14B-Instruct",
        "thinking":   False,
        "enabled":    True,    
    },
    {
        "name":       "Qwen/Qwen3-14B",
        "short_name": "Qwen3-14B",
        "thinking":   False,
        "enabled":    True,   
    },
    {
        "name":       "Qwen/Qwen3-4B",
        "short_name": "Qwen3-4B",
        "thinking":   False,
        "enabled":    True,
    },
    # ── Add new models below this line ──────────────────────
   
]

# ── Shared generation settings ──────────────────────────────
GENERATION = {
    #"max_new_tokens":    786,
    "max_new_tokens":    512,
    "min_new_tokens":    80,
    "do_sample":         True,
    "temperature":       0.7,
    "top_p":             0.9,
    "top_k":             20,
}

# ── Shared data / I/O settings ──────────────────────────────
DATA_CSV          = "data/processed/pubmed_train_clean_tokens(10000).csv" #"data/processed/pubmed_train_clean.csv"
N_SAMPLES         = 1000
SEED              = 42
MAX_INPUT_TOKENS  = 7000    #16384
MAX_ARTICLE_CHARS = 65000   #65000   
OUTPUT_BASE_DIR   = "outputs({model})_pubmed_abstract"
ACTIVE_MODEL = ["Qwen3-14B"]