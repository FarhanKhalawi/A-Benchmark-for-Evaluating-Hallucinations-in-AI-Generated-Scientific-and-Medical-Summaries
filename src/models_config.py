# ============================================================
# models_config.py
# ============================================================

MODELS = [
    # ── Local GPU models ────────────────────────────────────   
    {
        "name":       "Qwen/Qwen3-4B",
        "short_name": "Qwen3-4B",
        "thinking":   False,
        "enabled":    False,
    },
    {
        "name":       "Qwen/Qwen3-14B",
        "short_name": "Qwen3-14B",
        "thinking":   False,
        "enabled":    False,
    },
    {
        "name":           "Qwen/Qwen3-4B-Thinking-2507",
        "short_name":     "Qwen3-4B-Thinking-2507",
        "thinking":       True,
        "enabled":        False,
        "max_new_tokens": 4096,
    },
    

    # ── Together.ai API models ───────────────────────────────
    # Requires:  pip install together
    #            TOGETHER_API_KEY in .env file
    
    # ────────────────── Main MODELS  ────────────────────────────────────────────────────────
    
    {
        "name":       "Qwen/Qwen3-0.6B",
        "short_name": "Qwen3-0.6B",
        "thinking":   False,
        "enabled":    True,
    },
    {
        "name":       "Qwen/Qwen3-4B",
        "short_name": "Qwen3-4B",
        "thinking":   False,
        "enabled":    True,   
    },
    {
        "short_name":     "Qwen3.5-9B",
        "api":            "together",
        "together_model": "Qwen/Qwen3.5-9B",
        "thinking":       False,
        "enabled":        True,
        "n_threads":      4,
        "max_new_tokens": 512,
    },
    {
        "short_name":     "GPT-4.1-mini",
        "api":            "openai",        
        "openai_model":   "gpt-4.1-mini",
        "thinking":       False,
        "enabled":        True,
        "n_threads":      4,
        "max_new_tokens": 512,
    },
    {
        "short_name":     "DeepSeek-V3.1",
        "api":            "together",
        "together_model": "deepseek-ai/DeepSeek-V3.1",
        "thinking":       False,
        "enabled":        True,
        "n_threads":      4,
        "max_new_tokens": 512,
    },
    {
        "short_name":     "Llama3-8B-Instruct-Lite",
        "api":            "together",
        "together_model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "thinking":       False,
        "enabled":        True,
        "n_threads":      4,
        "max_new_tokens": 512,
    },
    {
        "short_name":     "DeepSeek-V3.1-Thinking",
        "api":            "together",
        "together_model": "deepseek-ai/DeepSeek-V3.1",
        "thinking":       True,
        "enabled":        True,
        "n_threads":      4,
        "max_new_tokens": 4096,
    },
    {
        "short_name":     "GPT-5.4-nano",
        "api":            "openai",
        "openai_model":   "gpt-5.4-nano",
        "thinking":       False,
        "enabled":        True,
        "n_threads":      4,
        "max_new_tokens": 1024,
        "reasoning_effort": "none",   
    },

    {
        "short_name":     "GPT-5.4-nano-reasoning",
        "api":            "openai",
        "openai_model":   "gpt-5.4-nano",
        "thinking":       False,
        "enabled":        True,
        "n_threads":      4,
        "max_new_tokens": 4096,
        "reasoning_effort": "medium",   # ← change here: none, low, medium, high
    },



    # ── Other MODELS  ────────────────────────────

    {
        "short_name":     "Qwen2.5-7B-Instruct-Turbo",  
        "api":            "together",
        "together_model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "thinking":       False,
        "enabled":        False,
        "n_threads":      4,
        "max_new_tokens": 512,
    },
    {
        "short_name":     "Mistral-Small-24B-Instruct",
        "api":            "together",
        "together_model": "mistralai/Mistral-Small-24B-Instruct-2501",
        "thinking":       False,
        "enabled":        False,
        "n_threads":      4,
        "max_new_tokens": 512,
    },
    {
        "short_name":     "Llama3.3-70B-Instruct-Turbo", 
        "api":            "together",
        "together_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "thinking":       False,
        "enabled":        False,
        "n_threads":      4,
        "max_new_tokens": 512,
    },
    {
        "short_name":           "o4-mini",
        "api":                  "openai",
        "openai_model":         "o4-mini",
        "thinking":             True,
        "enabled":              False,
        "n_threads":            4,
        "max_new_tokens":       4096,   
    },

]

# ── Shared generation settings ──────────────────────────────
# Used by both local GPU models and Together.ai API models
GENERATION = {
    "max_new_tokens": 512,
    "min_new_tokens": 80,
    "do_sample":      True,
    "temperature":    0.7,
    "top_p":          0.9,
    "top_k":          20,
}

# ── Shared data / I/O settings ──────────────────────────────
DATA_CSV          = "data/processed/pubmed_train_clean_tokens(1000).csv"
N_SAMPLES         = 30
SEED              = 42
MAX_INPUT_TOKENS  = 7000
MAX_ARTICLE_CHARS = 65000
OUTPUT_BASE_DIR   = "outputs({model})_pubmed_abstract"
ACTIVE_MODEL = [ "Qwen3-0.6B", "Qwen3-4B" , "Llama3-8B-Instruct-Lite" , "Qwen3.5-9B", "DeepSeek-V3.1-Thinking", "GPT-4.1-mini", "DeepSeek-V3.1", "GPT-5.4-nano", "GPT-5.4-nano-reasoning"] 
#ACTIVE_MODEL = ["Qwen2.5-7B-Instruct-Turbo", "Qwen3.5-9B", "Llama3-8B-Instruct-Lite", "o4-mini", "Mistral-Small-24B-Instruct", "GPT-4.1-mini", "Llama3.3-70B-Instruct-Turbo", "DeepSeek-V3.1"]

#ACTIVE_MODEL = ["Qwen3-0.6B" , "Qwen3.5-9B", "GPT-4.1-mini", "DeepSeek-V3.1-Thinking", "DeepSeek-V3.1", "Llama3-8B-Instruct-Lite", "GPT-5.4-nano", "GPT-5.4-nano-reasoning"]
# gpt-5-mini, gpt-5.4-nano, DeepSeek-V3.1(Thinking), GPT-5.4-nano-reasoning, "Qwen3-4B"