import torch

# Check GPU info
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Available VRAM: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")

