import os, math
import numpy as np
from pathlib import Path

def build_inv_freq_llama3(head_dim: int, rope_theta: float, rope_scaling: dict, dtype=np.float32):
    # base inv_freq
    inv = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=dtype) / head_dim))

    # llama3 scaling (piecewise + smooth blend)
    scale_factor = rope_scaling.get("factor", 8.0)
    low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
    old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    new = []
    for inv_freq in inv:
        wavelen = 2 * math.pi / float(inv_freq)
        if wavelen < high_freq_wavelen:
            new.append(inv_freq)
        elif wavelen > low_freq_wavelen:
            new.append(inv_freq / scale_factor)
        else:
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new.append((1 - smooth) * inv_freq / scale_factor + smooth * inv_freq)

    return np.array(new, dtype=dtype)

def get_weights_dir(model_name: str) -> Path:
    base_dir = os.environ.get("LLAMA_WEIGHTS_DIR", "~/llama_weights")
    return Path(base_dir).expanduser().resolve() / f"{model_name}-np"

def write_inv_freq_files(model_name: str, num_layers: int, head_dim: int):
    weights_dir = get_weights_dir(model_name)
    rope_theta = 500000.0
    rope_scaling = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }

    inv = build_inv_freq_llama3(head_dim, rope_theta, rope_scaling)

    weights_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_layers):
        fn = weights_dir / f"layers.{i}.self_attn.rotary_emb.inv_freq"
        if not fn.exists():
            with open(fn, "wb") as f:
                np.save(f, inv)

# Llama 3.1 8B
write_inv_freq_files("Llama-3.1-8B", num_layers=32, head_dim=128)

# Llama 3.1 70B
# write_inv_freq_files("Llama-3.1-70B", num_layers=80, head_dim=128)
