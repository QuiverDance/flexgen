"""
The Llama model configurations and weight downloading utilities.

adopted from opt_config.py

#this codes were based on: PR#135 https://github.com/FMInference/FlexLLMGen/pull/135
"""

import dataclasses
import glob
import os
from typing import Any, Dict, Any, Tuple, Union
import numpy as np
from tqdm import tqdm


@dataclasses.dataclass(frozen=True)
class LlamaConfig:
    name: str = "Llama-3.1-8B"
    hf_token: str = ''
    
    model_type: str = "llama"
    architectures: Tuple[str, ...] = ("LlamaForCausalLM",)

    hidden_act: str = "silu"
    input_dim: int = 4096 # == hidden_size
    initializer_range: float = 0.02
    intermediate_size: int = 14336
    max_position_embeddings: int = 131072

    n_head: int = 32 # == num_attention_heads
    num_hidden_layers: int = 32
    num_key_value_heads: int = 8

    rms_norm_eps: float = 1e-05
    
    # Llama 3.1 attention/MLP flags
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False

    # RoPE (Llama 3.1 long context)
    rope_theta: float = 500000.0
    rope_scaling: Dict[str, Any] = dataclasses.field(default_factory=lambda: {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    })
    
    tie_word_embeddings: bool = False
    use_cache: bool = True
    pretraining_tp: int = 1

    # Tokenizer-related
    vocab_size: int = 128256
    bos_token_id: int = 128000
    eos_token_id: Union[int, Tuple[int, ...]] = 128001  # base default
    pad_token_id: int = 128004

    # Dtypes: keep your existing convention (numpy dtype used by your runtime)
    dtype: type = np.float16
    # HF config.json also carries torch_dtype="bfloat16" (info/interop)
    torch_dtype: str = "bfloat16"

    def model_bytes(self):
        """
        Rough parameter-size estimator (bytes) for Llama 3.1-style blocks.
        
        - Token embedding + LM head (untied in Llama 3.1): 2 * vocab * h
        - Per layer:
            Attention: Wq(h*h) + Wk(h*kv_dim) + Wv(h*kv_dim) + Wo(h*h)
            MLP (SwiGLU): gate(h*intermediate) + up(h*intermediate) + down(intermediate*h)
            RMSNorm: 2*h
        - Final RMSNorm: h
        """
        h = self.input_dim
        intermediate = self.intermediate_size
        n_head = self.n_head
        n_kv = self.num_key_value_heads
        head_dim = h // n_head
        kv_dim = head_dim * n_kv

        bytes_per_element = 2 # float16

        embed_and_head = 2 * self.vocab_size * h

        attn = (h * h) + (h * kv_dim) + (h * kv_dim) + (h * h)  # q,k,v,out
        mlp = 3 * h * intermediate # gate, up, down
        norms = 2 * h # before attn and mlp

        final_norm = h

        total_elems = embed_and_head + self.num_hidden_layers * (attn + mlp + norms) + final_norm
        return total_elems * bytes_per_element

    def cache_bytes(self, batch_size, seq_len):
        # for RoPE caching (if implemented)
        head_dim = self.input_dim // self.n_head
        kv_dim = self.num_key_value_heads * head_dim
        bytes_per_element = 2  # float16
        return batch_size * seq_len * self.num_hidden_layers * kv_dim * 2 * bytes_per_element

    def hidden_bytes(self, batch_size, seq_len):
        bytes_per_element = 2  # float16
        return batch_size * seq_len * self.input_dim * bytes_per_element


def get_llama_config(name, **kwargs):
    if "/" in name:
        name = name.split("/")[-1]

    is_instruct = "-Instruct" in name
    if is_instruct:
        arch_name = name.replace("-Instruct", "")
    else:
        arch_name = name

    if arch_name in ("Llama-3.1-8B", "Meta-Llama-3.1-8B"):
        config = LlamaConfig(
            name=name,
            hf_token=kwargs.get("hf_token", ""),
            input_dim=4096,
            intermediate_size=14336,
            n_head=32,
            num_hidden_layers=32,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            vocab_size=128256,
            bos_token_id=128000,
            pad_token_id=128004,
            eos_token_id=(128001, 128008, 128009) if is_instruct else 128001,
        )
    elif arch_name in ("Llama-3.1-70B", "Meta-Llama-3.1-70B"):
        config = LlamaConfig(
            name=name,
            hf_token=kwargs.get("hf_token", ""),
            input_dim=8192,
            intermediate_size=28672,
            n_head=64,
            num_hidden_layers=80,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            vocab_size=128256,
            bos_token_id=128000,
            pad_token_id=128004,
            eos_token_id=(128001, 128008, 128009) if is_instruct else 128001,
        )
    else:
        raise ValueError(f"Invalid model name: {name} (only Llama 3.1 8B/70B supported)")

    return dataclasses.replace(config, **kwargs)


def download_llama_weights(model_name, path, hf_token):
    """
    Downloads HF weights and converts them into per-tensor .npy files under:
      <path>/<model_name>-np/

    Updated for Llama 3.1: supports .safetensors (primary) and .bin (legacy).
    """
    from huggingface_hub import snapshot_download
    import torch

    try:
        from safetensors.torch import load_file as safetensors_load_file
    except ImportError:
        safetensors_load_file = None

    print(
        f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
        f"The downloading and cpu loading can take dozens of minutes. "
        f"If it seems to get stuck, you can monitor the progress by "
        f"checking the memory usage of this process."
    )

    # Accept full repo id or short name
    if "/" in model_name:
        hf_model_name = model_name
    else:
        hf_model_name = "meta-llama/" + model_name

    folder = snapshot_download(
        hf_model_name,
        allow_patterns=["*.safetensors", "*.bin"],
        token=hf_token,
    )

    # Prefer safetensors if available
    st_files = glob.glob(os.path.join(folder, "*.safetensors"))
    bin_files = glob.glob(os.path.join(folder, "*.bin"))
    weight_files = st_files if len(st_files) > 0 else bin_files

    if len(weight_files) == 0:
        raise FileNotFoundError(
            f"No weight shards found in {folder}. "
            f"Expected *.safetensors or *.bin."
        )

    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    out_dir = os.path.join(path, f"{model_name}-np")
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    for w_file in tqdm(weight_files, desc="Convert format"):
        if w_file.endswith(".safetensors"):
            if safetensors_load_file is None:
                raise ImportError(
                    "safetensors is required to load .safetensors weights. "
                    "Please install it: pip install safetensors"
                )
            state = safetensors_load_file(w_file, device="cpu")
        else:
            state = torch.load(w_file, map_location="cpu")

        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            param_path = os.path.join(out_dir, name)
            t = param.detach().cpu().contiguous()
            if t.dtype == torch.bfloat16:
                # Save exact bf16 bits as uint16 (numpy compatible)
                arr = t.view(torch.uint16).numpy()
                with open(param_path, "wb") as f:
                    np.save(f, arr)
                with open(param_path + ".dtype", "w") as f:
                    f.write("bfloat16")
            else:
                with open(param_path, "wb") as f:
                    np.save(f, t.numpy())
                with open(param_path + ".dtype", "w") as f:
                    f.write(str(t.dtype).replace("torch.", ""))