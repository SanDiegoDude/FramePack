# utils/lora_utils.py
# Adapted from https://github.com/neph1/FramePack/tree/main
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
from diffusers.loaders.lora_pipeline import _fetch_state_dict
from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING
from dataclasses import dataclass
from typing import Optional, List

def load_lora(transformer, lora_path: str, adapter_name: str):
    """
    Load a LoRA weights file into the transformer model under a unique adapter name.
    """
    from diffusers.loaders.lora_pipeline import _fetch_state_dict
    from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers
    state_dict = _fetch_state_dict(
        lora_path,
        "pytorch_lora_weights.safetensors",
        True, True, None, None, None, None, None, None, None, None
    )
    state_dict = _convert_hunyuan_video_lora_to_diffusers(state_dict)
    transformer.load_lora_adapter(state_dict, network_alphas=None, adapter_name=adapter_name)
    print(f"LoRA {adapter_name} loaded from {lora_path}")
    return transformer


@dataclass
class LoRAConfig:
    path: str
    weight: float = 1.0
    adapter_name: Optional[str] = None
    error: Optional[str] = None

def parse_lora_arg(lora_arg: str) -> List[LoRAConfig]:
    """
    Parse comma-separated LoRA list, with optional :weight per item.
    Supports Linux/Windows paths.
    Ex:
      --lora "/a/b/foo.safetensors:0.72,/c/bar.safetensors,/d/zzz.saf:1.2"
      => List of LoRAConfig(path, weight)
    """
    if not lora_arg:
        return []
    out: List[LoRAConfig] = []
    for entry in lora_arg.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry and not entry.endswith(":"):  # allow for paths with : in drive letter (Windows)
            # rsplit, so C:\foo\bar:0.8 works, and C:\foo\baz.safetensors works
            lpath, lweight = entry.rsplit(":", 1)
            try:
                weight = float(lweight)
            except Exception:
                lpath = entry
                weight = 1.0
        else:
            lpath, weight = entry, 1.0
        out.append(LoRAConfig(path=lpath.strip(), weight=weight))
    return out


def set_adapters(
        transformer,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,
    ):
    adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names
    # Expand weights into a list, one entry per adapter
    if not isinstance(weights, list):
        weights = [weights] * len(adapter_names)
    if len(adapter_names) != len(weights):
        raise ValueError(
            f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
        )
    # Set None values to default of 1.0
    weights = [w if w is not None else 1.0 for w in weights]
    # Get scale expansion function
    scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING["HunyuanVideoTransformer3DModel"]
    weights = scale_expansion_fn(transformer, weights)
    set_weights_and_activate_adapters(transformer, adapter_names, weights)

def safe_adapter_name(name):
    """
    Return a string safe for use as a PyTorch module name (adapter_name).
    Forbids dots/periods and weird chars.
    """
    # Remove periods, replace with underscore
    name = name.replace('.', '_')
    # Only allow [a-zA-Z0-9_-], replace other chars with _
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    return name


# === Multi-LoRA Support ===
def load_all_loras(transformer, lora_configs, skip_fail: bool = False):
    """
    Loads multiple LoRAs onto a transformer with unique adapter names per LoRA.
    Args:
        transformer: The model to attach lora adapters to
        lora_configs: List[LoRAConfig] objects
        skip_fail: If True, continues on failure, else raises on first error.
    Returns:
        (applied_configs, failed_configs) lists
    """
    loaded_adapter_names = []
    applied_configs = []
    failed_configs = []
    for idx, cfg in enumerate(lora_configs):
        # Create a unique adapter name for each LoRA
        base_name = os.path.splitext(os.path.basename(cfg.path))[0]
        safe_name = safe_adapter_name(base_name)
        adapter_name = f"lora_{idx}_{safe_name}"
        try:
            load_lora(transformer, cfg.path, adapter_name=adapter_name)
            cfg.adapter_name = adapter_name
            applied_configs.append(cfg)
            loaded_adapter_names.append(adapter_name)
        except Exception as e:
            cfg.error = str(e)
            failed_configs.append(cfg)
            if not skip_fail:
                raise RuntimeError(f"Failed to load LoRA '{cfg.path}': {e}")
    # Activate all at once
    if applied_configs:
        set_adapters(
            transformer,
            [c.adapter_name for c in applied_configs],
            [c.weight for c in applied_configs],
        )
    return applied_configs, failed_configs

def lora_diagnose(lora_path):
    """
    Loads a LoRA and prints its key/shape info, for debugging.
    """
    try:
        from diffusers.loaders.lora_pipeline import _fetch_state_dict
        state_dict = _fetch_state_dict(
            lora_path, "pytorch_lora_weights.safetensors",
            True, True, None, None, None, None, None, None, None, None)
        print(f"LoRA '{lora_path}' metadata:")
        for k in state_dict:
            print(f"  {k}: {state_dict[k].shape}")
        print("LoRA appears loadable and valid.")
    except Exception as e:
        print(f"Failed to load LoRA '{lora_path}': {e}")
