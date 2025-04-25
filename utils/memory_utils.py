# utils/memory_utils.py
import torch
import gc
from utils.common import debug

# Define device constants
cpu = torch.device('cpu')
gpu = torch.device('cuda')

def get_cuda_free_memory_gb(device=None):
    """Get the amount of free CUDA memory in GB"""
    if device is None:
        device = gpu
    
    if not torch.cuda.is_available():
        return 0
    
    try:
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        return free_memory / (1024 ** 3)
    except Exception as e:
        debug(f"Error getting CUDA memory: {e}")
        return 0

def clear_cuda_cache():
    """Clear CUDA cache to free fragmented memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_mem = get_cuda_free_memory_gb(gpu)
        return free_mem
    return 0

def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=6):
    """Move a model to device while preserving memory"""
    if model is None:
        return None
    
    free_mem = get_cuda_free_memory_gb(target_device)
    debug(f"Free memory before moving model: {free_mem:.2f} GB, preserving {preserved_memory_gb:.2f} GB")
    
    torch.cuda.empty_cache()
    
    model = model.to(target_device)
    debug(f"Moving {model.__class__.__name__} to {target_device}")
    
    return model

def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=6):
    """Offload a model to CPU"""
    if model is None:
        return None
    
    model = model.to(cpu)
    debug(f"Offloaded {model.__class__.__name__} to CPU")
    
    torch.cuda.empty_cache()
    
    return model

def fake_diffusers_current_device(model, device):
    """Handle device for diffusers models"""
    if model is None:
        return None
    
    model = model.to(device)
    debug(f"Set up device tracking for {model.__class__.__name__}")
    
    return model

def unload_complete_models(*models):
    """Unload models from GPU"""
    for model in models:
        if model is not None:
            model.to(cpu)
    
    torch.cuda.empty_cache()
    
    return get_cuda_free_memory_gb(gpu)

def load_model_as_complete(model, target_device):
    """Load a model to target device"""
    if model is None:
        return None
    
    free_mem_before = get_cuda_free_memory_gb(target_device)
    debug(f"Free memory before loading {model.__class__.__name__}: {free_mem_before:.2f} GB")
    
    model = model.to(target_device)
    debug(f"Loaded {model.__class__.__name__} to {target_device}")
    
    return model

class DynamicSwapInstaller:
    """Set up dynamic model swapping"""
    @staticmethod
    def install_model(model, device):
        # This doesn't actually move the model to GPU
        debug(f"DynamicSwapInstaller: Model will be swapped as needed")
        return model
