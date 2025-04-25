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
    """
    Move a model to a device while ensuring a minimum amount of memory is preserved.
    This is critical for low-VRAM operation.
    """
    if model is None:
        return None
        
    if target_device == cpu:
        model.to(target_device)
        return model
    
    # Check available memory before moving
    free_mem = get_cuda_free_memory_gb(target_device)
    debug(f"Free memory before moving model: {free_mem:.2f} GB, preserving {preserved_memory_gb:.2f} GB")
    
    # Clear cache if we need more memory
    if free_mem < preserved_memory_gb + 2.0:  # Add buffer
        debug("Not enough memory - clearing CUDA cache")
        torch.cuda.empty_cache()
        gc.collect()
        free_mem = get_cuda_free_memory_gb(target_device)
        
    # Move model to device
    try:
        model.to(target_device)
        debug(f"Moving {model.__class__.__name__} to {target_device}")
        
        # Check memory after moving
        new_free_mem = get_cuda_free_memory_gb(target_device)
        used_mem = free_mem - new_free_mem
        debug(f"Model used {used_mem:.2f} GB, {new_free_mem:.2f} GB remaining")
        
        return model
    except Exception as e:
        debug(f"Error moving model to device: {e}")
        model.to(cpu)  # Make sure it's on CPU if GPU failed
        torch.cuda.empty_cache()
        return model

def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=6):
    """
    Offload a model from GPU to CPU to preserve memory
    """
    if model is None:
        return None
    
    # Monitor memory
    free_mem_before = get_cuda_free_memory_gb(gpu)
    debug(f"Free memory before offloading {model.__class__.__name__}: {free_mem_before:.2f} GB")
    
    # Move to CPU
    try:
        model.to(cpu)
        debug(f"Offloaded {model.__class__.__name__} to CPU")
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        
        # Check memory again
        free_mem_after = get_cuda_free_memory_gb(gpu)
        debug(f"After offload: {free_mem_after:.2f} GB free ({free_mem_after - free_mem_before:.2f} GB freed)")
        
        return model
    except Exception as e:
        debug(f"Error offloading model: {e}")
        return model

def fake_diffusers_current_device(model, device):
    """
    Special handling for HuggingFace models that track device separately from PyTorch
    
    Note: This does NOT move the model to GPU, it just sets up tracking
    """
    if model is None:
        return None
    
    # Don't actually move the model - just set device tracking
    try:
        # Store target device for later use, without changing properties
        if not hasattr(model, '_target_device'):
            setattr(model, '_target_device', device)
            
        # Use model's own device tracking if available
        if hasattr(model, 'set_device') and callable(getattr(model, 'set_device')):
            model.set_device(device)
            
        debug(f"Set up device tracking for {model.__class__.__name__}")
        return model
    except Exception as e:
        debug(f"Warning: Could not set up device tracking: {e}")
        return model

def unload_complete_models(*models):
    """
    Unload multiple models from GPU to free memory
    
    Args:
        *models: Models to unload
    """
    debug(f"Unloading {len(models)} models from GPU")
    
    for i, model in enumerate(models):
        if model is not None:
            try:
                model.to(cpu)
                debug(f"Moved {model.__class__.__name__} to CPU")
            except Exception as e:
                debug(f"Error moving model {i} to CPU: {e}")
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Report free memory
    free_mem = get_cuda_free_memory_gb(gpu)
    debug(f"After unloading, free VRAM: {free_mem:.2f} GB")
    
    return free_mem

def load_model_as_complete(model, target_device):
    """
    Load a specific model to target device
    
    This handles loading the model to device and reporting memory usage
    """
    if model is None:
        return None
    
    # Check memory before loading if loading to GPU
    if target_device == gpu:
        free_mem_before = get_cuda_free_memory_gb(target_device)
        debug(f"Free memory before loading {model.__class__.__name__}: {free_mem_before:.2f} GB")
    
    # Move model to target device
    try:
        model.to(target_device)
        debug(f"Loaded {model.__class__.__name__} to {target_device}")
        
        # Report memory after loading
        if target_device == gpu:
            free_mem_after = get_cuda_free_memory_gb(target_device)
            used_mem = free_mem_before - free_mem_after
            debug(f"Model used {used_mem:.2f} GB, {free_mem_after:.2f} GB remaining")
        
        return model
    except Exception as e:
        debug(f"Error loading model: {e}")
        model.to(cpu)  # Ensure it's on CPU if GPU failed
        return model

class DynamicSwapInstaller:
    """
    Marks models for dynamic swapping without actually moving them to GPU
    """
    @staticmethod
    def install_model(model, device):
        """
        Install a model for dynamic swapping
        
        Important: This does NOT actually move the model to GPU!
        It just sets up bookkeeping for later dynamic loading.
        """
        debug(f"DynamicSwapInstaller: Model will be swapped as needed")
        
        try:
            # Just tag the model - don't actually move it to GPU yet
            if not hasattr(model, '_dynamic_swap_target'):
                setattr(model, '_dynamic_swap_target', device)
                
            return model
        except Exception as e:
            debug(f"Warning: Dynamic swapping setup failed: {e}")
            return model
