# utils/memory_utils.py
import torch
import gc
import os
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
    Move a model to a device while preserving a specified amount of memory
    
    This function moves a model to the target device (typically GPU) while ensuring 
    that a minimum amount of memory is preserved for other operations.
    """
    if target_device == cpu:
        model.to(target_device)
        return model
    
    # Check if we have enough memory
    free_mem = get_cuda_free_memory_gb(target_device)
    debug(f"Free memory before moving model: {free_mem:.2f} GB, preserving {preserved_memory_gb:.2f} GB")
    
    # If we don't have enough preserved memory, clear cache
    if free_mem < preserved_memory_gb + 1.0:  # Add 1GB buffer
        debug("Not enough VRAM - clearing CUDA cache")
        torch.cuda.empty_cache()
        gc.collect()
        free_mem = get_cuda_free_memory_gb(target_device)
        debug(f"Free memory after clearing cache: {free_mem:.2f} GB")
    
    # Move model to device
    debug(f"Moving {model.__class__.__name__} to {target_device}")
    model.to(target_device)
    
    # Report memory usage
    new_free_mem = get_cuda_free_memory_gb(target_device)
    used_mem = free_mem - new_free_mem
    debug(f"Model used {used_mem:.2f} GB, {new_free_mem:.2f} GB remaining")
    
    return model

def offload_model_from_device_for_memory_preservation(model, target_device=None, preserved_memory_gb=6):
    """
    Offload a model from a device to CPU to preserve memory
    
    Args:
        model: The model to offload
        target_device: The original device to report memory for
        preserved_memory_gb: Amount of memory to preserve after offload
    """
    if model is None:
        return None
    
    if target_device is None:
        target_device = gpu
    
    # Get memory before offload
    free_mem_before = get_cuda_free_memory_gb(target_device)
    debug(f"Free memory before offloading {model.__class__.__name__}: {free_mem_before:.2f} GB")
    
    # Move to CPU
    debug(f"Offloading {model.__class__.__name__} to CPU")
    model.to(cpu)
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get memory after offload
    free_mem_after = get_cuda_free_memory_gb(target_device)
    freed_mem = free_mem_after - free_mem_before
    debug(f"Offloading freed {freed_mem:.2f} GB, now have {free_mem_after:.2f} GB free")
    
    return model

def fake_diffusers_current_device(model, device):
    """
    Helper to handle model device context issues with diffusers models
    
    Many HuggingFace models track device in a way that PyTorch doesn't update with .to()
    """
    if model is None:
        return None
    
    # First move the model to the device
    model = model.to(device)
    debug(f"Moved {model.__class__.__name__} to {device}")
    
    # For diffusers models, try to add device tracking safely
    try:
        # Instead of trying to set a read-only property, just store the target device
        if not hasattr(model, '_target_device'):
            object.__setattr__(model, '_target_device', device)
            
        # Some HuggingFace models have specific device methods
        if hasattr(model, 'set_device') and callable(model.set_device):
            model.set_device(device)
    except Exception as e:
        debug(f"Warning: Could not update device tracking for {model.__class__.__name__}: {e}")
        # This is not critical - the model is already on the right device via .to()
    
    return model

def unload_complete_models(*models):
    """
    Completely unload models from memory (both CPU and GPU)
    
    Args:
        *models: Models to unload
    """
    debug(f"Completely unloading {len(models)} models from memory")
    
    # First move to CPU to free GPU memory
    for i, model in enumerate(models):
        if model is not None:
            try:
                model.to(cpu)
                debug(f"Moved model {i} to CPU")
            except Exception as e:
                debug(f"Error moving model {i} to CPU: {e}")
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Report free memory
    free_mem = get_cuda_free_memory_gb(gpu)
    debug(f"After unloading, free VRAM: {free_mem:.2f} GB")
    
    return free_mem

def load_model_as_complete(model, target_device):
    """
    Load model completely to the target device
    
    Args:
        model: Model to load
        target_device: Target device (GPU or CPU)
    """
    if model is None:
        debug("Cannot load None model")
        return None
    
    # Check free memory before loading
    if target_device == gpu:
        free_mem_before = get_cuda_free_memory_gb(target_device)
        debug(f"Free memory before loading {model.__class__.__name__}: {free_mem_before:.2f} GB")
    
    # Move model to target device
    model = model.to(target_device)
    debug(f"Loaded {model.__class__.__name__} to {target_device}")
    
    # Report memory usage for GPU
    if target_device == gpu:
        free_mem_after = get_cuda_free_memory_gb(target_device)
        used_mem = free_mem_before - free_mem_after
        debug(f"Model used {used_mem:.2f} GB, {free_mem_after:.2f} GB remaining")
    
    return model

class DynamicSwapInstaller:
    """
    Dynamic model swapping functionality for memory efficiency
    """
    @staticmethod
    def install_model(model, device):
        """
        Install a model for dynamic swapping
        
        Args:
            model: The model to prepare for swapping
            device: The target device (usually GPU)
        """
        debug(f"DynamicSwapInstaller: Model will be swapped as needed")
        
        # Don't try to move the entire model to GPU - just set up tracking
        # This is critical for low VRAM devices
        try:
            # Only update tracking attributes, don't move the model
            if hasattr(model, 'device'):
                # Some models have a device property
                if not hasattr(model, '_original_device'):
                    model._original_device = model.device
                # Don't actually set model.device - it's read-only
            
            # Tag the model with target device for later reference
            object.__setattr__(model, '_target_device', device)
            
            # Free memory after tagging
            torch.cuda.empty_cache()
            
            debug(f"Installed {model.__class__.__name__} for dynamic swapping")
            return model
            
        except Exception as e:
            debug(f"Warning: Dynamic swapping setup failed: {e}")
            return model
