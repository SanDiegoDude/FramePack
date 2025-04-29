# utils/memory_utils.py
import torch
import gc
import time
from utils.common import debug, DEBUG
# Import original memory functions
from diffusers_helper.memory import (
    cpu, gpu, get_cuda_free_memory_gb as _original_get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation as _original_move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation as _original_offload_model_from_device_for_memory_preservation,
    fake_diffusers_current_device as _original_fake_diffusers_current_device,
    DynamicSwapInstaller, unload_complete_models as _original_unload_complete_models,
    load_model_as_complete as _original_load_model_as_complete
)

def get_cuda_free_memory_gb(device=None):
    """Get the amount of free CUDA memory in GB with enhanced debugging"""
    result = _original_get_cuda_free_memory_gb(device)
    debug(f"MEMORY: Free CUDA memory: {result:.2f} GB")
    return result

def clear_cuda_cache():
    """Clear CUDA cache with memory tracking"""
    before = get_cuda_free_memory_gb(gpu)
    debug(f"MEMORY: Before clear_cache: {before:.2f} GB free")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        after = _original_get_cuda_free_memory_gb(gpu)
        debug(f"MEMORY: After clear_cache: {after:.2f} GB free (freed {after-before:.2f} GB)")
        return after
    return 0

def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=6):
    if model is None:
        debug("MEMORY: Attempted to move None model")
        raise RuntimeError("Tried to move_model_to_device_with_memory_preservation with model=None")
    model_name = model.__class__.__name__
    # Only show message if not in debug mode
    if not DEBUG and target_device == gpu:
        print(f"Moving {model_name} to {target_device} with preserved memory: {preserved_memory_gb} GB")
    debug(f"MEMORY: ===== MOVING {model_name} TO {target_device} =====")
        
    # Get memory stats before
    free_mem_before = get_cuda_free_memory_gb(target_device)
    debug(f"MEMORY: Before moving {model_name}: {free_mem_before:.2f} GB free, preserving {preserved_memory_gb:.2f} GB")
    # Clear cache
    debug(f"MEMORY: Clearing cache before moving {model_name}")
    torch.cuda.empty_cache()
    gc.collect()
    # Get memory after clearing
    free_mem_after_clear = get_cuda_free_memory_gb(target_device)
    debug(f"MEMORY: After clearing cache: {free_mem_after_clear:.2f} GB free (freed {free_mem_after_clear-free_mem_before:.2f} GB)")
    # Track timing
    start_time = time.time()
    try:
        # Use original implementation
        debug(f"MEMORY: Calling original move_model_to_device for {model_name}")
        model = _original_move_model_to_device_with_memory_preservation(
            model, target_device, preserved_memory_gb
        )
        if model is None:
            debug(f"MEMORY: [FATAL] _original_move_model_to_device_with_memory_preservation returned None (model_name={model_name})")
            raise RuntimeError(f"_original_move_model_to_device_with_memory_preservation returned None for '{model_name}'!")
        # After moving, ensure ALL submodules (including LoRA) are now on the target device
        model.to(target_device)
        # Get memory after moving
        free_mem_after_move = get_cuda_free_memory_gb(target_device)
        used_mem = free_mem_after_clear - free_mem_after_move
        elapsed = time.time() - start_time
        debug(f"MEMORY: Moved {model_name} in {elapsed:.2f}s, used {used_mem:.2f} GB, {free_mem_after_move:.2f} GB remaining")
        # Check for potential memory leak
        if used_mem > 20:  # Unusually high memory usage
            debug(f"MEMORY: WARNING - High memory usage ({used_mem:.2f} GB) for {model_name}")
        return model
    except Exception as e:
        debug(f"MEMORY: ERROR moving {model_name} to {target_device}: {e}")
        # Try to diagnose
        debug(f"MEMORY: Attempting to diagnose OOM for {model_name}")
        try:
            debug(f"MEMORY: Model size estimate: {sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3):.2f} GB")
        except Exception:
            debug("MEMORY: Cannot print model size; model is None or missing parameters")
        debug(f"MEMORY: CUDA Memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        debug(f"MEMORY: CUDA Memory reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        # Try to recover
        try:
            debug(f"MEMORY: Attempting to return model to CPU")
            model = model.to(cpu)
        except:
            debug(f"MEMORY: Failed to return model to CPU")
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        raise

def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=6):
    """Offload a model with detailed memory tracking"""
    if model is None:
        debug("MEMORY: Attempted to offload None model")
        return None
        
    model_name = model.__class__.__name__
    
    # Only show message if not in debug mode
    if not DEBUG:
        print(f"Offloading {model_name} from {model.device if hasattr(model, 'device') else 'device'} to preserve memory: {preserved_memory_gb} GB")
        
    debug(f"MEMORY: ===== OFFLOADING {model_name} FROM GPU =====")
    # Get memory before offload
    free_mem_before = get_cuda_free_memory_gb(gpu)
    debug(f"MEMORY: Before offloading {model_name}: {free_mem_before:.2f} GB free")
    # Track timing
    start_time = time.time()
    try:
        # Use original implementation
        model = _original_offload_model_from_device_for_memory_preservation(
            model, target_device, preserved_memory_gb
        )
        # Get memory after offload
        free_mem_after = get_cuda_free_memory_gb(gpu)
        freed_mem = free_mem_after - free_mem_before
        elapsed = time.time() - start_time
        debug(f"MEMORY: Offloaded {model_name} in {elapsed:.2f}s, freed {freed_mem:.2f} GB, now {free_mem_after:.2f} GB free")
        return model
    except Exception as e:
        debug(f"MEMORY: ERROR offloading {model_name}: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        raise

def fake_diffusers_current_device(model, device):
    """Handle device for diffusers models with detailed tracking"""
    if model is None:
        debug("MEMORY: Attempted to set device for None model")
        return None
    model_name = model.__class__.__name__
    debug(f"MEMORY: Setting up device tracking for {model_name} to {device}")
    try:
        # Use original implementation
        model = _original_fake_diffusers_current_device(model, device)
        debug(f"MEMORY: Successfully set up device tracking for {model_name}")
        return model
    except Exception as e:
        debug(f"MEMORY: ERROR setting device for {model_name}: {e}")
        raise

def unload_complete_models(*models):
    """Unload models with detailed memory tracking"""
    debug(f"MEMORY: ===== UNLOADING {len(models)} MODELS =====")
    
    # Get list of model names for minimal non-debug output
    model_names = []
    for model in models:
        if model is not None:
            model_names.append(model.__class__.__name__)
    
    # Only show message if not in debug mode and we have models to unload
    if not DEBUG and model_names:
        print(f"Unloaded {', '.join(model_names)} as complete.")
        
    # Get memory before unload
    free_mem_before = get_cuda_free_memory_gb(gpu)
    debug(f"MEMORY: Before unloading: {free_mem_before:.2f} GB free")
    # Track each model
    for i, model in enumerate(models):
        if model is not None:
            model_name = model.__class__.__name__
            debug(f"MEMORY: Unloading model {i+1}/{len(models)}: {model_name}")
    try:
        # Use original implementation
        _original_unload_complete_models(*models)
        # Get memory after unload
        free_mem_after = get_cuda_free_memory_gb(gpu)
        freed_mem = free_mem_after - free_mem_before
        debug(f"MEMORY: Unloaded {len(models)} models, freed {freed_mem:.2f} GB, now {free_mem_after:.2f} GB free")
        return free_mem_after
    except Exception as e:
        debug(f"MEMORY: ERROR unloading models: {e}")
        torch.cuda.empty_cache()
        raise

def load_model_as_complete(model, target_device):
    """Load a model with detailed memory tracking"""
    if model is None:
        debug("MEMORY: Attempted to load None model")
        return None
        
    model_name = model.__class__.__name__
    
    # Only show message if not in debug mode
    if not DEBUG:
        print(f"Loaded {model_name} to {target_device} as complete.")
        
    debug(f"MEMORY: ===== LOADING {model_name} TO {target_device} =====")
    # Get memory before loading
    if target_device == gpu:
        free_mem_before = get_cuda_free_memory_gb(target_device)
        debug(f"MEMORY: Before loading {model_name}: {free_mem_before:.2f} GB free")
    # Track timing
    start_time = time.time()
    try:
        # Use original implementation
        model = _original_load_model_as_complete(model, target_device)
        # Get memory after loading
        if target_device == gpu:
            free_mem_after = get_cuda_free_memory_gb(target_device)
            used_mem = free_mem_before - free_mem_after
            elapsed = time.time() - start_time
            debug(f"MEMORY: Loaded {model_name} in {elapsed:.2f}s, used {used_mem:.2f} GB, {free_mem_after:.2f} GB remaining")
            # Check for potential memory leak
            if used_mem > 20:  # Unusually high memory usage
                debug(f"MEMORY: WARNING - High memory usage ({used_mem:.2f} GB) for {model_name}")
        else:
            elapsed = time.time() - start_time
            debug(f"MEMORY: Loaded {model_name} to {target_device} in {elapsed:.2f}s")
        return model
    except Exception as e:
        debug(f"MEMORY: ERROR loading {model_name} to {target_device}: {e}")
        torch.cuda.empty_cache()
        raise
