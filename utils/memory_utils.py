# utils/memory_utils.py
import torch
import gc
import time
from utils.common import debug, DEBUG
# NOTE: Removed the direct LoraLayer import as it's not strictly needed for the hasattr checks
# from peft.tuners.lora.layer import LoraLayer
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
    """Move a model to device with detailed memory tracking, including PEFT adapters"""
    # debug(f"MEMORY: Pass to _original_move_model_to_device_with_memory_preservation: id={id(model)}, type={type(model)}") # Redundant with line below
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
        debug(f"MEMORY: Calling original move_model_to_device for {model_name}")
        # Use original implementation - THIS handles device placement for DynamicSwap models correctly.
        model = _original_move_model_to_device_with_memory_preservation(
            model, target_device, preserved_memory_gb
        )

        # --- BEGIN LoRA Device Handling ---
        # After moving the base model, ensure any attached PEFT LoRA adapters are also moved
        if target_device == gpu: # Only move LoRAs if the target is GPU
            debug(f"MEMORY: Ensuring PEFT LoRA adapters are on {target_device} for {model_name}")
            moved_lora_adapter_count = 0
            for module in model.modules():
                # Check if the module has LoRA layers attached (common PEFT pattern)
                if hasattr(module, 'lora_A') and isinstance(module.lora_A, torch.nn.ModuleDict):
                    for adapter_name in module.lora_A:
                        if module.lora_A[adapter_name].weight.device != target_device:
                            # debug(f"  - Moving LoRA A adapter '{adapter_name}' in {type(module).__name__} to {target_device}")
                            module.lora_A[adapter_name].to(target_device)
                            moved_lora_adapter_count += 1
                if hasattr(module, 'lora_B') and isinstance(module.lora_B, torch.nn.ModuleDict):
                     for adapter_name in module.lora_B:
                        if module.lora_B[adapter_name].weight.device != target_device:
                            # debug(f"  - Moving LoRA B adapter '{adapter_name}' in {type(module).__name__} to {target_device}")
                            module.lora_B[adapter_name].to(target_device)
                            moved_lora_adapter_count += 1
                # Add checks for other LoRA types if necessary (e.g., LoRA embeddings)
                if hasattr(module, 'lora_embedding_A') and isinstance(module.lora_embedding_A, torch.nn.ParameterDict):
                     for adapter_name in module.lora_embedding_A:
                        if module.lora_embedding_A[adapter_name].device != target_device:
                             module.lora_embedding_A[adapter_name] = torch.nn.Parameter(module.lora_embedding_A[adapter_name].to(target_device))
                             moved_lora_adapter_count += 1
                if hasattr(module, 'lora_embedding_B') and isinstance(module.lora_embedding_B, torch.nn.ParameterDict):
                     for adapter_name in module.lora_embedding_B:
                         if module.lora_embedding_B[adapter_name].device != target_device:
                             module.lora_embedding_B[adapter_name] = torch.nn.Parameter(module.lora_embedding_B[adapter_name].to(target_device))
                             moved_lora_adapter_count += 1

            if moved_lora_adapter_count > 0:
                debug(f"MEMORY: Moved {moved_lora_adapter_count} LoRA adapter components to {target_device}")
        # --- END LoRA Device Handling ---


        free_mem_after_move = get_cuda_free_memory_gb(target_device)
        used_mem = free_mem_after_clear - free_mem_after_move # Note: This might slightly underestimate usage now as LoRA move happens after measure
        elapsed = time.time() - start_time
        debug(f"MEMORY: Moved {model_name} (incl. LoRAs) in {elapsed:.2f}s, used approx {used_mem:.2f} GB, {free_mem_after_move:.2f} GB remaining")
        # Check for potential memory leak
        if used_mem > 20:  # Unusually high memory usage
            debug(f"MEMORY: WARNING - High memory usage ({used_mem:.2f} GB) for {model_name}")
        return model
    except Exception as e:
        debug(f"MEMORY: ERROR moving {model_name} to {target_device}: {e}")
        debug(f"MEMORY: Attempting to diagnose OOM for {model_name}")
        try: # Safely calculate model size
            model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
            debug(f"MEMORY: Model size estimate: {model_size_gb:.2f} GB")
        except Exception:
             debug("MEMORY: Could not estimate model size.")
        try: # Safely get memory stats
            debug(f"MEMORY: CUDA Memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
            debug(f"MEMORY: CUDA Memory reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        except Exception:
            debug("MEMORY: Could not get CUDA memory stats.")

        try:
            debug(f"MEMORY: Attempting to return model to CPU")
            model = model.to(cpu) # Try moving back to CPU
            # Also try moving adapters back
            for module in model.modules():
                 if hasattr(module, 'lora_A') and isinstance(module.lora_A, torch.nn.ModuleDict):
                     for adapter_name in module.lora_A: module.lora_A[adapter_name].to(cpu)
                 if hasattr(module, 'lora_B') and isinstance(module.lora_B, torch.nn.ModuleDict):
                     for adapter_name in module.lora_B: module.lora_B[adapter_name].to(cpu)
                 # Add embeddings if needed
                 if hasattr(module, 'lora_embedding_A') and isinstance(module.lora_embedding_A, torch.nn.ParameterDict):
                     for adapter_name in module.lora_embedding_A: module.lora_embedding_A[adapter_name] = torch.nn.Parameter(module.lora_embedding_A[adapter_name].to(cpu))
                 if hasattr(module, 'lora_embedding_B') and isinstance(module.lora_embedding_B, torch.nn.ParameterDict):
                     for adapter_name in module.lora_embedding_B: module.lora_embedding_B[adapter_name] = torch.nn.Parameter(module.lora_embedding_B[adapter_name].to(cpu))
        except Exception as offload_e:
            debug(f"MEMORY: Failed to return model/LoRAs fully to CPU: {offload_e}")

        torch.cuda.empty_cache()
        gc.collect()
        raise

def offload_model_from_device_for_memory_preservation(model, offload_target_device, preserved_memory_gb=6):
    """Offload a model with detailed memory tracking, including PEFT adapters"""
    if model is None:
        debug("MEMORY: Attempted to offload None model")
        return None

    model_name = model.__class__.__name__
    try:
        current_device = next(model.parameters()).device # Get current device
    except StopIteration:
        debug(f"MEMORY: Model {model_name} has no parameters, cannot determine current device.")
        current_device = cpu # Assume CPU if no parameters

    # Only show message if not in debug mode
    if not DEBUG:
        print(f"Offloading {model_name} from {current_device} to CPU to preserve memory")

    debug(f"MEMORY: ===== OFFLOADING {model_name} FROM {current_device} =====")
    if current_device.type == 'cuda':
        free_mem_before = get_cuda_free_memory_gb(current_device)
        debug(f"MEMORY: Before offloading {model_name}: {free_mem_before:.2f} GB free")
    else:
        free_mem_before = 0 # Cannot track GPU memory if starting on CPU

    start_time = time.time()
    try:
        model_after_offload = _original_offload_model_from_device_for_memory_preservation(
            model, offload_target_device, preserved_memory_gb
        )

        # --- BEGIN LoRA Offload Handling ---
        if model_after_offload is None:
            debug(f"MEMORY: Base model became None after original offload call for {model_name}, skipping LoRA offload.")
        else:
            # Now it's safe to proceed with model_after_offload
            debug(f"MEMORY: Ensuring PEFT LoRA adapters are on cpu for {model_name}")
            offloaded_lora_adapter_count = 0
            # Iterate using the potentially modified model object returned by the original function
            for module in model_after_offload.modules():
                if hasattr(module, 'lora_A') and isinstance(module.lora_A, torch.nn.ModuleDict):
                    for adapter_name in module.lora_A:
                        if module.lora_A[adapter_name].weight.device != cpu:
                            module.lora_A[adapter_name].to(cpu)
                            offloaded_lora_adapter_count += 1
                if hasattr(module, 'lora_B') and isinstance(module.lora_B, torch.nn.ModuleDict):
                     for adapter_name in module.lora_B:
                        if module.lora_B[adapter_name].weight.device != cpu:
                            module.lora_B[adapter_name].to(cpu)
                            offloaded_lora_adapter_count += 1
                # Add checks for other LoRA types if necessary (e.g., LoRA embeddings)
                if hasattr(module, 'lora_embedding_A') and isinstance(module.lora_embedding_A, torch.nn.ParameterDict):
                     for adapter_name in module.lora_embedding_A:
                         if module.lora_embedding_A[adapter_name].device != cpu:
                             module.lora_embedding_A[adapter_name] = torch.nn.Parameter(module.lora_embedding_A[adapter_name].to(cpu))
                             offloaded_lora_adapter_count += 1
                if hasattr(module, 'lora_embedding_B') and isinstance(module.lora_embedding_B, torch.nn.ParameterDict):
                     for adapter_name in module.lora_embedding_B:
                         if module.lora_embedding_B[adapter_name].device != cpu:
                             module.lora_embedding_B[adapter_name] = torch.nn.Parameter(module.lora_embedding_B[adapter_name].to(cpu))
                             offloaded_lora_adapter_count += 1
            if offloaded_lora_adapter_count > 0:
                 debug(f"MEMORY: Offloaded {offloaded_lora_adapter_count} LoRA adapter components to cpu")
        # --- END LoRA Offload Handling ---


        elapsed = time.time() - start_time
        if current_device.type == 'cuda':
            free_mem_after = get_cuda_free_memory_gb(current_device)
            freed_mem = free_mem_after - free_mem_before
            debug(f"MEMORY: Offloaded {model_name} (incl. LoRAs) in {elapsed:.2f}s, freed approx {freed_mem:.2f} GB, now {free_mem_after:.2f} GB free")
        else:
             debug(f"MEMORY: Offloaded {model_name} (incl. LoRAs) from {current_device} in {elapsed:.2f}s")

        # Return the result from the original function (which might be None)
        return model_after_offload
    except Exception as e:
        debug(f"MEMORY: ERROR offloading {model_name}: {e}")
        if current_device.type == 'cuda':
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
