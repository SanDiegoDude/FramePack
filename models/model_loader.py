# models/model_loader.py
import torch
import os
from utils.common import debug
from tqdm import tqdm # Add tqdm for progress bars
from utils.memory_utils import (
    cpu, gpu, unload_complete_models, load_model_as_complete,
    DynamicSwapInstaller, fake_diffusers_current_device, get_cuda_free_memory_gb,
    clear_cuda_cache # Import clear_cuda_cache
)
# Import LoRA utilities
from utils.lora_utils import LoRAConfig, load_lora, set_adapters, safe_adapter_name
from typing import List, Tuple

class ModelManager:
    """Manages loading, unloading, and access to AI models"""
    def __init__(self, high_vram=False, lora_configs=None, lora_skip_fail=False):
        self.high_vram = high_vram
        self.lora_skip_fail = lora_skip_fail
        debug(f"ModelManager initialized with high_vram={high_vram}")

        # Command-line LoRAs (loaded at startup)
        self.cli_lora_configs: List[LoRAConfig] = lora_configs or []
        self.applied_cli_loras: List[LoRAConfig] = [] # Track successfully applied CLI LoRAs
        self.failed_cli_loras: List[LoRAConfig] = []  # Track failed CLI LoRAs

        # Dynamic LoRAs (loaded via prompt)
        self.dynamic_lora_configs: List[LoRAConfig] = [] # Track currently active dynamic LoRAs
        self.failed_dynamic_loras: List[LoRAConfig] = [] # Track failed dynamic LoRAs from last attempt

        # Combined active LoRAs (used for setting adapters)
        self._active_loras: List[LoRAConfig] = [] # Internal combined list

        # Model objects
        self.text_encoder = None
        self.text_encoder_2 = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.vae = None
        self.feature_extractor = None
        self.image_encoder = None
        self.transformer = None

        # Model loading status
        self.models_loaded = False

    def _update_active_loras(self):
        """Updates the internal _active_loras list from CLI and dynamic sources."""
        self._active_loras = self.applied_cli_loras + self.dynamic_lora_configs
        debug(f"Updated _active_loras: {len(self._active_loras)} total active adapters.")
        # You could add more detailed logging here if needed

    def _get_active_adapter_names(self) -> List[str]:
        """Returns a list of adapter names currently considered active."""
        return [lora.adapter_name for lora in self._active_loras if lora.adapter_name]

    def load_cli_loras(self):
        """
        Load Command-Line Interface (CLI) specified LoRAs during initial setup.
        """
        if not self.cli_lora_configs:
            debug("No CLI LoRAs specified to load.")
            return

        if not self.transformer:
            debug("Cannot load CLI LoRAs: Transformer model not loaded yet.")
            return

        debug(f"Attempting to load {len(self.cli_lora_configs)} CLI LoRAs...")
        applied_count = 0
        failed_count = 0

        # Make sure transformer is on CPU for initial LoRA loading
        original_device = next(self.transformer.parameters()).device
        if original_device != cpu:
            debug(f"Temporarily moving transformer to CPU for CLI LoRA loading (currently on {original_device})")
            self.transformer.to(cpu)

        # Determine if we need a progress bar
        configs_to_load = self.cli_lora_configs
        use_tqdm = len(configs_to_load) >= 1
        iterable = tqdm(configs_to_load, desc="Loading CLI LoRAs", ncols=100, disable=not use_tqdm)
        
        for idx, cfg in enumerate(iterable): # Iterate using tqdm wrapper
            base_name = os.path.splitext(os.path.basename(cfg.path))[0]
            safe_name = safe_adapter_name(base_name)
            adapter_name = f"cli_{idx}_{safe_name}" # Prefix to distinguish
            cfg.adapter_name = adapter_name # Assign unique name
        
            # Update tqdm description if used
            if use_tqdm:
                 iterable.set_description(f"Loading CLI LoRA: {os.path.basename(cfg.path)[:25]}...")
        
            try:
                debug(f"Loading CLI LoRA: '{cfg.path}' as '{adapter_name}'")
                load_lora(self.transformer, cfg.path, adapter_name=adapter_name)
        
                # --- USER VISIBLE PRINT ---
                print(f"✅ Loaded CLI LoRA: '{os.path.basename(cfg.path)}' as adapter '{adapter_name}'")
                # --------------------------
        
                self.applied_cli_loras.append(cfg)
                applied_count += 1
            except Exception as e:
                cfg.error = str(e)
                self.failed_cli_loras.append(cfg)
                failed_count += 1
                error_msg = f"CLI LoRA Load Failed: Could not load '{os.path.basename(cfg.path)}' as '{adapter_name}'. Reason: {e}"
                # --- USER VISIBLE PRINT (Error) ---
                print(f"❌ Failed to load CLI LoRA: '{os.path.basename(cfg.path)}'. Reason: {e}")
                # ----------------------------------
                debug(f"[WARN] {error_msg}") # Keep debug for details
                if not self.lora_skip_fail:
                    debug("[ERROR] Stopping due to CLI LoRA load failure (--lora-skip-fail not set).")
                    if original_device != cpu:
                        debug(f"Moving transformer back to {original_device} before raising error.")
                        self.transformer.to(original_device)
                    raise RuntimeError(error_msg)
                else:
                    debug("[WARN] Skipping failed CLI LoRA as --lora-skip-fail is set.")
        
        # Clear tqdm description line if it was used
        if use_tqdm:
            iterable.set_description("Finished loading CLI LoRAs")
            iterable.close() # Ensure tqdm cleans up properly
        
        debug(f"Finished loading CLI LoRAs: {applied_count} succeeded, {failed_count} failed.")

        # Update the combined active list
        self._update_active_loras()

        # Activate the combined set of adapters
        if self._active_loras:
             debug(f"Setting adapters for {len(self._active_loras)} total LoRAs (CLI only at this stage).")
             set_adapters(
                 self.transformer,
                 [c.adapter_name for c in self._active_loras if c.adapter_name],
                 [c.weight for c in self._active_loras if c.adapter_name],
             )
        else:
             debug("No LoRAs active after CLI loading attempt.")

        # Move transformer back to its original device or intended device based on VRAM mode
        target_device = gpu if self.high_vram else cpu
        if self.transformer.device != target_device:
             debug(f"Moving transformer to target device: {target_device}")
             self.transformer.to(target_device)
        debug(f"Transformer final device after CLI LoRA load: {self.transformer.device}")

        # Diagnostic prints for CLI/log
        if self.applied_cli_loras:
            print("Applied CLI LoRAs: " + ", ".join([f"{os.path.basename(c.path)} (w={c.weight}, name={c.adapter_name})" for c in self.applied_cli_loras]))
        if self.failed_cli_loras:
             print("--- CLI LoRAs Failed to Load Summary ---")
             for c in self.failed_cli_loras:
                 print(f"  - {os.path.basename(c.path)} (Reason: {c.error})")
             print("-----------------------------------------")

    def set_dynamic_loras(self, requested_configs: List[LoRAConfig]) -> Tuple[List[LoRAConfig], List[LoRAConfig]]:
        """
        Manages dynamically loaded LoRAs based on prompt requests.
        Compares requested LoRAs with current dynamic LoRAs, unloads unused, loads new.

        Args:
            requested_configs: List of LoRAConfig objects extracted from the prompt.

        Returns:
            Tuple: (list of successfully applied/kept dynamic LoRAs, list of failed dynamic LoRAs)
        """
        if not self.transformer:
            debug("Cannot set dynamic LoRAs: Transformer not loaded.")
            return [], requested_configs # Return all as failed if no transformer

        debug(f"Setting dynamic LoRAs. Requested: {len(requested_configs)}")
        current_dynamic_configs = self.dynamic_lora_configs
        current_dynamic_paths = {cfg.path: cfg for cfg in current_dynamic_configs}
        requested_paths = {cfg.path: cfg for cfg in requested_configs}

        loras_to_unload = []
        loras_to_load = []
        loras_to_keep = [] # Keep track of those staying, might need weight update
        currently_applied_dynamic = []
        failed_dynamic = []

        # --- Identify changes ---
        # Check currently loaded dynamic LoRAs
        for path, current_cfg in current_dynamic_paths.items():
            if path not in requested_paths:
                loras_to_unload.append(current_cfg)
                debug(f"  - Unload requested: {os.path.basename(path)} (adapter: {current_cfg.adapter_name})")
            else:
                # Exists in both, check if weight changed
                requested_cfg = requested_paths[path]
                if current_cfg.weight != requested_cfg.weight:
                    debug(f"  - Weight change for {os.path.basename(path)}: {current_cfg.weight} -> {requested_cfg.weight}")
                    # Update weight in the config we'll keep
                    current_cfg.weight = requested_cfg.weight
                loras_to_keep.append(current_cfg) # Keep this one

        # Check requested LoRAs
        for path, requested_cfg in requested_paths.items():
             # Handle configs that failed normalization (don't try to load)
            if requested_cfg.error:
                debug(f"  - Skipping load for '{path}' due to previous error: {requested_cfg.error}")
                failed_dynamic.append(requested_cfg)
                continue # Skip to next requested config

            # If not already loaded dynamically, mark for loading
            if path not in current_dynamic_paths:
                loras_to_load.append(requested_cfg)
                debug(f"  - Load requested: {os.path.basename(path)} (weight: {requested_cfg.weight})")


        # --- Perform Unloading ---
        if loras_to_unload:
            debug(f"Unloading {len(loras_to_unload)} dynamic LoRAs...")
            original_device = next(self.transformer.parameters()).device
            if original_device != cpu: # PEFT might require CPU for delete? Be safe.
                debug("Moving transformer to CPU for adapter deletion.")
                self.transformer.to(cpu)

            for cfg in loras_to_unload:
                try:
                    if cfg.adapter_name:
                        debug(f"Deleting adapter: {cfg.adapter_name}")
                        self.transformer.delete_adapter(cfg.adapter_name)
                    else:
                         debug(f"Cannot delete LoRA for {cfg.path}, adapter name missing.")
                except Exception as e:
                    debug(f"Error deleting adapter {cfg.adapter_name} for {cfg.path}: {e}")
            # Move back if needed
            if original_device != cpu:
                self.transformer.to(original_device)
            debug("Finished unloading dynamic LoRAs.")


        # --- Perform Loading ---
        newly_loaded_loras = []
        if loras_to_load:
            debug(f"Loading {len(loras_to_load)} new dynamic LoRAs...")
            original_device = next(self.transformer.parameters()).device
            if original_device != cpu: # Move to CPU for loading consistency
                debug("Moving transformer to CPU for new dynamic LoRA loading.")
                self.transformer.to(cpu)

            # Inside set_dynamic_loras, within 'if loras_to_load:'
            newly_loaded_loras = [] # Initialize here
            
            # Determine if we need a progress bar for *new* loads
            use_tqdm = len(loras_to_load) >= 1
            iterable = tqdm(loras_to_load, desc="Loading Dynamic LoRAs", ncols=100, disable=not use_tqdm)
            
            for idx, cfg in enumerate(iterable): # Iterate using tqdm wrapper
                # Create a unique name
                base_name = os.path.splitext(os.path.basename(cfg.path))[0]
                safe_name = safe_adapter_name(base_name)
                adapter_name = f"dyn_{safe_name}_{hash(cfg.path) % 10000}"
                cfg.adapter_name = adapter_name
            
                # Update tqdm description if used
                if use_tqdm:
                     iterable.set_description(f"Loading Dynamic LoRA: {os.path.basename(cfg.path)[:25]}...")
            
                # Handle configs that failed normalization (don't try to load)
                if cfg.error:
                    debug(f"  - Skipping load for '{cfg.path}' due to previous error: {cfg.error}")
                    # Add to failed_dynamic here directly if not already done
                    if cfg not in failed_dynamic: # Avoid duplicates if error was set earlier
                         failed_dynamic.append(cfg)
                    # --- USER VISIBLE PRINT (Pre-fail) ---
                    # Optional: Print pre-existing failures? Could be noisy.
                    # print(f"⚠️ Skipping Dynamic LoRA '{os.path.basename(cfg.path)}': {cfg.error}")
                    # -------------------------------------
                    continue # Skip to next requested config
            
                try:
                    debug(f"Loading dynamic LoRA: '{cfg.path}' as '{adapter_name}'")
                    load_lora(self.transformer, cfg.path, adapter_name=adapter_name)
            
                    # --- USER VISIBLE PRINT ---
                    print(f"✅ Loaded Dynamic LoRA: '{os.path.basename(cfg.path)}' as adapter '{adapter_name}' (Weight: {cfg.weight})")
                    # --------------------------
            
                    newly_loaded_loras.append(cfg)
                except Exception as e:
                    cfg.error = str(e)
                    failed_dynamic.append(cfg)
                    error_msg = f"Dynamic LoRA Load Failed: Could not load '{os.path.basename(cfg.path)}' as '{adapter_name}'. Reason: {e}"
                    # --- USER VISIBLE PRINT (Error) ---
                    print(f"❌ Failed to load Dynamic LoRA: '{os.path.basename(cfg.path)}'. Reason: {e}")
                    # ----------------------------------
                    debug(f"[WARN] {error_msg}") # Keep debug for details
                    # No need to raise RuntimeError here, it's handled after the loop
            
            # Clear tqdm description line if it was used
            if use_tqdm:
                iterable.set_description("Finished loading dynamic LoRAs")
                iterable.close()
            
            # --- END Perform Loading ---

            # Move back to original device
            if original_device != cpu:
                 debug(f"Moving transformer back to {original_device} after loading.")
                 self.transformer.to(original_device)
            debug("Finished loading new dynamic LoRAs.")


        # --- Update State and Apply Adapters ---
        # New set of dynamic configs = kept ones + newly loaded ones
        self.dynamic_lora_configs = loras_to_keep + newly_loaded_loras
        self.failed_dynamic_loras = failed_dynamic # Store failures from this run

        # Update the combined active list
        self._update_active_loras()

        # Re-apply *all* currently active adapters (CLI + dynamic) with potentially updated weights
        if self._active_loras:
            adapter_names = [c.adapter_name for c in self._active_loras if c.adapter_name]
            adapter_weights = [c.weight for c in self._active_loras if c.adapter_name]
            debug(f"Applying {len(adapter_names)} total adapters (CLI + Dynamic) with weights: {adapter_weights}")
            # Ensure transformer is on the correct target device before setting adapters
            target_device = gpu if self.high_vram else cpu
            if self.transformer.device != target_device:
                debug(f"Ensuring transformer is on {target_device} before set_adapters.")
                self.transformer.to(target_device)

            set_adapters(self.transformer, adapter_names, adapter_weights)
            debug("Adapters set successfully.")
        else:
            # If no LoRAs are active, ensure PEFT knows (might need explicit disabling)
            try:
                # Use PEFT's way to disable adapters if possible, otherwise this might be implicit
                debug("No active LoRAs. Attempting to disable all adapters.")
                self.transformer.disable_adapter_layers()
                self.transformer.enable_adapter_layers() # Re-enable base layers? Check PEFT docs. Or maybe set_adapters with empty lists works?
                # Safest might be:
                # set_adapters(self.transformer, [], []) # Call with empty lists
            except Exception as e:
                 debug(f"Note: Could not explicitly disable adapters (might be okay): {e}")


        # --- Final Check for Failures ---
        # After attempting to load all requested dynamic LoRAs, check if any failed
        if failed_dynamic and not self.lora_skip_fail:
             # ... (error message creation) ...
             combined_error_msg = f"Failed to apply {len(failed_dynamic)} required dynamic LoRA(s): {', '.join([f'{os.path.basename(c.path)} ({c.error})' for c in failed_dynamic])}" # Simplified formatting
             debug(f"[ERROR] {combined_error_msg} -- Halting generation as --lora-skip-fail is not set.")
             
             # --- Ensure Correct Device Before Raising ---
             # Move to intended final device *before* raising the error to leave state consistent
             final_target_device = gpu if self.high_vram else cpu
             if hasattr(self.transformer, 'device') and self.transformer.device != final_target_device:
                 debug(f"Ensuring transformer is on {final_target_device} before raising LoRA error.")
                 try:
                     self.transformer.to(final_target_device)
                 except Exception as move_e:
                      debug(f"Could not move transformer to {final_target_device} before raising error: {move_e}")
             # --- End Ensure Device Before Raising ---

             raise RuntimeError(combined_error_msg) # Raise the error

        # --- Apply Adapters (if successful or skipping failures) ---
        # Update the combined active list (should happen regardless of failure if skipping)
        self._update_active_loras()

        # Re-apply *all* currently active adapters (CLI + dynamic) with potentially updated weights
        if self._active_loras:
            adapter_names = [c.adapter_name for c in self._active_loras if c.adapter_name]
            adapter_weights = [c.weight for c in self._active_loras if c.adapter_name]
            debug(f"Applying {len(adapter_names)} total adapters (CLI + Dynamic) with weights: {adapter_weights}")
            
            # Ensure transformer is on the correct target device before setting adapters
            target_device = gpu if self.high_vram else cpu # Re-calculate target here
            if not hasattr(self.transformer, 'device') or self.transformer.device != target_device:
                 debug(f"Ensuring transformer is on {target_device} before set_adapters.")
                 try: # Add try-except for robustness
                      self.transformer.to(target_device)
                 except Exception as move_e:
                      debug(f"ERROR moving transformer to {target_device} before set_adapters: {move_e}")
                      # If move fails here, set_adapters will likely fail too, but let it try
            
            try: # Add try-except for robustness
                set_adapters(self.transformer, adapter_names, adapter_weights)
                debug("Adapters set successfully.")
            except Exception as set_adapter_e:
                 debug(f"ERROR setting adapters: {set_adapter_e}")
                 # Decide how to handle this - maybe raise another error?
                 # For now, just log it. Generation might fail later.

        else:
            # If no LoRAs are active, attempt to disable adapters
            try:
                debug("No active LoRAs. Attempting to disable adapter layers.")
                if hasattr(self.transformer, 'disable_adapter_layers'):
                    self.transformer.disable_adapter_layers()
                # Maybe need transformer.enable_adapter_layers() here if disable affects base weights? Test needed.
            except Exception as e:
                 debug(f"Note: Could not explicitly disable adapters (might be okay): {e}")

        # --- Final Device Placement ---
        # ENSURE the transformer is on the correct device *before returning* to generate_video
        final_target_device = gpu if self.high_vram else cpu
        if hasattr(self.transformer, 'device') and self.transformer.device != final_target_device:
             debug(f"Final check: Moving transformer to target device {final_target_device} before returning.")
             try:
                  self.transformer.to(final_target_device)
             except Exception as move_e:
                  debug(f"ERROR during final move to {final_target_device}: {move_e}")
                  # If this fails, subsequent steps in generate_video will likely fail.
        elif not hasattr(self.transformer, 'device'):
             debug("Transformer has no device attribute at final check.") # Should not happen if loaded
        else:
             debug(f"Transformer already on final target device {final_target_device}.")


        debug(f"Dynamic LoRA update complete. Active dynamic: {len(self.dynamic_lora_configs)}, Failed this run: {len(failed_dynamic)}")
        # Return the lists of applied and failed dynamic LoRAs for this run
        return self.dynamic_lora_configs, self.failed_dynamic_loras

    def ensure_all_models_loaded(self):
        """If any required model is None, reload all models (auto-heals from accidental None)."""
        required_attrs = [
            "text_encoder", "text_encoder_2", "tokenizer",
            "tokenizer_2", "vae", "feature_extractor",
            "image_encoder", "transformer"
        ]
        missing = [name for name in required_attrs if getattr(self, name, None) is None]
        if missing:
            debug(f"[AutoReload] ModelManager reloading all models due to missing: {missing}")
            # Always reload all at once
            self.load_all_models() # This should also reload CLI LoRAs
        # Recheck
        still_missing = [name for name in required_attrs if getattr(self, name, None) is None]
        if still_missing:
            debug(f"[FATAL] After reload, still missing: {still_missing}")
            raise RuntimeError(f"Failed to reload required models: {still_missing}")

    def load_all_models(self):
        """Load all required models based on VRAM configuration"""
        try:
            debug("Starting model loading process")
            # --- Model Loading Code (existing, unchanged) ---
            from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
            from diffusers import AutoencoderKLHunyuanVideo
            from transformers import SiglipImageProcessor, SiglipVisionModel
            from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

            # Load all models to CPU initially
            debug("Loading text encoders and tokenizers")
            self.text_encoder = LlamaModel.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='text_encoder',
                torch_dtype=torch.float16
            ).cpu()
            self.text_encoder_2 = CLIPTextModel.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='text_encoder_2',
                torch_dtype=torch.float16
            ).cpu()

            self.tokenizer = LlamaTokenizerFast.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='tokenizer'
            )

            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='tokenizer_2'
            )

            debug("Loading VAE")
            self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='vae',
                torch_dtype=torch.float16
            ).cpu()

            debug("Loading image encoder components")
            self.feature_extractor = SiglipImageProcessor.from_pretrained(
                "lllyasviel/flux_redux_bfl",
                subfolder='feature_extractor'
            )

            self.image_encoder = SiglipVisionModel.from_pretrained(
                "lllyasviel/flux_redux_bfl",
                subfolder='image_encoder',
                torch_dtype=torch.float16
            ).cpu()

            debug("Loading transformer model")
            self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                'lllyasviel/FramePackI2V_HY',
                torch_dtype=torch.bfloat16
            ).cpu()
             # --- End Model Loading Code ---

            # Configure models
            debug("Configuring models")
            for m in [self.vae, self.text_encoder, self.text_encoder_2, self.image_encoder, self.transformer]:
                if m is not None:
                    m.eval()

            # Configure VAE for low VRAM
            if not self.high_vram:
                self.vae.enable_slicing()
                self.vae.enable_tiling()

            # Configure transformer for high quality output
            self.transformer.high_quality_fp32_output_for_inference = True
            debug('transformer.high_quality_fp32_output_for_inference = True')

            # Set model dtypes
            self.transformer.to(dtype=torch.bfloat16)
            self.vae.to(dtype=torch.float16)
            self.image_encoder.to(dtype=torch.float16)
            self.text_encoder.to(dtype=torch.float16)
            self.text_encoder_2.to(dtype=torch.float16)

            # Make sure no gradients are computed
            for m in [self.vae, self.text_encoder, self.text_encoder_2, self.image_encoder, self.transformer]:
                if m is not None:
                    m.requires_grad_(False)

            # Handle device placement based on VRAM
            if not self.high_vram:
                debug("Low VRAM mode: Using dynamic swapping")
                DynamicSwapInstaller.install_model(self.transformer, device=gpu)
                DynamicSwapInstaller.install_model(self.text_encoder, device=gpu)
                # Add others if necessary for swapping
                # DynamicSwapInstaller.install_model(self.text_encoder_2, device=gpu)
                # DynamicSwapInstaller.install_model(self.image_encoder, device=gpu)
                # DynamicSwapInstaller.install_model(self.vae, device=gpu)
            else:
                debug("High VRAM mode: Moving all models to GPU")
                # Move base models first
                for m in [self.text_encoder, self.text_encoder_2, self.image_encoder, self.vae, self.transformer]:
                    if m is not None:
                        m.to(gpu)

            # Load CLI LoRAs after transformer is ready and on its initial device
            if self.cli_lora_configs:
                 self.load_cli_loras() # This handles moving transformer and setting adapters

            self.models_loaded = True
            debug("All models loaded successfully")
            # Final check on transformer device
            expected_device = gpu if self.high_vram else cpu
            if self.transformer and self.transformer.device != expected_device:
                 debug(f"[WARN] Transformer device is {self.transformer.device}, expected {expected_device} after load_all_models.")
                 # Force move if needed, though load_cli_loras should handle it
                 # self.transformer.to(expected_device)
            return True

        except Exception as e:
            debug(f"Error loading models: {e}")
            import traceback
            debug(traceback.format_exc())
            self.models_loaded = False
            # Attempt cleanup
            self.unload_all_models()
            return False

    def initialize_teacache(self, enable_teacache=True, num_steps=0):
        """Initialize transformer TeaCache setting"""
        if self.transformer is not None:
            self.transformer.initialize_teacache(
                enable_teacache=enable_teacache,
                num_steps=num_steps if enable_teacache else 0
            )
            debug(f"TeaCache initialized: enable_teacache={enable_teacache}, num_steps={num_steps}")
            return True
        else:
            debug("Cannot initialize TeaCache: transformer not loaded")
            return False

    # --- Updated unload_all_models ---
    def unload_all_models(self):
        """
        Unload all models completely from memory (both CPU and GPU),
        including deleting any attached LoRA adapters.
        """
        debug("--- Starting Full Model Unload ---")
        try:
            # 1. Delete LoRA Adapters from Transformer (if it exists)
            if hasattr(self, 'transformer') and self.transformer is not None:
                debug("Attempting to delete LoRA adapters from transformer...")
                original_device = cpu
                try:
                    # Check if transformer has parameters to get device
                    if next(self.transformer.parameters(), None) is not None:
                        original_device = next(self.transformer.parameters()).device
                        if original_device != cpu:
                            debug("Moving transformer to CPU before deleting adapters.")
                            self.transformer.to(cpu)
                    else:
                         debug("Transformer has no parameters, assuming CPU.")

                    # Get all potential adapter names from our tracked configs
                    all_adapter_names = set()
                    for cfg in self.applied_cli_loras + self.dynamic_lora_configs:
                        if cfg.adapter_name:
                            all_adapter_names.add(cfg.adapter_name)

                    if all_adapter_names:
                        debug(f"Found adapter names to delete: {list(all_adapter_names)}")
                        # Use PEFT's method to see currently loaded adapters
                        if hasattr(self.transformer, 'delete_adapter'):
                             # It's safer to check what PEFT thinks is loaded
                            loaded_adapters = []
                            if hasattr(self.transformer, 'active_adapters'):
                                loaded_adapters.extend(self.transformer.active_adapters)
                            if hasattr(self.transformer, 'adapters'): # Backup check
                                loaded_adapters.extend(list(self.transformer.adapters.keys()))

                            adapters_to_delete = set(loaded_adapters) & all_adapter_names
                            debug(f"Adapters PEFT knows about that we tracked: {list(adapters_to_delete)}")

                            for name in adapters_to_delete:
                                try:
                                    debug(f"Deleting adapter '{name}'...")
                                    self.transformer.delete_adapter(name)
                                    debug(f"Successfully deleted adapter '{name}'.")
                                except Exception as e:
                                    debug(f"Error deleting adapter '{name}': {e}")
                            # Also try disabling just in case delete leaves things active
                            if hasattr(self.transformer, 'disable_adapter_layers'):
                                self.transformer.disable_adapter_layers()
                        else:
                             debug("Transformer does not have 'delete_adapter' method. Skipping deletion.")
                    else:
                        debug("No tracked adapter names found to delete.")

                except Exception as e:
                    debug(f"Error during adapter deletion preparation: {e}")
                finally:
                    # Clear our tracking lists regardless of success
                    self.applied_cli_loras = []
                    self.dynamic_lora_configs = []
                    self._active_loras = []
                    debug("Cleared internal LoRA tracking lists.")
                    # No need to move transformer back here, it will be deleted next

            # 2. Move models to CPU (redundant if already done for adapters, but safe)
            if self.models_loaded:
                debug("Moving any remaining loaded models to CPU...")
                for model_attr in ['text_encoder', 'text_encoder_2', 'image_encoder', 'vae', 'transformer']:
                    model = getattr(self, model_attr, None)
                    if model is not None and hasattr(model, 'device') and model.device != cpu:
                        try:
                            model.to(cpu)
                            debug(f"Moved {model_attr} to CPU")
                        except Exception as e:
                            debug(f"Error moving {model_attr} to CPU during unload: {e}")

            # 3. Clear CUDA Cache (important before deleting refs)
            if torch.cuda.is_available():
                debug("Clearing CUDA cache...")
                clear_cuda_cache()

            # 4. Delete model references
            debug("Deleting model references...")
            self.text_encoder = None
            self.text_encoder_2 = None
            self.image_encoder = None
            self.vae = None
            self.transformer = None
            # Also clear tokenizers/processors
            self.tokenizer = None
            self.tokenizer_2 = None
            self.feature_extractor = None

            # Mark models as unloaded
            self.models_loaded = False

            # 5. Force Garbage Collection
            debug("Running garbage collection...")
            import gc
            gc.collect()

            # Report free memory
            free_mem = get_cuda_free_memory_gb(gpu) if torch.cuda.is_available() else 0
            debug(f"--- Full Model Unload Complete --- Free VRAM: {free_mem:.2f} GB")
            return True
        except Exception as e:
            debug(f"Error during full model unload: {e}")
            import traceback
            debug(traceback.format_exc())
            # Attempt partial cleanup
            self.models_loaded = False
            self.text_encoder = None
            self.text_encoder_2 = None
            # ... etc ...
            self.transformer = None
            self.applied_cli_loras = []
            self.dynamic_lora_configs = []
            self._active_loras = []
            return False
