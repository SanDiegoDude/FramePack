# models/model_loader.py
import torch
import os
from utils.common import debug
from utils.memory_utils import (
    cpu, gpu, unload_complete_models, load_model_as_complete,
    DynamicSwapInstaller, fake_diffusers_current_device, get_cuda_free_memory_gb
)

class ModelManager:
    """Manages loading, unloading, and access to AI models"""
    def __init__(self, high_vram=False, lora_configs=None, lora_skip_fail=False):
        self.high_vram = high_vram
        self.lora_configs = lora_configs or []
        self.lora_skip_fail = lora_skip_fail
        self.active_loras = []
        self.failed_loras = []
        self.lora_path = lora_path
        self.lora_weight = lora_weight
        self.lora_name = None  # Will be set if LoRA is loaded
        debug(f"ModelManager initialized with high_vram={high_vram}")
        if lora_path:
            debug(f"LoRA path: {lora_path}, weight: {lora_weight}")
            
        # Model objects (existing code)
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

        def load_loras(self):
            """
            Load all requested LoRAs and activate their adapters.
            """
            if not self.lora_configs:
                return
            from utils.lora_utils import load_all_loras
            self.active_loras, self.failed_loras = load_all_loras(
                self.transformer,
                self.lora_configs,
                skip_fail=self.lora_skip_fail
            )
            # Diagnostic/verbose prints for CLI/log
            if self.active_loras:
                print("Loaded LoRAs: " + ", ".join([f"{c.path} (w={c.weight})" for c in self.active_loras]))
            if self.failed_loras:
                print("LoRAs failed to load:")
                for c in self.failed_loras:
                    print(f"  {c.path} reason: {c.error}")
    
    def load_all_models(self):
        """Load all required models based on VRAM configuration"""
        try:
            debug("Starting model loading process")
            # Load models from pretrained
            from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
            from diffusers import AutoencoderKLHunyuanVideo
            from transformers import SiglipImageProcessor, SiglipVisionModel
            from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
            
            # Load all models to CPU initially
            debug("Loading text encoders and tokenizers")
            self.text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
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
            
            # After loading the transformer model, load LoRA if specified
            if self.lora_path:
                try:
                    from utils.lora_utils import load_lora
                    debug(f"Loading LoRA from {self.lora_path}")
                    
                    # Extract filename to use as adapter name
                    _, filename = os.path.split(self.lora_path)
                    self.lora_name = filename.split('.')[0]
                    
                    # Load the LoRA adapter
                    self.transformer = load_lora(self.transformer, self.lora_path, filename)
                    debug(f"Successfully loaded LoRA: {self.lora_name}")
                except Exception as e:
                    debug(f"Error loading LoRA: {e}")
                    import traceback
                    debug(traceback.format_exc())
            
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
            
            # Handle device placement based on VRAM - like original code
            if not self.high_vram:
                debug("Low VRAM mode: Using dynamic swapping")
                # Use DynamicSwapInstaller from the original code
                DynamicSwapInstaller.install_model(self.transformer, device=gpu)
                DynamicSwapInstaller.install_model(self.text_encoder, device=gpu)
            else:
                debug("High VRAM mode: Moving all models to GPU")
                for m in [self.text_encoder, self.text_encoder_2, self.image_encoder, self.vae, self.transformer]:
                    if m is not None:
                        m.to(gpu)

            # Load LoRAs after transformer is ready
            if self.lora_configs:
                self.load_loras()
            self.models_loaded = True
            debug("All models loaded successfully")
            return True
            
        except Exception as e:
            debug(f"Error loading models: {e}")
            import traceback
            debug(traceback.format_exc())
            self.models_loaded = False
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

    def apply_lora_weight(self):
        """Apply LoRA weight to the loaded adapter"""
        if not self.lora_name or not hasattr(self, 'transformer') or self.transformer is None:
            return False
            
        try:
            from utils.lora_utils import set_adapters
            debug(f"Applying LoRA weight {self.lora_weight} to {self.lora_name}")
            set_adapters(self.transformer, self.lora_name, self.lora_weight)
            return True
        except Exception as e:
            debug(f"Error applying LoRA weight: {e}")
            import traceback
            debug(traceback.format_exc())
            return False
    
    def unload_all_models(self):
        """
        Unload all models completely from memory (both CPU and GPU)
        
        This clears all model references to fully free memory
        """
        try:
            # First move everything to CPU to free GPU memory
            if self.models_loaded:
                for model_attr in ['text_encoder', 'text_encoder_2', 'image_encoder', 'vae', 'transformer']:
                    if hasattr(self, model_attr) and getattr(self, model_attr) is not None:
                        try:
                            model = getattr(self, model_attr)
                            model.to('cpu')
                            debug(f"Moved {model_attr} to CPU")
                        except Exception as e:
                            debug(f"Error moving {model_attr} to CPU: {e}")
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Set all model references to None
            self.text_encoder = None
            self.text_encoder_2 = None 
            self.image_encoder = None
            self.vae = None
            self.transformer = None
            
            # Mark models as unloaded
            self.models_loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Report free memory
            free_mem = get_cuda_free_memory_gb(gpu)
            debug(f"All models completely unloaded. Free VRAM: {free_mem:.2f} GB")
            return True
        except Exception as e:
            debug(f"Error completely unloading models: {e}")
            import traceback
            debug(traceback.format_exc())
            return False
