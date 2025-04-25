# models/model_loader.py
import torch
from utils.common import debug
from utils.memory_utils import (
    cpu, gpu, unload_complete_models, load_model_as_complete,
    DynamicSwapInstaller, fake_diffusers_current_device
)

class ModelManager:
    """Manages loading, unloading, and access to AI models"""
    
    def __init__(self, high_vram=False):
        self.high_vram = high_vram
        debug(f"ModelManager initialized with high_vram={high_vram}")
        
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
    
    def load_all_models(self):
        """Load all required models based on VRAM configuration"""
        try:
            debug("Starting model loading process")
            from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
            from diffusers import AutoencoderKLHunyuanVideo
            from transformers import SiglipImageProcessor, SiglipVisionModel
            from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
            
            # Load text encoders and tokenizers
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
            
            # Load VAE
            debug("Loading VAE")
            self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo", 
                subfolder='vae', 
                torch_dtype=torch.float16
            ).cpu()
            
            # Load image encoder components
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
            
            # Load transformer
            debug("Loading transformer model")
            self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                'lllyasviel/FramePackI2V_HY', 
                torch_dtype=torch.bfloat16
            ).cpu()
            
            # Configure models
            debug("Configuring models")
            for m in [self.vae, self.text_encoder, self.text_encoder_2, self.image_encoder, self.transformer]:
                if m is not None:
                    m.eval()
            
            # Configure VAE for low VRAM if needed
            if not self.high_vram:
                self.vae.enable_slicing()
                self.vae.enable_tiling()
            
            # Configure transformer for high quality output
            self.transformer.high_quality_fp32_output_for_inference = True
            debug('transformer.high_quality_fp32_output_for_inference = True')
            
            # Set model dtypes
            for m, dtype in zip(
                [self.transformer, self.vae, self.image_encoder, self.text_encoder, self.text_encoder_2], 
                [torch.bfloat16, torch.float16, torch.float16, torch.float16, torch.float16]
            ):
                if m is not None:
                    m.to(dtype=dtype)
                    m.requires_grad_(False)
            
            # Handle device placement based on VRAM
            if not self.high_vram:
                debug("Low VRAM mode: Using dynamic swapping")
                try:
                    DynamicSwapInstaller.install_model(self.transformer, device=gpu)
                    DynamicSwapInstaller.install_model(self.text_encoder, device=gpu)
                except Exception as e:
                    debug(f"Warning: Dynamic swapping setup failed: {e}")
                    debug("Falling back to standard CPU/GPU transfers")
            else:
                debug("High VRAM mode: Moving all models to GPU")
                for m in [self.text_encoder, self.text_encoder_2, self.image_encoder, self.vae, self.transformer]:
                    if m is not None:
                        m.to(gpu)
            
            self.models_loaded = True
            debug("All models loaded successfully")
            return True
            
        except Exception as e:
            debug(f"Error loading models: {e}")
            import traceback
            debug(traceback.format_exc())
            self.models_loaded = False
            return False
            
            self.models_loaded = True
            debug("All models loaded successfully")
            return True
            
        except Exception as e:
            debug(f"Error loading models: {e}")
            import traceback
            debug(traceback.format_exc())
            self.models_loaded = False
            return False
    
    def unload_all_models(self):
        """Unload all models from GPU to free memory"""
        try:
            if self.models_loaded:
                unload_complete_models(
                    self.text_encoder, 
                    self.text_encoder_2, 
                    self.image_encoder, 
                    self.vae, 
                    self.transformer
                )
                torch.cuda.empty_cache()
                debug("All models unloaded from GPU and memory cleared")
            return True
        except Exception as e:
            debug(f"Error unloading models: {e}")
            return False
    
    def load_model_to_device(self, model_name, target_device=None):
        """
        Load a specific model to the target device
        
        Args:
            model_name: String name of the model (text_encoder, text_encoder_2, etc.)
            target_device: Device to load to (defaults to GPU)
        """
        if target_device is None:
            target_device = gpu
            
        try:
            model = getattr(self, model_name)
            if model is not None:
                if model_name == "text_encoder":
                    # Special handling for text_encoder
                    fake_diffusers_current_device(model, target_device)
                else:
                    load_model_as_complete(model, target_device=target_device)
                debug(f"Model {model_name} loaded to {target_device}")
                return True
            else:
                debug(f"Model {model_name} not found")
                return False
        except Exception as e:
            debug(f"Error loading model {model_name} to device: {e}")
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
    
    def get_required_models_for_task(self, task):
        """
        Return a list of model names required for a specific task
        
        Args:
            task: String name of the task (encode_prompt, vae_encode, etc.)
        """
        task_model_map = {
            "encode_prompt": ["text_encoder", "text_encoder_2"],
            "vae_encode": ["vae"],
            "vae_decode": ["vae"],
            "clip_vision_encode": ["image_encoder"],
            "sampling": ["transformer"]
        }
        
        return task_model_map.get(task, [])
