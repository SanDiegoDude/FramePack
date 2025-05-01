# core/generation.py
import os
import time
import torch
import uuid
import math
import einops
import numpy as np
import traceback
import gc
import subprocess
from PIL import Image
from utils.common import debug, generate_timestamp
import utils.memory_utils as mem_utils
# Keep other specific imports from memory_utils if needed, or access them via mem_utils
from utils.memory_utils import cpu, gpu, get_cuda_free_memory_gb
from utils.video_utils import (
    save_bcthw_as_mp4, find_nearest_bucket, resize_and_center_crop,
    extract_frames_from_video, fix_video_compatibility,
    make_mp4_faststart, apply_gaussian_blur
)
from diffusers_helper.thread_utils import AsyncStream
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.utils import crop_or_pad_yield_mask, soft_append_bcthw
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from ui.style import make_progress_bar_html

from utils.prompt_parser import LoraPromptProcessor, SequentialPromptProcessor, apply_prompt_processors
from utils.lora_utils import LoRAConfig # Potentially needed for type hints if used directly

class VideoGenerator:
    """Handles all video generation functionality"""
    def __init__(self, model_manager, outputs_folder='./outputs/'):
        self.model_manager = model_manager
        self.output_folder = outputs_folder
        os.makedirs(outputs_folder, exist_ok=True)
        self.stream = None
        self.graceful_stop_requested = False
        debug(f"VideoGenerator initialized with output folder: {outputs_folder}")
        
    def setup_stream(self):
        """Initialize async stream for generation"""
        self.stream = AsyncStream()
        return self.stream
    
    def get_dims_from_aspect(self, aspect, custom_w, custom_h):
        """Calculate dimensions based on aspect ratio"""
        presets = {
            "16:9": (1280, 720), "9:16": (720, 1280),
            "1:1": (768, 768), "4:5": (720, 900),
            "3:2": (900, 600), "2:3": (600, 900),
            "21:9": (1260, 540), "4:3": (800, 600),
        }
        if aspect == "Custom...":
            width, height = int(custom_w), int(custom_h)
        else:
            width, height = presets.get(aspect, (768, 768))
        max_pixels = 1024 * 1024
        px = width * height
        if px > max_pixels:
            scale = math.sqrt(max_pixels / px)
            width = int(width * scale)
            height = int(height * scale)
        width = (width // 8) * 8
        height = (height // 8) * 8
        return width, height
    
    def parse_hex_color(self, hexcode):
        """Parse hex color code to RGB values"""
        hexcode = hexcode.lstrip('#')
        if len(hexcode) == 6:
            r = int(hexcode[0:2], 16) / 255
            g = int(hexcode[2:4], 16) / 255
            b = int(hexcode[4:6], 16) / 255
            return r, g, b
        # fallback to gray
        return 0.5, 0.5, 0.5
    
    def extract_frames_from_video(self, video_path, num_frames=8, from_end=True, max_resolution=640):
        """
        Extract frames from a video file
        This is a wrapper around the utility function to maintain API compatibility
        """
        from utils.video_utils import extract_frames_from_video as extract_frames
        return extract_frames(video_path, num_frames, from_end, max_resolution)
    
    def encode_multiple_prompts(self, cleaned_prompt_text, n_prompt, cfg, llm_weight=1.0, clip_weight=1.0):
        """
        Encode multiple prompts for sequential generation.
        Returns encoded prompts, poolers, masks, and negative versions.
        """
        from utils.prompt_parser import parse_sequential_prompts

        prompts = parse_sequential_prompts(cleaned_prompt_text)
        if not prompts:
            debug("[Encode Multi] Warning: Cleaned prompt resulted in no sequential prompts. Using the cleaned prompt as a single prompt.")
            prompts = [cleaned_prompt_text] # Fallback if splitting failed but text exists
        elif len(prompts) == 1:
             debug(f"[Encode Multi] Only one prompt after sequential parse: '{prompts[0]}'")
        else:
             debug(f"[Encode Multi] Encoding {len(prompts)} sequential prompts derived from cleaned prompt.")
        
        prompts = parse_sequential_prompts(prompt_text)
        debug(f"Encoding {len(prompts)} sequential prompts")
        
        # Initialize lists to store encodings
        llama_vecs = []
        clip_poolers = []
        llama_masks = []
        
        # --- Memory Management: Offload everything ---
        if not self.model_manager.high_vram:
            mem_utils.unload_complete_models(
                self.model_manager.text_encoder,
                self.model_manager.text_encoder_2
            )
            mem_utils.fake_diffusers_current_device(self.model_manager.text_encoder, gpu)
            mem_utils.load_model_as_complete(self.model_manager.text_encoder_2, target_device=gpu)
        
        # Encode each prompt
        for i, p in enumerate(prompts):
            debug(f"Encoding prompt {i+1}/{len(prompts)}: '{p}'")
            # Encode prompt
            llama_vec, clip_pooler = encode_prompt_conds(
                p,
                self.model_manager.text_encoder,
                self.model_manager.text_encoder_2,
                self.model_manager.tokenizer,
                self.model_manager.tokenizer_2
            )
            
            # Process masks
            llama_vec, llama_mask = crop_or_pad_yield_mask(llama_vec, 512)
            
            # Apply weights
            if llm_weight != 1.0:
                llama_vec = llama_vec * llm_weight
                
            if clip_weight == 0.0:
                clip_pooler = clip_pooler * 1e-7  # Near-zero value
            elif clip_weight != 1.0:
                clip_pooler = clip_pooler * clip_weight
                
            # Store encodings
            llama_vecs.append(llama_vec)
            clip_poolers.append(clip_pooler)
            llama_masks.append(llama_mask)
        
        # Handle negative prompt
        if cfg > 1:
            debug(f"Encoding negative prompt: '{n_prompt}'")
            llama_vec_n, clip_pooler_n = encode_prompt_conds(
                n_prompt,
                self.model_manager.text_encoder,
                self.model_manager.text_encoder_2,
                self.model_manager.tokenizer,
                self.model_manager.tokenizer_2
            )
            llama_vec_n, llama_mask_n = crop_or_pad_yield_mask(llama_vec_n, 512)
            
            # Apply weights to negative prompt
            if llm_weight != 1.0:
                llama_vec_n = llama_vec_n * llm_weight
            
            if clip_weight == 0.0:
                clip_pooler_n = clip_pooler_n * 1e-7
            elif clip_weight != 1.0:
                clip_pooler_n = clip_pooler_n * clip_weight
        else:
            debug("Using zero negative embeddings for CFG=1")
            llama_vec_n = torch.zeros_like(llama_vecs[0])
            clip_pooler_n = torch.zeros_like(clip_poolers[0])
            llama_mask_n = torch.ones_like(llama_masks[0])
        
        # Convert to appropriate dtype
        target_dtype = self.model_manager.transformer.dtype
        for i in range(len(llama_vecs)):
            llama_vecs[i] = llama_vecs[i].to(target_dtype)
            clip_poolers[i] = clip_poolers[i].to(target_dtype)
        
        llama_vec_n = llama_vec_n.to(target_dtype)
        clip_pooler_n = clip_pooler_n.to(target_dtype)
        
        return llama_vecs, clip_poolers, llama_masks, llama_vec_n, clip_pooler_n, llama_mask_n
    
    def prepare_inputs(self, input_image, cleaned_prompt, n_prompt, cfg, gaussian_blur_amount=0.0,
                   llm_weight=1.0, clip_weight=1.0):
        """Prepare text embeddings, poolers, masks, and image tensors"""
        if input_image is None:
            raise ValueError("Input image required for this mode.")
        if not hasattr(input_image, 'shape'):
            raise ValueError("Input image is not a valid numpy array!")
        H, W, C = input_image.shape
        
        # --- Memory Management: Offload everything ---
        if not self.model_manager.high_vram:
            mem_utils.unload_complete_models(
                self.model_manager.text_encoder,
                self.model_manager.text_encoder_2,
                self.model_manager.image_encoder,
                self.model_manager.vae,
                self.model_manager.transformer
            )
            
        # --- Text Encoding ---
        debug(f"[Prepare Inputs] Using cleaned prompt for encoding: '{cleaned_prompt}'")
        debug(f"[Prepare Inputs] Encoding negative prompt: '{n_prompt}'")
        if not self.model_manager.high_vram:
            mem_utils.fake_diffusers_current_device(self.model_manager.text_encoder, gpu)
            mem_utils.load_model_as_complete(self.model_manager.text_encoder_2, target_device=gpu)
        
        debug(f"[Prepare Inputs] Text Encoder device: {self.model_manager.text_encoder.device if self.model_manager.text_encoder else 'None'}")
        debug(f"[Prepare Inputs] Text Encoder 2 device: {self.model_manager.text_encoder_2.device if self.model_manager.text_encoder_2 else 'None'}")                          
        debug(f"[Prepare Inputs] BEFORE POS ENCODE - Prompt: '{prompt}'")
        
        lv, cp = encode_prompt_conds(
            cleaned_prompt, # Use the cleaned prompt here
            self.model_manager.text_encoder,
            self.model_manager.text_encoder_2,
            self.model_manager.tokenizer,
            self.model_manager.tokenizer_2
        )
        
        debug(f"[Prepare Inputs] Neg Text Encoder device: {self.model_manager.text_encoder.device if self.model_manager.text_encoder else 'None'}")
        debug(f"[Prepare Inputs] Neg Text Encoder 2 device: {self.model_manager.text_encoder_2.device if self.model_manager.text_encoder_2 else 'None'}")
        debug(f"[Prepare Inputs] AFTER POS ENCODE - lv: {lv.shape} ({lv.dtype}), mean: {lv.mean():.4f}, std: {lv.std():.4f}, isfinite: {torch.isfinite(lv).all()}")
        debug(f"[Prepare Inputs] AFTER POS ENCODE - cp: {cp.shape} ({cp.dtype}), mean: {cp.mean():.4f}, std: {cp.std():.4f}, isfinite: {torch.isfinite(cp).all()}")
        
        if cfg == 1:
            lv_n, cp_n = torch.zeros_like(lv), torch.zeros_like(cp)
            debug(f"[Prepare Inputs] Using ZERO negative embeddings for CFG=1")
        else:
            debug(f"[Prepare Inputs] BEFORE NEG ENCODE - Prompt: '{n_prompt}'")
            lv_n, cp_n = encode_prompt_conds(
                n_prompt,
                self.model_manager.text_encoder,
                self.model_manager.text_encoder_2,
                self.model_manager.tokenizer,
                self.model_manager.tokenizer_2
            )
            debug(f"[Prepare Inputs] AFTER NEG ENCODE - lv_n: {lv_n.shape} ({lv_n.dtype}), mean: {lv_n.mean():.4f}, std: {lv_n.std():.4f}, isfinite: {torch.isfinite(lv_n).all()}")
            debug(f"[Prepare Inputs] AFTER NEG ENCODE - cp_n: {cp_n.shape} ({cp_n.dtype}), mean: {cp_n.mean():.4f}, std: {cp_n.std():.4f}, isfinite: {torch.isfinite(cp_n).all()}")
            
        # --- Mask Generation (using the _correct_ crop_or_pad_yield_mask) ---
        # This function should return lv/lv_n with shape [B, 512, D]
        # and m/m_n with shape [B, 512] and dtype torch.bool
        debug(f"[Prepare Inputs] BEFORE PADDING - lv mean: {lv.mean():.4f}, std: {lv.std():.4f}")                          
        lv, m = crop_or_pad_yield_mask(lv, 512)
        debug(f"[Prepare Inputs] AFTER PADDING - lv mean: {lv.mean():.4f}, std: {lv.std():.4f}") # Check if padding affects stats
        debug(f"[Prepare Inputs] BEFORE PADDING - lv_n mean: {lv_n.mean():.4f}, std: {lv_n.std():.4f}")
        lv_n, m_n = crop_or_pad_yield_mask(lv_n, 512)
        debug(f"[Prepare Inputs] AFTER PADDING - lv_n mean: {lv_n.mean():.4f}, std: {lv_n.std():.4f}") # Check if padding affects stats
        
        debug(f"After crop_or_pad - lv: {lv.shape} ({lv.dtype}), m: {m.shape} ({m.dtype})")
        debug(f"lv mean: {lv.mean():.4f}, std: {lv.std():.4f}, isfinite: {torch.isfinite(lv).all()}")
        debug(f"cp mean: {cp.mean():.4f}, std: {cp.std():.4f}, isfinite: {torch.isfinite(cp).all()}")
        debug(f"lv_n mean: {lv_n.mean():.4f}, std: {lv_n.std():.4f}, isfinite: {torch.isfinite(lv_n).all()}")
        debug(f"cp_n mean: {cp_n.mean():.4f}, std: {cp_n.std():.4f}, isfinite: {torch.isfinite(cp_n).all()}")
        
        # --- Apply Weights ---
        if llm_weight != 1.0:
            lv = lv * llm_weight
            lv_n = lv_n * llm_weight
            debug(f"[Prepare Inputs] Applied LLM weight: {llm_weight}")
            
        # Even with zero weight, preserve tensor structure
        if clip_weight == 0.0:
            # Use a tiny value instead of true zero
            cp = cp * 1e-7  # Small enough to have negligible impact
            cp_n = cp_n * 1e-7
            debug(f"[Prepare Inputs] Applied near-zero CLIP weight (1e-7)")
        elif clip_weight != 1.0:
            cp = cp * clip_weight
            cp_n = cp_n * clip_weight
            debug(f"[Prepare Inputs] Applied CLIP weight: {clip_weight}")
            
        # --- Image Processing ---
        h, w = find_nearest_bucket(H, W, resolution=640)
        input_np = resize_and_center_crop(input_image, target_width=w, target_height=h)
        Image.fromarray(input_np).save(os.path.join(self.output_folder, f'{generate_timestamp()}_prep.png'))
        input_tensor = torch.from_numpy(input_np).float() / 127.5 - 1.0
        input_tensor = input_tensor.permute(2, 0, 1)[None, :, None] # Add Batch and Time dims
        
        # --- Gaussian Blur ---
        if gaussian_blur_amount > 0.0:
            input_tensor = apply_gaussian_blur(input_tensor, gaussian_blur_amount)
            
        return input_np, input_tensor, lv, cp, lv_n, cp_n, m, m_n, h, w
    
    def cleanup_temp_files(self, job_id, keep_final=True, final_output_path=None):
        """Clean up temporary files created during generation"""
        import glob
        debug(f"[CLEANUP] Starting cleanup for job {job_id}")
        preview_pattern = os.path.join(self.output_folder, f"{job_id}_preview_*.mp4")
        preview_files = glob.glob(preview_pattern)
        extension_file = os.path.join(self.output_folder, f"{job_id}_extension.mp4")
        compat_files = glob.glob(os.path.join(self.output_folder, f"{job_id}_*_compat.mp4"))
        filelist = os.path.join(self.output_folder, f"{job_id}_filelist.txt")
        
        for f in preview_files:
            if os.path.exists(f):
                try: os.remove(f); debug(f"[CLEANUP] Removed preview: {f}")
                except Exception as e: debug(f"[CLEANUP] Failed to remove {f}: {e}")
                
        temp_files = compat_files + [filelist]
        if not keep_final or final_output_path != extension_file:
            temp_files.append(extension_file)
            
        for f in temp_files:
            if os.path.exists(f):
                try: os.remove(f); debug(f"[CLEANUP] Removed temp file: {f}")
                except Exception as e: debug(f"[CLEANUP] Failed to remove {f}: {e}")
        
        debug(f"[CLEANUP] Completed cleanup for job {job_id}")
        
    @torch.no_grad()
    def generate_video(self, config):
        """Main video generation function"""
        
        # Import prompt parser
        from utils.prompt_parser import parse_sequential_prompts
        
        # Extract parameters
        mode = config['mode']
        input_image = config.get('input_image', None) # Can be None for text2video
        start_frame = config.get('start_frame', None)
        end_frame = config.get('end_frame', None)
        aspect = config.get('aspect', '1:1')
        custom_w = config.get('custom_w', 768)
        custom_h = config.get('custom_h', 768)
        prompt = config['prompt']
        n_prompt = config['n_prompt']
        seed = config['seed']
        adv_window = config['latent_window_size']
        segment_count = config.get('segment_count', 1)
        adv_seconds = config.get('adv_seconds', 0) # Calculated in process callback
        selected_frames = config.get('selected_frames', 0) # Calculated in process callback
        steps = config['steps']
        cfg = config['cfg']
        gs = config['gs']
        rs = config['rs']
        gpu_memory_preservation = config['gpu_memory_preservation']
        use_teacache = config['use_teacache']
        init_color = config.get('init_color', None)
        keyframe_weight = config.get('keyframe_weight', 0.7)
        input_video = config.get('input_video', None)
        extension_direction = config.get('extension_direction', "Forward")
        extension_frames = config.get('extension_frames', 8)
        original_mode = config.get('original_mode', None) # Keep track if it was video_extension
        gaussian_blur_amount = config.get('gaussian_blur_amount', 0.0)
        llm_weight = config.get('llm_weight', 1.0)
        clip_weight = config.get('clip_weight', 1.0)
        trim_percentage = config.get('trim_percentage', 0.2)  # Default to 0.2 if not provided
        frame_overlap = config.get('frame_overlap', 0)
        total_generated_latent_frames = 0
        job_id = generate_timestamp()
        
        debug(f"[Generator] Received prompt: '{prompt}'")
        debug(f"[Generator] Received negative prompt: '{n_prompt}'")
        debug(f"generate_video(): started {mode}, job_id: {job_id}")
        
        output_filename = os.path.join(self.output_folder, f'{job_id}_final.mp4')
        debug(f"Default output filename set to: {output_filename}")
        
        t_start = time.time()
        timings = {"latent_encoding": 0, "step_times": [], "generation_time": 0, "vae_decode_time": 0, "total_time": 0}

        # ============ Prompt Processing & Dynamic LoRA Loading ============
        debug(f"[Generator] Raw prompt received: '{prompt}'")
        
        # 1. Initialize Prompt Processors
        prompt_processors = [
            LoraPromptProcessor(),       # Extracts LoRAs and cleans the prompt
            SequentialPromptProcessor() # Extracts sequential info (doesn't clean further)
            # Add other processors here if needed in the future
        ]
        
        # 2. Apply Processors
        cleaned_prompt, extracted_data = apply_prompt_processors(prompt, prompt_processors)
        
        # 3. Extract LoRA Configs
        dynamic_lora_configs = extracted_data.get("lora", {}).get("lora_configs", [])
        debug(f"[Generator] Extracted {len(dynamic_lora_configs)} dynamic LoRA configs from prompt.")
        
        # 4. Apply Dynamic LoRAs via ModelManager
        # Ensure models (especially transformer) are loaded before applying LoRAs
        self.model_manager.ensure_all_models_loaded() # Crucial check
        
        try:
            applied_dynamic_loras, failed_dynamic_loras = self.model_manager.set_dynamic_loras(dynamic_lora_configs)
            # Optional: Handle failures if needed (e.g., notify user)
            if failed_dynamic_loras:
                debug(f"[Generator] WARNING: Failed to apply {len(failed_dynamic_loras)} dynamic LoRAs.")
                # You could add a UI message here if the stream exists
                # if self.stream:
                #     fail_msg = ", ".join([f"'{os.path.basename(c.path)}' ({c.error})" for c in failed_dynamic_loras])
                #     self.stream.output_queue.push(('progress', (None, f"⚠️ Failed LoRAs: {fail_msg}", "")))
        except RuntimeError as lora_error:
             # Handle critical failure if --lora-skip-fail is False
            debug(f"[Generator] CRITICAL LoRA Error: {lora_error}")
            debug(traceback.format_exc())
            if self.stream:
                 self.stream.output_queue.push(('progress', (None, f"❌ LoRA Error: {lora_error}. Generation stopped.", "")))
                 self.stream.output_queue.push(('end', 'lora_error'))
            # Clean up models before returning None
            if not self.model_manager.high_vram:
                mem_utils.unload_complete_models(
                    self.model_manager.text_encoder, self.model_manager.text_encoder_2,
                    self.model_manager.image_encoder, self.model_manager.vae,
                    self.model_manager.transformer
                )
            return None # Stop generation
        
        debug(f"[Generator] Using cleaned prompt for generation: '{cleaned_prompt}'")
        # ===============================================================

        
        # Extract the frame_overlap
        frame_overlap = config.get('frame_overlap', 0)
        debug(f"Using frame overlap: {frame_overlap}")

        
        # Calculate sections based on calculated seconds/frames
        # NOTE: Assuming adv_seconds/selected_frames are correctly calculated in ui.callbacks.process
        use_adv = True # Always use "advanced" calculations now
        if use_adv:
            latent_window_size = adv_window
            frames_per_section = latent_window_size * 4 - 3
            total_frames = int(round(adv_seconds * 30))
            total_sections = int(segment_count) if segment_count is not None else math.ceil(total_frames / frames_per_section)
            debug(f"Gen Params | window={latent_window_size} | frames/sec={frames_per_section} | total_frames={total_frames} | sections={total_sections}")
        else: # Fallback, should not happen with current UI
            latent_window_size = 9
            frames_per_section = 33
            total_frames = int(selected_frames)
            total_sections = total_frames // frames_per_section
            debug(f"Simple Mode | window=9 | frames/sec=33 | total_frames={total_frames} | sections={total_sections}")

        # Check if we're using sequential prompting
        prompts = parse_sequential_prompts(cleaned_prompt)
        use_sequential = len(prompts) > 1
        
        # Initialize variables for sequential prompts
        seq_llama_vecs = None
        seq_clip_poolers = None 
        seq_llama_masks = None
        
        if use_sequential:
            # Determine if we need to reverse the prompt order based on mode
            # For most modes, we need to reverse prompt order since generation is back-to-front
            should_reverse_prompts = (
                mode == "image2video" or
                mode == "text2video" or
                (mode == "keyframes" and start_frame is not None) or
                (original_mode == "video_extension" and extension_direction == "Forward")
            )

            # reversed to correct order of output
            if should_reverse_prompts:
                debug(f"Reversing sequential prompts for {mode} mode")
                prompts = list(reversed(prompts))
                
            debug(f"Using sequential prompting with {len(prompts)} prompts")
        
        if self.stream:
            self.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

        try:
            if hasattr(self.model_manager, 'lora_name') and self.model_manager.lora_name:
                debug(f"Applying LoRA '{self.model_manager.lora_name}' with weight {self.model_manager.lora_weight}")
                self.model_manager.apply_lora_weight()
            # ============ Input Preparation ============
            clip_output = None
            lv, cp, lv_n, cp_n, m, m_n = None, None, None, None, None, None
            start_latent = None
            end_latent = None # Only for keyframes
            height, width = 0, 0

            if mode == "keyframes":
                if end_frame is None: raise ValueError("Keyframes mode requires End Frame.")
                end_H, end_W, _= end_frame.shape
                height, width = find_nearest_bucket(end_H, end_W, resolution=640)
                debug(f"Keyframes: Bucket={width}x{height}")
                
                if start_frame is not None:
                    s_np = resize_and_center_crop(start_frame, width, height)
                    anchor_np = s_np
                else:
                    anchor_np = np.ones((height, width, 3), dtype=np.uint8) * 128 # Gray
                    
                end_np = resize_and_center_crop(end_frame, width, height)
                
                # --- Memory: Offload ---
                if not self.model_manager.high_vram: mem_utils.unload_complete_models(
                    self.model_manager.text_encoder, self.model_manager.text_encoder_2,
                    self.model_manager.image_encoder, self.model_manager.vae, self.model_manager.transformer
                )
                
                # --- VAE Encode Start/End ---
                if not self.model_manager.high_vram: mem_utils.load_model_as_complete(self.model_manager.vae, gpu)
                t_vae_start = time.time()
                anchor_tensor = (torch.from_numpy(anchor_np).float()/127.5 - 1.0).permute(2,0,1)[None,:,None]
                start_latent = vae_encode(anchor_tensor, self.model_manager.vae)
                end_tensor = (torch.from_numpy(end_np).float()/127.5 - 1.0).permute(2,0,1)[None,:,None]
                end_latent = vae_encode(end_tensor, self.model_manager.vae)
                timings["latent_encoding"] += time.time() - t_vae_start
                if not self.model_manager.high_vram: mem_utils.unload_complete_models(self.model_manager.vae)
                
                # --- Text Encoding ---
                if not self.model_manager.high_vram:
                    mem_utils.fake_diffusers_current_device(self.model_manager.text_encoder, gpu)
                    mem_utils.load_model_as_complete(self.model_manager.text_encoder_2, gpu)
                    
                lv, cp = encode_prompt_conds(
                    cleaned_prompt, self.model_manager.text_encoder, self.model_manager.text_encoder_2,
                    self.model_manager.tokenizer, self.model_manager.tokenizer_2
                )
                
                lv_n, cp_n = encode_prompt_conds(
                    n_prompt, self.model_manager.text_encoder, self.model_manager.text_encoder_2,
                    self.model_manager.tokenizer, self.model_manager.tokenizer_2
                ) if cfg > 1 else (torch.zeros_like(lv), torch.zeros_like(cp))
                
                lv, m = crop_or_pad_yield_mask(lv, 512)
                lv_n, m_n = crop_or_pad_yield_mask(lv_n, 512)
                
                if not self.model_manager.high_vram: mem_utils.unload_complete_models(self.model_manager.text_encoder, self.model_manager.text_encoder_2)
                
                 # --- CLIP Vision Encoding ---
                if not self.model_manager.high_vram: mem_utils.load_model_as_complete(self.model_manager.image_encoder, gpu)
                
                end_clip = hf_clip_vision_encode(
                    end_np, self.model_manager.feature_extractor, self.model_manager.image_encoder
                ).last_hidden_state
                
                if start_frame is not None:
                    start_clip = hf_clip_vision_encode(
                        anchor_np, self.model_manager.feature_extractor, self.model_manager.image_encoder
                    ).last_hidden_state
                    clip_output = (keyframe_weight * start_clip + (1.0 - keyframe_weight) * end_clip)
                else:
                    clip_output = end_clip
                    
                if not self.model_manager.high_vram: mem_utils.unload_complete_models(self.model_manager.image_encoder)
                
            elif mode == "text2video":
                width, height = self.get_dims_from_aspect(aspect, custom_w, custom_h)
                debug(f"Text2Video: Target dims={width}x{height}")
                
                if init_color:
                    r, g, b = self.parse_hex_color(init_color)
                    anchor_np = np.zeros((height, width, 3), dtype=np.uint8)
                    anchor_np[:, :, 0] = int(r * 255); anchor_np[:, :, 1] = int(g * 255); anchor_np[:, :, 2] = int(b * 255)
                else:
                    anchor_np = np.zeros((height, width, 3), dtype=np.uint8) # Black
                    
                # --- Prepare inputs using common function ---
                # ensure prepare_inputs correctly calls encode_prompt_conds with all managers
                _, input_tensor, lv, cp, lv_n, cp_n, m, m_n, height, width = self.prepare_inputs(
                    anchor_np, cleaned_prompt, n_prompt, cfg, gaussian_blur_amount, llm_weight, clip_weight
                )
                
                # --- VAE Encode ---
                if not self.model_manager.high_vram: mem_utils.load_model_as_complete(self.model_manager.vae, gpu)
                t_vae_start = time.time()
                start_latent = vae_encode(input_tensor, self.model_manager.vae)
                timings["latent_encoding"] += time.time() - t_vae_start
                if not self.model_manager.high_vram: mem_utils.unload_complete_models(self.model_manager.vae)
                
                # --- CLIP Vision (Encode the blank/color frame) ---
                if not self.model_manager.high_vram: mem_utils.load_model_as_complete(self.model_manager.image_encoder, gpu)
                clip_output = hf_clip_vision_encode(
                    anchor_np, self.model_manager.feature_extractor, self.model_manager.image_encoder
                ).last_hidden_state
                if not self.model_manager.high_vram: mem_utils.unload_complete_models(self.model_manager.image_encoder)
                
            elif mode == "image2video" or original_mode == "video_extension": # Includes redirected extension
                if input_image is None: raise ValueError("Image input required for image2video/video_extension")
                debug(f"Image2Video/Extension: Processing input image")
                
                # --- Prepare inputs using common function ---
                inp_np, input_tensor, lv, cp, lv_n, cp_n, m, m_n, height, width = self.prepare_inputs(
                    input_image, cleaned_prompt, n_prompt, cfg, gaussian_blur_amount, llm_weight, clip_weight
                )
                
                # --- VAE Encode ---
                if self.stream: self.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
                if not self.model_manager.high_vram: mem_utils.load_model_as_complete(self.model_manager.vae, gpu)
                t_vae_start = time.time()
                start_latent = vae_encode(input_tensor, self.model_manager.vae)
                timings["latent_encoding"] += time.time() - t_vae_start
                if not self.model_manager.high_vram: mem_utils.unload_complete_models(self.model_manager.vae)
                debug("VAE encoded")
                
                # --- CLIP Vision ---
                if self.stream: self.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
                if not self.model_manager.high_vram: mem_utils.load_model_as_complete(self.model_manager.image_encoder, gpu)
                clip_output = hf_clip_vision_encode(
                    inp_np, self.model_manager.feature_extractor, self.model_manager.image_encoder
                ).last_hidden_state
                if not self.model_manager.high_vram: mem_utils.unload_complete_models(self.model_manager.image_encoder)
                debug("CLIP Vision encoded")
                
            # ============ Dtype Conversion ============
            target_dtype = self.model_manager.transformer.dtype # Should be torch.bfloat16
            if lv is not None: lv = lv.to(target_dtype)
            if lv_n is not None: lv_n = lv_n.to(target_dtype)
            if cp is not None: cp = cp.to(target_dtype)
            if cp_n is not None: cp_n = cp_n.to(target_dtype)
            if clip_output is not None: clip_output = clip_output.to(target_dtype)
            # Masks (m, m_n) should be torch.bool and shape [B, 512], NO dtype conversion needed.
            
            debug(f"Final Dtypes - lv: {lv.dtype}, cp: {cp.dtype}, m: {m.dtype if m is not None else 'None'}, clip: {clip_output.dtype if clip_output is not None else 'None'}")


            # ============ Sequential Prompt Handling ============
            if use_sequential:
                debug(f"Encoding sequential prompts after main prompt preparation")
                # Store the original encodings 
                first_lv, first_cp, first_m = lv, cp, m
                
                # Now encode all prompts
                seq_llama_vecs, seq_clip_poolers, seq_llama_masks, seq_llama_vec_n, seq_clip_pooler_n, seq_llama_mask_n = \
                    self.encode_multiple_prompts( cleaned_prompt, n_prompt, cfg, llm_weight, clip_weight)
                
                # We'll keep using the original negative prompt encodings
                # as they apply to all prompts
                debug(f"Encoded {len(seq_llama_vecs)} sequential prompts")

            
            # ============ Sampling Setup ============
            if self.stream: self.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
            
            rnd = torch.Generator("cpu").manual_seed(seed)
            history_latents = torch.zeros((1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            history_pixels = None
            loop_iterator = reversed(range(total_sections))
            graceful_stop = False
            
            # ============ Main Generation Loop ============
            for section in loop_iterator:
                is_last_section = section == 0
                latent_padding_size = section * latent_window_size
                is_first_iteration = (section == total_sections - 1)
                debug(f'Section {section}/{total_sections-1}, padding={latent_padding_size}, last={is_last_section}, first={is_first_iteration}')
                
                if self.stream and self.stream.input_queue.top() == 'graceful_end': graceful_stop = True; debug("Graceful stop requested.")
                if graceful_stop and not is_last_section: debug("Stopping before next section."); break # Allow last section to finish
                
                # --- Prepare Indices and Clean Latents ---
                split_sizes = [1, latent_padding_size, latent_window_size, 1, 2, 16]
                total_indices = sum(split_sizes)
                indices = torch.arange(total_indices).unsqueeze(0)
                clean_latent_indices_pre, _, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split(split_sizes, dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
                clean_latents_pre = start_latent.to(history_latents.device, dtype=history_latents.dtype)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
                
                if mode == "keyframes" and is_first_iteration:
                    debug("Keyframes: Using end_latent for first iteration's clean_latents_post.")
                    clean_latents_post_override = end_latent.to(history_latents.device, dtype=history_latents.dtype)
                    clean_latents = torch.cat([clean_latents_pre, clean_latents_post_override], dim=2)
                
                # --- Mask Fallback (Correct Shape/Dtype) ---
                current_m = m if m is not None else torch.ones((lv.shape[0], lv.shape[1]), dtype=torch.bool, device=lv.device)
                current_m_n = m_n if m_n is not None else torch.ones((lv_n.shape[0], lv_n.shape[1]), dtype=torch.bool, device=lv_n.device)
                debug(f"Loop {section} - Using mask m: {current_m.shape} ({current_m.dtype})")

                if self.model_manager.transformer is None:
                    self.model_manager.transformer = mem_utils.load_model_as_complete(
                        self.model_manager.transformer, gpu
                    )

                # --- Memory: Load Transformer ---
                if not self.model_manager.high_vram:
                    # Only offload models you want to! Don't call with no args.
                    mem_utils.unload_complete_models(
                        # self.model_manager.text_encoder,
                        self.model_manager.text_encoder_2,
                        self.model_manager.image_encoder,
                        self.model_manager.vae,
                        # self.model_manager.transformer
                    )
                    if self.model_manager.transformer is None:
                        raise RuntimeError("ModelManager.transformer is None before GPU move attempt!")
                    debug(f"[GEN] About to move managed transformer to GPU: id={id(self.model_manager.transformer)}, type={type(self.model_manager.transformer)}")
                    mem_utils.move_model_to_device_with_memory_preservation(self.model_manager.transformer, mem_utils.gpu, gpu_memory_preservation)
                
                # --- Initialize TeaCache ---
                self.model_manager.initialize_teacache(use_teacache, steps if use_teacache else 0)
                debug(f"TeaCache initialized: {use_teacache}")
                
                # --- Callback Definition ---
                def callback(d):
                    nonlocal graceful_stop # Allow modification
                    # Preview generation
                    preview = d['denoised']
                    preview = vae_decode_fake(preview)
                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
                    
                    # Check for stop conditions
                    if self.stream and self.stream.input_queue.top() == 'end':
                        debug("callback: received 'end', stopping generation.")
                        self.stream.output_queue.push(('end', None))
                        raise KeyboardInterrupt('User ends the task.')
                    if self.stream and self.stream.input_queue.top() == 'graceful_end':
                        debug("callback: received 'graceful_end', will stop after current section.")
                        graceful_stop = True
                        
                    # Progress calculation
                    current_step = d['i'] + 1
                    
                    # Step timing
                    if current_step > 1:  # Skip first step for timing (initialization can be slow)
                        current_time = time.time()
                        if hasattr(callback, 'last_time'):
                            step_time = current_time - callback.last_time
                            timings["step_times"].append(step_time)
                            timings["generation_time"] += step_time
                        callback.last_time = current_time
                    elif current_step == 1:
                        callback.last_time = time.time()
                        
                    # Section progress
                    section_percentage = int(100.0 * current_step / steps)
                    
                    # Overall progress - ADAPTED FROM ORIGINAL
                    sections_completed = (total_sections - 1) - section  # How many sections came _before_ this one
                    if total_sections > 0:
                        overall_percentage = int(100.0 * (sections_completed + (current_step / steps)) / total_sections)
                    else:
                        overall_percentage = 0
                        
                    # Frame count
                    actual_pixel_frames = history_pixels.shape[2] if history_pixels is not None else 0
                    actual_seconds = actual_pixel_frames / 30.0
                    
                    hint = f'Section {sections_completed+1}/{total_sections} - Step {current_step}/{steps}'
                    total_expected_frames = 0
                    if total_sections == 1:
                        total_expected_frames = (latent_window_size * 2 + 1) + 4  # +4 for initial frame 
                    else:
                        total_expected_frames = (latent_window_size * 2 + 1) + 4 + (latent_window_size * 2) * (total_sections - 1)
                    
                    desc = f'Pixel frames generated: {actual_pixel_frames}, Length: {actual_seconds:.2f}s (FPS-30), Expected total: {total_expected_frames} frames'
                    
                    # HTML progress display
                    progress_html = f"""
                    <div class="dual-progress-container">
                        <div class="progress-label">
                            <span>Current Section:</span>
                            <span>{section_percentage}%</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fg" style="width: {section_percentage}%"></div>
                        </div>
                        <div class="progress-label">
                            <span>Overall Progress:</span>
                            <span>{overall_percentage}%</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fg" style="width: {overall_percentage}%"></div>
                        </div>
                        <div style="font-size:0.9em; opacity:0.8;">{hint}</div>
                    </div>
                    """
                    
                    debug(f"In callback, section: {section_percentage}%, overall: {overall_percentage}%")
                    if self.stream:
                        self.stream.output_queue.push(('progress', (preview, desc, progress_html)))
                
                # DEBUG PROMPT HANDLING
                debug(f"Loop {section} - Calling sample_hunyuan...")
                debug(f"  prompt_embeds=lv: {lv.shape} ({lv.dtype}), mean: {lv.mean():.4f}")
                debug(f"  prompt_embeds_mask=current_m: {current_m.shape} ({current_m.dtype})")
                debug(f"  prompt_poolers=cp: {cp.shape} ({cp.dtype}), mean: {cp.mean():.4f}")
                debug(f"  neg_prompt_embeds=lv_n: {lv_n.shape} ({lv_n.dtype}), mean: {lv_n.mean():.4f}")
                debug(f"  neg_prompt_embeds_mask=current_m_n: {current_m_n.shape} ({current_m_n.dtype})")
                debug(f"  neg_prompt_poolers=cp_n: {cp_n.shape} ({cp_n.dtype}), mean: {cp_n.mean():.4f}")
                if clip_output is not None:
                    debug(f"  image_embeddings=clip_output: {clip_output.shape} ({clip_output.dtype}), mean: {clip_output.mean():.4f}")
                else:
                    debug(f"  image_embeddings=clip_output: None")
                debug(f"  CFG scales: cfg={cfg}, gs={gs}, rs={rs}")

                # Find the section to determine which prompt to use
                if use_sequential and seq_llama_vecs is None:
                    # Calculate which prompt to use
                    # Remember sections are processed in reverse order (end→start)
                    # section is from your loop_iterator which should match the current section index
                    section_idx = total_sections - 1 - section  # Convert to forward index
                    prompt_idx = min(section_idx, len(seq_llama_vecs) - 1)  # Use last prompt if we run out
                    
                    if section_idx == prompt_idx:
                        debug(f"Section {section}/{total_sections-1} using prompt {prompt_idx+1}/{len(seq_llama_vecs)}: '{prompts[prompt_idx][:30]}...'")
                    else:
                        debug(f"Section {section}/{total_sections-1} using final prompt {len(seq_llama_vecs)}/{len(seq_llama_vecs)}: '{prompts[prompt_idx][:30]}...'")
                    
                    # Use the corresponding encodings
                    current_lv = seq_llama_vecs[prompt_idx]
                    current_cp = seq_clip_poolers[prompt_idx]
                    current_m = seq_llama_masks[prompt_idx]
                else:
                    # Use the default single prompt encodings
                    current_lv = lv
                    current_cp = cp
                    current_m = m
                
                # Run sampling based on mode
                if mode == "keyframes":
                    generated_latents = sample_hunyuan(
                        transformer=self.model_manager.transformer,
                        sampler="unipc",
                        width=width,
                        height=height,
                        frames=frames_per_section,
                        real_guidance_scale=cfg,
                        distilled_guidance_scale=gs,
                        guidance_rescale=rs,
                        num_inference_steps=steps,
                        generator=rnd,
                        prompt_embeds=current_lv,
                        prompt_embeds_mask=current_m,
                        prompt_poolers=current_cp,
                        negative_prompt_embeds=lv_n,
                        negative_prompt_embeds_mask=m_n,
                        negative_prompt_poolers=cp_n,
                        device=gpu,
                        dtype=torch.bfloat16,
                        image_embeddings=clip_output,
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        callback=callback,
                    )
                else:  # image2video, text2video, video_extension (redirected to image2video)
                    generated_latents = sample_hunyuan(
                        transformer=self.model_manager.transformer,
                        sampler='unipc',
                        width=width,
                        height=height,
                        frames=frames_per_section,
                        real_guidance_scale=cfg,
                        distilled_guidance_scale=gs,
                        guidance_rescale=rs,
                        num_inference_steps=steps,
                        generator=rnd,
                        prompt_embeds=current_lv,
                        prompt_embeds_mask=current_m,
                        prompt_poolers=current_cp,
                        negative_prompt_embeds=lv_n,
                        negative_prompt_embeds_mask=m_n,
                        negative_prompt_poolers=cp_n,
                        device=gpu,
                        dtype=torch.bfloat16,
                        image_embeddings=clip_output,
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        callback=callback
                    )
                
                # Handle last section specially
                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents.device, dtype=generated_latents.dtype), generated_latents], dim=2)
                    debug(f"is_last_section => concatenated latent, new shape: {generated_latents.shape}")
                    
                # Update history latents
                history_latents = torch.cat([generated_latents.to(history_latents.device, dtype=history_latents.dtype), history_latents], dim=2)
                debug(f"history_latents.shape after concat: {history_latents.shape}")
                
                # VAE Decoding Section - CRITICAL ROTARY DECODER
                if not self.model_manager.high_vram:
                    mem_utils.offload_model_from_device_for_memory_preservation(
                        self.model_manager.transformer,
                        offload_target_device=mem_utils.gpu,
                        preserved_memory_gb=8
                    )
                    mem_utils.load_model_as_complete(self.model_manager.vae, target_device=gpu)
                    debug("loaded vae to gpu")
                
                # Clear CUDA cache to reduce fragmentation
                torch.cuda.empty_cache()
                debug("Cleared CUDA cache before decoding")
                
                # Update total frame count
                total_generated_latent_frames += int(generated_latents.shape[2])
                debug(f"Current number of total generated frames: {total_generated_latent_frames}")
                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
                
                # VAE decoding with original approach
                try:
                    t_vae_start = time.time()
                    if history_pixels is None:
                        history_pixels = vae_decode(real_history_latents, self.model_manager.vae).cpu()
                        debug(f"First section: Decoded full latents to pixels with shape: {history_pixels.shape}")
                    else:
                        # Calculate correct section frames
                        section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                        overlapped_frames = latent_window_size * 4 - 3
                        debug(f"Decoding section: frames={section_latent_frames}, overlap={overlapped_frames}")
                        
                        # Decode current section only
                        current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], self.model_manager.vae).cpu()
                        
                        # Use the original soft_append_bcthw with correct order
                        history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
                        debug(f"Blended with original method. New history shape: {history_pixels.shape}")
                    
                    timings["vae_decode_time"] += time.time() - t_vae_start
                    
                    # Save preview video
                    preview_filename = os.path.join(self.output_folder, f'{job_id}_preview_{uuid.uuid4().hex}.mp4')
                    try:
                        save_bcthw_as_mp4(history_pixels, preview_filename, fps=30, quiet=True)
                        debug(f"[FILE] Preview video saved: {preview_filename} ({os.path.exists(preview_filename)})")
                        if self.stream:
                            self.stream.output_queue.push(('preview_video', preview_filename))
                            debug(f"[QUEUE] Queued preview_video event: {preview_filename}")
                    except Exception as e:
                        debug(f"[ERROR] Failed to save preview video: {e}")
                    
                    # Clean up memory
                    if 'current_pixels' in locals():
                        del current_pixels
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    debug(f"Error during VAE decoding: {str(e)}")
                    debug(traceback.format_exc())
                    raise
                
                # Check if we should gracefully stop after this section
                if graceful_stop:
                    debug("graceful stop requested, ending generation after completing section.")
                    break
                    
            # After all sections, process the video
            # Handle video extension mode
            if original_mode == "video_extension" and input_video is not None and 'history_pixels' in locals() and history_pixels is not None:
                # Save the generated extension
                extension_filename = os.path.join(self.output_folder, f'{job_id}_extension.mp4')
                save_bcthw_as_mp4(history_pixels, extension_filename, fps=30, quiet=True)  # Added quiet parameter
                debug(f"[FILE] Extension video saved as {extension_filename}")
                
                # Now combine with the original video
                combined_filename = os.path.join(self.output_folder, f'{job_id}_combined.mp4')
                try:
                    if extension_direction == "Backward":
                        debug(f"[FFMPEG] Processing for backward extension - need to apply trim to extension")
                        # Create a trimmed version of the extension with trim_percentage applied
                        trim_ext = os.path.join(self.output_folder, f'{job_id}_ext_trimmed.mp4')
                        # Calculate the number of frames to keep (trim from start)
                        trim_seconds = trim_percentage * 5.0  # Approximate trim in seconds (assuming ~30fps and usual length)
                        
                        # Use quiet ffmpeg
                        trim_cmd = [
                            "ffmpeg", "-y", "-loglevel", "error",
                            "-i", extension_filename,
                            "-ss", str(trim_seconds),  # Start time offset to skip initial frames
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-vsync", "cfr", "-r", "30",
                            trim_ext
                        ]
                        
                        try:
                            subprocess.run(trim_cmd, check=True, capture_output=True)
                            debug(f"[FFMPEG] Successfully created trimmed extension for backward mode")
                            # Use the trimmed file instead of the original extension
                            extension_filename = trim_ext
                        except Exception as e:
                            debug(f"[FFMPEG] Failed to trim extension: {e}")
                            # Continue with original file
                            
                    # Prepare videos for concatenation
                    debug("[FFMPEG] Preparing videos for concatenation...")
                    temp_ext = os.path.join(self.output_folder, f'{job_id}_ext_compat.mp4')
                    temp_orig = os.path.join(self.output_folder, f'{job_id}_orig_compat.mp4')
                    
                    # Convert extension to compatible format (quiet mode)
                    convert_cmd = [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", extension_filename,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vsync", "cfr", "-r", "30",
                        temp_ext
                    ]
                    
                    try:
                        subprocess.run(convert_cmd, check=True, capture_output=True)
                    except subprocess.CalledProcessError as e:
                        debug(f"[FFMPEG] Extension conversion failed: {e}")
                        temp_ext = extension_filename
                        
                    # Convert original video to compatible format (quiet mode)
                    convert_cmd = [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", input_video,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vsync", "cfr", "-r", "30",
                        temp_orig
                    ]
                    
                    try:
                        subprocess.run(convert_cmd, check=True, capture_output=True)
                    except subprocess.CalledProcessError as e:
                        debug(f"[FFMPEG] Original conversion failed: {e}")
                        temp_orig = input_video
                        
                    # Create file list for concating
                    filelist_path = os.path.join(self.output_folder, f'{job_id}_filelist.txt')
                    with open(filelist_path, 'w') as f:
                        if extension_direction == "Forward":
                            f.write(f"file '{os.path.abspath(temp_orig)}'\n")
                            f.write(f"file '{os.path.abspath(temp_ext)}'\n")
                        else:  # Backward
                            f.write(f"file '{os.path.abspath(temp_ext)}'\n")
                            f.write(f"file '{os.path.abspath(temp_orig)}'\n")
                            
                    # Run concat (quiet mode)
                    concat_cmd = [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-f", "concat",
                        "-safe", "0",
                        "-i", filelist_path,
                        "-c", "copy",
                        combined_filename
                    ]
                    
                    result = subprocess.run(concat_cmd, capture_output=True)
                    if result.returncode != 0:
                        debug(f"[FFMPEG] Concat failed: {result.stderr.decode() if result.stderr else 'Unknown error'}")
                        debug("[FFMPEG] Falling back to using just the extension video")
                        output_filename = extension_filename
                    else:
                        debug(f"[FFMPEG] Concat succeeded!")
                        make_mp4_faststart(combined_filename, quiet=True)  # Added quiet parameter
                        output_filename = combined_filename
                        
                    # Clean up temp files
                    for f in [filelist_path, temp_ext, temp_orig]:
                        if os.path.exists(f) and f != extension_filename and f != input_video:
                            try:
                                os.remove(f)
                            except:
                                pass
                                
                except Exception as e:
                    debug(f"[ERROR] Failed to combine videos: {e}")
                    debug(traceback.format_exc())
                    output_filename = extension_filename
                    debug(f"[FILE] Using extension video as fallback: {output_filename}")
                    
            # Final export logic (text2video special handling)
            elif mode == "text2video":
                N_actual = history_pixels.shape[2]
                # Special case: single image
                if latent_window_size <= 2 and total_sections <= 1 and total_frames <= 8:
                    debug("txt2img branch: pulling last frame, skipping video trim (window=2, adv=0.1)")
                    last_img_tensor = history_pixels[0, :, -1]
                    last_img = np.clip((np.transpose(last_img_tensor.cpu().numpy(), (1, 2, 0)) + 1) * 127.5, 0, 255).astype(np.uint8)
                    img_filename = os.path.join(self.output_folder, f'{job_id}_final_image.png')
                    try:
                        Image.fromarray(last_img).save(img_filename)
                        debug(f"[FILE] Image saved: {img_filename}")
                        if self.stream:
                            html_link = f'<a href="file/{img_filename}" target="_blank"><img src="file/{img_filename}" style="max-width:100%;border:3px solid orange;border-radius:8px;" title="Click for full size"></a>'
                            self.stream.output_queue.push(('file_img', (img_filename, html_link)))
                            self.stream.output_queue.push(('end', "img"))
                        return img_filename
                    except Exception as e:
                        debug(f"[ERROR] Save failed for txt2img: {e}")
                        debug(traceback.format_exc())
                        if self.stream:
                            self.stream.output_queue.push(('end', "img"))
                        return None
                        
                # Normal text2video: trim initial frames
                if latent_window_size == 3:
                    drop_n = int(N_actual * 0.75)
                    debug(f"special trim for 3: dropping first {drop_n} frames of {N_actual}")
                elif latent_window_size == 4:
                    drop_n = int(N_actual * 0.5)
                    debug(f"special trim for 4: dropping first {drop_n} frames of {N_actual}")
                else:
                    # Only modify this line to use trim_percentage instead of hardcoded 0.2
                    drop_n = math.floor(N_actual * trim_percentage)
                    debug(f"normal trim: dropping first {drop_n} frames ({trim_percentage*100:.1f}%) of {N_actual}")
                    
                history_pixels = history_pixels[:, :, drop_n:, :, :]
                N_after = history_pixels.shape[2]
                debug(f"Final video after trim for txt2vid, {N_after} frames left")
                
                # Handle case where only one frame remains
                if N_after == 1:
                    last_img_tensor = history_pixels[0, :, 0]
                    last_img = np.clip((np.transpose(last_img_tensor.cpu().numpy(), (1, 2, 0)) + 1) * 127.5, 0, 255).astype(np.uint8)
                    img_filename = os.path.join(self.output_folder, f'{job_id}_final_image.png')
                    try:
                        Image.fromarray(last_img).save(img_filename)
                        debug(f"[FILE] Image saved: {img_filename}")
                        if self.stream:
                            html_link = f'<a href="file/{img_filename}" target="_blank"><img src="file/{img_filename}" style="max-width:100%;border:3px solid orange;border-radius:8px;" title="Click for full size"></a>'
                            self.stream.output_queue.push(('file_img', (img_filename, html_link)))
                            self.stream.output_queue.push(('end', "img"))
                        return img_filename
                    except Exception as e:
                        debug(f"[ERROR] Save failed for trimmed single image: {e}")
                        debug(traceback.format_exc())
                        if self.stream:
                            self.stream.output_queue.push(('end', "img"))
                        return None
                        
            # Special handling for keyframes with no start frame
            elif mode == "keyframes" and start_frame is None:
                N_actual = history_pixels.shape[2]
                debug(f"keyframe mode with no start frame: considering trimming {N_actual} frames")
                
                # Use similar trim logic as text2video
                if latent_window_size == 3:
                    drop_n = int(N_actual * 0.75)
                    debug(f"keyframe special trim for 3: dropping first {drop_n} frames of {N_actual}")
                elif latent_window_size == 4:
                    drop_n = int(N_actual * 0.5)
                    debug(f"keyframe special trim for 4: dropping first {drop_n} frames of {N_actual}")
                else:
                    # Only modify this line to use trim_percentage instead of hardcoded 0.2
                    drop_n = math.floor(N_actual * trim_percentage)
                    debug(f"keyframe normal trim: dropping first {drop_n} frames ({trim_percentage*100:.1f}%) of {N_actual}")
                    
                history_pixels = history_pixels[:, :, drop_n:, :, :]
                N_after = history_pixels.shape[2]
                debug(f"Final video after trim for keyframe (no start frame), {N_after} frames left")
                
                # Handle case where trimming leaves just one frame
                if N_after == 1:
                    debug("After trimming keyframe video, only one frame remains - will save as image")
                    last_img_tensor = history_pixels[0, :, 0]
                    last_img = np.clip((np.transpose(last_img_tensor.cpu().numpy(), (1,2,0)) + 1) * 127.5, 0, 255).astype(np.uint8)
                    img_filename = os.path.join(self.output_folder, f'{job_id}_keyframe_final.png')
                    try:
                        Image.fromarray(last_img).save(img_filename)
                        debug(f"[FILE] Image saved: {img_filename}")
                        if self.stream:
                            html_link = f'<a href="file/{img_filename}" target="_blank"><img src="file/{img_filename}" style="max-width:100%;border:3px solid orange;border-radius:8px;" title="Click for full size"></a>'
                            self.stream.output_queue.push(('file_img', (img_filename, html_link)))
                            self.stream.output_queue.push(('end', "img"))
                        return img_filename
                    except Exception as e:
                        debug(f"[ERROR] Save failed for keyframe single image: {e}")
                        debug(traceback.format_exc())
                        if self.stream:
                            self.stream.output_queue.push(('end', "img"))
                        return None
                        
            # Final MP4 Export
            if 'history_pixels' in locals() and history_pixels is not None and history_pixels.shape[2] > 0:
                # Check if we've already handled an image or special case
                image_likely_saved = False
                if (mode == "text2video" or (mode == "keyframes" and start_frame is None)):
                    if history_pixels.shape[2] <= 1:
                        image_likely_saved = True
                        
                # Handle the extension case first
                if original_mode == "video_extension" and 'combined_filename' in locals() and os.path.exists(combined_filename):
                    debug(f"[FILE] Using pre-combined video file for video_extension mode: {combined_filename}")
                    fix_video_compatibility(combined_filename, fps=30, quiet=True)  # Added quiet parameter
                    if self.stream:
                        self.stream.output_queue.push(('file', combined_filename))
                    self.cleanup_temp_files(job_id, keep_final=True, final_output_path=combined_filename)
                    return combined_filename
                
                # Standard video export
                elif not image_likely_saved:
                    debug(f"[FILE] Attempting to save final video to {output_filename}")
                    try:
                        save_bcthw_as_mp4(history_pixels, output_filename, fps=30, quiet=True)  # Added quiet parameter
                        debug(f"[FILE] Video successfully saved to {output_filename}: {os.path.exists(output_filename)}")
                        fix_video_compatibility(output_filename, fps=30, quiet=True)  # Added quiet parameter
                        if self.stream:
                            self.stream.output_queue.push(('file', output_filename))
                        self.cleanup_temp_files(job_id, keep_final=True, final_output_path=output_filename)
                        return output_filename
                    except Exception as e:
                        debug(f"[ERROR] FAILED to save final video {output_filename}: {e}")
                        debug(traceback.format_exc())
                        return None
                else:
                    debug(f"[FILE] Skipping final video save, likely handled by image export logic.")
                    return None
            else:
                debug(f"[FILE] Skipping final video save: No valid history_pixels found.")
                return None
                
        except KeyboardInterrupt:
            debug("KeyboardInterrupt received, performing clean recovery")
            # Clean up resources
            if not self.model_manager.high_vram:
                mem_utils.unload_complete_models(
                    self.model_manager.text_encoder,
                    self.model_manager.text_encoder_2,
                    self.model_manager.image_encoder,
                    self.model_manager.vae,
                    self.model_manager.transformer
                )
            torch.cuda.empty_cache()
            # Cleanup temporary files
            self.cleanup_temp_files(job_id, keep_final=False)
            # Push messages to UI
            if self.stream:
                self.stream.output_queue.push(('progress', (None, "Generation stopped by user.", "")))
                self.stream.output_queue.push(('end', "interrupted"))
            return None
            
        except Exception as ex:
            debug(f"EXCEPTION THROWN: {ex}")
            debug(traceback.format_exc())
            if not self.model_manager.high_vram:
                mem_utils.unload_complete_models(
                    self.model_manager.text_encoder,
                    self.model_manager.text_encoder_2,
                    self.model_manager.image_encoder,
                    self.model_manager.vae,
                    self.model_manager.transformer
                )
            debug("after exception and possible unload, exiting generation.")
            if self.stream:
                self.stream.output_queue.push(('end', None))
            return None
            
        finally:
            debug("in finally block, writing final summary/progress/end")
            t_end = time.time()
            
            # Use final state of history_pixels for summary
            if 'history_pixels' in locals() and history_pixels is not None:
                # Account for trimming possibly reducing frames before final save
                trimmed_frames = history_pixels.shape[2]
                video_seconds = trimmed_frames / 30.0
            else:
                trimmed_frames = 0
                video_seconds = 0.0
                
            # Calculate final timings
            timings["total_time"] = t_end - t_start
            misc_time = timings["total_time"] - (timings["latent_encoding"] + timings["generation_time"] + timings["vae_decode_time"])
            
            # Calculate avg step time
            avg_step_time = sum(timings["step_times"]) / len(timings["step_times"]) if timings["step_times"] else 0
            
            # Format timing display
            minutes = int(timings["total_time"] // 60)
            seconds = timings["total_time"] % 60
            summary_string = f"""
            ### Performance Summary
            **Total generated frames:** {trimmed_frames}  
            **Video length:** {video_seconds:.2f} seconds (FPS-30)  
            **Time taken:** {minutes}m {seconds:.2f}s  
            
            ### ⏱️ Performance Breakdown:
            • **Latent encoding:** {timings['latent_encoding']:.2f}s  
            • **Average step time:** {avg_step_time:.2f}s  
            • **Active generation:** {timings['generation_time']:.2f}s  
            • **VAE decoding:** {timings['vae_decode_time']:.2f}s  
            • **Other processing:** {misc_time:.2f}s
            """
            
            if self.stream:
                self.stream.output_queue.push(('progress', (None, summary_string, "")))
                self.stream.output_queue.push(('final_stats', summary_string))
                self.stream.output_queue.push(('end', None))
                
            return output_filename
