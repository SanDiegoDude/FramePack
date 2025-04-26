# core/generation.py
import os
import time
import torch
import uuid
import math
import einops
import numpy as np
import traceback
from PIL import Image
from utils.common import debug, generate_timestamp
from utils.memory_utils import (
    cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation, load_model_as_complete,
    unload_complete_models, fake_diffusers_current_device
)
from utils.video_utils import (
    save_bcthw_as_mp4, find_nearest_bucket, resize_and_center_crop,
    crop_or_pad_yield_mask, extract_frames_from_video, fix_video_compatibility,
    make_mp4_faststart, apply_gaussian_blur
)
from diffusers_helper.thread_utils import AsyncStream
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from ui.style import make_progress_bar_html

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
    
    def prepare_inputs(self, input_image, prompt, n_prompt, cfg, gaussian_blur_amount=0.0,
                      llm_weight=1.0, clip_weight=1.0):
        """Prepare inputs for generation"""
        if input_image is None:
            raise ValueError(
                "No input image provided! For text2video, a blank will be created in worker -- "
                "but for image2video, you must upload an image."
            )
        
        if hasattr(input_image, 'shape'):
            H, W, C = input_image.shape
        else:
            raise ValueError("Input image is not a valid numpy array!")
        
        # Clear GPU first
        if not self.model_manager.high_vram:
            unload_complete_models(
                self.model_manager.text_encoder, 
                self.model_manager.text_encoder_2, 
                self.model_manager.image_encoder, 
                self.model_manager.vae, 
                self.model_manager.transformer
            )
        
        # Load only text encoders for prompt processing
        fake_diffusers_current_device(self.model_manager.text_encoder, gpu)
        load_model_as_complete(self.model_manager.text_encoder_2, target_device=gpu)
        
        # Encode prompts
        llama_vec, clip_pool = encode_prompt_conds(
            prompt, 
            self.model_manager.text_encoder, 
            self.model_manager.text_encoder_2, 
            self.model_manager.tokenizer, 
            self.model_manager.tokenizer_2
        )
        
        # Create negative prompts
        if cfg == 1:
            llama_vec_n, clip_pool_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_pool)
        else:
            llama_vec_n, clip_pool_n = encode_prompt_conds(
                n_prompt,
                self.model_manager.text_encoder, 
                self.model_manager.text_encoder_2, 
                self.model_manager.tokenizer, 
                self.model_manager.tokenizer_2
            )
        
        # Process masks
        llama_vec, mask = crop_or_pad_yield_mask(llama_vec, 512)
        llama_vec_n, mask_n = crop_or_pad_yield_mask(llama_vec_n, 512)
        
        # Apply weights
        if llm_weight != 1.0:
            llama_vec = llama_vec * llm_weight
            llama_vec_n = llama_vec_n * llm_weight
        
        if clip_weight != 1.0:
            clip_pool = clip_pool * clip_weight
            clip_pool_n = clip_pool_n * clip_weight
        
        # Process image
        h, w = find_nearest_bucket(H, W, resolution=640)
        input_np = resize_and_center_crop(input_image, target_width=w, target_height=h)
        
        # Save a copy
        Image.fromarray(input_np).save(os.path.join(self.output_folder, f'{generate_timestamp()}.png'))
        
        # Convert to tensor
        input_tensor = torch.from_numpy(input_np).float() / 127.5 - 1
        input_tensor = input_tensor.permute(2, 0, 1)[None, :, None]
        
        # Apply blur if needed
        if gaussian_blur_amount > 0.0:
            input_tensor = apply_gaussian_blur(input_tensor, gaussian_blur_amount)
        
        return input_np, input_tensor, llama_vec, clip_pool, llama_vec_n, clip_pool_n, mask, mask_n, h, w
    
    def cleanup_temp_files(self, job_id, keep_final=True, final_output_path=None):
        """Clean up temporary files created during generation"""
        import glob
        
        debug(f"[CLEANUP] Starting cleanup for job {job_id}")
        
        # Find all preview videos for this job
        preview_pattern = os.path.join(self.output_folder, f"{job_id}_preview_*.mp4")
        preview_files = glob.glob(preview_pattern)
        
        # Find extension/temporary files
        extension_file = os.path.join(self.output_folder, f"{job_id}_extension.mp4")
        compat_files = glob.glob(os.path.join(self.output_folder, f"{job_id}_*_compat.mp4"))
        filelist = os.path.join(self.output_folder, f"{job_id}_filelist.txt")
        
        # Delete preview videos
        for f in preview_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    debug(f"[CLEANUP] Removed preview: {f}")
                except Exception as e:
                    debug(f"[CLEANUP] Failed to remove {f}: {e}")
        
        # Delete temporary files used for video concatenation
        temp_files = compat_files + [filelist]
        if not keep_final or final_output_path != extension_file:
            temp_files.append(extension_file)
        
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    debug(f"[CLEANUP] Removed temp file: {f}")
                except Exception as e:
                    debug(f"[CLEANUP] Failed to remove {f}: {e}")
        
        debug(f"[CLEANUP] Completed cleanup for job {job_id}")

    @torch.no_grad()
    def generate_video(self, config):
        """
        Main video generation function - this is the new worker function
        
        Args:
            config: Dictionary containing all generation parameters
            
        Returns:
            str: Output filename
        """
        # Extract parameters
        mode = config['mode']
        input_image = config['input_image']
        start_frame = config['start_frame']
        end_frame = config['end_frame']
        aspect = config['aspect']
        custom_w = config['custom_w']
        custom_h = config['custom_h']
        prompt = config['prompt']
        n_prompt = config['n_prompt']
        seed = config['seed']
        use_adv = config.get('use_adv', True)  # Default to True
        adv_window = config['latent_window_size']
        adv_seconds = config.get('adv_seconds', 0)
        selected_frames = config.get('selected_frames', 0)
        segment_count = config.get('segment_count', 1)
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
        original_mode = config.get('original_mode', None)
        frame_overlap = config.get('frame_overlap', 0)
        gaussian_blur_amount = config.get('gaussian_blur_amount', 0.0)
        llm_weight = config.get('llm_weight', 1.0)
        clip_weight = config.get('clip_weight', 1.0)
        
        # Create job ID and setup
        job_id = generate_timestamp()
        debug(f"generate_video(): started {mode}, job_id: {job_id}")
        
        # Setup output filename
        output_filename = os.path.join(self.output_folder, f'{job_id}_final.mp4')
        debug(f"Default output filename set to: {output_filename}")
        
        # Initialize timing statistics
        t_start = time.time()
        timings = {
            "latent_encoding": 0,
            "step_times": [],
            "generation_time": 0,
            "vae_decode_time": 0,
            "total_time": 0
        }
        
        # Setup section/frames logic - PRESERVING ORIGINAL CALCULATIONS
        if use_adv:
            latent_window_size = adv_window
            frames_per_section = latent_window_size * 4 - 3
            total_frames = int(round(adv_seconds * 30))
            # Ensure we generate the exact number of requested segments
            total_sections = int(segment_count) if segment_count is not None else math.ceil(total_frames / frames_per_section)
            debug(f"Advanced mode | latent_window_size={latent_window_size} "
                  f"| frames_per_section={frames_per_section} | total_frames={total_frames} | total_sections={total_sections}")
        else:
            latent_window_size = 9
            frames_per_section = latent_window_size * 4 - 3
            total_frames = int(selected_frames)
            total_sections = total_frames // frames_per_section
            debug(f"Simple mode | latent_window_size=9 | frames_per_section=33 | "
                  f"total_frames={total_frames} | total_sections={total_sections}")
        
        # Initialize stream queue progress
        if self.stream:
            self.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))
        
        try:
            #----------- Mode and input setup & prompts -------------
            clip_output = None  # Initialize clip_output

            if mode == "keyframes":
                # Keyframes mode processing
                if end_frame is None:
                    raise ValueError("Keyframes mode requires End Frame to be set!")
                
                # Get original dimensions and find bucket size
                end_H, end_W, end_C = end_frame.shape
                height, width = find_nearest_bucket(end_H, end_W, resolution=640)
                debug(f"Using bucket dimensions for keyframe mode: {width}x{height} (from original {end_W}x{end_H})")
                
                # Process start frame
                if start_frame is not None:
                    debug(f"Resizing start frame from {start_frame.shape[1]}x{start_frame.shape[0]} to bucket size {width}x{height}")
                    s_np = resize_and_center_crop(start_frame, target_width=width, target_height=height)
                    input_anchor_np = s_np
                else:
                    # Use gray color with bucket dimensions
                    input_anchor_np = np.ones((height, width, 3), dtype=np.uint8) * 128
                    debug(f"Created gray start frame with bucket dimensions {width}x{height}")
                
                # Process end frame
                end_np = resize_and_center_crop(end_frame, target_width=width, target_height=height)
                
                # VAE encode start frame
                t_encode_start = time.time()
                input_anchor_tensor = torch.from_numpy(input_anchor_np).float() / 127.5 - 1
                input_anchor_tensor = input_anchor_tensor.permute(2, 0, 1)[None, :, None].float()
                start_latent = vae_encode(input_anchor_tensor, self.model_manager.vae.float())
                timings["latent_encoding"] += time.time() - t_encode_start
                
                # VAE encode end frame
                end_tensor = torch.from_numpy(end_np).float() / 127.5 - 1
                end_tensor = end_tensor.permute(2, 0, 1)[None, :, None].float()
                end_latent = vae_encode(end_tensor, self.model_manager.vae.float())
                
                # Text prompt encoding
                if not self.model_manager.high_vram:
                    # Clear GPU memory before loading transformer
                    unload_complete_models(
                        self.model_manager.text_encoder, 
                        self.model_manager.text_encoder_2, 
                        self.model_manager.image_encoder, 
                        self.model_manager.vae, 
                        self.model_manager.transformer
                    )
                    debug("explicitly unloaded all models (before sampling)")
                    
                    # Load only the transformer, with memory preservation
                    move_model_to_device_with_memory_preservation(
                        self.model_manager.transformer, 
                        target_device=gpu, 
                        preserved_memory_gb=gpu_memory_preservation
                    )
                    debug("moved transformer to gpu (memory preservation)")
                
                # Then do CLIP encoding
                if mode == "text2video" or mode == "image2video":
                    clip_output = hf_clip_vision_encode(
                        inp_np, 
                        self.model_manager.feature_extractor, 
                        self.model_manager.image_encoder
                    ).last_hidden_state
                
                lv, cp = encode_prompt_conds(
                    prompt, 
                    self.model_manager.text_encoder, 
                    self.model_manager.text_encoder_2, 
                    self.model_manager.tokenizer, 
                    self.model_manager.tokenizer_2
                )
                lv, mask = crop_or_pad_yield_mask(lv, 512)
                
                lv_n, cp_n = encode_prompt_conds(
                    n_prompt,
                    self.model_manager.text_encoder, 
                    self.model_manager.text_encoder_2, 
                    self.model_manager.tokenizer, 
                    self.model_manager.tokenizer_2
                )
                lv_n, mask_n = crop_or_pad_yield_mask(lv_n, 512)
                
                m = mask
                m_n = mask_n
                
                # CLIP Vision feature extraction
                # ---- Unload all models before sampling
                if not self.model_manager.high_vram:
                    debug("------ MEMORY DEBUG BEFORE TRANSFORMER LOAD ------")
                    debug(f"Section {section}/{total_sections}, preparing to load transformer")
                    # Clear GPU memory
                    debug("Clearing CUDA cache multiple times")
                    for _ in range(5):
                        torch.cuda.empty_cache()
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Print detailed memory stats
                    debug(f"CUDA memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
                    debug(f"CUDA memory reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
                    debug(f"Free memory: {get_cuda_free_memory_gb(gpu):.2f} GB")
                    
                    # Try to diagnose if any tensors are leaking
                    tensor_sizes = {}
                    for obj in gc.get_objects():
                        try:
                            if torch.is_tensor(obj) and obj.device.type == 'cuda':
                                size_mb = obj.nelement() * obj.element_size() / (1024 * 1024)
                                if size_mb > 10:  # Only log large tensors
                                    shape_key = str(tuple(obj.shape))
                                    if shape_key not in tensor_sizes:
                                        tensor_sizes[shape_key] = 0
                                    tensor_sizes[shape_key] += size_mb
                        except:
                            pass
                    
                    if tensor_sizes:
                        debug("Large tensors still in CUDA memory:")
                        for shape, size_mb in sorted(tensor_sizes.items(), key=lambda x: x[1], reverse=True):
                            debug(f"  Shape {shape}: {size_mb:.1f} MB")
                    
                    unload_complete_models(
                        self.model_manager.text_encoder,
                        self.model_manager.text_encoder_2,
                        self.model_manager.image_encoder,
                        self.model_manager.vae,
                        self.model_manager.transformer
                    )
                    debug("explicitly unloaded all models (before sampling)")
                    
                    # Load only the transformer, with memory preservation
                    debug("------ ATTEMPTING TO LOAD TRANSFORMER ------")
                    try:
                        move_model_to_device_with_memory_preservation(
                            self.model_manager.transformer,
                            target_device=gpu,
                            preserved_memory_gb=gpu_memory_preservation
                        )
                        debug("moved transformer to gpu (memory preservation)")
                    except Exception as e:
                        debug(f"CRITICAL ERROR loading transformer: {e}")
                        debug("Attempting emergency memory cleanup...")
                        # Emergency cleanup
                        torch.cuda.empty_cache()
                        gc.collect()
                        # Try to diagnose
                        debug(f"After error: Free memory: {get_cuda_free_memory_gb(gpu):.2f} GB")
                        # Try less memory-intensive approach
                        if get_cuda_free_memory_gb(gpu) > 2.0:
                            debug("Trying alternative approach with less memory")
                            # Continue with CPU processing or smaller batch

                # Set up teacache
                self.model_manager.initialize_teacache(enable_teacache=use_teacache, num_steps=steps if use_teacache else 0)
                
                # Process end frame with CLIP
                end_clip_output = hf_clip_vision_encode(
                    end_np, 
                    self.model_manager.feature_extractor, 
                    self.model_manager.image_encoder
                ).last_hidden_state
                
                if start_frame is not None:
                    # Process start frame with CLIP
                    start_clip_output = hf_clip_vision_encode(
                        input_anchor_np, 
                        self.model_manager.feature_extractor, 
                        self.model_manager.image_encoder
                    ).last_hidden_state
                    
                    # Use weighted combination based on slider value
                    clip_output = (keyframe_weight * start_clip_output + (1.0 - keyframe_weight) * end_clip_output)
                    debug(f"Using weighted combination: {keyframe_weight:.1f} start frame, {1.0-keyframe_weight:.1f} end frame")
                else:
                    # No start frame provided - use 100% end frame embedding
                    clip_output = end_clip_output
                    debug("No start frame provided - using 100% end frame embedding")
                
            elif mode == "text2video":
                # Text2video mode processing
                width, height = self.get_dims_from_aspect(aspect, custom_w, custom_h)
                
                if init_color is not None:
                    r, g, b = self.parse_hex_color(init_color)
                    debug(f"Using color picker value {init_color} -> RGB {r},{g},{b}")
                    input_image_arr = np.zeros((height, width, 3), dtype=np.uint8)
                    input_image_arr[:, :, 0] = int(r * 255)
                    input_image_arr[:, :, 1] = int(g * 255)
                    input_image_arr[:, :, 2] = int(b * 255)
                else:
                    input_image_arr = np.zeros((height, width, 3), dtype=np.uint8)
                    debug("No color provided, defaulting to black")
                
                input_image = input_image_arr
                
                # Continue with general image preparation below
            
            # Handle video extension mode
            if mode == "video_extension":
                if input_video is None:
                    raise ValueError("Video extension mode requires a video to be uploaded!")
                
                debug(f"Processing video extension: direction={extension_direction}")
                
                # Extract frames from the video
                extracted_frames, video_fps, original_dims = extract_frames_from_video(
                    input_video,
                    num_frames=int(extension_frames),
                    from_end=(extension_direction == "Forward"),
                    max_resolution=640
                )
                
                if len(extracted_frames) == 0:
                    raise ValueError("Failed to extract frames from the input video")
                
                # Set input_image based on direction
                if extension_direction == "Forward":
                    # Use last frame as input for forward extension
                    input_image = extracted_frames[-1]
                    debug(f"Using last frame as input_image for forward extension")
                else:
                    # Use first frame as input for backward extension
                    input_image = extracted_frames[0]
                    debug(f"Using first frame as input_image for backward extension")
                
                # Store the extracted frames for later use in video stitching
                all_extracted_frames = extracted_frames
                
                # Redirect to use image2video processing path
                mode = "image2video"
                debug(f"Redirecting to image2video path with selected frame as input")
            
            # General preparation for image2video or text2video
            if mode != "keyframes":
                debug("Preparing inputs")
                if self.stream:
                    self.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
                
                # Use prepare_inputs for standard processing
                inp_np, inp_tensor, lv, cp, lv_n, cp_n, m, m_n, height, width = self.prepare_inputs(
                    input_image, prompt, n_prompt, cfg,
                    gaussian_blur_amount=gaussian_blur_amount,
                    llm_weight=llm_weight,
                    clip_weight=clip_weight
                )
                
                # VAE encode
                t_encode_start = time.time()
                start_latent = vae_encode(inp_tensor.float(), self.model_manager.vae.float())
                timings["latent_encoding"] += time.time() - t_encode_start
                
                debug("VAE encoded")
                if self.stream:
                    self.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
                
                # CLIP vision encoding for image2video and text2video
                if not self.model_manager.high_vram:
                    # Offload transformer
                    offload_model_from_device_for_memory_preservation(
                        self.model_manager.transformer, 
                        target_device=gpu, 
                        preserved_memory_gb=8
                    )
                    
                    # Load VAE for decoding
                    load_model_as_complete(self.model_manager.vae, target_device=gpu)
                    debug("loaded vae to gpu")
                
                if mode == "text2video" or mode == "image2video":
                    clip_output = hf_clip_vision_encode(
                        inp_np, 
                        self.model_manager.feature_extractor, 
                        self.model_manager.image_encoder
                    ).last_hidden_state
                
                debug("Got CLIP output last_hidden_state")
            
            # Convert tensors to proper dtype
            lv = lv.to(self.model_manager.transformer.dtype)
            lv_n = lv_n.to(self.model_manager.transformer.dtype)
            cp = cp.to(self.model_manager.transformer.dtype)
            cp_n = cp_n.to(self.model_manager.transformer.dtype)
            
            if clip_output is not None:
                clip_output = clip_output.to(self.model_manager.transformer.dtype)
            
            if self.stream:
                self.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
            
            # Setup generator for reproducibility
            rnd = torch.Generator("cpu").manual_seed(seed)
            
            # Initialize history tensors
            history_latents = torch.zeros(
                size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32
            ).cpu()
            history_pixels = None
            
            # Calculate latent paddings list for info - PRESERVING ORIGINAL LOGIC
            if total_sections > 4:
                latent_paddings_list_for_info = [3] + [2] * (total_sections - 3) + [1, 0]
                debug(f"Calculated padding list {latent_paddings_list_for_info} for info (len={len(latent_paddings_list_for_info)})")
            else:
                latent_paddings_list_for_info = list(reversed(range(total_sections)))
                debug(f"Calculated standard padding list {latent_paddings_list_for_info} for info (len={len(latent_paddings_list_for_info)})")
            
            # Use the original iteration scheme - CRITICAL
            debug(f"[TESTING] Forcing old iteration scheme: reversed(range(total_sections={total_sections}))")
            loop_iterator = reversed(range(total_sections))
            
            graceful_stop = False
            
            # Process each section - PRESERVING ORIGINAL SECTION LOOP
            for section in loop_iterator:  # Iterates from total_sections-1 down to 0
                # Determine section properties (Unchanged)
                is_last_section = section == 0
                latent_padding_size = section * latent_window_size
                is_first_iteration = (section == total_sections - 1)
                debug(f'section = {section}, latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')
                
                # Check for graceful stop
                if self.stream and self.stream.input_queue.top() == 'graceful_end':
                    debug("input_queue 'graceful_end' received. Will stop after current section.")
                    graceful_stop = True
                
                # Check at end of loop too
                if graceful_stop:
                    debug("graceful stop requested, ending generation after completing section.")
                    break
                
                # Setup indices and clean latents - PRESERVING ORIGINAL LOGIC
                split_sizes = [1, latent_padding_size, latent_window_size, 1, 2, 16]
                total_indices = sum(split_sizes)
                indices = torch.arange(total_indices).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split(split_sizes, dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
                
                # Setup the clean latents for ALL modes first
                clean_latents_pre = start_latent.to(history_latents.device, dtype=history_latents.dtype)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
                
                # Special handling for keyframes mode
                if mode == "keyframes" and is_first_iteration:
                    debug("Keyframes mode: Overriding clean_latents_post with end_latent for first iteration.")
                    clean_latents_post = end_latent.to(history_latents.device, dtype=history_latents.dtype)
                    clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
                
                # Mask fallback safeguard (Unchanged)
                m = m if m is not None else torch.ones_like(lv)
                m_n = m_n if m_n is not None else torch.ones_like(lv_n)
                
                # Memory management before sampling
                if not self.model_manager.high_vram:
                    unload_complete_models(
                        self.model_manager.text_encoder, 
                        self.model_manager.text_encoder_2, 
                        self.model_manager.image_encoder, 
                        self.model_manager.vae, 
                        self.model_manager.transformer
                    )
                    debug("explicitly unloaded all models (before sampling)")
                    
                    move_model_to_device_with_memory_preservation(
                        self.model_manager.transformer, 
                        target_device=gpu, 
                        preserved_memory_gb=gpu_memory_preservation
                    )
                    debug("moved transformer to gpu (memory preservation)")
                
                # Initialize TeaCache
                self.model_manager.initialize_teacache(
                    enable_teacache=use_teacache, 
                    num_steps=steps if use_teacache else 0
                )
                debug("teacache initialized", "use_teacache", use_teacache)
                
                # Define callback function for this section
                def callback(d):
                    nonlocal graceful_stop
                    
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
                    desc = f'Pixel frames generated: {actual_pixel_frames}, Length: {actual_seconds:.2f}s (FPS-30)'
                    
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
                        prompt_embeds=lv,
                        prompt_embeds_mask=m,
                        prompt_poolers=cp,
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
                        prompt_embeds=lv,
                        prompt_embeds_mask=m,
                        prompt_poolers=cp,
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
                    offload_model_from_device_for_memory_preservation(
                        self.model_manager.transformer, 
                        target_device=gpu, 
                        preserved_memory_gb=8
                    )
                    load_model_as_complete(self.model_manager.vae, target_device=gpu)
                    debug("loaded vae to gpu")
                
                # Clear CUDA cache to reduce fragmentation
                torch.cuda.empty_cache()
                debug("Cleared CUDA cache before decoding")
                
                # Calculate theoretical max overlap frames
                max_overlapped_frames = latent_window_size * 4 - 3
                debug(f"Theoretical max overlapped_frames={max_overlapped_frames}")
                
                try:
                    # Decode the newly generated latents in chunks - CRITICAL LOGIC
                    t_vae_start = time.time()
                    debug(f"Decoding generated_latents with shape: {generated_latents.shape}")
                    
                    chunk_size = 4  # Small enough to avoid OOM
                    current_pixels_chunks = []
                    
                    for i in range(0, generated_latents.shape[2], chunk_size):
                        end_idx = min(i + chunk_size, generated_latents.shape[2])
                        debug(f"Decoding chunk {i}:{end_idx} of {generated_latents.shape[2]}")
                        
                        # Move chunk to GPU, decode, then immediately move result to CPU
                        chunk = generated_latents[:, :, i:end_idx].to(self.model_manager.vae.device, dtype=self.model_manager.vae.dtype)
                        chunk_pixels = vae_decode(chunk, self.model_manager.vae).cpu()
                        current_pixels_chunks.append(chunk_pixels)
                        
                        # Force cleanup of chunk tensors
                        del chunk, chunk_pixels
                        torch.cuda.empty_cache()
                    
                    # Combine all chunks on CPU
                    current_pixels = torch.cat(current_pixels_chunks, dim=2)
                    debug(f"Successfully decoded in chunks: final shape {current_pixels.shape}")
                    
                    del current_pixels_chunks
                    torch.cuda.empty_cache()
                    
                    # End VAE decode timing
                    timings["vae_decode_time"] += time.time() - t_vae_start
                    
                    debug(f"Decoded newly generated latents to pixels with shape: {current_pixels.shape}")
                    
                    # Initialize or update history_pixels
                    if history_pixels is None:
                        history_pixels = current_pixels
                        debug(f"First section: Set history_pixels directly with shape: {history_pixels.shape}")
                    else:
                        # Important: simple concatenation to avoid blending issues
                        history_pixels = torch.cat([current_pixels, history_pixels], dim=2)
                        debug(f"Concatenated new frames without blending. New history shape: {history_pixels.shape}")
                    
                    # Save preview video
                    preview_filename = os.path.join(self.output_folder, f'{job_id}_preview_{uuid.uuid4().hex}.mp4')
                    try:
                        save_bcthw_as_mp4(history_pixels, preview_filename, fps=30)
                        debug(f"[FILE] Preview video saved: {preview_filename} ({os.path.exists(preview_filename)})")
                        if self.stream:
                            self.stream.output_queue.push(('preview_video', preview_filename))
                            debug(f"[QUEUE] Queued preview_video event: {preview_filename}")
                    except Exception as e:
                        debug(f"[ERROR] Failed to save preview video: {e}")
                    
                    # Clean up memory
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
                save_bcthw_as_mp4(history_pixels, extension_filename, fps=30)
                debug(f"[FILE] Extension video saved as {extension_filename}")
                
                # Now combine with the original video
                combined_filename = os.path.join(self.output_folder, f'{job_id}_combined.mp4')
                
                try:
                    import subprocess
                    
                    # Prepare videos for concatenation
                    debug("[FFMPEG] Preparing videos for concatenation...")
                    temp_ext = os.path.join(self.output_folder, f'{job_id}_ext_compat.mp4')
                    temp_orig = os.path.join(self.output_folder, f'{job_id}_orig_compat.mp4')
                    
                    # Convert extension to compatible format
                    convert_cmd = [
                        "ffmpeg", "-y",
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
                    
                    # Convert original video to compatible format
                    convert_cmd = [
                        "ffmpeg", "-y",
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
                    
                    # Run concat
                    concat_cmd = [
                        "ffmpeg", "-y",
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
                        make_mp4_faststart(combined_filename)
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
                    drop_n = math.floor(N_actual * 0.2)
                    debug(f"normal trim: dropping first {drop_n} of {N_actual}")
                
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
                    drop_n = math.floor(N_actual * 0.2)  # Default to 20% trim
                    debug(f"keyframe normal trim: dropping first {drop_n} of {N_actual}")
                
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
                    fix_video_compatibility(combined_filename, fps=30)
                    
                    if self.stream:
                        self.stream.output_queue.push(('file', combined_filename))
                    
                    self.cleanup_temp_files(job_id, keep_final=True, final_output_path=combined_filename)
                    return combined_filename
                
                # Standard video export
                elif not image_likely_saved:
                    debug(f"[FILE] Attempting to save final video to {output_filename}")
                    
                    try:
                        save_bcthw_as_mp4(history_pixels, output_filename, fps=30)
                        debug(f"[FILE] Video successfully saved to {output_filename}: {os.path.exists(output_filename)}")
                        fix_video_compatibility(output_filename, fps=30)
                        
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
                unload_complete_models(
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
                unload_complete_models(
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
            
            summary_string = (
                f"Finished!\n"
                f"Total generated frames: {trimmed_frames}, "
                f"Video length: {video_seconds:.2f} seconds (FPS-30), "
                f"Time taken: {minutes}m {seconds:.2f}s.\n\n"
                f"⏱️ Performance Breakdown:\n"
                f"• Latent encoding: {timings['latent_encoding']:.2f}s\n"
                f"• Average step time: {avg_step_time:.2f}s\n"
                f"• Active generation: {timings['generation_time']:.2f}s\n"
                f"• VAE decoding: {timings['vae_decode_time']:.2f}s\n"
                f"• Other processing: {misc_time:.2f}s"
            )
            
            if self.stream:
                self.stream.output_queue.push(('progress', (None, summary_string, "")))
                self.stream.output_queue.push(('end', None))
            
            return output_filename
