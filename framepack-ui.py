from diffusers_helper.hf_login import login

import os
import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import random
import math
from PIL import Image
import torchvision # Keep this import

from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument('--hf-cache', choices=['local', 'global'], default='local')
parser.add_argument('--host', type=str, default='0.0.0.0') # Changed from --server
parser.add_argument('--port', type=int, default=7800) # Default port if not specified
parser.add_argument('--inbrowser', action='store_true')
args = parser.parse_args()



ASPECT_RATIOS = {
    "1:1": (768, 768),
    "16:9": (1024, 576),
    "9:16": (576, 1024),
    "4:3": (896, 672),
    "3:4": (672, 896),
    "21:9": (1280, 576), # Approx 2.22:1, common ultrawide
    "9:21": (576, 1280),
    # Add more if desired, ensure W/H are divisible by 8, preferably 64
}
DEFAULT_ASPECT_RATIO = "16:9" # Or choose another default like 1:1


if args.hf_cache == 'local':
    if not os.path.exists('./hf_download'):
        os.makedirs('./hf_download')
    os.environ['HF_HOME'] = os.path.abspath(os.path.join(os.path.dirname(__file__), './hf_download'))

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60
print(f'Free VRAM {free_mem_gb:.2f} GB')
print(f'Using CUDA device: {gpu}')
print(f'High-VRAM Mode: {high_vram}')

# --- Load Models ---
print("-" * 20)
print("Loading models to CPU...")

hf_hub_kwargs = {"resume_download": True} # Optional: Add resume download capability

model_repo_hy = "hunyuanvideo-community/HunyuanVideo"
model_repo_flux = "lllyasviel/flux_redux_bfl"
model_repo_fp = "lllyasviel/FramePackI2V_HY"

print(f"Loading Llama text_encoder from: {model_repo_hy}")
text_encoder = LlamaModel.from_pretrained(model_repo_hy, subfolder='text_encoder', torch_dtype=torch.float16, **hf_hub_kwargs).cpu()

print(f"Loading CLIP text_encoder_2 from: {model_repo_hy}")
text_encoder_2 = CLIPTextModel.from_pretrained(model_repo_hy, subfolder='text_encoder_2', torch_dtype=torch.float16, **hf_hub_kwargs).cpu()

print(f"Loading Llama tokenizer from: {model_repo_hy}")
tokenizer = LlamaTokenizerFast.from_pretrained(model_repo_hy, subfolder='tokenizer', **hf_hub_kwargs)

print(f"Loading CLIP tokenizer_2 from: {model_repo_hy}")
tokenizer_2 = CLIPTokenizer.from_pretrained(model_repo_hy, subfolder='tokenizer_2', **hf_hub_kwargs)

print(f"Loading VAE (HunyuanVideo) from: {model_repo_hy}")
vae = AutoencoderKLHunyuanVideo.from_pretrained(model_repo_hy, subfolder='vae', torch_dtype=torch.float16, **hf_hub_kwargs).cpu()

print(f"Loading Siglip feature_extractor from: {model_repo_flux}")
feature_extractor = SiglipImageProcessor.from_pretrained(model_repo_flux, subfolder='feature_extractor', **hf_hub_kwargs)

print(f"Loading Siglip image_encoder from: {model_repo_flux}")
image_encoder = SiglipVisionModel.from_pretrained(model_repo_flux, subfolder='image_encoder', torch_dtype=torch.float16, **hf_hub_kwargs).cpu()

print(f"Loading FramePack Transformer from: {model_repo_fp}")
transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(model_repo_fp, torch_dtype=torch.bfloat16, **hf_hub_kwargs).cpu()

print("All models loaded to CPU.")
print("-" * 20)

# --- Configure Models ---
all_models = (vae, text_encoder, text_encoder_2, image_encoder, transformer)
for m in all_models:
    m.eval().requires_grad_(False)

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()
    # Install dynamic swap for models that will be moved on/off GPU
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    # image_encoder and vae will be loaded fully when needed
else:
    print("Moving models to GPU (High VRAM mode)...")
    for m in all_models:
        m.to(gpu)
    print("Models moved to GPU.")

# Set dtypes and other settings AFTER potential move to GPU
transformer.high_quality_fp32_output_for_inference = True
transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

stream = AsyncStream()
outputs_folder = './outputs/'; os.makedirs(outputs_folder, exist_ok=True)

# -------- Worker (handles all modes) --------
@torch.no_grad()
def worker(mode, input_image, input_video, aspect_ratio_str,
           prompt, n_prompt, 
           shift, cfg, gs, rs,
           strength, seed, total_second_length, latent_window_size,
           steps, gpu_memory_preservation, use_teacache):

    job_id = generate_timestamp()
    is_image_mode = mode in ['txt2img', 'img2img']
    output_type_flag = 'image_file' if is_image_mode else 'video_file' # Flag for process_fn

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting job...'))))
    print(f"Starting worker job {job_id} with mode: {mode}")

    try:
        # --- Clean GPU and Load Text Encoders ---
        if not high_vram:
            unload_complete_models(*all_models)
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding...'))))

        # --- Prepare prompt embeddings ---
        llama_vec, clip_pool = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        if cfg == 1:
            llama_vec_n, clip_pool_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_pool)
        else:
            llama_vec_n, clip_pool_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        llama_vec, llama_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        llama_vec = llama_vec.to(transformer.dtype); llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_pool = clip_pool.to(transformer.dtype); clip_pool_n = clip_pool_n.to(transformer.dtype)

        # --- Prepare latents and image embeddings per mode ---
        height, width = None, None
        init_latent = None      # Latent passed to sampler's initial_latent arg (used by txt2vid, vid2vid)
        concat_latent = None    # Latent passed to sampler's concat_latent arg (used by vid2vid, extend_vid)
        image_embeddings = None # CLIP vision embeddings (used by img2vid)
        first_frame_conditioning_latent = None # Latent used only for clean_latents conditioning on step 0 (img2vid, extend_vid)
        black_frame_latent = None # Variable to store black frame VAE latent if needed
        black_frame_embeddings = None # Variable to store black frame CLIP embeddings if needed

        
        # --- Load VAE and Image Encoder (conditionally) ---
        # Ensure Image Encoder is loaded for ALL modes now (to encode black frame)
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu) # Load VAE if needed by any mode
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE ready...'))))
            # Image encoder needed for img2vid, img2img, AND the black frame for others
            load_model_as_complete(image_encoder, target_device=gpu)
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image Encoder ready...'))))

        # --- Mode Specific Setup ---
        # Get target W/H for modes that use Aspect Ratio
        if mode in ['txt2img', 'img2img', 'txt2vid']:
            if aspect_ratio_str in ASPECT_RATIOS: width, height = ASPECT_RATIOS[aspect_ratio_str]
            else: width, height = ASPECT_RATIOS[DEFAULT_ASPECT_RATIO]; print(f"Warning: Invalid aspect ratio...")
            print(f"Mode {mode}: Using aspect ratio {aspect_ratio_str} -> {width}x{height}")

        # --- Generate Black Frame Latent & Embeddings (if needed) ---
        if mode in ['txt2img', 'txt2vid']:
            print(f"Mode {mode}: Generating black frame latent & embeddings for {width}x{height}...")
            # Create black frame numpy array
            black_np = np.zeros((height, width, 3), dtype=np.uint8)
            # VAE Encode black frame (using float32 fix)
            original_vae_dtype = vae.dtype
            try:
                vae.to(dtype=torch.float32)
                inp_black = (torch.from_numpy(black_np).float() / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device=gpu)
                black_frame_latent = vae_encode(inp_black, vae).to(transformer.dtype)
            finally:
                vae.to(dtype=original_vae_dtype)
            print(f"  Black frame VAE latent shape: {black_frame_latent.shape}")
            # Calculate CLIP embeddings for black frame
            if not high_vram: load_model_as_complete(image_encoder, target_device=gpu) # Ensure loaded
            black_frame_embeddings_out = hf_clip_vision_encode(black_np, feature_extractor, image_encoder)
            black_frame_embeddings = black_frame_embeddings_out.last_hidden_state.to(transformer.dtype)
            print(f"  Black frame CLIP embeddings shape: {black_frame_embeddings.shape}")

        # --- Specific Mode Setup Logic ---
        if mode == 'txt2img':
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Preparing for txt2img...'))))
            # Conditioning only mode, starts from noise. Uses black frame latent/embeddings.
            init_latent = None
            concat_latent = None
            first_frame_conditioning_latent = black_frame_latent # Use black frame latent for initial conditioning
            image_embeddings = black_frame_embeddings # Use black frame embeddings
            print(f"Txt2Img: Setup complete.")

        elif mode == 'img2img':
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Preparing for img2img...'))))
            # Uses initial_latent + strength. Needs REAL image embeddings.
            if input_image is None: raise ValueError("Input image required.")
            np_img = np.array(input_image)
            img_resized = resize_and_center_crop(np_img, target_width=width, target_height=height) # Uses target AR size
            print(f"Img2Img: Resized image to {width}x{height}")
            # VAE Encode for init_latent (using float32 fix)
            original_vae_dtype = vae.dtype
            try:
                vae.to(dtype=torch.float32)
                inp_for_vae = (torch.from_numpy(img_resized).float() / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device=gpu)
                init_latent = vae_encode(inp_for_vae, vae).to(transformer.dtype)
                first_frame_conditioning_latent = init_latent # Also use for clean_latents
            finally:
                vae.to(dtype=original_vae_dtype)
            print(f"Img2Img: Initial latent shape {init_latent.shape}")
            # Calculate REAL Image Embeddings
            if not high_vram: load_model_as_complete(image_encoder, target_device=gpu)
            image_encoder_output = hf_clip_vision_encode(img_resized, feature_extractor, image_encoder)
            image_embeddings = image_encoder_output.last_hidden_state.to(transformer.dtype)
            print(f"Img2Img: Image embedding shape {image_embeddings.shape}")
            concat_latent = None

        elif mode == 'txt2vid':
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Preparing for txt2vid...'))))
            # Conditioning only mode, starts from noise. Uses black frame latent/embeddings.
            init_latent = None
            concat_latent = None
            first_frame_conditioning_latent = black_frame_latent # Use black frame
            image_embeddings = black_frame_embeddings # Use black frame
            print(f"Txt2Vid: Setup complete.")

        elif mode == 'img2vid':
            # Conditioning only mode, starts from noise. Uses REAL image latent/embeddings.
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Preparing for img2vid...'))))
            if input_image is None: raise ValueError("Input image required.")
            np_img = np.array(input_image)
            H, W, _ = np_img.shape
            height, width = find_nearest_bucket(H, W, resolution=640) # Size from input
            img_resized = resize_and_center_crop(np_img, target_width=width, target_height=height)
            print(f"Img2Vid: Resized/bucketed image to {width}x{height}")
            # VAE Encode real first frame for conditioning (using float32 fix)
            original_vae_dtype = vae.dtype
            try:
                vae.to(dtype=torch.float32)
                inp_for_vae = (torch.from_numpy(img_resized).float() / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device=gpu)
                first_frame_conditioning_latent = vae_encode(inp_for_vae, vae).to(transformer.dtype)
            finally:
                vae.to(dtype=original_vae_dtype)
            print(f"Img2Vid: First frame latent for conditioning: {first_frame_conditioning_latent.shape}")
            # Calculate REAL Image Embeddings
            if not high_vram: load_model_as_complete(image_encoder, target_device=gpu)
            image_encoder_output = hf_clip_vision_encode(img_resized, feature_extractor, image_encoder)
            image_embeddings = image_encoder_output.last_hidden_state.to(transformer.dtype)
            print(f"Img2Vid: Image embedding shape {image_embeddings.shape}")
            init_latent = None; concat_latent = None

        elif mode == 'extend_vid':
            # Conditioning only mode, starts from noise. Uses REAL first frame latent/concat. Needs CLIP embeddings.
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Preparing for {mode}...'))))
            if input_video is None or not hasattr(input_video, 'name'): raise ValueError("Input video file required.")
            video_path = input_video.name; print(f"Video Mode: Loading video {video_path}")
            vid_frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
            vid = vid_frames.permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0
            _, _, T, H, W = vid.shape; height, width = H, W # Derive size from input
            print(f"Video Mode: Loaded video with {T} frames, {W}x{H}")
            original_vae_dtype = vae.dtype
            try:
                vae.to(dtype=torch.float32) # Apply fix
                first_frame_np = vid_frames[0].numpy()
                inp_first_frame = (torch.from_numpy(first_frame_np).float() / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device=gpu)
                first_frame_conditioning_latent = vae_encode(inp_first_frame, vae).to(transformer.dtype)
                print(f"ExtendVid: First frame latent for conditioning: {first_frame_conditioning_latent.shape}")
                print(f"ExtendVid: Encoding full video for concat_latent...")
                concat_latent = vae_encode(vid.to(device=gpu), vae).to(transformer.dtype)
                print(f"ExtendVid: Conditioning concat_latent shape {concat_latent.shape}")
            finally:
                vae.to(dtype=original_vae_dtype)
             # Needs CLIP embeddings from the first frame for the sampler
             print(f"ExtendVid: Calculating CLIP embeddings from first frame...")
             if not high_vram: load_model_as_complete(image_encoder, target_device=gpu)
             first_frame_np = vid_frames[0].numpy() # Already loaded
             image_encoder_output = hf_clip_vision_encode(first_frame_np, feature_extractor, image_encoder)
             image_embeddings = image_encoder_output.last_hidden_state.to(transformer.dtype)
             print(f"ExtendVid: Image embedding shape {image_embeddings.shape}")

             init_latent = None # Start from noise

        elif mode == 'vid2vid':
             # (Keep logic from previous step, but get size from input, apply VAE fix)
             stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Preparing for {mode}...'))))
             if input_video is None or not hasattr(input_video, 'name'): raise ValueError("Input video file required.")
             video_path = input_video.name; print(f"Video Mode: Loading video {video_path}")
             vid_frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
             vid = vid_frames.permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0
             _, _, T, H, W = vid.shape; height, width = H, W # Derive size from input
             print(f"Video Mode: Loaded video with {T} frames, {W}x{H}")
             original_vae_dtype = vae.dtype
             try:
                 vae.to(dtype=torch.float32) # Apply fix
                 first_frame_np = vid_frames[0].numpy()
                 inp_first_frame = (torch.from_numpy(first_frame_np).float() / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device=gpu)
                 init_latent = vae_encode(inp_first_frame, vae).to(transformer.dtype) # This IS the initial latent
                 first_frame_conditioning_latent = init_latent # Use for clean_latents too
                 print(f"Vid2Vid: Initial latent for sampler: {init_latent.shape}")
                 print(f"Vid2Vid: Encoding full video for concat_latent...")
                 concat_latent = vae_encode(vid.to(device=gpu), vae).to(transformer.dtype)
                 print(f"Vid2Vid: Conditioning concat_latent shape {concat_latent.shape}")
             finally:
                 vae.to(dtype=original_vae_dtype)
             print(f"Vid2Vid: Calculating CLIP embeddings from first frame...")
             if not high_vram: load_model_as_complete(image_encoder, target_device=gpu)
             first_frame_np = vid_frames[0].numpy() # Already loaded
             image_encoder_output = hf_clip_vision_encode(first_frame_np, feature_extractor, image_encoder)
             image_embeddings = image_encoder_output.last_hidden_state.to(transformer.dtype)
             print(f"Vid2Vid: Image embedding shape {image_embeddings.shape}")

        # Fallback sizing (should only hit if AR lookup failed)
        if height is None or width is None: height, width = 512, 512; print(f"Warning: Using fallback size {width}x{height}")

        # --- Sampling loop Setup ---
        effective_latent_window_size = 1 if is_image_mode else latent_window_size
        if is_image_mode:
            total_latent_sections = 1
            latent_paddings = [0] # Force single run, no padding needed
            print(f"Image Mode: Forcing single generation section with effective window size 1.")
            num_frames_per_window = 1 # <<< SET TO 1 FOR IMAGE MODE
            print(f"Image Mode: Setting num_frames_per_window to {num_frames_per_window}")
        else: # Video modes
            # Use UI value for window size
            effective_latent_window_size = latent_window_size
            total_latent_sections = max(int(round((total_second_length * 30) / (effective_latent_window_size * 4))), 1)
            if total_latent_sections > 4: latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            else: latent_paddings = list(reversed(range(total_latent_sections)))
            num_frames_per_window = effective_latent_window_size * 4 - 3 # <<< Use effective size
        stream.output_queue.push(('progress', (None, f'Preparing for {total_latent_sections} sections...', make_progress_bar_html(0, 'Initializing Sampler...'))))
        rnd = torch.Generator("cpu").manual_seed(int(seed))
        # Use default window size even for images, loop runs once anyway
        num_frames_per_window = latent_window_size * 4 - 3
        history_latents = torch.zeros((1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu() # Use determined H/W
        history_pixels = None
        total_generated_latent_frames = 0
        print(f"Latent padding sequence: {latent_paddings}")

        # --- Main Generation Loop (runs once for image modes) ---
        for section_index, latent_padding in enumerate(latent_paddings):
            is_last_section = (latent_padding == 0)
            latent_padding_size = latent_padding * effective_latent_window_size # <<< Use effective size

            print(f"\n--- Section {section_index + 1}/{total_latent_sections} (Padding: {latent_padding}, Last: {is_last_section}) ---")

            if stream.input_queue.top() == 'end': print("User requested stop."); stream.output_queue.push(('end', None)); return

            # Prepare clean latents for conditioning
            indices = torch.arange(0, sum([1, latent_padding_size, effective_latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, _, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = \
                indices.split([1, latent_padding_size, effective_latent_window_size, 1, 2, 16], dim=1) # Use effective size
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
    
            # Determine the 'clean' reference latent for the *conditioning* input's 'pre' part
            # For modes starting from an image/video, ALWAYS use the first frame's VAE latent as the main reference.
            if mode in ['img2vid', 'extend_vid', 'vid2vid']:
                 if first_frame_conditioning_latent is not None:
                     clean_latents_pre_cond = first_frame_conditioning_latent.to(history_latents.device, dtype=history_latents.dtype)
                     print(f"Using first frame VAE latent for clean_latents_pre conditioning (Section {section_index+1}).")
                 else:
                     # Fallback if first frame latent wasn't prepared (shouldn't happen for these modes)
                     print(f"Warning: Expected first_frame_conditioning_latent for {mode} but not found, using history.")
                     clean_latents_pre_cond = history_latents[:, :, :1] # Use history as fallback ONLY
            elif mode == 'txt2img': # Treat txt2img similar to txt2vid here
                # Use the zero history buffer as reference (effectively no structural prior)
                print(f"{mode}: Using history[:1] (zeros) for clean_latents_pre conditioning (Section {section_index+1}).")
                clean_latents_pre_cond = history_latents[:, :, :1]
            else: # Should only be txt2vid now based on earlier setup
                if init_latent is not None: # txt2vid (method with strength=1) uses zero init_latent as reference
                     clean_latents_pre_cond = init_latent.to(history_latents.device, dtype=history_latents.dtype)
                     print(f"{mode}: Using zero init_latent for clean_latents_pre conditioning (Section {section_index+1}).")
                else: # Fallback for txt2vid if init_latent somehow None
                     print(f"{mode}: Using history[:1] (zeros) for clean_latents_pre conditioning (Section {section_index+1}).")
                     clean_latents_pre_cond = history_latents[:, :, :1]
    
    
            # Get the 'post', '2x', and '4x' parts from history buffer (represents recently generated context)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16].split([1, 2, 16], dim=2)
    
            # Combine the constant first-frame reference with the recent history context
            clean_latents_for_cond = torch.cat([clean_latents_pre_cond, clean_latents_post], dim=2)
            print(f"Shape of clean_latents_for_cond: {clean_latents_for_cond.shape}")

            # --- Load Transformer (conditionally) ---
            if not high_vram:
                unload_complete_models(vae, image_encoder, text_encoder, text_encoder_2) # Keep transformer target device in mind
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Transformer ready...'))))

            # Configure TeaCache
            transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=steps)

            # Progress Callback
            def callback(d):
                if stream.input_queue.top() == 'end':
                    print("Stop signal received in callback, allowing current step/section to finish.")
                    # DO NOT raise KeyboardInterrupt here.
                    # Just prevent further progress updates for this stopped task.
                    return # Exit the callback early

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling Section {section_index + 1}/{total_latent_sections}, Step {current_step}/{steps}'
                desc = f'Total generated frames: ~{int(max(0, total_generated_latent_frames * 4 - 3))}/{int(total_second_length * 30)}. Running...'

                try:
                    preview = d.get('denoised', None)
                    if preview is not None:
                        preview = vae_decode_fake(preview) # Quick decode for preview
                        preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                        preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
                    else:
                         preview = None # Or a placeholder image?
                except Exception as e:
                    print(f"Error during preview generation: {e}")
                    preview = None # Fallback

                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))


            # --- Run Sampling ---
            # Pass the correct initial_latent based on mode
            print(f"Calling sample_hunyuan with: mode={mode}, shift={shift}, strength={strength if mode in ['txt2vid', 'vid2vid'] else 'N/A'}, cfg={cfg}, gs={gs}, rs={rs}")
            print(f"  initial_latent provided: {'Yes' if init_latent is not None else 'No'}")
            print(f"  concat_latent provided: {'Yes' if concat_latent is not None else 'No'}")
            print(f"  image_embeddings provided: {'Yes' if image_embeddings is not None else 'No'} (Shape: {image_embeddings.shape if image_embeddings is not None else 'N/A'})") # Log shape

            generated_latents = sample_hunyuan(
                transformer=transformer,
                initial_latent=init_latent, 
                concat_latent=concat_latent,
                strength=strength, 
                width=width, height=height,
                frames=num_frames_per_window,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                shift=shift if shift is not None and shift > 0 else 3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_mask,
                prompt_poolers=clip_pool,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_mask_n,
                negative_prompt_poolers=clip_pool_n,
                device=gpu,
                dtype=transformer.dtype,
                image_embeddings=image_embeddings,
                latent_indices=latent_indices,
                clean_latents=clean_latents_for_cond.to(gpu, dtype=transformer.dtype), # Use the prepared conditioning latent
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x.to(gpu, dtype=transformer.dtype),
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x.to(gpu, dtype=transformer.dtype),
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback
            )
            print(f"Section {section_index + 1} sampling complete. Output latent shape: {generated_latents.shape}")

            # --- Post-processing and History Update ---
            # Prepend the clean first frame latent for img2vid on the last section
            if is_last_section:
                 if mode in ['img2vid', 'extend_vid'] or (mode in ['txt2img', 'txt2vid'] and black_frame_latent is not None):
                      # Prepend the conditioning frame (real or black)
                      cond_latent_to_prepend = first_frame_conditioning_latent # This holds black_frame_latent for txt modes
                      if cond_latent_to_prepend is not None:
                           print(f"Prepending conditioning latent for {mode} final output.")
                           generated_latents = torch.cat([cond_latent_to_prepend.to(generated_latents.device, dtype=generated_latents.dtype), generated_latents], dim=2)
                      else: print(f"Warning: Cannot prepend conditioning latent for {mode}, as it's missing.")
                 elif init_latent is not None and mode == 'vid2vid':
                      # Prepend initial latent for vid2vid
                      print(f"Prepending initial latent to the final output (vid2vid).")
                      generated_latents = torch.cat([init_latent.to(generated_latents.device, dtype=generated_latents.dtype), generated_latents], dim=2)

            # (History update logic - keep as is)
            current_section_latent_frames = generated_latents.shape[2]
            total_generated_latent_frames += current_section_latent_frames
            print(f"Total generated latent frames so far: {total_generated_latent_frames}")
            history_latents = torch.cat([generated_latents.cpu().to(history_latents.dtype), history_latents], dim=2)


            # --- VAE Decode Section (Using simple demo logic) ---
            # (Keep this section as is, using the reverted logic from previous step)
            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(100, 'VAE Decoding...'))))
            real_history_latents_gpu = history_latents[:, :, :total_generated_latent_frames].to(gpu, dtype=vae.dtype)
            print(f"Decoding latents for append. Full history shape on GPU: {real_history_latents_gpu.shape}")
            
            
            # Decode ALL generated latents for this chunk
            decoded_pixels = vae_decode(real_history_latents_gpu, vae).cpu()

            if is_image_mode:
                 # For image mode, take frame (handle if black frame was prepended)
                 frame_index_to_save = 0
                 if mode == 'txt2img':
                      if decoded_pixels.shape[2] > 1: # Check if black frame was prepended + generated frame exists
                           frame_index_to_save = 1 # Save the second frame (index 1)
                           print(f"Image Mode (txt2img): Extracting frame {frame_index_to_save} (skipping prepended black frame).")
                      else: # Only one frame generated (shouldn't happen if prepend worked?)
                           frame_index_to_save = 0
                           print(f"Image Mode (txt2img): Only one frame found, extracting frame {frame_index_to_save}.")
                 else: # img2img
                      frame_index_to_save = 0 # Save the first frame
                      print(f"Image Mode (img2img): Extracting frame {frame_index_to_save}.")

                 if decoded_pixels.shape[2] > frame_index_to_save:
                      output_image_pixels = decoded_pixels[:, :, frame_index_to_save]
                      output_image_np = ((output_image_pixels[0].permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
                      output_filename = os.path.join(outputs_folder, f'{job_id}_final.png')
                      print(f"Saving final image to {output_filename}")
                      Image.fromarray(output_image_np).save(output_filename)
                      stream.output_queue.push((output_type_flag, output_filename))
                 else:
                      print(f"Error: Could not extract frame index {frame_index_to_save} from decoded pixels with shape {decoded_pixels.shape}")
                      stream.output_queue.push(('error', f"Failed to extract output frame for {mode}."))

                 history_pixels = None
            else: # Video Mode VAE/Append logic
                 # For txt2vid, trim the first (black) frame AFTER appending is done
                 if history_pixels is None:
                     history_pixels = decoded_pixels
                 else:
                     # Use the same soft append logic as before
                     overlap_pixel_frames = effective_latent_window_size * 4 - 3 # <<< Use effective size
                     section_latent_frames = (effective_latent_window_size * 2 + 1) if is_last_section else (effective_latent_window_size * 2) # <<< Use effective size
                     slice_to_decode = real_history_latents_gpu[:, :, :min(section_latent_frames, real_history_latents_gpu.shape[2])]
                     current_pixels = vae_decode(slice_to_decode, vae).cpu() # Decode the slice again for append context
                     history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlap=overlap_pixel_frames)
                 print(f"Current video history_pixels shape: {history_pixels.shape}")
                 output_filename = os.path.join(outputs_folder, f'{job_id}_section_{section_index+1}.mp4')

                 # Trim black frame for txt2vid only on the very last save
                 final_pixels_to_save = history_pixels
                 if is_last_section and mode == 'txt2vid':
                      if history_pixels.shape[2] > 1: # Make sure there's more than just the black frame
                           print("Trimming first (black) frame for final txt2vid output.")
                           final_pixels_to_save = history_pixels[:, :, 1:] # Save from second frame onwards
                      else:
                           print("Warning: txt2vid output has only one frame, cannot trim black frame.")

                 print(f"Saving video section to {output_filename} with {final_pixels_to_save.shape[2]} frames.")
                 save_bcthw_as_mp4(final_pixels_to_save, output_filename, fps=30)
                 stream.output_queue.push((output_type_flag, output_filename))

            if not high_vram: unload_complete_models(vae)

            if is_last_section: print("Last section processed."); break

        # --- End of Loop ---
        print(f"Worker job {job_id} finished processing sections.")

    except Exception as e:
        print(f"!!!!!!!!!! Error in worker job {job_id} !!!!!!!!!!"); traceback.print_exc()
        stream.output_queue.push(('error', str(e)))
    finally:
        if not high_vram: unload_complete_models(*all_models); print("Models unloaded (low VRAM mode).")
        stream.output_queue.push(('end', None))
        print(f"Worker job {job_id} cleanup complete.")


# -------- Gradio UI --------
def process_fn(mode, img, vid, aspect_ratio_str,
               prompt, n_prompt, shift, cfg, gs, rs,
               strength, seed_from_ui, lock_seed_val,
               seconds, window, steps, gpu_mem, tea,
               progress=gr.Progress(track_tqdm=True)):


    # --- Seed Handling ---
    if not lock_seed_val:
        actual_seed = random.randint(0, 2**32 - 1) # Generate new seed
        print(f"Lock Seed unchecked. Generated new seed: {actual_seed}")
    else:
        actual_seed = int(seed_from_ui) # Use the value from the UI
        print(f"Lock Seed checked. Using seed from UI: {actual_seed}")

    # Reset UI elements and UPDATE SEED field immediately
    yield None, None, gr.update(visible=False, value=None), gr.update(value=''), gr.update(value=''), gr.update(interactive=False), gr.update(interactive=True), gr.update(value=actual_seed)

    # Clear previous stream queues if any
    stream.input_queue = AsyncStream().input_queue # Reset input queue
    stream.output_queue = AsyncStream().output_queue # Reset output queue

    # Launch worker thread - PASS THE DETERMINED actual_seed
    print("Starting process_fn...")
    async_run(worker, mode, img, vid, aspect_ratio_str, # Pass aspect ratio
              prompt, n_prompt, shift, cfg, gs, rs,
              strength, actual_seed, # Pass actual_seed here
              seconds, window, steps, gpu_mem, tea)

    # Handle outputs from worker thread
    output_video_path = None
    output_image_path = None # Added for image output
    last_preview = None
    while True:
        try:
            flag, data = stream.output_queue.next()

            # --- Output Handling ---
            if flag == 'video_file': # Video output
                output_video_path = data
                output_image_path = None # Clear image path
                # Yield updates: video=path, image=None, preview, status, bar, buttons, seed
                yield gr.update(value=output_video_path), gr.update(value=None, visible=False), gr.update(value=last_preview, visible=last_preview is not None), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update()
            elif flag == 'image_file': # Image output
                output_image_path = data
                output_video_path = None # Clear video path
                # Yield updates: video=None, image=path, preview, status, bar, buttons, seed
                yield gr.update(value=None, visible=False), gr.update(value=output_image_path, visible=True), gr.update(value=last_preview, visible=last_preview is not None), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update()
            elif flag == 'progress':
                preview_img, status_text, html_bar = data
                last_preview = preview_img
                # Yield keeps existing video/image paths, updates preview/status/bar/buttons/seed
                yield gr.update(), gr.update(), gr.update(value=preview_img, visible=preview_img is not None), status_text, html_bar, gr.update(interactive=False), gr.update(interactive=True), gr.update()
            elif flag == 'error':
                 error_message = data
                 print(f"Gradio UI received error: {error_message}")
                 # Yield keeps existing outputs, shows error, enables start button
                 yield gr.update(), gr.update(), gr.update(value=last_preview, visible=last_preview is not None), f"Error: {error_message}", '', gr.update(interactive=True), gr.update(interactive=False), gr.update()
                 break
            elif flag == 'end':
                print("Gradio UI received end signal.")
                # Final update: Keep final video/image, hide preview, clear status/bar, enable start button
                # Determine final visibility based on which output path is set
                final_vid_visible = output_video_path is not None
                final_img_visible = output_image_path is not None
                yield gr.update(value=output_video_path, visible=final_vid_visible), gr.update(value=output_image_path, visible=final_img_visible), gr.update(visible=False, value=None), '', '', gr.update(interactive=True), gr.update(interactive=False), gr.update()
                break
            else:
                 print(f"Received unexpected flag: {flag}")

        except Exception as e:
            # (Keep error handling as is, ensuring yield signature matches new outputs)
            if "FIFOQueue object" in str(e) and "'next' of" in str(e):
                print("Waiting for worker output...")
                yield gr.update(), gr.update(), gr.update(value=last_preview, visible=last_preview is not None), "Processing, please wait...", gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update()
                continue
            else:
                 print(f"Error processing output queue: {e}")
                 traceback.print_exc()
                 yield gr.update(), gr.update(), gr.update(value=last_preview, visible=last_preview is not None), f"UI Error: {e}", '', gr.update(interactive=True), gr.update(interactive=False), gr.update()
                 break

def end_process_early():
     print("Stop button clicked.")
     if stream and stream.input_queue:
        stream.input_queue.push('end')
     # Disable stop button, re-enable start button maybe? Depends on desired behavior
     return gr.update(interactive=False) # Disable stop button itself


# --- Gradio Interface Definition ---
css = make_progress_bar_css()
with gr.Blocks(css=css).queue() as demo:
    gr.Markdown("# FramePack – Advanced UI")
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                ["txt2vid", "img2vid", "vid2vid", "extend_vid", "txt2img", "img2img"], # ADDED MODES
                value="img2vid",
                label="Mode"
            )
            # Inputs controlled by mode
            input_image = gr.Image(sources='upload', type="numpy", label="Input Image (for img2vid/img2img)", visible=True, height=400)
            input_video = gr.Video(label="Input Video (for vid2vid/extend_vid)", sources='upload', visible=False)

            # Common Inputs
            prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Enter your prompt here...")
            n_prompt = gr.Textbox(label="Negative Prompt", lines=2, value="ugly, blurry, deformed, text, watermark, signature")

            # Mode-dependent inputs
            aspect_ratio = gr.Dropdown(list(ASPECT_RATIOS.keys()), value=DEFAULT_ASPECT_RATIO, label="Aspect Ratio (txt2img/img2img/txt2vid)", visible=False) # ADDED, initially hidden
            seconds = gr.Slider(1, 120, value=5, step=0.1, label="Output Video Length (sec)", visible=True) # Initially visible

            # Seed Row
            with gr.Row():
                seed = gr.Number(label="Seed", value=random.randint(0, 2**32 - 1), precision=0)
                lock_seed = gr.Checkbox(label="Lock Seed", value=False)

            with gr.Accordion("Advanced Settings", open=False):
                sampler = gr.Dropdown(
                    ["unipc", "unipc_bh2", "dpmpp_2m", "dpmpp_sde", "dpmpp_2m_sde", "dpmpp_3m_sde", "ddim", "plms", "euler", "euler_ancestral"],
                    value="unipc", label="Sampler",
                    visible=False # Hidden until other sampler support can be implemented in the library
                )
                # Strength slider - interactive based on mode (img2img, vid2vid)
                strength = gr.Slider(0., 1., value=0.7, label="Denoise Strength (img2img/vid2vid)", info="How much to change input image/video frame. 0.7 = default. Only used in img2img/vid2vid.", visible=True, interactive=False)

                # Video-specific settings - may hide for img modes
                shift = gr.Slider(0., 10., value=3.0, label="Shift μ (Temporal Consistency - Video)", info="Higher values might increase consistency but affect motion. Default 3.0.", visible=True)
                window = gr.Slider(1, 33, value=9, step=1, label="Latent Window Size (Video)", info="Affects temporal range per step. Default 9.", visible=True)

                # Common Advanced
                cfg = gr.Slider(1., 32., value=1.0, label="CFG Scale (Prompt Guidance)", info="Only effective if > 1.0. FramePack default is 1.0.")
                gs = gr.Slider(1., 32., value=10.0, label="Distilled CFG Scale", info="Main guidance scale for FramePack. Default 10.0.")
                rs = gr.Slider(0., 1., value=0., label="Rescale Guidance", info="Guidance rescale factor. Default 0.0.")
                steps = gr.Slider(1, 100, value=25, step=1, label="Sampling Steps")
                if not high_vram:
                    gpu_mem = gr.Slider(3, max(10, int(free_mem_gb)), value=min(6, int(free_mem_gb)-2), step=0.1, label="GPU Memory to Preserve (GB)")
                else:
                     gpu_mem = gr.Number(label="GPU Memory Preservation (N/A in High VRAM mode)", value=0, interactive=False)
                tea = gr.Checkbox(label="Use TeaCache Optimization", value=False)

            with gr.Row():
                 start_btn = gr.Button("Generate", variant="primary")
                 end_btn = gr.Button("Stop", interactive=False)

        with gr.Column(scale=1):
            # Use two components, show/hide based on mode
            result_vid = gr.Video(label="Output Video", interactive=False, height=512, autoplay=True, loop=True, visible=True) # Start visible
            result_img = gr.Image(label="Output Image", interactive=False, height=512, visible=False) # Start hidden
            preview = gr.Image(label="Live Preview", interactive=False, visible=False, height=256)
            status_md = gr.Markdown("")
            bar_html = gr.HTML("")

    # --- UI Control Logic ---
    def update_ui_for_mode(selected_mode):
        is_img_input_mode = selected_mode in ["img2vid", "img2img"]
        is_vid_input_mode = selected_mode in ["vid2vid", "extend_vid"]
        is_video_output_mode = selected_mode in ["txt2vid", "img2vid", "vid2vid", "extend_vid"]
        is_image_output_mode = selected_mode in ["txt2img", "img2img"]
        # AR relevant ONLY for txt2img, txt2vid (where size isn't from input)
        is_ar_relevant = selected_mode in ["txt2img", "txt2vid"]
        # Strength relevant ONLY for img2img, vid2vid (modes using init_latent + strength)
        is_strength_relevant = selected_mode in ["img2img", "vid2vid"]
        is_video_params_relevant = is_video_output_mode

        return {
            input_image: gr.update(visible=is_img_input_mode),
            input_video: gr.update(visible=is_vid_input_mode),
            aspect_ratio: gr.update(visible=is_ar_relevant),
            seconds: gr.update(visible=is_video_output_mode),
            result_vid: gr.update(visible=is_video_output_mode),
            result_img: gr.update(visible=is_image_output_mode),
            strength: gr.update(interactive=is_strength_relevant),
            shift: gr.update(visible=is_video_params_relevant),
            window: gr.update(visible=is_video_params_relevant),
        }

    mode.change(
        update_ui_for_mode,
        inputs=[mode],
        # Update all components controlled by mode
        outputs=[input_image, input_video, aspect_ratio, seconds, result_vid, result_img, strength, shift, window],
        queue=False
    )

    # --- Button Actions ---
    # Add aspect_ratio to inputs list
    inputs = [mode, input_image, input_video, aspect_ratio, # ADDED aspect_ratio
              prompt, n_prompt, shift, cfg, gs, rs,
              strength, seed, lock_seed, seconds, window, steps, gpu_mem, tea]

    # Add result_img to outputs list
    outputs = [result_vid, result_img, preview, status_md, bar_html, start_btn, end_btn, seed] # Added result_img

    start_btn.click(process_fn, inputs=inputs, outputs=outputs)
    end_btn.click(end_process_early, outputs=[end_btn])

# --- Launch App ---
print(f"Launching Gradio app on {args.host}:{args.port}")
demo.launch(
    server_name=args.host, # Use args.host
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser
)
