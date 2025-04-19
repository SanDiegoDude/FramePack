# -*- coding: utf-8 -*-
# Note: Added encoding hint for potentially wider compatibility

# --- Imports ---
import os
import argparse
import random
import math
import traceback
import numpy as np
from PIL import Image
import torch
import torchvision
import gradio as gr
import einops
import safetensors.torch as sf

# Diffusers & Transformers specific
from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer,
    SiglipImageProcessor, SiglipVisionModel
)

# Helper library specific
from diffusers_helper.hf_login import login # Assuming this is needed somewhere, though not called directly here
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import (
    save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw,
    resize_and_center_crop, generate_timestamp
    # state_dict_weighted_merge, state_dict_offset_merge # Not used currently
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    cpu, gpu, get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    fake_diffusers_current_device, DynamicSwapInstaller,
    unload_complete_models, load_model_as_complete
)
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# --- Constants and Argument Parsing ---
ASPECT_RATIOS = {
    "1:1": (768, 768),
    "16:9": (1024, 576),
    "9:16": (576, 1024),
    "4:3": (896, 672),
    "3:4": (672, 896),
    "21:9": (1280, 576), # Approx 2.22:1
    "9:21": (576, 1280),
}
DEFAULT_ASPECT_RATIO = "16:9"

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument('--hf-cache', choices=['local', 'global'], default='local')
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=7800)
parser.add_argument('--inbrowser', action='store_true')
args = parser.parse_args()

# --- Environment Setup ---
if args.hf_cache == 'local':
    hf_home_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './hf_download'))
    if not os.path.exists(hf_home_dir):
        os.makedirs(hf_home_dir)
    os.environ['HF_HOME'] = hf_home_dir

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60
print(f'Free VRAM {free_mem_gb:.2f} GB')
print(f'Using CUDA device: {gpu}')
print(f'High-VRAM Mode: {high_vram}')

# --- Load Models ---
print("-" * 20)
print("Loading models to CPU...")
hf_hub_kwargs = {"resume_download": True}

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
    print("Low VRAM mode: Enabling VAE slicing/tiling and dynamic swapping.")
    vae.enable_slicing()
    vae.enable_tiling()
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    print("High VRAM mode: Moving models to GPU.")
    for m in all_models:
        m.to(gpu)
    print("Models moved to GPU.")

transformer.high_quality_fp32_output_for_inference = True
transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16) # Keep VAE float16 default, handle encode exceptions
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)
print("Model dtypes configured.")

# --- Global Variables ---
stream = AsyncStream()
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

# -------- Worker (handles all modes) --------
@torch.no_grad()
def worker(mode, input_image, input_video, aspect_ratio_str,
           prompt, n_prompt,
           shift, cfg, gs, rs,
           strength, seed, total_second_length, latent_window_size,
           steps, gpu_memory_preservation, use_teacache):

    job_id = generate_timestamp()
    is_image_mode = mode in ['txt2img', 'img2img']
    output_type_flag = 'image_file' if is_image_mode else 'video_file'

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
        init_latent = None # Passed to sampler if not None (txt2vid-like, img2img, vid2vid)
        concat_latent = None # Video conditioning (extend_vid, vid2vid)
        image_embeddings = None # CLIP conditioning (required by transformer)
        first_frame_conditioning_latent = None # VAE latent used for clean_latents guide (all modes)
        black_frame_latent = None
        black_frame_embeddings = None

        # --- Load VAE and Image Encoder (conditionally) ---
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE ready...'))))
            load_model_as_complete(image_encoder, target_device=gpu) # Needed by all modes now
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image Encoder ready...'))))

        # --- Determine Target Size ---
        if mode in ['txt2img', 'img2img', 'txt2vid']:
            if aspect_ratio_str in ASPECT_RATIOS: width, height = ASPECT_RATIOS[aspect_ratio_str]
            else: width, height = ASPECT_RATIOS[DEFAULT_ASPECT_RATIO]; print(f"Warning: Invalid aspect ratio...")
            print(f"Mode {mode}: Using aspect ratio {aspect_ratio_str} -> {width}x{height}")

        # --- Generate Black Frame Latent & Embeddings (for text input modes) ---
        if mode in ['txt2img', 'txt2vid']:
            print(f"Mode {mode}: Generating black frame latent & embeddings for {width}x{height}...")
            black_np = np.zeros((height, width, 3), dtype=np.uint8)
            original_vae_dtype = vae.dtype
            try:
                vae.to(dtype=torch.float32)
                inp_black = (torch.from_numpy(black_np).float() / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device=gpu)
                black_frame_latent = vae_encode(inp_black, vae).to(transformer.dtype)
            finally:
                vae.to(dtype=original_vae_dtype)
            print(f"  Black frame VAE latent shape: {black_frame_latent.shape}")
            if not high_vram: load_model_as_complete(image_encoder, target_device=gpu)
            black_frame_embeddings_out = hf_clip_vision_encode(black_np, feature_extractor, image_encoder)
            black_frame_embeddings = black_frame_embeddings_out.last_hidden_state.to(transformer.dtype)
            print(f"  Black frame CLIP embeddings shape: {black_frame_embeddings.shape}")

        # --- Specific Mode Setup Logic ---
        if mode == 'txt2img':
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Preparing for txt2img...'))))
            init_latent = None; concat_latent = None
            first_frame_conditioning_latent = black_frame_latent
            image_embeddings = black_frame_embeddings
            print(f"Txt2Img: Setup complete.")

        elif mode == 'img2img':
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Preparing for img2img...'))))
            if input_image is None: raise ValueError("Input image required.")
            np_img = np.array(input_image)
            img_resized = resize_and_center_crop(np_img, target_width=width, target_height=height) # Uses target AR size
            print(f"Img2Img: Resized image to {width}x{height}")
            original_vae_dtype = vae.dtype
            try:
                vae.to(dtype=torch.float32)
                inp_for_vae = (torch.from_numpy(img_resized).float() / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device=gpu)
                init_latent = vae_encode(inp_for_vae, vae).to(transformer.dtype) # Used for start noise
                first_frame_conditioning_latent = init_latent # Used for clean_latents guide
            finally:
                vae.to(dtype=original_vae_dtype)
            print(f"Img2Img: Initial latent shape {init_latent.shape}")
            if not high_vram: load_model_as_complete(image_encoder, target_device=gpu)
            image_encoder_output = hf_clip_vision_encode(img_resized, feature_extractor, image_encoder)
            image_embeddings = image_encoder_output.last_hidden_state.to(transformer.dtype) # Real embeddings
            print(f"Img2Img: Image embedding shape {image_embeddings.shape}")
            concat_latent = None

        elif mode == 'txt2vid':
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Preparing for txt2vid...'))))
            init_latent = None; concat_latent = None
            first_frame_conditioning_latent = black_frame_latent
            image_embeddings = black_frame_embeddings
            print(f"Txt2Vid: Setup complete.")

        elif mode == 'img2vid':
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Preparing for img2vid...'))))
            if input_image is None: raise ValueError("Input image required.")
            np_img = np.array(input_image)
            H, W, _ = np_img.shape; height, width = find_nearest_bucket(H, W, resolution=640) # Size from input
            img_resized = resize_and_center_crop(np_img, target_width=width, target_height=height)
            print(f"Img2Vid: Resized/bucketed image to {width}x{height}")
            original_vae_dtype = vae.dtype
            try:
                vae.to(dtype=torch.float32)
                inp_for_vae = (torch.from_numpy(img_resized).float() / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device=gpu)
                first_frame_conditioning_latent = vae_encode(inp_for_vae, vae).to(transformer.dtype) # Used for clean_latents guide
            finally:
                vae.to(dtype=original_vae_dtype)
            print(f"Img2Vid: First frame latent for conditioning: {first_frame_conditioning_latent.shape}")
            if not high_vram: load_model_as_complete(image_encoder, target_device=gpu)
            image_encoder_output = hf_clip_vision_encode(img_resized, feature_extractor, image_encoder)
            image_embeddings = image_encoder_output.last_hidden_state.to(transformer.dtype) # Real embeddings
            print(f"Img2Vid: Image embedding shape {image_embeddings.shape}")
            init_latent = None; concat_latent = None # Conditioning only mode

        elif mode == 'extend_vid':
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Preparing for {mode}...'))))
            if input_video is None or not hasattr(input_video, 'name'): raise ValueError("Input video file required.")
            video_path = input_video.name; print(f"Video Mode: Loading video {video_path}")
            vid_frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
            vid = vid_frames.permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0
            _, _, T, H, W = vid.shape; height, width = H, W # Size from input
            print(f"Video Mode: Loaded video with {T} frames, {W}x{H}")
            original_vae_dtype = vae.dtype
            try:
                vae.to(dtype=torch.float32)
                first_frame_np = vid_frames[0].numpy()
                inp_first_frame = (torch.from_numpy(first_frame_np).float() / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device=gpu)
                first_frame_conditioning_latent = vae_encode(inp_first_frame, vae).to(transformer.dtype) # Used for clean_latents guide
                print(f"ExtendVid: First frame latent for conditioning: {first_frame_conditioning_latent.shape}")
                print(f"ExtendVid: Encoding full video for concat_latent...")
                concat_latent = vae_encode(vid.to(device=gpu), vae).to(transformer.dtype) # Used for concat conditioning
                print(f"ExtendVid: Conditioning concat_latent shape {concat_latent.shape}")
            finally:
                vae.to(dtype=original_vae_dtype)
            print(f"ExtendVid: Calculating CLIP embeddings from first frame...")
            if not high_vram: load_model_as_complete(image_encoder, target_device=gpu)
            image_encoder_output = hf_clip_vision_encode(first_frame_np, feature_extractor, image_encoder)
            image_embeddings = image_encoder_output.last_hidden_state.to(transformer.dtype) # Real embeddings from first frame
            print(f"ExtendVid: Image embedding shape {image_embeddings.shape}")
            init_latent = None # Conditioning only mode

        elif mode == 'vid2vid':
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Preparing for {mode}...'))))
            if input_video is None or not hasattr(input_video, 'name'): raise ValueError("Input video file required.")
            video_path = input_video.name; print(f"Video Mode: Loading video {video_path}")
            vid_frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
            vid = vid_frames.permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0
            _, _, T, H, W = vid.shape; height, width = H, W # Size from input
            print(f"Video Mode: Loaded video with {T} frames, {W}x{H}")
            original_vae_dtype = vae.dtype
            try:
                vae.to(dtype=torch.float32)
                first_frame_np = vid_frames[0].numpy()
                inp_first_frame = (torch.from_numpy(first_frame_np).float() / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device=gpu)
                init_latent = vae_encode(inp_first_frame, vae).to(transformer.dtype) # Used for start noise
                first_frame_conditioning_latent = init_latent # Used for clean_latents guide
                print(f"Vid2Vid: Initial latent for sampler: {init_latent.shape}")
                print(f"Vid2Vid: Encoding full video for concat_latent...")
                concat_latent = vae_encode(vid.to(device=gpu), vae).to(transformer.dtype) # Used for concat conditioning
                print(f"Vid2Vid: Conditioning concat_latent shape {concat_latent.shape}")
            finally:
                vae.to(dtype=original_vae_dtype)
            print(f"Vid2Vid: Calculating CLIP embeddings from first frame...")
            if not high_vram: load_model_as_complete(image_encoder, target_device=gpu)
            image_encoder_output = hf_clip_vision_encode(first_frame_np, feature_extractor, image_encoder)
            image_embeddings = image_encoder_output.last_hidden_state.to(transformer.dtype) # Real embeddings from first frame
            print(f"Vid2Vid: Image embedding shape {image_embeddings.shape}")

        # --- Fallback sizing ---
        if height is None or width is None:
            height, width = 512, 512
            print(f"Warning: Using fallback size {width}x{height}")

        # --- Sampling loop Setup ---
        num_frames_per_window = latent_window_size * 4 - 3 # Use UI window size for internal processing
        if is_image_mode:
            total_latent_sections = 1
            latent_paddings = [0] # Force single run
            print("Image Mode: Forcing single generation section.")
            print(f"Image Mode: Using num_frames_per_window={num_frames_per_window} based on latent_window_size={latent_window_size}")
        else: # Video modes
            total_latent_sections = max(int(round((total_second_length * 30) / (latent_window_size * 4))), 1)
            if total_latent_sections > 4: latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            else: latent_paddings = list(reversed(range(total_latent_sections)))
        stream.output_queue.push(('progress', (None, f'Preparing for {total_latent_sections} sections...', make_progress_bar_html(0, 'Initializing Sampler...'))))
        rnd = torch.Generator("cpu").manual_seed(int(seed))
        history_latents = torch.zeros((1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0
        print(f"Latent padding sequence: {latent_paddings}")

        # --- Main Generation Loop ---
        for section_index, latent_padding in enumerate(latent_paddings):
            is_last_section = (latent_padding == 0)
            latent_padding_size = latent_padding * latent_window_size

            print(f"\n--- Section {section_index + 1}/{total_latent_sections} (Padding: {latent_padding}, Last: {is_last_section}) ---")

            if stream.input_queue.top() == 'end':
                print("User requested stop.")
                stream.output_queue.push(('end', None))
                return

            # Prepare clean latents for conditioning
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, _, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = \
                indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            # Determine the 'clean' reference latent for the *conditioning* input's 'pre' part.
            # Reflects the state the *next* generated frame should transition FROM (due to backward generation).
            if section_index == 0:
                # First section run (generates LATEST chunk in time). Reference is the input image/video start frame (or black frame).
                if mode in ['img2vid', 'extend_vid', 'vid2vid', 'img2img']: # Modes with a real or derived first frame
                    if first_frame_conditioning_latent is not None:
                         clean_latents_pre_cond = first_frame_conditioning_latent.to(history_latents.device, dtype=history_latents.dtype)
                         print(f"Using input first frame VAE latent for clean_latents_pre conditioning (Section {section_index+1} - Final Time Chunk).")
                    else: # Should not happen if setup is correct
                         print(f"Warning: Expected first_frame_conditioning_latent for {mode} but not found, using history.")
                         clean_latents_pre_cond = history_latents[:, :, :1]
                else: # txt2img, txt2vid (should use black frame latent stored in first_frame_conditioning_latent)
                     if first_frame_conditioning_latent is not None:
                          clean_latents_pre_cond = first_frame_conditioning_latent.to(history_latents.device, dtype=history_latents.dtype)
                          print(f"Using black frame VAE latent for clean_latents_pre conditioning (Section {section_index+1} - Final Time Chunk).")
                     else: # Fallback if black frame failed
                          print(f"Warning: Expected black frame latent for {mode} but not found, using history.")
                          clean_latents_pre_cond = history_latents[:, :, :1]
            else:
                # Subsequent sections (generate EARLIER chunks). Reference is the START frame of the PREVIOUSLY generated (later) chunk.
                print(f"Using history[:1] (start of prev. chunk) for clean_latents_pre conditioning (Section {section_index+1}).")
                clean_latents_pre_cond = history_latents[:, :, :1]

            # Get the 'post', '2x', and '4x' parts from history buffer
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16].split([1, 2, 16], dim=2)
            clean_latents_for_cond = torch.cat([clean_latents_pre_cond, clean_latents_post], dim=2)
            print(f"Shape of clean_latents_for_cond: {clean_latents_for_cond.shape}")

            # --- Load Transformer (conditionally) ---
            if not high_vram:
                unload_complete_models(vae, image_encoder, text_encoder, text_encoder_2)
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Transformer ready...'))))

            # --- Configure TeaCache ---
            transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=steps)

            # --- Progress Callback ---
            def callback(d):
                if stream.input_queue.top() == 'end':
                    print("Stop signal received in callback, allowing current step/section to finish.")
                    return # Exit the callback early
                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling Section {section_index + 1}/{total_latent_sections}, Step {current_step}/{steps}'
                desc = f'Total generated frames: ~{int(max(0, total_generated_latent_frames * 4 - 3))}/{int(total_second_length * 30)}. Running...'
                preview_img_data = None
                try:
                    preview = d.get('denoised', None)
                    if preview is not None:
                        preview_vae = vae_decode_fake(preview)
                        preview_np = (preview_vae * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                        preview_img_data = einops.rearrange(preview_np, 'b c t h w -> (b h) (t w) c')
                except Exception as e:
                    print(f"Error during preview generation: {e}")
                stream.output_queue.push(('progress', (preview_img_data, desc, make_progress_bar_html(percentage, hint))))

            # --- Run Sampling ---
            print(f"Calling sample_hunyuan with: mode={mode}, shift={shift}, strength={strength if mode in ['img2img', 'vid2vid'] else 'N/A'}, cfg={cfg}, gs={gs}, rs={rs}")
            print(f"  initial_latent provided: {'Yes' if init_latent is not None else 'No'}")
            print(f"  concat_latent provided: {'Yes' if concat_latent is not None else 'No'}")
            print(f"  image_embeddings provided: {'Yes' if image_embeddings is not None else 'No'} (Shape: {image_embeddings.shape if image_embeddings is not None else 'N/A'})")

            generated_latents = sample_hunyuan(
                transformer=transformer,
                # sampler='unipc', # Hardcoded in function
                initial_latent=init_latent,
                concat_latent=concat_latent,
                strength=strength,
                width=width, height=height,
                frames=num_frames_per_window, # Use standard window size
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
                image_embeddings=image_embeddings, # Pass real or black frame embeddings
                latent_indices=latent_indices,
                clean_latents=clean_latents_for_cond.to(gpu, dtype=transformer.dtype),
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x.to(gpu, dtype=transformer.dtype),
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x.to(gpu, dtype=transformer.dtype),
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback
            )
            print(f"Section {section_index + 1} sampling complete. Output latent shape: {generated_latents.shape}")

            # --- Post-processing and History Update ---
            if is_last_section:
                if mode in ['img2vid', 'extend_vid'] or (mode in ['txt2img', 'txt2vid'] and black_frame_latent is not None):
                    cond_latent_to_prepend = first_frame_conditioning_latent
                    if cond_latent_to_prepend is not None:
                        print(f"Prepending conditioning latent for {mode} final output.")
                        generated_latents = torch.cat([cond_latent_to_prepend.to(generated_latents.device, dtype=generated_latents.dtype), generated_latents], dim=2)
                    else: print(f"Warning: Cannot prepend conditioning latent for {mode}, as it's missing.")
                elif init_latent is not None and mode == 'vid2vid':
                    print(f"Prepending initial latent to the final output (vid2vid).")
                    generated_latents = torch.cat([init_latent.to(generated_latents.device, dtype=generated_latents.dtype), generated_latents], dim=2)

            current_section_latent_frames = generated_latents.shape[2]
            total_generated_latent_frames += current_section_latent_frames
            print(f"Total generated latent frames so far: {total_generated_latent_frames}")
            history_latents = torch.cat([generated_latents.cpu().to(history_latents.dtype), history_latents], dim=2)

            # --- VAE Decode Section ---
            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(100, 'VAE Decoding...'))))

            real_history_latents_gpu = history_latents[:, :, :total_generated_latent_frames].to(gpu, dtype=vae.dtype)
            print(f"Decoding latents shape: {real_history_latents_gpu.shape}")
            decoded_pixels = vae_decode(real_history_latents_gpu, vae).cpu() # B, C, T, H, W

            if is_image_mode:
                # Extract desired frame (index 0 for img2img, index 1 for txt2img after prepend)
                frame_index_to_save = 1 if mode == 'txt2img' and decoded_pixels.shape[2] > 1 else 0
                print(f"Image Mode ({mode}): Extracting frame index {frame_index_to_save} from decoded shape {decoded_pixels.shape}.")

                if decoded_pixels.shape[2] > frame_index_to_save:
                     output_image_pixels = decoded_pixels[:, :, frame_index_to_save] # B, C, H, W
                     output_image_np = ((output_image_pixels[0].permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8) # H, W, C
                     output_filename = os.path.join(outputs_folder, f'{job_id}_final.png')
                     print(f"Saving final image to {output_filename}")
                     Image.fromarray(output_image_np).save(output_filename)
                     stream.output_queue.push((output_type_flag, output_filename))
                else:
                     print(f"Error: Could not extract frame index {frame_index_to_save} from decoded pixels with shape {decoded_pixels.shape}")
                     stream.output_queue.push(('error', f"Failed to extract output frame for {mode}."))
                history_pixels = None # No history needed for img mode
            else: # Video Mode VAE/Append logic
                if history_pixels is None:
                    history_pixels = decoded_pixels # First chunk
                else:
                    overlap_pixel_frames = latent_window_size * 4 - 3
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    # Need slice from history for append context
                    slice_to_decode = real_history_latents_gpu[:, :, :min(section_latent_frames, real_history_latents_gpu.shape[2])]
                    current_pixels_for_append = vae_decode(slice_to_decode, vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels_for_append, history_pixels, overlap=overlap_pixel_frames)

                print(f"Current video history_pixels shape: {history_pixels.shape}")
                output_filename = os.path.join(outputs_folder, f'{job_id}_section_{section_index+1}.mp4')
                final_pixels_to_save = history_pixels
                if is_last_section and mode == 'txt2vid':
                     if history_pixels.shape[2] > 1:
                          print("Trimming first (black) frame for final txt2vid output.")
                          final_pixels_to_save = history_pixels[:, :, 1:]
                     else: print("Warning: txt2vid output has only one frame, cannot trim black frame.")
                print(f"Saving video section to {output_filename} with {final_pixels_to_save.shape[2]} frames.")
                save_bcthw_as_mp4(final_pixels_to_save, output_filename, fps=30)
                stream.output_queue.push((output_type_flag, output_filename))

            if not high_vram:
                unload_complete_models(vae)

            if is_last_section:
                print("Last section processed.")
                break
        # --- End of Loop ---
        print(f"Worker job {job_id} finished processing sections.")

    except Exception as e:
        print(f"!!!!!!!!!! Error in worker job {job_id} !!!!!!!!!!")
        traceback.print_exc()
        stream.output_queue.push(('error', str(e)))
    finally:
        if not high_vram:
            unload_complete_models(*all_models)
            print("Models unloaded (low VRAM mode).")
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
        actual_seed = random.randint(0, 2**32 - 1)
        print(f"Lock Seed unchecked. Generated new seed: {actual_seed}")
    else:
        actual_seed = int(seed_from_ui)
        print(f"Lock Seed checked. Using seed from UI: {actual_seed}")

    # Reset UI elements and UPDATE SEED field immediately
    yield (
        gr.update(value=None, visible=False), # result_vid
        gr.update(value=None, visible=False), # result_img
        gr.update(visible=False, value=None), # preview
        gr.update(value=''), # status_md
        gr.update(value=''), # bar_html
        gr.update(interactive=False), # start_btn
        gr.update(interactive=True), # end_btn
        gr.update(value=actual_seed) # seed
    )

    # --- Clear queues and run worker ---
    stream.input_queue = AsyncStream().input_queue
    stream.output_queue = AsyncStream().output_queue
    print("Starting process_fn...")
    async_run(worker, mode, img, vid, aspect_ratio_str,
              prompt, n_prompt, shift, cfg, gs, rs,
              strength, actual_seed, seconds, window, steps, gpu_mem, tea)

    # --- Handle outputs from worker ---
    output_video_path = None
    output_image_path = None
    last_preview = None
    while True:
        try:
            flag, data = stream.output_queue.next()

            if flag == 'video_file':
                output_video_path = data
                output_image_path = None
                yield (
                    gr.update(value=output_video_path), # result_vid
                    gr.update(value=None, visible=False), # result_img
                    gr.update(value=last_preview, visible=last_preview is not None), # preview
                    gr.update(), # status_md
                    gr.update(), # bar_html
                    gr.update(interactive=False), # start_btn
                    gr.update(interactive=True), # end_btn
                    gr.update() # seed
                )
            elif flag == 'image_file':
                output_image_path = data
                output_video_path = None
                yield (
                    gr.update(value=None, visible=False), # result_vid
                    gr.update(value=output_image_path, visible=True), # result_img
                    gr.update(value=last_preview, visible=last_preview is not None), # preview
                    gr.update(), # status_md
                    gr.update(), # bar_html
                    gr.update(interactive=False), # start_btn
                    gr.update(interactive=True), # end_btn
                    gr.update() # seed
                )
            elif flag == 'progress':
                preview_img, status_text, html_bar = data
                last_preview = preview_img
                yield (
                    gr.update(), # result_vid
                    gr.update(), # result_img
                    gr.update(value=preview_img, visible=preview_img is not None), # preview
                    status_text, # status_md
                    html_bar, # bar_html
                    gr.update(interactive=False), # start_btn
                    gr.update(interactive=True), # end_btn
                    gr.update() # seed
                )
            elif flag == 'error':
                 error_message = data
                 print(f"Gradio UI received error: {error_message}")
                 yield (
                     gr.update(), # result_vid
                     gr.update(), # result_img
                     gr.update(value=last_preview, visible=last_preview is not None), # preview
                     f"Error: {error_message}", # status_md
                     '', # bar_html
                     gr.update(interactive=True), # start_btn
                     gr.update(interactive=False), # end_btn
                     gr.update() # seed
                 )
                 break
            elif flag == 'end':
                print("Gradio UI received end signal.")
                final_vid_visible = output_video_path is not None
                final_img_visible = output_image_path is not None
                yield (
                    gr.update(value=output_video_path, visible=final_vid_visible), # result_vid
                    gr.update(value=output_image_path, visible=final_img_visible), # result_img
                    gr.update(visible=False, value=None), # preview
                    '', # status_md
                    '', # bar_html
                    gr.update(interactive=True), # start_btn
                    gr.update(interactive=False), # end_btn
                    gr.update() # seed
                )
                break
            else:
                 print(f"Received unexpected flag: {flag}")

        except Exception as e:
            if "FIFOQueue object" in str(e) and "'next' of" in str(e): # Should not happen now
                print("Waiting for worker output...")
                yield (
                    gr.update(), gr.update(), gr.update(value=last_preview, visible=last_preview is not None),
                    "Processing, please wait...", gr.update(),
                    gr.update(interactive=False), gr.update(interactive=True), gr.update()
                )
                continue
            else:
                 print(f"Error processing output queue: {e}")
                 traceback.print_exc()
                 yield (
                     gr.update(), gr.update(), gr.update(value=last_preview, visible=last_preview is not None),
                     f"UI Error: {e}", '',
                     gr.update(interactive=True), gr.update(interactive=False), gr.update()
                 )
                 break

def end_process_early():
     print("Stop button clicked.")
     if stream and stream.input_queue:
        stream.input_queue.push('end')
     return gr.update(interactive=False)

# --- Gradio Interface Definition ---
css = make_progress_bar_css()
with gr.Blocks(css=css).queue() as demo:
    gr.Markdown("# FramePack – Advanced UI")
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                ["txt2vid", "img2vid", "vid2vid", "extend_vid", "txt2img", "img2img"],
                value="img2vid",
                label="Mode"
            )
            input_image = gr.Image(sources='upload', type="numpy", label="Input Image (for img2vid/img2img)", visible=True, height=400)
            input_video = gr.Video(label="Input Video (for vid2vid/extend_vid)", sources='upload', visible=False)
            prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Enter your prompt here...")
            n_prompt = gr.Textbox(label="Negative Prompt", lines=2, value="ugly, blurry, deformed, text, watermark, signature")
            aspect_ratio = gr.Dropdown(list(ASPECT_RATIOS.keys()), value=DEFAULT_ASPECT_RATIO, label="Aspect Ratio (txt2img/txt2vid)", visible=False) # Visibility controlled
            seconds = gr.Slider(1, 120, value=5, step=0.1, label="Output Video Length (sec)", visible=True) # Visibility controlled
            with gr.Row():
                seed = gr.Number(label="Seed", value=random.randint(0, 2**32 - 1), precision=0)
                lock_seed = gr.Checkbox(label="Lock Seed", value=False)

            with gr.Accordion("Advanced Settings", open=False):
                # Sampler hidden for now
                # sampler = gr.Dropdown(...)
                strength = gr.Slider(0., 1., value=0.7, label="Denoise Strength (img2img/vid2vid)", info="How much to change input image/video frame. 0.7 = default. Only used in img2img/vid2vid.", visible=True, interactive=False) # Interactivity controlled
                shift = gr.Slider(0., 10., value=3.0, label="Shift μ (Temporal Consistency - Video)", info="Higher values might increase consistency but affect motion. Default 3.0.", visible=True) # Visibility controlled
                window = gr.Slider(1, 33, value=9, step=1, label="Latent Window Size", info="Affects temporal range per step. Default 9.", visible=True) # Visibility controlled
                cfg = gr.Slider(1., 32., value=1.0, label="CFG Scale (Prompt Guidance)")
                gs = gr.Slider(1., 32., value=10.0, label="Distilled CFG Scale")
                rs = gr.Slider(0., 1., value=0., label="Rescale Guidance")
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
            result_vid = gr.Video(label="Output Video", interactive=False, height=512, autoplay=True, loop=True, visible=True)
            result_img = gr.Image(label="Output Image", interactive=False, height=512, visible=False)
            preview = gr.Image(label="Live Preview", interactive=False, visible=False, height=256)
            status_md = gr.Markdown("")
            bar_html = gr.HTML("")

    # --- UI Control Logic ---
    def update_ui_for_mode(selected_mode):
        is_img_input_mode = selected_mode in ["img2vid", "img2img"]
        is_vid_input_mode = selected_mode in ["vid2vid", "extend_vid"]
        is_video_output_mode = selected_mode in ["txt2vid", "img2vid", "vid2vid", "extend_vid"]
        is_image_output_mode = selected_mode in ["txt2img", "img2img"]
        is_ar_relevant = selected_mode in ["txt2img", "txt2vid"]
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
        outputs=[input_image, input_video, aspect_ratio, seconds, result_vid, result_img, strength, shift, window],
        queue=False
    )

    # --- Button Actions ---
    inputs = [mode, input_image, input_video, aspect_ratio,
              prompt, n_prompt, shift, cfg, gs, rs,
              strength, seed, lock_seed, seconds, window, steps, gpu_mem, tea]
    outputs = [result_vid, result_img, preview, status_md, bar_html, start_btn, end_btn, seed]
    start_btn.click(process_fn, inputs=inputs, outputs=outputs)
    end_btn.click(end_process_early, outputs=[end_btn])

# --- Launch App ---
print(f"Launching Gradio app on {args.host}:{args.port}")
demo.launch(
    server_name=args.host,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser
)
