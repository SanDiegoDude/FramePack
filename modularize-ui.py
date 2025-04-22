from diffusers_helper.hf_login import login
import os
import time
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
import gradio as gr
import torch
import traceback
import einops
import uuid
import safetensors.torch as sf
import numpy as np
import argparse
import math
from PIL import Image
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
import random

import subprocess

def make_mp4_faststart(mp4_path):
    tmpfile = mp4_path + ".tmp"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", mp4_path,
        "-c", "copy",
        "-movflags", "+faststart",
        tmpfile
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.replace(tmpfile, mp4_path)
        debug(f"[FFMPEG] Moved moov atom to front of {mp4_path} (faststart applied)")
    except Exception as e:
        debug(f"[FFMPEG] Faststart failed for {mp4_path}: {e}")
        if os.path.exists(tmpfile):
            os.remove(tmpfile)

def get_valid_frame_stops(latent_window_size, max_seconds=120, fps=30):
    frames_per_section = latent_window_size * 4 - 3
    max_sections = int((max_seconds * fps) // frames_per_section)
    stops = [frames_per_section * i for i in range(1, max_sections + 1)]
    return stops

DEBUG = True
def debug(*a, **k):
    if DEBUG:
        print("[DEBUG]", *a, **k)

# ---- CLI args ----
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()
print(args)

# ---- VRAM Check ----
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60
print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# ---- Model Load ----
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
for m in [vae, text_encoder, text_encoder_2, image_encoder, transformer]:
    m.eval()
if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()
transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')
for m, dtype in zip([transformer, vae, image_encoder, text_encoder, text_encoder_2], [torch.bfloat16, torch.float16, torch.float16, torch.float16, torch.float16]):
    m.to(dtype=dtype)
    m.requires_grad_(False)
if not high_vram:
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    for m in [text_encoder, text_encoder_2, image_encoder, vae, transformer]:
        m.to(gpu)
stream = AsyncStream()
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

def extract_frames_from_video(video_path, num_frames=8, from_end=True):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        from_end: If True, extract from the end of the video, otherwise from the beginning
    
    Returns:
        numpy array of frames and the video fps
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for video extension. Please install it with 'pip install opencv-python'")
        
    debug(f"Extracting frames from video: {video_path}, num_frames={num_frames}, from_end={from_end}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    debug(f"Video properties: total_frames={total_frames}, fps={fps}, dimensions={width}x{height}")
    
    # Calculate frame indices to extract
    if from_end:
        start_idx = max(0, total_frames - num_frames)
        frame_indices = list(range(start_idx, total_frames))
    else:
        frame_indices = list(range(min(num_frames, total_frames)))
    
    # Extract the frames
    frames = []
    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame in frame_indices:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        current_frame += 1
        
        # Break if we've extracted all needed frames
        if len(frames) == len(frame_indices):
            break
    
    cap.release()
    
    if not frames:
        raise ValueError(f"Failed to extract frames from video: {video_path}")
    
    # Convert to numpy array
    frames = np.array(frames)
    debug(f"Extracted {len(frames)} frames with shape {frames.shape}")
    
    return frames, fps

# ---- Color Helper ----
def parse_hex_color(hexcode):
    hexcode = hexcode.lstrip('#')
    if len(hexcode) == 6:
        r = int(hexcode[0:2], 16) / 255
        g = int(hexcode[2:4], 16) / 255
        b = int(hexcode[4:6], 16) / 255
        return r, g, b
    # fallback to gray
    return 0.5, 0.5, 0.5

# ---- Worker Utility Split ----
def prepare_inputs(input_image, prompt, n_prompt, cfg):
    if input_image is None:
        raise ValueError(
            "No input image provided! For text2video, a blank will be created in worker -- "
            "but for image2video, you must upload an image."
        )
    if hasattr(input_image, 'shape'):
        H, W, C = input_image.shape
    else:
        raise ValueError("Input image is not a valid numpy array!")
    if not high_vram:
        unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
    fake_diffusers_current_device(text_encoder, gpu)
    load_model_as_complete(text_encoder_2, target_device=gpu)
    llama_vec, clip_pool = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    if cfg == 1:
        llama_vec_n, clip_pool_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_pool)
    else:
        llama_vec_n, clip_pool_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    llama_vec, mask = crop_or_pad_yield_mask(llama_vec, 512)
    llama_vec_n, mask_n = crop_or_pad_yield_mask(llama_vec_n, 512)
    h, w = find_nearest_bucket(H, W, resolution=640)
    input_np = resize_and_center_crop(input_image, target_width=w, target_height=h)
    Image.fromarray(input_np).save(os.path.join(outputs_folder, f'{generate_timestamp()}.png'))
    input_tensor = torch.from_numpy(input_np).float() / 127.5 - 1
    input_tensor = input_tensor.permute(2, 0, 1)[None, :, None]
    return input_np, input_tensor, llama_vec, clip_pool, llama_vec_n, clip_pool_n, mask, mask_n, h, w

def get_dims_from_aspect(aspect, custom_w, custom_h):
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
    import math
    max_pixels = 1024 * 1024
    px = width * height
    if px > max_pixels:
        scale = math.sqrt(max_pixels / px)
        width = int(width * scale)
        height = int(height * scale)
    width = (width // 8) * 8
    height = (height // 8) * 8
    return width, height
    

# ---- WORKER ----
@torch.no_grad()
def worker(
    mode, input_image, start_frame, end_frame, aspect, custom_w, custom_h,
    prompt, n_prompt, seed,
    use_adv, adv_window, adv_seconds, selected_frames,
    steps, cfg, gs, rs, gpu_memory_preservation, use_teacache,
    init_color, keyframe_weight
):
    job_id = generate_timestamp()
    debug("worker(): started", mode, "job_id:", job_id)

    # -- section/frames logic
    if use_adv:
        latent_window_size = adv_window
        frames_per_section = latent_window_size * 4 - 3
        total_frames = int(round(adv_seconds * 30))
        total_sections = math.ceil(total_frames / frames_per_section)
        debug(f"worker: Advanced mode | latent_window_size={latent_window_size} "
              f"| frames_per_section={frames_per_section} | total_frames={total_frames} | total_sections={total_sections}")
    else:
        latent_window_size = 9
        frames_per_section = latent_window_size * 4 - 3
        total_frames = int(selected_frames)
        total_sections = total_frames // frames_per_section
        debug(f"worker: Simple mode | latent_window_size=9 | frames_per_section=33 | "
              f"total_frames={total_frames} | total_sections={total_sections}")

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))
    debug("worker: pushed progress event 'Starting ...'")

    try:
        t_start = time.time()

        # ----------- Mode and input setup & prompts -------------
        if mode == "keyframes":
            if end_frame is None:
                raise ValueError("Keyframes mode requires End Frame to be set!")
            
            # Get original dimensions
            end_H, end_W, end_C = end_frame.shape
            
            # Find nearest bucket size for the end frame
            height, width = find_nearest_bucket(end_H, end_W, resolution=640)
            debug(f"Using bucket dimensions for keyframe mode: {width}x{height} (from original {end_W}x{end_H})")
            
            # --- Build start-frame numpy image ---
            if start_frame is not None:
                # Resize start_frame to match bucket dimensions
                debug(f"Resizing start frame from {start_frame.shape[1]}x{start_frame.shape[0]} to bucket size {width}x{height}")
                s_np = resize_and_center_crop(start_frame, target_width=width, target_height=height)
                input_anchor_np = s_np
            else:
                # Use gray color with bucket dimensions
                input_anchor_np = np.ones((height, width, 3), dtype=np.uint8) * 128
                debug(f"Created gray start frame with bucket dimensions {width}x{height}")
            
            # --- End frame processing with bucket dimensions ---
            end_np = resize_and_center_crop(end_frame, target_width=width, target_height=height)
            
            # --- VAE encode ---
            input_anchor_tensor = torch.from_numpy(input_anchor_np).float() / 127.5 - 1
            input_anchor_tensor = input_anchor_tensor.permute(2, 0, 1)[None, :, None].float()
            start_latent = vae_encode(input_anchor_tensor, vae.float())
            
            # --- End frame processing (no need to resize, use directly) ---
            end_np = resize_and_center_crop(end_frame, target_width=width, target_height=height)
            end_tensor = torch.from_numpy(end_np).float() / 127.5 - 1
            end_tensor = end_tensor.permute(2, 0, 1)[None, :, None].float()
            end_latent = vae_encode(end_tensor, vae.float())
            
            # --- Text prompt encoding & mask logic ---
            if not high_vram:
                unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)
            lv, cp = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
            lv, mask = crop_or_pad_yield_mask(lv, 512)
            lv_n, cp_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
            lv_n, mask_n = crop_or_pad_yield_mask(lv_n, 512)
            m = mask
            m_n = mask_n
            
            # --- CLIP Vision feature extraction ---
            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu)
            
            if mode == "keyframes":
                # Process end frame with CLIP
                end_clip_output = hf_clip_vision_encode(end_np, feature_extractor, image_encoder).last_hidden_state
                
                if start_frame is not None:
                    # Process start frame with CLIP
                    start_clip_output = hf_clip_vision_encode(input_anchor_np, feature_extractor, image_encoder).last_hidden_state
                    
                    # Use weighted combination based on slider value
                    clip_output = (keyframe_weight * start_clip_output + (1.0 - keyframe_weight) * end_clip_output)
                    debug(f"Using weighted combination: {keyframe_weight:.1f} start frame, {1.0-keyframe_weight:.1f} end frame")
                else:
                    # No start frame provided - use 100% end frame embedding
                    clip_output = end_clip_output
                    debug("No start frame provided - using 100% end frame embedding")

        
        elif mode == "video_extension":
            if input_video is None:
                raise ValueError("Video extension mode requires a video to be uploaded!")
            
            debug(f"Processing video extension: direction={extension_direction}, length={extension_length}s")
            
            # Extract frames from the video
            extracted_frames, video_fps = extract_frames_from_video(
                input_video, 
                num_frames=int(extension_frames),
                from_end=(extension_direction == "Forward")
            )
            
            # Determine dimensions based on extracted frames
            frame_h, frame_w, _ = extracted_frames[0].shape
            height, width = find_nearest_bucket(frame_h, frame_w, resolution=640)
            debug(f"Using bucket dimensions for video extension: {width}x{height}")
            
            if extension_direction == "Forward":
                # For forward extension, use the last frame as start frame
                input_anchor_np = resize_and_center_crop(extracted_frames[-1], target_width=width, target_height=height)
                # Create a blank end frame
                end_np = np.ones((height, width, 3), dtype=np.uint8) * 128  # Gray
            else:  # Backward
                # For backward extension, use the first frame as end frame
                end_np = resize_and_center_crop(extracted_frames[0], target_width=width, target_height=height)
                # Create a blank start frame
                input_anchor_np = np.ones((height, width, 3), dtype=np.uint8) * 128  # Gray
            
            # Convert to tensors and encode with VAE (similar to keyframes mode)
            input_anchor_tensor = torch.from_numpy(input_anchor_np).float() / 127.5 - 1
            input_anchor_tensor = input_anchor_tensor.permute(2, 0, 1)[None, :, None].float()
            start_latent = vae_encode(input_anchor_tensor, vae.float())
            
            end_tensor = torch.from_numpy(end_np).float() / 127.5 - 1
            end_tensor = end_tensor.permute(2, 0, 1)[None, :, None].float()
            end_latent = vae_encode(end_tensor, vae.float())
            
            # Text prompt encoding & mask logic (same as keyframes mode)
            if not high_vram:
                unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)
            lv, cp = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
            lv, mask = crop_or_pad_yield_mask(lv, 512)
            lv_n, cp_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
            lv_n, mask_n = crop_or_pad_yield_mask(lv_n, 512)
            m = mask
            m_n = mask_n
            
            # CLIP Vision feature extraction
            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu)
            
            # Extract CLIP features (focuses on the main conditioning frame)
            if extension_direction == "Forward":
                clip_output = hf_clip_vision_encode(input_anchor_np, feature_extractor, image_encoder).last_hidden_state
            else:  # Backward
                clip_output = hf_clip_vision_encode(end_np, feature_extractor, image_encoder).last_hidden_state

        
        elif mode == "text2video":
            width, height = get_dims_from_aspect(aspect, custom_w, custom_h)
            if init_color is not None:
                r, g, b = parse_hex_color(init_color)
                debug(f"worker: Using color picker value {init_color} -> RGB {r},{g},{b}")
                input_image_arr = np.zeros((height, width, 3), dtype=np.uint8)
                input_image_arr[:, :, 0] = int(r * 255)
                input_image_arr[:, :, 1] = int(g * 255)
                input_image_arr[:, :, 2] = int(b * 255)
            else:
                input_image_arr = np.zeros((height, width, 3), dtype=np.uint8)
                debug("worker: No color provided, defaulting to black")
            input_image = input_image_arr
        
        # ---------- Text & Prompt encodings ----------
        debug("worker: preparing inputs")
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        debug("worker: pushed 'Text encoding ...' progress event")
        if mode != "keyframes":
            # --- Image2Video/Text2Video (legacy/unchanged)
            inp_np, inp_tensor, lv, cp, lv_n, cp_n, m, m_n, height, width = prepare_inputs(
                input_image, prompt, n_prompt, cfg
            )
            start_latent = vae_encode(inp_tensor.float(), vae.float())

        debug("worker: VAE encoded")
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        debug("worker: pushed 'CLIP Vision encoding ...' progress event")
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)
            debug("worker: loaded image_encoder to gpu")
        if mode == "text2video" or mode == "image2video":
            clip_output = hf_clip_vision_encode(inp_np, feature_extractor, image_encoder).last_hidden_state
        debug("worker: got clip output last_hidden_state")
        lv = lv.to(transformer.dtype)
        lv_n = lv_n.to(transformer.dtype)
        cp = cp.to(transformer.dtype)
        cp_n = cp_n.to(transformer.dtype)
        if clip_output is not None:
            clip_output = clip_output.to(transformer.dtype)

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
        debug("worker: pushed 'Start sampling ...' progress event")
        rnd = torch.Generator("cpu").manual_seed(seed)

        history_latents = torch.zeros(
            size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32
        ).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        # -------- SECTION PATCH/STITCH LOOP ----------
        # Calculate latent paddings with special pattern
        if total_sections > 4:
            # Special pattern for longer videos that improves transitions
            latent_paddings = [3] + [2] * (total_sections - 3) + [1, 0]
            debug(f"worker: Using special padding pattern for {total_sections} sections: {latent_paddings}")
        else:
            # For 4 or fewer sections, use the standard reversed range
            latent_paddings = list(reversed(range(total_sections)))
            debug(f"worker: Using standard padding for {total_sections} sections: {latent_paddings}")
        
        # Initialize video tracking variables
        history_latents = torch.zeros(
            size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32
        ).cpu()
        history_pixels = None
        total_generated_latent_frames = 0
        
        # Process each section with the padding pattern
        for latent_padding in latent_paddings:
            is_first_section = latent_padding == latent_paddings[0]
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size
            
            debug(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')
            
            split_sizes = [1, latent_padding_size, latent_window_size, 1, 2, 16]
            total_indices = sum(split_sizes)
            indices = torch.arange(total_indices).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split(split_sizes, dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            
            # Setup the clean latents for ALL modes first
            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            
            # Then selectively override for keyframes mode first section
            if mode == "keyframes" and is_first_section:
                clean_latents_post = end_latent.to(history_latents)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            
            # ------- mask fallback safeguard -------
            m   = m if m is not None else torch.ones_like(lv)
            m_n = m_n if m_n is not None else torch.ones_like(lv_n)

            # -- memory mgmt
            if not high_vram:
                unload_complete_models()
                debug("worker: unloaded complete models")
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
                debug("worker: moved transformer to gpu (memory preservation)")

            transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=steps if use_teacache else 0)
            debug("worker: teacache initialized", "use_teacache", use_teacache)

            # --- sampling ---
            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)
                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
                
                if stream.input_queue.top() == 'end':
                    debug("worker: callback: received 'end', stopping generation.")
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')
                
                # Section progress
                current_step = d['i'] + 1
                section_percentage = int(100.0 * current_step / steps)
                
                # Overall progress
                current_section_index = latent_paddings.index(latent_padding)
                sections_completed = current_section_index 
                overall_percentage = int(100.0 * (sections_completed + (current_step / steps)) / len(latent_paddings))
                
                hint = f'Section {sections_completed+1}/{len(latent_paddings)} - Step {current_step}/{steps}'
                desc = f'Generated frames: {total_generated_latent_frames}, Length: {total_generated_latent_frames/30.0:.2f}s (FPS-30)'
                
                # Create dual progress bar HTML
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
                
                debug(f"worker: In callback, section: {section_percentage}%, overall: {overall_percentage}%")
                stream.output_queue.push(('progress', (preview, desc, progress_html)))

            if mode == "keyframes":
                generated_latents = sample_hunyuan(
                    transformer=transformer,
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
                    # IMPORTANT: Use all indices instead of None
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )   
                
            else: 
                generated_latents = sample_hunyuan(
                transformer=transformer,
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
            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
                debug(f"worker: is_last_section => concatenated latent, new shape: {generated_latents.shape}")
            
            total_generated_latent_frames += int(generated_latents.shape[2])
            debug(f"worker: Added {generated_latents.shape[2]} frames, total now: {total_generated_latent_frames}")
            
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
            debug(f"worker: history_latents.shape after concat: {history_latents.shape}")

           # ------- decode & video preview -----
            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                debug("worker: offloaded transformer")
                load_model_as_complete(vae, target_device=gpu)
                debug("worker: loaded vae to gpu (again)")
            
            # ---- Guarantee the last N latent frames match encoded end_frame ----
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents.float(), vae.float()).cpu()
                debug("worker: vae decoded (first time)")
                preview_filename = os.path.join(outputs_folder, f'{job_id}_preview_{uuid.uuid4().hex}.mp4')
                try:
                    save_bcthw_as_mp4(history_pixels, preview_filename, fps=30)
                    debug(f"[FILE] Preview video saved: {preview_filename} ({os.path.exists(preview_filename)})")
                    stream.output_queue.push(('preview_video', preview_filename))
                except Exception as e:
                    debug(f"[ERROR] Failed to save preview video: {e}")
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = frames_per_section
                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames].float(), vae.float()).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
                debug("worker: vae decoded + soft_append_bcthw")
                preview_filename = os.path.join(outputs_folder, f'{job_id}_preview_{uuid.uuid4().hex}.mp4')
                try:
                    save_bcthw_as_mp4(history_pixels, preview_filename, fps=30)
                    debug(f"[FILE] Preview video saved: {preview_filename} ({os.path.exists(preview_filename)})")
                    stream.output_queue.push(('preview_video', preview_filename))
                except Exception as e:
                    debug(f"[ERROR] Failed to save preview video: {e}")
            
            if not high_vram:
                unload_complete_models()
                debug("worker: unloaded complete models (end section)")
            
            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
            if is_last_section:
                debug("worker: is_last_section - break")
                break

        # After history_pixels is fully processed and before final video export
        if mode == "video_extension" and 'history_pixels' in locals() and history_pixels is not None:
            # Save the generated extension
            extension_filename = os.path.join(outputs_folder, f'{job_id}_extension.mp4')
            save_bcthw_as_mp4(history_pixels, extension_filename, fps=30)
            debug(f"[FILE] Extension video saved as {extension_filename}")
            
            # Calculate extension length for ffmpeg trimming
            if extension_direction == "Forward":
                # For forward extension, keep all generated frames
                extension_keep_frames = history_pixels.shape[2]
            else:  # Backward
                # For backward extension, trim a bit from end to avoid overlap
                extension_keep_frames = max(1, history_pixels.shape[2] - 2)  # Avoid duplicating first frame
            
            # Now combine with the original video
            combined_filename = os.path.join(outputs_folder, f'{job_id}_combined.mp4')
            try:
                import subprocess
                
                if extension_direction == "Forward":
                    # Append the extension to the original video
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-i", input_video,
                        "-i", extension_filename,
                        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[outv]",
                        "-map", "[outv]",
                        combined_filename
                    ]
                else:  # Backward
                    # Prepend the extension to the original video
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-i", extension_filename,
                        "-i", input_video,
                        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[outv]",
                        "-map", "[outv]",
                        combined_filename
                    ]
                    
                debug(f"[FFMPEG] Running command: {' '.join(cmd)}")
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                make_mp4_faststart(combined_filename)
                output_filename = combined_filename
                debug(f"[FILE] Combined video saved as {output_filename}")
            except Exception as e:
                debug(f"[ERROR] Failed to combine videos: {e}")
                traceback.print_exc()
                output_filename = extension_filename
                debug(f"[FILE] Using extension video as fallback: {output_filename}")

        
        # ---- Final export logic (txt2video special handling) ----
        if mode == "text2video":
            N_actual = history_pixels.shape[2]
            # ----- txt2img edge-case: make a single image if very short/low window -----
            if latent_window_size == 2 and total_sections == 1 and total_frames <= 8:
                debug("txt2img branch: pulling last frame, skipping video trim (window=2, adv=0.1)")
                last_img_tensor = history_pixels[0, :, -1]
                last_img = np.clip((np.transpose(last_img_tensor.cpu().numpy(), (1, 2, 0)) + 1) * 127.5, 0, 255).astype(np.uint8)
                img_filename = os.path.join(outputs_folder, f'{job_id}_final_image.png')
                debug(f"[FILE] Saving single-frame image {img_filename}")
                try:
                    Image.fromarray(last_img).save(img_filename)
                    debug(f"[FILE] Image saved: {img_filename}")
                    html_link = f'<a href="file/{img_filename}" target="_blank"><img src="file/{img_filename}" style="max-width:100%;border:3px solid orange;border-radius:8px;" title="Click for full size"></a>'
                    stream.output_queue.push(('file_img', (img_filename, html_link)))
                    debug(f"[QUEUE] Queued file_img event: {img_filename}")
                    stream.output_queue.push(('end', "img"))
                    debug("[QUEUE] Queued event 'end' for image")
                except Exception as e:
                    debug(f"[ERROR] Save failed for txt2img: {e}")
                    traceback.print_exc()
                    stream.output_queue.push(('end', "img"))
                return
                # ----- else: normal text2video, trim initial frames -----
            else:
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
                if N_after == 1:
                    last_img_tensor = history_pixels[0, :, 0]
                    last_img = np.clip((np.transpose(last_img_tensor.cpu().numpy(), (1,2,0)) + 1) * 127.5, 0, 255).astype(np.uint8)
                    img_filename = os.path.join(outputs_folder, f'{job_id}_final_image.png')
                    debug(f"[FILE] Final single-frame image due to trim: {img_filename}")
                    try:
                        Image.fromarray(last_img).save(img_filename)
                        debug(f"[FILE] Image saved: {img_filename}")
                        html_link = f'<a href="file/{img_filename}" target="_blank"><img src="file/{img_filename}" style="max-width:100%;border:3px solid orange;border-radius:8px;" title="Click for full size"></a>'
                        stream.output_queue.push(('file_img', (img_filename, html_link)))
                        debug(f"[QUEUE] Queued file_img event: {img_filename}")
                        stream.output_queue.push(('end', "img"))
                        debug("[QUEUE] Queued event 'end' for image")
                    except Exception as e:
                        debug(f"[ERROR] Save failed for trimmed single image: {e}")
                        traceback.print_exc()
                        stream.output_queue.push(('end', "img"))
                    return
                    
        # ----- keyframes trim when no start frame provided -----
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
                img_filename = os.path.join(outputs_folder, f'{job_id}_keyframe_final.png')
                try:
                    Image.fromarray(last_img).save(img_filename)
                    debug(f"[FILE] Image saved: {img_filename}")
                    html_link = f'<a href="file/{img_filename}" target="_blank"><img src="file/{img_filename}" style="max-width:100%;border:3px solid orange;border-radius:8px;" title="Click for full size"></a>'
                    stream.output_queue.push(('file_img', (img_filename, html_link)))
                    stream.output_queue.push(('end', "img"))
                    return
                except Exception as e:
                    debug(f"[ERROR] Save failed for keyframe single image: {e}")
                    traceback.print_exc()
                    stream.output_queue.push(('end', "img"))
                    return
        
        # --------- Final MP4 Export ---------
        debug(f"[FILE] Attempting to save video to {output_filename}")
        try:
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30)
            debug(f"[FILE] Video successfully saved to {output_filename}: {os.path.exists(output_filename)}")
            make_mp4_faststart(output_filename)
            debug(f"[FILE] Faststart patch applied to {output_filename}: {os.path.exists(output_filename)}")
            stream.output_queue.push(('file', output_filename))
            debug(f"[QUEUE] Queued event 'file' with data: {output_filename}")
        except Exception as e:
            debug(f"[ERROR] FAILED to save video {output_filename}: {e}")
            traceback.print_exc()

    except Exception as ex:
        debug("worker: EXCEPTION THROWN", ex)
        traceback.print_exc()
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        debug("worker: after exception and possible unload, exiting worker.")
        stream.output_queue.push(('end', None))
        debug("worker: exception: pushed end event")
        return
    finally:
        debug("worker: in finally block, writing final summary/progress/end")
        t_end = time.time()
        if 'history_pixels' in locals() and history_pixels is not None:
            trimmed_frames = history_pixels.shape[2]
            video_seconds = trimmed_frames / 30.0
        else:
            trimmed_frames = 0
            video_seconds = 0.0
        summary_string = (
            f"Finished!\n"
            f"Total generated frames: {trimmed_frames}, "
            f"Video length: {video_seconds:.2f} seconds (FPS-30), "
            f"Time taken: {t_end - t_start:.2f}s."
        )
        stream.output_queue.push(('progress', (None, summary_string, "")))
        debug("worker: pushed final progress event")
        stream.output_queue.push(('end', None))
        debug("worker: pushed end event in finally (done)")
            
def process(
    mode, input_image, start_frame, end_frame, aspect_selector, custom_w, custom_h,
    prompt, n_prompt, seed,
    use_adv, adv_window, adv_seconds, selected_frames,
    steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, lock_seed, init_color,
    keyframe_weight
):
    global stream
    debug("process: called with mode", mode)
    assert mode in ['image2video', 'text2video', 'keyframes'], "Invalid mode"
    
    # Create initial empty progress bars HTML
    empty_progress = """
    <div class="dual-progress-container">
        <div class="progress-label">
            <span>Current Section:</span>
            <span>0%</span>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fg" style="width: 0%"></div>
        </div>
        
        <div class="progress-label">
            <span>Overall Progress:</span>
            <span>0%</span>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fg" style="width: 0%"></div>
        </div>
        
        <div style="font-size:0.9em; opacity:0.8;">Preparing...</div>
    </div>
    """
    
    if mode == 'image2video' and input_image is None:
        debug("process: Aborting early -- no input image for image2video")
        yield (
            None, None, None,
            "Please upload an input image!", None,
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update()
        )
        return
    if not lock_seed:
        seed = int(time.time()) % 2**32
    debug("process: entering main async_run yield cycle, seed:", seed)
    
    yield (
        None, None, None, '', gr.update(visible=False),  # Progress bar hidden at start
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(value=seed)
    )
    stream = AsyncStream()
    async_run(
        worker,
        mode,
        input_image,
        start_frame,
        end_frame,
        aspect_selector,
        custom_w,
        custom_h,
        prompt,
        n_prompt,
        seed,
        use_adv, adv_window, adv_seconds, selected_frames,
        steps,
        cfg,
        gs,
        rs,
        gpu_memory_preservation,
        use_teacache,
        init_color,
        keyframe_weight
    )
    output_filename = None
    last_desc = ""
    last_is_image = False
    last_img_path = None
    
    while True:
        flag, data = stream.output_queue.next()
        debug(f"process: got queue event: {flag}, type(data): {type(data)}")
        if flag == 'file':
            output_filename = data
            yield (
                gr.update(value=output_filename, visible=True), # result_video (show final video)
                gr.update(visible=False),                       # result_image_html
                gr.update(visible=False),                       # preview_image
                gr.update(value="", visible=False),             # progress_desc
                gr.update(value="", visible=False),             # progress_bar
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update()
            )
            last_is_image = False
            last_img_path = None
        elif flag == 'preview_video':
            preview_filename = data
            debug(f"[UI] Got preview_video event: {preview_filename}")
            yield (
                gr.update(value=preview_filename, visible=True), # result_video keep VISIBLE and update with preview
                gr.update(visible=False),                        # result_image_html
                gr.update(visible=False),                        # preview_image
                "Generating preview...",                         # progress_desc
                gr.update(value="", visible=False),              # progress_bar
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update()
            )
        elif flag == 'progress':
            preview, desc, html = data
            if desc:
                last_desc = desc
            debug(f"process: yielding progress output: desc={desc}")
            # Make sure to set visible=True for progress bar
            yield (
                gr.update(),                           # result_video
                gr.update(),                           # result_image_html
                gr.update(visible=True, value=preview), # preview_image
                desc,                                  # progress_desc
                gr.update(value=html, visible=True),   # Make progress bar visible with EACH update
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update()
            )
        elif flag == 'file_img':
            (img_filename, html_link) = data
            debug("process: yielding file_img/single image output", img_filename)
            yield (
                gr.update(visible=False),                           # result_video
                gr.update(value=img_filename, visible=True),        # result_image_html as Image!
                gr.update(visible=False),                           # preview_image
                f"Generated single image!<br>Saved as <code>{img_filename}</code>",  # progress_desc
                gr.update(visible=False),                           # progress_bar
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update()
            )
            last_is_image = True
            last_img_path = img_filename
        elif flag == 'end':
            debug("process: yielding end event. output_filename =", output_filename)
            if data == "img" or last_is_image:  # special image end
                yield (
                    gr.update(visible=False),               # result_video
                    gr.update(visible=True),                # result_image_html (keep image visible)
                    gr.update(visible=False),               # preview_image
                    f"Generated single image!<br><a href=\"file/{img_filename}\" target=\"_blank\">Click here to open full size in new tab.</a><br><code>{img_filename}</code>",  # progress_desc
                    gr.update(visible=False),               # progress_bar
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    gr.update()
                )
            else:
                yield (
                    gr.update(value=output_filename, visible=True), # result_video
                    gr.update(visible=False),                       # result_image_html
                    gr.update(visible=False),                       # preview_image
                    gr.update(value=last_desc, visible=True),       # progress_desc
                    gr.update(value="", visible=False),             # progress_bar
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    gr.update()
                )
            debug("process: end event, breaking loop.")
            break
        else:
            last_is_image = False
            last_img_path = None

def end_process():
    stream.input_queue.push('end')

css = """
.gr-box, .gr-image, .gr-video {
    border: 2px solid orange !important;
    border-radius: 8px !important;
    margin-bottom: 16px;
    background: #222 !important;
}

/* Progress Bar Styling */
.dual-progress-container {
    background: #222;
    border: 2px solid orange;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
}

.progress-label {
    font-weight: bold;
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
}

.progress-bar-bg {
    background: #333;
    border-radius: 4px;
    height: 20px;
    overflow: hidden;
    margin-bottom: 12px;
}

.progress-bar-fg {
    height: 100%;
    background: linear-gradient(90deg, #ff8800, #ff2200);
    border-radius: 4px;
    transition: width 0.3s ease;
}
"""

block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column(scale=2):
            mode_selector = gr.Radio(
                ["image2video", "text2video", "keyframes", "video_extension"],
                value="image2video", 
                label="Mode"
            )
            input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)  # always present, sometimes hidden
            start_frame = gr.Image(sources='upload', type="numpy", label="Start Frame (Optional)", height=320, visible=False)
            end_frame = gr.Image(sources='upload', type="numpy", label="End Frame (Required)", height=320, visible=False)
            aspect_selector = gr.Dropdown(
                ["16:9", "9:16", "1:1", "4:5", "3:2", "2:3", "21:9", "4:3", "Custom..."],
                label="Aspect Ratio",
                value="1:1",
                visible=False
            )
            custom_w = gr.Number(label="Width", value=768, visible=False)
            custom_h = gr.Number(label="Height", value=768, visible=False)
            prompt = gr.Textbox(label="Prompt", value='', lines=3)
            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)
            advanced_mode = gr.Checkbox(label="Advanced Mode", value=False)
            latent_window_size = gr.Slider(label="Latent Window Size", minimum=2, maximum=33, value=9, step=1, visible=False)
            adv_seconds = gr.Slider(label="Video Length (Seconds)", minimum=0.1, maximum=120.0, value=5.0, step=0.1, visible=False)
            total_frames_dropdown = gr.Dropdown(
                label="Output Video Frames",
                choices=[str(x) for x in get_valid_frame_stops(9)],
                value=str(get_valid_frame_stops(9)[0]),
                visible=True
            )
            init_color = gr.ColorPicker(label="Initial Frame Color", value="#808080", visible=False)
            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True)
                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                seed = gr.Number(label="Seed", value=random.randint(0, 2**32-1), precision=0)
                lock_seed = gr.Checkbox(label="Lock Seed", value=False)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)
                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB)", minimum=6, maximum=128, value=6, step=0.1)
        with gr.Column(scale=2):
            progress_bar = gr.HTML(visible=False)  # Start hidden
            progress_desc = gr.Markdown(visible=False)
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            result_image_html = gr.Image(label='Single Frame Image', visible=False)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            with gr.Group(visible=False) as keyframes_options:
                keyframe_weight = gr.Slider(
                    label="Start Frame Influence", 
                    minimum=0.0, 
                    maximum=1.0, 
                    value=0.7, 
                    step=0.1,
                    info="Higher values prioritize start frame characteristics (0 = end frame only, 1 = start frame only)"
                )

            with gr.Group(visible=False) as video_extension_options:
                input_video = gr.Video(
                    label="Upload Video to Extend", 
                    format="mp4"
                )
                extension_direction = gr.Radio(
                    ["Forward", "Backward"], 
                    label="Extension Direction",
                    value="Forward",
                    info="Forward extends the end, Backward extends the beginning"
                )
                extension_length = gr.Slider(
                    label="Extension Length (seconds)", 
                    minimum=1.0, 
                    maximum=10.0, 
                    value=3.0, 
                    step=0.1
                )
                extension_frames = gr.Slider(
                    label="Context Frames", 
                    minimum=1, 
                    maximum=16, 
                    value=8, 
                    step=1,
                    info="Number of frames to extract from video for continuity"
                )

    # --- calllbacks ---
    def update_frame_dropdown(window):
        stops = get_valid_frame_stops(window)
        # Only let user select from valid numbers of frames
        if stops:
            return gr.update(choices=[str(x) for x in stops], value=str(stops[0]))
        else:
            return gr.update(choices=[''], value='')
    def show_hide_advanced(show, window):
        lw_vis = gr.update(visible=show)
        secs_vis = gr.update(visible=show)
        dropdown_vis = gr.update(visible=not show)
        if not show:
            dropdown_update = update_frame_dropdown(window)
            dropdown_update["visible"] = True
            return lw_vis, secs_vis, dropdown_update
        else:
            return lw_vis, secs_vis, dropdown_vis
    def switch_mode(mode):
        return (
            gr.update(visible=mode == "image2video"),  # input_image
            gr.update(visible=(mode == "keyframes")),  # start_frame
            gr.update(visible=(mode == "keyframes")),  # end_frame
            gr.update(visible=(mode == "text2video")),  # aspect_selector
            gr.update(visible=(mode == "text2video" and aspect_selector.value == "Custom...")), # custom_w
            gr.update(visible=(mode == "text2video" and aspect_selector.value == "Custom...")), # custom_h
            gr.update(visible=(mode == "keyframes")),  # keyframes_options
            gr.update(visible=(mode == "video_extension"))  # video_extension_options
        )
    def show_custom(aspect):
        show = aspect == "Custom..."
        return gr.update(visible=show), gr.update(visible=show)
    advanced_mode.change(
        show_hide_advanced,
        inputs=[advanced_mode, latent_window_size],
        outputs=[latent_window_size, adv_seconds, total_frames_dropdown],
    )
    latent_window_size.change(
        lambda window, adv: update_frame_dropdown(window) if not adv else gr.update(),
        inputs=[latent_window_size, advanced_mode],
        outputs=[total_frames_dropdown],
    )
    mode_selector.change(
        switch_mode,
        inputs=[mode_selector],
        outputs=[input_image, start_frame, end_frame, aspect_selector, custom_w, custom_h,  keyframes_options]
    )
    def show_init_color(mode):
        return gr.update(visible=(mode == "text2video"))
    
    mode_selector.change(
        show_init_color,
        inputs=[mode_selector],
        outputs=[init_color]
    )
    aspect_selector.change(
        show_custom,
        inputs=[aspect_selector],
        outputs=[custom_w, custom_h],
    )
    ips = [
        mode_selector,
        input_image,    # For im2vid/txt2vid
        start_frame,    # For keyframes (optional)
        end_frame,      # For keyframes (required)
        aspect_selector,
        custom_w,
        custom_h,
        prompt,
        n_prompt,
        seed,
        advanced_mode,
        latent_window_size,
        adv_seconds,
        total_frames_dropdown,
        steps,
        cfg,
        gs,
        rs,
        gpu_memory_preservation,
        use_teacache,
        lock_seed,
        init_color,
        keyframe_weight,
        input_video,
        extension_direction,
        extension_length,
        extension_frames,
    ]
    prompt.submit(
        fn=process,
        inputs=ips,
        outputs=[
            result_video,
            result_image_html,
            preview_image,
            progress_desc,
            progress_bar,
            start_button,
            end_button,
            seed
        ]
    )

      
    start_button.click(
        fn=process,
        inputs=ips,
        outputs=[
            result_video,      # 0
            result_image_html, # 1 (HTML for clickable img)
            preview_image,     # 2
            progress_desc,     # 3
            progress_bar,      # 4
            start_button,      # 5
            end_button,        # 6
            seed               # 7
        ]
    )
    end_button.click(fn=end_process)
block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
