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

def get_valid_frame_stops(latent_window_size, max_seconds=120, fps=30):
    frames_per_section = latent_window_size * 4 - 3
    max_sections = int((max_seconds * fps) // frames_per_section)
    stops = [frames_per_section * i for i in range(1, max_sections + 1)]
    return stops
    
def fix_video_compatibility(video_path, fps=30):
    """Fix compatibility issues with MP4 videos for QuickTime and Windows Media Player."""
    import os
    import subprocess
    
    if not os.path.exists(video_path):
        debug(f"[VIDEO FIX] Video not found: {video_path}")
        return False
    
    # Create a temporary file path
    temp_path = video_path + ".compatible.mp4"
    
    # Use ffmpeg to convert with proper settings
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-c:v", "libx264",           # Use H.264 codec
        "-pix_fmt", "yuv420p",       # Use widely supported pixel format
        "-crf", "23",                # Reasonable quality level (lower = better)
        "-preset", "medium",         # Encoding speed/quality tradeoff
        "-movflags", "+faststart",   # Enable faststart
        "-metadata", f"encoder=FeatureScope",
        temp_path
    ]
    
    try:
        debug(f"[VIDEO FIX] Running compatibility fix for {video_path}")
        process = subprocess.run(
            cmd, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Check if output file exists and has valid size
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1000:
            # Replace original with compatible version
            os.replace(temp_path, video_path)
            debug(f"[VIDEO FIX] Successfully fixed compatibility for {video_path}")
            return True
        else:
            debug(f"[VIDEO FIX] Failed to create valid output file")
            return False
            
    except subprocess.CalledProcessError as e:
        debug(f"[VIDEO FIX] FFMPEG error: {e}")
        stderr = e.stderr.decode('utf-8') if e.stderr else "Unknown error"
        debug(f"[VIDEO FIX] FFMPEG stderr: {stderr}")
        return False
    except Exception as e:
        debug(f"[VIDEO FIX] Unexpected error: {e}")
        return False
    finally:
        # Clean up temp file if it still exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
                
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

def apply_gaussian_blur(image_tensor, blur_amount):
    """Apply gaussian blur to input tensor with specified amount (0.0-1.0)"""
    if blur_amount <= 0.0:
        return image_tensor
    
    # Try multiple methods to apply blur
    try:
        # Method 1: Try torchvision (most reliable)
        import torchvision.transforms.functional as TF
        
        kernel_size = int(blur_amount * 20) + 1
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        sigma = [blur_amount * 3.0]
        
        if len(image_tensor.shape) == 5:  # [B,C,T,H,W]
            b, c, t, h, w = image_tensor.shape
            result = torch.zeros_like(image_tensor)
            
            for bi in range(b):
                for ti in range(t):
                    frame = image_tensor[bi, :, ti]
                    result[bi, :, ti] = TF.gaussian_blur(frame, kernel_size, sigma)
            
            return result
        else:
            return TF.gaussian_blur(image_tensor, kernel_size, sigma)
            
    except (ImportError, AttributeError, ModuleNotFoundError):
        # Method 2: Use OpenCV as fallback
        try:
            import cv2
            import numpy as np
            
            kernel_size = int(blur_amount * 20) + 1
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            sigma = blur_amount * 3.0
            
            # We need to handle the numpy conversion
            if len(image_tensor.shape) == 5:  # [B,C,T,H,W]
                b, c, t, h, w = image_tensor.shape
                result = torch.zeros_like(image_tensor)
                
                for bi in range(b):
                    for ti in range(t):
                        # Get frame, convert to numpy, apply blur, convert back
                        frame = image_tensor[bi, :, ti].cpu().numpy()  # C,H,W
                        
                        # OpenCV expects HWC format
                        frame = np.transpose(frame, (1, 2, 0))  # H,W,C
                        
                        # Apply blur (make sure kernel size is odd)
                        blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
                        
                        # Convert back to tensor format CHW
                        blurred = np.transpose(blurred, (2, 0, 1))  # C,H,W
                        result[bi, :, ti] = torch.from_numpy(blurred).to(image_tensor.device)
                
                return result
            else:
                # Assume BCHW
                b, c, h, w = image_tensor.shape
                result = torch.zeros_like(image_tensor)
                
                for bi in range(b):
                    frame = image_tensor[bi].cpu().numpy()  # C,H,W
                    frame = np.transpose(frame, (1, 2, 0))  # H,W,C
                    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
                    blurred = np.transpose(blurred, (2, 0, 1))  # C,H,W
                    result[bi] = torch.from_numpy(blurred).to(image_tensor.device)
                
                return result
        
        except (ImportError, AttributeError, ModuleNotFoundError):
            debug("WARNING: Could not apply blur - no suitable method found. Returning original image.")
            return image_tensor

# ---- CLI Debug Output ----
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

# Step 1: Define the extract_frames_from_video function
def extract_frames_from_video(video_path, num_frames=8, from_end=True, max_resolution=640):
    """
    Extract frames from a video file with bucket resizing for memory efficiency.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for video extension. Please install it with 'pip install opencv-python'")
    
    # Safety check to ensure at least 1 frame
    num_frames = max(1, int(num_frames))
    
    debug(f"Extracting frames from video: {video_path}, num_frames={num_frames}, from_end={from_end}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    debug(f"Video properties: total_frames={total_frames}, fps={fps}, dimensions={orig_width}x{orig_height}")
    
    # Additional safety check for video with no frames
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")
        
    # Calculate bucket dimensions
    bucket_height, bucket_width = find_nearest_bucket(orig_height, orig_width, resolution=max_resolution)
    debug(f"Using bucket dimensions: {bucket_width}x{bucket_height}")
    
    # Calculate frame indices to extract
    if from_end:
        start_idx = max(0, total_frames - num_frames)
        frame_indices = list(range(start_idx, total_frames))
    else:
        frame_indices = list(range(min(num_frames, total_frames)))
    
    debug(f"Will extract frames at indices: {frame_indices}")
    
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
            # Resize to bucket dimensions
            frame = resize_and_center_crop(frame, target_width=bucket_width, target_height=bucket_height)
            frames.append(frame)
            debug(f"Extracted frame {current_frame}")
        
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
    
    return frames, fps, (orig_height, orig_width)

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
def prepare_inputs(input_image, prompt, n_prompt, cfg, gaussian_blur_amount=0.0, 
                   llm_weight=1.0, clip_weight=1.0):
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

    # Apply encoder weights
    if llm_weight != 1.0:
        llama_vec = llama_vec * llm_weight
        llama_vec_n = llama_vec_n * llm_weight
    
    if clip_weight != 1.0:
        clip_pool = clip_pool * clip_weight
        clip_pool_n = clip_pool_n * clip_weight
        
    h, w = find_nearest_bucket(H, W, resolution=640)
    input_np = resize_and_center_crop(input_image, target_width=w, target_height=h)
    Image.fromarray(input_np).save(os.path.join(outputs_folder, f'{generate_timestamp()}.png'))
    input_tensor = torch.from_numpy(input_np).float() / 127.5 - 1
    input_tensor = input_tensor.permute(2, 0, 1)[None, :, None]
    # Apply gaussian blur if needed
    if gaussian_blur_amount > 0.0:
        input_tensor = apply_gaussian_blur(input_tensor, gaussian_blur_amount)
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
# ---- WORKER ----
@torch.no_grad()
def worker(
    mode, input_image, start_frame, end_frame, aspect, custom_w, custom_h,
    prompt, n_prompt, seed,
    use_adv, adv_window, adv_seconds, selected_frames,
    steps, cfg, gs, rs, gpu_memory_preservation, use_teacache,
    init_color, keyframe_weight,
    input_video=None, extension_direction="Forward", extension_frames=8,
    original_mode=None 
):
    job_id = generate_timestamp()
    debug("worker(): started", mode, "job_id:", job_id)

    # --- DEFINE DEFAULT OUTPUT FILENAME ---
    output_filename = os.path.join(outputs_folder, f'{job_id}_final.mp4')
    debug(f"worker: Default output filename set to: {output_filename}")
    
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
            
            if mode == "video_extension":
                if input_video is None:
                    debug("process: Aborting early -- no input video for video_extension")
                    yield (
                        None, None, None,
                        "Please upload a video to extend!", None,
                        gr.update(interactive=True),
                        gr.update(interactive=False),
                        gr.update()
                    )
                    return
                    
                try:
                    # Make sure extension_frames is at least 1
                    extension_frames_val = max(1, int(extension_frames))
                    debug(f"Extracting frames from video for {extension_direction} extension (frames={extension_frames_val})")
                    
                    # Extract frames from the video
                    extracted_frames, video_fps, _ = extract_frames_from_video(
                        input_video,
                        num_frames=extension_frames_val,
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
                
                # Let processing continue with normal mode handling
                mode = "image2video"  # Redirect to use image2video processing path
                debug(f"Redirecting to image2video path with selected frame as input")
            
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
        # Calculate latent paddings list - *ONLY FOR CALLBACK INFO*
        # This calculation is optional if total_sections is reliably calculated above.
        # If you keep it, ensure it uses 'total_sections'
        if total_sections > 4:
            latent_paddings_list_for_info = [3] + [2] * (total_sections - 3) + [1, 0]
            debug(f"worker: Calculated padding list {latent_paddings_list_for_info} for info (len={len(latent_paddings_list_for_info)})")
        else:
            latent_paddings_list_for_info = list(reversed(range(total_sections)))
            debug(f"worker: Calculated standard padding list {latent_paddings_list_for_info} for info (len={len(latent_paddings_list_for_info)})")
        # Check length (optional)
        if len(latent_paddings_list_for_info) != total_sections:
             debug(f"WARNING: Mismatch between total_sections ({total_sections}) and latent_paddings_list_for_info length ({len(latent_paddings_list_for_info)})")

        # --- FORCE OLD ITERATION SCHEME ---
        debug(f"worker: [TESTING] Forcing old iteration scheme: reversed(range(total_sections={total_sections}))")
        loop_iterator = reversed(range(total_sections))
        # --- END FORCE OLD ITERATION ---

        # Process each section using the old iteration method
        for section in loop_iterator: # Iterates from total_sections-1 down to 0
            # Determine section properties (Unchanged)
            is_last_section = section == 0
            latent_padding_size = section * latent_window_size
            is_first_iteration = (section == total_sections - 1)
            debug(f'section = {section}, latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            # Check for abort signal (Unchanged)
            if stream.input_queue.top() == 'end':
                 debug("worker: input_queue 'end' received. Aborting generation.")
                 stream.output_queue.push(('end', None))
                 return

            # Setup indices and clean latents for sample_hunyuan (Uses latent_padding_size)
            split_sizes = [1, latent_padding_size, latent_window_size, 1, 2, 16]
            total_indices = sum(split_sizes)
            indices = torch.arange(total_indices).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split(split_sizes, dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            # Setup the clean latents for ALL modes first
            clean_latents_pre = start_latent.to(history_latents.device, dtype=history_latents.dtype)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            # Selectively override for keyframes mode FIRST ITERATION
            # 'is_first_iteration' now correctly flags the start of the reversed loop
            if mode == "keyframes" and is_first_iteration:
                debug("Keyframes mode: Overriding clean_latents_post with end_latent for first iteration.")
                clean_latents_post = end_latent.to(history_latents.device, dtype=history_latents.dtype)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            # Mask fallback safeguard (Unchanged)
            m   = m if m is not None else torch.ones_like(lv)
            m_n = m_n if m_n is not None else torch.ones_like(lv_n)

            # Memory mgmt before sampling (Unchanged)
            if not high_vram:
                unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
                # Note: The debug message here might be slightly confusing as it says "(end section)"
                # but it runs *before* the section's sampling. This was in your provided code.
                debug("worker: explicitly unloaded all models (before sampling)")
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
                debug("worker: moved transformer to gpu (memory preservation)")

            transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=steps if use_teacache else 0)
            debug("worker: teacache initialized", "use_teacache", use_teacache)

            # --- Define the *callback adapted for the 'section' index* ---
            def callback(d):
                # Preview generation (Unchanged)
                preview = d['denoised']
                preview = vae_decode_fake(preview)
                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
                if stream.input_queue.top() == 'end':
                    debug("worker: callback: received 'end', stopping generation.")
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')
    
                # Section progress (Unchanged)
                current_step = d['i'] + 1
                section_percentage = int(100.0 * current_step / steps)
    
                # Overall progress (ADAPTED - FIX HERE)
                # 'section' goes from total_sections-1 down to 0
                # Use total_sections instead of local_total_sections
                sections_completed = (total_sections - 1) - section # How many sections came *before* this one

                # FIX HERE: Guard against division by zero if total_sections could potentially be 0
                if total_sections > 0:
                    overall_percentage = int(100.0 * (sections_completed + (current_step / steps)) / total_sections)
                else:
                    overall_percentage = 0 # Or handle as appropriate if total_sections is 0
                
                # Calculate actual frame count (Based on history_pixels, as before)
                actual_pixel_frames = history_pixels.shape[2] if history_pixels is not None else 0
                actual_seconds = actual_pixel_frames / 30.0

                # FIX HERE: Use total_sections in the f-string
                hint = f'Section {sections_completed+1}/{total_sections} - Step {current_step}/{steps}'
                desc = f'Pixel frames generated: {actual_pixel_frames}, Length: {actual_seconds:.2f}s (FPS-30)'
    
                # Create dual progress bar HTML (Unchanged - uses calculated percentages)
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
    
                # Debug message and pushing to queue (Unchanged)
                debug(f"worker: In callback, section: {section_percentage}%, overall: {overall_percentage}%")
                stream.output_queue.push(('progress', (preview, desc, progress_html)))
            # --- End adapted callback definition ---

            # --- Run sampling (Passes the correct indices/latents based on padding_size) ---
            # The mode check here ensures correct arguments like image_embeddings are passed
            if mode == "keyframes":
                generated_latents = sample_hunyuan(
                    transformer=transformer, sampler="unipc", width=width, height=height, frames=frames_per_section,
                    real_guidance_scale=cfg, distilled_guidance_scale=gs, guidance_rescale=rs,
                    num_inference_steps=steps, generator=rnd, prompt_embeds=lv, prompt_embeds_mask=m,
                    prompt_poolers=cp, negative_prompt_embeds=lv_n, negative_prompt_embeds_mask=m_n,
                    negative_prompt_poolers=cp_n, device=gpu, dtype=torch.bfloat16, image_embeddings=clip_output,
                    latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )
            else: # image2video, text2video, video_extension redirected to image2video
                generated_latents = sample_hunyuan(
                    transformer=transformer, sampler='unipc', width=width, height=height, frames=frames_per_section,
                    real_guidance_scale=cfg, distilled_guidance_scale=gs, guidance_rescale=rs,
                    num_inference_steps=steps, generator=rnd, prompt_embeds=lv, prompt_embeds_mask=m,
                    prompt_poolers=cp, negative_prompt_embeds=lv_n, negative_prompt_embeds_mask=m_n,
                    negative_prompt_poolers=cp_n, device=gpu, dtype=torch.bfloat16, image_embeddings=clip_output,
                    latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback
                )

            # --- Post-sampling ---
            # Handle the last section specially (add start latent if needed)
            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents.device, dtype=generated_latents.dtype), generated_latents], dim=2)
                debug(f"worker: is_last_section => concatenated latent, new shape: {generated_latents.shape}")

            # Update latent frame count (Tracks only newly added frames)
            new_latent_frames = generated_latents.shape[2]
            # Note: This 'total_generated_latent_frames' might be slightly misleading if only used in desc.
            # The VAE decode step below uses the actual length of history_latents.
            # total_generated_latent_frames += new_latent_frames # Commenting out as it wasn't used correctly
            # debug(f"worker: Added {new_latent_frames} latent frames this section.")

            # Update history latents (CPU tensor)
            history_latents = torch.cat([generated_latents.to(history_latents.device, dtype=history_latents.dtype), history_latents], dim=2)
            debug(f"worker: history_latents.shape after concat: {history_latents.shape}")

            # --- VAE Decoding Section (COMBINED BEST VERSION) ---
            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)
                debug("worker: loaded vae to gpu")
            
            # Clear CUDA cache to reduce fragmentation
            torch.cuda.empty_cache()
            debug("Cleared CUDA cache before decoding")
            
            # Calculate theoretical max overlap frames (keep this for reference)
            max_overlapped_frames = latent_window_size * 4 - 3
            debug(f"Theoretical max overlapped_frames={max_overlapped_frames}")
            
            try:
                # Decode the newly generated latents in chunks
                debug(f"Decoding generated_latents with shape: {generated_latents.shape}")
                chunk_size = 4  # Small enough to avoid OOM
                current_pixels_chunks = []
                
                for i in range(0, generated_latents.shape[2], chunk_size):
                    end_idx = min(i + chunk_size, generated_latents.shape[2])
                    debug(f"Decoding chunk {i}:{end_idx} of {generated_latents.shape[2]}")
                    
                    # Move chunk to GPU, decode, then immediately move result to CPU
                    chunk = generated_latents[:, :, i:end_idx].to(vae.device, dtype=vae.dtype)
                    chunk_pixels = vae_decode(chunk, vae).cpu()
                    current_pixels_chunks.append(chunk_pixels)
                    
                    # Force cleanup of chunk tensors
                    del chunk, chunk_pixels
                    torch.cuda.empty_cache()
                
                # Combine all chunks on CPU
                current_pixels = torch.cat(current_pixels_chunks, dim=2)
                debug(f"Successfully decoded in chunks: final shape {current_pixels.shape}")
                del current_pixels_chunks
                torch.cuda.empty_cache()
                
                debug(f"Decoded newly generated latents to pixels with shape: {current_pixels.shape}")
                
                # Initialize history_pixels if this is the first section
                if history_pixels is None:
                    history_pixels = current_pixels
                    debug(f"First section: Set history_pixels directly with shape: {history_pixels.shape}")
                else:
                    # For sequential sections, simply prepend the new frames 
                    # (since we're generating from end to beginning)
                    # IMPORTANT: Use simple concatenation instead of soft_append_bcthw
                    # This eliminates the double exposure effect
                    history_pixels = torch.cat([current_pixels, history_pixels], dim=2)
                    debug(f"Concatenated new frames without blending. New history shape: {history_pixels.shape}")
                
                # === RESTORE PREVIEW VIDEO FUNCTIONALITY ===
                # Save preview for progress updates
                preview_filename = os.path.join(outputs_folder, f'{job_id}_preview_{uuid.uuid4().hex}.mp4')
                try:
                    save_bcthw_as_mp4(history_pixels, preview_filename, fps=30)
                    debug(f"[FILE] Preview video saved: {preview_filename} ({os.path.exists(preview_filename)})")
                    stream.output_queue.push(('preview_video', preview_filename))
                    debug(f"[QUEUE] Queued preview_video event: {preview_filename}")
                except Exception as e:
                    debug(f"[ERROR] Failed to save preview video: {e}")
                
                # Clean up memory
                del current_pixels
                torch.cuda.empty_cache()
                
            except Exception as e:
                debug(f"Error during VAE decoding: {str(e)}")
                import traceback
                debug(traceback.format_exc())
                raise
        # --- Loop finished ---

        # After history_pixels is fully processed and before final video export
        if original_mode == "video_extension" and input_video is not None and 'history_pixels' in locals() and history_pixels is not None:
            # Save the generated extension
            extension_filename = os.path.join(outputs_folder, f'{job_id}_extension.mp4')
            save_bcthw_as_mp4(history_pixels, extension_filename, fps=30)
            debug(f"[FILE] Extension video saved as {extension_filename}")
            
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
                
                # IMPORTANT: Update the output_filename to use the combined file
                output_filename = combined_filename
                debug(f"[FILE] Combined video saved as {output_filename} - using as final output")
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
        if 'history_pixels' in locals() and history_pixels is not None and history_pixels.shape[2] > 0:
            # A simple flag to check if image logic might have run.
            image_likely_saved = False
            if (mode == "text2video" or (mode == "keyframes" and start_frame is None)):
                if history_pixels.shape[2] <= 1: # This condition was used in your image saving logic
                    image_likely_saved = True
                    
            # Special handling for video_extension - use the combined file directly
            if original_mode == "video_extension" and 'combined_filename' in locals() and os.path.exists(combined_filename):
                debug(f"[FILE] Using pre-combined video file for video_extension mode: {combined_filename}")
                fix_video_compatibility(combined_filename, fps=30)
                stream.output_queue.push(('file', combined_filename))
                debug(f"[QUEUE] Queued event 'file' with data: {combined_filename}")
            elif not image_likely_saved:
                debug(f"[FILE] Attempting to save final video to {output_filename}")
                try:
                    save_bcthw_as_mp4(history_pixels, output_filename, fps=30)
                    debug(f"[FILE] Video successfully saved to {output_filename}: {os.path.exists(output_filename)}")
                    
                    fix_video_compatibility(output_filename, fps=30)
                    stream.output_queue.push(('file', output_filename))
                    debug(f"[QUEUE] Queued event 'file' with data: {output_filename}")
                except Exception as e:
                    debug(f"[ERROR] FAILED to save final video {output_filename}: {e}")
                    traceback.print_exc()
            else:
                debug(f"[FILE] Skipping final video save, likely handled by image export logic.")
        else:
            debug(f"[FILE] Skipping final video save: No valid history_pixels found.")

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
        # Use final state of history_pixels for summary
        if 'history_pixels' in locals() and history_pixels is not None:
            # Account for trimming possibly reducing frames before final save
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
    latent_window_size, segment_count,  # New parameters
    steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, lock_seed, init_color,
    keyframe_weight,
    input_video=None, extension_direction="Forward", extension_frames=8,
    frame_overlap=0, trim_pct=0.2, gaussian_blur_amount=0.0,
    llm_weight=1.0, clip_weight=1.0, clean_latent_weight=1.0
):
    global stream
    debug("process: called with mode", mode)
    assert mode in ['image2video', 'text2video', 'keyframes', 'video_extension'], "Invalid mode"
    
    # Map our new UI values to what the original worker expects
    use_adv = True  # Always use advanced mode
    adv_window = latent_window_size
    
    # Calculate frames per section (for selected_frames mapping)
    frames_per_section = latent_window_size * 4 - 3
    effective_frames = frames_per_section - min(frame_overlap, frames_per_section-1)
    
    # Map segment_count to appropriate parameter
    adv_seconds = (segment_count * effective_frames) / 30.0
    selected_frames = segment_count * effective_frames
    
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
    
    # Special handling for video_extension mode: Extract frames and set input_image
    original_mode = mode
    original_video = input_video
    
    if mode == "video_extension":
        if input_video is None:
            debug("process: Aborting early -- no input video for video_extension")
            yield (
                None, None, None,
                "Please upload a video to extend!", None,
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update()
            )
            return
            
        try:
            debug(f"Extracting frames from video for {extension_direction} extension")
            # Extract frames from the video
            extracted_frames, video_fps, _ = extract_frames_from_video(
                input_video,
                num_frames=int(extension_frames),
                from_end=(extension_direction == "Forward"),
                max_resolution=640
            )
            
            # Set input_image based on direction
            if extension_direction == "Forward":
                input_image = extracted_frames[-1]  # Use last frame
                debug(f"Using last frame as input_image for forward extension")
                mode = "image2video"  # Use image2video processing path
            else:
                # For backward extension, set up as keyframe generation
                end_frame = extracted_frames[0]  # First frame becomes the target
                start_frame = None  # No start frame needed (we're generating TO this frame)
                mode = "keyframes"  # Use keyframes mode
                debug(f"Setting up backward extension as keyframe targeting first frame of video")
            
        except Exception as e:
            debug(f"Video frame extraction error: {str(e)}")
            traceback.print_exc()
            yield (
                None, None, None,
                f"Error processing video: {str(e)}", None,
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update()
            )
            return
    
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
        use_adv, adv_window, adv_seconds, selected_frames,  # Feed mapped values to worker
        steps,
        cfg,
        gs,
        rs,
        gpu_memory_preservation,
        use_teacache,
        init_color,
        keyframe_weight,
        input_video,
        extension_direction,
        extension_frames,
        original_mode
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
            debug(f"[UI] Handling preview_video event for: {preview_filename}")
            
            # Verify file exists before setting in UI
            if os.path.exists(preview_filename):
                debug(f"[UI] Preview file exists, updating video display")
                yield (
                    gr.update(value=preview_filename, visible=True), # result_video
                    gr.update(visible=False),                      # result_image_html
                    gr.update(visible=False),                      # preview_image
                    "Generating video...",                         # progress_desc
                    gr.update(visible=True),                       # progress_bar 
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    gr.update()
                )
            else:
                debug(f"[UI] Warning: Preview file not found: {preview_filename}")
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
/* Image and Video Container Styling */
.input-image-container img {
    max-height: 512px !important;
    width: auto !important;
    object-fit: contain !important;
}

.keyframe-image-container img {
    max-height: 320px !important;
    width: auto !important;
    object-fit: contain !important;
}

.result-container img, .result-container video {
    max-height: 512px !important;
    width: auto !important;
    object-fit: contain !important;
    margin: 0 auto !important;
    display: block !important;
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

.stats-box {
    background: #333;
    border: 2px solid orange;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
}

.stats-box table {
    width: 100%;
}

.stats-box td {
    padding: 4px 8px;
}

.stats-box td:first-child {
    width: 40%;
}

.stats-box {
    background: #222;
    border: 2px solid orange;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
}

.stats-box table {
    width: 100%;
}

.stats-box td {
    padding: 4px 8px;
}

.stats-box td:first-child {
    width: 40%;
}
"""

block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack Advanced Playground by SCG')
    with gr.Row():
        with gr.Column(scale=2):
            mode_selector = gr.Radio(
                ["image2video", "text2video", "keyframes", "video_extension"],
                value="image2video", 
                label="Mode"
            )
            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)
            input_image = gr.Image(sources='upload', type="numpy", label="Image", elem_classes="input-image-container")  # always present, sometimes hidden
            start_frame = gr.Image(sources='upload', type="numpy", label="Start Frame (Optional)", elem_classes="keyframe-image-container", visible=False)
            with gr.Group(visible=False) as keyframes_options:
                    keyframe_weight = gr.Slider(label="Start Frame Influence", minimum=0.0, maximum=1.0, value=0.7, step=0.1, info="Higher values prioritize start frame characteristics (0 = end frame only, 1 = start frame only)")
            end_frame = gr.Image(sources='upload', type="numpy", label="End Frame (Required)", elem_classes="keyframe-image-container", visible=False)
            with gr.Group(visible=False) as video_extension_options:
                input_video = gr.Video(
                    label="Upload Video to Extend", 
                    format="mp4",
                    height=320,
                    elem_classes="video-container"
                )
                extension_direction = gr.Radio(
                    ["Forward", "Backward"], 
                    label="Extension Direction",
                    value="Forward",
                    info="Forward extends the end, Backward extends the beginning"
                )
                extension_frames = gr.Slider(
                    label="Context Frames", 
                    minimum=1, 
                    maximum=16, 
                    value=0, 
                    step=1,
                    info="Number of frames to extract from video for continuity"
                )
            aspect_selector = gr.Dropdown(
                ["16:9", "9:16", "1:1", "4:5", "3:2", "2:3", "21:9", "4:3", "Custom..."],
                label="Aspect Ratio",
                value="1:1",
                visible=False
            )
            custom_w = gr.Number(label="Width", value=768, visible=False)
            custom_h = gr.Number(label="Height", value=768, visible=False)
            prompt = gr.Textbox(label="Prompt", value='', lines=3)
            with gr.Accordion("Negative Prompt", open=False):
                n_prompt = gr.Textbox(
                    label="Negative Prompt - Requires CFG higher than 1.0 to take effect", 
                    value="", 
                    lines=2
                )
            
            # Realtime calculation display
            video_stats = gr.HTML(
                value="<div class='stats-box'>Estimated video length: calculating...</div>",
                label="Approximate Output Length"
            )
            
            latent_window_size = gr.Slider(
                label="Latent Window Size", 
                minimum=2, 
                maximum=33, 
                value=9, 
                step=1
            )
            
            segment_count = gr.Slider(
                label="Number of Segments", 
                minimum=1, 
                maximum=50, 
                value=5, 
                step=1,
                info="More segments = longer video"
            )
            
            overlap_slider = gr.Slider(
                label="Frame Overlap",
                minimum=0,
                maximum=33,
                value=8,
                step=1,
                info="Controls how many frames overlap between sections"
            )
            
            trim_percentage = gr.Slider(
                label="Segment Trim Percentage",
                minimum=0.0,
                maximum=1.0,
                value=0.2,
                step=0.01,
                info="Percentage of frames to trim (0.0 = keep all, 1.0 = maximum trim)"
            )
            
            # -- Add encoder weight controls --
            with gr.Accordion("Advanced Model Parameters", open=False):
                llm_encoder_weight = gr.Slider(
                    label="LLM Encoder Weight", 
                    minimum=0.0, 
                    maximum=5.0, 
                    value=1.0, 
                    step=0.1,
                    info="0.0 to disable LLM encoder"
                )
                
                clip_encoder_weight = gr.Slider(
                    label="CLIP Encoder Weight", 
                    minimum=0.0, 
                    maximum=5.0, 
                    value=1.0, 
                    step=0.1,
                    info="0.0 to disable CLIP encoder"
                )
                
                clean_latent_weight = gr.Slider(
                    label="Clean Latent Weight", 
                    minimum=0.0, 
                    maximum=2.0, 
                    value=1.0, 
                    step=0.01,
                    info="Controls influence of anchor/initial frame"
                )
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, info="Must be >1.0 for negative prompts to work")
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                rs = gr.Slider(
                    label="CFG Re-Scale", 
                    minimum=0.0, 
                    maximum=1.0, 
                    value=0.0, 
                    step=0.01
                )
            
            # -- Add Gaussian blur control --
            gaussian_blur = gr.Slider(
                label="Gaussian Blur", 
                minimum=0.0, 
                maximum=1.0, 
                value=0.0, 
                step=0.01, 
                visible=False,
                info="Apply blur to input images before processing"
            )
            init_color = gr.ColorPicker(label="Initial Frame Color", value="#808080", visible=False)
            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True)
                seed = gr.Number(label="Seed", value=random.randint(0, 2**32-1), precision=0)
                lock_seed = gr.Checkbox(label="Lock Seed", value=False)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB)", minimum=6, maximum=128, value=6, step=0.1)
        with gr.Column(scale=2):
            progress_bar = gr.HTML(visible=False)  # Start hidden
            progress_desc = gr.Markdown(visible=False)
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, elem_classes="result-container", loop=True)
            result_image_html = gr.Image(label='Single Frame Image', elem_classes="result-container", visible=False)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
           

            

    # --- calllbacks ---
    def update_frame_dropdown(window):
        stops = get_valid_frame_stops(window)
        # Only let user select from valid numbers of frames
        if stops:
            return gr.update(choices=[str(x) for x in stops], value=str(stops[0]))
        else:
            return gr.update(choices=[''], value='')


    # Add new function to update overlap slider maximum
    def update_overlap_slider(window_size):
        """Update maximum overlap based on window size"""
        # Convert input to value if it's a Gradio component
        window_size_val = window_size if not hasattr(window_size, 'value') else window_size.value
        window_size_val = float(window_size_val)
        
        # Calculate max overlap
        frames_per_section = int(window_size_val * 4 - 3)
        max_overlap = max(0, frames_per_section - 1)
        
        return gr.update(maximum=max_overlap)
    
    # Add function to calculate and display video stats
    def update_video_stats(window_size, segments, overlap):
        """Calculate and format video statistics based on current settings"""
        # Convert inputs to values if they're Gradio components
        window_size_val = window_size if not hasattr(window_size, 'value') else window_size.value
        segments_val = segments if not hasattr(segments, 'value') else segments.value
        overlap_val = overlap if not hasattr(overlap, 'value') else overlap.value
        
        # Ensure we're working with numbers
        window_size_val = float(window_size_val)
        segments_val = float(segments_val)
        overlap_val = float(overlap_val)
        
        # Calculate frames
        frames_per_section = int(window_size_val * 4 - 3)
        max_overlap = max(0, frames_per_section - 1)
        actual_overlap = min(overlap_val, max_overlap)
        effective_frames = frames_per_section - actual_overlap
        total_frames = segments_val * effective_frames
        
        # Calculate time (at 30fps)
        seconds = total_frames / 30.0
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        
        # Format the output
        stats_html = f"""
        <div class="stats-box">
            <table>
                <tr>
                    <td><b>Frames per segment:</b></td>
                    <td>{int(effective_frames)} frames</td>
                </tr>
                <tr>
                    <td><b>Total frames:</b></td>
                    <td>{int(total_frames)} frames</td>
                </tr>
                <tr>
                    <td><b>Video length:</b></td>
                    <td>{minutes}m {remaining_seconds:.1f}s (at 30fps)</td>
                </tr>
            </table>
        </div>
        """
        return stats_html
    
    # Connect callbacks
    latent_window_size.change(
        update_overlap_slider,
        inputs=[latent_window_size],
        outputs=[overlap_slider]
    )
    
    latent_window_size.change(
        update_video_stats,
        inputs=[latent_window_size, segment_count, overlap_slider],
        outputs=[video_stats]
    )
    
    segment_count.change(
        update_video_stats,
        inputs=[latent_window_size, segment_count, overlap_slider],
        outputs=[video_stats]
    )
    
    overlap_slider.change(
        update_video_stats,
        inputs=[latent_window_size, segment_count, overlap_slider],
        outputs=[video_stats]
    )
        
    def switch_mode(mode):
        is_img2vid = mode == "image2video"
        is_txt2vid = mode == "text2video"
        is_keyframes = mode == "keyframes"
        is_video_ext = mode == "video_extension"
        
        # Show blur for img2vid, keyframes, and extend video
        show_blur = is_img2vid or is_keyframes or is_video_ext
        
        return (
            gr.update(visible=is_img2vid),  # input_image
            gr.update(visible=is_keyframes),  # start_frame
            gr.update(visible=is_keyframes),  # end_frame
            gr.update(visible=is_txt2vid),  # aspect_selector
            gr.update(visible=(is_txt2vid and aspect_selector.value == "Custom...")),  # custom_w
            gr.update(visible=(is_txt2vid and aspect_selector.value == "Custom...")),  # custom_h
            gr.update(visible=is_keyframes),  # keyframes_options
            gr.update(visible=is_video_ext),  # video_extension_options
            gr.update(visible=show_blur)  # gaussian_blur
        )
    def show_custom(aspect):
        show = aspect == "Custom..."
        return gr.update(visible=show), gr.update(visible=show)
    
    
    latent_window_size.change(
        update_overlap_slider,
        inputs=[latent_window_size],
        outputs=[overlap_slider]
    )
    
    latent_window_size.change(
        update_video_stats,
        inputs=[latent_window_size, segment_count, overlap_slider],
        outputs=[video_stats]
    )
    
    segment_count.change(
        update_video_stats,
        inputs=[latent_window_size, segment_count, overlap_slider],
        outputs=[video_stats]
    )
    
    overlap_slider.change(
        update_video_stats,
        inputs=[latent_window_size, segment_count, overlap_slider],
        outputs=[video_stats]
    )
    mode_selector.change(
        switch_mode,
        inputs=[mode_selector],
        outputs=[
            input_image,
            start_frame,
            end_frame,
            aspect_selector,
            custom_w,
            custom_h,
            keyframes_options,
            video_extension_options,
            gaussian_blur  # Add this output
        ]
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
        input_image,
        start_frame,
        end_frame,
        aspect_selector,
        custom_w,
        custom_h,
        prompt,
        n_prompt,
        seed,
        # No advanced_mode
        latent_window_size,
        segment_count,  # Replace adv_seconds
        # No total_frames_dropdown
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
        extension_frames,
        # Add new parameters:
        overlap_slider,
        trim_percentage,
        gaussian_blur,
        llm_encoder_weight,
        clip_encoder_weight,
        clean_latent_weight,
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
