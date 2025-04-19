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
def worker(mode, input_image, input_video,
           prompt, n_prompt, sampler, shift, cfg, gs, rs,
           strength, seed, total_second_length, latent_window_size,
           steps, gpu_memory_preservation, use_teacache):

    job_id = generate_timestamp()
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting job...'))))
    print(f"Starting worker job {job_id} with mode: {mode}")

    try:
        # --- Clean GPU and Load Text Encoders ---
        # (Keep this section as is)
        if not high_vram:
            unload_complete_models(*all_models)
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding...'))))

        # --- Prepare prompt embeddings ---
        # (Keep this section as is)
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

        # --- Load VAE and Image Encoder (conditionally) ---
        if not high_vram:
            if mode in ['txt2vid', 'img2vid', 'vid2vid', 'extend_vid', 'video_inpaint']:
                load_model_as_complete(vae, target_device=gpu)
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE ready...'))))
            if mode == 'img2vid':
                load_model_as_complete(image_encoder, target_device=gpu)
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image Encoder ready...'))))

        # --- Mode Specific Setup ---
        if mode == 'txt2vid':
            # Uses initial_latent + strength method (effectively starts from noise)
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Preparing for txt2vid...'))))
            height, width = 512, 512 # Or make configurable
            dummy = torch.zeros((1, 3, 1, height, width), device=gpu, dtype=vae.dtype)
            # Need to handle potential VAE dtype issues even for dummy encode if VAE is float16
            original_vae_dtype = vae.dtype
            try:
                vae.to(torch.float32) # Temporarily change VAE
                init_latent_shape = vae_encode(dummy.to(torch.float32), vae).shape
            finally:
                vae.to(original_vae_dtype) # Change back
            # The actual initial_latent passed to sampler is zeros, sampler mixes noise based on strength
            init_latent = torch.zeros(init_latent_shape, device=gpu, dtype=transformer.dtype)
            # No specific first frame conditioning needed
            print(f"Txt2Vid: Initial latent shape for sampler {init_latent.shape}")

        elif mode == 'img2vid':
            # Uses conditioning method (like simple demo), strength is ignored for start noise
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Preparing for img2vid...'))))
            if input_image is None: raise ValueError("Input image required for img2vid.")
            np_img = np.array(input_image)
            H, W, _ = np_img.shape
            height, width = find_nearest_bucket(H, W, resolution=640)
            img_resized = resize_and_center_crop(np_img, target_width=width, target_height=height)
            print(f"Img2Vid: Resized/bucketed image to {width}x{height}")

            # VAE Encode first frame for conditioning (using float32 fix)
            original_vae_dtype = vae.dtype
            try:
                vae.to(dtype=torch.float32)
                inp_for_vae = (torch.from_numpy(img_resized).float() / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device=gpu)
                first_frame_conditioning_latent = vae_encode(inp_for_vae, vae).to(transformer.dtype)
            finally:
                vae.to(dtype=original_vae_dtype)
            print(f"Img2Vid: First frame latent for conditioning: {first_frame_conditioning_latent.shape}, dtype {first_frame_conditioning_latent.dtype}")

            # Image Embeddings
            if not high_vram: load_model_as_complete(image_encoder, target_device=gpu)
            image_encoder_output = hf_clip_vision_encode(img_resized, feature_extractor, image_encoder)
            image_embeddings = image_encoder_output.last_hidden_state.to(transformer.dtype)
            print(f"Img2Vid: Image embedding shape {image_embeddings.shape}")

            # IMPORTANT: No initial_latent passed to sampler for this mode
            init_latent = None
            concat_latent = None

        elif mode == 'extend_vid':
            # Uses conditioning method (like simple demo), strength ignored for start noise
            # Needs conditioning from first frame AND whole video (for concat)
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Preparing for {mode}...'))))
            if input_video is None or not hasattr(input_video, 'name'): raise ValueError("Input video file required.")
            video_path = input_video.name
            print(f"Video Mode: Loading video {video_path}")
            vid_frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
            vid = vid_frames.permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0 # B, C, T, H, W
            _, _, T, H, W = vid.shape
            height, width = H, W
            print(f"Video Mode: Loaded video with {T} frames, {W}x{H}")

            original_vae_dtype = vae.dtype
            try:
                # Temporarily set VAE to float32 for ALL encodes in this block
                vae.to(dtype=torch.float32)

                # VAE Encode first frame for initial conditioning
                first_frame_np = vid_frames[0].numpy() # Use original frame data
                inp_first_frame = (torch.from_numpy(first_frame_np).float() / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device=gpu)
                first_frame_conditioning_latent = vae_encode(inp_first_frame, vae).to(transformer.dtype)
                print(f"ExtendVid: First frame latent for conditioning: {first_frame_conditioning_latent.shape}, dtype {first_frame_conditioning_latent.dtype}")

                # VAE Encode whole video for concat_latent conditioning
                print(f"ExtendVid: Encoding full video for concat_latent...")
                concat_latent = vae_encode(vid.to(device=gpu), vae).to(transformer.dtype)
                print(f"ExtendVid: Conditioning concat_latent shape {concat_latent.shape}")

            finally:
                vae.to(dtype=original_vae_dtype) # Restore VAE dtype

            # IMPORTANT: No initial_latent passed to sampler for this mode
            init_latent = None
            # No image embeddings needed
            image_embeddings = None

        elif mode == 'vid2vid':
            # Uses initial_latent + strength method (like txt2vid but starting from video frame)
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Preparing for {mode}...'))))
            if input_video is None or not hasattr(input_video, 'name'): raise ValueError("Input video file required.")
            video_path = input_video.name
            print(f"Video Mode: Loading video {video_path}")
            vid_frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
            vid = vid_frames.permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0 # B, C, T, H, W
            _, _, T, H, W = vid.shape
            height, width = H, W
            print(f"Video Mode: Loaded video with {T} frames, {W}x{H}")

            original_vae_dtype = vae.dtype
            try:
                vae.to(dtype=torch.float32) # Use float32 for safety

                # VAE Encode first frame to use as initial_latent for the sampler
                first_frame_np = vid_frames[0].numpy()
                inp_first_frame = (torch.from_numpy(first_frame_np).float() / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device=gpu)
                # This IS the initial latent passed to the sampler
                init_latent = vae_encode(inp_first_frame, vae).to(transformer.dtype)
                print(f"Vid2Vid: Initial latent for sampler (from first frame): {init_latent.shape}, dtype {init_latent.dtype}")
                # Also store for clean_latents conditioning on first step
                first_frame_conditioning_latent = init_latent

                # VAE Encode whole video for concat_latent conditioning
                print(f"Vid2Vid: Encoding full video for concat_latent...")
                concat_latent = vae_encode(vid.to(device=gpu), vae).to(transformer.dtype)
                print(f"Vid2Vid: Conditioning concat_latent shape {concat_latent.shape}")

            finally:
                vae.to(dtype=original_vae_dtype)

            # No image embeddings needed
            image_embeddings = None

        # Fallback sizing
        if height is None or width is None: height, width = 512, 512; print(f"Warning: Using fallback size {width}x{height}")

        # --- Sampling loop ---
        # (Calculation of total_latent_sections, rnd, num_frames_per_window, history_latents, paddings is unchanged)
        total_latent_sections = max(int(round((total_second_length * 30) / (latent_window_size * 4))), 1)
        stream.output_queue.push(('progress', (None, f'Preparing for {total_latent_sections} sections...', make_progress_bar_html(0, 'Initializing Sampler...'))))
        rnd = torch.Generator("cpu").manual_seed(int(seed))
        num_frames_per_window = latent_window_size * 4 - 3
        history_latents = torch.zeros((1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0
        if total_latent_sections > 4: latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
        else: latent_paddings = list(reversed(range(total_latent_sections)))
        print(f"Latent padding sequence: {latent_paddings}")

        # --- Main Generation Loop ---
        for section_index, latent_padding in enumerate(latent_paddings):
            is_last_section = (latent_padding == 0)
            latent_padding_size = latent_padding * latent_window_size

            print(f"\n--- Section {section_index + 1}/{total_latent_sections} (Padding: {latent_padding}, Last: {is_last_section}) ---")

            if stream.input_queue.top() == 'end': print("User requested stop."); stream.output_queue.push(('end', None)); return

            # Prepare clean latents for conditioning
            # Use specific first frame latent for step 0 in img2vid/extend_vid/vid2vid
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, _, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = \
                indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            # Determine the 'clean' reference latent for the *conditioning* input
            if section_index == 0:
                if mode in ['img2vid', 'extend_vid', 'vid2vid']:
                    # Use the specially prepared VAE latent of the first frame for conditioning
                    clean_latents_pre_cond = first_frame_conditioning_latent.to(history_latents.device, dtype=history_latents.dtype)
                    print("Using first frame VAE latent for clean_latents_pre conditioning.")
                elif mode == 'txt2vid':
                    # txt2vid uses the passed init_latent (zeros) for conditioning ref too
                    clean_latents_pre_cond = init_latent.to(history_latents.device, dtype=history_latents.dtype)
                    print("Using zero init_latent for clean_latents_pre conditioning.")
                else: # Fallback? Should not happen
                    clean_latents_pre_cond = history_latents[:, :, :1]
            else: # Subsequent steps use history
                clean_latents_pre_cond = history_latents[:, :, :1]
                print("Using history[:1] for clean_latents_pre conditioning.")

            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16].split([1, 2, 16], dim=2)
            clean_latents_for_cond = torch.cat([clean_latents_pre_cond, clean_latents_post], dim=2) # Rename for clarity

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
            print(f"Calling sample_hunyuan with: mode={mode}, sampler={sampler}, shift={shift}, strength={strength if mode in ['txt2vid', 'vid2vid'] else 'N/A'}, cfg={cfg}, gs={gs}, rs={rs}")
            print(f"  initial_latent provided: {'Yes' if init_latent is not None else 'No'}")
            print(f"  concat_latent provided: {'Yes' if concat_latent is not None else 'No'}")
            print(f"  image_embeddings provided: {'Yes' if image_embeddings is not None else 'No'}")

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler=sampler,
                initial_latent=init_latent, # Will be None for img2vid, extend_vid
                concat_latent=concat_latent, # Will be None for txt2vid, img2vid
                strength=strength, # Only relevant if initial_latent is not None
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
                image_embeddings=image_embeddings, # Only for img2vid
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
            if is_last_section and mode == 'img2vid':
                 if first_frame_conditioning_latent is not None:
                     print("Prepending clean first frame latent for img2vid final output.")
                     # Ensure dtype and device match
                     generated_latents = torch.cat([first_frame_conditioning_latent.to(generated_latents.device, dtype=generated_latents.dtype), generated_latents], dim=2)
                 else:
                      print("Warning: Cannot prepend first frame latent for img2vid, as it's missing.")
            # Prepending initial latent for txt2vid/vid2vid (if initial_latent was used)
            elif is_last_section and init_latent is not None and mode in ['txt2vid', 'vid2vid']:
                 print("Prepending initial latent to the final output (txt2vid/vid2vid).")
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
            if history_pixels is None:
                print("Decoding first section directly.")
                history_pixels = vae_decode(real_history_latents_gpu, vae).cpu()
            else:
                print("Decoding current section slice and appending.")
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlap_pixel_frames = latent_window_size * 4 - 3
                print(f"Decoding slice of size {section_latent_frames} latents for soft append.")
                slice_to_decode = real_history_latents_gpu[:, :, :min(section_latent_frames, real_history_latents_gpu.shape[2])]
                current_pixels = vae_decode(slice_to_decode, vae).cpu()
                print(f"Soft appending with overlap: {overlap_pixel_frames} pixel frames.")
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlap=overlap_pixel_frames)
            print(f"Current history_pixels shape: {history_pixels.shape}")
            if not high_vram: unload_complete_models(vae)


            # --- Save Intermediate/Final Video ---
            output_filename = os.path.join(outputs_folder, f'{job_id}_section_{section_index+1}.mp4')
            print(f"Saving video to {output_filename} with {history_pixels.shape[2]} frames.")
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30)
            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                print("Last section processed.")
                break

        print(f"Worker job {job_id} finished.")

    # --- Error Handling & Finally Block ---               
    except Exception as e:
        print(f"!!!!!!!!!! Error in worker job {job_id} !!!!!!!!!!"); traceback.print_exc()
        stream.output_queue.push(('error', str(e)))
    finally:
        if not high_vram: unload_complete_models(*all_models); print("Models unloaded (low VRAM mode).")
        stream.output_queue.push(('end', None))
        print(f"Worker job {job_id} cleanup complete.")


# -------- Gradio UI --------
def process_fn(mode, img, vid, # Removed mask
               prompt, n_prompt, sampler, shift, cfg, gs, rs,
               strength, seed_from_ui, lock_seed_val, # Renamed seed -> seed_from_ui, added lock_seed_val
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
    yield None, gr.update(visible=False, value=None), gr.update(value=''), gr.update(value=''), gr.update(interactive=False), gr.update(interactive=True), gr.update(value=actual_seed)

    # Clear previous stream queues if any
    stream.input_queue = AsyncStream().input_queue # Reset input queue
    stream.output_queue = AsyncStream().output_queue # Reset output queue

    # Launch worker thread - PASS THE DETERMINED actual_seed
    print("Starting process_fn...")
    async_run(worker, mode, img, vid, # Removed mask
              prompt, n_prompt, sampler, shift, cfg, gs, rs,
              strength, actual_seed, # Pass actual_seed here
              seconds, window, steps, gpu_mem, tea)

    # Handle outputs from worker thread
    output_video_path = None
    last_preview = None
    while True:
        try:
            flag, data = stream.output_queue.next() # Removed timeout

            if flag == 'file':
                output_video_path = data
                # Update includes None for the seed output, as it's already updated
                yield output_video_path, gr.update(value=last_preview, visible=last_preview is not None), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update() # Keep seed field as is
            elif flag == 'progress':
                preview_img, status_text, html_bar = data
                last_preview = preview_img # Store the latest preview
                # Update includes None for the seed output
                yield output_video_path, gr.update(value=preview_img, visible=preview_img is not None), status_text, html_bar, gr.update(interactive=False), gr.update(interactive=True), gr.update() # Keep seed field as is
            elif flag == 'error':
                 error_message = data
                 print(f"Gradio UI received error: {error_message}")
                 # Update includes None for the seed output
                 yield output_video_path, gr.update(value=last_preview, visible=last_preview is not None), f"Error: {error_message}", '', gr.update(interactive=True), gr.update(interactive=False), gr.update() # Keep seed field as is
                 break # Stop processing on error
            elif flag == 'end':
                print("Gradio UI received end signal.")
                # Final update: show final video, hide preview, clear status/bar
                # Update includes None for the seed output
                yield output_video_path, gr.update(visible=False, value=None), '', '', gr.update(interactive=True), gr.update(interactive=False), gr.update() # Keep seed field as is
                break # Exit loop
            else:
                 print(f"Received unexpected flag: {flag}")

        except Exception as e:
            if "FIFOQueue object" in str(e) and "'next' of" in str(e): # More specific check if needed
                print("Waiting for worker output...")
                # Update includes None for the seed output
                yield output_video_path, gr.update(value=last_preview, visible=last_preview is not None), "Processing, please wait...", gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update() # Keep seed field as is
                continue
            else:
                 print(f"Error processing output queue: {e}")
                 traceback.print_exc()
                 # Update includes None for the seed output
                 yield output_video_path, gr.update(value=last_preview, visible=last_preview is not None), f"UI Error: {e}", '', gr.update(interactive=True), gr.update(interactive=False), gr.update() # Keep seed field as is
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
                ["txt2vid", "img2vid", "vid2vid", "extend_vid"], # Removed video_inpaint
                value="img2vid",
                label="Mode"
            )
            input_image = gr.Image(
                sources='upload',
                type="numpy",
                label="Input Image (for img2vid)",
                visible=True, # Initially visible, controlled by mode change
                height=400 # <<< SET DISPLAY HEIGHT
            )
            input_video = gr.Video(
                label="Input Video (for vid2vid, extend_vid)",
                sources='upload',
                visible=False # Initially hidden
            )
            prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Enter your prompt here...")
            n_prompt = gr.Textbox(label="Negative Prompt", lines=2, value="ugly, blurry, deformed, text, watermark, signature")

            # --- Moved Seed and Length outside Advanced ---
            seconds = gr.Slider(1, 120, value=5, step=0.1, label="Output Video Length (sec)")
            with gr.Row():
                seed = gr.Number(label="Seed", value=random.randint(0, 2**32 - 1), precision=0) # Start with random seed
                lock_seed = gr.Checkbox(label="Lock Seed", value=False) # <<< ADDED Lock Seed checkbox

            with gr.Accordion("Advanced Settings", open=False):
                sampler = gr.Dropdown(
                    ["unipc", "unipc_bh2", "dpmpp_2m", "dpmpp_sde", "dpmpp_2m_sde", "dpmpp_3m_sde", "ddim", "plms", "euler", "euler_ancestral"],
                    value="unipc",
                    label="Sampler"
                )
                shift = gr.Slider(0., 10., value=3.0, label="Shift μ (Temporal Consistency)", info="Higher values might increase consistency but affect motion. Default 3.0.")
                cfg = gr.Slider(1., 32., value=1.0, label="CFG Scale (Prompt Guidance)", info="Only effective if > 1.0. FramePack default is 1.0.")
                gs = gr.Slider(1., 32., value=10.0, label="Distilled CFG Scale", info="Main guidance scale for FramePack. Default 10.0.")
                rs = gr.Slider(0., 1., value=0., label="Rescale Guidance", info="Guidance rescale factor. Default 0.0.")
                strength = gr.Slider(0., 1., value=1.0, label="Denoise Strength (for img2vid/vid2vid)", info="How much to change the input image/video. 1.0 = max change. Default 0.7.")
                # seed = gr.Number(label="Seed", value=31337, precision=0) # --- MOVED ---
                # seconds = gr.Slider(1, 120, value=5, step=0.1, label="Output Video Length (sec)") # --- MOVED ---
                window = gr.Slider(1, 33, value=9, step=1, label="Latent Window Size", info="Affects temporal range per step. Default 9.")
                steps = gr.Slider(1, 100, value=25, step=1, label="Sampling Steps", info="Number of denoising steps. Default 25.")
                if not high_vram:
                    gpu_mem = gr.Slider(3, max(10, int(free_mem_gb)), value=min(6, int(free_mem_gb)-2), step=0.1, label="GPU Memory to Preserve (GB)", info="Leave this much VRAM free for other apps. Higher = Slower.")
                else:
                     gpu_mem = gr.Number(label="GPU Memory Preservation (N/A in High VRAM mode)", value=0, interactive=False)
                tea = gr.Checkbox(label="Use TeaCache Optimization", value=False, info="May speed up sampling but can slightly affect quality (e.g., hands).") # <<< CHANGED DEFAULT

            with gr.Row():
                 start_btn = gr.Button("Generate", variant="primary")
                 end_btn = gr.Button("Stop", interactive=False)

        with gr.Column(scale=1):
            result_vid = gr.Video(label="Output Video", interactive=False, height=512, autoplay=True, loop=True)
            preview = gr.Image(label="Live Preview", interactive=False, visible=False, height=256)
            status_md = gr.Markdown("") # For text status updates
            bar_html = gr.HTML("") # For progress bar

    # --- UI Control Logic ---
    def update_ui_for_mode(selected_mode):
        is_img_mode = selected_mode == "img2vid"
        is_vid_mode = selected_mode in ["vid2vid", "extend_vid"]
        is_strength_relevant = selected_mode in ["vid2vid"]

        return {
            input_image: gr.update(visible=is_img_mode),
            input_video: gr.update(visible=is_vid_mode),
            strength: gr.update(interactive=is_strength_relevant)
        }

    mode.change(
        update_ui_for_mode,
        inputs=[mode],
        outputs=[input_image, input_video, strength], # Removed mask_image
        queue=False
    )

    # --- Button Actions ---
    # ADD lock_seed to inputs list
    inputs = [mode, input_image, input_video,
              prompt, n_prompt, sampler, shift, cfg, gs, rs,
              strength, seed, lock_seed, seconds, window, steps, gpu_mem, tea] # Added lock_seed

    # ADD seed to outputs list (so we can update it)
    outputs = [result_vid, preview, status_md, bar_html, start_btn, end_btn, seed] # Added seed

    start_btn.click(process_fn, inputs=inputs, outputs=outputs) # Pass updated lists
    end_btn.click(end_process_early, outputs=[end_btn])

# --- Launch App ---
print(f"Launching Gradio app on {args.host}:{args.port}")
demo.launch(
    server_name=args.host, # Use args.host
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser
)
