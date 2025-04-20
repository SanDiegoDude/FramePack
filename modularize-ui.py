from diffusers_helper.hf_login import login
import os
import time
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
import gradio as gr
import torch
import traceback
import einops
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

def get_valid_frame_stops(latent_window_size, max_seconds=120, fps=30):
    frames_per_section = latent_window_size * 4 - 3
    max_sections = int((max_seconds * fps) // frames_per_section)
    stops = [frames_per_section * i for i in range(1, max_sections + 1)]
    return stops

# ==== (model/stream/load code same as before) ====

# ---- Worker Utility Split ----
def prepare_inputs(input_image, prompt, n_prompt, cfg):
    global high_vram
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
    H, W, C = input_image.shape
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

@torch.no_grad()
def worker(
    mode, input_image, aspect, custom_w, custom_h,
    prompt, n_prompt, seed,
    use_adv, adv_window, adv_seconds, selected_frames,
    steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, display_final_frame
):
    t_start = time.time()             # <--- Add this as the FIRST statement
    history_pixels = None             # <--- Init for safety in finally
    # -- deterministic output choice --
    if use_adv:
        latent_window_size = adv_window
        frames_per_section = latent_window_size * 4 - 3
        total_frames = int(round(adv_seconds * 30))
    else:
        latent_window_size = 9
        frames_per_section = latent_window_size * 4 - 3
        total_frames = int(selected_frames)
    extra_frames = frames_per_section if mode == "text2video" else 0
    run_frames = total_frames + extra_frames
    total_sections = math.ceil((run_frames + 3) / frames_per_section)
    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))
    try:
        if mode == "text2video":
            width, height = get_dims_from_aspect(aspect, custom_w, custom_h)
            input_image_arr = np.zeros((height, width, 3), dtype=np.uint8)
            input_image = input_image_arr
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        inp_np, inp_tensor, lv, cp, lv_n, cp_n, m, m_n, height, width = prepare_inputs(input_image, prompt, n_prompt, cfg)
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)
        start_latent = vae_encode(inp_tensor, vae)
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)
        clip_output = hf_clip_vision_encode(inp_np, feature_extractor, image_encoder).last_hidden_state
        lv = lv.to(transformer.dtype)
        lv_n = lv_n.to(transformer.dtype)
        cp = cp.to(transformer.dtype)
        cp_n = cp_n.to(transformer.dtype)
        clip_output = clip_output.to(transformer.dtype)
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
        rnd = torch.Generator("cpu").manual_seed(seed)
        history_latents = torch.zeros(
            size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32
        ).cpu()
        history_pixels = None
        first_patch_pixels = None
        t_start = time.time()
        total_generated_latent_frames = 0
        for section in reversed(range(total_sections)):
            is_last_section = section == 0
            latent_padding_size = section * latent_window_size
            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
            transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=steps if use_teacache else 0)
            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)
                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')
                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {total_generated_latent_frames}, Video length: {total_generated_latent_frames / 30.0:.2f} seconds (FPS-30).'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
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
                callback=callback,
            )
            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
            decoded = vae_decode(real_history_latents, vae).cpu()
            if history_pixels is None:
                history_pixels = decoded
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = frames_per_section
                current_pixels = decoded[:, :, :section_latent_frames, :, :]
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
            # Capture final frame for user if requested (the last frame of this patch/section)
            if display_final_frame and first_patch_pixels is None:
                # decoded: (1, 3, T, H, W)
                first_patch_pixels = decoded[:, :, -1, :, :].squeeze().permute(1, 2, 0).clamp(0, 1).numpy()
                # Save image to file so the UI can serve (so it's clickable)
                final_path = os.path.join(outputs_folder, f"{job_id}_finalframe.png")
                Image.fromarray((first_patch_pixels*255).astype(np.uint8)).save(final_path)
                final_frame_img = final_path
            else:
                final_frame_img = None
            if not high_vram:
                unload_complete_models()
            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
            if mode == "text2video" and is_last_section:
                if history_pixels.shape[2] > extra_frames + total_frames:
                    history_pixels = history_pixels[:, :, extra_frames:extra_frames+total_frames, :, :]
                else:
                    history_pixels = history_pixels[:, :, extra_frames:, :, :]
            elif is_last_section:
                if history_pixels.shape[2] > total_frames:
                    history_pixels = history_pixels[:, :, :total_frames, :, :]
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30)
            stream.output_queue.push(('file', output_filename))
            if is_last_section:
                break
    except Exception:
        traceback.print_exc()
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        return
    finally:
        t_end = time.time()
        total_time = t_end - t_start
        trimmed_frames = history_pixels.shape[2] if history_pixels is not None else 0
        video_seconds = trimmed_frames / 30.0
        summary_string = (
            f"Finished!\n"
            f"Total generated frames: {trimmed_frames}, "
            f"Video length: {video_seconds:.2f} seconds (FPS-30), "
            f"Time taken: {total_time:.2f}s."
        )
        # Final frame served here for outside loop if none earlier
        ffimg = None
        if display_final_frame and first_patch_pixels is not None:
            ffimg = final_path if os.path.exists(final_path) else None
        stream.output_queue.push(('progress', (None, summary_string, "", ffimg)))
        stream.output_queue.push(('end', None))

def process(
    mode, input_image, aspect_selector, custom_w, custom_h,
    prompt, n_prompt, seed,
    use_adv, adv_window, adv_seconds, selected_frames,
    steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, lock_seed, display_final_frame
):
    global stream
    assert mode in ['image2video', 'text2video'], "Invalid mode"
    if not lock_seed:
        seed = int(time.time()) % 2**32
    yield (
        None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update(value=seed), None
    )
    stream = AsyncStream()
    async_run(
        worker,
        mode,
        input_image,
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
        use_teacache, display_final_frame
    )
    output_filename = None
    last_desc = ""
    final_frame_src = None
    while True:
        flag, data = stream.output_queue.next()
        if flag == 'file':
            output_filename = data
            yield (
                gr.update(value=output_filename), gr.update(), gr.update(), gr.update(),
                gr.update(interactive=False), gr.update(interactive=True), gr.update(), final_frame_src
            )
        elif flag == 'progress':
            # progress now can have 4 items: preview, desc, html, final_frame_img
            if len(data) == 4:
                preview, desc, html, final_frame_img = data
            else:
                preview, desc, html = data
                final_frame_img = None
            if desc:
                last_desc = desc
            if final_frame_img is not None:
                final_frame_src = final_frame_img
            yield (
                gr.update(), gr.update(visible=True, value=preview), desc, html,
                gr.update(interactive=False), gr.update(interactive=True), gr.update(), final_frame_src
            )
        elif flag == 'end':
            yield (
                gr.update(value=output_filename), gr.update(visible=False),
                gr.update(value=last_desc), '',
                gr.update(interactive=True), gr.update(interactive=False), gr.update(), final_frame_src
            )
            break

def end_process():
    stream.input_queue.push('end')

css = """
.gr-box, .gr-image, .gr-video {
    border: 2px solid orange !important;
    border-radius: 8px !important;
    margin-bottom: 16px;
    background: #222 !important;
}
"""

block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column(scale=2):
            mode_selector = gr.Radio(
                ["image2video", "text2video"], value="image2video", label="Mode"
            )
            input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
            aspect_selector = gr.Dropdown(
                ["16:9", "9:16", "1:1", "4:5", "3:2", "2:3", "21:9", "4:3", "Custom..."],
                label="Aspect Ratio",
                value="1:1",
                visible=False
            )
            custom_w = gr.Number(label="Width", value=768, visible=False)
            custom_h = gr.Number(label="Height", value=768, visible=False)
            prompt = gr.Textbox(label="Prompt", value='')
            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)
            advanced_mode = gr.Checkbox(label="Advanced Mode", value=False)
            display_final_frame = gr.Checkbox(label="Display Final Frame", value=False)
            latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
            adv_seconds = gr.Slider(label="Video Length (Seconds)", minimum=0.1, maximum=120.0, value=5.0, step=0.1, visible=False)
            total_frames_dropdown = gr.Dropdown(
                label="Output Video Frames",
                choices=[str(x) for x in get_valid_frame_stops(9)],
                value=str(get_valid_frame_stops(9)[0]),
                visible=True
            )
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
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            final_frame_out = gr.Image(label="Final Frame", visible=False, interactive=False, type="filepath")

    def update_frame_dropdown(window):
        stops = get_valid_frame_stops(window)
        if stops:
            return gr.update(choices=[str(x) for x in stops], value=str(stops[0]))
        else:
            return gr.update(choices=[''], value='')
    def show_hide_advanced(show):
        return (
            gr.update(visible=show),    # latent_window_size
            gr.update(visible=show),    # adv_seconds
            gr.update(visible=not show),# total_frames_dropdown
        )
    def switch_mode(mode):
        return (
            gr.update(visible=mode=="image2video"),
            gr.update(visible=mode=="text2video"),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    def show_custom(aspect):
        show = aspect == "Custom..."
        return gr.update(visible=show), gr.update(visible=show)

    def show_final_frame(checked):
        return gr.update(visible=checked)

    display_final_frame.change(show_final_frame, [display_final_frame], [final_frame_out])
    latent_window_size.change(update_frame_dropdown, inputs=[latent_window_size], outputs=[total_frames_dropdown])
    advanced_mode.change(show_hide_advanced, inputs=[advanced_mode], outputs=[latent_window_size, adv_seconds, total_frames_dropdown])
    mode_selector.change(
        switch_mode,
        inputs=[mode_selector],
        outputs=[input_image, aspect_selector, custom_w, custom_h],
    )
    aspect_selector.change(
        show_custom,
        inputs=[aspect_selector],
        outputs=[custom_w, custom_h],
    )
    ips = [
        mode_selector,
        input_image,
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
        display_final_frame
    ]
    start_button.click(
        fn=process,
        inputs=ips,
        outputs=[
            result_video,
            preview_image,
            progress_desc,
            progress_bar,
            start_button,
            end_button,
            seed,
            final_frame_out
        ]
    )
    end_button.click(fn=end_process)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument("--server", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, required=False)
    parser.add_argument("--inbrowser", action='store_true')
    args = parser.parse_args()
    print(args)

    block.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
    )
