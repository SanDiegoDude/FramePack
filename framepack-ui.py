from diffusers_helper.hf_login import login

import os
import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
from PIL import Image
import torchvision

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
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=7800)
parser.add_argument('--inbrowser', action='store_true')
args = parser.parse_args()

if args.hf_cache == 'local':
    os.environ['HF_HOME'] = os.path.abspath(os.path.join(os.path.dirname(__file__), './hf_download'))

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60
print(f'Free VRAM {free_mem_gb:.2f} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

for m in (vae, text_encoder, text_encoder_2, image_encoder, transformer):
    m.eval().requires_grad_(False)

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    for m in (vae, text_encoder, text_encoder_2, image_encoder, transformer):
        m.to(gpu)

transformer.high_quality_fp32_output_for_inference = True
transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

stream = AsyncStream()
outputs_folder = './outputs/'; os.makedirs(outputs_folder, exist_ok=True)

# -------- Worker (handles all modes) --------
@torch.no_grad()
def worker(mode, input_image, input_video, mask_image,
           prompt, n_prompt, sampler, shift, cfg, gs, rs,
           strength, seed, total_second_length, latent_window_size,
           steps, gpu_memory_preservation, use_teacache):

    # --- prepare prompt embeddings ---
    llama_vec, clip_pool = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    if cfg == 1:
        llama_vec_n = torch.zeros_like(llama_vec); clip_pool_n = torch.zeros_like(clip_pool)
    else:
        llama_vec_n, clip_pool_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    llama_vec, llama_mask = crop_or_pad_yield_mask(llama_vec, length=512)
    llama_vec_n, llama_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

    # --- prepare latents per mode ---
    height, width = None, None
    init_latent, concat_latent = None, None

    if mode == 'txt2vid':
        # black image
        dummy = torch.zeros((1,3,1,512,512), device=gpu, dtype=torch.float16)
        init_latent = vae_encode(dummy, vae)

    elif mode == 'img2vid':
        np_img = np.array(input_image)
        H,W,_=np_img.shape
        height,width = find_nearest_bucket(H,W,640)
        img = resize_and_center_crop(np_img, width, height)
        inp = (torch.from_numpy(img).float()/127.5-1.0).permute(2,0,1)[None,:,None].to(device=gpu)
        init_latent = vae_encode(inp, vae)

    elif mode in ('vid2vid','extend_vid','video_inpaint'):
        # read video
        vid, _, _ = torchvision.io.read_video(input_video.name if hasattr(input_video,'name') else input_video)
        # vid: T,H,W,C
        vid = vid.permute(3,0,1,2)[None].float()/127.5-1.0
        _,_,T,H,W = vid.shape
        height,width = H,W
        concat_latent = vae_encode(vid.to(device=gpu), vae)
        if mode=='video_inpaint':
            # mask not implemented—stub
            pass

    # fallback sizing
    if height is None:
        height, width = 512,512

    # --- sampling loop replicated from original, but passing new args ---
    total_latent_sections = max(int(round((total_second_length*30)/(latent_window_size*4))),1)
    job_id = generate_timestamp()
    stream.output_queue.push(('progress',(None,'',make_progress_bar_html(0,'Starting...'))))

    rnd = torch.Generator("cpu").manual_seed(int(seed))
    num_frames = latent_window_size*4 - 3
    history_latents = torch.zeros((1,16,1+2+16,height//8,width//8),dtype=torch.float32).cpu()
    history_pixels, total_gen = None,0
    paddings = ([3]+[2]*(total_latent_sections-3)+[1,0]) if total_latent_sections>4 else list(reversed(range(total_latent_sections)))

    for section, latent_padding in enumerate(paddings):
        is_last = (latent_padding==0)
        padding_size = latent_padding*latent_window_size

        # compute clean_latents as original code...
        clean_pre = init_latent if init_latent is not None else history_latents
        clean_post,cl2,cl4 = history_latents[:,:, :1+2+16].split([1,2,16],dim=2)
        clean_all = torch.cat([clean_pre.to(history_latents), clean_post], dim=2)
        indices = torch.arange(0,1+padding_size+latent_window_size+1+2+16).unsqueeze(0)
        pre,_,li,post,i2,i4 = indices.split([1,padding_size,latent_window_size,1,2,16],dim=1)
        clean_indices = torch.cat([pre,post],dim=1)

        # offload/load as before...
        if not high_vram:
            unload_complete_models(); move_model_to_device_with_memory_preservation(transformer,gpu,gpu_memory_preservation)
        transformer.initialize_teacache(use_teacache, steps) if use_teacache else transformer.initialize_teacache(False)

        def cb(d):
            pr = vae_decode_fake(d['denoised'])
            pr = (pr*255).detach().cpu().numpy().clip(0,255).astype(np.uint8)
            pr = einops.rearrange(pr,'b c t h w -> (b h) (t w) c')
            pct = int(100*(d['i']+1)/steps)
            desc = f'Frames: {total_gen*4-3}/{int(total_second_length*30)}, running...'
            stream.output_queue.push(('progress',(pr,desc,make_progress_bar_html(pct,f"Step {d['i']+1}/{steps}"))))

        generated = sample_hunyuan(
            transformer,
            sampler=sampler,
            initial_latent=init_latent,
            concat_latent=concat_latent,
            strength=strength,
            width=width, height=height,
            frames=num_frames,
            real_guidance_scale=cfg,
            distilled_guidance_scale=gs,
            guidance_rescale=rs,
            shift=shift,
            num_inference_steps=steps,
            generator=rnd,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_mask,
            prompt_poolers=clip_pool,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_mask_n,
            negative_prompt_poolers=clip_pool_n,
            device=gpu, dtype=torch.bfloat16,
            image_embeddings=None,
            latent_indices=li,
            clean_latents=clean_all,
            clean_latent_indices=clean_indices,
            clean_latents_2x=cl2,
            clean_latent_2x_indices=i2,
            clean_latents_4x=cl4,
            clean_latent_4x_indices=i4,
            callback=cb
        )

        if is_last and init_latent is not None:
            generated = torch.cat([init_latent.to(generated), generated], dim=2)

        total_gen += generated.shape[2]
        history_latents = torch.cat([generated.to(history_latents), history_latents], dim=2)
        real_latents = history_latents[:,:,:total_gen]

        if history_pixels is None:
            history_pixels = vae_decode(real_latents, vae).cpu()
        else:
            overlap = latent_window_size*4-3
            sec_frames = (latent_window_size*2+1 if is_last else latent_window_size*2)
            cur = vae_decode(real_latents[:,:,:sec_frames], vae).cpu()
            history_pixels = soft_append_bcthw(cur, history_pixels, overlap)

        if not high_vram:
            offload_model_from_device_for_memory_preservation(transformer,gpu,8)
            load_model_as_complete(vae,gpu)

        out_path = os.path.join(outputs_folder,f'{job_id}_{total_gen}.mp4')
        save_bcthw_as_mp4(history_pixels, out_path, fps=30)
        stream.output_queue.push(('file', out_path))
        if is_last: break

    stream.output_queue.push(('end',None))

# -------- Gradio UI --------
def process_fn(mode, img, vid, mask, prompt, n_prompt, sampler, shift, cfg, gs, rs,
               strength, seed, seconds, window, steps, gpu_mem, tea):
    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)
    stream.input_queue = AsyncStream().input_queue
    async_run(worker, mode, img, vid, mask, prompt, n_prompt, sampler, shift, cfg, gs, rs,
              strength, seed, seconds, window, steps, gpu_mem, tea)
    out, prev = None, None
    while True:
        flag, data = stream.output_queue.next()
        if flag=='file':
            out = data; yield out, prev, '', '', gr.update(interactive=False), gr.update(interactive=True)
        if flag=='progress':
            pr,ds,html = data; yield gr.update(), gr.update(visible=True,value=pr), ds, html, gr.update(interactive=False), gr.update(interactive=True)
        if flag=='end':
            yield out, gr.update(visible=False), '', '', gr.update(interactive=True), gr.update(interactive=False)
            break

css = make_progress_bar_css()
with gr.Blocks(css=css).queue() as demo:
    gr.Markdown("# FramePack – Advanced")
    with gr.Row():
        with gr.Column():
            mode       = gr.Radio(["txt2vid","img2vid","video_inpaint","vid2vid","extend_vid"], value="txt2vid", label="Mode")
            input_image  = gr.Image(sources='upload', type="numpy", label="Image (for img2vid)")
            input_video  = gr.Video(label="Video (for vid2vid/…)", visible=False)
            mask_image   = gr.Image(shape=(None,None), type="numpy", label="Mask (video inpaint)", visible=False)
            prompt     = gr.Textbox(label="Prompt")
            n_prompt   = gr.Textbox(label="Negative Prompt", value="")
            sampler    = gr.Dropdown(["unipc","unipc_bh2","dpm2","dpm2_ancestral","ddpm","plms"], value="unipc", label="Sampler")
            shift      = gr.Slider(0.,10., value=None, label="Shift μ (None=auto)")
            cfg        = gr.Slider(1.,32., value=1., label="CFG Scale")
            gs         = gr.Slider(1.,32., value=10.,label="Distilled CFG Scale")
            rs         = gr.Slider(0.,1.,  value=0., label="Rescale")
            strength   = gr.Slider(0.,1., value=1., label="Denoise Strength")
            seed       = gr.Number(label="Seed", value=31337, precision=0)
            seconds    = gr.Slider(1,120, value=5, step=0.1, label="Length (sec)")
            window     = gr.Slider(1,33,  value=9, step=1, label="Latent window")
            steps      = gr.Slider(1,100,value=25, step=1, label="Steps")
            gpu_mem    = gr.Slider(6,128,value=6, step=0.1, label="GPU preserve (GB)")
            tea        = gr.Checkbox(label="Use TeaCache", value=True)
            start_btn  = gr.Button("Start")
            end_btn    = gr.Button("End", interactive=False)
        with gr.Column():
            preview    = gr.Image(label="Preview", visible=False)
            result_vid = gr.Video(label="Output", height=512, loop=True)
            status_md  = gr.Markdown("")
            bar_html   = gr.HTML("")
    # show/hide inputs by mode
    mode.change(lambda m: {'visible': m=="img2vid"}, input_image, input_image, queue=False)
    mode.change(lambda m: {'visible': m in ["vid2vid","extend_vid","video_inpaint"]}, input_video, input_video, queue=False)
    mode.change(lambda m: {'visible': m=="video_inpaint"}, mask_image, mask_image, queue=False)

    inputs = [mode, input_image, input_video, mask_image,
              prompt, n_prompt, sampler, shift, cfg, gs, rs,
              strength, seed, seconds, window, steps, gpu_mem, tea]
    start_btn.click(process_fn, inputs, [result_vid, preview, status_md, bar_html, start_btn, end_btn])
    end_btn.click(lambda: stream.input_queue.push('end'))

demo.launch(server_name=args.server, server_port=args.port, share=args.share, inbrowser=args.inbrowser)
