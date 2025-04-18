# framepack-ui.py (patched version)

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

# --- New: Enhanced CLI Parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument('--hf-cache', choices=['local', 'global'], default='local')
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=7800)
parser.add_argument('--inbrowser', action='store_true')
args = parser.parse_args()

# HF cache logic
if args.hf_cache == 'local':
    os.environ['HF_HOME'] = os.path.abspath(os.path.join(os.path.dirname(__file__), './hf_download'))

# --- Device VRAM Check ---
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60
print(f'Free VRAM {free_mem_gb:.2f} GB')
print(f'High-VRAM Mode: {high_vram}')

# model loading and interface definition would follow...

# --- Fix Gradio mask_image crash ---
# Replace this line (buggy):
# mask_image = gr.Image(shape=(None,None), type="numpy", label="Mask (video inpaint)", visible=False)
# With:
mask_image = gr.Image(type="numpy", label="Mask (video inpaint)", visible=False)

# --- Gradio Launch Block Patch ---
block.launch(
    server_name=args.host,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
