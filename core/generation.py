# core/generation.py
import os
import time
import torch
import uuid
import numpy as np
from utils.common import debug, generate_timestamp
from utils.memory_utils import cpu, gpu, get_cuda_free_memory_gb
from utils.video_utils import (
    save_bcthw_as_mp4, find_nearest_bucket, resize_and_center_crop,
    crop_or_pad_yield_mask, extract_frames_from_video, fix_video_compatibility
)
from diffusers_helper.thread_utils import AsyncStream

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
        from diffusers_helper.thread_utils import AsyncStream
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
    
    def get_valid_frame_stops(self, latent_window_size, max_seconds=120, fps=30):
        """Get valid frame stop points"""
        frames_per_section = latent_window_size * 4 - 3
        max_sections = int((max_seconds * fps) // frames_per_section)
        stops = [frames_per_section * i for i in range(1, max_sections + 1)]
        return stops
    
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
        """
        Prepare inputs for generation
        
        This is a placeholder - we'll implement the full functionality later
        """
        # This will be implemented with the actual generation functionality
        return None
    
    def generate_video(self, config):
    """
    Main video generation function
    
    Args:
        config: Dictionary containing generation parameters
    
    Returns:
        dict: Results and status information
    """
    # This is a placeholder for the worker function
    # We'll implement the full functionality as we build out the module
    
    job_id = generate_timestamp()
    debug(f"Starting video generation job {job_id}")
    
    # Eventually this will contain the logic from the worker function
    return {
        "job_id": job_id,
        "status": "placeholder",
        "message": "VideoGenerator.generate_video not yet implemented"
    }
