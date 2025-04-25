# utils/video_utils.py
import os
import subprocess
import cv2
import numpy as np
from PIL import Image
import torch
import math
from utils.common import debug

def save_bcthw_as_mp4(tensor, filename, fps=30):
    """
    Save a tensor with shape [batch, channels, time, height, width] as an MP4 video file
    
    Args:
        tensor: Tensor with shape [b,c,t,h,w]
        filename: Output filename
        fps: Frames per second
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    try:
        import tempfile
        import shutil
        
        # Create a temporary directory to store frames
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract dimensions
            b, c, t, h, w = tensor.shape
            
            # Process each batch and time frame
            for bi in range(b):
                # Process all frames in this batch
                for ti in range(t):
                    # Get frame and convert to image
                    frame = tensor[bi, :, ti]
                    frame = (frame.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                    
                    # Save frame as an image file
                    frame_path = os.path.join(tmpdir, f"frame_{ti:05d}.png")
                    Image.fromarray(frame).save(frame_path)
            
            # Use ffmpeg to convert the frames to an MP4 video
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmpdir, "frame_%05d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23
