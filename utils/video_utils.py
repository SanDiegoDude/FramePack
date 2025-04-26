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
                "-crf", "23",  # Adjust quality (lower is better)
                "-preset", "medium",  # Adjust speed/compression trade-off
                filename
            ]
            
            subprocess.run(cmd, check=True)
            
            # Make the video compatible with QuickTime and Windows Media Player
            fix_video_compatibility(filename, fps)
            
            return True
    
    except Exception as e:
        debug(f"Error saving video: {e}")
        import traceback
        debug(traceback.format_exc())
        return False

def fix_video_compatibility(video_path, fps=30):
    """Fix compatibility issues with MP4 videos for QuickTime and Windows Media Player."""
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

def extract_video_frames(video_path, first_and_last=True):
    """Extract first and/or last frame from a video file"""
    try:
        debug(f"Extracting frames from {video_path}")
        if not os.path.exists(video_path):
            debug(f"Video file not found: {video_path}")
            return None, None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            debug(f"Could not open video: {video_path}")
            return None, None
        
        # Get first frame
        ret, first = cap.read()
        first_frame = cv2.cvtColor(first, cv2.COLOR_BGR2RGB) if ret else None
        
        if first_and_last:
            # Get frame count and jump to last frame
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            ret, last = cap.read()
            last_frame = cv2.cvtColor(last, cv2.COLOR_BGR2RGB) if ret else None
        else:
            last_frame = None
        
        cap.release()
        return first_frame, last_frame
    
    except Exception as e:
        debug(f"Error extracting video frames: {e}")
        return None, None

def extract_frames_from_video(video_path, num_frames=8, from_end=True, max_resolution=640):
    """
    Extract frames from a video file with bucket resizing for memory efficiency.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        from_end: If True, extract from the end of the video
        max_resolution: Maximum resolution for the extracted frames
    
    Returns:
        tuple: (frames, fps, original_dimensions)
    """
    try:
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
        
        # Calculate bucket dimensions
        bucket_height, bucket_width = find_nearest_bucket(orig_height, orig_width, resolution=max_resolution)
        debug(f"Using bucket dimensions: {bucket_width}x{bucket_height}")
        
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
                
                # Resize to bucket dimensions
                frame = resize_and_center_crop(frame, target_width=bucket_width, target_height=bucket_height)
                
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
        
        return frames, fps, (orig_height, orig_width)
    
    except Exception as e:
        debug(f"Error extracting frames from video: {e}")
        import traceback
        debug(traceback.format_exc())
        raise

def find_nearest_bucket(height, width, resolution=640, bucket_step=8):
    """
    Find the nearest bucket dimensions for a given height and width.
    
    Args:
        height: Original height
        width: Original width
        resolution: Target resolution
        bucket_step: Step size for bucket dimensions
    
    Returns:
        tuple: (bucket_height, bucket_width)
    """
    # Calculate aspect ratio
    aspect = width / height
    
    # Scale to target resolution (the longest side will be scaled to this value)
    if height > width:
        new_height = resolution
        new_width = int(new_height * aspect)
    else:
        new_width = resolution
        new_height = int(new_width / aspect)
    
    # Round to nearest bucket_step
    bucket_height = math.ceil(new_height / bucket_step) * bucket_step
    bucket_width = math.ceil(new_width / bucket_step) * bucket_step
    
    return bucket_height, bucket_width

def resize_and_center_crop(image, target_width, target_height):
    """
    Resize an image and center crop to the target dimensions
    
    Args:
        image: Input image (numpy array)
        target_width: Target width
        target_height: Target height
    
    Returns:
        numpy array: Resized and cropped image
    """
    h, w = image.shape[:2]
    aspect = w / h
    target_aspect = target_width / target_height
    
    # Resize to match either width or height
    if aspect > target_aspect:
        # Image is wider than target aspect, match height
        new_h = target_height
        new_w = int(new_h * aspect)
    else:
        # Image is taller than target aspect, match width
        new_w = target_width
        new_h = int(new_w / aspect)
    
    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Center crop
    y_start = max(0, (new_h - target_height) // 2)
    x_start = max(0, (new_w - target_width) // 2)
    
    cropped = resized[y_start:y_start + target_height, x_start:x_start + target_width]
    
    # Handle any size discrepancies (should be rare)
    h, w = cropped.shape[:2]
    
    # If cropped image is smaller than target, pad it
    if h < target_height or w < target_width:
        pad_h = max(0, target_height - h)
        pad_w = max(0, target_width - w)
        
        # Create padding
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # Pad the image (with border replication)
        cropped = cv2.copyMakeBorder(
            cropped, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    
    return cropped

def make_mp4_faststart(mp4_path):
    """
    Move moov atom to start of file for fast streaming
    
    Args:
        mp4_path: Path to MP4 file
    """
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
            from utils.common import debug
            debug("WARNING: Could not apply blur - no suitable method found. Returning original image.")
            return image_tensor
