# ui/callbacks.py
import torch
import random
import time
import numpy as np
from utils.common import debug
from ui.style import make_progress_bar_html

def process(*args):
    """
    Main processing function for all generation modes
    This is a placeholder that mimics the functionality without actual generation
    """
    debug("UI Process: Started (placeholder)")
    
    # Extract parameters (partial list)
    mode = args[0]
    input_image = args[1]
    start_frame = args[2]
    end_frame = args[3]
    
    debug(f"Process mode: {mode}")
    
    # For now, just return some placeholder data
    return (
        gr.update(visible=False),                   # result_video
        gr.update(visible=False),                   # result_image_html
        gr.update(visible=False),                   # preview_image
        "This is a placeholder UI. Actual generation not implemented yet.",  # progress_desc
        gr.update(visible=False),                   # progress_bar
        gr.update(interactive=True),                # start_button
        gr.update(interactive=False),               # end_button
        gr.update(),                                # seed
        gr.update(),                                # first_frame
        gr.update(),                                # last_frame
        gr.update()                                 # extend_button
    )

def end_process():
    """Placeholder for the end_process function"""
    debug("UI End Process: Called (placeholder)")
    return gr.update(value="End Generation", variant="secondary")

def update_video_stats(window_size, segments, overlap):
    """Calculate and format video statistics based on current settings"""
    # Convert inputs to values if needed
    window_size_val = float(window_size)
    segments_val = float(segments)
    overlap_val = float(overlap)
    
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

def update_overlap_slider(window_size):
    """Update maximum overlap based on window size"""
    window_size_val = float(window_size)
    frames_per_section = int(window_size_val * 4 - 3)
    max_overlap = max(0, frames_per_section - 1)
    return gr.update(maximum=max_overlap)

def switch_mode(mode):
    """Switch visibility of UI elements based on mode"""
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
        gr.update(visible=False),  # custom_w - will be updated by show_custom
        gr.update(visible=False),  # custom_h - will be updated by show_custom
        gr.update(visible=is_keyframes),  # keyframes_options
        gr.update(visible=is_video_ext),  # video_extension_options
        gr.update(visible=show_blur)  # gaussian_blur
    )

def show_custom(aspect):
    """Show/hide custom width/height inputs"""
    show = aspect == "Custom..."
    return gr.update(visible=show), gr.update(visible=show)

def show_init_color(mode):
    """Show/hide initial color picker"""
    return gr.update(visible=(mode == "text2video"))

def setup_for_extension(video_path):
    """Setup UI for video extension"""
    if not video_path:
        return [gr.update() for _ in range(6)]
    
    return [
        gr.update(value="video_extension"),  # mode_selector
        gr.update(value=video_path),         # input_video
        gr.update(visible=False),            # input_image
        gr.update(visible=False),            # start_frame
        gr.update(visible=False),            # end_frame
        gr.update(visible=True)              # video_extension_options
    ]
