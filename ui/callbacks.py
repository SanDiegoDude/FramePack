# ui/callbacks.py
import gradio as gr
import torch
import random
import time
import numpy as np
import os
from utils.common import debug
from ui.style import make_progress_bar_html
from diffusers_helper.thread_utils import AsyncStream, async_run
from utils.video_utils import extract_video_frames

# Global stream for async communication
stream = None
graceful_stop_requested = False

def process(
    mode, input_image, start_frame, end_frame, aspect_selector, custom_w, custom_h,
    prompt, n_prompt, seed,
    latent_window_size, segment_count,
    steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, lock_seed, init_color,
    keyframe_weight,
    input_video=None, extension_direction="Forward", extension_frames=8,
    frame_overlap=0, trim_pct=0.2, gaussian_blur_amount=0.0,
    llm_weight=1.0, clip_weight=1.0, clean_latent_weight=1.0,
    video_generator=None, model_manager=None
):
    """
    Main processing function for all generation modes
    
    Args:
        All UI parameters + video_generator and model_manager references
    """
    global stream
    output_filename = None
    final_output_path = None
    
    debug(f"Process called with mode: {mode}")
    assert mode in ['image2video', 'text2video', 'keyframes', 'video_extension'], "Invalid mode"
    
    # Extract values from gradio components if needed
    latent_window_size_val = latent_window_size if not hasattr(latent_window_size, 'value') else latent_window_size.value
    segment_count_val = segment_count if not hasattr(segment_count, 'value') else segment_count.value
    frame_overlap_val = frame_overlap if not hasattr(frame_overlap, 'value') else frame_overlap.value
    
    # Calculate frames per section (for selected_frames mapping)
    frames_per_section = latent_window_size_val * 4 - 3
    effective_frames = frames_per_section - min(frame_overlap_val, frames_per_section-1)
    
    # Map segment_count to appropriate parameter
    adv_seconds = (segment_count_val * effective_frames) / 30.0
    selected_frames = segment_count_val * effective_frames
    
    # Validate inputs based on mode
    if mode == 'image2video' and input_image is None:
        debug("Aborting early -- no input image for image2video")
        return (
            None, None, None,
            "Please upload an input image!", None,
            gr.update(interactive=True),
            gr.update(interactive=False),  # end_graceful_button (replaces end_button)
            gr.update(interactive=False),  # force_stop_button (new)
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update()
        )
    
    # Special handling for video_extension mode
    original_mode = mode
    original_video = input_video
    
    if mode == "video_extension":
        if input_video is None:
            debug("Aborting early -- no input video for video_extension")
            return (
                None, None, None,
                "Please upload a video to extend!", None,
                gr.update(interactive=True),
                gr.update(interactive=False),  # end_graceful_button (replaces end_button)
                gr.update(interactive=False),  # force_stop_button (new)
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update()
            )
        try:
            debug(f"Extracting frames from video for {extension_direction} extension")
            # Extract frames from the video
            extracted_frames, video_fps, _ = video_generator.extract_frames_from_video(
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
            import traceback
            debug(traceback.format_exc())
            return (
                None, None, None,
                f"Error processing video: {str(e)}", None,
                gr.update(interactive=True),
                gr.update(interactive=False),  # end_graceful_button (replaces end_button)
                gr.update(interactive=False),  # force_stop_button (new)
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update()
            )
    
    # Update seed if not locked
    if not lock_seed:
        seed = int(time.time()) % 2**32
    
    debug(f"Starting generation with seed: {seed}")
    debug(f"[UI Callback] Prompt received: '{prompt}'")
    debug(f"[UI Callback] Negative Prompt received: '{n_prompt}'")
    
    # Initial UI update
    yield (
        None, None, None, '', gr.update(visible=False),  # Progress bar hidden at start
        gr.update(interactive=False),
        gr.update(interactive=True),  # end_graceful_button (replaces end_button)
        gr.update(interactive=True),  # force_stop_button (new)
        gr.update(value=seed),
        gr.update(visible=False, elem_classes=""),  # first_frame - explicitly hide
        gr.update(visible=False, elem_classes=""),  # last_frame - explicitly hide
        gr.update(visible=False),   # extend_button - explicitly hide
        gr.update(visible=False),   # note_message - hidden at start
        gr.update(visible=False)    # generation_stats - hidden at start
    )
    # Setup async stream
    stream = AsyncStream()
    video_generator.stream = stream
    
    # Prepare config dictionary for the generator
    config = {
        'mode': mode,
        'input_image': input_image,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'aspect': aspect_selector,
        'custom_w': custom_w,
        'custom_h': custom_h,
        'prompt': prompt,
        'n_prompt': n_prompt,
        'seed': seed,
        'use_adv': True,  # Always use advanced mode
        'latent_window_size': latent_window_size_val,
        'adv_seconds': adv_seconds,
        'selected_frames': selected_frames,
        'segment_count': segment_count_val,
        'steps': steps,
        'cfg': cfg,
        'gs': gs,
        'rs': rs,
        'gpu_memory_preservation': gpu_memory_preservation,
        'use_teacache': use_teacache,
        'init_color': init_color,
        'keyframe_weight': keyframe_weight,
        'input_video': input_video,
        'extension_direction': extension_direction,
        'extension_frames': extension_frames,
        'original_mode': original_mode,
        'frame_overlap': frame_overlap_val,
        'gaussian_blur_amount': gaussian_blur_amount,
        'llm_weight': llm_weight,
        'clip_weight': clip_weight,
        'clean_latent_weight': clean_latent_weight
    }
    
    # Launch async generation
    async_run(video_generator.generate_video, config)
    
    # State tracking variables
    output_filename = None
    last_desc = ""
    last_is_image = False
    last_img_path = None
    
    # Process events from stream
    while True:
        flag, data = stream.output_queue.next()
        debug(f"Process: got queue event: {flag}, type(data): {type(data)}")
        
        if flag == 'file':
            # Assign data first
            output_filename = data # Use a local variable for clarity in this block
            final_output_path = output_filename # Update the outer scope variable too
            debug(f"[UI] Received final file: {output_filename}")

            # Initialize variables before try/call
            first_frame_img = None
            last_frame_img = None
            
            # Extract first and last frames using the corrected path
            try: # Add try-except for robustness
                first_frame_img, last_frame_img = extract_video_frames(output_filename)
                debug(f"[UI] Frame extraction result: first_frame exists={first_frame_img is not None}, last_frame exists={last_frame_img is not None}")
            except Exception as e:
                debug(f"[UI] Error during extract_video_frames: {e}")
                # Keep them as None if extraction fails

            # Handle case where extraction fails or returns None
            if first_frame_img is None:
                 debug("[UI] First frame is None, using placeholder.")
                 first_frame_img = np.zeros((64, 64, 3), dtype=np.uint8) # Placeholder
            if last_frame_img is None:
                 debug("[UI] Last frame is None, using placeholder.")
                 last_frame_img = np.zeros((64, 64, 3), dtype=np.uint8) # Placeholder

            yield (
                gr.update(value=output_filename, visible=True), 
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(interactive=True, value="Start Generation"), 
                gr.update(interactive=False),           # end_graceful_button (replaces end_button)
                gr.update(interactive=False),           # force_stop_button (new)
                gr.update(),
                gr.update(value=first_frame_img, visible=True, elem_classes="show-thumbnail"),
                gr.update(value=last_frame_img, visible=True, elem_classes="show-thumbnail"),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True, value=f"""
            ### Generation Complete
            
            **Video saved as:** `{os.path.basename(output_filename)}`
            
            {last_desc if last_desc else ""}
                """)
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
                    gr.update(interactive=True),           # end_graceful_button (replaces end_button)
                    gr.update(interactive=True),           # force_stop_button (new)
                    gr.update(),                    # seed
                    gr.update(),                    # first_frame
                    gr.update(),                    # last_frame
                    gr.update(),                    # extend_button
                    gr.update(visible=(segment_count_val > 1), value="Note: The ending actions will be generated before the starting actions due to the inverted sampling."),
                    gr.update(visible=False)        # generation_stats
                )
            else:
                debug(f"[UI] Warning: Preview file not found: {preview_filename}")
                
        elif flag == 'progress':
            preview, desc, html = data
            if desc:
                last_desc = desc
            debug(f"Process: yielding progress output: desc={desc}")
            
            # Make sure to set visible=True for progress bar
            yield (
                gr.update(),                           # result_video
                gr.update(),                           # result_image_html
                gr.update(visible=True, value=preview), # preview_image
                desc,                                  # progress_desc
                gr.update(value=html, visible=True),   # progress_bar
                gr.update(interactive=False),          # start_button
                gr.update(interactive=True),           # end_graceful_button (replaces end_button)
                gr.update(interactive=True),           # force_stop_button (new)
                gr.update(),                           # seed
                gr.update(),                           # first_frame
                gr.update(),                           # last_frame
                gr.update(),                           # extend_button
                gr.update(visible=(segment_count_val > 1), value="Note: The ending actions will be generated before the starting actions due to the inverted sampling."),
                gr.update(visible=False)               # generation_stats
            )
            
        elif flag == 'file_img':
            (img_filename, html_link) = data
            debug(f"Process: yielding file_img/single image output: {img_filename}")
            yield (
                gr.update(visible=False),                           # result_video
                gr.update(value=img_filename, visible=True),        # result_image_html
                gr.update(visible=False),                           # preview_image
                f"Generated single image!<br>Saved as <code>{img_filename}</code>",  # progress_desc
                gr.update(visible=False),                           # progress_bar
                gr.update(interactive=True),                        # start_button
                gr.update(interactive=False),           # end_graceful_button (replaces end_button)
                gr.update(interactive=False),           # force_stop_button (new)
                gr.update(),                                        # seed
                gr.update(),                                        # first_frame
                gr.update(),                                        # last_frame
                gr.update(),                                        # extend_button
                gr.update(visible=False),                           # note_message
                gr.update(visible=False)                            # generation_stats
            )
            last_is_image = True
            last_img_path = img_filename
            
        elif flag == 'end':
            debug(f"Process: yielding end event. final_output_path = {final_output_path}, data = {data}")
            if data == "interrupted":
                yield (
                    gr.update(visible=False),       # result_video
                    gr.update(visible=False),       # result_image_html
                    gr.update(visible=False),       # preview_image
                    "Generation stopped by user.",  # progress_desc
                    gr.update(visible=False),       # progress_bar
                    gr.update(interactive=True, value="Start Generation"),
                    gr.update(interactive=False),           # end_graceful_button (replaces end_button)
                    gr.update(interactive=False),           # force_stop_button (new)
                    gr.update(),                    # seed
                    gr.update(),                    # first_frame
                    gr.update(),                    # last_frame
                    gr.update(),                    # extend_button
                    gr.update(visible=False),       # note_message
                    gr.update(visible=False)        # generation_stats
                )
                
            elif data == "img" or last_is_image:  # special image end
                yield (
                    gr.update(visible=False),               # result_video
                    gr.update(visible=True),                # result_image_html (keep image visible)
                    gr.update(visible=False),               # preview_image
                    f"Generated single image!<br><a href=\"file/{last_img_path}\" target=\"_blank\">Click here to open full size in new tab.</a><br><code>{last_img_path}</code>",  # progress_desc
                    gr.update(visible=False),               # progress_bar
                    gr.update(interactive=True),
                    gr.update(interactive=False),           # end_graceful_button (replaces end_button)
                    gr.update(interactive=False),           # force_stop_button (new)
                    gr.update(),                    # seed
                    gr.update(),                    # first_frame
                    gr.update(),                    # last_frame
                    gr.update(),                    # extend_button
                    gr.update(visible=False),       # note_message
                    gr.update(visible=False)        # generation_stats
                )
                
            else:
                yield (
                    gr.update(value=output_filename, visible=True), # result_video
                    gr.update(visible=False),                       # result_image_html
                    gr.update(visible=False),                       # preview_image
                    gr.update(value="", visible=False),             # progress_desc - hide as we use formatted stats
                    gr.update(value="", visible=False),             # progress_bar
                    gr.update(interactive=True),
                    gr.update(interactive=False),           # end_graceful_button (replaces end_button)
                    gr.update(interactive=False),           # force_stop_button (new)
                    gr.update(),                    # seed
                    gr.update(),                    # first_frame
                    gr.update(),                    # last_frame
                    gr.update(),                    # extend_button
                    gr.update(visible=False),       # note_message
                    gr.update(visible=True)         # generation_stats - keep visible
                )
                
            debug("Process: end event, breaking loop.")
            break
            
        else:
            last_is_image = False
            last_img_path = None

# Replace the old end_process function with two new functions
def request_graceful_end():
    """Request a graceful end to the generation process"""
    global stream
    if stream:
        stream.input_queue.push('graceful_end')
    # Return updates for both buttons
    return gr.update(interactive=False), gr.update(interactive=True)

def force_immediate_stop():
    """Force an immediate stop to the generation process"""
    global stream
    if stream:
        stream.input_queue.push('end')
    # Return updates for both buttons
    return gr.update(interactive=False), gr.update(interactive=False)

def end_process():
    """Compatibility function that redirects to force_immediate_stop"""
    global stream
    if stream:
        stream.input_queue.push('end')
    return gr.update(value="Force Stop", variant="stop")

# function to handle extension direction changes for video_extension mode
def toggle_init_color_for_backward(extension_direction, mode):
    """Show color picker when Backward is selected in video_extension mode"""
    show_color = (extension_direction == "Backward" and mode == "video_extension")
    return gr.update(visible=show_color)

def update_video_stats(window_size, segments, overlap):
    """Calculate and format video statistics based on current settings"""
    # Convert inputs to values if needed
    window_size_val = float(window_size)
    segments_val = float(segments)
    overlap_val = float(overlap)
    
    # Calculate frames using the original algorithm logic
    overlapped_frames = int(window_size_val * 4 - 3)
    
    # Calculate frames per segment using the correct formula
    # Last section (first in reverse generation order)
    last_section_frames = int(window_size_val * 2 + 1)
    # Regular sections
    regular_section_frames = int(window_size_val * 2)
    
    # Calculate total frames
    if segments_val == 1:
        total_frames = last_section_frames + 4  # +4 for initial frame when only one segment
    else:
        # First segment (last in generation order) gets the extra frame
        # and we add 4 for the initial frame
        total_frames = last_section_frames + 4 + (regular_section_frames * (segments_val - 1))
    
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
                <td>{regular_section_frames} frames (regular), {last_section_frames} frames (last)</td>
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
        gr.update(visible=is_img2vid),              # input_image
        gr.update(visible=is_keyframes),            # start_frame
        gr.update(visible=is_keyframes),            # end_frame
        gr.update(visible=is_txt2vid),              # aspect_selector
        gr.update(visible=False),                   # custom_w - updated by show_custom
        gr.update(visible=False),                   # custom_h - updated by show_custom
        gr.update(visible=is_keyframes),            # keyframes_options
        gr.update(visible=is_video_ext),            # video_container
        gr.update(visible=is_video_ext),            # video_extension_controls
        gr.update(visible=show_blur)                # gaussian_blur
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
        return [gr.update() for _ in range(7)]  # Update number of return values
    
    return [
        gr.update(value="video_extension"),     # mode_selector
        gr.update(value=video_path),            # input_video
        gr.update(visible=False),               # input_image
        gr.update(visible=False),               # start_frame
        gr.update(visible=False),               # end_frame
        gr.update(visible=True),                # video_container
        gr.update(visible=True)                 # video_extension_controls
    ]
