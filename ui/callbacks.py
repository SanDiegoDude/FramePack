# ui/callbacks.py
import gradio as gr
import torch
import random
import time
import numpy as np
import os
from utils.common import debug # Keep this for debug
# Correct the import for memory functions
from utils.memory_utils import get_cuda_free_memory_gb, gpu # IMPORT FROM memory_utils
from ui.style import make_progress_bar_html
from diffusers_helper.thread_utils import AsyncStream, async_run
from utils.video_utils import extract_video_frames
import json

# Global stream for async communication
stream = None
_stop_requested_flag = False
_graceful_stop_batch_flag = False

def process(
    mode, input_image, start_frame, end_frame, aspect_selector, custom_w, custom_h,
    prompt, n_prompt, seed,
    latent_window_size, segment_count,
    steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, lock_seed, init_color,
    keyframe_weight,
    input_video=None, extension_direction="Forward", extension_frames=8,
    frame_overlap=0, trim_pct=0.2, gaussian_blur_amount=0.0,
    llm_weight=1.0, clip_weight=1.0, clean_latent_weight=1.0,
    batch_count=1, endless_run=False, unload_on_end_flag=False,
    video_generator=None, model_manager=None
):
    """
    Main processing function for all generation modes with batch/endless support.
    """
    global stream, _stop_requested_flag, _graceful_stop_batch_flag
  
    # --- Reset global flags at the start of a new 'Start Generation' click ---
    # IMPORTANT: Always reset flags at the beginning of a new run sequence
    _stop_requested_flag = False
    _graceful_stop_batch_flag = False
    debug("Reset stop flags for new run sequence.")
    # --------------------------------------------------------------------

    current_batch_item = 0
    total_batch_count = int(batch_count) if not endless_run else float('inf')
    run_seed = seed
    last_successful_output = None # Store the last good output path/data

    while True: # Outer loop for batch/endless

        # --- IMMEDIATE EXIT CHECKS ---
        if _stop_requested_flag:
            debug("IMMEDIATE EXIT: Force stop flag detected at start of outer loop.")
            break
        if _graceful_stop_batch_flag:
            debug("IMMEDIATE EXIT: Graceful stop flag detected at start of outer loop.")
            break
        if not endless_run and current_batch_item >= total_batch_count:
            debug(f"IMMEDIATE EXIT: Completed {current_batch_item}/{int(batch_count)} batch items.")
            break
        # --- END IMMEDIATE EXIT CHECKS ---

        current_batch_item += 1
        batch_progress_text = f" (Batch {current_batch_item}/{int(batch_count)})" if not endless_run and batch_count > 1 else \
                              f" (Endless Run - Item {current_batch_item})" if endless_run else ""
        debug(f"--- Starting Batch Item {current_batch_item} ---")

        # Reset state for this *specific generation*
        output_filename = None
        final_output_path = None
        stream = None # Crucial: Reset stream for the new generation task

        # Update seed (as before)
        if not lock_seed and current_batch_item > 1:
            run_seed = int(time.time()) % 2**32
            debug(f"New seed for batch item {current_batch_item}: {run_seed}")
        elif current_batch_item > 1:
            debug(f"Using locked seed {run_seed} for batch item {current_batch_item}")

        # --- Prep logic (validation, video extract - keep user's code) ---
        # ... (Your existing validation and video_extension logic here) ...
        # Ensure `mode`, `input_image`, `start_frame`, `end_frame`, etc. are set correctly for this iteration
        latent_window_size_val = latent_window_size if not hasattr(latent_window_size, 'value') else latent_window_size.value
        segment_count_val = segment_count if not hasattr(segment_count, 'value') else segment_count.value
        frame_overlap_val = frame_overlap if not hasattr(frame_overlap, 'value') else frame_overlap.value
        frames_per_section = latent_window_size_val * 4 - 3
        effective_frames = frames_per_section - min(frame_overlap_val, frames_per_section-1)
        adv_seconds = (segment_count_val * effective_frames) / 30.0
        selected_frames = segment_count_val * effective_frames
        original_mode = mode # Store original mode before potential change in video_extension
        if mode == "video_extension":
             if input_video is None: return # Abort (error handled earlier)
             try:
                  extracted_frames, _, _ = video_generator.extract_frames_from_video(input_video, int(extension_frames), (extension_direction == "Forward"), 640)
                  if extension_direction == "Forward":
                       input_image = extracted_frames[-1]; mode = "image2video"
                  else:
                       end_frame = extracted_frames[0]; start_frame = None; mode = "keyframes"
             except Exception as e: return # Abort (error handled earlier)
        # --------------------------------------------------------------------

        # --- Initial UI Yield for this Batch Item ---
        # Keep previous final outputs visible, hide previews/progress for the new one
        yield (
            gr.update(), # result_video (no change yet)
            gr.update(), # result_image_html (no change yet)
            gr.update(visible=False), # preview_image (hide previous)
            f"Starting Generation{batch_progress_text}...", # progress_desc
            gr.update(visible=False), # progress_bar (hide previous)
            gr.update(interactive=False), # start_button
            gr.update(interactive=True), # end_graceful_button
            gr.update(interactive=True), # force_stop_button
            gr.update(value=run_seed), # seed display
            gr.update(), # first_frame (no change yet)
            gr.update(), # last_frame (no change yet)
            gr.update(), # extend_button (no change yet)
            gr.update(visible=False), # note_message (hide previous)
            gr.update(), # generation_stats (no change yet)
            gr.update(), # generation_stats_accordion (no change yet)
            gr.update(), # frame_thumbnails_group (no change yet)
            gr.update(), # final_processed_prompt_display (no change yet)
            gr.update() # final_prompt_accordion (no change yet)
        )
        # ------------------------------------------------

        # Setup and run generation (ensure video_generator.stream is set!)
        stream = AsyncStream()
        video_generator.stream = stream # *** IMPORTANT ***
        config = { 'mode': mode, 'input_image': input_image, 'start_frame': start_frame, 'end_frame': end_frame, 'aspect': aspect_selector, 'custom_w': custom_w, 'custom_h': custom_h, 'prompt': prompt, 'n_prompt': n_prompt, 'seed': run_seed, 'use_adv': True, 'latent_window_size': latent_window_size_val, 'adv_seconds': adv_seconds, 'selected_frames': selected_frames, 'segment_count': segment_count_val, 'steps': steps, 'cfg': cfg, 'gs': gs, 'rs': rs, 'gpu_memory_preservation': gpu_memory_preservation, 'use_teacache': use_teacache, 'init_color': init_color, 'keyframe_weight': keyframe_weight, 'input_video': input_video, 'extension_direction': extension_direction, 'extension_frames': extension_frames, 'original_mode': original_mode, 'frame_overlap': frame_overlap_val, 'gaussian_blur_amount': gaussian_blur_amount, 'llm_weight': llm_weight, 'clip_weight': clip_weight, 'clean_latent_weight': clean_latent_weight, 'trim_percentage': trim_pct } # Added trim_pct
        async_run(video_generator.generate_video, config)

        # --- Inner Loop: Process Stream Events for *this* generation ---
        last_desc = ""
        last_is_image = False
        last_img_path = None
        final_prompt_text = "Prompt not received"
        last_stats = ""
        generation_interrupted_flag = False # Reset for this generation

        while True:
            # Check for external stop signals FIRST
            if _stop_requested_flag:
                debug("INNER LOOP: Force stop flag detected.")
                if stream: stream.input_queue.push('end') # Signal generator
                generation_interrupted_flag = True
                break # Exit inner loop

            # Graceful batch stop doesn't break inner loop, generator handles 'graceful_end'
            if _graceful_stop_batch_flag:
                 debug("INNER LOOP: Graceful batch stop requested, allowing current generation to potentially finish gracefully.")
                 # No break here, let 'end' event handle it or generator callback interrupt

            if stream is None: debug("INNER LOOP: Stream is None."); generation_interrupted_flag = True; break

            # Get next event
            flag, data = stream.output_queue.next()
            debug(f"Inner Loop: Event: {flag}, Data Type: {type(data)}")

            # Handle non-UI updates first
            if flag == 'final_prompt': final_prompt_text = data; continue
            if flag == 'final_stats': last_stats = data; continue

            # --- UI Updates ---
            elif flag == 'preview_video' or flag == 'progress':
                preview_val = None
                desc_val = last_desc # Use previous desc if only preview video comes
                html_val = gr.update() # Keep existing progress bar if only preview

                if flag == 'preview_video':
                     preview_val = data # Data is the preview image/video path
                elif flag == 'progress':
                     preview_val, desc_val, html_val = data # Unpack progress data
                     if desc_val: last_desc = desc_val

                # Yield updates: CLEAR old outputs, show new progress/preview
                yield (
                    gr.update(value=data if flag=='preview_video' else None, visible=(flag=='preview_video')), # result_video (show preview if video)
                    gr.update(visible=False),                       # result_image_html (clear image)
                    gr.update(value=preview_val, visible=True),     # preview_image (show new preview)
                    f"{desc_val}{batch_progress_text}",             # progress_desc
                    gr.update(value=html_val, visible=True),        # progress_bar
                    gr.update(interactive=False),                   # start_button
                    gr.update(interactive=not _graceful_stop_batch_flag), # end_graceful (allow stopping batch)
                    gr.update(interactive=True),                    # force_stop (always allow)
                    gr.update(),                                    # seed
                    gr.update(visible=False),                       # first_frame (clear)
                    gr.update(visible=False),                       # last_frame (clear)
                    gr.update(visible=False),                       # extend_button (clear)
                    gr.update(visible=(segment_count_val > 1), value="Note: Generating end before start..."), # note_message
                    gr.update(visible=False),                       # generation_stats (clear)
                    gr.update(visible=False, open=False),           # generation_stats_accordion
                    gr.update(visible=False),                       # frame_thumbnails_group (clear)
                    gr.update(visible=False),                       # final_processed_prompt_display
                    gr.update(visible=False)                        # final_prompt_accordion
                )

            elif flag == 'file':
                output_filename = data
                last_successful_output = output_filename # Store for next iteration start
                first_frame_img, last_frame_img = extract_video_frames(output_filename)
                # Handle None frames with placeholders if needed
                first_frame_img = first_frame_img if first_frame_img is not None else np.zeros((64, 64, 3), dtype=np.uint8)
                last_frame_img = last_frame_img if last_frame_img is not None else np.zeros((64, 64, 3), dtype=np.uint8)

                # Yield updates: Show final video, hide preview, show thumbnails
                yield (
                    gr.update(value=output_filename, visible=True), # result_video
                    gr.update(visible=False),                       # result_image_html
                    gr.update(visible=False),                       # preview_image (hide)
                    gr.update(visible=False),                       # progress_desc (hide)
                    gr.update(visible=False),                       # progress_bar (hide)
                    gr.update(), # Buttons updated at 'end' event
                    gr.update(),
                    gr.update(),
                    gr.update(), # seed
                    gr.update(value=first_frame_img, visible=True), # first_frame
                    gr.update(value=last_frame_img, visible=True),  # last_frame
                    gr.update(visible=True),                        # extend_button
                    gr.update(visible=False),                       # note_message
                    gr.update(), # Stats updated at 'end'
                    gr.update(),
                    gr.update(visible=True),                        # frame_thumbnails_group
                    gr.update(),
                    gr.update()
                )
                last_is_image = False
                last_img_path = None

            elif flag == 'file_img':
                (img_filename, html_link) = data
                last_successful_output = img_filename # Store for next iteration start

                # Yield updates: Show final image, hide video/preview
                yield (
                    gr.update(visible=False),                       # result_video
                    gr.update(value=img_filename, visible=True),    # result_image_html
                    gr.update(visible=False),                       # preview_image (hide)
                    gr.update(visible=False),                       # progress_desc (hide)
                    gr.update(visible=False),                       # progress_bar (hide)
                    gr.update(), # Buttons updated at 'end'
                    gr.update(),
                    gr.update(),
                    gr.update(), # seed
                    gr.update(visible=False),                       # first_frame (hide)
                    gr.update(visible=False),                       # last_frame (hide)
                    gr.update(visible=False),                       # extend_button (hide)
                    gr.update(visible=False),                       # note_message
                    gr.update(), # Stats updated at 'end'
                    gr.update(),
                    gr.update(visible=False),                       # frame_thumbnails_group (hide)
                    gr.update(),
                    gr.update()
                )
                last_is_image = True
                last_img_path = img_filename

            elif flag == 'end':
                debug(f"Inner Loop: Received 'end'. data = {data}")
                is_error = (data == "wildcard_error" or data == "lora_error" or data == "prompt_error")
                generation_interrupted_flag = generation_interrupted_flag or (data == "interrupted") or is_error

                if is_error:
                    debug(f"Generation error '{data}'. Signaling force stop.")
                    _stop_requested_flag = True # Signal outer loop
                    error_message = f"Error: {data}. Batch stopped."
                    yield ( # Yield immediate error state
                         gr.update(), gr.update(), gr.update(visible=False), error_message, gr.update(visible=False),
                         gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False),
                         gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                         gr.update(visible=True, value=error_message), gr.update(visible=True, open=True), gr.update(visible=False),
                         gr.update(value=final_prompt_text, visible=True), gr.update(visible=True)
                    )
                # No 'else' here, break happens below

                # Break inner loop on any 'end' event (normal or interrupted/errored)
                debug("Inner Loop: Breaking on 'end' event.")
                break

            else: # Handle unknown flags?
                 debug(f"Inner Loop: Ignoring unknown flag '{flag}'")

        # --- End of inner stream processing loop ---
        debug(f"Inner loop finished for batch item {current_batch_item}. Interrupted={generation_interrupted_flag}")

        # --- Post-Generation UI Update (before next loop iteration or final exit) ---
        final_status_message = ""
        # Use last_successful_output to retain visibility if current gen failed/was stopped
        current_output_path = output_filename if output_filename else last_img_path
        output_to_display = current_output_path if current_output_path else last_successful_output

        show_final_output = bool(output_to_display) # Show if we have *any* output
        show_as_image = last_is_image if current_output_path else (isinstance(output_to_display, str) and output_to_display.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
        show_thumbnails = show_final_output and not show_as_image

        # Determine final status message
        if generation_interrupted_flag:
            final_status_message = f"Generation Interrupted{batch_progress_text}."
            if _graceful_stop_batch_flag: final_status_message += " Finishing batch."
            if _stop_requested_flag: final_status_message = f"Generation Force Stopped{batch_progress_text}."
        elif show_as_image:
            final_status_message = f"Generated Image!{batch_progress_text}<br><code>{output_to_display}</code>"
        elif show_thumbnails: # Video completed
             stats_display = last_stats if last_stats else "No performance stats."
             final_status_message = f"### Generation Complete{batch_progress_text}\n**Video:** `{os.path.basename(output_to_display)}`\n{stats_display}"
        else: # Completed but no output saved? Or only preview?
             stats_display = last_stats if last_stats else ""
             final_status_message = f"### Generation Finished{batch_progress_text}\n{stats_display}"

        # Determine button states for the *next* potential iteration or final state
        is_last_item_planned = not endless_run and (current_batch_item >= total_batch_count) # Check if this was the last *planned*
        is_stopping = _stop_requested_flag or _graceful_stop_batch_flag or is_last_item_planned

        # Update UI with final status for *this* item
        yield (
            gr.update(value=output_to_display if show_thumbnails else None, visible=show_thumbnails),
            gr.update(value=output_to_display if show_as_image else None, visible=show_as_image),
            gr.update(visible=False), # preview_image (hide)
            gr.update(visible=False), # progress_desc (hide)
            gr.update(visible=False), # progress_bar (hide)
            gr.update(interactive=is_stopping), # start_button
            gr.update(interactive=not is_stopping), # end_graceful_button
            gr.update(interactive=not is_stopping), # force_stop_button
            gr.update(), # seed
            gr.update(visible=show_thumbnails), # first_frame
            gr.update(visible=show_thumbnails), # last_frame
            gr.update(visible=show_thumbnails), # extend_button
            gr.update(visible=False), # note_message
            gr.update(visible=True, value=final_status_message), # generation_stats
            gr.update(visible=True, open=generation_interrupted_flag), # Open accordion if interrupted/errored
            gr.update(visible=show_thumbnails), # frame_thumbnails_group
            gr.update(value=final_prompt_text, visible=True), # final_processed_prompt_display
            gr.update(visible=True) # final_prompt_accordion
        )

        # --- Loop control checks (redundant with top checks, but safe) ---
        if _stop_requested_flag: debug("Outer loop: Breaking due to force stop flag."); break
        if _graceful_stop_batch_flag: debug("Outer loop: Breaking due to graceful stop flag."); break
        if not endless_run and current_batch_item >= total_batch_count: debug("Outer loop: Breaking after finishing planned batches."); break
        # --- End Loop control checks ---

        debug(f"Outer loop: Continuing to next iteration.")
        # Optional delay between batches: time.sleep(1)

    # --- End of outer batch/endless loop ---
    debug("--- Batch/Endless Run Finished ---")
    final_message = "Generation sequence finished."
    if _stop_requested_flag: final_message = "Generation sequence force stopped."
    if _graceful_stop_batch_flag: final_message = "Generation sequence stopped gracefully."

    # Final unload logic (remains the same as previous version)
    if unload_on_end_flag and not _stop_requested_flag:
        debug("Unloading models as unload_on_end is set and batch finished.")
        try:
            yield ( gr.update(),gr.update(),gr.update(), f"{final_message} Models unloaded.", gr.update(visible=False), gr.update(interactive=True),gr.update(interactive=False),gr.update(interactive=False), gr.update(),gr.update(),gr.update(),gr.update(),gr.update(), gr.update(),gr.update(),gr.update(),gr.update(),gr.update() )
            model_manager.unload_all_models()
            free_mem = get_cuda_free_memory_gb(gpu)
            mem_status_update = f"Models unloaded. Free VRAM: {free_mem:.1f} GB"
            debug(mem_status_update)
            # Yield final state *after* unload
            yield (
                gr.update(), gr.update(), gr.update(), # outputs
                mem_status_update, # progress desc
                gr.update(visible=False), # progress bar
                gr.update(interactive=True), # start button enabled
                gr.update(interactive=False), # end graceful disabled
                gr.update(interactive=False), # force stop disabled
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), # etc
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update() # stats, frames, prompt
            )
        except Exception as e:
            debug(f"Error unloading models at end: {e}")
            yield (
                 # ... yield error state ...
                 gr.update(value=f"Error during final model unload: {e}"), # progress_desc
                 # ... ensure buttons are interactive ...
            )
    else:
         debug("Not unloading models.")
         # Final UI yield to ensure buttons are in correct final state
         yield ( gr.update(),gr.update(),gr.update(), final_message, gr.update(visible=False), gr.update(interactive=True),gr.update(interactive=False),gr.update(interactive=False), gr.update(),gr.update(),gr.update(),gr.update(),gr.update(), gr.update(),gr.update(),gr.update(),gr.update(),gr.update() )

    # Clean up stream reference
    stream = None

# --- Modify Stop Handlers ---

def request_graceful_end():
    """Request a graceful end to the current generation OR the entire batch/endless run."""
    global stream, graceful_stop_requested
    global _graceful_stop_batch_flag # Need access to the flag controlling the outer loop

    if stream:
        debug("Requesting graceful stop for *current* generation via stream.")
        stream.input_queue.push('graceful_end')
        graceful_stop_requested = True # Signal generator callback
    else:
        debug("No active stream, assuming request is for the *batch/endless* loop.")

    # Signal the outer batch loop to stop after this item finishes
    _graceful_stop_batch_flag = True
    debug("Signalled graceful stop for the entire batch/endless sequence.")

    # Update button states: Disable graceful, keep force active (until current gen finishes)
    return gr.update(interactive=False), gr.update(interactive=True)

def force_immediate_stop():
    """Force an immediate stop to the generation process and the entire batch."""
    global stream
    global _stop_requested_flag # Need access to the flag controlling the outer loop
  
    debug("Requesting FORCE STOP.")
    _stop_requested_flag = True # Signal the outer loop and inner checks
  
    if stream:
        debug("Pushing 'end' to stream for immediate generator stop.")
        stream.input_queue.push('end') # Tell the generator to stop ASAP
  
    # Return updates for both buttons: Disable both immediately
    return gr.update(interactive=False), gr.update(interactive=False)

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
    
    # Calculate frames based on the original demo's approach
    last_section_frames = int(window_size_val * 4 + 1)  # Last section (first in generation order)
    regular_section_frames = int(window_size_val * 4) # Regular sections
    
    # Maximum possible overlap is limited by section sizes
    max_possible_overlap = min(regular_section_frames, last_section_frames)
    # User's requested overlap (limited by what's possible)
    effective_overlap = min(int(overlap_val), max_possible_overlap)
    
    # Calculate total frames
    if segments_val <= 1:
        # Just one section (include the input frame)
        total_frames = last_section_frames
    else:
        # First section gets full frame count plus input frame
        total_frames = last_section_frames
        
        # Each additional section adds (length - overlap) frames
        for i in range(1, int(segments_val)):
            total_frames += regular_section_frames
    
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
