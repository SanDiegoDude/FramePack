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
    batch_count=1, endless_run=False, unload_on_end_flag=False, # ADDED inputs
    video_generator=None, model_manager=None
):
    """
    Main processing function for all generation modes
    
    Args:
        All UI parameters + batch/endless flags + video_generator and model_manager references
    """
    global stream, graceful_stop_requested # Use global flag for graceful stop signal

    # --- Outer loop for Batch/Endless ---
    current_batch_item = 0
    total_batch_count = int(batch_count) if not endless_run else float('inf') # Use inf for endless
    run_seed = seed # Store initial seed

    # Flags to control the loop and stop logic
    _stop_requested_flag = False # Internal flag for forced stop
    _graceful_stop_batch_flag = False # Internal flag for graceful batch stop

    while (not endless_run and current_batch_item < total_batch_count) or \
          (endless_run and not _graceful_stop_batch_flag and not _stop_requested_flag):

        current_batch_item += 1
        batch_progress_text = f" (Batch {current_batch_item}/{total_batch_count})" if total_batch_count > 1 and not endless_run else \
                              f" (Endless Run - Item {current_batch_item})" if endless_run else ""
        debug(f"--- Starting Batch Item {current_batch_item} ---")

        # --- Reset state for this iteration ---
        output_filename = None
        final_output_path = None
        graceful_stop_requested = False # Reset graceful stop for the *generation* part
        stream = None # Ensure stream is reset

        # Update seed for this run if not locked and not the very first run
        if not lock_seed and current_batch_item > 1:
             run_seed = int(time.time()) % 2**32
             debug(f"New seed for batch item {current_batch_item}: {run_seed}")
        elif lock_seed and current_batch_item > 1:
             debug(f"Using locked seed {run_seed} for batch item {current_batch_item}")

        # --- Start of original 'process' function logic ---
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
            None, None, None, # outputs
            f"Starting Generation{batch_progress_text}...", # progress desc
            gr.update(visible=False),  # Progress bar hidden at start
            gr.update(interactive=False), # start button
            gr.update(interactive=True),  # end_graceful_button
            gr.update(interactive=True),  # force_stop_button
            gr.update(value=run_seed),    # Update seed display
            gr.update(visible=False, elem_classes=""),  # first_frame
            gr.update(visible=False, elem_classes=""),  # last_frame
            gr.update(visible=False),   # extend_button
            gr.update(visible=False),   # note_message
            gr.update(visible=False),   # generation_stats
            gr.update(visible=False), # generation_stats_accordian
            gr.update(visible=False), # frame_thumbnails_group
            gr.update(visible=False), # final_processed_prompt_display
            gr.update(visible=False) # final_prompt_accordion
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
            'seed': run_seed,
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
        
        # State tracking variables for *this* generation
        output_filename = None
        last_desc = ""
        last_is_image = False
        last_img_path = None
        final_prompt_text = "Not available"
        last_stats = ""
        generation_interrupted = False # Track if *this* generation was stopped early
        generation_error = False # Track errors like wildcards

        # Process events from stream for *this* generation
        while True:
            if _stop_requested_flag: # Check for immediate stop signal
                 debug("Force stop detected during stream processing.")
                 if stream: stream.input_queue.push('end') # Force stop the async generator
                 generation_interrupted = True
                 break # Exit inner stream loop

            if stream is None: # Should not happen, but safety check
                debug("Stream became None unexpectedly.")
                break

            flag, data = stream.output_queue.next()
            debug(f"Process: got queue event: {flag}, type(data): {type(data)}")
    
            # --- Handle final_prompt flag ---
            if flag == 'final_prompt':
                final_prompt_text = data
                debug(f"[UI] Received final prompt: '{final_prompt_text}'")
                # No UI update needed here, just store it for the 'end' event
                continue # Go to next event
    
            # --- Handle file/preview/progress --- 
            elif flag == 'file':
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
                    gr.update(interactive=False, value="End Generation"),           # end_graceful_button (replaces end_button)
                    gr.update(interactive=False, value="Force Stop"),           # force_stop_button (new)
                    gr.update(),
                    gr.update(value=first_frame_img, visible=True, elem_classes="show-thumbnail"),
                    gr.update(value=last_frame_img, visible=True, elem_classes="show-thumbnail"),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True, value=f"""
                ### Generation Complete
                
                **Video saved as:** `{os.path.basename(output_filename)}`
                
                {last_desc if last_desc else ""}
                    """),
                    gr.update(visible=True, open=False),            # generation_stats_accordion 
                    gr.update(visible=True),                        # frame_thumbnails_group
                    gr.update(visible=False), # final_processed_prompt_display
                    gr.update(visible=False) # final_prompt_accordion
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
                        f"Generating video{batch_progress_text}...", # progress_desc
                        gr.update(visible=True),                       # progress_bar
                        gr.update(interactive=False),
                        gr.update(interactive=True),           # end_graceful_button (replaces end_button)
                        gr.update(interactive=True),           # force_stop_button (new)
                        gr.update(),                    # seed
                        gr.update(),                    # first_frame
                        gr.update(),                    # last_frame
                        gr.update(),                    # extend_button
                        gr.update(visible=(segment_count_val > 1), value="Note: The ending actions will be generated before the starting actions due to the inverted sampling."),
                        gr.update(visible=False),        # generation_stats
                        gr.update(visible=False, open=False),            # generation_stats_accordion 
                        gr.update(visible=False), # frame_thumbnails_group
                        gr.update(visible=False), # final_processed_prompt_display
                        gr.update(visible=False)  # final_prompt_accordion
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
                    f"{batch_progress_text}", # progress_desc
                    gr.update(value=html, visible=True),   # progress_bar
                    gr.update(interactive=False),          # start_button
                    gr.update(interactive=True),           # end_graceful_button (replaces end_button)
                    gr.update(interactive=True),           # force_stop_button (new)
                    gr.update(),                           # seed
                    gr.update(),                           # first_frame
                    gr.update(),                           # last_frame
                    gr.update(),                           # extend_button
                    gr.update(visible=(segment_count_val > 1), value="Note: The ending actions will be generated before the starting actions due to the inverted sampling."),
                    gr.update(visible=False),               # generation_stats
                    gr.update(visible=False, open=False),    # generation_stats_accordion 
                    gr.update(visible=False), # frame_thumbnails_group
                    gr.update(visible=False), # final_processed_prompt_display
                    gr.update(visible=False)  # final_prompt_accordion
                )
                
            elif flag == 'file_img':
                (img_filename, html_link) = data
                debug(f"Process: yielding file_img/single image output: {img_filename}")
                yield (
                    gr.update(visible=False),                           # result_video
                    gr.update(value=img_filename, visible=True),        # result_image_html
                    gr.update(visible=False),                           # preview_image
                    f"Generated single image!{batch_progress_text}<br>Saved as <code>{img_filename}</code>", # progress_desc (updated)
                    gr.update(visible=False),                           # progress_bar
                    gr.update(interactive=True),                        # start_button
                    gr.update(interactive=False),           # end_graceful_button (replaces end_button)
                    gr.update(interactive=False),           # force_stop_button (new)
                    gr.update(),                                        # seed
                    gr.update(),                                        # first_frame
                    gr.update(),                                        # last_frame
                    gr.update(),                                        # extend_button
                    gr.update(visible=False),                           # note_message
                    gr.update(visible=False),                            # generation_stats
                    gr.update(visible=False, open=False),            # generation_stats_accordion 
                    gr.update(visible=False), # frame_thumbnails_group
                    gr.update(value=final_prompt_text), # final_processed_prompt_display
                    gr.update()  # final_prompt_accordion
                )
                last_is_image = True
                last_img_path = img_filename
    
            elif flag == 'final_stats':
                last_stats = data
                debug(f"[UI] Received final stats: {last_stats[:50]}...")
                continue #Go to next event
            
            elif flag == 'end':
                debug(f"Process: yielding 'end' event for batch item {current_batch_item}. final_output_path = {final_output_path}, data = {data}")

                # Determine if this specific generation ended due to interruption/error
                generation_interrupted = (data == "interrupted" or data == "wildcard_error" or data == "lora_error" or data == "prompt_error" or _stop_requested_flag)
                generation_error = (data == "wildcard_error" or data == "lora_error" or data == "prompt_error")

                if _stop_requested_flag:
                    debug("Force stop finalized this generation.")
                    # Break inner loop, outer loop condition will handle exit
                    break

                if _graceful_stop_batch_flag:
                    debug("Graceful stop finalized this generation.")
                    generation_interrupted = True # Treat as interrupted for UI update purposes
                    # Break inner loop, outer loop condition will handle exit

                if generation_error:
                    debug(f"Generation error '{data}' occurred. Stopping batch.")
                    _stop_requested_flag = True # Treat errors like force stop for the batch
                    # Update UI with error message
                    error_message = f"Generation Error: {data}. Batch stopped."
                    yield (
                        gr.update(visible=False), # result_video
                        gr.update(visible=False), # result_image_html
                        gr.update(visible=False), # preview_image
                        error_message,            # progress_desc
                        gr.update(visible=False), # progress_bar
                        gr.update(interactive=True, value="Start Generation"),
                        gr.update(interactive=False, value="End Generation"),
                        gr.update(interactive=False, value="Force Stop"),           # force_stop_button (new)
                        gr.update(),                    # seed
                        gr.update(),                    # first_frame
                        gr.update(),                    # last_frame
                        gr.update(),                    # extend_button
                        gr.update(visible=False),       # note_message
                        gr.update(visible=False),        # generation_stats
                        gr.update(visible=False, open=False),            # generation_stats_accordion 
                        gr.update(visible=False),       # frame_thumbnails_group
                        gr.update(value=final_prompt_text, visible=True), # final_processed_prompt_display - show prompt even on error
                        gr.update(visible=True)
                    )
                    break # Exit inner loop
                    
                final_status_message = ""
                show_outputs = False
                show_thumbnails = False

                if generation_interrupted:
                    final_status_message = f"Generation Interrupted{batch_progress_text}."
                    if _graceful_stop_batch_flag:
                        final_status_message += " Finishing batch run."
                    if _stop_requested_flag:
                        final_status_message = f"Generation Force Stopped{batch_progress_text}."
                    show_outputs = False
                elif last_is_image:
                    final_status_message = f"Generated single image!{batch_progress_text}<br><a href=\"file/{last_img_path}\" target=\"_blank\">Open</a> <code>{last_img_path}</code>"
                    show_outputs = True # Show the image result
                else: # Normal video completion for this item
                    stats_display = last_stats if last_stats else "No performance stats."
                    if output_filename and os.path.exists(output_filename):
                        final_status_message = f"""
            ### Generation Complete{batch_progress_text}
            **Video saved as:** `{os.path.basename(output_filename)}`
            {stats_display}
                        """
                        show_outputs = True
                        show_thumbnails = True
                    else:
                         final_status_message = f"""
            ### Generation Finished{batch_progress_text} (No Output Saved)
            {stats_display}
                         """
                         show_outputs = False

                # Check if this is the *absolute last* item in the sequence
                is_last_item_overall = (not endless_run and current_batch_item >= total_batch_count) or \
                                       (endless_run and (_graceful_stop_batch_flag or _stop_requested_flag)) or \
                                       _stop_requested_flag # Force stop always ends the sequence

                # Decide button states based on whether the whole sequence is ending
                final_start_interactive = is_last_item_overall
                final_graceful_interactive = not is_last_item_overall
                final_force_interactive = not is_last_item_overall

                yield (
                    gr.update(value=output_filename if output_filename and show_outputs and not last_is_image else None, visible=show_outputs and not last_is_image), # result_video
                    gr.update(value=last_img_path if last_img_path and show_outputs and last_is_image else None, visible=show_outputs and last_is_image), # result_image_html
                    gr.update(visible=False),                      # preview_image
                    gr.update(value="", visible=False),            # progress_desc (use stats)
                    gr.update(value="", visible=False),            # progress_bar
                    gr.update(interactive=final_start_interactive, value="Start Generation"),
                    gr.update(interactive=final_graceful_interactive, value="End Generation"),
                    gr.update(interactive=final_force_interactive, value="Force Stop"),
                    gr.update(), # seed (keeps last run's seed)
                    gr.update(visible=show_thumbnails), # first_frame visibility tied to thumbnails
                    gr.update(visible=show_thumbnails), # last_frame visibility tied to thumbnails
                    gr.update(visible=show_thumbnails), # extend_button visibility tied to thumbnails
                    gr.update(visible=False),           # note_message
                    gr.update(visible=True, value=final_status_message), # generation_stats
                    gr.update(visible=True, open=False), # generation_stats_accordion
                    gr.update(visible=show_thumbnails),  # frame_thumbnails_group
                    gr.update(value=final_prompt_text, visible=True), # final_processed_prompt_display
                    gr.update(visible=True)              # final_prompt_accordion
                )

                debug(f"Process: end event processed for batch item {current_batch_item}.")
                break # Exit the inner stream processing loop

            # --- Handle other flags --- (existing)
            else:
                last_is_image = False
                last_img_path = None
# --- End of inner stream processing loop ---
 # --- After a generation finishes (or is interrupted) ---
        debug(f"End of batch item {current_batch_item}. Stop flags: stop={_stop_requested_flag}, graceful={_graceful_stop_batch_flag}")

        # If a force stop was requested, break the outer loop immediately
        if _stop_requested_flag:
            debug("Force stop requested, breaking batch loop.")
            # Reset UI elements associated with batch/endless
            yield (
                 # Re-yield UI state to ensure buttons are correct after forced stop
                 gr.update(), # result_video
                 gr.update(), # result_image_html
                 gr.update(), # preview_image
                 gr.update(value=f"Batch Force Stopped after item {current_batch_item}."), # progress_desc
                 gr.update(visible=False), # progress_bar
                 gr.update(interactive=True),  # start_button - enable
                 gr.update(interactive=False), # end_graceful_button - disable
                 gr.update(interactive=False), # force_stop_button - disable
                 gr.update(), # seed
                 gr.update(), # first_frame
                 gr.update(), # last_frame
                 gr.update(), # extend_button
                 gr.update(), # note_message
                 gr.update(), # generation_stats
                 gr.update(), # generation_stats_accordion
                 gr.update(), # frame_thumbnails_group
                 gr.update(), # final_processed_prompt_display
                 gr.update() # final_prompt_accordion
            )
            break # Exit outer batch loop

        # If a graceful stop for the *whole batch* was requested, break outer loop
        if _graceful_stop_batch_flag:
            debug("Graceful batch stop requested, breaking batch loop after current item.")
            break # Exit outer batch loop

        # If we finished the planned batches (and not endless)
        if not endless_run and current_batch_item >= total_batch_count:
             debug("Finished all planned batch items.")
             break # Exit outer batch loop

        # If endless, check the actual UI checkbox state (if possible, Gradio limitation might apply)
        # For now, rely on the _graceful_stop_batch_flag set by the button click
        if endless_run and _graceful_stop_batch_flag:
             debug("Endless run stopped gracefully.")
             break # Exit outer batch loop

        # Otherwise, continue to the next iteration of the outer loop
        debug(f"Proceeding to next batch item or continuing endless run.")
        # Optional: Short delay between batches? time.sleep(1)

    # --- End of outer batch/endless loop ---
    debug("--- Batch/Endless Run Finished ---")

    # Final cleanup/actions after the entire sequence is done
    is_fully_complete = not _stop_requested_flag # Check if we didn't force stop

    # Unload models if requested AND the sequence completed naturally or gracefully stopped
    if unload_on_end_flag and is_fully_complete:
        debug("Unloading models as unload_on_end is set and batch finished.")
        try:
            yield ( # Yield status update BEFORE unload
                gr.update(), gr.update(), gr.update(), # outputs
                "Generation sequence complete. Unloading models...", # progress desc
                gr.update(visible=False), # progress bar
                gr.update(), gr.update(), gr.update(), # buttons
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), # etc
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update() # stats, frames, prompt
            )
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
         debug("Not unloading models (unload_on_end not set or batch was force stopped).")
         # Ensure final UI state is yielded correctly if no unload happens
         # This might already be covered by the last yield in the loop's 'end' handler
         # Add a final yield here just in case the loop exited without a final status yield
         yield (
            gr.update(), gr.update(), gr.update(), # outputs
            "Generation sequence finished.", # progress desc
            gr.update(visible=False), # progress bar
            gr.update(interactive=True), # start button enabled
            gr.update(interactive=False), # end graceful disabled
            gr.update(interactive=False), # force stop disabled
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), # etc
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update() # stats, frames, prompt
         )

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

    # Reset batch/endless UI elements immediately (handled by the yield in process)
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
