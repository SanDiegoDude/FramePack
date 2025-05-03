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
    _stop_requested_flag = False
    _graceful_stop_batch_flag = False
    debug("Reset stop flags for new run sequence.")
    # --------------------------------------------------------------------
  
    current_batch_item = 0
    total_batch_count = int(batch_count) if not endless_run else float('inf')
    run_seed = seed
    
    # Batch state tracking variables
    last_output_filename = None
    last_output_is_image = False
    last_output_image_path = None
    last_first_frame = None
    last_last_frame = None
    
    # Main batch loop
    while True:
        # Check for exit conditions
        if _stop_requested_flag:
            debug("IMMEDIATE EXIT: Force stop flag detected at start of outer loop.")
            break
        if _graceful_stop_batch_flag:
            debug("GRACEFUL EXIT: Graceful stop flag detected at start of outer loop.")
            break
        if not endless_run and current_batch_item >= total_batch_count:
            debug(f"NATURAL EXIT: Completed {current_batch_item}/{int(batch_count)} batch items.")
            break
        
        # Increment batch counter
        current_batch_item += 1
        batch_progress_text = f" (Batch {current_batch_item}/{int(batch_count)})" if not endless_run and batch_count > 1 else \
                             f" (Endless Run - Item {current_batch_item})" if endless_run else ""
        debug(f"--- Starting Batch Item {current_batch_item} ---")
        
        # Reset per-generation state
        stream = None
        final_output_path = None
        
        # Update seed for this batch item
        if not lock_seed and current_batch_item > 1:
            run_seed = int(time.time()) % 2**32
            debug(f"New seed for batch item {current_batch_item}: {run_seed}")
        
        # Process video extraction for video_extension mode
        if mode == "video_extension":
            if input_video is None:
               yield (
                  gr.update(), gr.update(), gr.update(visible=False),
                  "Error: Input video required for video extension mode.", gr.update(visible=False),
                  gr.update(interactive=True), gr.update(interactive=True), # Include endless_run_button
                  gr.update(interactive=False), gr.update(interactive=False),
                  gr.update(), gr.update(visible=False), gr.update(visible=False),
                  gr.update(visible=False), gr.update(visible=False),
                  gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
               )
               break
                
            try:
                extracted_frames, _, _ = video_generator.extract_frames_from_video(
                    input_video, int(extension_frames), (extension_direction == "Forward"), 640
                )
                if extension_direction == "Forward":
                    input_image = extracted_frames[-1]; orig_mode = mode; mode = "image2video"
                else:
                    end_frame = extracted_frames[0]; start_frame = None; orig_mode = mode; mode = "keyframes"
            except Exception as e:
                yield (
                    gr.update(), gr.update(), gr.update(visible=False),
                    f"Error extracting frames from video: {e}", gr.update(visible=False),
                    gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(), gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                )
                break
        
        # --- Input validation ---
        if mode == "image2video" and input_image is None:
          yield (
              gr.update(), gr.update(), gr.update(visible=False),
              "Error: Image input required for image2video mode", gr.update(visible=False),
              gr.update(interactive=True), gr.update(interactive=True), # Include endless_run_button
              gr.update(interactive=False), gr.update(interactive=False),
              gr.update(), gr.update(visible=False), gr.update(visible=False),
              gr.update(visible=False), gr.update(visible=False),
              gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
          )
          break
            
        if mode == "keyframes" and end_frame is None:
          yield (
              gr.update(), gr.update(), gr.update(visible=False),
              "Error: End Frame required for keyframes mode", gr.update(visible=False),
              gr.update(interactive=True), gr.update(interactive=True), # Include endless_run_button 
              gr.update(interactive=False), gr.update(interactive=False),
              gr.update(), gr.update(visible=False), gr.update(visible=False),
              gr.update(visible=False), gr.update(visible=False),
              gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
          )
          break
        
        # --- Initial UI Yield for this Batch Item ---
        # Keep previous outputs visible during new generation
        yield (
            # Keep previous video visible if it exists
            gr.update(value=last_output_filename if not last_output_is_image and last_output_filename else None, 
                    visible=not last_output_is_image and last_output_filename is not None),
            # Keep previous image visible if it exists
            gr.update(value=last_output_image_path if last_output_is_image else None,
                    visible=last_output_is_image),
            gr.update(visible=False),  # preview_image
            f"Starting Generation{batch_progress_text}...",  # progress_desc
            gr.update(visible=False),  # progress_bar
            gr.update(interactive=False),  # start_button
            gr.update(interactive=False),  # endless_run_button 
            gr.update(interactive=True),  # end_graceful_button
            gr.update(interactive=True),  # force_stop_button
            gr.update(value=run_seed),  # seed display
            # Keep frame thumbnails if they exist
            gr.update(value=last_first_frame if last_first_frame is not None else None, 
                    visible=last_first_frame is not None),
            gr.update(value=last_last_frame if last_last_frame is not None else None, 
                    visible=last_last_frame is not None),
            gr.update(visible=not last_output_is_image and last_output_filename is not None),  # extend_button
            gr.update(visible=False),  # note_message
            gr.update(visible=True),  # generation_stats
            gr.update(visible=True),  # generation_stats_accordion
            gr.update(visible=not last_output_is_image and last_output_filename is not None),  # frame_thumbnails
            gr.update(visible=True),  # final_processed_prompt
            gr.update(visible=True)  # final_prompt_accordion
        )
        
        # Calculate params for generation
        latent_window_size_val = latent_window_size if not hasattr(latent_window_size, 'value') else latent_window_size.value
        segment_count_val = segment_count if not hasattr(segment_count, 'value') else segment_count.value
        frame_overlap_val = frame_overlap if not hasattr(frame_overlap, 'value') else frame_overlap.value
        frames_per_section = latent_window_size_val * 4 - 3
        effective_frames = frames_per_section - min(frame_overlap_val, frames_per_section-1)
        adv_seconds = (segment_count_val * effective_frames) / 30.0
        selected_frames = segment_count_val * effective_frames
        original_mode = mode if 'orig_mode' not in locals() else orig_mode
        
        # Setup and run generation
        stream = AsyncStream()
        video_generator.stream = stream
        config = {
            'mode': mode, 'input_image': input_image, 'start_frame': start_frame, 'end_frame': end_frame, 
            'aspect': aspect_selector, 'custom_w': custom_w, 'custom_h': custom_h, 
            'prompt': prompt, 'n_prompt': n_prompt, 'seed': run_seed, 
            'use_adv': True, 'latent_window_size': latent_window_size_val, 
            'adv_seconds': adv_seconds, 'selected_frames': selected_frames, 
            'segment_count': segment_count_val, 'steps': steps, 
            'cfg': cfg, 'gs': gs, 'rs': rs, 
            'gpu_memory_preservation': gpu_memory_preservation, 'use_teacache': use_teacache, 
            'init_color': init_color, 'keyframe_weight': keyframe_weight, 
            'input_video': input_video, 'extension_direction': extension_direction, 
            'extension_frames': extension_frames, 'original_mode': original_mode, 
            'frame_overlap': frame_overlap_val, 'gaussian_blur_amount': gaussian_blur_amount, 
            'llm_weight': llm_weight, 'clip_weight': clip_weight, 
            'clean_latent_weight': clean_latent_weight, 'trim_percentage': trim_pct
        }
        
        async_run(video_generator.generate_video, config)
        
        # Per-generation state variables
        output_filename = None
        last_desc = ""
        is_image_output = False
        image_path = None
        final_prompt_text = "Prompt not received"
        last_stats = ""
        generation_interrupted = False
        first_frame_img = None
        last_frame_img = None
        preview_video_visible = False
        
        # --- Inner Loop: Process Stream Events for this generation ---
        while True:
            # Check for stop flags
            if _stop_requested_flag:
                debug("INNER LOOP: Force stop flag detected.")
                if stream: stream.input_queue.push('end')
                generation_interrupted = True
                break
            
            # Check for graceful stop (don't break inner loop)
            if _graceful_stop_batch_flag and stream:
                stream.input_queue.push('graceful_end')
                debug("INNER LOOP: Sent graceful_end signal to generator.")
            
            # Get next event
            flag, data = stream.output_queue.next()
            debug(f"Inner Loop: Event: {flag}, Data Type: {type(data)}")
            
            # Handle special flags that don't need UI updates
            if flag == 'final_prompt':
                final_prompt_text = data
                debug(f"Received final prompt: '{final_prompt_text}'")
                continue
                
            if flag == 'final_stats':
                last_stats = data
                debug(f"Received final stats: {last_stats[:50]}...")
                continue
            
            # Handle UI update flags
            if flag == 'preview_video':
                debug(f"Received preview video: {data}")
                preview_video_visible = True
                yield (
                    gr.update(value=data, visible=True),  # result_video
                    gr.update(visible=False),  # result_image_html
                    gr.update(visible=False),  # preview_image
                    "Generating video...",  # progress_desc
                    gr.update(visible=True),  # progress_bar
                    gr.update(interactive=False),  # start_button
                    gr.update(interactive=False),  # endless_run_button 
                    gr.update(interactive=True),  # end_graceful_button
                    gr.update(interactive=True),  # force_stop_button
                    gr.update(),  # seed
                    gr.update(visible=False),  # first_frame
                    gr.update(visible=False),  # last_frame
                    gr.update(visible=False),  # extend_button
                    gr.update(visible=(segment_count_val > 1), value="Note: Generating end before start..."),
                    gr.update(visible=False),  # generation_stats
                    gr.update(visible=False),  # generation_stats_accordion
                    gr.update(visible=False),  # frame_thumbnails
                    gr.update(visible=False),  # final_processed_prompt
                    gr.update(visible=False)  # final_prompt_accordion
                )
            
            elif flag == 'progress':
                if isinstance(data, tuple) and len(data) == 3:
                    preview_val, desc_val, html_val = data
                    if desc_val is not None and not isinstance(desc_val, str):
                        desc_val = str(desc_val)
                    if desc_val:
                        last_desc = desc_val
                else:
                    debug(f"Unexpected progress data format: {type(data)}")
                    preview_val = None
                    desc_val = last_desc
                    html_val = gr.update()
                
                yield (
                    gr.update(visible=preview_video_visible),  # Keep preview video visible if it was showing
                    gr.update(visible=False),  # result_image_html
                    gr.update(value=preview_val, visible=True),  # preview_image
                    f"{desc_val}{batch_progress_text}",  # progress_desc
                    gr.update(value=html_val, visible=True),  # progress_bar
                    gr.update(interactive=False),  # start_button
                    gr.update(interactive=False),  # endless_run_button 
                    gr.update(interactive=True),  # end_graceful_button
                    gr.update(interactive=True),  # force_stop_button
                    gr.update(),  # seed
                    gr.update(visible=False),  # first_frame
                    gr.update(visible=False),  # last_frame
                    gr.update(visible=False),  # extend_button
                    gr.update(visible=(segment_count_val > 1), value="Note: Generating end before start..."),
                    gr.update(visible=False),  # generation_stats
                    gr.update(visible=False),  # generation_stats_accordion
                    gr.update(visible=False),  # frame_thumbnails
                    gr.update(visible=False),  # final_processed_prompt
                    gr.update(visible=False)  # final_prompt_accordion
                )
            
            elif flag == 'file':
                output_filename = data
                is_image_output = False
                image_path = None
                
                # Save for next batch iteration
                last_output_filename = output_filename
                last_output_is_image = False
                last_output_image_path = None
                
                # Extract frame thumbnails
                first_frame_img, last_frame_img = extract_video_frames(output_filename)
                if first_frame_img is None:
                    first_frame_img = np.zeros((64, 64, 3), dtype=np.uint8)
                if last_frame_img is None:
                    last_frame_img = np.zeros((64, 64, 3), dtype=np.uint8)
                    
                # Save for next batch iteration
                last_first_frame = first_frame_img
                last_last_frame = last_frame_img
                
                yield (
                    gr.update(value=output_filename, visible=True),  # result_video
                    gr.update(visible=False),  # result_image_html
                    gr.update(visible=False),  # preview_image
                    gr.update(visible=False),  # progress_desc
                    gr.update(visible=False),  # progress_bar
                    gr.update(interactive=False),  # start_button - updated in 'end'
                    gr.update(interactive=False),  # endless_run_button 
                    gr.update(interactive=True),  # end_graceful_button
                    gr.update(interactive=True),  # force_stop_button
                    gr.update(),  # seed
                    gr.update(value=first_frame_img, visible=True),  # first_frame
                    gr.update(value=last_frame_img, visible=True),  # last_frame
                    gr.update(visible=True),  # extend_button
                    gr.update(visible=False),  # note_message
                    gr.update(visible=False),  # generation_stats - updated in 'end'
                    gr.update(visible=False),  # generation_stats_accordion
                    gr.update(visible=True),  # frame_thumbnails
                    gr.update(visible=False),  # final_processed_prompt
                    gr.update(visible=False)  # final_prompt_accordion
                )
            
            elif flag == 'file_img':
                # Unpack image path and HTML link
                (img_filename, html_link) = data
                debug(f"Process: yielding file_img/single image output: {img_filename}")
                
                # Save state for batch tracking
                is_image_output = True
                image_path = img_filename
                last_output_filename = None
                last_output_is_image = True
                last_output_image_path = img_filename
                
                # Clear frame thumbnails for image outputs
                last_first_frame = None
                last_last_frame = None
                
                # Match your working code exactly
                yield (
                    gr.update(visible=False),                           # result_video
                    gr.update(value=img_filename, visible=True),        # result_image_html - IMPORTANT: Set value AND visible
                    gr.update(visible=False),                           # preview_image
                    f"Generated single image!<br>Saved as <code>{img_filename}</code>",  # progress_desc
                    gr.update(visible=False),                           # progress_bar
                    gr.update(interactive=True),                        # start_button - IMPORTANT: Make interactive immediately
                    gr.update(interactive=True),                        # endless_run_button 
                    gr.update(interactive=False),                       # end_graceful_button
                    gr.update(interactive=False),                       # force_stop_button
                    gr.update(),                                        # seed
                    gr.update(visible=False),                           # first_frame
                    gr.update(visible=False),                           # last_frame
                    gr.update(visible=False),                           # extend_button
                    gr.update(visible=False),                           # note_message
                    gr.update(visible=False),                           # generation_stats
                    gr.update(visible=False, open=False),               # generation_stats_accordion
                    gr.update(visible=False),                           # frame_thumbnails
                    gr.update(value=final_prompt_text, visible=True),   # final_processed_prompt
                    gr.update(visible=True)                             # final_prompt_accordion
                )
            
            elif flag == 'end':
                debug(f"Inner Loop: Received 'end' event with data={data}")
                
                # Check if generation was error/interrupted
                is_error = (data == "wildcard_error" or data == "lora_error" or data == "prompt_error" or
                           data == "validation_error")
                generation_interrupted = generation_interrupted or (data == "interrupted") or is_error
                
                if is_error:
                    debug(f"Generation error '{data}'. Signaling force stop.")
                    _stop_requested_flag = True
                    yield (
                        gr.update(visible=False),  # result_video
                        gr.update(visible=False),  # result_image_html
                        gr.update(visible=False),  # preview_image
                        f"Error: {data}. Batch stopped.",  # progress_desc
                        gr.update(visible=False),  # progress_bar
                        gr.update(interactive=True),  # start_button
                        gr.update(interactive=True),  # endless_run_button 
                        gr.update(interactive=False),  # end_graceful_button
                        gr.update(interactive=False),  # force_stop_button
                        gr.update(),  # seed
                        gr.update(visible=False),  # first_frame
                        gr.update(visible=False),  # last_frame
                        gr.update(visible=False),  # extend_button
                        gr.update(visible=False),  # note_message
                        gr.update(visible=True, value=f"Error: {data}. Batch stopped."),  # generation_stats
                        gr.update(visible=True, open=True),  # generation_stats_accordion
                        gr.update(visible=False),  # frame_thumbnails
                        gr.update(value=final_prompt_text, visible=True),  # final_processed_prompt
                        gr.update(visible=True)  # final_prompt_accordion
                    )
                elif generation_interrupted:
                    # Handle normal interruptions
                    yield (
                        gr.update(visible=False),  # result_video
                        gr.update(visible=False),  # result_image_html
                        gr.update(visible=False),  # preview_image
                        "Generation interrupted by user.",  # progress_desc
                        gr.update(visible=False),  # progress_bar
                        gr.update(interactive=True),  # start_button
                        gr.update(interactive=True),  # endless_run_button 
                        gr.update(interactive=False),  # end_graceful_button
                        gr.update(interactive=False),  # force_stop_button
                        gr.update(),  # seed
                        gr.update(visible=False),  # first_frame
                        gr.update(visible=False),  # last_frame
                        gr.update(visible=False),  # extend_button
                        gr.update(visible=False),  # note_message
                        gr.update(visible=True, value="Generation interrupted by user."),  # generation_stats
                        gr.update(visible=True, open=True),  # generation_stats_accordion
                        gr.update(visible=False),  # frame_thumbnails
                        gr.update(value=final_prompt_text, visible=True),  # final_processed_prompt
                        gr.update(visible=True)  # final_prompt_accordion
                    )
                
                elif is_image_output:
                    # For image output, focus on displaying the stats, but don't try to set image value again
                    debug(f"End event for image output: {image_path}")
                    
                    # Match your working code more closely
                    yield (
                        gr.update(visible=False),                           # result_video
                        gr.update(visible=True),                            # result_image_html - KEEP VISIBLE
                        gr.update(visible=False),                           # preview_image
                        f"Generated single image!<br><a href=\"file/{image_path}\" target=\"_blank\">Click here to open full size in new tab.</a><br><code>{image_path}</code>",  # progress_desc
                        gr.update(visible=False),                           # progress_bar
                        gr.update(interactive=True, value="Start Generation"), # start_button
                        gr.update(interactive=True),                        # endless_run_button 
                        gr.update(interactive=False),                       # end_graceful_button
                        gr.update(interactive=False),                       # force_stop_button
                        gr.update(),                                        # seed
                        gr.update(visible=False),                           # first_frame
                        gr.update(visible=False),                           # last_frame
                        gr.update(visible=False),                           # extend_button
                        gr.update(visible=False),                           # note_message
                        gr.update(visible=True, value=f"Generated single image!<br><code>{image_path}</code>\n{last_stats}"),  # generation_stats
                        gr.update(visible=True, open=False),                # generation_stats_accordion
                        gr.update(visible=False),                           # frame_thumbnails
                        gr.update(value=final_prompt_text, visible=True),   # final_processed_prompt
                        gr.update(visible=True)                             # final_prompt_accordion
                    )
                else:
                    # Handle video output completion
                    stats_display = ""
                    if output_filename and os.path.exists(output_filename):
                        stats_display = f"""
### Generation Complete{batch_progress_text}
**Video saved as:** `{os.path.basename(output_filename)}`
{last_stats}
                        """
                    else:
                        stats_display = f"""
### Generation Stopped
No complete output was generated.
{last_stats}
                        """
                        
                    yield (
                        gr.update(value=output_filename if output_filename else None, visible=bool(output_filename)),  # result_video
                        gr.update(visible=False),  # result_image_html
                        gr.update(visible=False),  # preview_image
                        gr.update(visible=False),  # progress_desc
                        gr.update(visible=False),  # progress_bar
                        gr.update(interactive=True),  # start_button
                        gr.update(interactive=True),  # endless_run_button 
                        gr.update(interactive=False),  # end_graceful_button
                        gr.update(interactive=False),  # force_stop_button
                        gr.update(),  # seed
                        gr.update(visible=bool(output_filename)),  # first_frame
                        gr.update(visible=bool(output_filename)),  # last_frame
                        gr.update(visible=bool(output_filename)),  # extend_button
                        gr.update(visible=False),  # note_message
                        gr.update(visible=True, value=stats_display),  # generation_stats
                        gr.update(visible=True, open=False),  # generation_stats_accordion
                        gr.update(visible=bool(output_filename)),  # frame_thumbnails
                        gr.update(value=final_prompt_text, visible=True),  # final_processed_prompt
                        gr.update(visible=True)  # final_prompt_accordion
                    )
                
                # Exit inner loop for this generation
                debug("Inner loop: breaking on 'end' event.")
                break
        
        # --- End of inner stream loop for this generation ---
        debug(f"Inner loop finished for batch item {current_batch_item}. Interrupted={generation_interrupted}")
        
        # Check for early exits in the batch sequence
        if _stop_requested_flag:
            debug("Outer loop: Breaking due to force stop flag.")
            break
            
        # Check endless run checkbox state 
        if endless_run and _graceful_stop_batch_flag:
            debug("Graceful stop requested, ending endless run after current item.")
        
        # Continue to next batch item if there are more and we're not stopping
        if (not endless_run and current_batch_item >= total_batch_count) or _graceful_stop_batch_flag:
            debug("Outer loop: Breaking after finishing planned batches.")
            break
            
        debug(f"Outer loop: Continuing to next batch iteration {current_batch_item + 1}.")
    
    # --- End of outer batch/endless loop ---
    debug("--- Batch/Endless Run Finished ---")
    
    # Final message based on how we exited
    final_message = "Generation sequence finished."
    if _stop_requested_flag: 
        final_message = "Generation sequence force stopped."
    if _graceful_stop_batch_flag: 
        final_message = "Generation sequence stopped gracefully."
    
    # Reset flags for next run
    _stop_requested_flag = False
    _graceful_stop_batch_flag = False
    
    # Perform final unload if needed
    if unload_on_end_flag:
        debug("Unloading models as unload_on_end is set and batch finished.")
        try:
            yield (
                gr.update(), gr.update(), gr.update(),
                f"{final_message} Unloading models...",
                gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=False),
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            )
            model_manager.unload_all_models()
            free_mem = get_cuda_free_memory_gb(gpu)
            mem_status_update = f"Models unloaded. Free VRAM: {free_mem:.1f} GB"
            debug(mem_status_update)
            
            # Final UI state after unload
            yield (
                gr.update(), gr.update(), gr.update(), 
                mem_status_update, 
                gr.update(visible=False), gr.update(interactive=True), 
                gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), 
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            )
        except Exception as e:
            debug(f"Error unloading models at end: {e}")
            yield (
                gr.update(), gr.update(), gr.update(), 
                f"Error during final model unload: {e}", 
                gr.update(visible=False), gr.update(interactive=True), gr.update(interactive=True), 
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), 
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            )
    else:
        # Final UI state without unload
        yield (
            gr.update(), gr.update(), gr.update(), 
            final_message, 
            gr.update(visible=False), gr.update(interactive=True), gr.update(interactive=True), 
            gr.update(interactive=False), gr.update(interactive=False),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), 
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        )
    
    # Clean up stream
    stream = None

# --- Modify Stop Handlers ---

def request_graceful_end():
    """Request a graceful end to the current generation OR the entire batch/endless run."""
    global stream, _graceful_stop_batch_flag
  
    debug("Requesting graceful stop for current generation and batch run.")
    
    # Signal the generator to end gracefully if running
    if stream:
        stream.input_queue.push('graceful_end')
    
    # Signal the batch loop to stop after current generation finishes
    _graceful_stop_batch_flag = True
    
    # Update button states
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
