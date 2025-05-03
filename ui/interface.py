import sys
import json
import random
import os
import contextlib
import gradio as gr
import torch
from ui.style import get_css
from utils.common import debug
from utils.memory_utils import get_cuda_free_memory_gb, gpu


# Hush noisy mpeg output.
@contextlib.contextmanager
def suppress_output_if_not_debug():
    """Context manager to suppress stdout and stderr only when debug is disabled"""
    if debug:
        # In debug mode, don't suppress anything
        yield
    else:
        # Not in debug mode, suppress output
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Open null files
        null_out = open(os.devnull, 'w')
        null_err = open(os.devnull, 'w')

        try:
            # Redirect stdout/stderr to null
            sys.stdout = null_out
            sys.stderr = null_err
            yield
        finally:
            # Restore original stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            # Close null files
            null_out.close()
            null_err.close()

# Ensure the original_video_postprocess is captured correctly before patching
try:
    original_video_postprocess = gr.Video.postprocess
except AttributeError:
     # Fallback if gradio structure changes, unlikely but safe
    original_video_postprocess = lambda self, y: y
    debug("Warning: Could not find original gr.Video.postprocess")


def debug_aware_video_postprocess(self, y):
    """Wrapper around Gradio's Video postprocess that respects the debug flag"""
    with suppress_output_if_not_debug():
        # Ensure the function we call is the original one we saved
        return original_video_postprocess(self, y)

# Patch the postprocess method
gr.Video.postprocess = debug_aware_video_postprocess


def create_interface(model_manager, video_generator, unload_on_end_flag=False):
    """Create and configure the Gradio interface"""
    from ui.callbacks import (
        process, request_graceful_end, force_immediate_stop, update_video_stats,
        switch_mode, show_custom, show_init_color,
        update_overlap_slider, setup_for_extension,
        toggle_init_color_for_backward
    )

    debug("Creating UI interface")

    # Create a wrapper function that has access to the model_manager and video_generator
    def process_wrapper(*args):
        # This needs to be a generator function too
        for values in process(*args, video_generator=video_generator, model_manager=model_manager):
            yield values

    # --- JavaScript Definitions ---
    paste_js = """
async function handlePaste(event, promptElementId, modeElementId, imgInputId, startFrameId, endFrameId, hiddenTextboxId) {
    const items = (event.clipboardData || event.originalEvent.clipboardData).items;
    let imageBlob = null;
    for (const item of items) {
        if (item.type.startsWith('image/')) {
            imageBlob = item.getAsFile();
            break;
        }
    }

    if (imageBlob) {
        event.preventDefault(); // Prevent default paste action
        console.log('Image pasted!');
        const reader = new FileReader();
        reader.onload = function(e) {
            const base64Image = e.target.result;
            const hiddenTextbox = document.getElementById(hiddenTextboxId);
            if (hiddenTextbox) {
                const textarea = hiddenTextbox.querySelector('textarea');
                if (textarea) {
                    const pasteData = JSON.stringify({
                        imageData: base64Image,
                        mode: document.querySelector('#' + modeElementId + ' input[type=radio]:checked')?.value || 'image2video',
                        imgInputHasValue: !!document.querySelector('#' + imgInputId + ' img'),
                        startFrameHasValue: !!document.querySelector('#' + startFrameId + ' img'),
                        endFrameHasValue: !!document.querySelector('#' + endFrameId + ' img')
                    });
                    textarea.value = pasteData;
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    console.log('Sent base64 data to hidden Gradio textbox.');
                } else {
                     console.error('Could not find textarea within hidden Gradio component:', hiddenTextboxId);
                }
            } else {
                 console.error('Could not find hidden Gradio component:', hiddenTextboxId);
            }
        };
        reader.readAsDataURL(imageBlob);
    } else {
        console.log('Pasted content is not an image.');
    }
}

function attachPasteListener(promptElementId, modeElementId, imgInputId, startFrameId, endFrameId, hiddenTextboxId) {
    const promptElement = document.getElementById(promptElementId);
    if (promptElement) {
        const textarea = promptElement.querySelector('textarea');
        if (textarea) {
             textarea.addEventListener('paste', (event) => handlePaste(event, promptElementId, modeElementId, imgInputId, startFrameId, endFrameId, hiddenTextboxId));
             console.log('Paste listener attached to prompt textarea.');
        } else {
            console.error('Could not find textarea within prompt element:', promptElementId);
            setTimeout(() => attachPasteListener(promptElementId, modeElementId, imgInputId, startFrameId, endFrameId, hiddenTextboxId), 500);
        }
    } else {
        console.error('Could not find prompt element:', promptElementId);
        setTimeout(() => attachPasteListener(promptElementId, modeElementId, imgInputId, startFrameId, endFrameId, hiddenTextboxId), 500);
    }
}
"""

    lightbox_css = """
/* Lightbox Structure (omitted for brevity, assumed correct) */
.lightbox-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.8); display: none; justify-content: center; align-items: center; z-index: 1000; cursor: pointer; }
.lightbox-content { position: relative; background: #111; padding: 20px; border-radius: 5px; max-width: 90vw; max-height: 90vh; display: flex; justify-content: center; align-items: center; overflow: hidden; cursor: default; }
.lightbox-content img { display: block; max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain; cursor: pointer; user-select: none; -webkit-user-drag: none; }
.lightbox-close { position: absolute; top: 10px; right: 15px; font-size: 2em; color: white; cursor: pointer; text-shadow: 0 0 5px black; }
.lightbox-controls { position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.6); padding: 5px 10px; border-radius: 5px; display: flex; gap: 10px; }
.lightbox-controls button { background: #555; color: white; border: none; padding: 5px 10px; cursor: pointer; border-radius: 3px; }
.lightbox-controls button:hover { background: #777; }
"""

    lightbox_js = """
let isZoomed = false;
let originalWidth, originalHeight;

function openLightbox(imgElement) {
    const overlay = document.getElementById('lightbox-overlay');
    const contentImg = document.getElementById('lightbox-image');
    if (!overlay || !contentImg || !imgElement) return;
    const src = imgElement.src;
    if (!src || src.includes('placeholder.png') || !src.startsWith('http') && !src.startsWith('file:') && !src.startsWith('data:')) return; // Check for valid src

    contentImg.src = src;
    isZoomed = false;
    contentImg.style.maxWidth = '100%'; contentImg.style.maxHeight = '100%';
    contentImg.style.width = 'auto'; contentImg.style.height = 'auto';
    contentImg.style.cursor = 'zoom-in';

    const tempImg = new Image();
    tempImg.onload = () => { originalWidth = tempImg.naturalWidth; originalHeight = tempImg.naturalHeight; console.log(`Original dims: ${originalWidth}x${originalHeight}`); };
    tempImg.src = src;
    overlay.style.display = 'flex';
}
function closeLightbox() { const overlay = document.getElementById('lightbox-overlay'); if (overlay) { overlay.style.display = 'none'; } }
function toggleZoom() {
    const contentImg = document.getElementById('lightbox-image'); if (!contentImg || !originalWidth || !originalHeight) return;
    if (isZoomed) { contentImg.style.maxWidth = '100%'; contentImg.style.maxHeight = '100%'; contentImg.style.width = 'auto'; contentImg.style.height = 'auto'; contentImg.style.cursor = 'zoom-in'; }
    else { contentImg.style.maxWidth = originalWidth + 'px'; contentImg.style.maxHeight = originalHeight + 'px'; contentImg.style.width = originalWidth + 'px'; contentImg.style.height = originalHeight + 'px'; contentImg.style.cursor = 'zoom-out'; }
    isZoomed = !isZoomed;
}
function handleOverlayClick(event) { if (event.target.id === 'lightbox-overlay') { closeLightbox(); } }
document.addEventListener('keydown', function(event) { if (event.key === 'Escape') { closeLightbox(); } });

// Define handleImageClick globally for listener attachment/removal
function handleImageClick(event) { openLightbox(event.target); }

// Function to add lightbox listeners (global scope needed for block.load)
function addLightboxListeners() {
    document.querySelectorAll('.clickable-image').forEach(elem => {
        const imgTag = elem.querySelector('img');
        if (imgTag) {
            imgTag.removeEventListener('click', handleImageClick); // Remove previous
            imgTag.addEventListener('click', handleImageClick); // Add new
        }
    });
    // console.log('Lightbox click listeners updated.'); // Optional: for debugging
}

// Attach listener function (for JS)
function attachAllListeners(promptId, modeId, imgId, startId, endId, hiddenId) {
     attachPasteListener(promptId, modeId, imgId, startId, endId, hiddenId);
     addLightboxListeners(); // Attach lightbox listeners initially

     // Use MutationObserver to re-attach lightbox listeners when Gradio updates images
    const observerTargetIds = ['result_image_output', 'first_frame_output', 'last_frame_output'];
    observerTargetIds.forEach(id => {
        const targetNode = document.getElementById(id);
        if (targetNode) {
            const observer = new MutationObserver(() => { setTimeout(addLightboxListeners, 150); }); // Use the global function, slight delay
            observer.observe(targetNode, { childList: true, subtree: true });
            // console.log(`MutationObserver attached to #${id}`); // Optional: for debugging
        } else {
            // console.warn(`Could not find node #${id} for MutationObserver, retrying...`); // Optional
            setTimeout(() => { // Retry attachment later
                const laterNode = document.getElementById(id);
                if(laterNode) {
                    const observer = new MutationObserver(() => { setTimeout(addLightboxListeners, 150); });
                    observer.observe(laterNode, { childList: true, subtree: true });
                    // console.log(`MutationObserver attached later to #${id}`); // Optional
                } else {
                    // console.error(`Still could not find node #${id} for observer.`); // Optional
                }
            }, 1500);
        }
    });
}
"""

    # Combine JS
    combined_js = paste_js + "\n\n" + lightbox_js

    # Add CSS parameter to Blocks
    block = gr.Blocks(css=get_css() + lightbox_css, js=combined_js).queue()

    # --- Start Building the UI Layout ---
    with block:
        gr.Markdown('# FramePack Advanced Video Generator')
        hidden_paste_textbox = gr.Textbox(visible=False, elem_id="hidden_paste_box") # For JS communication

        with gr.Row():
            mode_selector = gr.Radio(
                ["image2video", "text2video", "keyframes", "video_extension"],
                value="image2video", label="Mode", elem_classes="mode-selector", elem_id="mode_selector_radio"
            )

        with gr.Row():
            # Left column for inputs
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", value='', lines=3, elem_id="prompt_textbox") # Listener target

                with gr.Accordion("Negative Prompt", open=False):
                    n_prompt = gr.Textbox(label="Negative Prompt - Requires CFG higher than 1.0 to take effect", value="", lines=2)

                input_image = gr.Image(sources=['upload', 'clipboard'], type="numpy", label="Image", elem_classes="input-image-container", elem_id="input_image_component")
                start_frame = gr.Image(sources=['upload', 'clipboard'], type="numpy", label="Start Frame (Optional)", elem_classes="keyframe-image-container", visible=False, elem_id="start_frame_component")

                with gr.Group(visible=False) as keyframes_options:
                    keyframe_weight = gr.Slider(label="Start Frame Influence", minimum=0.0, maximum=1.0, value=0.7, step=0.1, info="Higher values prioritize start frame characteristics")

                end_frame = gr.Image(sources=['upload', 'clipboard'], type="numpy", label="End Frame (Required)", elem_classes="keyframe-image-container", visible=False, elem_id="end_frame_component")

                with gr.Group(visible=False) as video_container:
                    input_video = gr.Video(label="Upload Video to Extend", format="mp4", elem_classes="video-container", include_audio=False, interactive=True, show_share_button=False, show_download_button=True)

                with gr.Group(visible=False) as video_extension_controls:
                    extension_direction = gr.Radio(["Forward", "Backward"], label="Extension Direction", value="Forward", info="Forward extends the end, Backward extends the beginning")
                    extension_frames = gr.Slider(label="Context Frames", minimum=1, maximum=16, value=8, step=1, info="Number of frames to extract from video for continuity")

                aspect_selector = gr.Dropdown(["16:9", "9:16", "1:1", "4:5", "3:2", "2:3", "21:9", "4:3", "Custom..."], label="Aspect Ratio", value="1:1", visible=False)
                custom_w = gr.Number(label="Width", value=768, visible=False)
                custom_h = gr.Number(label="Height", value=768, visible=False)
                init_color = gr.ColorPicker(label="Initial Frame Color", value="#808080", visible=False)

                with gr.Accordion("Generation Parameters", open=True):
                    lock_seed = gr.Checkbox(label="Lock Seed", value=False)
                    seed = gr.Number(label="Seed", value=random.randint(0, 2**32-1), precision=0)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                    video_stats = gr.HTML(value="<div class='stats-box'>Calculating...</div>", label="Approximate Output Length")
                    latent_window_size = gr.Slider(label="Latent Window Size", minimum=2, maximum=33, value=9, step=1)
                    segment_count = gr.Slider(label="Number of Segments", minimum=1, maximum=50, value=5, step=1, info="More segments = longer video")
                    overlap_slider = gr.Slider(label="Frame Overlap (Currently Locked to 8)", minimum=0, maximum=33, value=8, step=1, info="Controls how many frames overlap between sections", interactive=False)
                    trim_percentage = gr.Slider(label="Segment Trim Percentage", minimum=0.0, maximum=1.0, value=0.2, step=0.01, info="Percentage of frames to trim (0.0 = keep all, 1.0 = maximum trim)")
                    gaussian_blur = gr.Slider(label="Gaussian Blur", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=True, info="Apply blur to input images before processing")

                with gr.Accordion("Advanced Model Parameters", open=False):
                    use_teacache = gr.Checkbox(label='Use TeaCache', value=True)
                    # User specified moving unload_on_end here:
                    unload_on_end_state = gr.Checkbox(label='Unload all models when queue is empty', value=unload_on_end_flag, visible=True)
                    llm_encoder_weight = gr.Slider(label="LLM Encoder Weight", minimum=0.0, maximum=5.0, value=1.0, step=0.1, info="0.0 to disable LLM encoder")
                    clip_encoder_weight = gr.Slider(label="CLIP Encoder Weight", minimum=0.0, maximum=5.0, value=1.0, step=0.1, info="0.0 to disable CLIP encoder")
                    clean_latent_weight = gr.Slider(label="Clean Latent Weight", minimum=0.0, maximum=2.0, value=1.0, step=0.01, info="Controls influence of anchor/initial frame")
                    cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.1, info="Must be >1.0 for negative prompts to work")
                    gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.1)
                    rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                    gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB)", minimum=6, maximum=128, value=6, step=0.1)

                with gr.Row():
                    unload_button = gr.Button(value="Unload All Models", variant="secondary")
                    clear_cache_button = gr.Button(value="Clear CUDA Cache", variant="secondary")
                mem_status = gr.Markdown("")

            # Right column for outputs
            with gr.Column(scale=2):
                with gr.Row():
                    start_button = gr.Button(value="Start Generation", elem_classes="start-button")

                # --- NEW Layout for Batch Count ---
                with gr.Row(equal_height=True, variant="compact"): # Use compact variant for less padding
                    with gr.Column(scale=3):
                         gr.Markdown("""
                         **Batch Count**
                         <span style='font-size:0.9em; color:#999;'>Number of videos to generate sequentially.</span>
                         """)
                    with gr.Column(scale=1, min_width=80): # Ensure number input has some minimum width
                        batch_count = gr.Number(
                            value=1, minimum=1, step=1, precision=0,
                            show_label=False, # Label is now in Markdown
                            container=False # Remove container background/border
                        )
                # --- END NEW Layout ---

                # --- NEW Layout for Endless Run ---
                with gr.Row(equal_height=True, variant="compact"):
                    with gr.Column(scale=3):
                         gr.Markdown("""
                         **Endless Run**
                         <span style='font-size:0.9em; color:#999;'>Keep generating until unchecked or stopped.</span>
                         """)
                    with gr.Column(scale=1, min_width=80): # Align checkbox
                        endless_run = gr.Checkbox(
                            value=False,
                            show_label=False, # Label is now in Markdown
                            container=False # Remove container background/border
                        )
                # --- END NEW Layout ---

                # Stop buttons row
                with gr.Row():
                    with gr.Column(scale=1):
                        end_graceful_button = gr.Button(value="End Generation", interactive=False, elem_classes="end-graceful-button")
                    with gr.Column(scale=1):
                        force_stop_button = gr.Button(value="Force Stop", interactive=False, elem_classes="force-stop-button")

                progress_bar = gr.HTML(visible=False)
                progress_desc = gr.Markdown(visible=False)
                preview_image = gr.Image(label="Next Latents", height=200, visible=False)

                result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, elem_classes="result-container", interactive=False, loop=True, visible=False)
                result_image_html = gr.Image(label='Single Frame Image', elem_classes="result-container clickable-image", visible=False, elem_id="result_image_output")
                extend_button = gr.Button(value="Extend This Video", visible=False, elem_classes="extend-button")

                with gr.Accordion("Final Processed Prompt", open=False, visible=False) as final_prompt_accordion:
                    final_processed_prompt_display = gr.Textbox(label="Prompt Sent to Model", lines=4, interactive=False, show_copy_button=True)

                with gr.Accordion("Generation Complete (Expand for details)", open=False, visible=False) as generation_stats_accordion:
                    generation_stats = gr.Markdown(value="", elem_id="generation_stats")

                note_message = gr.Markdown(value="", visible=False, elem_id="sampling_note")

                with gr.Group(visible=False, elem_classes="frame-thumbnail-container") as frame_thumbnails_group:
                    gr.Markdown("### Frame Comparison", elem_classes="frame-header")
                    with gr.Row(elem_classes="frame-thumbnail-row"):
                        first_frame = gr.Image(label="First Frame", elem_classes="frame-thumbnail clickable-image", visible=True, show_download_button=True, container=True, elem_id="first_frame_output")
                        last_frame = gr.Image(label="Last Frame", elem_classes="frame-thumbnail clickable-image", visible=True, show_download_button=True, container=True, elem_id="last_frame_output")

                # Lightbox HTML (doesn't need `with gr.HTML()` wrapper when defining layout)
                gr.HTML("""
                <div id="lightbox-overlay" class="lightbox-overlay" onclick="handleOverlayClick(event)">
                    <div class="lightbox-content">
                        <span class="lightbox-close" onclick="closeLightbox()">×</span>
                        <img id="lightbox-image" src="" alt="Lightbox Image" onclick="toggleZoom()" />
                        <div class="lightbox-controls">
                             <button onclick="toggleZoom()">Toggle Zoom</button>
                             <button onclick="closeLightbox()">Close</button>
                         </div>
                    </div>
                </div>
                """)

        # --- End of UI Layout Definition ---

        # --- Python Callbacks & Event Handlers (Correctly Indented) ---
        def handle_image_paste_data(paste_data_json):
            try:
                if not paste_data_json or paste_data_json.strip() == "":
                    return gr.update(), gr.update(), gr.update()

                paste_data = json.loads(paste_data_json)
                image_data = paste_data.get("imageData")
                mode = paste_data.get("mode")
                end_has_val = paste_data.get("endFrameHasValue")
                start_has_val = paste_data.get("startFrameHasValue")

                if not image_data:
                    return gr.update(), gr.update(), gr.update()

                debug(f"Python: Handling pasted image for mode: {mode}")

                if mode == "image2video":
                    debug("Python: Pasting into image2video input.")
                    return gr.update(value=image_data), gr.update(), gr.update()
                elif mode == "keyframes":
                    if not end_has_val:
                        debug("Python: Pasting into keyframes END frame (empty).")
                        return gr.update(), gr.update(), gr.update(value=image_data)
                    elif not start_has_val:
                        debug("Python: Pasting into keyframes START frame (end has value).")
                        return gr.update(), gr.update(value=image_data), gr.update()
                    else:
                        debug("Python: Pasting into keyframes END frame (replacing).")
                        return gr.update(), gr.update(), gr.update(value=image_data)
                else:
                    debug(f"Python: Image pasted in mode '{mode}', no image input target.")
                    return gr.update(), gr.update(), gr.update()

            except Exception as e:
                debug(f"Error processing pasted image data: {e}")
                # Optionally yield an error message to the user via gr.Info or gr.Warning
                gr.Warning(f"Failed to process pasted image: {e}")
                return gr.update(), gr.update(), gr.update()

        hidden_paste_textbox.input(
            fn=handle_image_paste_data,
            inputs=[hidden_paste_textbox],
            outputs=[input_image, start_frame, end_frame],
            show_progress="hidden" # Hide Gradio's default progress for this silent update
        )

        def unload_all_models():
            """Unload all models completely from memory (CPU and GPU)"""
            try:
                model_manager.unload_all_models()
                free_mem = get_cuda_free_memory_gb(gpu)
                return f"All models completely unloaded. Free VRAM: {free_mem:.1f} GB"
            except Exception as e:
                return f"Error unloading models: {str(e)}"

        def clear_cuda_cache():
            """Clear CUDA cache to free fragmented memory"""
            try:
                torch.cuda.empty_cache()
                free_mem = get_cuda_free_memory_gb(gpu)
                return f"CUDA cache cleared. Free VRAM: {free_mem:.1f} GB"
            except Exception as e:
                return f"Error clearing CUDA cache: {str(e)}"

        unload_button.click(fn=unload_all_models, outputs=mem_status)
        clear_cache_button.click(fn=clear_cuda_cache, outputs=mem_status)

        latent_window_size.change(update_overlap_slider, inputs=[latent_window_size], outputs=[overlap_slider])
        latent_window_size.change(update_video_stats, inputs=[latent_window_size, segment_count, overlap_slider], outputs=[video_stats])
        segment_count.change(update_video_stats, inputs=[latent_window_size, segment_count, overlap_slider], outputs=[video_stats])
        overlap_slider.change(update_video_stats, inputs=[latent_window_size, segment_count, overlap_slider], outputs=[video_stats])

        mode_selector.change(
            switch_mode, inputs=[mode_selector],
            outputs=[input_image, start_frame, end_frame, aspect_selector, custom_w, custom_h, keyframes_options, video_container, video_extension_controls, gaussian_blur]
        )
        mode_selector.change(show_init_color, inputs=[mode_selector], outputs=[init_color])
        aspect_selector.change(show_custom, inputs=[aspect_selector], outputs=[custom_w, custom_h])
        extension_direction.change(fn=toggle_init_color_for_backward, inputs=[extension_direction, mode_selector], outputs=[init_color])
        extend_button.click(
            fn=setup_for_extension, inputs=[result_video],
            outputs=[mode_selector, input_video, input_image, start_frame, end_frame, video_container, video_extension_controls]
        )

        # Define process inputs list (ensure order matches process function)
        ips = [
            mode_selector, input_image, start_frame, end_frame, aspect_selector, custom_w, custom_h,
            prompt, n_prompt, seed, latent_window_size, segment_count, steps, cfg, gs, rs,
            gpu_memory_preservation, use_teacache, lock_seed, init_color, keyframe_weight,
            input_video, extension_direction, extension_frames, overlap_slider, trim_percentage,
            gaussian_blur, llm_encoder_weight, clip_encoder_weight, clean_latent_weight,
            batch_count, endless_run, unload_on_end_state, # Added batch/endless/unload controls
        ]

        # Define process outputs list (ensure order matches process function yield)
        output_list = [
            result_video, result_image_html, preview_image, progress_desc, progress_bar,
            start_button, end_graceful_button, force_stop_button, seed,
            first_frame, last_frame, extend_button, note_message,
            generation_stats, generation_stats_accordion, frame_thumbnails_group,
            final_processed_prompt_display, final_prompt_accordion
        ]

        prompt.submit(fn=process_wrapper, inputs=ips, outputs=output_list)
        start_button.click(fn=process_wrapper, inputs=ips, outputs=output_list)
        end_graceful_button.click(fn=request_graceful_end, outputs=[end_graceful_button, force_stop_button])
        force_stop_button.click(fn=force_immediate_stop, outputs=[end_graceful_button, force_stop_button])

        # --- Initialization on Load ---
        # Initialize video stats
        block.load(fn=update_video_stats, inputs=[latent_window_size, segment_count, overlap_slider], outputs=[video_stats])
        # Attach JS listeners (Try using 'js' instead of '_js')
        block.load(
            None, [], [],
            # Use the combined attachAllListeners function
            js="() => { attachAllListeners('prompt_textbox', 'mode_selector_radio', 'input_image_component', 'start_frame_component', 'end_frame_component', 'hidden_paste_box'); }"
        )

    # Return the fully constructed block
    return block
