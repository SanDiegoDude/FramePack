# ui/interface.py
import gradio as gr
import random
import torch
import os
import contextlib
import sys
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

import gradio as gr
original_video_postprocess = gr.Video.postprocess

def debug_aware_video_postprocess(self, y):
    """Wrapper around Gradio's Video postprocess that respects the debug flag"""
    with suppress_output_if_not_debug():
        return original_video_postprocess(self, y)

gr.Video.postprocess = debug_aware_video_postprocess


def create_interface(model_manager, video_generator):
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
    
    block = gr.Blocks(css=get_css()).queue()
    
    with block:
        gr.Markdown('# FramePack Advanced Video Generator')
        
        # Mode selector across the top
        with gr.Row():
            mode_selector = gr.Radio(
                ["image2video", "text2video", "keyframes", "video_extension"],
                value="image2video",
                label="Mode",
                elem_classes="mode-selector"
            )
        
        with gr.Row():
            # Left column for inputs
            with gr.Column(scale=2):
                # Prompts at the top of left column
                prompt = gr.Textbox(label="Prompt", value='', lines=3)
                with gr.Accordion("Negative Prompt", open=False):
                    n_prompt = gr.Textbox(
                        label="Negative Prompt - Requires CFG higher than 1.0 to take effect",
                        value="",
                        lines=2
                    )
                
                # Input sections for different modes
                input_image = gr.Image(sources='upload', type="numpy", label="Image", elem_classes="input-image-container")
                
                # Keyframes mode controls
                start_frame = gr.Image(sources='upload', type="numpy", label="Start Frame (Optional)",
                                     elem_classes="keyframe-image-container", visible=False)
                
                with gr.Group(visible=False) as keyframes_options:
                    keyframe_weight = gr.Slider(
                        label="Start Frame Influence",
                        minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                        info="Higher values prioritize start frame characteristics"
                    )
                
                end_frame = gr.Image(sources='upload', type="numpy", label="End Frame (Required)",
                                   elem_classes="keyframe-image-container", visible=False)
                
                # Wrap the video in a custom div for scrolling
                with gr.Group(visible=False) as video_container:
                    input_video = gr.Video(
                        label="Upload Video to Extend",
                        format="mp4",
                        elem_classes="video-container",
                        include_audio=False,
                        interactive=True,
                        show_share_button=False,
                        show_download_button=True
                    )
                
                # Then, controls in a separate group below
                with gr.Group(visible=False) as video_extension_controls:
                    extension_direction = gr.Radio(
                        ["Forward", "Backward"],
                        label="Extension Direction",
                        value="Forward",
                        info="Forward extends the end, Backward extends the beginning"
                    )
                    
                    extension_frames = gr.Slider(
                        label="Context Frames",
                        minimum=1,
                        maximum=16,
                        value=8,
                        step=1,
                        info="Number of frames to extract from video for continuity"
                    )
                
                # Text2video specific controls
                aspect_selector = gr.Dropdown(
                    ["16:9", "9:16", "1:1", "4:5", "3:2", "2:3", "21:9", "4:3", "Custom..."],
                    label="Aspect Ratio",
                    value="1:1",
                    visible=False
                )
                custom_w = gr.Number(label="Width", value=768, visible=False)
                custom_h = gr.Number(label="Height", value=768, visible=False)
                init_color = gr.ColorPicker(label="Initial Frame Color", value="#808080", visible=False)
                
                # Generation Parameters accordion
                with gr.Accordion("Generation Parameters", open=True):
                    lock_seed = gr.Checkbox(label="Lock Seed", value=False)
                    seed = gr.Number(label="Seed", value=random.randint(0, 2**32-1), precision=0)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                    
                    # Video stats display
                    video_stats = gr.HTML(
                        value="<div class='stats-box'>Calculating...</div>",
                        label="Approximate Output Length"
                    )
                    
                    latent_window_size = gr.Slider(
                        label="Latent Window Size",
                        minimum=2, maximum=33, value=9, step=1
                    )
                    
                    segment_count = gr.Slider(
                        label="Number of Segments",
                        minimum=1, maximum=50, value=5, step=1,
                        info="More segments = longer video"
                    )
                    
                    overlap_slider = gr.Slider(
                        label="Frame Overlap (Currently Locked to 8)",
                        minimum=0, maximum=33, value=8, step=1,
                        info="Controls how many frames overlap between sections",
                        interactive=False  # Set to non-interactive
                    )
                    
                    trim_percentage = gr.Slider(
                        label="Segment Trim Percentage",
                        minimum=0.0, maximum=1.0, value=0.2, step=0.01,
                        info="Percentage of frames to trim (0.0 = keep all, 1.0 = maximum trim)"
                    )
                    
                    # Blur control
                    gaussian_blur = gr.Slider(
                        label="Gaussian Blur",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.01,
                        visible=True,
                        info="Apply blur to input images before processing"
                    )
                
                # Advanced Model Parameters
                with gr.Accordion("Advanced Model Parameters", open=False):
                    use_teacache = gr.Checkbox(label='Use TeaCache', value=True)
                    gpu_memory_preservation = gr.Slider(
                        label="GPU Inference Preserved Memory (GB)",
                        minimum=6, maximum=128, value=6, step=0.1
                    )
                    
                    llm_encoder_weight = gr.Slider(
                        label="LLM Encoder Weight",
                        minimum=0.0,
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        info="0.0 to disable LLM encoder"
                    )
                    
                    clip_encoder_weight = gr.Slider(
                        label="CLIP Encoder Weight",
                        minimum=0.0,
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        info="0.0 to disable CLIP encoder"
                    )
                    
                    clean_latent_weight = gr.Slider(
                        label="Clean Latent Weight",
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.01,
                        info="Controls influence of anchor/initial frame"
                    )
                    
                    cfg = gr.Slider(
                        label="CFG Scale", 
                        minimum=1.0, maximum=32.0, value=1.0, step=0.1,
                        info="Must be >1.0 for negative prompts to work"
                    )
                    
                    gs = gr.Slider(
                        label="Distilled CFG Scale", 
                        minimum=1.0, maximum=32.0, value=10.0, step=0.1
                    )
                    
                    rs = gr.Slider(
                        label="CFG Re-Scale",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.01
                    )
                
                # Memory management
                with gr.Row():
                    unload_button = gr.Button(value="Unload All Models", variant="secondary")
                    clear_cache_button = gr.Button(value="Clear CUDA Cache", variant="secondary")
                
                mem_status = gr.Markdown("")
            
            # Right column for outputs
            with gr.Column(scale=2):
                # Start/End buttons at top of right column
                with gr.Row():
                    start_button = gr.Button(value="Start Generation", elem_classes="start-button")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        end_graceful_button = gr.Button(value="End Generation", 
                                                        interactive=False, 
                                                        elem_classes="end-graceful-button")
                    with gr.Column(scale=1):
                        force_stop_button = gr.Button(value="Force Stop", 
                                                      interactive=False, 
                                                      elem_classes="force-stop-button")
                
                # Progress indicators
                progress_bar = gr.HTML(visible=False)
                progress_desc = gr.Markdown(visible=False)
                preview_image = gr.Image(label="Next Latents", height=200, visible=False)
                
                # Results section
                result_video = gr.Video(
                    label="Finished Frames", 
                    autoplay=True, 
                    show_share_button=False,
                    elem_classes="result-container",
                    interactive=False,  # Add this line
                    loop=True,
                    visible=False # Hide on startup
                )
                result_image_html = gr.Image(
                    label='Single Frame Image', 
                    elem_classes="result-container", 
                    visible=False
                )
                
                extend_button = gr.Button(value="Extend This Video", visible=False, elem_classes="extend-button")
                
                
                with gr.Accordion("Generation Complete (Expand for details)", open=False, visible=False) as generation_stats_accordion:
                    generation_stats = gr.Markdown(
                        value="",
                        elem_id="generation_stats"
                    )
                
                # Note message
                note_message = gr.Markdown(
                    value="",
                    visible=False,
                    elem_id="sampling_note"
                )
                
                # First/Last frame displays in their own group
                with gr.Group(visible=False, elem_classes="frame-thumbnail-container") as frame_thumbnails_group:
                    with gr.Row(elem_classes="frame-thumbnail-row"):
                        first_frame = gr.Image(
                            label="First Frame",
                            elem_classes="frame-thumbnail",
                            visible=True
                        )
                        last_frame = gr.Image(
                            label="Last Frame",
                            elem_classes="frame-thumbnail",
                            visible=True
                        )
        
        # Memory management functions
        def unload_all_models():
            """Unload all models completely from memory (CPU and GPU)"""
            try:
                model_manager.unload_all_models()
                free_mem = get_cuda_free_memory_gb(gpu)
                return f"All models completely unloaded from memory. Free VRAM: {free_mem:.1f} GB"
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
        
        # Connect button callbacks for memory management
        unload_button.click(fn=unload_all_models, outputs=mem_status)
        clear_cache_button.click(fn=clear_cuda_cache, outputs=mem_status)
        
        # Connect callbacks for UI functionality
        latent_window_size.change(
            update_overlap_slider,
            inputs=[latent_window_size],
            outputs=[overlap_slider]
        )
        
        latent_window_size.change(
            update_video_stats,
            inputs=[latent_window_size, segment_count, overlap_slider],
            outputs=[video_stats]
        )
        
        segment_count.change(
            update_video_stats,
            inputs=[latent_window_size, segment_count, overlap_slider],
            outputs=[video_stats]
        )
        
        overlap_slider.change(
            update_video_stats,
            inputs=[latent_window_size, segment_count, overlap_slider],
            outputs=[video_stats]
        )
        
        mode_selector.change(
            switch_mode,
            inputs=[mode_selector],
            outputs=[
                input_image,
                start_frame,
                end_frame,
                aspect_selector,
                custom_w,
                custom_h,
                keyframes_options,
                video_container,          # Changed from video_extension_options
                video_extension_controls, # Added this new component
                gaussian_blur
            ]
        )
        
        mode_selector.change(
            show_init_color,
            inputs=[mode_selector],
            outputs=[init_color]
        )
        
        aspect_selector.change(
            show_custom,
            inputs=[aspect_selector],
            outputs=[custom_w, custom_h],
        )

        extension_direction.change(
            fn=toggle_init_color_for_backward,
            inputs=[extension_direction, mode_selector],
            outputs=[init_color]
        )
        
        extend_button.click(
            fn=setup_for_extension,
            inputs=[result_video],
            outputs=[
                mode_selector,
                input_video,
                input_image,
                start_frame,
                end_frame,
                video_container,          # Changed from video_extension_options
                video_extension_controls  # Added this new component
            ]
        )
        
        # Define process inputs
        ips = [
            mode_selector,
            input_image,
            start_frame,
            end_frame,
            aspect_selector,
            custom_w,
            custom_h,
            prompt,
            n_prompt,
            seed,
            latent_window_size,
            segment_count,
            steps,
            cfg,
            gs,
            rs,
            gpu_memory_preservation,
            use_teacache,
            lock_seed,
            init_color,
            keyframe_weight,
            input_video,
            extension_direction,
            extension_frames,
            overlap_slider,
            trim_percentage,
            gaussian_blur,
            llm_encoder_weight,
            clip_encoder_weight,
            clean_latent_weight,
        ]
        
        # Define output list including first/last frame images
        output_list = [
            result_video,      # 0
            result_image_html, # 1
            preview_image,     # 2
            progress_desc,     # 3
            progress_bar,      # 4
            start_button,      # 5
            end_graceful_button,# 6 - replaced end_button
            force_stop_button, # 7 - new button
            seed,              # 8
            first_frame,       # 9
            last_frame,        # 10
            extend_button,     # 11
            note_message,      # 12
            generation_stats,   # 13
            generation_stats_accordion, #14
            frame_thumbnails_group #15
        ]
        
        prompt.submit(
            fn=process_wrapper,
            inputs=ips,
            outputs=output_list,
        )
        
        start_button.click(
            fn=process_wrapper,
            inputs=ips,
            outputs=output_list,
        )

        end_graceful_button.click(fn=request_graceful_end, 
                                 outputs=[end_graceful_button, force_stop_button])
        force_stop_button.click(fn=force_immediate_stop, 
                               outputs=[end_graceful_button, force_stop_button])
        
        # Initialize the video stats on page load
        block.load(
            fn=update_video_stats,
            inputs=[latent_window_size, segment_count, overlap_slider],
            outputs=[video_stats]
        )
    
    return block
