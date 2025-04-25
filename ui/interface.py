# ui/interface.py
import gradio as gr
from ui.style import get_css
from utils.common import debug

def create_interface(model_manager, video_generator):
    """Create and configure the Gradio interface"""
    # Import UI callbacks when implemented
    # from ui.callbacks import process, end_process, update_video_stats, etc.
    
    # For now, just create a placeholder interface
    debug("Creating initial placeholder UI")
    
    block = gr.Blocks(css=get_css()).queue()
    
    with block:
        gr.Markdown("# Video Generation UI (Placeholder)")
        
        with gr.Row():
            gr.Markdown("This is a placeholder UI. Implementation coming soon!")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Available Models")
                for model_name in ["text_encoder", "vae", "transformer", "image_encoder"]:
                    if hasattr(model_manager, model_name) and getattr(model_manager, model_name) is not None:
                        gr.Markdown(f"✅ {model_name}")
                    else:
                        gr.Markdown(f"❌ {model_name}")
                        
        with gr.Row():
            with gr.Column():
                unload_button = gr.Button(value="Unload All Models", variant="secondary")
            
            with gr.Column():
                clear_cache_button = gr.Button(value="Clear CUDA Cache", variant="secondary")
                
        mem_status = gr.Markdown("")
        
        def unload_all_models():
            """Unload all models from GPU to free memory"""
            try:
                model_manager.unload_all_models()
                from utils.memory_utils import get_cuda_free_memory_gb, gpu
                free_mem = get_cuda_free_memory_gb(gpu)
                return f"All models unloaded from GPU. Free VRAM: {free_mem:.1f} GB"
            except Exception as e:
                return f"Error unloading models: {str(e)}"
        
        def clear_cuda_cache():
            """Clear CUDA cache to free fragmented memory"""
            try:
                torch.cuda.empty_cache()
                from utils.memory_utils import get_cuda_free_memory_gb, gpu
                free_mem = get_cuda_free_memory_gb(gpu)
                return f"CUDA cache cleared. Free VRAM: {free_mem:.1f} GB"
            except Exception as e:
                return f"Error clearing CUDA cache: {str(e)}"
        
        # Connect button callbacks
        unload_button.click(fn=unload_all_models, outputs=mem_status)
        clear_cache_button.click(fn=clear_cuda_cache, outputs=mem_status)
    
    return block
