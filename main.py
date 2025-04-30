import argparse
import os
import torch
from models.model_loader import ModelManager
from core.generation import VideoGenerator
from ui.interface import create_interface
from utils.memory_utils import get_cuda_free_memory_gb, gpu
from utils.common import debug, setup_debug

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Video Generation Application")
    parser.add_argument('--share', action='store_true', help="Enable Gradio sharing")
    parser.add_argument("--server", type=str, default='0.0.0.0', help="Server address to bind to")
    parser.add_argument("--port", type=int, required=False, help="Port to run the server on")
    parser.add_argument("--inbrowser", action='store_true', help="Open in browser automatically")
    parser.add_argument("--debug", action='store_true', help="Enable debug output")
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA files, seperate by commas. Append ':0.85' to name to adjust weight, default 1.0 weight.") 
    parser.add_argument("--lora-weight", type=float, default=1.0, help="Weight for LoRA (0.0-1.0)")
    parser.add_argument("--lora-skip-fail", action='store_true', help="If set, skip lora failures instead of aborting generation")
    parser.add_argument("--lora-diagnose", action='store_true', help="Diagnose LoRA weights/loadability and exit (no generation)")
    args = parser.parse_args()


    # --- BEGIN BACKEND SWITCH ---
    try:
        # The error involves cusolver, so let's try preferring 'magma'
        torch.backends.cuda.preferred_linalg_library("magma")
        debug("Set preferred CUDA linear algebra library to 'magma'")
    except Exception as e:
        debug(f"Could not set preferred CUDA linalg library: {e}. Using default.")
    # --- END BACKEND SWITCH ---

    
    # Multi-LoRA configuration
    from utils.lora_utils import parse_lora_arg, lora_diagnose
    lora_arg = args.lora or ""
    lora_configs = parse_lora_arg(lora_arg) if lora_arg else []
    lora_skip_fail = args.lora_skip_fail

    if getattr(args, 'lora_diagnose', False):
        for conf in lora_configs:
            lora_diagnose(conf.path)
        exit(0)
    
    # Set up debugging based on arguments (True by default for now during development)
    setup_debug(args.debug)
    debug(f"Command line arguments: {args}")
    
    # Set environment variables
    os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(
        os.path.join(os.path.dirname(__file__), './hf_download')))
    debug(f"HF_HOME set to: {os.environ['HF_HOME']}")
    
    # HF login if needed
    try:
        from diffusers_helper.hf_login import login
        debug("Trying to login to Hugging Face...")
        # Check if HF_TOKEN environment variable is set
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            login(hf_token)
            debug("Logged in to Hugging Face successfully")
        else:
            debug("No HF_TOKEN environment variable found, skipping login")
    except Exception as e:
        debug(f"Error during Hugging Face login: {e}")
        debug("Continuing without login...")
    
    # Create output directory
    outputs_folder = './outputs/'
    os.makedirs(outputs_folder, exist_ok=True)
    debug(f"Output folder set to: {outputs_folder}")
    
    # Check VRAM availability
    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 60
    debug(f'Free VRAM: {free_mem_gb:.2f} GB')
    debug(f'High-VRAM Mode: {high_vram}')
    
    # Initialize model manager and load models
    model_manager = ModelManager(
        high_vram=high_vram,
        lora_configs=lora_configs,
        lora_skip_fail=lora_skip_fail
    )
    
    try:
        # Load all models
        debug("Loading models...")
        model_manager.load_all_models()
        debug("Models loaded successfully")
        
        # Create video generator
        debug("Initializing video generator...")
        video_generator = VideoGenerator(model_manager, outputs_folder=outputs_folder)
        
        # Create and launch Gradio interface
        debug("Setting up user interface...")
        interface = create_interface(model_manager, video_generator)
        
        # Launch the interface
        debug(f"Launching Gradio interface on {args.server}, port={args.port}, share={args.share}")
        interface.launch(
            server_name=args.server,
            server_port=args.port,
            share=args.share,
            inbrowser=args.inbrowser,
        )
        
    except Exception as e:
        debug(f"Error during startup: {e}")
        import traceback
        debug(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
