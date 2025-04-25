class VideoGenerator:
    def __init__(self, model_manager, output_folder='./outputs/'):
        self.model_manager = model_manager
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
    def prepare_inputs(self, input_image, prompt, n_prompt, cfg, 
                        gaussian_blur_amount=0.0, llm_weight=1.0, clip_weight=1.0):
        # Process inputs and prepare for generation
        pass
        
    def generate_video(self, config, callback=None):
        # Main generation function, broken down into helper methods
        # This replaces the monolithic worker function
        pass
        
    def _process_image_to_video(self, params, callback):
        # Image2Video specific logic
        pass
        
    def _process_text_to_video(self, params, callback):
        # Text2Video specific logic
        pass
        
    def _process_keyframes(self, params, callback):
        # Keyframe interpolation logic
        pass
        
    def _process_video_extension(self, params, callback):
        # Video extension logic
        pass
