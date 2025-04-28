# utils/prompt_parser.py
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
from utils.common import debug

class PromptProcessor:
    """Base class for prompt processors"""
    def __init__(self, name: str):
        self.name = name
        
    def process(self, prompt_text: str) -> str:
        """Process a prompt and return the modified prompt"""
        return prompt_text
        
    def extract_data(self, prompt_text: str) -> Dict[str, Any]:
        """Extract data from a prompt without modifying it"""
        return {}

def parse_sequential_prompts(prompt_text: str) -> List[str]:
    """
    Parse a prompt text with semicolons into a list of sequential prompts.
    
    Args:
        prompt_text: Full prompt text with semicolons
        
    Returns:
        List of individual prompts, stripped of whitespace
    """
    if not prompt_text:
        return []
        
    # Split by semicolons and strip whitespace
    prompts = [p.strip() for p in prompt_text.split(';') if p.strip()]
    
    if not prompts:
        return [prompt_text.strip()]  # Return original if splitting resulted in empty list
        
    debug(f"Parsed {len(prompts)} sequential prompts: {prompts}")
    return prompts

def apply_prompt_processors(prompt_text: str, processors: List[PromptProcessor]) -> Tuple[str, Dict[str, Any]]:
    """
    Apply a series of prompt processors to extract data and transform a prompt.
    This generic function allows for multiple types of processing in sequence.
    
    Args:
        prompt_text: Original prompt text
        processors: List of PromptProcessor objects to apply
        
    Returns:
        Tuple of (modified_prompt, extracted_data)
    """
    modified_prompt = prompt_text
    extracted_data = {}
    
    for processor in processors:
        # Extract data without modifying the prompt
        processor_data = processor.extract_data(modified_prompt)
        extracted_data[processor.name] = processor_data
        
        # Apply the processor to modify the prompt
        modified_prompt = processor.process(modified_prompt)
        
    return modified_prompt, extracted_data

# Example specialized processor for sequential prompts
class SequentialPromptProcessor(PromptProcessor):
    """Process prompts with semicolons for sequential generation"""
    def __init__(self):
        super().__init__("sequential")
        
    def extract_data(self, prompt_text: str) -> Dict[str, Any]:
        """Extract sequential prompts without modifying the original"""
        prompts = parse_sequential_prompts(prompt_text)
        return {
            "prompts": prompts, 
            "is_sequential": len(prompts) > 1
        }
        
    def process(self, prompt_text: str) -> str:
        """For sequential processing, we keep the original prompt"""
        return prompt_text
