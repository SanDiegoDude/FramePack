# utils/prompt_parser.py
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
import re
import os
from pathlib import Path # <-- ADD THIS LINE
from utils.common import debug
# Import LoRAConfig from lora_utils
from utils.lora_utils import LoRAConfig

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

class LoraPromptProcessor(PromptProcessor):
    """
    Extracts LoRA specifications like [path/to/lora:weight] or [/abs/path/lora]
    from the prompt text and removes them.
    """
    # Regex explanation:
    # \[             # Match opening square bracket
    # (              # Start capture group 1 (path)
    #   [^:\]]+     # Match one or more characters that are NOT ':' or ']'
    # )              # End capture group 1
    # (?:            # Start non-capturing group for optional weight
    #   :            # Match the colon separator
    #   (\d+\.?\d*) # Capture group 2 (weight): one or more digits, optional decimal part
    # )?             # End non-capturing group, make it optional
    # \]             # Match closing square bracket
    LORA_REGEX = re.compile(r"\[([^:\]]+?)(?::(\d+\.?\d*))?\]")

    def __init__(self):
        super().__init__("lora")

    def _normalize_path(self, path_fragment: str) -> Optional[str]:
        """Adds .safetensors if missing and checks existence."""
        path_fragment = path_fragment.strip()
        # Try absolute path first
        potential_path_abs = Path(path_fragment)
        potential_path_abs_st = Path(f"{path_fragment}.safetensors")

        # Try relative path (assuming a standard 'loras' folder might exist)
        # Adjust './loras/' if your LoRA directory is different or configurable
        lora_dir = Path('./loras/') 
        potential_path_rel = lora_dir / path_fragment
        potential_path_rel_st = lora_dir / f"{path_fragment}.safetensors"


        # Check existence in order: abs, abs+.st, rel, rel+.st
        if potential_path_abs.is_file():
            return str(potential_path_abs.resolve())
        if potential_path_abs_st.is_file():
             return str(potential_path_abs_st.resolve())
        if potential_path_rel.is_file():
             return str(potential_path_rel.resolve())
        if potential_path_rel_st.is_file():
             return str(potential_path_rel_st.resolve())

        debug(f"[LoraParser] LoRA file not found for spec: '{path_fragment}' (checked . and .safetensors)")
        return None # Indicate not found

    def extract_data(self, prompt_text: str) -> Dict[str, Any]:
        """Extracts LoRAConfig objects from the prompt."""
        lora_configs = []
        matches = self.LORA_REGEX.finditer(prompt_text)
        for match in matches:
            path_fragment = match.group(1)
            weight_str = match.group(2)

            normalized_path = self._normalize_path(path_fragment)
            if normalized_path:
                weight = float(weight_str) if weight_str else 1.0
                lora_configs.append(LoRAConfig(path=normalized_path, weight=weight))
            else:
                # Store the failure attempt for potential error reporting later
                lora_configs.append(LoRAConfig(path=path_fragment, weight=1.0, error=f"File not found"))


        debug(f"[LoraParser] Extracted LoRA configs: {lora_configs}")
        return {"lora_configs": lora_configs}

    def process(self, prompt_text: str) -> str:
        """Removes the LoRA specifications from the prompt text."""
        cleaned_prompt = self.LORA_REGEX.sub("", prompt_text).strip()
        # Clean up potential extra spaces left by removal
        cleaned_prompt = re.sub(r'\s{2,}', ' ', cleaned_prompt)
        debug(f"[LoraParser] Cleaned prompt: '{cleaned_prompt}'")
        return cleaned_prompt


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
        # If splitting results in nothing, but original had content, return original
        original_stripped = prompt_text.strip()
        return [original_stripped] if original_stripped else []

    # debug(f"Parsed {len(prompts)} sequential prompts: {prompts}") # Debug moved to generation.py
    return prompts


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
        """For sequential processing, we keep the original prompt for splitting later"""
        # The actual splitting happens in generate_video after LoRAs are handled
        return prompt_text


# --- Updated apply_prompt_processors ---
def apply_prompt_processors(prompt_text: str, processors: List[PromptProcessor]) -> Tuple[str, Dict[str, Any]]:
    """
    Apply a series of prompt processors to extract data and transform a prompt.
    Ensures LoRA processing happens first if included.

    Args:
        prompt_text: Original prompt text
        processors: List of PromptProcessor objects to apply

    Returns:
        Tuple of (modified_prompt, extracted_data)
    """
    modified_prompt = prompt_text
    extracted_data = {}

    # Ensure LoraPromptProcessor runs first if present
    lora_processor = next((p for p in processors if isinstance(p, LoraPromptProcessor)), None)
    other_processors = [p for p in processors if not isinstance(p, LoraPromptProcessor)]

    # Process LoRAs first
    if lora_processor:
        debug(f"Applying LoRA processor: {lora_processor.name}")
        processor_data = lora_processor.extract_data(modified_prompt)
        extracted_data[lora_processor.name] = processor_data
        modified_prompt = lora_processor.process(modified_prompt) # Clean the prompt
        debug(f"After LoRA processing - Cleaned Prompt: '{modified_prompt}', Data: {processor_data}")


    # Process remaining processors on the (potentially cleaned) prompt
    for processor in other_processors:
        debug(f"Applying processor: {processor.name}")
        # Extract data based on the current state of modified_prompt
        processor_data = processor.extract_data(modified_prompt)
        extracted_data[processor.name] = processor_data
        # Process modifies the prompt further (if the processor does so)
        modified_prompt = processor.process(modified_prompt)
        debug(f"After {processor.name} processing - Prompt: '{modified_prompt}', Data: {processor_data}")


    debug(f"Final result - Modified Prompt: '{modified_prompt}', All Extracted Data: {extracted_data}")
    return modified_prompt, extracted_data
