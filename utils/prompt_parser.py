# utils/prompt_parser.py
import random
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
import re
import os
from pathlib import Path # <-- ADD THIS LINE
from utils.common import debug
# Import LoRAConfig from lora_utils
from utils.lora_utils import LoRAConfig

# --- Custom Exception ---
class WildcardFileNotFoundError(Exception):
    """Custom exception for missing wildcard files."""
    def __init__(self, filename):
        self.filename = filename
        super().__init__(f"Wildcard file not found: {filename}")

# --- Base Class ---
class PromptProcessor:
    """Base class for prompt processors"""
    def __init__(self, name: str):
        self.name = name

    def process(self, prompt_text: str, seed: Optional[int] = None) -> str:
        """Process a prompt and return the modified prompt"""
        return prompt_text

    def extract_data(self, prompt_text: str) -> Dict[str, Any]:
        """Extract data from a prompt without modifying it"""
        return {}

# --- NEW: Wildcard Processor ---
class WildcardProcessor(PromptProcessor):
    """
    Replaces __wildcard_name__ placeholders with random lines from files
    in the ./wildcards/ directory. Requires seed for deterministic selection.
    """
    WILDCARD_REGEX = re.compile(r"__([a-zA-Z0-9_-]+)__")
    WILDCARD_DIR = Path("./wildcards")

    def __init__(self):
        super().__init__("wildcard")
        # Ensure wildcard directory exists
        self.WILDCARD_DIR.mkdir(exist_ok=True)

    def _get_wildcard_options(self, name: str) -> List[str]:
        """Loads lines from a wildcard file."""
        filepath = self.WILDCARD_DIR / f"{name.lower()}.txt"
        if not filepath.is_file():
            raise WildcardFileNotFoundError(str(filepath))

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            if not lines:
                debug(f"Warning: Wildcard file '{filepath}' is empty or contains only whitespace.")
                # --- USER VISIBLE PRINT ---
                print(f"⚠️ Warning: Wildcard file '{filepath.name}' is empty.")
                # --------------------------
                return []
            return lines
        except Exception as e:
            debug(f"Error reading wildcard file {filepath}: {e}")
            # --- USER VISIBLE PRINT ---
            print(f"❌ Error reading wildcard file '{filepath.name}': {e}")
            # --------------------------
            # Treat as empty if read fails
            return []

    def process(self, prompt_text: str, seed: Optional[int] = None) -> str:
        """Processes wildcards in the prompt."""
        if seed is None:
            # Use system time if no seed provided, less reproducible
            rng = random.Random()
            debug("Warning: WildcardProcessor processing without a specific seed.")
        else:
            rng = random.Random(seed)

        processed_prompt = prompt_text
        # Process iteratively to handle cases where wildcard result might contain another wildcard
        processed_something = True
        iterations = 0
        max_iterations = 10 # Safety break

        while processed_something and iterations < max_iterations:
            processed_something = False
            matches = list(self.WILDCARD_REGEX.finditer(processed_prompt))
            if not matches:
                break

            # Process one match per iteration to simplify replacement
            match = matches[0]
            wildcard_name = match.group(1)
            debug(f"Processing wildcard: __{wildcard_name}__")

            try:
                options = self._get_wildcard_options(wildcard_name)
                if options:
                    replacement = rng.choice(options)
                    debug(f"Selected: '{replacement}'")
                else:
                    replacement = "" # Replace with empty if file is empty or not found via error path
                    debug(f"Wildcard '{wildcard_name}' resulted in empty replacement.")

                # Replace only the first occurrence found in this iteration
                processed_prompt = processed_prompt[:match.start()] + replacement + processed_prompt[match.end():]
                processed_something = True

            except WildcardFileNotFoundError as wf_err:
                # Propagate this error up to be caught by generate_video
                raise wf_err
            except Exception as e:
                debug(f"Unexpected error processing wildcard '{wildcard_name}': {e}")
                # Replace with empty string to avoid infinite loops on errors
                processed_prompt = processed_prompt[:match.start()] + "" + processed_prompt[match.end():]
                processed_something = True # Mark as processed to continue, but it's an error state

            iterations += 1

        if iterations >= max_iterations:
            debug(f"Warning: Wildcard processing reached max iterations ({max_iterations}). Potential recursive wildcards?")

        return processed_prompt

# --- NEW: Randomizer Processor ---
class RandomizerProcessor(PromptProcessor):
    """
    Processes {option1|option2} syntax, including multi-select (N$$)
    and custom separators ($$sep$$). Requires seed.
    """
    # Regex to find the outermost curly braces non-greedily
    # This is simple and might fail on complex nesting, but works for common cases.
    RANDOMIZER_REGEX = re.compile(r"\{([^\{\}]+?)\}")
    # Regex to parse count and separator inside the braces
    # Optional count (digits) followed by $$
    COUNT_REGEX = re.compile(r"^\s*(\d+)\$\$")
    # Optional separator ($$ any non-$$ chars $$)
    SEP_REGEX = re.compile(r"\s*\$\$(.*?)\$\$")

    def __init__(self):
        super().__init__("randomizer")

    def _process_match(self, match: re.Match, rng: random.Random) -> str:
        """Processes a single {..} block."""
        content = match.group(1)

        # 1. Parse Count (N$$)
        count = 1
        count_match = self.COUNT_REGEX.match(content)
        if count_match:
            count = int(count_match.group(1))
            # Remove the count specifier from the content
            content = content[count_match.end():].lstrip()
            debug(f"Randomizer: Found count specifier N={count}")

        # 2. Parse Separator ($$sep$$)
        separator = ", " # Default separator
        sep_match = self.SEP_REGEX.match(content)
        if sep_match:
            separator = sep_match.group(1)
            # Remove the separator specifier from the content
            content = content[sep_match.end():].lstrip()
            debug(f"Randomizer: Found separator specifier: '{separator}'")

        # 3. Split options
        options = [opt.strip() for opt in content.split('|') if opt.strip()]
        debug(f"Randomizer: Options: {options}")

        if not options:
            debug("Randomizer: No valid options found, returning empty string.")
            return ""

        # 4. Validate count
        num_options = len(options)
        actual_count = max(1, min(count, num_options))
        if actual_count != count:
            debug(f"Randomizer: Adjusted count from {count} to {actual_count} (num options: {num_options})")

        # 5. Select and Join
        selected_options = rng.sample(options, actual_count)
        result = separator.join(selected_options)
        debug(f"Randomizer: Selected: {selected_options}, Result: '{result}'")
        return result

    def process(self, prompt_text: str, seed: Optional[int] = None) -> str:
        """Processes randomizer blocks in the prompt."""
        if seed is None:
            rng = random.Random()
            debug("Warning: RandomizerProcessor processing without a specific seed.")
        else:
            rng = random.Random(seed)

        processed_prompt = prompt_text
        iterations = 0
        max_iterations = 50 # Safety break for potential complex replacements

        # Keep processing as long as we find matches
        while iterations < max_iterations:
            match = self.RANDOMIZER_REGEX.search(processed_prompt)
            if not match:
                break # No more matches found

            try:
                replacement = self._process_match(match, rng)
                # Replace the found match block with the processed result
                processed_prompt = processed_prompt[:match.start()] + replacement + processed_prompt[match.end():]
            except Exception as e:
                 debug(f"Error processing randomizer block '{match.group(0)}': {e}")
                 # Replace the block with empty string on error to prevent loops
                 processed_prompt = processed_prompt[:match.start()] + "" + processed_prompt[match.end():]

            iterations += 1

        if iterations >= max_iterations:
            debug(f"Warning: Randomizer processing reached max iterations ({max_iterations}). Potential complex prompt structure?")

        # Final cleanup of potential extra spaces
        processed_prompt = re.sub(r'\s{2,}', ' ', processed_prompt).strip()
        return processed_prompt

class LoraPromptProcessor(PromptProcessor):
    """
    Extracts LoRA specifications like [path/to/lora:weight] or [/abs/path/lora]
    from the prompt text and removes them.
    """
    # Regex explanation:
    # \[             # Match opening square bracket
    # (              # Start capture group 1 (path)
    #   .+?          # Match one or more characters (.+?) non-greedily
    # )              # End capture group 1
    # (?:            # Start non-capturing group for optional weight
    #   :            # Match the colon separator
    #   (\d+\.?\d*) # Capture group 2 (weight): one or more digits, optional decimal part
    # )?             # End non-capturing group, make it optional
    # \]             # Match closing square bracket
    LORA_REGEX = re.compile(r"\[(.+?)(?::(\d+\.?\d*))?\]")
    LORA_DIRS = [Path("./loras/"), Path("./models/Lora/")]
    
    def __init__(self):
        super().__init__("lora")
        # Ensure potential directories exist (optional, helps user setup)
        # for lora_dir in self.LORA_DIRS:
        #     lora_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_path(self, path_fragment: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Tries to find the LoRA file, adding .safetensors if needed.
        Checks absolute path first, then relative paths in common LoRA dirs.
        Returns (found_path, error_message)
        """
        path_fragment = path_fragment.strip()

        # 1. Check if the provided path is absolute and exists
        potential_path_abs = Path(path_fragment)
        potential_path_abs_st = Path(f"{path_fragment}.safetensors")

        if potential_path_abs.is_file():
            return str(potential_path_abs.resolve()), None
        if potential_path_abs_st.is_file():
             return str(potential_path_abs_st.resolve()), None

        # 2. Check relative paths in predefined LoRA directories
        for lora_dir in self.LORA_DIRS:
            potential_path_rel = lora_dir / path_fragment
            potential_path_rel_st = lora_dir / f"{path_fragment}.safetensors"

            if potential_path_rel.is_file():
                 return str(potential_path_rel.resolve()), None
            if potential_path_rel_st.is_file():
                 return str(potential_path_rel_st.resolve()), None

        # 3. If not found anywhere
        error_msg = f"LoRA file not found for spec: '{path_fragment}' (checked absolute and relative in {self.LORA_DIRS})"
        debug(f"[LoraParser] {error_msg}")
        # --- USER VISIBLE PRINT ---
        print(f"⚠️ Warning: Could not find LoRA file specified as '{path_fragment}'. Will skip loading this LoRA.")
        # --------------------------
        return None, error_msg # Indicate not found and provide reason

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
                lora_configs.append(LoRAConfig(path=path_fragment, weight=weight, error=error_msg or "File not found"))


        debug(f"[LoraParser] Extracted LoRA configs: {lora_configs}")
        return {"lora_configs": lora_configs}

    def process(self, prompt_text: str, seed: Optional[int] = None) -> str: # Added seed arg for consistency
        """Removes the LoRA specifications from the prompt text."""
        # Seed is not used here, but included for consistent interface
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


def apply_prompt_processors(prompt_text: str, processors: List[PromptProcessor], seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Apply a series of prompt processors to extract data and transform a prompt.
    Enforces the order: Wildcard -> Randomizer -> LoRA -> Sequential.
    Passes the seed to processors that need it.

    Args:
        prompt_text: Original prompt text
        processors: List of PromptProcessor objects to apply (order in list doesn't matter)
        seed: Optional random seed for processors like Wildcard and Randomizer.

    Returns:
        Tuple of (modified_prompt, extracted_data)

    Raises:
        WildcardFileNotFoundError: If a required wildcard file is not found.
    """
    modified_prompt = prompt_text
    extracted_data = {}

    # Define the strict processing order
    processing_order = [
        WildcardProcessor,
        RandomizerProcessor,
        LoraPromptProcessor,
        SequentialPromptProcessor
        # Add other processor types here in their desired order if needed
    ]

    # Create a map of processor types provided in the input list
    processor_map = {type(p): p for p in processors}

    for processor_type in processing_order:
        processor = processor_map.get(processor_type)
        if processor:
            debug(f"Applying processor: {processor.name}")
            try:
                # Extract data based on the *current* state of modified_prompt
                processor_data = processor.extract_data(modified_prompt)
                if processor_data: # Only add if data was extracted
                    extracted_data[processor.name] = processor_data

                # --- CORRECTED CALL ---
                # Only pass seed if the processor type needs it
                if isinstance(processor, (WildcardProcessor, RandomizerProcessor)):
                    modified_prompt = processor.process(modified_prompt, seed=seed)
                else:
                    modified_prompt = processor.process(modified_prompt) # No seed needed
                # --- END CORRECTION ---

                debug(f"After {processor.name} processing - Prompt: '{modified_prompt}'")
                if processor.name in extracted_data:
                     debug(f"  Extracted Data: {extracted_data[processor.name]}")

            except WildcardFileNotFoundError as e:
                 # Let this specific error bubble up to be caught in generate_video
                 debug(f"WildcardFileNotFoundError caught during processing: {e}")
                 raise e
            except Exception as e:
                 # Catch other errors like the TypeError we saw
                 debug(f"Error during {processor.name} processing: {e}")
                 print(f"⚠️ Error applying prompt processor '{processor.name}': {e}")
                 # Re-raise other errors to stop processing if they occur
                 raise e

    debug(f"Final result - Modified Prompt: '{modified_prompt}', All Extracted Data: {extracted_data}")
    return modified_prompt, extracted_data
