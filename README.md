![image](https://github.com/user-attachments/assets/5134a2b2-4b46-41bc-880b-b75c16d13756)




# FramePack Advanced Video Generator ✨

This project is a significantly refactored and enhanced version of the original [FramePack](https://github.com/lllyasviel/FramePack) video generation tool. It maintains the core next-frame-section prediction concept while introducing a modular architecture, improved memory management, multi-LoRA support, dynamic prompt processing (wildcards, randomizers), and various UI/UX enhancements.

The goal of this fork is to provide a more stable, extensible, and feature-rich platform for experimenting with FramePack-based video generation.

## Key Features

*   **Modular Architecture:** Codebase refactored into distinct UI, Core Generation, Model Management, and Utility layers for easier maintenance and development.
*   **Dynamic Prompt Processing:**
    *   **Wildcards:** Use `__wildcard_name__` syntax to randomly select lines from corresponding `.txt` files in the `./wildcards/` directory. Uses the main generation seed for deterministic selection.
    *   **Randomizers:** Use `{option1 | option2}` syntax for random selection. Supports multi-select (`N$$`) and custom separators (`$$sep$$`), also tied to the generation seed.
*   **Multi-LoRA Support:** Load and apply multiple LoRA adapters simultaneously with adjustable weights via the command line **and** dynamically via the prompt.
*   **Dynamic LoRA Loading:** Specify LoRAs directly in the prompt text using `[path:weight]` syntax for on-the-fly experimentation *after* wildcard/randomizer processing. These LoRAs are loaded for the current generation and unloaded automatically if not requested again.
*   **LoRA Diagnostics:** Inspect LoRA compatibility and structure without running a full generation using the `--lora-diagnose` flag.
*   **Enhanced UI:** Includes first/last frame previews, drag-and-drop support for frames, a dedicated "Extend Video" button, display of the final processed prompt, and improved layout.
*   **Improved Memory Management:** Refactored model manager aims for more robust handling of model loading/unloading, especially in low-VRAM scenarios.
*   **Video Extension Modes:** Supports both forward (extending the end) and backward (extending the beginning) video generation.
*   **Video Compatibility:** Improved output MP4 compatibility for wider playback support.
*   **Sequential Prompting:** Control generation segments individually by separating distinct prompts with a semicolon (`;`) in the main prompt box. Each prompt will be applied to subsequent segments of the video generation process.
*   **Batch Count & Endless Generation:**
    *   **Batch Count:** Process multiple videos sequentially with the same settings.
    *   **Endless Run:** Generate videos continuously until manually stopped.
*   **Memory Management:** Automatically unload models when idle with the `--unload-on-end` flag, ideal for server environments.

![image](https://github.com/user-attachments/assets/880a8535-a066-4f5a-8b94-9c44366dd29e)


## Requirements

*   **GPU:** NVIDIA GPU (RTX 30XX series or newer recommended) with support for fp16/bf16. At least **6GB VRAM** is required for basic operation, more recommended for longer videos or higher resolutions.
*   **OS:** Linux or Windows.
*   **Storage:** Sufficient disk space for models (can exceed 30GB) and wildcard files.

## Installation

It is **strongly recommended** to install FramePack Advanced within a Python virtual environment (`venv`) to avoid conflicts with system packages.

1.  **Create Wildcards Directory (if needed):**
    Before installing requirements, create a directory named `wildcards` in the root of the project folder. This is where your wildcard `.txt` files will live.
    ```bash
    # In the project's root directory
    mkdir wildcards
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Navigate to the project directory
    cd /path/to/FramePackAdvanced

    # Create venv (e.g., named 'venv')
    python -m venv venv

    # Activate venv
    # Linux/macOS:
    source venv/bin/activate
    # Windows (cmd):
    .\venv\Scripts\activate.bat
    # Windows (PowerShell):
    .\venv\Scripts\Activate.ps1
    ```
    *(You should see `(venv)` preceding your command prompt)*

3.  **Install PyTorch with CUDA:**
    *   Visit the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    *   Select the appropriate settings for your system (Stable, OS, Package: Pip, Language: Python, CUDA version matching your driver).
    *   Copy and run the provided installation command. It will look something like this ( **DO NOT** use this exact command, get the correct one from the website):
        ```bash
        # Example command - GET YOURS FROM PYTORCH.ORG!
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

4.  **Install Project Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Install Optional Performance Enhancements (Recommended):**

    *   **Triton:** (Required for Flash Attention, significantly speeds up attention layers on compatible GPUs)
        ```bash
        # Linux:
        pip install triton

        # Windows (Requires C++ Build Tools, see Triton docs for details):
        pip install triton-windows
        ```
    *   **SageAttention:** (Alternative attention mechanism)
        ```bash
        pip install sageattention
        ```

## Usage

Launch the Gradio interface using `main.py`:

```bash
python main.py [OPTIONS]
```

**Common Command-Line Options:**

*   `--inbrowser`: Launch internet browser to the UI automatically.
*   `--port <number>`: Run on a specific network port.
*   `--debug`: Enable detailed console logging for troubleshooting.
*   `--lora <lora_list>`: Load one or more LoRA adapters **at startup** (these remain active for the session, see below).
*   `--lora-skip-fail`: If loading multiple LoRAs, skip any that fail instead of stopping.
*   `--lora-diagnose`: Load specified LoRAs, print info, and exit (no generation).
*   `--unload-on-end`: Unload models from GPU and CPU memory when generation queue is empty, freeing resources when idle.

**Prompt Processing Order:**

When you enter text into the **Prompt** box, it undergoes several processing steps *before* being sent to the text encoders. Understanding this order is crucial:

1.  **Wildcards (`__name__`)**: Replaced with random lines from `./wildcards/name.txt`.
2.  **Randomizers (`{options}`)**: Options are randomly selected based on syntax (`N$$`, `$$sep$$`).
3.  **LoRA Extraction (`[path:weight]`)**: LoRA syntax is detected and removed; LoRAs are prepared for loading.
4.  **Sequential Splitting (`;`)**: If semicolons remain, the prompt is split for sequential generation.

The final, processed prompt used by the model can be viewed in the **"Final Processed Prompt"** accordion in the UI after generation.

**Wildcard Support (`__wildcard__`)**

You can insert dynamic content into your prompts using wildcard files.

*   **Syntax:** Use double underscores around a name: `__wildcard_name__`.
*   **File Location:** The system looks for a file named `wildcard_name.txt` (lowercase) inside the `./wildcards/` directory in your project root.
*   **File Content:** Create a plain text file (`.txt`). Each line in the file is treated as a potential replacement option. Empty lines or lines with only whitespace are ignored.
*   **Processing:** When the prompt is processed, each `__wildcard_name__` instance is replaced by one randomly selected line from the corresponding file. This selection uses the main **Seed** value for reproducibility.
*   **Error Handling:** If a specified wildcard file (e.g., `./wildcards/my_colors.txt`) does not exist, generation will **stop**, and an error message will be displayed in the console and UI.
*   **Example:**
    *   Create `./wildcards/artists.txt` with lines like `Monet`, `Van Gogh`, `Picasso`.
    *   Prompt: `A painting by __artists__ in a __style__ style.`
    *   If `./wildcards/style.txt` also exists, the final prompt might become (depending on the seed): `A painting by Van Gogh in a impressionist style.`

**Prompt Randomization (`{options}`)**

Inject randomness into parts of your prompt using curly braces `{}` and pipes `|`.

*   **Basic Syntax:** Separate choices with a pipe `|`. One option will be randomly selected.
    *   Example: `A photograph of a {dog|cat|rabbit}` might become `A photograph of a cat`.
*   **Multi-Select (`N$$`)**: Choose *N* distinct options. Place `N$$` at the *very beginning* inside the braces (before any separator definition).
    *   Example: `{2$$ red|green|blue|yellow}` might become `green, blue` (order depends on seed).
    *   If `N` is greater than the number of options, all options will be selected. Default is `1$$` (select one) if omitted.
*   **Custom Separator (`$$sep$$`)**: Define the text used to join selected options. Place `$$separator_text$$` at the beginning, *after* the multi-select syntax if used.
    *   Example: `{2$$ and also $$ red|green|blue}` might become `blue and also red`.
    *   The default separator if omitted is a comma and space: `, `.
*   **Combined Example:** `{3$$ with $$ option A|option B|option C|option D|option E}` - Selects 3 distinct options and joins them with ` with `.
*   **Processing:** Random selection uses the main **Seed** value. This happens *after* wildcards are processed, so a wildcard result can become one of the options within the braces.
*   **Nesting:** LoRA syntax `[path:weight]` or wildcards `__name__` *inside* the braces will only be processed further if that specific option containing them is randomly selected.

**Startup LoRA Configuration (`--lora`)**

The `--lora` argument accepts a comma-separated list of LoRA file paths to be loaded when the application starts. These LoRAs will remain active for all generations during the session unless the application is restarted. Append `:<weight>` to a path to specify a custom weight (default is `1.0`).

```bash
 # Load a single LoRA with default weight
 python main.py --lora "/path/to/style.safetensors" ...

 # Load multiple LoRAs with default weights
 python main.py --lora "/path/to/style.safetensors,/path/to/char.safetensors" ...

 # Load multiple LoRAs with custom weights
 python main.py --lora "/path/to/style.safetensors:0.7,/path/to/char.safetensors:0.9" ...

 # Load one default, one custom weight
 python main.py --lora "/path/to/style.safetensors,/path/to/char.safetensors:0.5" ...

 # Diagnose LoRAs without generating
 python main.py --lora "/path/to/style.safetensors,/path/to/maybe_broken.safetensors" --lora-diagnose --lora-skip-fail
```

*(Note: The old `--lora-weight` argument has been removed.)*

**Dynamic LoRA Loading (Prompt Syntax)**

In addition to loading LoRAs at startup, you can dynamically load specific LoRAs for individual generations directly within the main **Prompt** text box. This processing step happens *after* Wildcards and Randomizers.

*   **Syntax:** Use square brackets `[]` around the LoRA path. A weight can optionally be added after a colon `:`.
    *   `[path/to/your/lora]` (Loads with default weight 1.0)
    *   `[path/to/your/lora:0.75]` (Loads with weight 0.75)
*   **Multiple LoRAs:** You can include multiple dynamic LoRAs anywhere in your prompt.
    ```
    [style_anime:0.6] [char_catgirl] A catgirl exploring a neon city [environment_cyberpunk:0.9]
    ```
*   **Path Details:**
    *   Paths can be **absolute** (e.g., `[C:/Users/Me/loras/mylora]`, `[/home/user/loras/mylora]`) or **relative**.
    *   Relative paths are searched within configured LoRA directories (e.g., `./loras/`, `./models/Lora/`). See `utils/prompt_parser.py` for the current list.
    *   The `.safetensors` file extension is **optional** and will be added automatically if omitted.
*   **Automatic Cleaning:** The `[lora:weight]` syntax will be automatically removed from the prompt text before it's sent to the text encoders.
*   **Lifecycle:** Dynamically loaded LoRAs are active *only* for the generation requested by that specific prompt. They are automatically unloaded afterwards if not included in the next prompt, helping manage resources. LoRAs loaded via the `--lora` command-line flag remain active throughout the session.
*   **Case Sensitivity:** LoRA path matching follows operating system rules:
    *   **Windows:** Case-insensitive (e.g., `[MyLoRa]` can match `mylora.safetensors`).
    *   **Linux/macOS:** Case-sensitive (e.g., `[MyLoRa]` only matches `MyLoRa.safetensors`).
    *   **Best Practice:** Always use the exact filename case as it appears on your file system for consistency and to avoid issues.
*   **Feedback:** The application console will print messages indicating which dynamic LoRAs are being loaded or if loading fails (e.g., file not found). TQDM progress bars may show for multiple loads.

**Example Prompt Combining Features:**

```
A {close up|medium shot} of __character__ {dancing|fighting|meditating} [style_cinematic:0.7]
in a __location__. {2$$ wearing sunglasses| wearing a hat| wearing a sombrero}.
```

This prompt will:
1.  Replace `__character__` and `__location__` using files from `./wildcards/`.
2.  Randomly select either `close up` or `medium shot`.
3.  Randomly select `dancing`, `fighting`, or `meditating`.
4.  Randomly select *two* options between `wearing sunglasses` and `wearing a hat` and `wearing a sombrero` (because of `2$$`).
5.  Attempt to load `style_cinematic` LoRA dynamically with weight 0.7.
6.  Use the final resulting text (e.g., "A close up of Luke Skywalker fighting wearing sunglasses, wearing a sombrero in a desert temple.") for generation.

**Batch & Endless Generation**

The interface provides three ways to run multiple generations:

* **Batch Count:** Set a number in the input field next to the Start Generation button to automatically run that many generations sequentially. Each generation uses the same prompt and settings (but with a new random seed unless "Lock Seed" is checked).

* **Endless Run:** Click this button instead of Start Generation to continuously create videos until manually stopped. This is useful for exploring variations with the same prompt.

* **Controlling Generations:**
  * **End Generation:** Gracefully finishes the current generation and stops the batch/endless sequence.
  * **Force Stop:** Immediately terminates the current generation and sequence.

**Server Mode Example**

To run FramePack on a server with automatic resource cleanup when idle:

```
python main.py --server 0.0.0.0 --port 7860 --unload-on-end
```
This configuration will free GPU and CPU memory whenever the generation queue becomes empty, making it ideal for shared computing environments.

## Modes & Workarounds

*   **Image-to-Video:** Standard mode, uses an input image and prompt.
*   **Text-to-Video:** Generates video from only a text prompt.
    *   *Workaround:* Starts with a solid color frame by default (configurable via the "Initial Frame Color" picker). Generating directly from noise often produces poor results currently.
*   **Keyframes:** Attempts to interpolate between an optional start frame and a required end frame.
    *   *Limitation:* Quality can be lower and less coherent compared to Image-to-Video. If no start frame is provided, it may start with a grey/colored frame similar to Text-to-Video.
*   **Video Extension:** Takes an existing video and extends it.
    *   *Forward:* Uses the last frame of the input video as the starting point for Image-to-Video generation.
    *   *Backward:* Uses the first frame of the input video as the *target* end frame for Keyframe generation (effectively generating *towards* the start of the video). May exhibit a fade-in from a grey/color frame due to the keyframing mechanism. Increasing the "Segment Trim Percentage" can reduce this fade.

## Known Issues & Limitations

*   **Txt2Vid / Backward Extension Start:** As noted above, these modes initialize with a basic frame rather than pure noise, which is a workaround. The "Initial Frame Color" picker helps for Txt2Vid. Backward Extension relies on Keyframing and inherits its limitations.
*   **LoRA Loading Errors** Some Hunyuan LoRAs won't load. You can try running the lora diagnostic tool which checks basic compatibility, but doesn't detect issues with missing network rank data (yet).
*   **Complex Randomizer Nesting:** Very complex nested `{}` structures might not parse correctly. Keep randomization blocks relatively straightforward.

## Attribution

This project builds heavily upon the original FramePack research and codebase by Lvmin Zhang.

*   **Original FramePack Project:** [https://github.com/lllyasviel/FramePack](https://github.com/lllyasviel/FramePack)
*   **Original Paper:** ["Packing Input Frame Context in Next-Frame Prediction Models for Video Generation"](https://lllyasviel.github.io/frame_pack_gitpage/)

Please refer to the original project for foundational concepts and examples.

## Cite Original Work

If you use the core concepts or models from FramePack in your research, please cite the original paper:

```bibtex
@article{zhang2025framepack,
    title={Packing Input Frame Contexts in Next-Frame Prediction Models for Video Generation},
    author={Lvmin Zhang and Maneesh Agrawala},
    journal={Arxiv},
    year={2025}
}
```
