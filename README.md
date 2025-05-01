![image](https://github.com/user-attachments/assets/d24d0989-e734-45ab-b11e-464e81c5c461)


# FramePack Advanced Video Generator ✨

This project is a significantly refactored and enhanced version of the original [FramePack](https://github.com/lllyasviel/FramePack) video generation tool. It maintains the core next-frame-section prediction concept while introducing a modular architecture, improved memory management, multi-LoRA support, and various UI/UX enhancements.

The goal of this fork is to provide a more stable, extensible, and feature-rich platform for experimenting with FramePack-based video generation.

## Key Features

*   **Modular Architecture:** Codebase refactored into distinct UI, Core Generation, Model Management, and Utility layers for easier maintenance and development.
*   **Multi-LoRA Support:** Load and apply multiple LoRA adapters simultaneously with adjustable weights via the command line **and** dynamically via the prompt.
*   **Dynamic LoRA Loading:** Specify LoRAs directly in the prompt text using `[path:weight]` syntax for on-the-fly experimentation. These LoRAs are loaded for the current generation and unloaded automatically if not requested again.
*   **LoRA Diagnostics:** Inspect LoRA compatibility and structure without running a full generation using the `--lora-diagnose` flag.
*   **Enhanced UI:** Includes first/last frame previews, drag-and-drop support for frames, a dedicated "Extend Video" button, and improved layout.
*   **Improved Memory Management:** Refactored model manager aims for more robust handling of model loading/unloading, especially in low-VRAM scenarios.
*   **Video Extension Modes:** Supports both forward (extending the end) and backward (extending the beginning) video generation.
*   **Video Compatibility:** Improved output MP4 compatibility for wider playback support.
*   **Sequential Prompting:** Control generation segments individually by separating distinct prompts with a semicolon (`;`) in the main prompt box. Each prompt will be applied to subsequent segments of the video generation process.

## Requirements

*   **GPU:** NVIDIA GPU (RTX 30XX series or newer recommended) with support for fp16/bf16. At least **6GB VRAM** is required for basic operation, more recommended for longer videos or higher resolutions.
*   **OS:** Linux or Windows.
*   **Storage:** Sufficient disk space for models (can exceed 30GB).

## Installation

It is **strongly recommended** to install FramePack Advanced within a Python virtual environment (`venv`) to avoid conflicts with system packages.

1.  **Create and Activate Virtual Environment:**
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

2.  **Install PyTorch with CUDA:**
    *   Visit the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    *   Select the appropriate settings for your system (Stable, OS, Package: Pip, Language: Python, CUDA version matching your driver).
    *   Copy and run the provided installation command. It will look something like this ( **DO NOT** use this exact command, get the correct one from the website):
        ```bash
        # Example command - GET YOURS FROM PYTORCH.ORG!
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

3.  **Install Project Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Optional Performance Enhancements (Recommended):**

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

**Startup LoRA Configuration (`--lora`):**

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


**Dynamic LoRA Loading (Prompt Syntax):**

In addition to loading LoRAs at startup, you can dynamically load specific LoRAs for individual generations directly within the main **Prompt** text box.

*   **Syntax:** Use square brackets `[]` around the LoRA path. A weight can optionally be added after a colon `:`.
    *   `[path/to/your/lora]` (Loads with default weight 1.0)
    *   `[path/to/your/lora:0.75]` (Loads with weight 0.75)
*   **Multiple LoRAs:** You can include multiple dynamic LoRAs anywhere in your prompt.
    ```
    [style_anime:0.6] [char_catgirl] A catgirl exploring a neon city [environment_cyberpunk:0.9]
    ```
*   **Path Details:**
    *   Paths can be **absolute** (e.g., `[C:/Users/Me/loras/mylora]`, `[/home/user/loras/mylora]`) or **relative**.
    *   Relative paths are searched relative to a `./loras/` directory within the project folder by default.
    *   The `.safetensors` file extension is **optional** and will be added automatically if omitted.
*   **Automatic Cleaning:** The `[lora:weight]` syntax will be automatically removed from the prompt text before it's sent to the text encoders. The cleaned prompt is used for generation.
*   **Lifecycle:** Dynamically loaded LoRAs are active *only* for the generation requested by that specific prompt. They are automatically unloaded afterwards if not included in the next prompt, helping manage resources. LoRAs loaded via the `--lora` command-line flag remain active throughout the session.
*   **Case Sensitivity:** LoRA path matching follows operating system rules:
    *   **Windows:** Case-insensitive (e.g., `[MyLoRa]` can match `mylora.safetensors`).
    *   **Linux/macOS:** Case-sensitive (e.g., `[MyLoRa]` only matches `MyLoRa.safetensors`).
    *   **Best Practice:** Always use the exact filename case as it appears on your file system for consistency and to avoid issues.
*   **Feedback:** The application console will print messages indicating which dynamic LoRAs are being loaded or if loading fails (e.g., file not found). TQDM progress bars may show for multiple loads.

**Example Prompt:**

```
[style_illustration:0.8] A dragon flying over a castle; the dragon landing on a turret [char_dragon_detailed] [setting_fantasy_castle:0.5]
```

This prompt will:
1.  Attempt to load `style_illustration` (weight 0.8), `char_dragon_detailed` (weight 1.0), and `setting_fantasy_castle` (weight 0.5) dynamically.
2.  Use these LoRAs *in addition* to any LoRAs loaded via the `--lora` startup argument.
3.  Use the cleaned prompts `"A dragon flying over a castle"` and `"the dragon landing on a turret"` for sequential generation.
4.  Unload the three dynamic LoRAs after generation completes (unless the next prompt requests them again).
```


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
