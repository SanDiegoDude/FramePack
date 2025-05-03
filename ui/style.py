# ui/style.py
def get_css():
    """Return CSS styling for the UI"""
    return """
    .gr-box, .gr-image, .gr-video {
        border: 2px solid orange !important;
        border-radius: 8px !important;
        margin-bottom: 16px;
        background: #222 !important;
    }
    /* Mode Selector */
    .mode-selector label span {
        font-size: 1.1em !important;
        padding: 8px 12px !important;
    }
    /* Start Button (Green) */
    .start-button button {
        width: 100% !important;
        background: linear-gradient(to bottom, #28a745, #218838) !important;
        border-color: #1e7e34 !important;
        color: white !important;
        font-size: 1.2em !important;
        padding: 16px !important;
        margin-bottom: 10px !important;
        border-radius: 5px !important;
        text-shadow: 1px 1px 1px rgba(0,0,0,0.3);
    }
    .start-button button:disabled {
        background: linear-gradient(to bottom, #5a8f69, #4d7a5a) !important; /* Desaturated green */
        border-color: #40664a !important;
        color: #ccc !important;
        cursor: not-allowed !important;
    }
    /* Endless Run Button (Blue) */
    .endless-run-button button {
        width: 100% !important;
        background: linear-gradient(to bottom, #3575D3, #2955A3) !important; 
        border-color: #1F4890 !important;
        color: white !important;
        font-size: 1.2em !important;
        padding: 16px !important;
        margin-bottom: 10px !important;
        border-radius: 5px !important;
        text-shadow: 1px 1px 1px rgba(0,0,0,0.3);
    }
    
    .endless-run-button button:disabled {
        background: linear-gradient(to bottom, #496F9E, #37517A) !important;
        border-color: #2F4368 !important;
        color: #ccc !important;
        cursor: not-allowed !important;
    }
    
    .endless-run-button button:hover:not(:disabled) {
         background: linear-gradient(to bottom, #4185E6, #3064B7) !important;
         border-color: #223D75 !important;
    }
    
    .start-button button:hover:not(:disabled) {
         background: linear-gradient(to bottom, #2ebf4f, #24973e) !important;
         border-color: #1a702f !important;
    }

    /* End Graceful Button (Yellow) */
    .end-graceful-button button {
        width: 100% !important;
        background: linear-gradient(to bottom, #ffc107, #e0a800) !important; /* Amber yellow gradient */
        border-color: #d39e00 !important;
        color: #333 !important; /* Darker text for yellow */
        font-size: 1.1em !important;
        padding: 10px !important;
        border-radius: 5px !important;
    }
    .end-graceful-button button:disabled {
        background: linear-gradient(to bottom, #a68b3a, #8c732f) !important; /* Desaturated yellow */
        border-color: #796429 !important;
        color: #888 !important;
        cursor: not-allowed !important;
    }
     .end-graceful-button button:hover:not(:disabled) {
         background: linear-gradient(to bottom, #ffd03a, #e6b20a) !important;
         border-color: #c79200 !important;
     }


    /* Force Stop Button (Crimson/Red) */
    .force-stop-button button {
        width: 100% !important;
        background: linear-gradient(to bottom, #dc3545, #c82333) !important; /* Red gradient */
        border-color: #bd2130 !important;
        color: white !important;
        font-size: 1.1em !important;
        padding: 10px !important;
        border-radius: 5px !important;
        text-shadow: 1px 1px 1px rgba(0,0,0,0.3);
    }
    .force-stop-button button:disabled {
        background: linear-gradient(to bottom, #9a535a, #824750) !important; /* Desaturated red */
        border-color: #703b42 !important;
        color: #ccc !important;
        cursor: not-allowed !important;
    }
    .force-stop-button button:hover:not(:disabled) {
        background: linear-gradient(to bottom, #e44d5a, #d12b3a) !important;
        border-color: #b01f2c !important;
    }
    
/* Frame Thumbnails - maximize height 460px, side by side */
.frame-thumbnail-container {
    margin-top: 5px !important;
    margin-bottom: 5px !important;
    height: 460px !important;       /* Increase from 350px */
    max-height: 460px !important;
    overflow: visible !important;
}

.frame-thumbnail-row {
    display: flex !important;
    gap: 20px !important;
    max-height: 460px !important;
}

.frame-thumbnail {
    flex: 1 1 0% !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    background: #222 !important;
    padding: 5px !important;
    overflow: hidden !important;
    height: 100% !important;
}

.frame-thumbnail img, .frame-thumbnail .gr-image img {
    max-height: 440px !important;   /* Slightly smaller than container */
    max-width: 100% !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
    display: block !important;
    margin: 0 auto !important;
}
    
    /* Position extend button next to video */
    .extend-button button {
        margin-top: 5px !important;
        margin-bottom: 15px !important;
        background-color: #0078D7 !important; /* Microsoft blue */
        border-color: #0078D7 !important;
    }
    
    /* Make accordions look better */
    .gradio-accordion {
        margin-top: 10px !important;
        margin-bottom: 10px !important;
    }
    
    /* Input Image Container - proper scaling */
    .input-image-container, .keyframe-image-container {
        max-height: 340px !important;
        overflow: hidden !important;
    }
    .input-image-container img, .keyframe-image-container img {
        max-height: 340px !important;
        width: auto !important;
        object-fit: contain !important;
        margin: 0 auto !important;
        display: block !important;
    }
    
    /* Result Container - proper scaling */
    .result-container {
    max-height: 512px !important;
    min-height: 128px !important;
    overflow: hidden !important;
}

.result-container video, .gr-video video {
    max-height: 512px !important;
    height: 100% !important;
    width: auto !important;
    object-fit: contain !important;
    margin: 0 auto !important;
    display: block !important;
}
    
    /* Video Container - proper scaling */
    .video-container {
        position: relative;
        max-height: 596px !important;
        overflow: hidden !important;
        margin-bottom: 20px;
    }
    .video-container video {
        max-height: 512px !important;
        width: auto !important;
        object-fit: contain !important;
        margin: 0 auto !important;
        display: block !important;
    }
    
    /* Progress Bar Styling */
    .dual-progress-container {
        background: #222;
        border: 2px solid orange;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 16px;
    }
    .progress-label {
        font-weight: bold;
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
    }
    .progress-bar-bg {
        background: #333;
        border-radius: 4px;
        height: 20px;
        overflow: hidden;
        margin-bottom: 12px;
    }
    .progress-bar-fg {
        height: 100%;
        background: linear-gradient(90deg, #ff8800, #ff2200);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    .stats-box {
        background: #222;
        border: 2px solid orange;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 16px;
    }
    .stats-box table {
        width: 100%;
    }
    .stats-box td {
        padding: 4px 8px;
    }
    .stats-box td:first-child {
        width: 40%;
    }
    
    /* Better formatting for generation stats */
    #generation_stats {
        margin-top: 20px;
        padding: 15px;
        background: #333;
        border-radius: 8px;
        border-left: 4px solid orange;
    }
    #generation_stats strong {
        color: #ff8800;
    }
    
    /* Fix for trim controls */
    .video-container.editing {
        padding-bottom: 180px !important;
    }
    """

def make_progress_bar_css():
    """Return CSS for progress bar"""
    return """
    .progress-container {
        width: 100%;
        background-color: #ddd;
        border-radius: 5px;
    }
    .progress-bar {
        height: 20px;
        background-color: #4CAF50;
        text-align: center;
        line-height: 20px;
        color: white;
        border-radius: 5px;
    }
    """

def make_progress_bar_html(percent, label=''):
    """Generate HTML for a progress bar"""
    return f"""
    <div style="width:100%; background-color:#444; border-radius:5px; margin-bottom:10px;">
        <div style="height:24px; width:{percent}%; background:linear-gradient(90deg, #ff8800, #ff2200); 
             text-align:center; line-height:24px; color:white; border-radius:5px;">
            {percent}%
        </div>
    </div>
    <div style="text-align:center; margin-top:5px;"><b>{label}</b></div>
    """
