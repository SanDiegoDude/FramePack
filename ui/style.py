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
    /* Start Button - Spotify green, taller */
    .start-button button {
        width: 100% !important;
        background-color: #1DB954 !important; /* Spotify green */
        border-color: #1DB954 !important;
        font-size: 1.2em !important;
        padding: 16px !important; /* Taller button */
        margin-bottom: 10px !important;
    }
    .start-button button:disabled {
        background-color: #107C33 !important; /* Darker Spotify green */
        border-color: #107C33 !important;
    }

    /* End Generation Button - Windows Explorer mellow yellow */
    .end-graceful-button button {
        width: 100% !important;
        background-color: #FFCC00 !important; /* Windows Explorer yellow */
        border-color: #FFCC00 !important;
        font-size: 1.1em !important;
        padding: 10px !important;
    }
    .end-graceful-button button:disabled {
        background-color: #D4AA00 !important; /* Darker yellow */
        border-color: #D4AA00 !important;
    }

    /* Force Stop Button - Crimson */
    .force-stop-button button {
        width: 100% !important;
        background-color: #DC143C !important; /* Crimson */
        border-color: #DC143C !important;
        font-size: 1.1em !important;
        padding: 10px !important;
    }
    .force-stop-button button:disabled {
        background-color: #AA0F2D !important; /* Darker crimson */
        border-color: #AA0F2D !important;
    }
    
    /* Remove old button classes that are no longer needed */
    .end-button-warning, .end-button-force {
        display: none !important;
    }
    /* Frame Thumbnails */
    .frame-thumbnail img {
        height: 256px !important;
        width: 256px !important;
        object-fit: cover !important;
    }
    /* Image and Video Container Styling */
    .input-image-container img {
        max-height: 512px !important;
        width: auto !important;
        object-fit: contain !important;
    }
    .keyframe-image-container img {
        max-height: 320px !important;
        width: auto !important;
        object-fit: contain !important;
    }
    .result-container img, .result-container video {
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
    
    /* Video container styling */
    .video-container {
        position: relative;
        max-height: 400px;           /* Fixed height container */
        overflow-y: auto;            /* Allow vertical scrolling */
        overflow-x: hidden;          /* Prevent horizontal scrolling */
        margin-bottom: 20px;         /* Space below container */
        padding-bottom: 150px;       /* Extra padding for trim interface */
    }
    
    /* Make sure the video itself doesn't overflow horizontally */
    .video-container video {
        max-width: 100%;
        object-fit: contain;
    }
    
    /* Ensure trim controls stay with the video */
    .video-container .video-wrap {
        position: relative;
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
    /* Hide frame thumbnails by default */
    .frame-thumbnail {
        display: none;
    }
    
    /* Show frame thumbnails only when they have content */
    .frame-thumbnail:not(:empty) {
        display: block;
    }

    <script>
    function detectVideoTrimmingUI() {
        const style = `
            <style>
                .video-container.editing {
                    padding-bottom: 180px !important;
                }
            </style>
        `;
        document.head.insertAdjacentHTML('beforeend', style);
        
        // Monitor for trim UI appearance
        setInterval(function() {
            const videoContainers = document.querySelectorAll('.video-container');
            videoContainers.forEach(container => {
                const trimControls = container.querySelector('.trim-control');
                if (trimControls) {
                    container.classList.add('editing');
                } else {
                    container.classList.remove('editing');
                }
            });
        }, 500);
    }
    // Initialize when page loads
    window.addEventListener('DOMContentLoaded', detectVideoTrimmingUI);
    </script>
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
