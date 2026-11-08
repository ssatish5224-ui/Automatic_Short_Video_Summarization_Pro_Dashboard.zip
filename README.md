# ClipSense AI – Automatic Short-Video Summarization

A Streamlit dashboard that creates a shorter highlight video using MobileNetV2 visual features and frame-motion analysis.

## Features
- Upload MP4, AVI, MOV and MKV video files
- Inspect duration, resolution, FPS and file size
- Configure segment length, target summary duration and frame sampling rate
- Choose Balanced, Visual-change focused or Motion focused highlight scoring
- Generate a short MP4 made from selected non-overlapping segments
- View selected key-moment thumbnails and scores
- Explore score charts, segment details and timeline data
- Export the summary MP4, CSV report, JSON report and thumbnail ZIP
- Clear the current session from the sidebar

## Requirements
Python 3.10–3.12 is recommended. TensorFlow installation compatibility depends on the Python version and operating system.

## Run in VS Code (Windows PowerShell)
Open the folder that contains `app.py` and `requirements.txt`, then run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## How the scoring works
MobileNetV2 extracts visual features from sampled frames. Each time segment receives a visual-novelty score and a frame-motion score. The selected scoring profile combines these normalized values, then a greedy selection step chooses non-overlapping segments up to the requested duration. This is a visual highlight-ranking prototype, not a semantic understanding system.

## Notes
- The first run may download pretrained MobileNetV2 weights and can take longer.
- Use a short test video first.
- The generated MP4 uses OpenCV's `mp4v` codec; playback compatibility can vary by system.
- Results depend on video content, sampling rate and scoring settings.
