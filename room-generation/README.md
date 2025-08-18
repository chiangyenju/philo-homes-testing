# Video to 3D Model Reconstruction

Convert videos into 3D models using COLMAP and computer vision.

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the app:**
```bash
python app.py
```

3. **Open browser at** `http://127.0.0.1:7860`

## Features

- 📹 Extract frames from video with quality filtering
- 🔨 Automatic COLMAP setup and configuration  
- 🎛️ Three quality presets (Fast/Balanced/Quality)
- 📊 3D point cloud visualization
- 💾 Export models as ZIP files

## Files

- `app.py` - Main unified application with Gradio UI
- `requirements.txt` - Python dependencies
- `tools/` - COLMAP binaries (auto-downloads if missing)

## Usage Tips

- Record 30-60 second videos
- Move camera slowly and steadily
- Capture all angles of the room/object
- Ensure good lighting
- Start with "Fast" mode for testing

## Requirements

- Windows 10/11 (COLMAP auto-downloads)
- Python 3.8+
- 8GB+ RAM recommended
- No GPU required (CPU-only processing)