# Local Model Processing - Room Object Removal

High-quality object removal from room images using YOLO + SAM + Big-LaMa.

## Features

- **Object Detection**: YOLOv8 with confidence as low as 0.001
- **Segmentation**: SAM (Segment Anything Model) for precise masks
- **Inpainting**: Big-LaMa with iterative refinement
- **GPU Support**: DirectML for AMD GPUs, CUDA for NVIDIA
- **Full Control**: Detailed parameter adjustment for all models

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download models:
```bash
python download_models.py
```

3. Download Big-LaMa:
```bash
# Download from: https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
# Extract to models/ directory
```

4. Clone LaMa repository (for inference code):
```bash
git clone https://github.com/advimman/lama.git
# Copy saicinpainting folder to models/
```

## Usage

```bash
python room_removal_ultimate.py
```

Open http://localhost:7860 in your browser.

## Workflow

1. **Detection Tab**: Upload image and detect objects (use confidence 0.001 for maximum detection)
2. **Segmentation Tab**: Select object indices or "all", create masks
3. **Inpainting Tab**: Remove objects with Big-LaMa

## Parameters

### YOLO Detection
- Confidence: 0.001-1.0 (lower = more objects)
- IoU Threshold: Non-maximum suppression overlap
- Max Detections: Maximum objects per image

### SAM Segmentation
- Model Type: vit_b (fast) or vit_h (quality)
- Mask Dilation: Expand masks to include shadows (10-20 recommended)

### LaMa Inpainting
- Refinement Iterations: 1 for clean result, 2-5 for edge refinement
- Based on PR #112 iterative refinement approach

## Tips

- Use 1 iteration for best results (avoids gray mask artifacts)
- Set mask dilation to 10-20 to remove object shadows
- Use vit_h for best segmentation quality