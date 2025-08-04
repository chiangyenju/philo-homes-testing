# Room 3D Reconstruction System Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Environment Setup](#environment-setup)
4. [Core Components](#core-components)
5. [Implementation Guide](#implementation-guide)
6. [API Reference](#api-reference)
7. [Testing & Validation](#testing--validation)
8. [Performance Optimization](#performance-optimization)
9. [Deployment Options](#deployment-options)
10. [Troubleshooting](#troubleshooting)

## Overview

### Project Goals
Transform 2D room images into 3D spatial representations by:
- Removing furniture and objects from room images
- Detecting architectural elements (doors, windows, walls)
- Estimating real-world scale using standard door dimensions
- Generating 3D room layouts for visualization and planning

### Key Features
- **Object Detection & Removal**: Identify and seamlessly remove furniture/objects
- **Architectural Analysis**: Detect doors, windows, and room boundaries
- **Scale Estimation**: Use door dimensions for metric measurements
- **3D Reconstruction**: Generate 3D room models from single 2D images
- **Interactive Processing**: Support for batch and real-time processing

### Technology Stack
- **Computer Vision**: OpenCV, PIL/Pillow
- **Deep Learning**: PyTorch, Ultralytics YOLO, Segment Anything Model (SAM)
- **Image Inpainting**: LaMa, EdgeConnect, Stable Diffusion
- **3D Processing**: Open3D, Trimesh, NumPy
- **Visualization**: Matplotlib, Plotly, Three.js (web interface)

## System Architecture

### High-Level Pipeline
```
Input Image → Object Detection → Segmentation → Inpainting → Structure Detection → Scale Estimation → 3D Generation → Output
```

### Component Breakdown
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Layer   │    │ Processing Core │    │  Output Layer   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Image Upload  │───▶│ • Object Det.   │───▶│ • 3D Model      │
│ • Video Frames  │    │ • Segmentation  │    │ • Clean Image   │
│ • Batch Files   │    │ • Inpainting    │    │ • Measurements  │
└─────────────────┘    │ • Structure Det.│    │ • Visualization │
                       │ • 3D Recon.     │    └─────────────────┘
                       └─────────────────┘
```

### Data Flow
1. **Input Processing**: Image validation, preprocessing, format conversion
2. **Object Analysis**: YOLO detection → SAM segmentation → mask generation
3. **Content Removal**: LaMa inpainting → quality validation → refinement
4. **Structure Analysis**: Edge detection → door/window identification → geometric analysis
5. **Scale Calculation**: Door measurement → pixel-to-meter conversion → validation
6. **3D Reconstruction**: Vanishing point detection → perspective calculation → mesh generation

## Environment Setup

### System Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ for models and cache
- **Python**: 3.8+ (recommended 3.10)

### Installation Guide

#### 1. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n room3d python=3.10
conda activate room3d

# Or using venv
python -m venv room3d_env
source room3d_env/bin/activate  # Linux/Mac
# room3d_env\Scripts\activate  # Windows
```

#### 2. Install Core Dependencies
```bash
# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Computer Vision & ML
pip install ultralytics>=8.0.0
pip install opencv-python>=4.8.0
pip install pillow>=10.0.0
pip install numpy>=1.24.0
pip install scipy>=1.11.0
pip install scikit-image>=0.21.0

# Segmentation
pip install segment-anything
pip install git+https://github.com/facebookresearch/detectron2.git

# Inpainting
pip install lama-cleaner
pip install diffusers>=0.21.0
pip install transformers>=4.35.0

# 3D Processing
pip install open3d>=0.17.0
pip install trimesh>=3.23.0
pip install plotly>=5.17.0

# Utilities
pip install tqdm
pip install matplotlib
pip install seaborn
pip install pyyaml
pip install requests
```

#### 3. Model Downloads
```bash
# Create models directory
mkdir models
cd models

# Download pre-trained models (automated in code)
# YOLO models will auto-download on first use
# SAM models will auto-download on first use
# LaMa models will auto-download on first use
```

### Project Structure
```
room3d_reconstruction/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── detector.py          # Object detection
│   │   ├── segmenter.py         # Image segmentation
│   │   ├── inpainter.py         # Image inpainting
│   │   ├── structure_analyzer.py # Room structure detection
│   │   ├── scale_estimator.py   # Scale calculation
│   │   └── reconstructor_3d.py  # 3D generation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py       # Image processing utilities
│   │   ├── geometry_utils.py    # Geometric calculations
│   │   ├── visualization.py     # Plotting and visualization
│   │   └── config.py           # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_manager.py    # Model loading and caching
│   └── pipeline.py             # Main pipeline orchestrator
├── tests/
│   ├── test_detector.py
│   ├── test_inpainter.py
│   ├── test_structure.py
│   └── test_pipeline.py
├── examples/
│   ├── basic_usage.py
│   ├── batch_processing.py
│   └── interactive_demo.py
├── config/
│   ├── default_config.yaml
│   └── model_config.yaml
├── data/
│   ├── sample_images/
│   ├── test_cases/
│   └── outputs/
├── models/                     # Downloaded model weights
├── requirements.txt
├── setup.py
└── README.md
```

## Core Components

### 1. Object Detection Module (`detector.py`)

#### Purpose
Identify and locate furniture, decor, and removable objects in room images.

#### Key Classes
```python
class ObjectDetector:
    def __init__(self, model_type='yolov8n', confidence_threshold=0.5)
    def detect_objects(self, image) -> List[Detection]
    def filter_room_objects(self, detections) -> List[Detection]
    def visualize_detections(self, image, detections)
```

#### Configuration
```yaml
detector:
  model: "yolov8n"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  confidence_threshold: 0.5
  room_object_classes:
    - "chair"
    - "couch"
    - "bed"
    - "dining table"
    # ... (see full list in config)
```

### 2. Segmentation Module (`segmenter.py`)

#### Purpose
Create precise pixel-level masks for detected objects using SAM.

#### Key Classes
```python
class ImageSegmenter:
    def __init__(self, sam_model='sam_vit_b_01ec64')
    def segment_objects(self, image, detections) -> np.ndarray
    def refine_masks(self, masks, image) -> np.ndarray
    def combine_masks(self, mask_list) -> np.ndarray
```

#### Features
- Point prompting for interactive segmentation
- Box prompting from YOLO detections
- Mask refinement and post-processing
- Multi-object mask combination

### 3. Inpainting Module (`inpainter.py`)

#### Purpose
Remove objects and fill in background seamlessly.

#### Key Classes
```python
class ImageInpainter:
    def __init__(self, method='lama', device='cuda')
    def inpaint_image(self, image, mask) -> np.ndarray
    def batch_inpaint(self, images, masks) -> List[np.ndarray]
    def validate_inpainting(self, original, inpainted, mask) -> float
```

#### Supported Methods
- **LaMa**: Best for large objects and complex backgrounds
- **EdgeConnect**: Good for preserving structural details
- **Stable Diffusion**: High quality but slower
- **OpenCV Telea/NS**: Fast but lower quality

### 4. Structure Analyzer (`structure_analyzer.py`)

#### Purpose
Detect architectural elements like doors, windows, and walls.

#### Key Classes
```python
class StructureAnalyzer:
    def __init__(self, door_detector_model=None)
    def detect_doors(self, image) -> List[Door]
    def detect_windows(self, image) -> List[Window]
    def detect_walls(self, image) -> List[Wall]
    def find_room_boundaries(self, image) -> RoomBoundary
```

#### Detection Strategies
- **Geometric**: Edge detection + contour analysis + aspect ratio filtering
- **Learning-based**: Custom trained CNN for door/window detection
- **Hybrid**: Combine geometric and learned features

### 5. Scale Estimator (`scale_estimator.py`)

#### Purpose
Calculate real-world scale using standard door dimensions.

#### Key Classes
```python
class ScaleEstimator:
    def __init__(self, standard_door_width=0.8, standard_door_height=2.0)
    def estimate_scale(self, door_detections) -> float
    def validate_scale(self, scale, image_dimensions) -> bool
    def get_measurements(self, pixel_distances, scale) -> List[float]
```

#### Standard Dimensions
- Interior doors: 80cm width × 200cm height
- Exterior doors: 90cm width × 200cm height
- Window heights: 120-150cm typically
- Ceiling heights: 240-270cm standard

### 6. 3D Reconstructor (`reconstructor_3d.py`)

#### Purpose
Generate 3D room layout from processed 2D image.

#### Key Classes
```python
class Room3DReconstructor:
    def __init__(self, method='geometric')
    def detect_vanishing_points(self, image) -> List[Point2D]
    def estimate_room_layout(self, image, doors, scale) -> RoomLayout
    def generate_3d_mesh(self, layout) -> trimesh.Trimesh
    def export_model(self, mesh, format='obj') -> str
```

#### 3D Generation Methods
- **Geometric**: Vanishing point detection + perspective geometry
- **Layout Networks**: Deep learning for room layout estimation
- **Hybrid**: Combine geometric constraints with learned priors

## Implementation Guide

### Phase 1: Basic Pipeline (Week 1-2)

#### Step 1: Set up core detection
```python
# src/core/detector.py
from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_type='yolov8n', confidence_threshold=0.5):
        self.model = YOLO(f'{model_type}.pt')
        self.confidence_threshold = confidence_threshold
        self.room_objects = [
            'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear'
        ]
    
    def detect_objects(self, image):
        results = self.model(image)
        detections = []
        
        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                confidence = float(box.conf)
                
                if (class_name in self.room_objects and 
                    confidence > self.confidence_threshold):
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'area': self._calculate_area(box.xyxy[0])
                    })
        
        return detections
    
    def _calculate_area(self, bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
```

#### Step 2: Implement basic inpainting
```python
# src/core/inpainter.py
import cv2
import numpy as np

class ImageInpainter:
    def __init__(self, method='opencv'):
        self.method = method
    
    def inpaint_image(self, image, mask):
        if self.method == 'opencv':
            return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        elif self.method == 'lama':
            return self._lama_inpaint(image, mask)
    
    def create_mask_from_detections(self, image_shape, detections):
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for detection in detections:
            bbox = detection['bbox'].astype(int)
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255
        
        return mask
```

#### Step 3: Basic structure detection
```python
# src/core/structure_analyzer.py
import cv2
import numpy as np

class StructureAnalyzer:
    def __init__(self):
        self.door_aspect_ratio_range = (1.5, 4.0)
        self.min_door_area = 1000
    
    def detect_doors(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        doors = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_door_area:
                rect = cv2.boundingRect(contour)
                x, y, w, h = rect
                aspect_ratio = h / w if w > 0 else 0
                
                if (self.door_aspect_ratio_range[0] < aspect_ratio < 
                    self.door_aspect_ratio_range[1]):
                    
                    doors.append({
                        'bbox': rect,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'confidence': self._calculate_door_confidence(contour, rect)
                    })
        
        return sorted(doors, key=lambda x: x['confidence'], reverse=True)
```

### Phase 2: Advanced Features (Week 3-4)

#### Integrate SAM for better segmentation
```python
# Enhanced segmenter.py
from segment_anything import SamPredictor, sam_model_registry

class ImageSegmenter:
    def __init__(self, sam_checkpoint="sam_vit_b_01ec64.pth"):
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(sam)
    
    def segment_from_boxes(self, image, boxes):
        self.predictor.set_image(image)
        masks = []
        
        for box in boxes:
            mask, _, _ = self.predictor.predict(box=box)
            masks.append(mask[0])  # Take first mask
        
        return np.array(masks)
```

#### Add LaMa inpainting
```python
# Enhanced inpainter.py with LaMa
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config

class ImageInpainter:
    def __init__(self, method='lama'):
        if method == 'lama':
            self.model = ModelManager(name="lama", device="cuda")
    
    def lama_inpaint(self, image, mask):
        config = Config(
            ldm_steps=20,
            ldm_sampler='plms',
            hd_strategy='Original',
            hd_strategy_crop_margin=32,
        )
        
        result = self.model(image, mask, config)
        return result
```

### Phase 3: 3D Generation (Week 5-6)

#### Implement vanishing point detection
```python
# Enhanced reconstructor_3d.py
import cv2
import numpy as np
from scipy.optimize import least_squares

class Room3DReconstructor:
    def detect_vanishing_points(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return []
        
        # Group lines by orientation
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if abs(angle) < 30 or abs(angle) > 150:  # Horizontal-ish
                horizontal_lines.append(line[0])
            elif 60 < abs(angle) < 120:  # Vertical-ish
                vertical_lines.append(line[0])
        
        # Find intersection points (vanishing points)
        vanishing_points = self._find_line_intersections(horizontal_lines + vertical_lines)
        return vanishing_points
    
    def generate_room_layout(self, doors, scale, vanishing_points):
        if not doors or not scale:
            return None
        
        # Use largest door as reference
        main_door = max(doors, key=lambda x: x['area'])
        door_width_m = main_door['bbox'][2] / scale
        door_height_m = main_door['bbox'][3] / scale
        
        # Estimate room dimensions based on door position and vanishing points
        # This is a simplified approach - real implementation would be more complex
        
        return {
            'width': door_width_m * 6,  # Rough estimation
            'height': door_height_m,
            'depth': door_width_m * 8,
            'door_position': main_door['bbox']
        }
```

## API Reference

### Main Pipeline Class
```python
class Room3DPipeline:
    def __init__(self, config_path='config/default_config.yaml'):
        """Initialize the complete pipeline"""
        
    def process_image(self, image_path: str) -> Dict:
        """
        Process a single room image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dict containing:
                - original_image: Original input image
                - clean_image: Image with objects removed
                - detections: List of detected objects
                - doors: List of detected doors
                - scale: Estimated pixels per meter
                - layout_3d: 3D room layout parameters
                - mesh: 3D mesh object (if generated)
        """
        
    def batch_process(self, image_paths: List[str]) -> List[Dict]:
        """Process multiple images in batch"""
        
    def export_results(self, results: Dict, output_dir: str, formats: List[str]):
        """Export results in various formats (obj, ply, json, etc.)"""
```

### Configuration System
```yaml
# config/default_config.yaml
detector:
  model: "yolov8n"
  confidence_threshold: 0.5
  device: "cuda"

segmenter:
  model: "sam_vit_b"
  use_sam: true
  refinement: true

inpainter:
  method: "lama"  # lama, opencv, stable_diffusion
  quality: "high"
  batch_size: 1

structure_analyzer:
  door_detection: "geometric"  # geometric, learned, hybrid
  min_door_area: 1000
  aspect_ratio_tolerance: 0.3

scale_estimator:
  standard_door_width: 0.8
  standard_door_height: 2.0
  validation_enabled: true

reconstructor_3d:
  method: "geometric"  # geometric, learned, hybrid
  generate_mesh: true
  mesh_quality: "medium"

output:
  save_intermediate: true
  formats: ["obj", "ply", "json"]
  visualization: true
```

## Testing & Validation

### Unit Tests
```python
# tests/test_detector.py
import pytest
from src.core.detector import ObjectDetector

class TestObjectDetector:
    def setup_method(self):
        self.detector = ObjectDetector()
    
    def test_object_detection(self):
        # Test with sample image
        image_path = "data/test_cases/living_room_1.jpg"
        detections = self.detector.detect_objects(image_path)
        assert len(detections) > 0
        assert all('bbox' in d for d in detections)
    
    def test_confidence_filtering(self):
        # Test confidence threshold filtering
        pass
```

### Integration Tests
```python
# tests/test_pipeline.py
from src.pipeline import Room3DPipeline

class TestPipeline:
    def test_end_to_end_processing(self):
        pipeline = Room3DPipeline()
        results = pipeline.process_image("data/test_cases/bedroom_1.jpg")
        
        assert 'clean_image' in results
        assert 'layout_3d' in results
        assert results['scale'] is not None
```

### Validation Metrics
```python
# Validation criteria
def validate_results(results):
    checks = {
        'object_removal_quality': validate_inpainting_quality(results),
        'door_detection_accuracy': validate_door_detection(results),
        'scale_estimation_error': validate_scale_accuracy(results),
        'layout_plausibility': validate_3d_layout(results)
    }
    return checks
```

## Performance Optimization

### GPU Memory Management
```python
# Memory optimization strategies
import torch

def optimize_gpu_memory():
    # Clear cache between operations
    torch.cuda.empty_cache()
    
    # Use gradient checkpointing for large models
    torch.utils.checkpoint.checkpoint_sequential()
    
    # Process in batches
    batch_size = 4 if torch.cuda.get_device_properties(0).total_memory > 8e9 else 2
```

### Caching Strategy
```python
# Model caching
class ModelCache:
    def __init__(self):
        self.models = {}
    
    def get_model(self, model_name):
        if model_name not in self.models:
            self.models[model_name] = self._load_model(model_name)
        return self.models[model_name]
```

### Parallel Processing
```python
# Batch processing optimization
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

def batch_process_optimized(image_paths, num_workers=None):
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(image_paths))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_single_image, image_paths))
    
    return results
```

## Deployment Options

### Local Development
```bash
# Run locally
python examples/basic_usage.py --input data/sample_images/ --output results/
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY src/ /app/src/
COPY config/ /app/config/
WORKDIR /app

CMD ["python3", "-m", "src.pipeline"]
```

### Cloud Deployment
```python
# API server with FastAPI
from fastapi import FastAPI, UploadFile
from src.pipeline import Room3DPipeline

app = FastAPI()
pipeline = Room3DPipeline()

@app.post("/process_room/")
async def process_room_image(file: UploadFile):
    # Save uploaded file
    # Process with pipeline
    # Return results
    pass
```

### Web Interface
```html
<!-- Simple web interface -->
<!DOCTYPE html>
<html>
<head>
    <title>Room 3D Reconstruction</title>
</head>
<body>
    <div id="upload-area">
        <input type="file" id="image-input" accept="image/*">
        <button onclick="processImage()">Process Room</button>
    </div>
    <div id="results">
        <!-- Results display -->
    </div>
    
    <script>
        async function processImage() {
            // Upload and process image
            // Display results
        }
    </script>
</body>
</html>
```

## Troubleshooting

### Common Issues

#### GPU Memory Errors
```python
# Solution: Reduce batch size or model size
RuntimeError: CUDA out of memory

# Fix:
- Use smaller YOLO model (yolov8n instead of yolov8x)
- Process images at lower resolution
- Clear GPU cache: torch.cuda.empty_cache()
- Reduce batch size
```

#### Model Download Failures
```python
# Solution: Manual model download
URLError: <urlopen error [Errno 11001] getaddrinfo failed>

# Fix:
- Check internet connection
- Use manual model download
- Set up local model cache
```

#### Poor Object Detection
```python
# Solution: Adjust parameters
# Too many false positives

# Fix:
- Increase confidence threshold
- Filter by object size
- Use custom trained model for room-specific objects
```

#### Inaccurate Scale Estimation
```python
# Solution: Improve door detection
# Scale estimation fails

# Fix:
- Validate door detection manually
- Use multiple reference objects
- Implement door detection validation
- Fall back to average room dimensions
```

### Debug Mode
```python
# Enable detailed logging
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Room3DPipeline:
    def __init__(self, debug=False):
        self.debug = debug
        if debug:
            self.setup_debug_mode()
    
    def setup_debug_mode(self):
        # Save intermediate results
        # Enable detailed logging
        # Visualization of each step
        pass
```

### Performance Monitoring
```python
# Performance profiling
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
    
    def start_timer(self, operation):
        self.timings[operation] = time.time()
    
    def end_timer(self, operation):
        if operation in self.timings:
            elapsed = time.time() - self.timings[operation]
            print(f"{operation}: {elapsed:.2f}s")
    
    def log_memory_usage(self):
        gpu_memory = GPUtil.getGPUs()[0].memoryUsed
        ram_memory = psutil.virtual_memory().percent
        print(f"GPU Memory: {gpu_memory}MB, RAM: {ram_memory}%")
```

---

## Getting Started Checklist

- [ ] Set up Python environment (3.8+)
- [ ] Install dependencies from requirements.txt
- [ ] Download sample room images
- [ ] Run basic detection test
- [ ] Implement Phase 1 components
- [ ] Test with sample images
- [ ] Add advanced features (SAM, LaMa)
- [ ] Implement 3D reconstruction
- [ ] Set up evaluation metrics
- [ ] Optimize for your hardware
- [ ] Deploy and test

## Next Steps

1. **Start with Phase 1**: Get basic object detection and removal working
2. **Test extensively**: Use diverse room images to validate approach
3. **Iterate improvements**: Add SAM, LaMa, and other advanced components
4. **Scale up**: Move to batch processing and deployment
5. **Specialize**: Adapt for specific room types or use cases

This documentation provides the foundation for building a complete room 3D reconstruction system. Start with the basic components and gradually add complexity as you validate each piece of the pipeline.