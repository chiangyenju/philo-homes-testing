# Room-to-3D Virtual Environment Pipeline

## Project Overview

Transform a single photo of a room into a 3D virtual environment for AI-driven furniture placement. The system will detect room geometry, remove existing furniture, identify doors and windows, and create a clean 3D space ready for virtual furniture design.

## Ultimate Goal

**Input**: Single photo of a furnished room  
**Output**: 3D virtual environment with:
- Accurate room dimensions and geometry
- Clean, empty room structure
- Properly positioned doors and windows
- Spatial awareness for intelligent furniture placement

## Technical Architecture

### Sequential Pipeline Approach

The system uses a 5-step sequential pipeline that builds complexity incrementally:

```
Photo Input → Room Layout → Depth/Scale → Opening Detection → Door/Window Classification → 3D Environment
```

## Phase 1: Core Pipeline (Priority Implementation)

### Step 1: Object Removal and Room Cleaning
**Status**: ✅ Already solved  
**Description**: Remove furniture and objects using YOLO detection
- Use pre-trained YOLO models to detect furniture
- Apply inpainting techniques to fill removed areas
- Preserve wall, floor, and ceiling structures

### Step 2: Room Layout Estimation
**Implementation**: Use deep learning models for room boundary detection
- **Primary Model**: HorizonNet or LayoutNet
- **Alternative**: DuLa-Net or LED2-Net
- **Output**: Wall positions, floor boundaries, ceiling intersections
- **Format**: Corner coordinates and wall orientations

**Key Models to Implement**:
```python
# HorizonNet for room layout estimation
- Input: Single RGB image
- Output: Room corner positions, wall boundaries
- Inference: ~100ms on GPU
```

### Step 3: Depth Estimation and Scale Recovery
**Implementation**: Monocular depth estimation for 3D understanding
- **Primary Model**: MiDaS or DPT (Dense Prediction Transformer)
- **Purpose**: Convert 2D layout to scaled 3D dimensions
- **Output**: Depth map aligned with room layout
- **Scale Recovery**: Use known architectural constraints (door heights, etc.)

**Integration Requirements**:
```python
# Depth estimation pipeline
- Align depth maps with layout estimation
- Apply scale normalization using architectural priors
- Generate 3D point cloud of room structure
```

### Step 4: Opening Detection (Generic)
**Implementation**: Detect openings in walls without classification
- **Method**: Depth discontinuity analysis + geometric constraints
- **Features**: Size filtering, position validation, aspect ratio checks
- **Output**: Bounding boxes of all openings (doors + windows)

**Geometric Constraints**:
- Minimum opening size: 60cm width, 120cm height
- Maximum opening size: 80% of wall area
- Position validation: Reasonable placement on walls

## Phase 2: Enhanced Classification (Secondary Priority)

### Step 5: Door vs Window Classification
**Implementation**: Binary classifier for detected openings
- **Input**: Cropped opening regions + contextual features
- **Model**: Lightweight CNN or feature-based classifier
- **Context Features**:
  - Distance from floor
  - Size relative to room
  - Aspect ratio
  - Wall position (corner/center)
  - Room layout context

**Training Data Requirements**:
- 200-500 door examples
- 200-500 window examples  
- Augmentation: rotation, brightness, scale variations

### Step 6: 3D Environment Generation
**Implementation**: Create navigable 3D space
- **Room Geometry**: Convert layout + depth to 3D mesh
- **Opening Integration**: Place doors/windows in 3D structure
- **Optimization**: Ensure geometric consistency
- **Output Format**: 3D model compatible with furniture placement AI

## Google Colab Implementation Specifications

### Environment Setup
```python
# Required libraries and installations
!pip install torch torchvision
!pip install opencv-python
!pip install trimesh
!pip install open3d
!pip install matplotlib plotly
!pip install huggingface_hub
!pip install ultralytics  # For YOLO
```

### Model Downloads and Setup
```python
# Pre-trained model requirements
models_to_download = {
    "HorizonNet": "sunset1995/HorizonNet",
    "MiDaS": "Intel/dpt-large", 
    "YOLO": "yolov8n.pt",
    "InPainting": "runwayml/stable-diffusion-inpainting"
}
```

### Memory and Compute Considerations
- **GPU Requirements**: T4 or better (available in Colab Pro)
- **RAM Usage**: ~12-15GB peak (manageable in Colab)
- **Processing Time**: ~30-60 seconds per image
- **Batch Processing**: Process images sequentially to manage memory

## Technical Implementation Details

### Data Flow Architecture
```python
class RoomPipeline:
    def __init__(self):
        self.object_remover = YOLOInpainter()
        self.layout_estimator = HorizonNet()
        self.depth_estimator = MiDaS()
        self.opening_detector = GeometricAnalyzer()
        self.classifier = OpeningClassifier()
        self.renderer = Room3DGenerator()
    
    def process_image(self, image_path):
        # Step 1: Remove furniture
        clean_room = self.object_remover(image_path)
        
        # Step 2: Estimate layout
        layout = self.layout_estimator(clean_room)
        
        # Step 3: Depth estimation
        depth_map = self.depth_estimator(clean_room)
        
        # Step 4: Detect openings
        openings = self.opening_detector(layout, depth_map)
        
        # Step 5: Classify openings
        classified = self.classifier(openings)
        
        # Step 6: Generate 3D environment
        room_3d = self.renderer(layout, depth_map, classified)
        
        return room_3d
```

### Output Specifications

#### Room Geometry Data Structure
```python
room_data = {
    "dimensions": {
        "length": float,  # meters
        "width": float,   # meters  
        "height": float   # meters
    },
    "walls": [
        {
            "id": int,
            "start_point": [x, y, z],
            "end_point": [x, y, z], 
            "normal": [x, y, z]
        }
    ],
    "openings": [
        {
            "type": "door" | "window",
            "wall_id": int,
            "position": [x, y, z],
            "dimensions": [width, height],
            "confidence": float
        }
    ],
    "floor_polygon": [[x, y] coordinates],
    "ceiling_height": float
}
```

## Success Metrics and Validation

### Phase 1 Metrics
- **Room Layout Accuracy**: Compare estimated room dimensions to ground truth
- **Opening Detection Recall**: Percentage of actual openings detected
- **3D Reconstruction Quality**: Visual inspection and geometric consistency

### Phase 2 Metrics  
- **Door/Window Classification**: Accuracy on manually labeled test set
- **Furniture Placement Quality**: Spatial reasoning validation
- **End-to-End Performance**: Complete pipeline success rate

## Development Priorities

### Immediate Tasks (Week 1-2)
1. Set up Google Colab environment with all dependencies
2. Implement and test object removal pipeline
3. Integrate HorizonNet for room layout estimation
4. Test depth estimation with MiDaS

### Medium-term Goals (Week 3-4)
1. Develop opening detection algorithms
2. Create basic 3D room visualization
3. Validate geometric consistency
4. Test with various room types

### Long-term Enhancements (Week 5+)
1. Train door/window classifier
2. Optimize processing speed
3. Add robust error handling
4. Scale to batch processing

## Error Handling and Edge Cases

### Common Failure Modes
- **Complex room layouts**: L-shaped rooms, irregular geometry
- **Poor lighting conditions**: Dark rooms, strong shadows
- **Cluttered scenes**: Too many objects obscuring structure
- **Unusual architecture**: Non-standard door/window designs

### Mitigation Strategies
- Confidence scoring for all predictions
- Fallback to simpler geometric assumptions
- User validation interface for critical decisions
- Graceful degradation when components fail

## Integration with Furniture Placement AI

### Spatial Constraints Export
```python
constraints = {
    "walkable_area": polygon_coordinates,
    "wall_adjacency_zones": wall_buffer_areas,
    "door_clearance_zones": door_swing_areas,
    "window_interaction_zones": window_access_areas,
    "room_center": [x, y],
    "traffic_flow_paths": path_coordinates
}
```

### API Interface for Design AI
- **Room query functions**: Get dimensions, wall positions
- **Spatial validation**: Check if furniture placement is valid
- **Constraint checking**: Ensure door clearances, traffic flow
- **Optimization suggestions**: Recommend furniture arrangements

This pipeline provides a robust foundation for transforming room photos into intelligent 3D environments while maintaining flexibility for iterative improvements and extensions.