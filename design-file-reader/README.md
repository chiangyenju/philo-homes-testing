# Layout Extractor - Floor Plan Analysis Tool

🏠 Advanced computer vision tool for extracting furniture, walls, and doors from 2D floor plan images, with support for manual adjustments via Gradio interface.

## Features

- **🪑 Furniture Detection**: Multiple detection methods for accurate furniture boundary extraction
- **🧱 Wall Detection**: Multi-scale edge detection with orientation classification
- **🚪 Door Detection**: Intelligent gap analysis in walls (max 2 doors per image)
- **📍 Manual Adjustment**: Click-to-pin functionality for unidentified items
- **🎨 Interactive Interface**: Gradio web app for easy use and adjustments
- **📊 Multiple Export Formats**: JSON, CSV, and visual annotations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/layout-extractor.git
cd layout-extractor

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

```python
# Basic extraction
python test_extraction.py

# This will process 'layout.png' and generate:
# - extraction_results.json (detailed data)
# - extraction_results.csv (tabular format)
# - extraction_visualization.png (annotated image)
```

### Gradio Web Interface

```python
# Launch the web app
python layout_extractor_app.py

# Opens browser at http://localhost:7860
# Upload image → Process → Click to add pins → Export
```

## Algorithm Details

### Furniture Detection
- **Adaptive Thresholding**: Multiple block sizes (11, 15, 21) for varying lighting conditions
- **Connected Components Analysis**: Identifies distinct objects
- **Shape Analysis**: Rectangularity and circularity metrics
- **Confidence Scoring**: Based on shape properties and detection method

### Wall Detection
- **Multi-scale Edge Detection**: Canny edge detection at multiple thresholds
- **Line Merging**: Combines similar lines to reduce noise
- **Orientation Classification**: Horizontal, vertical, or diagonal
- **Thickness Estimation**: Analyzes perpendicular pixels

### Door Detection
- **Gap Analysis**: Identifies breaks in collinear walls
- **Size Filtering**: 50-200px gaps considered as doors
- **Confidence Based on Size**: 70-150px gaps have highest confidence
- **Maximum 2 Doors**: As per typical floor plan constraints

## Output Format

### JSON Structure
```json
{
  "furniture": [
    {
      "id": 1,
      "center": [496, 457],
      "bbox": {"x": 100, "y": 200, "width": 50, "height": 80},
      "area": 4000,
      "confidence": 0.8,
      "has_text": false
    }
  ],
  "walls": [
    {
      "id": 1,
      "start": [0, 100],
      "end": [500, 100],
      "orientation": "horizontal",
      "thickness": 10
    }
  ],
  "doors": [
    {
      "id": 1,
      "position": [250, 100],
      "size": 80,
      "orientation": "horizontal",
      "confidence": 0.9
    }
  ]
}
```

### CSV Format
- `Type, ID, Center_X, Center_Y, Width, Height, Area, Confidence`
- Includes furniture, doors, and manual pins

## API Usage

```python
from layout_extractor_app import LayoutExtractor

# Initialize extractor
extractor = LayoutExtractor()

# Process image
image = cv2.imread('floor_plan.png')
results = extractor.process_image(image)

# Access results
furniture = results['furniture']
walls = results['walls']
doors = results['doors']

# Add manual pin
extractor.add_manual_pin(x=100, y=200, label="Missed Item")

# Export
json_data = extractor.export_to_json()
csv_df = extractor.export_to_csv()

# Visualize
vis_image = extractor.visualize(
    show_walls=True, 
    show_furniture=True, 
    show_doors=True, 
    show_pins=True
)
```

## Working with Design Files

### Supported Formats
- **Images**: TIFF, PNG, JPG, BMP
- **Better with originals**: DWG, DXF, IFC (via conversion)

### If You Have .RAP Files (ARCHICAD)
1. Use BIMx Desktop Viewer (free) to open
2. Export to DXF or high-res image
3. Process with this tool

### Recommended Workflow
1. **Best**: Use original CAD files (DWG/DXF) with coordinate data
2. **Good**: High-resolution floor plan images with clear lines
3. **OK**: Screenshots or photos of floor plans

## Configuration

### Detection Parameters
```python
# In layout_extractor_app.py

# Furniture detection
MIN_FURNITURE_AREA = 200  # Minimum area in pixels
MAX_FURNITURE_AREA = width * height * 0.15  # Maximum as percentage

# Wall detection
MIN_LINE_LENGTH = 50  # Minimum wall segment length
MAX_LINE_GAP = 20  # Maximum gap to merge lines

# Door detection
MIN_DOOR_GAP = 50  # Minimum gap size for doors
MAX_DOOR_GAP = 200  # Maximum gap size for doors
DOOR_CONFIDENCE_RANGE = (70, 150)  # High confidence range
```

## Examples

### Processing Multiple Images
```python
import glob

for image_path in glob.glob('floor_plans/*.png'):
    extractor = LayoutExtractor()
    image = cv2.imread(image_path)
    results = extractor.process_image(image)
    
    # Save with image name
    base_name = Path(image_path).stem
    extractor.export_to_json(f'{base_name}_results.json')
```

### Coordinate Systems
The tool uses pixel coordinates with origin at top-left:
- X increases rightward (0 to image_width)
- Y increases downward (0 to image_height)

For CAD export, consider transforming to bottom-left origin.

## Limitations

- **Door Detection**: Limited to 2 doors per image (can be modified)
- **Text Recognition**: Currently identifies text regions but doesn't OCR
- **Scale**: No automatic scale detection - pixel coordinates only
- **Curved Walls**: Best with straight walls, limited curve support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Future Improvements

- [ ] OCR integration for text/number extraction
- [ ] Machine learning model training on floor plan datasets
- [ ] Automatic scale detection from dimension annotations
- [ ] Support for curved walls and irregular shapes
- [ ] Room boundary detection and labeling
- [ ] Window detection separate from doors
- [ ] 3D extrusion from 2D plans

## License

MIT License - See LICENSE file for details

## Acknowledgments

- OpenCV for computer vision algorithms
- Gradio for the web interface
- scikit-image for image processing utilities

## Contact

For questions or support, please open an issue on GitHub.

---

Made with ❤️ for architecture and interior design automation