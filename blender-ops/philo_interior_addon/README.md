# Philo Interior Generator

A professional Blender addon for creating photorealistic interior visualizations with automated room generation, lighting presets, and intelligent furniture placement.

## Features

### 🏠 Room Generation
- **Customizable room size**: 4-50 meters with 6-meter tall ceilings
- **Material presets**:
  - Floors: Wood, Marble, Concrete, Carpet
  - Walls: Paint, Wallpaper, Brick, Plaster
- Auto-generated walls, floor, and ceiling with proper UV mapping

### 💡 Professional Lighting Presets
- **Natural**: Warm daylight through windows with realistic sun and sky
- **Studio**: Clean, even lighting perfect for product photography
- **Dramatic**: High contrast with accent lighting for luxury presentations
- Photorealistic render settings with enhanced light bounces (128 max)
- No post-processing effects - clean, direct output

### 📦 Smart Model Import
- **Single model import**: Import GLB, GLTF, FBX, or OBJ files
- **Batch folder import**: Process entire furniture libraries at once
- **Intelligent placement**:
  - Automatic floor positioning
  - Linear X-axis arrangement for folder imports (no overlapping)
  - Optional collision detection with adjustable spacing
  - Dynamic room size recommendations based on imported models

### 📸 Camera & Rendering
- **Optimized camera setup**: Fixed at 3.5m height with distance scaling
- **Quality presets**:
  - Preview: 64 samples (75% resolution for quick tests)
  - Medium: 256 samples (full resolution)
  - High: 2048 samples (production quality)
- OpenImageDenoise for all quality levels
- Adaptive sampling for optimal performance

### 🔧 Additional Tools
- **Model scaling**: Resize selected models (0.1x to 5x) with automatic floor repositioning
- **Physics collision**: Add rigid body physics to objects for simulations

## Installation

### Quick Installation (macOS with script)
```bash
cd blender-ops
./quick_install.sh
```

### Manual Installation
1. Copy `philo_interior_addon` folder to Blender's addon directory:
   - macOS: `~/Library/Application Support/Blender/4.4/scripts/addons/`
   - Windows: `%APPDATA%\Blender Foundation\Blender\4.4\scripts\addons\`
   - Linux: `~/.config/blender/4.4/scripts/addons/`
2. In Blender: Edit → Preferences → Add-ons → Enable "Philo Interior Generator"

## Quick Start Guide

1. **Create Room**
   - Open Philo panel in 3D viewport (press N)
   - Set room size (default: 8m, max: 50m)
   - Choose floor and wall materials
   - Click "Generate Room"

2. **Setup Lighting**
   - Select preset: Natural (residential), Studio (product), or Dramatic (luxury)
   - Click "Apply Lighting"

3. **Import Furniture**
   - Single model: Browse and select file
   - Multiple models: Select folder
   - Models auto-arrange along X-axis with proper spacing

4. **Camera & Render**
   - Click "Setup Camera" (auto-adjusts to room size)
   - Choose quality: Preview/Medium/High
   - Click "Render Snapshot"

## File Structure

```
philo_interior_addon/
├── __init__.py      # Addon registration and metadata
├── operators.py     # Core functionality (900+ lines)
├── props.py         # Scene properties and settings
├── ui.py           # User interface panels
└── README.md       # Documentation
```

## Technical Specifications

### Rendering
- Engine: Cycles (GPU recommended)
- Light bounces: 128 (diffuse: 8, glossy: 8, transmission: 128)
- Sampling: Sobol-Burley pattern with light tree
- Color management: Filmic tone mapping

### Room Defaults
- Ceiling height: 6 meters
- Camera height: 3.5 meters (fixed)
- Wall margin: 0.5 meters
- Furniture spacing: 1.5 meters (folder import)

### Import Details
- Supported formats: GLB, GLTF, FBX, OBJ
- Auto 180° rotation for proper facing
- Smooth shading applied automatically
- Original scale preserved (no auto-scaling)

## Tips for Best Results

1. **Room Size**: Generate room AFTER importing to get size recommendations
2. **Lighting**: Each preset is optimized for specific use cases
3. **Performance**: Enable GPU compute in Blender preferences
4. **Quality**: Start with Preview, use High for final renders

## Troubleshooting

**Import errors**: Check file paths and verify 3D file validity
**Models overlapping**: Ensure adequate room size for all models
**Slow rendering**: Reduce quality or enable GPU compute
**Dark renders**: Apply lighting preset after room generation

## Version History

- **v1.0.0**: Complete interior visualization workflow
  - Room generation with materials
  - Three lighting presets
  - Smart furniture import
  - Camera optimization
  - High-quality rendering

## Code Architecture

The addon follows Blender's best practices with clear separation:
- **operators.py**: Contains 8 main operators and helper functions
- **props.py**: Defines 9 scene properties for settings storage  
- **ui.py**: Creates 6 UI panels organized by workflow
- **__init__.py**: Handles registration and addon metadata

Created by Philo Homes for professional interior visualization.