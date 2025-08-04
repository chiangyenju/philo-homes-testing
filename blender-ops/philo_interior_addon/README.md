# Philo Interior Generator

A streamlined Blender addon for creating photorealistic furniture visualizations with professional lighting presets.

## Features

1. **Room Generation with Material Options**
   - Creates a complete room with ceiling, floor, and walls
   - **Floor materials**: Wood, Marble, Concrete, Carpet
   - **Wall materials**: Paint, Wallpaper, Brick, Plaster
   - Optimized 8x8m dimensions for interior visualization
   - Furniture automatically positioned above floor level

2. **Professional Lighting Presets**
   - **Natural**: Warm daylight through windows - perfect for residential interiors
   - **Studio**: Clean, even lighting - ideal for product photography
   - **Dramatic**: High contrast with accent lights - luxury furniture presentation
   - Each preset includes optimized:
     - Light placement and color temperatures
     - Photorealistic render settings
     - Subtle post-processing effects (bloom, vignette)
     - Professional color grading

3. **3D Model Import with Smart Placement**
   - Single model import with file browser
   - Batch import from folder
   - Supports: GLB, GLTF, FBX, OBJ formats
   - **Automatic floor positioning** - models placed at correct height
   - **Collision detection** - prevents furniture overlap
   - **Smart placement algorithm** - finds valid positions automatically
   - Adjustable spacing between objects
   - Automatic smooth shading

4. **Model Tools**
   - **Scale control** - Resize selected models (0.1x to 5x)
   - Works on single or multiple selected objects
   - Maintains proportions when scaling
   - **Automatic floor repositioning** after scaling to prevent sinking

5. **Camera & Rendering**
   - **Elevated camera view** positioned further back (8m distance)
   - Height of 2.5m with slight downward angle
   - Wide 24mm lens for complete room coverage
   - Perfect for architectural visualization
   - Three quality presets:
     - Preview: 64 samples (fast)
     - Medium: 256 samples (balanced)
     - High: 1024 samples (production)
   - One-click snapshot rendering

6. **Physics Tools**
   - Add collision physics to furniture
   - Enable physics simulations
   - Rigid body support for realistic interactions

## Installation

### Quick Installation (macOS with script)

1. Navigate to the `blender-ops` directory
2. Run the installation script:
   ```bash
   ./quick_install.sh
   ```
3. Restart Blender
4. Enable the addon:
   - Go to Edit > Preferences > Add-ons
   - Search for "Philo Interior Generator"
   - Check the box to enable it

### Direct Installation (All platforms)

1. Download the `philo_interior_addon` folder
2. In Blender, go to Edit > Preferences > Add-ons
3. Click "Install..." and select the addon folder
4. Enable "Philo Interior Generator"

### Manual Installation

1. Copy `philo_interior_addon` to Blender's addon directory:
   - macOS: `~/Library/Application Support/Blender/4.4/scripts/addons/`
   - Windows: `%APPDATA%\Blender Foundation\Blender\4.4\scripts\addons\`
   - Linux: `~/.config/blender/4.4/scripts/addons/`

2. Restart Blender and enable the addon in Preferences

## Usage

1. **Open the Panel**: In 3D Viewport, press N and find the "Philo" tab

2. **Simple Workflow**:
   - Choose floor and wall materials in "Room Materials" panel
   - Click "Generate Room" to create the room
   - Select a lighting style (Natural, Studio, or Dramatic)
   - Click "Apply Lighting" to set up photorealistic lighting
   - Import your 3D models (single file or entire folder)
   - Select models and use "Model Tools" to scale if needed
   - Click "Setup Camera" for elevated view
   - Choose render quality (Preview/Medium/High)
   - Click "Render Snapshot" for final image

3. **Room Materials**:
   - **Floor**: Wood, Marble, Concrete, or Carpet
   - **Walls**: Paint, Wallpaper, Brick, or Plaster
   - Set before generating room

4. **Lighting Styles**:
   - **Natural**: Warm sunlight through windows for residential feel
   - **Studio**: Clean, professional lighting for product showcase
   - **Dramatic**: Moody lighting with strong contrasts for luxury items

5. **Import Settings**:
   - **Avoid Overlaps**: Toggle collision detection
   - **Spacing**: Adjust minimum distance between objects (0-1m)

6. **Model Scaling**:
   - Select one or more models
   - Adjust scale slider (0.1x to 5x)
   - Click "Scale Selected Model"

## Tips

- **Getting Started**:
  - Choose room materials before generating the room
  - Select lighting style based on your intended use:
    - Natural for homey, residential feeling
    - Studio for clean product photography
    - Dramatic for high-end furniture showcase
  
- **Performance**:
  - Use Preview quality (64 samples) for testing
  - Medium quality (256 samples) for client reviews
  - High quality (1024 samples) for final deliverables
  
- **Best Practices**:
  - The lighting presets are optimized for furniture photography
  - Each preset includes appropriate post-processing effects
  - No manual adjustment needed - just select and apply
  - Enable "Avoid Overlaps" when importing multiple models
  - Models are automatically positioned on the floor

## Keyboard Shortcuts

- **N**: Toggle side panel in 3D Viewport
- **F12**: Render image
- **Shift+Z**: Toggle rendered viewport shading

## Requirements

- Blender 4.4 or higher
- GPU with Cycles support recommended for faster rendering
- 8GB+ RAM recommended for complex scenes