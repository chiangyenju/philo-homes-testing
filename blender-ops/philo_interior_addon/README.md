# Philo Interior Generator

A clean, minimalist Blender addon for generating realistic interior scenes with furniture import and rendering capabilities.

## Features

1. **Room Generation with Material Options**
   - Creates a complete room with ceiling, floor, and walls
   - **Floor materials**: Wood, Marble, Concrete, Carpet
   - **Wall materials**: Paint, Wallpaper, Brick, Plaster
   - Optimized 8x8m dimensions for interior visualization
   - Furniture automatically positioned above floor level

2. **Advanced Lighting System**
   - Environment lighting with adjustable strength
   - Key light (main window light) with warm daylight color
   - Fill light (ceiling bounce) for soft shadows
   - Shadow control toggle
   - Bloom effect for realistic glow
   - Exposure and contrast controls

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

### Direct Installation (Recommended)

1. Download the `philo_interior_addon` folder
2. In Blender, go to Edit > Preferences > Add-ons
3. Click "Install..." and select the addon folder
4. Enable "Philo Interior Generator"

### Manual Installation

1. Copy `philo_interior_addon` to Blender's addon directory:
   - macOS: `~/Library/Application Support/Blender/4.0/scripts/addons/`
   - Windows: `%APPDATA%\Blender Foundation\Blender\4.0\scripts\addons\`
   - Linux: `~/.config/blender/4.0/scripts/addons/`

2. Restart Blender and enable the addon in Preferences

## Usage

1. **Open the Panel**: In 3D Viewport, press N and find the "Philo" tab

2. **Workflow**:
   - Choose floor and wall materials in "Room Materials" panel
   - Click "Generate Room" to create the room
   - Click "Setup Lighting" to add lights and effects
   - Import your 3D models (single file or entire folder)
   - Select models and use "Model Tools" to scale if needed
   - Click "Setup Camera" for elevated view with slight angle
   - Adjust lighting and effects as needed
   - Click "Render Snapshot" for final image

3. **Room Materials**:
   - **Floor**: Wood, Marble, Concrete, or Carpet
   - **Walls**: Paint, Wallpaper, Brick, or Plaster
   - Set before generating room

4. **Import Settings**:
   - **Avoid Overlaps**: Toggle collision detection
   - **Spacing**: Adjust minimum distance between objects (0-1m)

5. **Model Scaling**:
   - Select one or more models
   - Adjust scale slider (0.1x to 5x)
   - Click "Scale Selected Model"

6. **Lighting Adjustments**:
   - Environment Strength: Overall ambient light
   - Key Light Power: Main directional light (0-10000)
   - Fill Light Power: Soft fill light (0-5000)
   - Enable Shadows: Toggle shadow rendering
   - Bloom Effect: Add glow to bright areas
   - Exposure: Overall brightness (-5 to +5)
   - Contrast: Low/Medium/High options

## Tips

- Choose room materials before clicking "Generate Room"
- Start with default lighting settings and adjust gradually
- Use Preview quality for testing, High for final renders
- Enable "Avoid Overlaps" when importing multiple models
- Models are automatically positioned on the floor
- Select and scale models after import if needed
- Scaled models are automatically repositioned on floor
- Camera provides elevated view perfect for showcasing entire room
- The addon automatically applies smooth shading to imported models
- Bloom effect works best with bright light sources
- Use Physics Tools to add realistic collision for animations

## Requirements

- Blender 4.0 or higher
- GPU with Cycles support recommended for faster rendering