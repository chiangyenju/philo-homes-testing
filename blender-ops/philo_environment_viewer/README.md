# Philo Environment Viewer - Simplified

A streamlined Blender addon for loading room environments and placing furniture.

## Simple 4-Step Workflow

### 1. Load Room
- Browse to your mesh_texture.glb file
- Adjust room scale if needed
- Click "Load Room"

### 2. Apply Lighting
- Click "Apply Interior Lighting" 
- Automatically sets up 3-point lighting for interiors

### 3. Add Furniture
- Browse to furniture model (GLB/FBX/OBJ)
- Adjust furniture scale to fit room
- Click "Import Furniture"
- Use G to move, R to rotate

### 4. Camera & Render
- Click "Add Camera" to place camera inside room
- Press Numpad 0 to see camera view
- Click "Render" for final image

## Installation

Run the provided install script:
```bash
./quick_install_env_viewer.sh
```

Then in Blender:
1. Edit > Preferences > Add-ons
2. Enable "Philo Environment Viewer"
3. Find it in 3D Viewport > Sidebar > Environment tab

## Tips

- Start with room scale at 1.0, adjust if needed
- Furniture scale usually needs to be 0.1-0.5
- Camera is placed at eye level (1.6m)
- Renders at 1920x1080 with denoising