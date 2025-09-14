# Dynamic Transparency in Philo Environment Viewer

## How It Works

The dynamic transparency feature creates a Spline-like effect where walls automatically become transparent based on their distance from the camera. This allows you to see inside rooms while maintaining the sense of depth and space.

## Technical Implementation

### Node-Based Shader System

The transparency effect is achieved through a custom shader node setup that's applied to all room materials:

1. **Camera Data Node** - Captures the view depth (distance from camera to surface)
2. **Math Operations** - Process the depth value to create a transparency gradient:
   - Divide by threshold (2.0) to normalize distance
   - Smooth transition function for natural falloff  
   - Invert values (closer = more transparent)
   - Power function (8.0) for sharp cutoff
   - Greater-than threshold for complete transparency
3. **Mix Shader** - Blends between transparent and original material based on calculated value
4. **Material Settings** - Configured for proper transparency rendering

### Key Parameters

- **Distance Threshold**: 2.0 units - Walls closer than this start becoming transparent
- **Power Falloff**: 8.0 - Creates sharp transition (higher = sharper edge)
- **Smoothness**: 0.5 - Blend factor for smooth transitions
- **Transparency Threshold**: 0.1 - Minimum value for complete transparency

## Usage

### Applying Dynamic Transparency

1. Load your room model first
2. Click "Make Walls Transparent" button in the UI
3. The viewport will automatically switch to Material Preview mode
4. Move the camera around - walls will become transparent as you get closer

### Resetting Transparency

Click the "Reset" button to restore all walls to their original opaque state.

## How It Achieves Spline-Like Behavior

The system mimics Spline's transparency by:

1. **Real-time Updates** - Transparency recalculates every frame based on camera position
2. **Sharp Cutoff** - High power value (8.0) creates distinct transparent/opaque boundary
3. **Backface Culling** - Hides inside faces of walls for cleaner see-through effect
4. **Distance-Based** - Uses actual 3D distance, not just viewport position

## Viewport Settings

For best results:
- Use Material Preview shading mode (automatically set)
- Enable Cycles renderer for accurate transparency
- Transparent background enabled in render settings

## Performance Notes

- The effect runs entirely on GPU through shader nodes
- No Python scripts running per frame
- Minimal performance impact even with complex scenes
- Works with any number of wall materials

## Troubleshooting

If transparency isn't working:
1. Ensure room is loaded as "Room_Environment" parent
2. Check that materials have "Use Nodes" enabled
3. Verify Cycles renderer is active
4. Try resetting and reapplying the effect

## Technical Details for Developers

The shader node tree structure:
```
Camera Data → Math Divide → Smooth Min → Subtract → Power → Greater Than → Mix Shader → Output
                ↓             ↓            ↓          ↓         ↓              ↑
             (depth/2.0)   (smooth 0.5)  (1.0-x)   (x^8.0)   (x>0.1)    Transparent BSDF
                                                                         Original BSDF
```

Material blend settings:
- Blend Method: BLEND
- Backface Culling: True  
- Show Transparent Back: True
- Transparent Shadows: True