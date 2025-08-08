"""
Philo Interior Generator - Blender Addon for Interior Visualization

This addon provides a complete workflow for creating photorealistic interior renders:
- Automated room generation with customizable materials
- Professional lighting presets (Natural, Studio, Dramatic)
- Batch furniture import with intelligent placement
- Camera setup optimized for interior visualization
- High-quality rendering with realistic settings

Main Components:
- operators.py: Core functionality (room generation, lighting, import, rendering)
- props.py: Scene properties and settings storage
- ui.py: User interface panels in the 3D viewport sidebar
"""

bl_info = {
    "name": "Philo Interior Generator",
    "blender": (4, 0, 0),
    "category": "3D View",
    "version": (1, 0, 0),
    "author": "Philo Homes",
    "description": "Generate realistic interior scenes with furniture import and rendering",
    "location": "View3D > Sidebar > Philo Tab",
}

import bpy
from . import operators, ui, props

def register():
    props.register()
    operators.register()
    ui.register()

def unregister():
    ui.unregister()
    operators.unregister()
    props.unregister()

if __name__ == "__main__":
    register()