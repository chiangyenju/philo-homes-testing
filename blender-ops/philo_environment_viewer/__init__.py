"""
Philo Environment Viewer - Blender Addon for 3D Interior Environment Visualization

This addon provides tools for loading and exploring 3D interior environments:
- Load GLB/GLTF environment meshes
- Place and arrange 3D models within the environment
- Apply professional lighting setups
- Configure cameras for architectural visualization
- Render high-quality snapshots with effects

Main Components:
- operators.py: Core functionality for environment and model operations
- props.py: Scene properties and settings storage
- ui.py: User interface panels in the 3D viewport sidebar
"""

bl_info = {
    "name": "Philo Environment Viewer",
    "blender": (4, 0, 0),
    "category": "3D View",
    "version": (1, 0, 0),
    "author": "Philo Homes",
    "description": "Load and explore 3D interior environments with model placement",
    "location": "View3D > Sidebar > Environment Tab",
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