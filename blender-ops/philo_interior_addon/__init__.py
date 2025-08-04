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