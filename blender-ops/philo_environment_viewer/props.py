"""
Simplified Properties for Environment Viewer
"""

import bpy
from bpy.props import StringProperty, FloatProperty

def register():
    # Room environment
    bpy.types.Scene.env_environment_path = StringProperty(
        name="Room GLB",
        description="Path to room environment GLB file",
        default="",
        subtype='FILE_PATH'
    )
    
    bpy.types.Scene.env_room_scale = FloatProperty(
        name="Room Scale",
        description="Scale the room environment",
        default=1.0,
        min=0.1,
        max=10.0,
        precision=2
    )
    
    # Furniture
    bpy.types.Scene.env_furniture_path = StringProperty(
        name="Furniture Model",
        description="Path to furniture model file",
        default="",
        subtype='FILE_PATH'
    )
    
    bpy.types.Scene.env_furniture_scale = FloatProperty(
        name="Furniture Scale",
        description="Scale furniture to fit room",
        default=0.01,
        min=0.001,
        max=1.0,
        precision=3
    )

def unregister():
    del bpy.types.Scene.env_environment_path
    del bpy.types.Scene.env_room_scale
    del bpy.types.Scene.env_furniture_path
    del bpy.types.Scene.env_furniture_scale