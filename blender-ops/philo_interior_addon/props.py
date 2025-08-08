"""
Properties Module - Scene Properties for Philo Interior Generator

This module defines all the custom properties that store addon settings:

Lighting Properties:
- philo_lighting_preset: Choose between Natural, Studio, or Dramatic lighting

Import Properties:
- philo_model_path: Path to single 3D model file
- philo_folder_path: Path to folder containing multiple models

Material Properties:
- philo_floor_material: Floor type (Wood, Marble, Concrete, Carpet)
- philo_wall_material: Wall type (Paint, Wallpaper, Brick, Plaster)

Transform Properties:
- philo_model_scale: Scale factor for selected models
- philo_room_size: Room dimensions in meters (4-50m)

Collision Properties:
- philo_use_collision: Enable/disable overlap prevention
- philo_collision_margin: Minimum spacing between objects

Render Properties:
- philo_render_quality: Preview (64), Medium (256), or High (2048) samples
"""

import bpy
from bpy.props import StringProperty, FloatProperty, EnumProperty, BoolProperty, IntProperty

def register():
    # Lighting preset
    bpy.types.Scene.philo_lighting_preset = EnumProperty(
        name="Lighting Preset",
        description="Photorealistic lighting setups",
        items=[
            ('NATURAL', 'Natural', 'Warm daylight through windows'),
            ('STUDIO', 'Studio', 'Clean product photography lighting'),
            ('DRAMATIC', 'Golden Hour', 'Warm sunset lighting for cozy atmosphere'),
        ],
        default='NATURAL'
    )
    
    # Import properties
    bpy.types.Scene.philo_model_path = StringProperty(
        name="Model Path",
        description="Path to 3D model file",
        default="",
        subtype='FILE_PATH'
    )
    
    bpy.types.Scene.philo_folder_path = StringProperty(
        name="Folder Path",
        description="Path to folder containing 3D models",
        default="",
        subtype='DIR_PATH'
    )
    
    # Material properties
    bpy.types.Scene.philo_floor_material = EnumProperty(
        name="Floor Material",
        description="Choose floor material type",
        items=[
            ('WOOD', 'Wood', 'Wooden floor'),
            ('MARBLE', 'Marble', 'Marble floor'),
            ('CONCRETE', 'Concrete', 'Concrete floor'),
            ('CARPET', 'Carpet', 'Carpeted floor'),
        ],
        default='WOOD'
    )
    
    bpy.types.Scene.philo_wall_material = EnumProperty(
        name="Wall Material",
        description="Choose wall material type",
        items=[
            ('PAINT', 'Paint', 'Painted walls'),
            ('WALLPAPER', 'Wallpaper', 'Wallpaper texture'),
            ('BRICK', 'Brick', 'Exposed brick'),
            ('PLASTER', 'Plaster', 'Textured plaster'),
        ],
        default='PAINT'
    )
    
    # Model scale property
    bpy.types.Scene.philo_model_scale = FloatProperty(
        name="Model Scale",
        description="Scale factor for selected model",
        default=1.0,
        min=0.1,
        max=5.0,
        precision=2
    )
    
    # Collision properties
    bpy.types.Scene.philo_use_collision = BoolProperty(
        name="Avoid Overlaps",
        description="Prevent furniture from overlapping",
        default=True
    )
    
    bpy.types.Scene.philo_collision_margin = FloatProperty(
        name="Spacing",
        description="Minimum space between objects",
        default=0.2,
        min=0.0,
        max=1.0
    )
    
    # Render properties
    bpy.types.Scene.philo_render_quality = EnumProperty(
        name="Quality",
        description="Render quality preset",
        items=[
            ('PREVIEW', 'Preview', 'Fast preview - 64 samples'),
            ('MEDIUM', 'Medium', 'Balanced - 256 samples'),
            ('HIGH', 'High', 'Production - 1024 samples'),
        ],
        default='MEDIUM'
    )
    
    # Room size property
    bpy.types.Scene.philo_room_size = FloatProperty(
        name="Room Size",
        description="Size of the room in meters",
        default=8.0,
        min=4.0,
        max=50.0,
        precision=1
    )
    
    # HDRI path property
    bpy.types.Scene.philo_hdri_path = StringProperty(
        name="HDRI Path",
        description="Path to HDRI environment texture (optional)",
        default="",
        subtype='FILE_PATH'
    )

def unregister():
    del bpy.types.Scene.philo_lighting_preset
    del bpy.types.Scene.philo_model_path
    del bpy.types.Scene.philo_folder_path
    del bpy.types.Scene.philo_floor_material
    del bpy.types.Scene.philo_wall_material
    del bpy.types.Scene.philo_model_scale
    del bpy.types.Scene.philo_use_collision
    del bpy.types.Scene.philo_collision_margin
    del bpy.types.Scene.philo_render_quality
    del bpy.types.Scene.philo_room_size
    del bpy.types.Scene.philo_hdri_path