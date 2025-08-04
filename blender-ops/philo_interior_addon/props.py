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
            ('DRAMATIC', 'Dramatic', 'Moody luxury furniture lighting'),
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