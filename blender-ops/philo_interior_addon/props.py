import bpy
from bpy.props import StringProperty, FloatProperty, EnumProperty, BoolProperty, IntProperty

def register():
    # Lighting properties
    bpy.types.Scene.philo_hdri_strength = FloatProperty(
        name="Environment Strength",
        description="Strength of HDRI lighting",
        default=0.5,
        min=0.0,
        max=5.0
    )
    
    bpy.types.Scene.philo_key_light_strength = FloatProperty(
        name="Key Light Power",
        description="Main light intensity",
        default=1000,
        min=0,
        max=10000
    )
    
    bpy.types.Scene.philo_fill_light_strength = FloatProperty(
        name="Fill Light Power",
        description="Fill light intensity",
        default=300,
        min=0,
        max=5000
    )
    
    bpy.types.Scene.philo_use_shadows = BoolProperty(
        name="Enable Shadows",
        description="Enable shadow rendering",
        default=True
    )
    
    # Effects properties
    bpy.types.Scene.philo_use_bloom = BoolProperty(
        name="Bloom Effect",
        description="Add glow to bright areas",
        default=True
    )
    
    bpy.types.Scene.philo_bloom_intensity = FloatProperty(
        name="Bloom Intensity",
        description="Strength of bloom effect",
        default=0.02,
        min=0.0,
        max=0.5,
        precision=3
    )
    
    bpy.types.Scene.philo_exposure = FloatProperty(
        name="Exposure",
        description="Overall brightness adjustment",
        default=0.0,
        min=-5.0,
        max=5.0
    )
    
    bpy.types.Scene.philo_contrast = EnumProperty(
        name="Contrast",
        description="Color contrast level",
        items=[
            ('LOW', 'Low', 'Low contrast for soft look'),
            ('MEDIUM', 'Medium', 'Balanced contrast'),
            ('HIGH', 'High', 'High contrast for dramatic look'),
        ],
        default='MEDIUM'
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
    del bpy.types.Scene.philo_hdri_strength
    del bpy.types.Scene.philo_key_light_strength
    del bpy.types.Scene.philo_fill_light_strength
    del bpy.types.Scene.philo_use_shadows
    del bpy.types.Scene.philo_use_bloom
    del bpy.types.Scene.philo_bloom_intensity
    del bpy.types.Scene.philo_exposure
    del bpy.types.Scene.philo_contrast
    del bpy.types.Scene.philo_model_path
    del bpy.types.Scene.philo_folder_path
    del bpy.types.Scene.philo_floor_material
    del bpy.types.Scene.philo_wall_material
    del bpy.types.Scene.philo_model_scale
    del bpy.types.Scene.philo_use_collision
    del bpy.types.Scene.philo_collision_margin
    del bpy.types.Scene.philo_render_quality