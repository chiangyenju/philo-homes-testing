"""
Operators Module - Core functionality for the Philo Interior Generator

This module contains all the Blender operators that implement the addon's features:

Room Generation:
- PHILO_OT_generate_room: Creates a customizable room with walls, floor, and ceiling

Lighting:
- PHILO_OT_setup_lighting: Applies professional lighting presets for different moods
  - Natural: Warm daylight through windows
  - Studio: Clean, even lighting for product showcase
  - Dramatic: High contrast with accent lighting

Model Import:
- PHILO_OT_import_model: Import single 3D model with automatic floor placement
- PHILO_OT_import_folder: Batch import with intelligent X-axis arrangement

Tools:
- PHILO_OT_scale_model: Scale selected models with floor repositioning
- PHILO_OT_setup_camera: Position camera for optimal interior view
- PHILO_OT_render_snapshot: Render with quality presets
- PHILO_OT_add_collision: Add physics collision to objects

Helper Functions:
- get_object_bounds: Calculate object bounding box
- check_collision: Detect overlapping objects
- position_on_floor: Place objects on ground level
- find_valid_position: Find non-overlapping placement
"""

import bpy
import os
import math
import random
from mathutils import Vector
from bpy.types import Operator

# Global list to track placed furniture positions
placed_furniture = []

class PHILO_OT_generate_room(Operator):
    bl_idname = "philo.generate_room"
    bl_label = "Generate Room"
    bl_description = "Generate a room with ceiling, floors, and walls"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Clear scene and reset furniture tracking
        global placed_furniture
        placed_furniture = []
        
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Room dimensions (can be overridden by custom property)
        scene = context.scene
        room_size = getattr(scene, 'philo_room_size', 8)
        wall_height = 6
        
        # Create floor
        bpy.ops.mesh.primitive_plane_add(size=room_size, location=(0, 0, 0))
        floor = context.active_object
        floor.name = "Floor"
        
        # Create walls
        walls = []
        # Back wall
        bpy.ops.mesh.primitive_plane_add(
            size=room_size, 
            location=(0, -room_size/2, wall_height/2), 
            rotation=(math.radians(90), 0, 0)
        )
        walls.append(context.active_object)
        context.active_object.name = "Wall_Back"
        
        # Left wall
        bpy.ops.mesh.primitive_plane_add(
            size=room_size, 
            location=(-room_size/2, 0, wall_height/2), 
            rotation=(math.radians(90), 0, math.radians(90))
        )
        walls.append(context.active_object)
        context.active_object.name = "Wall_Left"
        
        # Right wall
        bpy.ops.mesh.primitive_plane_add(
            size=room_size, 
            location=(room_size/2, 0, wall_height/2), 
            rotation=(math.radians(90), 0, math.radians(-90))
        )
        walls.append(context.active_object)
        context.active_object.name = "Wall_Right"
        
        # Ceiling
        bpy.ops.mesh.primitive_plane_add(
            size=room_size, 
            location=(0, 0, wall_height),
            rotation=(0, 0, 0)
        )
        ceiling = context.active_object
        ceiling.name = "Ceiling"
        
        # Apply materials
        self._apply_room_materials(floor, walls, ceiling, context.scene)
        
        self.report({'INFO'}, "Room generated successfully")
        return {'FINISHED'}
    
    def _apply_room_materials(self, floor, walls, ceiling, scene):
        # Floor material based on selection
        floor_mat = self._create_floor_material(scene.philo_floor_material)
        floor.data.materials.append(floor_mat)
        
        # Wall material based on selection
        wall_mat = self._create_wall_material(scene.philo_wall_material)
        for wall in walls:
            wall.data.materials.append(wall_mat)
        
        # Ceiling material (always white)
        ceiling.data.materials.append(wall_mat)
    
    def _create_floor_material(self, material_type):
        mat = bpy.data.materials.new(name=f"Floor_{material_type}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        principled = nodes["Principled BSDF"]
        
        # Add texture coordinate and mapping nodes for all materials
        tex_coord = nodes.new('ShaderNodeTexCoord')
        mapping = nodes.new('ShaderNodeMapping')
        mapping.inputs['Scale'].default_value = (10, 10, 10)  # Tile scale
        links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
        
        if material_type == 'WOOD':
            # Realistic wood with grain
            principled.inputs['Base Color'].default_value = (0.4, 0.3, 0.25, 1)
            principled.inputs['Roughness'].default_value = 0.4
            principled.inputs['Specular IOR Level'].default_value = 0.5
            # Add subtle bump for wood grain
            bump = nodes.new('ShaderNodeBump')
            bump.inputs['Strength'].default_value = 0.1
            links.new(bump.outputs['Normal'], principled.inputs['Normal'])
        elif material_type == 'MARBLE':
            # Luxurious marble with veining
            principled.inputs['Base Color'].default_value = (0.95, 0.95, 0.93, 1)
            principled.inputs['Roughness'].default_value = 0.05
            principled.inputs['Specular IOR Level'].default_value = 1.0
            principled.inputs['IOR'].default_value = 1.486  # Marble IOR
        elif material_type == 'CONCRETE':
            # Realistic concrete with texture
            principled.inputs['Base Color'].default_value = (0.6, 0.6, 0.58, 1)
            principled.inputs['Roughness'].default_value = 0.7
            principled.inputs['Specular IOR Level'].default_value = 0.3
        elif material_type == 'CARPET':
            # Soft carpet with fabric feel
            principled.inputs['Base Color'].default_value = (0.5, 0.4, 0.35, 1)
            principled.inputs['Roughness'].default_value = 0.85
            principled.inputs['Sheen Weight'].default_value = 0.8
            principled.inputs['Sheen Roughness'].default_value = 0.5
            principled.inputs['Specular IOR Level'].default_value = 0.1
        
        return mat
    
    def _create_wall_material(self, material_type):
        mat = bpy.data.materials.new(name=f"Wall_{material_type}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        principled = nodes["Principled BSDF"]
        
        if material_type == 'PAINT':
            # Modern matte paint
            principled.inputs['Base Color'].default_value = (0.92, 0.92, 0.9, 1)
            principled.inputs['Roughness'].default_value = 0.5
            principled.inputs['Specular IOR Level'].default_value = 0.3
            principled.inputs['Sheen Weight'].default_value = 0.1
        elif material_type == 'WALLPAPER':
            # Textured wallpaper
            principled.inputs['Base Color'].default_value = (0.88, 0.86, 0.82, 1)
            principled.inputs['Roughness'].default_value = 0.6
            principled.inputs['Specular IOR Level'].default_value = 0.2
            # Add subtle pattern bump
            bump = nodes.new('ShaderNodeBump')
            bump.inputs['Strength'].default_value = 0.05
            links.new(bump.outputs['Normal'], principled.inputs['Normal'])
        elif material_type == 'BRICK':
            # Realistic brick with mortar
            principled.inputs['Base Color'].default_value = (0.65, 0.35, 0.25, 1)
            principled.inputs['Roughness'].default_value = 0.8
            principled.inputs['Specular IOR Level'].default_value = 0.2
        elif material_type == 'PLASTER':
            # Smooth plaster finish
            principled.inputs['Base Color'].default_value = (0.96, 0.95, 0.92, 1)
            principled.inputs['Roughness'].default_value = 0.4
            principled.inputs['Specular IOR Level'].default_value = 0.25
        
        return mat

class PHILO_OT_setup_lighting(Operator):
    bl_idname = "philo.setup_lighting"
    bl_label = "Setup Lighting"
    bl_description = "Setup photorealistic lighting for furniture visualization"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        
        # Setup world environment with HDRI support
        world = bpy.data.worlds.new("Interior_World")
        scene.world = world
        world.use_nodes = True
        
        # Clear default nodes
        world.node_tree.nodes.clear()
        
        # Create node setup for HDRI
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        
        # Create nodes
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        tex_coord.location = (-800, 300)
        
        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.location = (-600, 300)
        
        env_texture = nodes.new(type='ShaderNodeTexEnvironment')
        env_texture.location = (-400, 300)
        
        # Background shader for HDRI
        bg_hdri = nodes.new(type='ShaderNodeBackground')
        bg_hdri.location = (-200, 300)
        bg_hdri.label = "HDRI Background"
        
        # Background shader for solid color fallback
        bg_solid = nodes.new(type='ShaderNodeBackground')
        bg_solid.location = (-200, 100)
        bg_solid.label = "Solid Color"
        
        # Mix shader to blend between HDRI and solid
        mix_shader = nodes.new(type='ShaderNodeMixShader')
        mix_shader.location = (0, 200)
        
        # Output
        output_node = nodes.new(type='ShaderNodeOutputWorld')
        output_node.location = (200, 200)
        
        # Connect nodes
        links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], env_texture.inputs['Vector'])
        links.new(env_texture.outputs['Color'], bg_hdri.inputs['Color'])
        links.new(bg_solid.outputs['Background'], mix_shader.inputs[1])
        links.new(bg_hdri.outputs['Background'], mix_shader.inputs[2])
        links.new(mix_shader.outputs['Shader'], output_node.inputs['Surface'])
        
        # Check if HDRI file is provided and exists
        hdri_path = scene.philo_hdri_path
        if hdri_path and os.path.exists(bpy.path.abspath(hdri_path)):
            # Load HDRI texture
            try:
                hdri_image = bpy.data.images.load(bpy.path.abspath(hdri_path))
                env_texture.image = hdri_image
                # If HDRI is loaded, we'll use it
                hdri_loaded = True
            except:
                hdri_loaded = False
                self.report({'WARNING'}, "Failed to load HDRI file")
        else:
            hdri_loaded = False
        
        # Remove existing lights
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                bpy.data.objects.remove(obj)
        
        # Create lights based on preset
        if scene.philo_lighting_preset == 'NATURAL':
            self._setup_natural_lighting(context, scene)
        elif scene.philo_lighting_preset == 'STUDIO':
            self._setup_studio_lighting(context, scene)
        else:  # DRAMATIC
            self._setup_dramatic_lighting(context, scene)
        
        # Setup render settings
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
        
        # Enhanced photorealistic light paths
        scene.cycles.max_bounces = 12  # Balanced for performance
        scene.cycles.diffuse_bounces = 4
        scene.cycles.glossy_bounces = 6
        scene.cycles.transmission_bounces = 12
        scene.cycles.transparent_max_bounces = 8
        scene.cycles.volume_bounces = 2
        
        # Additional realism settings
        scene.cycles.blur_glossy = 0.5  # Softer reflections
        scene.cycles.sample_clamp_direct = 0
        scene.cycles.sample_clamp_indirect = 10
        
        # Advanced render settings for photorealism
        scene.render.film_transparent = False
        scene.cycles.film_exposure = 1.0
        scene.cycles.use_preview_denoising = True
        
        # Better sampling
        scene.cycles.use_light_tree = True
        scene.cycles.sampling_pattern = 'TABULATED_SOBOL'
        
        # Color management for photorealism
        scene.view_settings.view_transform = 'Filmic'
        scene.sequencer_colorspace_settings.name = 'sRGB'
        
        # Apply preset-specific settings
        if scene.philo_lighting_preset == 'NATURAL':
            # Natural daylight settings
            bg_solid.inputs['Color'].default_value = (0.85, 0.9, 1.0, 1)  # Sky blue
            bg_solid.inputs['Strength'].default_value = 0.5
            bg_hdri.inputs['Strength'].default_value = 1.0
            # Use HDRI if loaded, otherwise solid color
            mix_shader.inputs['Fac'].default_value = 1.0 if hdri_loaded else 0.0
            scene.view_settings.exposure = 0.3
            scene.view_settings.look = 'Medium Contrast'
            scene.view_settings.gamma = 1.0
            self._setup_natural_effects(scene)
        elif scene.philo_lighting_preset == 'STUDIO':
            # Clean studio backdrop
            bg_solid.inputs['Color'].default_value = (0.95, 0.95, 0.95, 1)  # Light gray
            bg_solid.inputs['Strength'].default_value = 1.0
            bg_hdri.inputs['Strength'].default_value = 1.0
            # Use HDRI if loaded, otherwise solid color
            mix_shader.inputs['Fac'].default_value = 1.0 if hdri_loaded else 0.0
            scene.view_settings.exposure = 0.0
            scene.view_settings.look = 'None'  # Clean, no contrast adjustment
            scene.view_settings.gamma = 1.0
            self._setup_studio_effects(scene)
        else:  # DRAMATIC/Golden Hour
            # Warm sunset/golden hour lighting
            bg_solid.inputs['Color'].default_value = (0.9, 0.7, 0.5, 1)  # Warm sunset color
            bg_solid.inputs['Strength'].default_value = 0.4
            bg_hdri.inputs['Strength'].default_value = 1.0
            # Use HDRI if loaded, otherwise solid color
            mix_shader.inputs['Fac'].default_value = 1.0 if hdri_loaded else 0.0
            scene.view_settings.exposure = 0.2
            scene.view_settings.look = 'Medium High Contrast'
            scene.view_settings.gamma = 0.95
            self._setup_dramatic_effects(scene)
        
        self.report({'INFO'}, f"{scene.philo_lighting_preset.title()} lighting setup complete")
        return {'FINISHED'}
    
    def _setup_natural_lighting(self, context, scene):
        """Photorealistic natural daylight for furniture photography"""
        # Sun light - warm afternoon sun
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        sun = context.active_object
        sun.name = "Sun_Light"
        sun.data.energy = 3.0
        sun.data.angle = math.radians(0.526)  # Soft shadows
        sun.rotation_euler = (math.radians(55), math.radians(-30), 0)
        sun.data.color = (1, 0.95, 0.9)  # Warm sunlight
        
        # Large window light - main key light
        bpy.ops.object.light_add(type='AREA', location=(-3.8, 0, 3))
        window = context.active_object
        window.name = "Window_Light"
        window.data.shape = 'RECTANGLE'
        window.data.size = 3
        window.data.size_y = 4.5
        window.data.energy = 1200  # Increased for more realistic daylight
        window.rotation_euler = (math.radians(90), 0, math.radians(90))
        window.data.color = (1, 0.98, 0.95)
        window.data.cycles.is_portal = True
        # Spread angle not available in Blender 4.4
        
        # Soft ceiling bounce - fill light
        bpy.ops.object.light_add(type='AREA', location=(0, 0, 5.95))
        fill = context.active_object
        fill.name = "Ceiling_Bounce"
        fill.data.shape = 'DISK'
        fill.data.size = 8  # Larger for softer shadows
        fill.data.energy = 200
        fill.data.color = (0.95, 0.97, 1)  # Slightly cool
        fill.rotation_euler = (math.radians(180), 0, 0)
        fill.data.use_shadow = False
        # Spread angle not available in Blender 4.4
        
        # Subtle rim light from opposite side
        bpy.ops.object.light_add(type='AREA', location=(3, -2, 2))
        rim = context.active_object
        rim.name = "Rim_Light"
        rim.data.size = 1.5
        rim.data.energy = 200
        rim.data.color = (1, 0.97, 0.93)
        rim.rotation_euler = (math.radians(60), math.radians(45), 0)
    
    def _setup_studio_lighting(self, context, scene):
        """Professional studio lighting for product photography"""
        # Large softbox key light
        bpy.ops.object.light_add(type='AREA', location=(-2.5, 2, 3))
        key = context.active_object
        key.name = "Softbox_Key"
        key.data.shape = 'RECTANGLE'
        key.data.size = 2.5
        key.data.size_y = 2.5
        key.data.energy = 1500
        key.rotation_euler = (math.radians(30), math.radians(-40), 0)
        key.data.color = (1, 1, 1)  # Pure white
        
        # Large fill light - opposite side
        bpy.ops.object.light_add(type='AREA', location=(3, 1.5, 2.5))
        fill = context.active_object
        fill.name = "Fill_Light"
        fill.data.shape = 'RECTANGLE'
        fill.data.size = 3
        fill.data.size_y = 3
        fill.data.energy = 600
        fill.rotation_euler = (math.radians(45), math.radians(50), 0)
        fill.data.color = (1, 1, 1)
        
        # Top light for even illumination
        bpy.ops.object.light_add(type='AREA', location=(0, 0, 5.5))
        top = context.active_object
        top.name = "Top_Light"
        top.data.shape = 'DISK'
        top.data.size = 2
        top.data.energy = 400
        top.rotation_euler = (math.radians(180), 0, 0)
        top.data.color = (1, 1, 1)
        
        # Background light
        bpy.ops.object.light_add(type='AREA', location=(0, -3.5, 1))
        bg_light = context.active_object
        bg_light.name = "Background_Light"
        bg_light.data.shape = 'RECTANGLE'
        bg_light.data.size = 4
        bg_light.data.size_y = 2
        bg_light.data.energy = 300
        bg_light.rotation_euler = (math.radians(90), 0, 0)
        bg_light.data.color = (1, 1, 1)
    
    def _setup_dramatic_lighting(self, context, scene):
        """Golden hour lighting for warm, inviting atmosphere"""
        # Warm golden sun light - main key
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        sun = context.active_object
        sun.name = "Golden_Sun"
        sun.data.energy = 2.5
        sun.data.angle = math.radians(2.5)  # Larger sun disk for softer shadows
        sun.rotation_euler = (math.radians(35), math.radians(-45), 0)
        sun.data.color = (1, 0.85, 0.6)  # Golden hour color
        
        # Large warm window light
        bpy.ops.object.light_add(type='AREA', location=(-3.8, -2, 3))
        window = context.active_object
        window.name = "Sunset_Window"
        window.data.shape = 'RECTANGLE'
        window.data.size = 3.5
        window.data.size_y = 4.5
        window.data.energy = 1500
        window.rotation_euler = (math.radians(75), math.radians(15), math.radians(90))
        window.data.color = (1, 0.75, 0.5)  # Deep golden sunset
        
        # Soft ambient fill from opposite side
        bpy.ops.object.light_add(type='AREA', location=(3.5, 2, 4))
        fill = context.active_object
        fill.name = "Ambient_Fill"
        fill.data.shape = 'RECTANGLE'
        fill.data.size = 4
        fill.data.size_y = 3
        fill.data.energy = 300
        fill.data.color = (0.9, 0.85, 1)  # Slightly cool to balance warmth
        fill.rotation_euler = (math.radians(45), math.radians(-30), 0)
        
        # Warm ceiling bounce
        bpy.ops.object.light_add(type='AREA', location=(0, 0, 5.9))
        bounce = context.active_object
        bounce.name = "Ceiling_Glow"
        bounce.data.shape = 'DISK'
        bounce.data.size = 6
        bounce.data.energy = 250
        bounce.data.color = (1, 0.9, 0.8)  # Warm bounce light
        bounce.rotation_euler = (math.radians(180), 0, 0)
        bounce.data.use_shadow = False
    
    def _setup_natural_effects(self, scene):
        """Setup effects for natural lighting preset"""
        # Disable compositor effects - direct output only
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
        
        # Simple passthrough - no effects
        render_layers = tree.nodes.new('CompositorNodeRLayers')
        composite = tree.nodes.new('CompositorNodeComposite')
        tree.links.new(render_layers.outputs['Image'], composite.inputs['Image'])
        
        # Enhanced render settings for photorealism
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.01
        scene.cycles.use_fast_gi = True
        
        # Better light sampling
        scene.cycles.light_sampling_threshold = 0.01
        scene.cycles.sample_clamp_indirect = 10.0
        
        # Caustics for realistic glass/water
        scene.cycles.caustics_reflective = True
        scene.cycles.caustics_refractive = True
    
    def _setup_studio_effects(self, scene):
        """Setup effects for studio lighting preset"""
        # Disable compositor effects - direct output only
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
        
        # Simple passthrough - no effects
        render_layers = tree.nodes.new('CompositorNodeRLayers')
        composite = tree.nodes.new('CompositorNodeComposite')
        tree.links.new(render_layers.outputs['Image'], composite.inputs['Image'])
        
        # Studio-specific render settings
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.005  # Higher quality for studio
        
        # Clean shadows
        scene.cycles.sample_clamp_direct = 0  # No clamping for clean highlights
        scene.cycles.sample_clamp_indirect = 0
    
    def _setup_dramatic_effects(self, scene):
        """Setup effects for dramatic lighting preset"""
        # Disable compositor effects - direct output only
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
        
        # Simple passthrough - no effects
        render_layers = tree.nodes.new('CompositorNodeRLayers')
        composite = tree.nodes.new('CompositorNodeComposite')
        tree.links.new(render_layers.outputs['Image'], composite.inputs['Image'])
        
        # Dramatic-specific render settings
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.01
        
        # Allow some light bleeding for mood
        scene.cycles.sample_clamp_indirect = 20.0
        scene.cycles.sample_clamp_direct = 0

def get_object_bounds(obj):
    """Get the bounding box of an object in world space"""
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    min_co = Vector((
        min(corner.x for corner in bbox_corners),
        min(corner.y for corner in bbox_corners),
        min(corner.z for corner in bbox_corners)
    ))
    max_co = Vector((
        max(corner.x for corner in bbox_corners),
        max(corner.y for corner in bbox_corners),
        max(corner.z for corner in bbox_corners)
    ))
    
    return min_co, max_co

def check_collision(new_min, new_max, margin=0.2):
    """Check if a new object would collide with existing furniture"""
    global placed_furniture
    
    for existing_min, existing_max in placed_furniture:
        # Check overlap in all axes with margin
        if (new_min.x - margin < existing_max.x and new_max.x + margin > existing_min.x and
            new_min.y - margin < existing_max.y and new_max.y + margin > existing_min.y and
            new_min.z - margin < existing_max.z and new_max.z + margin > existing_min.z):
            return True
    return False

def find_valid_position(obj, room_size=8, margin=0.2, max_attempts=50):
    """Find a valid position for an object that doesn't collide with others"""
    obj_min, obj_max = get_object_bounds(obj)
    obj_size = obj_max - obj_min
    
    # Room boundaries (leaving space from walls)
    wall_margin = 0.5
    x_range = (-room_size/2 + wall_margin + obj_size.x/2, room_size/2 - wall_margin - obj_size.x/2)
    y_range = (-room_size/2 + wall_margin + obj_size.y/2, room_size/2 - wall_margin - obj_size.y/2)
    
    for attempt in range(max_attempts):
        # Try random positions
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        
        # Calculate new position
        old_location = obj.location.copy()
        obj.location = (x, y, obj.location.z)
        
        # Update and check bounds
        bpy.context.view_layer.update()
        new_min, new_max = get_object_bounds(obj)
        
        if not check_collision(new_min, new_max, margin):
            return True
        
        # Restore old position
        obj.location = old_location
    
    return False

def position_on_floor(obj):
    """Position object so it sits on the floor"""
    # Update to get accurate bounds
    bpy.context.view_layer.update()
    
    # Find lowest point
    if obj.type == 'MESH':
        if obj.data.vertices:
            min_z = min((obj.matrix_world @ v.co).z for v in obj.data.vertices)
            obj.location.z -= min_z
    elif obj.type == 'EMPTY' and obj.children:
        # For groups, find lowest point among all children
        min_z = float('inf')
        for child in obj.children:
            if child.type == 'MESH' and child.data.vertices:
                child_min_z = min((child.matrix_world @ v.co).z for v in child.data.vertices)
                min_z = min(min_z, child_min_z)
        if min_z != float('inf'):
            obj.location.z -= min_z

class PHILO_OT_import_model(Operator):
    bl_idname = "philo.import_model"
    bl_label = "Import Single Model"
    bl_description = "Import a single 3D model"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        global placed_furniture
        scene = context.scene
        filepath = scene.philo_model_path
        
        if not filepath or not os.path.exists(bpy.path.abspath(filepath)):
            self.report({'ERROR'}, "Please select a valid model file")
            return {'CANCELLED'}
        
        filepath = bpy.path.abspath(filepath)
        
        # Store objects before import
        objects_before = set(context.scene.objects)
        
        # Import based on file extension
        if filepath.lower().endswith('.glb') or filepath.lower().endswith('.gltf'):
            bpy.ops.import_scene.gltf(filepath=filepath)
        elif filepath.lower().endswith('.fbx'):
            bpy.ops.import_scene.fbx(filepath=filepath)
        elif filepath.lower().endswith('.obj'):
            bpy.ops.wm.obj_import(filepath=filepath)
        else:
            self.report({'ERROR'}, "Unsupported file format")
            return {'CANCELLED'}
        
        # Get newly imported objects
        new_objects = list(set(context.scene.objects) - objects_before)
        
        if new_objects:
            # Create parent if multiple objects
            if len(new_objects) > 1:
                bpy.ops.object.empty_add(location=(0, 0, 0))
                parent = context.active_object
                parent.name = f"Imported_{os.path.basename(filepath).split('.')[0]}"
                for obj in new_objects:
                    obj.parent = parent
                main_obj = parent
            else:
                main_obj = new_objects[0]
            
            # Apply default 180-degree Z rotation to face correct direction
            main_obj.rotation_euler = (0, 0, math.radians(180))
            
            # Position on floor first
            position_on_floor(main_obj)
            
            # Find valid position if collision avoidance is enabled
            if scene.philo_use_collision and placed_furniture:
                if find_valid_position(main_obj, margin=scene.philo_collision_margin):
                    self.report({'INFO'}, f"Placed {os.path.basename(filepath)} at valid position")
                else:
                    # If no valid position found, place at center
                    main_obj.location.x = 0
                    main_obj.location.y = 0
                    self.report({'WARNING'}, "No valid position found, placed at center")
            
            # Update bounds and add to placed furniture list
            bpy.context.view_layer.update()
            bounds = get_object_bounds(main_obj)
            placed_furniture.append(bounds)
            
            # Apply smooth shading
            for obj in new_objects:
                if obj.type == 'MESH':
                    obj.select_set(True)
                    context.view_layer.objects.active = obj
                    bpy.ops.object.shade_smooth()
        
        self.report({'INFO'}, f"Imported: {os.path.basename(filepath)}")
        return {'FINISHED'}

class PHILO_OT_import_folder(Operator):
    bl_idname = "philo.import_folder"
    bl_label = "Import Folder"
    bl_description = "Import all 3D models from a folder"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        global placed_furniture
        scene = context.scene
        folder_path = scene.philo_folder_path
        
        if not folder_path or not os.path.exists(bpy.path.abspath(folder_path)):
            self.report({'ERROR'}, "Please select a valid folder")
            return {'CANCELLED'}
        
        folder_path = bpy.path.abspath(folder_path)
        
        # Find all supported files
        supported_extensions = ['.glb', '.gltf', '.fbx', '.obj']
        model_files = []
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                model_files.append(os.path.join(folder_path, file))
        
        if not model_files:
            self.report({'ERROR'}, "No supported 3D files found in folder")
            return {'CANCELLED'}
        
        # Import models and measure their sizes first
        imported_objects = []
        imported_count = 0
        
        for i, filepath in enumerate(model_files):
            # Store objects before import
            objects_before = set(context.scene.objects)
            
            try:
                # Import model
                if filepath.lower().endswith('.glb') or filepath.lower().endswith('.gltf'):
                    bpy.ops.import_scene.gltf(filepath=filepath)
                elif filepath.lower().endswith('.fbx'):
                    bpy.ops.import_scene.fbx(filepath=filepath)
                elif filepath.lower().endswith('.obj'):
                    bpy.ops.wm.obj_import(filepath=filepath)
                
                # Get newly imported objects
                new_objects = list(set(context.scene.objects) - objects_before)
                
                if new_objects:
                    # Create parent if multiple objects
                    if len(new_objects) > 1:
                        bpy.ops.object.empty_add(location=(0, 0, 0))
                        parent = context.active_object
                        parent.name = f"Model_{os.path.basename(filepath).split('.')[0]}"
                        for obj in new_objects:
                            obj.parent = parent
                        main_obj = parent
                    else:
                        main_obj = new_objects[0]
                    
                    # Apply default 180-degree Z rotation to face correct direction
                    main_obj.rotation_euler = (0, 0, math.radians(180))
                    
                    # Position on floor first (before storing in list)
                    position_on_floor(main_obj)
                    
                    # Store imported object info
                    imported_objects.append(main_obj)
                    
                    # Apply smooth shading
                    for obj in new_objects:
                        if obj.type == 'MESH':
                            obj.select_set(True)
                            context.view_layer.objects.active = obj
                            bpy.ops.object.shade_smooth()
                    
                    imported_count += 1
                    
            except Exception as e:
                self.report({'WARNING'}, f"Failed to import: {os.path.basename(filepath)} - {str(e)}")
        
        # Now arrange all imported objects in a line along X-axis
        if imported_objects:
            # Calculate total width needed
            total_width = 0
            model_widths = []
            
            for obj in imported_objects:
                obj_min, obj_max = get_object_bounds(obj)
                width = obj_max.x - obj_min.x
                model_widths.append(width)
                total_width += width
            
            # Add spacing between models
            spacing = 3.0  # Increased space between models for better visibility
            total_width += spacing * (len(imported_objects) - 1)
            
            # Check if we need a larger room
            current_room_size = getattr(scene, 'philo_room_size', 8)
            needed_room_size = max(current_room_size, total_width + 2)  # +2 for wall margins
            
            if needed_room_size > current_room_size:
                # Store the needed room size for next room generation
                scene.philo_room_size = needed_room_size
                self.report({'WARNING'}, f"Room size should be at least {needed_room_size:.1f}m. Please regenerate the room.")
            
            # Position models in a line
            current_x = -total_width / 2
            
            for i, obj in enumerate(imported_objects):
                # Get current bounds
                obj_min, obj_max = get_object_bounds(obj)
                width = model_widths[i]
                
                # Calculate center offset
                center_offset = (obj_min.x + obj_max.x) / 2
                
                # Position model accounting for its center
                obj.location.x = current_x + width / 2 - center_offset
                obj.location.y = 0  # Center along Y
                
                # Ensure it's still on the floor after moving
                position_on_floor(obj)
                
                # Update for next model
                current_x += width + spacing
                
                # Update bounds
                bpy.context.view_layer.update()
                bounds = get_object_bounds(obj)
                placed_furniture.append(bounds)
        
        self.report({'INFO'}, f"Imported {imported_count} models")
        return {'FINISHED'}

class PHILO_OT_scale_model(Operator):
    bl_idname = "philo.scale_model"
    bl_label = "Scale Selected Model"
    bl_description = "Scale the selected model"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        selected_objects = context.selected_objects
        
        if not selected_objects:
            self.report({'ERROR'}, "Please select a model first")
            return {'CANCELLED'}
        
        # Scale all selected objects
        for obj in selected_objects:
            obj.scale = (scene.philo_model_scale, scene.philo_model_scale, scene.philo_model_scale)
            
            # Also scale children if it's an empty
            if obj.type == 'EMPTY' and obj.children:
                for child in obj.children:
                    child.scale = (scene.philo_model_scale, scene.philo_model_scale, scene.philo_model_scale)
        
        # Update scene
        bpy.context.view_layer.update()
        
        # Reposition on floor after scaling to prevent sinking
        for obj in selected_objects:
            position_on_floor(obj)
        
        self.report({'INFO'}, f"Scaled {len(selected_objects)} objects to {scene.philo_model_scale}x and repositioned on floor")
        return {'FINISHED'}

class PHILO_OT_setup_camera(Operator):
    bl_idname = "philo.setup_camera"
    bl_label = "Setup Camera"
    bl_description = "Setup camera to view room from a distance with slight downward angle"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        room_size = getattr(scene, 'philo_room_size', 8)
        
        # Remove existing cameras
        for obj in bpy.data.objects:
            if obj.type == 'CAMERA':
                bpy.data.objects.remove(obj)
        
        # Create new camera
        cam_data = bpy.data.cameras.new("Main_Camera")
        cam_obj = bpy.data.objects.new("Main_Camera", cam_data)
        context.scene.collection.objects.link(cam_obj)
        context.scene.camera = cam_obj
        
        # Position camera at fixed location
        cam_obj.location = (0, 12.5, 5)  # X=0, Y=12.5, Z=5
        
        # Set camera rotation
        # X rotation: 75 degrees
        # Y rotation: 0 degrees
        # Z rotation: 180 degrees to fix orientation
        cam_obj.rotation_euler = (math.radians(75), 0, math.radians(180))
        
        # Camera settings for interior view
        # Adjust lens based on room size
        cam_data.lens = min(35, 24 + (room_size - 8) * 0.5)  # Wider for larger rooms
        cam_data.sensor_width = 36
        cam_data.dof.use_dof = True
        cam_data.dof.aperture_fstop = 5.6
        cam_data.dof.focus_distance = 12.5  # Focus distance matches Y position
        
        self.report({'INFO'}, "Camera positioned at Y=12.5, Z=5 with 75° angle")
        return {'FINISHED'}

class PHILO_OT_render_snapshot(Operator):
    bl_idname = "philo.render_snapshot"
    bl_label = "Render Snapshot"
    bl_description = "Take a snapshot from camera view"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        
        # Ensure camera exists
        if not scene.camera:
            bpy.ops.philo.setup_camera()
        
        # Apply quality settings before rendering
        quality_settings = {
            'PREVIEW': {'samples': 128, 'denoising': True, 'denoise_type': 'OPENIMAGEDENOISE'},
            'MEDIUM': {'samples': 512, 'denoising': True, 'denoise_type': 'OPENIMAGEDENOISE'},
            'HIGH': {'samples': 2048, 'denoising': True, 'denoise_type': 'OPENIMAGEDENOISE'}
        }
        
        settings = quality_settings[scene.philo_render_quality]
        scene.cycles.samples = settings['samples']
        scene.cycles.use_denoising = settings['denoising']
        scene.cycles.denoiser = settings['denoise_type']
        
        # Quality-specific settings
        if scene.philo_render_quality == 'HIGH':
            scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
            scene.cycles.denoising_prefilter = 'ACCURATE'
            scene.render.resolution_percentage = 100
        elif scene.philo_render_quality == 'MEDIUM':
            scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
            scene.cycles.denoising_prefilter = 'FAST'
            scene.render.resolution_percentage = 100
        else:  # PREVIEW
            scene.cycles.denoising_input_passes = 'RGB_ALBEDO'
            scene.cycles.denoising_prefilter = 'NONE'
            scene.render.resolution_percentage = 75  # Lower res for preview
        
        # Start render
        bpy.ops.render.render('INVOKE_DEFAULT')
        
        self.report({'INFO'}, f"Rendering with {settings['samples']} samples")
        return {'FINISHED'}

class PHILO_OT_add_collision(Operator):
    bl_idname = "philo.add_collision"
    bl_label = "Add Physics Collision"
    bl_description = "Add collision physics to selected objects"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        selected_objects = context.selected_objects
        
        if not selected_objects:
            self.report({'ERROR'}, "Please select objects first")
            return {'CANCELLED'}
        
        for obj in selected_objects:
            if obj.type == 'MESH':
                # Enable rigid body physics
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_add()
                
                # Set to passive (doesn't fall)
                obj.rigid_body.type = 'PASSIVE'
                
                # Use mesh collision shape for accuracy
                obj.rigid_body.collision_shape = 'MESH'
                
        self.report({'INFO'}, f"Added collision to {len(selected_objects)} objects")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(PHILO_OT_generate_room)
    bpy.utils.register_class(PHILO_OT_setup_lighting)
    bpy.utils.register_class(PHILO_OT_import_model)
    bpy.utils.register_class(PHILO_OT_import_folder)
    bpy.utils.register_class(PHILO_OT_scale_model)
    bpy.utils.register_class(PHILO_OT_setup_camera)
    bpy.utils.register_class(PHILO_OT_render_snapshot)
    bpy.utils.register_class(PHILO_OT_add_collision)

def unregister():
    bpy.utils.unregister_class(PHILO_OT_add_collision)
    bpy.utils.unregister_class(PHILO_OT_render_snapshot)
    bpy.utils.unregister_class(PHILO_OT_setup_camera)
    bpy.utils.unregister_class(PHILO_OT_scale_model)
    bpy.utils.unregister_class(PHILO_OT_import_folder)
    bpy.utils.unregister_class(PHILO_OT_import_model)
    bpy.utils.unregister_class(PHILO_OT_setup_lighting)
    bpy.utils.unregister_class(PHILO_OT_generate_room)