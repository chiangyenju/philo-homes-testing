"""
Simplified Environment Viewer Operators
"""

import bpy
import os
import math
from mathutils import Vector
from bpy.types import Operator
from collections import Counter

class ENV_OT_load_environment(Operator):
    bl_idname = "env.load_environment"
    bl_label = "Load Room Environment"
    bl_description = "Load room environment GLB file"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        filepath = scene.env_environment_path
        
        if not filepath or not os.path.exists(bpy.path.abspath(filepath)):
            self.report({'ERROR'}, "Please select a valid GLB file")
            return {'CANCELLED'}
        
        filepath = bpy.path.abspath(filepath)
        
        # Clear scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Import GLB
        bpy.ops.import_scene.gltf(filepath=filepath)
        
        # Group all imported objects under one parent
        imported_objects = context.selected_objects
        if imported_objects:
            bpy.ops.object.empty_add(location=(0, 0, 0))
            room_parent = context.active_object
            room_parent.name = "Room_Environment"
            room_parent["is_environment"] = True
            
            for obj in imported_objects:
                obj.parent = room_parent
                obj["is_environment"] = True
            
            # Apply scale
            room_parent.scale = (scene.env_room_scale,) * 3
            
            # Rotate 90 degrees on X axis to stand upright
            room_parent.rotation_euler = (math.radians(90), 0, 0)
            
            # Apply rotation and scale
            bpy.ops.object.select_all(action='DESELECT')
            room_parent.select_set(True)
            context.view_layer.objects.active = room_parent
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
            
            # Set origin to center of geometry for all room objects
            for obj in room_parent.children:
                if obj.type == 'MESH':
                    bpy.ops.object.select_all(action='DESELECT')
                    obj.select_set(True)
                    context.view_layer.objects.active = obj
                    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            
            # Now set parent origin to center
            bpy.ops.object.select_all(action='DESELECT')
            room_parent.select_set(True)
            context.view_layer.objects.active = room_parent
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            
            # Find floor level (minimum Z)
            min_z = float('inf')
            for obj in room_parent.children:
                if obj.type == 'MESH':
                    for vert in obj.data.vertices:
                        world_z = (obj.matrix_world @ vert.co).z
                        min_z = min(min_z, world_z)
            
            # Position so floor is at Z=0
            if min_z != float('inf'):
                room_parent.location = (0, 0, -min_z)
            else:
                room_parent.location = (0, 0, 0)
            
            self.report({'INFO'}, "Room environment loaded")
        
        return {'FINISHED'}

class ENV_OT_apply_lighting(Operator):
    bl_idname = "env.apply_lighting"
    bl_label = "Apply Lighting"
    bl_description = "Setup realistic interior lighting"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene

        # Use Cycles for best quality lighting
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 128
        scene.cycles.use_denoising = True
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.01

        # Enable light bounces for realistic indirect lighting
        scene.cycles.max_bounces = 8
        scene.cycles.diffuse_bounces = 4
        scene.cycles.glossy_bounces = 4
        scene.cycles.transmission_bounces = 8
        scene.cycles.volume_bounces = 0
        scene.cycles.transparent_max_bounces = 8

        # Remove all existing lights
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                bpy.data.objects.remove(obj)

        # Find room bounds to position lights properly
        room_parent = None
        for obj in bpy.data.objects:
            if obj.name == "Room_Environment":
                room_parent = obj
                break

        # Calculate room center and size
        room_center = Vector((0, 0, 2))  # Default
        room_size = Vector((5, 5, 3))    # Default

        if room_parent:
            room_min = Vector((float('inf'),) * 3)
            room_max = Vector((float('-inf'),) * 3)

            for child in room_parent.children:
                if child.type == 'MESH':
                    for vert in child.data.vertices:
                        world_co = child.matrix_world @ vert.co
                        room_min = Vector((min(room_min[i], world_co[i]) for i in range(3)))
                        room_max = Vector((max(room_max[i], world_co[i]) for i in range(3)))

            room_center = (room_min + room_max) / 2
            room_size = room_max - room_min

        # 1. Main ceiling light (bright interior illumination)
        bpy.ops.object.light_add(
            type='AREA',
            location=(room_center.x, room_center.y, room_max.z - 0.5)
        )
        main_light = context.active_object
        main_light.name = "Main_Ceiling_Light"
        main_light.data.energy = 200  # Much brighter for interior
        main_light.data.color = (1, 0.95, 0.85)  # Warm white like indoor lighting
        main_light.data.size = min(room_size.x * 0.4, 2.5)
        main_light.data.shape = 'DISK'
        main_light.rotation_euler = (math.radians(180), 0, 0)  # Point down
        # Soft shadows are automatic with area lights based on size

        # 2. Secondary ceiling lights for even coverage
        ceiling_positions = [
            (room_center.x - room_size.x * 0.25, room_center.y - room_size.y * 0.25, room_max.z - 0.5),
            (room_center.x + room_size.x * 0.25, room_center.y - room_size.y * 0.25, room_max.z - 0.5),
            (room_center.x - room_size.x * 0.25, room_center.y + room_size.y * 0.25, room_max.z - 0.5),
            (room_center.x + room_size.x * 0.25, room_center.y + room_size.y * 0.25, room_max.z - 0.5),
        ]

        for i, pos in enumerate(ceiling_positions):
            bpy.ops.object.light_add(type='POINT', location=pos)
            ceiling_light = context.active_object
            ceiling_light.name = f"Ceiling_Light_{i+1}"
            ceiling_light.data.energy = 500  # Bright point lights
            ceiling_light.data.color = (1, 0.97, 0.9)  # Warm white
            # Point lights in Cycles automatically have soft shadows

        # 3. Wall wash lights (placed inside room, illuminating walls)
        wall_positions = [
            (room_min.x + room_size.x * 0.15, room_center.y, room_center.z),  # Left wall
            (room_max.x - room_size.x * 0.15, room_center.y, room_center.z),  # Right wall
            (room_center.x, room_min.y + room_size.y * 0.15, room_center.z),  # Front wall
            (room_center.x, room_max.y - room_size.y * 0.15, room_center.z),  # Back wall
        ]

        for i, pos in enumerate(wall_positions):
            bpy.ops.object.light_add(type='AREA', location=pos)
            wall_light = context.active_object
            wall_light.name = f"Wall_Wash_{i+1}"
            wall_light.data.energy = 100  # Moderate brightness
            wall_light.data.color = (1, 0.98, 0.95)  # Neutral warm
            wall_light.data.size = min(room_size.z * 0.5, 2)
            wall_light.data.shape = 'RECTANGLE'
            wall_light.data.size_y = min(room_size.z * 0.4, 1.8)

            # Point toward the nearest wall
            if i == 0:  # Left wall
                wall_light.rotation_euler = (0, math.radians(90), 0)
            elif i == 1:  # Right wall
                wall_light.rotation_euler = (0, math.radians(-90), 0)
            elif i == 2:  # Front wall
                wall_light.rotation_euler = (math.radians(90), 0, 0)
            else:  # Back wall
                wall_light.rotation_euler = (math.radians(-90), 0, 0)

        # 4. Ambient fill light (soft overall illumination)
        bpy.ops.object.light_add(
            type='AREA',
            location=(room_center.x, room_center.y, room_center.z)
        )
        ambient_light = context.active_object
        ambient_light.name = "Ambient_Fill"
        ambient_light.data.energy = 50
        ambient_light.data.color = (1, 1, 1)  # Pure white for neutral fill
        ambient_light.data.size = min(max(room_size.x, room_size.y) * 0.8, 4)
        ambient_light.data.shape = 'DISK'

        # 5. Floor uplight for depth (subtle)
        bpy.ops.object.light_add(
            type='AREA',
            location=(room_center.x, room_center.y, room_min.z + 0.2)
        )
        floor_light = context.active_object
        floor_light.name = "Floor_Uplight"
        floor_light.data.energy = 30
        floor_light.data.color = (0.9, 0.85, 0.8)  # Warm tone
        floor_light.data.size = min(room_size.x * 0.3, 1.5)
        floor_light.data.shape = 'DISK'
        floor_light.rotation_euler = (0, 0, 0)  # Point up

        # Setup realistic world environment
        world = bpy.data.worlds.get("World")
        if not world:
            world = bpy.data.worlds.new("World")
        scene.world = world
        world.use_nodes = True

        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()

        # Create HDRI-like gradient environment
        # Texture Coordinate node
        tex_coord = nodes.new('ShaderNodeTexCoord')
        tex_coord.location = (-800, 0)

        # Mapping node for gradient direction
        mapping = nodes.new('ShaderNodeMapping')
        mapping.location = (-600, 0)

        # Gradient texture for sky
        gradient = nodes.new('ShaderNodeTexGradient')
        gradient.location = (-400, 0)
        gradient.gradient_type = 'SPHERICAL'

        # Color ramp for sky colors
        color_ramp = nodes.new('ShaderNodeValToRGB')
        color_ramp.location = (-200, 0)
        color_ramp.color_ramp.elements[0].position = 0.3
        color_ramp.color_ramp.elements[0].color = (0.7, 0.85, 1.0, 1)  # Horizon color
        color_ramp.color_ramp.elements[1].position = 1.0
        color_ramp.color_ramp.elements[1].color = (0.4, 0.6, 0.9, 1)  # Sky blue

        # Mix with ground color
        mix_shader = nodes.new('ShaderNodeMixRGB')
        mix_shader.location = (0, 0)
        mix_shader.blend_type = 'MIX'
        mix_shader.inputs['Color2'].default_value = (0.15, 0.12, 0.1, 1)  # Ground color

        # Separate XYZ to detect ground
        separate_xyz = nodes.new('ShaderNodeSeparateXYZ')
        separate_xyz.location = (-400, -200)

        # Math node to create ground mask
        math_less = nodes.new('ShaderNodeMath')
        math_less.location = (-200, -200)
        math_less.operation = 'LESS_THAN'
        math_less.inputs[1].default_value = 0.0

        # Background shader
        background = nodes.new('ShaderNodeBackground')
        background.location = (200, 0)
        background.inputs['Strength'].default_value = 1.0  # Brighter environment for interior

        # Output
        output = nodes.new('ShaderNodeOutputWorld')
        output.location = (400, 0)

        # Connect nodes for realistic sky
        links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], gradient.inputs['Vector'])
        links.new(gradient.outputs['Fac'], color_ramp.inputs['Fac'])
        links.new(mapping.outputs['Vector'], separate_xyz.inputs['Vector'])
        links.new(separate_xyz.outputs['Z'], math_less.inputs['Value'])
        links.new(math_less.outputs['Value'], mix_shader.inputs['Fac'])
        links.new(color_ramp.outputs['Color'], mix_shader.inputs['Color1'])
        links.new(mix_shader.outputs['Color'], background.inputs['Color'])
        links.new(background.outputs['Background'], output.inputs['Surface'])

        # Add volume scatter for atmospheric effect (subtle haze)
        volume_scatter = nodes.new('ShaderNodeVolumeScatter')
        volume_scatter.location = (200, -200)
        volume_scatter.inputs['Color'].default_value = (1, 0.98, 0.95, 1)
        volume_scatter.inputs['Density'].default_value = 0.001  # Very subtle

        links.new(volume_scatter.outputs['Volume'], output.inputs['Volume'])

        # Set color management for photorealistic look
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0.5
        scene.view_settings.gamma = 1.0

        # Enable ambient occlusion in Cycles
        scene.cycles.use_fast_gi = True
        scene.world.cycles_visibility.scatter = True

        # Set viewport to rendered
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces[0]
                space.shading.type = 'RENDERED'
                space.overlay.show_overlays = False  # Hide overlays for clean view

        self.report({'INFO'}, "Natural interior lighting applied")
        return {'FINISHED'}

class ENV_OT_import_furniture(Operator):
    bl_idname = "env.import_furniture"
    bl_label = "Import Furniture"
    bl_description = "Import furniture model (GLB/GLTF/FBX)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        filepath = scene.env_furniture_path

        # Validate file
        if not filepath:
            self.report({'ERROR'}, "Please select a model file")
            return {'CANCELLED'}

        filepath = bpy.path.abspath(filepath)
        if not os.path.exists(filepath):
            self.report({'ERROR'}, f"File not found: {filepath}")
            return {'CANCELLED'}

        # Get file extension
        ext = os.path.splitext(filepath)[1].lower()

        # Store objects before import
        before_import = set(context.scene.objects)

        # Import based on file type
        try:
            if ext in ['.glb', '.gltf']:
                bpy.ops.import_scene.gltf(filepath=filepath)
            elif ext == '.fbx':
                bpy.ops.import_scene.fbx(filepath=filepath)
            elif ext == '.obj':
                bpy.ops.wm.obj_import(filepath=filepath)
            else:
                self.report({'ERROR'}, f"Unsupported format: {ext}")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Import failed: {str(e)}")
            return {'CANCELLED'}

        # Get newly imported objects
        new_objects = list(set(context.scene.objects) - before_import)

        if not new_objects:
            self.report({'WARNING'}, "No objects imported")
            return {'CANCELLED'}

        # Collect all mesh objects and preserve hierarchy
        all_meshes = []
        all_objects = []
        root_objects = []  # Top-level imported objects
        empties = []  # Track empty objects

        for obj in new_objects:
            all_objects.append(obj)
            if obj.parent is None or obj.parent not in new_objects:
                root_objects.append(obj)
            if obj.type == 'MESH':
                all_meshes.append(obj)
            elif obj.type == 'EMPTY':
                empties.append(obj)
            # Also check children
            for child in obj.children_recursive:
                if child.type == 'MESH' and child not in all_meshes:
                    all_meshes.append(child)
                if child not in all_objects:
                    all_objects.append(child)

        if not all_meshes:
            self.report({'ERROR'}, "No mesh objects found in imported file")
            return {'CANCELLED'}

        # Determine the furniture object based on what was imported
        furniture_object = None

        # Case 1: Single mesh object
        if len(all_meshes) == 1 and len(root_objects) == 1:
            furniture_object = all_meshes[0]
            furniture_object.name = f"Furniture_{os.path.basename(filepath).split('.')[0]}"

        # Case 2: Multiple objects or existing empty parent
        elif len(root_objects) == 1 and root_objects[0].type == 'EMPTY':
            # Use existing empty parent
            furniture_object = root_objects[0]
            furniture_object.name = f"Furniture_{os.path.basename(filepath).split('.')[0]}"

        # Case 3: Multiple root objects - create parent
        else:
            # Create a parent empty to group all furniture parts
            bpy.ops.object.empty_add(location=(0, 0, 0))
            furniture_parent = context.active_object
            furniture_parent.name = f"Furniture_{os.path.basename(filepath).split('.')[0]}"

            # Parent all root objects to the new empty, preserving internal hierarchy
            for obj in root_objects:
                obj.parent = furniture_parent

            furniture_object = furniture_parent

        # Apply transformations to all mesh objects to get correct dimensions
        for mesh_obj in all_meshes:
            bpy.ops.object.select_all(action='DESELECT')
            mesh_obj.select_set(True)
            context.view_layer.objects.active = mesh_obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Calculate real dimensions from all mesh children
        bounds_min = Vector((float('inf'),) * 3)
        bounds_max = Vector((float('-inf'),) * 3)

        for mesh_obj in all_meshes:
            mesh_obj.data.update()
            for vert in mesh_obj.data.vertices:
                world_co = mesh_obj.matrix_world @ vert.co
                for i in range(3):
                    bounds_min[i] = min(bounds_min[i], world_co[i])
                    bounds_max[i] = max(bounds_max[i], world_co[i])

        # Original size calculation
        original_size = bounds_max - bounds_min

        # Determine likely furniture type and auto-scale
        auto_scale = scene.env_furniture_scale

        # Auto-detect furniture type based on proportions
        width = original_size.x
        depth = original_size.y
        height = original_size.z

        # If dimensions seem wrong (e.g., in cm or mm), auto-correct
        if max(width, depth, height) > 10:  # Likely in wrong units
            # Assume it's in cm, convert to meters
            auto_scale = 0.01
            self.report({'INFO'}, "Auto-detected centimeter units, converting to meters")
        elif max(width, depth, height) > 100:  # Likely in mm
            auto_scale = 0.001
            self.report({'INFO'}, "Auto-detected millimeter units, converting to meters")
        elif max(width, depth, height) < 0.1:  # Too small, likely wrong scale
            auto_scale = 10
            self.report({'INFO'}, "Model too small, scaling up")

        # Apply user scale on top of auto-scale
        final_scale = auto_scale * scene.env_furniture_scale
        furniture_object.scale = (final_scale, final_scale, final_scale)

        # Apply scale to the parent empty
        bpy.ops.object.select_all(action='DESELECT')
        furniture_object.select_set(True)
        context.view_layer.objects.active = furniture_object
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # Recalculate bounds after scaling from all mesh children
        bounds_min = Vector((float('inf'),) * 3)
        bounds_max = Vector((float('-inf'),) * 3)

        for mesh_obj in all_meshes:
            mesh_obj.data.update()
            for vert in mesh_obj.data.vertices:
                world_co = mesh_obj.matrix_world @ vert.co
                for i in range(3):
                    bounds_min[i] = min(bounds_min[i], world_co[i])
                    bounds_max[i] = max(bounds_max[i], world_co[i])

        final_size = bounds_max - bounds_min

        # Store sizes as custom properties
        furniture_object["original_size_x"] = original_size.x
        furniture_object["original_size_y"] = original_size.y
        furniture_object["original_size_z"] = original_size.z
        furniture_object["final_size_x"] = final_size.x
        furniture_object["final_size_y"] = final_size.y
        furniture_object["final_size_z"] = final_size.z
        furniture_object["auto_scale_applied"] = auto_scale

        # Find room floor level
        room_parent = None
        for obj in bpy.data.objects:
            if obj.name == "Room_Environment":
                room_parent = obj
                break

        floor_z = 0  # Default floor level
        room_center = Vector((0, 0, 0))

        if room_parent:
            # Find actual floor level from room geometry
            room_min = Vector((float('inf'),) * 3)
            room_max = Vector((float('-inf'),) * 3)

            for child in room_parent.children:
                if child.type == 'MESH':
                    for vert in child.data.vertices:
                        world_co = child.matrix_world @ vert.co
                        room_min = Vector((min(room_min[i], world_co[i]) for i in range(3)))
                        room_max = Vector((max(room_max[i], world_co[i]) for i in range(3)))

            floor_z = room_min.z
            room_center = (room_min + room_max) / 2
            room_center.z = floor_z  # Keep at floor level

        # Position furniture: center in room, sitting on floor
        furniture_object.location = room_center
        # Adjust Z so bottom of furniture sits on floor
        furniture_object.location.z = floor_z - bounds_min.z

        # Clean up only unnecessary empty objects (not part of furniture hierarchy)
        for obj in new_objects:
            if obj.type == 'EMPTY' and obj != furniture_object:
                # Check if this empty has any children or is part of the furniture
                if not obj.children and obj.parent != furniture_object:
                    # Safe to remove if it's not connected to anything
                    bpy.data.objects.remove(obj, do_unlink=True)

        # Report results
        self.report({'INFO'}, f"Imported: {furniture_object.name}")
        self.report({'INFO'}, f"Original size: {original_size.x:.3f}m x {original_size.y:.3f}m x {original_size.z:.3f}m")
        self.report({'INFO'}, f"Final size: {final_size.x:.2f}m W x {final_size.y:.2f}m D x {final_size.z:.2f}m H")
        self.report({'INFO'}, f"Positioned at floor level (Z={floor_z:.2f})")

        # Guess furniture type
        if final_size.z > 0.7 and final_size.z < 0.8:
            self.report({'INFO'}, "➔ Detected: Table/Desk (standard height ~75cm)")
        elif final_size.z > 0.4 and final_size.z < 0.5:
            self.report({'INFO'}, "➔ Detected: Chair/Seating (standard seat height ~45cm)")
        elif final_size.z > 1.8:
            self.report({'INFO'}, "➔ Detected: Wardrobe/Tall furniture")
        elif final_size.x > 1.8 and final_size.y > 1.5:
            self.report({'INFO'}, "➔ Detected: Bed")

        return {'FINISHED'}

class ENV_OT_setup_camera(Operator):
    bl_idname = "env.setup_camera"
    bl_label = "Setup Camera"
    bl_description = "Add camera inside room"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        
        # Remove old cameras
        for obj in bpy.data.objects:
            if obj.type == 'CAMERA':
                bpy.data.objects.remove(obj)
        
        # Find room bounds to place camera inside
        room_parent = None
        for obj in bpy.data.objects:
            if obj.name == "Room_Environment":
                room_parent = obj
                break
        
        if room_parent:
            # Calculate room bounds
            room_min = Vector((float('inf'),) * 3)
            room_max = Vector((float('-inf'),) * 3)
            
            for child in room_parent.children:
                if child.type == 'MESH':
                    for vert in child.data.vertices:
                        world_co = child.matrix_world @ vert.co
                        room_min = Vector((min(room_min[i], world_co[i]) for i in range(3)))
                        room_max = Vector((max(room_max[i], world_co[i]) for i in range(3)))
            
            room_center = (room_min + room_max) / 2
            room_size = room_max - room_min
            
            # Place camera on wall for wide room view
            # Try to position camera near a corner, backed against the wall
            cam_x = room_min.x + room_size.x * 0.05  # 5% from wall (very close)
            cam_y = room_min.y + room_size.y * 0.05  # 5% from wall (very close)
            cam_z = 1.7  # Slightly above eye level
            
            # Create camera
            bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z))
            camera = context.active_object
            camera.name = "Room_Camera"
            scene.camera = camera
            
            # Point camera towards opposite corner for best view
            target = Vector((
                room_max.x - room_size.x * 0.2,
                room_max.y - room_size.y * 0.2,
                0.8  # Look slightly down into room
            ))
            direction = target - camera.location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()
        else:
            # Fallback if no room found
            bpy.ops.object.camera_add(location=(3, -3, 1.6))
            camera = context.active_object
            camera.name = "Room_Camera"
            scene.camera = camera
            camera.rotation_euler = (math.radians(85), 0, math.radians(45))
        
        # Camera settings for interior photography
        camera.data.lens = 18  # Ultra-wide for full room view
        camera.data.clip_start = 0.01  # Very close clipping for tight spaces
        camera.data.clip_end = 100
        camera.data.dof.use_dof = True
        camera.data.dof.focus_distance = 5.0  # Mid-room focus
        camera.data.dof.aperture_fstop = 8  # Good depth of field
        
        self.report({'INFO'}, "Camera added - Use Numpad 0 to view")
        return {'FINISHED'}

class ENV_OT_recenter_origin(Operator):
    bl_idname = "env.recenter_origin"
    bl_label = "Center Origin"
    bl_description = "Set origin to center-bottom of selected object"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        selected = context.selected_objects
        
        if not selected:
            self.report({'ERROR'}, "Please select an object")
            return {'CANCELLED'}
        
        for obj in selected:
            # Store cursor location
            cursor_loc = context.scene.cursor.location.copy()
            
            # Set origin to center
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            
            # Get bounds and move to bottom center
            if obj.type == 'MESH' or (obj.type == 'EMPTY' and obj.bound_box):
                bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
                min_z = min(corner.z for corner in bbox)
                center_x = sum(corner.x for corner in bbox) / 8
                center_y = sum(corner.y for corner in bbox) / 8
                
                # Move origin to bottom center
                context.scene.cursor.location = (center_x, center_y, min_z)
                bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            
            # Restore cursor
            context.scene.cursor.location = cursor_loc
        
        self.report({'INFO'}, "Origin set to center-bottom")
        return {'FINISHED'}


class ENV_OT_adjust_furniture_scale(Operator):
    bl_idname = "env.adjust_furniture_scale"
    bl_label = "Adjust Furniture Scale"
    bl_description = "Scale selected furniture object"
    bl_options = {'REGISTER', 'UNDO'}
    
    scale_factor: bpy.props.FloatProperty(
        name="Scale Factor",
        description="Scale multiplier for selected furniture",
        default=1.0,
        min=0.1,
        max=10.0,
        step=0.1,
        precision=2
    )
    
    uniform_scale: bpy.props.BoolProperty(
        name="Uniform Scale",
        description="Apply uniform scaling to all axes",
        default=True
    )
    
    def execute(self, context):
        selected = context.selected_objects
        
        if not selected:
            self.report({'ERROR'}, "Please select a furniture object")
            return {'CANCELLED'}
        
        for obj in selected:
            # Skip environment objects, but scale everything else (furniture)
            if obj.get("is_environment") or obj.name == "Room_Environment":
                continue
            
            # Apply scale to the object
            if self.uniform_scale:
                obj.scale *= self.scale_factor
            else:
                # Non-uniform scaling could be added here if needed
                obj.scale *= self.scale_factor
            
            # Apply the transformation to make it permanent
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        self.report({'INFO'}, f"Scaled furniture by {self.scale_factor}x")
        return {'FINISHED'}

class ENV_OT_check_size(Operator):
    bl_idname = "env.check_size"
    bl_label = "Check Object Size"
    bl_description = "Display real-world dimensions of selected objects"
    bl_options = {'REGISTER'}

    def execute(self, context):
        selected = context.selected_objects

        if not selected:
            self.report({'ERROR'}, "Please select an object to check size")
            return {'CANCELLED'}

        for obj in selected:
            # Check if object has stored size data (for imported furniture)
            if "original_size_x" in obj and "scaled_size_x" in obj:
                orig_x = obj["original_size_x"]
                orig_y = obj["original_size_y"]
                orig_z = obj["original_size_z"]
                scaled_x = obj["scaled_size_x"]
                scaled_y = obj["scaled_size_y"]
                scaled_z = obj["scaled_size_z"]

                self.report({'INFO'}, f"=== {obj.name} ===")
                self.report({'INFO'}, f"Original: {orig_x:.3f}m W x {orig_y:.3f}m D x {orig_z:.3f}m H")
                self.report({'INFO'}, f"Current: {scaled_x:.3f}m W x {scaled_y:.3f}m D x {scaled_z:.3f}m H")
                self.report({'INFO'}, f"In cm: {scaled_x*100:.1f}cm x {scaled_y*100:.1f}cm x {scaled_z*100:.1f}cm")
                continue

            # For other objects or meshes, calculate from vertices
            all_meshes = []

            if obj.type == 'MESH':
                all_meshes.append(obj)
            elif obj.type == 'EMPTY':
                # Check children for meshes
                for child in obj.children_recursive:
                    if child.type == 'MESH':
                        all_meshes.append(child)

            if not all_meshes:
                self.report({'WARNING'}, f"{obj.name}: No mesh data found")
                continue

            # Calculate bounding box in world space
            bounds_min = Vector((float('inf'),) * 3)
            bounds_max = Vector((float('-inf'),) * 3)

            for mesh_obj in all_meshes:
                # Ensure mesh is up to date
                mesh_obj.data.update()

                # Use vertices for more accurate bounds
                for vert in mesh_obj.data.vertices:
                    world_co = mesh_obj.matrix_world @ vert.co
                    for i in range(3):
                        bounds_min[i] = min(bounds_min[i], world_co[i])
                        bounds_max[i] = max(bounds_max[i], world_co[i])

            # Calculate dimensions
            size = bounds_max - bounds_min

            # Apply any scale from parent hierarchy
            total_scale = Vector((1, 1, 1))
            current_obj = obj
            while current_obj:
                total_scale.x *= current_obj.scale.x
                total_scale.y *= current_obj.scale.y
                total_scale.z *= current_obj.scale.z
                current_obj = current_obj.parent

            # Apply total scale to size
            width = size.x * total_scale.x
            depth = size.y * total_scale.y
            height = size.z * total_scale.z

            # Report dimensions
            self.report({'INFO'}, f"=== {obj.name} ===")
            self.report({'INFO'},
                f"Size: {width:.3f}m W x {depth:.3f}m D x {height:.3f}m H")
            self.report({'INFO'},
                f"In cm: {width*100:.1f}cm x {depth*100:.1f}cm x {height*100:.1f}cm")

            # Common furniture size references
            if height > 0.7 and height < 0.8 and width > 0.5:
                self.report({'INFO'}, "➔ Typical table/desk height")
            elif height > 0.4 and height < 0.5 and width > 0.4:
                self.report({'INFO'}, "➔ Typical chair seat height")
            elif height > 1.8 and height < 2.2:
                self.report({'INFO'}, "➔ Typical wardrobe/tall furniture")
            elif width > 1.8 and width < 2.2 and depth > 1.5:
                self.report({'INFO'}, "➔ Typical bed size")

        return {'FINISHED'}


class ENV_OT_reset_wall_transparency(Operator):
    bl_idname = "env.reset_wall_transparency"
    bl_label = "Reset Wall Transparency"
    bl_description = "Reset all walls to opaque"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Find room environment
        room_parent = None
        for obj in bpy.data.objects:
            if obj.name == "Room_Environment":
                room_parent = obj
                break
        
        if not room_parent:
            self.report({'ERROR'}, "No room environment found")
            return {'CANCELLED'}
        
        # Reset all walls
        for child in room_parent.children:
            if child.type == 'MESH' and (child.get("has_transparency") or child.get("has_dynamic_transparency")):
                for mat in child.data.materials:
                    if mat and mat.use_nodes:
                        nodes = mat.node_tree.nodes
                        links = mat.node_tree.links
                        
                        # Find the original BSDF and output
                        original_bsdf = None
                        output = None
                        
                        for node in nodes:
                            if node.type == 'BSDF_PRINCIPLED' and not node.name.startswith('Dynamic') and not node.name.startswith('Advanced'):
                                original_bsdf = node
                            elif node.type == 'OUTPUT_MATERIAL':
                                output = node
                        
                        # Remove all transparency-related nodes
                        nodes_to_remove = []
                        for node in nodes:
                            if any(prefix in node.name for prefix in ['Transparent_', 'Dynamic_', 'Advanced_', 'Transparency_']):
                                nodes_to_remove.append(node)
                        
                        # Remove the nodes
                        for node in nodes_to_remove:
                            nodes.remove(node)
                        
                        # Reconnect original material to output if both exist
                        if original_bsdf and output:
                            # Clear existing connections to output
                            for link in list(links):
                                if link.to_node == output and link.to_socket.name == 'Surface':
                                    links.remove(link)
                            # Connect original BSDF directly to output
                            links.new(original_bsdf.outputs['BSDF'], output.inputs['Surface'])
                        
                        # Reset material settings
                        mat.blend_method = 'OPAQUE'
                        mat.use_backface_culling = False
                        mat.show_transparent_back = False
                        mat.use_screen_refraction = False
                
                # Clear transparency flags
                if "has_transparency" in child:
                    del child["has_transparency"]
                if "has_dynamic_transparency" in child:
                    del child["has_dynamic_transparency"]
        
        self.report({'INFO'}, "All transparency effects reset")
        return {'FINISHED'}


class ENV_OT_apply_dynamic_transparency(Operator):
    bl_idname = "env.apply_dynamic_transparency"
    bl_label = "Make Walls Transparent"
    bl_description = "Make walls transparent based on camera distance - closer walls become transparent"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Find room environment
        room_parent = None
        for obj in bpy.data.objects:
            if obj.name == "Room_Environment":
                room_parent = obj
                break
        
        if not room_parent:
            self.report({'ERROR'}, "No room environment found")
            return {'CANCELLED'}
        
        # Process each wall mesh
        walls_processed = 0
        for child in room_parent.children:
            if child.type == 'MESH':
                # Apply distance-based transparency to each material
                for mat in child.data.materials:
                    if mat:
                        # Enable nodes if not already
                        if not mat.use_nodes:
                            mat.use_nodes = True
                        
                        nodes = mat.node_tree.nodes
                        links = mat.node_tree.links
                        
                        # Skip if already has transparency
                        if any('Trans_' in node.name for node in nodes):
                            continue
                        
                        # Find existing principled BSDF and output
                        principled = None
                        output = None
                        for node in nodes:
                            if node.type == 'BSDF_PRINCIPLED':
                                principled = node
                            elif node.type == 'OUTPUT_MATERIAL':
                                output = node
                        
                        if not principled or not output:
                            continue
                        
                        # Create distance-based transparency using Alpha channel
                        # 1. Camera Data for distance
                        camera_data = nodes.new('ShaderNodeCameraData')
                        camera_data.location = (-600, -200)
                        camera_data.name = "Trans_CameraData"
                        
                        # 2. Map Range to control distance (0-3 units mapped to 0-1)
                        map_range = nodes.new('ShaderNodeMapRange')
                        map_range.location = (-400, -200)
                        map_range.name = "Trans_MapRange"
                        map_range.inputs['From Min'].default_value = 0.0
                        map_range.inputs['From Max'].default_value = 2.0  # Walls within 2 units become transparent
                        map_range.inputs['To Min'].default_value = 0.0
                        map_range.inputs['To Max'].default_value = 1.0
                        map_range.clamp = True
                        
                        # 3. Invert (closer = less alpha = more transparent)
                        invert = nodes.new('ShaderNodeMath')
                        invert.operation = 'SUBTRACT'
                        invert.location = (-200, -200)
                        invert.name = "Trans_Invert"
                        invert.inputs[0].default_value = 1.0
                        
                        # 4. Multiply to control transparency range
                        multiply = nodes.new('ShaderNodeMath')
                        multiply.operation = 'MULTIPLY'
                        multiply.location = (0, -200)
                        multiply.name = "Trans_Multiply"
                        multiply.inputs[1].default_value = 0.5  # Range of transparency change
                        
                        # 5. Add to ensure minimum opacity
                        add = nodes.new('ShaderNodeMath')
                        add.operation = 'ADD'
                        add.location = (200, -200)
                        add.name = "Trans_Add"
                        add.inputs[1].default_value = 0.5  # Minimum 50% opacity (never too transparent)
                        
                        # Connect the distance-based alpha
                        links.new(camera_data.outputs['View Z Depth'], map_range.inputs['Value'])
                        links.new(map_range.outputs['Result'], invert.inputs[1])
                        links.new(invert.outputs['Value'], multiply.inputs[0])
                        links.new(multiply.outputs['Value'], add.inputs[0])
                        links.new(add.outputs['Value'], principled.inputs['Alpha'])
                        
                        # Configure material for transparency
                        mat.blend_method = 'BLEND'
                        mat.use_backface_culling = False
                        mat.show_transparent_back = True
                        
                        # For Eevee
                        if hasattr(mat, 'use_transparency_overlap'):
                            mat.use_transparency_overlap = True
                        
                        walls_processed += 1
                
                # Mark as having transparency
                child["has_dynamic_transparency"] = True
        
        # Use Eevee for faster viewport preview
        context.scene.render.engine = 'BLENDER_EEVEE_NEXT' if bpy.app.version >= (4, 2, 0) else 'BLENDER_EEVEE'
        
        # Configure Eevee for transparency and fix shadow buffer issues
        eevee = context.scene.eevee
        
        # Optimize shadow settings to prevent buffer overflow
        if hasattr(eevee, 'shadow_cube_size'):
            eevee.shadow_cube_size = '1024'  # Reduce from default to prevent overflow
        if hasattr(eevee, 'shadow_cascade_size'):
            eevee.shadow_cascade_size = '1024'  # Reduce cascade shadows
        if hasattr(eevee, 'use_soft_shadows'):
            eevee.use_soft_shadows = False  # Disable soft shadows to save memory
        
        # Limit light shadows to prevent buffer issues
        for obj in context.scene.objects:
            if obj.type == 'LIGHT':
                # Disable shadows for less important lights
                if 'Fill' in obj.name or 'Lamp' in obj.name:
                    obj.data.use_shadow = False
                # Reduce shadow resolution for remaining lights
                elif hasattr(obj.data, 'shadow_buffer_size'):
                    obj.data.shadow_buffer_size = 1024
        
        if hasattr(eevee, 'use_ssr'):
            eevee.use_ssr = False  # Disable SSR to save performance with transparency
        
        # DO NOT modify world colors - only ensure basic lighting exists
        # The grey/sandy look comes from changing world colors, so we avoid that
        
        # Set viewport shading to Material Preview to see transparency
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces[0]
                space.shading.type = 'MATERIAL'  # Material preview for transparency
                space.shading.use_scene_lights = True  # Use existing lights
                space.shading.use_scene_world = True  # Use existing world
        
        self.report({'INFO'}, f"Applied distance-based transparency to {walls_processed} wall materials")
        self.report({'INFO'}, "Walls become transparent as camera gets closer")
        return {'FINISHED'}


class ENV_OT_prepare_for_spline(Operator):
    bl_idname = "env.prepare_for_spline"
    bl_label = "Prepare for Spline Export"
    bl_description = "Optimize scene for best quality when importing to Spline"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        
        # 1. Ensure all materials have proper PBR setup for Spline
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                for mat_slot in obj.material_slots:
                    if mat_slot.material:
                        mat = mat_slot.material
                        if mat.use_nodes:
                            nodes = mat.node_tree.nodes
                            
                            # Find Principled BSDF
                            principled = None
                            for node in nodes:
                                if node.type == 'BSDF_PRINCIPLED':
                                    principled = node
                                    break
                            
                            if principled:
                                # Optimize for Spline's PBR pipeline
                                # Ensure metallic/roughness are set properly
                                if 'Metallic' in principled.inputs:
                                    # Keep metallic values for proper material response
                                    pass
                                
                                if 'Roughness' in principled.inputs:
                                    # Ensure roughness for realistic lighting
                                    if principled.inputs['Roughness'].default_value == 0:
                                        principled.inputs['Roughness'].default_value = 0.5
                                
                                # Set IOR for better reflections
                                if 'IOR' in principled.inputs:
                                    principled.inputs['IOR'].default_value = 1.45
        
        # 2. Add emission to light sources for Spline
        light_objects = []
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                # Create emissive mesh to represent light in Spline
                if obj.data.type == 'POINT':
                    bpy.ops.mesh.primitive_ico_sphere_add(location=obj.location, subdivisions=2)
                elif obj.data.type == 'AREA':
                    bpy.ops.mesh.primitive_plane_add(location=obj.location)
                else:
                    continue
                
                light_mesh = context.active_object
                light_mesh.name = f"EmissiveLight_{obj.name}"
                light_mesh.scale = (0.2, 0.2, 0.2)
                
                # Create emissive material
                mat = bpy.data.materials.new(name=f"Emissive_{obj.name}")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                nodes.clear()
                
                # Add emission shader
                emission = nodes.new('ShaderNodeEmission')
                emission.inputs['Color'].default_value = (*obj.data.color, 1.0)
                emission.inputs['Strength'].default_value = obj.data.energy
                
                output = nodes.new('ShaderNodeOutputMaterial')
                mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
                
                light_mesh.data.materials.append(mat)
                light_objects.append(light_mesh)
        
        # 3. Bake ambient occlusion for better depth in Spline
        self.report({'INFO'}, f"Prepared scene for Spline export - Added {len(light_objects)} emissive lights")
        self.report({'INFO'}, "Recommendations for Spline:")
        self.report({'INFO'}, "1. Use 'Realistic' or 'Studio' environment in Spline")
        self.report({'INFO'}, "2. Enable 'Global Illumination' in Spline")
        self.report({'INFO'}, "3. Add HDRI: Spline > Environment > HDRI > Choose preset")
        self.report({'INFO'}, "4. Adjust Exposure: Environment > Exposure (try 1.2-1.5)")
        
        return {'FINISHED'}


class ENV_OT_before_after_snapshot(Operator):
    bl_idname = "env.before_after_snapshot"
    bl_label = "Before/After Snapshots"
    bl_description = "Render room with and without furniture"
    bl_options = {'REGISTER', 'UNDO'}
    
    output_path: bpy.props.StringProperty(
        name="Output Path",
        description="Where to save the snapshots",
        default="//snapshots/",
        subtype='DIR_PATH'
    )
    
    def execute(self, context):
        import os
        from datetime import datetime
        
        if not context.scene.camera:
            self.report({'ERROR'}, "Please setup camera first")
            return {'CANCELLED'}
        
        # Create output directory
        output_dir = bpy.path.abspath(self.output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup render settings
        scene = context.scene
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Store furniture visibility states
        furniture_objects = []
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and not obj.get("is_environment"):
                # Assume non-environment meshes are furniture
                if not obj.name.startswith("Room_") and obj.parent != bpy.data.objects.get("Room_Environment"):
                    furniture_objects.append((obj, obj.hide_viewport, obj.hide_render))
        
        # Render WITH furniture (current state)
        with_furniture_path = os.path.join(output_dir, f"with_furniture_{timestamp}.png")
        scene.render.filepath = with_furniture_path
        bpy.ops.render.render(write_still=True)
        self.report({'INFO'}, f"Saved: {with_furniture_path}")
        
        # Hide all furniture
        for obj, _, _ in furniture_objects:
            obj.hide_viewport = True
            obj.hide_render = True
        
        # Render WITHOUT furniture
        without_furniture_path = os.path.join(output_dir, f"without_furniture_{timestamp}.png")
        scene.render.filepath = without_furniture_path
        bpy.ops.render.render(write_still=True)
        self.report({'INFO'}, f"Saved: {without_furniture_path}")
        
        # Restore furniture visibility
        for obj, original_viewport, original_render in furniture_objects:
            obj.hide_viewport = original_viewport
            obj.hide_render = original_render
        
        # Create comparison HTML file
        html_path = os.path.join(output_dir, f"comparison_{timestamp}.html")
        with open(html_path, 'w') as f:
            f.write(f'''<!DOCTYPE html>
<html>
<head>
    <title>Room Comparison - {timestamp}</title>
    <style>
        body {{ font-family: Arial; margin: 20px; background: #f0f0f0; }}
        .container {{ max-width: 1920px; margin: 0 auto; }}
        .slider-container {{ position: relative; overflow: hidden; background: white; }}
        .slider-container img {{ width: 100%; display: block; }}
        .after-image {{ position: absolute; top: 0; left: 0; width: 50%; overflow: hidden; }}
        .after-image img {{ width: 200%; max-width: 200%; }}
        .slider {{ position: absolute; top: 0; bottom: 0; left: 50%; width: 4px; background: white;
                  box-shadow: 0 0 4px rgba(0,0,0,0.5); cursor: ew-resize; }}
        .label {{ position: absolute; top: 20px; background: rgba(0,0,0,0.7); color: white; 
                  padding: 5px 10px; font-size: 14px; }}
        .before-label {{ left: 20px; }}
        .after-label {{ right: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Room Comparison - {timestamp}</h1>
        <div class="slider-container" id="comparison">
            <img src="without_furniture_{timestamp}.png" alt="Before">
            <div class="after-image" id="afterImage">
                <img src="with_furniture_{timestamp}.png" alt="After">
            </div>
            <div class="slider" id="slider"></div>
            <div class="label before-label">Empty Room</div>
            <div class="label after-label">With Furniture</div>
        </div>
    </div>
    <script>
        const slider = document.getElementById('slider');
        const afterImage = document.getElementById('afterImage');
        const comparison = document.getElementById('comparison');
        let isDown = false;
        
        slider.addEventListener('mousedown', () => isDown = true);
        document.addEventListener('mouseup', () => isDown = false);
        document.addEventListener('mousemove', (e) => {{
            if (!isDown) return;
            const rect = comparison.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const percent = (x / rect.width) * 100;
            if (percent >= 0 && percent <= 100) {{
                slider.style.left = percent + '%';
                afterImage.style.width = percent + '%';
            }}
        }});
    </script>
</body>
</html>''')
        
        self.report({'INFO'}, f"Comparison viewer saved: {html_path}")
        return {'FINISHED'}


class ENV_OT_save_template(Operator):
    bl_idname = "env.save_template"
    bl_label = "Save Room Template"
    bl_description = "Save current room setup as a reusable template"
    bl_options = {'REGISTER', 'UNDO'}
    
    template_name: bpy.props.StringProperty(
        name="Template Name",
        description="Name for this template",
        default="Room_Template"
    )
    
    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Path to save template",
        default="//templates/",
        subtype='FILE_PATH'
    )
    
    def execute(self, context):
        import json
        import os
        
        # Create templates directory
        template_dir = bpy.path.abspath(self.filepath)
        if not template_dir.endswith('/'):
            template_dir = os.path.dirname(template_dir) + '/'
        os.makedirs(template_dir, exist_ok=True)
        
        # Collect room data
        template_data = {
            'name': self.template_name,
            'room': {},
            'furniture': [],
            'lights': [],
            'camera': {},
            'settings': {}
        }
        
        # Save room environment data
        room_env = bpy.data.objects.get("Room_Environment")
        if room_env:
            template_data['room'] = {
                'location': list(room_env.location),
                'rotation': list(room_env.rotation_euler),
                'scale': list(room_env.scale)
            }
        
        # Save furniture data
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and not obj.get("is_environment"):
                if obj.name.startswith("Furniture_") or (not obj.name.startswith("Room_") and obj.parent != room_env):
                    furniture_data = {
                        'name': obj.name,
                        'location': list(obj.location),
                        'rotation': list(obj.rotation_euler),
                        'scale': list(obj.scale),
                        'source_file': obj.get('source_file', ''),  # Store original file path if available
                        'materials': []
                    }
                    
                    # Store material properties
                    for mat_slot in obj.material_slots:
                        if mat_slot.material:
                            mat = mat_slot.material
                            mat_data = {
                                'name': mat.name,
                                'base_color': list(mat.diffuse_color) if mat else [1, 1, 1, 1]
                            }
                            
                            # Store principled BSDF settings if available
                            if mat.use_nodes:
                                for node in mat.node_tree.nodes:
                                    if node.type == 'BSDF_PRINCIPLED':
                                        mat_data['metallic'] = node.inputs['Metallic'].default_value
                                        mat_data['roughness'] = node.inputs['Roughness'].default_value
                                        if 'Base Color' in node.inputs:
                                            col = node.inputs['Base Color'].default_value
                                            mat_data['base_color'] = [col[0], col[1], col[2], col[3]]
                                        break
                            
                            furniture_data['materials'].append(mat_data)
                    
                    template_data['furniture'].append(furniture_data)
        
        # Save light data
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                light_data = {
                    'name': obj.name,
                    'type': obj.data.type,
                    'location': list(obj.location),
                    'rotation': list(obj.rotation_euler),
                    'energy': obj.data.energy,
                    'color': list(obj.data.color)
                }
                template_data['lights'].append(light_data)
        
        # Save camera data
        if context.scene.camera:
            cam = context.scene.camera
            template_data['camera'] = {
                'location': list(cam.location),
                'rotation': list(cam.rotation_euler),
                'lens': cam.data.lens if cam.data else 50
            }
        
        # Save scene settings
        template_data['settings'] = {
            'room_scale': context.scene.env_room_scale if hasattr(context.scene, 'env_room_scale') else 1.0,
            'furniture_scale': context.scene.env_furniture_scale if hasattr(context.scene, 'env_furniture_scale') else 1.0
        }
        
        # Save to JSON file
        template_file = os.path.join(template_dir, f"{self.template_name}.json")
        with open(template_file, 'w') as f:
            json.dump(template_data, f, indent=2)
        
        self.report({'INFO'}, f"Template saved: {template_file}")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class ENV_OT_load_template(Operator):
    bl_idname = "env.load_template"
    bl_label = "Load Room Template"
    bl_description = "Load a saved room template"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: bpy.props.StringProperty(
        name="Template File",
        description="Select template to load",
        default="",
        subtype='FILE_PATH'
    )
    
    clear_scene: bpy.props.BoolProperty(
        name="Clear Scene",
        description="Clear existing objects before loading template",
        default=True
    )
    
    def execute(self, context):
        import json
        import os
        
        if not os.path.exists(self.filepath):
            self.report({'ERROR'}, "Template file not found")
            return {'CANCELLED'}
        
        # Load template data
        with open(self.filepath, 'r') as f:
            template_data = json.load(f)
        
        # Clear scene if requested
        if self.clear_scene:
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)
        
        # Recreate room environment
        if template_data.get('room'):
            # Create empty for room parent
            bpy.ops.object.empty_add(location=(0, 0, 0))
            room_parent = context.active_object
            room_parent.name = "Room_Environment"
            room_parent["is_environment"] = True
            
            room = template_data['room']
            room_parent.location = room['location']
            room_parent.rotation_euler = room['rotation']
            room_parent.scale = room['scale']
        
        # Recreate furniture
        # Note: This is a simplified version - in real use, you'd need the actual GLB files
        for furniture in template_data.get('furniture', []):
            # Create placeholder cube for furniture
            bpy.ops.mesh.primitive_cube_add()
            obj = context.active_object
            obj.name = furniture['name']
            obj.location = furniture['location']
            obj.rotation_euler = furniture['rotation']
            obj.scale = furniture['scale']
            
            # Apply materials if available
            for mat_data in furniture.get('materials', []):
                mat = bpy.data.materials.new(name=mat_data['name'])
                mat.use_nodes = True
                
                # Setup principled BSDF
                nodes = mat.node_tree.nodes
                principled = nodes.get("Principled BSDF")
                if principled:
                    if 'base_color' in mat_data:
                        principled.inputs['Base Color'].default_value = mat_data['base_color']
                    if 'metallic' in mat_data:
                        principled.inputs['Metallic'].default_value = mat_data['metallic']
                    if 'roughness' in mat_data:
                        principled.inputs['Roughness'].default_value = mat_data['roughness']
                
                obj.data.materials.append(mat)
            
            self.report({'WARNING'}, f"Created placeholder for {furniture['name']} - Original model file needed for full restoration")
        
        # Recreate lights
        for light_data in template_data.get('lights', []):
            bpy.ops.object.light_add(
                type=light_data['type'],
                location=light_data['location']
            )
            light = context.active_object
            light.name = light_data['name']
            light.rotation_euler = light_data['rotation']
            light.data.energy = light_data['energy']
            light.data.color = light_data['color']
        
        # Recreate camera
        if template_data.get('camera'):
            cam_data = template_data['camera']
            bpy.ops.object.camera_add(location=cam_data['location'])
            camera = context.active_object
            camera.rotation_euler = cam_data['rotation']
            camera.data.lens = cam_data.get('lens', 50)
            context.scene.camera = camera
        
        # Apply settings
        if template_data.get('settings'):
            settings = template_data['settings']
            if hasattr(context.scene, 'env_room_scale'):
                context.scene.env_room_scale = settings.get('room_scale', 1.0)
            if hasattr(context.scene, 'env_furniture_scale'):
                context.scene.env_furniture_scale = settings.get('furniture_scale', 1.0)
        
        self.report({'INFO'}, f"Template loaded: {template_data['name']}")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class ENV_OT_export_baked_glb(Operator):
    bl_idname = "env.export_baked_glb"
    bl_label = "Export Baked GLB"
    bl_description = "Export room as GLB with lighting baked in"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: bpy.props.StringProperty(
        name="File Path",
        default="//room_export.glb",
        subtype='FILE_PATH'
    )
    
    bake_lighting: bpy.props.BoolProperty(
        name="Bake Lighting",
        default=True,
        description="Bake lighting into textures"
    )
    
    texture_size: bpy.props.IntProperty(
        name="Texture Size",
        default=2048,
        min=512,
        max=4096,
        description="Size of baked textures"
    )
    
    def execute(self, context):
        import os
        
        # Store original selection
        original_selection = context.selected_objects.copy()
        original_active = context.view_layer.objects.active
        
        # Select all mesh objects
        bpy.ops.object.select_all(action='DESELECT')
        for obj in context.scene.objects:
            if obj.type == 'MESH':
                obj.select_set(True)
        
        if self.bake_lighting:
            # Create bake material for each object
            for obj in context.selected_objects:
                if obj.type != 'MESH':
                    continue
                    
                # Make active
                context.view_layer.objects.active = obj
                
                # Ensure UV map exists
                if not obj.data.uv_layers:
                    bpy.ops.mesh.uv_texture_add()
                
                # Create new material for baking
                mat_name = f"{obj.name}_baked"
                mat = bpy.data.materials.new(name=mat_name)
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                
                # Clear default nodes
                nodes.clear()
                
                # Add nodes
                bsdf = nodes.new('ShaderNodeBsdfPrincipled')
                output = nodes.new('ShaderNodeOutputMaterial')
                tex_node = nodes.new('ShaderNodeTexImage')
                
                # Create texture
                tex_name = f"{obj.name}_baked_texture"
                tex = bpy.data.images.new(
                    tex_name, 
                    self.texture_size, 
                    self.texture_size
                )
                tex_node.image = tex
                
                # Connect nodes
                mat.node_tree.links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
                mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
                
                # Assign material
                obj.data.materials.clear()
                obj.data.materials.append(mat)
                
                # Select texture node for baking
                nodes.active = tex_node
                
                # Bake
                context.scene.render.engine = 'CYCLES'
                context.scene.cycles.bake_type = 'COMBINED'
                bpy.ops.object.bake(type='COMBINED')
        
        # Export GLB
        filepath = bpy.path.abspath(self.filepath)
        
        bpy.ops.export_scene.gltf(
            filepath=filepath,
            use_selection=True,
            export_format='GLB',
            export_texcoords=True,
            export_normals=True,
            export_materials='EXPORT',
            export_attributes=True,
            export_draco_mesh_compression_enable=False,
            export_apply=True
        )
        
        # Restore selection
        bpy.ops.object.select_all(action='DESELECT')
        for obj in original_selection:
            obj.select_set(True)
        context.view_layer.objects.active = original_active
        
        self.report({'INFO'}, f"Exported to {filepath}")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class ENV_OT_export_furniture(Operator):
    bl_idname = "env.export_furniture"
    bl_label = "Export Selected Furniture"
    bl_description = "Export selected furniture as GLB with proper hierarchy"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(
        name="File Path",
        default="//exported_furniture.glb",
        subtype='FILE_PATH'
    )

    def execute(self, context):
        selected = context.selected_objects

        if not selected:
            self.report({'ERROR'}, "Please select furniture to export")
            return {'CANCELLED'}

        # Store original selection
        original_selection = selected.copy()
        original_active = context.view_layer.objects.active

        # Select the furniture and all its children
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected:
            obj.select_set(True)
            # Also select all children
            for child in obj.children_recursive:
                child.select_set(True)

        # Export with proper settings
        filepath = bpy.path.abspath(self.filepath)

        try:
            bpy.ops.export_scene.gltf(
                filepath=filepath,
                use_selection=True,
                export_format='GLB',
                export_texcoords=True,
                export_normals=True,
                export_materials='EXPORT',
                export_attributes=True,
                export_draco_mesh_compression_enable=False,
                export_apply=False,  # Don't apply modifiers
                export_animations=False,
                export_hierarchy_full_collections=False
            )
            self.report({'INFO'}, f"Exported furniture to: {filepath}")
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {str(e)}")
            return {'CANCELLED'}

        # Restore selection
        bpy.ops.object.select_all(action='DESELECT')
        for obj in original_selection:
            obj.select_set(True)
        context.view_layer.objects.active = original_active

        return {'FINISHED'}

    def invoke(self, context, event):
        # Auto-generate filename based on selection
        if context.selected_objects:
            obj_name = context.selected_objects[0].name.replace("Furniture_", "")
            self.filepath = f"//exported_{obj_name}.glb"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class ENV_OT_render_snapshot(Operator):
    bl_idname = "env.render_snapshot"
    bl_label = "Render Snapshot"
    bl_description = "Render current view"
    bl_options = {'REGISTER'}

    def execute(self, context):
        if not context.scene.camera:
            self.report({'ERROR'}, "Please setup camera first")
            return {'CANCELLED'}

        # Render settings
        context.scene.render.resolution_x = 1920
        context.scene.render.resolution_y = 1080

        # Start render
        bpy.ops.render.render('INVOKE_DEFAULT')

        return {'FINISHED'}


def register():
    bpy.utils.register_class(ENV_OT_load_environment)
    bpy.utils.register_class(ENV_OT_apply_lighting)
    bpy.utils.register_class(ENV_OT_import_furniture)
    bpy.utils.register_class(ENV_OT_setup_camera)
    bpy.utils.register_class(ENV_OT_recenter_origin)
    bpy.utils.register_class(ENV_OT_adjust_furniture_scale)
    bpy.utils.register_class(ENV_OT_check_size)
    bpy.utils.register_class(ENV_OT_reset_wall_transparency)
    bpy.utils.register_class(ENV_OT_apply_dynamic_transparency)
    bpy.utils.register_class(ENV_OT_prepare_for_spline)
    bpy.utils.register_class(ENV_OT_export_baked_glb)
    bpy.utils.register_class(ENV_OT_export_furniture)
    bpy.utils.register_class(ENV_OT_before_after_snapshot)
    bpy.utils.register_class(ENV_OT_save_template)
    bpy.utils.register_class(ENV_OT_load_template)
    bpy.utils.register_class(ENV_OT_render_snapshot)

def unregister():
    bpy.utils.unregister_class(ENV_OT_render_snapshot)
    bpy.utils.unregister_class(ENV_OT_load_template)
    bpy.utils.unregister_class(ENV_OT_save_template)
    bpy.utils.unregister_class(ENV_OT_before_after_snapshot)
    bpy.utils.unregister_class(ENV_OT_export_furniture)
    bpy.utils.unregister_class(ENV_OT_export_baked_glb)
    bpy.utils.unregister_class(ENV_OT_prepare_for_spline)
    bpy.utils.unregister_class(ENV_OT_apply_dynamic_transparency)
    bpy.utils.unregister_class(ENV_OT_reset_wall_transparency)
    bpy.utils.unregister_class(ENV_OT_check_size)
    bpy.utils.unregister_class(ENV_OT_adjust_furniture_scale)
    bpy.utils.unregister_class(ENV_OT_recenter_origin)
    bpy.utils.unregister_class(ENV_OT_setup_camera)
    bpy.utils.unregister_class(ENV_OT_import_furniture)
    bpy.utils.unregister_class(ENV_OT_apply_lighting)
    bpy.utils.unregister_class(ENV_OT_load_environment)
