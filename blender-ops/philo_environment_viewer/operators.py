"""
Simplified Environment Viewer Operators
"""

import bpy
import os
import math
from mathutils import Vector
from bpy.types import Operator

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
            
            # Fix rotation - rotate 90 degrees on X axis to stand upright
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
    bl_description = "Setup interior lighting"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        
        # Setup Cycles
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
        scene.cycles.samples = 512
        scene.cycles.use_denoising = True
        
        # Remove old lights
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                bpy.data.objects.remove(obj)
        
        # Find room bounds to place lights inside
        room_parent = None
        for obj in bpy.data.objects:
            if obj.name == "Room_Environment":
                room_parent = obj
                break
        
        if room_parent:
            # Calculate room interior bounds
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
            
            # Calculate safe interior positions (20% margin from walls)
            margin = 0.2
            safe_x_min = room_min.x + room_size.x * margin
            safe_x_max = room_max.x - room_size.x * margin
            safe_y_min = room_min.y + room_size.y * margin
            safe_y_max = room_max.y - room_size.y * margin
            safe_z_max = room_max.z - room_size.z * 0.1  # 10% from ceiling
            
            # Key light - warm, positioned inside room
            key_pos = Vector((
                safe_x_min + room_size.x * 0.3,  # 30% from left wall
                safe_y_min + room_size.y * 0.2,  # 20% from front wall
                safe_z_max - 1.0  # Near ceiling
            ))
            bpy.ops.object.light_add(type='AREA', location=key_pos)
            key = context.active_object
            key.name = "Key_Light"
            key.data.energy = 800
            key.data.size = min(room_size.x, room_size.y) * 0.3  # Scale to room
            key.data.color = (1, 0.95, 0.9)
            key.rotation_euler = (math.radians(45), 0, math.radians(-45))
            
            # Fill light - cooler, opposite corner
            fill_pos = Vector((
                safe_x_max - room_size.x * 0.3,  # 30% from right wall
                safe_y_max - room_size.y * 0.3,  # 30% from back wall
                safe_z_max - 1.5  # Slightly lower than key
            ))
            bpy.ops.object.light_add(type='AREA', location=fill_pos)
            fill = context.active_object
            fill.name = "Fill_Light"
            fill.data.energy = 500
            fill.data.size = min(room_size.x, room_size.y) * 0.4
            fill.data.color = (0.9, 0.95, 1)
            fill.rotation_euler = (math.radians(50), 0, math.radians(135))
            
            # Ceiling bounce - center of room
            ceiling_pos = Vector((
                room_center.x,
                room_center.y,
                safe_z_max - 0.2  # Just below ceiling
            ))
            bpy.ops.object.light_add(type='AREA', location=ceiling_pos)
            ceiling = context.active_object
            ceiling.name = "Ceiling_Light"
            ceiling.data.energy = 400
            ceiling.data.size = min(room_size.x, room_size.y) * 0.6  # Large diffuse light
            ceiling.rotation_euler = (math.radians(180), 0, 0)  # Point down
            
            # Add accent light for depth
            accent_pos = Vector((
                room_center.x,
                safe_y_min + room_size.y * 0.1,  # Near front wall
                1.8  # Table/counter height
            ))
            bpy.ops.object.light_add(type='POINT', location=accent_pos)
            accent = context.active_object
            accent.name = "Accent_Light"
            accent.data.energy = 150
            accent.data.color = (1, 0.9, 0.8)
            accent.data.shadow_soft_size = 0.5
            
        else:
            # Fallback if no room found - use default positions
            bpy.ops.object.light_add(type='AREA', location=(2, -2, 3))
            key = context.active_object
            key.name = "Key_Light"
            key.data.energy = 500
            key.data.size = 2
            key.data.color = (1, 0.95, 0.9)
            
            bpy.ops.object.light_add(type='AREA', location=(-2, 1, 2.5))
            fill = context.active_object
            fill.name = "Fill_Light"
            fill.data.energy = 300
            fill.data.size = 3
            fill.data.color = (0.9, 0.95, 1)
            
            bpy.ops.object.light_add(type='AREA', location=(0, 0, 4))
            ceiling = context.active_object
            ceiling.name = "Ceiling_Light"
            ceiling.data.energy = 200
            ceiling.data.size = 4
            ceiling.rotation_euler = (math.radians(180), 0, 0)
        
        # Setup world with subtle ambient light
        world = bpy.data.worlds.new("Interior_World")
        scene.world = world
        world.use_nodes = True
        
        # Create a more sophisticated world setup
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Add nodes for better ambient lighting
        background = nodes.new('ShaderNodeBackground')
        background.inputs['Color'].default_value = (0.15, 0.15, 0.18, 1)  # Slight blue tint
        background.inputs['Strength'].default_value = 0.3  # Subtle ambient
        
        # Add environment texture node for future HDRI support
        env_tex = nodes.new('ShaderNodeTexEnvironment')
        env_tex.location = (-300, 0)
        
        # Mix shader to blend HDRI and solid color
        mix_shader = nodes.new('ShaderNodeMixShader')
        mix_shader.inputs['Fac'].default_value = 0  # Use solid color by default
        
        # Output
        output = nodes.new('ShaderNodeOutputWorld')
        output.location = (200, 0)
        
        # Connect nodes
        links.new(background.outputs['Background'], mix_shader.inputs[1])
        links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])
        
        # Set viewport shading
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces[0]
                space.shading.type = 'RENDERED'
        
        self.report({'INFO'}, "Lighting applied")
        return {'FINISHED'}

class ENV_OT_import_furniture(Operator):
    bl_idname = "env.import_furniture"
    bl_label = "Import Furniture"
    bl_description = "Import furniture model"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        filepath = scene.env_furniture_path
        
        if not filepath or not os.path.exists(bpy.path.abspath(filepath)):
            self.report({'ERROR'}, "Please select a valid model file")
            return {'CANCELLED'}
        
        filepath = bpy.path.abspath(filepath)
        
        # Store current objects
        before = set(context.scene.objects)
        
        # Import model
        if filepath.lower().endswith(('.glb', '.gltf')):
            bpy.ops.import_scene.gltf(filepath=filepath)
        elif filepath.lower().endswith('.fbx'):
            bpy.ops.import_scene.fbx(filepath=filepath)
        elif filepath.lower().endswith('.obj'):
            bpy.ops.wm.obj_import(filepath=filepath)
        
        # Get new objects
        new_objects = list(set(context.scene.objects) - before)
        
        if new_objects:
            # Create furniture parent
            bpy.ops.object.empty_add(location=(0, 0, 0))
            furniture = context.active_object
            furniture.name = f"Furniture_{os.path.basename(filepath).split('.')[0]}"
            
            # Parent all parts
            for obj in new_objects:
                obj.parent = furniture
            
            # Apply scale
            furniture.scale = (scene.env_furniture_scale,) * 3
            
            # Apply scale to get correct bounds
            bpy.ops.object.select_all(action='DESELECT')
            furniture.select_set(True)
            context.view_layer.objects.active = furniture
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            
            # Set origin to center for all furniture parts
            for obj in new_objects:
                if obj.type == 'MESH':
                    bpy.ops.object.select_all(action='DESELECT')
                    obj.select_set(True)
                    context.view_layer.objects.active = obj
                    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            
            # Set furniture parent origin to center bottom
            bpy.ops.object.select_all(action='DESELECT')
            furniture.select_set(True)
            context.view_layer.objects.active = furniture
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            
            # Get the current bounds to adjust origin to bottom
            bbox = [furniture.matrix_world @ Vector(corner) for corner in furniture.bound_box]
            min_z = min(corner.z for corner in bbox)
            center_x = sum(corner.x for corner in bbox) / 8
            center_y = sum(corner.y for corner in bbox) / 8
            
            # Move origin to bottom center
            cursor_loc = context.scene.cursor.location.copy()
            context.scene.cursor.location = (center_x, center_y, min_z)
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            context.scene.cursor.location = cursor_loc
            
            # Now furniture bottom is at origin Z
            min_z = 0
            
            # Get room bounds to place furniture inside
            room_parent = None
            for obj in bpy.data.objects:
                if obj.name == "Room_Environment":
                    room_parent = obj
                    break
            
            if room_parent:
                # Calculate room center and safe placement area
                room_min = Vector((float('inf'),) * 3)
                room_max = Vector((float('-inf'),) * 3)
                
                for child in room_parent.children:
                    if child.type == 'MESH':
                        for vert in child.data.vertices:
                            world_co = child.matrix_world @ vert.co
                            room_min = Vector((min(room_min[i], world_co[i]) for i in range(3)))
                            room_max = Vector((max(room_max[i], world_co[i]) for i in range(3)))
                
                # Place furniture in center of room, on floor
                room_center = (room_min + room_max) / 2
                furniture.location.x = room_center.x
                furniture.location.y = room_center.y
                furniture.location.z = 0  # Origin is already at bottom
            else:
                # Fallback if no room found
                furniture.location.z = 0
            
            # Select for easy positioning
            bpy.ops.object.select_all(action='DESELECT')
            furniture.select_set(True)
            context.view_layer.objects.active = furniture
            
            self.report({'INFO'}, f"Imported {furniture.name} - Use G to move")
        
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
            
            # Place camera inside room, offset from corner
            cam_x = room_center.x + room_size.x * 0.3  # 30% from center
            cam_y = room_center.y - room_size.y * 0.3  # 30% from center
            cam_z = 1.6  # Eye level
            
            # Create camera
            bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z))
            camera = context.active_object
            camera.name = "Room_Camera"
            scene.camera = camera
            
            # Point camera towards room center
            direction = room_center - camera.location
            direction.z = 0  # Keep looking horizontally
            rot_quat = direction.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()
        else:
            # Fallback if no room found
            bpy.ops.object.camera_add(location=(3, -3, 1.6))
            camera = context.active_object
            camera.name = "Room_Camera"
            scene.camera = camera
            camera.rotation_euler = (math.radians(85), 0, math.radians(45))
        
        # Camera settings
        camera.data.lens = 24  # Wide angle for interior
        camera.data.clip_end = 100
        
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
    bpy.utils.register_class(ENV_OT_render_snapshot)

def unregister():
    bpy.utils.unregister_class(ENV_OT_render_snapshot)
    bpy.utils.unregister_class(ENV_OT_recenter_origin)
    bpy.utils.unregister_class(ENV_OT_setup_camera)
    bpy.utils.unregister_class(ENV_OT_import_furniture)
    bpy.utils.unregister_class(ENV_OT_apply_lighting)
    bpy.utils.unregister_class(ENV_OT_load_environment)