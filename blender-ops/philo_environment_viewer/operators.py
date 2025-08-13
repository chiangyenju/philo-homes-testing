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
        
        # Setup Cycles with realistic settings
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
        scene.cycles.samples = 512
        scene.cycles.use_denoising = True
        scene.cycles.caustics_reflective = True
        scene.cycles.caustics_refractive = True
        scene.cycles.max_bounces = 12
        
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
            
            # Key light - warm sunlight from window
            key_pos = Vector((
                safe_x_min + room_size.x * 0.3,  # 30% from left wall
                safe_y_min + room_size.y * 0.2,  # 20% from front wall
                safe_z_max - 0.5  # Very close to ceiling
            ))
            bpy.ops.object.light_add(type='AREA', location=key_pos)
            key = context.active_object
            key.name = "Key_Light"
            key.data.energy = 1200
            key.data.size = min(room_size.x, room_size.y) * 0.25
            key.data.color = (1, 0.95, 0.85)  # Warm daylight
            key.rotation_euler = (math.radians(65), 0, math.radians(-30))
            
            # Fill light - soft sky light
            fill_pos = Vector((
                safe_x_max - room_size.x * 0.25,
                safe_y_max - room_size.y * 0.25,
                safe_z_max - 0.8
            ))
            bpy.ops.object.light_add(type='AREA', location=fill_pos)
            fill = context.active_object
            fill.name = "Fill_Light"
            fill.data.energy = 600
            fill.data.size = min(room_size.x, room_size.y) * 0.5  # Larger for softer shadows
            fill.data.color = (0.85, 0.9, 1)  # Cool sky light
            fill.rotation_euler = (math.radians(70), 0, math.radians(120))
            
            # Ceiling bounce light for ambient fill
            ceiling_pos = Vector((
                room_center.x,
                room_center.y,
                safe_z_max - 0.1  # Very close to ceiling
            ))
            bpy.ops.object.light_add(type='AREA', location=ceiling_pos)
            ceiling = context.active_object
            ceiling.name = "Ceiling_Bounce"
            ceiling.data.energy = 300
            ceiling.data.size = min(room_size.x, room_size.y) * 0.7  # Very large for soft light
            ceiling.data.color = (0.98, 0.98, 1)  # Neutral white
            ceiling.rotation_euler = (math.radians(180), 0, 0)  # Point down
            
            # Add practical lights (lamp simulation)
            lamp_pos1 = Vector((
                safe_x_min + room_size.x * 0.15,
                safe_y_min + room_size.y * 0.15,
                1.2  # Table lamp height
            ))
            bpy.ops.object.light_add(type='POINT', location=lamp_pos1)
            lamp1 = context.active_object
            lamp1.name = "Table_Lamp_1"
            lamp1.data.energy = 100
            lamp1.data.color = (1, 0.9, 0.75)  # Warm incandescent
            lamp1.data.shadow_soft_size = 0.3
            
            # Second practical light
            lamp_pos2 = Vector((
                safe_x_max - room_size.x * 0.15,
                safe_y_max - room_size.y * 0.15,
                2.2  # Floor lamp height
            ))
            bpy.ops.object.light_add(type='POINT', location=lamp_pos2)
            lamp2 = context.active_object
            lamp2.name = "Floor_Lamp"
            lamp2.data.energy = 120
            lamp2.data.color = (1, 0.95, 0.9)
            lamp2.data.shadow_soft_size = 0.4
            
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
        
        # Add nodes for realistic ambient lighting
        background = nodes.new('ShaderNodeBackground')
        background.inputs['Color'].default_value = (0.05, 0.05, 0.06, 1)  # Very subtle ambient
        background.inputs['Strength'].default_value = 0.1  # Minimal ambient for realism
        
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
            
            # Find the actual bottom of the furniture model by checking all vertices
            furniture_min_z = float('inf')
            furniture_max_z = float('-inf')
            furniture_min_x = float('inf')
            furniture_max_x = float('-inf')
            furniture_min_y = float('inf')
            furniture_max_y = float('-inf')
            
            for obj in new_objects:
                if obj.type == 'MESH':
                    # Update mesh to ensure correct vertex positions
                    obj.data.update()
                    # Get world coordinates of all vertices
                    for vert in obj.data.vertices:
                        world_co = obj.matrix_world @ vert.co
                        furniture_min_z = min(furniture_min_z, world_co.z)
                        furniture_max_z = max(furniture_max_z, world_co.z)
                        furniture_min_x = min(furniture_min_x, world_co.x)
                        furniture_max_x = max(furniture_max_x, world_co.x)
                        furniture_min_y = min(furniture_min_y, world_co.y)
                        furniture_max_y = max(furniture_max_y, world_co.y)
            
            # Calculate furniture center
            furniture_center_x = (furniture_min_x + furniture_max_x) / 2
            furniture_center_y = (furniture_min_y + furniture_max_y) / 2
            
            # Set furniture parent origin to bottom center
            bpy.ops.object.select_all(action='DESELECT')
            furniture.select_set(True)
            context.view_layer.objects.active = furniture
            
            # Move origin to the actual bottom center of the furniture
            cursor_loc = context.scene.cursor.location.copy()
            context.scene.cursor.location = (furniture_center_x, furniture_center_y, furniture_min_z)
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            context.scene.cursor.location = cursor_loc
            
            # After changing origin, recalculate the actual bottom position
            # The furniture's location has changed after setting origin
            furniture_actual_min_z = float('inf')
            for obj in new_objects:
                if obj.type == 'MESH':
                    obj.data.update()
                    for vert in obj.data.vertices:
                        world_co = obj.matrix_world @ vert.co
                        furniture_actual_min_z = min(furniture_actual_min_z, world_co.z)
            
            # Get room bounds to place furniture inside
            room_parent = None
            for obj in bpy.data.objects:
                if obj.name == "Room_Environment":
                    room_parent = obj
                    break
            
            if room_parent:
                # Find room bounds including the actual floor level
                room_min_x = float('inf')
                room_max_x = float('-inf')
                room_min_y = float('inf')
                room_max_y = float('-inf')
                room_min_z = float('inf')
                room_max_z = float('-inf')
                
                # Find all room vertices to get bounds
                all_z_values = []
                floor_vertices = []  # Store potential floor vertices
                
                for child in room_parent.children:
                    if child.type == 'MESH':
                        child.data.update()  # Ensure mesh is updated
                        for vert in child.data.vertices:
                            world_co = child.matrix_world @ vert.co
                            room_min_x = min(room_min_x, world_co.x)
                            room_max_x = max(room_max_x, world_co.x)
                            room_min_y = min(room_min_y, world_co.y)
                            room_max_y = max(room_max_y, world_co.y)
                            room_min_z = min(room_min_z, world_co.z)
                            room_max_z = max(room_max_z, world_co.z)
                            all_z_values.append(world_co.z)
                            
                            # Store vertices that might be part of the floor
                            if world_co.z < room_min_z + 0.5:  # Within 0.5 units of minimum
                                floor_vertices.append(world_co.z)
                
                # Detect floor surface more accurately
                room_height = room_max_z - room_min_z
                
                # Method 1: Find the most common Z value in the bottom region (likely the floor)
                if floor_vertices:
                    # Round values to handle minor variations
                    rounded_floor_z = [round(z, 2) for z in floor_vertices]
                    # Find most common Z value (the actual floor surface)
                    from collections import Counter
                    z_counts = Counter(rounded_floor_z)
                    most_common_z = z_counts.most_common(1)[0][0]
                    room_floor_z = most_common_z
                else:
                    # Method 2: Use percentile approach
                    all_z_values.sort()
                    # Find floor surface: vertices in the bottom 10% of room height
                    floor_threshold = room_min_z + (room_height * 0.1)
                    floor_surface_z_values = [z for z in all_z_values if z <= floor_threshold]
                    
                    if floor_surface_z_values:
                        # Get the top of the floor (highest point in floor region)
                        room_floor_z = max(floor_surface_z_values)
                    else:
                        # Fallback to absolute minimum if no floor detected
                        room_floor_z = room_min_z
                
                room_center_x = (room_min_x + room_max_x) / 2
                room_center_y = (room_min_y + room_max_y) / 2
                
                # Calculate the offset needed to place furniture on floor
                # The furniture's actual bottom might be below its origin after the origin change
                furniture_bottom_offset = furniture_actual_min_z - furniture.location.z
                
                # Debug information
                print(f"Room bounds: Z min={room_min_z:.3f}, Z max={room_max_z:.3f}")
                print(f"Floor detected at Z={room_floor_z:.3f}")
                print(f"Furniture actual bottom at Z={furniture_actual_min_z:.3f}")
                print(f"Furniture origin at Z={furniture.location.z:.3f}")
                print(f"Bottom offset: {furniture_bottom_offset:.3f}")
                
                # Place furniture so its actual bottom sits on the floor surface
                furniture.location.x = room_center_x
                furniture.location.y = room_center_y
                # Adjust Z position so the actual bottom (not origin) sits on floor
                furniture.location.z = room_floor_z - furniture_bottom_offset
            else:
                # Fallback if no room found - place at origin
                furniture.location = (0, 0, 0.001)
            
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


class ENV_OT_dynamic_transparency(Operator):
    bl_idname = "env.dynamic_transparency"
    bl_label = "Enable Dynamic Transparency"
    bl_description = "Automatically make walls transparent based on viewing angle - preserves textures"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        
        # Find room environment
        room_parent = None
        for obj in bpy.data.objects:
            if obj.name == "Room_Environment":
                room_parent = obj
                break
        
        if not room_parent:
            self.report({'ERROR'}, "No room environment found")
            return {'CANCELLED'}
        
        # Apply dynamic transparency to all room meshes
        for child in room_parent.children:
            if child.type == 'MESH':
                # Ensure object has a material
                if not child.data.materials:
                    mat = bpy.data.materials.new(name=f"Dynamic_{child.name}")
                    child.data.materials.append(mat)
                else:
                    mat = child.data.materials[0]
                
                # Setup dynamic transparency with shader nodes
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                
                # Store original shader setup if exists
                original_bsdf = None
                original_output = None
                original_links = []
                
                # Find existing Principled BSDF and preserve it
                for node in nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        original_bsdf = node
                    elif node.type == 'OUTPUT_MATERIAL':
                        original_output = node
                
                # If we have existing material setup, preserve it
                if original_bsdf:
                    # Store all links to the BSDF
                    for link in links:
                        if link.to_node == original_bsdf:
                            original_links.append({
                                'from_socket': link.from_socket,
                                'to_socket': link.to_socket
                            })
                else:
                    # Create new Principled BSDF
                    original_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
                    original_bsdf.location = (0, 0)
                
                # Create or get output node
                if not original_output:
                    original_output = nodes.new('ShaderNodeOutputMaterial')
                    original_output.location = (500, 0)
                
                # Add transparency nodes without clearing existing ones
                
                # Check if dynamic transparency already exists
                if "Dynamic_Mix" in nodes:
                    # Already has dynamic setup, skip
                    continue
                
                # Geometry node - provides backfacing info
                geometry = nodes.new('ShaderNodeNewGeometry')
                geometry.location = (-600, -400)
                geometry.name = "Dynamic_Geometry"
                
                # Color Ramp to control transparency gradient
                ramp = nodes.new('ShaderNodeValToRGB')
                ramp.name = "Transparency_Ramp"
                ramp.location = (-400, -400)
                # Adjust ramp for smooth transition
                ramp.color_ramp.elements[0].position = 0.3  # Start fading
                ramp.color_ramp.elements[1].position = 0.7  # Fully transparent
                ramp.color_ramp.elements[0].color = (1, 1, 1, 1)  # White = opaque
                ramp.color_ramp.elements[1].color = (0, 0, 0, 1)  # Black = transparent
                
                # Math node to invert backfacing (1 - backfacing)
                invert = nodes.new('ShaderNodeMath')
                invert.operation = 'SUBTRACT'
                invert.location = (-200, -400)
                invert.inputs[0].default_value = 1.0
                invert.name = "Dynamic_Invert"
                
                # Transparent shader
                transparent = nodes.new('ShaderNodeBsdfTransparent')
                transparent.location = (0, -200)
                transparent.name = "Dynamic_Transparent"
                
                # Mix shader controlled by facing ratio
                mix = nodes.new('ShaderNodeMixShader')
                mix.name = "Dynamic_Mix"
                mix.location = (300, 0)
                
                # Disconnect existing connection to output
                for link in list(links):
                    if link.to_node == original_output and link.to_socket.name == 'Surface':
                        links.remove(link)
                
                # Connect nodes for dynamic transparency
                # Backfacing output gives 1 for faces pointing away from camera
                links.new(geometry.outputs['Backfacing'], ramp.inputs['Fac'])
                links.new(ramp.outputs['Color'], invert.inputs[1])
                links.new(invert.outputs['Value'], mix.inputs['Fac'])
                
                # Connect original material to mix
                links.new(original_bsdf.outputs['BSDF'], mix.inputs[1])
                links.new(transparent.outputs['BSDF'], mix.inputs[2])
                links.new(mix.outputs['Shader'], original_output.inputs['Surface'])
                
                # Set material properties
                mat.blend_method = 'BLEND'
                mat.show_transparent_back = False
                mat.use_backface_culling = False  # Show both sides
                
                # Mark as having dynamic transparency
                child["has_dynamic_transparency"] = True
        
        self.report({'INFO'}, "Dynamic view-based transparency enabled")
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
    bpy.utils.register_class(ENV_OT_dynamic_transparency)
    bpy.utils.register_class(ENV_OT_reset_wall_transparency)
    bpy.utils.register_class(ENV_OT_render_snapshot)

def unregister():
    bpy.utils.unregister_class(ENV_OT_render_snapshot)
    bpy.utils.unregister_class(ENV_OT_reset_wall_transparency)
    bpy.utils.unregister_class(ENV_OT_dynamic_transparency)
    bpy.utils.unregister_class(ENV_OT_recenter_origin)
    bpy.utils.unregister_class(ENV_OT_setup_camera)
    bpy.utils.unregister_class(ENV_OT_import_furniture)
    bpy.utils.unregister_class(ENV_OT_apply_lighting)
    bpy.utils.unregister_class(ENV_OT_load_environment)