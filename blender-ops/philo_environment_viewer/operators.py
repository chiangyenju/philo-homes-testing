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
            key.data.energy = 400
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
            fill.data.energy = 200
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
            ceiling.data.energy = 100
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
            lamp1.data.energy = 30
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
            lamp2.data.energy = 40
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
            # Create a collection for this furniture
            furniture_name = f"Furniture_{os.path.basename(filepath).split('.')[0]}"
            furniture_collection = bpy.data.collections.new(name=furniture_name)
            context.scene.collection.children.link(furniture_collection)

            # Find the bounding box of all imported objects BEFORE moving them
            # This preserves their relative positions
            combined_min = [float('inf')] * 3
            combined_max = [float('-inf')] * 3

            for obj in new_objects:
                if obj.type == 'MESH':
                    # Get world matrix for this object
                    world_matrix = obj.matrix_world.copy()
                    # Get bounding box in world space
                    for corner in obj.bound_box:
                        world_co = world_matrix @ Vector(corner)
                        for i in range(3):
                            combined_min[i] = min(combined_min[i], world_co[i])
                            combined_max[i] = max(combined_max[i], world_co[i])

            # Calculate center of the combined bounding box
            if combined_min[0] != float('inf'):
                center_x = (combined_min[0] + combined_max[0]) / 2
                center_y = (combined_min[1] + combined_max[1]) / 2
                center_z = combined_min[2]  # Use bottom of bounding box
            else:
                center_x = center_y = center_z = 0

            # Create furniture parent empty at the center-bottom of all objects
            bpy.ops.object.empty_add(location=(center_x, center_y, center_z))
            furniture = context.active_object
            furniture.name = furniture_name
            furniture.empty_display_type = 'CUBE'
            furniture.empty_display_size = 0.5

            # Store original transforms and parent all objects WITHOUT moving them
            for obj in new_objects:
                # Store the world matrix before parenting
                original_matrix = obj.matrix_world.copy()

                # Remove from current collections
                for coll in list(obj.users_collection):
                    coll.objects.unlink(obj)

                # Add to furniture collection
                furniture_collection.objects.link(obj)

                # Parent to the empty while keeping transform
                obj.parent = furniture
                # Restore the world transform to maintain position
                obj.matrix_world = original_matrix

            # Move the parent empty to the collection
            for coll in list(furniture.users_collection):
                coll.objects.unlink(furniture)
            furniture_collection.objects.link(furniture)
            
            # Apply scale to the parent (this will scale all children)
            furniture.scale = (scene.env_furniture_scale,) * 3
            
            # Recalculate bounds after scaling to find the actual bottom
            furniture_min_z = float('inf')

            for obj in new_objects:
                if obj.type == 'MESH':
                    # Update mesh to ensure correct vertex positions
                    obj.data.update()
                    # Get world coordinates of all vertices
                    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
                    for corner in bbox:
                        furniture_min_z = min(furniture_min_z, corner.z)
            
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
                furniture_bottom_offset = furniture_min_z - furniture.location.z

                # Place furniture so its actual bottom sits on the floor surface
                furniture.location.x = room_center_x
                furniture.location.y = room_center_y
                # Adjust Z position so the actual bottom sits on floor
                furniture.location.z = room_floor_z - furniture_bottom_offset
            else:
                # Fallback if no room found - place at origin with slight elevation
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
    
    def invoke(self, context, event):
        # Show dialog with scale factor input
        return context.window_manager.invoke_props_dialog(self)


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
    bpy.utils.register_class(ENV_OT_reset_wall_transparency)
    bpy.utils.register_class(ENV_OT_apply_dynamic_transparency)
    bpy.utils.register_class(ENV_OT_prepare_for_spline)
    bpy.utils.register_class(ENV_OT_export_baked_glb)
    bpy.utils.register_class(ENV_OT_before_after_snapshot)
    bpy.utils.register_class(ENV_OT_save_template)
    bpy.utils.register_class(ENV_OT_load_template)
    bpy.utils.register_class(ENV_OT_render_snapshot)

def unregister():
    bpy.utils.unregister_class(ENV_OT_render_snapshot)
    bpy.utils.unregister_class(ENV_OT_load_template)
    bpy.utils.unregister_class(ENV_OT_save_template)
    bpy.utils.unregister_class(ENV_OT_before_after_snapshot)
    bpy.utils.unregister_class(ENV_OT_export_baked_glb)
    bpy.utils.unregister_class(ENV_OT_prepare_for_spline)
    bpy.utils.unregister_class(ENV_OT_apply_dynamic_transparency)
    bpy.utils.unregister_class(ENV_OT_reset_wall_transparency)
    bpy.utils.unregister_class(ENV_OT_adjust_furniture_scale)
    bpy.utils.unregister_class(ENV_OT_recenter_origin)
    bpy.utils.unregister_class(ENV_OT_setup_camera)
    bpy.utils.unregister_class(ENV_OT_import_furniture)
    bpy.utils.unregister_class(ENV_OT_apply_lighting)
    bpy.utils.unregister_class(ENV_OT_load_environment)
