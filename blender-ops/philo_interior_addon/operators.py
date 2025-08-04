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
        
        # Room dimensions
        room_size = 8
        wall_height = 3
        
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
        principled = mat.node_tree.nodes["Principled BSDF"]
        
        if material_type == 'WOOD':
            principled.inputs['Base Color'].default_value = (0.3, 0.25, 0.2, 1)
            principled.inputs['Roughness'].default_value = 0.3
        elif material_type == 'MARBLE':
            principled.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1)
            principled.inputs['Roughness'].default_value = 0.1
            principled.inputs['Specular IOR Level'].default_value = 0.8
        elif material_type == 'CONCRETE':
            principled.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1)
            principled.inputs['Roughness'].default_value = 0.8
        elif material_type == 'CARPET':
            principled.inputs['Base Color'].default_value = (0.4, 0.3, 0.3, 1)
            principled.inputs['Roughness'].default_value = 0.9
            principled.inputs['Sheen Weight'].default_value = 0.5
        
        return mat
    
    def _create_wall_material(self, material_type):
        mat = bpy.data.materials.new(name=f"Wall_{material_type}")
        mat.use_nodes = True
        principled = mat.node_tree.nodes["Principled BSDF"]
        
        if material_type == 'PAINT':
            principled.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1)
            principled.inputs['Roughness'].default_value = 0.8
        elif material_type == 'WALLPAPER':
            principled.inputs['Base Color'].default_value = (0.85, 0.85, 0.8, 1)
            principled.inputs['Roughness'].default_value = 0.7
        elif material_type == 'BRICK':
            principled.inputs['Base Color'].default_value = (0.6, 0.3, 0.2, 1)
            principled.inputs['Roughness'].default_value = 0.9
        elif material_type == 'PLASTER':
            principled.inputs['Base Color'].default_value = (0.95, 0.95, 0.9, 1)
            principled.inputs['Roughness'].default_value = 0.85
        
        return mat

class PHILO_OT_setup_lighting(Operator):
    bl_idname = "philo.setup_lighting"
    bl_label = "Setup Lighting"
    bl_description = "Setup realistic lighting and effects"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        
        # Setup world environment
        world = bpy.data.worlds.new("Interior_World")
        scene.world = world
        world.use_nodes = True
        
        # Clear default nodes
        world.node_tree.nodes.clear()
        
        # Add environment nodes
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        
        bg_node = nodes.new(type='ShaderNodeBackground')
        output_node = nodes.new(type='ShaderNodeOutputWorld')
        
        # Set environment color and strength
        bg_node.inputs['Color'].default_value = (0.1, 0.1, 0.15, 1)  # Slight blue tint
        bg_node.inputs['Strength'].default_value = scene.philo_hdri_strength
        
        links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
        
        # Remove existing lights
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                bpy.data.objects.remove(obj)
        
        # Key light (main window light)
        bpy.ops.object.light_add(type='AREA', location=(-3, 3, 2.5))
        key_light = context.active_object
        key_light.name = "Key_Light"
        key_light.data.size = 2
        key_light.data.energy = scene.philo_key_light_strength
        key_light.data.color = (1.0, 0.95, 0.85)  # Warm daylight
        key_light.rotation_euler = (math.radians(45), math.radians(-45), 0)
        key_light.data.use_shadow = scene.philo_use_shadows
        
        # Fill light (ceiling bounce)
        bpy.ops.object.light_add(type='AREA', location=(0, 0, 2.8))
        fill_light = context.active_object
        fill_light.name = "Fill_Light"
        fill_light.data.size = 4
        fill_light.data.energy = scene.philo_fill_light_strength
        fill_light.data.color = (1.0, 1.0, 1.0)
        fill_light.rotation_euler = (math.radians(180), 0, 0)
        fill_light.data.use_shadow = scene.philo_use_shadows
        
        # Setup render settings
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
        
        # Apply quality settings
        quality_settings = {
            'PREVIEW': {'samples': 64, 'denoising': True},
            'MEDIUM': {'samples': 256, 'denoising': True},
            'HIGH': {'samples': 1024, 'denoising': True}
        }
        
        settings = quality_settings[scene.philo_render_quality]
        scene.cycles.samples = settings['samples']
        scene.cycles.use_denoising = settings['denoising']
        
        # Enhanced light paths for realism
        scene.cycles.max_bounces = 12
        scene.cycles.diffuse_bounces = 4
        scene.cycles.glossy_bounces = 4
        scene.cycles.transmission_bounces = 12
        scene.cycles.transparent_max_bounces = 8
        
        # Color management
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.exposure = scene.philo_exposure
        
        contrast_map = {
            'LOW': 'Low Contrast',
            'MEDIUM': 'Medium Contrast',
            'HIGH': 'High Contrast'
        }
        scene.view_settings.look = contrast_map[scene.philo_contrast]
        
        # Bloom effect (compositor)
        if scene.philo_use_bloom:
            scene.use_nodes = True
            tree = scene.node_tree
            tree.nodes.clear()
            
            # Add nodes
            render_layers = tree.nodes.new('CompositorNodeRLayers')
            glare = tree.nodes.new('CompositorNodeGlare')
            composite = tree.nodes.new('CompositorNodeComposite')
            
            # Configure bloom
            glare.glare_type = 'GHOSTS'
            glare.quality = 'HIGH'
            glare.threshold = 1.0  # Higher threshold = less bloom on darker areas
            glare.mix = scene.philo_bloom_intensity
            glare.size = 7  # Smaller size for tighter bloom
            
            # Connect nodes
            tree.links.new(render_layers.outputs['Image'], glare.inputs['Image'])
            tree.links.new(glare.outputs['Image'], composite.inputs['Image'])
        
        self.report({'INFO'}, "Lighting setup complete")
        return {'FINISHED'}

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
        min_z = min((obj.matrix_world @ v.co).z for v in obj.data.vertices)
        obj.location.z -= min_z
    elif obj.type == 'EMPTY' and obj.children:
        # For groups, find lowest point among all children
        min_z = float('inf')
        for child in obj.children:
            if child.type == 'MESH':
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
        
        # Import and arrange models
        imported_count = 0
        spacing = 3  # Space between models
        cols = int(math.sqrt(len(model_files)))
        
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
                    
                    # Position on floor
                    position_on_floor(main_obj)
                    
                    if scene.philo_use_collision:
                        # Try to find valid position
                        if not find_valid_position(main_obj, margin=scene.philo_collision_margin):
                            # Fall back to grid position
                            row = i // cols
                            col = i % cols
                            main_obj.location.x = (col - cols/2) * spacing
                            main_obj.location.y = (row - len(model_files)//cols/2) * spacing
                    else:
                        # Use grid arrangement
                        row = i // cols
                        col = i % cols
                        main_obj.location.x = (col - cols/2) * spacing
                        main_obj.location.y = (row - len(model_files)//cols/2) * spacing
                    
                    # Update bounds and add to placed furniture
                    bpy.context.view_layer.update()
                    bounds = get_object_bounds(main_obj)
                    placed_furniture.append(bounds)
                    
                    # Apply smooth shading
                    for obj in new_objects:
                        if obj.type == 'MESH':
                            obj.select_set(True)
                            context.view_layer.objects.active = obj
                            bpy.ops.object.shade_smooth()
                    
                    imported_count += 1
                    
            except Exception as e:
                self.report({'WARNING'}, f"Failed to import: {os.path.basename(filepath)} - {str(e)}")
        
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
        # Remove existing cameras
        for obj in bpy.data.objects:
            if obj.type == 'CAMERA':
                bpy.data.objects.remove(obj)
        
        # Create new camera
        cam_data = bpy.data.cameras.new("Main_Camera")
        cam_obj = bpy.data.objects.new("Main_Camera", cam_data)
        context.scene.collection.objects.link(cam_obj)
        context.scene.camera = cam_obj
        
        # Position camera further away and higher
        cam_obj.location = (0, 8, 2.5)  # Centered, further back, higher up
        
        # Point into the room with slight downward angle
        # X rotation: slight downward angle (around 80 degrees instead of 90)
        # Z rotation: 180 degrees to fix orientation
        cam_obj.rotation_euler = (math.radians(80), 0, math.radians(180))
        
        # Camera settings for interior view
        cam_data.lens = 24  # Wider angle to capture more of the room from distance
        cam_data.sensor_width = 36
        cam_data.dof.use_dof = True
        cam_data.dof.aperture_fstop = 5.6
        cam_data.dof.focus_distance = 8  # Focus on middle of room
        
        self.report({'INFO'}, "Camera setup complete - elevated view with slight downward angle")
        return {'FINISHED'}

class PHILO_OT_render_snapshot(Operator):
    bl_idname = "philo.render_snapshot"
    bl_label = "Render Snapshot"
    bl_description = "Take a snapshot from camera view"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Ensure camera exists
        if not context.scene.camera:
            bpy.ops.philo.setup_camera()
        
        # Start render
        bpy.ops.render.render('INVOKE_DEFAULT')
        
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