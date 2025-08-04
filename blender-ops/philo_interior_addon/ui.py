import bpy
from bpy.types import Panel

class PHILO_PT_main_panel(Panel):
    bl_label = "Philo Interior Generator"
    bl_idname = "PHILO_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Philo"
    
    def draw(self, context):
        layout = self.layout
        
        # Room generation
        box = layout.box()
        box.label(text="1. Create Room", icon='HOME')
        box.operator("philo.generate_room", icon='MESH_CUBE')

class PHILO_PT_room_materials_panel(Panel):
    bl_label = "Room Materials"
    bl_idname = "PHILO_PT_room_materials_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Philo"
    bl_parent_id = "PHILO_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        col = layout.column(align=True)
        col.label(text="Floor Type:")
        col.prop(scene, "philo_floor_material", text="")
        
        col.separator()
        
        col.label(text="Wall Type:")
        col.prop(scene, "philo_wall_material", text="")
        
        layout.label(text="Apply materials before generating room", icon='INFO')

class PHILO_PT_lighting_panel(Panel):
    bl_label = "2. Lighting & Effects"
    bl_idname = "PHILO_PT_lighting_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Philo"
    bl_parent_id = "PHILO_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Lighting preset
        box = layout.box()
        box.label(text="Lighting Style:", icon='LIGHT')
        box.prop(scene, "philo_lighting_preset", text="")
        
        # Preset descriptions
        info_box = box.box()
        info_box.scale_y = 0.8
        if scene.philo_lighting_preset == 'NATURAL':
            info_box.label(text="Warm sunlight through windows")
            info_box.label(text="Perfect for residential interiors")
        elif scene.philo_lighting_preset == 'STUDIO':
            info_box.label(text="Clean, even lighting")
            info_box.label(text="Ideal for product showcase")
        else:  # DRAMATIC
            info_box.label(text="High contrast with accents")
            info_box.label(text="Luxury furniture presentation")
        
        # Setup button
        layout.operator("philo.setup_lighting", icon='LIGHT', text="Apply Lighting")

class PHILO_PT_import_panel(Panel):
    bl_label = "3. Import Models"
    bl_idname = "PHILO_PT_import_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Philo"
    bl_parent_id = "PHILO_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Collision settings
        box = layout.box()
        box.label(text="Placement Settings:", icon='PHYSICS')
        box.prop(scene, "philo_use_collision")
        if scene.philo_use_collision:
            box.prop(scene, "philo_collision_margin")
        
        # Single model import
        box = layout.box()
        box.label(text="Single Model:", icon='FILE_3D')
        box.prop(scene, "philo_model_path", text="")
        box.operator("philo.import_model", icon='IMPORT')
        
        # Folder import
        box = layout.box()
        box.label(text="Import Folder:", icon='FILE_FOLDER')
        box.prop(scene, "philo_folder_path", text="")
        box.operator("philo.import_folder", icon='IMPORT')
        
        layout.label(text="Supports: GLB, GLTF, FBX, OBJ", icon='INFO')

class PHILO_PT_model_tools_panel(Panel):
    bl_label = "Model Tools"
    bl_idname = "PHILO_PT_model_tools_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Philo"
    bl_parent_id = "PHILO_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Scale controls
        box = layout.box()
        box.label(text="Scale Selected Model:", icon='OBJECT_DATAMODE')
        box.prop(scene, "philo_model_scale")
        box.operator("philo.scale_model", icon='TRANSFORM_ORIGINS')
        
        layout.label(text="Select model(s) first", icon='INFO')

class PHILO_PT_camera_panel(Panel):
    bl_label = "4. Camera & Render"
    bl_idname = "PHILO_PT_camera_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Philo"
    bl_parent_id = "PHILO_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Camera setup
        col = layout.column(align=True)
        col.operator("philo.setup_camera", icon='CAMERA_DATA')
        col.label(text="Elevated view with slight angle", icon='INFO')
        
        col.separator()
        
        # Render quality
        col.label(text="Render Quality:")
        col.prop(scene, "philo_render_quality", text="")
        
        # Quality info
        box = col.box()
        box.scale_y = 0.8
        if scene.philo_render_quality == 'PREVIEW':
            box.label(text="64 samples - Fast preview")
        elif scene.philo_render_quality == 'MEDIUM':
            box.label(text="256 samples - Balanced")
        elif scene.philo_render_quality == 'HIGH':
            box.label(text="1024 samples - Production")
        
        col.separator()
        
        # Render button
        col.operator("philo.render_snapshot", text="Render Snapshot", icon='RENDER_STILL')

class PHILO_PT_physics_panel(Panel):
    bl_label = "Physics Tools"
    bl_idname = "PHILO_PT_physics_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Philo"
    bl_parent_id = "PHILO_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        # Physics info
        box = layout.box()
        box.label(text="Collision Physics:", icon='PHYSICS')
        box.label(text="1. Select furniture objects")
        box.label(text="2. Click to add collision")
        box.label(text="3. Use for physics simulations")
        
        layout.operator("philo.add_collision", icon='RIGID_BODY')

def register():
    bpy.utils.register_class(PHILO_PT_main_panel)
    bpy.utils.register_class(PHILO_PT_room_materials_panel)
    bpy.utils.register_class(PHILO_PT_lighting_panel)
    bpy.utils.register_class(PHILO_PT_import_panel)
    bpy.utils.register_class(PHILO_PT_model_tools_panel)
    bpy.utils.register_class(PHILO_PT_camera_panel)
    bpy.utils.register_class(PHILO_PT_physics_panel)

def unregister():
    bpy.utils.unregister_class(PHILO_PT_physics_panel)
    bpy.utils.unregister_class(PHILO_PT_camera_panel)
    bpy.utils.unregister_class(PHILO_PT_model_tools_panel)
    bpy.utils.unregister_class(PHILO_PT_import_panel)
    bpy.utils.unregister_class(PHILO_PT_lighting_panel)
    bpy.utils.unregister_class(PHILO_PT_room_materials_panel)
    bpy.utils.unregister_class(PHILO_PT_main_panel)