"""
Simplified UI for Environment Viewer
"""

import bpy
from bpy.types import Panel

class ENV_PT_main(Panel):
    bl_label = "Room Environment"
    bl_idname = "ENV_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Environment"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Step 1: Load Room
        box = layout.box()
        box.label(text="1. Load Room", icon='HOME')
        box.prop(scene, "env_environment_path", text="")
        box.prop(scene, "env_room_scale")
        box.operator("env.load_environment", text="Load Room", icon='IMPORT')
        
        layout.separator()
        
        # Step 2: Lighting
        box = layout.box()
        box.label(text="2. Interior Lighting", icon='LIGHT')
        box.operator("env.apply_lighting", text="Apply Magazine Lighting", icon='LIGHT_SUN')
        
        layout.separator()
        
        # Step 3: Import Furniture
        box = layout.box()
        box.label(text="3. Add Furniture", icon='OBJECT_DATA')
        box.prop(scene, "env_furniture_path", text="")
        box.prop(scene, "env_furniture_scale")
        box.operator("env.import_furniture", text="Import Furniture", icon='IMPORT')
        
        box.separator()
        box.label(text="Adjust Selected:", icon='MODIFIER')
        row = box.row(align=True)
        row.operator("env.adjust_furniture_scale", text="Scale", icon='FULLSCREEN_ENTER')
        row.operator("env.recenter_origin", text="Center Origin", icon='PIVOT_MEDIAN')
        
        box.label(text="Tip: Use G to move, R to rotate", icon='INFO')
        
        layout.separator()
        
        # Step 4: Camera & Render
        box = layout.box()
        box.label(text="4. Camera & Render", icon='CAMERA_DATA')
        row = box.row(align=True)
        row.operator("env.setup_camera", text="Setup Camera", icon='CAMERA_DATA')
        row.operator("env.render_snapshot", text="Render", icon='RENDER_STILL')
        
        box.separator()
        box.label(text="Transparency:", icon='SHADING_RENDERED')
        row = box.row(align=True)
        row.operator("env.apply_dynamic_transparency", text="Make Walls Transparent", icon='MOD_OPACITY')
        row.operator("env.reset_wall_transparency", text="Reset", icon='RECOVER_LAST')
        
        box.separator()
        box.label(text="Comparison:", icon='IMAGE_DATA')
        box.operator("env.before_after_snapshot", text="Before/After Snapshots", icon='RENDER_ANIMATION')
        
        layout.separator()
        
        # Step 5: Templates
        box = layout.box()
        box.label(text="5. Room Templates", icon='FILE_FOLDER')
        row = box.row(align=True)
        row.operator("env.save_template", text="Save Template", icon='FILE_TICK')
        row.operator("env.load_template", text="Load Template", icon='FILE_FOLDER')
        box.label(text="Save/load complete room setups", icon='INFO')

def register():
    bpy.utils.register_class(ENV_PT_main)

def unregister():
    bpy.utils.unregister_class(ENV_PT_main)