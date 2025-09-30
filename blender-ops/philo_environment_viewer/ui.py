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
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Compact layout
        col = layout.column(align=True)

        # Step 1: Load Room
        box = col.box()
        box.label(text="1. Load Room", icon='HOME')
        box.prop(scene, "env_environment_path", text="")
        box.prop(scene, "env_room_scale")
        box.operator("env.load_environment", text="Load Room", icon='IMPORT')

        # Step 2: Lighting
        box = col.box()
        box.label(text="2. Interior Lighting", icon='LIGHT')
        box.operator("env.apply_lighting", text="Apply Magazine Lighting", icon='LIGHT_SUN')

        # Step 3: Import Furniture
        box = col.box()
        box.label(text="3. Add Furniture", icon='OBJECT_DATA')
        box.prop(scene, "env_furniture_path", text="")
        box.prop(scene, "env_furniture_scale")
        box.operator("env.import_furniture", text="Import Furniture", icon='IMPORT')

        box.separator()
        box.label(text="Adjust Selected:", icon='MODIFIER')
        row = box.row(align=True)
        row.operator("env.adjust_furniture_scale", text="Scale", icon='FULLSCREEN_ENTER')
        row.operator("env.recenter_origin", text="Center Origin", icon='PIVOT_MEDIAN')
        row = box.row(align=True)
        row.operator("env.check_size", text="Check Size", icon='EMPTY_ARROWS')
        row.operator("env.export_furniture", text="Export", icon='EXPORT')

class ENV_PT_tools(Panel):
    bl_label = "Tools"
    bl_idname = "ENV_PT_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Environment"
    bl_parent_id = "ENV_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout

        # Camera & Render
        box = layout.box()
        box.label(text="Camera & Render", icon='CAMERA_DATA')
        row = box.row(align=True)
        row.operator("env.setup_camera", text="Camera", icon='CAMERA_DATA')
        row.operator("env.render_snapshot", text="Render", icon='RENDER_STILL')

        # Transparency
        box = layout.box()
        box.label(text="Transparency", icon='SHADING_RENDERED')
        row = box.row(align=True)
        row.operator("env.apply_dynamic_transparency", text="Walls Transparent", icon='MOD_OPACITY')
        row.operator("env.reset_wall_transparency", text="Reset", icon='RECOVER_LAST')

        # Templates
        box = layout.box()
        box.label(text="Templates", icon='FILE_FOLDER')
        row = box.row(align=True)
        row.operator("env.save_template", text="Save", icon='FILE_TICK')
        row.operator("env.load_template", text="Load", icon='FILE_FOLDER')

        # Extras
        box = layout.box()
        box.operator("env.before_after_snapshot", text="Before/After", icon='RENDER_ANIMATION')

def register():
    bpy.utils.register_class(ENV_PT_main)
    bpy.utils.register_class(ENV_PT_tools)

def unregister():
    bpy.utils.unregister_class(ENV_PT_tools)
    bpy.utils.unregister_class(ENV_PT_main)