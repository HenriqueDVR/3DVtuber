"""OC2VRM: turn an AI-generated character mesh into a VRM avatar."""

import bpy
from bpy.props import FloatProperty, StringProperty
from bpy_extras.io_utils import ExportHelper

from .pipeline import build, face, landmarks, vrm_export


class VTUBER_MAKER_OT_make_vrm(bpy.types.Operator, ExportHelper):
    """Rig the selected character, add blink, mouth shapes and hair physics, and save a VRM"""

    bl_idname = "vtuber_maker.make_vrm"
    bl_label = "Make VRM Avatar"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".vrm"
    filter_glob: StringProperty(default="*.vrm", options={"HIDDEN"})
    avatar_name: StringProperty(name="Avatar Name", default="My VTuber")
    author: StringProperty(name="Author", default="")
    height: FloatProperty(
        name="Height", description="Avatar height in metres", default=1.6, min=0.5, max=3.0, unit="LENGTH"
    )

    @classmethod
    def poll(cls, context):
        return any(obj.type == "MESH" for obj in context.view_layer.objects.selected)

    def execute(self, context):
        meshes = [obj for obj in context.view_layer.objects.selected if obj.type == "MESH"]
        try:
            report = build.build_avatar(
                meshes, self.filepath, name=self.avatar_name, author=self.author or "Unknown", height=self.height
            )
        except (vrm_export.VrmAddonMissing, landmarks.LandmarkError, face.FaceNotFound, ValueError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        summary = f"Saved {bpy.path.basename(self.filepath)} with {report['hair_chains']} hair chains"
        if report["mouth_estimated"]:
            summary += "; no painted mouth found, so its position was estimated"
        self.report({"INFO"}, summary)
        return {"FINISHED"}


class VTUBER_MAKER_PT_panel(bpy.types.Panel):
    bl_label = "OC2VRM"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "VTuber"

    def draw(self, context):
        steps = self.layout.column(align=True)
        steps.label(text="1. Import your character model")
        steps.label(text="2. Select it in the viewport")
        self.layout.operator(VTUBER_MAKER_OT_make_vrm.bl_idname, icon="ARMATURE_DATA")


CLASSES = (VTUBER_MAKER_OT_make_vrm, VTUBER_MAKER_PT_panel)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    from . import hand_setup_ui
    hand_setup_ui.register()


def unregister():
    from . import hand_setup_ui
    hand_setup_ui.unregister()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
