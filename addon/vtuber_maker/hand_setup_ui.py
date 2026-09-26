"""A small rest-pose hand wizard in the VTuber sidebar. No external UI dependencies."""
import bpy
from bpy.props import EnumProperty, FloatProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Vector

from .pipeline import fingers, hand_setup, vrm_export
from .pipeline.rig import make_active

_DRAW_HANDLE = None


def _update_curl(settings, context):
    arm = settings.armature
    if arm and hand_setup.PREVIEW_KEY in arm:
        try:
            hand_setup.preview(arm, settings.side, settings.preview_pose, settings.curl)
            settings.message = "Preview only — Reset before editing, saving or exporting"
        except ValueError as error:
            settings.message = str(error)


def _change_selection(settings, context):
    if settings.armature:
        hand_setup.reset_preview(settings.armature)


class VTUBER_MAKER_PG_hand_setup(bpy.types.PropertyGroup):
    armature: PointerProperty(name="Rig", type=bpy.types.Object, poll=lambda self, obj: obj.type == "ARMATURE")
    mesh: PointerProperty(name="Mesh", type=bpy.types.Object, poll=lambda self, obj: obj.type == "MESH")
    side: EnumProperty(name="Hand", items=[("left", "Left", "Character's left hand"),
                                          ("right", "Right", "Character's right hand")], update=_change_selection)
    finger: EnumProperty(name="Finger", items=[(f, f.title(), "") for f in fingers.FINGERS], update=_change_selection)
    preview_pose: StringProperty(default="fist")
    curl: FloatProperty(name="Curl", default=.7, min=0, max=1, update=_update_curl)
    message: StringProperty(default="Select your rigged mesh. Work in a saved setup copy.")


def _targets(context):
    settings = context.scene.vtuber_hand_setup
    if settings.mesh is None:
        obj = context.object
        if obj and obj.type == "MESH":
            settings.mesh = obj
    if settings.armature is None and settings.mesh:
        settings.armature = settings.mesh.find_armature()
    if not settings.armature or not settings.mesh:
        raise ValueError("Choose the rig and its skinned mesh above")
    if context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    return settings, settings.mesh, settings.armature


def _focus(context, arm, side, finger, point=0, view=""):
    objects = hand_setup.guide_objects(arm, side)
    for obj in hand_setup.guide_objects(arm, "left") + hand_setup.guide_objects(arm, "right"):
        obj.hide_set(obj.get("vtuber_hand_side") != side or obj.get("finger") != finger)
    target = next(o for o in objects if o.get("finger") == finger and o.get("point") == point)
    make_active(target)
    if context.area and context.area.type == "VIEW_3D":
        bpy.ops.wm.tool_set_by_id(name="builtin.move")
        region = context.space_data.region_3d
        data = hand_setup.read_guides(arm, side)[finger]
        points = [arm.matrix_world @ Vector(p) for p in data["points"]]
        region.view_location = sum(points, Vector()) / 4
        region.view_distance = max((points[-1] - points[0]).length * 3, .04)
        if view:
            normal = arm.matrix_world.to_3x3() @ (Vector(data["bend"]) - Vector(data["points"][0]))
            if view == "SIDE":
                normal = (points[1] - points[0]).cross(normal)
            if normal.length > 1e-6:
                region.view_rotation = normal.to_track_quat("Z", "Y")
                region.view_perspective = "ORTHO"


class VTUBER_MAKER_OT_hand_action(bpy.types.Operator):
    """Edit rest-space guides; apply only when their placement and depth are correct"""
    bl_idname = "vtuber_maker.hand_action"
    bl_label = "Hand Setup"
    bl_options = {"REGISTER", "UNDO"}
    action: StringProperty()
    point: bpy.props.IntProperty(default=0)

    def invoke(self, context, event):
        if self.action in ("APPLY", "MIRROR"):
            return context.window_manager.invoke_confirm(self, event)
        return self.execute(context)

    def execute(self, context):
        try:
            settings, mesh, arm = _targets(context)
            side, finger = settings.side, settings.finger
            if self.action == "START":
                hand_setup.create_guides(arm, side)
                _focus(context, arm, side, finger)
                settings.message = "Move Base, Tip, then both joints inside the finger. Check Side view."
            elif self.action in ("SELECT", "PALM", "SIDE"):
                hand_setup.require_rest(arm)
                _focus(context, arm, side, finger, self.point, self.action if self.action != "SELECT" else "")
            elif self.action == "INTERPOLATE":
                hand_setup.require_rest(arm)
                data = hand_setup.read_guides(arm, side)
                row = data[finger]
                base, tip = Vector(row["points"][0]), Vector(row["points"][-1])
                row["points"] = [list(base.lerp(tip, t)) for t in (0, .45, .75, 1)]
                hand_setup.write_guides(arm, side, data)
            elif self.action == "FLIP":
                hand_setup.require_rest(arm)
                data = hand_setup.read_guides(arm, side)
                row = data[finger]
                row["bend"] = list(2 * Vector(row["points"][0]) - Vector(row["bend"]))
                hand_setup.write_guides(arm, side, data)
            elif self.action == "APPLY":
                count = hand_setup.apply(mesh, arm, side, hand_setup.read_guides(arm, side))
                settings.message = f"Rebuilt {side} fingers ({count} weighted vertices). Test the mesh below."
            elif self.action == "MIRROR":
                hand_setup.mirror_guides(arm, side)
                settings.message = "Copied across rig X=0. Switch hands, review, then Apply that hand."
            elif self.action == "RESET":
                hand_setup.reset_preview(arm)
                for obj in hand_setup.guide_objects(arm, side):
                    obj.hide_set(obj.get("finger") != finger)
                settings.message = "Preview restored; guides can be edited again."
            elif self.action in ("FIST", "POINT", "RELAXED", "REST", "FINGER"):
                pose = finger if self.action == "FINGER" else self.action.lower()
                hand_setup.preview(arm, side, pose, settings.curl)
                settings.preview_pose = pose
                for obj in hand_setup.guide_objects(arm, side):
                    obj.hide_set(True)
                make_active(mesh)
                settings.message = "Preview only — Reset before editing, saving or exporting"
            elif self.action == "PAINT":
                hand_setup.require_rest(arm)
                make_active(mesh)
                name = fingers.bone_name(side, finger, fingers._segments(finger)[0])
                if name not in mesh.vertex_groups:
                    raise ValueError("Apply guides before painting")
                mesh.vertex_groups.active_index = mesh.vertex_groups[name].index
                bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
                settings.message = "Advanced: choose segment in Vertex Groups. Apply will replace painted hand weights."
            else:
                raise ValueError("Unknown hand setup action")
        except (ValueError, KeyError, StopIteration) as error:
            self.report({"ERROR"}, str(error) or "Create guides for this hand first")
            return {"CANCELLED"}
        return {"FINISHED"}


class VTUBER_MAKER_OT_export_hand_vrm(bpy.types.Operator, ExportHelper):
    """Export the existing avatar without rerigging it; first restore the preview"""
    bl_idname = "vtuber_maker.export_hand_vrm"
    bl_label = "Export Existing Avatar"
    filename_ext = ".vrm"
    filter_glob: StringProperty(default="*.vrm", options={"HIDDEN"})

    def execute(self, context):
        try:
            _, _, arm = _targets(context)
            hand_setup.require_rest(arm)
            vrm_export.ensure_vrm_addon()
            vrm_export.bind_fingers(arm)
            vrm_export.export(self.filepath, arm)
        except (ValueError, RuntimeError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class VTUBER_MAKER_PT_hand_setup(bpy.types.Panel):
    bl_label = "Guided Hand Setup"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "VTuber"

    def draw(self, context):
        layout = self.layout
        s = context.scene.vtuber_hand_setup
        layout.label(text="Use a saved copy, in rest pose.", icon="INFO")
        layout.prop(s, "mesh")
        layout.prop(s, "armature")
        layout.prop(s, "side", expand=True)
        self.button(layout, "1. Create / Resume Guides", "START")
        layout.prop(s, "finger")
        box = layout.box()
        box.label(text="2. Position inside the finger")
        for label, point in (("Base", 0), ("Tip", 3), ("Joint 1", 1), ("Joint 2", 2), ("Bend toward palm", 4)):
            op = box.operator(VTUBER_MAKER_OT_hand_action.bl_idname, text=label)
            op.action, op.point = "SELECT", point
        row = box.row(align=True)
        self.button(row, "Palm view", "PALM")
        self.button(row, "Side view", "SIDE")
        self.button(box, "Suggest joints between base / tip", "INTERPOLATE")
        self.button(box, "Flip bend direction", "FLIP")
        box.label(text="Drag arrows; use both views for depth.")
        if s.armature:
            base = next((o for o in hand_setup.guide_objects(s.armature, s.side)
                         if o.get("finger") == s.finger and o.get("point") == 0), None)
            if base:
                box.prop(base, '["radius"]', text="Influence radius (rig units)")
        self.button(layout, "3. Apply Guides + Rebuild Weights", "APPLY")
        self.button(layout, "Copy Guides to Other Hand", "MIRROR")
        box = layout.box()
        box.label(text="4. Test actual mesh deformation")
        row = box.row(align=True)
        self.button(row, "Rest", "REST")
        self.button(row, "Relaxed", "RELAXED")
        row = box.row(align=True)
        self.button(row, "Fist", "FIST")
        self.button(row, "Point", "POINT")
        self.button(box, "Only Selected Finger", "FINGER")
        box.prop(s, "curl", slider=True)
        self.button(box, "Reset Preview", "RESET")
        self.button(layout, "Advanced: Paint Finger Weights", "PAINT")
        layout.label(text="Fused fingers may need mesh repair.", icon="ERROR")
        layout.label(text="Save .blend to keep editable guides.")
        layout.operator(VTUBER_MAKER_OT_export_hand_vrm.bl_idname)
        # Split long status messages at word boundaries for narrow sidebars.
        import textwrap
        for line in textwrap.wrap(s.message, width=42):
            layout.label(text=line)

    @staticmethod
    def button(layout, text, action):
        layout.operator(VTUBER_MAKER_OT_hand_action.bl_idname, text=text).action = action


def _draw_guides():
    """Colored centerlines connect the draggable handles; never exported as avatar geometry."""
    import gpu
    from gpu_extras.batch import batch_for_shader
    context = bpy.context
    s = getattr(context.scene, "vtuber_hand_setup", None)
    if not s or not s.armature or hand_setup.PREVIEW_KEY in s.armature:
        return
    arm = s.armature
    objects = hand_setup.guide_objects(arm, s.side)
    visible = {o.get("point"): o for o in objects if o.get("finger") == s.finger and not o.hide_get()}
    if set(visible) != set(range(5)):
        return
    points = [visible[i].matrix_world.translation for i in range(5)]
    lines = [points[i] for edge in ((0, 1), (1, 2), (2, 3), (0, 4)) for i in edge]
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "LINES", {"pos": lines})
    shader.bind()
    shader.uniform_float("color", hand_setup.COLORS[fingers.FINGERS.index(s.finger)])
    batch.draw(shader)


CLASSES = (VTUBER_MAKER_PG_hand_setup, VTUBER_MAKER_OT_hand_action,
           VTUBER_MAKER_OT_export_hand_vrm, VTUBER_MAKER_PT_hand_setup)


def register():
    global _DRAW_HANDLE
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.vtuber_hand_setup = PointerProperty(type=VTUBER_MAKER_PG_hand_setup)
    if not bpy.app.background:
        _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(_draw_guides, (), "WINDOW", "POST_VIEW")


def unregister():
    global _DRAW_HANDLE
    if _DRAW_HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE, "WINDOW")
        _DRAW_HANDLE = None
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            hand_setup.reset_preview(obj)
    del bpy.types.Scene.vtuber_hand_setup
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
