"""Render a VRM the way it looks as a VTuber: MToon shading, relaxed pose, bust and full shots.

    blender -b -P scripts/render_vrm.py -- avatar.vrm out_dir [shot,shot,...]

Needs the VRM Add-on for Blender (for the importer and its MToon preview shader), so
don't pass --factory-startup. Writes transparent PNGs named <shot>.png.
"""
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, OUT = os.path.abspath(argv[0]), os.path.abspath(argv[1])
ONLY = set(argv[2].split(",")) if len(argv) > 2 else None
os.makedirs(OUT, exist_ok=True)

ARM_DROP = math.radians(62)   # T-pose arms down to a relaxed stance

bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.import_scene.vrm(filepath=SRC)
armature = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bones = armature.pose.bones


def bone(name):
    for b in bones:
        if b.name.lower().replace("_", "").replace(".", "") == name.lower():
            return b
    raise KeyError(name)


def rotate_world(pb, axis, angle):
    """Rotate a pose bone about a world axis through its head."""
    rest = armature.matrix_world @ pb.bone.matrix_local
    world = Matrix.Rotation(angle, 4, axis) @ rest.to_3x3().to_4x4()
    local = rest.to_3x3().inverted() @ world.to_3x3()
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = local.to_quaternion() @ pb.rotation_quaternion


for side in ("Left", "Right"):
    pb = bone(f"{side}UpperArm")
    x = (armature.matrix_world @ pb.bone.head_local).x
    rotate_world(pb, "Y", ARM_DROP if x > 0 else -ARM_DROP)
    rotate_world(bone(f"{side}LowerArm"), "Z", math.radians(-12 if x > 0 else 12))
bpy.context.view_layer.update()

shape_keys = {}
for m in meshes:
    if m.data.shape_keys:
        for kb in m.data.shape_keys.key_blocks:
            shape_keys.setdefault(kb.name, []).append(kb)
print("shape keys:", sorted(shape_keys))


def set_keys(values):
    for blocks in shape_keys.values():
        for kb in blocks:
            kb.value = 0.0
    for name, value in values.items():
        for key, blocks in shape_keys.items():
            if key.lower() == name.lower():
                for kb in blocks:
                    kb.value = value


scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT" if bpy.app.version < (5, 0, 0) else "BLENDER_EEVEE"  # renamed back in 5.0
scene.render.resolution_x, scene.render.resolution_y = 1600, 1600
scene.render.film_transparent = True
scene.view_settings.view_transform = "Standard"
scene.eevee.taa_render_samples = 32
world = bpy.data.worlds.new("w")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.9, 0.9, 0.95, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.6

sun_data = bpy.data.lights.new("key", "SUN")
sun_data.energy = 2.5
sun = bpy.data.objects.new("key", sun_data)
scene.collection.objects.link(sun)
sun.rotation_euler = (math.radians(55), 0, math.radians(-25))

head = armature.matrix_world @ bone("Head").head
top = max((m.matrix_world @ Vector(c)).z for m in meshes for c in m.bound_box)
bottom = min((m.matrix_world @ Vector(c)).z for m in meshes for c in m.bound_box)

cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam


def aim(location, target, lens):
    cam_data.lens = lens
    cam.location = location
    cam.rotation_euler = (target - location).to_track_quat("-Z", "Y").to_euler()


def shot(name, location, target, lens, keys=None, size=(1600, 1600)):
    if ONLY and name not in ONLY:
        return
    set_keys(keys or {})
    aim(Vector(location), Vector(target), lens)
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.filepath = os.path.join(OUT, f"{name}.png")
    bpy.ops.render.render(write_still=True)


face = head + Vector((0, 0, 0.07))
bust_target = head + Vector((0, 0, -0.02))
front = Vector((0, -1, 0))
three_q = Vector((math.sin(math.radians(-30)), -math.cos(math.radians(30)), 0))

shot("bust_front", bust_target + front * 1.1 + Vector((0, 0, 0.04)), bust_target, 50, size=(1600, 1200))
shot("bust_34", bust_target + three_q * 1.1 + Vector((0, 0, 0.04)), bust_target, 50, size=(1600, 1200))
shot("face_neutral", face + front * 0.55, face, 50)
shot("face_blink", face + front * 0.55, face, 50, {"blink": 1.0, "Blink": 1.0, "blink_left": 1.0, "blink_right": 1.0})
shot("face_halfblink", face + front * 0.55, face, 50, {"blink_left": 0.5, "blink_right": 0.5})
shot("face_aa", face + front * 0.55, face, 50, {"aa": 1.0, "A": 1.0, "a": 1.0})
for vowel in ("ih", "ou", "ee", "oh"):
    shot(f"face_{vowel}", face + front * 0.55, face, 50, {vowel: 1.0})
mid = Vector((head.x, head.y, (top + bottom) / 2))
shot("full_front", mid + front * 4.2, mid, 50, size=(1200, 1800))
shot("full_34", mid + three_q * 4.2, mid, 50, size=(1200, 1800))
print("rendered to", OUT)
