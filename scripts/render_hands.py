"""Solid, textured renders of one hand in app-like poses, from cameras fixed on the rest pose.

blender -b baked.blend --python-exit-code 1 -P scripts/render_hands.py -- out/stem left

Writes <stem>_<pose>_<view>.png. Judge hands from these, not x-ray renders. Read-only.
Poses (degrees per joint, fingers / thumb): open 0/0, relaxed 15/8, half 35/15, fist 70/25.
Views: back of the hand, palm, and from the thumb side. Camera is set once, on the rest pose.
"""
import math
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
out, side = argv[0], argv[1].capitalize()
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
mesh = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")
names = [b.name for b in arm.pose.bones if b.name.startswith(side) and any(f in b.name for f in FINGERS)]
POSES = (("open", 0, 0), ("relaxed", 15, 8), ("half", 35, 15), ("fist", 70, 25))

hand = arm.pose.bones[side + "Hand"]
wrist = arm.matrix_world @ hand.head
tips = [arm.matrix_world @ arm.pose.bones[n].tail for n in names if "Distal" in n]
centre = (wrist + sum(tips, Vector()) / len(tips)) / 2
along = (sum(tips, Vector()) / len(tips) - wrist).normalized()
# palm normal from the knuckle row
idx = arm.pose.bones[side + "IndexProximal"].head
lit = arm.pose.bones[side + "LittleProximal"].head
across = (idx - lit)
across = (across - along * across.dot(along)).normalized()
normal = along.cross(across).normalized() * (1 if side == "Left" else -1)  # toward the palm
extent = max((t - wrist).length for t in tips) * 1.5

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
sh = scene.display.shading
sh.light, sh.color_type, sh.show_xray, sh.show_cavity = "STUDIO", "TEXTURE", False, True
sh.show_backface_culling = False
scene.render.resolution_x = scene.render.resolution_y = 420
cam_data = bpy.data.cameras.new("c")
cam_data.type, cam_data.ortho_scale = "ORTHO", extent
cam = bpy.data.objects.new("c", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
# hide the rest of the body far from the hand: clip to a slab around it
cam_data.clip_start, cam_data.clip_end = .5 - extent * .6, .5 + extent * .6

views = {"back": -normal, "palm": normal, "thumb": across}
tiles = []
for pose, fdeg, tdeg in POSES:
    for n in names:
        pb = arm.pose.bones[n]
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (math.radians(tdeg if "Thumb" in n else fdeg), 0, 0)
    bpy.context.view_layer.update()
    for label, d in views.items():
        z = d.normalized()
        y = (along - z * along.dot(z)).normalized()
        cam.matrix_world = Matrix.Translation(centre + z * .5) @ Matrix((y.cross(z), y, z)).transposed().to_4x4()
        path = f"{out}_{pose}_{label}.png"
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        tiles.append(path)
print("TILES", ";".join(tiles))

