"""Measure each hand against the VRM T-pose, in world axes. Read-only.

blender -b avatar.blend --python-exit-code 1 -P scripts/check_hand_tpose.py

Tracking apps (UniVRM, so Warudo) apply finger tracking assuming the rest hand is the spec's
(VRMC_vrm-1.0/tpose.md 1.6-1.8): hand straight out along the arm and level, palm facing the
ground, the four fingers along the arm, the thumb level and 45 degrees between straight out and
straight forward. Finger bones can be perfectly consistent with the hand's own frame and still
come out wrong in the app when the whole hand droops or rolls. Degrees, per side:
  droop       - hand direction below/above horizontal
  yaw         - hand direction toward the front/back of the character
  palm_roll   - palm normal away from straight down
  finger_*    - each finger's proximal direction away from the arm direction
  thumb_*     - thumb direction: elevation from level, and angle out from the arm (spec 45)
Blender frame here: the character faces -Y, up is +Z, its left is +X.
"""
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from vtuber_maker.pipeline import fingers
from vtuber_maker.pipeline.rig import BONE_NAMES


def deg(v):
    return round(math.degrees(v), 1)


arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
world = arm.matrix_world
for side, sign in (("left", 1), ("right", -1)):
    bones = arm.data.bones
    hand = bones.get(BONE_NAMES[f"{side}_hand"])
    if hand is None:
        continue
    out = Vector((sign, 0, 0))
    forward = Vector((0, -1, 0))
    wrist = world @ hand.head_local
    heads = [world @ bones[fingers.bone_name(side, f, "proximal")].head_local
             for f in ("index", "middle", "ring", "little") if fingers.bone_name(side, f, "proximal") in bones]
    knuckles = sum(heads, Vector()) / len(heads)
    along = (knuckles - wrist).normalized()
    index = world @ bones[fingers.bone_name(side, "index", "proximal")].head_local
    little = world @ bones[fingers.bone_name(side, "little", "proximal")].head_local
    across = (index - little).normalized()  # toward the thumb side
    palm = (along.cross(across) * sign).normalized()  # toward the palm
    row = {"droop": deg(math.asin(max(-1, min(1, -along.z)))),
           "yaw": deg(math.atan2(along.dot(forward), along.dot(out))),
           "palm_roll": deg(palm.angle(Vector((0, 0, -1))))}
    for f in ("index", "middle", "ring", "little"):
        b = bones.get(fingers.bone_name(side, f, "proximal"))
        if b:
            d = (world.to_3x3() @ (b.tail_local - b.head_local)).normalized()
            row[f"finger_{f}"] = deg(d.angle(out))
    t = bones.get(fingers.bone_name(side, "thumb", "proximal"))
    if t:
        d = (world.to_3x3() @ (t.tail_local - t.head_local)).normalized()
        row["thumb_elevation"] = deg(math.asin(max(-1, min(1, d.z))))
        flat = Vector((d.x, d.y, 0)).normalized()
        row["thumb_out_from_arm"] = deg(math.atan2(flat.dot(forward), flat.dot(out)))
    print(side.upper(), "  ".join(f"{k} {v}" for k, v in row.items()))
