"""Add chest jiggle (two spring bones) to a rigged avatar .blend and export it as VRM.

    blender -b in.blend -P scripts/add_chest_jiggle.py -- out.blend out.vrm [--strength 0.8]

Works on a .blend saved by make_vrm.py --blend, including one edited by hand: the
original file is left alone. Needs the VRM Add-on for Blender.
"""
import argparse
import sys

import bpy
import numpy as np
from mathutils import Vector

ARM_BONES = {f"{s}{b}" for s in ("Left", "Right") for b in ("UpperArm", "LowerArm", "Hand", "Shoulder")}
# Search for each breast's front-most point this far either side of the centre line (m).
SIDE_RANGE = (0.035, 0.13)
# The bone starts this far behind that point and its tip bone sticks out this far in front.
BONE_DEPTH = 0.07
TIP_LENGTH = 0.03
# Weighted region: an ellipsoid round the front point, full weight in its inner FULL share.
RADIUS = Vector((0.065, 0.06, 0.07))
FULL = 0.35
# Spring settings (VRM 1.0 units).
STIFFNESS = 0.35
DRAG = 0.15
GRAVITY = 0.05


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_blend")
    ap.add_argument("out_vrm")
    ap.add_argument("--strength", type=float, default=0.8, help="max share of weight the jiggle bones take (0..1)")
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:])


def main():
    args = parse_args()
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = next(o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm)
    groups = {g.index: g.name for g in mesh.vertex_groups}
    mw = mesh.matrix_world
    verts = np.array([mw @ v.co for v in mesh.data.vertices])
    bones = arm.data.bones
    aw = arm.matrix_world
    chest_z = (aw @ bones["Chest"].head_local).z
    neck_z = (aw @ bones["Neck"].head_local).z
    centre_x = (aw @ bones["Hips"].head_local).x

    def excluded(v):
        return any(
            (groups[g.group] in ARM_BONES or groups[g.group].startswith("Hair") or groups[g.group] == "Head")
            and g.weight > 0.1
            for g in v.groups
        )

    blocked = np.array([excluded(v) for v in mesh.data.vertices])
    band = (verts[:, 2] > chest_z) & (verts[:, 2] < neck_z) & ~blocked

    apexes = {}
    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        off = sign * (verts[:, 0] - centre_x)
        pick = band & (off > SIDE_RANGE[0]) & (off < SIDE_RANGE[1])
        if not pick.any():
            sys.exit(f"no chest surface found on the {side.lower()} side")
        front = np.nonzero(pick)[0][np.argmin(verts[pick, 1])]
        apexes[side] = Vector(verts[front])
        print(f"{side} front point {tuple(round(c, 3) for c in apexes[side])}")

    bpy.context.view_layer.objects.active = arm
    for o in bpy.context.selected_objects:
        o.select_set(False)
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    ebones = arm.data.edit_bones
    inverse = aw.inverted()
    for side, apex in apexes.items():
        for name in (f"{side}Breast", f"{side}Breast_end"):
            if name in ebones:
                ebones.remove(ebones[name])
        bone = ebones.new(f"{side}Breast")
        bone.head = inverse @ (apex + Vector((0.0, BONE_DEPTH, 0.0)))
        bone.tail = inverse @ apex
        bone.parent = ebones["UpperChest"]
        tip = ebones.new(f"{side}Breast_end")
        tip.head = bone.tail
        tip.tail = inverse @ (apex - Vector((0.0, TIP_LENGTH, 0.0)))
        tip.parent = bone
        tip.use_connect = True
    bpy.ops.object.mode_set(mode="OBJECT")

    for side, apex in apexes.items():
        name = f"{side}Breast"
        group = mesh.vertex_groups.get(name) or mesh.vertex_groups.new(name=name)
        centre = apex + Vector((0.0, 0.5 * BONE_DEPTH, 0.0))
        scaled = (verts - np.array(centre)) / np.array(RADIUS)
        dist = np.linalg.norm(scaled, axis=1)
        weight = np.clip((1.0 - dist) / (1.0 - FULL), 0.0, 1.0)
        weight = weight * weight * (3 - 2 * weight) * args.strength
        weight[blocked | (verts[:, 1] > centre.y + 0.5 * BONE_DEPTH)] = 0.0
        count = 0
        for index in np.nonzero(weight > 1e-3)[0]:
            v = mesh.data.vertices[int(index)]
            w = float(weight[index])
            for g in list(v.groups):
                if g.group != group.index:
                    mesh.vertex_groups[g.group].add([int(index)], g.weight * (1.0 - w), "REPLACE")
            group.add([int(index)], w, "REPLACE")
            count += 1
        print(f"{name}: {count} vertices")

    spring_bone = arm.data.vrm_addon_extension.spring_bone1
    for spring in list(spring_bone.springs):
        if spring.vrm_name == "Chest":
            spring_bone.springs.remove(list(spring_bone.springs).index(spring))
    for side in apexes:
        spring = spring_bone.add_spring()
        spring.vrm_name = "Chest"
        for name in (f"{side}Breast", f"{side}Breast_end"):
            joint = spring.joints.add()
            joint.node.bone_name = name
            joint.stiffness = STIFFNESS
            joint.drag_force = DRAG
            joint.gravity_power = GRAVITY
            joint.hit_radius = 0.02

    bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(args.out_blend), copy=True)
    result = bpy.ops.export_scene.vrm(filepath=bpy.path.abspath(args.out_vrm), armature_object_name=arm.name)
    print("exported", result)


main()
