"""Retune a rigged avatar .blend: hair springs, mouth size, head pivot. Exports VRM.

    blender -b in.blend -P scripts/tune_avatar.py -- out.blend out.vrm [--mouth 1.35] [--head-up 0.03]

Brings .blend files rigged with older settings in line with make_vrm.py's current defaults.
The original file is left alone. Needs the VRM Add-on for Blender.
"""
import argparse
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import hair_physics, rig  # noqa: E402
from vtuber_maker.pipeline.face_rig import VOWELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_blend")
    ap.add_argument("out_vrm")
    ap.add_argument("--mouth", type=float, default=1.35, help="scale the vowel mouth shapes by this")
    ap.add_argument("--head-up", type=float, default=0.0, help="raise the head pivot by this (m)")
    ap.add_argument("--chest-stiffness", type=float, help="chest spring stiffness (lower = more jiggle)")
    ap.add_argument("--chest-drag", type=float, help="chest spring drag (lower = bouncier)")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])

    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = next(o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm)

    springs = arm.data.vrm_addon_extension.spring_bone1.springs
    for spring in springs:
        joints = list(spring.joints)
        if not joints or not joints[0].node.bone_name.startswith("Hair"):
            continue
        for i, joint in enumerate(joints):
            along = i / max(len(joints) - 1, 1)
            joint.stiffness = hair_physics.HAIR_STIFFNESS + (hair_physics.HAIR_TIP_STIFFNESS - hair_physics.HAIR_STIFFNESS) * along
            joint.gravity_power = hair_physics.HAIR_GRAVITY_POWER
            joint.drag_force = hair_physics.HAIR_DRAG_FORCE
    print("retuned hair springs")
    for spring in springs:
        joints = list(spring.joints)
        if not joints or "Breast" not in joints[0].node.bone_name:
            continue
        for joint in joints:
            if args.chest_stiffness is not None:
                joint.stiffness = args.chest_stiffness
            if args.chest_drag is not None:
                joint.drag_force = args.chest_drag
    print("retuned chest springs")

    keys = mesh.data.shape_keys.key_blocks
    count = len(mesh.data.vertices)
    basis = np.empty(count * 3)
    keys["Basis"].data.foreach_get("co", basis)
    for name in VOWELS:
        if name not in keys:
            continue
        co = np.empty(count * 3)
        keys[name].data.foreach_get("co", co)
        keys[name].data.foreach_set("co", basis + (co - basis) * args.mouth)
    print(f"mouth shapes x{args.mouth}")

    if args.head_up:
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode="EDIT")
        edit = arm.data.edit_bones
        head, neck = edit[rig.BONE_NAMES["head"]], edit[rig.BONE_NAMES["neck"]]
        inverse = arm.matrix_world.inverted()
        world = arm.matrix_world @ head.head
        world.z += args.head_up
        head.head = inverse @ world
        neck.tail = head.head
        bpy.ops.object.mode_set(mode="OBJECT")
        print(f"head pivot raised {args.head_up} m")

    bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(args.out_blend), copy=True)
    print("exported", bpy.ops.export_scene.vrm(filepath=bpy.path.abspath(args.out_vrm), armature_object_name=arm.name))


main()
