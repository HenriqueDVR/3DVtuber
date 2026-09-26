"""Bend the knees of an existing rig forward a little, then export VRM.

    blender -b in.blend -P scripts/bend_knees.py -- out.blend out.vrm

Apps bend legs with IK in the direction the rest pose already bends them (crouching when the
head drops, sitting). Straight or back-bent knees leave that to chance and the legs fold
sideways. New builds do this themselves (landmarks.bend_knees); this fixes a .blend built
before, keeping its weights and hand edits. The original file is left alone.
"""
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import landmarks, rig  # noqa: E402


def main():
    out_blend, out_vrm = sys.argv[sys.argv.index("--") + 1:]
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    matrix = arm.matrix_world
    bones = arm.data.bones
    keys = [f"{side}_{k}" for side in ("left", "right") for k in ("upper_leg", "lower_leg", "foot")]
    joints = {k: np.array(matrix @ bones[rig.BONE_NAMES[k]].head_local) for k in keys}
    bent = landmarks.bend_knees(joints)

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    inverse = matrix.inverted()
    for side in ("left", "right"):
        knee = inverse @ Vector(bent[f"{side}_lower_leg"].tolist())
        before = joints[f"{side}_lower_leg"]
        arm.data.edit_bones[rig.BONE_NAMES[f"{side}_upper_leg"]].tail = knee
        arm.data.edit_bones[rig.BONE_NAMES[f"{side}_lower_leg"]].head = knee
        print(f"{side} knee moved {np.round((bent[f'{side}_lower_leg'] - before) * 100, 1)} cm")
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(out_blend), copy=True)
    print("exported", bpy.ops.export_scene.vrm(filepath=bpy.path.abspath(out_vrm), armature_object_name=arm.name))


main()
