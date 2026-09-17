"""Rebuild hair spring settings and body colliders on a rigged avatar .blend, then export VRM.

    blender -b in.blend -P scripts/rebuild_hair_physics.py -- out.blend out.vrm

Replaces the hair springs and all colliders with make_vrm.py's current design (softer hair,
colliders fitted to head, neck, shoulders, torso and limbs), and adds colliders on chest jiggle
bones if there are any. Chest springs are kept. The original file is left alone.
"""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import hair_physics, rig, spring_sim, vrm_export  # noqa: E402

# Chest colliders: a sphere this big round each breast bone's tip, set this far back along it.
BREAST_RADIUS = 0.045
BREAST_BACK = 0.035


def main():
    out_blend, out_vrm = sys.argv[sys.argv.index("--") + 1:]
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = next(o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm)
    bones = arm.data.bones
    joints = {key: np.array(arm.matrix_world @ bones[name].head_local) for key, name in rig.BONE_NAMES.items()
              if name in bones}
    spring_bone = arm.data.vrm_addon_extension.spring_bone1

    chains = [[j.node.bone_name for j in s.joints] for s in spring_bone.springs
              if len(s.joints) and s.joints[0].node.bone_name.startswith("Hair")]
    # chains as build.py hands them over: the tip joint isn't its own bone in the design
    chains = [names if names[-1] in bones else names[:-1] for names in chains]
    print(f"{len(chains)} hair chains")

    # drop hair springs, all collider groups and colliders (the chest springs stay)
    for index in reversed(range(len(spring_bone.springs))):
        joints_ = spring_bone.springs[index].joints
        if len(joints_) and joints_[0].node.bone_name.startswith("Hair"):
            spring_bone.springs.remove(index)
    for index in reversed(range(len(spring_bone.collider_groups))):
        bpy.ops.vrm.remove_spring_bone1_collider_group(armature_object_name=arm.name, collider_group_index=index)
    for index in reversed(range(len(spring_bone.colliders))):
        bpy.ops.vrm.remove_spring_bone1_collider(armature_object_name=arm.name, collider_index=index)

    design = hair_physics.design(mesh, arm, chains, joints)
    for side in ("Left", "Right"):
        name = f"{side}Breast"
        if name in bones:
            tip = np.array(arm.matrix_world @ bones[name].tail_local)
            head = np.array(arm.matrix_world @ bones[name].head_local)
            direction = (head - tip) / np.linalg.norm(head - tip)
            design.colliders.append(spring_sim.Collider(name, BREAST_RADIUS, offset=tip + BREAST_BACK * direction))
    for c in design.colliders:
        print(f"collider {c.bone:14s} r={c.radius:.3f} {'capsule' if c.tail is not None else 'sphere'}")
    vrm_export.bind_spring_bones(arm, design)

    bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(out_blend), copy=True)
    print("exported", bpy.ops.export_scene.vrm(filepath=bpy.path.abspath(out_vrm), armature_object_name=arm.name))


main()
