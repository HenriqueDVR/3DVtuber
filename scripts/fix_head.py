"""Move a rigged avatar's head joint to the chin and make the face follow the head fully.

    blender -b in.blend -P scripts/fix_head.py -- out.blend out.vrm

For .blend files rigged before make_vrm.py placed the head joint at the chin; the original
file is left alone. Needs the VRM Add-on for Blender.
"""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import parts, rig  # noqa: E402
from vtuber_maker.pipeline.meshdata import world_vertices  # noqa: E402


def main():
    out_blend, out_vrm = sys.argv[sys.argv.index("--") + 1:]
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = next(o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm)
    verts = world_vertices(mesh)
    edges = np.empty(len(mesh.data.edges) * 2, dtype=np.int64)
    mesh.data.edges.foreach_get("vertices", edges)
    labels = parts.split(verts, edges.reshape(-1, 2))

    bones = arm.data.bones
    head_bone = bones[rig.BONE_NAMES["head"]]
    neck_z = (arm.matrix_world @ bones[rig.BONE_NAMES["neck"]].head_local).z
    head_joint = arm.matrix_world @ head_bone.head_local
    # the face shell: the part holding the front-most point near the head joint's height
    near = (np.abs(verts[:, 0] - head_joint.x) < 0.02) & (np.abs(verts[:, 2] - head_joint.z) < 0.02)
    face = labels == labels[np.nonzero(near)[0][np.argmin(verts[near, 1])]]
    chin = rig.chin_height(verts, face, head_joint.x, neck_z, head_joint.z)
    if chin is None or chin <= neck_z + 0.01:
        sys.exit(f"couldn't find the chin (got {chin})")
    print(f"head joint {head_joint.z:.3f} -> chin {chin:.3f}")

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    edit = arm.data.edit_bones[rig.BONE_NAMES["head"]]
    inverse = arm.matrix_world.inverted()
    world_head = arm.matrix_world @ edit.head
    world_head.z = chin
    edit.head = inverse @ world_head
    neck = arm.data.edit_bones[rig.BONE_NAMES["neck"]]
    world_neck_tail = arm.matrix_world @ neck.tail
    world_neck_tail.z = chin
    neck.tail = inverse @ world_neck_tail
    bpy.ops.object.mode_set(mode="OBJECT")

    rig.weight_head(mesh, arm, face, labels)
    bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(out_blend), copy=True)
    print("exported", bpy.ops.export_scene.vrm(filepath=bpy.path.abspath(out_vrm), armature_object_name=arm.name))


main()
