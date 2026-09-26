"""Read joint positions from a model someone else rigged, as manifest joint overrides.

    blender -b --python-exit-code 1 -P scripts/joints_from_rig.py -- rigged.fbx --json joints.json [--height 1.6]

For characters whose limbs can't be told apart from what is beside them (model C's wings touch
her arms everywhere, and detection falls back to typical proportions): an auto-rigger that saw
the whole model, like Tripo's, places them from what it learnt instead. Its bones (VRoid J_Bip_*
names) are read at their heads and moved into the pipeline's normalized frame -- the same scale
and offset normalize.prepare gives the rigged mesh -- so they sit on the unrigged source too, if
it is the same generation. Writes {joint key: [x, y, z]} for `python -m app joint`. Changes nothing.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import meshdata, normalize  # noqa: E402

KEYS = {"hips": "C_Hips", "spine": "C_Spine", "chest": "C_Chest", "upper_chest": "C_UpperChest",
        "neck": "C_Neck", "head": "C_Head"}
for letter, side in (("L", "left"), ("R", "right")):
    KEYS.update({f"{side}_{key}": f"{letter}_{name}" for key, name in
                 (("upper_arm", "UpperArm"), ("lower_arm", "LowerArm"), ("hand", "Hand"),
                  ("upper_leg", "UpperLeg"), ("lower_leg", "LowerLeg"), ("foot", "Foot"))})


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("rigged", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--height", type=float, default=1.6)
    args = parser.parse_args(argv)

    bpy.ops.wm.read_homefile(use_empty=True)
    if args.rigged.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(args.rigged.resolve()))
    else:
        bpy.ops.import_scene.gltf(filepath=str(args.rigged.resolve()))
    armature = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
    world = np.array(armature.matrix_world)
    heads = {key: (world @ np.array([*armature.data.bones[f"J_Bip_{name}"].head_local, 1.0]))[:3]
             for key, name in KEYS.items() if f"J_Bip_{name}" in armature.data.bones}
    if np.isnan(np.array(list(heads.values()))).any():
        raise SystemExit("the rig's joints don't import (NaN): Tripo's GLB export has broken bind matrices; use its FBX")

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    before = np.concatenate([meshdata.world_vertices(m) for m in meshes])
    for m in meshes:  # normalize the mesh on its own; the armature stays where it was
        world_matrix = m.matrix_world.copy()
        m.parent = None
        m.matrix_world = world_matrix
        m.modifiers.clear()
    mesh = normalize.prepare(meshes, height=args.height)
    after = meshdata.world_vertices(mesh)
    if not np.allclose(np.ptp(before, axis=0) / np.ptp(before[:, 2]), np.ptp(after, axis=0) / np.ptp(after[:, 2]), atol=1e-3):
        raise SystemExit("normalizing turned the model; joints can't be carried over by scale alone")
    scale = np.ptp(after[:, 2]) / np.ptp(before[:, 2])
    offset = after.min(axis=0) - scale * before.min(axis=0)
    joints = {key: [round(float(v), 4) for v in scale * p + offset] for key, p in heads.items()}
    args.json.write_text(json.dumps(joints, indent=2), encoding="utf-8")
    print(f"scale {scale:.4f}, offset {np.round(offset, 4)}; {len(joints)} joints written to {args.json}")
    for key, p in joints.items():
        print(f"JOINT {key:16s} {p}")


main()
