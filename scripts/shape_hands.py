"""Reshape both hands of a finished T-pose .blend after a reference VRM hand, then export.

    blender -b in_rest.blend --python-exit-code 1 -P scripts/shape_hands.py -- out.blend out.vrm
        [--reference samples/reference/VRM1_Constraint_Twist_Sample.vrm] [--amount 1.0]

Input: a `*_rest.blend` from `to_tpose.py --spec-hands` (the reference must share that rest
pose). The reference hand is fitted to each hand's bones and thickness and the hand is moved
toward it (`pipeline/hand_shape.py`). Weights, bones, UVs and shape keys are kept; nothing but
hand vertices and their shading normals changes. Never overwrites an existing file.
"""
import argparse
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import hand_shape, vrm_export  # noqa: E402

REFERENCE = Path(__file__).resolve().parents[1] / "samples" / "reference" / "VRM1_Constraint_Twist_Sample.vrm"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_blend", type=Path)
    parser.add_argument("out_vrm", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--amount", type=float, default=1.0, help="0 keeps the hand, 1 takes the reference form")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:])


def main():
    args = parse_args()
    for path in (args.out_blend, args.out_vrm):
        if path.exists():
            sys.exit(f"{path} exists: choose a new name; files are never overwritten")
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers)]
    vrm_export.ensure_vrm_addon()
    reference = hand_shape.read_reference(args.reference)
    for mesh in meshes:
        for side in ("left", "right"):
            if not any(n.startswith(side.capitalize()) for n in mesh.vertex_groups.keys()):
                continue
            moved = hand_shape.shape_hand(mesh, armature, side, reference[side], amount=args.amount)
            print(f"{mesh.name} {side}: reshaped {moved} vertices")
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_blend.resolve()), copy=True)
    vrm_export.export(args.out_vrm.resolve(), armature)
    print("wrote", args.out_blend.name, "and", args.out_vrm.name)


if __name__ == "__main__":
    main()
