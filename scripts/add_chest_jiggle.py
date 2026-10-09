"""Add chest bounce (two spring bones) to a rigged avatar .blend and export it as VRM.

    blender -b in.blend -P scripts/add_chest_jiggle.py -- out.blend out.vrm [--amount 0.5] [--strength 0.8] [--height 1.6]

Works on a .blend saved by make_vrm.py --blend or any later step (the app runs it after the
T-pose, option `chest`). The original file is left alone. Needs the VRM Add-on for Blender.
See pipeline/chest.py for what it touches.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import chest, improve, vrm_export  # noqa: E402
from vtuber_maker.pipeline.rig import BONE_NAMES  # noqa: E402


def _on_chest(mesh):
    """How many of a mesh's vertices the chest bones hold: of a body in several meshes (a .vrm
    taken in: face, body, hair), the one with the chest on it."""
    groups = {mesh.vertex_groups[BONE_NAMES[key]].index for key in ("chest", "upper_chest")
              if BONE_NAMES[key] in mesh.vertex_groups}
    return sum(any(g.group in groups and g.weight > 0.3 for g in v.groups) for v in mesh.data.vertices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_blend")
    ap.add_argument("out_vrm")
    ap.add_argument("--strength", type=float, default=chest.DEFAULT_STRENGTH,
                    help="max share of weight the chest bones take (0..1)")
    ap.add_argument("--amount", type=float, default=0.5,
                    help=f"how much it bounces, {chest.AMOUNT_RANGE[0]} to {chest.AMOUNT_RANGE[1]}: softer springs holding more of the chest")
    ap.add_argument("--height", type=float, default=1.6)
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = max((o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm),
               key=_on_chest if improve.imported(arm) else lambda o: len(o.data.vertices))
    try:
        weighted = chest.add(mesh, arm, args.strength, args.height, args.amount)
    except ValueError as error:
        sys.exit(f"chest: {error}")
    for side, count in weighted.items():
        print(f"{side} chest bone: {count} vertices")
    bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(args.out_blend), copy=True)
    vrm_export.export(Path(bpy.path.abspath(args.out_vrm)), arm)
    print(f"VTUBER_MAKER_REPORT {json.dumps({'chest': weighted})}")


main()
