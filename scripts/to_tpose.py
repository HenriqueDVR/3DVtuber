"""Swing the arms out to the VRM rest pose (a T-pose), then export VRM.

    blender -b in.blend -P scripts/to_tpose.py -- out.blend out.vrm [--angle 0]

VRM 1.0 says the rest pose must be a T-pose: arms straight out to the sides. Image-to-3D
characters come in an A-pose, and an app that assumes a T-pose applies its tracking rotations
against a rest pose 40-70 degrees off, so the arms shoot outwards the moment arm tracking
starts. This bakes the T-pose into the mesh and makes it the armature's rest pose; the A-pose
becomes just another pose. `--angle` tilts the arms down from horizontal in degrees (0 is the
spec T-pose). Runs last, after weights and face tracking, so it moves the finished mesh. The
original file is left alone. Needs the VRM Add-on for Blender.

`--keep-finger-pose` preserves the hand's modeled curvature rather than flattening fingers.
This is a tracking experiment, not the fully open reference hand: verify offsets in the app.
`--repair-shoulder-seams` closes coincident shoulder weight seams without changing topology.
It does not fill pre-existing holes or alter disconnected clothing further down the body.
"""
import argparse
import math
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import hand_setup, loose, tpose, vrm_export  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_blend", type=Path)
    parser.add_argument("out_vrm", type=Path)
    parser.add_argument("--angle", type=float, default=0.0, help="degrees below horizontal (0 = spec T-pose)")
    parser.add_argument("--keep-finger-pose", action="store_true",
                        help="raise arms but preserve modeled finger curvature; tracking may retain a curl offset")
    parser.add_argument("--spec-hands", action="store_true",
                        help="hands exactly in the VRM T-pose: level, palm down, fingers straight, "
                             "thumb 45 degrees (after fix_fingers.py)")
    parser.add_argument("--relaxed-fingers", action="store_true",
                        help="rest every finger in one even, lightly bent pose (after fix_fingers.py)")
    parser.add_argument("--open-thumb", action="store_true",
                        help="with --keep-finger-pose: still open the thumb to the VRM rest angle")
    parser.add_argument("--repair-shoulder-seams", action="store_true",
                        help="synchronize shoulder seam weights without welding or cutting the mesh")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if argv is None else argv)


def main():
    args = parse_args()
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers)]
    if not meshes:
        sys.exit("no skinned mesh in the file")

    if args.repair_shoulder_seams:
        for mesh in meshes:
            changed = loose.synchronize_shoulder_seams(mesh, armature)
            print(f"{mesh.name}: synchronized {changed} shoulder seam vertices (topology unchanged)")
    down = math.radians(args.angle)
    direction = (math.cos(down), 0.0, -math.sin(down))
    turned = tpose.to_t_pose(meshes[0], armature, direction=direction, extra_meshes=meshes[1:],
                            straighten="spec" if args.spec_hands else "relaxed" if args.relaxed_fingers else
                            (("thumb",) if args.open_thumb else False) if args.keep_finger_pose else True)
    if args.keep_finger_pose:
        print("preserved modeled finger curvature; verify tracking offsets in the target app")
    for side in hand_setup.reseed_guides(armature):
        print(f"{side} hand guides moved onto the baked finger bones")
    for side, angle in sorted(turned.items()):
        print(f"{side} arm raised {angle:.1f} degrees")
    if not any(turned.values()):
        print("arms were already in the rest pose; nothing moved")

    vrm_export.ensure_vrm_addon()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_blend.resolve()), copy=True)
    vrm_export.export(args.out_vrm.resolve(), armature)
    print("wrote", args.out_blend.name, "and", args.out_vrm.name)


if __name__ == "__main__":
    main()
