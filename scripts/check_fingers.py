"""Measure whether a baked rig's fingers will curl cleanly in a tracking app.

blender -b avatar.blend --python-exit-code 1 -P scripts/check_fingers.py -- [--json out.json]

Unity-based apps (Warudo, VNyan) derive each finger's bend axis from the rest pose, so a finger
that is curled, kinked or rolled at rest bends sideways when tracked. For every finger this
reports, in degrees:
  bent       - largest angle between neighbouring segments at rest (VRM wants straight fingers)
  out_of_palm - how far the finger leaves the palm plane at rest
  hinge      - largest angle between a segment's X axis and the finger's shared hinge
  sideways   - when every segment curls 30 degrees about its own X, how far the fingertip's
               path leans out of the plane the finger should fold in
  toward_palm - whether that curl closes the hand (True) or bends the finger backwards
Prints PASS/FAIL per hand. Does not modify or save the file.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from vtuber_maker.pipeline import fingers, tpose
from vtuber_maker.pipeline.rig import BONE_NAMES

# Limits for a finger an app can drive without it folding sideways, in degrees.
LIMITS = {"bent": 3.0, "out_of_palm": 5.0, "hinge": 3.0, "sideways": 8.0}


def measure(arm, side):
    hand = arm.pose.bones.get(BONE_NAMES[f"{side}_hand"])
    if hand is None:
        return None
    along = (hand.tail - hand.head).normalized()
    across = tpose._across_palm(arm, side, along)
    if across is None:
        return None
    normal = along.cross(across).normalized()
    palm = normal * (1 if side == "left" else -1)  # the side the fingers close toward
    result = {}
    for finger in fingers.FINGERS:
        chain = [arm.pose.bones.get(fingers.bone_name(side, finger, s)) for s in fingers._segments(finger)]
        if not all(chain):
            continue
        dirs = [(b.tail - b.head).normalized() for b in chain]
        bent = max(math.degrees(a.angle(b)) for a, b in zip(dirs, dirs[1:]))
        out = max(abs(math.degrees(math.asin(max(-1, min(1, d.dot(normal)))))) for d in dirs)
        hinges = [b.matrix.to_3x3().col[0].normalized() for b in chain]
        # the thumb folds across the palm, so only its own segments must agree on a hinge
        shared = hinges[0] if finger == "thumb" else (dirs[0].cross(normal).normalized() * -1)
        if finger != "thumb" and shared.dot(hinges[0]) < 0:
            shared = -shared
        hinge = max(math.degrees(h.angle(shared)) for h in hinges)
        tip = chain[-1].tail.copy()
        try:
            for b in chain:
                b.rotation_mode = "XYZ"
                b.rotation_euler = (math.radians(30), 0, 0)
            bpy.context.view_layer.update()
            moved = chain[-1].tail - tip
            sideways = math.degrees(math.asin(min(1, abs(moved.normalized().dot(hinges[0])))))
            # a thumb closes across the palm, toward the little finger
            toward = moved.dot(palm if finger != "thumb" else palm - across) > 0
        finally:
            for b in chain:
                b.matrix_basis = Matrix.Identity(4)
            bpy.context.view_layer.update()
        result[finger] = {"bent": round(bent, 2), "out_of_palm": round(out, 2), "hinge": round(hinge, 2),
                          "sideways": round(sideways, 2), "toward_palm": bool(toward)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    if any(not np.allclose(np.array(b.matrix_basis), np.eye(4), atol=1e-6) for b in arm.pose.bones):
        raise ValueError("clear the pose first: fingers are measured at rest")
    report = {}
    for side in ("left", "right"):
        rows = measure(arm, side)
        if rows is None:
            print(f"SKIP {side}: no hand or fingers")
            continue
        report[side] = rows
        failed = []
        for finger, row in rows.items():
            print(f"{side:5s} {finger:6s} " + "  ".join(f"{k} {v}" for k, v in row.items()))
            # the thumb sits 45 degrees out of line by design; only its hinge and curl count
            keys = ("hinge", "sideways") if finger == "thumb" else LIMITS
            failed += [f"{finger} {k} {row[k]}" for k in keys if row[k] > LIMITS[k]]
            if not row["toward_palm"]:
                failed.append(f"{finger} curls backwards")
        print(("PASS " if not failed else "FAIL ") + side + ("" if not failed else ": " + "; ".join(failed)))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
