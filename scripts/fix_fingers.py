"""Rebuild both hands' finger bones and weights from the hand's surface, on the A-pose rig.

    blender -b in_apose.blend --python-exit-code 1 -P scripts/fix_fingers.py -- out.blend

Lays each finger's bones down the middle of the finger (`pipeline/finger_guides.py`) and weights
them through the Guided Hand Setup editor, whose guides are left in the file for adjusting by
hand. Then bake WITHOUT `--keep-finger-pose`, so the fingers are straight at rest:

    blender -b out.blend --python-exit-code 1 -P scripts/to_tpose.py -- final.blend final.vrm

and measure with `scripts/check_fingers.py`. Refuses an existing output and a posed rig; the
input file is left alone. Writes a JSON report next to the output.
"""
import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from vtuber_maker.pipeline import finger_guides  # noqa: E402


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    if len(argv) != 1:
        sys.exit(__doc__)
    out = Path(argv[0]).resolve()
    if out.exists():
        sys.exit(f"{out} exists: choose a new name; originals are never overwritten")
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
    if any(not np.allclose(np.array(b.matrix_basis), np.eye(4), atol=1e-6) for b in arm.pose.bones):
        sys.exit("clear the pose first: finger guides are read on the rest pose")
    report = finger_guides.rebuild(mesh, arm)
    for side, row in report.items():
        print(side, json.dumps(row))
    out.with_suffix(".json").write_text(json.dumps({"source": bpy.data.filepath, **report}, indent=2),
                                        encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(out), copy=True)
    print("wrote", out.name)


if __name__ == "__main__":
    main()
