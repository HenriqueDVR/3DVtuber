"""After reshaping fingers by hand in a finished rest pose, lay their bones down them again.

    blender -b in_rest.blend --python-exit-code 1 -P scripts/fit_finger_bones.py -- out.blend out.vrm right_index [left_thumb ...]

Fingers are named side_finger: left/right and thumb/index/middle/ring/little. Each named finger's
bones are moved onto the middle of its surface as it now is and its weight split among them
again (`tpose.fit_finger_bones`); the mesh does not move and nothing else changes. Prints how far
the bones sat from the finger before and after. Exports like the pipeline (`vrm_export.export`).
Never overwrites an existing file.
"""
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import fingers, hand_setup, tpose, vrm_export  # noqa: E402


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) < 3:
        sys.exit(__doc__)
    out_blend, out_vrm = Path(args[0]), Path(args[1])
    for path in (out_blend, out_vrm):
        if path.exists():
            sys.exit(f"{path} exists: choose a new name; files are never overwritten")
    wanted = []
    for name in args[2:]:
        side, _, finger = name.partition("_")
        if side not in ("left", "right") or finger not in fingers.FINGERS:
            sys.exit(f"'{name}' is not a finger: use e.g. right_index or left_thumb")
        wanted.append((side, finger))
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers)]
    for side, finger in wanted:
        before, after = tpose.fit_finger_bones(meshes, armature, side, finger)
        print(f"{side} {finger}: bones were up to {before * 1000:.1f} mm off the finger's middle, now {after * 1000:.1f} mm")
    tpose._align_finger_hinges(armature)
    hand_setup.reseed_guides(armature)
    vrm_export.ensure_vrm_addon()
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend.resolve()), copy=True)
    vrm_export.export(out_vrm.resolve(), armature)
    print("wrote", out_blend.name, "and", out_vrm.name)


if __name__ == "__main__":
    main()
