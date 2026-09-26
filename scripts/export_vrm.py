"""Export a finished .blend as a .vrm the way the pipeline does.

    blender -b in.blend --python-exit-code 1 -P scripts/export_vrm.py -- out.vrm

Use this after editing a `*_rest.blend` by hand instead of File > Export > VRM. The add-on's own
export uses its "Auto" T-pose unless told otherwise and writes Blender's bone frames, which
Warudo's finger tracking twists; `vrm_export.export` exports the rest pose and normalizes the
joints. Also reports hand vertices that a shape key would move, which is what sculpting a mesh
with the wrong key active leaves behind. Never overwrites an existing file.
"""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import hand_setup, vrm_export  # noqa: E402


def stray_hand_keys(mesh):
    """Shape keys (other than the basis) that move vertices weighted to a hand."""
    keys = mesh.data.shape_keys
    if keys is None:
        return {}
    hand = {mesh.vertex_groups[n].index for side in ("left", "right") for n in hand_setup._names(side)
            if n in mesh.vertex_groups}
    on_hand = np.array([any(g.group in hand and g.weight > 0 for g in v.groups) for v in mesh.data.vertices])
    basis = np.zeros(len(mesh.data.vertices) * 3)
    keys.reference_key.data.foreach_get("co", basis)
    found = {}
    for key in keys.key_blocks:
        if key == keys.reference_key:
            continue
        co = np.zeros_like(basis)
        key.data.foreach_get("co", co)
        moved = np.linalg.norm((co - basis).reshape(-1, 3), axis=1) > 1e-5
        if np.any(moved & on_hand):
            found[key.name] = int((moved & on_hand).sum())
    return found


def main():
    out = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    if out.exists():
        sys.exit(f"{out} exists: choose a new name; files are never overwritten")
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    for mesh in (o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers)):
        for name, count in stray_hand_keys(mesh).items():
            print(f"WARNING: shape key '{name}' on {mesh.name} moves {count} hand vertices")
    vrm_export.ensure_vrm_addon()
    vrm_export.export(out, armature)
    print("wrote", out)


if __name__ == "__main__":
    main()
