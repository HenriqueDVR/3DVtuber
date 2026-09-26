"""Assert a finished bake preserves the modeled hand surface in hand-local coordinates.

blender -b source.blend --python-exit-code 1 -P scripts/check_hand_preservation.py -- candidate.blend
Checks the actual vertex positions in EVERY shape key, not just bone direction or edge stretch.
Both files must retain the same topology; no files are written.
"""
import sys
from pathlib import Path

import bpy
import numpy as np


def coordinates_in_hand(obj, bone, armature, coords):
    matrix = np.array((armature.matrix_world @ bone.matrix_local).inverted() @ obj.matrix_world)
    return coords @ matrix[:3, :3].T + matrix[:3, 3]


def snapshot():
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
    coords = {key.name: np.array([p.co[:] for p in key.data]) for key in mesh.data.shape_keys.key_blocks}
    hands = {}
    for side in ("Left", "Right"):
        names = {side + "Hand"} | {g.name for g in mesh.vertex_groups
                                         if any(g.name.startswith(side + f) for f in ("Thumb", "Index", "Middle", "Ring", "Little"))}
        groups = {g.index for g in mesh.vertex_groups if g.name in names}
        selected = np.array([v.index for v in mesh.data.vertices
                             if sum(g.weight for g in v.groups if g.group in groups) > .5])
        assert len(selected), f"no {side} hand vertices"
        bone = arm.data.bones[side + "Hand"]
        hands[side] = (selected, {name: coordinates_in_hand(mesh, bone, arm, points[selected]) for name, points in coords.items()})
    return mesh.name, len(mesh.data.vertices), [tuple(p.vertices) for p in mesh.data.polygons], hands


def main():
    name, count, topology, source = snapshot()
    target = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    bpy.ops.wm.open_mainfile(filepath=str(target))
    mesh = bpy.data.objects[name]
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    assert len(mesh.data.vertices) == count
    assert topology == [tuple(p.vertices) for p in mesh.data.polygons], "topology changed"
    for side, (selected, keys) in source.items():
        assert set(keys) == set(mesh.data.shape_keys.key_blocks.keys())
        maximum = 0
        for name, original in keys.items():
            coords = np.array([p.co[:] for p in mesh.data.shape_keys.key_blocks[name].data])[selected]
            actual = coordinates_in_hand(mesh, arm.data.bones[side + "Hand"], arm, coords)
            error = float(np.linalg.norm(actual - original, axis=1).max())
            maximum = max(maximum, error)
            assert error < 1e-5, f"{side} {name}: hand surface changed by {error * 1000:.3f} mm"
        print(f"PASS {side}: {len(selected)} vertices, {len(keys)} keys, maximum change {maximum * 1000:.6f} mm")


if __name__ == "__main__":
    main()
