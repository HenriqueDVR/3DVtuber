"""Exercise the hand wizard against a real blend, without saving or exporting it.

blender -b --factory-startup avatar.blend --python-exit-code 1 -P scripts/check_hand_setup.py
This checks mechanics, not whether the suggested guide positions look right.
"""
import argparse
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
import vtuber_maker
from vtuber_maker.pipeline import hand_setup, meshdata


def evaluated(mesh):
    graph = bpy.context.evaluated_depsgraph_get()
    graph.update()
    return meshdata.world_vertices(mesh.evaluated_get(graph))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-test", type=Path, help="Optional new VRM for export verification, not an approved avatar")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if args.export_test and args.export_test.exists():
        raise ValueError("Choose an unused test export path")
    vtuber_maker.register()
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
    settings = bpy.context.scene.vtuber_hand_setup
    settings.mesh, settings.armature = mesh, arm
    before = evaluated(mesh)
    keys = {k.name: np.array([v.co[:] for v in k.data]) for k in mesh.data.shape_keys.key_blocks}
    # Shoulder/clothing and body groups must not change during either hand's setup.
    protected = {g.index for g in mesh.vertex_groups if g.name not in hand_setup._names("left") + hand_setup._names("right")}
    weights = [[(g.group, g.weight) for g in v.groups if g.group in protected] for v in mesh.data.vertices]
    for side in ("left", "right"):
        settings.side = side
        assert bpy.ops.vtuber_maker.hand_action(action="START") == {"FINISHED"}
        data = hand_setup.read_guides(arm, side)
        assert bpy.ops.vtuber_maker.hand_action(action="APPLY") == {"FINISHED"}
        np.testing.assert_allclose(evaluated(mesh), before, atol=1e-5)
        assert bpy.ops.vtuber_maker.hand_action(action="FIST") == {"FINISHED"}
        movement = np.linalg.norm(evaluated(mesh) - before, axis=1)
        assert movement.max() > .001, (side, "preview did not move the mesh")
        print("PASS", side, "preview maximum movement cm", float(movement.max() * 100))
        assert bpy.ops.vtuber_maker.hand_action(action="RESET") == {"FINISHED"}
        np.testing.assert_allclose(evaluated(mesh), before, atol=1e-5)
        assert data == hand_setup.read_guides(arm, side)
    for k in mesh.data.shape_keys.key_blocks:
        np.testing.assert_array_equal(keys[k.name], [v.co[:] for v in k.data])
    assert weights == [[(g.group, g.weight) for g in v.groups if g.group in protected] for v in mesh.data.vertices]
    print("PASS wizard mechanics on", bpy.data.filepath, "— clothing weights and shape keys unchanged; no blend saved")
    if args.export_test:
        import json
        import struct
        from vtuber_maker.pipeline import fingers
        assert bpy.ops.vtuber_maker.export_hand_vrm(filepath=str(args.export_test.resolve())) == {"FINISHED"}
        raw = args.export_test.read_bytes()
        length = struct.unpack_from("<I", raw, 12)[0]
        document = json.loads(raw[20:20 + length])
        human = document["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"]
        for side in ("left", "right"):
            for finger in fingers.FINGERS:
                for key, segment in zip(fingers.humanoid_keys(side, finger), fingers._segments(finger)):
                    first, *rest = key.split("_")
                    node = human[first + "".join(p.title() for p in rest)]["node"]
                    assert document["nodes"][node]["name"] == fingers.bone_name(side, finger, segment)
        print("PASS exported all 30 finger humanoid bindings to", args.export_test)
    vtuber_maker.unregister()


if __name__ == "__main__":
    main()
