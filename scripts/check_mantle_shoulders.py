"""Read-only model B mantle motion checks, including an older rig of the same model.

Run inside Blender. Reference matching is on modeled A-pose files. For a baked derivative,
pass --baked after its matching --source; topology is checked before reusing vertex masks.
Never saves a blend or changes the user's existing files.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vtuber_maker.pipeline import meshdata
from repair_clothing import reference_masks


def targets():
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
    return mesh, arm


def measure(mesh, arm, masks):
    saved = {b.name: b.matrix_basis.copy() for b in arm.pose.bones}
    def coords():
        graph = bpy.context.evaluated_depsgraph_get()
        graph.update()
        return meshdata.world_vertices(mesh.evaluated_get(graph))
    rest = coords()
    result = {}
    try:
        for bone_kind in ("Shoulder", "UpperArm"):
            for axis in ("X", "Z"):
                for side in ("Left", "Right"):
                    for name, basis in saved.items():
                        arm.pose.bones[name].matrix_basis = basis
                    bone = arm.pose.bones[side + bone_kind]
                    bone.matrix_basis = bone.matrix_basis @ Matrix.Rotation(math.radians(30), 4, axis)
                    now = coords()
                    movement = np.linalg.norm(now - rest, axis=1) * 100
                    result[side + bone_kind + "_" + axis] = {
                        role: {"vertices": int(mask.sum()), "moving_over_1mm": int((movement[mask] > .1).sum()),
                               "mean_cm": float(movement[mask].mean()), "max_cm": float(movement[mask].max())}
                        for role, mask in masks.items()}
    finally:
        for name, basis in saved.items():
            arm.pose.bones[name].matrix_basis = basis
        bpy.context.view_layer.update()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("samples/model_matte.glb"))
    parser.add_argument("--baked", type=Path)
    parser.add_argument("--vrm", type=Path, help="Reimport the baked VRM and check its exported skinning too")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.report.exists():
        raise ValueError("Choose an unused report path")
    bpy.ops.wm.open_mainfile(filepath=str(args.source.resolve()))
    mesh, arm = targets()
    masks = reference_masks(mesh, args.reference, "mantle")
    topology = [tuple(p.vertices) for p in mesh.data.polygons]
    count = len(mesh.data.vertices)
    results = {str(args.source): measure(mesh, arm, masks)}
    if args.baked:
        bpy.ops.wm.open_mainfile(filepath=str(args.baked.resolve()))
        mesh, arm = targets()
        if count != len(mesh.data.vertices) or topology != [tuple(p.vertices) for p in mesh.data.polygons]:
            raise ValueError("Baked topology differs; cannot reuse mantle membership")
        results[str(args.baked)] = measure(mesh, arm, masks)
        for side in ("Left", "Right"):
            checks = [results[str(args.baked)][side + "UpperArm_" + axis]["mantle"] for axis in ("X", "Z")]
            assert all(c["moving_over_1mm"] > 50 and c["max_cm"] > 1 for c in checks), checks
    if args.vrm:
        if not args.baked:
            raise ValueError("--vrm requires its matching --baked file")
        from mathutils.kdtree import KDTree
        from vtuber_maker.pipeline import vrm_export
        original = meshdata.world_vertices(mesh)
        tree = KDTree(len(original))
        for i, p in enumerate(original):
            tree.insert(p, i)
        tree.balance()
        # The VRM exporter can normalize humanoid rest transforms. UVs identify cloth
        # even when positions change. Keep only unambiguous, exact texture-coordinate matches.
        uv_roles = {}
        source_uv = mesh.data.uv_layers.active
        if source_uv is None:
            raise ValueError("The source needs UVs for exported garment identification")
        for loop in mesh.data.loops:
            key = tuple(round(float(v), 5) for v in source_uv.data[loop.index].uv)
            membership = tuple(bool(mask[loop.vertex_index]) for mask in masks.values())
            uv_roles.setdefault(key, set()).add(membership)
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        vrm_export.ensure_vrm_addon()
        assert bpy.ops.import_scene.vrm(filepath=str(args.vrm.resolve())) == {"FINISHED"}
        mesh, arm = targets()
        matches = [tree.find(p) for p in meshdata.world_vertices(mesh)]
        distances = np.array([m[2] for m in matches])
        print("VRM_MATCH", mesh.name, "distance_mm", np.percentile(distances * 1000, [50, 95, 99, 100]).tolist(),
              "source_bounds", [original.min(axis=0).tolist(), original.max(axis=0).tolist()],
              "import_bounds", [meshdata.world_vertices(mesh).min(axis=0).tolist(), meshdata.world_vertices(mesh).max(axis=0).tolist()])
        if distances.max() < .0001:
            exported_masks = {role: mask[[m[1] for m in matches]] for role, mask in masks.items()}
        else:
            candidates = [set() for _ in mesh.data.vertices]
            uv = mesh.data.uv_layers.active
            if uv is None:
                raise ValueError("Exported mesh has no UVs")
            for loop in mesh.data.loops:
                key = tuple(round(float(v), 5) for v in uv.data[loop.index].uv)
                candidates[loop.vertex_index].update(uv_roles.get(key, set()))
            exported_masks = {role: np.array([len(c) == 1 and next(iter(c))[i] for c in candidates])
                              for i, role in enumerate(masks)}
            print("UV_IDENTIFICATION", {role: int(mask.sum()) for role, mask in exported_masks.items()})
            assert exported_masks["mantle"].sum() > 900, "Too few unambiguous mantle vertices"
        results[str(args.vrm)] = measure(mesh, arm, exported_masks)
        for side in ("Left", "Right"):
            for axis in ("X", "Z"):
                check = results[str(args.vrm)][side + "UpperArm_" + axis]["mantle"]
                assert check["moving_over_1mm"] > 50 and check["max_cm"] > 1, check
    args.report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print("PASS measured actual mantle deformation; no avatar files saved")


if __name__ == "__main__":
    main()
