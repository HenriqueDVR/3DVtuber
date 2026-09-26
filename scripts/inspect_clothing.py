"""Read-only clothing/component comparison against an unrigged GLB, in the modelled pose.

blender -b before_tpose.blend --python-exit-code 1 -P scripts/inspect_clothing.py -- original.glb report.json
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils.kdtree import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from vtuber_maker.pipeline import meshdata, normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
    old = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(args.reference.resolve()))
    ref = normalize.prepare(list(set(bpy.data.objects) - old))
    original = meshdata.world_vertices(ref)
    labels = meshdata.pieces(ref, original)
    tree = KDTree(len(original))
    for i, point in enumerate(original):
        tree.insert(point, i)
    tree.balance()
    coords = meshdata.world_vertices(mesh)
    matches = [tree.find(point) for point in coords]
    nearest = np.array([m[1] for m in matches])
    distances = np.array([m[2] for m in matches])
    mapped = labels[nearest]
    names = [g.name for g in mesh.vertex_groups]
    weights = np.zeros((len(coords), len(names)))
    for v in mesh.data.vertices:
        for g in v.groups:
            weights[v.index, g.group] = g.weight
    components = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        indices = np.flatnonzero(mapped == label)
        mean = weights[indices].mean(axis=0) if len(indices) else np.zeros(len(names))
        components.append({"label": int(label), "original_vertices": len(members), "mapped_vertices": len(indices),
                           "bounds": [original[members].min(axis=0).tolist(), original[members].max(axis=0).tolist()],
                           "centre": original[members].mean(axis=0).tolist(),
                           "weights": {names[i]: round(float(mean[i]), 4) for i in np.flatnonzero(mean > .005)}})
    result = {"source": bpy.data.filepath, "reference": str(args.reference),
              "match_distance_mm": np.percentile(distances * 1000, [50, 95, 99, 100]).tolist(),
              "bones": {b.name: [list(arm.matrix_world @ b.head_local), list(arm.matrix_world @ b.tail_local)]
                        for b in arm.data.bones if b.name in ("Hips", "Chest", "UpperChest", "LeftUpperArm", "LeftLowerArm", "LeftHand", "RightUpperArm", "RightLowerArm", "RightHand")},
              "components": components}
    from mathutils import Vector
    from vtuber_maker.pipeline import tpose
    for side, sign in (("Left", 1), ("Right", -1)):
        tpose._point_arm(arm, side + "UpperArm", side + "Hand", Vector((sign, 0, 0)))
    graph = bpy.context.evaluated_depsgraph_get()
    graph.update()
    evaluated = mesh.evaluated_get(graph)
    posed = meshdata.world_vertices(evaluated)
    separated = meshdata.pieces(mesh, posed)
    fragments = []
    for label in np.unique(separated):
        indices = np.flatnonzero(separated == label)
        mean = weights[indices].mean(axis=0)
        fragments.append({"label": int(label), "vertices": len(indices),
                          "reference_parts": np.unique(mapped[indices]).tolist(),
                          "rest_bounds": [coords[indices].min(axis=0).tolist(), coords[indices].max(axis=0).tolist()],
                          "posed_centre": posed[indices].mean(axis=0).tolist(),
                          "weights": {names[i]: round(float(mean[i]), 4) for i in np.flatnonzero(mean > .005)}})
    result["posed_components"] = fragments
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("WROTE", args.report, "components", len(components), "match mm", result["match_distance_mm"])


if __name__ == "__main__":
    main()
