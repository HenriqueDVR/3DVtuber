"""Read-only comparison of a model's original surface and its arms-out deformation.

blender -b source.blend --python-exit-code 1 -P scripts/inspect_deformation.py -- report.json
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from vtuber_maker.pipeline import meshdata, tpose


def evaluated_coords(obj):
    graph = bpy.context.evaluated_depsgraph_get()
    graph.update()
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh()
    points = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    evaluated.to_mesh_clear()
    return points


def inspect(mesh, arm):
    before = meshdata.world_vertices(mesh)
    labels = meshdata.pieces(mesh, before)
    weld, _, _ = meshdata.welded_edges(mesh, before)
    names = {g.index: g.name for g in mesh.vertex_groups}

    def weights(i):
        return {names[g.group]: round(g.weight, 4) for g in mesh.data.vertices[i].groups if g.weight > .01}

    for side, sign in (("Left", 1), ("Right", -1)):
        tpose._point_arm(arm, side + "UpperArm", side + "Hand", Vector((sign, 0, 0)))
    after = evaluated_coords(mesh)
    gaps = np.linalg.norm(after - after[weld], axis=1)
    seam_indices = np.flatnonzero(gaps > .002)
    counts = {int(label): int((labels == label).sum()) for label in np.unique(labels)}
    large = np.flatnonzero(np.array([counts[int(l)] > .02 * len(before) for l in labels]))
    tree = KDTree(len(large))
    for index in large:
        tree.insert(before[index], int(index))
    tree.balance()
    components = []
    for label, count in counts.items():
        if count > .02 * len(before):
            continue
        members = np.flatnonzero(labels == label)
        anchor_candidates = []
        for index in members:
            _, near, distance = tree.find(before[index])
            anchor_candidates.append((distance, int(index), near))
        if not anchor_candidates:
            continue
        distance, index, near = min(anchor_candidates)
        new_distance = float(np.linalg.norm(after[index] - after[near]))
        if new_distance - distance < .005:
            continue
        components.append({"label": label, "vertices": count, "centre": before[members].mean(axis=0).tolist(),
                           "bounds_cm": (np.ptp(before[members], axis=0) * 100).tolist(),
                           "gap_before_cm": distance * 100, "gap_after_cm": new_distance * 100,
                           "attachment": [index, near], "weights": [weights(index), weights(near)]})
    edges = np.array([e.vertices[:] for e in mesh.data.edges])
    initial = np.linalg.norm(before[edges[:, 0]] - before[edges[:, 1]], axis=1)
    final = np.linalg.norm(after[edges[:, 0]] - after[edges[:, 1]], axis=1)
    return {"mesh": mesh.name, "vertices": len(before), "faces": len(mesh.data.polygons),
            "shape_keys": len(mesh.data.shape_keys.key_blocks) if mesh.data.shape_keys else 0,
            "pieces": len(counts), "split_seam_vertices": len(seam_indices),
            "worst_seams": [{"vertices": [int(i), int(weld[i])], "position": before[i].tolist(),
                             "gap_cm": float(gaps[i] * 100), "weights": [weights(i), weights(weld[i])]}
                            for i in seam_indices[np.argsort(-gaps[seam_indices])][:25]],
            "floating_components": sorted(components, key=lambda p: -p["gap_after_cm"]),
            "stretched_edges": [{"vertices": edges[i].tolist(), "position": before[edges[i]].mean(axis=0).tolist(),
                                 "growth_cm": float((final[i] - initial[i]) * 100),
                                 "weights": [weights(v) for v in edges[i]]}
                                for i in np.argsort(-(final - initial))[:12]]}


def compare_original(mesh, path):
    """Compare hands to the unrigged input, before any pose/weight operation."""
    from vtuber_maker.pipeline import normalize
    old_objects = set(bpy.data.objects)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = [name for name in archive.namelist() if name.lower().endswith(".obj")]
            if len(candidates) != 1:
                raise ValueError("reference ZIP must have exactly one OBJ")
            vertices, faces = [], []
            for line in archive.read(candidates[0]).decode("utf-8-sig").splitlines():
                fields = line.split()
                if not fields:
                    continue
                if fields[0] == "v":
                    vertices.append(tuple(float(v) for v in fields[1:4]))
                elif fields[0] == "f":
                    indices = [int(v.split("/")[0]) for v in fields[1:]]
                    faces.append(tuple(i - 1 if i > 0 else len(vertices) + i for i in indices))
        data = bpy.data.meshes.new("original_reference")
        data.from_pydata(vertices, [], faces)
        obj = bpy.data.objects.new("original_reference", data)
        bpy.context.scene.collection.objects.link(obj)
    else:
        bpy.ops.import_scene.gltf(filepath=str(path.resolve()))
    reference = normalize.prepare(list(set(bpy.data.objects) - old_objects))
    coords = meshdata.world_vertices(reference)
    tree = KDTree(len(coords))
    for i, point in enumerate(coords):
        tree.insert(point, i)
    tree.balance()
    source = meshdata.world_vertices(mesh)
    result = {"reference": str(path), "vertices": len(coords), "faces": len(reference.data.polygons), "hands": {}}
    for side in ("Left", "Right"):
        groups = {g.index for g in mesh.vertex_groups if g.name == side + "Hand" or
                  any(g.name.startswith(side + f) for f in ("Thumb", "Index", "Middle", "Ring", "Little"))}
        selected = [v.index for v in mesh.data.vertices if sum(g.weight for g in v.groups if g.group in groups) > .5]
        distances = [tree.find(source[i])[2] for i in selected]
        result["hands"][side] = {"vertices": len(selected),
                                "nearest_original_vertex_mm": np.percentile(distances, [50, 95, 100]).tolist() if distances else []}
        result["hands"][side]["nearest_original_vertex_mm"] = [v * 1000 for v in result["hands"][side]["nearest_original_vertex_mm"]]
    bpy.data.objects.remove(reference, do_unlink=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--repair-shoulders", action="store_true")
    parser.add_argument("--max-edge-growth-cm", type=float,
                        help="fail if raising the arms stretches an edge more than this")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
    comparison = compare_original(mesh, args.reference) if args.reference else None
    if args.repair_shoulders:
        from vtuber_maker.pipeline import loose
        print("seam vertices synchronized:", loose.synchronize_shoulder_seams(mesh, arm))
    result = inspect(mesh, arm)
    if comparison:
        result["original_comparison"] = comparison
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.max_edge_growth_cm is not None:
        worst = max(e["growth_cm"] for e in result["stretched_edges"])
        assert worst <= args.max_edge_growth_cm, f"an edge grew {worst:.2f} cm (limit {args.max_edge_growth_cm})"


if __name__ == "__main__":
    main()
