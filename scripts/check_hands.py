"""Measure finger rest/bake/curl deformation without saving changes to the input blend.

blender -b model.blend --python-exit-code 1 -P scripts/check_hands.py -- [--bake] [--json report.json]
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
from vtuber_maker.pipeline import fingers, rig, tpose


def coordinates(mesh):
    graph = bpy.context.evaluated_depsgraph_get()
    graph.update()
    evaluated = mesh.evaluated_get(graph)
    data = evaluated.to_mesh()
    result = np.array([mesh.matrix_world @ v.co for v in data.vertices])
    evaluated.to_mesh_clear()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bake", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and any(
        m.type == "ARMATURE" and m.object == arm for m in o.modifiers)]
    mesh = max(meshes, key=lambda o: len(o.data.vertices))
    before = coordinates(mesh)
    edges = np.array([e.vertices[:] for e in mesh.data.edges])
    lengths = np.linalg.norm(before[edges[:, 0]] - before[edges[:, 1]], axis=1)
    regions = {}
    for side in ("left", "right"):
        names = {rig.BONE_NAMES[f"{side}_hand"]} | {
            fingers.bone_name(side, finger, segment)
            for finger in fingers.FINGERS for segment in fingers._segments(finger)}
        indices = {g.index for g in mesh.vertex_groups if g.name in names}
        regions[side] = np.array([sum(g.weight for g in v.groups if g.group in indices) > 0.5
                                  for v in mesh.data.vertices])
    report = {"file": bpy.data.filepath, "hands": {}}
    if args.bake:
        report["arms_turned"] = tpose.to_t_pose(mesh, arm, extra_meshes=[o for o in meshes if o != mesh])
    rest = coordinates(mesh)
    rest_lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    for side, region in regions.items():
        selected = region[edges[:, 0]] & region[edges[:, 1]]
        hand = arm.pose.bones[rig.BONE_NAMES[f"{side}_hand"]]
        along = (hand.tail - hand.head).normalized()
        across = tpose._across_palm(arm, side, along)
        normal = along.cross(across).normalized() if across is not None else None
        bones = []
        for finger in fingers.FINGERS:
            for segment in fingers._segments(finger):
                b = arm.pose.bones.get(fingers.bone_name(side, finger, segment))
                if b is not None:
                    bones.append({"name": b.name, "head": list(b.head), "tail": list(b.tail),
                                  "length_cm": b.length * 100,
                                  "hinge": list(b.matrix.to_3x3().col[0]),
                                  "palm_normal": list(normal) if normal else None})
        chain_angles = []
        for finger in fingers.FINGERS:
            chain = [arm.pose.bones.get(fingers.bone_name(side, finger, s)) for s in fingers._segments(finger)]
            if all(b is not None for b in chain):
                first = (chain[0].tail - chain[0].head).normalized()
                chain_angles.extend(math.degrees(first.angle((b.tail - b.head).normalized())) for b in chain[1:])
        entry = {"vertices": int(region.sum()), "bones": bones,
                 "chain_bend_degrees": max(chain_angles, default=0),
                 "hinge_out_of_palm": max((abs(np.dot(b["hinge"], normal)) for b in bones), default=0) if normal else None,
                 "bake_edge_growth_cm": float((rest_lengths - lengths)[selected].max() * 100) if selected.any() else None,
                 "curls": {}}
        for degrees in (20, 45):
            for b in bones:
                bone = arm.pose.bones[b["name"]]
                bone.rotation_mode = "XYZ"
                bone.rotation_euler = (math.radians(degrees), 0, 0)
            bpy.context.view_layer.update()
            posed = coordinates(mesh)
            posed_lengths = np.linalg.norm(posed[edges[:, 0]] - posed[edges[:, 1]], axis=1)
            growth = (posed_lengths - rest_lengths)[selected]
            entry["curls"][degrees] = {"max_edge_growth_cm": float(growth.max() * 100),
                                       "edges_growing_over_1cm": int((growth > .01).sum())}
            for b in bones:
                arm.pose.bones[b["name"]].matrix_basis = Matrix.Identity(4)
            bpy.context.view_layer.update()
        report["hands"][side] = entry
    for side, entry in report["hands"].items():
        print(side, json.dumps({k: v for k, v in entry.items() if k != "bones"}))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
