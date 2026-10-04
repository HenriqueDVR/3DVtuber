"""Close small holes in a rigged avatar's surface and export it as VRM.

    blender -b in.blend -P scripts/fix_holes.py -- out.blend out.vrm [--box x0 y0 z0 x1 y1 z1] [--size 0.04] [--list]

A hole is a closed ring of edges with a face on one side only, no larger than --size (m)
across (loose.find_small_holes). Those that are cracks, with border points under --gap (m)
apart that should be one point, are closed by bringing those points together
(loose.close_small_holes): no vertex is added or removed, and shape keys, weights and springs
stand. A hole with real area is left alone and counted as left. --box (world space, as the
.blend stands) keeps to one place; look at --list first. --list only prints the holes and writes nothing. The original file is left alone.

Made for one model whose T-pose bake of 2026-09-21 left a row of sliver holes
across the chest, seen as dark lines in every build since.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import loose, meshdata, vrm_export  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_blend", nargs="?")
    ap.add_argument("out_vrm", nargs="?")
    ap.add_argument("--box", type=float, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"))
    ap.add_argument("--size", type=float, default=loose.HOLE_SIZE)
    ap.add_argument("--gap", type=float, default=loose.HOLE_GAP)
    ap.add_argument("--list", action="store_true", help="print the holes and write nothing")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = max((o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm), key=lambda o: len(o.data.vertices))
    low, high = (args.box[:3], args.box[3:]) if args.box else (None, None)
    verts = meshdata.world_vertices(mesh)
    holes = loose.find_small_holes(mesh, low, high, args.size, verts)
    edges = np.empty(len(mesh.data.edges) * 2, dtype=np.int64)
    mesh.data.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)
    for hole in holes:
        points = verts[np.unique(edges[hole])]
        print(f"hole: {len(hole)} edges, {np.round(np.ptp(points, axis=0) * 1000, 1).tolist()} mm, "
              f"at {np.round(points.mean(axis=0), 3).tolist()}")
    print(f"{len(holes)} small holes")
    if args.list:
        return
    if not (args.out_blend and args.out_vrm):
        sys.exit("fix_holes: give out.blend and out.vrm, or --list")
    moved = loose.close_small_holes(mesh, low, high, args.size, args.gap)
    left = len(loose.find_small_holes(mesh, low, high, args.size))
    print(f"moved {moved} vertices; {left} holes left")
    bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(args.out_blend), copy=True)
    vrm_export.export(Path(bpy.path.abspath(args.out_vrm)), arm)
    print(f"VTUBER_MAKER_REPORT {json.dumps({'holes': len(holes), 'moved': moved, 'left': left})}")


main()
