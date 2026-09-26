"""Hang the tracked mouth from its line and let small openings grow from the middle, then export VRM.

    blender -b in.blend -P scripts/fix_mouth.py -- out.blend out.vrm [--scale 1.3] [--free]
        [--rest 0.25] [--rest-height 0.07] [--lift 0.02] [--jaw 0.008] [--renders dir]

For a blend with the mouth patch, its vowel keys and add_face_tracking.py's ARKit mouth shapes.
The vowels are moved down so the open mouth's top stays at the mouth line (face_rig.reshape_mouth),
--scale makes every opening that much bigger, the closed patch becomes a line --rest of the widest
vowel, and the ARKit mouth shapes are rebuilt
from the new vowels as add_face_tracking.py builds them (--jaw as it was built). Prints how much the
rebuilt shapes differ off the patch from the old ones: that should be nothing. Eyes, blink and
everything else are left alone. The input file is not touched. Needs the VRM Add-on.

--free paints the drawn mouth out of the face patch round the patch's own line and makes the
patch the mouth even when closed: a drawn line --rest-height mouth widths tall and, unless --rest is
given, as wide as the painted mouth was (FREE_REST_WIDTH). Without it the painted lips stay on the
face and an opening appears over them from nowhere (model A, 2026-09-24).
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vtuber_maker.pipeline import face_rig, vrm_export  # noqa: E402
from add_face_tracking import FaceShapes  # noqa: E402

# --free: the closed line's width, in mouth widths, and height (face_rig.MOUTH_REST_SHAPE's)
FREE_REST_WIDTH = 0.8
FREE_REST_HEIGHT = 2 * face_rig.MOUTH_REST_SHAPE[1]
MOUTH_KEYS = ("jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
              "mouthStretchLeft", "mouthStretchRight", "mouthPucker", "mouthFunnel", "mouthLeft", "mouthRight")
# View name -> shape key values: what a tracker sends while talking, and the full shapes.
VIEWS = {"rest": {}, **{f"jaw_{a}": {"jawOpen": a} for a in (0.15, 0.3, 0.5, 0.75, 1.0)},
         **{v: {v: 1.0} for v in face_rig.VOWELS},
         "talk_ou": {"jawOpen": 0.3, "mouthFunnel": 0.3}, "talk_smile": {"jawOpen": 0.4, "mouthSmileLeft": 0.6, "mouthSmileRight": 0.6}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_blend")
    ap.add_argument("out_vrm")
    ap.add_argument("--scale", type=float, default=1.0, help="size of every opening, times its present size")
    ap.add_argument("--free", action="store_true", help="paint out the drawn mouth; the patch draws the closed mouth")
    ap.add_argument("--rest", type=float, help=f"closed line, share of the widest vowel (default {face_rig.MOUTH_REST_SHARE}, "
                    f"or {FREE_REST_WIDTH} mouth widths with --free)")
    ap.add_argument("--rest-height", type=float, help=f"closed line's height with --free, mouth widths (default {FREE_REST_HEIGHT})")
    ap.add_argument("--lift", type=float, default=face_rig.MOUTH_TOP_LIFT, help="open mouth's top above the line, mouth widths")
    ap.add_argument("--jaw", type=float, default=0.008, help="chin drop on jawOpen (m), as add_face_tracking.py was run")
    ap.add_argument("--renders", type=Path, help="write mouth close-ups here, before and after")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])

    vrm_export.ensure_vrm_addon()
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    obj = next(o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm)
    keys = obj.data.shape_keys.key_blocks
    missing = [name for name in MOUTH_KEYS if name not in keys]
    if face_rig.MOUTH_GRID_ATTRIBUTE not in obj.data.attributes or missing:
        raise SystemExit(f"needs the mouth patch and add_face_tracking.py's mouth shapes (missing {missing})")
    renders = args.renders.resolve() if args.renders else None
    if renders:
        render_views(obj, renders / "before")

    patch = mouth_patch(obj)
    old = {name: deltas(obj, name) for name in MOUTH_KEYS}
    rest, rest_height = args.rest, 0.0
    if args.free:
        free = face_rig.use_free_mouth(obj)
        print("painted the drawn mouth out of face patch pixels", free["box"] and [round(v) for v in free["box"]],
              f"; {free['polygons']} polygons in front there took the face patch")
        rest_height = FREE_REST_HEIGHT if args.rest_height is None else args.rest_height
        if rest is None:
            width = obj[face_rig.MOUTH_COVER_PROPERTY]["width"]
            widest = max(np.ptp((deltas(obj, v) + base(obj))[patch, 0]) for v in face_rig.VOWELS if v in keys)
            rest = FREE_REST_WIDTH * width / (args.scale * widest)
    done = face_rig.reshape_mouth(obj, face_rig.MOUTH_REST_SHARE if rest is None else rest, args.lift, args.scale, rest_height)
    for name, (was, now) in done["tops"].items():
        w, h = done["sizes"][name]
        print(f"{name}: top {1000 * was:+.1f} mm above the line, now {1000 * now:+.1f} mm; {1000 * w:.1f} x {1000 * h:.1f} mm")
    print("closed mouth {:.1f} x {:.1f} mm".format(*(1000 * v for v in done["rest_size"])))
    FaceShapes(obj).add_mouth(args.jaw)
    for name in MOUTH_KEYS:
        off = np.abs(deltas(obj, name) - old[name])[~patch].max()
        print(f"{name}: off the patch differs by at most {1000 * off:.3f} mm")

    bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(args.out_blend), copy=True)
    vrm_export.export(Path(bpy.path.abspath(args.out_vrm)), arm)
    print("wrote", args.out_blend, "and", args.out_vrm)
    if renders:
        render_views(obj, renders / "after")
        print("rendered close-ups to", renders)


def mouth_patch(obj):
    grid = np.empty(len(obj.data.vertices) * 3)
    obj.data.attributes[face_rig.MOUTH_GRID_ATTRIBUTE].data.foreach_get("vector", grid)
    return grid.reshape(-1, 3)[:, 2] > 0.5


def base(obj):
    key = obj.data.shape_keys.reference_key
    co = np.empty(len(key.data) * 3)
    key.data.foreach_get("co", co)
    return co.reshape(-1, 3)


def deltas(obj, name):
    keys = obj.data.shape_keys
    values = []
    for key in (keys.key_blocks[name], keys.reference_key):
        co = np.empty(len(key.data) * 3)
        key.data.foreach_get("co", co)
        values.append(co.reshape(-1, 3))
    return values[0] - values[1]


def render_views(obj, out_dir):
    """Flat-lit front close-ups of the mouth with each view's shape keys on."""
    cover = obj[face_rig.MOUTH_COVER_PROPERTY]
    width = cover["width"]
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "TEXTURE"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x = scene.render.resolution_y = 400
    data = bpy.data.cameras.new("mouth")
    data.type = "ORTHO"
    data.ortho_scale = 3.5 * width
    camera = bpy.data.objects.new("mouth", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.location = obj.matrix_world @ Vector((cover["x"], -1.0, cover["z"] + 0.3 * width))
    camera.rotation_euler = (math.pi / 2, 0, 0)
    keys = obj.data.shape_keys.key_blocks
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, values in VIEWS.items():
        for key in keys:
            key.value = values.get(key.name, 0.0)
        scene.render.filepath = str(out_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    for key in keys:
        key.value = 0.0
    bpy.data.objects.remove(camera)


main()
