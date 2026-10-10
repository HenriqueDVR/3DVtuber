"""Make the eyes follow face tracking, then export VRM.

    blender -b in.blend -P scripts/add_eye_tracking.py -- out.blend out.vrm --face face.json [--slide [H V]] [--still-iris] [--lid-travel F] [--renders dir]

By default (eyes with room around the iris, like model A) each painted iris moves onto its own disc,
the face under it has the iris painted out, and a copy of the lids in front keeps it under them.
`--slide` is for a face with no room for an iris (a mascot's ball): the whole painted eye slides a
little over the face instead, H of its width sideways and V of its height up and down (default
eyes.SLIDE). Either way the eyes move on the eight ARKit eyeLook keys, which face trackers drive
directly, bound to the VRM look expressions; in Warudo set eye movement to blendshapes.
Needs face points (viewer/setup.html) on a face patch, and a blend with the face tracking keys
(scripts/add_face_tracking.py): the lids blink with them. Measures the result and exits 1 if an
iris could come through the lids or sink into the face, or a patch into the ball, in any look;
the files are written either way. The input file is left alone. Needs the VRM Add-on.
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import eyes, face_points, face_rig, meshdata, vrm_export  # noqa: E402

# The look-at input angle (degrees) at which the look expressions reach full weight, as VRoid exports.
LOOK_INPUT = 90.0
RANGE_NAMES = ("horizontal_outer", "horizontal_inner", "vertical_up", "vertical_down")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_blend")
    ap.add_argument("out_vrm")
    ap.add_argument("--face", type=Path, required=True, help="face.json from viewer/setup.html")
    ap.add_argument("--slide", type=float, nargs="*", metavar=("H", "V"),
                    help="slide the whole eye instead of moving an iris (a mascot); optional shares of its width and height")
    ap.add_argument("--modelled", action="store_true",
                    help="the irises are modelled discs in their sockets: slide the discs themselves, each with a picture of "
                         "its own (the app tries this when the moving iris is refused)")
    ap.add_argument("--still-iris", action="store_true",
                    help="moving iris: in a blink the lids slide down over a still iris (default: the iris squeezes with the eye)")
    ap.add_argument("--lid-travel", type=float, default=1.0,
                    help="moving iris: how far the lids come down in a blink, times what the build's blink does (with --still-iris: up to about 1.6)")
    ap.add_argument("--renders", type=Path, help="write eye close-ups here, before and after, looking each way")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.slide is not None and len(args.slide) not in (0, 2):
        raise SystemExit("--slide takes no values or two (sideways and up-down shares)")
    renders = args.renders.resolve() if args.renders else None

    vrm_export.ensure_vrm_addon()
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    obj = next(o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm)
    mesh = obj.data
    if eyes.get_int(mesh, eyes.IRIS_ATTRIBUTE).any() or eyes.get_int(mesh, eyes.PATCH_ATTRIBUTE).any():
        raise SystemExit("this blend already has tracked eyes: start from the one before")
    if mesh.shape_keys is None:
        raise SystemExit("no shape keys: run scripts/add_face_tracking.py first")
    points = face_points.load(args.face)
    if renders:
        render_views(obj, points, renders, {"before": {}})

    if args.slide is not None:
        built = eyes.add_sliding_eyes(obj, points, tuple(args.slide) if args.slide else eyes.SLIDE)
        parts = built["patches"]
        numbers, failures = eyes.check_sliding(obj, built)
        report_sliding(built, numbers)
    elif args.modelled:
        if not eyes.modelled_irises(obj, points):
            print("FAILED: the eyes aren't modelled as iris discs narrower than the eyes: nothing to slide")
            sys.exit(1)
        # the blink first, so what is added blinks with the face: the whole marked eye squeezes, as
        # painted eyes do (the lid blink is sized from the insert, here only the iris)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import add_face_tracking
        add_face_tracking.FaceShapes(obj, painted_eyes=True).add_blink("", face_rig.PAINTED_BLINK)
        built = eyes.add_modelled_irises(obj, points)
        if built is None:
            print("FAILED: too little of the iris discs shows between the lids to paint whole irises from")
            sys.exit(1)
        parts = built["eyes"]
        numbers, failures = eyes.check_modelled(obj, built, points)
        report_modelled(built, numbers)
    else:
        built = eyes.add_moving_irises(obj, points, still_iris=args.still_iris)
        if args.lid_travel != 1.0:
            eyes.scale_blinks(obj, args.lid_travel)
        parts = built["eyes"]
        numbers, failures = eyes.check_irises(obj, built)
        report_irises(built, numbers)
    for name, crop in built["crops"].items():
        x0, y0, x1, y1 = crop["box"]
        print(f"{name} image: {x1 - x0} x {y1 - y0} px of the face patch's {crop['full'][0]} x {crop['full'][1]}")
    vrm_export.bind_look_expressions(arm, obj)
    rest = meshdata.world_vertices(obj)
    origin = np.mean([rest[info["vertices"]].mean(axis=0) for info in parts.values()], axis=0)
    vrm_export.set_look_at(arm, "expression", {name: (LOOK_INPUT, 1.0) for name in RANGE_NAMES}, origin=origin)

    bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(args.out_blend), copy=True)
    vrm_export.export(Path(bpy.path.abspath(args.out_vrm)), arm)
    print("wrote", args.out_blend, "and", args.out_vrm)

    if renders:
        render_views(obj, points, renders, VIEWS)
        print("rendered close-ups to", renders)
        compare(renders / "before.png", renders / "rest.png")
    for failure in failures:
        print("FAILED:", failure)
    if failures:
        sys.exit(1)


def report_modelled(built, numbers):
    for side, info in built["eyes"].items():
        w, h = info["opening_size"]
        iw, ih = info["iris_size"]
        across, up = info["slide"]
        dw, dh = info["disc_size"]
        print(f"{side} eye: marked {1000 * w:.1f} x {1000 * h:.1f} mm, iris {1000 * iw:.1f} mm across on a disc {1000 * dw:.1f} x {1000 * dh:.1f} mm, "
              f"room beside it {1000 * info['white_beside'][0]:.1f} and {1000 * info['white_beside'][1]:.1f} mm, "
              f"slides {1000 * across:.1f} mm sideways, {1000 * up:.1f} mm up/down, {len(info['vertices'])} iris vertices")
    shown = ", ".join(f"{side} {info['shown_px'] / max(info['outline_px'], 1):.0%}" for side, info in built["eyes"].items())
    print(f"each iris painted whole from the part that shows ({shown} of its disc), {built['repainted_polygons']} polygons "
          f"round them moved onto the face patch; "
          f"checked {numbers['points']} iris points over every look and blink: most showing outside the eye in one look "
          f"{numbers['outside']}, fewest showing at all {numbers['shown_least']}")


def report_irises(built, numbers):
    for side, info in built["eyes"].items():
        w, h = info["opening_size"]
        iw, ih = info["iris_size"]
        across, up = info["slide"]
        print(f"{side} eye: opening {1000 * w:.1f} x {1000 * h:.1f} mm, iris {1000 * iw:.1f} x {1000 * ih:.1f} mm, "
              f"white beside it {1000 * info['white_inner']:.1f} mm in / {1000 * info['white_outer']:.1f} mm out, "
              f"slides {1000 * across:.1f} mm sideways, {1000 * up:.1f} mm up/down, {len(info['vertices'])} iris vertices")
    spreads = ", ".join(f"{side} {1000 * info['track_spread']:.1f} mm" for side, info in built["eyes"].items())
    print(f"lid overlay: {len(built['overlay_vertices'])} vertices; the face's spread about the iris track: {spreads}")
    print(f"checked {numbers['points']} iris points over every look and blink: most showing over the lids in one "
          f"look {numbers['over_lids']}, most hidden by the face in the window {numbers['sunk']}")


def report_sliding(built, numbers):
    for side, info in built["patches"].items():
        across, up = info["slide"]
        print(f"{side} eye: paint {info['paint_mm2']:.0f} mm2, slides {1000 * across:.1f} mm sideways, "
              f"{1000 * up:.1f} mm up/down, blink squeezes {built['blink'][side]:.0%}, {len(info['vertices'])} vertices")
    print(f"{built['repainted_polygons']} polygons under the eyes moved onto the face patch; checked {numbers['points']} "
          f"painted patch points over every look and blink: most hidden by the face in one look {numbers['hidden']}")


# View name -> shape key values. The character's left is the viewer's right.
VIEWS = {
    "rest": {},
    "look_left": {"eyeLookOutLeft": 1.0, "eyeLookInRight": 1.0},
    "look_right": {"eyeLookInLeft": 1.0, "eyeLookOutRight": 1.0},
    "look_up": {"eyeLookUpLeft": 1.0, "eyeLookUpRight": 1.0},
    "look_down": {"eyeLookDownLeft": 1.0, "eyeLookDownRight": 1.0},
    "look_up_left": {"eyeLookOutLeft": 1.0, "eyeLookInRight": 1.0, "eyeLookUpLeft": 1.0, "eyeLookUpRight": 1.0},
    "blink": {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0},
    "half_blink_look_down": {"eyeBlinkLeft": 0.5, "eyeBlinkRight": 0.5, "eyeLookDownLeft": 1.0, "eyeLookDownRight": 1.0},
}


def eye_camera(obj, points):
    """An orthographic camera straight in front of the eyes, framing both."""
    face = eyes.FacePatch(obj)
    centres = []
    for eye in eyes.marked_eyes(points, face.size).values():
        p = eye.points
        centres.append(face.world([[(p["inner"][0] + p["outer"][0]) / 2, (p["top"][1] + p["bottom"][1]) / 2]])[0])
    (lx, lz), (rx, rz) = centres
    data = bpy.data.cameras.new("eyes")
    data.type = "ORTHO"
    data.ortho_scale = 2.0 * abs(lx - rx)
    camera = bpy.data.objects.new("eyes", data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = Vector(((lx + rx) / 2, -1.0, (lz + rz) / 2))
    camera.rotation_euler = (math.pi / 2, 0, 0)
    return camera


def render_views(obj, points, out_dir, named):
    """Flat-lit renders of the eyes with each view's shape keys on; the camera is removed after."""
    camera = eye_camera(obj, points)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "TEXTURE"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x, scene.render.resolution_y = 1400, 700
    scene.camera = camera
    keys = obj.data.shape_keys.key_blocks
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, values in named.items():
        for key in keys:
            key.value = values.get(key.name, 0.0)
        scene.render.filepath = str(out_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    for key in keys:
        key.value = 0.0
    bpy.data.objects.remove(camera)


def compare(before, after):
    """Print how much the rest view changed: it should look as it did."""
    a, b = (bpy.data.images.load(str(path)) for path in (before, after))
    pa, pb = (np.array(image.pixels[:]).reshape(-1, 4)[:, :3] for image in (a, b))
    difference = np.abs(pa - pb).max(axis=1)
    print(f"rest vs before: mean difference {difference.mean():.4f}, pixels off by more than 10%: "
          f"{(difference > 0.1).mean():.2%}")


main()
