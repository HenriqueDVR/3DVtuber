"""Add finger bones to a rigged hand for hand tracking, then export VRM.

    blender -b in.blend -P scripts/add_fingers.py -- out.blend out.vrm [--renders dir]

Webcam hand tracking (Warudo, VSeeFace) drives the VRM finger bones; the build rigs the hand as
one bone, so the hand tracks as a paddle. This finds the fingers in the hand's own shape -- the
grooves between them show as lanes across the palm past the knuckles -- puts three bones down
each, and moves the hand's weight onto them (see pipeline/fingers.py). Fused fingers, like a
gloved hand's, work: the webbing between them stretches instead of tearing.

Run before `to_tpose.py`, on the modelled pose. `--renders` writes a close-up of each hand at
rest and curled, which is the quickest way to see whether the lanes landed on real fingers.
The original file is left alone. Needs the VRM Add-on for Blender.
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import fingers, rig, vrm_export  # noqa: E402

# How far each finger bone is curled for the render, in degrees.
CURL = 45.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_blend", type=Path)
    parser.add_argument("out_vrm", type=Path)
    parser.add_argument("--renders", type=Path, help="write close-ups of both hands here")
    parser.add_argument("--curl", type=float, default=CURL, help="how far each finger bone bends in the render")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:])


def main():
    args = parse_args()
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = next(o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers))

    report = fingers.build_fingers(mesh, armature)
    for side, found in sorted(report.items()):
        if not found.get("fingers"):
            print(f"{side} hand: no fingers found ({found.get('reason', 'unknown')})")
            continue
        reach, width, thick = found["hand_cm"]
        print(f"{side} hand: {', '.join(found['named'])}; hand {reach} x {width} x {thick} cm, "
              f"lengths {found['finger_length_cm']} cm")
    if not any(found.get("fingers") for found in report.values()):
        sys.exit("no hand had fingers to rig; nothing written")

    vrm_export.ensure_vrm_addon()
    bound = vrm_export.bind_fingers(armature)
    print(f"bound {bound} finger bones to the VRM humanoid")
    if args.renders:
        render_hands(mesh, armature, report, args.renders, args.curl)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_blend.resolve()), copy=True)
    vrm_export.export(args.out_vrm.resolve(), armature)
    print("wrote", args.out_blend.name, "and", args.out_vrm.name)


def render_hands(mesh, armature, report, out_dir, curl=CURL):
    """Flat-lit close-ups of each rigged hand, at rest and with every finger curled."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "TEXTURE"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x = scene.render.resolution_y = 800
    scene.render.film_transparent = True
    data = bpy.data.cameras.new("hand")
    data.type = "ORTHO"
    camera = bpy.data.objects.new("hand", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    out_dir.mkdir(parents=True, exist_ok=True)

    for side, found in sorted(report.items()):
        if not found.get("fingers"):
            continue
        hand = armature.pose.bones[rig.BONE_NAMES[f"{side}_hand"]]
        reach = max(0.12, 2.5 * (hand.tail - hand.head).length)
        data.ortho_scale = reach
        centre = armature.matrix_world @ hand.tail
        for view, (offset, rotation) in {
            "front": ((0.0, -reach, 0.0), (math.pi / 2, 0.0, 0.0)),
            "side": ((reach, 0.0, 0.0), (math.pi / 2, 0.0, math.pi / 2)),
        }.items():
            camera.location = centre + Vector(offset)
            camera.rotation_euler = rotation
            for pose in ("rest", "curl"):
                _curl(armature, side, curl if pose == "curl" else 0.0)
                scene.render.filepath = str((out_dir / f"{side}_{view}_{pose}.png").resolve())
                bpy.ops.render.render(write_still=True)
        _curl(armature, side, 0.0)
    print("rendered hand close-ups to", out_dir)


def _curl(armature, side, degrees):
    """Bend every finger bone on one hand by the same angle, the way an app curls a finger.

    A bone's own X axis is the hinge: its Y runs along the finger, so turning about X folds the
    finger towards the palm, which is the rotation a hand tracker sends.
    """
    for finger in fingers.FINGERS:
        for segment in fingers._segments(finger):
            bone = armature.pose.bones.get(fingers.bone_name(side, finger, segment))
            if bone is None:
                continue
            bone.rotation_mode = "XYZ"
            bone.rotation_euler = (math.radians(degrees), 0.0, 0.0)
    bpy.context.view_layer.update()


main()
