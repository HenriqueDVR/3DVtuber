"""Make a VRM avatar from an image-to-3D character model.

    blender -b -P scripts/make_vrm.py -- <model.glb|.gltf|.fbx|.obj> <output.vrm> [--name NAME] [--author AUTHOR] [--height METRES]

Needs the VRM Add-on for Blender installed, so don't pass --factory-startup.
The model should be a standing A-pose character facing the front view.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "addon"))

from vtuber_maker.pipeline import build, manifest  # noqa: E402

IMPORTERS = {
    ".glb": lambda path: bpy.ops.import_scene.gltf(filepath=path),
    ".gltf": lambda path: bpy.ops.import_scene.gltf(filepath=path),
    ".fbx": lambda path: bpy.ops.import_scene.fbx(filepath=path),
    ".obj": lambda path: bpy.ops.wm.obj_import(filepath=path),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Turn an image-to-3D character into a VRM avatar.")
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", help="avatar name (defaults to the model file name)")
    parser.add_argument("--author", default="Unknown")
    parser.add_argument("--height", type=float, default=1.6, help="avatar height in metres")
    parser.add_argument("--blend", type=Path, help="also save the rigged scene here, to fix weights by hand and re-export")
    parser.add_argument("--keep-eyes", action="store_true", help="leave the eyes as modelled: no blink")
    parser.add_argument("--face", type=Path, help="face.json with eye and mouth points, from viewer/setup.html")
    parser.add_argument("--joints", type=Path,
                        help="joints.json of {joint: [x, y, z]} in the scaled avatar's space, replacing detected joints")
    parser.add_argument("--rigid-head", action="store_true",
                        help="hat, helmet or solid head instead of hair: no hair chains, head pieces rigid")
    parser.add_argument("--cloth", action="store_true", help="spring chains for hanging capes, coat tails and skirts")
    parser.add_argument("--hat-tip", action="store_true", help="with --rigid-head: a spring chain for a pointed hat's tip")
    parser.add_argument("--hair-head-only", action="store_true",
                        help="hair springs collide with the head alone, not the body (hair lying on the shoulders)")
    parser.add_argument("--no-sleeve-springs", action="store_true",
                        help="sleeves hanging deep under the arms stay on the arms, with no chains")
    parser.add_argument("--free-mouth", action="store_true",
                        help="with --face: paint the drawn mouth out and make the mouth its own shape, "
                             "so face tracking can slide and curl it")
    parser.add_argument("--mouth-colour",
                        help='tint the inside of the mouth: "skin" (the face colour) or a hex colour like "#3a7bd5"')
    parser.add_argument("--skinning", choices=("heat", "template", "voxel"), default="heat",
                        help="bone heat and rules, weights from a fitted template body (template_skin.py), "
                             "or geodesic voxel weights and the same rules (voxel_skin.py)")
    parser.add_argument("--manifest", type=Path,
                        help="character manifest: its hair, hat, swing_cloth, appendage and body pieces override detection")
    parser.add_argument("--closed-eyes", type=Path, help="closed-eyes version of the face patch texture, for blink lids")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])


def _colour(value):
    """None, "skin", or a hex colour as sRGB floats."""
    if value is None or value == "skin":
        return value
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))


def main():
    args = parse_args()
    model = args.model.resolve()
    importer = IMPORTERS.get(model.suffix.lower())
    if importer is None:
        sys.exit(f"Unsupported model type {model.suffix}; use one of {', '.join(IMPORTERS)}")

    data = manifest.load(args.manifest) if args.manifest else None
    bpy.ops.wm.read_homefile(use_empty=True)
    importer(str(model))
    report = build.build_avatar(
        list(bpy.context.scene.objects),
        args.output.resolve(),
        name=args.name or model.stem,
        author=args.author,
        height=args.height,
        closed_eyes_image=args.closed_eyes.resolve() if args.closed_eyes else None,
        face_points_path=args.face.resolve() if args.face else None,
        keep_eyes=args.keep_eyes,
        joints_override=json.loads(args.joints.read_text()) if args.joints else None,
        rigid_head=args.rigid_head,
        cloth_physics=args.cloth,
        hat_tip=args.hat_tip,
        mouth_color=_colour(args.mouth_colour),
        free_mouth=args.free_mouth,
        pieces=manifest.pieces_for(data, "build") if data else None,
        pieces_reference=manifest.path_of(data, "reference") if data else None,
        rig_path=manifest.path_of(data, "rig") if data else None,
        skinning=args.skinning,
        hair_hits_body=not args.hair_head_only,
        sleeve_springs=not args.no_sleeve_springs,
        still_fingers=bool(data) and (data.get("options") or {}).get("fingers") == "none",
    )
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.blend.resolve()))
    for warning in report.get("warnings", []):
        print(f"BUILD WARNING {warning}")
    print("VTUBER_MAKER_REPORT " + json.dumps(report))


main()
