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

from vtuber_maker.pipeline import build  # noqa: E402

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
    parser.add_argument("--closed-eyes", type=Path, help="closed-eyes version of the face patch texture, for blink lids")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])


def main():
    args = parse_args()
    model = args.model.resolve()
    importer = IMPORTERS.get(model.suffix.lower())
    if importer is None:
        sys.exit(f"Unsupported model type {model.suffix}; use one of {', '.join(IMPORTERS)}")

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
    )
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.blend.resolve()))
    print("VTUBER_MAKER_REPORT " + json.dumps(report))


main()
