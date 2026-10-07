"""Take a finished .vrm in as it is: the first step of a build that improves one instead of making one.

    blender -b --python-exit-code 1 -P scripts/import_vrm.py -- in.vrm out.vrm [--blend out.blend]

Reads the file with the VRM Add-on, so its skeleton, weights, expressions, springs, colliders and
toon materials all stay (pipeline/improve.py); the humanoid bones take the pipeline's names. The
.blend is what the later steps (chest bounce, the checks, the export) read. The export writes every
joint unrotated, as VRoid does: a file saved by the VRM Add-on's own menu has Blender's bone frames,
which Warudo's finger tracking twists. VRM 1.0 only. Never overwrites an existing file.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import improve, vrm_export  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Take a finished VRM 1.0 file in as it is.")
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--blend", type=Path, help="the scene the later steps read")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def main():
    args = parse_args()
    for path in (args.output, args.blend):
        if path is not None and path.exists():
            sys.exit(f"{path} exists: choose a new name; files are never overwritten")
    bpy.ops.wm.read_homefile(use_empty=True)
    try:
        armature, meshes = improve.load(args.model.resolve())
    except improve.NotImprovable as error:
        sys.exit(f"VRM NOT TAKEN IN {error}")
    report = improve.holds(armature, meshes)
    report["renamed_bones"] = improve.rename_bones(armature)
    if args.blend:
        # the pictures live in the .vrm: without packing them the .blend would point back into it
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(args.blend.resolve()))
    vrm_export.export(args.output.resolve(), armature)
    print("VTUBER_MAKER_REPORT " + json.dumps(report))


main()
