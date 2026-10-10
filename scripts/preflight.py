"""Look a model over before rigging it, and print the build command it asks for.

    blender -b -P scripts/preflight.py -- <model.glb|.gltf|.fbx|.obj> [--json report.json]

Reports how the model is standing, what it is made of, and which build options fit it:
a solid head under a hat, cloth hanging below the hips, legs a robe hides, whether the
generator left eye inserts to blink with. Nothing is changed or written unless --json
is given. The last line is a ready-made make_vrm.py command.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "addon"))

from vtuber_maker.pipeline import face_points, landmarks, normalize, survey  # noqa: E402

IMPORTERS = {
    ".glb": lambda path: bpy.ops.import_scene.gltf(filepath=path),
    ".gltf": lambda path: bpy.ops.import_scene.gltf(filepath=path),
    ".fbx": lambda path: bpy.ops.import_scene.fbx(filepath=path),
    ".obj": lambda path: bpy.ops.wm.obj_import(filepath=path),
}
# Options that are worth putting on the command line as flags.
FLAGS = {"rigid_head": "--rigid-head", "hat_tip": "--hat-tip", "cloth_physics": "--cloth"}


def parse_args():
    parser = argparse.ArgumentParser(description="Survey a character model before rigging it.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--json", type=Path, help="also write the whole survey here")
    parser.add_argument("--height", type=float, default=1.6, help="avatar height in metres")
    parser.add_argument("--face", type=Path, help="face.json, once the eyes are marked: the eye inserts are found at the marks")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])


def main():
    args = parse_args()
    model = args.model.resolve()
    importer = IMPORTERS.get(model.suffix.lower())
    if importer is None:
        sys.exit(f"Unsupported model type {model.suffix}; use one of {', '.join(IMPORTERS)}")

    bpy.ops.wm.read_homefile(use_empty=True)
    importer(str(model))
    mesh = normalize.prepare(list(bpy.context.scene.objects), height=args.height)
    orientation = normalize.orientation_of(mesh)
    try:
        report = survey.survey(mesh, marked=face_points.load(args.face) if args.face else None)
    except landmarks.LandmarkError as error:
        # the size too: a model that fails here never reaches the survey's notes, and a tester's
        # was Tripo's high-poly export, twenty times what any build here has been made from
        triangles = sum(len(p.vertices) - 2 for p in mesh.data.polygons)
        heavy = (f"; and {triangles:,} triangles is heavy: export the low-poly version, about 50k"
                 if triangles > survey.HEAVY_TRIANGLES else "")
        arms = landmarks.ARMS_AT_SIDES in str(error)
        print()
        print(f"MODEL {model.name}")
        print(f"  standing   {_orientation_line(orientation)}")
        print(f"  mesh       {triangles:,} triangles")
        print(f"  This doesn't read as a standing humanoid: {error}{heavy}")
        if arms:
            print("  Generate the model in a T-pose (arms straight out to the sides): the arms have to")
            print("  show apart from the body for their joints to be found.")
        else:
            print("  Check it stands upright and apart in the viewer, that it is one character, and")
            print("  that it isn't posed. Nothing else here can run until joint detection works.")
        print("VTUBER_MAKER_ERROR " + json.dumps({"code": "VM-ARMS-AT-SIDES" if arms else "VM-NOT-HUMANOID", "detail": f"{error}{heavy}"}))
        sys.exit(2)
    report["orientation"] = orientation
    report["model"] = str(model)

    print()
    print(f"MODEL {model.name}")
    print(f"  standing   {_orientation_line(orientation)}")
    print(f"  size       {report['size'][0]:.2f} x {report['size'][1]:.2f} x {report['size'][2]:.2f} m")
    print(f"  mesh       {report['triangles']:,} triangles, {report['vertices']:,} vertices, "
          f"{report['pieces']} pieces, texture {_texture_line(report['texture'])}")
    print(f"  pose       {report['pose']['kind']}, arms {report['pose']['arm_drop_degrees']:.0f}° below "
          f"horizontal, symmetry {report['symmetry']:.2f}")
    print(f"  head       {_head_line(report['head'])}")
    print(f"  face       {_face_line(report['parts'])}")
    print(f"  below hips {report['cloth']['share'] * 100:.0f}% of the surface hangs clear of the legs")
    print()
    if report["notes"]:
        print("NOTES")
        for note in report["notes"]:
            print(f"  - {note}")
        print()
    print("BUILD")
    print(f"  {_command(model, report)}")
    if not report["parts"]["read"] or not report["parts"].get("eye_inserts"):
        print("  Mark the eyes and mouth first (viewer/setup.html) and add --face face.json.")
    print()

    if args.json:
        args.json.resolve().write_text(json.dumps(report, indent=2))
        print(f"survey written to {args.json}")
    print("VTUBER_MAKER_SURVEY " + json.dumps(report))


def _orientation_line(orientation):
    if not orientation:
        return "unknown"
    line = orientation["describe"] if "describe" in orientation else (
        f"up {_axis(orientation['up'])}, facing {_axis(orientation['facing'])}")
    if orientation.get("turned"):
        line += " (turned upright for the build)"
    if not (orientation.get("up_sure") and orientation.get("facing_sure")):
        line += " -- UNSURE: check the model stands up and faces you in the viewer"
    return line


def _axis(vector):
    index = max(range(3), key=lambda i: abs(vector[i]))
    return f"{'+' if vector[index] > 0 else '-'}{'XYZ'[index]}"


def _texture_line(texture):
    return f"{texture[0]}x{texture[1]}" if texture else "none"


def _head_line(head):
    if head["hair_share"] is None:
        return f"{head['height'] * 100:.0f} cm tall (no parts to read)"
    return (f"{head['height'] * 100:.0f} cm tall, hair hangs {head['hair_hang']:.1f} head heights below it, "
            f"tip {head['tip_ratio']:.2f} of its widest")


def _face_line(parts):
    if not parts["read"]:
        return "one fused mesh: the face is read from the painted texture"
    bits = ["eye inserts" if parts["eye_inserts"] else "no eye inserts",
            "mouth insert" if parts["mouth_insert"] else "no mouth insert"]
    return ", ".join(bits)


def _command(model, report):
    options = report["options"]
    flags = [flag for key, flag in FLAGS.items() if options.get(key)]
    if options.get("mouth_color") == "skin":
        flags.append("--mouth-colour skin")
    if report["triangles"] < survey.COARSE_TRIANGLES:
        flags.append("--free-mouth")
    name = model.stem
    return (f"blender -b -P scripts/make_vrm.py -- {model.name} out/{name}.vrm "
            f"--blend out/{name}.blend {' '.join(flags)}".rstrip())


main()
