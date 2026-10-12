"""Model in, finished VRM out: survey, build, fix, check, in one command.

    python scripts/make_avatar.py my_oc.glb [--face face.json] [--out-dir out] [options]

Runs, each in its own Blender:

1. `preflight.py`   -- measures the model and picks the build options it asks for
2. `make_vrm.py`    -- rigs it, with those options plus anything you passed here
3. `fix_hair_weights.py` -- fills unweighted vertices, reattaches hair ties and strands
4. `add_ahoge_spring.py` -- springs the strand standing up on the head, when there is one
5. `add_face_tracking.py` -- ARKit shapes and a partial blink, when face points were given
6. `fix_loose_pieces.py` -- puts straps and panels torn between two limbs on one bone
7. `add_fingers.py`  -- finger bones for hand tracking, found in the hand's own surface
8. `to_tpose.py`     -- swings the arms out to the T-pose VRM wants, fingers straight
9. `check_avatar.py` -- poses the result and measures what moved wrong

Every step reads the last step's `.blend` and writes a new one, so nothing is overwritten
and any step can be rerun by hand afterwards. The last lines say what was built, what the
model asked for, and whether the check found anything worth opening Blender for.

This is a plain Python script: run it with python, not inside Blender.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
# Where Blender usually is on Windows, when it isn't on PATH or in $BLENDER.
WINDOWS_BLENDER = Path(r"C:\Program Files\Blender Foundation")
# Flags preflight can decide on its own, as (survey option, make_vrm flag).
DETECTED = (("rigid_head", "--rigid-head"), ("cloth_physics", "--cloth"))
# Under this many triangles a face has too few vertices to bend without denting.
COARSE_FACE = 3_000


def find_blender(given=None):
    for candidate in (given, os.environ.get("BLENDER"), shutil.which("blender")):
        if candidate and Path(candidate).exists():
            return str(candidate)
    found = sorted(WINDOWS_BLENDER.glob("*/blender.exe")) if WINDOWS_BLENDER.exists() else []
    if found:
        return str(found[-1])
    sys.exit("Blender not found: put it on PATH, set $BLENDER, or pass --blender")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out-dir", type=Path, default=REPO / "out")
    parser.add_argument("--name", help="avatar name (defaults to the model file name)")
    parser.add_argument("--author", default="Unknown")
    parser.add_argument("--face", type=Path, help="face.json from viewer/setup.html")
    parser.add_argument("--joints", type=Path, help="joints.json, when detection needs correcting")
    parser.add_argument("--blender", help="path to the Blender executable")
    parser.add_argument("--keep-eyes", action="store_true", default=True,
                        help="leave the eyes as modelled (the default; blink comes from face tracking)")
    parser.add_argument("--blink-lids", dest="keep_eyes", action="store_false",
                        help="cut eyelids into the mesh instead: only works on a fused, painted face")
    parser.add_argument("--extra", default="", help="anything else to pass make_vrm.py, quoted")
    parser.add_argument("--no-fixes", action="store_true", help="stop after the build")
    parser.add_argument("--no-fingers", action="store_true",
                        help="leave the hands as one bone each instead of rigging fingers")
    parser.add_argument("--no-tpose", action="store_true",
                        help="keep the pose as modelled instead of the T-pose VRM asks for")
    parser.add_argument("--no-check", action="store_true", help="skip the final measurement pass")
    return parser.parse_args()


def main():
    args = parse_args()
    blender = find_blender(args.blender)
    model = args.model.resolve()
    if not model.exists():
        sys.exit(f"no such model: {model}")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or model.stem
    steps = []

    survey = run_preflight(blender, model, out, name)
    flags = [flag for key, flag in DETECTED if survey["options"].get(key)]
    if survey["options"].get("mouth_color") == "skin":
        flags += ["--mouth-colour", "skin"]

    blend = out / f"{name}.blend"
    vrm = out / f"{name}.vrm"
    build = [str(model), str(vrm), "--name", name, "--author", args.author, "--blend", str(blend)]
    if args.keep_eyes:
        build.append("--keep-eyes")
    if args.face:
        build += ["--face", str(args.face.resolve())]
    if args.joints:
        build += ["--joints", str(args.joints.resolve())]
    if args.face and _mouth_only(survey) and survey.get("face_patch"):
        # a coarse or solid face dents when ARKit shapes bend it, so the mouth moves on its own
        # over a still face; that needs the drawn mouth painted out of the texture at build time
        build.append("--free-mouth")
    build += flags + (args.extra.split() if args.extra else [])
    report = run_build(blender, build)
    steps.append(("build", blend))

    if not args.no_fixes:
        blend = run_fix(blender, "fix_hair_weights.py", blend, out / f"{name}_hairfix.blend", steps, "hair weights")
        if not survey["options"].get("rigid_head"):
            blend = run_fix(blender, "add_ahoge_spring.py", blend, out / f"{name}_ahoge.blend", steps, "ahoge spring",
                            optional=True)
        if args.face:
            blend = run_fix(blender, "add_face_tracking.py", blend, out / f"{name}_tracking.blend", steps,
                            "face tracking", extra=["--mouth-only"] if _mouth_only(survey) else [])
        blend = run_fix(blender, "fix_loose_pieces.py", blend, out / f"{name}_loose.blend", steps,
                        "loose pieces", optional=True)
        if not args.no_fingers:
            blend = run_fix(blender, "add_fingers.py", blend, out / f"{name}_fingers.blend", steps, "fingers",
                            optional=True)
        if not args.no_tpose:
            blend = run_fix(blender, "to_tpose.py", blend, out / f"{name}_tpose.blend", steps, "t-pose")
        vrm = blend.with_suffix(".vrm")

    check = {} if args.no_check else run_check(blender, blend, out / f"{name}_check.json")
    summarise(name, model, survey, report, steps, blend, vrm, check)


def _mouth_only(survey):
    """Whether the face is too coarse or too solid to bend: only the mouth patch should move."""
    return survey["triangles"] < COARSE_FACE or bool(survey["options"].get("rigid_head"))


def run(blender, blend_in, script, script_args, label):
    """Run one Blender step; returns its stdout, or None when the step failed."""
    command = [blender, "-b"]
    if blend_in:
        command.append(str(blend_in))
    command += ["-P", str(SCRIPTS / script), "--"] + [str(a) for a in script_args]
    print(f"--- {label}")
    # Blender's Python writes UTF-8 whatever the system's code page: read as that, a path with a letter
    # the code page lacks no longer costs the step its whole output
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:], file=sys.stderr)
        return None
    return result.stdout


def run_preflight(blender, model, out, name):
    output = run(blender, None, "preflight.py", [model, "--json", out / f"{name}_survey.json"], "surveying the model")
    if output is None:
        sys.exit("preflight failed; the model can't be read")
    print(_between(output, "MODEL", "VTUBER_MAKER_SURVEY"))
    return json.loads(_tagged(output, "VTUBER_MAKER_SURVEY"))


def run_build(blender, build_args):
    output = run(blender, None, "make_vrm.py", build_args, "rigging")
    if output is None:
        sys.exit("the build failed")
    return json.loads(_tagged(output, "VTUBER_MAKER_REPORT"))


def run_fix(blender, script, blend_in, blend_out, steps, label, extra=(), optional=False):
    """Run a fix script on a .blend; returns the new .blend, or the old one when it didn't apply."""
    output = run(blender, blend_in, script, [blend_out, blend_out.with_suffix(".vrm")] + list(extra), label)
    if output is None or not blend_out.exists():
        print(f"    {label}: nothing to do" if optional else f"    {label}: FAILED, carrying on with the last build")
        return blend_in
    steps.append((label, blend_out))
    return blend_out


def run_check(blender, blend, json_path):
    output = run(blender, blend, "check_avatar.py", ["--json", json_path], "checking")
    if output is None:
        return {}
    for line in output.splitlines():
        if line.startswith("CHECK FAIL") or line.startswith("CHECK VERDICT"):
            print(f"    {line[6:]}")
    return json.loads(json_path.read_text()) if json_path.exists() else {}


def summarise(name, model, survey, report, steps, blend, vrm, check):
    print()
    print(f"AVATAR {name} from {model.name}")
    for label, path in steps:
        print(f"  {label:14s} {path.name}")
    print(f"  vrm            {vrm.name}")
    print(f"  rig            {report['hair_chains']} hair chains, {report['cloth_chains']} cloth chains, "
          f"{'eye inserts' if report['eye_inserts'] else 'painted face'}"
          f"{', face points' if report.get('face_points') else ''}")
    if survey["notes"]:
        print("  worth knowing")
        for note in survey["notes"]:
            print(f"    - {note}")
    if check:
        failed = check.get("failed", [])
        print("  check          " + ("passed: nothing over the limits" if not failed else "found problems"))
        for line in failed:
            print(f"    - {line}")
        if failed:
            print(f"    Open {blend.name} in Blender and fix the weights by hand.")
    print()
    print(f"Load {vrm.name} in the viewer:  python -m http.server 8321 --bind 127.0.0.1")
    print(f"  http://localhost:8321/viewer/index.html?model=/out/{vrm.name}")


def _tagged(output, tag):
    for line in output.splitlines():
        if line.startswith(tag):
            return line[len(tag):].strip()
    sys.exit(f"no {tag} line in the output; the step didn't finish")


def _between(output, start, end):
    match = re.search(rf"^{start}.*?(?=^{end})", output, re.S | re.M)
    return match.group(0).rstrip() if match else ""


main()
