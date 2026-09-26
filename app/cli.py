"""Drive a character build from the command line, or from an agent.

    python -m app <command> ...        (python -m app --help for the list)

Every command prints one JSON object on stdout; progress of long runs goes to stderr as JSON
lines. The exit code is 1 only when the command itself failed ({"error": ...}); a build whose
checks found problems still exits 0 -- read `check_failed`.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from . import engine
from .engine import REPO, Project, StepFailed, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m app", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blender", help="path to blender.exe (default: $BLENDER, PATH, Program Files)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="start a character manifest for a model")
    p.add_argument("model", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--author", default="Unknown")
    p.add_argument("--manifest", type=Path, help="where to write it (default out/<name>/character.json)")
    p.add_argument("--face", type=Path, help="face.json from viewer/setup.html")
    p.add_argument("--reference", type=Path, help="the unmodified model, when the source had its texture edited")
    p.add_argument("--rig", type=Path, help="the same model rigged by Tripo (its FBX): its skeleton and weights are used")

    p = sub.add_parser("status", help="options, plan and what each step produced")
    p.add_argument("manifest", type=Path)

    p = sub.add_parser("run", help="run the build chain, or part of it")
    p.add_argument("manifest", type=Path)
    p.add_argument("--from", dest="start", choices=engine.NAMES)
    p.add_argument("--to", dest="stop", choices=engine.NAMES)
    p.add_argument("--only", choices=engine.NAMES)

    p = sub.add_parser("set", help="set an option (value read as JSON when it parses, else text)")
    p.add_argument("manifest", type=Path)
    p.add_argument("key", choices=sorted(manifest.OPTIONS))
    p.add_argument("value")

    p = sub.add_parser("joint", help="override a detected joint, or --clear it")
    p.add_argument("manifest", type=Path)
    p.add_argument("key", choices=manifest.JOINTS)
    p.add_argument("xyz", type=float, nargs="*")
    p.add_argument("--clear", action="store_true")

    p = sub.add_parser("pieces", help="list the source's pieces with suggested roles (runs Blender)")
    p.add_argument("manifest", type=Path)

    p = sub.add_parser("piece", help="name a piece and give it a role, or --remove it")
    p.add_argument("manifest", type=Path)
    p.add_argument("name")
    p.add_argument("--role", choices=manifest.ROLES)
    p.add_argument("--point", type=float, nargs=3)
    p.add_argument("--count", type=int)
    p.add_argument("--label", type=int, help="take point and count from this label of the last `pieces` listing")
    p.add_argument("--bone", help="for follow_bone, e.g. LeftUpperArm; for appendage, what it grows from (default: nearest)")
    p.add_argument("--anchor", help="for detail: the piece it rides")
    p.add_argument("--tip", action="store_true", help="for hat: swing its pointed tip")
    p.add_argument("--remove", action="store_true")
    p.add_argument("--region", type=Path, help="a JSON list of source vertex numbers: only this part of the piece")
    p.add_argument("--rename", metavar="NEW", help="give the named piece a new name (details anchored to it follow)")

    p = sub.add_parser("render", help="render the latest (or a step's) rig in a pose; returns the PNGs")
    p.add_argument("manifest", type=Path)
    p.add_argument("--pose", default="raised")
    p.add_argument("--step", choices=engine.NAMES, help="render this step's output instead of the latest")
    p.add_argument("--run", help="render the last .blend of this run folder (a name from `runs`) instead")
    p.add_argument("--size", type=int, default=800)

    p = sub.add_parser("movement", help="which pieces stay behind or tear when the arms and head move (runs Blender)")
    p.add_argument("manifest", type=Path)
    p.add_argument("--step", choices=engine.NAMES, help="check this step's output (default: the last before the T-pose)")

    p = sub.add_parser("springs", help="run the spring bones as apps do: what droops, floats off or sinks in (runs Blender)")
    p.add_argument("manifest", type=Path)
    p.add_argument("--step", choices=engine.NAMES, help="check this step's output (default: the latest)")

    p = sub.add_parser("face-patch", help="give the model a front-mapped face texture to mark points on (runs Blender)")
    p.add_argument("manifest", type=Path)

    p = sub.add_parser("face", help="save face points (a face.json from viewer/setup.html) into the character")
    p.add_argument("manifest", type=Path)
    p.add_argument("points", type=Path)

    p = sub.add_parser("approve", help="keep this build's checks as the baseline later builds are compared with")
    p.add_argument("manifest", type=Path)
    p.add_argument("--note", default="", help="why it is good, e.g. what was checked in Warudo")

    p = sub.add_parser("compare", help="compare two check reports (scripts/compare_checks.py)")
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)

    p = sub.add_parser("runs", help="every build run, newest first: its steps, .vrm and rig check")
    p.add_argument("manifest", type=Path)

    p = sub.add_parser("bundle", help="gather everything an agent needs to look at a problem into one folder")
    p.add_argument("manifest", type=Path)
    p.add_argument("--note", default="", help="what looks wrong, in the user's words")

    args = parser.parse_args(argv)
    try:
        result = COMMANDS[args.command](args)
    except (StepFailed, ValueError, FileNotFoundError) as error:
        print(json.dumps({"error": str(error)}))
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_init(args):
    path = (args.manifest or REPO / "out" / args.name / "character.json").resolve()
    if path.exists():
        raise ValueError(f"{path} exists: edit it with set/joint/piece, or pick another --manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = manifest.new(_relative(args.model, path), args.name, args.author)
    if args.face:
        data["face"] = _relative(args.face, path)
    if args.reference:
        data["reference"] = _relative(args.reference, path)
    if getattr(args, "rig", None):
        data["rig"] = _relative(args.rig, path)
    manifest.save(data, path)
    return {"manifest": str(path), "next": f"python -m app run {path} --only survey"}


def cmd_status(args):
    return _project(args).status()


def cmd_runs(args):
    return {"runs": _project(args).runs()}


def cmd_approve(args):
    return _project(args).approve(args.note)


def cmd_run(args):
    project = _project(args)
    events = lambda event: print(json.dumps(event, default=str), file=sys.stderr, flush=True)  # noqa: E731
    return project.run(start=args.start, stop=args.stop, only=args.only, on_event=events)


def cmd_set(args):
    data = manifest.load(args.manifest)
    try:
        value = json.loads(args.value)
    except json.JSONDecodeError:
        value = args.value
    data["options"][args.key] = value
    manifest.save(data, args.manifest)
    return {"options": data["options"]}


def cmd_joint(args):
    data = manifest.load(args.manifest)
    if args.clear:
        data["joints"].pop(args.key, None)
    elif len(args.xyz) == 3:
        data["joints"][args.key] = args.xyz
    else:
        raise ValueError("give x y z, or --clear")
    manifest.save(data, args.manifest)
    return {"joints": data["joints"]}


def cmd_pieces(args):
    project = _project(args)
    out = project.out / "pieces.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    code, output = engine.run_script(project.blender, "list_pieces.py",
                                     # points and counts are the reference's, when there is one
                                     [manifest.path_of(project.data, "reference") or manifest.path_of(project.data, "source"),
                                      "--json", out,
                                      "--manifest", project.manifest_path, "--glb", project.out / "pieces.glb"],
                                     log=project.out / "pieces.log")
    if code != 0:
        raise StepFailed(f"list_pieces failed (see {project.out / 'pieces.log'})\n{output[-2000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def cmd_piece(args):
    data = manifest.load(args.manifest)
    rename = getattr(args, "rename", None)
    if rename:
        # anchors name their piece, so a detail keeps riding it under the new name
        if not any(p["name"] == args.name for p in data["pieces"]):
            raise ValueError(f"no piece named {args.name!r}")
        if any(p["name"] == rename for p in data["pieces"]):
            raise ValueError(f"a piece is already named {rename!r}")
        for p in data["pieces"]:
            if p["name"] == args.name:
                p["name"] = rename
            if p.get("anchor") == args.name:
                p["anchor"] = rename
        manifest.save(data, args.manifest)
        return {"pieces": data["pieces"]}
    rows = [p for p in data["pieces"] if p["name"] != args.name]
    if args.remove:
        data["pieces"] = rows
        manifest.save(data, args.manifest)
        return {"pieces": data["pieces"]}
    old = next((p for p in data["pieces"] if p["name"] == args.name), {})
    piece = dict(old, name=args.name)
    if args.label is not None:
        listing = engine._read_json(Project(args.manifest).out / "pieces.json", None)
        if listing is None:
            raise ValueError("run `pieces` first to list the labels")
        row = next((r for r in listing["pieces"] if r["label"] == args.label), None)
        if row is None:
            raise ValueError(f"no label {args.label} in the last listing")
        piece.update(point=row["point"], count=row["count"])
    for key in ("role", "point", "count", "bone", "anchor"):
        if getattr(args, key) is not None:
            piece[key] = getattr(args, key)
    if args.tip:
        piece["tip"] = True
    region = getattr(args, "region", None)
    if region is not None:
        piece["region"] = sorted({int(i) for i in (region if isinstance(region, list) else json.loads(Path(region).read_text()))})
    if piece.get("bone") == "auto":  # an appendage back to the bone nearest where it grows
        piece.pop("bone")
    # in place: pieces are repaired in the order listed
    names = [p["name"] for p in data["pieces"]]
    data["pieces"] = ([piece if p["name"] == args.name else p for p in data["pieces"]] if args.name in names
                      else rows + [piece])
    manifest.save(data, args.manifest)
    return {"piece": piece}


def cmd_render(args):
    project = _project(args)
    if getattr(args, "run", None):
        blends = sorted((project.out / "runs" / Path(args.run).name).glob("[0-9][0-9]_*.blend"))
        blend = blends[-1] if blends else None
    else:
        blend = Path(project.state["steps"][args.step]["blend"]) if args.step else project.latest()
    if blend is None:
        raise StepFailed("nothing built yet")
    # keyed by run too: every run has a 13_tpose.blend, and one run's render replaced another's
    out = project.out / "renders" / f"{blend.parent.name}_{blend.stem}_{args.pose}"
    code, output = engine.run_script(project.blender, "render_pose.py",
                                     [out, "--pose", args.pose, "--size", args.size], blend_in=blend,
                                     log=out.with_suffix(".log"))
    if code != 0:
        raise StepFailed(f"render failed\n{output[-2000:]}")
    return {"blend": str(blend), "run": blend.parent.name, "pose": args.pose, "images": sorted(str(p) for p in out.glob("*.png"))}


def cmd_movement(args):
    """Run scripts/check_movement.py on the modelled pose; the result is also kept as movement.json."""
    project = _project(args)
    # before the T-pose bake: a sleeve left at the ribs is no longer near the arm after it
    blend = Path(project.state["steps"][args.step]["blend"]) if args.step else project.latest(before="tpose")
    if blend is None:
        raise StepFailed("nothing rigged yet: run the build first")
    out = project.out / "movement.json"
    reference = manifest.path_of(project.data, "reference") or manifest.path_of(project.data, "source")
    code, output = engine.run_script(project.blender, "check_movement.py",
                                     ["--reference", reference, "--json", out, "--manifest", project.manifest_path,
                                      "--height", project.options()["height"], "--listing", project.out / "pieces.json"],
                                     blend_in=blend, log=project.out / "movement.log")
    if code != 0:
        raise StepFailed(f"check_movement failed (see {project.out / 'movement.log'})\n{output[-2000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def cmd_springs(args):
    """Run scripts/check_springs.py on the finished rig; the result is also kept as springs.json."""
    project = _project(args)
    blend = Path(project.state["steps"][args.step]["blend"]) if args.step else project.latest()
    if blend is None:
        raise StepFailed("nothing rigged yet: run the build first")
    out = project.out / "springs.json"
    code, output = engine.run_script(project.blender, "check_springs.py",
                                     ["--json", out, "--height", project.options()["height"]],
                                     blend_in=blend, log=project.out / "springs.log")
    if code != 0:
        raise StepFailed(f"check_springs failed (see {project.out / 'springs.log'})\n{output[-2000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def cmd_face_patch(args):
    """Render the head from the front and map that render onto the face as its own texture.

    Face points are pixels in that texture (see AVATAR_GUIDE.md step 6). The patched model
    becomes the manifest's source; the model before it becomes the reference, which piece
    points and vertex counts are read on.
    """
    project = _project(args)
    before = manifest.path_of(project.data, "source")
    reference = manifest.path_of(project.data, "reference") or before
    folder = project.out / "face"
    if folder.resolve() in before.resolve().parents:
        before = reference  # remaking: from the model before the last patch, never a patch of a patch
    views, image, patched = folder / "views", folder / "face_patch_head_000.png", folder / f"{project.name}_face.glb"
    head_at = []
    survey = project.survey()
    height = project.options()["height"]
    if survey and survey.get("eyes"):
        # the modelled eyes, a little under the face's middle, as a share of the height
        head_at = ["--head-at", round((survey["eyes"][2] - FACE_BELOW_EYES * height) / height, 3)]
    elif survey and "head" in survey.get("joints", {}):
        # the face, a little above the head joint (a guess where clothing or a tail hides the neck)
        head_at = ["--head-at", round((survey["joints"]["head"][2] + FACE_ABOVE_HEAD * height) / height, 3)]
    code, output = engine.run_script(project.blender, "retexture/render_views.py", [before, views, 1024, *head_at],
                                     log=folder / "render_views.log")
    if code != 0 or not (views / "color_head_000.png").exists():
        raise StepFailed(f"rendering the head failed (see {folder / 'render_views.log'})\n{output[-2000:]}")
    from PIL import Image
    Image.open(views / "color_head_000.png").convert("RGB").resize((2048, 2048), Image.LANCZOS).save(image)
    code, output = engine.run_script(project.blender, "retexture/face_patch.py", [before, views, "head_000", image, patched],
                                     log=folder / "face_patch.log")
    if code != 0 or not patched.exists():
        raise StepFailed(f"making the face patch failed (see {folder / 'face_patch.log'})\n{output[-2000:]}")
    data = manifest.load(project.manifest_path)
    data["reference"] = _relative(reference, project.manifest_path)
    data["source"] = _relative(patched, project.manifest_path)
    manifest.save(data, project.manifest_path)
    return {"source": str(patched), "reference": str(reference), "image": str(image),
            "next": "mark the face points on the image, then rerun from survey"}


# The face's middle sits this share of the height above the head joint (the head bone's base).
FACE_ABOVE_HEAD = 0.04
# ... and this share under the eyes, so the mouth and chin are in the frame too.
FACE_BELOW_EYES = 0.02

FACE_KEYS = {"right_eye": ("inner", "outer", "top", "bottom"), "left_eye": ("inner", "outer", "top", "bottom"),
             "mouth": ("left", "right", "top", "bottom")}


def cmd_face(args):
    return save_face(args.manifest, json.loads(Path(args.points).read_text(encoding="utf-8")), blender=args.blender)


def save_face(manifest_path, points, blender=None):
    """Check face points and keep them as <out>/face.json, named in the manifest."""
    for key, names in FACE_KEYS.items():
        missing = [n for n in names if not isinstance((points.get(key) or {}).get(n), list)]
        if missing:
            raise ValueError(f"{key} is missing {', '.join(missing)}")
    if not points.get("size"):
        raise ValueError("no image size")
    project = Project(manifest_path, blender=blender)
    keep = {"image": points.get("image") or "face_patch.png", "size": points["size"],
            **{key: {n: [round(float(v), 1) for v in points[key][n]] for n in names} for key, names in FACE_KEYS.items()}}
    path = project.out / "face.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(keep, indent=2), encoding="utf-8")
    data = manifest.load(project.manifest_path)
    data["face"] = _relative(path, project.manifest_path)
    manifest.save(data, project.manifest_path)
    return {"face": str(path), "next": "rerun from build: the face points shape the head, eyes and mouth"}


def cmd_compare(args):
    import subprocess
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "compare_checks.py"), str(args.before),
                             str(args.after)], capture_output=True, text=True)
    return {"worse": result.returncode == 1, "report": result.stdout}


# Screenshots a user attaches from Warudo: kinds, and the most one note may carry (bytes).
SCREENSHOT_KINDS = {"image/png": ".png", "image/jpeg": ".jpg"}
SCREENSHOT_MAX = 8 * 1024 * 1024


def save_feedback(manifest_path, note, images=()):
    """What the user saw in Warudo, kept with the exact .vrm it was about. Returns the entry.

    `images` are data URLs (PNG or JPEG) from the page. Feedback given earlier was lost between
    sessions or tied to the wrong build; each note now sits in its own folder beside the build.
    """
    import base64
    project = Project(manifest_path)
    if not (note or "").strip() and not images:
        raise ValueError("say what you saw, or attach a screenshot")
    shots = []
    for i, url in enumerate(images):
        head, _, data = str(url).partition(",")
        kind = head.removeprefix("data:").split(";")[0]
        if kind not in SCREENSHOT_KINDS or not head.endswith(";base64"):
            raise ValueError("screenshots must be PNG or JPEG images")
        raw = base64.b64decode(data)
        if len(raw) > SCREENSHOT_MAX:
            raise ValueError(f"screenshot {i + 1} is over {SCREENSHOT_MAX // 2**20} MB")
        shots.append((f"screenshot_{i + 1}{SCREENSHOT_KINDS[kind]}", raw))
    stamp, n = datetime.now().strftime("%Y%m%d-%H%M%S"), 0
    while (project.out / "feedback" / (stamp + (f"-{n}" if n else ""))).exists():
        n += 1
    folder = project.out / "feedback" / (stamp + (f"-{n}" if n else ""))
    folder.mkdir(parents=True)
    saved = []
    for name, raw in shots:
        (folder / name).write_bytes(raw)
        saved.append(name)
    entry = {"when": datetime.now().isoformat(timespec="seconds"), "note": note or "",
             "vrm": str(project._last_vrm() or ""), "screenshots": saved}
    (folder / "feedback.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return dict(entry, folder=str(folder))


def list_feedback(manifest_path):
    """Every note given on this character, newest first, with its folder."""
    root = Project(manifest_path).out / "feedback"
    entries = []
    for path in sorted(root.glob("*/feedback.json"), reverse=True) if root.exists() else []:
        entries.append(dict(json.loads(path.read_text(encoding="utf-8")), folder=str(path.parent)))
    return entries


def cmd_bundle(args):
    """Manifest, state, survey, checks, piece listing, step logs and renders in one folder."""
    project = _project(args)
    folder = project.out / "bundles" / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True)
    shutil.copy2(project.manifest_path, folder / "character.json")
    if project.state_path.exists():
        shutil.copy2(project.state_path, folder / "state.json")
    for name, entry in project.state["steps"].items():
        for key in ("json", "log"):
            if entry.get(key) and Path(entry[key]).exists():
                shutil.copy2(entry[key], folder / f"{name}{Path(entry[key]).suffix}")
    for extra in ("pieces.json", "movement.json", "springs.json", "face.json"):
        if (project.out / extra).exists():
            shutil.copy2(project.out / extra, folder / extra)
    feedback = project.out / "feedback"
    if feedback.exists():
        shutil.copytree(feedback, folder / "feedback")
    renders = project.out / "renders"
    if renders.exists():
        shutil.copytree(renders, folder / "renders", ignore=shutil.ignore_patterns("*.log"))
    (folder / "NOTE.md").write_text(
        f"# {project.name}: what looks wrong\n\n{args.note or '(no note given)'}\n\n"
        f"Latest file: `{project.latest()}`\n\nStart with docs/agent/app.md, then "
        f"docs/agent/troubleshooting.md. Measure before changing any rule.\n", encoding="utf-8")
    return {"bundle": str(folder), "files": sorted(p.name for p in folder.iterdir())}


COMMANDS = {"init": cmd_init, "status": cmd_status, "run": cmd_run, "set": cmd_set, "joint": cmd_joint,
            "pieces": cmd_pieces, "piece": cmd_piece, "render": cmd_render, "movement": cmd_movement, "springs": cmd_springs,
            "face-patch": cmd_face_patch, "face": cmd_face, "compare": cmd_compare, "approve": cmd_approve, "bundle": cmd_bundle,
            "runs": cmd_runs}


def _project(args):
    return Project(args.manifest, blender=getattr(args, "blender", None))


def _relative(target, manifest_path):
    """A path as the manifest should store it: relative to the manifest's folder when it can be."""
    target = Path(target).resolve()
    try:
        return str(Path(target).relative_to(manifest_path.parent))
    except ValueError:
        import os
        try:
            return os.path.relpath(target, manifest_path.parent)
        except ValueError:  # another drive
            return str(target)
