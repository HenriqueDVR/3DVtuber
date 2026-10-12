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
from types import SimpleNamespace

from . import engine, face_guess, inside, paint, preflight, report, sheet, update, vendor
from .engine import REPO, Project, StepFailed, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m app", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blender", help="path to blender.exe (default: $BLENDER, PATH, Program Files)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="what the app needs and what it found: Blender, the VRM Add-on, disk, memory")
    p.add_argument("--use", metavar="BLENDER", help="run builds with this blender.exe from now on, wherever it is (\"\" to go back to the app picking one)")
    p.add_argument("--forget", metavar="BLENDER", help="take a blender.exe added with --use off the list")
    p.add_argument("--blenders", metavar="N", help="never more than N Blenders at once (1: one at a time); auto: as many as the computer has room for")

    p = sub.add_parser("detector", help="the face detector behind \"Place them for me\": where it is, or --install it")
    p.add_argument("--install", action="store_true", help="into tools/face_detector: about 250 MB to download, 1.5 GB on disk")

    p = sub.add_parser("update", help="is a newer version of the app published (update.json says where)? --apply gets it")
    p.add_argument("--apply", action="store_true", help="download it and put it in place (about 8 MB; out/ and tools/ are not touched)")
    p = sub.add_parser("pack", help="the zip a tester starts from: the committed files, their version, and update.json when there is one")
    p.add_argument("--out", type=Path, help="folder for the zip (default: dist/)")
    p.add_argument("--at", help="the commit to pack: a hash, a branch, origin/testers (default: the one checked out)")
    sub.add_parser("vendor", help="keep a copy of three.js, three-vrm and the face tracker with the app, so it runs offline (17 MB)")

    p = sub.add_parser("init", help="start a character manifest for a model")
    p.add_argument("model", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--author", default="Unknown")
    p.add_argument("--manifest", type=Path, help="where to write it (default out/<name>/character.json)")
    p.add_argument("--face", type=Path, help="face.json from viewer/setup.html")
    p.add_argument("--reference", type=Path, help="the unmodified model, when the source had its texture edited")
    p.add_argument("--rig", type=Path, help="the same model rigged by Tripo (its FBX), when that is a separate file: "
                   "its skeleton and weights are used (a rigged FBX model is its own rig without this)")

    p = sub.add_parser("status", help="options, plan and what each step produced")
    p.add_argument("manifest", type=Path)

    p = sub.add_parser("prepare", help="all a new character needs before its face is marked: survey, piece listing, face patch")
    p.add_argument("manifest", type=Path)

    p = sub.add_parser("preflight", help="what is known to go wrong with a build before it starts: stop, warn and ok lines, "
                       "each with its code and what mends it (no Blender, nothing written)")
    p.add_argument("manifest", type=Path)
    p.add_argument("--from", dest="start", choices=engine.NAMES)
    p.add_argument("--to", dest="stop", choices=engine.NAMES)
    p.add_argument("--only", choices=engine.NAMES)

    p = sub.add_parser("run", help="run the build chain, or part of it")
    p.add_argument("manifest", type=Path)
    p.add_argument("--from", dest="start", choices=engine.NAMES)
    p.add_argument("--to", dest="stop", choices=engine.NAMES)
    p.add_argument("--only", choices=engine.NAMES)
    p.add_argument("--strict", action="store_true", help="don't start when `preflight` has a stop: the build is known to go wrong")

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
    p.add_argument("--tip", action="store_true", default=None, help="for hat: swing its pointed tip")
    p.add_argument("--remove", action="store_true")
    p.add_argument("--region", type=Path, help="a JSON list of source vertex numbers: only this part of the piece")
    p.add_argument("--rename", metavar="NEW", help="give the named piece a new name (details anchored to it follow)")

    p = sub.add_parser("render", help="render the latest (or a step's) rig in a pose; returns the PNGs")
    p.add_argument("manifest", type=Path)
    p.add_argument("--pose", default="raised")
    p.add_argument("--step", choices=engine.NAMES, help="render this step's output instead of the latest")
    p.add_argument("--run", help="render the last .blend of this run folder (a name from `runs`) instead")
    p.add_argument("--size", type=int, default=800)

    p = sub.add_parser("portraits", help="draw the character in the poses the app's menus show it in (after a build they are drawn by themselves)")
    p.add_argument("manifest", type=Path)

    p = sub.add_parser("sheet", help="contact sheets of the finished .vrm as apps draw it: face, body, details (headless Edge or Chrome)")
    p.add_argument("target", type=Path, help="a character manifest (its last .vrm) or a .vrm file")
    p.add_argument("--out", type=Path, help="folder for the PNGs (default: the character's renders/, or beside the .vrm)")
    p.add_argument("--browser", help="msedge.exe or chrome.exe to use")

    p = sub.add_parser("inside", help="what shows through a hole cut in the finished .vrm's surface, as apps draw it (the eye white deep in its socket); a build runs it with its vrm_check")
    p.add_argument("target", type=Path, help="a character manifest (its last .vrm) or a .vrm file")
    p.add_argument("--out", type=Path, help="folder for the report and pictures (default: the character's renders/, or beside the .vrm)")
    p.add_argument("--browser", help="msedge.exe or chrome.exe to use")

    p = sub.add_parser("movement", help="which pieces stay behind or tear when the arms and head move (runs Blender)")
    p.add_argument("manifest", type=Path)
    p.add_argument("--step", choices=engine.NAMES, help="check this step's output (default: the last before the T-pose)")

    p = sub.add_parser("springs", help="run the spring bones as apps do: what droops, floats off or sinks in (runs Blender)")
    p.add_argument("manifest", type=Path)
    p.add_argument("--step", choices=engine.NAMES, help="check this step's output (default: the latest)")

    p = sub.add_parser("face-patch", help="give the model a front-mapped face texture to mark points on (runs Blender)")
    p.add_argument("manifest", type=Path)

    p = sub.add_parser("duplicate", help="the character again under another name: same model, own output folder, to build a choice both ways")
    p.add_argument("manifest", type=Path)
    p.add_argument("name")

    p = sub.add_parser("rename", help="give a character another name; its builds stay where they are")
    p.add_argument("manifest", type=Path)
    p.add_argument("name")

    p = sub.add_parser("hide", help="take a character off the app's list (--show puts it back); no file is deleted")
    p.add_argument("manifest", type=Path)
    p.add_argument("--show", action="store_true")

    p = sub.add_parser("delete", help="delete a character: its folder in out/ and its manifest (--dry: what would go)")
    p.add_argument("manifest", type=Path)
    p.add_argument("--dry", action="store_true")

    p = sub.add_parser("paint", help="the texture paint kept with a character (Paint page): what there is, or --clear it")
    p.add_argument("manifest", type=Path)
    p.add_argument("--clear", action="store_true", help="drop all of it: the build reads the generated texture again")

    p = sub.add_parser("face-guess", help="face points placed by an anime face detector, to check and nudge (needs a Python with anime-face-detector)")
    p.add_argument("manifest", type=Path)
    p.add_argument("--save", action="store_true", help="keep them as the character's face points")
    p.add_argument("--python", help="a Python that has anime-face-detector")

    p = sub.add_parser("face", help="save face points (a face.json from viewer/setup.html) into the character")
    p.add_argument("manifest", type=Path)
    p.add_argument("points", type=Path)

    p = sub.add_parser("approve", help="keep this build's checks as the baseline later builds are compared with")
    p.add_argument("manifest", type=Path)
    p.add_argument("--note", default="", help="why it is good, e.g. what was checked in Warudo")

    p = sub.add_parser("warudo", help="copy the finished .vrm into Warudo's Characters folder")
    p.add_argument("manifest", type=Path, nargs="?")
    p.add_argument("--folder", help="send avatars to this folder from now on ('' = the one found in Steam)")

    p = sub.add_parser("compare", help="compare two check reports (scripts/compare_checks.py)")
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)

    p = sub.add_parser("runs", help="every build run, newest first: its steps, .vrm and rig check")
    p.add_argument("manifest", type=Path)

    p = sub.add_parser("bundle", help="gather everything an agent needs to look at a problem into one folder")
    p.add_argument("manifest", type=Path)
    p.add_argument("--note", default="", help="what looks wrong, in the user's words")

    p = sub.add_parser("report", help="a zip for the developers: settings, logs, checks and numbers, paths rewritten; nothing is sent")
    p.add_argument("manifest", type=Path, nargs="?")
    p.add_argument("--note", default="", help="what happened, in the user's words")
    p.add_argument("--about", default="", help="what the app was showing (an error's words)")
    p.add_argument("--pictures", action="store_true", help="also the rendered pictures of the character")
    p.add_argument("--model", action="store_true", help="also the model the build reads, its rig, reference and paint (sent: textures made smaller to fit)")
    p.add_argument("--send", action="store_true", help="post it to the developers' channel (the address in update.json)")
    p.add_argument("--address", help="keep this Discord webhook as where reports are sent ('' forgets it)")
    p.add_argument("--check", type=Path, metavar="ZIP", help="a report that came in: is it one this app writes? Nothing in it is opened or run")
    p.add_argument("--unpack", action="store_true", help="with --check: unpack it into out/received/<name>/ when nothing is wrong with it")

    p = sub.add_parser("regress", help="rebuild characters in a scratch folder and compare each with its good build")
    p.add_argument("manifests", type=Path, nargs="*", help="which characters (default: every one with a finished build)")
    p.add_argument("--keep", action="store_true", help="keep the scratch builds' .blend and .vrm files")

    p = sub.add_parser("inputs", help="the character's model handed over otherwise (turned, in centimetres, in parts, decimated, "
                       "as FBX or OBJ, arms down, no picture), each copy built in a scratch folder and held against the model's own build")
    p.add_argument("manifest", type=Path)
    p.add_argument("--only", help="names of the copies to make and build, with commas (scripts/make_variants.py; default: all)")
    p.add_argument("--rigged", type=Path, help="a rigged FBX (Tripo's), for the two copies with their arms lowered")
    p.add_argument("--keep", action="store_true", help="keep the copies and the scratch builds' .blend and .vrm files")

    p = sub.add_parser("disk", help="how much disk a character's runs take, and what `prune` would free")
    p.add_argument("manifest", type=Path)

    p = sub.add_parser("prune", help="delete the .blend and .vrm files of runs nothing reads any more (reports and logs stay)")
    p.add_argument("manifest", type=Path)
    p.add_argument("--dry-run", action="store_true", help="only say what would go")

    args = parser.parse_args(argv)
    try:
        result = COMMANDS[args.command](args)
    except (StepFailed, ValueError, FileNotFoundError) as error:
        known = getattr(error, "known", None)
        print(json.dumps({"error": str(error), **({"code": known["code"], "fix": known.get("fix", "")} if known and known.get("code") else {})}))
        return 1
    except OSError as error:
        # the app's own write refused by a full drive: said as that, not as a traceback
        if not engine.no_space(error):
            raise
        known = engine.errors.describe("VM-DISK-FULL")
        print(json.dumps({"error": f"{known['what']}: {error}", "code": known["code"], "fix": known["fix"]}))
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_setup(args):
    if args.forget:
        return engine.forget_blender(args.forget)
    if getattr(args, "blenders", None) is not None:
        return engine.use_blenders(args.blenders)
    return engine.use_blender(args.use) if args.use is not None else engine.setup(args.blender)


def install_detector(on_event=None):
    """The face detector into tools/face_detector (face_guess.install), stoppable from the app."""
    try:
        return face_guess.install(on_event=on_event, cancelled=engine.CANCEL.is_set)
    except face_guess.NoDetector as error:
        if engine.CANCEL.is_set():
            raise engine.Cancelled(str(error)) from None
        raise StepFailed(str(error)) from None


def cmd_detector(args):
    if args.install:
        events = lambda event: print(json.dumps(event, default=str), file=sys.stderr, flush=True)  # noqa: E731
        return install_detector(on_event=events)
    found = face_guess.status()["python"]
    return {"python": found, "next": "" if found else "python -m app detector --install (about 250 MB to download, 1.5 GB on disk)"}


def cmd_update(args):
    if not args.apply:
        return update.check()
    events = lambda event: print(json.dumps(event, default=str), file=sys.stderr, flush=True)  # noqa: E731
    return update.apply(on_event=events)


def cmd_pack(args):
    return update.pack(args.out, at=args.at)


def cmd_vendor(args):
    events = lambda event: print(json.dumps(event, default=str), file=sys.stderr, flush=True)  # noqa: E731
    return vendor.fetch(on_event=events)


MODEL_TYPES = (".fbx", ".glb", ".gltf", ".obj", ".vrm")
# What Windows lets no file or folder be named with.
UNNAMEABLE = '<>:"/\\|?*'


def cmd_init(args):
    # the name is the character's folder: said here, where Windows' own refusal of one ("Yuki?",
    # "Miku: winter") came back as the app's crash, "[WinError 123] ... syntax is incorrect"
    odd = sorted(set(args.name) & set(UNNAMEABLE)) + (["a tab or a line break"] if any(ord(c) < 32 for c in args.name) else [])
    if args.manifest is None and odd:
        raise ValueError(f"the name can't have {' '.join(odd)} in it: it is also the name of the character's folder")
    path = (args.manifest or REPO / "out" / args.name / "character.json").resolve()
    if path.exists():
        raise ValueError(f"{path} exists: edit it with set/joint/piece, or pick another --manifest")
    # said here, with what was typed: left to the survey it was Blender's "can't be read" a minute
    # later, the same for a .zip, a folder and a path pasted with its quotes on
    model = Path(args.model)
    if not model.is_file():
        raise ValueError(f"no model file at {model}" + (": that is a folder, name the file in it" if model.is_dir() else ""))
    if model.suffix.lower() not in MODEL_TYPES:
        raise ValueError(f"{model.name} is a {model.suffix or 'file with no type'}: the model must be one of {', '.join(MODEL_TYPES)}"
                         + (" (unpack the .zip and name the model inside it)" if model.suffix.lower() in (".zip", ".rar", ".7z") else ""))
    if Path(args.model).suffix.lower() == ".vrm" and engine.shipped(args.model) is None:
        raise ValueError(f"{args.model} isn't a VRM file the app can read (VRM 1.0 or 0.x)")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = manifest.new(_relative(args.model, path), args.name, args.author)
    if args.face:
        data["face"] = _relative(args.face, path)
    if args.reference:
        data["reference"] = _relative(args.reference, path)
    if getattr(args, "rig", None):
        data["rig"] = _relative(args.rig, path)
    elif (rig := manifest.rig_of(args.model)) is not None:
        # Tripo's rigged FBX is model and rig in one file; a GLB the texture steps made from it names it
        data["rig"] = data["source"] if rig.resolve() == Path(args.model).resolve() else _relative(rig, path)
    manifest.save(data, path)
    # another tool's skeleton in the model: readable (adopt.py) but not measured, so not taken unasked
    other = None if data.get("rig") or manifest.keeps_rig(manifest.load(path)) else manifest.rig_kind(args.model)
    improving = manifest.taken_in(manifest.load(path))
    return {"manifest": str(path), "source": data["source"], "rig": data.get("rig"), "other_rig": other,
            # whose avatar it is and whether its terms have to be read first (`set licence_read true` says they were)
            "licence": engine.licence(args.model) if manifest.improves(data) else None,
            # a model that came whole (an MMD model as FBX) is taken in with its own rig, as a .vrm is
            "kept_rig": manifest.keeps_rig(manifest.load(path)),
            "improving": improving, "next": f"python -m app {'run' if improving else 'prepare'} {path}"}


def cmd_status(args):
    return _project(args).status()


def cmd_runs(args):
    return {"runs": _project(args).runs()}


def cmd_approve(args):
    return _project(args).approve(args.note)


def cmd_warudo(args):
    if args.folder is not None:
        return engine.use_warudo(args.folder)
    if args.manifest is None:
        return {"warudo": str(engine.warudo_folder() or "")}
    return _project(args).send_to_warudo()


def cmd_regress(args):
    """Rebuild each character from its manifest into out/regress/<name>/ and compare the reports
    with its baseline: the build marked good, else its last build. Its own files are not touched.

    Every pipeline change so far was A/B'd by hand in scratch folders (out/_regress_0928,
    out/compare/*); this is that, as one command and one table. As many characters at once as the
    computer has room for, one Blender each (`engine.at_once`; one after another on a computer
    without the room, or told to keep to one Blender), a minute or two a character. `worse` > 0
    means something measured worse than the baseline.
    """
    events = getattr(args, "on_event", None) or (lambda event: print(json.dumps(event, default=str), file=sys.stderr, flush=True))
    chosen = [Path(m).resolve() for m in args.manifests] or [m for m in engine.characters() if Project(m).baseline()[0]]
    rows, jobs, took = [], [], {}
    for path in chosen:
        project = Project(path, blender=getattr(args, "blender", None))
        what, before = project.baseline()
        row = {"name": project.name, "manifest": str(path), "baseline": what, "worse": [], "better": [], "error": None}
        rows.append(row)
        if what is None:
            row["error"] = "never built to a rig check: nothing to compare with"
            continue
        jobs.append((row, before, _scratch_copy(project)))

    def rebuild(scratch, emit):
        emit({"step": scratch.name, "status": "running", "reason": f"rebuilding in {scratch.out}"})
        started = datetime.now()
        try:
            scratch.run(on_event=emit)
        finally:
            took[scratch.out] = round((datetime.now() - started).total_seconds())
            # as each one ends, not when all have: twenty characters' models are 12 GB
            if not args.keep:
                for model in (p for p in scratch.out.rglob("*") if p.is_file() and p.suffix.lower() in engine.MODELS):
                    model.unlink()

    ended = engine.at_once([scratch for _, _, scratch in jobs], rebuild, on_event=events)
    for (row, before, scratch), end in zip(jobs, ended):
        if isinstance(end, StepFailed):
            row["error"] = str(end).splitlines()[0]
        elif isinstance(end, BaseException):
            raise end
        row["seconds"] = took.get(scratch.out)
        row.update(engine.changes(before, scratch.reports()))
        row["out"] = str(scratch.out)
    return {"characters": rows, "worse": sum(len(r["worse"]) for r in rows), "failed": [r["name"] for r in rows if r["error"]]}


def _scratch_copy(project):
    """The character again under out/regress/<name>/: the same manifest with its paths made
    absolute and its own output folder, plus the piece listing the movement check names pieces by."""
    out = REPO / "out" / "regress" / project.name
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(json.dumps(project.data))
    for key in manifest.PATHS:
        if key != "out_dir" and data.get(key):
            data[key] = str(manifest.path_of(project.data, key))
    data["out_dir"] = str(out)
    manifest.save(data, out / "character.json")
    if (project.out / "pieces.json").exists():
        shutil.copy2(project.out / "pieces.json", out / "pieces.json")
    scratch = Project(out / "character.json", blender=project._blender)
    # the folder is kept between runs, and with it an eye step's refusal by an older pipeline: model A's
    # T-pose build was rebuilt with sliding eyes for three days after its iris worked, against an iris baseline
    if scratch.state.pop("eyes_fallback", None):
        scratch._save_state()
    return scratch


def cmd_inputs(args):
    from . import inputs
    events = lambda event: print(json.dumps(event, default=str), file=sys.stderr, flush=True)  # noqa: E731
    return inputs.run(args.manifest, only=args.only.split(",") if args.only else None, rigged=args.rigged, keep=args.keep,
                      on_event=events, blender=getattr(args, "blender", None))


def cmd_disk(args):
    return _project(args).usage()


def cmd_prune(args):
    return _project(args).prune(dry=args.dry_run)


def cmd_preflight(args):
    return _project(args).preflight(start=args.start, stop=args.stop, only=args.only)


def said_before(project, emit, start=None, stop=None, only=None):
    """Say what `preflight` found as the run's first events (status "preflight": the page writes them
    into its progress, an agent reads them on stderr); returns the stops. Nothing is refused here."""
    found = [line for line in project.preflight(start=start, stop=stop, only=only)["lines"] if line["kind"] != "ok"]
    for line in found:
        emit({"step": line["at"] or only or start or engine.NAMES[0], "status": "preflight", "kind": line["kind"],
              "code": line["code"], "reason": line["text"], "fix": line["fix"]})
    return [line for line in found if line["kind"] == "stop"]


def cmd_run(args):
    project = _project(args)
    events = lambda event: print(json.dumps(event, default=str), file=sys.stderr, flush=True)  # noqa: E731
    stops = said_before(project, events, start=args.start, stop=args.stop, only=args.only)
    if stops and getattr(args, "strict", False):
        raise StepFailed("not started: " + "; ".join(f"{line['text']} [{line['code']}]" for line in stops))
    return project.run(start=args.start, stop=args.stop, only=args.only, on_event=events)


def cmd_prepare(args):
    events = lambda event: print(json.dumps(event, default=str), file=sys.stderr, flush=True)  # noqa: E731
    return prepare(args.manifest, on_event=events, blender=getattr(args, "blender", None))


def prepare(manifest_path, on_event=None, blender=None):
    """All a new character needs before its face can be marked, in one go: the survey, the piece
    listing and the face patch, each only when it isn't there yet. Returns what it did.

    These were three buttons on three pages, and a first build without the face patch had to be
    redone from the start once it was made (the patched copy becomes the model). The survey runs
    again after the patch, since it is of the model the build will read.
    """
    emit = on_event or (lambda event: None)
    args = lambda: SimpleNamespace(manifest=Path(manifest_path), blender=blender)  # noqa: E731
    did = []
    if _project(args()).improving:
        # a finished .vrm has its pieces, face and weights already: nothing to get ready
        return {"did": did, "status": _project(args()).status(), "next": "`run`: the file is taken in as it is and added to"}
    if _project(args()).survey() is None:
        _project(args()).run(only="survey", on_event=emit)
        did.append("survey")
    listing = _project(args()).out / "pieces.json"
    if not listing.exists() or listing_stale(_project(args()), engine._read_json(listing, None)):
        emit({"step": "pieces", "status": "running"})
        cmd_pieces(args())
        emit({"step": "pieces", "status": "done"})
        did.append("pieces")
    if not _project(args()).survey().get("face_patch"):
        emit({"step": "face patch", "status": "running"})
        cmd_face_patch(args())
        emit({"step": "face patch", "status": "done"})
        _project(args()).run(only="survey", on_event=emit)
        did.append("face patch")
    return {"did": did, "status": _project(args()).status(),
            "next": "mark the face points (Face page, or `face`), then `run`; `run` alone builds without face tracking"}


def cmd_set(args):
    data = manifest.load(args.manifest)
    try:
        value = json.loads(args.value)
    except json.JSONDecodeError:
        value = args.value
    data["options"][args.key] = value
    manifest.save(data, args.manifest)
    if manifest.improves(data) and args.key not in manifest.VRM_OPTIONS:
        # kept all the same: the manifest stays whole, and its source may become a model
        return {"options": data["options"],
                "note": f"this character's source is a finished .vrm, taken in as it is: {args.key} changes nothing in it "
                        f"(the options that do: {', '.join(manifest.VRM_OPTIONS)})"}
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


def listing_stale(project, listing):
    """Whether the piece listing was made before the eyes were marked: its suggestions then read the
    face by size alone (the fox girl's face came out as hair, her mouth and an eye as a hat)."""
    face = manifest.path_of(project.data, "face")
    points = engine._read_json(face, None) if face else None
    return bool(listing and points and engine.marks_eyes(points) and not listing.get("eyes_marked")
                and (project.survey() or {}).get("face_patch"))


def cmd_pieces(args):
    project = _project(args)
    out = project.out / "pieces.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    face = manifest.path_of(project.data, "face")
    code, output = engine.run_script(project.blender, "list_pieces.py",
                                     # points and counts are the reference's, when there is one
                                     [manifest.path_of(project.data, "reference") or manifest.path_of(project.data, "source"),
                                      "--json", out,
                                      "--manifest", project.manifest_path, "--glb", project.out / "pieces.glb",
                                      "--picture", project.out / engine.FIRST_PICTURE,
                                      # the marks are on the face patch, which the reference doesn't have
                                      *(["--face", face, "--marked-on", manifest.path_of(project.data, "source")]
                                        if face and Path(face).exists() else [])],
                                     log=project.out / "pieces.log", extra=project.frame())
    if code != 0:
        raise StepFailed(f"list_pieces failed (see {project.out / 'pieces.log'})\n{output[-2000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


# What a piece is told beside its role, each read by one role or two (follow_bone and appendage: bone;
# detail: anchor; hat: tip).
ROLE_DETAILS = ("bone", "anchor", "tip")


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
    if args.role is not None and old.get("role") not in (None, args.role):
        # what the role it had was given stays with that role: a hanging sleeve's arm bone, kept, made the
        # tail it was then called grow from the arm, with the Parts page showing "grows from: nearest"
        for key in ROLE_DETAILS:
            piece.pop(key, None)
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
    # None: not said, the tip stays as it was. False takes it off: the Parts page's "tip swings" could be
    # ticked and never unticked
    if args.tip:
        piece["tip"] = True
    elif args.tip is False:
        piece.pop("tip", None)
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
        blend = _blend_of(project, args.step) if args.step else project.latest()
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


def cmd_portraits(args):
    """The character's posed pictures (scripts/render_portraits.py), for one built before the app drew them."""
    project = _project(args)
    if project.latest() is None:
        raise StepFailed("nothing built yet: the pictures need the rig")
    drawn = project.portraits(on_event=getattr(args, "on_event", None))
    if not drawn:
        raise StepFailed(f"the pictures could not be drawn: see {project.out / 'portraits' / 'portraits.log'}")
    return {"portraits": drawn}


def cmd_sheet(args):
    """viewer/sheet.html's sheets of an exported file, drawn by three-vrm as apps draw it: the poses,
    expressions and camera angles where Warudo showed defects Blender's renders never did."""
    if args.target.suffix.lower() == ".vrm":
        vrm, out = args.target, args.out or args.target.with_name(args.target.stem + "_sheets")
    else:
        project = Project(args.target, blender=getattr(args, "blender", None))
        vrm = project._last_vrm()
        if vrm is None:
            raise StepFailed("nothing exported yet")
        out = args.out or project.out / "renders" / f"sheets_{vrm.parent.name}"
    try:
        return {"vrm": str(vrm), **sheet.render(vrm, out, browser=args.browser)}
    except (sheet.NoBrowser, TimeoutError, RuntimeError) as error:
        raise StepFailed(f"sheet: {error}")


def cmd_inside(args):
    """viewer/inside.html's count on an exported file (app/inside.py): the worst camera's area of what shows
    over 12 mm behind a window cut in a surface, and what of it fails; the pictures are in `out`."""
    if args.target.suffix.lower() == ".vrm":
        vrm, out = args.target, args.out or args.target.with_name(args.target.stem + "_inside")
    else:
        project = Project(args.target, blender=getattr(args, "blender", None))
        vrm = project._last_vrm()
        if vrm is None:
            raise StepFailed("nothing exported yet")
        out = args.out or project.out / "renders" / f"inside_{vrm.parent.name}"
    try:
        report = inside.measure(vrm, out, browser=args.browser, pictures=True)
    except (sheet.NoBrowser, TimeoutError, RuntimeError) as error:
        raise StepFailed(f"inside: {error}")
    return {"vrm": str(vrm), "out": str(out), "images": report["images"], **inside.judge(report)}


def _blend_of(project, step):
    """The .blend a named step left; StepFailed when it hasn't run or leaves none (survey, the checks)."""
    blend = project.state["steps"].get(step, {}).get("blend")
    if not blend:
        raise StepFailed(f"{step} has no .blend yet: run the build up to it first")
    return Path(blend)


def cmd_movement(args):
    """Run scripts/check_movement.py on the modelled pose; the result is also kept as movement.json."""
    project = _project(args)
    # before the T-pose bake: a sleeve left at the ribs is no longer near the arm after it
    blend = _blend_of(project, args.step) if args.step else project.latest(before="tpose")
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
    blend = _blend_of(project, args.step) if args.step else project.latest()
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
    if folder.resolve() in before.resolve().parents or _patched(before):
        before = reference  # remaking: from the model before the last patch, never a patch of a patch
    views, image, patched = folder / "views", folder / "face_patch_head_000.png", folder / f"{project.name}_face.glb"
    at, body = _frame(project.survey(), project.options()["height"])
    head_at = [] if at is None else ["--head-at", at]
    # a remake frames the head as the patch before it did: the face points are pixels in that picture,
    # and a survey of the patched model put the frame a millimetre lower (8 px) than the first one
    kept_in = folder
    kept = _framing(folder) if project.data.get("face") else None
    if kept is None and project.data.get("face") and _patched(manifest.path_of(project.data, "source")):
        # a copy of a character (`duplicate`) wears the original's patch and has its points and
        # paint: its first patch of its own is framed as that one was. Framed anew, the mouth of
        # model D's copy opened at the nose and the paint lay a face part too high
        kept_in = manifest.path_of(project.data, "source").parent
        kept = _framing(kept_in)
    if kept is not None:
        head_at = ["--head-at", kept]
        body = (engine._read_json(kept_in / "head_at.json", None) or {}).get("body", 1.0)
    if head_at:
        head_at += ["--body", body]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "head_at.json").write_text(json.dumps({"head_at": head_at[1], "body": body}), encoding="utf-8")
    code, output = engine.run_script(project.blender, "retexture/render_views.py",
                                     [before, views, 1024, *head_at, "--only", ",".join(PATCH_VIEWS), "--depth-of", PATCH_VIEW],
                                     log=folder / "render_views.log")
    if code != 0 or not (views / "color_head_000.png").exists():
        raise StepFailed(f"rendering the head failed (see {folder / 'render_views.log'})\n{output[-2000:]}")
    from PIL import Image
    Image.open(views / "color_head_000.png").convert("RGB").resize((2048, 2048), Image.LANCZOS).save(image)
    code, output = engine.run_script(project.blender, "retexture/face_patch.py",
                                     [before, views, "head_000", image, patched, *_keep(project)],
                                     log=folder / "face_patch.log")
    if code != 0 or not patched.exists():
        raise StepFailed(f"making the face patch failed (see {folder / 'face_patch.log'})\n{output[-2000:]}")
    (folder / "patch.json").write_text(json.dumps({"version": PATCH_VERSION}), encoding="utf-8")
    data = manifest.load(project.manifest_path)
    data["reference"] = _relative(reference, project.manifest_path)
    data["source"] = _relative(patched, project.manifest_path)
    manifest.save(data, project.manifest_path)
    return {"source": str(patched), "reference": str(reference), "image": str(image),
            "next": "mark the face points on the image, then rerun from survey"}


# How far past the marked eye or mouth the picture is left as rendered, as a share of its size.
KEEP_PAD = (0.35, 0.8)


def _keep(project):
    """`--keep` boxes for face_patch.py: the marked eyes and mouth, with room for lashes and brows,
    in the picture's own 0..1 (v up). None without face points: the patch is then left unfilled."""
    face = manifest.path_of(project.data, "face")
    points = engine._read_json(face, None) if face else None
    if not points or not points.get("size"):
        return []
    width, height = points["size"]
    boxes = []
    for part in FACE_KEYS:
        xy = [points[part][n] for n in FACE_KEYS[part] if isinstance((points.get(part) or {}).get(n), list)]
        if len(xy) < 2:
            continue
        x0, x1, y0, y1 = min(p[0] for p in xy), max(p[0] for p in xy), min(p[1] for p in xy), max(p[1] for p in xy)
        px, py = KEEP_PAD[0] * (x1 - x0), KEEP_PAD[1] * max(y1 - y0, 0.3 * (x1 - x0))
        boxes += ["--keep", ",".join(f"{v:.4f}" for v in ((x0 - px) / width, 1 - (y1 + py) / height, (x1 + px) / width, 1 - (y0 - py) / height))]
    return boxes


def _frame(survey, height):
    """(where the head view is centred, the body's share of the model's height) from a survey; the
    first None without a head to go by.

    The views are framed by shares of the whole model, and horns or a tail stand over the body's
    `height` (orient.head_top): by a share of the body a demon girl's patch was framed up among her
    horns, her mouth at the picture's lower edge. The head view is sized by the body too.
    """
    whole = max(float((survey or {}).get("size", [0, 0, height])[2]), height)
    body = round(height / whole, 3)
    if survey and survey.get("eyes"):
        # the modelled eyes, a little under the face's middle
        return round((survey["eyes"][2] - FACE_BELOW_EYES * height) / whole, 3), body
    if survey and "head" in survey.get("joints", {}):
        # the face, a little above the head joint (a guess where clothing or a tail hides the neck)
        return round((survey["joints"]["head"][2] + FACE_ABOVE_HEAD * height) / whole, 3), body
    return None, body


def _framing(folder):
    """The height (as a share of the model's) the last face patch in this folder centred the head
    view on: as recorded, else read back from that render's camera. None without one."""
    said = engine._read_json(folder / "head_at.json", None)
    if said and isinstance(said.get("head_at"), (int, float)):
        return said["head_at"]
    views = engine._read_json(folder / "views" / "views.json", None)
    try:
        head = next(v for v in views["views"] if v["name"] == "head_000")
        return round(head["matrix_world"][2][3] / views["height"], 3)
    except (TypeError, KeyError, StopIteration, IndexError, ZeroDivisionError):
        return None


def _patched(model):
    """Whether a model already wears a face patch (a duplicate's source is its original's patched copy)."""
    try:
        return any(m.get("name") == "face_patch" for m in paint.read_glb(model)[0].get("materials", []))
    except (paint.NotPaintable, OSError, ValueError):
        return False


# The face's middle sits this share of the height above the head joint (the head bone's base).
FACE_ABOVE_HEAD = 0.04
# ... and this share under the eyes, so the mouth and chin are in the frame too.
FACE_BELOW_EYES = 0.02

FACE_KEYS = {"right_eye": ("inner", "outer", "top", "bottom"), "left_eye": ("inner", "outer", "top", "bottom"),
             "mouth": ("left", "right", "top", "bottom")}


def guess_face(manifest_path, python=None, blender=None):
    """Face points for the character's face patch picture, found by the detector (app/face_guess.py)."""
    project = Project(manifest_path, blender=blender)
    # the picture the patch was made from, kept beside the patched model (a character may use another's patch)
    source = manifest.path_of(project.data, "source")
    folders = [project.out / "face", *([source.parent] if source else [])]
    picture = next((f / "face_patch_head_000.png" for f in folders if (f / "face_patch_head_000.png").exists()), None)
    if picture is None:
        raise StepFailed("no face patch yet: make it first (`face-patch`, or the Face page)")
    try:
        points = face_guess.guess(picture, python=python, cache=REPO / "tools" / "work" / "hf")
    except (face_guess.NoDetector, face_guess.NoFace, RuntimeError) as error:
        raise StepFailed(f"face-guess: {error}")
    return {**points, "image": "face_patch.png"}


def paint_state(manifest_path, blender=None):
    """What the Paint page opens on: the source's paintable pictures, each with its paint if any."""
    project = Project(manifest_path, blender=blender)
    source = manifest.path_of(project.data, "source")
    if source is None or source.suffix.lower() != ".glb" or not source.exists():
        # an FBX or OBJ becomes a .glb with the face patch ("Get it ready")
        return {"paintable": False, "why": "the model isn't a .glb yet: make the face patch first (Face page, or Get it ready)",
                "pictures": []}
    try:
        found = paint.pictures(paint.read_glb(source)[0])
    except (paint.NotPaintable, ValueError, KeyError, IndexError) as error:
        return {"paintable": False, "why": str(error), "pictures": []}
    kept = paint.overlays(manifest.path_of(project.data, "paint"))
    drafts = paint.overlays(_draft_folder(project))
    for picture in found:
        picture["painted"] = picture["overlay"] in kept
        # paint not saved when the page last closed: newer than what is kept, or with nothing kept
        draft = drafts.get(picture["overlay"])
        picture["draft"] = draft is not None and (picture["overlay"] not in kept
                                                  or draft.stat().st_mtime > kept[picture["overlay"]].stat().st_mtime)
    return {"paintable": bool(found), "why": "" if found else "the model has no texture to paint on", "pictures": found,
            "orphans": sorted(set(kept) - {p["overlay"] for p in found}), "old_patch": _old_patch(source)}


# The face patch as scripts/retexture/face_patch.py now makes it: hair lying over the face on its
# own picture (2). cmd_face_patch notes it beside the patched model.
PATCH_VERSION = 2
# What `face-patch` has drawn of the model: the head from the front, which the patch is made of (and
# its depth alone, by which the patch tells what lies hidden), and each part from the front and from
# behind, which a handoff bundle and a tester's report take (`_000`, `_180`). render_views.py draws 34
# views with their depths when not told otherwise, for re-texturing by hand: drawn so for every new
# character, 25 views and 33 depths were read by nothing, 9 of the face patch's 18 seconds and 30 of
# its 38 MB (the fox girl, 2026-10-12). The patched model is the same file byte for byte.
PATCH_VIEW = "head_000"
PATCH_VIEWS = (PATCH_VIEW, "head_180", "torso_000", "torso_180", "hips_000", "hips_180", "legs_000", "legs_180", "face_000")


def _old_patch(source):
    """Whether a model wears a face patch from before hair and face had a picture each: paint on
    the hair there shows on the face behind it, and only a remade patch mends that."""
    if not _patched(source):
        return False
    said = engine._read_json(Path(source).parent / "patch.json", None) or {}
    return said.get("version", 1) < PATCH_VERSION


def _draft_folder(project):
    return project.out / "paint" / "draft"


def save_paint_draft(manifest_path, images, blender=None):
    """Keep paint the user hasn't saved yet, so a closed window or a crash doesn't lose it. Nothing
    else changes: the build doesn't read drafts."""
    project = Project(manifest_path, blender=blender)
    return {"draft": paint.save(_draft_folder(project), manifest.path_of(project.data, "source"), images)}


def save_paint(manifest_path, images, blender=None):
    """Keep the page's paint ({overlay name: PNG data URL or None}) with the character, and mark
    what was built from it stale: the texture goes in at the build step."""
    project = Project(manifest_path, blender=blender)
    source = manifest.path_of(project.data, "source")
    folder = manifest.path_of(project.data, "paint") or project.out / "paint" / "overlays"
    kept = paint.save(folder, source, images)
    shutil.rmtree(_draft_folder(project), ignore_errors=True)  # saved: the draft has nothing more to say
    data = manifest.load(project.manifest_path)
    data["paint"] = _relative(folder, project.manifest_path) if kept else None
    manifest.save(data, project.manifest_path)
    project = Project(manifest_path, blender=blender)
    project.model()  # made now: a picture that won't composite fails here, not minutes into a build
    project.mark_stale("build")
    return {"painted": kept, "next": "rerun from build: the texture goes in there"}


def cmd_paint(args):
    if args.clear:
        project = _project(args)
        kept = paint.overlays(manifest.path_of(project.data, "paint"))
        return save_paint(args.manifest, {name: None for name in kept if name in
                                          {p["overlay"] for p in paint_state(args.manifest)["pictures"]}}, blender=args.blender)
    project = _project(args)
    return {**paint_state(args.manifest, blender=args.blender), "model": str(project.model())}


def cmd_face_guess(args):
    points = guess_face(args.manifest, python=args.python, blender=getattr(args, "blender", None))
    if args.save:
        return {"points": points, **save_face(args.manifest, points, blender=getattr(args, "blender", None))}
    return {"points": points, "next": "check them on the Face page, or keep them with --save"}


def cmd_face(args):
    return save_face(args.manifest, json.loads(Path(args.points).read_text(encoding="utf-8")), blender=args.blender)


def save_face(manifest_path, points, blender=None):
    """Check face points and keep them as <out>/face.json, named in the manifest."""
    # eyes hidden behind a visor or glasses are left out, both of them: the mouth alone is marked
    marked = {key: names for key, names in FACE_KEYS.items() if key == "mouth" or engine.marks_eyes(points)}
    for key, names in marked.items():
        missing = [n for n in names if not isinstance((points.get(key) or {}).get(n), list)]
        if missing:
            raise ValueError(f"{key} is missing {', '.join(missing)}")
    if not points.get("size"):
        raise ValueError("no image size")
    project = Project(manifest_path, blender=blender)
    keep = {"image": points.get("image") or "face_patch.png", "size": points["size"],
            **{key: {n: [round(float(v), 1) for v in points[key][n]] for n in names} for key, names in marked.items()}}
    # how the picture they were marked on frames the head: a patch made again framed otherwise leaves
    # the marks off the face, and the look before a build then says so (preflight._reframed)
    frame = preflight.patch_frame(project)
    if frame is not None:
        keep["frame"] = frame
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
                             str(args.after)], capture_output=True, text=True, errors="replace")
    return {"worse": result.returncode == 1, "report": result.stdout, **compare_rows(args.before, args.after)}


def compare_rows(before, after):
    """The same comparison pose by pose, for the app's table: each pose's stretch and squash in both
    reports, and which way it moved past the noise (compare_checks' own margins)."""
    checks = engine.compare_checks
    old, new = (engine._read_json(path, None) for path in (before, after))
    if not old or not new or "poses" not in old or "poses" not in new:
        return {"poses": [], "unweighted": None}
    rows = []
    for pose in sorted(set(old["poses"]) | set(new["poses"])):
        a, b = old["poses"].get(pose, {}), new["poses"].get(pose, {})
        stretch, squash = [a.get("stretch_cm"), b.get("stretch_cm")], [a.get("squash"), b.get("squash")]
        way = lambda pair, margin, small: ("worse" if checks._grew(*pair, margin, smaller_is_worse=small)  # noqa: E731
                                           else "better" if checks._grew(*pair[::-1], margin, smaller_is_worse=small) else "")
        rows.append({"pose": pose, "stretch": stretch, "squash": squash,
                     "stretch_way": way(stretch, checks.STRETCH_NOISE, False), "squash_way": way(squash, checks.SQUASH_NOISE, True)})
    return {"poses": rows, "unweighted": [old.get("unweighted"), new.get("unweighted")]}


# How a build looked to the user: the two answers a note can carry.
VERDICTS = ("right", "wrong")


def duplicate(manifest_path, name):
    """The character again under another name, to build one choice both ways (iris or sliding eyes,
    a bigger mouth) and compare them instead of writing over one manifest. Same model files; its
    own output folder, face points and paint; nothing built yet."""
    if not name or Path(name).name != name or not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError("the name is a folder name: letters, digits, _ and -")
    project = Project(manifest_path)
    out = REPO / "out" / name
    path = out / "character.json"
    if path.exists():
        raise ValueError(f"{name} exists already")
    out.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in json.loads(json.dumps(project.data)).items() if not k.startswith("_")}
    data["name"], data["out_dir"] = name, None
    for key in ("source", "reference", "rig"):
        if data.get(key):
            data[key] = _relative(manifest.path_of(project.data, key), path)
    face = manifest.path_of(project.data, "face")
    if face and face.exists():
        shutil.copy2(face, out / "face.json")
        data["face"] = "face.json"
    kept = paint.overlays(manifest.path_of(project.data, "paint"))
    data["paint"] = None
    if kept:
        (out / "paint" / "overlays").mkdir(parents=True)
        for overlay in kept.values():
            shutil.copy2(overlay, out / "paint" / "overlays" / overlay.name)
        data["paint"] = str(Path("paint") / "overlays")
    manifest.save(data, path)
    # the listing names the pieces on the Parts page, and the survey says what "auto" means: both are the model's
    for extra in ("pieces.json", "pieces.glb"):
        if (project.out / extra).exists():
            shutil.copy2(project.out / extra, out / extra)
    return {"manifest": str(path), "name": name, "from": project.name, "next": f"python -m app run {path}"}


def rename(manifest_path, name):
    """Another name for a character: in the app's list and inside the exported file. Its builds stay
    where they are (the output folder keeps its old name, now written into the manifest)."""
    if not name or Path(name).name != name or not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError("the name is a folder name: letters, digits, _ and -")
    project = Project(manifest_path)
    if any(_name_of(p) == name for p in engine.characters() if p.resolve() != project.manifest_path):
        raise ValueError(f"{name} exists already")
    data = manifest.load(project.manifest_path)
    if not data.get("out_dir"):
        data["out_dir"] = _relative(project.out, project.manifest_path)
    was, data["name"] = data["name"], name
    manifest.save(data, project.manifest_path)
    return {"name": name, "was": was, "next": "rebuild from build for the new name to be in the .vrm"}


def _name_of(path):
    return (engine._read_json(path, None) or {}).get("name")


def cmd_rename(args):
    return rename(args.manifest, args.name)


def cmd_hide(args):
    return engine.hide_character(args.manifest, hide=not args.show)


def cmd_delete(args):
    return engine.delete_character(args.manifest, dry=args.dry)


def cmd_duplicate(args):
    return duplicate(args.manifest, args.name)


# Screenshots a user attaches from Warudo: kinds, and the most one note may carry (bytes).
SCREENSHOT_KINDS = {"image/png": ".png", "image/jpeg": ".jpg"}
SCREENSHOT_MAX = 8 * 1024 * 1024


def save_feedback(manifest_path, note, images=(), checks=(), verdict=None):
    """What the user saw in Warudo, kept with the exact .vrm it was about. Returns the entry.

    `images` are data URLs (PNG or JPEG) from the page. Feedback given earlier was lost between
    sessions or tied to the wrong build; each note now sits in its own folder beside the build.
    `checks` are the poses tried that looked right, `verdict` "right" or "wrong" for the whole
    build: kept as their own fields, so the app can say which builds still wait for a look.
    """
    import base64
    project = Project(manifest_path)
    if verdict not in (None, *VERDICTS):
        raise ValueError(f"verdict is one of {', '.join(VERDICTS)}")
    checks = [str(c) for c in checks]
    if not (note or "").strip() and not images and not checks and verdict is None:
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
             "vrm": str(project._last_vrm() or ""), "screenshots": saved, "checks": checks, "verdict": verdict}
    (folder / "feedback.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return dict(entry, folder=str(folder))


def list_feedback(manifest_path):
    """Every note given on this character, newest first, with its folder."""
    root = Project(manifest_path).out / "feedback"
    entries = []
    for path in sorted(root.glob("*/feedback.json"), reverse=True) if root.exists() else []:
        # notes from before the ticks and the verdict were kept apart carry neither
        entries.append({"checks": [], "verdict": None, **json.loads(path.read_text(encoding="utf-8")), "folder": str(path.parent)})
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
        # a step that failed has its log there and nowhere else: "it won't build" came with no build log
        failed = (entry.get("last_failure") or {}).get("log")
        if entry.get("status") != "done" and failed and Path(failed).exists():
            shutil.copy2(failed, folder / f"{name}_failed.log")
    # the model as the face patch's views drew it, front and back: before a first build there are no
    # renders, and the bundle said "hat" and "A-pose" of a character nobody reading it could see
    views = sorted(p for p in (project.out / "face" / "views").glob("color_*.png") if p.stem.endswith(("_000", "_180")))
    if views:
        (folder / "model").mkdir()
        for picture in views:
            shutil.copy2(picture, folder / "model" / picture.name)
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


def cmd_report(args):
    if args.address is not None:
        return report.use_address(args.address)
    if args.check is not None:
        return report.unpack(args.check) if args.unpack else report.check(args.check)
    if args.send:
        return report.send(args.manifest, note=args.note, pictures=args.pictures, model=args.model, about=args.about,
                           blender=getattr(args, "blender", None))
    return report.make(args.manifest, note=args.note, pictures=args.pictures, model=args.model, about=args.about,
                       blender=getattr(args, "blender", None))


COMMANDS = {"report": cmd_report, "init": cmd_init, "status": cmd_status, "preflight": cmd_preflight, "run": cmd_run, "set": cmd_set, "joint": cmd_joint,
            "pieces": cmd_pieces, "piece": cmd_piece, "render": cmd_render, "portraits": cmd_portraits, "sheet": cmd_sheet, "inside": cmd_inside, "movement": cmd_movement, "springs": cmd_springs,
            "face-patch": cmd_face_patch, "face": cmd_face, "face-guess": cmd_face_guess, "paint": cmd_paint, "duplicate": cmd_duplicate, "rename": cmd_rename, "hide": cmd_hide, "delete": cmd_delete, "compare": cmd_compare, "approve": cmd_approve, "bundle": cmd_bundle,
            "runs": cmd_runs, "disk": cmd_disk, "prune": cmd_prune, "prepare": cmd_prepare, "setup": cmd_setup, "inputs": cmd_inputs,
            "regress": cmd_regress, "vendor": cmd_vendor, "update": cmd_update, "pack": cmd_pack, "detector": cmd_detector, "warudo": cmd_warudo}


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


if __name__ == "__main__":  # `python -m app.cli` too: it exited 0 having done nothing
    sys.exit(main())
