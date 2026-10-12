"""What the app makes of a model handed over otherwise: turned, in centimetres, in parts, cut along
every edge, decimated, in another file format, with its arms down, without its picture.

    python -m app inputs <manifest> [--only name,name] [--rigged x.fbx] [--keep]

Testers' models come every one of those ways, and each used to be found out one report at a time.
This makes the copies of one character's model (scripts/make_variants.py, one change a copy), builds
the model itself and every copy in scratch folders under out/inputs/<name>/, and holds each copy
against the model: its rig check, exported file, movement and spring checks within the noise between
two builds (`same`), or stopped before the build by a line of the look before a build
(app/preflight.py) or by a step that says why, with a code (`stopped`). What is left is the fault this
looks for: a build that finished and is worse than the model's own (`worse`), most of all one whose
own verdict says "ready" all the same (`quietly_worse`: no word of it anywhere), or one that fell over
with no code (`failed`).

No face points are carried over: a copy is built as a new character is before its face is marked, and
so is the model it is held against. One Blender at a time; a minute or two a copy.
"""

import json
import shutil
import time
from pathlib import Path

from . import engine, errors
from .engine import REPO, Project, StepFailed, manifest

# What a finished copy is held against the model's own build by: `engine.changes`' lines of these
# kinds. (Its movement lines name pieces by vertex number, which a file saved again renumbers: the
# pieces the movement check flags are counted instead.)
JUDGED = ("rig check", "exported file", "physics")


def run(manifest_path, only=None, rigged=None, keep=False, on_event=None, blender=None):
    """Make the copies of the character's model, build each, and say how each came out:
    {"name", "model", "baseline", "rows", "quietly_worse", "failed", "not_as_expected"}."""
    emit = on_event or (lambda event: None)
    base = Project(manifest_path, blender=blender)
    # the model as it was given, before the app's own face patch
    model = manifest.path_of(base.data, "reference") or manifest.path_of(base.data, "source")
    if model is None or not model.is_file():
        raise ValueError(f"{base.name} names no model file that is there: {model}")
    root = REPO / "out" / "inputs" / base.name
    files = root / "files"
    files.mkdir(parents=True, exist_ok=True)
    emit({"step": "copies", "status": "running", "reason": f"making the copies of {model.name} in {files}"})
    args = [model, files, *(["--rigged", rigged] if rigged else []), *(["--only", ",".join(only)] if only else [])]
    code, output = engine.run_script(base.blender, "make_variants.py", args, log=root / "make_variants.log")
    if code != 0:
        raise StepFailed(f"the copies could not be made (see {root / 'make_variants.log'})\n{output[-2000:]}")
    made = json.loads((files / "variants.json").read_text(encoding="utf-8"))["variants"]
    emit({"step": "copies", "status": "done", "reason": f"{len(made)} copies"})

    baseline = _built(root, "model_itself", model, base, None, emit, keep)
    reports = baseline.pop("_reports")
    rows = []
    for variant in made:
        row = _built(root, variant["name"], Path(variant["file"]), base, reports if baseline["outcome"] == "built" else None, emit, keep)
        row.pop("_reports", None)
        row.update(what=variant["what"], expects=variant["expects"], as_expected=_as_expected(row, variant["expects"]))
        rows.append(row)
    if not keep:
        shutil.rmtree(files, ignore_errors=True)  # the copies themselves: each as big as the model
    found = {"name": base.name, "model": str(model), "baseline": baseline, "rows": rows,
             # worse than the model's own build and its verdict says "ready": nothing anywhere says so
             "quietly_worse": [r["name"] for r in rows if r["outcome"] == "worse" and r["ready"]],
             # worse, and its own rig check or verdict says something the model's doesn't
             "worse_and_said": [r["name"] for r in rows if r["outcome"] == "worse" and not r["ready"]],
             "failed": [r["name"] for r in rows if r["outcome"] == "failed"],
             "not_as_expected": [r["name"] for r in rows if not r["as_expected"]]}
    (root / "inputs.json").write_text(json.dumps(found, indent=2, default=str), encoding="utf-8")
    return found


def _as_expected(row, expects):
    """Whether a copy came out as its maker said it should: the same as the model, or stopped with
    one of the codes named. Stopped where "same" was hoped for is the rule kept (it is said, not
    quiet) and an intake that could do better: not as expected."""
    if expects == "same":
        return row["outcome"] == "same"
    return row["outcome"] == "stopped" and bool(set(expects.split()) & set(row["codes"]))


def _character(folder, model, base, name):
    """A new character for `model` in `folder`, with the base character's options and nothing else
    of it: no face points, no roles, no joints (they are of the base's own mesh)."""
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    data = manifest.new(str(model), f"{base.name}_{name}", base.data.get("author") or "Unknown")
    data["options"] = json.loads(json.dumps(base.data["options"]))
    data["out_dir"] = str(folder)
    rig = manifest.rig_of(model)
    if rig is not None:
        data["rig"] = str(rig)
    manifest.save(data, folder / "character.json")
    return Project(folder / "character.json", blender=base._blender)


def _built(root, name, model, base, baseline, emit, keep):
    """One model surveyed, looked over and, when nothing stops it, built; how it came out."""
    started = time.time()
    project = _character(root / name, model, base, name)
    row = {"name": name, "file": str(model), "outcome": None, "codes": [], "said": [], "worse": [], "better": [],
           "check_failed": None, "ready": None, "_reports": {}}
    emit({"step": name, "status": "running", "reason": f"building {model.name}"})
    try:
        if not project.improving:
            project.run(only="survey", on_event=emit)
        survey = project.survey() or {}
        row["survey"] = {"triangles": survey.get("triangles"), "pose": (survey.get("pose") or {}).get("kind"),
                         "orientation": survey.get("orientation"), "texture": survey.get("texture"),
                         "pieces": survey.get("pieces")}
        stops = [line for line in project.preflight()["lines"] if line["kind"] == "stop"]
        if stops:
            row.update(outcome="stopped", codes=[line["code"] for line in stops], said=[line["text"] for line in stops])
        else:
            status = project.run(start="build", on_event=emit)
            row.update(outcome="built", check_failed=status["check_failed"], ready=status["verdict"]["ready"],
                       issues=[text for _, text in status["verdict"]["issues"]], _reports=project.reports())
    except StepFailed as error:
        failure = errors.failure(getattr(error, "step", None), error.known, str(error))
        # a step that says why, with a code, has stopped the build; one with no code has fallen over
        row.update(outcome="stopped" if failure["code"] else "failed", codes=[failure["code"]] if failure["code"] else [],
                   said=[f"{failure['what']}{' (' + failure['detail'] + ')' if failure['detail'] else ''}"])
    if row["outcome"] == "built" and baseline is not None:
        row.update(_held(baseline, row["_reports"]))
        row["outcome"] = "worse" if row["worse"] else "same"
    row["seconds"] = round(time.time() - started)
    if not keep:
        for file in (p for p in (root / name).rglob("*") if p.is_file() and p.suffix.lower() in engine.MODELS):
            file.unlink()
    emit({"step": name, "status": "done", "reason": f"{row['outcome']}{': ' + ', '.join(row['codes']) if row['codes'] else ''}"})
    return row


def _held(baseline, reports):
    """{"worse", "better"} of a copy's build against the model's own, by what can be compared across
    files that number their vertices otherwise."""
    changed = engine.changes(baseline, reports)
    worse = [line for line in changed["worse"] if line.startswith(JUDGED)]
    better = [line for line in changed["better"] if line.startswith(JUDGED)]
    flagged = lambda files: len((engine._read_json(files.get("movement"), None) or {}).get("pieces", []))  # noqa: E731
    was, now = flagged(baseline), flagged(reports)
    if now > was:
        worse.append(f"movement: {now} pieces flagged, the model's own build has {was}")
    elif now < was:
        better.append(f"movement: {now} pieces flagged, the model's own build has {was}")
    return {"worse": worse, "better": better}
