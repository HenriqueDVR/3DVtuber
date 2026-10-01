"""A character's build, step by step, from its manifest. Plain Python: each step is one Blender run.

The manifest (addon/vtuber_maker/pipeline/manifest.py) holds what the user decided; this holds
what happened. Each run writes into its own folder under the project's output folder, so
nothing earlier is overwritten, and `state.json` there records which file every step last
produced: a later step reads that, never "the newest file" (an intermediate A-pose can be newer
than a finished VRM). A step can be rerun alone, or the chain from any step on.

Only one Blender runs at a time (two at once hung the machine on 2026-09-23), and a run that
logs "Malloc returns null" is stopped: out of memory, Blender spins instead of exiting.
"""

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from . import errors

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
WINDOWS_BLENDER = Path(r"C:\Program Files\Blender Foundation")
LOCK = REPO / "out" / ".blender.lock"
# Longest a single step may run (s); the slowest real steps take about 3 minutes.
STEP_TIMEOUT = 30 * 60
OUT_OF_MEMORY = "Malloc returns null"
# Oldest Blender the pipeline runs on (EEVEE Next, extensions, the VRM Add-on's 4.2 builds), and the
# series the full test suite has passed on (tests/run_matrix.py). Others at or above the minimum run
# with a warning; keep in step with addon/vtuber_maker/pipeline/compat.py.
MIN_BLENDER = (4, 2, 0)
TESTED = {"4.2", "4.5", "5.2"}
# Under this many triangles a face has too few vertices to bend without denting.
COARSE_FACE = 3_000


def _manifest_module():
    # manifest.py is plain Python, but its package imports bpy: load the file on its own
    path = REPO / "addon" / "vtuber_maker" / "pipeline" / "manifest.py"
    spec = importlib.util.spec_from_file_location("vtuber_manifest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manifest = _manifest_module()


def _compare_module():
    # the check comparison the command-line tool prints, reused for "since the approved build"
    spec = importlib.util.spec_from_file_location("compare_checks", SCRIPTS / "compare_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_checks = _compare_module()
# The reports a build is judged by, kept as they were when the user said a build was good.
APPROVED = ("check", "vrm_check", "movement", "springs")
# Spring findings this much worse than the approved build's (cm) count as a change.
SPRING_NOISE = 1.0


class StepFailed(Exception):
    """`known`: the matching errors.KNOWN entry ({"code", "what", "fix", "options"}), when there is one."""

    def __init__(self, message, known=None):
        super().__init__(message)
        self.known = known


class Cancelled(Exception):
    """The user stopped the run. Not a StepFailed: a soft step must not carry on past it."""


# Set by the app's Stop button: the running Blender is killed at its next poll, and the chain
# stops before its next step. Only this process's runs; a CLI run elsewhere is untouched.
CANCEL = threading.Event()
CANCELLED = -2


# (name, label, what it writes): "vrm" = a .blend and a .vrm, "blend" = a .blend only,
# "json" = a report only. Order is the build order that produced the model A and model B candidates.
STEPS = [
    ("survey", "surveying the model", "json"),
    ("build", "rigging", "vrm"),
    ("hair", "hair weights", "vrm"),
    ("ahoge", "ahoge spring", "vrm"),
    ("face_tracking", "face tracking", "vrm"),
    ("loose", "loose pieces", "vrm"),
    ("fingers", "finger bones", "vrm"),
    ("pieces", "clothing pieces", "vrm"),
    ("finger_guides", "finger guides", "blend"),
    ("tpose", "T-pose", "vrm"),
    ("eyes", "eye tracking", "vrm"),
    ("mouth", "mouth shapes", "vrm"),
    ("check", "checking the rig", "json"),
    ("vrm_check", "checking the exported VRM", "json"),
    # what check_avatar can't see: pieces left behind as limbs move (on the modelled pose), and the
    # springs running as apps run them (on the finished rig); the verdict takes them in
    ("movement", "checking what moves with the body", "json"),
    ("springs", "running the physics", "json"),
]
NAMES = [name for name, _, _ in STEPS]
# Steps that exit nonzero when there is nothing for them to do (no ahoge, no loose strap):
# the chain carries on with the file before them.
OPTIONAL = {"ahoge", "loose", "fingers"}
# Checks that run after the build and report; a check that can't run doesn't take the avatar away.
CHECKS = {"movement", "springs"}
# Face features that, when their own checks refuse, leave the avatar without them rather than
# without an avatar: model C's eye step refused (iris over the lids) and the chain ended with no .vrm.
SOFT = {"face_tracking", "eyes", "mouth"}


def find_blender(given=None):
    """The Blender to run: --blender, $BLENDER, PATH, else the best installed one.

    Installed ones are ranked tested series first, then newest: a newer untested Blender beside the
    tested one must not silently take over the user's builds. Below MIN_BLENDER is never picked.
    """
    for candidate in (given, os.environ.get("BLENDER"), shutil.which("blender")):
        if candidate and Path(candidate).exists():
            return str(candidate)
    found = [(_folder_version(p), p) for p in installed_blenders()]
    found = [(v, p) for v, p in found if v and v >= MIN_BLENDER]
    if found:
        return str(max(found, key=lambda vp: (".".join(map(str, vp[0][:2])) in TESTED, vp[0]))[1])
    raise FileNotFoundError(f"Blender {'.'.join(map(str, MIN_BLENDER[:2]))} or later not found: "
                            "put it on PATH, set $BLENDER, or pass --blender")


def installed_blenders():
    """Every blender.exe in the usual Windows places: the installer's folder, Steam, the Store."""
    roots = [WINDOWS_BLENDER, Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Steam" / "steamapps" / "common",
             Path(os.environ.get("LOCALAPPDATA", "")) / "Programs"]
    found = []
    for root in roots:
        if root.is_dir():
            found += [p for p in root.glob("*/blender.exe") if "blender" in p.parent.name.lower()]
    return found


def _folder_version(path):
    # "Blender 4.2", "blender-4.5.3-windows-x64": the folder names the series, which is what matters
    found = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", Path(path).parent.name)
    return tuple(int(g or 0) for g in found.groups()) if found else None


class Project:
    """One character: its manifest, its output folder and what the steps produced."""

    def __init__(self, manifest_path, blender=None):
        self.manifest_path = Path(manifest_path).resolve()
        self.data = manifest.load(self.manifest_path)
        self.name = self.data["name"]
        self.out = manifest.path_of(self.data, "out_dir") or REPO / "out" / self.name
        self.state_path = self.out / "state.json"
        self.state = _load_state(self.state_path)
        self._blender = blender

    @property
    def blender(self):
        if self._blender is None:
            self._blender = find_blender()
        return self._blender

    # ----- what the steps will do

    def survey(self):
        entry = self.state["steps"].get("survey")
        if not entry or not entry.get("json") or not Path(entry["json"]).exists():
            return None
        return json.loads(Path(entry["json"]).read_text(encoding="utf-8"))

    def options(self):
        """The manifest's options with every "decide from the survey" filled in."""
        options = dict(self.data["options"])
        survey = self.survey() or {"options": {}, "triangles": 0, "face_patch": False}
        found = survey["options"]
        if options["rigid_head"] is None:
            options["rigid_head"] = bool(found.get("rigid_head"))
        if options["cloth"] is None:
            options["cloth"] = bool(found.get("cloth_physics"))
        if options["mouth_colour"] is None:
            options["mouth_colour"] = found.get("mouth_color")
        if options["mouth_only"] is None:
            options["mouth_only"] = survey["triangles"] < COARSE_FACE or options["rigid_head"]
        if options["eyes"] is None:
            # with face points, eyes move by default: modelled eyes as irises, painted ones as patches
            options["eyes"] = ("iris" if survey.get("parts", {}).get("eye_inserts") else "slide") if self.data.get("face") else "none"
        if options["free_mouth"] is None:
            # a coarse or solid face dents when ARKit shapes bend it, so the mouth moves on its own
            options["free_mouth"] = bool(self.data.get("face") and options["mouth_only"] and survey.get("face_patch"))
        return options

    def plan(self):
        """[(step, will it run, why not)] for the whole chain."""
        o = self.options()
        face = bool(self.data.get("face"))
        pieces = bool(manifest.pieces_for(self.data, "pieces"))
        rigged = bool(self.data.get("rig"))
        skip = {
            "ahoge": (not o["ahoge"] or o["rigid_head"]) and "no ahoge wanted, or a rigid head",
            "face_tracking": (not face or not o["face_tracking"]) and "no face points, or face tracking off",
            "loose": not o["loose_pieces"] and "loose-piece repair off",
            "fingers": (o["fingers"] == "none" and "fingers off") or (rigged and "the adopted rig has its own fingers"),
            "pieces": not pieces and "no clothing pieces named in the manifest",
            "finger_guides": (o["fingers"] != "guides" and "finger guides off") or (rigged and "the adopted rig has its own fingers"),
            "tpose": o["tpose"] == "none" and "T-pose off",
            "eyes": (o["eyes"] == "none" or not face) and "no eye tracking, or no face points",
            "mouth": (not o["mouth_fix"] or not face) and "no mouth fix asked for",
            "vrm_check": o["tpose"] == "none" and "no final VRM",
        }
        return [(name, not skip.get(name), skip.get(name) or "") for name in NAMES]

    def status(self):
        steps = []
        # a soft step that failed is reported, and the build carries on without it
        for name, runs, why in self.plan():
            entry = self.state["steps"].get(name, {})
            steps.append({"step": name, "runs": runs, "skip_reason": why, "status": entry.get("status", "not run"),
                          **{k: entry[k] for k in ("blend", "vrm", "json", "log", "finished", "seconds", "report", "reason",
                                                     "known", "last_failure")
                             if k in entry}})
        last = self.latest()
        check = self.state["steps"].get("check", {})
        failed = _read_json(check.get("json"), {}).get("failed")
        return {"name": self.name, "manifest": str(self.manifest_path), "out": str(self.out),
                "latest_blend": str(last) if last else None, "options": self.options(), "steps": steps,
                "check_failed": failed, "verdict": self.verdict(failed), "approved": self.approved(),
                "since_approved": self.since_approved()}

    def approve(self, note=""):
        """Keep the current build's reports as the ones later builds are compared with."""
        steps = self.state["steps"]
        if steps.get("vrm_check", {}).get("status") != "done":
            raise StepFailed("approve: build to the end first, a half-built avatar isn't a baseline")
        folder = self.out / "approved"
        folder.mkdir(parents=True, exist_ok=True)
        kept = {}
        for name in APPROVED:
            source = steps.get(name, {}).get("json")
            if steps.get(name, {}).get("status") == "done" and source and Path(source).exists():
                shutil.copy2(source, folder / f"{name}.json")
                kept[name] = source
        meta = {"when": datetime.now().isoformat(timespec="seconds"), "vrm": str(self._last_vrm() or ""),
                "note": note, "reports": kept}
        _write_json(folder / "approved.json", meta)
        return meta

    def approved(self):
        return _read_json(self.out / "approved" / "approved.json", None)

    def since_approved(self):
        """{"worse": [...], "better": [...]} against the approved build, or None without one.

        Compares only reports both builds have, and only once this build has them: a half-rebuilt
        chain is still carrying the old reports' numbers or none.
        """
        if self.approved() is None:
            return None
        folder, steps = self.out / "approved", self.state["steps"]
        worse, better = [], []
        for name in ("check", "vrm_check"):
            before, now = _read_json(folder / f"{name}.json", None), _read_json(steps.get(name, {}).get("json"), None)
            if before and now and steps[name].get("status") == "done":
                w, b = compare_checks.changes(before, now)
                label = "rig check" if name == "check" else "exported file"
                worse += [f"{label}, {line}" for line in w]
                better += [f"{label}, {line}" for line in b]
        before, now = _read_json(folder / "movement.json", None), _read_json(steps.get("movement", {}).get("json"), None)
        if before is not None and now is not None:
            named = lambda report: {str(r.get("name") or r["label"]) for r in report.get("pieces", [])}
            worse += [f"movement: {n} newly flagged" for n in sorted(named(now) - named(before))]
            better += [f"movement: {n} no longer flagged" for n in sorted(named(before) - named(now))]
        before, now = _read_json(folder / "springs.json", None), _read_json(steps.get("springs", {}).get("json"), None)
        if before is not None and now is not None:
            for key, field, what in (("sagging", "sag_cm", "droop at rest"), ("coming_away", "gap_cm", "left behind by a swing")):
                old = max((r[field] for r in before.get(key, [])), default=0.0)
                new = max((r[field] for r in now.get(key, [])), default=0.0)
                if new - old > SPRING_NOISE:
                    worse.append(f"physics: worst {what} {old} -> {new} cm")
                elif old - new > SPRING_NOISE:
                    better.append(f"physics: worst {what} {old} -> {new} cm")
        return {"worse": worse, "better": better}

    def verdict(self, failed=None):
        """{"ready", "issues", "looks"}: what must be fixed (the rig check's failures, features built
        without) and what is worth a look (the movement and spring checks, which are heuristics: on a
        T-pose, ribbons resting on a shoulder read as left behind by the arm), each with its page."""
        steps = self.state["steps"]
        if steps.get("vrm_check", {}).get("status") != "done":
            return {"ready": False, "issues": [("build", "not built to the end yet")], "looks": []}
        issues = [("review", f"rig check: {f}") for f in (failed or [])]
        for name in sorted(SOFT):
            if steps.get(name, {}).get("status") == "failed":
                issues.append(("build", f"built without {name.replace('_', ' ')}"))
        looks = []
        # the loose step measures itself and undoes a tidy-up that tore the rig: the avatar is safe,
        # but an armpit it meant to join may open
        loose = steps.get("loose", {})
        if loose.get("status") == "done":
            looks += [("build", f"loose pieces: {line}") for line in (loose.get("report") or {}).get("undone", [])]
        movement = _read_json(steps.get("movement", {}).get("json"), {})
        for row in movement.get("pieces", []):
            kinds = sorted({f["kind"] for f in row["findings"]})
            looks.append(("movement", f"{row.get('name') or row.get('where') or 'piece ' + str(row['label'])}: "
                                       f"{' and '.join(kinds)} ({row['findings'][0]['pose']})"))
        springs = _read_json(steps.get("springs", {}).get("json"), {})
        for s in springs.get("sagging", []):
            looks.append(("movement", f"{s['chain']} droops {s['sag_cm']} cm standing still"))
        for c in springs.get("coming_away", []):
            looks.append(("movement", f"a piece of {c['count']} vertices is left {c['gap_cm']} cm behind what it sits on"))
        for name in CHECKS:
            if steps.get(name, {}).get("status") == "failed":
                looks.append(("movement", f"the {name} check couldn't run"))
        if not self.data.get("face"):
            # without face points the build still finishes, and nothing said the face was dead
            looks.insert(0, ("face", "no face points: face and eye tracking were skipped, so only the build's basic "
                                     "blink and mouth ship (no ARKit shapes, eyes don't move); mark them on the Face page"))
        changed = self.since_approved()
        for line in (changed or {}).get("worse", []):
            looks.append(("build", f"worse than the approved build: {line}"))
        return {"ready": not issues, "issues": issues, "looks": looks}

    def latest(self, before=None):
        """The .blend the last finished step produced, looking only at steps before `before`."""
        names = NAMES[:NAMES.index(before)] if before else NAMES
        for name in reversed(names):
            entry = self.state["steps"].get(name, {})
            if entry.get("status") == "done" and entry.get("blend") and Path(entry["blend"]).exists():
                return Path(entry["blend"])
        return None

    # ----- history

    def runs(self):
        """Every run folder under runs/, newest first: its steps, its last .vrm and its rig check."""
        approved = (self.approved() or {}).get("vrm")
        approved = Path(approved).parent.resolve() if approved else None
        current = {Path(e[k]).parent.resolve() for e in self.state["steps"].values() for k in ("blend", "json", "vrm") if e.get(k)}
        found = []
        for folder in sorted((self.out / "runs").glob("*"), reverse=True):
            if not folder.is_dir():
                continue
            logs = sorted(folder.glob("[0-9][0-9]_*.log"))
            vrms = sorted(folder.glob("[0-9][0-9]_*.vrm"))
            checks = {p.stem[3:]: p for p in folder.glob("[0-9][0-9]_*check.json")}
            check = _read_json(checks.get("check"), None) or {}
            try:
                when = datetime.strptime(folder.name, "%Y%m%d-%H%M%S").isoformat()
            except ValueError:
                when = None
            found.append({"run": folder.name, "folder": str(folder), "when": when, "steps": [p.stem[3:] for p in logs],
                          "vrm": str(vrms[-1]) if vrms else None,
                          "check": str(checks["check"]) if "check" in checks else None,
                          "vrm_check": str(checks["vrm_check"]) if "vrm_check" in checks else None,
                          "failed": check.get("failed"), "approved": folder.resolve() == approved,
                          "current": folder.resolve() in current})
        return found

    def summary(self):
        """How far the build got and the newest render: a card on the app's Characters page."""
        steps = self.state["steps"]
        statuses = {name: e.get("status") for name, e in steps.items()}
        renders = sorted((p for p in (self.out / "renders").glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
        pictures = sorted(renders[-1].glob("*.png"), key=lambda p: ("front" not in p.stem, p.name)) if renders else []
        return {"surveyed": statuses.get("survey") == "done", "built": statuses.get("vrm_check") == "done",
                "stale": sorted(n for n, s in statuses.items() if s == "stale"),
                "failed": sorted(n for n, s in statuses.items() if s == "failed"),
                "finished": max((e.get("finished") or "" for e in steps.values()), default="") or None,
                "approved": self.approved() is not None, "picture": str(pictures[0]) if pictures else None}

    # ----- running

    def run(self, start=None, stop=None, only=None, on_event=None):
        """Run the chain (or `only` one step); returns the status. Raises StepFailed."""
        emit = on_event or (lambda event: None)
        first = NAMES.index(only or start) if (only or start) else 0
        last = NAMES.index(only or stop) if (only or stop) else len(NAMES) - 1
        folder = self.out / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
        folder.mkdir(parents=True, exist_ok=True)
        with _Lock():
            # what built this run, so "it was good last week" can be traced to a Blender, add-on and commit
            made_by = provenance(self.blender, self.manifest_path)
            _write_json(folder / "run.json", {**made_by, "steps": NAMES[first:last + 1]})
            version = made_by["blender_version"]
            if version and tuple(map(int, version.split("."))) < MIN_BLENDER:
                known = errors.describe("VM-BLENDER-OLD")
                raise StepFailed(f"Blender {version} [VM-BLENDER-OLD] {known['fix']}", known)
            for warning in made_by["warnings"]:
                emit({"step": NAMES[first], "status": "warning", "reason": warning})
            for index in range(first, last + 1):
                name = NAMES[index]
                if CANCEL.is_set():
                    raise Cancelled(f"stopped by the user before {name}")
                if name != "survey" and self.survey() is None:
                    raise StepFailed("run the survey step first")
                # planned afresh each time: the survey decides what later steps do
                _, runs, why = self.plan()[index]
                if not runs:
                    self._record(name, {"status": "skipped", "reason": why})
                    emit({"step": name, "status": "skipped", "reason": why})
                    continue
                emit({"step": name, "status": "running"})
                try:
                    entry = self._run_step(name, index, folder)
                except Cancelled:
                    # not recorded: the step keeps what it last finished, which later steps may still read
                    emit({"step": name, "status": "cancelled"})
                    raise
                except StepFailed as error:
                    failure = {"reason": str(error)[-1500:], "finished": datetime.now().isoformat(timespec="seconds"),
                               **({"known": error.known} if error.known else {})}
                    if name not in SOFT | CHECKS:
                        # the step keeps what it last finished (later steps may read it); the failure
                        # rides beside it so the app can say what went wrong and what to do
                        self._note_failure(name, failure, folder / f"{index:02d}_{name}.log")
                        raise
                    entry = {"status": "failed", **failure}
                    # the reason keeps only its end, where the log's name no longer is
                    log = folder / f"{index:02d}_{name}.log"
                    if log.exists():
                        entry["log"] = str(log)
                self._record(name, entry)
                emit({"step": name, **entry})
        return self.status()

    def _run_step(self, name, index, folder):
        stem = folder / f"{index:02d}_{name}"
        kind = dict((n, k) for n, _, k in STEPS)[name]
        fresh = name in ("survey", "build", "vrm_check")
        # the movement check reads the modelled pose: in a baked T-pose a sleeve left at the ribs
        # is no longer near the arm
        blend_in = None if fresh else self.latest(before="tpose" if name == "movement" else name)
        if not fresh and blend_in is None:
            raise StepFailed(f"{name}: no earlier step has produced a .blend yet")
        args, extra = self._arguments(name, stem, blend_in)
        log = stem.with_suffix(".log")
        started = time.time()
        code = _blender(self.blender, blend_in, args, log, extra)
        if code == CANCELLED:
            raise Cancelled(f"{name}: stopped by the user")
        entry = {"status": "done", "log": str(log), "input": str(blend_in) if blend_in else None,
                 "finished": datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - started)}
        output = log.read_text(encoding="utf-8", errors="replace")
        if kind == "json":
            entry["json"] = str(stem.with_suffix(".json"))
        else:
            entry["blend"] = str(stem.with_suffix(".blend"))
            if kind == "vrm":
                entry["vrm"] = str(stem.with_suffix(".vrm"))
        made = Path(entry.get("blend") or entry["json"]).exists()
        if code != 0 or not made:
            if name in OPTIONAL:
                return {"status": "nothing to do", "log": str(log), "finished": entry["finished"]}
            known = errors.classify(name, output)
            said = f" [{known['code']}] {known['what']}; {known['fix']}" if known else ""
            if OUT_OF_MEMORY in output:
                raise StepFailed(f"{name}: Blender ran out of memory (see {log}){said}", known)
            raise StepFailed(f"{name}: failed with exit code {code} (see {log}){said}\n{_tail(output)}", known)
        # checked here, at the step that made it: a truncated or half-exported file otherwise
        # surfaces steps later as someone else's traceback
        problem = output_problem(entry)
        if problem:
            known = errors.describe("VM-BAD-OUTPUT")
            raise StepFailed(f"{name}: {problem} (see {log}) [VM-BAD-OUTPUT] {known['fix']}", known)
        for tag in ("VTUBER_MAKER_REPORT", "VTUBER_MAKER_PIECES"):
            report = _tagged(output, tag)
            if report is not None:
                entry["report"] = report
        if name in ("check", "vrm_check"):
            entry["report"] = {k: v for k, v in _read_json(entry["json"], {}).items() if k in ("failed", "worst", "unweighted", "spots")}
        if name in CHECKS:
            # the Movement page shows the latest of each
            shutil.copy2(entry["json"], self.out / f"{name}.json")
        return entry

    def _arguments(self, name, stem, blend_in):
        """(script and its arguments, Blender arguments before -P)."""
        o, data = self.options(), self.data
        blend, vrm = stem.with_suffix(".blend"), stem.with_suffix(".vrm")
        source = manifest.path_of(data, "source")
        face = manifest.path_of(data, "face")
        if name == "survey":
            return ["preflight.py", source, "--json", stem.with_suffix(".json"), "--height", o["height"]], []
        if name == "build":
            args = ["make_vrm.py", source, vrm, "--name", self.name, "--author", data.get("author") or "Unknown",
                    "--blend", blend, "--height", o["height"], "--keep-eyes", "--manifest", self.manifest_path]
            if face:
                args += ["--face", face]
            if data["joints"]:
                joints = stem.with_name(stem.name + "_joints.json")
                joints.write_text(json.dumps(data["joints"], indent=2), encoding="utf-8")
                args += ["--joints", joints]
            args += ["--skinning", o["skinning"]]
            args += [flag for key, flag in (("rigid_head", "--rigid-head"), ("cloth", "--cloth"),
                                             ("hat_tip", "--hat-tip"), ("free_mouth", "--free-mouth")) if o[key]]
            if o["mouth_colour"]:
                args += ["--mouth-colour", o["mouth_colour"]]
            return args, []
        if name == "movement":
            reference = manifest.path_of(data, "reference") or source
            return ["check_movement.py", "--reference", reference, "--json", stem.with_suffix(".json"),
                    "--manifest", self.manifest_path, "--height", o["height"], "--listing", self.out / "pieces.json"], []
        if name == "springs":
            return ["check_springs.py", "--json", stem.with_suffix(".json"), "--height", o["height"]], []
        script = {"hair": "fix_hair_weights.py", "ahoge": "add_ahoge_spring.py", "face_tracking": "add_face_tracking.py",
                  "loose": "fix_loose_pieces.py", "fingers": "add_fingers.py", "pieces": "apply_pieces.py",
                  "finger_guides": "fix_fingers.py", "tpose": "to_tpose.py", "eyes": "add_eye_tracking.py",
                  "mouth": "fix_mouth.py", "check": "check_avatar.py", "vrm_check": "check_avatar.py"}[name]
        if name == "finger_guides":
            return [script, blend], []
        if name == "check":
            # close-ups of the worst spots, so the Build page shows where a failure is
            return [script, "--json", stem.with_suffix(".json"), "--spots", stem.with_name(stem.name + "_spots")], []
        if name == "vrm_check":
            # what really ships: the exported file, re-imported (the add-on once swapped the pose on export)
            exported = self._last_vrm()
            if exported is None:
                raise StepFailed("vrm_check: no exported .vrm yet")
            load = ("import bpy; bpy.ops.wm.read_homefile(use_empty=True); "
                    f"assert bpy.ops.import_scene.vrm(filepath={str(exported)!r}) == {{'FINISHED'}}")
            return [script, "--json", stem.with_suffix(".json")], ["--python-expr", load]
        args = [script, blend, vrm]
        if name == "loose" and o["keep_seams"]:
            args.append("--keep-seams")
        if name == "face_tracking" and o["mouth_only"]:
            args.append("--mouth-only")
        if name == "pieces":
            args += ["--manifest", self.manifest_path]
        if name == "tpose":
            if o["tpose"] == "spec":
                args.append("--spec-hands")
            if o["repair_shoulder_seams"]:
                args.append("--repair-shoulder-seams")
        if name == "eyes":
            args += ["--face", face]
            if o["eyes"] == "slide":
                args += ["--slide", *o["slide"]]
        if name == "mouth":
            fix = o["mouth_fix"]
            if fix.get("free"):
                args.append("--free")
            for key in ("scale", "rest", "lift"):
                if fix.get(key) is not None:
                    args += [f"--{key}", fix[key]]
        return args, []

    def _last_vrm(self):
        for name in reversed(NAMES):
            entry = self.state["steps"].get(name, {})
            if entry.get("status") == "done" and entry.get("vrm") and Path(entry["vrm"]).exists():
                return Path(entry["vrm"])
        return None

    def _record(self, name, entry):
        self.state["steps"][name] = entry
        if entry["status"] == "done":
            # built on an older file: kept, but no longer what a later step reads
            for later in NAMES[NAMES.index(name) + 1:]:
                if self.state["steps"].get(later, {}).get("status") == "done":
                    self.state["steps"][later]["status"] = "stale"
        self._save_state()

    def _note_failure(self, name, failure, log):
        entry = self.state["steps"].setdefault(name, {"status": "not run"})
        entry["last_failure"] = {**failure, **({"log": str(log)} if log.exists() else {})}
        self._save_state()

    def _save_state(self):
        self.out.mkdir(parents=True, exist_ok=True)
        _write_json(self.state_path, self.state, backup=True)


def run_script(blender, script, args, blend_in=None, log=None, extra=()):
    """Run any scripts/ file in Blender outside the chain (renders, checks); returns (code, output)."""
    log = Path(log) if log else Path(os.environ.get("TEMP", "/tmp")) / f"vtuber_{os.getpid()}.log"
    with _Lock():
        code = _blender(blender, blend_in, [script, *args], log, list(extra))
    if code == CANCELLED:
        raise Cancelled("stopped by the user")
    return code, log.read_text(encoding="utf-8", errors="replace")


def _blender(blender, blend_in, args, log, extra):
    """One Blender run, output to `log`; returns the exit code. Stops a run that hangs."""
    script, *script_args = args
    command = [blender, "-b"]
    if blend_in:
        command.append(str(blend_in))
    command += [*extra, "--python-exit-code", "1", "-P", str(SCRIPTS / script), "--", *(str(a) for a in script_args)]
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8", errors="replace") as handle:
        handle.write(" ".join(f'"{c}"' if " " in c else c for c in map(str, command)) + "\n")
        handle.flush()
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=REPO)
        started = time.time()
        while process.poll() is None:
            time.sleep(2)
            if CANCEL.is_set():
                process.kill()
                process.wait()
                with open(log, "a", encoding="utf-8") as tail:
                    tail.write(f"\nCANCELLED by the user after {round(time.time() - started)} s\n")
                return CANCELLED
            if time.time() - started > STEP_TIMEOUT or OUT_OF_MEMORY in _log_tail(log):
                process.kill()
                process.wait()
                with open(log, "a", encoding="utf-8") as tail:
                    tail.write(f"\nSTOPPED by the engine after {round(time.time() - started)} s\n")
                return -1
    return process.returncode


class _Lock:
    """One Blender at a time across the app, the CLI and any agent driving it."""

    def __enter__(self):
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            held = LOCK.read_text(errors="replace").strip()
            found = re.match(r"pid (\d+)", held)
            if not found or _alive(int(found.group(1))):
                raise StepFailed(f"another build is running ({held}); if none is, delete {LOCK}") from None
            # left by a process that crashed or was killed: without this the user had to find and
            # delete the file by hand. A second taker racing us still loses on O_EXCL.
            LOCK.unlink(missing_ok=True)
            try:
                handle = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                raise StepFailed(f"another build is running ({LOCK.read_text(errors='replace').strip()})") from None
        os.write(handle, f"pid {os.getpid()} since {datetime.now().isoformat(timespec='seconds')}".encode())
        os.close(handle)
        return self

    def __exit__(self, *exc):
        LOCK.unlink(missing_ok=True)


def _alive(pid):
    """Whether a process with this pid is running. Never os.kill(pid, 0) on Windows: it terminates."""
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            # no such process, or one we may not look at: only the first frees the lock
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED
        handle = ctypes.c_void_p(handle)
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# The humanoid bones VRM 1.0 requires; an exported file without them won't load in any app.
REQUIRED_BONES = {"hips", "spine", "head", *(f"{side}{part}" for side in ("left", "right")
                  for part in ("UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot"))}
# .blend magic: plain, zstd (Blender 3+ compression) and gzip (older compression)
BLEND_MAGIC = (b"BLENDER", b"\x28\xb5\x2f\xfd", b"\x1f\x8b")


def output_problem(entry):
    """What is wrong with a step's output files, or None. Cheap: headers and the VRM's JSON, no Blender."""
    if entry.get("blend"):
        path = Path(entry["blend"])
        with open(path, "rb") as handle:
            head = handle.read(7)
        if not head.startswith(BLEND_MAGIC):
            return f"{path.name} isn't a .blend file"
    if entry.get("json") and _read_json(entry["json"], None) is None:
        return f"{Path(entry['json']).name} isn't readable JSON"
    if entry.get("vrm"):
        path = Path(entry["vrm"])
        if not path.exists():
            return f"{path.name} wasn't exported"
        gltf = _glb_json(path)
        if gltf is None:
            return f"{path.name} isn't a readable .vrm (GLB)"
        vrm = (gltf.get("extensions") or {}).get("VRMC_vrm")
        if not vrm:
            return f"{path.name} has no VRM 1.0 data (VRMC_vrm)"
        bones = (vrm.get("humanoid") or {}).get("humanBones") or {}
        missing = sorted(REQUIRED_BONES - set(bones))
        if missing:
            return f"{path.name} lacks required humanoid bones: {', '.join(missing)}"
        count = len(gltf.get("nodes", []))
        broken = sorted(b for b, v in bones.items() if not isinstance(v.get("node"), int) or not 0 <= v["node"] < count)
        if broken:
            return f"{path.name} names bones with no node: {', '.join(broken)}"
    return None


def _glb_json(path):
    try:
        with open(path, "rb") as handle:
            head = handle.read(20)
            if len(head) < 20 or head[:4] != b"glTF" or head[16:20] != b"JSON":
                return None
            return json.loads(handle.read(struct.unpack("<I", head[12:16])[0]))
    except (OSError, ValueError):
        return None


_BLENDER_VERSIONS = {}


def provenance(blender, manifest_path):
    """What built a run: Blender and VRM Add-on versions, the code's commit, the manifest's hash.

    `warnings` lists what differs from the tested setup (Blender 4.2 LTS). Never raises: a missing
    git or a Blender that won't print its version must not stop a build.
    """
    warnings = []
    if blender not in _BLENDER_VERSIONS:
        try:
            out = subprocess.run([blender, "--version"], capture_output=True, text=True, timeout=60).stdout
            found = re.search(r"Blender (\d+\.\d+\.\d+)", out)
            _BLENDER_VERSIONS[blender] = found.group(1) if found else None
        except Exception:  # noqa: BLE001 -- provenance is a record, never a reason to stop
            _BLENDER_VERSIONS[blender] = None
    version = _BLENDER_VERSIONS[blender]
    if version and ".".join(version.split(".")[:2]) not in TESTED:
        warnings.append(f"Blender {version}: not yet tested with this pipeline (tested: {', '.join(sorted(TESTED))})")
    commit = dirty = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=20).stdout.strip() or None
        # tracked files only: the tree holds large untracked models
        changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                 capture_output=True, text=True, timeout=20).stdout
        dirty = bool(changed.strip())
    except Exception:  # noqa: BLE001 -- provenance is a record, never a reason to stop
        pass
    try:
        digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()[:16]
    except OSError:
        digest = None
    return {"started": datetime.now().isoformat(timespec="seconds"), "blender": str(blender),
            "blender_version": version, "vrm_addon": _vrm_addon_version(version), "commit": commit,
            "dirty": dirty, "manifest": str(manifest_path), "manifest_sha256": digest,
            "python": platform.python_version(), "warnings": warnings}


def _vrm_addon_version(blender_version):
    # the extension's own manifest, where Blender 4.2 installs it for this user
    base = Path(os.environ.get("APPDATA", Path.home() / ".config")) / "Blender Foundation" / "Blender"
    series = ".".join((blender_version or "4.2").split(".")[:2])
    for toml in sorted((base / series / "extensions").glob("*/vrm/blender_manifest.toml")):
        found = re.search(r'^version\s*=\s*"([^"]+)"', toml.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
        if found:
            return found.group(1)
    return None


def _write_json(path, data, backup=False):
    """Write JSON so a crash mid-write leaves the old file whole, not half of the new one."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if backup and path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    os.replace(temporary, path)


def _load_state(path):
    """state.json, or its backup when it was left unreadable; a fresh state when neither exists."""
    path = Path(path)
    if not path.exists():
        return {"steps": {}}
    state = _read_json(path, None)
    if isinstance(state, dict) and "steps" in state:
        return state
    backup = _read_json(path.with_name(path.name + ".bak"), None)
    if isinstance(backup, dict) and "steps" in backup:
        return backup
    raise StepFailed(f"{path} is unreadable and has no usable backup; move it aside to start the build afresh")


def _tagged(output, tag):
    for line in output.splitlines():
        if line.startswith(tag):
            try:
                return json.loads(line[len(tag):].strip())
            except json.JSONDecodeError:
                return None
    return None


def _log_tail(path, size=4000):
    with open(path, "rb") as handle:
        handle.seek(max(0, handle.seek(0, 2) - size))
        return handle.read().decode("utf-8", errors="replace")


def _tail(text, size=2000):
    return text[-size:]


def _read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (TypeError, OSError, json.JSONDecodeError):
        return default


if __name__ == "__main__":
    sys.exit("use python -m app")
