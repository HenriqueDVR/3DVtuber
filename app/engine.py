"""A character's build, step by step, from its manifest. Plain Python: each step is one Blender run.

The manifest (addon/vtuber_maker/pipeline/manifest.py) holds what the user decided; this holds
what happened. Each run writes into its own folder under the project's output folder, so
nothing earlier is overwritten, and `state.json` there records which file every step last
produced: a later step reads that, never "the newest file" (an intermediate A-pose can be newer
than a finished VRM). A step can be rerun alone, or the chain from any step on.

Only one Blender runs at a time (two at once hung the machine on 2026-09-23), and a run that
logs "Malloc returns null" is stopped: out of memory, Blender spins instead of exiting. So is one
that fills the drive or uses the last of the memory. Every Blender starts on factory settings
with only the VRM Add-on on (scripts/_startup.py), whatever else the user has installed.
"""

import contextlib
import hashlib
import importlib.util
import json
import math
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

from . import contracts, errors, face_guess, paint, vendor

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
WINDOWS_BLENDER = Path(r"C:\Program Files\Blender Foundation")
LOCK = REPO / "out" / ".blender.lock"
# Run before a step's file loads, on factory settings: see scripts/_startup.py.
STARTUP = SCRIPTS / "_startup.py"
FINGERPRINT = SCRIPTS / "_fingerprint.py"
# Longest a single step may run (s); the slowest real steps take about 3 minutes.
STEP_TIMEOUT = 30 * 60
OUT_OF_MEMORY = "Malloc returns null"
# Free space a step needs where it writes (bytes): a .blend or the .vrm, about 100 MB each on
# the models so far. On a full drive Blender's saves fail in ways that read as code bugs: four
# steps "failed" on 2026-09-29 with E: at 0 GB. Under the floor, a running step is stopped.
DISK_NEEDED = 1 << 30
DISK_FLOOR = 50 << 20
# Memory still to be had (RAM and page file, bytes). Under the floor a Blender that asks for more
# gets none and spins instead of exiting; under LOW the user is told before the run starts. The
# page file grows on the system drive, so that drive running out is the same problem (2026-09-23).
MEMORY_FLOOR = 200 << 20
MEMORY_LOW = 2 << 30
SYSTEM_DISK_LOW = 4 << 30
# A run's model files, which `prune` removes once nothing reads them (.blend1: Blender's backup).
MODELS = (".blend", ".blend1", ".vrm")
# Oldest Blender the pipeline runs on (EEVEE Next, extensions, the VRM Add-on's 4.2 builds), and the
# series the full test suite has passed on (tests/run_matrix.py). Others at or above the minimum run
# with a warning; keep in step with addon/vtuber_maker/pipeline/compat.py.
MIN_BLENDER = (4, 2, 0)
TESTED = {"4.2", "4.5", "5.2"}
# When each tested LTS series stops getting fixes (blender.org's release cycle), for the Setup page.
SUPPORT_ENDS = {"4.2": "2026-07", "4.5": "2027-07", "5.2": "2028-07"}
# What this machine's user chose in the app, e.g. which Blender to run. out/ is never committed.
SETTINGS = REPO / "out" / "settings.json"
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
# What _blender returns for a run the engine stopped itself: too long, out of memory, disk full.
STOPPED = -1


# (name, label, what it writes): "blend" = a .blend, "vrm" = a .vrm only, "json" = a report only.
# Order is the build order that produced the model A and model B candidates.
STEPS = [
    ("survey", "surveying the model", "json"),
    ("build", "rigging", "blend"),
    ("hair", "hair weights", "blend"),
    ("ahoge", "ahoge spring", "blend"),
    ("face_tracking", "face tracking", "blend"),
    ("loose", "loose pieces", "blend"),
    ("fingers", "finger bones", "blend"),
    ("pieces", "clothing pieces", "blend"),
    ("finger_guides", "finger guides", "blend"),
    ("tpose", "T-pose", "blend"),
    ("chest", "chest bounce", "blend"),
    ("eyes", "eye tracking", "blend"),
    ("mouth", "mouth shapes", "blend"),
    ("check", "checking the rig", "json"),
    # the one .vrm of a build, from the last .blend: every step used to export its own, about a
    # minute and 700 MB a build on model A for files nothing read. After the check, so its files keep
    # their number (13_check_spots).
    ("export", "exporting the VRM", "vrm"),
    ("vrm_check", "checking the exported VRM", "json"),
    # what check_avatar can't see: pieces left behind as limbs move (on the modelled pose), and the
    # springs running as apps run them (on the finished rig); the verdict takes them in
    ("movement", "checking what moves with the body", "json"),
    ("springs", "running the physics", "json"),
]
NAMES = [name for name, _, _ in STEPS]
# Blender arguments that tell a step's script to leave the .vrm to the export step
# (`vrm_export.skipped`); the scripts still take the file's name, for use by hand.
NO_EXPORT = ["--python-expr", "import os; os.environ['VTUBER_MAKER_NO_EXPORT'] = '1'"]
# Steps that exit nonzero when there is nothing for them to do (no ahoge, no loose strap):
# the chain carries on with the file before them.
OPTIONAL = {"ahoge", "loose", "fingers"}
# Checks that run after the build and report; a check that can't run doesn't take the avatar away.
CHECKS = {"movement", "springs"}
# Face features that, when their own checks refuse, leave the avatar without them rather than
# without an avatar: model C's eye step refused (iris over the lids) and the chain ended with no .vrm.
# Chest bounce too: a chest it can't find leaves the avatar without the bounce.
SOFT = {"face_tracking", "chest", "eyes", "mouth"}
EYES_FALLBACK_REASON = ("the moving iris showed over the lids (the drawn iris is as tall as the eye), "
                        "so the eyes slide as a painted patch instead")


def find_blender(given=None):
    """The Blender to run: --blender, $BLENDER, the one chosen on the Setup page, PATH, else the
    best installed one.

    Installed ones are ranked tested series first, then newest: a newer untested Blender beside the
    tested one must not silently take over the user's builds. Below MIN_BLENDER is never picked.
    The portable copies in tools/blender are never picked by themselves, only when chosen.
    """
    for candidate in (given, os.environ.get("BLENDER"), _read_json(SETTINGS, {}).get("blender"), shutil.which("blender")):
        if candidate and Path(candidate).exists():
            return str(candidate)
    found = [(_folder_version(p), p) for p in installed_blenders()]
    found = [(v, p) for v, p in found if v and v >= MIN_BLENDER]
    if found:
        return str(max(found, key=lambda vp: (".".join(map(str, vp[0][:2])) in TESTED, vp[0]))[1])
    raise FileNotFoundError(f"Blender {'.'.join(map(str, MIN_BLENDER[:2]))} or later not found: "
                            "put it on PATH, set $BLENDER, or pass --blender")


def installed_blenders():
    """Every Blender in the usual places: on Windows the installer's folder, Steam, the Store; on
    macOS the Applications folders; on Linux the unpacked folders under /opt and the home folder."""
    if sys.platform == "darwin":
        apps = [Path("/Applications"), Path.home() / "Applications"]
        return [p for root in apps if root.is_dir() for p in root.glob("Blender*.app/Contents/MacOS/Blender")]
    if os.name != "nt":
        roots = [Path("/opt"), Path.home(), Path.home() / "Applications", Path.home() / ".local" / "share"]
        return [p for root in roots if root.is_dir() for p in root.glob("blender*/blender") if p.is_file()]
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

    def model(self):
        """The model the steps read: the source, or its painted copy when the character has paint."""
        source = manifest.path_of(self.data, "source")
        return paint.painted(source, manifest.path_of(self.data, "paint"), self.out, self.name)

    def mark_stale(self, start="build"):
        """What was built from `start` on no longer shows the manifest: the app then offers the rebuild."""
        changed = False
        for name in NAMES[NAMES.index(start):]:
            if self.state["steps"].get(name, {}).get("status") == "done":
                self.state["steps"][name]["status"] = "stale"
                changed = True
        if changed:
            self._save_state()

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
            # an iris the eye step refused (taller than the opening: model A's T-pose FBX) slides instead
            if options["eyes"] == "iris" and self.state.get("eyes_fallback"):
                options["eyes"] = self.state["eyes_fallback"]
        if self.data.get("face") and not marks_eyes(_read_json(manifest.path_of(self.data, "face"), None)):
            # only the mouth is marked: there are no eyes to move or to close
            options["eyes"] = "hidden"
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
            "chest": not o["chest"] and "chest bounce off",
            "eyes": (o["eyes"] in ("none", "hidden") or not face) and "no eye tracking, or no face points",
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
                "check_failed": failed, "verdict": self.verdict(failed), "ships": self.ships(),
                "approved": self.approved(), "since_approved": self.since_approved()}

    def ships(self):
        """What the finished .vrm holds (`shipped`), or None until the build has run to its end."""
        entry = self.state["steps"].get("vrm_check", {})
        if entry.get("status") != "done":
            return None
        kept = (entry.get("report") or {}).get("ships")
        if kept:
            return kept
        # a build from before the facts were kept with the check
        vrm = self._last_vrm()
        return shipped(vrm) if vrm else None

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

    def send_to_warudo(self):
        """Copy the finished .vrm into Warudo's Characters folder as <name>_<run>.vrm, where its
        character list finds it. A copy per build, never written over: the note a user writes
        about what they saw has to stay about that file."""
        vrm = self._last_vrm()
        if vrm is None:
            raise StepFailed("no finished .vrm yet: build first")
        folder = warudo_folder()
        if folder is None:
            raise FileNotFoundError("Warudo wasn't found in a Steam library: give its Characters folder "
                                    r"(Warudo\Warudo_Data\StreamingAssets\Characters)")
        target = folder / f"{self.name}_{vrm.parent.name}.vrm"
        already = target.exists() and target.stat().st_size == vrm.stat().st_size
        if target.exists() and not already:
            raise StepFailed(f"{target} is there already and is a different file: rename or remove it in Warudo's folder")
        if not already:
            if free_disk(folder) < vrm.stat().st_size + DISK_FLOOR:
                known = errors.describe("VM-DISK-FULL")
                raise StepFailed(f"no room for the avatar on {folder.anchor} [VM-DISK-FULL] {known['fix']}", known)
            # under another name first: a Warudo that is open must not find half a file to load
            part = target.with_suffix(".part")
            shutil.copy2(vrm, part)
            part.replace(target)
        return {"sent": str(target), "name": target.name, "from": str(vrm), "already": already}

    def reports(self):
        """{report name: its JSON file} for the reports this build has finished (see APPROVED)."""
        steps = self.state["steps"]
        return {name: steps[name]["json"] for name in APPROVED
                if steps.get(name, {}).get("status") == "done" and steps[name].get("json")}

    def baseline(self):
        """(what it is, {report name: file}) of the build others are compared with: the one marked
        good, else the last build; (None, {}) for a character never built to a check."""
        if self.approved() is not None:
            return "the build marked good", {name: self.out / "approved" / f"{name}.json" for name in APPROVED}
        found = self.reports()
        return ("the last build", found) if "check" in found else (None, {})

    def since_approved(self):
        """{"worse": [...], "better": [...]} against the approved build, or None without one.

        Compares only reports both builds have, and only once this build has them: a half-rebuilt
        chain is still carrying the old reports' numbers or none.
        """
        if self.approved() is None:
            return None
        return changes(self.baseline()[1], self.reports())

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
        ships = self.ships()
        if ships:
            done = lambda name: steps.get(name, {}).get("status") == "done"  # noqa: E731
            in_file, to_see = shipping_problems(ships, {"tpose": done("tpose"), "eyes": done("eyes"),
                                                        "blink": (done("face_tracking") or done("eyes"))
                                                        and self.options()["eyes"] != "hidden"})
            issues += [("build", f"exported file: {line}") for line in in_file]
            looks += [("face", line) for line in to_see]
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
            # the build keeps the eyes as modelled (--keep-eyes): the blink comes with the face steps
            looks.insert(0, ("face", "no face points: face and eye tracking were skipped, so the avatar can't blink "
                                     "and its eyes don't move; only the vowel mouth shapes ship. Mark them on the Face page"))
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

    # ----- disk

    def _kept_runs(self):
        """The run folders whose model files something may still read: the ones the steps' latest
        files are in, the build marked good, and builds a Warudo note was written about."""
        named = [e[k] for e in self.state["steps"].values() for k in ("blend", "json", "vrm") if e.get(k)]
        named.append((self.approved() or {}).get("vrm"))
        named += [_read_json(note, {}).get("vrm") for note in (self.out / "feedback").glob("*/feedback.json")]
        return {Path(path).parent.resolve() for path in named if path}

    def usage(self):
        """How much disk this character takes, and what `prune` would free.

        Every run keeps a .blend per step and its .vrm, 400-850 MB a run on the models so far, and
        nothing removed them: E: filled up mid-build on 2026-09-29. A run's models are only needed
        while a step may read them (`_kept_runs`); its reports and logs are what comparing uses.
        """
        kept, runs = self._kept_runs(), []
        for folder in sorted((p for p in (self.out / "runs").glob("*") if p.is_dir()), reverse=True):
            files = [p for p in folder.iterdir() if p.is_file()]
            runs.append({"run": folder.name, "bytes": sum(p.stat().st_size for p in files),
                         "models": sum(p.stat().st_size for p in files if p.suffix.lower() in MODELS),
                         "kept": folder.resolve() in kept})
        total = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) if self.out.is_dir() else 0
        return {"name": self.name, "out": str(self.out), "bytes": total, "runs": runs,
                "prunable": sum(r["models"] for r in runs if not r["kept"]),
                "prunable_runs": sum(1 for r in runs if not r["kept"] and r["models"])}

    def prune(self, dry=False):
        """Delete the model files (.blend, .vrm) of runs nothing reads any more; reports, logs and
        renders stay. Refuses while a Blender run is going: its folder isn't in the state yet."""
        kept, removed, freed = self._kept_runs(), [], 0
        with contextlib.nullcontext() if dry else _Lock():
            for folder in (self.out / "runs").glob("*"):
                if not folder.is_dir() or folder.resolve() in kept:
                    continue
                for path in folder.iterdir():
                    if path.is_file() and path.suffix.lower() in MODELS:
                        freed += path.stat().st_size
                        removed.append(str(path))
                        if not dry:
                            path.unlink()
        return {"name": self.name, "removed": len(removed), "freed": freed, "dry": dry}

    def looked_at(self):
        """Whether the finished .vrm has a verdict: marked good, or a Warudo note written about it.
        None without a finished build. A build that only passed its checks is still waiting to be seen."""
        vrm = self._last_vrm()
        if vrm is None or self.state["steps"].get("vrm_check", {}).get("status") != "done":
            return None
        same = lambda other: bool(other) and Path(other).resolve() == vrm.resolve()  # noqa: E731
        if same((self.approved() or {}).get("vrm")):
            return True
        return any(same((_read_json(p, None) or {}).get("vrm")) for p in (self.out / "feedback").glob("*/feedback.json"))

    def summary(self):
        """How far the build got and the newest render: a card on the app's Characters page."""
        steps = self.state["steps"]
        statuses = {name: e.get("status") for name, e in steps.items()}
        renders = sorted((p for p in (self.out / "renders").glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
        pictures = sorted(renders[-1].glob("*.png"), key=lambda p: ("front" not in p.stem, p.name)) if renders else []
        # a still render from the front, arms down before arms out before the rest, the newest of its kind: the
        # figure alone on a clear ground, where a contact sheet is a grid
        # (never the T-pose: that is the rig's rest, not a picture of the character)
        fronts = [p for folder in renders if not folder.name.startswith("sheets") for p in folder.glob("*front*.png")
                  if not p.stem.startswith("t-pose")]
        # every still there is, by pose and view ("hang_back"), the newest of each: the app's menus show the
        # character in more than one pose
        stills = {p.stem: str(p) for folder in renders if not folder.name.startswith("sheets")
                  for p in folder.glob("*.png") if p.stem.rpartition("_")[2] in ("front", "back", "left", "right")}
        return {"surveyed": statuses.get("survey") == "done", "built": statuses.get("vrm_check") == "done",
                "stale": sorted(n for n, s in statuses.items() if s == "stale"),
                "failed": sorted(n for n, s in statuses.items() if s == "failed"),
                "finished": max((e.get("finished") or "" for e in steps.values()), default="") or None,
                "approved": self.approved() is not None, "picture": str(pictures[0]) if pictures else None,
                "portrait": str(min(fronts, key=lambda p: (next((i for i, pose in enumerate(("hang", "rest"))
                                                                 if p.stem.startswith(pose)), 2), -p.stat().st_mtime)))
                if fronts else None, "stills": stills, "portraits": self.portrait_set(),
                "waiting": self.looked_at() is False}

    # ----- the character's pictures

    def portrait_set(self):
        """The posed pictures drawn so far, by pose: the file, where the middle of the face and the floor
        are in it (shares of the picture, from its top left), and how tall the character and how wide
        its face are (shares of its side)."""
        folder = self.out / "portraits"
        drawn = _read_json(folder / "portraits.json", None) or {}
        return {name: {"file": str(folder / p["file"]), "head": p["head"], "floor": p["floor"], "height": drawn["height"],
                       "face": drawn.get("face")}
                for name, p in drawn.get("poses", {}).items() if (folder / p["file"]).exists()}

    def portraits(self, blend=None, on_event=None, locked=False):
        """Draw the character in the poses the app's menus show it in (scripts/render_portraits.py), from
        `blend` or the newest .blend there is. Returns the set, or None when there is no rig yet or
        Blender failed: these are pictures, and a build never fails for want of them. `locked`: called
        from inside a run, which holds the one-Blender-at-a-time lock already."""
        emit = on_event or (lambda event: None)
        blend = blend or self.latest()
        if blend is None:
            return None
        folder = self.out / "portraits"
        folder.mkdir(parents=True, exist_ok=True)
        emit({"step": "portraits", "status": "running", "log": str(folder / "portraits.log")})
        started = time.time()
        try:
            args, log = ["render_portraits.py", folder, "--size", 1400], folder / "portraits.log"
            code = _blender(self.blender, blend, args, log, []) if locked else run_script(self.blender, args[0], args[1:], blend_in=blend, log=log)[0]
        except Exception as error:  # noqa: BLE001: whatever stops a picture must not stop a build
            code = str(error)
        if code != 0:
            emit({"step": "portraits", "status": "skipped", "reason": "the character's pictures could not be drawn"})
            return None
        emit({"step": "portraits", "status": "done", "seconds": round(time.time() - started)})
        return self.portrait_set()

    def _portraits_from(self):
        return (_read_json(self.out / "portraits" / "portraits.json", None) or {}).get("blend")

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
            for warning in [*made_by["warnings"], *resource_warnings(self.out)]:
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
                # with its log's name: the app shows the log's end while the step runs
                emit({"step": name, "status": "running", "log": str(folder / f"{index:02d}_{name}.log")})
                try:
                    try:
                        entry = self._run_step(name, index, folder)
                    except StepFailed as error:
                        if not self._eyes_fall_back(name, error):
                            raise
                        emit({"step": name, "status": "warning", "reason": EYES_FALLBACK_REASON})
                        entry = self._run_step(name, index, folder)
                        entry["report"] = {**(entry.get("report") or {}), "eyes": "slide", "why": EYES_FALLBACK_REASON}
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
                # a new character gets its pictures as soon as it has a rig to pose
                if name == "build" and entry.get("status") == "done" and not self.portrait_set():
                    self.portraits(on_event=emit, locked=True)
            # ... and every build that changed the rig ends by drawing them again from the finished one
            newest = self.latest()
            if newest is not None and not CANCEL.is_set() and str(newest) != self._portraits_from():
                self.portraits(newest, on_event=emit, locked=True)
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
        free = free_disk(folder)
        if free < DISK_NEEDED:
            known = errors.describe("VM-DISK-FULL")
            raise StepFailed(f"{name}: only {free / 2**20:.0f} MB free on {folder.anchor} [VM-DISK-FULL] {known['fix']}", known)
        args, extra = self._arguments(name, stem, blend_in)
        prints = _fingerprint_of(stem.with_suffix(".blend"))
        if kind == "blend":
            # the numbers the step's file is judged by (contracts), written as the step saves it
            extra = [*extra, "--python-expr", f"import os; os.environ['VTUBER_MAKER_FINGERPRINT'] = {str(prints)!r}",
                     "-P", str(FINGERPRINT)]
        log = stem.with_suffix(".log")
        started = time.time()
        code = _blender(self.blender, blend_in, args, log, extra)
        if code == CANCELLED:
            raise Cancelled(f"{name}: stopped by the user")
        entry = {"status": "done", "log": str(log), "input": str(blend_in) if blend_in else None,
                 "finished": datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - started)}
        output = log.read_text(encoding="utf-8", errors="replace")
        entry[kind] = str(stem.with_suffix(f".{kind}"))
        made = Path(entry[kind]).exists()
        if code != 0 or not made:
            # a step the engine itself stopped (STOPPED) had something to do and didn't get to
            if name in OPTIONAL and code != STOPPED:
                return {"status": "nothing to do", "log": str(log), "finished": entry["finished"]}
            known = errors.classify(name, output)
            if known is None and free_disk(folder) < DISK_FLOOR:
                # Blender words a save that didn't fit many ways; on a full drive that is the reason
                known = errors.describe("VM-DISK-FULL")
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
        if kind == "blend":
            damage, warnings = contracts.compare(name, _read_json(_fingerprint_of(blend_in), None) if blend_in else None,
                                                 _read_json(prints, None), rigged=bool(self.data.get("rig")))
            if damage:
                known = errors.describe("VM-STEP-DAMAGE")
                raise StepFailed(f"{name}: {'; '.join(damage)} (see {log}) [VM-STEP-DAMAGE] {known['fix']}", known)
            if warnings:
                entry["warnings"] = warnings
        for tag in ("VTUBER_MAKER_REPORT", "VTUBER_MAKER_PIECES"):
            report = _tagged(output, tag)
            if report is not None:
                entry["report"] = report
        if name in ("check", "vrm_check"):
            entry["report"] = {k: v for k, v in _read_json(entry["json"], {}).items() if k in ("failed", "worst", "unweighted", "spots")}
        if name == "vrm_check":
            # the file the check just re-imported: what it holds is judged with the check (verdict)
            entry["report"]["ships"] = shipped(self._last_vrm())
        if name in CHECKS:
            # the Movement page shows the latest of each
            shutil.copy2(entry["json"], self.out / f"{name}.json")
        return entry

    def _arguments(self, name, stem, blend_in):
        """(script and its arguments, Blender arguments before -P)."""
        o, data = self.options(), self.data
        blend, vrm = stem.with_suffix(".blend"), stem.with_suffix(".vrm")
        source = self.model() if name in ("survey", "build") else manifest.path_of(data, "source")
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
            if not o["hair_hits_body"]:
                args.append("--hair-head-only")
            return args, NO_EXPORT
        if name == "movement":
            reference = manifest.path_of(data, "reference") or source
            return ["check_movement.py", "--reference", reference, "--json", stem.with_suffix(".json"),
                    "--manifest", self.manifest_path, "--height", o["height"], "--listing", self.out / "pieces.json"], []
        if name == "springs":
            return ["check_springs.py", "--json", stem.with_suffix(".json"), "--height", o["height"]], []
        if name == "export":
            return ["export_vrm.py", vrm], []
        script = {"hair": "fix_hair_weights.py", "ahoge": "add_ahoge_spring.py", "face_tracking": "add_face_tracking.py",
                  "loose": "fix_loose_pieces.py", "fingers": "add_fingers.py", "pieces": "apply_pieces.py",
                  "finger_guides": "fix_fingers.py", "tpose": "to_tpose.py", "chest": "add_chest_jiggle.py",
                  "eyes": "add_eye_tracking.py",
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
        if name == "face_tracking" and o["eyes"] == "hidden":
            args.append("--no-blink")
        if name == "pieces":
            args += ["--manifest", self.manifest_path]
        if name == "tpose":
            # an adopted rig's hands are left as modelled: the spec hands relay the fingers for bones
            # this build fitted, and on Tripo's own (flat, straight already) they fanned the fingers
            # out and bent the palm (model A (T-pose); the raw FBX's hands were clean)
            # Without either flag the step still straightens: it turned Tripo's thumb from 23 to 45
            # degrees, which is not the hand the user approved in Warudo (an earlier build, the raw hands)
            if self.data.get("rig"):
                args.append("--keep-finger-pose")
            elif o["tpose"] == "spec":
                args.append("--spec-hands")
            if o["repair_shoulder_seams"]:
                args.append("--repair-shoulder-seams")
        if name == "chest":
            args += ["--height", o["height"], "--amount", o["chest_amount"]]
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
        return args, NO_EXPORT

    def _eyes_fall_back(self, name, error):
        """Whether an automatic iris the eye step refused should be tried again as a sliding patch
        (remembered in the state, so later builds of the same model slide straight away)."""
        refused = name == "eyes" and (error.known or {}).get("code") == "VM-EYES-REFUSED"
        if not refused or self.data["options"]["eyes"] is not None or self.options()["eyes"] != "iris":
            return False
        self.state["eyes_fallback"] = "slide"
        self._save_state()
        return True

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


def changes(before, now):
    """{"worse": [...], "better": [...]} between two builds' reports ({"check": file, ...}; one
    either build lacks is left out): the rig check and exported file pose by pose, pieces the
    movement check newly flags, the worst spring droop and gap past SPRING_NOISE."""
    worse, better = [], []
    read = lambda reports, name: _read_json(reports.get(name), None)  # noqa: E731
    for name, label in (("check", "rig check"), ("vrm_check", "exported file")):
        old, new = read(before, name), read(now, name)
        if old and new:
            w, b = compare_checks.changes(old, new)
            worse += [f"{label}, {line}" for line in w]
            better += [f"{label}, {line}" for line in b]
    old, new = read(before, "movement"), read(now, "movement")
    if old is not None and new is not None:
        named = lambda report: {str(r.get("name") or r["label"]) for r in report.get("pieces", [])}  # noqa: E731
        worse += [f"movement: {n} newly flagged" for n in sorted(named(new) - named(old))]
        better += [f"movement: {n} no longer flagged" for n in sorted(named(old) - named(new))]
    old, new = read(before, "springs"), read(now, "springs")
    if old is not None and new is not None:
        for key, field, what in (("sagging", "sag_cm", "droop at rest"), ("coming_away", "gap_cm", "left behind by a swing")):
            was = max((r[field] for r in old.get(key, [])), default=0.0)
            now_ = max((r[field] for r in new.get(key, [])), default=0.0)
            if now_ - was > SPRING_NOISE:
                worse.append(f"physics: worst {what} {was} -> {now_} cm")
            elif was - now_ > SPRING_NOISE:
                better.append(f"physics: worst {what} {was} -> {now_} cm")
    return {"worse": worse, "better": better}


def characters():
    """Every character manifest the app lists: the samples' and one per output folder. Other JSON
    kept beside them isn't one: Name_face.json (face points) showed as a character named Name_face."""
    found = sorted({*REPO.glob("samples/manifests/*.json"), *REPO.glob("out/*/character.json")})
    hidden = set(hidden_characters())
    return [p for p in found if {"source", "options"} <= set(_read_json(p, None) or ()) and str(p.resolve()) not in hidden]


def hidden_characters():
    """Manifests the user took off the Characters page (out/settings.json): nothing of theirs is deleted."""
    return [h for h in _read_json(SETTINGS, {}).get("hidden", []) if Path(h).exists()]


def hide_character(manifest_path, hide=True):
    settings = _read_json(SETTINGS, {})
    path = str(Path(manifest_path).resolve())
    hidden = [h for h in settings.get("hidden", []) if h != path and Path(h).exists()]
    settings["hidden"] = [*hidden, path] if hide else hidden
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    _write_json(SETTINGS, settings)
    return {"hidden": settings["hidden"]}


def run_script(blender, script, args, blend_in=None, log=None, extra=()):
    """Run any scripts/ file in Blender outside the chain (renders, checks); returns (code, output)."""
    log = Path(log) if log else Path(os.environ.get("TEMP", "/tmp")) / f"vtuber_{os.getpid()}.log"
    with _Lock():
        code = _blender(blender, blend_in, [script, *args], log, list(extra))
    if code == CANCELLED:
        raise Cancelled("stopped by the user")
    return code, log.read_text(encoding="utf-8", errors="replace")


def _command(blender, blend_in, args, extra):
    """The command line for one script run. Factory settings and the startup script come before
    the file, so it opens with the VRM Add-on on and nothing else of the user's loaded."""
    script, *script_args = args
    command = [blender, "-b", "--factory-startup", "-P", str(STARTUP)]
    if blend_in:
        command.append(str(blend_in))
    return command + [*extra, "--python-exit-code", "1", "-P", str(SCRIPTS / script), "--", *(str(a) for a in script_args)]


def _blender(blender, blend_in, args, log, extra):
    """One Blender run, output to `log`; returns the exit code. Stops a run that hangs, fills the
    drive or uses the last of the memory (STOPPED), and one the user stops (CANCELLED)."""
    command = _command(blender, blend_in, args, extra)
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
            why = _stop_reason(log, started)
            if why is not None:
                process.kill()
                process.wait()
                with open(log, "a", encoding="utf-8") as tail:
                    tail.write(f"\nSTOPPED by the engine after {round(time.time() - started)} s{why}\n")
                return STOPPED
    return process.returncode


def _stop_reason(log, started):
    """Why the running Blender has to be stopped now, as the log line's ending ("" for a plain
    timeout or Blender's own out-of-memory message), or None to let it run. app/errors.py reads it."""
    if time.time() - started > STEP_TIMEOUT or OUT_OF_MEMORY in _log_tail(log):
        return ""
    if free_disk(log) < DISK_FLOOR:
        return ": disk full"
    memory = free_memory()
    if memory is not None and memory < MEMORY_FLOOR:
        return ": out of memory"
    return None


def free_disk(path):
    """Bytes free on the drive `path` is, or will be, on."""
    path = Path(path).resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


def free_memory():
    """Bytes of memory still to be had, RAM and page file together, or None where the system
    doesn't say. On Windows it is what a process may still commit: at zero, allocations fail."""
    if os.name == "nt":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("size", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        *((name, ctypes.c_ulonglong) for name in ("ram", "free_ram", "commit", "free_commit",
                                                                    "virtual", "free_virtual", "extended"))]
        status = Status(size=ctypes.sizeof(Status))
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.free_commit)
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return (int(fields["MemAvailable"].split()[0]) + int(fields["SwapFree"].split()[0])) * 1024
    except (OSError, KeyError, ValueError):
        return None


def resource_warnings(out):
    """What is short before a run starts, in the user's words: memory, room for the page file,
    room for the build. A warning, not a refusal: the run may still fit."""
    gb = lambda size: f"{size / 2**30:.1f} GB"  # noqa: E731
    notes = []
    memory = free_memory()
    if memory is not None and memory < MEMORY_LOW:
        notes.append(f"only {gb(memory)} of memory is free: close other programs, or Blender may run out mid-build")
    drive = Path(out).resolve().anchor
    free = free_disk(out)
    if free < 3 * DISK_NEEDED:
        notes.append(f"{drive} has {gb(free)} free and a full build writes several hundred MB: "
                     "free up space (Characters page) before it runs out")
    system = os.environ.get("SystemDrive")
    if os.name == "nt" and system and Path(system + "\\").resolve().anchor != drive:
        free = free_disk(system + "\\")
        if free < SYSTEM_DISK_LOW:
            notes.append(f"{system} has {gb(free)} free: Windows can't grow its page file there, "
                         "so Blender may run out of memory; free some space on it")
    return notes


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
        skins = gltf.get("skins", [])
        joints = {j for skin in skins for j in skin.get("joints", [])}
        loose = sorted(b for b, v in bones.items() if v["node"] not in joints)
        if skins and loose:
            return f"{path.name} maps bones the mesh isn't skinned to: {', '.join(loose)}"
        # Tripo's rigged GLB came with bind matrices like 1e31 and every bone imported at NaN
        if any(not math.isfinite(v) or abs(v) > BIND_LIMIT for skin in skins for v in _bind_matrices(path, gltf, skin)):
            return f"{path.name} has bind matrices that aren't numbers a skeleton can have"
    return None


# Larger than any inverse bind matrix entry of a model a few metres tall.
BIND_LIMIT = 1e4
VOWELS = ("aa", "ih", "ou", "ee", "oh")
LOOKS = ("lookUp", "lookDown", "lookLeft", "lookRight")
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")
# Degrees the exported arms may be off level and still be the T-pose apps start tracking from;
# the bake leaves them at 0.0 on every model so far.
T_POSE_SLACK = 10.0


def shipped(path):
    """What an exported .vrm holds, from its JSON alone (no Blender), or None when it isn't a
    readable VRM 1.0: the facts the Build page lists and `shipping_problems` judges.

    Read from the file because the .blend can look right while the file doesn't: the add-on's
    automatic pose replaced the baked rest pose in every export until 2026-09-22, and joints left
    with Blender's bone frames made Warudo twist every finger.
    """
    gltf = _glb_json(path)
    vrm = ((gltf or {}).get("extensions") or {}).get("VRMC_vrm")
    if not vrm:
        return None
    nodes = gltf.get("nodes", [])
    bones = (vrm.get("humanoid") or {}).get("humanBones") or {}
    joints = sorted({j for skin in gltf.get("skins", []) for j in skin.get("joints", [])})
    presets = (vrm.get("expressions") or {}).get("preset") or {}
    bound = {name for name, e in presets.items() if e.get("morphTargetBinds")}
    springs = gltf["extensions"].get("VRMC_springBone") or {}
    materials = gltf.get("materials", [])
    return {"bones": len(bones), "fingers": sum(1 for b in bones if any(f in b for f in FINGERS)),
            "turned_joints": [nodes[j].get("name", str(j)) for j in joints if not _unrotated(nodes[j])],
            "arms_off_level": _arms_off_level(nodes, bones),
            "blink": "blink" in bound, "vowels": [v for v in VOWELS if v in bound],
            "look": (vrm.get("lookAt") or {}).get("type"), "looks_bound": all(k in bound for k in LOOKS),
            "shapes": sorted({n for mesh in gltf.get("meshes", []) for n in (mesh.get("extras") or {}).get("targetNames", [])}),
            "springs": len(springs.get("springs", [])), "colliders": len(springs.get("colliders", [])),
            "materials": len(materials),
            "toon": sum("VRMC_materials_mtoon" in (m.get("extensions") or {}) for m in materials),
            "textures": len(gltf.get("images", [])), "megabytes": round(Path(path).stat().st_size / 2**20, 1)}


def marks_eyes(points):
    """Whether face points mark the eyes (face_points.has_eyes; this module stays plain Python).
    Points that can't be read count as marking them: the build then says what is wrong."""
    return points is None or bool(points.get("right_eye") or points.get("left_eye"))


def shipping_problems(ships, wanted):
    """([what must be fixed], [what is worth a look]) in the file that ships.

    `wanted`: what the build finished and so the file should show ({"tpose", "eyes", "blink"}).
    Only invariants every build so far holds are must-fix; a missing blink or vowel is a look,
    since a model can lack the face for it.
    """
    issues, looks = [], []
    turned = ships["turned_joints"]
    if turned:
        issues.append(f"{len(turned)} joints rest rotated in the exported file ({turned[0]}, ...): Warudo's finger tracking "
                      "twists such bones. Export through the app, never the VRM Add-on's menu")
    off = ships["arms_off_level"]
    if wanted.get("tpose") and off is not None and off > T_POSE_SLACK:
        issues.append(f"the exported rest pose isn't a T-pose (arms {off:g} degrees off level): tracking apps start from one")
    if wanted.get("eyes") and (ships["look"] != "expression" or not ships["looks_bound"]):
        issues.append("eye tracking was built, but in the exported file the eyes aren't bound to the gaze")
    if wanted.get("blink") and not ships["blink"]:
        looks.append("the exported file has no blink, though the face was built to have one")
    missing = [v for v in VOWELS if v not in ships["vowels"]]
    if missing:
        looks.append(f"the exported file has no mouth shape for {', '.join(missing)}: the avatar can't say "
                     f"{'them' if len(missing) > 1 else 'it'}")
    return issues, looks


def _unrotated(node):
    """A glTF node that only moves its children: no matrix, no turn, no scale."""
    if "matrix" in node:
        return False
    x, y, z, _ = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    return max(abs(x), abs(y), abs(z)) < 1e-6 and all(abs(s - 1.0) < 1e-6 for s in node.get("scale", (1.0, 1.0, 1.0)))


def _arms_off_level(nodes, bones):
    """Degrees the worse arm (upper arm to hand, at rest) is off the model's side axis: 0 in a
    T-pose. None when a node up the chain is turned, so the positions can't be read off the
    translations, or the file has no arms."""
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    worst = 0.0
    for side, sign in (("left", 1.0), ("right", -1.0)):
        arm, hand = (bones.get(f"{side}{part}", {}).get("node") for part in ("UpperArm", "Hand"))
        if arm is None or hand is None:
            return None
        reach, at = [0.0, 0.0, 0.0], hand
        while at != arm:
            if at not in parent:
                return None
            reach = [a + b for a, b in zip(reach, nodes[at].get("translation", (0.0, 0.0, 0.0)))]
            at = parent[at]
            if not _unrotated(nodes[at]):
                return None
        while at in parent:
            at = parent[at]
            if not _unrotated(nodes[at]):
                return None
        length = math.hypot(*reach)
        if not length:
            return None
        # VRM faces +Z with Y up: the character's left is +X
        worst = max(worst, math.degrees(math.acos(max(-1.0, min(1.0, sign * reach[0] / length)))))
    return round(worst, 1)


def _bind_matrices(path, gltf, skin):
    """The floats of a skin's inverse bind matrices, read from the file's binary chunk; () when
    the skin has none or they aren't laid out plainly."""
    try:
        accessor = gltf["accessors"][skin["inverseBindMatrices"]]
        view = gltf["bufferViews"][accessor["bufferView"]]
        count = accessor["count"] * 16
        with open(path, "rb") as handle:
            json_length = struct.unpack("<I", handle.read(20)[12:16])[0]
            handle.seek(20 + json_length + 8 + view.get("byteOffset", 0) + accessor.get("byteOffset", 0))
            raw = handle.read(count * 4)
        return struct.unpack(f"<{count}f", raw) if len(raw) == count * 4 else ()
    except (KeyError, IndexError, OSError, struct.error):
        return ()


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
    version = blender_version(blender)
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
            "blender_version": version, "vrm_addon": vrm_addon_version(blender, version), "commit": commit,
            "dirty": dirty, "manifest": str(manifest_path), "manifest_sha256": digest,
            "python": platform.python_version(), "warnings": warnings}


def blender_version(blender):
    """"4.2.1" as that Blender prints it, or None when it won't say. Asked once per path."""
    blender = str(blender)
    if blender not in _BLENDER_VERSIONS:
        try:
            out = subprocess.run([blender, "--version"], capture_output=True, text=True, timeout=60).stdout
            found = re.search(r"Blender (\d+\.\d+\.\d+)", out)
            _BLENDER_VERSIONS[blender] = found.group(1) if found else None
        except Exception:  # noqa: BLE001 -- a record and a Setup line, never a reason to stop
            _BLENDER_VERSIONS[blender] = None
    return _BLENDER_VERSIONS[blender]


def vrm_addon_version(blender, version):
    """The VRM Add-on installed for this Blender, from the extension's own manifest, or None.
    A portable Blender keeps its extensions in its own `portable` folder, an installed one in
    the user's AppData under its series."""
    series = ".".join((version or "4.2").split(".")[:2])
    user = Path(os.environ.get("APPDATA", Path.home() / ".config")) / "Blender Foundation" / "Blender" / series
    for home in (Path(blender).parent / "portable", user):
        for toml in sorted((home / "extensions").glob("*/vrm/blender_manifest.toml")):
            found = re.search(r'^version\s*=\s*"([^"]+)"', toml.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
            if found:
                return found.group(1)
    return None


def known_blenders():
    """Every Blender the Setup page can offer, newest first: the installed ones, the portable
    copies in tools/blender, and the ones the user added by hand (`use_blender`: a Blender kept
    where the app doesn't look). Each with its series, whether it has the VRM Add-on, and whether
    it was `added` and is `missing` (added once, gone since)."""
    numbers = lambda version: tuple(map(int, version.split("."))) if version else (0,)  # noqa: E731
    found, seen = [], set()
    listed = [(path, False) for path in [*installed_blenders(), *portable_blenders()]]
    listed += [(Path(path), True) for path in _read_json(SETTINGS, {}).get("blenders") or []]
    for path, added in listed:
        if _same_file(path) in seen:
            continue
        seen.add(_same_file(path))
        missing = not Path(path).is_file()
        # a found one is named by its folder ("4.2", "5.2.2"): asking each Blender itself takes
        # seconds. An added one may sit in any folder, so it is asked, once
        named = re.search(r"\d+\.\d+(?:\.\d+)?", Path(path).parent.name)
        version = (None if missing else blender_version(path)) if added else named.group(0) if named else None
        if not added and numbers(version) + (0, 0) < MIN_BLENDER:
            continue
        series = ".".join(version.split(".")[:2]) if version else None
        found.append({"path": str(path), "version": version, "series": series, "tested": series in TESTED,
                      "portable": (Path(path).parent / "portable").is_dir(), "added": added, "missing": missing,
                      "vrm_addon": vrm_addon_version(path, series) if series else None})
    return sorted(found, key=lambda b: numbers(b["version"]), reverse=True)


def portable_blenders():
    """The portable copies unzipped under tools/blender, each with its own `portable` settings
    folder. Offered on the Setup page, never picked unasked (`find_blender`)."""
    return sorted((REPO / "tools" / "blender").glob("*/blender.exe"))


def _same_file(path):
    """A path as a key: two spellings of one file on Windows (case, slashes) are one Blender."""
    return os.path.normcase(os.path.abspath(str(path)))


def setup(given=None):
    """What the app needs and what it found: {"blender", "blenders", "checks", "ready"}.

    Each check is {"name", "state", "text", "fix"}, state "ok", "warn" (works, worth knowing) or
    "bad" (a build can't run). Before this a missing add-on or an old Blender showed up as a
    failed step some minutes into the first build.
    """
    checks = []
    add = lambda name, state, text, fix="": checks.append({"name": name, "state": state, "text": text, "fix": fix})  # noqa: E731
    blenders = known_blenders()
    try:
        blender = find_blender(given)
    except FileNotFoundError:
        blender = None
    version = blender_version(blender) if blender else None
    series = ".".join(version.split(".")[:2]) if version else None
    newer = next((b for b in blenders if b["tested"] and b["vrm_addon"] and series and
                  tuple(map(int, b["series"].split("."))) > tuple(map(int, series.split(".")))), None)
    if blender is None:
        add("Blender", "bad", "Blender 4.2 or later wasn't found",
            "install Blender 5.2 LTS from blender.org/download; the app finds it in its usual place")
    elif version is None:
        add("Blender", "bad", f"{blender} doesn't start", "install Blender 5.2 LTS from blender.org/download")
    elif tuple(map(int, version.split("."))) < MIN_BLENDER:
        add("Blender", "bad", f"Blender {version} is older than 4.2, which the app needs", "install Blender 5.2 LTS from blender.org/download")
    elif series not in TESTED:
        add("Blender", "warn", f"Blender {version}: not yet tested with the app (tested: {', '.join(sorted(TESTED))})",
            "builds run, with a warning; pick a tested one below if something goes wrong")
    elif SUPPORT_ENDS.get(series, "9999") < datetime.now().strftime("%Y-%m"):
        add("Blender", "warn", f"Blender {version} works, but its series stopped getting fixes in {SUPPORT_ENDS[series]}",
            f"Blender {newer['version']} is here, tested, and gave the same avatars a little faster: use it below"
            if newer else "Blender 5.2 LTS is tested and supported until 2028-07")
    else:
        add("Blender", "ok", f"Blender {version}")
    chosen = _read_json(SETTINGS, {}).get("blender")
    if chosen and not Path(chosen).exists() and not given:
        add("Chosen Blender", "warn", f"the Blender chosen here is gone ({chosen}); the app picked the one above itself",
            "choose another below, or let the app pick")
    if blender and version:
        addon = vrm_addon_version(blender, version)
        if addon:
            add("VRM Add-on", "ok", f"VRM Add-on {addon}")
        else:
            add("VRM Add-on", "bad", "the VRM Add-on isn't installed in this Blender",
                "open Blender, Edit > Preferences > Get Extensions, search for \"VRM format\" and install it")
    for module, name, state, what, fix in (
            ("PIL", "Pictures", "bad", "Pillow isn't installed: the face patch can't be made", "pip install pillow"),
            ("tkinter", "File dialogs", "warn", "tkinter is missing: Browse buttons won't open, type paths instead", ""),
            ("webview", "Own window", "warn", "pywebview isn't installed: the app opens in your browser instead of its own window",
             "optional: pip install pywebview")):
        if not _installed(module):
            add(name, state, what, fix)
    detector = face_guess.status()["python"]
    if detector:
        add("Auto face marking", "ok", "the face detector is installed: the Face page's \"Place them for me\" works")
    else:
        add("Auto face marking", "warn", "the face detector isn't installed: the eyes and mouth are marked by hand on the Face page",
            "optional: the button below installs it (about 250 MB to download, 1.5 GB on disk), or python -m app detector --install")
    for note in resource_warnings(REPO / "out"):
        add("Space", "warn", note)
    kept = vendor.status()
    if not kept["views"]:
        add("Internet", "warn", "the 3D views load three.js from the internet: without a connection they stay empty",
            "keep a copy with the app (17 MB, from jsDelivr and Google): the button below, or python -m app vendor")
    elif not kept["tracker"]:
        add("Internet", "warn", "the 3D views work without a connection; the viewer's webcam test still loads its face tracker "
            "from the internet", "keep a copy with the app (14 MB): the button below, or python -m app vendor")
    else:
        add("Internet", "ok", "the 3D views and the webcam test are kept with the app: no connection needed")
    return {"blender": blender, "version": version, "chosen": chosen, "blenders": blenders,
            "checks": checks, "ready": not any(c["state"] == "bad" for c in checks)}


def _installed(module):
    """Whether the Python running the app has this package."""
    return importlib.util.find_spec(module) is not None


def use_blender(path):
    """Run builds with this blender.exe from now on (Setup page); "" goes back to the app picking
    one. A Blender the app wouldn't list by itself is remembered, so it stays on the list.
    Returns setup().

    The path comes from a person: a file dialog, or typed. Only a file named blender is ever
    started (to ask its version), never whatever else was picked.
    """
    settings = _read_json(SETTINGS, {})
    if path:
        path = Path(path)
        if not path.is_file() or path.stem.lower() != "blender":
            raise ValueError(f"{path} isn't a blender.exe: pick the blender.exe inside Blender's folder")
        version = blender_version(path)
        if version is None or tuple(map(int, version.split("."))) < MIN_BLENDER:
            raise ValueError(f"{path} isn't a Blender 4.2 or later" + (f" (it is {version})" if version else " (it doesn't start)"))
        if _same_file(path) not in {_same_file(b["path"]) for b in known_blenders()}:
            settings["blenders"] = [*(settings.get("blenders") or []), str(path)]
    settings["blender"] = str(path) if path else None
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    _write_json(SETTINGS, settings)
    return setup()


def forget_blender(path):
    """Take a Blender the user added off the Setup page's list; if builds ran with it, the app
    picks one itself again. Returns setup()."""
    settings = _read_json(SETTINGS, {})
    settings["blenders"] = [p for p in settings.get("blenders") or [] if _same_file(p) != _same_file(path)]
    if settings.get("blender") and _same_file(settings["blender"]) == _same_file(path):
        settings["blender"] = None
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    _write_json(SETTINGS, settings)
    return setup()


WARUDO_CHARACTERS = Path("steamapps") / "common" / "Warudo" / "Warudo_Data" / "StreamingAssets" / "Characters"


def steam_libraries():
    """Steam's library folders: its own, and the ones on other drives it lists in libraryfolders.vdf."""
    steam = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Steam"
    try:
        listed = re.findall(r'"path"\s+"([^"]+)"', (steam / "steamapps" / "libraryfolders.vdf").read_text(encoding="utf-8", errors="replace"))
    except OSError:
        listed = []
    return [steam, *(Path(p.replace("\\\\", "\\")) for p in listed)]


def warudo_folder():
    """Warudo's Characters folder, where it lists avatars: the one set by hand (`use_warudo`), else
    the one in a Steam library; None when there is neither."""
    chosen = _read_json(SETTINGS, {}).get("warudo")
    for folder in ([Path(chosen)] if chosen else []) + [library / WARUDO_CHARACTERS for library in steam_libraries()]:
        if folder.is_dir():
            return folder
    return None


def use_warudo(folder):
    """Send avatars to this folder from now on; "" goes back to the one found in Steam."""
    if folder and not Path(folder).is_dir():
        raise ValueError(f"{folder} isn't a folder: pick Warudo's Characters folder "
                         r"(Warudo\Warudo_Data\StreamingAssets\Characters)")
    settings = _read_json(SETTINGS, {})
    settings["warudo"] = str(folder) if folder else None
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    _write_json(SETTINGS, settings)
    return {"warudo": str(warudo_folder() or "")}


def _fingerprint_of(blend):
    """Where the numbers of a step's .blend are kept: beside it, and still there once the
    Characters page has removed the model files of an old run."""
    blend = Path(blend)
    return blend.with_name(blend.stem + "_fingerprint.json")


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
