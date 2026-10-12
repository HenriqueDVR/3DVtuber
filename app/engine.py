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
import tarfile
import threading
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from . import contracts, errors, face_guess, inside, paint, slim, trust, update, vendor

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
WINDOWS_BLENDER = Path(r"C:\Program Files\Blender Foundation")
LOCK = REPO / "out" / ".blender.lock"
# Run before a step's file loads, on factory settings: see scripts/_startup.py.
STARTUP = SCRIPTS / "_startup.py"
FINGERPRINT = SCRIPTS / "_fingerprint.py"
# Run after it in the Blender a run keeps for its steps: see `_Worker`.
WORKER = SCRIPTS / "_worker.py"
# Longest a single step may run (s); the slowest real steps take about 3 minutes.
STEP_TIMEOUT = 30 * 60
# How often a running Blender is looked at (s), and every how many looks its log, the drive and the memory
# are read (`_stop_reason`: every 2 s, as before). Asleep 2 s between looks, a step came back up to 2 s after
# its Blender had ended: a second a step on average, some fifteen a build, for nothing.
POLL, WATCH = 0.1, 20
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
# A character's picture before it has a build to pose: the model as it came, from the front, drawn
# when its pieces are listed (scripts/list_pieces.py --picture). The menus stood empty until the first build.
FIRST_PICTURE = "first_look.png"
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
# Between two builds, a shape key has changed when its furthest move differs by this share of
# itself, or the middle of what it moves is this far (m) from where it was. Rebuilds of three
# characters with nothing changed: 0% and 0 mm; the mouth marked on a throat: 42% and 6 cm.
KEY_REACH = 0.25
KEY_PLACE = 0.01


class StepFailed(Exception):
    """`known`: the matching errors.KNOWN entry ({"code", "what", "fix", "options"}), when there is one.
    `carries_on`: the step is one a build does without (an optional repair that crashed): the chain
    goes on with the file before it and the step is kept as failed."""

    def __init__(self, message, known=None, carries_on=False):
        super().__init__(message)
        self.known = known
        self.carries_on = carries_on


class NoWarudo(FileNotFoundError):
    """Warudo's Characters folder isn't known: the page asks for it."""
    known = errors.describe("VM-NO-WARUDO")


class NoBlender(FileNotFoundError):
    """No Blender to run a step with: the Setup page gets one."""
    known = errors.describe("VM-NO-BLENDER")


def refused(code, message, detail=""):
    """What the engine refuses before a step starts, as a StepFailed with its code (app/errors.py): the
    page draws what happened and what to do from the code, the message stays for a command line."""
    return StepFailed(message, {**errors.describe(code), **({"detail": detail} if detail else {})})


# What says a step's script fell over (Python's own words, which Blender prints before it exits with
# the code a script that had nothing to do also exits with).
CRASHED = "Traceback (most recent call last)"


class Cancelled(Exception):
    """The user stopped the run. Not a StepFailed: a soft step must not carry on past it."""


# Set by the app's Stop button: the running Blender is killed at its next poll, and the chain
# stops before its next step. Only this process's runs; a CLI run elsewhere is untouched.
CANCEL = threading.Event()
CANCELLED = -2
# What _blender returns for a run the engine stopped itself: too long, out of memory, disk full.
STOPPED = -1
# ... and for one ended to make room for the Blenders running beside it: run again when there is room.
ASIDE = -3


class SetAside(Exception):
    """A step's Blender was ended to make room (`Project._side_by_side`). Nothing of the step failed."""


# The build's last steps read what the steps before them left and none reads another's, but for the
# exported file's check, which reads the export: with room for it they run side by side
# (`Project._side_by_side`), each in a Blender of its own.
TAIL = ("check", "export", "vrm_check", "movement", "springs")
# The order they start in when not all may run at once: the export first (the longest line hangs from
# it: its check, then the browser's look at the file), then the longer ones; the portraits last.
LEAD = ("export", "vrm_check", "check", "springs", "movement", "portraits")
# Never more Blenders at once than this, whatever the computer (the last steps are four lines of work).
MOST_BLENDERS = 4
# What a Blender is taken to need (bytes) when this character's last build doesn't say what the step
# took: the heaviest step measured took 2.9 GB (the export of a 417-bone model), most take 1; and how
# much over what the last build took, for a model that has changed since.
BLENDER_NEEDS = 3 << 30
NEED_MARGIN = 1.25
# Another Blender starts only with this many processor cores idle (a step keeps one to two busy), and
# one running beside others gives way when the memory still to be had falls under MEMORY_TIGHT (bytes).
CORES_SPARE = 2.0
MEMORY_TIGHT = 1 << 30


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
# And the finger guides: hands they can't read (a tester's game model, gloves with trims: "found 2
# digits standing out from the hand, need 5") keep the fingers the step before gave them.
SOFT = {"face_tracking", "chest", "eyes", "mouth", "finger_guides"}
# Steps that make what a finished .vrm already has, with what of the file's stays instead: a
# character whose source is a .vrm (manifest.improves) is taken in as it is and these don't run.
KEPT = {"survey": "a finished .vrm is read as it is, not surveyed",
        "hair": "the file's own hair weights and springs stay", "ahoge": "the file's own springs stay",
        "face_tracking": "the file's own face shapes stay", "loose": "the file's own weights stay",
        "fingers": "the file's own fingers stay", "pieces": "the file's own weights stay",
        "finger_guides": "the file's own fingers stay", "tpose": "the file's own rest pose stays",
        "eyes": "the file's own eyes stay", "mouth": "the file's own mouth stays",
        "movement": "it reads the modelled pose of a model built here"}
# A model that came with its own rig (manifest.keeps_rig, pipeline/kept.py) is taken in the same way,
# but for its rest pose: it is baked to the T-pose like any model's.
KEPT_RIG = {**{name: why.replace("file's", "model's") for name, why in KEPT.items() if name != "tpose"},
            "survey": "a model that came rigged is read as it is, not surveyed",
            "chest": "the model's own bones stay: chest bounce is for a rig made here"}
# Run before a check on a character taken in from a .vrm: its meshes joined in memory (the checks
# measure one mesh, and a VRoid file's body, face and hair are three).
WHOLE_BODY = ["--python-exit-code", "1", "-P", str(SCRIPTS / "_whole_body.py")]
# The piece roles the movement check may give by itself (option `auto_pieces`): the ones the pieces
# step acts on. A hair, hat or cloth role changes what the build step makes and stays the user's.
AUTO_ROLES = ("follow_bone", "shoulder_cloth", "fitted_skirt")
# An automatic iris the eye step refuses is tried again, in this order (see _eyes_fall_back).
EYES_FALLBACK = {
    "modelled": ("the moving iris showed over the lids (the drawn iris is as tall as the eye, or its lashes read as "
                 "iris), so the model's own iris discs slide instead"),
    "slide": ("the moving iris was refused and the model's own iris discs couldn't be used either, "
              "so the eyes slide as a painted patch instead"),
    "iris": ("the sliding patch an earlier build fell back to was refused, so the moving iris is tried again"),
}
# What the page offers when the eye step refuses the very mode its own advice names (errors.KNOWN: "set
# eyes to slide"): a tester's fox girl failed with `eyes: slide` and was told to set eyes to slide.
EYES_REFUSED_AGAIN = {
    True: ("the sliding patch was refused too. This model has eye pieces of its own: set eyes to modelled",
           {"eyes": "modelled"}),
    False: ("the sliding patch was refused too: check the eye marks on the Face page against the eye close-ups "
            "in the step's log folder", None),
}


def find_blender(given=None):
    """The Blender to run: --blender, $BLENDER, the one chosen on the Setup page, PATH, else the
    best installed one.

    Installed ones are ranked tested series first, then newest: a newer untested Blender beside the
    tested one must not silently take over the user's builds. Below MIN_BLENDER is never picked.
    The portable copies in tools/blender are never picked by themselves, only when chosen.
    """
    # (each asked only when the one before it names none: looking along PATH is some forty milliseconds on
    # Windows, and this is asked with every look at a character)
    for candidate in (lambda: given, lambda: os.environ.get("BLENDER"), lambda: _read_json(SETTINGS, {}).get("blender"),
                      lambda: shutil.which("blender")):
        candidate = candidate()
        if candidate and Path(candidate).exists():
            return str(candidate)
    found = [(_folder_version(p), p) for p in installed_blenders()]
    found = [(v, p) for v, p in found if v and v >= MIN_BLENDER]
    if found:
        return str(max(found, key=lambda vp: (".".join(map(str, vp[0][:2])) in TESTED, vp[0]))[1])
    raise NoBlender(f"Blender {'.'.join(map(str, MIN_BLENDER[:2]))} or later not found: "
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
    # "Blender 4.2", "blender-4.5.3-windows-x64": the folder names the series, which is what matters.
    # A folder that names none is asked: macOS keeps every Blender in Blender.app/Contents/MacOS, and
    # read by its folder alone no Blender was ever found on a Mac
    found = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", Path(path).parent.name)
    if not found:
        found = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", blender_version(path) or "")
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
        self._worker = None  # the Blender a run keeps for its steps, while one runs (`_kept_blender`)

    @property
    def blender(self):
        if self._blender is None:
            self._blender = find_blender()
        return self._blender

    # ----- what the steps will do

    @property
    def improving(self):
        """Whether the character is taken in as it came (a finished .vrm being added to, or a model
        with its own rig), not a bare model being built."""
        return manifest.taken_in(self.data)

    @property
    def keeping(self):
        """Whether the character is a model that came whole, taken in with its own rig (kept.py)."""
        return manifest.keeps_rig(self.data)

    def source_holds(self):
        """What the .vrm this character started from holds (`shipped`); None for a model."""
        source = manifest.path_of(self.data, "source")
        return shipped(source) if manifest.improves(self.data) and source and source.exists() else None

    def model(self):
        """The model the steps read: the source, or its painted copy when the character has paint."""
        source = manifest.path_of(self.data, "source")
        return paint.painted(source, manifest.path_of(self.data, "paint"), self.out, self.name)

    def frame(self):
        """Blender arguments for the manifest's `facing`: the model taken to face as its file has it
        (normalize._turn_upright), in every Blender that reads the model."""
        said = "import os; os.environ['VTUBER_MAKER_FACING'] = 'file'"
        return ["--python-expr", said] if self.data["options"].get("facing") == "file" else []

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
        if self.improving:
            # the file keeps its size: what scales with height goes by the height it has
            tall = (self.state["steps"].get("build", {}).get("report") or {}).get("height")
            options["height"] = tall or options["height"]
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
        pieces = bool(manifest.pieces_for(self.data, "pieces")) or o["auto_pieces"]
        rigged = bool(self.data.get("rig"))
        skip = {
            "ahoge": (not o["ahoge"] or o["rigid_head"]) and "no ahoge wanted, or a rigid head",
            "face_tracking": (not face or not o["face_tracking"]) and "no face points, or face tracking off",
            "loose": not o["loose_pieces"] and "loose-piece repair off",
            "fingers": (o["fingers"] == "none" and "fingers off") or (rigged and "the adopted rig has its own fingers"),
            "pieces": not pieces and "no clothing pieces named in the manifest",
            # a model with mitten hands: the finger step found none to rig, and the guides step, which
            # relays fingers, stopped the whole build with no avatar (the synthetic character, default options)
            "finger_guides": (o["fingers"] != "guides" and "finger guides off") or (rigged and "the adopted rig has its own fingers")
            or (self.state["steps"].get("fingers", {}).get("status") == "nothing to do" and "no separate fingers were found on this model"),
            "tpose": o["tpose"] == "none" and "T-pose off",
            "chest": not o["chest"] and "chest bounce off",
            "eyes": (o["eyes"] in ("none", "hidden") or not face) and "no eye tracking, or no face points",
            "mouth": (not o["mouth_fix"] or not face) and "no mouth fix asked for",
            "vrm_check": o["tpose"] == "none" and "no final VRM",
        }
        if self.improving:
            skip.update(KEPT_RIG if self.keeping else KEPT)
        return [(name, not skip.get(name), skip.get(name) or "") for name in NAMES]

    def status(self):
        steps = []
        # a soft step that failed is reported, and the build carries on without it
        for name, runs, why in self.plan():
            entry = self.state["steps"].get(name, {})
            steps.append({"step": name, "runs": runs, "skip_reason": why, "status": entry.get("status", "not run"),
                          **{k: entry[k] for k in ("blend", "vrm", "json", "log", "finished", "seconds", "report", "reason",
                                                     "known", "last_failure", "failure")
                             if k in entry}})
            # a failure kept by an app from before failures were kept drawn: drawn here, so the page never
            # reads a log's words itself (nor looks in them for the log's own name, as it did)
            if entry.get("status") == "failed" and "failure" not in entry:
                steps[-1]["failure"] = errors.failure(name, entry.get("known"), entry.get("reason", ""))
                if steps[-1].get("log") is None and _log_named(entry.get("reason")):
                    steps[-1]["log"] = _log_named(entry.get("reason"))
            last = entry.get("last_failure")
            if last and "failure" not in last:
                steps[-1]["last_failure"] = {**({"log": _log_named(last.get("reason"))} if _log_named(last.get("reason")) else {}), **last,
                                             "failure": errors.failure(name, last.get("known"), last.get("reason", ""))}
        last = self.latest()
        check = self.state["steps"].get("check", {})
        failed = _read_json(check.get("json"), {}).get("failed")
        return {"name": self.name, "manifest": str(self.manifest_path), "out": str(self.out),
                "improving": self.improving, "inert": manifest.inert(self.data), "source_holds": self.source_holds(),
                "gained": gained(self.source_holds(), self.ships()),
                "latest_blend": str(last) if last else None, "options": self.options(), "steps": steps,
                "check_failed": failed, "verdict": self.verdict(failed), "ships": self.ships(),
                "approved": self.approved(), "since_approved": self.since_approved()}

    def preflight(self, start=None, stop=None, only=None):
        """What is known to go wrong with the steps `start` to `stop` (or `only` one) before a Blender
        starts: lines of kind stop, warn or ok, each with its code and what mends it (app/preflight.py).
        Read from the disk alone; nothing is run and nothing refused."""
        from . import preflight
        return preflight.check(self, start=start, stop=stop, only=only)

    def forget_eyes_fallback(self):
        """Let the next build try the moving iris first again: the way an earlier build's eye step fell
        back to is forgotten. The eye step and what follows it are to be built again."""
        fell = self.state.pop("eyes_fallback", None)
        if fell:
            self._save_state()
            self.mark_stale("eyes")
        return {"forgot": fell}

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
            raise refused("VM-NO-EXPORT", "approve: build to the end first, a half-built avatar isn't a baseline")
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
            raise refused("VM-NO-EXPORT", "no finished .vrm yet: build first")
        folder = warudo_folder()
        if folder is None:
            raise NoWarudo("Warudo wasn't found in a Steam library: give its Characters folder "
                           r"(Warudo\Warudo_Data\StreamingAssets\Characters)")
        target = folder / f"{self.name}_{vrm.parent.name}.vrm"
        already = target.exists() and target.stat().st_size == vrm.stat().st_size
        if target.exists() and not already:
            raise refused("VM-WARUDO-FILE", f"{target} is there already and is a different file: rename or remove it in Warudo's folder",
                          str(target))
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
        looks = []
        if self.improving:
            # the skinning came with the file and no step here remakes it: said, not held against the build.
            # Without the check's guess at the cause, which is about rigs made here: "a limb's weights reach
            # onto the body" was said of a VRoid hip whose weights only changed too fast across the crease
            issues, looks = [], [("review", f"rig check: {f.split(': ')[0]} (the file's own skinning, kept as it came)")
                                 for f in (failed or [])]
        # the exported file drawn as apps draw it (inside.py); a .vrm taken in keeps what it came with
        seen = (steps["vrm_check"].get("report") or {}).get("seen") or {}
        (looks if self.improving else issues).extend(("review", f"as apps draw it: {line}") for line in seen.get("failed", []))
        if seen.get("left") and self._last_vrm() is not None:
            looks.append(("review", f"the avatar wasn't drawn to see what shows through it: {seen['left']}"))
        for name in sorted(SOFT):
            if steps.get(name, {}).get("status") == "failed":
                issues.append(("build", f"built without {name.replace('_', ' ')}"))
        ships = self.ships()
        if ships:
            done = lambda name: steps.get(name, {}).get("status") == "done"  # noqa: E731
            in_file, to_see = shipping_problems(ships, {"tpose": done("tpose"), "eyes": done("eyes"),
                                                        "blink": (done("face_tracking") or done("eyes"))
                                                        and self.options()["eyes"] != "hidden"})
            issues += [("build", f"exported file: {line}") for line in in_file]
            looks += [("face", line) for line in to_see]
            # a .vrm taken in must come out with everything it had
            issues += [("build", f"exported file: {line}") for line in lost(self.source_holds(), ships)]
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
        if not self.data.get("face") and not self.improving:
            # without face points the build still finishes, and nothing said the face was dead
            # the build keeps the eyes as modelled (--keep-eyes): the blink comes with the face steps
            looks.insert(0, ("face", "no face points: face and eye tracking were skipped, so the avatar can't blink "
                                     "and its eyes don't move; only the vowel mouth shapes ship. Mark them on the Face page"))
        # The fox girl's first build found none of her pale hair, hung no chain on it and said "ready"
        # (2026-10-10). Nothing in the mesh parts a bob that should swing from short hair or a hat that
        # shouldn't (the plain boy, model B: built the same, and rightly), so it is said, not judged;
        # and not to someone who asked for a rigid head themselves.
        built = steps.get("build", {}).get("report") or {}
        if built.get("hair_chains") == 0 and not self.improving and self.data["options"].get("rigid_head") is not True:
            looks.append(("parts", "no hair was found to hang on springs, so none swings: right for short hair, a hat or a "
                                   "hood. If this character's hair should move, paint it on the Parts page and call it hair"))
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
            runs.append({"run": folder.name, "bytes": sum(_size(p) for p in files),
                         "models": sum(_size(p) for p in files if p.suffix.lower() in MODELS),
                         "kept": folder.resolve() in kept})
        total = tree_size(self.out)
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
        first = self.out / FIRST_PICTURE
        first = str(first) if first.exists() else None
        # a finished .vrm isn't surveyed: it is as ready to build as a surveyed model
        return {"surveyed": statuses.get("survey") == "done" or self.improving, "built": statuses.get("vrm_check") == "done",
                "stale": sorted(n for n, s in statuses.items() if s == "stale"),
                "failed": sorted(n for n, s in statuses.items() if s == "failed"),
                "finished": max((e.get("finished") or "" for e in steps.values()), default="") or None,
                "approved": self.approved() is not None, "picture": str(pictures[0]) if pictures else first,
                "portrait": str(min(fronts, key=lambda p: (next((i for i, pose in enumerate(("hang", "rest"))
                                                                 if p.stem.startswith(pose)), 2), -p.stat().st_mtime)))
                if fronts else first, "stills": stills, "portraits": self.portrait_set(),
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
            code = _blender(self.blender, blend, args, log, [], **({"worker": self._worker} if self._worker else {})) if locked else run_script(self.blender, args[0], args[1:], blend_in=blend, log=log)[0]
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
        with _Lock(), self._kept_blender(folder):
            # inside the lock: refused because another build ran, it left an empty run on the Runs page
            folder.mkdir(parents=True, exist_ok=True)
            # what built this run, so "it was good last week" can be traced to a Blender, add-on and commit
            made_by = provenance(self.blender, self.manifest_path)
            _write_json(folder / "run.json", {**made_by, "steps": NAMES[first:last + 1]})
            version = made_by["blender_version"]
            if version and tuple(map(int, version.split("."))) < MIN_BLENDER:
                known = errors.describe("VM-BLENDER-OLD")
                raise StepFailed(f"Blender {version} [VM-BLENDER-OLD] {known['fix']}", known)
            for warning in [*made_by["warnings"], *resource_warnings(self.out)]:
                emit({"step": NAMES[first], "status": "warning", "reason": warning})
            ahead = None  # the last steps' results, when they were run side by side (`_ahead`)
            for index in range(first, last + 1):
                name = NAMES[index]
                if CANCEL.is_set():
                    raise Cancelled(f"stopped by the user before {name}")
                if name != "survey" and self.survey() is None and not self.improving:
                    raise refused("VM-NO-SURVEY", "run the survey step first")
                # planned afresh each time: the survey decides what later steps do
                _, runs, why = self.plan()[index]
                if not runs:
                    self._record(name, {"status": "skipped", "reason": why})
                    emit({"step": name, "status": "skipped", "reason": why})
                    continue
                if name == "build" and manifest.inert(self.data):
                    # said in the run itself: `skinning: voxel` on a .vrm built the same file with no word of it
                    emit({"step": name, "status": "warning", "reason": "a finished .vrm is taken in as it is: these options "
                          f"change nothing in it: {', '.join(manifest.inert(self.data))}"})
                if name in TAIL and ahead is None:
                    ahead = self._ahead(index, last, folder, emit)
                # with its log's name: the app shows the log's end while the step runs
                if name not in (ahead or ()):
                    emit({"step": name, "status": "running", "log": str(folder / f"{index:02d}_{name}.log")})
                try:
                    try:
                        self._eyes_tried = set()
                        # (one run ahead is taken as it came out: its file, or what it raised)
                        entry = _taken(ahead.pop(name)) if name in (ahead or ()) else self._run_step(name, index, folder)
                    except StepFailed as error:
                        while True:
                            if not self._eyes_fall_back(name, error):
                                raise self._eyes_advice(name, error)
                            mode = self.options()["eyes"]
                            emit({"step": name, "status": "warning", "reason": EYES_FALLBACK[mode]})
                            try:
                                entry = self._run_step(name, index, folder)
                                break
                            except StepFailed as again:
                                error = again
                        entry["report"] = {**(entry.get("report") or {}), "eyes": mode, "why": EYES_FALLBACK[mode]}
                except Cancelled:
                    # not recorded: the step keeps what it last finished, which later steps may still read
                    emit({"step": name, "status": "cancelled"})
                    raise
                except StepFailed as error:
                    # `failure`: the code, what happened, what to do and the option that does it, as the
                    # page draws them (errors.failure); `reason` and `known` as they always were
                    failure = {"reason": str(error)[-1500:], "finished": datetime.now().isoformat(timespec="seconds"),
                               **({"known": error.known} if error.known else {}),
                               "failure": errors.failure(name, error.known, str(error))}
                    if name not in SOFT | CHECKS and not error.carries_on:
                        # the step keeps what it last finished (later steps may read it); the failure
                        # rides beside it so the app can say what went wrong and what to do
                        self._note_failure(name, failure, folder / f"{index:02d}_{name}.log")
                        error.step = name  # for whoever started the run: which step it ended at
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
            if newest is not None and not CANCEL.is_set() and str(newest) != self._portraits_from() and "portraits" not in (ahead or ()):
                self.portraits(newest, on_event=emit, locked=True)
        return self.status()

    @contextlib.contextmanager
    def _kept_blender(self, folder):
        """The run's one Blender (`_Worker`), there for as long as the run and never past it: the lock
        is let go with no Blender of the run left alive."""
        self._worker = _Worker(self.blender, folder) if one_blender() else None
        try:
            yield
        finally:
            worker, self._worker = self._worker, None
            if worker is not None:
                worker.close()

    def _need(self, name):
        """Bytes a step's Blender is taken to need: what it took in this character's last build and a
        margin, else BLENDER_NEEDS."""
        took = self.state["steps"].get(name, {}).get("peak_mb")
        return int(took * NEED_MARGIN) << 20 if took else BLENDER_NEEDS

    def _room(self, name, running, budget, calm=False):
        """Whether `name`'s Blender may start now beside the ones running ({step: what it needs}).

        One always may. Another only while: the user hasn't said otherwise (`blenders_at_once`); nothing
        has had to give way yet in this run (`calm`); what the memory still to be had leaves after it is
        over MEMORY_LOW, and all of them together stay inside `budget` (what there was when the first
        started); and CORES_SPARE processor cores stand idle. All read as they are now: a computer busy
        with something else gets one Blender at a time, a big one gets four.
        """
        if not running:
            return True
        if calm or len(running) >= min(blenders_at_once() or MOST_BLENDERS, MOST_BLENDERS):
            return False
        need, free = self._need(name), free_memory()
        # (a computer that doesn't say what memory it has left gets one at a time)
        if free is None or free - need < MEMORY_LOW or sum(running.values()) + need > budget:
            return False
        idle = LOAD.idle_cores()
        return idle is None or idle >= CORES_SPARE

    def _ahead(self, index, last, folder, emit):
        """The build's last steps from `index` on, run side by side (`_side_by_side`) when more than one
        is to run and the computer has room for two Blenders now: {step: its entry, or what it raised}.
        {} when it hasn't: the chain then runs them itself, one after another."""
        if SHARED.held and SHARED.crowd() > 1:
            # other characters are building beside this one or waiting to (`at_once`): each is one
            # Blender then, and the computer's room is counted by characters
            return {}
        plan = self.plan()
        group = [name for name in NAMES[index:last + 1] if name in TAIL and plan[NAMES.index(name)][1]]
        newest = self.latest()
        if newest is not None and str(newest) != self._portraits_from():
            group.append("portraits")
        first = [name for name in LEAD if name in group and name != "vrm_check"][:2]
        free = free_memory()
        LOAD.idle_cores(wait=0.3)
        if len(first) < 2 or not self._room(first[1], {first[0]: self._need(first[0])}, float("inf") if free is None else free - MEMORY_LOW):
            return {}
        # each gets a Blender of its own: the kept one would hold its memory for nothing meanwhile
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        return self._side_by_side(group, folder, emit, newest)

    def _side_by_side(self, group, folder, emit, newest=None):
        """Run `group` (steps of TAIL in build order, and "portraits") as many at once as there is room
        for; returns {step: its entry, or what it raised} for the chain to take in its own order, as
        if it had just run each (`run`). The portraits are drawn and said here.

        What keeps the result the chain's own: the exported file's check waits for the export and is
        handed its file; a step that fails as would end the chain (not a check that only reports)
        ends the ones after it in the build's order, as they would never have run, and lets the ones
        before it finish. What keeps the computer standing: a Blender starts only when `_room` says so,
        asked again each time one ends; when the memory runs short all the same (MEMORY_TIGHT), the
        Blender started last is ended and its step run again once it can run alone, and from then on
        the run has one Blender at a time. So it has after a step fails that had company: the step is
        run once more by itself before its failure is believed (a graphics card or a drive that gave
        one Blender what it wanted may not give four).
        """
        waiting, running, free = [name for name in LEAD if name in group], {}, free_memory()
        budget = float("inf") if free is None else free - MEMORY_LOW

        def work(name, lane):
            try:
                if name == "portraits":
                    lane.result = self.portraits(newest, on_event=lane.events.append, locked=True)
                else:
                    lane.result = self._run_step(name, NAMES.index(name), folder, kept=False, abandon=lane.abandon, exported=lane.exported)
            except BaseException as error:  # noqa: BLE001 -- kept, for the chain to raise in its own order
                lane.result = error

        try:
            return self._lanes(group, folder, emit, work, waiting, running, budget)
        finally:
            # whatever ended this early, no Blender of the run outlives it
            for lane in running.values():
                lane.abandon.set()
            for lane in running.values():
                lane.thread.join(timeout=60)

    def _lanes(self, group, folder, emit, work, waiting, running, budget):
        """`_side_by_side`'s round, until nothing waits or runs: what is over is taken in, what the
        memory can't bear is set aside, what there is room for is started."""
        outcome, again, calm, exported, started = {}, set(), False, None, 0
        while waiting or running:
            for name, lane in list(running.items()):
                if lane.thread.is_alive():
                    continue
                del running[name]
                for event in lane.events:
                    emit(event)
                if name == "portraits":
                    outcome[name] = lane.result  # drawn or not, they were tried: the chain doesn't try again
                    continue
                if lane.dropped:
                    continue
                if isinstance(lane.result, SetAside):
                    waiting.insert(0, name)  # again, when it can run alone
                    continue
                if isinstance(lane.result, StepFailed) and lane.company and name not in again:
                    again.add(name)
                    waiting.insert(0, name)
                    calm = True
                    emit({"step": name, "status": "warning", "reason": f"{name.replace('_', ' ')} failed with other Blenders going "
                          "beside it: it is run again by itself, and the rest of this build one at a time"})
                    continue
                outcome[name] = lane.result
                # over, ahead of its turn: for the page's bar. How it ended is said when the chain takes it
                emit({"step": name, "status": "ahead"})
                if name == "export" and isinstance(lane.result, dict) and lane.result.get("vrm"):
                    exported = Path(lane.result["vrm"])
                if isinstance(lane.result, BaseException) and not (isinstance(lane.result, StepFailed) and name in SOFT | CHECKS):
                    # the chain ends at this step: what comes after it would not have run
                    after = group[group.index(name) + 1:]
                    waiting = [other for other in waiting if other not in after]
                    for other in after:
                        if other in running and other != "portraits":
                            running[other].dropped = True
                            running[other].abandon.set()
            free = free_memory()
            crowd = [name for name in running if name != "portraits" and not running[name].dropped]
            if len(running) > 1 and crowd and not calm and free is not None and free < MEMORY_TIGHT:
                last = max(crowd, key=lambda name: running[name].started)
                running[last].abandon.set()
                calm = True
                emit({"step": last, "status": "warning", "reason": f"the memory ran short with {len(running)} Blenders going at once: "
                      f"{last.replace('_', ' ')} is run again by itself, and the rest of this build one at a time"})
            for name in list(waiting):
                if name == "vrm_check" and "export" in group:
                    if "export" in waiting or "export" in running:
                        continue
                    if exported is None:
                        waiting.remove(name)  # no export to check: the chain says why when it comes to it
                        continue
                if not self._room(name, {other: lane.need for other, lane in running.items()}, budget, calm):
                    break
                waiting.remove(name)
                started += 1
                lane = SimpleNamespace(abandon=threading.Event(), started=started, need=self._need(name), result=None,
                                       events=[], dropped=False, exported=exported, company=bool(running))
                lane.thread = threading.Thread(target=work, args=(name, lane), daemon=True)
                running[name] = lane
                if len(running) > 1:
                    for other in running.values():
                        other.company = True
                if name != "portraits":
                    emit({"step": name, "status": "running", "log": str(folder / f"{NAMES.index(name):02d}_{name}.log")})
                lane.thread.start()
            time.sleep(0.1)
        return outcome

    def _run_step(self, name, index, folder, kept=True, abandon=None, exported=None):
        """One step, to its entry in the state. `kept`: told to the run's one Blender when there is one.
        The other two are for a step run beside others (`_side_by_side`): `abandon`, an Event that ends
        its Blender to make room (SetAside), and `exported`, the .vrm the export beside it has just
        written, which the state doesn't name yet."""
        stem = folder / f"{index:02d}_{name}"
        kind = dict((n, k) for n, _, k in STEPS)[name]
        fresh = name in ("survey", "build", "vrm_check")
        # the movement check reads the modelled pose: in a baked T-pose a sleeve left at the ribs
        # is no longer near the arm
        blend_in = None if fresh else self.latest(before="tpose" if name == "movement" else name)
        if not fresh and blend_in is None:
            raise refused("VM-NOTHING-BUILT", f"{name}: no earlier step has produced a .blend yet")
        free = free_disk(folder)
        if free < DISK_NEEDED:
            known = errors.describe("VM-DISK-FULL")
            raise StepFailed(f"{name}: only {free / 2**20:.0f} MB free on {folder.anchor} [VM-DISK-FULL] {known['fix']}", known)
        found = self._find_pieces(stem, blend_in) if name == "pieces" and self.options()["auto_pieces"] else []
        if name == "pieces" and not found and not manifest.pieces_for(self.data, "pieces"):
            return {"status": "nothing to do", "log": str(stem.with_name(stem.name + "_movement.log")),
                    "finished": datetime.now().isoformat(timespec="seconds")}
        args, extra = self._arguments(name, stem, blend_in, exported)
        extra = [*extra, *self.frame()]
        prints = _fingerprint_of(stem.with_suffix(".blend"))
        if kind == "blend":
            # the numbers the step's file is judged by (contracts), written as the step saves it
            extra = [*extra, "--python-expr", f"import os; os.environ['VTUBER_MAKER_FINGERPRINT'] = {str(prints)!r}",
                     "-P", str(FINGERPRINT)]
        log = stem.with_suffix(".log")
        started = time.time()
        # (what a stand-in for `_blender` isn't given, it needn't take: the tests' take five things)
        more = {key: value for key, value in (("worker", self._worker if kept else None), ("abandon", abandon)) if value is not None}
        code = _blender(self.blender, blend_in, args, log, extra, **more)
        if code == CANCELLED:
            raise Cancelled(f"{name}: stopped by the user")
        if code == ASIDE:
            raise SetAside(name)
        entry = {"status": "done", "log": str(log), "input": str(blend_in) if blend_in else None,
                 "finished": datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - started)}
        # what its Blender took, for the next build to go by (`_need`); a step the kept Blender ran
        # can't say for itself, and keeps what it took the last time it had a Blender of its own
        took = getattr(_USED, "memory", None)
        took = round(took / 2**20) if took else self.state["steps"].get(name, {}).get("peak_mb")
        if took:
            entry["peak_mb"] = took
        output = log.read_text(encoding="utf-8", errors="replace")
        entry[kind] = str(stem.with_suffix(f".{kind}"))
        if code == STOPPED and free_disk(folder) < DISK_FLOOR:
            # stopped on a full drive: what it had half written is no file of any use, and is the room
            # the user is about to be asked to find (a .blend, Blender's own "@" of one being saved, a .vrm)
            for half in (stem.with_suffix(".blend"), stem.with_suffix(".blend@"), stem.with_suffix(".blend1"), stem.with_suffix(".vrm")):
                with contextlib.suppress(OSError):
                    half.unlink()
        made = Path(entry[kind]).exists()
        if (code != 0 or not made) and found and code not in (STOPPED, CANCELLED):
            # the roles found by themselves must never cost the avatar: without them, as if the option were off
            _found_manifest(stem).replace(_found_undone(stem))
            log.replace(log.with_name(stem.name + "_found.log"))
            again = self._run_step(name, index, folder)
            again["warnings"] = [*again.get("warnings", []), "the roles found for pieces by themselves were not given "
                                 f"({', '.join(p['name'] for p in found)}): the step failed with them (see {stem.name}_found.log)"]
            return again
        if code != 0 or not made:
            # a step the engine itself stopped (STOPPED) had something to do and didn't get to. So had
            # one whose script fell over: read as "nothing to do" too, a repair that crashed was a
            # build quietly without it. The build still goes on without it (`carries_on`), and says so
            crashed = name in OPTIONAL and code != STOPPED and CRASHED in output
            if name in OPTIONAL and code != STOPPED and not crashed:
                return {"status": "nothing to do", "log": str(log), "finished": entry["finished"]}
            known = errors.classify(name, output)
            if known is None and free_disk(folder) < DISK_FLOOR:
                # Blender words a save that didn't fit many ways; on a full drive that is the reason
                known = errors.describe("VM-DISK-FULL")
            said = f" [{known['code']}] {known['what']}{' (' + known['detail'] + ')' if known.get('detail') else ''}; {known['fix']}" if known else ""
            if OUT_OF_MEMORY in output:
                raise StepFailed(f"{name}: Blender ran out of memory (see {log}){said}", known)
            raise StepFailed(f"{name}: failed with exit code {code} (see {log}){said}\n{_tail(output)}", known, carries_on=crashed)
        pictures = None
        if name == "export" and (not self.improving or self.keeping):
            # the export's last act, with the app's own Python: Blender's has no picture codec. Before
            # the file is checked, so what is checked is what ships; a .vrm taken in keeps its pictures
            pictures = slim.pictures(entry["vrm"])
            with open(log, "a", encoding="utf-8") as handle:
                handle.write(f"\npictures: {json.dumps(pictures)}\n")
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
                # what the step did otherwise than asked (a role it left off the face)
                if report.get("notes"):
                    entry["warnings"] = [*entry.get("warnings", []), *report["notes"]]
        if pictures is not None:
            entry["report"] = {**(entry.get("report") or {}), "pictures": pictures}
        if name in ("check", "vrm_check"):
            entry["report"] = {k: v for k, v in _read_json(entry["json"], {}).items() if k in ("failed", "worst", "unweighted", "spots")}
        if name == "vrm_check":
            # the file the check just re-imported: what it holds is judged with the check (verdict)
            entry["report"]["ships"] = shipped(exported or self._last_vrm())
            # ... and the file as apps draw it, by the app's own Python and a browser: what shows through
            # a hole cut in its surface (inside.py). Kept in the check's own report too, so that builds
            # are compared by it; a file that couldn't be drawn says why and the build goes on
            seen = inside.check(exported or self._last_vrm(), stem.with_name(stem.name + "_seen"))
            entry["report"]["seen"] = seen
            _write_json(entry["json"], {**_read_json(entry["json"], {}), "seen": seen})
            with open(log, "a", encoding="utf-8") as handle:
                handle.write(f"\nseen: {json.dumps(seen)}\n")
        if name in CHECKS:
            # the Movement page shows the latest of each
            shutil.copy2(entry["json"], self.out / f"{name}.json")
        return entry

    def _arguments(self, name, stem, blend_in, exported=None):
        """(script and its arguments, Blender arguments before -P). `exported`: the .vrm the exported
        file's check reads, when the state doesn't name it yet (`_side_by_side`)."""
        o, data = self.options(), self.data
        blend, vrm = stem.with_suffix(".blend"), stem.with_suffix(".vrm")
        source = self.model() if name in ("survey", "build") else manifest.path_of(data, "source")
        face = manifest.path_of(data, "face")
        if name == "survey":
            # the marked eyes say which pieces are the eyes (by size the fox girl's ears were, and her face was hair)
            return ["preflight.py", source, "--json", stem.with_suffix(".json"), "--height", o["height"],
                    *(["--face", face] if face else [])], []
        if name == "build" and self.keeping:
            # the height as the manifest has it: `o` reads it back from this step's own report
            return ["import_rigged.py", source, vrm, "--blend", blend, "--name", self.name, "--author", data.get("author") or "Unknown",
                    "--height", data["options"]["height"], *([] if o["hair_hits_body"] else ["--hair-head-only"])], NO_EXPORT
        if name == "build" and self.improving:
            return ["import_vrm.py", source, vrm, "--blend", blend, *(["--licence-read"] if o["licence_read"] else [])], NO_EXPORT
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
                    "--manifest", self._named(), "--height", o["height"], "--listing", self.out / "pieces.json"], []
        if name == "springs":
            return (["check_springs.py", "--json", stem.with_suffix(".json"), "--height", o["height"]],
                    WHOLE_BODY if self.improving else [])
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
            return ([script, "--json", stem.with_suffix(".json"), "--spots", stem.with_name(stem.name + "_spots")],
                    WHOLE_BODY if self.improving else [])
        if name == "vrm_check":
            # what really ships: the exported file, re-imported (the add-on once swapped the pose on export)
            exported = exported or self._last_vrm()
            if exported is None:
                raise refused("VM-NO-EXPORT", "vrm_check: no exported .vrm yet")
            if self.improving:
                # the file has its own bone names back: read with the pipeline's, all its meshes as one
                return ([script, exported, "--json", stem.with_suffix(".json")],
                        ["--python-exit-code", "1", "-P", str(SCRIPTS / "load_vrm_as_ours.py"), *WHOLE_BODY])
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
        elif name == "face_tracking" and o["eye_shown"]:
            args += ["--covered", "right" if o["eye_shown"] == "left" else "left"]
        if name == "pieces":
            # with the pieces the movement check found for itself, when there are any (_find_pieces)
            args += ["--manifest", _found_manifest(stem) if _found_manifest(stem).exists() else self.manifest_path]
        if name == "tpose":
            # an adopted rig's hands are left as modelled: the spec hands relay the fingers for bones
            # this build fitted, and on Tripo's own (flat, straight already) they fanned the fingers
            # out and bent the palm (model A (T-pose); the raw FBX's hands were clean)
            # Without either flag the step still straightens: it turned Tripo's thumb from 23 to 45
            # degrees, which is not the hand the user approved in Warudo (an earlier build, the raw hands)
            if self.data.get("rig") or self.keeping:
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
            elif o["eyes"] == "modelled":
                args.append("--modelled")
                if o["iris_shape"]:
                    args += ["--iris-shape", o["iris_shape"]]
                if o["iris_size"] != 1.0:
                    args += ["--iris-size", o["iris_size"]]
            if o["eye_shown"]:
                args += ["--only", o["eye_shown"]]
            for flag, key in (("--travel", "look_travel"), ("--blink", "blink_amount")):
                if o[key] != 1.0:
                    args += [flag, o[key]]
        if name == "mouth":
            fix = o["mouth_fix"]
            if fix.get("free"):
                args.append("--free")
            for key in ("scale", "rest", "lift"):
                if fix.get(key) is not None:
                    args += [f"--{key}", fix[key]]
        return args, NO_EXPORT

    def _named(self):
        """The manifest the movement check names pieces by: the pieces step's own when that step gave
        roles to pieces it found for itself (`auto_pieces`), so the check knows a mantle it sees
        stretch was made one; else the user's."""
        entry = self.state["steps"].get("pieces", {})
        report = entry.get("report") or {}
        given = Path(report.get("manifest") or "")
        return given if entry.get("status") == "done" and report.get("found") and given.is_file() else self.manifest_path

    def _find_pieces(self, stem, blend_in):
        """With `auto_pieces`: the pieces the arms leave behind or tear on the model as it stands before
        the pieces step, each with the role the movement check suggests (found_pieces), written with
        the user's own into `<stem>_manifest.json` for the step to read. [] when the check finds
        none, can't run, or its roles were undone on a first try: the user's pieces alone then."""
        if _found_undone(stem).exists():
            return []
        probe = stem.with_name(stem.name + "_movement")
        args, extra = self._arguments("movement", probe, blend_in)
        # by the user's own pieces alone: what an earlier run found is found again, or isn't
        args[args.index("--manifest") + 1] = self.manifest_path
        code = _blender(self.blender, blend_in, args, probe.with_suffix(".log"), [*extra, *self.frame()],
                        **({"worker": self._worker} if self._worker else {}))
        if code == CANCELLED:
            raise Cancelled("pieces: stopped by the user")
        found = found_pieces(_read_json(probe.with_suffix(".json"), {}), self.data["pieces"]) if code == 0 else []
        if found:
            data = json.loads(json.dumps({k: v for k, v in self.data.items() if not k.startswith("_")}))
            for key in manifest.PATHS:
                if key != "out_dir" and data.get(key):
                    data[key] = str(manifest.path_of(self.data, key))
            data["pieces"] = [*data["pieces"], *found]
            manifest.save(data, _found_manifest(stem))
        return found

    def _eyes_fall_back(self, name, error):
        """Whether an automatic iris the eye step refused should be tried again as a sliding patch
        (remembered in the state, so later builds of the same model slide straight away)."""
        refused = name == "eyes" and (error.known or {}).get("code") == "VM-EYES-REFUSED"
        now = self.options()["eyes"]
        if not refused or self.data["options"]["eyes"] is not None or now not in ("iris", "modelled", "slide"):
            return False
        if now == "slide" and self.state.get("eyes_fallback") != "slide":
            return False  # painted eyes, which slide from the start: there is nothing else to try
        # the established moving iris first (model A's builds, seen good in Warudo, keep it); then the
        # model's own iris discs, where it has them (the plain boy, the catgirl); the painted patch last.
        # A patch remembered from an older build and refused now (the fox girl's, from before her discs
        # could slide) leaves the two before it to try: the build stayed without eyes for good
        tried = self.__dict__.setdefault("_eyes_tried", set())
        tried.add(now)
        left = [mode for mode in ("iris", "modelled", "slide") if mode not in tried]
        if not left:
            return False
        if left[0] == "iris":
            self.state.pop("eyes_fallback", None)
        else:
            self.state["eyes_fallback"] = left[0]
        self._save_state()
        return True

    def _eyes_advice(self, name, error):
        """`error`, its advice changed when the eye step refused the mode that advice names."""
        known = error.known or {}
        advised = (known.get("options") or {}).get("eyes")
        # (the patch refused has no way advised with it, errors.EYES_REFUSED: this is where it gets one)
        patch = "face over the eye patch" in (known.get("detail") or "")
        if (name != "eyes" or known.get("code") != "VM-EYES-REFUSED"
                or not (patch or advised in {self.options()["eyes"], *getattr(self, "_eyes_tried", ())})):
            return error
        inserts = bool((self.survey() or {}).get("parts", {}).get("eye_inserts"))
        fix, options = EYES_REFUSED_AGAIN[inserts and self.data["options"]["eyes"] is not None]
        return StepFailed(str(error).replace(known["fix"], fix), {**known, "fix": fix, "options": options})

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


def _log_named(reason):
    """The log a failure's words name ("... (see <log>)"), for one kept before its log was kept with it."""
    found = re.search(r"\(see (.+?\.log)\)", str(reason or ""))
    return found.group(1) if found else None


def set_state_aside(manifest_path):
    """Move a character's unreadable state.json (and its backup) aside, so that the character opens
    again with its record started afresh: {"aside": [files]}. Nothing is deleted: the files keep
    their content under another name, for a report, and every run folder stays where it is. Only for
    a record that really can't be read: one that reads is left alone (ValueError)."""
    data = manifest.load(Path(manifest_path).resolve())
    out = manifest.path_of(data, "out_dir") or REPO / "out" / data["name"]
    state = out / "state.json"
    try:
        _load_state(state)
    except StepFailed:
        # (without the one-Blender lock: a build of another character holds it, and none of this one
        # can be running, since a character whose record doesn't read can't be opened to start one)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        moved = []
        for path in (state, state.with_name("state.json.bak")):
            if path.exists():
                aside = path.with_name(f"{path.name}.unreadable-{stamp}")
                path.replace(aside)
                moved.append(str(aside))
        return {"aside": moved}
    raise ValueError(f"{state} reads: nothing to set aside")


def _taken(result):
    """A step's result as the chain takes it when the step was run ahead: its entry, or what it raised."""
    if isinstance(result, BaseException):
        raise result
    return result


def found_pieces(movement, named):
    """Manifest pieces for what the movement check (its report, `movement`) found the arms leave
    behind or tear and has a role for among AUTO_ROLES: what "Apply this fix" on the Movement page
    writes, for every such piece at once. A piece the user named (`named`: the manifest's pieces)
    keeps the user's role, right or not. Each carries `found`, which the pieces step reads: what it
    gives by itself it measures, and undoes when it tears the rig."""
    taken = {piece["name"] for piece in named}
    found = []
    for row in movement.get("pieces", []):
        role = row.get("suggested")
        if role not in AUTO_ROLES or row.get("role"):
            continue
        name = f"{role}_{row['label']}"
        while name in taken:
            name += "_found"
        piece = {"name": name, "point": row["point"], "count": row["count"], "role": role, "found": True}
        if role == "follow_bone":
            bone = next((f["bone"] for f in row["findings"] if f.get("bone")), "")
            piece["bone"] = ("Left" if bone.startswith("Left") else "Right") + "UpperArm"
        taken.add(name)
        found.append(piece)
    return found


def _found_manifest(stem):
    return stem.with_name(stem.name + "_manifest.json")


def _found_undone(stem):
    return stem.with_name(stem.name + "_manifest_undone.json")


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
    worse += _face_changes(_face_keys(before), _face_keys(now))
    old, new = read(before, "movement"), read(now, "movement")
    if old is not None and new is not None:
        named =lambda report: {str(r.get("name") or r["label"]) for r in report.get("pieces", [])}  # noqa: E731
        worse += [f"movement: {n} newly flagged" for n in sorted(named(new) - named(old))]
        better += [f"movement: {n} no longer flagged" for n in sorted(named(old) - named(new))]
    old, new = read(before, "springs"), read(now, "springs")
    if old is not None and new is not None:
        # droop chain by chain: the report lists only chains over their limit, each kind with its own
        # (a hat tip 3 cm, cloth 8), and the largest of one build against the largest of the other
        # read model B's hat tip at 4.0 cm against its cape at 8.6 as one chain drooping twice as far
        sag_was = {r.get("chain"): r["sag_cm"] for r in old.get("sagging", [])}
        sag_now = {r.get("chain"): r["sag_cm"] for r in new.get("sagging", [])}
        # of the chains both list, the one that droops most further and the one that droops most less
        both = sorted(set(sag_was) & set(sag_now), key=lambda c: sag_now[c] - sag_was[c])
        rows = [(sag_was[c], sag_now[c], "droop at rest") for c in dict.fromkeys(both[-1:] + both[:1])]
        newly = sorted(set(sag_now) - set(sag_was), key=lambda c: -sag_now[c])
        gone = sorted(set(sag_was) - set(sag_now), key=lambda c: -sag_was[c])
        if newly:
            worse.append(f"physics: {len(newly)} chain{'s' if len(newly) > 1 else ''} newly over the droop limit at rest "
                         f"({newly[0]} {sag_now[newly[0]]} cm)")
        if gone:
            better.append(f"physics: {len(gone)} chain{'s' if len(gone) > 1 else ''} no longer over the droop limit at rest "
                          f"({gone[0]} was {sag_was[gone[0]]} cm)")
        rows.append((max((r["gap_cm"] for r in old.get("coming_away", [])), default=0.0),
                     max((r["gap_cm"] for r in new.get("coming_away", [])), default=0.0), "left behind by a swing"))
        # and piece by piece, each known by where it sits: the worst gap alone hid model A's chest
        # panels, newly turning with her head (22.5 and 14.4 cm), behind a bit of hair that had stood
        # still since September (29.8 cm)
        for found, others, where, what in ((new, old, worse, "newly"), (old, new, better, "no longer")):
            pieces = sorted((r for r in found.get("coming_away", []) if not any(_same_piece(r, o) for o in others.get("coming_away", []))),
                            key=lambda r: -r["gap_cm"])
            if pieces:
                where.append(f"physics: {len(pieces)} piece{'s' if len(pieces) > 1 else ''} {what} left behind by a swing "
                             f"({pieces[0]['count']} vertices, {pieces[0]['gap_cm']} cm)")
        for was, now_, what in rows:
            if now_ - was > SPRING_NOISE:
                worse.append(f"physics: worst {what} {was} -> {now_} cm")
            elif was - now_ > SPRING_NOISE:
                better.append(f"physics: worst {what} {was} -> {now_} cm")
    return {"worse": worse, "better": better}


def _face_keys(reports):
    """{shape key: {"moved", "max_mm", maybe "at"}} of a build, from its rig check; of a build older
    than that part of the check, from the numbers kept beside its last .blend (no places in those).
    A build marked good is read where its reports came from."""
    check = reports.get("check")
    if check is None:
        return {}
    keys = ((_read_json(check, None) or {}).get("face") or {}).get("keys")
    if keys is not None:
        return keys
    folder = Path(check).parent
    if folder.name == "approved":
        source = ((_read_json(folder / "approved.json", None) or {}).get("reports") or {}).get("check")
        folder = Path(source).parent if source else folder
    prints = sorted(folder.glob("*_fingerprint.json"))
    found = {}
    for mesh in (_read_json(prints[-1], None) or {}).get("keys", {}).values() if prints else ():
        found.update(mesh)
    return found


def _face_changes(old, new):
    """One line when shape keys both builds have no longer do what they did: gone, moving under half
    or over twice the vertices, reaching KEY_REACH further or less, or working KEY_PLACE (m) away.

    Model D's face points were saved again with the mouth on her throat: jawOpen went from 1116
    vertices and 18.5 mm to 133 and 10.8, through two days of regress runs that said "nothing worse".
    Which way is better can't be told from numbers, so every such change counts as worse: look.
    """
    changed = []
    for name in sorted(set(old) & set(new)):
        was, now = old[name], new[name]
        if not was.get("moved"):
            continue
        if not now.get("moved"):
            changed.append((9.0, f"{name} no longer moves anything (it moved {was['moved']} vertices)"))
            continue
        count = max(now["moved"], was["moved"]) / min(now["moved"], was["moved"])
        reach = abs(now["max_mm"] - was["max_mm"]) / max(was["max_mm"], 1e-9)
        place = (sum((a - b) ** 2 for a, b in zip(was["at"], now["at"])) ** 0.5 if "at" in was and "at" in now else 0.0)
        if place > KEY_PLACE:
            changed.append((place / KEY_PLACE, f"{name} works {place * 100:.1f} cm from where it did"))
        elif count > 2 or reach > KEY_REACH:
            changed.append((max(count / 2, reach / KEY_REACH),
                            f"{name} moves {now['moved']} vertices by up to {now['max_mm']} mm, was {was['moved']} by {was['max_mm']}"))
    gone = sorted(name for name, was in old.items() if was.get("moved") and name not in new)
    if gone and new:
        changed.append((9.0, f"{gone[0]} is gone"))
    if not changed:
        return []
    return [f"face shapes: {len(changed)} of {len(set(old) & set(new)) + len(gone) * bool(new)} changed, "
            f"{max(changed)[1]}"]


def _same_piece(a, b, reach=0.05):
    """Whether two builds' spring reports name one piece: by its place (labels are vertex numbers,
    which a rebuild changes), or by its size in a report without places."""
    if "point" in a and "point" in b:
        return sum((x - y) ** 2 for x, y in zip(a["point"], b["point"])) < reach ** 2
    return a.get("count") == b.get("count")


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


def delete_character(manifest_path, dry=False):
    """Delete a character: its output folder (builds, face points, paint, notes) and its manifest.

    Only a folder inside the app's own `out` is deleted: an `out_dir` written by hand can name any
    folder on the disk, and that one is left where it is and named (`left`). The model the
    character was made from stays unless it lies in the folder (a face patch does). Refuses while a
    build runs, and while another character reads a file in the folder: a copy (`duplicate`) is
    built from the original's face patch. `dry`: what would go, nothing touched.
    """
    path = Path(manifest_path).resolve()
    data = _read_json(path, None)
    if not isinstance(data, dict) or not {"source", "options"} <= set(data):
        raise ValueError(f"{path} isn't a character")
    data["_path"] = str(path)
    name = data.get("name") or path.stem
    out = (manifest.path_of(data, "out_dir") or REPO / "out" / name).resolve()
    inside = lambda p, folder: folder in Path(p).resolve().parents  # noqa: E731
    folder = out if out.parent == (REPO / "out").resolve() and out.is_dir() else None
    if folder is not None:
        for other in {*characters(), *map(Path, hidden_characters())}:
            theirs = _read_json(other, None) or {}
            if other.resolve() == path or inside(other, folder):
                continue
            theirs["_path"] = str(other.resolve())
            read = [key for key in manifest.PATHS if key != "out_dir" and theirs.get(key) and inside(manifest.path_of(theirs, key), folder)]
            if read:
                raise ValueError(f"{theirs.get('name') or other.stem} is built from files in {folder} ({', '.join(read)}): "
                                 "delete that character first, or take this one off the list instead")
    freed = tree_size(folder) if folder else 0
    freed += path.stat().st_size if folder is None or not inside(path, folder) else 0
    model = manifest.path_of(data, "reference") or manifest.path_of(data, "source")
    done = {"name": name, "folder": str(folder) if folder else None, "manifest": str(path), "freed": freed, "dry": dry,
            "left": str(out) if folder is None and out.is_dir() else None,
            "model": str(model) if model and model.exists() and not (folder and inside(model, folder)) else None}
    if dry:
        return done
    with _Lock():
        # the manifest last: a folder that wouldn't go (a file open in another program) leaves the
        # character on the list, to be deleted again
        if folder is not None:
            try:
                shutil.rmtree(folder)
            except OSError as error:
                raise StepFailed(f"{folder} couldn't be deleted whole ({error}): close what has its files open and delete it again") from None
        path.unlink(missing_ok=True)
    return done


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


def _orders(extra, script):
    """A step's Blender arguments (`extra`, then its script) as what scripts/_worker.py runs, in their
    order; None when one of them is neither a script nor a line of Python: that step keeps a Blender
    of its own."""
    run, tokens = [], [str(token) for token in extra]
    while tokens:
        token = tokens.pop(0)
        if token == "--python-expr" and tokens:
            run.append(["expr", tokens.pop(0)])
        elif token == "-P" and tokens:
            run.append(["file", tokens.pop(0)])
        elif token == "--python-exit-code" and tokens:
            tokens.pop(0)  # an error ends the step with 1 there too
        else:
            return None
    return [*run, ["file", str(SCRIPTS / script)]]


class _Worker:
    """One Blender kept through a run for the steps that start from a .blend (scripts/_worker.py).

    A Blender takes 1.6 s to start and switch the VRM Add-on on, and most of a build's steps are two or
    three seconds of work: started anew for each, a quarter of a build was Blenders starting. Each step
    is told to this one instead, which opens the step's .blend and runs its scripts as its own Blender
    would have, into the step's own log; the files it leaves are the same byte for byte (SETUP.md "One
    Blender for a build's steps"). Watched as any Blender is: stopped with the run, when it hangs, and
    when the memory or the drive runs out; one that dies under a step fails that step, and the next
    step starts another.
    """

    def __init__(self, blender, folder):
        self.blender, self.folder, self.process = str(blender), Path(folder), None

    def _start(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        with open(self.folder / "worker.log", "a", encoding="utf-8", errors="replace") as handle:
            self.process = subprocess.Popen([self.blender, "-b", "--factory-startup", "-P", str(STARTUP), "-P", str(WORKER)],
                                            stdin=subprocess.PIPE, stdout=handle, stderr=subprocess.STDOUT, cwd=REPO)
            dies_with_us(self.process)

    def run(self, blend_in, args, log, extra):
        """As `_blender`: the step's exit code, STOPPED or CANCELLED; None when the step can't be handed
        over (then nothing was started and nothing written)."""
        run = _orders(extra, args[0])
        if run is None:
            return None
        try:
            if self.process is None or self.process.poll() is not None:
                self._start()
        except OSError:
            self.process = None
            return None
        command, log = _command(self.blender, blend_in, args, extra), Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(" ".join(f'"{c}"' if " " in c else c for c in map(str, command)) + "\n", encoding="utf-8")
        done = log.with_name(log.name + ".done")
        done.unlink(missing_ok=True)
        order = {"blend": str(blend_in), "argv": [str(c) for c in command], "run": run, "log": str(log), "done": str(done)}
        started, looks = time.time(), 0
        try:
            self.process.stdin.write((json.dumps(order) + "\n").encode("utf-8"))
            self.process.stdin.flush()
        except OSError:
            pass  # it is gone: found dead just below
        while not done.exists():
            if self.process.poll() is not None:
                # a crash takes this Blender as it would have taken the step's own
                code, self.process = self.process.returncode, None
                with open(log, "a", encoding="utf-8") as tail:
                    tail.write(f"\nBlender ended under the step with exit code {code}\n")
                return code or 1
            time.sleep(POLL)
            looks += 1
            if CANCEL.is_set():
                self.close(kill=True)
                with open(log, "a", encoding="utf-8") as tail:
                    tail.write(f"\nCANCELLED by the user after {round(time.time() - started)} s\n")
                return CANCELLED
            if looks % WATCH:
                continue
            why = _stop_reason(log, started)
            if why is not None:
                self.close(kill=True)
                with open(log, "a", encoding="utf-8") as tail:
                    tail.write(f"\nSTOPPED by the engine after {round(time.time() - started)} s{why}\n")
                return STOPPED
        code = (_read_json(done, None) or {}).get("code", 1)
        done.unlink(missing_ok=True)
        return code

    def close(self, kill=False):
        """Ends the Blender: its input closed, it leaves by itself; one that doesn't is made to."""
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if kill:
                process.kill()
            else:
                process.stdin.close()
            process.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait()


def one_blender():
    """Whether a run keeps one Blender for its steps (`_Worker`): yes, unless out/settings.json says
    "one_blender": false or $OC2VRM_ONE_BLENDER is 0. The way back, should a step ever behave otherwise
    in company than alone."""
    if not WORKER.is_file():
        return False  # (a copy of the app without its script: every step would fail for want of it)
    said = os.environ.get("OC2VRM_ONE_BLENDER")
    if said is not None:
        return said.strip().lower() not in ("0", "no", "false", "off")
    return _read_json(SETTINGS, {}).get("one_blender") is not False


def no_space(error):
    """Whether an error of the system's says the drive is full (Python's ENOSPC; Windows' own two codes
    for it, which reach Python as other numbers when a write is refused half way)."""
    import errno
    return isinstance(error, OSError) and (error.errno == errno.ENOSPC or getattr(error, "winerror", None) in (39, 112))


_JOB = []


def dies_with_us(process):
    """Tie a Blender to this process: when the app ends, however it ends, so does the Blender.

    The app's window closed in the middle of a build (or the app killed in Task Manager) left its
    Blender running: Windows ends no child with its parent. It went on for minutes holding two or
    three gigabytes, wrote into a run nothing would read, and the next build, which finds the lock
    left by a dead process and takes it, ran beside it on a computer with room for one. A Windows job
    whose last handle is this process's: the system ends what is in it when the handle goes. Elsewhere
    nothing is done (Linux and macOS were never run; a terminal closed there takes its children along).
    Returns whether the process was tied."""
    if os.name != "nt" or not isinstance(getattr(process, "_handle", None), int):
        return False
    import ctypes
    from ctypes import wintypes
    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        if not _JOB:
            class Basic(ctypes.Structure):
                _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64), ("flags", wintypes.DWORD),
                            ("least_set", ctypes.c_size_t), ("most_set", ctypes.c_size_t), ("processes", wintypes.DWORD),
                            ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

            class Extended(ctypes.Structure):
                _fields_ = [("basic", Basic), ("io", ctypes.c_uint64 * 6), ("process_memory", ctypes.c_size_t),
                            ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
            kernel.CreateJobObjectW.restype = wintypes.HANDLE
            kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
            kernel.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
            kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
            job = kernel.CreateJobObjectW(None, None)
            limits = Extended()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not job or not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                return False
            _JOB.append(job)  # kept for as long as this process lives: its closing is what ends them
        return bool(kernel.AssignProcessToJobObject(_JOB[0], wintypes.HANDLE(process._handle)))
    except (OSError, AttributeError, ValueError):
        return False


def _blender(blender, blend_in, args, log, extra, worker=None, abandon=None):
    """One Blender run, output to `log`; returns the exit code. Stops a run that hangs, fills the
    drive or uses the last of the memory (STOPPED), and one the user stops (CANCELLED). With a
    `worker` (the run's one Blender) a step that starts from a .blend is told to it instead. With
    `abandon` (an Event), set to end this Blender for the sake of others (ASIDE). The most memory
    its own Blender had committed is left in `_USED.memory` for the caller, where it can be told."""
    _USED.memory = None
    if worker is not None and blend_in is not None:
        code = worker.run(blend_in, args, log, extra)
        if code is not None:
            return code
    command = _command(blender, blend_in, args, extra)
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8", errors="replace") as handle:
        handle.write(" ".join(f'"{c}"' if " " in c else c for c in map(str, command)) + "\n")
        handle.flush()
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=REPO)
        dies_with_us(process)
        started, looks = time.time(), 0
        while process.poll() is None:
            time.sleep(POLL)
            looks += 1
            if CANCEL.is_set():
                process.kill()
                process.wait()
                with open(log, "a", encoding="utf-8") as tail:
                    tail.write(f"\nCANCELLED by the user after {round(time.time() - started)} s\n")
                return CANCELLED
            if abandon is not None and abandon.is_set():
                process.kill()
                process.wait()
                with open(log, "a", encoding="utf-8") as tail:
                    tail.write(f"\nSET ASIDE by the engine after {round(time.time() - started)} s: to run again when there is room\n")
                return ASIDE
            if looks % WATCH:
                continue
            why = _stop_reason(log, started)
            if why is not None:
                process.kill()
                process.wait()
                with open(log, "a", encoding="utf-8") as tail:
                    tail.write(f"\nSTOPPED by the engine after {round(time.time() - started)} s{why}\n")
                return STOPPED
        _USED.memory = _peak_memory(process)
    return process.returncode


# What the last Blender this thread ran took (`_blender`): its peak of committed memory, in bytes.
_USED = threading.local()


def _peak_memory(process):
    """The most memory a process had committed at once (bytes), asked of Windows while its handle is
    held; None elsewhere, and for a stand-in process. What the next build goes by when it weighs
    whether another Blender fits beside the ones running (`Project._need`)."""
    if os.name != "nt" or not isinstance(getattr(process, "_handle", None), int):
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD),
                    *((name, ctypes.c_size_t) for name in ("peak_set", "set", "peak_paged", "paged", "peak_unpaged", "unpaged",
                                                           "commit", "peak_commit"))]
    counters = Counters(size=ctypes.sizeof(Counters))
    try:
        told = ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(int(process._handle)), ctypes.byref(counters), counters.size)
    except (OSError, AttributeError, ValueError):
        return None
    return int(counters.peak_commit) if told else None


class _Load:
    """How many processor cores stand idle, from two looks at the system's own count of its time."""

    def __init__(self):
        self.last = self.at = self.idle = None

    @staticmethod
    def _times():
        """(idle, all) processor time so far, in one unit; None where the system doesn't say."""
        if os.name == "nt":
            import ctypes
            idle, kernel, user = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
            if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                return None
            return idle.value, kernel.value + user.value  # (the kernel's time holds the idle time)
        try:
            fields = [int(value) for value in Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()[1:]]
            return fields[3] + fields[4], sum(fields[:8])
        except (OSError, ValueError, IndexError):
            return None

    def idle_cores(self, wait=0.0):
        """Cores idle since the last look, or over `wait` seconds from now when there was none; None
        where it can't be told. Looked at no oftener than twice a second: shorter than that is noise."""
        cores, now = os.cpu_count() or 1, self._times()
        if now is None:
            try:
                return max(cores - os.getloadavg()[0], 0.0)
            except (OSError, AttributeError):
                return None
        if wait:  # a look of its own, from now: the last one may be an hour old
            self.last, self.at = now, time.time()
            time.sleep(wait)
            now = self._times() or now
        elif self.last is None or time.time() - self.at < 0.5:
            if self.last is None:
                self.last, self.at = now, time.time()
            return self.idle
        if now[1] > self.last[1]:
            self.idle = cores * (now[0] - self.last[0]) / (now[1] - self.last[1])
        self.last, self.at = now, time.time()
        return self.idle


LOAD = _Load()


def blenders_at_once():
    """The most Blenders a run may have going at once, as the user set it (out/settings.json
    "blenders", or $OC2VRM_BLENDERS): a number, 1 for never more than one. None, the default: as many
    as the computer has room for when they are wanted (`Project._room`)."""
    said = os.environ.get("OC2VRM_BLENDERS") or _read_json(SETTINGS, {}).get("blenders")
    try:
        return max(int(said), 1)
    except (TypeError, ValueError):
        return None


def blenders_now():
    """(how many Blenders a build could have going at once as the computer stands, what holds it to
    that): for Setup to say. Each is taken to need BLENDER_NEEDS here; a build asks again as it goes,
    and by what its own steps took the last time (`Project._room`)."""
    limit = blenders_at_once()
    if limit == 1:
        return 1, "set to one at a time"
    most, free = min(limit or MOST_BLENDERS, MOST_BLENDERS), free_memory()
    if free is None:
        return 1, "this system doesn't say how much memory is left"
    idle = LOAD.idle_cores()
    if idle is None:
        idle = LOAD.idle_cores(wait=0.1)
    by_memory = int((free - MEMORY_LOW) // BLENDER_NEEDS)
    by_cores = most if idle is None else int(idle // CORES_SPARE)
    count = max(1, min(most, by_memory, by_cores))
    why = ("" if count == most else f"{free / 2**30:.0f} GB of memory to be had" if by_memory <= by_cores
           else f"{idle:.0f} of {os.cpu_count()} processor cores idle")
    return count, why


def use_blenders(count):
    """Never more than this many Blenders at once from now on (1: one at a time); "auto" or nothing:
    as many as there is room for. Returns setup()."""
    settings = _read_json(SETTINGS, {})
    if count in (None, "", "auto"):
        settings.pop("blenders", None)
    else:
        try:
            settings["blenders"] = max(int(count), 1)
        except (TypeError, ValueError):
            raise ValueError(f"{count!r} isn't a number of Blenders: give 1, 2, 3, 4 or auto") from None
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    _write_json(SETTINGS, settings)
    return setup()


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


def _size(path):
    """A file's size in bytes; 0 for one that went since it was listed."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def tree_size(folder):
    """Bytes of every file under a folder, as it stands; 0 for one that isn't there.

    Counted while a build runs, what goes meanwhile counts for nothing: Blender saves under another
    name and renames, and the browser that draws the exported file fills and empties a profile of
    thousands of files inside the run's folder. Listed first and measured after, one such file ended
    the whole count: the character missing from the disk table, or the table not drawn at all.
    """
    total = 0
    for base, _, names in os.walk(folder):
        total += sum(_size(os.path.join(base, name)) for name in names)
    return total


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
        # held once for several runs of this process side by side (`together`): their own takings pass
        self.passed = SHARED.held
        if self.passed:
            return self
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                held = LOCK.read_text(errors="replace").strip()
            except FileNotFoundError:  # let go between our try and this look: free, as a dead holder's
                held = "pid 0"
            found = re.match(r"pid (\d+)", held)
            if not found or (int(found.group(1)) and _alive(int(found.group(1)))):
                raise refused("VM-BUSY", f"another build is running ({held}); if none is, delete {LOCK}", held) from None
            # left by a process that crashed or was killed: without this the user had to find and
            # delete the file by hand. A second taker racing us still loses on O_EXCL.
            LOCK.unlink(missing_ok=True)
            try:
                handle = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                raise refused("VM-BUSY", f"another build is running ({LOCK.read_text(errors='replace').strip()})") from None
        os.write(handle, f"pid {os.getpid()} since {datetime.now().isoformat(timespec='seconds')}".encode())
        os.close(handle)
        return self

    def __exit__(self, *exc):
        if not self.passed:
            LOCK.unlink(missing_ok=True)


# The lock held once for several runs of this process (`together`), and how many characters `at_once`
# has building or waiting to as it stands now.
SHARED = SimpleNamespace(held=False, crowd=lambda: 0)


@contextlib.contextmanager
def together():
    """The one-build lock held once for several runs of this process side by side (`at_once`): each
    run's own taking of it passes. While more than one character is building or waiting to
    (`SHARED.crowd`), each keeps to one Blender (`Project._ahead`), so that what is counted against
    the computer's room is characters."""
    with _Lock():
        SHARED.held = True
        try:
            yield
        finally:
            SHARED.held, SHARED.crowd = False, lambda: 0


def at_once(projects, work, on_event=None, pause=0.5):
    """`work(project, emit)` for each of `projects`, as many at once as the computer has room for:
    what each returned, or what it raised, in their order.

    For a command that builds several characters (`regress`): they waited for each other, a minute
    and a half each, on a computer with room for two or three. The lock is held once for all of them
    (`together`) and each run keeps to one Blender while another builds or waits, so a character is
    a Blender: another starts only when `Project._room` says so of its rigging step, the heaviest
    (2.2 GB on every model of about 50,000 triangles), asked again whenever one ends. As a build's
    own last steps (`_lanes`): one that fails while others run beside it is run once more by itself,
    last, before its failure is believed. A computer with no room, or told to keep to one Blender
    (`blenders`), runs them one after another; the last one left, and one asked for alone, runs its
    last steps side by side as any build. Every event says its `character`.
    """
    emit = on_event or (lambda event: None)
    results = [None] * len(projects)
    waiting, alone, running, tried = list(range(len(projects))), [], {}, set()
    free = free_memory()
    budget = float("inf") if free is None else free - MEMORY_LOW

    def run(index, lane):
        try:
            lane.result = work(projects[index], lambda event: emit({**event, "character": projects[index].name}))
        except BaseException as error:  # noqa: BLE001 -- kept, to be handed back in its place
            lane.result = error

    def start(index):
        lane = SimpleNamespace(need=projects[index]._need("build"), result=None, company=bool(running))
        lane.thread = threading.Thread(target=run, args=(index, lane), daemon=True)
        running[index] = lane
        if len(running) > 1:
            for other in running.values():
                other.company = True
        lane.thread.start()

    with together():
        SHARED.crowd = lambda: len(waiting) + len(alone) + len(running)
        while waiting or alone or running:
            for index, lane in list(running.items()):
                if lane.thread.is_alive():
                    continue
                del running[index]
                if isinstance(lane.result, StepFailed) and lane.company and index not in tried and not CANCEL.is_set():
                    tried.add(index)
                    alone.append(index)
                    emit({"step": projects[index].name, "character": projects[index].name, "status": "warning",
                          "reason": f"{projects[index].name} failed with other characters building beside it: "
                                    "it is built again by itself, last, before that is believed"})
                    continue
                results[index] = lane.result
            if CANCEL.is_set():
                for index in waiting + alone:
                    results[index] = Cancelled("stopped by the user")
                waiting, alone = [], []
            for index in list(waiting):
                if not projects[index]._room("build", {other: lane.need for other, lane in running.items()}, budget):
                    break
                waiting.remove(index)
                start(index)
            if alone and not waiting and not running:
                start(alone.pop(0))
            if waiting or alone or running:
                time.sleep(pause)
    return results


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
        return _shipped_0(path, gltf)
    nodes = gltf.get("nodes", [])
    bones = (vrm.get("humanoid") or {}).get("humanBones") or {}
    joints = sorted({j for skin in gltf.get("skins", []) for j in skin.get("joints", [])})
    presets = (vrm.get("expressions") or {}).get("preset") or {}
    bound = {name for name, e in presets.items() if e.get("morphTargetBinds")}
    springs = gltf["extensions"].get("VRMC_springBone") or {}
    materials = gltf.get("materials", [])
    return {"bones": len(bones), "fingers": sum(1 for b in bones if any(f in b for f in FINGERS)),
            "turned_joints": _turned(nodes, joints),
            "arms_off_level": _arms_off_level(nodes, bones),
            "blink": "blink" in bound, "vowels": [v for v in VOWELS if v in bound],
            "look": (vrm.get("lookAt") or {}).get("type"), "looks_bound": all(k in bound for k in LOOKS),
            "shapes": sorted({n for mesh in gltf.get("meshes", []) for n in (mesh.get("extras") or {}).get("targetNames", [])}),
            "springs": len(springs.get("springs", [])), "colliders": len(springs.get("colliders", [])),
            "materials": len(materials),
            "toon": sum("VRMC_materials_mtoon" in (m.get("extensions") or {}) for m in materials),
            "textures": len(gltf.get("images", [])), "megabytes": round(Path(path).stat().st_size / 2**20, 1)}


# VRM 0.x preset names as the 1.0 ones `shipped` speaks in.
PRESETS_0 = {"a": "aa", "i": "ih", "u": "ou", "e": "ee", "o": "oh", "blink": "blink", "lookup": "lookUp",
             "lookdown": "lookDown", "lookleft": "lookLeft", "lookright": "lookRight"}


def _shipped_0(path, gltf):
    """`shipped` for a VRM 0.x file, in the same words, so what a converted avatar came with can
    be held against what left (`lost`); None when the file is no VRM at all. A spring per root
    bone and every collider counted singly, as the VRM Add-on makes of them on the way to 1.0."""
    vrm = ((gltf or {}).get("extensions") or {}).get("VRM")
    if not isinstance(vrm, dict) or "humanoid" not in vrm:
        return None
    nodes = gltf.get("nodes", [])
    bones = {b["bone"]: {"node": b["node"]} for b in vrm["humanoid"].get("humanBones", []) if "bone" in b and "node" in b}
    joints = sorted({j for skin in gltf.get("skins", []) for j in skin.get("joints", [])})
    groups = (vrm.get("blendShapeMaster") or {}).get("blendShapeGroups") or []
    bound = {PRESETS_0.get(g.get("presetName")) for g in groups if g.get("binds")} - {None}
    springs = vrm.get("secondaryAnimation") or {}
    named = [mesh.get("extras") or {} for mesh in gltf.get("meshes", [])]
    named += [p.get("extras") or {} for mesh in gltf.get("meshes", []) for p in mesh.get("primitives", [])]
    return {"bones": len(bones), "fingers": sum(1 for b in bones if any(f in b for f in FINGERS)),
            # (the arms' level is read on 1.0's axes: a 0.x file faces the other way)
            "turned_joints": _turned(nodes, joints), "arms_off_level": None,
            "blink": "blink" in bound, "vowels": [v for v in VOWELS if v in bound],
            "look": {"Bone": "bone", "BlendShape": "expression"}.get((vrm.get("firstPerson") or {}).get("lookAtTypeName")),
            "looks_bound": all(k in bound for k in LOOKS),
            "shapes": sorted({n for extras in named for n in extras.get("targetNames", [])}),
            "springs": sum(len(g.get("bones", [])) for g in springs.get("boneGroups", [])),
            "colliders": sum(len(g.get("colliders", [])) for g in springs.get("colliderGroups", [])),
            "materials": len(gltf.get("materials", [])),
            "toon": sum(p.get("shader") == "VRM/MToon" for p in vrm.get("materialProperties", [])),
            "textures": len(gltf.get("images", [])), "megabytes": round(Path(path).stat().st_size / 2**20, 1)}


# Licences a VRM 0.x file can name that forbid changing it, and the two licence pages the VRM
# Add-on reads by itself (with the answer on each that still wants a person to look).
NO_CHANGES = ("CC_BY_ND", "CC_BY_NC_ND")
READ_PAGES = {"hub.vroid.com": ("modification", "disallow"), "uv-license.com": ("remarks", "true")}


def licence(path):
    """What a .vrm says of who made it and what may be done with it, for the page that takes it in:
    {"spec", "name", "author", "licence", "pages", "confirm", "why"}, or None when it isn't one.

    `confirm`: the user has to read the terms and say they may change the avatar before it is
    taken in. The VRM Add-on's own rule (importer/license_validation.py), kept here so it can be
    asked before a Blender starts: a VRM 0.x file that forbids derivatives, or whose terms are a
    web page the add-on can't read. A VRM 1.0 file's terms the add-on doesn't judge, and neither
    does this. The build asks the add-on itself (improve.licence_stop): that answer is the one that counts.
    """
    from urllib.parse import parse_qsl, urlsplit
    extensions = (_glb_json(path) or {}).get("extensions") or {}
    if "VRMC_vrm" in extensions:
        meta = extensions["VRMC_vrm"].get("meta") or {}
        return {"spec": "1.0", "name": meta.get("name"), "author": ", ".join(meta.get("authors") or []),
                "licence": meta.get("modification"), "pages": [u for u in (meta.get("otherLicenseUrl"),) if u],
                "confirm": False, "why": ""}
    meta = (extensions.get("VRM") or {}).get("meta")
    if not isinstance(meta, dict):
        return None
    name = meta.get("licenseName")
    pages = [u for u in dict.fromkeys((meta.get("otherLicenseUrl") if name == "Other" else None, meta.get("otherPermissionUrl"))) if u]
    why = []
    if name in NO_CHANGES:
        why.append(f"its licence ({name.replace('_', ' ')}) doesn't allow changed versions")
    if name == "Other" and not meta.get("otherLicenseUrl"):
        why.append("it names its own licence and gives no address for it")
    for page in pages:
        key = stop = None
        try:
            url = urlsplit(page)
            key, stop = READ_PAGES.get(url.hostname if url.path.endswith("/license") else None, (None, None))
        except ValueError:
            pass
        if key is None:
            why.append(f"its terms are on a page to read: {page}")
        elif dict(parse_qsl(url.query)).get(key) == stop:
            why.append(f"its licence page says changes need a look: {page}")
    return {"spec": "0.x", "name": meta.get("title"), "author": meta.get("author"), "licence": name, "pages": pages,
            "confirm": bool(why), "why": "; ".join(why)}


def gained(before, after):
    """What the exported file has over the .vrm it was made from (both `shipped`), as lines for
    the Build page; [] before the build has run to its end, or for a character built from a model."""
    if not before or not after:
        return []
    lines = []
    was, now = len(before["turned_joints"]), len(after["turned_joints"])
    if now < was:
        lines.append(f"{was} joints rested turned in the file, {now or 'none'} now: tracking apps turn a bone about "
                     "its own axes, and Warudo twists the fingers of a file whose joints rest turned")
    for key, what in (("springs", "springs"), ("colliders", "spring colliders"), ("fingers", "finger bones")):
        if after[key] > before[key]:
            lines.append(f"{after[key] - before[key]} {what} added ({before[key]} before, {after[key]} now)")
    new = sorted(set(after["shapes"]) - set(before["shapes"]))
    if new:
        lines.append(f"{len(new)} face shapes added")
    return lines


def lost(before, after):
    """What a .vrm taken in had (`shipped`) that the exported one no longer has, as lines; [] when
    nothing went missing, or for a character that didn't start from a .vrm (`before` None)."""
    if not before or not after:
        return []
    lines = []
    for key, what in (("bones", "humanoid bones"), ("fingers", "finger bones"), ("springs", "springs"),
                      ("colliders", "spring colliders"), ("materials", "materials"), ("toon", "toon materials"),
                      ("textures", "pictures")):
        if after[key] < before[key]:
            lines.append(f"{after[key]} {what}, the file taken in had {before[key]}")
    gone = sorted(set(before["shapes"]) - set(after["shapes"]))
    if gone:
        lines.append(f"{len(gone)} face shapes of the file taken in are gone ({', '.join(gone[:4])}{', ...' if len(gone) > 4 else ''})")
    if before["blink"] and not after["blink"]:
        lines.append("the file taken in could blink; this one can't")
    mute = [v for v in before["vowels"] if v not in after["vowels"]]
    if mute:
        lines.append(f"the file taken in had mouth shapes for {', '.join(mute)}; this one doesn't")
    if before["look"] != after["look"] or (before["looks_bound"] and not after["looks_bound"]):
        lines.append("the eyes no longer follow the gaze the way they did in the file taken in")
    return lines


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


def _turned(nodes, joints):
    """Names of the joints written with a turn or a scale of their own. Two kinds are left out,
    both only in a .vrm taken in from elsewhere: joints a node constraint is on or reads (an aim
    joint's own axis has to point at its source at rest, vrm_export.normalize_bones), and a joint
    under one of those that is turned back against it and so rests on the model's axes."""
    held = set()
    for i, node in enumerate(nodes):
        for constraint in ((node.get("extensions") or {}).get("VRMC_node_constraint") or {}).get("constraint", {}).values():
            held |= {i, constraint.get("source")}
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    worlds = {}

    def world(i):
        if i not in worlds:
            local = _rotation(nodes[i])
            above = world(parent[i]) if i in parent else None
            worlds[i] = local if above is None else [[sum(above[r][k] * local[k][c] for k in range(3)) for c in range(3)]
                                                     for r in range(3)]
        return worlds[i]

    straight = lambda m: all(abs(m[r][c] - (r == c)) < 1e-5 for r in range(3) for c in range(3))  # noqa: E731
    return [nodes[j].get("name", str(j)) for j in joints
            if j not in held and not _unrotated(nodes[j]) and not straight(world(j))]


def _rotation(node):
    """A glTF node's turn and scale as a 3x3 matrix (rows), from its matrix or its rotation and scale."""
    if "matrix" in node:
        m = node["matrix"]  # column-major
        return [[m[c * 4 + r] for c in range(3)] for r in range(3)]
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    sx, sy, sz = node.get("scale", (1.0, 1.0, 1.0))
    return [[(1 - 2 * (y * y + z * z)) * sx, 2 * (x * y - z * w) * sy, 2 * (x * z + y * w) * sz],
            [2 * (x * y + z * w) * sx, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z - x * w) * sz],
            [2 * (x * z - y * w) * sx, 2 * (y * z + x * w) * sy, (1 - 2 * (x * x + y * y)) * sz]]


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
    # git's word in a checkout, the version file's in a tester's copy: asked of git alone, every run a
    # tester sent said "commit": null, and which version built it was read off the size of its check
    try:
        here = update.current()
    except Exception:  # noqa: BLE001 -- provenance is a record, never a reason to stop
        here = {}
    commit, dirty = here.get("commit"), here.get("changed") if here.get("commit") else None
    try:
        digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()[:16]
    except OSError:
        digest = None
    return {"started": datetime.now().isoformat(timespec="seconds"), "blender": str(blender),
            "blender_version": version, "vrm_addon": vrm_addon_version(blender, version), "commit": commit,
            "dirty": dirty, "app": here.get("version") or None, "manifest": str(manifest_path), "manifest_sha256": digest,
            "python": platform.python_version(), "warnings": warnings}


def blender_version(blender):
    """"4.2.1" as that Blender prints it, or None when it won't say. Asked once per path."""
    blender = str(blender)
    if blender not in _BLENDER_VERSIONS:
        try:
            # (errors="replace", as `install`: a byte Windows' code page has no letter for, in a warning's
            # path, and nothing came back at all: a Blender that runs read as one that doesn't start)
            out = subprocess.run([blender, "--version"], capture_output=True, text=True, errors="replace", timeout=60).stdout
            found = re.search(r"Blender (\d+\.\d+\.\d+)", out)
            _BLENDER_VERSIONS[blender] = found.group(1) if found else None
        except Exception:  # noqa: BLE001 -- a record and a Setup line, never a reason to stop
            _BLENDER_VERSIONS[blender] = None
    return _BLENDER_VERSIONS[blender]


def vrm_addon_version(blender, version):
    """The VRM Add-on installed for this Blender, from the extension's own manifest, or None.
    A portable Blender keeps its extensions in its own `portable` folder, an installed one in
    the user's settings folder under its series (`blender_settings`)."""
    series = ".".join((version or "4.2").split(".")[:2])
    for home in (Path(blender).parent / "portable", blender_settings(series)):
        for toml in sorted((home / "extensions").glob("*/vrm/blender_manifest.toml")):
            found = re.search(r'^version\s*=\s*"([^"]+)"', toml.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
            if found:
                return found.group(1)
    return None


def blender_settings(series, system=None, env=None, home=None):
    """Where an installed Blender of this series ("5.2") keeps the user's settings and extensions.

    Windows: %APPDATA%\\Blender Foundation\\Blender; macOS: ~/Library/Application Support/Blender;
    Linux: $XDG_CONFIG_HOME/blender, else ~/.config/blender. Until 2026-10-07 the Windows shape was
    looked in everywhere, so on macOS and Linux an installed add-on read as missing.
    """
    system, env, home = system or sys.platform, os.environ if env is None else env, Path(home or Path.home())
    if system == "win32":
        return Path(env.get("APPDATA") or home / "AppData" / "Roaming") / "Blender Foundation" / "Blender" / series
    if system == "darwin":
        return home / "Library" / "Application Support" / "Blender" / series
    return Path(env.get("XDG_CONFIG_HOME") or home / ".config") / "blender" / series


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
        # ... as is one whose folder names no version (macOS: Blender.app/Contents/MacOS)
        version = (None if missing else blender_version(path)) if added or not named else named.group(0)
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
    home = REPO / "tools" / "blender"
    return sorted([*home.glob("*/blender.exe"), *(p for p in home.glob("*/blender") if p.is_file())])


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
    # `action`: what the first-run page's button does about it ("get_blender", "install_addon", "install:<package>")
    add = lambda name, state, text, fix="", action="": checks.append(  # noqa: E731
        {"name": name, "state": state, "text": text, "fix": fix, **({"action": action} if action else {})})
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
            f"the app can fetch Blender {blender_build()['version']} for itself ({blender_build()['bytes'] >> 20} MB, kept in its own folder), "
            "or install it from blender.org/download" if blender_build()
            else "install Blender 5.2 LTS from blender.org/download; the app finds it in its usual place",
            "install_blender" if blender_build() else "get_blender")
    elif version is None:
        add("Blender", "bad", f"{blender} doesn't start", "install Blender 5.2 LTS from blender.org/download", "get_blender")
    elif tuple(map(int, version.split("."))) < MIN_BLENDER:
        add("Blender", "bad", f"Blender {version} is older than 4.2, which the app needs", "install Blender 5.2 LTS from blender.org/download",
            "get_blender")
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
                "open Blender, Edit > Preferences > Get Extensions, search for \"VRM format\" and install it", "install_addon")
    for module, name, state, what, fix, action in (
            ("PIL", "Pictures", "bad", "Pillow isn't installed: the face patch can't be made", "pip install pillow", "install:pillow"),
            ("tkinter", "File dialogs", "warn", "tkinter is missing: Browse buttons won't open, type paths instead", "", ""),
            ("webview", "Own window", "warn", "pywebview isn't installed: the app opens in your browser instead of its own window",
             "optional: pip install pywebview", "install:pywebview")):
        if not _installed(module):
            add(name, state, what, fix, action)
    detector = face_guess.status()["python"]
    if detector:
        add("Auto face marking", "ok", "the face detector is installed: the Face page's \"Place them for me\" works")
    else:
        add("Auto face marking", "warn", "the face detector isn't installed: the eyes and mouth are marked by hand on the Face page",
            "optional: the button below installs it (about 250 MB to download, 1.5 GB on disk), or python -m app detector --install")
    for note in resource_warnings(REPO / "out"):
        add("Space", "warn", note)
    room, held = blenders_now()
    add("Speed", "ok", f"room for {room} Blender{'s' if room > 1 else ''} at once as the computer stands" + (f" ({held})" if held else "")
        + ": the last steps of a build run side by side when there is room for more than one")
    kept = vendor.status()
    if not kept["views"]:
        add("Fully local", "warn", "not yet: the 3D views load three.js from the internet each time, and stay empty without a connection",
            "keep a copy with the app (17 MB, from jsDelivr and Google): the button below, or python -m app vendor", "vendor")
    elif not kept["tracker"]:
        add("Fully local", "warn", "nearly: the 3D views are kept with the app; the viewer's webcam test still loads its face tracker "
            "from the internet", "keep a copy with the app (14 MB): the button below, or python -m app vendor", "vendor")
    else:
        add("Fully local", "ok", "everything runs on this computer: your models and avatars never leave it, and no connection is needed")
    return {"blender": blender, "version": version, "chosen": chosen, "blenders": blenders,
            "checks": checks, "ready": not any(c["state"] == "bad" for c in checks),
            # for the first-run page: whether it was finished, and what a tester quotes in a report
            "welcomed": bool(_read_json(SETTINGS, {}).get("welcomed")), "python": platform.python_version(),
            **open_in(),
            "warudo": str(warudo_folder() or ""), "app": app_version()}


def app_version():
    """The app's version as people say it and its commit, "v1 (6473e9a)", with "+" when tracked files
    differ from that commit; "" when it can't be told. A git checkout says so itself, a packed copy (testers') reads the file its zip or its last
    update wrote (update.current). What a tester quotes so a report can be traced to its code."""
    try:
        here = update.current()
        code = here["short"] + ("+" if here["short"] and here["changed"] else "")
        return f"{here['version']} ({code})" if here["version"] and code else here["version"] or code
    except Exception:  # noqa: BLE001 -- a label, never a reason to stop
        return ""


def open_in(choice=None):
    """Where the app opens: "window" (its own, with pywebview; the default) or "browser". With a
    `choice`, keeps it for the next start. {"open_in", "can_window"}: without pywebview the browser
    is what opens, whatever is chosen."""
    settings = _read_json(SETTINGS, {})
    if choice is not None:
        if choice not in ("window", "browser"):
            raise ValueError("the app opens in its own window (window) or in the browser (browser)")
        settings["open_in"] = choice
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        _write_json(SETTINGS, settings)
    return {"open_in": "browser" if settings.get("open_in") == "browser" else "window", "can_window": _installed("webview")}


def welcome(done=True):
    """Remember that the first-run page was finished (or, False, show it again at the next start)."""
    settings = _read_json(SETTINGS, {})
    settings["welcomed"] = bool(done)
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    _write_json(SETTINGS, settings)
    return {"welcomed": bool(done)}


# The app's own Python packages the first-run page may install, by the module that shows they are there.
PACKAGES = {"pillow": "PIL", "pywebview": "webview"}


def package_command(name):
    """The pip command that installs one of PACKAGES into the Python running the app."""
    if name not in PACKAGES:
        raise ValueError(f"{name} isn't a package the app installs ({', '.join(PACKAGES)})")
    return [sys.executable, "-m", "pip", "install", name]


def addon_command(blender):
    """The command that has a Blender install the VRM Add-on from its own extensions site and
    switch it on: what Get Extensions > "VRM format" > Install does, without opening Blender."""
    return [str(blender), "--online-mode", "--command", "extension", "install", "--sync", "--enable", "vrm"]


def install(command, what, on_event=None, timeout=900):
    """Run an install command the user asked for and say how it went: {"installed", "log"}.
    Raises StepFailed with the command's last lines when it fails."""
    emit = on_event or (lambda event: None)
    emit({"step": what, "status": "running"})
    try:
        # errors="replace": pip and Blender print paths as UTF-8, and read strictly in Windows' own
        # code page a user name with an accent turned a good install into a traceback
        done = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout, cwd=REPO)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise StepFailed(f"{what}: {error}") from error
    log = (done.stdout or "") + (done.stderr or "")
    if done.returncode != 0:
        raise StepFailed(f"{what} failed (exit code {done.returncode}):\n{_tail(log, 1200)}")
    emit({"step": what, "status": "done"})
    return {"installed": what, "log": _tail(log, 1200)}


# The Blender the app fetches for itself: the tested LTS as blender.org's own Windows zip, with the
# checksum blender.org publishes beside it (release/Blender5.2/blender-5.2.2.sha256). Unzipped into
# tools/blender it is a portable copy: nothing is installed on the computer, and deleting the folder removes it.
BLENDER_ZIP = {"version": "5.2.2", "folder": "blender-5.2.2-windows-x64", "bytes": 404453484,
               "url": "https://download.blender.org/release/Blender5.2/blender-5.2.2-windows-x64.zip",
               "sha256": "3849d17a682cba006075aaa3f3597ecb5c9c30ec31035b2e092c53e40679b535"}
# The same for Linux (x64): blender.org's tarball, unpacked the same way. Not run on a Linux computer yet.
BLENDER_TAR = {"version": "5.2.2", "folder": "blender-5.2.2-linux-x64", "bytes": 383295504, "exe": "blender",
               "url": "https://download.blender.org/release/Blender5.2/blender-5.2.2-linux-x64.tar.xz",
               "sha256": "84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168"}
# By sys.platform. macOS has none: its Blender is a disk image to drag into Applications, where the app finds it.
BLENDER_BUILDS = {"win32": BLENDER_ZIP, "linux": BLENDER_TAR}
# Free space the zip and its 1.3 GB of files need together, with room to spare.
BLENDER_SPACE = 3 << 30


def blender_build(system=None, machine=None):
    """The Blender the app can fetch for this computer (BLENDER_BUILDS), or None: another system,
    or a processor blender.org's x64 build doesn't run on."""
    machine = (machine or platform.machine()).lower()
    return BLENDER_BUILDS.get(system or sys.platform) if machine in ("amd64", "x86_64") else None


def _unpack(archive, folder):
    """Unpack blender.org's zip or tarball into `folder`. A tarball's members are kept inside it
    (Python 3.12's "data" filter where there is one; else any path leading out is refused)."""
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as packed:
            packed.extractall(folder)
        return
    with tarfile.open(archive) as packed:
        if hasattr(tarfile, "data_filter"):
            packed.extractall(folder, filter="data")
            return
        root = os.path.realpath(folder)
        for member in packed.getmembers():
            inside = os.path.realpath(os.path.join(root, member.name))
            if os.path.commonpath([root, inside]) != root or member.islnk() or member.issym() and os.path.isabs(member.linkname):
                raise StepFailed(f"{member.name} in Blender's archive leads outside its folder: not unpacked")
        packed.extractall(folder)


def install_blender(on_event=None, open_url=trust.open_url, source=None, addon=True):
    """Fetch Blender into tools/blender (BLENDER_BUILDS), choose it for the builds and, with `addon`,
    have it install the VRM Add-on: everything a new computer lacks, at one press. Returns setup().

    The archive goes to a .part file, is hashed as it arrives and is only unpacked when the hash is
    the pinned one. A copy already there is used as it is. The copy gets a `portable` folder, so its
    add-ons and settings stay inside it and no Blender installed on the computer is touched.
    Windows and Linux (x64): elsewhere Blender comes from blender.org or the system's packages.
    """
    source = source or blender_build()
    emit = on_event or (lambda event: None)
    if source is None:
        raise StepFailed("the app only fetches Blender on Windows and Linux (x64): install Blender 4.2 or later from blender.org/download")
    home = REPO / "tools" / "blender"
    target = home / source["folder"]
    exe = target / source.get("exe", "blender.exe")
    if not exe.is_file():
        home.mkdir(parents=True, exist_ok=True)
        if free_disk(home) < BLENDER_SPACE:
            raise StepFailed(f"Blender needs about {BLENDER_SPACE >> 30} GB free on {home.anchor} "
                             f"({free_disk(home) / 2**30:.1f} GB there now)")
        part = home / (source["folder"] + ".part")
        unzipped = home / (source["folder"] + ".unzipping")
        digest, got, said = hashlib.sha256(), 0, -1
        emit({"step": "downloading Blender", "status": "running"})
        try:
            request = urllib.request.Request(source["url"], headers={"User-Agent": "oc2vrm"})
            with open_url(request, timeout=120) as response, open(part, "wb") as out:
                while chunk := response.read(1 << 20):
                    if CANCEL.is_set():
                        raise Cancelled("stopped by the user while downloading Blender")
                    digest.update(chunk)
                    out.write(chunk)
                    got += len(chunk)
                    tenth = got * 10 // max(source["bytes"], 1)
                    if tenth != said:
                        said = tenth
                        emit({"step": "downloading Blender", "status": "progress", "reason": f"{got >> 20} of {source['bytes'] >> 20} MB"})
            if digest.hexdigest() != source["sha256"]:
                raise StepFailed(f"{source['url']} isn't the file blender.org lists (SHA-256 {digest.hexdigest()}, "
                                 f"expected {source['sha256']}): not unzipped")
            emit({"step": "downloading Blender", "status": "done"})
            emit({"step": "unzipping Blender", "status": "running"})
            shutil.rmtree(unzipped, ignore_errors=True)
            _unpack(part, unzipped)
            # the archive holds one folder named as itself; one of loose files is taken as that folder
            inner = unzipped / source["folder"]
            os.replace(inner if inner.is_dir() else unzipped, target)
            emit({"step": "unzipping Blender", "status": "done"})
        except (OSError, zipfile.BadZipFile, tarfile.TarError) as error:
            raise StepFailed(f"fetching Blender: {error}") from error
        finally:
            part.unlink(missing_ok=True)
            shutil.rmtree(unzipped, ignore_errors=True)
        if not exe.is_file():
            raise StepFailed(f"the archive was unpacked, but there is no {exe.name} in {target}")
    (target / "portable").mkdir(exist_ok=True)
    _BLENDER_VERSIONS.pop(str(exe), None)
    use_blender(exe)
    if addon and not vrm_addon_version(exe, blender_version(exe)):
        with _Lock():
            install(addon_command(exe), "installing the VRM Add-on", on_event)
    return setup()


def install_package(name, on_event=None):
    """pip-install one of PACKAGES. The user's click is the consent: it downloads from PyPI."""
    found = install(package_command(name), f"installing {name}", on_event)
    importlib.invalidate_caches()
    return found


def install_vrm_addon(given=None, on_event=None):
    """Install the VRM Add-on into the Blender the builds run on (`addon_command`): a download
    from extensions.blender.org, at the user's click. Raises StepFailed when Blender says no, or
    when the add-on still isn't there afterwards."""
    blender = find_blender(given)
    with _Lock():
        found = install(addon_command(blender), "installing the VRM Add-on", on_event)
    if not vrm_addon_version(blender, blender_version(blender)):
        raise StepFailed("Blender ran, but the VRM Add-on still isn't there. Install it by hand: open Blender, "
                         "Edit > Preferences > Get Extensions, search for \"VRM format\".\n" + found["log"])
    return found


def _installed(module):
    """Whether the Python running the app has this package. tkinter is imported to find out: a
    Linux Python without python3-tk still has the tkinter folder, and only importing it fails."""
    if module == "tkinter":
        try:
            import tkinter  # noqa: F401
        except ImportError:
            return False
        return True
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
    temporary = manifest.temporary(path)
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if backup and path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    # the page asks for the status while a build writes its state: Windows refuses the replace then
    manifest.replace(temporary, path)


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
    raise refused("VM-STATE-UNREADABLE", f"{path} is unreadable and has no usable backup; move it aside to start the build afresh", str(path))


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
    # utf-8-sig: a file written by Notepad or PowerShell starts with a mark that json refuses. ValueError:
    # one saved in another encoding doesn't read as text at all, which is no JSONDecodeError, and a single
    # such character.json took the whole Characters page down with it
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (TypeError, OSError, ValueError):
        return default


if __name__ == "__main__":
    sys.exit("use python -m app")
