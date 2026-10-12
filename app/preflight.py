"""What is known to go wrong with a build before a Blender starts, said in one sentence each.

A build takes minutes and used to be offered whatever the character looked like: a model with its
arms at its sides, roles that put the face on hair springs, an eye mode refused on the last build
and not changed since. Each was found out by building. Everything here is read from what is on the
disk already (the manifest, the survey, the piece listing, the face points, the state, the model
file's own header), by the app's Python alone: no Blender, a few milliseconds, so the page asks it
with every look at a character.

`check` returns lines of three kinds, each with a code of app/errors.py:

- `stop`: the build is known to fail or to come out wrong. The Build page asks before running past
  one (`python -m app run --strict` refuses); nothing is ever refused outright, since the user may
  know better than a measurement.
- `warn`: worth knowing before the minutes are spent; the build goes on.
- `ok`: what was looked at and found right, for an agent reading the JSON.

A line says what it is about (`text`), what to do (`fix`), the page that mends it (`page`), and where
there is one the manifest options (`options`) or the page's action (`action`, with `args`) that does it.

Every limit here was held against the characters on the developers' computer before it was let stop
anything (SETUP.md "Preflight"): it fires on the manifest or model that had the problem and says
nothing of the good ones. Where a problem can't be told before the build it is not guessed at: a
mouth marked on the throat with no eyes marked beside it reads like any other (the rig check says so
after the build, `mouth_on_head`).
"""

import hashlib
import math
import os
from datetime import datetime
from pathlib import Path

from . import engine, errors, update
from .engine import manifest

STOP, WARN, OK = "stop", "warn", "ok"
# Roles that hang a piece on springs, which the face and its eye and mouth pieces must never be on
# (build._off_the_face leaves them off and says so; said here before the minutes are spent).
SWINGING = ("hair", "swing_cloth", "appendage")
# Roles the listing's word on the face decides anything about: the ones the build step acts on.
HEAD_ROLES = (*SWINGING, "hat")
# How scripts/list_pieces.py says a piece is the face, or an eye or mouth piece (its `why`; a test
# holds these to that script's own words).
THE_FACE, AN_INSERT = "the face", "an eye or mouth insert"
# Hair named a hat: this share of the listed hair's vertices under a hat role. A tester's fox girl
# had 94% (four of her six hair pieces, "Accept all" on a listing made before her eyes were marked)
# and rigid hair; no other character here has any.
HAT_SHARE = 0.5
# survey.HEAVY_TRIANGLES and COARSE_TRIANGLES (that module imports bpy; a test holds these to it).
HEAVY_TRIANGLES, COARSE_TRIANGLES = 300_000, 3_000
# Separate surface pieces (the survey's count, after vertices at one place are joined). The characters
# here have 25 to 239 (model B the most). Model A's stand-in decimated in Blender to 20,000 triangles
# has 1,740 and to 5,000 has 1,794: a generated model is cut along every texture seam, the decimator
# doesn't close the cuts, and each patch shrinks away from its neighbours. The first built with torn
# seams, hair drooping 24 cm and no finger guides; the second fell over in the T-pose step.
FRAGMENTS = 1_000
# Face points, in distances between the two eyes' middles (ten marked faces here, SETUP.md "Preflight"):
# the mouth lies 0.39 to 0.91 below the eyes' line and within 0.02 of their middle, the eyes within
# 0.01 of level, an eye is 0.37 to 0.59 wide and the mouth 0.13 to 0.69. The limits stand well outside
# all of them; a mouth dragged 300 px down model A's picture (model D's fault, on a face whose eyes
# are marked) reads 1.22.
MOUTH_DROP = (0.15, 1.15)
MOUTH_ASIDE = 0.4
EYES_TILT = 0.3
EYES_APART = 0.6   # the eyes' middles, in eye widths: under it both are marked on one eye
SAME_SPOT = 0.02   # corners this share of the eye distance apart are one spot
OUTSIDE = 0.02     # marks this share of the picture's side outside it
# A framing differs when the head view's middle moved this share of the model's height (cli._frame
# rounds to 0.001; 0.001 of 1.6 m is 8 px of the picture).
FRAME_MOVED = 0.0015
# The longest name a build writes under a character's folder: a rig check's close-up,
# runs\20261012-155109\13_check_spots\stretch_front_posed.png (of the 8,909 files under the characters
# on the developers' computer none a build wrote is longer). With the folder's own path it has to stay
# inside what Windows gives a path (260 characters with the end mark): Blender saves a .blend the old
# way, whatever Windows' long-path switch says (5.2.2 with the switch on: saved at 258 characters,
# "Cannot open file ... .blend@ for writing" from 259; SETUP.md "The tester's computer").
LONGEST_WRITTEN = len("\\runs\\20261012-155109\\13_check_spots\\stretch_front_posed.png")
WINDOWS_PATH = 259
# OneDrive's marks on a file it keeps online only (Windows file attributes: offline, recall on open,
# recall on data access).
ONLINE_ONLY = 0x1000 | 0x40000 | 0x400000
# What a whole build writes when this character has none to go by (bytes): the runs here take 400 to 850 MB.
BUILD_WRITES = 800 << 20
# ... and what its heaviest step is taken to commit: the rigging step, 2.2 to 2.3 GB on every model of
# about 50,000 triangles (SETUP.md "One Blender for a build's steps").
STEP_COMMITS = 2300 << 20


def check(project, start=None, stop=None, only=None):
    """Look a character over before the steps `start` to `stop` (or `only` one; all of them when
    none is named) run: {"lines", "stops", "warns", "from", "to", "skipped"}.

    Never raises: a part that can't be looked at is named in `skipped` and the rest still said."""
    names = engine.NAMES
    first = names.index(only or start) if (only or start) else 0
    last = names.index(only or stop) if (only or stop) else len(names) - 1
    lines, skipped = [], []
    try:
        c = _Context(project, first, last)
    except Exception as error:  # noqa: BLE001 -- a look that fails must never stand between a user and a build
        return {"lines": [], "stops": 0, "warns": 0, "from": names[first], "to": names[last],
                "skipped": [f"everything: {type(error).__name__}: {error}"]}
    for part in (_tools, _files, _licence, _model, _face, _roles, _eyes, _again, _newer_blend, _place, _room):
        try:
            found = [line for line in part(c) if c.covers(line["at"])]
            if not any(line["kind"] != OK for line in found) and part in _FOUND_RIGHT:
                said = _FOUND_RIGHT[part](c)
                if said:
                    found.append(_line(OK, f"OK-{part.__name__.strip('_').upper()}", said, at=None))
        except Exception as error:  # noqa: BLE001
            skipped.append(f"{part.__name__.strip('_')}: {type(error).__name__}: {error}")
            continue
        lines += found
    # one line a code and place: the licence is found by the file and by the last failure alike
    seen, kept = set(), []
    for line in lines:
        key = (line["code"], line["text"])
        if key not in seen:
            seen.add(key)
            kept.append(line)
    accepted = project.state.get("accepted") or {}
    for line in kept:
        if line["kind"] == STOP and accepted.get(line["code"]) == line["text"]:
            # the user was asked this of this character and built on: said still, not asked again
            line["kind"], line["accepted"] = WARN, True
    order = {STOP: 0, WARN: 1, OK: 2}
    kept.sort(key=lambda line: order[line["kind"]])
    return {"lines": kept, "stops": sum(line["kind"] == STOP for line in kept),
            "warns": sum(line["kind"] == WARN for line in kept),
            "from": names[first], "to": names[last], "skipped": skipped}


def accept(project, codes):
    """Remember that the user was asked the stops with these codes, as they read now, and built on:
    they are warnings of this character from now on, until what they say changes."""
    now = {line["code"]: line["text"] for line in check(project)["lines"] if line["kind"] == STOP and line["code"] in set(codes)}
    if now:
        project.state["accepted"] = {**(project.state.get("accepted") or {}), **now}
        project._save_state()
    return {"accepted": sorted(now)}


class _Context:
    """What the looks read, each thing once."""

    def __init__(self, project, first, last):
        self.project, self.first, self.last = project, first, last
        self.data, self.state = project.data, project.state
        self.steps = project.state.get("steps", {})
        self.options = project.options()
        self.survey = None if project.improving else project.survey()
        self.plan = {name: runs for name, runs, _ in project.plan()}
        self.source = manifest.path_of(self.data, "source")
        self.face_path = manifest.path_of(self.data, "face")
        self.points = engine._read_json(self.face_path, None) if self.face_path else None
        self.listing = engine._read_json(project.out / "pieces.json", None)

    def covers(self, step):
        """Whether the run goes through `step` and the step is one this character runs. None: a
        finding of the run as a whole (no Blender, no room on the drive), whatever its steps."""
        if step is None:
            return True
        index = engine.NAMES.index(step)
        return self.first <= index <= self.last and self.plan.get(step, True)

    @property
    def marks_eyes(self):
        return bool(isinstance(self.points, dict) and (self.points.get("right_eye") or self.points.get("left_eye")))

    @property
    def built(self):
        """Whether the character stands built to its end from the survey it has now: its user has seen
        what the build made of the model."""
        check, survey = self.steps.get("vrm_check", {}), self.steps.get("survey", {})
        return check.get("status") == "done" and (check.get("finished") or "") > (survey.get("finished") or "")

    def built_after(self, path):
        """Whether the character stands built to its end since `path` (the face points) was written."""
        check = self.steps.get("vrm_check", {})
        return check.get("status") == "done" and bool(check.get("finished")) and not _after(path, check["finished"])


def _line(kind, code, text=None, at="build", fix=None, **more):
    """One finding. `text` is the sentence for this character; without one, the code's own."""
    known = errors.describe(code) or {"what": "", "fix": "", "options": None}
    line = {"kind": kind, "code": code, "text": text or known["what"], "fix": fix if fix is not None else known["fix"],
            "page": more.pop("page", known.get("page")), "options": more.pop("options", known["options"]),
            "action": more.pop("action", known.get("action")), "at": at}
    return {**line, **more}


# ----- the tools


def _tools(c):
    at = None  # of the run as a whole
    try:
        blender = c.project.blender
    except FileNotFoundError:
        yield _line(STOP, "VM-NO-BLENDER", "Blender 4.2 or later wasn't found on this computer", at=at)
        return
    version = engine.blender_version(blender)
    if version is None:
        yield _line(STOP, "VM-NO-BLENDER", f"the Blender chosen doesn't start: {blender}", at=at,
                    fix="choose another on the Setup page, or let the app fetch one")
    elif tuple(map(int, version.split("."))) < engine.MIN_BLENDER:
        yield _line(STOP, "VM-BLENDER-OLD", f"Blender {version} is older than 4.2, which the build needs", at=at,
                    fix="choose a newer one on the Setup page, or let the app fetch Blender 5.2")
    elif not engine.vrm_addon_version(blender, version):
        yield _line(STOP, "VM-NO-VRM-ADDON", f"the VRM Add-on isn't installed in Blender {version}, and nothing can be exported without it",
                    at=at, fix="install it on the Setup page: one press")


def _tools_right(c):
    blender = c.project.blender
    version = engine.blender_version(blender)
    return f"Blender {version} with the VRM Add-on {engine.vrm_addon_version(blender, version)}"


# ----- the files the character names


def _files(c):
    from . import cli
    at = None  # of the run as a whole
    source = c.source
    if source is None or not source.is_file():
        yield _line(STOP, "VM-NO-MODEL", f"the model file isn't where this character says it is: {source}", at=at)
        return
    if source.suffix.lower() not in cli.MODEL_TYPES:
        yield _line(STOP, "VM-UNSUPPORTED-FILE", f"{source.name} is a {source.suffix or 'file with no type'}, which can't be read", at=at)
    for key, what, kind in (("rig", "the rig file", STOP), ("reference", "the model before its face patch", STOP),
                            ("face", "the face points file", STOP), ("paint", "the folder of texture paint", WARN)):
        path = manifest.path_of(c.data, key)
        if path is not None and not path.exists():
            yield _line(kind, "VM-FILE-GONE", f"{what} isn't where this character says it is: {path}", page="model",
                        **({"fix": "paint it again on the Paint page: the build uses the texture as it came"} if kind == WARN else {}))
    rig = manifest.path_of(c.data, "rig")
    if rig is not None and rig.suffix.lower() == ".glb" and rig.is_file() and _broken_binds(rig):
        yield _line(STOP, "VM-RIG-GLB", f"{rig.name} is a rigged GLB whose bind matrices no skeleton can have: every bone would import broken")
    dimmed = _colour_factor(source) if source.suffix.lower() == ".glb" else None
    if dimmed:
        # Tripo's newer exports: the picture multiplied by a grey factor, a Mix node once imported. It stopped
        # the texture steps once (VM-TRIPO-MIX); every step reads through it now (models E and E2 here carry
        # it and build, face patch and all), so it is only said to have been seen
        yield _line(OK, "OK-COLOUR-FACTOR", f"the model's colour is multiplied by {dimmed:g} in its file (newer Tripo exports): read through", at=None)


def _colour_factor(path):
    """The grey factor a GLB's textured material multiplies its picture by, when it is not 1; else None."""
    for material in (engine._glb_json(path) or {}).get("materials", []):
        colour = material.get("pbrMetallicRoughness") or {}
        factor = colour.get("baseColorFactor")
        if "baseColorTexture" in colour and isinstance(factor, list) and len(factor) >= 3 and any(abs(v - 1.0) > 1e-3 for v in factor[:3]):
            return round(float(factor[0]), 3)
    return None


def _broken_binds(path):
    gltf = engine._glb_json(path) or {}
    return any(not math.isfinite(v) or abs(v) > engine.BIND_LIMIT
               for skin in gltf.get("skins", []) for v in engine._bind_matrices(path, gltf, skin))


def _licence(c):
    if not manifest.improves(c.data) or c.source is None or not c.source.is_file():
        return
    said = engine.licence(c.source)
    if said and said["confirm"] and not c.options["licence_read"]:
        # the user's to say, never set for them: no option rides with this line
        yield _line(STOP, "VM-LICENCE", f"this avatar's licence has to be read before the app changes it: {said['why']}", options=None)


# ----- the model, as the survey measured it


def _model(c):
    s = c.survey
    if s is None:
        return
    at = "build"
    if s.get("texture") is None:
        yield _line(STOP, "VM-NO-TEXTURE", at=at)
    pose = s.get("pose") or {}
    if pose.get("kind") == "arms down":
        yield _line(STOP, "VM-ARMS-DOWN", f"the arms hang at the sides ({pose.get('arm_drop_degrees', 0):.0f}° below level), "
                    "where sleeves are fused with the body: shoulders and sleeves come out torn or stuck", at=at)
    triangles = s.get("triangles") or 0
    if triangles > HEAVY_TRIANGLES:
        yield _line(STOP, "VM-HEAVY-MODEL", f"{triangles:,} triangles is far more than an avatar needs: every step crawls, "
                    "Blender may run out of memory and tracking apps struggle", at=at)
    elif 0 < triangles < COARSE_TRIANGLES:
        yield _line(WARN, "VM-COARSE-MODEL", f"only {triangles:,} triangles: the face is too coarse to bend, so only the mouth is moved", at=at)
    pieces = s.get("pieces") or 0
    if isinstance(pieces, int) and pieces > FRAGMENTS:
        yield _line(STOP, "VM-FRAGMENTS", f"the surface is in {pieces:,} separate pieces (a model decimated across its texture seams, or exported "
                    "in fragments): it tears apart between them, and hair, clothing and fingers can't be read from it", at=at)
    turned = s.get("orientation") or {}
    if turned and not (turned.get("up_sure") and turned.get("facing_sure")):
        which = "which way up it stands" if not turned.get("up_sure") else "which way it faces"
        # a doubt, not a fault: asked until the character has been built from this survey and seen
        yield _line(WARN if c.built else STOP, "VM-ORIENT-UNSURE",
                    f"the app couldn't tell {which}: if the picture on the Model page shows it upright and facing you, it is right", at=at,
                    fix="if it lies down or shows its back, turn it upright facing you in Blender and export it again"
                    + ("" if not turned.get("up_sure") else "; or say it faces as its file has it (Options: facing)"))
    if (s.get("parts") or {}).get("read") is False:
        yield _line(WARN, "VM-PARTS-NOT-READ", at=at)
    # (not said: joints "placed from typical proportions". The survey notes it of the shoulders of half
    # the characters here, model A and the plain boy among them, and their builds are right)


def _model_right(c):
    s = c.survey
    if s is None:
        return None
    return f"the model: {(s.get('pose') or {}).get('kind', 'pose not read')}, {s.get('triangles', 0):,} triangles, textured"


# ----- the face points


def _face(c):
    if c.project.improving:
        return
    if not c.data.get("face"):
        yield _line(WARN, "VM-NO-FACE-POINTS", at="build")
        return
    points = c.points
    wrong = _unreadable(points)
    if wrong:
        yield _line(STOP, "VM-FACE-FILE", f"the face points can't be read: {wrong}", at="build")
        return
    # asked of the face steps: `build` reads the points too, to shape the head round them
    odd = _implausible(points)
    if odd:
        yield _line(STOP, "VM-FACE-MARKS", odd, at="build")
    moved = _reframed(c, points)
    if moved:
        yield _line(STOP, "VM-FACE-REFRAMED", moved, at="build")


def _unreadable(points):
    """What stops face_points.load reading these, in words; None when it would read them."""
    if not isinstance(points, dict):
        return "the file is missing or isn't JSON"
    eyes = bool(points.get("right_eye") or points.get("left_eye"))
    for key, names in (("right_eye", ("inner", "outer", "top", "bottom")), ("left_eye", ("inner", "outer", "top", "bottom")),
                       ("mouth", ("left", "right", "top", "bottom"))):
        if key != "mouth" and not eyes:
            continue
        part = points.get(key) if isinstance(points.get(key), dict) else {}
        missing = [n for n in names if not _is_xy(part.get(n))]
        if missing:
            return f"{key.replace('_', ' ')} has no {', '.join(missing)}"
    size = points.get("size")
    if not (isinstance(size, list) and len(size) == 2 and all(isinstance(v, (int, float)) and v > 0 for v in size)):
        return "no picture size"
    return None


def _is_xy(value):
    return (isinstance(value, list) and len(value) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value))


def _implausible(points):
    """The one sentence that says why these marks can't be a face's, or None. Plausible is wide:
    a chibi, a pirate with a beard, a slime whose eyes are stored under each other's names."""
    width, height = points["size"]
    marked = [xy for key in ("right_eye", "left_eye", "mouth") for xy in (points.get(key) or {}).values() if _is_xy(xy)]
    outside = sum(1 for x, y in marked if not (-OUTSIDE * width <= x <= (1 + OUTSIDE) * width and -OUTSIDE * height <= y <= (1 + OUTSIDE) * height))
    if outside:
        return f"{outside} of the {len(marked)} face points lie outside the picture of the face"
    middle = lambda part: (sum(p[0] for p in part.values()) / len(part), sum(p[1] for p in part.values()) / len(part))  # noqa: E731
    mouth = points["mouth"]
    mouth_at, mouth_wide = middle(mouth), math.dist(mouth["left"], mouth["right"])
    if not (points.get("right_eye") and points.get("left_eye")):
        # the eyes are covered: there is nothing to hold the mouth against (see the module's words)
        return "the mouth's two corners are marked on the same spot" if mouth_wide < 1.0 else None
    right, left = points["right_eye"], points["left_eye"]
    right_at, left_at = middle(right), middle(left)
    apart = math.dist(right_at, left_at)
    wide = (math.dist(right["inner"], right["outer"]) + math.dist(left["inner"], left["outer"])) / 2
    if min(math.dist(right["inner"], right["outer"]), math.dist(left["inner"], left["outer"])) < SAME_SPOT * max(apart, 1.0):
        return "an eye's two corners are marked on the same spot"
    if apart < EYES_APART * wide:
        return "both eyes are marked on the same eye"
    if abs(left_at[1] - right_at[1]) > EYES_TILT * apart:
        return "one eye is marked far above the other"
    eyes_at = ((right_at[0] + left_at[0]) / 2, (right_at[1] + left_at[1]) / 2)
    aside, drop = abs(mouth_at[0] - eyes_at[0]) / apart, (mouth_at[1] - eyes_at[1]) / apart
    if mouth_wide < SAME_SPOT * apart:
        return "the mouth's two corners are marked on the same spot"
    if aside > MOUTH_ASIDE:
        return "the mouth is marked off to one side of the face, not under the middle of the eyes"
    if drop < MOUTH_DROP[0]:
        return "the mouth is marked at the height of the eyes, or above them"
    if drop > MOUTH_DROP[1]:
        return (f"the mouth is marked {drop:.1f} times the distance between the eyes below them: on the chin or the throat, "
                "not the lips (it would open there)")
    return None


def _reframed(c, points):
    """Why the marks are of another picture than the face patch the model wears now, or None. Told
    by the frame kept with the marks when they were saved (cli.save_face): marks from before that
    was kept say nothing, and are taken as they are."""
    frame = points.get("frame")
    if not isinstance(frame, dict) or not isinstance(frame.get("head_at"), (int, float)):
        return None
    now = patch_frame(c.project)
    if now is None or abs(now["head_at"] - frame["head_at"]) <= FRAME_MOVED:
        return None
    return ("the face patch was made again framed otherwise than the picture the eyes and mouth were marked on "
            f"(its middle at {now['head_at']:.3f} of the model's height, {frame['head_at']:.3f} when marked): the marks are off the face")


def patch_frame(project):
    """{"head_at", "body"} of the face patch the model wears: kept beside the patched model, else
    in the character's own face folder. None without a patch that says."""
    source = manifest.path_of(project.data, "source")
    for folder in ([source.parent] if source else []) + [project.out / "face"]:
        said = engine._read_json(folder / "head_at.json", None)
        if isinstance(said, dict) and isinstance(said.get("head_at"), (int, float)):
            return {"head_at": said["head_at"], "body": said.get("body", 1.0)}
    return None


def _face_right(c):
    if c.project.improving or not c.data.get("face"):
        return None
    return "the face points: " + ("both eyes and the mouth" if c.marks_eyes else "the mouth alone (the eyes are covered)")


# ----- the roles given to pieces


def _roles(c):
    if c.project.improving:
        return
    pieces = [p for p in c.data.get("pieces", []) if not p.get("region")]
    listing = c.listing if isinstance(c.listing, dict) and isinstance(c.listing.get("pieces"), list) else None
    rows = listing["pieces"] if listing else []
    row_of = {p["name"]: next((r for r in rows if _same_piece(p, r)), None) for p in pieces}
    if listing and _listed_of(c, listing):
        # the build refuses a piece whose point or vertex count isn't the model's (roles.find): here, one
        # no listed piece of its size holds the point of (all 104 named pieces of the characters on the
        # developers' computer have such a row)
        gone = [p["name"] for p in pieces if not any(_holds(r, p) for r in rows)]
        if gone:
            yield _line(STOP, "VM-PIECE-GONE", f"{_names(gone)} {'is' if len(gone) == 1 else 'are'} named on the Parts page, and this "
                        "model has no piece of that size there: the build would stop at it", args={"names": gone})
    if not c.marks_eyes:
        return
    if listing and listing.get("eyes_marked"):
        face = [p["name"] for p in pieces if p["role"] in SWINGING and _is_face(row_of[p["name"]])]
        if face:
            what = "the face the eyes are marked on" if any((row_of[n].get("why") or "") == THE_FACE for n in face) else "an eye or mouth piece"
            yield _line(STOP, "VM-ROLE-ON-FACE", f"{_names(face)} {'is' if len(face) == 1 else 'are'} {what}, and "
                        f"{'is' if len(face) == 1 else 'are'} given a swinging role: the face would hang on springs", args={"names": face})
        hair = [r for r in rows if r.get("suggested") == "hair"]
        hats = [p["name"] for p in pieces if p["role"] == "hat" and row_of[p["name"]] in hair and row_of[p["name"]] is not None]
        listed = sum(r["count"] for r in hair)
        named = sum(row_of[n]["count"] for n in hats)
        if hats and named >= HAT_SHARE * listed and c.data["options"].get("rigid_head") is not True:
            yield _line(STOP, "VM-HAIR-AS-HAT", f"{len(hats)} piece{'s' if len(hats) > 1 else ''} that read as hair ({named:,} of its "
                        f"{listed:,} vertices) {'are' if len(hats) > 1 else 'is'} named a hat: the hair would be stuck to the head and not swing",
                        args={"names": hats})
    elif any(p["role"] in HEAD_ROLES for p in pieces) and (c.survey is None or c.survey.get("face_patch")):
        # Once the character stands built to its end with these marks its user has seen the face and
        # the hair move, and the build has left a swinging role off the face by itself
        # (build._off_the_face): then only hats on a head no hair swings from are still worth the look
        # (a tester's fox girl: 13 hats, no hair chain)
        hung = (c.steps.get("build", {}).get("report") or {}).get("hair_chains")
        rigid = any(p["role"] == "hat" for p in pieces) and hung == 0
        if rigid or not (c.face_path and c.built_after(c.face_path)):
            yield _line(WARN, "VM-LISTING-STALE",
                        "the parts were listed before the eyes were marked: the roles given can't be checked against the face"
                        if listing else "the parts aren't listed on this computer: the roles given can't be checked against the face")


def _same_piece(piece, row):
    """Whether a manifest piece is this row of the listing (core.js samePiece): its size and its point."""
    point, at = piece.get("point"), row.get("point")
    return (piece.get("count") == row.get("count") and isinstance(point, list) and isinstance(at, list)
            and len(point) == 3 and len(at) == 3 and math.dist(point, at) < 1e-4)


def _holds(row, piece, reach=0.002):
    """Whether a listed piece is of the named piece's size and has its point within its box (m)."""
    if row.get("count") != piece.get("count"):
        return False
    low, high, point = row.get("min"), row.get("max"), piece.get("point")
    if not (isinstance(low, list) and isinstance(high, list) and isinstance(point, list) and len(low) == len(high) == len(point) == 3):
        return True  # a listing that doesn't say where its pieces are: the size is all there is to go by
    return all(low[i] - reach <= point[i] <= high[i] + reach for i in range(3))


def _is_face(row):
    why = (row or {}).get("why") or ""
    return why == THE_FACE or why.startswith(AN_INSERT)


def _listed_of(c, listing):
    """Whether the listing is of the model the character names now (the reference, else the source)
    and was made after that file was last written: only then is a piece it lacks one the model lacks."""
    model = manifest.path_of(c.data, "reference") or c.source
    listed = c.project.out / "pieces.json"
    try:
        same = model is not None and listing.get("model") and os.path.samefile(listing["model"], model)
        return bool(same) and listed.stat().st_mtime >= model.stat().st_mtime
    except OSError:
        return False


def _names(names, most=3):
    shown = ", ".join(f"\"{n}\"" for n in names[:most])
    return shown + (f" and {len(names) - most} more" if len(names) > most else "")


def _roles_right(c):
    pieces = c.data.get("pieces", [])
    if c.project.improving or not pieces:
        return None
    return f"the {len(pieces)} role{'s' if len(pieces) > 1 else ''} given to pieces"


# ----- the eyes


def _eyes(c):
    if c.project.improving:
        return
    chosen = c.data["options"].get("eyes")
    if chosen in ("iris", "modelled", "slide") and not (c.data.get("face") and c.marks_eyes):
        yield _line(WARN, "VM-EYES-UNMARKED", at="build")
    fell = c.state.get("eyes_fallback")
    ran = c.steps.get("eyes", {}).get("finished") or (c.steps.get("eyes", {}).get("last_failure") or {}).get("finished")
    if fell == "slide" and chosen is None and ran and c.face_path and c.face_path.exists() and _after(c.face_path, ran):
        yield _line(WARN, "VM-EYES-FALLBACK", "the eyes slide as a painted patch because an earlier build's moving iris was refused, "
                    "and the face points have been saved again since", at="eyes")


def _after(path, when):
    """Whether a file was written after an ISO time the state kept."""
    try:
        return path.stat().st_mtime > datetime.fromisoformat(when).timestamp()
    except (OSError, ValueError):
        return False


# ----- what failed the last time and would again


def _again(c):
    """A step that failed in a way its inputs decide (errors `again`) and whose inputs stand as they
    did: the same manifest to the byte, the same model, rig and face points, the same app. A tester
    pressed Build on a refused eye mode until the report was sent."""
    for name in engine.NAMES[c.first:c.last + 1]:
        entry = c.steps.get(name, {})
        failure = entry.get("last_failure") or (entry if entry.get("status") == "failed" else None)
        known = (failure or {}).get("known") or {}
        if not known.get("code") or not errors.repeats(known["code"]) or not _unchanged(c, failure):
            continue
        if known["code"] == "VM-EYES-REFUSED" and _eyes_have_another_way(c):
            continue  # eyes left on automatic: the build tries the other ways by itself
        # said as the page says the failure itself (errors.failure: a state from before keeps less)
        drawn = failure.get("failure") or errors.failure(name, known, failure.get("reason", ""))
        yield _line(STOP, known["code"], f"\"{name.replace('_', ' ')}\" failed this way on the last build, and nothing it reads has changed "
                    f"since: {drawn['what']}", at=name, fix=drawn["fix"], options=drawn["options"],
                    **({"page": drawn["page"]} if drawn.get("page") else {}))


def _eyes_have_another_way(c):
    """engine.Project._eyes_fall_back's own rule: an automatic iris, disc or remembered patch that
    is refused leaves another way to try; painted eyes that slide from the start leave none."""
    now = c.options["eyes"]
    if c.data["options"].get("eyes") is not None or now not in ("iris", "modelled", "slide"):
        return False
    return not (now == "slide" and c.state.get("eyes_fallback") != "slide")


def _unchanged(c, failure):
    """Whether what a failed step read stands as it did when it failed. Told by the run's own record
    (run.json beside the step's log): a failure from before runs kept one says nothing."""
    log, when = failure.get("log"), failure.get("finished")
    if not log or not when:
        return False
    made = engine._read_json(Path(log).parent / "run.json", None)
    if not isinstance(made, dict) or not made.get("manifest_sha256"):
        return False
    try:
        now = hashlib.sha256(c.project.manifest_path.read_bytes()).hexdigest()[:16]
    except OSError:
        return False
    if now != made["manifest_sha256"] or _commit() != made.get("commit"):
        return False
    read = [manifest.path_of(c.data, key) for key in ("source", "reference", "rig", "face")]
    return not any(path is not None and _after(path, when) for path in read)


def _commit():
    try:
        return update.current().get("commit")
    except Exception:  # noqa: BLE001 -- as engine.provenance: a record, never a reason to stop
        return None


# ----- files of a newer Blender


def _newer_blend(c):
    """A run from the middle of the chain opens the .blend the step before left: saved by a newer
    Blender than the one chosen now, it doesn't open (a file saved by 5.2 can't be opened by 4.2)."""
    name = engine.NAMES[c.first]
    if name in ("survey", "build", "vrm_check"):
        return
    blend = c.project.latest(before=name)
    if blend is None:
        return
    saved = (engine._read_json(blend.parent / "run.json", None) or {}).get("blender_version")
    try:
        now = engine.blender_version(c.project.blender)
    except FileNotFoundError:
        return
    series = lambda version: tuple(map(int, version.split(".")[:2]))  # noqa: E731
    if saved and now and series(saved) > series(now):
        yield _line(STOP, "VM-BLEND-NEWER", f"the files of this build were saved by Blender {saved}, and Blender {now}, chosen now, can't open them",
                    at=name)


# ----- where the build writes, and what draws its result


def _place(c):
    out = Path(c.project.out).resolve()
    if os.name == "nt" and len(str(out)) + LONGEST_WRITTEN > WINDOWS_PATH:
        yield _line(STOP, "VM-PATH-LONG", f"the character's folder is {len(str(out))} characters deep on the disk ({out}): files a build "
                    f"writes there would pass the {WINDOWS_PATH + 1} characters Windows gives a path, and Blender can't save them", at=None)
    kept = _onedrive(out)
    if kept:
        yield _line(WARN, "VM-ONEDRIVE", f"the builds are written inside OneDrive ({kept}), which uploads every file of a build, "
                    "hundreds of megabytes, and holds each while it does", at=None)
    if c.source is not None and c.source.is_file() and _online_only(c.source):
        yield _line(WARN, "VM-MODEL-ONLINE", f"{c.source.name} is kept online only by OneDrive: it is downloaded when the build reads it, "
                    "and can't be read without a connection", at="build")
    from . import sheet
    try:
        sheet.find_browser()
    except sheet.NoBrowser:
        yield _line(WARN, "VM-NO-BROWSER", at="vrm_check")


def _onedrive(path):
    """The OneDrive folder `path` lies in (the user's own, a company's), or None."""
    for name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        root = os.environ.get(name)
        if not root:
            continue
        try:
            if os.path.commonpath([os.path.normcase(os.path.abspath(root)), os.path.normcase(str(path))]) == os.path.normcase(os.path.abspath(root)):
                return root
        except ValueError:  # another drive
            continue
    return None


def _online_only(path):
    try:
        return bool(getattr(path.stat(), "st_file_attributes", 0) & ONLINE_ONLY)
    except OSError:
        return False


# ----- room on the drive and in memory


def _room(c):
    at = None  # of the run as a whole
    gb = lambda size: f"{size / 2**30:.1f} GB"  # noqa: E731
    out = c.project.out
    drive = Path(out).resolve().anchor
    free, writes = engine.free_disk(out), _writes(c)
    if free < engine.DISK_NEEDED:
        yield _line(STOP, "VM-DISK-FULL", f"only {free / 2**20:.0f} MB are free on {drive}: no step starts with under 1 GB free where it writes", at=at)
    elif free < engine.DISK_NEEDED + writes:
        yield _line(STOP, "VM-DISK-LOW", f"{drive} has {gb(free)} free and this build writes about {writes / 2**20:.0f} MB: "
                    "it would be stopped part way, under 1 GB", at=at)
    elif free < 3 * engine.DISK_NEEDED:
        yield _line(WARN, "VM-DISK-LOW", f"{drive} has {gb(free)} free and a build writes several hundred MB", at=at)
    memory, commits = engine.free_memory(), _commits(c)
    if memory is not None and memory < commits:
        yield _line(STOP, "VM-MEMORY-LOW", f"{gb(memory)} of memory is to be had and this build's heaviest step "
                    f"{'took' if _measured(c) else 'is taken to need'} {gb(commits)}: Blender would run out part way", at=at)
    elif memory is not None and memory < engine.MEMORY_LOW:
        yield _line(WARN, "VM-MEMORY-LOW", f"only {gb(memory)} of memory is free: Blender may run out mid-build", at=at)
    system = os.environ.get("SystemDrive")
    if os.name == "nt" and system and Path(system + "\\").resolve().anchor != drive:
        left = engine.free_disk(system + "\\")
        if left < engine.SYSTEM_DISK_LOW:
            yield _line(WARN, "VM-SYSTEM-DISK-LOW", f"{system} has {gb(left)} free: Windows can't grow its page file there, "
                        "so Blender may run out of memory", at=at)


def _steps_run(c):
    return [name for name in engine.NAMES[c.first:c.last + 1] if c.plan.get(name, True)]


def _writes(c):
    """Bytes the steps to run are taken to write: what each one's files took in this character's
    last build, else its share of BUILD_WRITES."""
    kinds = {name: kind for name, _, kind in engine.STEPS}
    total = 0
    for name in _steps_run(c):
        entry = c.steps.get(name, {})
        took = sum(engine._size(entry[key]) for key in ("blend", "vrm") if entry.get(key))
        total += took or (BUILD_WRITES // 12 if kinds[name] in ("blend", "vrm") else 0)
    return total


def _measured(c):
    return any(c.steps.get(name, {}).get("peak_mb") for name in _steps_run(c))


def _commits(c):
    """Bytes the heaviest of the steps to run is taken to commit: what it took in this character's
    last build (engine `peak_mb`), else STEP_COMMITS for a run that rigs and a third of it for one
    that doesn't."""
    took = max((c.steps.get(name, {}).get("peak_mb") or 0 for name in _steps_run(c)), default=0)
    if took:
        return took << 20
    return STEP_COMMITS if "build" in _steps_run(c) else STEP_COMMITS // 3


def _room_right(c):
    memory = engine.free_memory()
    said = f"{engine.free_disk(c.project.out) / 2**30:.1f} GB free on {Path(c.project.out).resolve().anchor}"
    return said + (f", {memory / 2**30:.1f} GB of memory to be had" if memory is not None else "")


_FOUND_RIGHT = {_tools: _tools_right, _model: _model_right, _face: _face_right, _roles: _roles_right, _room: _room_right}
