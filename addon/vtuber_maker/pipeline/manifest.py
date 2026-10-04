"""The character manifest: every choice made about one avatar, in one JSON file.

A build reads nothing else character-specific, so a new character is rebuilt from its
manifest instead of from code written for it (model A's and model B's clothing repairs were first
hard-coded profiles in scripts/repair_clothing.py). Coordinates are in the normalized modelled
A-pose that `normalize.prepare` puts the source in: 1.6 m tall, on Z=0, facing -Y, character
left +X. Never a baked T-pose.

A piece is named by a point on its surface and its vertex count in the `reference` model
(else the `source`): the point picks the connected piece, the count refuses a different or
re-exported model. A source whose texture was edited after generating can gain a vertex or
two along new UV seams (model A's face patch did), so its unmodified model is the reference.

Plain Python, no bpy: the desktop app and agents read and write manifests outside Blender.
"""

import json
import os
import struct
from pathlib import Path

VERSION = 1
# {old version: function returning the manifest one version on}. Applied on load, so a character
# made by an older app keeps opening after the schema changes. Each VERSION bump adds one, with a test.
MIGRATIONS = {}

# What a piece does, and the stage that acts on it: "build" while rigging (make_vrm.py),
# "pieces" on the finished A-pose rig before the T-pose bake (scripts/apply_pieces.py).
ROLES = {
    "body": "build",            # ordinary skinning; listing it keeps it off hair and cloth chains
    "hair": "build",            # spring strands
    "hat": "build",             # wholly on the head; "tip": true also swings a pointed tip
    "swing_cloth": "build",     # cape, coat tail, loose skirt: cloth spring chains
    "appendage": "build",       # wing, tail, big ear: on the core bone it grows from ("bone" to choose), springs
    "fitted_skirt": "pieces",   # stays on the torso, cut free of the hands resting on it
    "shoulder_cloth": "pieces", # mantle: blended onto the upper arms, collar and hem stay
    "follow_bone": "pieces",    # wholly on one bone ("bone"): hanging sleeve, bracelet, prop
    "detail": "pieces",         # rides another piece ("anchor"): cuff trim, button
}

# The keys of `joints`, as landmarks.detect_body names them.
JOINTS = ("hips", "spine", "chest", "upper_chest", "neck", "head",
          *(f"{side}_{key}" for side in ("left", "right")
            for key in ("upper_arm", "lower_arm", "hand", "upper_leg", "lower_leg", "foot")))

# Every option with its default. None means "decide from the survey", as make_avatar.py does.
OPTIONS = {
    "height": 1.6,
    "rigid_head": None,       # hats, helmets, solid heads: the whole head moves as one
    "cloth": None,            # cloth chains for pieces the survey thinks hang free
    "hat_tip": False,         # with rigid_head: swing the highest head piece's tip
    "mouth_colour": None,     # "skin" or "#rrggbb" inside the mouth
    "free_mouth": None,       # paint the drawn mouth out and move it on its own
    "mouth_only": None,       # face tracking moves only the mouth (coarse or solid faces)
    "ahoge": True,            # spring the strand standing on top of the head, when there is one
    "face_tracking": True,    # ARKit shapes; needs `face`
    "loose_pieces": True,     # scripts/fix_loose_pieces.py
    "keep_seams": False,      # with it: don't join touching surfaces (model A's verified build kept them)
    "fingers": "guides",      # "guides" (surface guides, the model A/model B route), "auto" or "none"
    "skinning": "heat",       # "heat" (bone heat and the arm/body rules), "template" (template_skin.py) or "voxel" (voxel_skin.py, same rules)
    "tpose": "spec",          # "spec" (spec hands, straight fingers), "plain" or "none"
    "repair_shoulder_seams": True,
    "eyes": None,             # eye tracking: "iris" (moving iris), "slide" (painted patch), "none"; None: from the survey
    "slide": [],              # with "slide": [horizontal, vertical] travel in mm
    "mouth_fix": None,        # {"scale": 1.3, "free": true} runs scripts/fix_mouth.py
    "chest": False,           # chest bounce: two spring bones on the chest (pipeline/chest.py)
    "chest_amount": 1.0,      # how much it bounces, 0.25 (barely) to 2 (a lot)
}
EYES = (None, "none", "iris", "slide")
CHEST_AMOUNT = (0.25, 2.0)  # chest.AMOUNT_RANGE (this module stays plain Python)
FINGERS = ("guides", "auto", "none")
TPOSE = ("spec", "plain", "none")
SKINNING = ("heat", "template", "voxel")
# Paths, relative to the manifest's folder when not absolute. `paint`: the folder of texture paint
# (one transparent overlay per picture of the source, app/paint.py), laid over the source for a build.
PATHS = ("source", "reference", "face", "out_dir", "rig", "paint")
# The glTF scene extra through which a GLB names the rigged FBX it was made from: the texture
# steps (matte, tint, face patch) write Tripo's FBX out as the mesh alone.
RIG_KEY = "vtuber_rig"


def new(source, name, author="Unknown"):
    """A manifest for `source` with every option at its default and no pieces."""
    return {"version": VERSION, "name": name, "author": author, "source": str(source),
            "reference": None, "face": None, "out_dir": None, "joints": {}, "options": dict(OPTIONS),
            "pieces": []}


def carries_rig(path):
    """Whether a model file is an FBX with a VRoid-style skeleton, as Tripo's "VRM 1.0 humanoid" rig
    exports: then it is its own `rig` (adopt.py). Read as bytes: binary and ASCII FBX both keep
    bone names as plain text, and the desktop app asks this without Blender."""
    return rig_kind(path) == TRIPO_RIG


# The hip bone's name in the skeletons adopt.py reads, by the tool that names it so.
TRIPO_RIG = "Tripo / VRoid"
RIG_MARKERS = {b"J_Bip_C_Hips": TRIPO_RIG, b"mixamorig:Hips": "Mixamo", b"CC_Base_Hip": "Character Creator / AccuRig"}


def rig_kind(path):
    """Which tool's skeleton an FBX carries (a RIG_MARKERS value), or None.

    Only Tripo's is taken as the rig by itself (`carries_rig`): it was measured against our own
    rigging on model C. The others are read by `adopt.py` when the user names the file as the rig,
    and the app says so when it sees one.
    """
    path = Path(path)
    if path.suffix.lower() != ".fbx" or not path.is_file():
        return None
    keep, tail = max(map(len, RIG_MARKERS)), b""
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            for marker, kind in RIG_MARKERS.items():
                if marker in tail + chunk:
                    return kind
            tail = chunk[-keep:]
    return None


def rig_of(path):
    """The rig file that goes with a model file, or None.

    Tripo's rigged FBX is its own (carries_rig). A GLB the texture steps made from one names it
    (RIG_KEY), found where it was or, moved along with the GLB, beside it; a named file that no
    longer carries a rig counts for nothing. Plain bytes, no Blender.
    """
    path = Path(path)
    if carries_rig(path):
        return path
    named = _scene_extras(path).get(RIG_KEY)
    if not isinstance(named, str) or not named:
        return None
    for candidate in (Path(named), path.parent / Path(named).name):
        if carries_rig(candidate):
            return candidate
    return None


def _scene_extras(path):
    """The extras on a .glb or .gltf file's scene; {} for any other file, or one that doesn't read."""
    try:
        if path.suffix.lower() == ".gltf":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".glb":
            with path.open("rb") as f:
                head = f.read(20)
                if len(head) < 20 or head[:4] != b"glTF" or head[16:20] != b"JSON":
                    return {}
                data = json.loads(f.read(struct.unpack("<I", head[12:16])[0]))
        else:
            return {}
        extras = (data.get("scenes") or [{}])[data.get("scene", 0)].get("extras")
        return extras if isinstance(extras, dict) else {}
    except (OSError, ValueError, IndexError, AttributeError, TypeError):
        return {}


def load(path):
    """Read and check a manifest; paths stay as written, resolve them with `path_of`."""
    path = Path(path)
    data = migrate(json.loads(path.read_text(encoding="utf-8")))
    data["options"] = {**OPTIONS, **data.get("options", {})}
    data.setdefault("joints", {})
    data.setdefault("pieces", [])
    for key in PATHS:
        data.setdefault(key, None)
    validate(data)
    data["_path"] = str(path.resolve())
    return data


def migrate(data):
    """Carry an older manifest forward to VERSION; refuse one from a newer app rather than misread it."""
    version = data.get("version")
    if isinstance(version, int) and version > VERSION:
        raise ValueError(f"manifest: made by a newer version of the app (version {version}, this one reads {VERSION})")
    while isinstance(version, int) and version < VERSION and version in MIGRATIONS:
        data = MIGRATIONS[version](dict(data))
        version = data["version"]
    return data


def save(data, path):
    path = Path(path)
    validate(data)
    clean = {k: v for k, v in data.items() if not k.startswith("_")}
    # through a temporary file: a crash mid-write must not leave the user's decisions half written
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def path_of(data, key):
    """A manifest path as an absolute Path, or None when unset."""
    value = data.get(key)
    if not value:
        return None
    value = Path(value)
    if value.is_absolute() or "_path" not in data:
        return value
    return (Path(data["_path"]).parent / value).resolve()


def validate(data):
    """Raise ValueError naming every problem, so a hand-edited manifest is fixed in one go."""
    problems = []
    if data.get("version") != VERSION:
        problems.append(f"version must be {VERSION}")
    for key in ("name", "source"):
        if not data.get(key):
            problems.append(f"{key} is required")
    for key, point in data.get("joints", {}).items():
        if key not in JOINTS:
            problems.append(f"unknown joint {key!r}")
        elif not _is_point(point):
            problems.append(f"joint {key} must be [x, y, z]")
    options = data.get("options", {})
    for key in options:
        if key not in OPTIONS:
            problems.append(f"unknown option {key!r}")
    for key, allowed in (("eyes", EYES), ("fingers", FINGERS), ("tpose", TPOSE), ("skinning", SKINNING)):
        if options.get(key, OPTIONS[key]) not in allowed:
            problems.append(f"option {key} must be one of {', '.join('null (automatic)' if a is None else a for a in allowed)}")
    if not isinstance(options.get("chest", OPTIONS["chest"]), bool):
        problems.append("option chest must be true or false")
    amount = options.get("chest_amount", OPTIONS["chest_amount"])
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not CHEST_AMOUNT[0] <= amount <= CHEST_AMOUNT[1]:
        problems.append(f"option chest_amount must be a number from {CHEST_AMOUNT[0]} to {CHEST_AMOUNT[1]}")
    if options.get("eyes") != "slide" and options.get("slide"):
        problems.append("option slide only applies with eyes: slide")
    fix = options.get("mouth_fix")
    if fix is not None and (not isinstance(fix, dict) or set(fix) - {"scale", "free", "rest", "lift"}):
        problems.append("option mouth_fix must be null or {scale, free, rest, lift}")
    names = [piece.get("name") for piece in data.get("pieces", [])]
    for piece in data.get("pieces", []):
        label = piece.get("name") or "a piece"
        if not piece.get("name"):
            problems.append("every piece needs a name")
        if names.count(piece.get("name")) > 1:
            problems.append(f"piece name {label!r} is used twice")
        if not _is_point(piece.get("point")):
            problems.append(f"{label}: point must be [x, y, z]")
        if not isinstance(piece.get("count"), int) or piece["count"] <= 0:
            problems.append(f"{label}: count must be the piece's vertex count")
        role = piece.get("role")
        if role not in ROLES:
            problems.append(f"{label}: role must be one of {', '.join(ROLES)}")
        region = piece.get("region")
        if region is not None and (not isinstance(region, list) or not all(isinstance(i, int) and i >= 0 for i in region)):
            problems.append(f"{label}: region must be a list of vertex numbers")
        if role == "follow_bone" and not piece.get("bone"):
            problems.append(f"{label}: follow_bone needs a bone (a VRM bone name, e.g. LeftUpperArm)")
        if role == "detail" and piece.get("anchor") not in names:
            problems.append(f"{label}: detail needs an anchor naming another piece")
        if piece.get("anchor") == piece.get("name") and piece.get("name"):
            problems.append(f"{label}: a detail can't anchor to itself")
    if problems:
        raise ValueError("manifest: " + "; ".join(dict.fromkeys(problems)))


def pieces_for(data, stage):
    """The pieces a stage acts on ("build" or "pieces")."""
    return [p for p in data.get("pieces", []) if ROLES[p["role"]] == stage]


def _is_point(value):
    return (isinstance(value, (list, tuple)) and len(value) == 3
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value))
