"""Look a character over before rigging it, and say how to build it.

Every build option in the pipeline (`--rigid-head`, `--cloth`, `--hat-tip`,
`--mouth-colour skin`, a hand-written `--joints`) was learned by rigging a model, seeing
what broke and reading the workflow doc. The mesh knows most of it already: whether the
head is hair or a solid ball under a hat, whether cloth hangs below the hips, whether the
legs are hidden by a robe, whether the generator left eye inserts to blink with.

This measures those, and the build reports the same survey, so a bad rig can be traced
back to what the model looked like going in.
"""

import numpy as np

from . import face_points, landmarks, parts, texture_space
from .face import _dominant_color
from .meshdata import pieces, sample_surface, triangles, world_vertices

SAMPLES = 120_000
# Head: hair worth swinging hangs below the head joint by this share of the head's height.
# Colour can't tell hair from headwear -- the crown of a mascot in a witch hat samples the hat,
# so every dark piece on it reads as hair -- but how far a piece hangs can: a brim reaches a
# third of a head down, a bob or a ponytail much further.
HAIR_HANG = 0.6
# Pieces smaller than this share of the hair aren't worth measuring (clips, ties, a bobble).
HAIR_PIECE = 0.05
# The top of the head narrowing to under this share of its widest point is worth mentioning
# --hat-tip for. It isn't chosen automatically: a witch hat's cone and a round hood measured
# too close together (0.52 against 0.4 of the widest) to tell apart from the silhouette.
TIP_NARROW = 0.7
TIP_BAND = 0.1
# Cloth: surface below the hips and this far outside the leg capsules, as a share of the
# surface below the hips. A skirt, a cape or coat tails clear that; bare legs don't.
CLOTH_CLEAR = 0.05
CLOTH_SHARE = 0.25
# Below the share that asks for cloth chains, but worth mentioning (a short skirt).
CLOTH_MAYBE = 0.1
# Arms: how far below horizontal the arm hangs. A T-pose is recommended: the arms stand clear of
# the body (no sleeve fused to the ribs) and the bake to the VRM rest pose barely moves them.
T_POSE = 15
ARMS_DOWN = 80
# Poly count: below this the face rig has nothing to cut, above it every step crawls.
COARSE_TRIANGLES = 3_000
HEAVY_TRIANGLES = 300_000
# Skin: a face colour this far from any believable skin tone means a mouth painted red
# inside would look wrong (a mascot's face).
SKIN_TONES = np.array([[1.0, 0.87, 0.78], [0.94, 0.76, 0.63], [0.76, 0.57, 0.44], [0.45, 0.31, 0.24]])
SKIN_DISTANCE = 0.25
# Symmetry: how far the silhouette's width may differ between the sides, as a share of width.
LOPSIDED = 0.08


def survey(mesh_obj, joint_warnings=()):
    """Measure a prepared character and work out how it should be built.

    Returns a dict of measurements, `notes` worth telling the user, and the build
    `options` the mesh asks for (the keys build_avatar takes). Nothing here changes
    the mesh, and nothing here is fatal: a model that measures badly still builds, with
    the notes saying what to expect.
    """
    verts = world_vertices(mesh_obj)
    tris = triangles(mesh_obj.data)
    points = sample_surface(verts, tris, SAMPLES)
    warnings = list(dict.fromkeys(joint_warnings))

    joints = landmarks.detect_body(verts, tris, warnings)
    joints = landmarks.refine_limbs(verts, tris, joints)
    found = _find_parts(mesh_obj, joints)

    report = {
        "size": [round(float(v), 3) for v in np.ptp(verts, axis=0)],
        "vertices": len(mesh_obj.data.vertices),
        "triangles": len(tris),
        "pieces": int(len(np.unique(pieces(mesh_obj, verts)))),
        "materials": [slot.material.name for slot in mesh_obj.material_slots if slot.material],
        "texture": _texture_size(mesh_obj),
        "joints": {k: [round(float(c), 4) for c in v] for k, v in joints.items()},
        "pose": _pose(joints),
        "symmetry": _symmetry(points),
        "head": _head(points, verts, joints, found),
        "cloth": _cloth(points, joints),
        "parts": _parts_report(found),
        # where the modelled eyes are (the face patch's head view is framed on them), or None
        "eyes": _eyes_centre(verts, found),
        "face_colour": _face_colour(mesh_obj, found),
        # a face patch (see scripts/retexture/face_patch.py) is what --free-mouth needs to paint
        "face_patch": face_points._patch_image(mesh_obj) is not None,
    }
    report["options"] = _options(report, found)
    report["notes"] = list(dict.fromkeys(warnings + _notes(report, found)))
    return report


def _find_parts(mesh_obj, joints):
    """parts.find, or None on a model it can't read (no texture, no separate pieces)."""
    try:
        return parts.find(mesh_obj, joints)
    except (ValueError, KeyError, IndexError):
        return None


def _eyes_centre(verts, found):
    if found is None or not found.has_eye_inserts:
        return None
    eyes = np.isin(found.labels, [int(p) for p in found.eyes.values() if p >= 0])
    return [round(float(v), 4) for v in verts[eyes].mean(axis=0)] if eyes.any() else None


def _texture_size(mesh_obj):
    try:
        image = texture_space.base_color_image(mesh_obj)
    except ValueError:
        return None
    return list(image.size)


def _pose(joints):
    """How the arms are held: T-pose (recommended), A-pose, or arms down."""
    drops = []
    for side in ("left", "right"):
        arm = np.asarray(joints[f"{side}_hand"]) - np.asarray(joints[f"{side}_upper_arm"])
        flat = float(np.linalg.norm(arm[:2]))
        drops.append(float(np.degrees(np.arctan2(-arm[2], max(flat, 1e-6)))))
    drop = float(np.mean(drops))
    kind = "A-pose" if T_POSE <= drop <= ARMS_DOWN else ("T-pose" if drop < T_POSE else "arms down")
    return {"arm_drop_degrees": round(drop, 1), "kind": kind}


def _symmetry(points):
    """How alike the two sides are: 1.0 is a mirror image, lower means the pose is lopsided."""
    centre = float(np.median(points[:, 0]))
    bands = np.linspace(points[:, 2].min(), points[:, 2].max(), 24)
    scores = []
    for low, high in zip(bands[:-1], bands[1:]):
        band = points[(points[:, 2] >= low) & (points[:, 2] < high)]
        if len(band) < 50:
            continue
        left = float(np.percentile(band[:, 0], 98) - centre)
        right = float(centre - np.percentile(band[:, 0], 2))
        widest = max(left, right, 1e-6)
        scores.append(1.0 - abs(left - right) / widest)
    return round(float(np.mean(scores)) if scores else 1.0, 3)


def _head(points, verts, joints, found):
    """What the head is made of: hair to swing, or one solid piece, and whether it comes to a tip."""
    head_z = float(joints["head"][2])
    top = float(points[:, 2].max())
    head = points[points[:, 2] > head_z]
    centre = np.array([float(joints["head"][0]), float(np.median(head[:, 1]))]) if len(head) else np.zeros(2)
    radius = np.linalg.norm(head[:, :2] - centre, axis=1) if len(head) else np.zeros(1)
    tip_band = head[:, 2] > top - TIP_BAND * (top - head_z) if len(head) else np.zeros(0, dtype=bool)
    widest = float(np.percentile(radius, 98)) if len(head) else 0.0
    tip = float(np.percentile(radius[tip_band], 98)) if tip_band.any() else widest
    return {
        "hair_share": _hair_share(verts, head_z, found),
        "hair_hang": _hair_hang(verts, joints, found),
        "tip_ratio": round(tip / widest, 3) if widest else 1.0,
        "height": round(top - head_z, 3),
    }


def _hair_share(verts, head_z, found):
    if found is None:
        return None
    above = verts[:, 2] > head_z
    return round(float(found.hair[above].mean()), 3) if above.any() else 0.0


def _hair_hang(verts, joints, found):
    """How far the hair hangs below the head joint, in head heights; None with no parts to read.

    Measured per piece, because one strand reaching the waist is what asks for spring chains,
    while a hat brim and a bobble sitting on the crown don't.
    """
    if found is None or not found.hair.any():
        return 0.0 if found is not None else None
    head_z = float(joints["head"][2])
    head_height = max(float(verts[:, 2].max()) - head_z, 1e-6)
    hair_labels = found.labels[found.hair]
    hang = 0.0
    for piece in np.unique(hair_labels):
        piece_verts = verts[found.labels == piece]
        if len(piece_verts) < HAIR_PIECE * len(hair_labels):
            continue
        hang = max(hang, (head_z - float(piece_verts[:, 2].min())) / head_height)
    return round(hang, 2)


def _cloth(points, joints):
    """How much surface hangs below the hips clear of the legs: a skirt, a cape or coat tails."""
    hip_z = float(joints["hips"][2])
    below = points[points[:, 2] < hip_z]
    if len(below) < 100:
        return {"share": 0.0, "lowest": None}
    clear = np.ones(len(below), dtype=bool)
    for side in ("left", "right"):
        hip = np.asarray(joints[f"{side}_upper_leg"], dtype=float)
        ankle = np.asarray(joints[f"{side}_foot"], dtype=float)
        shaft = ankle - hip
        along = np.clip((below - hip) @ shaft / float(shaft @ shaft), 0.0, 1.0)
        distance = np.linalg.norm(below - (hip + along[:, None] * shaft), axis=1)
        clear &= distance > CLOTH_CLEAR + _leg_radius(joints)
    return {"share": round(float(clear.mean()), 3),
            "lowest": round(float(below[clear, 2].min()), 3) if clear.any() else None}


def _leg_radius(joints):
    """Half the gap between the legs, as a stand-in for how thick they are."""
    return abs(float(joints["left_upper_leg"][0]) - float(joints["right_upper_leg"][0])) / 2


def _parts_report(found):
    if found is None:
        return {"read": False}
    return {
        "read": True,
        "eye_inserts": bool(found.eyes),
        "mouth_insert": found.mouth >= 0,
        "face_part": found.face >= 0,
        "hair_parts": int(np.count_nonzero(found.hair)),
    }


def _options(report, found):
    """The build_avatar options this mesh asks for."""
    hang = report["head"]["hair_hang"]
    rigid_head = hang is not None and hang < HAIR_HANG
    options = {
        "rigid_head": rigid_head,
        "cloth_physics": report["cloth"]["share"] > CLOTH_SHARE,
    }
    if rigid_head and report["face_colour"] is not None and _is_unskinlike(report["face_colour"]):
        options["mouth_color"] = "skin"
    return options


def _face_colour(mesh_obj, found):
    """The face's own colour, to tell a human face from a mascot's."""
    if found is None or found.face < 0:
        return None
    try:
        colours = texture_space.vertex_colors(mesh_obj, texture_space.base_color_image(mesh_obj))
    except ValueError:
        return None
    face = colours[found.labels == found.face]
    return [round(float(c), 3) for c in _dominant_color(face)] if len(face) else None


def _is_unskinlike(colour):
    """Whether a colour is nothing like skin, so a red mouth inside would jar."""
    return bool(np.linalg.norm(SKIN_TONES - np.asarray(colour, dtype=float), axis=1).min() > SKIN_DISTANCE)


def _notes(report, found):
    """Plain-language warnings about what this model will cost to rig."""
    notes = []
    pose = report["pose"]
    if pose["kind"] == "arms down":
        notes.append(f"the arms hang at the sides ({pose['arm_drop_degrees']:.0f}° below horizontal): sleeves fuse "
                     "with the body there. Generate the model in a T-pose if you can, and check the shoulders and sleeves")
    elif pose["kind"] == "A-pose":
        notes.append(f"an A-pose ({pose['arm_drop_degrees']:.0f}° below horizontal) builds, but a T-pose builds "
                     "better: the arms stand clear of the body and barely move when baked to the VRM rest pose")
    if report["symmetry"] < 1 - LOPSIDED:
        notes.append(f"the two sides differ by {(1 - report['symmetry']) * 100:.0f}% of the body's width: "
                     "bones are placed symmetrically, so one side may sit off its limb")
    if report["triangles"] > HEAVY_TRIANGLES:
        notes.append(f"{report['triangles']:,} triangles is heavy; every step will be slow and VTubing apps "
                     "will struggle. Decimate to about 50k first")
    if report["triangles"] < COARSE_TRIANGLES:
        notes.append(f"only {report['triangles']:,} triangles: the face is too coarse to cut a mouth into, "
                     "so build with --free-mouth and add face tracking with --mouth-only")
    if report["texture"] is None:
        notes.append("no base colour texture: hair, face and skin colour can't be read, so parts detection "
                     "and the face rig are off. Export the model with its texture")
    if found is None:
        notes.append("the model isn't split into parts (one fused mesh): the face is rigged from the painted "
                     "texture instead, and there are no eye inserts to blink")
    elif not found.eyes:
        notes.append("no eye inserts found: mark the eyes and mouth with viewer/setup.html and pass --face face.json")
    if report["options"]["rigid_head"]:
        notes.append(f"nothing on the head hangs more than {report['head']['hair_hang']:.1f} head heights below it "
                     "(a helmet, a hat or a solid head, not hair to swing): building with --rigid-head")
    if report["options"]["rigid_head"] and report["head"]["tip_ratio"] < TIP_NARROW:
        notes.append(f"the head narrows to {report['head']['tip_ratio']:.0%} of its width at the top: if that is a "
                     "pointed hat rather than a round hood, add --hat-tip to swing its cone")
    if report["options"]["cloth_physics"]:
        notes.append(f"{report['cloth']['share'] * 100:.0f}% of the surface below the hips hangs clear of the legs "
                     "(a skirt, a cape or coat tails): building with --cloth")
    elif report["cloth"]["share"] > CLOTH_MAYBE:
        notes.append(f"{report['cloth']['share'] * 100:.0f}% of the surface below the hips hangs clear of the legs: "
                     "if that is a skirt or coat tails rather than boots, --cloth will swing it")
    if report["pieces"] > 40:
        notes.append(f"{report['pieces']} separate pieces: loose straps, buckles and clips are where weighting "
                     "goes wrong, so check them once it is rigged")
    return notes
