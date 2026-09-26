"""List a model's separate pieces, with the role each one looks like, for a character manifest.

    blender -b -P scripts/list_pieces.py -- <model.glb> --json pieces.json [--manifest character.json]

Each piece gets what a manifest names it by (a point on its surface and its vertex count, in
the normalized A-pose), its box, and a suggested role with the reason: hair and face inserts as
parts.find reads them, head pieces that aren't hair or face as a hat, and pieces hanging free
of the legs below the knee as swinging cloth. Suggestions only: the user decides (a hanging
sleeve looks like cloth until an arm moves). With --manifest its joints override detection
and its pieces show their chosen role. Changes nothing.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "addon"))

from vtuber_maker.pipeline import cloth, landmarks, manifest, meshdata, normalize, parts  # noqa: E402

# Pieces smaller than this share of the vertices are listed only when named in the manifest,
# or when they are hair or a face insert: a model can have hundreds of specks.
SPECK = 0.001
# Suggested as an appendage (wing, tail): its middle this far behind the body's, and this big, as
# shares of the height; model C's wings and tails sit 0.1-0.25 behind her, her twin-tails 0.03.
BEHIND = 0.06
BIG = 0.12


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--glb", type=Path, help="also export the normalized model, each vertex carrying its "
                                                 "piece label as the _PIECE attribute, for the app's 3D view")
    args = parser.parse_args(argv)
    data = manifest.load(args.manifest) if args.manifest else None
    height = data["options"]["height"] if data else 1.6

    bpy.ops.wm.read_homefile(use_empty=True)
    normalize.import_model(args.model)
    mesh = normalize.prepare(list(bpy.context.scene.objects), height=height)
    verts = meshdata.world_vertices(mesh)
    joints = landmarks.detect_body(verts, meshdata.triangles(mesh.data), [])
    if data:
        joints.update({k: np.asarray(v, dtype=float) for k, v in data["joints"].items()})
    found = parts.find(mesh, joints)
    labels = meshdata.pieces(mesh, verts)
    named = {}
    for piece in (data["pieces"] if data else []):
        near = int(np.argmin(np.linalg.norm(verts - np.asarray(piece["point"]), axis=1)))
        named[int(labels[near])] = piece

    knee_z = float((joints["left_lower_leg"][2] + joints["right_lower_leg"][2]) / 2)
    leg_distance = np.min([cloth._segment_distance(verts, np.asarray(joints[f"{side}_{a}"], dtype=float),
                                                   np.asarray(joints[f"{side}_{b}"], dtype=float))
                           for side in ("left", "right")
                           for a, b in (("upper_leg", "lower_leg"), ("lower_leg", "foot"))], axis=0)
    inserts = {int(p) for p in list(found.eyes.values()) + [found.mouth] if p >= 0}
    face_labels = set(np.unique(labels[found.labels == found.face]).tolist()) if found.face >= 0 else set()
    neck_z = float(joints["neck"][2])
    rigid_head = bool(data and data["options"]["rigid_head"])
    # the body itself: the biggest piece. Model C's (arms, legs and all) hangs below her knees clear of
    # her legs' guessed joints, and was suggested as swinging cloth
    body = int(np.bincount(labels).argmax())
    body_y = float(np.median(verts[labels == body, 1]))
    centre_x = float(joints["hips"][0])

    listed = []
    for label in np.unique(labels):
        members = labels == label
        count = int(members.sum())
        hair = bool(found.hair[members].mean() > .5)
        insert = bool(set(np.unique(found.labels[members]).tolist()) & inserts)
        if count < SPECK * len(verts) and not (hair or insert or int(label) in named):
            continue
        points = verts[members]
        # a vertex on the piece near its middle: the point a manifest names it by
        at = points[np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1))]
        if insert:
            role, why = "body", "an eye or mouth insert (face tracking moves it)"
        elif hair and rigid_head:
            # a solid head wears no hair: what reads as hair is headwear (model B's hat did)
            role, why = "hat", "on a rigid head, where nothing is hair"
        elif hair:
            role, why = "hair", "reads as hair"
        elif int(label) in face_labels:
            role, why = "body", "the face"
        elif points[:, 2].mean() > neck_z:
            role, why = "hat", "on the head and not hair or face"
        elif int(label) == body:
            role, why = "body", "the body itself"
        elif (np.median(points[:, 1]) > body_y + BEHIND * height and np.ptp(points, axis=0).max() > BIG * height
              and points[:, 2].min() > knee_z):
            role, why = "appendage", "big, and grows out of the back: a wing or a tail?"
        elif (points[:, 2].min() < knee_z and np.ptp(points[:, 2]) > cloth.MIN_CLOTH_LENGTH * height
              and np.median(leg_distance[members]) > cloth.OFF_LEGS * height):
            role, why = "swing_cloth", "hangs below the knees, clear of the legs"
        else:
            role, why = "body", "nothing special"
        row = {"label": int(label), "count": count, "point": [round(float(v), 9) for v in at],
               "min": points.min(axis=0).round(4).tolist(), "max": points.max(axis=0).round(4).tolist(),
               "suggested": role, "why": why, "where": _where(points, joints, centre_x, body_y, height)}
        if int(label) in named:
            row["manifest"] = named[int(label)]
        listed.append(row)
    # what the user most likely has to decide about first: named pieces, then the suggestions
    listed.sort(key=lambda row: ("manifest" not in row, row["suggested"] == "body", -row["count"]))
    report = {"model": str(args.model.resolve()), "vertices": len(verts), "pieces": listed,
              "joints": {k: np.asarray(v, dtype=float).round(4).tolist() for k, v in joints.items()}}
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.glb:
        export_labelled(mesh, labels, args.glb)
    print(f"VTUBER_MAKER_PIECE_LIST {len(listed)} pieces written to {args.json}")


def _where(points, joints, centre_x, body_y, height):
    """Where a piece is, in words: "head, left", "back", "right leg, front"..."""
    x, y, z = np.median(points, axis=0)
    neck, hips, knee = (float(joints[k][2]) for k in ("neck", "hips", "left_lower_leg"))
    side = "left" if x - centre_x > 0.04 * height else "right" if x - centre_x < -0.04 * height else ""
    if z > neck:
        area = "head"
    elif z > hips:
        area = "arm" if abs(x - centre_x) > abs(float(joints["left_upper_arm"][0]) - centre_x) else "chest"
    elif z > knee:
        area = "hips" if abs(x - centre_x) < 0.06 * height else "thigh"
    else:
        area = "leg"
    depth = "back" if y > body_y + 0.05 * height else "front" if y < body_y - 0.05 * height else ""
    if depth == "back" and area != "head":
        # behind the body a piece isn't on the arm or thigh it is level with: a wing, a tail
        return ", ".join(w for w in ("upper back" if z > hips else "lower back", side) if w)
    return ", ".join(w for w in ((f"{side} {area}" if side else area), depth) if w)


def export_labelled(mesh, labels, path):
    """The normalized model as GLB (Y up: Blender's (x, y, z) is glTF's (x, z, -y)), with _PIECE."""
    attribute = mesh.data.attributes.new("_PIECE", "FLOAT", "POINT")
    attribute.data.foreach_set("value", labels.astype(np.float32))
    # each vertex's number in the source, for a region (part of a piece) chosen in the app
    numbers = mesh.data.attributes.new("_VERTEX", "FLOAT", "POINT")
    numbers.data.foreach_set("value", np.arange(len(labels), dtype=np.float32))
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.export_scene.gltf(filepath=str(path.resolve()), export_format="GLB", use_selection=True,
                              export_attributes=True, export_yup=True)


if __name__ == "__main__":
    main()
