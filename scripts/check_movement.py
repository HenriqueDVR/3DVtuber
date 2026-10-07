"""Which pieces don't move with the body: left behind by an arm or the head, or torn by it.

    blender -b rigged.blend -P scripts/check_movement.py -- --reference source.glb --json movement.json
        [--manifest character.json] [--height 1.6]

Run on the rigged model *before* the T-pose bake (the modelled pose): the check reads what
each piece touches at rest, and in a baked T-pose a sleeve left behind at the ribs has already
come away from the arm it hung from. Model B's hanging sleeve was taken for the mantle and
"repaired" twice for want of this.

The arms are pointed about world axes (raised, forward, T-pose, at the sides) and the head
turned and nodded. Every piece of the source (labelled as scripts/list_pieces.py labels them)
is followed where it touches another piece at rest: when the other piece moves away and this
one stays, it is left behind -- a sleeve the arm leaves hanging in the air, a hat the head
turns out from under. Nearness to a bone can't tell that apart: an A-pose arm hangs beside the
chest, and model B's brooch and model A's corset trims sat nearer the upper arm than any torso bone
while rightly staying with the chest they sit on. A piece whose own edges stretch is torn
(part goes, part stays: a skirt pulled up by the hands resting on it). Each finding gets the
manifest role that fixed it last time. A piece left behind by a forearm or hand gets none: the
hanging hand rested on it, and on model A (its 762-vertex waist piece) what was wrong was the part
of the skirt the hand took along -- following the arm would have put the waist on it. Changes
nothing.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "addon"))

from vtuber_maker.pipeline import manifest, meshdata, roles  # noqa: E402
from vtuber_maker.pipeline.rig import BONE_NAMES  # noqa: E402

# Where the left arm is pointed (shoulder to wrist; the right mirrors), as in check_avatar.py.
ARM_POSES = {"arms raised": (0.5, 0, 1), "arms forward": (0.3, -1, -0.3), "arms T-pose": (1, 0, 0),
             "arms at sides": (0.25, 0, -1)}
HEAD_TURNS = {"head turn": ((0, 0, 1), 0.6), "head nod": ((1, 0, 0), 0.4)}
# Two pieces touch where a vertex of one lies within this (m, at 1.6 m tall) of the other.
CONTACT = 0.015
# A touching vertex comes away when it ends up this far (m) from where its neighbour on the other
# piece went, and it is the one left behind when the neighbour moved at least LAG of that.
SEPARATE = 0.05
LAG = 0.5
# A piece stays behind when at least MIN_LEFT of its touching vertices, and LEFT_SHARE of them,
# are left behind: model B's hanging sleeves lost 60-76% of their contact, the robe beside its
# A-pose forearm 23% (it rightly stays), a torso touching a sleeve at the armpit less.
MIN_LEFT = 10
LEFT_SHARE = 0.35
# Cloth draped over the shoulder (a mantle) doesn't come away from anything as the arm moves: the
# arm slides under it. So the vertices of a piece over an upper arm (on its outer or top half,
# past its first 10%, within DRAPE of it) are compared with where the upper arm would carry them rigidly (1 = with it, 0 = not
# at all); a median under DRAPE_FOLLOWS over at least MIN_LEFT of them and DRAPE_SHARE of the piece
# stays behind. Model B's mantle: 43 such vertices (4%), following 0 before its repair.
DRAPE = 0.12
DRAPE_FOLLOWS = 0.3
DRAPE_SHARE = 0.02
MIN_TRAVEL = 0.03
# A piece draped over both shoulders that reaches below this share of the hips' height is a cape,
# not a mantle: it hangs from the chest and swings, and isn't blended onto the arms.
CAPE_BELOW_HIPS = 0.75
# Pieces with these roles are meant to lag behind: their springs swing them.
SWINGS = ("hair", "swing_cloth", "appendage")
# A piece whose median vertex is at least this much on spring bones swings, as those roles do.
SWINGING = 0.5
# A piece is torn when one of its own edges grows by more than this (m): check_avatar.py's limit.
TORN = 0.06


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True, help="the source model the manifest's pieces are on")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--height", type=float, default=1.6)
    parser.add_argument("--listing", type=Path, help="scripts/list_pieces.py's JSON: what each piece looks like")
    args = parser.parse_args(argv)
    data = manifest.load(args.manifest) if args.manifest else None
    looks = ({r["label"]: r["suggested"] for r in json.loads(args.listing.read_text(encoding="utf-8"))["pieces"]}
             if args.listing and args.listing.exists() else {})
    scale = args.height / 1.6

    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = max((o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm), key=lambda o: len(o.data.vertices))
    with roles.reference(args.reference.resolve(), height=args.height) as source:
        labels, mapped, _, tree = roles.source_labels(mesh, source)
        source_verts = meshdata.world_vertices(source)
    rest = meshdata.world_vertices(mesh)
    count = len(rest)
    edges = np.empty(len(mesh.data.edges) * 2, dtype=np.int64)
    mesh.data.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)
    same_piece = mapped[edges[:, 0]] == mapped[edges[:, 1]]
    rest_length = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    partner = _partners(rest, mapped, CONTACT * scale)

    # the body bone each vertex is nearest, to name the bone a left-behind piece should follow
    body = [b for b in BONE_NAMES.values() if b in arm.data.bones]
    world = np.array(arm.matrix_world)
    ends = {b: [(world @ np.array([*getattr(arm.data.bones[b], k), 1.0]))[:3] for k in ("head_local", "tail_local")]
            for b in body}
    nearest = np.array(body)[np.argmin(np.stack([_segment_distance(rest, *ends[b]) for b in body], axis=1), axis=1)]

    def posed():
        bpy.context.view_layer.update()
        evaluated = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
        flat = evaluated.to_mesh()
        coords = np.empty(len(flat.vertices) * 3)
        flat.vertices.foreach_get("co", coords)
        evaluated.to_mesh_clear()
        m = np.array(mesh.matrix_world)
        return coords.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]

    def reset():
        for bone in arm.pose.bones:
            bone.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()

    def turn(name, axis, angle):
        bone = arm.pose.bones[name]
        head = bone.head.copy()
        bone.matrix = Matrix.Translation(head) @ Matrix.Rotation(angle, 4, axis) @ Matrix.Translation(-head) @ bone.matrix
        bpy.context.view_layer.update()

    draped = {}
    for side in ("Left", "Right"):
        if f"{side}UpperArm" in ends:
            head, tail = ends[f"{side}UpperArm"]
            axis = (tail - head) / np.linalg.norm(tail - head)
            t = np.clip((rest - head) @ axis / np.linalg.norm(tail - head), 0, 1)
            offset = rest - (head + t[:, None] * (tail - head))
            # over the arm is outward for an arm hanging beside the body and up for one held out:
            # the torso's side beside a hanging arm is also outside the shoulder joint, but under it
            over = np.array([np.sign(head[0]), 0.0, 1.0])
            over -= (over @ axis) * axis
            over /= np.linalg.norm(over)
            on_top = offset @ over > 0
            draped[side] = on_top & (t > 0.1) & (np.linalg.norm(offset, axis=1) < DRAPE * scale)

    def carried(side):
        """Where the upper arm would carry every vertex, rigidly."""
        name = f"{side}UpperArm"
        motion = np.array(arm.matrix_world @ arm.pose.bones[name].matrix
                          @ arm.data.bones[name].matrix_local.inverted() @ arm.matrix_world.inverted())
        return rest @ motion[:3, :3].T + motion[:3, 3]

    poses, rigid = {}, {}
    for label, direction in ARM_POSES.items():
        reset()
        for side, mirror in (("Left", 1.0), ("Right", -1.0)):
            upper, hand = f"{side}UpperArm", f"{side}Hand"
            if upper not in arm.pose.bones or hand not in arm.pose.bones:
                continue
            now = (arm.pose.bones[hand].head - arm.pose.bones[upper].head).normalized()
            want = Vector((mirror * direction[0], direction[1], direction[2])).normalized()
            axis = now.cross(want)
            if axis.length > 1e-6:
                turn(upper, axis.normalized(), now.angle(want))
        poses[label] = posed()
        rigid[label] = {side: carried(side) for side in draped}
    if "Head" in arm.pose.bones:
        for label, (axis, angle) in HEAD_TURNS.items():
            reset()
            turn("Head", Vector(axis), angle)
            poses[label] = posed()
    reset()

    # pieces on spring bones swing on their own when a limb moves under them, role or not (model A's
    # T-pose locks over the shoulders, found as hair by the build)
    springs = {joint.node.bone_name for spring in arm.data.vrm_addon_extension.spring_bone1.springs for joint in spring.joints}
    spring_groups = {g.index for g in mesh.vertex_groups if g.name in springs}
    on_springs = np.array([sum(g.weight for g in v.groups if g.group in spring_groups) for v in mesh.data.vertices])
    named = {}
    for piece in (data["pieces"] if data else []):
        _, i, _ = tree.find(piece["point"])
        named[int(labels[i])] = piece
    shoulder_x = {s: abs(ends[f"{s}UpperArm"][0][0]) for s in ("Left", "Right") if f"{s}UpperArm" in ends}
    hips_z = ends["Hips"][0][2] if "Hips" in ends else rest[:, 2].mean()
    neck_z = ends["Neck"][0][2] if "Neck" in ends else rest[:, 2].max()
    contacts = {}
    touching = partner >= 0
    report = []
    for label in np.unique(mapped):
        members = mapped == label
        source_members = labels == label
        points = source_verts[source_members]
        at = points[np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1))]
        row = {"label": int(label), "count": int(source_members.sum()), "point": [round(float(v), 9) for v in at],
               "touching": int((members & touching).sum()), "poses": {}, "findings": []}
        if int(label) in named:
            row["name"], row["role"] = named[int(label)]["name"], named[int(label)]["role"]
        piece_edges = same_piece & members[edges[:, 0]]
        contact = np.flatnonzero(members & touching)
        other = partner[contact]
        for pose, now in poses.items():
            moved = now - rest
            gap = np.linalg.norm(moved[contact] - moved[other], axis=1)
            lag = np.linalg.norm(moved[other], axis=1) - np.linalg.norm(moved[contact], axis=1)
            left = (gap > SEPARATE * scale) & (lag > LAG * gap)
            length = np.linalg.norm(now[edges[:, 0]] - now[edges[:, 1]], axis=1)
            growth = np.where(piece_edges & (rest_length > 1e-6), length - rest_length, 0.0)
            entry = {"left_behind": int(left.sum()), "stretch_cm": round(float(growth.max()) * 100, 1),
                     "moved_cm": round(float(np.linalg.norm(moved[members], axis=1).max()) * 100, 1)}
            row["poses"][pose] = entry
            behind = bool(left.sum() >= max(MIN_LEFT, LEFT_SHARE * len(contact)))
            drape_bone = None
            # a collar rightly stays on the chest while the head turns out of it: only what sits
            # on the head (a hat, hair) should turn with it
            if pose.startswith("head") and rest[members, 2].mean() < neck_z:
                behind = False
            seen = contacts.setdefault(int(label), [len(contact), 0])
            seen[1] = max(seen[1], int(left.sum()))
            for side, over in (rigid.get(pose) or {}).items():
                on = members & draped[side]
                travel = np.linalg.norm(over - rest, axis=1)
                on &= travel > MIN_TRAVEL * scale
                if on.sum() >= max(MIN_LEFT, DRAPE_SHARE * members.sum()):
                    follow = np.clip(1 - np.linalg.norm(now[on] - over[on], axis=1) / travel[on], 0, 1)
                    entry[f"follow_{side.lower()}"] = round(float(np.median(follow)), 2)
                    if np.median(follow) < DRAPE_FOLLOWS and not behind:
                        behind, left = True, np.zeros(len(contact), dtype=bool)
                        drape_bone = f"{side}UpperArm"
            if behind and row.get("role") not in SWINGS and np.median(on_springs[members]) < SWINGING:
                taken = None
                if left.any():
                    bones, votes = np.unique(nearest[other[left]], return_counts=True)
                    bone = str(bones[np.argmax(votes)])
                    # the piece that moved away from it
                    labels_, votes = np.unique(mapped[other[left]], return_counts=True)
                    taken = int(labels_[np.argmax(votes)])
                else:
                    bone = drape_bone
                row["findings"].append({"pose": pose, "kind": "stays behind", "bone": bone, "taken_by": taken,
                                        "share": round(float(left.mean()), 2) if left.any() else None})
            elif entry["stretch_cm"] > TORN * scale * 100:
                row["findings"].append({"pose": pose, "kind": "torn", "stretch_cm": entry["stretch_cm"]})
        if row["findings"]:
            row["suggested"], row["why"] = _suggest(row, rest[members], shoulder_x, hips_z, looks.get(int(label)), named)
            # seen, and right as it is (a cape the arms move under): listed so the user knows it was
            # looked at, but not a problem. Shown as one, "the fix: as it should be" read as an error
            row["expected"] = not row["suggested"] and row["why"].endswith("as they should")
            report.append(row)
    report.sort(key=lambda r: -len(r["findings"]))
    # every piece's touching vertices and the most of them left behind in any pose, for tuning
    contacts = {k: v for k, v in contacts.items() if v[1]}
    args.json.write_text(json.dumps({"blend": bpy.data.filepath, "vertices": count, "pieces": report,
                                     "contacts": contacts}, indent=2),
                         encoding="utf-8")
    for row in report:
        print(f"MOVEMENT piece {row['label']} ({row['count']} vertices{', ' + row['name'] if 'name' in row else ''}): "
              + "; ".join(f"{f['kind']} with {f['pose']}" + (f" ({f['share']:.0%} of its contact)" if f.get("share") else "")
                          for f in row["findings"]) + f" -> {row['suggested'] or '?'}")
    print(f"VTUBER_MAKER_REPORT {json.dumps({'pieces_flagged': len(report)})}")


def _partners(rest, labels, reach):
    """For each vertex, the nearest vertex of another piece within `reach`, or -1."""
    tree = KDTree(len(rest))
    for i, p in enumerate(rest):
        tree.insert(p, i)
    tree.balance()
    partner = np.full(len(rest), -1)
    for i, p in enumerate(rest):
        for _, j, distance in tree.find_range(p, reach):
            if labels[j] != labels[i] and (partner[i] < 0 or distance < np.linalg.norm(rest[partner[i]] - p)):
                partner[i] = j
    return partner


def _suggest(row, points, shoulder_x, hips_z, looks=None, named=None):
    """(role, why): the manifest role that fixed this kind of finding before, or "" for none."""
    behind = [f for f in row["findings"] if f["kind"] == "stays behind"]
    torn = [f for f in row["findings"] if f["kind"] == "torn"]
    if any(f["pose"].startswith("head") for f in behind):
        if looks == "hair":
            return "hair", "hair that stays when the head turns: its strands need their own springs"
        return "hat", "on the head but stays when it turns: a hat or headwear, rigid on the head"
    if behind and looks == "hair":
        # model A's T-pose locks over the shoulders: the arm moves under them, and following the arm
        # would carry hair hanging from the head away with it
        return "hair", "hair the arm moves under: its strands need their own springs"
    if behind:
        bone = behind[0]["bone"]
        side = "Left" if bone.startswith("Left") else "Right"
        # a piece spanning from inside one shoulder joint to inside the other drapes the shoulders
        if points[:, 0].min() < -shoulder_x.get("Right", 0) * .5 and points[:, 0].max() > shoulder_x.get("Left", 0) * .5:
            if points[:, 2].min() < CAPE_BELOW_HIPS * hips_z:
                # model D's cape: named a mantle on this advice, its collar's tips went with the arms
                return "", "a cape hanging from the shoulders: the arms move free under it, as they should"
            return "shoulder_cloth", "drapes over both shoulders but stays when the arms move: a mantle"
        if bone.endswith(("LowerArm", "Hand")):
            taken = behind[0].get("taken_by")
            taken = (named or {}).get(taken, {}).get("name") or f"piece {taken}"
            return "", (f"the hanging {side.lower()} hand rested on it: if it belongs to the body it stays, and {taken}, "
                        f"which the hand took along, may hang off the forearm (T-pose render): Fitted skirt on {taken}; "
                        f"if it is part of the sleeve or glove, give it Follows a bone, {side}LowerArm, on Parts")
        return "follow_bone", f"hangs from the {side.lower()} arm but stays when it moves: follow {side}UpperArm"
    # a skirt the hands rest on (model A's) sits about the hips; a coat tearing at the armpit isn't one
    if torn and points[:, 2].mean() < hips_z:
        return "fitted_skirt", "part of it goes with a hand resting on it and part stays: a skirt the hands touch"
    if torn and row.get("role"):
        # a mantle blended onto the arms stretches by design; how much is too much is for the eye
        return "", f"already {row['role']}, and stretches: look at it in the Review renders"
    if torn:
        return "", "tears when it moves; no role covers this yet: gather it for an agent (Stuck?)"
    return "", ""


def _segment_distance(points, a, b):
    ab = b - a
    t = np.clip((points - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


main()
