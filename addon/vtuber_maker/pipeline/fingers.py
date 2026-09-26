"""Finger bones for hand tracking, read off the hand's own shape.

Warudo and VSeeFace drive the VRM finger bones (three per finger, plus the thumb's metacarpal)
from webcam hand tracking. Image-to-3D characters come with no bones at all, and their hands are
one lump: the fingers are ridges separated by grooves, not separate pieces, and on a gloved hand
like model A's they are fused solid.

The fingers are found by walking the hand's surface from the wrist, not by slicing it. These
hands are modelled with the fingers curled, so a fingertip can sit level with the palm and even
touch it: anything measured along the hand's axis puts palm vertices on fingertip bones, and a
2 mm crease then tears into 9 cm when an app closes the hand. Surface distance can't be fooled
that way -- the fingertips are the patches furthest from the wrist along the surface, each
finger is what lies nearest its own tip, and a chain laid down that path follows the curl.

Weights fade between neighbouring fingers, which keeps the webbing between fused ones from
tearing when only one of them curls.
"""

import bpy
import numpy as np
from mathutils import Vector

from .meshdata import welded_edges, world_vertices
from .rig import BONE_NAMES, make_active

# VRM finger bones, in order from the palm out. The thumb's first bone is its metacarpal,
# which lies inside the palm; the other fingers start at the knuckle.
FINGERS = ("thumb", "index", "middle", "ring", "little")
SEGMENTS = {
    "thumb": ("metacarpal", "proximal", "distal"),
    "other": ("proximal", "intermediate", "distal"),
}
# Where each bone's head sits along a finger, as a fraction of the finger's length.
JOINTS_AT = (0.0, 0.45, 0.75)
# A vertex under this weight on the hand bone isn't part of the hand.
HAND_WEIGHT = 0.4
# Surface distance from the wrist, as a share of the furthest: fingertips are looked for past
# the first, and nothing nearer than the second can belong to a finger.
TIP_FROM = 0.9
TIP_LOWEST = 0.6
FINGER_FROM = 0.45
# A fingertip patch holds at least this many vertices.
TIP_MIN = 6
# A vertex belongs to a finger while it is within this much of that finger's own length from
# its tip; what lies in the overlap past 1 fades between neighbours.
FINGER_REACH = 1.15
FADE = 0.25
# Segment weights blend over this fraction of a finger's length either side of a joint.
SEGMENT_FADE = 0.12
# Passes of weight smoothing over the hand, so no weight has a hard edge across the webbing.
SMOOTHING = 3
# No bone in a chain is shorter than this share of the finger, end to end. Two joints that land
# on the same spot make a bone of no length, which Blender drops when it leaves edit mode, and
# VRM then refuses the hand for a missing proximal bone.
JOINT_APART = 0.12
# A patch whose chain would be shorter than this share of the hand's reach is a bump, not a
# finger: its joints land on each other and Blender drops the bones as having no length.
FINGER_MIN = 0.15
# ... and a chain longer than this share of the reach has run off the hand and up the arm.
FINGER_MAX = 0.6
# A chain is the thumb, rather than a fifth finger, when it is this much shorter than the rest.
THUMB_SHORTER = 0.8


def bone_name(side, finger, segment):
    return f"{side.capitalize()}{finger.capitalize()}{segment.capitalize()}"


def humanoid_keys(side, finger):
    """The VRM humanoid keys for one finger, in order (left_index_proximal, ...)."""
    return tuple(f"{side}_{finger}_{segment}" for segment in _segments(finger))


def build_fingers(mesh_obj, armature_obj, sides=("left", "right")):
    """Add finger bones to both hands and move the hands' weights onto them.

    Returns a report per side: which fingers were found, how big the hand is and how long they
    are. Run after skinning and before the T-pose bake: it reads the modelled shape.
    """
    report = {}
    verts = world_vertices(mesh_obj)
    welded = welded_edges(mesh_obj, verts)
    for side in sides:
        hand_bone = BONE_NAMES[f"{side}_hand"]
        if hand_bone not in armature_obj.data.bones or hand_bone not in mesh_obj.vertex_groups:
            continue
        found = _find_fingers(mesh_obj, armature_obj, verts, welded, side)
        if found is None or not found["fingers"]:
            report[side] = {"fingers": 0, "reason": "no fingers stand out from the hand's surface"}
            continue
        _add_bones(armature_obj, side, found)
        _reweight(mesh_obj, side, found)
        _smooth(mesh_obj, side, found, welded)
        report[side] = {
            "fingers": len(found["fingers"]),
            "named": [finger["finger"] for finger in found["fingers"]],
            "hand_cm": [round(float(found[key] * 100), 1) for key in ("reach", "hand_width", "hand_thick")],
            "finger_length_cm": [round(float(finger["straight"] * 100), 1) for finger in found["fingers"]],
        }
    return report


def _segments(finger):
    return SEGMENTS["thumb"] if finger == "thumb" else SEGMENTS["other"]


def _find_fingers(mesh_obj, armature_obj, verts, welded, side):
    """Measure one hand: its frame, the surface distance from the wrist, and a path per finger."""
    hand_bone = BONE_NAMES[f"{side}_hand"]
    bone = armature_obj.data.bones[hand_bone]
    wrist = np.array((armature_obj.matrix_world @ bone.head_local)[:])
    tip = np.array((armature_obj.matrix_world @ bone.tail_local)[:])
    along = (tip - wrist) / np.linalg.norm(tip - wrist)

    weight = _weights(mesh_obj, hand_bone, len(verts))
    local = verts - wrist
    depth = local @ along
    hand = _hand_piece(verts, tip, (weight >= HAND_WEIGHT) & (depth > 0.0), welded)
    if hand.sum() < 60:
        return None
    graph = _graph(verts, hand, welded)
    reach = float(np.percentile(depth[hand], 97))

    from_wrist = _walk(graph, hand & (depth < 0.12 * reach), len(verts))
    walked = hand & np.isfinite(from_wrist)
    if walked.sum() < 60:
        return None
    span = float(from_wrist[walked].max())
    patches = _tip_patches(graph, walked, from_wrist, span)
    if len(patches) < 2:
        return None

    across, through = _palm_axes(local, along, hand & (depth > 0.15 * reach) & (depth < 0.55 * reach))
    across = _thumb_side(across, wrist)
    coords = np.column_stack((depth, local @ across, local @ through))
    fingers = _name_fingers(_fingers_from_patches(graph, walked, from_wrist, span, patches, coords, reach))
    return {
        "wrist": wrist, "along": along, "across": across, "through": through,
        "hand": hand, "weight": weight, "coords": coords, "reach": reach, "fingers": fingers,
        "hand_width": float(np.percentile(coords[hand, 1], 98) - np.percentile(coords[hand, 1], 2)),
        "hand_thick": float(np.percentile(coords[hand, 2], 98) - np.percentile(coords[hand, 2], 2)),
    }


def _weights(mesh_obj, group_name, count):
    group = mesh_obj.vertex_groups[group_name].index
    weight = np.zeros(count)
    for index, vertex in enumerate(mesh_obj.data.vertices):
        for entry in vertex.groups:
            if entry.group == group:
                weight[index] = entry.weight
    return weight


def _hand_piece(verts, tip, selected, welded):
    """Only the surface the hand bone holds that is joined to the hand, not stray scraps near it."""
    weld, edge_a, edge_b = welded
    inside = selected[edge_a] & selected[edge_b]
    edge_a, edge_b = edge_a[inside], edge_b[inside]
    distance = np.where(selected, np.linalg.norm(verts - tip, axis=1), np.inf)
    reached = np.zeros(len(verts), dtype=bool)
    reached[weld[int(np.argmin(distance))]] = True
    while True:
        # buffered, because a vertex appears in many edges and a plain grown[edges] |= ... lets
        # one of those edges write False over a True another just wrote
        grown = reached.copy()
        np.logical_or.at(grown, edge_b, reached[edge_a])
        np.logical_or.at(grown, edge_a, reached[edge_b])
        if np.array_equal(grown, reached):
            return selected & reached[weld]
        reached = grown


def _graph(verts, hand, welded):
    """The hand's surface as edges between welded vertices, with their lengths."""
    weld, edge_a, edge_b = welded
    inside = hand[edge_a] & hand[edge_b] & (edge_a != edge_b)
    edge_a, edge_b = edge_a[inside], edge_b[inside]
    return weld, edge_a, edge_b, np.linalg.norm(verts[edge_a] - verts[edge_b], axis=1)


def _walk(graph, sources, count):
    """Distance from a set of vertices over the surface, spread edge by edge until it settles."""
    weld, edge_a, edge_b, lengths = graph
    distance = np.full(count, np.inf)
    distance[np.nonzero(sources)[0]] = 0.0
    distance = np.minimum(distance, distance[weld])
    for _ in range(3000):
        updated = distance.copy()
        np.minimum.at(updated, edge_b, distance[edge_a] + lengths)
        np.minimum.at(updated, edge_a, distance[edge_b] + lengths)
        settled = np.isfinite(updated) == np.isfinite(distance)
        if settled.all() and np.allclose(updated[np.isfinite(updated)], distance[np.isfinite(distance)]):
            break
        distance = updated
    return distance[weld]


def _tip_patches(graph, hand, from_wrist, span):
    """The fingertips: joined patches of the surface furthest from the wrist.

    How far out the fingers come apart depends on the model: a hand with its fingers pressed
    together only splits deep down, a spread hand splits at once. Every cut is tried and the one
    that splits into the most patches wins, the furthest out among ties, because a cut that
    reaches further down a fused hand starts swallowing the palm.
    """
    _, edge_a, edge_b, _ = graph
    best = []
    for cut in np.arange(TIP_FROM, TIP_LOWEST - 1e-9, -0.03):
        far = hand & (from_wrist >= cut * span)
        patches = [patch for patch in _patches(far, edge_a, edge_b) if patch.sum() >= TIP_MIN]
        if len(patches) > len(best):
            best = patches
        if len(best) >= 5:
            break
    return sorted(best, key=lambda patch: -patch.sum())[:5]


def _patches(selected, edge_a, edge_b):
    """Split a selection into joined patches."""
    outside = len(selected)
    label = np.where(selected, np.arange(outside), outside)
    inside = selected[edge_a] & selected[edge_b]
    edge_a, edge_b = edge_a[inside], edge_b[inside]
    while True:
        updated = label.copy()
        np.minimum.at(updated, edge_a, label[edge_b])
        np.minimum.at(updated, edge_b, label[edge_a])
        updated[~selected] = outside
        if np.array_equal(updated, label):
            break
        label = updated
    return [label == value for value in np.unique(label[selected])]


def _fingers_from_patches(graph, hand, from_wrist, span, patches, coords, reach):
    """Grow each fingertip back down the surface: what lies nearest a tip is that finger."""
    zone = hand & (from_wrist >= FINGER_FROM * span)
    walks = np.array([_walk(graph, patch, len(hand)) for patch in patches])
    walks[~np.isfinite(walks)] = np.inf
    owner = np.argmin(walks, axis=0)
    fingers = []
    for index, patch in enumerate(patches):
        walk = walks[index]
        # how long the finger is, measured over its own vertices: the whole zone is within reach
        # of every tip through the palm, so measuring there gives the distance across the hand
        own = zone & (owner == index) & np.isfinite(walk)
        if own.sum() < TIP_MIN * 2:
            continue
        # a chain whose territory leaks down into the palm is cut back to a finger's length
        # rather than thrown away: the rest of it is palm, and the fingertip still needs bones
        length = min(float(np.percentile(walk[own], 99)), FINGER_MAX * reach)
        members = own & (walk <= FINGER_REACH * length)
        if members.sum() < TIP_MIN * 2 or length <= 0.0:
            continue
        path = _path(coords, members, walk, length)
        straight = float(np.linalg.norm(path[-1] - path[0]))
        if not FINGER_MIN * reach <= straight <= FINGER_MAX * reach:
            continue
        fingers.append({
            "members": members, "from_tip": walk, "length": length, "straight": straight,
            "tip": coords[patch].mean(axis=0), "path": path,
        })
    return fingers


def _path(coords, members, from_tip, length):
    """The finger's own middle line, from its knuckle to its tip, at each joint."""
    path = []
    for at in JOINTS_AT + (1.0,):
        here = (1.0 - at) * length
        band = members & (np.abs(from_tip - here) <= 0.2 * length + 1e-4)
        if band.sum() < 3:
            band = members
        path.append(np.median(coords[band], axis=0))
    return _spread(path)


def _spread(path):
    """Push joints that landed on the same spot apart, so every bone has some length."""
    straight = path[-1] - path[0]
    span = float(np.linalg.norm(straight))
    if span < 1e-6:
        return path
    step = JOINT_APART * span
    direction = straight / span
    spread = [path[0]]
    for point in path[1:]:
        gap = point - spread[-1]
        if float(np.linalg.norm(gap)) < step:
            point = spread[-1] + step * direction
        spread.append(point)
    return spread


def _palm_axes(local, along, palm_slab):
    """Across the palm and through it: the palm is wider than it is thick, so its widest axis."""
    flat = local[palm_slab] - np.outer(local[palm_slab] @ along, along)
    _, _, directions = np.linalg.svd(flat - flat.mean(axis=0), full_matrices=False)
    across = directions[0] - along * (directions[0] @ along)
    across /= np.linalg.norm(across)
    return across, np.cross(along, across)


def _thumb_side(across, wrist):
    """Point `across` at the thumb: forward and towards the body, where a hanging hand has it."""
    towards = np.array([-1.0 if wrist[0] > 0 else 1.0, -1.0, 0.0])
    return across if across @ towards >= 0 else -across


def _name_fingers(found):
    """Name them across the hand: the thumb is the one furthest round it, then index outwards.

    A hand gives at most five chains and a VRM bone can only be used once, so any extra chain is
    dropped rather than sharing a name -- bones nothing drives would still carry mesh weight.
    """
    if not found:
        return found
    by_side = sorted(found, key=lambda finger: -finger["tip"][1])
    thumb = None
    if len(by_side) >= 5:
        thumb = by_side[0]
    elif len(by_side) >= 2:
        # with four or fewer, one of them is only the thumb when it is clearly the short one
        others = by_side[1:]
        if by_side[0]["length"] < THUMB_SHORTER * float(np.median([f["length"] for f in others])):
            thumb = by_side[0]
    rest = [finger for finger in by_side if finger is not thumb][:4]
    if thumb is not None:
        thumb["finger"] = "thumb"
    for name, finger in zip(("index", "middle", "ring", "little"), rest):
        finger["finger"] = name
    return ([thumb] if thumb is not None else []) + rest


def _point(found, at):
    """A (depth, width, palm) triple back in world space."""
    depth, width, palm = at
    return Vector((found["wrist"] + depth * found["along"] + width * found["across"]
                   + palm * found["through"]).tolist())


def _add_bones(armature_obj, side, found):
    make_active(armature_obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = armature_obj.data.edit_bones
    for finger in found["fingers"]:
        parent = bones[BONE_NAMES[f"{side}_hand"]]
        for index, segment in enumerate(_segments(finger["finger"])):
            bone = bones.new(bone_name(side, finger["finger"], segment))
            bone.head = _point(found, finger["path"][index])
            bone.tail = _point(found, finger["path"][index + 1])
            # every finger hinges about the same axis across the palm, so an app that curls a
            # hand by turning each bone about its own X folds the fingers together
            bone.align_roll(Vector(found["through"].tolist()))
            bone.parent = parent
            bone.use_connect = index > 0
            parent = bone
    bpy.ops.object.mode_set(mode="OBJECT")


def _reweight(mesh_obj, side, found):
    """Hand the hand bone's weight on to the fingers, fading between neighbours."""
    hand_group = mesh_obj.vertex_groups[BONE_NAMES[f"{side}_hand"]]
    groups = {finger["finger"]: [mesh_obj.vertex_groups.get(bone_name(side, finger["finger"], segment))
                                 or mesh_obj.vertex_groups.new(name=bone_name(side, finger["finger"], segment))
                                 for segment in _segments(finger["finger"])]
              for finger in found["fingers"]}
    claimed = np.zeros(len(found["hand"]), dtype=bool)
    for finger in found["fingers"]:
        claimed |= finger["members"]
    for index in np.nonzero(claimed)[0]:
        shares = {}
        for finger in found["fingers"]:
            distance = finger["from_tip"][index] / finger["length"]
            near = 1.0 if distance <= 1.0 else max(0.0, 1.0 - (distance - 1.0) / FADE)
            if near > 0.0:
                shares[finger["finger"]] = (near, float(np.clip(1.0 - distance, 0.0, 1.0)))
        total = sum(near for near, _ in shares.values())
        if total <= 0.0:
            continue
        # only what the hand bone held is handed on; the rest of this vertex may be the forearm's
        holds = float(found["weight"][index])
        for name, (near, along) in shares.items():
            for group, part in zip(groups[name], _segment_shares(along)):
                if part > 0.0:
                    group.add([int(index)], float(holds * near / total * part), "REPLACE")
        hand_group.add([int(index)], 0.0, "REPLACE")


def _smooth(mesh_obj, side, found, welded):
    """Average each hand vertex's weights over its neighbours, so no weight has a hard edge."""
    weld, edge_a, edge_b = welded
    region = found["hand"]
    inside = region[edge_a] & region[edge_b]
    edge_a, edge_b = weld[edge_a[inside]], weld[edge_b[inside]]
    names = [BONE_NAMES[f"{side}_hand"]]
    names += [bone_name(side, finger["finger"], segment)
              for finger in found["fingers"] for segment in _segments(finger["finger"])]
    groups = [mesh_obj.vertex_groups[name] for name in names if name in mesh_obj.vertex_groups]

    weights = np.array([_weights(mesh_obj, group.name, len(region)) for group in groups])
    neighbours = np.bincount(np.concatenate([edge_a, edge_b]), minlength=len(region))
    for _ in range(SMOOTHING):
        summed = weights.copy()
        for row in range(len(groups)):
            np.add.at(summed[row], edge_a, weights[row][edge_b])
            np.add.at(summed[row], edge_b, weights[row][edge_a])
        weights = summed / np.maximum(neighbours + 1, 1)
        weights[:, ~region] = 0.0
    # UV seams split a vertex in two and only one of them carries the edges, so the smoothed
    # weight is read back from whichever vertex stands for that position
    weights = weights[:, weld]
    total = weights.sum(axis=0)
    share = found["weight"]  # the hand's share of each vertex, which smoothing must not change
    for row, group in enumerate(groups):
        for index in np.nonzero(region)[0]:
            if total[index] > 1e-6:
                group.add([int(index)], float(share[index] * weights[row, index] / total[index]), "REPLACE")


def _segment_shares(along):
    """How one point along a finger divides between its three bones."""
    shares = [0.0, 0.0, 0.0]
    for k, edge in enumerate(JOINTS_AT[1:]):
        if along < edge - SEGMENT_FADE:
            shares[k] = 1.0
            return shares
        if along <= edge + SEGMENT_FADE:
            mix = (along - (edge - SEGMENT_FADE)) / (2 * SEGMENT_FADE)
            shares[k], shares[k + 1] = 1.0 - mix, mix
            return shares
    shares[2] = 1.0
    return shares
