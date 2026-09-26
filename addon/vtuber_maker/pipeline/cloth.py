"""Bone chains for hanging cloth (capes, coat tails, skirts) and a pointed hat's tip.

Hanging cloth is found by shape: separate pieces that reach below the knees but mostly
stand off the legs (a cape does, trousers and knee pads don't). The part of each piece
below the hips becomes side-by-side chains rooted on the hips, split by angle round the
body like wide hair, and blended between neighbouring chains so the cloth doesn't tear
between them. A pointed hat's tip is the part of the headwear furthest over its surface
from the brim; it gets one chain from the head, and small pieces hanging off its end (a
tassel) follow the last bone.
"""

import math
from typing import NamedTuple

import bpy
import numpy as np
from mathutils import Vector

from .meshdata import pieces, welded_edges, world_vertices
from .rig import BONE_NAMES, make_active

BONES_PER_CHAIN = 4
# Cloth spanning more than this angle round the body is split into several chains.
CHAIN_SPREAD = 2 * math.pi / 8
MIN_CHAIN_VERTICES = 20
MAX_INFLUENCES = 4
# A piece is hanging cloth if its median distance from the leg bones is at least this
# fraction of the character's height, and it reaches below the knees ...
OFF_LEGS = 0.06
# ... and less than this share of its weight is on the arms (wide sleeves hang from the arm).
MAX_ARM_SHARE = 0.5
# Within a piece that is cloth, a vertex the arms already drive this much stays on the arms.
# One robe can be a cape at the back and a sleeve by the wrist, and the piece's average hides
# it: a sleeve cuff pulled onto a cape's chain tore 22 cm when the arm lifted.
VERTEX_ARM_SHARE = 0.4
# ... and it is at least this fraction of the height long (boots and soles aren't cloth).
MIN_CLOTH_LENGTH = 0.2
# Chains hanging less than this fraction of the height below the hips aren't worth swinging.
MIN_HANG = 0.08
# Fraction of a chain's length over which weight hands over from the body.
ROOT_BLEND = 0.2

TIP_BONES = 4
# The tip chain covers the whole cone: the hat above the head and within CONE_WIDTH of the
# head's radius from its middle. Its weight fades in over the lowest TIP_ROOT_BLEND of the
# cone's height; the part within TIP_DROOP of the way from the end follows fully even where a
# drooping tip hangs out past the head.
TIP_ROOT_BLEND = 0.3
TIP_DROOP = 0.5
CONE_WIDTH = 0.95
# ... fading in over this share of that radius past it
DROOP_BLEND = 0.3
# Without a head under the headwear, the brim is this lowest percentage of its vertices.
BRIM_SHARE = 30
# Small pieces within TASSEL_GAP (m) of the tip's end that are no bigger than TASSEL_SIZE
# (fraction of the height) hang off it and follow its last bone.
TASSEL_GAP = 0.02
TASSEL_SIZE = 0.06


class Chain(NamedTuple):
    names: list
    positions: np.ndarray  # joint positions, root first, one more than bones
    parent: str
    angle: float = 0.0
    width: float = 1.0


def build_cloth_chains(mesh_obj, armature, joints, skip=None, force=None, detect=True):
    """Add chains for hanging cloth under the hips and reweight the cloth onto them.

    Run after skinning. `skip` masks vertices never to touch (the head, hair). `force` masks
    pieces the user named as swinging cloth (see manifest.py): they skip the tests below that
    decide what hangs free. With `detect` False only forced pieces swing. Returns the chains as
    lists of bone names, root first.
    """
    verts = world_vertices(mesh_obj)
    n = len(verts)
    skip = np.zeros(n, dtype=bool) if skip is None else np.asarray(skip, dtype=bool)
    force = np.zeros(n, dtype=bool) if force is None else np.asarray(force, dtype=bool)
    height = float(np.ptp(verts[:, 2]))
    j = {key: np.asarray(value, dtype=float) for key, value in joints.items()}
    root_z = float(j["hips"][2])
    knee_z = float((j["left_lower_leg"][2] + j["right_lower_leg"][2]) / 2)
    centre = j["hips"][:2]
    leg_distance = np.min([_segment_distance(verts, j[f"{side}_{a}"], j[f"{side}_{b}"])
                           for side in ("left", "right")
                           for a, b in (("upper_leg", "lower_leg"), ("lower_leg", "foot"))], axis=0)
    arm_share = _share(mesh_obj, n, {BONE_NAMES[f"{side}_{key}"] for side in ("left", "right")
                                     for key in ("upper_arm", "lower_arm", "hand")})
    labels = pieces(mesh_obj, verts)
    angle = np.arctan2(verts[:, 1] - centre[1], verts[:, 0] - centre[0])

    chains, strands = [], []
    for label in np.unique(labels[~skip]):
        members = (labels == label) & ~skip
        forced = bool(force[members].any())
        if not forced and not detect:
            continue
        if not forced and (verts[members, 2].min() > knee_z or np.ptp(verts[members, 2]) < MIN_CLOTH_LENGTH * height
                or np.median(leg_distance[members]) < OFF_LEGS * height or arm_share[members].mean() > MAX_ARM_SHARE):
            continue
        hanging = np.nonzero(members & (verts[:, 2] < root_z) & (arm_share < VERTEX_ARM_SHARE))[0]
        if len(hanging) < MIN_CHAIN_VERTICES:
            continue
        mid = _circular_mean(angle[hanging])
        offsets = _angle_between(angle[hanging], mid)
        low, high = float(offsets.min()), float(offsets.max())
        count = max(1, math.ceil((high - low) / CHAIN_SPREAD))
        width = max((high - low) / count, 1e-3)
        piece_chains = []
        for k in range(count):
            start = low + k * width
            pick = (offsets >= start) & ((offsets < start + width) | (k == count - 1))
            points = verts[hanging[pick]]
            if len(points) < MIN_CHAIN_VERTICES or root_z - points[:, 2].min() < MIN_HANG * height:
                continue
            index = len(chains) + len(piece_chains)
            names = [f"Cloth{index}_{b}" for b in range(BONES_PER_CHAIN)]
            piece_chains.append(Chain(names, _hanging_positions(points, root_z), BONE_NAMES["hips"],
                                      _circular_mean(np.array([mid + start + width / 2])), width))
        if piece_chains:
            chains.extend(piece_chains)
            strands.append((hanging, piece_chains))

    _add_bones(armature, chains)
    new_weights = {}
    for hanging, piece_chains in strands:
        for vertex in hanging:
            shares = [(c, 1.0 - abs(_angle_between(angle[vertex], c.angle)) / c.width) for c in piece_chains]
            shares = [(c, s) for c, s in shares if s > 0] or [
                (min(piece_chains, key=lambda c: abs(_angle_between(angle[vertex], c.angle))), 1.0)]
            total = sum(s for _, s in shares)
            chain_weights = {}
            along = 0.0
            for c, s in shares:
                tip_z = c.positions[-1][2]
                along = min(max((root_z - verts[vertex, 2]) / max(root_z - tip_z, 1e-6), 0.0), 1.0)
                _along_chain(chain_weights, c.names, along, s / total)
            new_weights[int(vertex)] = (chain_weights, min(along / ROOT_BLEND, 1.0))
    _apply(mesh_obj, new_weights)
    return [c.names for c in chains]


def build_tip_chain(mesh_obj, armature, joints, rigid, face=None):
    """Add a chain along a pointed hat's tip under the head bone; returns [names] or [].

    `rigid` marks the vertices fully on the head (see rig.make_head_rigid), `face` the head's
    own surface. The headwear is the rigid piece reaching highest; its brim is the part below
    the top of the head under it.
    """
    verts = world_vertices(mesh_obj)
    height = float(np.ptp(verts[:, 2]))
    labels = pieces(mesh_obj, verts)
    rigid = np.asarray(rigid, dtype=bool)
    if not rigid.any():
        return []
    top = int(np.argmax(np.where(rigid, verts[:, 2], -np.inf)))
    hat = np.nonzero(labels == labels[top])[0]
    # the brim (tilted or not) is everything below the top of the head it sits on
    under = np.zeros(len(verts), dtype=bool) if face is None else np.asarray(face, dtype=bool) & (labels != labels[top])
    crown = verts[under, 2].max() if under.any() else np.percentile(verts[hat, 2], BRIM_SHARE)
    # ... reached from its lowest point without going over the cone (a drooping tip is lower too)
    low_part = hat[verts[hat, 2] <= crown]
    reach = _surface_distance(mesh_obj, verts, [low_part[np.argmin(verts[low_part, 2])]], low_part)
    brim = low_part[np.isfinite(reach[low_part])]

    distance = _surface_distance(mesh_obj, verts, brim, hat)
    far = float(distance[hat].max())
    if not np.isfinite(far) or far <= 0:
        return []
    # The cone sits on the head: it's the hat above the top of the head and within the head's
    # width, plus a drooping end hanging out past that (near the end over the surface). The
    # brim, tilted or not, is wider than the head.
    end = int(hat[np.argmax(distance[hat])])
    from_end = _surface_distance(mesh_obj, verts, [end], hat)
    if under.any():
        # round the head bone: the face may be only the front of the head
        centre = np.asarray(joints["head"], dtype=float)[:2]
        head_radius = float(np.percentile(np.linalg.norm(verts[under, :2] - centre, axis=1), 95))
        over_head = np.linalg.norm(verts[hat, :2] - centre, axis=1) <= CONE_WIDTH * head_radius
    else:
        over_head = np.ones(len(hat), dtype=bool)
    cone = hat[over_head & (verts[hat, 2] > crown)]
    cone = cone[_connected(mesh_obj, verts, cone, end)] if end in set(cone.tolist()) else cone
    droop = TIP_DROOP * float(from_end[cone].max()) if len(cone) else 0.0
    candidates = np.union1d(cone, hat[from_end[hat] <= droop])
    tip = candidates[_connected(mesh_obj, verts, candidates, end)]
    d = 1.0 - from_end[tip] / from_end[tip].max()
    blend_height = max(TIP_ROOT_BLEND * (float(verts[tip, 2].max()) - crown), 1e-6)
    share = np.clip((verts[tip, 2] - crown) / blend_height, 0.0, 1.0)
    # a drooping end hanging out past the head follows fully, however low it hangs
    if under.any():
        out = np.linalg.norm(verts[tip, :2] - centre, axis=1) / (CONE_WIDTH * head_radius)
        past_head = np.clip((out - 1.0) / DROOP_BLEND, 0.0, 1.0)
        share = np.where(from_end[tip] <= droop, np.maximum(share, past_head), share)
    positions = []
    for k in range(TIP_BONES + 1):
        at = k / TIP_BONES
        band = np.abs(d - at) <= 0.5 / TIP_BONES
        pick = tip[band] if band.any() else tip[np.argsort(np.abs(d - at))[:10]]
        positions.append(verts[pick].mean(axis=0))
    positions[-1] = verts[tip[np.argmax(d)]]
    names = [f"HatTip_{b}" for b in range(TIP_BONES)]
    chain = Chain(names, np.array(positions), BONE_NAMES["head"])
    _add_bones(armature, [chain])

    new_weights = {}
    for vertex, along, part in zip(tip, d, share):
        weights = {}
        _along_chain(weights, names, float(along), 1.0)
        new_weights[int(vertex)] = (weights, float(part))
    # a tassel hanging off the tip's end
    end = verts[tip[d > 1 - 1.0 / TIP_BONES]]
    for label in np.unique(labels):
        members = np.nonzero(labels == label)[0]
        if label == labels[top] or np.ptp(verts[members], axis=0).max() > TASSEL_SIZE * height:
            continue
        gap = np.min(np.linalg.norm(verts[members][:: max(1, len(members) // 30), None] - end[None], axis=2))
        if gap <= TASSEL_GAP:
            for vertex in members:
                new_weights[int(vertex)] = ({names[-1]: 1.0}, 1.0)
    _apply(mesh_obj, new_weights)
    return [names]


# The body's core, which appendages hang from: bones checked for the nearest to a piece's root.
CORE_BONES = ("hips", "spine", "chest", "upper_chest", "neck", "head")
APPENDAGE_BONES = 4
# An appendage's root: its vertices within this (m, at 1.6 m) of its nearest point to the body core.
APPENDAGE_ROOT = 0.03
# The first share of an appendage along it blends from its bone onto the chain, so the root
# doesn't tear from the back it grows out of.
APPENDAGE_ROOT_BLEND = 0.15
# It forks into two chains when a second end, at least this share of the length out, lies this
# share of the length away from the first over the surface.
APPENDAGE_FORK = 0.6
# Small pieces riding on an appendage (see attach_riders): size as a share of the height, and how
# far (m) their median vertex may lie from it. Model C's feather and patch decorations float 1.1 to
# 2.9 cm off her wings and tails (a 2 cm cut on half their vertices missed most, and they flew off
# in the viewer); her hair flowers and ears, beside the top tail, lie 4 cm and more away.
RIDER_SIZE = 0.15
RIDER_GAP = 0.035


def build_appendage_chain(mesh_obj, armature, joints, members, index, bone=None):
    """Put a wing, tail or big ear on one core bone and swing it on a chain; returns the names.

    `members` marks the piece. The whole piece first goes onto `bone` -- by default the core bone
    (hips to head) nearest to where the piece meets the body -- so no limb moves it: Tripo's
    auto-rig gave model C's wings to the upper arms and her tails to the thighs. Then one chain
    runs from that root to the point furthest from it over the piece's surface, and the piece
    is weighted along it, blending from the bone over the first APPENDAGE_ROOT_BLEND.
    """
    from .loose import follow_bone
    verts = world_vertices(mesh_obj)
    height = float(np.ptp(verts[:, 2]))
    piece = np.flatnonzero(np.asarray(members, dtype=bool))
    if len(piece) < MIN_CHAIN_VERTICES:
        return []
    world = np.array(armature.matrix_world)
    segments = []
    for key in CORE_BONES:
        if BONE_NAMES[key] in armature.data.bones:
            b = armature.data.bones[BONE_NAMES[key]]
            head, tail = ((world @ np.array([*p, 1.0]))[:3] for p in (b.head_local, b.tail_local))
            if key == "head":  # the head runs up to the top of the model, ears and all
                tail = np.array([head[0], head[1], verts[:, 2].max()])
            segments.append((key, head, tail))
    distance = np.stack([_segment_distance(verts[piece], a, b) for _, a, b in segments], axis=1)
    near = distance.min(axis=1)
    closest = int(np.argmin(near))
    if bone is None:
        bone = BONE_NAMES[segments[int(np.argmin(distance[closest]))][0]]
    follow_bone(mesh_obj, armature, piece, bone)
    seeds = piece[near <= near.min() + APPENDAGE_ROOT * height / 1.6]
    along = _surface_distance(mesh_obj, verts, seeds, piece)
    reach = along[piece]
    finite = np.isfinite(reach)
    if not finite.any() or reach[finite].max() <= 0:
        return []
    reach = np.where(finite, reach, reach[finite].max())
    # A piece forking into two ends far apart (model C's bottom pair of tails, one piece) gets a
    # chain down each: one chain down the middle swings both as one.
    tips = [int(piece[np.argmax(reach)])]
    from_first = _surface_distance(mesh_obj, verts, tips, piece)[piece]
    far = np.where(np.isfinite(from_first) & (reach > APPENDAGE_FORK * reach.max()), from_first, -1.0)
    if far.max() > APPENDAGE_FORK * reach.max():
        tips.append(int(piece[np.argmax(far)]))
        from_second = _surface_distance(mesh_obj, verts, [tips[1]], piece)[piece]
        branch = (from_second < from_first).astype(int)
    else:
        branch = np.zeros(len(piece), dtype=int)
    chains, new_weights = [], {}
    for b, tip in enumerate(tips):
        mine = branch == b
        d = reach / reach[piece == tip][0]
        positions = [verts[seeds].mean(axis=0)]
        for k in range(1, APPENDAGE_BONES + 1):
            at = k / APPENDAGE_BONES
            band = mine & (np.abs(d - at) <= 0.5 / APPENDAGE_BONES)
            pick = piece[band] if band.any() else piece[mine][np.argsort(np.abs(d[mine] - at))[:10]]
            positions.append(verts[pick].mean(axis=0))
        positions[-1] = verts[tip]
        positions = np.array(positions)
        if np.linalg.norm(np.diff(positions, axis=0), axis=1).min() < 0.002:
            # two joints on one spot make a zero-length bone, which Blender deletes: evenly apart instead
            positions = positions[0] + np.linspace(0.0, 1.0, APPENDAGE_BONES + 1)[:, None] * (positions[-1] - positions[0])
        names = [f"Appendage{index + b}_{k}" for k in range(APPENDAGE_BONES)]
        chains.append(Chain(names, np.array(positions), bone))
        for vertex, at in zip(piece[mine], np.clip(d[mine], 0.0, 1.0)):
            weights = {}
            _along_chain(weights, names, float(at), 1.0)
            new_weights[int(vertex)] = (weights, min(float(at) / APPENDAGE_ROOT_BLEND, 1.0))
    _add_bones(armature, chains)
    _apply(mesh_obj, new_weights)
    return [c.names for c in chains]


def attach_riders(mesh_obj, carriers, skip):
    """Small pieces stuck onto `carriers` (anything on a spring) take the weights of the carrier under them.

    Model C's tails carry some twenty painted patches as separate pieces; left on the body bones
    they would stay while the tail swings. A piece rides when it is smaller than RIDER_SIZE and
    its median vertex lies within RIDER_GAP of a carrier. `skip` masks pieces the user
    named (they keep what they were given). Returns the number of vertices moved.
    """
    from mathutils.kdtree import KDTree
    verts = world_vertices(mesh_obj)
    height = float(np.ptp(verts[:, 2]))
    carriers = np.flatnonzero(np.asarray(carriers, dtype=bool))
    if not len(carriers):
        return 0
    tree = KDTree(len(carriers))
    for i, v in enumerate(carriers):
        tree.insert(verts[v], i)
    tree.balance()
    labels = pieces(mesh_obj, verts)
    carried = set(np.unique(labels[carriers]).tolist())
    skip = np.asarray(skip, dtype=bool)
    groups = mesh_obj.vertex_groups
    moved = 0
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if label in carried or skip[members].any() or np.ptp(verts[members], axis=0).max() > RIDER_SIZE * height:
            continue
        found = [tree.find(verts[v]) for v in members]
        if np.median([f[2] for f in found]) > RIDER_GAP * height / 1.6:
            continue
        for vertex, (_, i, _) in zip(members, found):
            source = mesh_obj.data.vertices[int(carriers[i])]
            weights = {groups[g.group].name: g.weight for g in source.groups}
            for g in list(mesh_obj.data.vertices[int(vertex)].groups):
                groups[g.group].remove([int(vertex)])
            for name, w in weights.items():
                groups[name].add([int(vertex)], w, "REPLACE")
            moved += 1
    return moved


def _connected(mesh_obj, verts, members, seed):
    """Mask over `members`: those joined to `seed` over edges between members."""
    weld, a, b = welded_edges(mesh_obj, verts)
    inside = np.zeros(len(verts), dtype=bool)
    inside[weld[members]] = True
    keep = inside[a] & inside[b]
    a, b = a[keep], b[keep]
    reached = np.zeros(len(verts), dtype=bool)
    reached[weld[seed]] = True
    while True:
        grown = reached.copy()
        grown[b[reached[a]]] = True
        grown[a[reached[b]]] = True
        if np.array_equal(grown, reached):
            return reached[weld[members]]
        reached = grown


def _hanging_positions(points, root_z):
    """BONES_PER_CHAIN + 1 joint positions from the hips' height down to the cloth's hem."""
    bottom = float(points[:, 2].min())
    step = (root_z - bottom) / BONES_PER_CHAIN
    positions = []
    for k in range(BONES_PER_CHAIN + 1):
        z = root_z - k * step
        band = np.abs(points[:, 2] - z) <= 0.6 * step
        pick = points[band] if band.any() else points[np.argsort(np.abs(points[:, 2] - z))[:10]]
        x, y = pick[:, 0].mean(), pick[:, 1].mean()
        positions.append([x, y, z])
    return np.array(positions)


def _along_chain(weights, names, along, share):
    """Add `share` of weight split between the two chain bones around `along` (0 root .. 1 tip)."""
    count = len(names)
    position = along * count - 0.5
    lower = int(min(max(math.floor(position), 0), count - 1))
    upper = min(lower + 1, count - 1)
    blend = min(max(position - lower, 0.0), 1.0)
    for index, part in ((lower, 1.0 - blend), (upper, blend)):
        weights[names[index]] = weights.get(names[index], 0.0) + share * part


def _apply(mesh_obj, new_weights):
    """Blend each vertex's body weights with its chain weights ({vertex: (chain weights, chain share)})."""
    groups = mesh_obj.vertex_groups
    for vertex, (chain_weights, share) in new_weights.items():
        v = mesh_obj.data.vertices[vertex]
        body = {groups[g.group].name: g.weight for g in v.groups}
        total = sum(body.values())
        combined = {name: (1 - share) * w / total for name, w in body.items()} if total > 1e-9 else {}
        for name, w in chain_weights.items():
            combined[name] = combined.get(name, 0.0) + share * w
        strongest = sorted(combined.items(), key=lambda item: -item[1])[:MAX_INFLUENCES]
        norm = sum(w for _, w in strongest)
        for g in list(v.groups):
            groups[g.group].remove([vertex])
        for name, w in strongest:
            if w > 1e-6:
                group = groups.get(name) or groups.new(name=name)
                group.add([vertex], w / norm, "REPLACE")


def _add_bones(armature, chains):
    if not chains:
        return
    make_active(armature)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = armature.data.edit_bones
    matrix = armature.matrix_world.inverted()
    for chain in chains:
        parent = bones[chain.parent]
        for k, name in enumerate(chain.names):
            bone = bones.new(name)
            bone.head = matrix @ Vector(chain.positions[k].tolist())
            bone.tail = matrix @ Vector(chain.positions[k + 1].tolist())
            bone.parent = parent
            bone.use_connect = k > 0
            parent = bone
    bpy.ops.object.mode_set(mode="OBJECT")


def _surface_distance(mesh_obj, verts, sources, within):
    """Shortest distance over the surface from `sources` to each vertex of `within` (inf elsewhere)."""
    weld, a, b = welded_edges(mesh_obj, verts)
    inside = np.zeros(len(verts), dtype=bool)
    inside[weld[within]] = True
    keep = inside[a] & inside[b]
    a, b = a[keep], b[keep]
    length = np.linalg.norm(verts[a] - verts[b], axis=1)
    distance = np.full(len(verts), np.inf)
    distance[weld[sources]] = 0.0
    for _ in range(len(within)):
        updated = distance.copy()
        np.minimum.at(updated, a, distance[b] + length)
        np.minimum.at(updated, b, distance[a] + length)
        if np.array_equal(updated, distance):
            break
        distance = updated
    return distance[weld]


def _segment_distance(points, start, end):
    shaft = end - start
    along = np.clip((points - start) @ shaft / max(float(shaft @ shaft), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (start + along[:, None] * shaft), axis=1)


def _share(mesh_obj, n, names):
    share = np.zeros(n)
    groups = mesh_obj.vertex_groups
    for v in mesh_obj.data.vertices:
        total = sum(g.weight for g in v.groups)
        if total > 1e-9:
            share[v.index] = sum(g.weight for g in v.groups if groups[g.group].name in names) / total
    return share


def _circular_mean(angles):
    return math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))


def _angle_between(a, b):
    return (np.asarray(a) - b + np.pi) % (2 * np.pi) - np.pi
