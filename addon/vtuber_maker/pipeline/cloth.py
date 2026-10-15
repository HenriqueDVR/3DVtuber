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
# Arm weight below the hips this far (fraction of the height) from the arm's bones isn't a sleeve's
# (hang_from_torso): model D's cape hem is 0.5 from them, a cuff by a hanging hand under 0.1.
ARM_REACH = 0.15
# Cloth taken back onto the torso changes from one torso bone to the next over at least this share
# of its distance from the spine (HANG_SAMPLES heights about its own averaged): a turn of the spine
# moves a point by that distance, so far out a handover as short as the body's own shears the
# cloth. Model D's cape went from Hips to Spine within 7 cm of height, 29 cm out: 6.6 cm of stretch
# in a twist, 14 edges folded in a side bend, a step in its outline. At a half it shears about as
# her body under it does (7 cm of handover 12 cm out). Averaged upward only, to keep every bone below
# the cloth out of it, the cape left the hips above its chains' roots and stretched 5.6 cm there.
HANG_BLEND = 0.5
HANG_SAMPLES = 9
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
    world = armature.matrix_world
    torso = [(BONE_NAMES[key], float((world @ armature.data.bones[BONE_NAMES[key]].head_local).z))
             for key in ("hips", "spine", "chest", "upper_chest") if BONE_NAMES[key] in armature.data.bones]

    chains, strands = [], []
    for label in np.unique(labels[~skip]):
        members = (labels == label) & ~skip
        forced = bool(force[members].any())
        if not forced and not detect:
            continue
        top, parent = root_z, BONE_NAMES["hips"]
        if forced and verts[members, 2].min() > root_z - MIN_HANG * height:
            # a cape on the shoulders that ends above the hips (the knight's, 1.40 to 0.94 m) hangs from
            # its own top, on the torso bone there: from the hips down there was nothing of it to swing
            top = float(verts[members, 2].max())
            parent = max(((name, z) for name, z in torso if z <= top), key=lambda bone: bone[1], default=(parent, 0))[0]
        if not forced and (verts[members, 2].min() > knee_z or np.ptp(verts[members, 2]) < MIN_CLOTH_LENGTH * height
                or np.median(leg_distance[members]) < OFF_LEGS * height or arm_share[members].mean() > MAX_ARM_SHARE):
            continue
        hanging = np.nonzero(members & (verts[:, 2] < top) & (arm_share < VERTEX_ARM_SHARE))[0]
        if len(hanging) < MIN_CHAIN_VERTICES:
            continue
        # what would swing must itself stand off the legs: a body that is one surface with its arms
        # and head has its middle far from the leg bones (model D), and its legs went onto chains
        if not forced and np.median(leg_distance[hanging]) < OFF_LEGS * height:
            continue
        if not forced:
            # ... each stretch of it by itself: a coat that is one surface with the body it is worn
            # on (a tester's pirate) hangs beside the legs it joins only above the hips, and the
            # trousers and boots went onto the coat's chains with it, 100% at the thighs
            stretch = _connected_stretches(mesh_obj, verts, hanging)
            legs = [s for s in np.unique(stretch) if np.median(leg_distance[hanging[stretch == s]]) < OFF_LEGS * height]
            hanging = hanging[~np.isin(stretch, legs)]
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
            if len(points) < MIN_CHAIN_VERTICES or top - points[:, 2].min() < MIN_HANG * height:
                continue
            index = len(chains) + len(piece_chains)
            names = [f"Cloth{index}_{b}" for b in range(BONES_PER_CHAIN)]
            piece_chains.append(Chain(names, _hanging_positions(points, top), parent,
                                      _circular_mean(np.array([mid + start + width / 2])), width))
        if piece_chains:
            chains.extend(piece_chains)
            strands.append((hanging, piece_chains, top))

    _add_bones(armature, chains)
    new_weights = {}
    for hanging, piece_chains, top in strands:
        for vertex in hanging:
            shares = [(c, 1.0 - abs(_angle_between(angle[vertex], c.angle)) / c.width) for c in piece_chains]
            shares = [(c, s) for c, s in shares if s > 0] or [
                (min(piece_chains, key=lambda c: abs(_angle_between(angle[vertex], c.angle))), 1.0)]
            total = sum(s for _, s in shares)
            chain_weights = {}
            along = 0.0
            for c, s in shares:
                tip_z = c.positions[-1][2]
                along = min(max((top - verts[vertex, 2]) / max(top - tip_z, 1e-6), 0.0), 1.0)
                _along_chain(chain_weights, c.names, along, s / total)
            new_weights[int(vertex)] = (chain_weights, min(along / ROOT_BLEND, 1.0))
    _apply(mesh_obj, new_weights)
    return [c.names for c in chains]


# Hanging sleeves (a miko's furisode): arm-weighted surface further than SLEEVE_OFF of the height
# from the arm's bones and hanging at least SLEEVE_HANG below the arm. On the arm alone they turned
# with it, and with the arms down stood out sideways "like side wings" in Warudo (the miko's: 68 cm
# deep, 1300 vertices a side). A chain every SLEEVE_SPAN along the arm, hanging from the flap's top:
# at 0.08 (three or four a sleeve) they swung apart in strips, nothing holding one to the next.
SLEEVE_OFF = 0.04
SLEEVE_HANG = 0.15
SLEEVE_SPAN = 0.16


def build_sleeve_chains(mesh_obj, armature, joints, skip=None):
    """Chains for sleeves hanging deep under the arms, on the upper or lower arm; returns their names.

    Run after skinning, on a rig whose arms stand out (T- or A-pose). `skip` masks vertices never to
    touch (hair, the head). A fitted sleeve, or a puffed one, hangs nowhere near SLEEVE_HANG: nothing.
    """
    verts = world_vertices(mesh_obj)
    n = len(verts)
    skip = np.zeros(n, dtype=bool) if skip is None else np.asarray(skip, dtype=bool)
    height = float(np.ptp(verts[:, 2]))
    j = {key: np.asarray(value, dtype=float) for key, value in joints.items()}
    chains, strands = [], []
    for side in ("left", "right"):
        shoulder, elbow, hand = (j[f"{side}_{key}"] for key in ("upper_arm", "lower_arm", "hand"))
        names = {BONE_NAMES[f"{side}_{key}"] for key in ("upper_arm", "lower_arm", "hand")}
        share = _share(mesh_obj, n, names)
        off = np.min([_segment_distance(verts, a, b) for a, b in ((shoulder, elbow), (elbow, hand))], axis=0)
        axis = (hand - shoulder) / max(float(np.linalg.norm(hand - shoulder)), 1e-9)
        t = (verts - shoulder) @ axis
        arm_z = shoulder[2] + np.clip(t, 0, None) * axis[2]
        flap = np.flatnonzero((share > 0.5) & ~skip & (off > SLEEVE_OFF * height) & (verts[:, 2] < arm_z)
                              & (t > 0) & (t < float((hand - shoulder) @ axis)))
        if len(flap) < MIN_CHAIN_VERTICES or float((arm_z[flap] - verts[flap, 2]).max()) < SLEEVE_HANG * height:
            continue
        low, high = float(t[flap].min()), float(t[flap].max())
        count = max(1, math.ceil((high - low) / (SLEEVE_SPAN * height)))
        width = max((high - low) / count, 1e-3)
        elbow_t = float((elbow - shoulder) @ axis)
        side_chains = []
        for k in range(count):
            mid = low + (k + 0.5) * width
            pick = flap[np.abs(t[flap] - mid) <= width / 2]
            points = verts[pick]
            top = float(points[:, 2].max()) if len(points) else 0.0
            if len(points) < MIN_CHAIN_VERTICES or top - points[:, 2].min() < SLEEVE_HANG * height:
                continue
            parent = BONE_NAMES[f"{side}_{'lower_arm' if mid > elbow_t else 'upper_arm'}"]
            index = len(chains) + len(side_chains)
            side_chains.append((Chain([f"Sleeve{index}_{b}" for b in range(BONES_PER_CHAIN)],
                                      _hanging_positions(points, top), parent, mid, width), top))
        chains += [chain for chain, _ in side_chains]
        strands.append((flap, t, side_chains, names))
    _add_bones(armature, chains)
    for flap, t, side_chains, arm in strands:
        for vertex in flap:
            shares = [(c, top, 1.0 - abs(t[vertex] - c.angle) / c.width) for c, top in side_chains]
            shares = [s for s in shares if s[2] > 0] or [min(((c, top, 1.0) for c, top in side_chains),
                                                            key=lambda s: abs(t[vertex] - s[0].angle))]
            total = sum(s for _, _, s in shares)
            chain_weights, along = {}, 0.0
            for c, top, s in shares:
                along = min(max((top - verts[vertex, 2]) / max(top - c.positions[-1][2], 1e-6), 0.0), 1.0)
                _along_chain(chain_weights, c.names, along, s / total)
            _arm_onto_chain(mesh_obj, int(vertex), arm, chain_weights, min(along / ROOT_BLEND, 1.0))
    return [c.names for c in chains]


def _arm_onto_chain(mesh_obj, vertex, arm, chain_weights, share):
    """Move `share` of the vertex's weight on the `arm` bones onto `chain_weights`; the rest stays.

    Only the arm's: a flap's torso weight by the armpit, given to the chains too (as `_apply`
    blends), tore the miko's kimono 6.1 cm from her side with the arms down.
    """
    groups = mesh_obj.vertex_groups
    v = mesh_obj.data.vertices[vertex]
    weights = {groups[g.group].name: g.weight for g in v.groups}
    total = sum(weights.values()) or 1.0
    moved = share * sum(w for name, w in weights.items() if name in arm) / total
    combined = {name: w / total * (1 - share if name in arm else 1.0) for name, w in weights.items()}
    for name, w in chain_weights.items():
        combined[name] = combined.get(name, 0.0) + moved * w
    strongest = sorted(combined.items(), key=lambda item: -item[1])[:MAX_INFLUENCES]
    norm = sum(w for _, w in strongest) or 1.0
    for g in list(v.groups):
        groups[g.group].remove([vertex])
    for name, w in strongest:
        (groups.get(name) or groups.new(name=name)).add([vertex], w / norm, "REPLACE")


def hang_from_torso(mesh_obj, armature, joints, skip=None, force=None):
    """Take hanging cloth an adopted rig gave to the arms back onto the torso; returns how many
    vertices changed.

    Tripo's rig weighted the whole of model D's cape, collar to hem, to the shoulders and upper
    arms it is draped over: it folded into the body when the arms came down, and with all its
    weight on the arms `build_cloth_chains` took it for a sleeve and gave it no chain. A piece
    counts when it passes that function's tests for hanging cloth (or is named in `force`) and
    most of what hangs below the hips is on an arm from further than ARM_REACH: a sleeve's cuff
    below the hips is by the hand. All of the piece's shoulder and arm weight then goes to the
    torso bones at its height (changing from one to the next the slower the further out it hangs,
    HANG_BLEND), so it hangs from the chest and the arms move free under it. Run
    before `build_cloth_chains`, on the rig's own weights; `skip` as there.
    """
    from .loose import _blend_skinning
    from .rig import _torso_weights
    verts = world_vertices(mesh_obj)
    n = len(verts)
    skip = np.zeros(n, dtype=bool) if skip is None else np.asarray(skip, dtype=bool)
    force = np.zeros(n, dtype=bool) if force is None else np.asarray(force, dtype=bool)
    height = float(np.ptp(verts[:, 2]))
    j = {key: np.asarray(value, dtype=float) for key, value in joints.items()}
    root_z = float(j["hips"][2])
    knee_z = float((j["left_lower_leg"][2] + j["right_lower_leg"][2]) / 2)
    leg_distance = np.min([_segment_distance(verts, j[f"{side}_{a}"], j[f"{side}_{b}"])
                           for side in ("left", "right")
                           for a, b in (("upper_leg", "lower_leg"), ("lower_leg", "foot"))], axis=0)
    arm_distance = np.min([_segment_distance(verts, j[f"{side}_{a}"], j[f"{side}_{b}"])
                           for side in ("left", "right")
                           for a, b in (("upper_arm", "lower_arm"), ("lower_arm", "hand"))], axis=0)
    bones = armature.data.bones
    arm_names = {b.name for b in bones if b.name.startswith(("Left", "Right"))
                 and not b.name.endswith(("UpperLeg", "LowerLeg", "Foot", "Toes"))}
    arm_groups = [g.index for g in mesh_obj.vertex_groups if g.name in arm_names]
    arm_share = _share(mesh_obj, n, arm_names)
    world = armature.matrix_world
    torso = [(BONE_NAMES[key], (world @ bones[BONE_NAMES[key]].head_local).z)
             for key in ("hips", "spine", "chest", "upper_chest") if BONE_NAMES[key] in bones]
    labels = pieces(mesh_obj, verts)
    changed = 0
    for label in np.unique(labels[~skip]):
        members = (labels == label) & ~skip
        below = members & (verts[:, 2] < root_z)
        if not below.any() or arm_share[members].max() <= 0:
            continue
        if not force[members].any() and (
                verts[members, 2].min() > knee_z or np.ptp(verts[members, 2]) < MIN_CLOTH_LENGTH * height
                or np.median(leg_distance[below]) < OFF_LEGS * height
                or np.mean((arm_share[below] > VERTEX_ARM_SHARE) & (arm_distance[below] > ARM_REACH * height)) < 0.5):
            continue
        for index in np.flatnonzero(members & (arm_share > 0)):
            span = HANG_BLEND * float(np.linalg.norm(verts[index, :2] - j["hips"][:2]))
            target = {}
            for z in np.linspace(verts[index, 2] - span / 2, verts[index, 2] + span / 2, HANG_SAMPLES):
                for group, weight in _torso_weights(mesh_obj.vertex_groups, torso, float(z)).items():
                    target[group] = target.get(group, 0.0) + weight / HANG_SAMPLES
            for group in arm_groups:
                mesh_obj.vertex_groups[group].remove([int(index)])
            _blend_skinning(mesh_obj, armature, int(index), target, float(arm_share[index]))
            changed += 1
    return changed


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
# What a piece lies on is read at up to this many of its vertices.
RIDER_SAMPLE = 200


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
    head_z = float(np.asarray(joints["head"], dtype=float)[2])
    on_head = bool(verts[piece, 2].min() > head_z)
    if on_head:
        # an ear or a horn standing on the head grows from its foot: the head bone runs up through
        # the crown beside it, the ear's middle was nearest to it, and a chain ran each way from there
        near = verts[piece, 2] - verts[piece, 2].min()
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
    # an ear is one thing, wide as it is: forked, a fox ear swung as a tip and a lobe
    if far.max() > APPENDAGE_FORK * reach.max() and not on_head:
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


def attach_riders(mesh_obj, carriers, skip, anchors=None):
    """Small pieces stuck onto `carriers` (anything on a spring) take the weights of the carrier under them.

    Model C's tails carry some twenty painted patches as separate pieces; left on the body bones
    they would stay while the tail swings. A piece rides when it is smaller than RIDER_SIZE, its
    median vertex lies within RIDER_GAP of a carrier, and what it lies on swings: of the vertices of
    other pieces nearest its own, most are carriers. Model A's front locks hang down over her chest:
    by nearness to them alone the shirt panels and trims under them (12 pieces, up to 9 cm across)
    went onto the locks' chains, swung over her chest with them, and the two chains pulling each had
    the tidy-up put it on the head, which turned them out of the shirt. A piece lying on one that
    rides (a button on a ribbon on a tail) rides in a later pass.

    `skip` masks pieces the user named (they keep what they were given). `anchors` masks a surface
    that doesn't swing (the face): a piece lying nearer to it than to any carrier stays where it is.
    Model A's T-pose FBX has her eyes and eyebrows as pieces under her fringe; on the fringe's chains
    they swung out of her face. Returns the number of vertices moved.
    """
    from mathutils.kdtree import KDTree

    def tree_of(indices):
        tree = KDTree(len(indices))
        for i, v in enumerate(indices):
            tree.insert(verts[v], i)
        tree.balance()
        return tree

    verts = world_vertices(mesh_obj)
    height = float(np.ptp(verts[:, 2]))
    riding = np.asarray(carriers, dtype=bool).copy()
    if not riding.any():
        return 0
    anchors = np.flatnonzero(np.asarray(anchors, dtype=bool)) if anchors is not None else ()
    anchored = tree_of(anchors) if len(anchors) else None
    labels = pieces(mesh_obj, verts)
    carried = set(np.unique(labels[riding]).tolist())
    skip = np.asarray(skip, dtype=bool)
    waiting = {}
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if label in carried or skip[members].any() or np.ptp(verts[members], axis=0).max() > RIDER_SIZE * height:
            continue
        waiting[int(label)] = members
    everything = tree_of(np.arange(len(verts)))
    groups = mesh_obj.vertex_groups
    moved = 0
    stuck = False
    while waiting:
        carriers = np.flatnonzero(riding)
        tree = tree_of(carriers)
        rides, settled = [], []
        for label, members in waiting.items():
            # what it lies on: the nearest vertex of another piece. Pieces lying on each other (two
            # beads on model B's hat tip) wait for one another for ever: then what lies under them all
            apart = (lambda i: int(labels[i]) not in waiting) if stuck else (lambda i: int(labels[i]) != label)
            sample = members[::max(1, len(members) // RIDER_SAMPLE)]
            under = [everything.find(verts[v], filter=apart)[1] for v in sample]
            under = np.array([i for i in under if i is not None], dtype=int)
            undecided = np.array([int(labels[i]) in waiting for i in under], dtype=bool)
            if len(under) and undecided.mean() > 0.5:
                continue
            settled.append(label)
            found = [tree.find(verts[v]) for v in members]
            gap = np.median([f[2] for f in found])
            if (gap > RIDER_GAP * height / 1.6 or not len(under) or riding[under[~undecided]].mean() <= 0.5
                    or anchored is not None and np.median([anchored.find(verts[v])[2] for v in members]) < gap):
                continue
            rides.append((members, found))
        for members, found in rides:
            for vertex, (_, i, _) in zip(members, found):
                source = mesh_obj.data.vertices[int(carriers[i])]
                weights = {groups[g.group].name: g.weight for g in source.groups}
                for g in list(mesh_obj.data.vertices[int(vertex)].groups):
                    groups[g.group].remove([int(vertex)])
                for name, w in weights.items():
                    groups[name].add([int(vertex)], w, "REPLACE")
                moved += 1
            riding[members] = True
        if stuck and not settled:
            break
        stuck = not settled
        for label in settled:
            del waiting[label]
    return moved


def _connected_stretches(mesh_obj, verts, members):
    """A label for each of `members` (indices): the stretch of surface it is joined to over edges
    between members, vertices sharing a place as one."""
    weld, a, b = welded_edges(mesh_obj, verts)
    inside = np.zeros(len(verts), dtype=bool)
    inside[weld[members]] = True
    keep = inside[a] & inside[b]
    a, b = a[keep], b[keep]
    labels = np.arange(len(verts))
    while True:
        updated = labels.copy()
        np.minimum.at(updated, a, labels[b])
        np.minimum.at(updated, b, labels[a])
        updated = updated[updated]
        if np.array_equal(updated, labels):
            return labels[weld[members]]
        labels = updated


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
