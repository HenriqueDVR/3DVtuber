"""Bone chains for hanging hair, so VRM spring bones can swing it.

Hair is found by its texture color. The part hanging below the head is split into
connected strands, and each strand becomes a chain of bones with its vertices
reweighted from the torso onto it. A strand wider than CHAIN_SPREAD around the
head, like hair covering the back, gets several side-by-side chains that share
weight with each other. Separate strands never share weight: blended across two
chains that swing apart, a strand would be averaged into the body.
"""

import math

import bpy
import numpy as np
from mathutils import Vector

from . import orient, texture_space
from .face import HAIR_DISTANCE
from .meshdata import world_vertices
from .rig import BONE_NAMES, make_active

# Hanging hair spanning more than this angle around the head is split into several chains.
CHAIN_SPREAD = 2 * math.pi / 8
BONES_PER_CHAIN = 4
MIN_CHAIN_VERTICES = 30
MAX_INFLUENCES = 4
# Fractions of the head radius.
HEAD_REACH = 1.3
HANGS_BELOW_CENTER = 0.35
# In front of the face (within FRONT_ANGLE of straight ahead) hair hangs from this far above the head's
# centre instead: bangs end at the brows, over the line above, and no build swung them (the demon girl's,
# seen in Warudo: her temple locks; her fringe is the face's own surface). A
# chain hung from it is marked FRONT_HUNG and springs twice as stiff (hair_physics): the synthetic
# character's temple locks, 46 degrees round and 35 cm long, drooped 11 mm standing still, 5.9 so.
FRONT_ANGLE = math.radians(60)
FRONT_HUNG = "front_hung"
FRONT_HANGS_ABOVE = 0.35
# ... and in front means no deeper than this (head radii) behind the eyes too: by angle alone the plain boy's
# spiky side tufts, level with the middle of his head, hung from there on three chains and read stretched in
# Warudo (none before), and the layer of hair under the miko's fringe ran her bangs' chains back into her head
FRONT_DEPTH = 0.4
# A chain hung from the front's line shorter than FRONT_TUFT (head radii, along it) and rooted over
# FRONT_TUFT_BEHIND (head radii) behind the eyes is a tuft at the side and stays on the head: the plain boy's,
# 0.49 to 0.75 long, 2.2 to 4.6 cm behind his eyes (his face shallow, so depth alone didn't part them), and the
# catgirl's two short side locks (0.65, 3.5 cm). Her bangs (0.7, level with her eyes), the miko's (0.86) and the
# demon girl's temple locks (0.80 to 0.89, 4.4 cm behind) swing.
FRONT_TUFT = 0.78
FRONT_TUFT_BEHIND = 0.125
MIN_CHAIN_LENGTH = 0.4
# A surface hanging past the jaw within this share of the bangs' half width is a lock, not bangs (_bangs).
LOCK_MIDDLE = 0.75
# The weights across a cut the bangs make in a surface of hair are smoothed this far from it (head radii), in this
# many rounds of averaging each vertex with its neighbours (_blend_cuts).
CUT_BLEND = 0.2
CUT_ROUNDS = 8
# Fraction of a chain's length over which weight hands over from the head bone.
ROOT_BLEND = 0.12
# And never over fewer of the strand's own edges than this: on a coarse mesh with short chains
# (model D: chains 23 cm, edges 15-27 mm at their roots, so 2.8 cm of blend) one edge took the whole
# hand-over, 277 of them in a ring round the head, and the hair below it moved off as a ledge.
ROOT_EDGES = 5
# ...up to this share of the chain.
ROOT_BLEND_MOST = 0.5
# A bit of hanging hair too small for a chain of its own and this close (m) to a strand's
# vertex moves with that vertex; further off it keeps its weights. Model C's twin-tail region
# (a box pick on a mesh fused to her body) left 168 vertices in such bits on Tripo's
# RightUpperArm and thigh weights.
STRAY_REACH = 0.02


def build_hair_bones(mesh_obj, armature, joints, features, hair_mask=None, off_arms=False):
    """Add hair bone chains under the head bone and reweight hanging hair onto them.

    Run after skinning. `hair_mask` marks the hair vertices when the mesh has separate
    hair parts; without it hair is found by texture color. Returns each chain as a list
    of bone names, root first. `off_arms` (for a rig made elsewhere) puts the hair that doesn't
    hang on the head alone: see `_on_the_head`.
    """
    verts = world_vertices(mesh_obj)
    head = _Head(verts, joints, features)
    if hair_mask is not None:
        hair = np.nonzero(hair_mask)[0]
    else:
        hair = _hair_vertices(mesh_obj, verts, features["hair_color"], head)
    in_front = (np.abs(head.angle(verts)) < FRONT_ANGLE) & (verts[:, 1] < head.center[1] - (1.0 - FRONT_DEPTH) * head.radius)
    hang_line = np.where(in_front, head.center[2] + FRONT_HANGS_ABOVE * head.radius,
                         head.center[2] - HANGS_BELOW_CENTER * head.radius)
    hanging = np.zeros(len(verts), dtype=bool)
    hanging[hair[verts[hair, 2] < hang_line[hair]]] = True
    if off_arms:
        _on_the_head(mesh_obj, hair[verts[hair, 2] >= hang_line[hair]])

    edges = np.empty(len(mesh_obj.data.edges) * 2, dtype=np.int64)
    mesh_obj.data.edges.foreach_get("vertices", edges)
    # Imported meshes split vertices along UV seams, cutting strands lengthwise into
    # unconnected strips, so vertices sharing a position are welded before grouping.
    weld = _welded(verts)
    a, b = weld[edges[0::2]], weld[edges[1::2]]
    welded_hanging = np.zeros(len(verts), dtype=bool)
    welded_hanging[weld[hanging]] = True
    within = welded_hanging[a] & welded_hanging[b]
    labels = _component_labels(len(verts), a[within], b[within])[weld]
    # bangs: hanging hair across the face (between the eyes' outer corners) above the jaw is a strand of
    # its own, as VRoid's bangs have chains of their own. One surface with the long front locks, the miko's
    # fringe was grouped with them by angle alone and lay in their root's hand-over: 60% on the head, still
    # in Warudo
    bangs = _bangs(verts, head, features, hanging & in_front, float(joints["head"][2]), labels)
    surfaces = labels
    labels = np.where(bangs, labels + labels.max() + 1, labels)

    chains, strands = [], []
    for label in np.unique(labels[hanging]):
        members = np.nonzero(hanging & (labels == label))[0]
        if len(members) < MIN_CHAIN_VERTICES:
            continue
        angles = head.angle(verts[members])
        center = _circular_mean(angles)
        offsets = _angle_between(angles, center)
        low, high = float(offsets.min()), float(offsets.max())
        count = max(1, math.ceil((high - low) / CHAIN_SPREAD))
        width = (high - low) / count
        strand_chains = []
        for piece in range(count):
            start = low + piece * width
            in_piece = (offsets >= start) & ((offsets < start + width) | (piece == count - 1))
            points = verts[members[in_piece]]
            if len(points) < MIN_CHAIN_VERTICES or np.ptp(points[:, 2]) < MIN_CHAIN_LENGTH * head.radius:
                continue
            positions = _chain_positions(points)
            if (positions[0][2] > head.center[2] - HANGS_BELOW_CENTER * head.radius
                    and positions[0][1] > head.center[1] - (1.0 - FRONT_TUFT_BEHIND) * head.radius
                    and np.linalg.norm(np.diff(positions, axis=0), axis=1).sum() < FRONT_TUFT * head.radius):
                continue
            names = [f"Hair{len(chains)}_{k}" for k in range(BONES_PER_CHAIN)]
            chain = _Chain(names, positions, center + start + width / 2, width)
            chains.append(chain)
            strand_chains.append(chain)
        if strand_chains:
            strands.append((members, strand_chains))

    covered = np.zeros(len(verts), dtype=bool)
    for members, _ in strands:
        covered[members] = True
    if off_arms:
        # hanging hair in no strand, still up on the head: the ring just under the hang line
        # (model D: 188 vertices, up to 17% on the arms); _reweight then takes those by a strand.
        # With no chain at all too: a fringe too short for one, under the line in front, kept the arms'
        _on_the_head(mesh_obj, np.flatnonzero(hanging & ~covered & (verts[:, 2] > joints["head"][2])))
    if chains:
        _add_bones(armature, chains)
        for chain in chains:
            if chain.root_z > head.center[2] - HANGS_BELOW_CENTER * head.radius:
                armature.data.bones[chain.names[0]][FRONT_HUNG] = True
        _reweight(mesh_obj, verts, head, strands, np.flatnonzero(hanging & ~covered))
        _blend_cuts(mesh_obj, verts, weld, edges, surfaces, labels, hanging, CUT_BLEND * head.radius)
    return [chain.names for chain in chains]


def _blend_cuts(mesh_obj, verts, weld, edges, surfaces, labels, hanging, reach):
    """Smooth the weights across each cut the bangs made in a surface of hair, within `reach` (m) of it.

    Weighted apart, the cut snapped from the fringe's chain to the locks' and opened as the hair swung:
    NerdGirl's fringe in shards over her face, 330 edges hung from chains sharing no bone where her build before
    the bangs rule had 29. Weighting the whole surface across all its chains closed the cut but pulled the
    fringe onto the locks (44% on its own chain): only the band by the cut is blended, the fringe keeps its own.
    """
    a, b = weld[edges[0::2]], weld[edges[1::2]]
    cut = hanging[a] & hanging[b] & (surfaces[a] == surfaces[b]) & (labels[a] != labels[b])
    if not cut.any():
        return
    from mathutils.kdtree import KDTree
    seam = np.unique(np.concatenate([a[cut], b[cut]]))
    tree = KDTree(len(seam))
    for k, vertex in enumerate(seam):
        tree.insert(verts[vertex], k)
    tree.balance()
    near = np.zeros(len(verts), dtype=bool)
    for vertex in np.flatnonzero(hanging & np.isin(surfaces, np.unique(surfaces[seam]))):
        near[vertex] = tree.find(verts[vertex])[2] <= reach
    band = np.unique(weld[near])  # welded vertices: copies along a UV seam move as one
    if not len(band):
        return
    keep = hanging[a] & hanging[b] & (a != b)
    neighbours = {int(v): set() for v in band}
    for i, j in zip(a[keep], b[keep]):
        if int(i) in neighbours:
            neighbours[int(i)].add(int(j))
        if int(j) in neighbours:
            neighbours[int(j)].add(int(i))
    groups = mesh_obj.vertex_groups
    names = {group.index: group.name for group in groups}
    involved = set(band) | {n for found in neighbours.values() for n in found}
    weights = {v: {names[g.group]: g.weight for g in mesh_obj.data.vertices[v].groups if g.weight > 0} for v in involved}
    for _ in range(CUT_ROUNDS):
        smoothed = {}
        for v in band:
            v = int(v)
            around = [weights[n] for n in neighbours[v]] or [weights[v]]
            mixed = {}
            for name, weight in weights[v].items():
                mixed[name] = mixed.get(name, 0.0) + 0.5 * weight
            for found in around:
                for name, weight in found.items():
                    mixed[name] = mixed.get(name, 0.0) + 0.5 * weight / len(around)
            smoothed[v] = mixed
        weights.update(smoothed)
    copies = {}
    for vertex, first in enumerate(weld):
        if near[vertex] or first in neighbours:
            copies.setdefault(int(first), []).append(vertex)
    for first in band:
        first = int(first)
        strongest = sorted(weights[first].items(), key=lambda item: -item[1])[:MAX_INFLUENCES]
        total = sum(weight for _, weight in strongest)
        for vertex in copies.get(first, [first]):
            for group in groups:
                group.remove([vertex])
            for name, weight in strongest:
                groups[name].add([vertex], weight / total, "REPLACE")


def _bangs(verts, head, features, front, jaw_z, surfaces):
    """Of the `front` vertices, those across the face (within the eyes' outer corners) above `jaw_z` (the head
    joint's height): what hangs lower in front is a lock over the chest. A surface that goes on down past the jaw
    across the face is such a lock, bangs and all: cut at the jaw, NerdGirl's side-swept fringe hung its upper
    part from a short chain that swung and its lower part from one rooted at her jaw that stayed, and folded to
    shards over her face between them as her head swung."""
    eyes = np.array([features["left_eye"], features["right_eye"]], dtype=float)
    span = np.ptp(eyes[:, 0]) / 2 + float(np.asarray(features.get("eye_size", (0.0, 0.0)))[0]) / 2
    across = front & (np.abs(verts[:, 0] - head.center[0]) < span)
    # down the face, not along its sides: the miko's locks hang past her chin at the eyes' outer corners
    middle = front & (np.abs(verts[:, 0] - head.center[0]) < LOCK_MIDDLE * span)
    below = np.unique(surfaces[middle & (verts[:, 2] <= jaw_z)], return_counts=True)
    locks = below[0][below[1] >= MIN_CHAIN_VERTICES]
    return across & (verts[:, 2] > jaw_z) & ~np.isin(surfaces, locks)


def _on_the_head(mesh_obj, members):
    """Hair on the head that doesn't hang moves with the head alone: no arm, and no neck.

    Tripo's rig had the top of model D's hair up to 38% on each upper arm, beside hair fully on
    the head: it tore 9 cm from it with the arms raised. Off the arms it was still 20% on the neck
    up to the crown (1812 vertices), while the chains below it hang from the head bone: a tilt
    shared by neck and head, as tracking gives, moved the two apart along the line where the chains
    start (291 edges stepping 10-30% of weight, 1.5 cm at 30 degrees), a ledge across every lock
    in Warudo. The face beside it is on the head alone too (rig.weight_head).
    """
    groups = mesh_obj.vertex_groups
    head = groups.get(BONE_NAMES["head"]) or groups.new(name=BONE_NAMES["head"])
    members = [int(index) for index in members]
    if not members:
        return
    for group in groups:
        if group.index != head.index:
            group.remove(members)
    head.add(members, 1.0, "REPLACE")


class _Head:
    def __init__(self, verts, joints, features):
        base = np.asarray(joints["head"], dtype=float)
        eyes = (np.asarray(features["left_eye"]) + np.asarray(features["right_eye"])) / 2
        top = orient.head_top(verts, base[1]) or float(verts[:, 2].max())
        self.radius = (top - base[2]) / 2
        self.center = np.array([base[0], eyes[1] + self.radius, (top + base[2]) / 2])

    def angle(self, points):
        """0 in front of the face, positive toward character left (+X), ±pi behind."""
        return np.arctan2(points[:, 0] - self.center[0], -(points[:, 1] - self.center[1]))


class _Chain:
    def __init__(self, names, positions, angle, width):
        self.names = names
        self.positions = positions
        self.angle = angle
        self.width = width
        self.root_z = positions[0][2]
        self.tip_z = positions[-1][2]


def _hair_vertices(mesh_obj, verts, hair_color, head):
    """Indices of hair-colored vertices in connected patches that reach the head.

    Strands in front of the face or shoulders often join the rest of the hair only
    through skin-colored vertices, so each patch is judged on its own: it counts as
    hair if any part of it lies near the head.
    """
    colors = texture_space.vertex_colors(mesh_obj, texture_space.base_color_image(mesh_obj))
    hair_colored = np.linalg.norm(colors - np.asarray(hair_color), axis=1) < HAIR_DISTANCE
    edges = np.empty(len(mesh_obj.data.edges) * 2, dtype=np.int64)
    mesh_obj.data.edges.foreach_get("vertices", edges)
    a, b = edges[0::2], edges[1::2]
    within_hair = hair_colored[a] & hair_colored[b]
    labels = _component_labels(len(verts), a[within_hair], b[within_hair])
    near_head = hair_colored & (np.linalg.norm(verts - head.center, axis=1) < HEAD_REACH * head.radius)
    return np.nonzero(hair_colored & np.isin(labels, labels[near_head]))[0]


def _welded(verts, tolerance=1e-5):
    """For each vertex, the index of the first vertex at the same position."""
    keys = np.round(verts / tolerance).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return first[inverse.ravel()]


def _component_labels(count, a, b):
    """Label each vertex with the smallest vertex index in its connected component."""
    labels = np.arange(count)
    while True:
        updated = labels.copy()
        np.minimum.at(updated, a, labels[b])
        np.minimum.at(updated, b, labels[a])
        updated = updated[updated]
        if np.array_equal(updated, labels):
            return labels
        labels = updated


def _chain_positions(points):
    """BONES_PER_CHAIN + 1 joint positions from the top of the strand to its tip."""
    top, bottom = points[:, 2].max(), points[:, 2].min()
    step = (top - bottom) / BONES_PER_CHAIN
    positions = []
    for k in range(BONES_PER_CHAIN + 1):
        z = top - k * step
        distance = np.abs(points[:, 2] - z)
        band = distance < step / 2
        if not band.any():
            band = distance <= distance.min()
        x, y = points[band, :2].mean(axis=0)
        positions.append(np.array([x, y, z]))
    return positions


def _add_bones(armature, chains):
    make_active(armature)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = armature.data.edit_bones
    for chain in chains:
        parent = bones[BONE_NAMES["head"]]
        for k, name in enumerate(chain.names):
            bone = bones.new(name)
            bone.head = Vector(chain.positions[k].tolist())
            bone.tail = Vector(chain.positions[k + 1].tolist())
            bone.parent = parent
            bone.use_connect = k > 0
            parent = bone
    bpy.ops.object.mode_set(mode="OBJECT")


def _reweight(mesh_obj, verts, head, strands, strays=()):
    """Move each strand's vertices onto its own chains, blending only between those chains.

    `strays` (hanging hair in no strand) within STRAY_REACH of a strand vertex copy its weights.
    """
    head_group = BONE_NAMES["head"]
    new_weights = {}
    edges = np.empty(len(mesh_obj.data.edges) * 2, dtype=np.int64)
    mesh_obj.data.edges.foreach_get("vertices", edges)
    edge_length = np.linalg.norm(verts[edges[0::2]] - verts[edges[1::2]], axis=1)
    for members, chains in strands:
        inside = np.zeros(len(verts), dtype=bool)
        inside[members] = True
        own = inside[edges[0::2]] & inside[edges[1::2]]
        step = float(np.median(edge_length[own])) if own.any() else 0.0
        for vertex, angle in zip(members, head.angle(verts[members])):
            shares = [(chain, 1.0 - abs(_angle_between(angle, chain.angle)) / chain.width) for chain in chains]
            shares = [(chain, share) for chain, share in shares if share > 0]
            if len(chains) == 1 or not shares:
                shares = [(min(chains, key=lambda chain: abs(_angle_between(angle, chain.angle))), 1.0)]
            total_share = sum(share for _, share in shares)
            weights = {}
            for chain, share in shares:
                share /= total_share
                along = min(max((chain.root_z - verts[vertex, 2]) / (chain.root_z - chain.tip_z), 0.0), 1.0)
                root_blend = min(max(ROOT_BLEND, ROOT_EDGES * step / (chain.root_z - chain.tip_z)), ROOT_BLEND_MOST)
                head_part = max(0.0, 1.0 - along / root_blend)
                weights[head_group] = weights.get(head_group, 0.0) + share * head_part
                position = along * BONES_PER_CHAIN - 0.5
                lower = int(min(max(math.floor(position), 0), BONES_PER_CHAIN - 1))
                upper = min(lower + 1, BONES_PER_CHAIN - 1)
                blend = min(max(position - lower, 0.0), 1.0)
                for index, part in ((lower, 1.0 - blend), (upper, blend)):
                    name = chain.names[index]
                    weights[name] = weights.get(name, 0.0) + share * (1.0 - head_part) * part
            strongest = sorted(weights.items(), key=lambda item: -item[1])[:MAX_INFLUENCES]
            total = sum(weight for _, weight in strongest)
            new_weights[int(vertex)] = {name: weight / total for name, weight in strongest if weight > 0}
    if len(strays) and new_weights:
        from mathutils.kdtree import KDTree
        done = np.fromiter(new_weights, dtype=np.int64)
        tree = KDTree(len(done))
        for k, vertex in enumerate(done):
            tree.insert(verts[vertex], k)
        tree.balance()
        for vertex in strays:
            _, k, gap = tree.find(verts[vertex])
            if gap <= STRAY_REACH:
                new_weights[int(vertex)] = new_weights[int(done[k])]

    groups = mesh_obj.vertex_groups
    for _members, chains in strands:
        for chain in chains:
            for name in chain.names:
                if name not in groups:
                    groups.new(name=name)
    reweighted = list(new_weights)
    for group in groups:
        group.remove(reweighted)
    for vertex, weights in new_weights.items():
        for name, weight in weights.items():
            groups[name].add([vertex], weight, "REPLACE")


def _circular_mean(angles):
    return math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))


def _angle_between(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi
