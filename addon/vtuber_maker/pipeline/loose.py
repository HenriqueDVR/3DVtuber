"""Mesh hygiene before posing: nothing may be torn between two bones that move apart.

Two faults do the same damage. A small separate piece (a buckle, a strap, a panel) can end up
half on one limb and half on another, and a stray vertex deep inside a big piece can be given a
bone from the other side of the body -- on model A a few vertices of the skirt were driven by the
forearm, and a 3 mm edge drew out to 41 cm the moment the arms went out to the T-pose. Both look
right in the pose the character was modelled in and only show when something moves.

Small loose pieces ride one bone whole, instead of being torn between two.

A character like model A comes in 163 separate pieces: buckles, clips, straps, panels. Bone heat
weights each vertex on its own, so a small panel lying between the arm and the hip ends up half
on the forearm and half on the body. Nothing shows in the pose it was modelled in, but the
moment the arm moves the piece stretches between the two: on model A two panels drew out into 20 cm
slivers hanging off the wrists as soon as the arms went out to the T-pose, and they followed the
hand around while tracking.

A piece small enough to be one prop can't bend anyway, so it is given the weights of the piece as
a whole -- the average over its vertices -- and from then on it moves rigidly with whatever holds
most of it. Big pieces (the body, the hair, a coat) are left alone: those are meant to deform.
"""

import re

import bmesh
import numpy as np

from .meshdata import pieces, welded_edges, world_vertices

# A piece counts as a prop when it holds no more than this share of the character's vertices.
PROP_SHARE = 0.02
# A bone holding at least this share of a piece counts as one of the bones moving it.
HOLDS = 0.12
# A patch of surface this small, held by a bone none of the surface around it uses, is a stray:
# bone heat picked the wrong bone for it (in vertices).
ISLAND = 40
# Passes of island clearing; each one takes the outermost strays first.
ISLAND_PASSES = 3
# Limb joints apps turn far. Two bones on one chain with one of these between them that holds none
# of a piece move apart as surely as two limbs do (see _torn).
LIMB_JOINTS = {side + name for side in ("Left", "Right")
               for name in ("UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot")}
# The tidy-up is measured in these poses (see worst_stretch): arms pointed about world axes as
# check_avatar.py points them, plus a head turn and a leg lift. A pose whose worst edge grows by
# more than TEARS (m) across the tidy-up tore something it glued, and the tidy-up is undone
# (see scripts/fix_loose_pieces.py): joining glued model A's cut sleeves to its skirt, 3.5 -> 77.6 cm
# with the arms raised. Joining shells at the armpit stretches too, by design: model B's 7.0 -> 13.6 cm.
GUARD_ARMS = {"arms T-pose": (1, 0, 0), "arms forward": (0.3, -1, -0.3), "arms raised": (0.5, 0, 1)}
TEARS = 0.2


def synchronize_shoulder_seams(mesh_obj, armature_obj, reach=0.4):
    """Give coincident shoulder vertices identical skinning, without altering topology.

    Sleeve splitting and UV seams can leave two copies of one surface point with different
    weights. Those copies part when the arm rises. Restrict this repair to the shoulder joint
    neighbourhood (a fraction of the arm's length), not contact between wrists and clothing.
    The original vertices, UVs, faces and shape keys are kept byte-for-byte.
    """
    verts = world_vertices(mesh_obj)
    selected = np.zeros(len(verts), dtype=bool)
    smooth_region = selected.copy()
    for side in ("Left", "Right"):
        upper = armature_obj.data.bones.get(side + "UpperArm")
        lower = armature_obj.data.bones.get(side + "LowerArm")
        if upper is None or lower is None:
            continue
        head = np.array(armature_obj.matrix_world @ upper.head_local)
        elbow = np.array(armature_obj.matrix_world @ lower.head_local)
        wrist = np.array(armature_obj.matrix_world @ lower.tail_local)
        length = np.linalg.norm(elbow - head) + np.linalg.norm(wrist - elbow)
        distance = np.linalg.norm(verts - head, axis=1)
        selected |= distance < reach * length
        # Leave room to blend beyond the repaired seam instead of putting a hard weight
        # transition exactly at its outer radius. This still stops well before the wrist.
        smooth_region |= distance < 1.5 * reach * length
    indices = np.flatnonzero(selected)
    keys = np.round(verts[indices] / 1e-5).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    ordered = indices[np.argsort(inverse, kind="stable")]
    labels = np.sort(inverse.ravel())
    runs = [part for members in np.split(ordered, np.flatnonzero(np.diff(labels)) + 1)
            for part in _not_far_apart(mesh_obj, armature_obj, members)]
    deform = {g.index for g in mesh_obj.vertex_groups if g.name in armature_obj.data.bones
              and armature_obj.data.bones[g.name].use_deform}
    changed = 0
    for members in runs:
        if len(members) < 2:
            continue
        rows = [{g.group: g.weight for g in mesh_obj.data.vertices[int(i)].groups
                 if g.group in deform and g.weight > 0} for i in members]
        groups = set().union(*rows)
        if not groups:
            continue
        average = {g: sum(row.get(g, 0) for row in rows) / len(rows) for g in groups}
        total = sum(average.values())
        average = {g: value / total for g, value in average.items()}
        if all(all(abs(row.get(g, 0) - average[g]) < 1e-6 for g in groups) for row in rows):
            continue
        for group, weight in average.items():
            mesh_obj.vertex_groups[group].add(members.tolist(), weight, "REPLACE")
        changed += len(members)
    if changed:
        # A seam-only correction creates a weight cliff one edge farther into the sleeve.
        # Spread the correction through the shoulder surface, keeping head/hair/forearm
        # weights and vertices outside this neighbourhood unchanged. The existing smoother
        # treats coincident UV vertices alike, so synchronized seams remain synchronized.
        from .rig import _smooth_body_weights
        _smooth_body_weights(mesh_obj, ~smooth_region)
    return changed


def _not_far_apart(mesh_obj, armature_obj, members):
    """`members` (copies of one point) in groups whose leading bones are under rig.FAR_BONES steps
    apart. The loose step cuts a twin-tail free of the arm it lies on (split_torn_seams); averaged
    back together at the shoulder, model C's twin-tail hung 35% on RightUpperArm. Neck and
    shoulder at a collar (2 steps) still close."""
    from .rig import FAR_BONES, bone_steps
    names = {g.index: g.name for g in mesh_obj.vertex_groups if g.name in armature_obj.data.bones}
    leaders = []
    for index in members:
        weights = [(g.weight, names[g.group]) for g in mesh_obj.data.vertices[int(index)].groups if g.group in names]
        leaders.append(max(weights)[1] if weights else None)
    bones = sorted({name for name in leaders if name is not None})
    steps = bone_steps(armature_obj, bones) if len(bones) > 1 else np.zeros((1, 1))
    at = {name: k for k, name in enumerate(bones)}
    groups = []
    for index, name in zip(members, leaders):
        home = next((g for g in groups if name is None or g[0] is None
                     or steps[at[name], at[g[0]]] < FAR_BONES), None)
        if home is None:
            groups.append((name, [index]))
        else:
            home[1].append(index)
    return [np.array(g[1]) for g in groups]


def join_touching_surfaces(mesh_obj, armature_obj, distance=0.0002, verts=None):
    """Merge vertices sitting in the same place, where the surfaces belong together.

    These models are built as separate shells resting against each other -- a sleeve against a
    torso, a collar against a neck. Nothing holds them together, so raising an arm opens a hole
    at the armpit. Merged, the skin stretches instead. UVs are stored per face corner, so the
    texture and its seams survive the merge, and the shape keys come through with the mesh.

    Only surfaces whose bones don't move apart are joined (see _together). In the pose a
    character is modelled in, things touch that have no business being joined -- a hand resting
    against a thigh -- and welding those glues the glove to the skirt: the first time the arm
    moves, a 3 mm edge draws out into a 40 cm ribbon.

    Nor are vertices of one piece that a shape key moves apart: the mouth patch and the lid covers
    lie folded on a line until their keys open them. Joined, model A (T-pose)'s mouth patch went from 99
    vertices to 36 and opened as a jagged block with skin showing through. Two pieces touching
    still join where a key moves one of them (the face against the shells round it).
    """
    verts = world_vertices(mesh_obj) if verts is None else verts
    representative = _together(mesh_obj, armature_obj, verts, distance)
    keys = mesh_obj.data.shape_keys
    edges = np.empty(len(mesh_obj.data.edges) * 2, dtype=np.int64)
    mesh_obj.data.edges.foreach_get("vertices", edges)
    piece = np.arange(len(verts))
    while keys:
        grown = piece.copy()
        np.minimum.at(grown, edges[0::2], piece[edges[1::2]])
        np.minimum.at(grown, edges[1::2], piece[edges[0::2]])
        grown = grown[grown]
        if np.array_equal(grown, piece):
            break
        piece = grown
    for key in keys.key_blocks if keys else ():
        if key == keys.reference_key:
            continue
        moved = np.empty(len(verts) * 3)
        key.data.foreach_get("co", moved)
        moved = moved.reshape(-1, 3)
        rest = np.empty(len(verts) * 3)
        keys.reference_key.data.foreach_get("co", rest)
        moved -= rest.reshape(-1, 3)
        apart = (np.abs(moved - moved[representative]).max(axis=1) > distance) & (piece == piece[representative])
        representative = np.where(apart, np.arange(len(verts)), representative)

    mesh = bmesh.new()
    mesh.from_mesh(mesh_obj.data)
    mesh.verts.ensure_lookup_table()
    targets = {mesh.verts[index]: mesh.verts[rep] for index, rep in enumerate(representative) if index != rep}
    if targets:
        bmesh.ops.weld_verts(mesh, targetmap=targets)
        mesh.to_mesh(mesh_obj.data)
        mesh_obj.data.update()
    mesh.free()
    return len(mesh_obj.data.vertices)


def _leading_bones(mesh_obj, count):
    """The vertex group holding most of each vertex, or -1 where nothing does."""
    best = np.zeros(count)
    leader = np.full(count, -1)
    for index, vertex in enumerate(mesh_obj.data.vertices):
        for entry in vertex.groups:
            if entry.weight > best[index]:
                best[index], leader[index] = entry.weight, entry.group
    return leader


def _together(mesh_obj, armature_obj, verts, distance):
    """Each vertex's representative: the first vertex in its spot (`distance` wide) whose leading
    bone doesn't move apart from its own (see _torn), or itself.

    A spot on a cut holds a copy of the vertex for each side -- the sleeve cut free of the skirt
    it rests on, each side with its own UV-seam twins -- and only copies on one side belong
    together. Joining them all to one copy glued model A's cut sleeves back onto the skirt.
    """
    leader = _leading_bones(mesh_obj, len(verts))
    names = {group.index: group.name for group in mesh_obj.vertex_groups}
    keys = np.round(verts / max(distance, 1e-9)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    representative = np.arange(len(verts))
    order = np.argsort(inverse, kind="stable")
    apart = {}

    def torn(first, second):
        if first < 0 or second < 0 or first == second:
            return False
        pair = (min(first, second), max(first, second))
        if pair not in apart:
            apart[pair] = _torn(armature_obj, [names[pair[0]], names[pair[1]]])
        return apart[pair]

    for members in np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1):
        reps = []
        for index in members:
            rep = next((r for r in reps if not torn(leader[index], leader[r])), None)
            if rep is None:
                reps.append(index)
            else:
                representative[index] = rep
    return representative


def _surface(mesh_obj, armature_obj, verts):
    """(weld, a, b) as meshdata.welded_edges, but a cut's two sides stay apart (see _together)."""
    weld = _together(mesh_obj, armature_obj, verts, 1e-5)
    edges = np.empty(len(mesh_obj.data.edges) * 2, dtype=np.int64)
    mesh_obj.data.edges.foreach_get("vertices", edges)
    return weld, weld[edges[0::2]], weld[edges[1::2]]


def split_torn_seams(mesh_obj, armature_obj, verts=None):
    """Cut the surface where it joins two limbs that move apart. Returns how many edges were cut.

    Image-to-3D generators fuse whatever touches in the pose they build: on model A the glove is one
    surface with the skirt it rests against. Weights can't save that -- one side has to follow
    the arm and the other the hips -- so a 2 cm edge across the join draws out to 38 cm as soon
    as the arms go out. Cutting the join lets each side go its own way; the slit it leaves sits
    where two pieces of clothing already meet.

    The cut runs between faces, each given the bone leading most of its corners: splitting the
    edges that cross the join (as this once did) opens slits beside them but leaves every
    crossing edge joining its two sides, so nothing came apart (172 hair-to-arm edges survived
    on model C's twin-tail). A vertex left holding faces only on the other side takes its weights
    from its neighbours there, so no corner of a cut face is still pulled by the other limb.

    Never between two hair chains or two cloth chains: side by side over one head of hair or one
    cape they are no chain of each other, and the cut slit model D's hair and cape along every
    chain boundary (476 of its 628 cut edges), open whenever they swung. Springs beside each other
    move together, and what is left between them stretches, as cloth does. Appendage chains are
    still cut apart: a fork's two ends (model C's pair of tails) swing their own ways.
    Nor between two fingers of one hand: the palm between their knuckles was slit on model D
    (60 edges, open in Warudo as the fingers moved).
    """
    leader = _leading_bones(mesh_obj, len(mesh_obj.data.vertices))
    names = {group.index: group.name for group in mesh_obj.vertex_groups}
    torn = {}

    def strand(name):
        match = _STRAND_CHAIN.match(name) or _FINGER.match(name)
        return match and match.group(1)

    def apart(first, second):
        if first < 0 or second < 0 or first == second:
            return False
        if strand(names[first]) and strand(names[first]) == strand(names[second]):
            return False
        pair = (min(first, second), max(first, second))
        if pair not in torn:
            torn[pair] = _torn(armature_obj, [names[pair[0]], names[pair[1]]])
        return torn[pair]

    mesh = bmesh.new()
    mesh.from_mesh(mesh_obj.data)
    side_layer = mesh.faces.layers.int.new("torn_seam_side")
    for face in mesh.faces:
        leaders = [leader[vertex.index] for vertex in face.verts if leader[vertex.index] >= 0]
        face[side_layer] = max(set(leaders), key=leaders.count) if leaders else -1
    cut = [edge for edge in mesh.edges if len(edge.link_faces) == 2
           and apart(edge.link_faces[0][side_layer], edge.link_faces[1][side_layer])]
    if not cut:
        mesh.free()
        return 0
    bmesh.ops.split_edges(mesh, edges=cut)
    mesh.verts.index_update()
    sides = [[face[side_layer] for face in vertex.link_faces] for vertex in mesh.verts]
    neighbours = [[edge.other_vert(vertex).index for edge in vertex.link_edges] for vertex in mesh.verts]
    mesh.faces.layers.int.remove(side_layer)
    mesh.to_mesh(mesh_obj.data)
    mesh_obj.data.update()
    mesh.free()

    count = len(mesh_obj.data.vertices)
    leader = _leading_bones(mesh_obj, count)
    weights = _weight_table(mesh_obj, count)
    moved = {}
    for index in range(count):
        held = [side for side in sides[index] if side >= 0]
        if not held or not all(apart(leader[index], side) for side in held):
            continue
        side = max(set(held), key=held.count)
        near = [other for other in neighbours[index] if not apart(leader[other], side)]
        moved[index] = weights[:, near].mean(axis=1) if near else np.eye(len(weights))[side]
    for index, row in moved.items():
        for group in mesh_obj.vertex_groups:
            if row[group.index] > 1e-4:
                group.add([index], float(row[group.index]), "REPLACE")
            elif weights[group.index, index] > 0:
                group.remove([index])
    return len(cut)


# The side-by-side chains hair.py and cloth.py lay over one head of hair or one cape.
_STRAND_CHAIN = re.compile(r"(Hair|Cloth)\d+_\d+$")
# The fingers of one hand: Left/Right, then the finger and its segment.
_FINGER = re.compile(r"(Left|Right)(?:Thumb|Index|Middle|Ring|Little)[A-Z][a-z]+$")


def rigidify_props(mesh_obj, armature_obj, verts=None):
    """Give every small piece torn across a joint one set of weights. Returns what was changed.

    A piece shared by a bone and its own parent bends the way it is meant to (a skirt panel on
    the hips and the spine, a strand on its spring chain) and is left alone. A piece shared by
    bones on different limbs -- a panel on the forearm and on the hips -- has no pose in which
    both are right, so it is put on whichever holds most of it.
    """
    verts = world_vertices(mesh_obj) if verts is None else verts
    labels = pieces(mesh_obj, verts)
    weights = _weight_table(mesh_obj, len(verts))
    changed = []
    for label in np.unique(labels):
        members = np.nonzero(labels == label)[0]
        if len(members) > PROP_SHARE * len(verts):
            continue
        size = float((verts[members].max(axis=0) - verts[members].min(axis=0)).max())
        average = weights[:, members].mean(axis=1)
        total = average.sum()
        if total <= 1e-6:
            continue
        average /= total
        holders = [mesh_obj.vertex_groups[index].name for index in np.nonzero(average >= HOLDS)[0]]
        if not _torn(armature_obj, holders):
            continue
        # one bone for the whole piece: the one every holder hangs off, so a fin reaching from
        # one arm to the other rides the chest rather than being pulled in two directions. On one
        # chain that is the torso (a cuff trim partly on the hips it rests on), which would leave
        # it behind when the arm lifts: there the bone holding most of it takes it.
        most = mesh_obj.vertex_groups[int(np.argmax(average))].name
        onto = most if _one_chain(armature_obj, holders) else _shared_bone(armature_obj, holders) or most
        _set_one(mesh_obj, members, onto)
        changed.append({
            "vertices": int(len(members)),
            "size_cm": round(size * 100, 1),
            "bone": onto,
            "was_torn_with": [name for name in holders if name != onto],
        })
    return changed


def _shared_bone(armature_obj, holders):
    """The nearest bone every holder descends from (itself counting), or None."""
    lines = []
    for name in holders:
        bone = armature_obj.data.bones.get(name)
        if bone is None:
            return None
        line = []
        while bone is not None:
            line.append(bone.name)
            bone = bone.parent
        lines.append(line)
    for candidate in lines[0]:
        if all(candidate in line for line in lines[1:]):
            return candidate
    return None


def _torn(armature_obj, holders):
    """Whether these bones pull a piece apart.

    Bones on one chain -- one of them with every other among its ancestors or descendants --
    bend a piece the way it is meant to (a skirt panel on the hips and the spine, a strand on its
    spring chain), unless a limb joint between two of them holds none of it (LIMB_JOINTS). The
    hips are the forearm's ancestor, but with the shoulder and elbow turning in between, a glove
    resting on the skirt moves 40 cm away from it when the arm lifts. Counting that pair as one
    chain welded model A's cut sleeves back onto the skirt.
    """
    bones = [armature_obj.data.bones.get(name) for name in holders]
    bones = [bone for bone in bones if bone is not None]
    if len(bones) < 2:
        return False
    names = {bone.name for bone in bones}
    for bone in bones:
        others = [other for other in bones if other != bone]
        if not all(_descends(bone, other) or _descends(other, bone) for other in others):
            continue
        if not any(_skips_joint(bone, other, names) if _descends(bone, other) else _skips_joint(other, bone, names)
                   for other in others):
            return False
    return True


def _one_chain(armature_obj, holders):
    """Whether one of these bones has every other among its ancestors or descendants."""
    bones = [bone for bone in (armature_obj.data.bones.get(name) for name in holders) if bone is not None]
    return any(all(_descends(bone, other) or _descends(other, bone) for other in bones if other != bone)
               for bone in bones)


def _skips_joint(bone, ancestor, holders):
    """Whether a limb joint from `bone` up to (not including) `ancestor` is missing from `holders`."""
    while bone is not None and bone != ancestor:
        if bone.name in LIMB_JOINTS and bone.name not in holders:
            return True
        bone = bone.parent
    return False


def _descends(bone, ancestor):
    parent = bone.parent
    while parent is not None:
        if parent == ancestor:
            return True
        parent = parent.parent
    return False


def _weight_table(mesh_obj, count):
    """Every vertex group's weight for every vertex, as one array."""
    weights = np.zeros((len(mesh_obj.vertex_groups), count))
    for index, vertex in enumerate(mesh_obj.data.vertices):
        for entry in vertex.groups:
            weights[entry.group, index] = entry.weight
    return weights


def _set_one(mesh_obj, members, bone_name):
    """Put the whole piece on one bone."""
    indices = [int(index) for index in members]
    for group in mesh_obj.vertex_groups:
        if group.name == bone_name:
            group.add(indices, 1.0, "REPLACE")
        else:
            group.remove(indices)


def fix_weight_islands(mesh_obj, armature_obj, verts=None):
    """Hand each stray patch of surface over to the bone the surface around it uses.

    Bone heat decides each vertex on its own, so it can hand a few vertices in the middle of the
    skirt to a forearm. Nothing shows until the arm moves and the edge between them draws out
    into a spike. A patch smaller than `ISLAND` whose bone is used by none of its neighbours is
    given the weights of the surface around it instead.

    A cut's two sides share their spots but not their surface (see _surface): walked across it,
    a glove and the strip of thigh it was cut from read as patches on each other, and one took
    the other's weights.
    """
    verts = world_vertices(mesh_obj) if verts is None else verts
    weld, edge_a, edge_b = _surface(mesh_obj, armature_obj, verts)
    weights = _weight_table(mesh_obj, len(verts))
    fixed, touched = [], []
    for _ in range(ISLAND_PASSES):
        leader = np.argmax(weights, axis=0)
        leader[weights.max(axis=0) <= 0.0] = -1
        same = leader[weld[edge_a]] == leader[weld[edge_b]]
        islands = _patches_of(leader, weld, edge_a[same], edge_b[same], len(verts))
        changed = False
        for members in islands:
            if len(members) > ISLAND:
                continue
            outside = _neighbours_outside(members, weld, edge_a, edge_b, len(verts))
            if not len(outside):
                continue  # a piece of its own: it is meant to move alone
            borrowed = weights[:, outside].mean(axis=1)
            if borrowed.sum() <= 1e-6:
                continue
            was = mesh_obj.vertex_groups[int(leader[members[0]])].name
            now = mesh_obj.vertex_groups[int(np.argmax(borrowed))].name
            # neighbouring bones of one chain share a patch of surface on purpose -- a hair strand
            # hands over from bone to bone down its length. Only a bone from somewhere else
            # entirely, which will one day move the other way, is a mistake worth undoing.
            if not _torn(armature_obj, [was, now]):
                continue
            borrowed = borrowed * (weights[:, members].sum(axis=0).mean() / borrowed.sum())
            weights[:, members] = borrowed[:, None]
            touched.extend(int(index) for index in members)
            fixed.append({"vertices": int(len(members)), "was": was, "now": now})
            changed = True
        if not changed:
            break
    if fixed:
        _write_weights(mesh_obj, weights, touched)
    return fixed


def _patches_of(leader, weld, edge_a, edge_b, count):
    """Connected runs of vertices that share a leading bone."""
    label = np.arange(count)
    while True:
        updated = label.copy()
        np.minimum.at(updated, edge_a, label[edge_b])
        np.minimum.at(updated, edge_b, label[edge_a])
        updated = updated[updated]
        if np.array_equal(updated, label):
            break
        label = updated
    label = label[weld]
    order = np.argsort(label, kind="stable")
    splits = np.nonzero(np.diff(label[order]))[0] + 1
    return [group for group in np.split(order, splits) if leader[group[0]] >= 0]


def _neighbours_outside(members, weld, edge_a, edge_b, count):
    """The vertices joined to this patch that aren't in it."""
    inside = np.zeros(count, dtype=bool)
    inside[members] = True
    inside = inside[weld] | inside
    crossing = inside[edge_a] != inside[edge_b]
    touching = np.concatenate([edge_a[crossing], edge_b[crossing]])
    return np.unique(touching[~inside[touching]])


def restore_skirt_weights(mesh_obj, armature_obj, garment):
    """Return the medial, below-elbow part of a known garment to the torso.

    Use on the modelled A-pose, BEFORE baking a T-pose. `garment` must identify the
    original garment surface, including copies made by sleeve cuts, not a spatial
    selection of everything near a wrist (which would include the hand itself).
    A cut can give those copies different weights; using the same position-based
    torso blend for both closes the false skirt seam without welding the hand.
    The lateral sleeve and upper arm are left alone, and so is the inner half of the
    forearm sleeve (see `_sleeve_tube`). No geometry or keys change.
    """
    from . import rig
    verts = world_vertices(mesh_obj)
    garment = np.asarray(garment, dtype=bool)
    strength = np.zeros(len(verts))
    bones, matrix = armature_obj.data.bones, armature_obj.matrix_world
    sleeve = _sleeve_tube(mesh_obj, armature_obj, verts, garment)
    for side, sign in (("Left", 1), ("Right", -1)):
        lower = bones[side + "LowerArm"]
        elbow = np.array(matrix @ lower.head_local)
        wrist = np.array(matrix @ lower.tail_local)
        shoulder = np.array(matrix @ bones[side + "UpperArm"].head_local)
        length = np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow)
        if elbow[2] - wrist[2] < .1 * length:
            raise ValueError("skirt repair requires the modelled arms-down pose, not a baked T-pose")
        along = np.clip((elbow[2] - verts[:, 2]) / (elbow[2] - wrist[2]), 0, 1)
        axis_x = elbow[0] + along * (wrist[0] - elbow[0])
        inward = sign * (axis_x - verts[:, 0]) / length
        medial = rig._smoothstep((inward - .015) / .055)
        below = rig._smoothstep((elbow[2] + .02 * length - verts[:, 2]) / (.10 * length))
        strength = np.maximum(strength, medial * below * (sign * verts[:, 0] >= 0))
    strength[sleeve] = 0
    torso = [(name, (matrix @ bones[name].head_local).z) for name in ("Hips", "Spine", "Chest", "UpperChest")]
    changed = 0
    for index in np.flatnonzero(garment & (strength > 0)):
        target = rig._torso_weights(mesh_obj.vertex_groups, torso, verts[index, 2])
        changed += _blend_skinning(mesh_obj, armature_obj, int(index), target, float(strength[index]))
    return changed


def _sleeve_tube(mesh_obj, armature_obj, verts, garment):
    """Garment vertices on the arm's sleeve tube, reached over the sleeve surface itself.

    The medial below-elbow box of `restore_skirt_weights` also holds the inner half of the
    forearm sleeve: on model A about 180 vertices per arm, 3.4-4.2 cm from the forearm axis, which
    were handed to the torso and left at the waist when the arm rose. Position cannot tell them
    from the skirt they rest against; connection can. Starting from sleeve surface outside the
    box (lateral or above the elbow), walk edges through vertices within the sleeve radius of
    an arm bone. Coincident vertices are joined across UV seams, but not across a skirt/sleeve
    contact whose two sides already ride limbs that move apart.
    """
    bones, matrix = armature_obj.data.bones, armature_obj.matrix_world
    weld, edge_a, edge_b = _surface(mesh_obj, armature_obj, verts)
    reached = np.zeros(len(verts), dtype=bool)
    for side, sign in (("Left", 1), ("Right", -1)):
        points = [np.array(matrix @ bones[side + name].head_local) for name in ("UpperArm", "LowerArm")]
        points.append(np.array(matrix @ bones[side + "LowerArm"].tail_local))
        (upper, upper_t), (fore, fore_t) = (_segment_distance(verts, a, b) for a, b in zip(points, points[1:]))
        this_side = garment & (sign * verts[:, 0] >= 0)
        # Sleeve radius from the lateral forearm sleeve, which no skirt reaches.
        axis_x = points[1][0] + fore_t * (points[2][0] - points[1][0])
        lateral = this_side & (sign * (verts[:, 0] - axis_x) > 0) & (fore_t > .1) & (fore_t < .9)
        length = np.linalg.norm(points[1] - points[0]) + np.linalg.norm(points[2] - points[1])
        lateral &= fore < .15 * length
        if not lateral.any():
            continue
        tube = this_side & (np.minimum(upper, fore) < 1.3 * np.median(fore[lateral]))
        # Seeds are sleeve surface the skirt box never touches: lateral of the forearm or
        # above the elbow.
        side_reached = np.zeros(len(verts), dtype=bool)
        side_reached[weld[tube & (lateral | (verts[:, 2] > points[1][2]))]] = True
        inside = np.zeros(len(verts), dtype=bool)
        inside[weld[tube]] = True
        while True:
            grow = inside[edge_a] & inside[edge_b] & (side_reached[edge_a] != side_reached[edge_b])
            if not grow.any():
                break
            side_reached[edge_a[grow]] = side_reached[edge_b[grow]] = True
        reached |= tube & side_reached[weld]
    return reached


def _segment_distance(points, a, b):
    """Distance from each point to segment a-b, and the position along it (0 at a)."""
    ab = b - a
    t = np.clip(((points - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1), t


def separate_skirt_contact(mesh_obj, armature_obj, garment):
    """Separate the corrected sleeve/body boundary below the elbows; return vertex origins.

    Weight blending cannot fix a surface fused between wrist and skirt. Reuse the
    skinning pipeline's interpolated cut, limited to the explicitly selected garment.
    The cut preserves face-corner UVs and shape-key layers and never welds a hand.
    """
    from . import rig
    verts = world_vertices(mesh_obj)
    below = np.zeros(len(verts), dtype=bool)
    for side, sign in (("Left", 1), ("Right", -1)):
        elbow = armature_obj.matrix_world @ armature_obj.data.bones[side + "LowerArm"].head_local
        below |= (sign * verts[:, 0] > 0) & (verts[:, 2] < elbow.z)
    return rig._split_sleeves(mesh_obj, armature_obj, np.asarray(garment, dtype=bool) & below)


def follow_shoulder_cloth(mesh_obj, armature_obj, garment):
    """Blend a selected mantle/upper cape onto upper arms, retaining its collar and hem.

    This is opt-in: a garment mask is required because hair and shoulder accessories
    can occupy the same space. The spatial ramps avoid a hard boundary in the cloth;
    lower cape spring weights, fingers, and unselected surfaces remain untouched.
    Run on the modelled A-pose, not on already displaced cloth in a baked T-pose.
    """
    from . import rig
    verts = world_vertices(mesh_obj)
    garment = np.asarray(garment, dtype=bool)
    changed = 0
    for side, sign in (("Left", 1), ("Right", -1)):
        bones, matrix = armature_obj.data.bones, armature_obj.matrix_world
        shoulder = np.array(matrix @ bones[side + "UpperArm"].head_local)
        elbow = np.array(matrix @ bones[side + "LowerArm"].head_local)
        wrist = np.array(matrix @ bones[side + "LowerArm"].tail_local)
        length = np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow)
        lateral = rig._smoothstep((sign * (verts[:, 0] - shoulder[0]) / length + .03) / .22)
        above = rig._smoothstep((verts[:, 2] - elbow[2]) / (.25 * length))
        distance = np.linalg.norm(verts - shoulder, axis=1) / length
        near = rig._smoothstep((.65 - distance) / .20)
        strength = lateral * above * near
        name = side + "UpperArm"
        group = mesh_obj.vertex_groups.get(name) or mesh_obj.vertex_groups.new(name=name)
        for index in np.flatnonzero(garment & (strength > 0)):
            changed += _blend_skinning(mesh_obj, armature_obj, int(index), {group.index: 1.0}, float(strength[index]))
    return changed


def attach_clothing_detail(mesh_obj, armature_obj, members, anchors):
    """Copy skinning from the nearest vertex of an explicitly identified clothing anchor.

    Explicit anchors avoid choosing the skirt just because a cuff touches it at rest.
    """
    from mathutils.kdtree import KDTree
    members, anchors = list(members), list(anchors)
    if not anchors or set(members) & set(anchors):
        raise ValueError("detail needs a non-empty, separate anchor surface")
    verts = world_vertices(mesh_obj)
    tree = KDTree(len(anchors))
    for i in anchors:
        tree.insert(verts[i], int(i))
    tree.balance()
    changed = 0
    for index in members:
        _, near, _ = tree.find(verts[index])
        target = {g.group: g.weight for g in mesh_obj.data.vertices[near].groups
                  if (bone := armature_obj.data.bones.get(mesh_obj.vertex_groups[g.group].name)) is not None
                  and bone.use_deform and g.weight > 0}
        if not target:
            raise ValueError("clothing anchor has no deform weights")
        changed += _blend_skinning(mesh_obj, armature_obj, int(index), target, 1.0)
    return changed


def transfer_clothing_weights(target, armature_obj, donor, target_indices, donor_indices, tolerance=1e-4):
    """Copy deform skinning vertex-for-vertex from an earlier build of the same surface.

    For restoring what a repair got wrong from the file before it, rather than recomputing
    from position. Every pair must sit at the same place (within `tolerance`, metres) and every
    donor vertex must carry deform weights; otherwise ValueError, raised before any weight is
    edited. Unrelated vertex groups on the target are kept.
    """
    target_indices, donor_indices = np.asarray(target_indices), np.asarray(donor_indices)
    if not len(target_indices) or len(target_indices) != len(donor_indices):
        raise ValueError("need matching, non-empty target and donor vertex lists")
    gap = np.linalg.norm(world_vertices(target)[target_indices] - world_vertices(donor)[donor_indices], axis=1)
    if gap.max() > tolerance:
        raise ValueError(f"donor does not match target geometry (up to {gap.max() * 1000:.1f} mm apart)")
    rows = []
    for index in donor_indices:
        row = {donor.vertex_groups[g.group].name: g.weight for g in donor.data.vertices[int(index)].groups
               if (bone := armature_obj.data.bones.get(donor.vertex_groups[g.group].name)) is not None
               and bone.use_deform and g.weight > 0}
        if not row:
            raise ValueError(f"donor vertex {int(index)} has no deform weights")
        rows.append(row)
    changed = 0
    for index, row in zip(target_indices, rows):
        groups = {(target.vertex_groups.get(name) or target.vertex_groups.new(name=name)).index: weight
                  for name, weight in row.items()}
        changed += _blend_skinning(target, armature_obj, int(index), groups, 1.0)
    return changed


def follow_bone(mesh_obj, armature_obj, members, bone_name):
    """Put a separate clothing piece wholly on one bone, keeping non-deform groups.

    For a piece that belongs to a limb but was skinned to what it hangs beside: model B's wide
    coat sleeves hang under the upper arms, entirely on Spine/Chest, and stayed at the ribs
    when the arms rose.
    """
    group = mesh_obj.vertex_groups.get(bone_name) or mesh_obj.vertex_groups.new(name=bone_name)
    return sum(_blend_skinning(mesh_obj, armature_obj, int(index), {group.index: 1.0}, 1.0) for index in members)


def worst_stretch(mesh_objs, armature_obj):
    """{pose: the most any edge of `mesh_objs` grows (m)} in the GUARD_ARMS poses, a head turn and
    a leg lift. The pose is put back afterwards."""
    import bpy
    from mathutils import Matrix, Vector
    bones = armature_obj.pose.bones
    rest = [world_vertices(obj) for obj in mesh_objs]
    edges = []
    for obj in mesh_objs:
        pairs = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
        obj.data.edges.foreach_get("vertices", pairs)
        edges.append(pairs.reshape(-1, 2))

    def turn(name, axis, angle):
        # about a world axis through the bone's head: a bone's own axes twist a limb instead
        bone = bones[name]
        head = bone.head.copy()
        bone.matrix = Matrix.Translation(head) @ Matrix.Rotation(angle, 4, axis) @ Matrix.Translation(-head) @ bone.matrix
        bpy.context.view_layer.update()

    def arms(direction):
        for side, mirror in (("Left", 1.0), ("Right", -1.0)):
            if f"{side}UpperArm" in bones and f"{side}Hand" in bones:
                now = (bones[f"{side}Hand"].head - bones[f"{side}UpperArm"].head).normalized()
                want = Vector((mirror * direction[0], direction[1], direction[2])).normalized()
                axis = now.cross(want)
                if axis.length > 1e-6:
                    turn(f"{side}UpperArm", axis.normalized(), now.angle(want))

    poses = {label: (lambda d=direction: arms(d)) for label, direction in GUARD_ARMS.items()}
    if "Head" in bones:
        poses["head turn"] = lambda: turn("Head", Vector((0, 0, 1)), 0.6)
    if "LeftUpperLeg" in bones:
        poses["leg lift"] = lambda: turn("LeftUpperLeg", Vector((1, 0, 0)), 0.8)
    kept = {bone.name: bone.matrix_basis.copy() for bone in bones}
    worst = {}
    try:
        for label, pose in poses.items():
            for bone in bones:
                bone.matrix_basis = Matrix.Identity(4)
            bpy.context.view_layer.update()
            pose()
            graph = bpy.context.evaluated_depsgraph_get()
            worst[label] = 0.0
            for obj, before, pairs in zip(mesh_objs, rest, edges):
                evaluated = obj.evaluated_get(graph)
                data = evaluated.to_mesh()
                coords = np.empty(len(data.vertices) * 3)
                data.vertices.foreach_get("co", coords)
                evaluated.to_mesh_clear()
                matrix = np.array(obj.matrix_world)
                now = coords.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
                length = np.linalg.norm(before[pairs[:, 0]] - before[pairs[:, 1]], axis=1)
                grown = np.linalg.norm(now[pairs[:, 0]] - now[pairs[:, 1]], axis=1) - length
                # edges with no length at rest are the stitches across sleeve cuts, made to stretch
                worst[label] = max(worst[label], float(grown[length > 1e-6].max(initial=0.0)))
    finally:
        for bone in bones:
            bone.matrix_basis = kept[bone.name]
        bpy.context.view_layer.update()
    return worst


def tore(before, after):
    """[(pose, before, after)] for the poses whose worst stretch grew by more than TEARS (see worst_stretch)."""
    return [(pose, before.get(pose, 0.0), value) for pose, value in after.items()
            if value - before.get(pose, 0.0) > TEARS]


def _blend_skinning(mesh_obj, armature_obj, index, target, strength):
    """Blend normalized deform weights only; preserve unrelated vertex groups."""
    groups = mesh_obj.vertex_groups
    old = {g.group: g.weight for g in mesh_obj.data.vertices[index].groups
           if (bone := armature_obj.data.bones.get(groups[g.group].name)) is not None and bone.use_deform}
    total, target_total = sum(old.values()), sum(target.values())
    new = {g: (1 - strength) * w / total for g, w in old.items()} if total > 0 else {}
    for g, w in target.items():
        new[g] = new.get(g, 0) + strength * w / target_total
    new = {g: w for g, w in new.items() if w > 1e-6}
    norm = sum(new.values())
    new = {g: w / norm for g, w in new.items()}
    if all(abs(old.get(g, 0) - new.get(g, 0)) < 1e-6 for g in old.keys() | new.keys()):
        return 0
    for g in old.keys() - new.keys():
        groups[g].remove([index])
    for g, w in new.items():
        groups[g].add([index], w, "REPLACE")
    return 1


def _write_weights(mesh_obj, weights, touched):
    """Write back only the vertices that were handed over."""
    for group in mesh_obj.vertex_groups:
        row = weights[group.index]
        keep = [index for index in touched if row[index] > 1e-4]
        drop = [index for index in touched if row[index] <= 1e-4]
        if drop:
            group.remove(drop)
        for index in keep:
            group.add([index], float(row[index]), "REPLACE")


# close_small_holes looks at holes no larger than this (m) across, and brings together border
# points nearer than HOLE_GAP (model A's pairs: 0.6 to 0.9 mm).
HOLE_SIZE = 0.04
HOLE_GAP = 0.0015


def find_small_holes(mesh_obj, low=None, high=None, size=HOLE_SIZE, verts=None):
    """Holes in the surface no larger than `size`: lists of mesh edge indices, one a hole.

    A hole is a closed ring of edges with a face on one side only, UV seams joined first. With
    `low`/`high` (world space) only holes wholly inside that box. Borders that don't close on
    themselves (a cloth's edge running out of the box) are not holes. Changes nothing.
    """
    mesh = mesh_obj.data
    verts = world_vertices(mesh_obj) if verts is None else verts
    weld, a, b = welded_edges(mesh_obj, verts)
    edge_of_loop = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("edge_index", edge_of_loop)
    faces = np.bincount(edge_of_loop, minlength=len(mesh.edges))
    _, same = np.unique(np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1), axis=0, return_inverse=True)
    same = same.ravel()
    lone = np.nonzero((np.bincount(same, weights=faces)[same] == 1) & (faces == 1))[0]
    labels = np.arange(len(verts))
    while True:
        updated = labels.copy()
        np.minimum.at(updated, a[lone], labels[b[lone]])
        np.minimum.at(updated, b[lone], labels[a[lone]])
        updated = updated[updated]
        if np.array_equal(updated, labels):
            break
        labels = updated
    holes = []
    for label in np.unique(labels[a[lone]]):
        edges = lone[labels[a[lone]] == label]
        ends = np.concatenate([a[edges], b[edges]])
        points = verts[np.unique(ends)]
        closed = (np.bincount(ends)[np.unique(ends)] % 2 == 0).all()
        inside = (low is None or (points >= np.asarray(low)).all()) and (high is None or (points <= np.asarray(high)).all())
        if closed and inside and np.ptp(points, axis=0).max() <= size:
            holes.append(edges.tolist())
    return holes


def close_small_holes(mesh_obj, low=None, high=None, size=HOLE_SIZE, gap=HOLE_GAP):
    """Close the surface's small holes (find_small_holes) that are cracks: border points within
    `gap` of each other are brought together. Returns how many vertices were moved.

    Model A's T-pose bake of 2026-09-21 left a row of them across the shirt's chest: pairs of
    vertices 0.9 mm apart where there had been one point, so a sliver 12 mm tall stood open above
    each pair and a crack ran between the pairs; dark lines in every build made from it. Filling
    them with faces smeared the texture (the crack is a UV seam, and its ring is far from convex),
    so the points are moved instead: each group onto its first vertex, in the basis and by the
    same step in every shape key. No vertex is added or removed and no weight changes. A hole
    with real area has no such pairs and is left as it is.
    """
    mesh = mesh_obj.data
    verts = world_vertices(mesh_obj)
    holes = find_small_holes(mesh_obj, low, high, size, verts)
    if not holes:
        return 0
    weld, _, _ = welded_edges(mesh_obj, verts)
    ends = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", ends)
    ends = ends.reshape(-1, 2)
    step = np.zeros((len(verts), 3))
    for hole in holes:
        points = np.unique(weld[ends[hole]])
        near = np.linalg.norm(verts[points][:, None] - verts[points][None], axis=2) < gap
        group = np.arange(len(points))
        for _ in range(len(points)):
            group = np.where(near, group[None, :], len(points)).min(axis=1)
        for leader in np.unique(group):
            for point in points[group == leader]:
                step[weld == point] = verts[points[leader]] - verts[point]
    moved = np.linalg.norm(step, axis=1) > 0
    if not moved.any():
        return 0
    step = step @ np.linalg.inv(np.array(mesh_obj.matrix_world)[:3, :3]).T
    blocks = mesh.shape_keys.key_blocks if mesh.shape_keys else []
    for data in [mesh.vertices] + [block.data for block in blocks]:
        coords = np.empty(len(verts) * 3, dtype=np.float32)
        data.foreach_get("co", coords)
        data.foreach_set("co", (coords.reshape(-1, 3) + step).astype(np.float32).ravel())
    mesh.update()
    return int(np.count_nonzero(moved))
