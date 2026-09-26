"""Reshape a straightened hand after a reference hand that looks and tracks well.

Image-to-3D hands are modelled curled, coarse, and with the thumb tucked; opened into the rest
pose apps expect (`tpose.spec_hands`) they came out lumpy, with a thick web and a thin or bulging
thumb, and smoothing alone could not find the right shape (2026-09-22). A VRoid hand has that
shape. Here the reference hand is fitted onto this rig bone by bone -- each segment stretched to
this hand's segment length and to its measured thickness and width, the palm to its length,
width and thickness -- and each vertex of this hand moves toward that fitted surface. Only the
overall form is taken: the move is smoothed over the surface first, so the glove's own seams,
plates and ridges ride along. Shading normals come from the fitted reference surface, which is
smooth where this mesh is coarse. Needs both hands in the same rest pose (palm down, fingers
along the arm, thumb 38 degrees out), which `spec_hands` and a VRoid T-pose share.
"""

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from . import finger_guides, fingers, meshdata
from .rig import BONE_NAMES

# Segment keys: the hand bone, then each finger's bones.
KEYS = ["hand"] + [f"{f}_{s}" for f in fingers.FINGERS for s in fingers._segments(f)]
# The wrist and the glove's cuff stay: the reshaping fades in between these distances from the
# wrist, of the wrist-to-knuckle length.
FADE_FROM = (.1, .25)
# Passes of averaging over the surface applied to the move, so only the overall form changes.
DETAIL_PASSES = 8
# Passes of averaging applied to the reference's weights before fitting it.
FIT_PASSES = 6
# Data-block collections an imported reference adds to, cleared again after reading it.
IMPORTED = ("objects", "meshes", "armatures", "materials", "images", "textures", "node_groups",
            "collections", "actions")


def _neighbours(key):
    """Segments whose surface a vertex of `key` may take its shape from: itself and its joints."""
    if key == "hand":
        return {"hand", "thumb_metacarpal"} | {f"{f}_proximal" for f in fingers.FINGERS[1:]}
    finger, segment = key.split("_")
    chain = [f"{finger}_{s}" for s in fingers._segments(finger)]
    k = chain.index(key)
    near = {key, "hand" if k == 0 else chain[k - 1]}
    if k + 1 < len(chain):
        near.add(chain[k + 1])
    return near


def hand_data(mesh_objs, groups, joints):
    """What reshaping needs of one hand: world vertices, triangles, weights over KEYS, joints.

    `groups` maps each key to its vertex-group name; `joints` maps each key to (head, tail) in
    world space. Only vertices mostly held by this hand's bones are kept.
    """
    verts, tris, weights, owners, offset = [], [], [], [], 0
    for obj in mesh_objs:
        index = {obj.vertex_groups[n].index: k for k, n in enumerate(groups[key] for key in KEYS)
                 if n in obj.vertex_groups}
        if not index:
            continue
        w = np.zeros((len(obj.data.vertices), len(KEYS)))
        for vertex in obj.data.vertices:
            for entry in vertex.groups:
                if entry.group in index:
                    w[vertex.index, index[entry.group]] += entry.weight
        verts.append(meshdata.world_vertices(obj))
        tris.append(meshdata.triangles(obj.data) + offset)
        weights.append(w)
        owners.append((obj, offset))
        offset += len(obj.data.vertices)
    verts, tris, weights = np.vstack(verts), np.vstack(tris), np.vstack(weights)
    budget = weights.sum(axis=1)
    return {"verts": verts, "tris": tris, "weights": weights / np.maximum(budget, 1e-9)[:, None],
            "budget": budget, "joints": joints, "owners": owners}


def rig_hand(mesh_objs, armature_obj, side):
    """`hand_data` for a hand rigged by this pipeline (bone tails are real)."""
    names = {"hand": BONE_NAMES[f"{side}_hand"]}
    names.update({f"{f}_{s}": fingers.bone_name(side, f, s) for f in fingers.FINGERS for s in fingers._segments(f)})
    world = armature_obj.matrix_world
    joints = {k: (np.array(world @ armature_obj.data.bones[n].head_local),
                  np.array(world @ armature_obj.data.bones[n].tail_local)) for k, n in names.items()}
    return hand_data(mesh_objs, names, joints)


def read_reference(path, sides=("left", "right")):
    """Import a VRM, read its hands with `hand_data`, and remove everything the import added."""
    before = {attr: set(getattr(bpy.data, attr)) for attr in IMPORTED}
    bpy.ops.import_scene.vrm(filepath=str(path))
    try:
        armature = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o not in before["objects"])
        human = armature.data.vrm_addon_extension.vrm1.humanoid.human_bones
        meshes = [o for o in bpy.data.objects if o.type == "MESH" and o not in before["objects"]]
        world = armature.matrix_world
        head = lambda n: np.array(world @ armature.data.bones[n].head_local)
        hands = {}
        for side in sides:
            names = {"hand": getattr(human, f"{side}_hand").node.bone_name}
            for f in fingers.FINGERS:
                for s in fingers._segments(f):
                    names[f"{f}_{s}"] = getattr(human, f"{side}_{f}_{s}").node.bone_name
            joints = {}
            for f in fingers.FINGERS:
                chain = [f"{f}_{s}" for s in fingers._segments(f)]
                for k, key in enumerate(chain):
                    bone = armature.data.bones[names[key]]
                    # imported tails are invented: end each bone at the next joint, the last at its
                    # end node (VRoid exports one past every fingertip)
                    tail = (head(names[chain[k + 1]]) if k + 1 < len(chain)
                            else head(bone.children[0].name) if bone.children else np.array(world @ bone.tail_local))
                    joints[key] = (head(names[key]), tail)
            joints["hand"] = (head(names["hand"]), head(names["middle_proximal"]))
            data = hand_data(meshes, names, joints)
            data.pop("owners")
            hands[side] = data
        return hands
    finally:
        for attr in IMPORTED:
            collection = getattr(bpy.data, attr)
            for block in list(collection):
                if block not in before[attr]:
                    collection.remove(block)


def _frames(data, side):
    """Per key: origin, unit axes (along the bone, toward the palm, across) and length."""
    joints = data["joints"]
    wrist = joints["hand"][0]
    hand = finger_guides._hand_frame(wrist, [joints[f"{f}_proximal"][0] for f in fingers.FINGERS[1:]], side)
    frames = {"hand": (wrist, hand["along"], hand["palm"], np.cross(hand["along"], hand["palm"]), hand["length"])}
    for key in KEYS[1:]:
        head, tail = joints[key]
        x = tail - head
        length = float(np.linalg.norm(x))
        x = x / length
        y = hand["palm"] - x * (hand["palm"] @ x)
        y /= np.linalg.norm(y)
        frames[key] = (head, x, y, np.cross(x, y), length)
    return frames


def _local(points, frame):
    origin, x, y, z, length = frame
    offset = points - origin
    return np.stack([offset @ x / length, offset @ y, offset @ z], axis=1)


def _spread(data, frames):
    """How thick each part of the hand is.

    The palm: (median, span) of the offsets toward the palm and across, half way along it. A
    finger: the median distance from its bones over the middle of each segment, one value for
    the whole finger (the thumb from its two outer segments). Measured over all a segment holds,
    the web and palm around a finger's base made the fitted fingers fat; measured per segment,
    too few vertices sat in some, and a glove's thick tips next to slimmer bases made a finger
    swing from thin to fat.
    """
    held = data["budget"] > .5
    owner = np.argmax(data["weights"], axis=1)
    out = {}
    hand = _local(data["verts"][held & (owner == 0)], frames["hand"])
    hand = hand[(hand[:, 0] > .3) & (hand[:, 0] < .7)]
    if len(hand) >= 8:
        low, mid, high = np.percentile(hand[:, 1:], (5, 50, 95), axis=0)
        out["hand"] = (mid, np.maximum(high - low, 1e-6))
    else:
        out["hand"] = None
    for finger in fingers.FINGERS:
        keys = [f"{finger}_{s}" for s in fingers._segments(finger)]
        radii = []
        for key in keys[1:] if finger == "thumb" else keys:
            k = KEYS.index(key)
            local = _local(data["verts"][held & (owner == k)], frames[key])
            local = local[(local[:, 0] > .2) & (local[:, 0] < .8)]
            radii.append(np.linalg.norm(local[:, 1:], axis=1))
        radii = np.concatenate(radii)
        for key in keys:
            out[key] = float(np.median(radii)) if len(radii) >= 8 else None
    return out


def fitted_reference(reference, target, side, passes=FIT_PASSES):
    """The reference hand's vertices moved onto `target`'s bones, lengths and thicknesses.

    Each vertex blends where each of its bones would put it. The blend uses the reference's
    weights averaged over its surface: its own weights switch sharply at the ball of the thumb,
    where the palm's and the thumb's fits disagree, and left a seam there.
    """
    ref_frames, our_frames = _frames(reference, side), _frames(target, side)
    ref_spread, our_spread = _spread(reference, ref_frames), _spread(target, our_frames)
    weights = _smoothed(reference["weights"], reference["verts"], reference["tris"], passes)
    fitted = np.zeros_like(reference["verts"])
    for k, key in enumerate(KEYS):
        weight = weights[:, k]
        if not np.any(weight > 0):
            continue
        local = _local(reference["verts"], ref_frames[key])
        if ref_spread[key] is not None and our_spread[key] is not None:
            if key == "hand":
                (ref_mid, ref_span), (our_mid, our_span) = ref_spread[key], our_spread[key]
                local[:, 1:] = (local[:, 1:] - ref_mid) * (our_span / ref_span) + our_mid
            else:
                local[:, 1:] *= our_spread[key] / ref_spread[key]
        origin, x, y, z, length = our_frames[key]
        placed = origin + np.outer(local[:, 0] * length, x) + np.outer(local[:, 1], y) + np.outer(local[:, 2], z)
        fitted += weight[:, None] * placed
    return fitted


def _smoothed(values, verts, tris, passes):
    """Average per-vertex `values` over a triangle mesh, points at one position counting as one."""
    keys = np.round(verts / 1e-5).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    weld = first[inverse.ravel()]
    edges = weld[np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])]
    edges = edges[edges[:, 0] != edges[:, 1]]
    degree = np.bincount(edges.ravel(), minlength=len(verts))
    out = np.zeros_like(values)
    np.add.at(out, weld, values)
    out /= np.maximum(np.bincount(weld, minlength=len(verts)), 1)[:, None]
    for _ in range(passes):
        summed = np.zeros_like(out)
        np.add.at(summed, edges[:, 0], out[edges[:, 1]])
        np.add.at(summed, edges[:, 1], out[edges[:, 0]])
        inside = degree > 0
        out[inside] = .5 * out[inside] + .5 * summed[inside] / degree[inside, None]
    out = out[weld]
    return out / np.maximum(out.sum(axis=1), 1e-12)[:, None]


def _vertex_normals(verts, tris):
    normals = np.zeros_like(verts)
    faces = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    for corner in range(3):
        np.add.at(normals, tris[:, corner], faces)
    return normals / np.maximum(np.linalg.norm(normals, axis=1), 1e-12)[:, None]


def shape_hand(mesh_obj, armature_obj, side, reference, amount=1.0, passes=DETAIL_PASSES):
    """Move this hand toward the fitted reference hand. Returns the number of vertices moved.

    `reference` is `read_reference(...)[side]`. Touches only vertices mostly held by this hand's
    bones, fading in from the wrist; other pieces of the mesh held by the hand (plates, rings)
    move with the nearest skin. Shape keys move alike; shading normals are taken from the
    fitted reference surface.
    """
    target = rig_hand([mesh_obj], armature_obj, side)
    fitted = fitted_reference(reference, target, side)
    keep = reference["budget"] > .5
    tris = reference["tris"][np.all(keep[reference["tris"]], axis=1)]
    ref_owner = np.argmax(reference["weights"], axis=1)
    tri_owner = ref_owner[tris[:, 0]]
    fitted_normals = _vertex_normals(fitted, tris)
    trees = {}
    for k, key in enumerate(KEYS):
        allowed = [KEYS.index(n) for n in _neighbours(key)]
        chosen = tris[np.isin(tri_owner, allowed)]
        if len(chosen):
            trees[k] = (BVHTree.FromPolygons([Vector(v) for v in fitted], chosen.tolist()), chosen)

    verts = target["verts"]
    count = len(verts)
    held = target["budget"] > .5
    owner = np.argmax(target["weights"], axis=1)
    frame = finger_guides._hand_frame(target["joints"]["hand"][0],
                                      [target["joints"][f"{f}_proximal"][0] for f in fingers.FINGERS[1:]], side)
    along = (verts - target["joints"]["hand"][0]) @ frame["along"] / frame["length"]
    fade = np.clip((along - FADE_FROM[0]) / (FADE_FROM[1] - FADE_FROM[0]), 0, 1) * amount
    piece = finger_guides.hand_piece(mesh_obj, side, verts)
    skin = held & piece & (fade > 0)
    move = np.zeros((count, 3))
    normal_at = np.zeros((count, 3))
    for i in np.flatnonzero(skin):
        tree = trees.get(int(owner[i]))
        if tree is None:
            continue
        location, _, face, _ = tree[0].find_nearest(Vector(verts[i]))
        if location is None:
            continue
        a, b, c = (fitted[j] for j in tree[1][face])
        u, v, w = _barycentric(np.array(location), a, b, c)
        corners = fitted_normals[tree[1][face]]
        n = u * corners[0] + v * corners[1] + w * corners[2]
        normal_at[i] = n / max(np.linalg.norm(n), 1e-12)
        move[i] = (np.array(location) - verts[i]) * fade[i]
    # only the overall form: average the move over the surface, the rest of the mesh held still
    weld, edge_a, edge_b = meshdata.welded_edges(mesh_obj, verts)
    free = np.zeros(count, bool)
    free[weld[skin]] = True
    step = edge_a != edge_b
    edge_a, edge_b = edge_a[step], edge_b[step]
    degree = np.bincount(np.r_[edge_a, edge_b], minlength=count)
    free &= degree > 0
    pooled = np.zeros((count, 3))
    np.add.at(pooled, weld, move)
    pooled /= np.maximum(np.bincount(weld, minlength=count), 1)[:, None]
    for _ in range(passes):
        summed = np.zeros_like(pooled)
        np.add.at(summed, edge_a, pooled[edge_b])
        np.add.at(summed, edge_b, pooled[edge_a])
        pooled[free] = .5 * pooled[free] + .5 * summed[free] / degree[free, None]
    move = np.where(skin[:, None], pooled[weld], 0.0)
    # plates and rings held by the hand ride on the nearest skin
    riders = held & ~piece & (fade > 0)
    if np.any(riders) and np.any(skin):
        index = np.flatnonzero(skin)
        tree = KDTree(len(index))
        for k, i in enumerate(index):
            tree.insert(Vector(verts[i]), k)
        tree.balance()
        for i in np.flatnonzero(riders):
            _, nearest, _ = tree.find(Vector(verts[i]))
            move[i] = move[index[nearest]]
    moved = np.linalg.norm(move, axis=1) > 1e-9
    if not moved.any():
        return 0
    data = mesh_obj.data
    corners = None
    if data.has_custom_normals:
        corners = np.zeros((len(data.loops), 3))
        data.corner_normals.foreach_get("vector", corners.ravel())
    local = move @ np.linalg.inv(np.array(mesh_obj.matrix_world)[:3, :3]).T
    for index in np.flatnonzero(moved):
        offset = Vector(local[index])
        data.vertices[int(index)].co += offset
        for key in (data.shape_keys.key_blocks if data.shape_keys else ()):
            key.data[int(index)].co += offset
    if corners is not None:
        loops = np.zeros(len(data.loops), int)
        data.loops.foreach_get("vertex_index", loops)
        hit = np.flatnonzero(skin[loops] & (np.linalg.norm(normal_at[loops], axis=1) > .5))
        share = fade[loops[hit]][:, None]
        mixed = corners[hit] * (1 - share) + normal_at[loops[hit]] * share
        corners[hit] = mixed / np.maximum(np.linalg.norm(mixed, axis=1), 1e-12)[:, None]
        data.normals_split_custom_set(corners.tolist())
    return int(moved.sum())


def _barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return 1.0, 0.0, 0.0
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return 1.0 - v - w, v, w
