"""Build a VRM humanoid armature from detected joints and skin a mesh to it.

Assumes the character stands on z=0 facing -Y.
"""

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

from .meshdata import world_vertices

# VRM humanoid bone key (as named by the VRM add-on) -> Blender bone name.
BONE_NAMES = {
    "hips": "Hips",
    "spine": "Spine",
    "chest": "Chest",
    "upper_chest": "UpperChest",
    "neck": "Neck",
    "head": "Head",
    "left_shoulder": "LeftShoulder",
    "left_upper_arm": "LeftUpperArm",
    "left_lower_arm": "LeftLowerArm",
    "left_hand": "LeftHand",
    "right_shoulder": "RightShoulder",
    "right_upper_arm": "RightUpperArm",
    "right_lower_arm": "RightLowerArm",
    "right_hand": "RightHand",
    "left_upper_leg": "LeftUpperLeg",
    "left_lower_leg": "LeftLowerLeg",
    "left_foot": "LeftFoot",
    "right_upper_leg": "RightUpperLeg",
    "right_lower_leg": "RightLowerLeg",
    "right_foot": "RightFoot",
}

# Where the spine joints sit along hips -> neck.
SPINE_AT, CHEST_AT, UPPER_CHEST_AT = 0.25, 0.55, 0.8

# Resolution of the skinning proxy, in voxels per character height.
PROXY_VOXELS = 160
# The head bone is at least this long (m), even when its joint sits low on the neck.
HEAD_MIN_LENGTH = 0.2

# A separate part keeps arm weights only if it runs along at least this fraction of the arm,
# counting its surface within ARM_REACH arm lengths of the arm's axis (a sleeve does, a coat
# tail or hair lock hanging beside the hand doesn't) ...
ARM_COVERAGE = 0.4
ARM_REACH = 0.25
# ... or if at least HUG_SHARE of it lies within ARM_HUG arm lengths of the axis (a glove, a cuff).
ARM_HUG = 0.1
HUG_SHARE = 0.6
# ... or if it is small (no longer than SMALL_PART arm lengths) and centred within
# SMALL_REACH arm lengths of the axis (a cuff ring or buckle wider than a glove).
SMALL_PART = 0.25
SMALL_REACH = 0.05
# Straps and buckles hanging off a cuff: separate parts no bigger than STRAP_SIZE (m) and
# STRAP_MAX_VERTS, within STRAP_GAP (m) of a part that hugs the arm.
STRAP_SIZE = 0.08
STRAP_MAX_VERTS = 2000
STRAP_GAP = 0.012
# Within a part that keeps arm weights, arm influence fades out between these distances from
# the axis (arm lengths), so a coat's skirt doesn't flare out with the sleeves it is joined to.
ARM_FADE = (0.08, 0.14)
# A sleeve wraps all the way round the arm; beyond the last stretch where a part still
# fills SLEEVE_SECTORS of SLEEVE_RING angular sectors round the axis (within ARM_REACH), the
# part is something beside the arm, like a coat tail hanging next to the hand, and gets no
# arm weight. Stretches are SLEEVE_STEP arm lengths long.
SLEEVE_RING = 12
SLEEVE_SECTORS = 9
SLEEVE_STEP = 0.04
# Sleeves get their arm weights set outright rather than trusting the skinning proxy, which
# fuses a sleeve with the torso wherever they touch or intersect in the A-pose. A vertex is on
# the arm inside the arm's tube: its radius at each step along the arm is the ARM_TUBE_PERCENTILE
# distance of the sleeve's surface on the side facing away from the torso, at most ARM_TUBE_MAX arm lengths. Arm weight fades out
# between TUBE_FADE times that radius, and blends into the proxy's weights over the first
# SHOULDER_BLEND of the arm. Between bones it blends over JOINT_BLEND (arm lengths).
ARM_TUBE_PERCENTILE = 85
ARM_TUBE_MAX = 0.12
TUBE_FADE = (1.55, 1.85)
SHOULDER_BLEND = 0.12
JOINT_BLEND = 0.05
# Smoothing passes over the arm share before a sleeve is cut free of the torso.
SPLIT_SMOOTHING = 8
TORSO_KEYS = ("hips", "spine", "chest", "upper_chest")


def build_armature(joints, name="Armature"):
    """Create and return an armature object with one bone per VRM humanoid joint."""
    j = {key: Vector(value) for key, value in joints.items()}
    hips, neck, head = j["hips"], j["neck"], j["head"]
    spine, chest, upper_chest = (hips.lerp(neck, t) for t in (SPINE_AT, CHEST_AT, UPPER_CHEST_AT))

    # bone key -> (head, tail, parent key); parents are listed before their children.
    segments = {
        "hips": (hips, spine, None),
        "spine": (spine, chest, "hips"),
        "chest": (chest, upper_chest, "spine"),
        "upper_chest": (upper_chest, neck, "chest"),
        "neck": (neck, head, "upper_chest"),
        "head": (head, head + Vector((0.0, 0.0, max(2.5 * (head - neck).length, HEAD_MIN_LENGTH))), "neck"),
    }
    for side in ("left", "right"):
        upper_arm, lower_arm, hand = j[f"{side}_upper_arm"], j[f"{side}_lower_arm"], j[f"{side}_hand"]
        clavicle = Vector((neck.x + 0.2 * (upper_arm.x - neck.x), upper_arm.y, (upper_arm.z + neck.z) / 2))
        segments[f"{side}_shoulder"] = (clavicle, upper_arm, "upper_chest")
        segments[f"{side}_upper_arm"] = (upper_arm, lower_arm, f"{side}_shoulder")
        segments[f"{side}_lower_arm"] = (lower_arm, hand, f"{side}_upper_arm")
        segments[f"{side}_hand"] = (hand, hand + 0.4 * (hand - lower_arm), f"{side}_lower_arm")

        upper_leg, lower_leg, foot = j[f"{side}_upper_leg"], j[f"{side}_lower_leg"], j[f"{side}_foot"]
        toe = Vector((foot.x, foot.y - 0.35 * (lower_leg - foot).length, foot.z * 0.3))
        segments[f"{side}_upper_leg"] = (upper_leg, lower_leg, "hips")
        segments[f"{side}_lower_leg"] = (lower_leg, foot, f"{side}_upper_leg")
        segments[f"{side}_foot"] = (foot, toe, f"{side}_lower_leg")

    data = bpy.data.armatures.new(name)
    armature = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(armature)
    make_active(armature)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = {}
    for key, (head_point, tail_point, parent) in segments.items():
        bone = data.edit_bones.new(BONE_NAMES[key])
        bone.head, bone.tail = head_point, tail_point
        if parent is not None:
            bone.parent = bones[parent]
            bone.use_connect = (bones[parent].tail - head_point).length < 1e-6
        bones[key] = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def skin(mesh_obj, armature_obj, parts=None):
    """Weight the mesh to the armature and deform it with an Armature modifier.

    Automatic (bone heat) weights fail on the self-intersecting surfaces that
    generators produce, so weights are solved on a voxel-remeshed proxy and
    copied back onto the original vertices.

    With `parts` (see parts.find), hair and face inserts are left out of the proxy
    and follow the head, and parts that don't run along an arm lose arm weights, so
    hair and coat tails hanging beside the hands stay put when the arms move. Sleeves
    joined to the torso are then cut free of it, which adds vertices: the returned
    parts cover the new vertex list (None without parts).
    """
    rigid = None
    if parts is not None:
        rigid = parts.hair | np.isin(parts.labels, [p for p in list(parts.eyes.values()) + [parts.mouth] if p >= 0])

    proxy = mesh_obj.copy()
    proxy.data = mesh_obj.data.copy()
    proxy.name = f"{mesh_obj.name}_skin_proxy"
    proxy.modifiers.clear()
    proxy.vertex_groups.clear()
    bpy.context.scene.collection.objects.link(proxy)
    if rigid is not None and rigid.any():
        _delete_vertices(proxy.data, np.nonzero(rigid)[0])
    # The proxy only needs its shape; reprojecting UVs and attributes onto a fine remesh
    # crashes Blender 4.2 on some imported meshes.
    for layer in list(proxy.data.uv_layers):
        proxy.data.uv_layers.remove(layer)
    for attribute in [a for a in proxy.data.attributes if not a.is_internal and not a.is_required]:
        proxy.data.attributes.remove(attribute)

    make_active(proxy)
    proxy.data.remesh_voxel_size = mesh_obj.dimensions.z / PROXY_VOXELS
    bpy.ops.object.voxel_remesh()

    make_active(armature_obj)
    proxy.select_set(True)
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")

    make_active(mesh_obj)
    mesh_obj.vertex_groups.clear()
    transfer = mesh_obj.modifiers.new("skin_transfer", "DATA_TRANSFER")
    transfer.object = proxy
    transfer.use_vert_data = True
    transfer.data_types_verts = {"VGROUP_WEIGHTS"}
    transfer.vert_mapping = "POLYINTERP_NEAREST"
    transfer.layers_vgroup_select_src = "ALL"
    transfer.layers_vgroup_select_dst = "NAME"
    bpy.ops.object.datalayout_transfer(modifier=transfer.name)
    bpy.ops.object.modifier_apply(modifier=transfer.name)
    bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
    bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)

    mesh_obj.parent = armature_obj
    mesh_obj.matrix_parent_inverse = armature_obj.matrix_world.inverted()
    deform = mesh_obj.modifiers.new("Armature", "ARMATURE")
    deform.object = armature_obj

    proxy_mesh = proxy.data
    bpy.data.objects.remove(proxy)
    bpy.data.meshes.remove(proxy_mesh)

    if parts is None:
        return None
    _follow_head(mesh_obj, np.nonzero(rigid)[0])
    cuttable = _keep_arms_off_parts(mesh_obj, armature_obj, parts.labels, rigid)
    origin = _split_sleeves(mesh_obj, armature_obj, cuttable)
    # vertices added by the split belong to the same parts as the ones they were split from
    return parts._replace(labels=parts.labels[origin], hair=parts.hair[origin])


def _split_sleeves(mesh_obj, armature_obj, cuttable):
    """Cut sleeves free where they share a surface with the torso; return each vertex's origin.

    Generators often fuse a sleeve's underside to the jacket's side. However that surface is
    weighted it stretches when the arm lifts, so it is split along the border between faces
    that follow the arm and faces that follow the torso, and each side keeps only its own
    bones. Only `cuttable` vertices (sleeve, below the shoulder) are split.
    """
    mesh = mesh_obj.data
    groups = mesh_obj.vertex_groups
    arm_names = {BONE_NAMES[f"{side}_{key}"] for side in ("left", "right") for key in ("upper_arm", "lower_arm", "hand")}
    arm = {groups[n].index for n in arm_names if n in groups}
    count = len(mesh.vertices)
    if not cuttable.any() or not arm:
        return np.arange(count)

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    deform = bm.verts.layers.deform.verify()
    origin_layer = bm.verts.layers.int.new("split_origin")
    for v in bm.verts:
        v[origin_layer] = v.index
    share = np.array([sum(w for g, w in v[deform].items() if g in arm) for v in bm.verts])
    # smooth the share across the surface first, so the cut runs along a clean line
    edge_pairs = np.array([[e.verts[0].index, e.verts[1].index] for e in bm.edges])
    if len(edge_pairs):
        smooth = share.copy()
        degree = np.bincount(edge_pairs.ravel(), minlength=len(share)).astype(float)
        for _ in range(SPLIT_SMOOTHING):
            total = np.zeros_like(smooth)
            np.add.at(total, edge_pairs[:, 0], smooth[edge_pairs[:, 1]])
            np.add.at(total, edge_pairs[:, 1], smooth[edge_pairs[:, 0]])
            smooth = np.where(cuttable & (degree > 0), 0.5 * smooth + 0.5 * total / np.maximum(degree, 1), smooth)
        share = smooth
    side_layer = bm.faces.layers.int.new("split_side")  # 1 follows the arm, 0 the torso
    arm_faces = []
    # faces with any cuttable corner count, so no face at the zone's edge stays half-joined
    zone = cuttable.copy()
    for f in bm.faces:
        f[side_layer] = int(np.mean([share[v.index] for v in f.verts]) >= 0.5)
        if any(cuttable[v.index] for v in f.verts):
            for v in f.verts:
                zone[v.index] = True
            if f[side_layer]:
                arm_faces.append(f)
    if arm_faces:
        # Split the arm side off as its own surface; this also separates faces that only
        # meet the torso side at a corner, which cutting edges alone would leave joined.
        bmesh.ops.split(bm, geom=arm_faces, use_only_faces=False)
        for v in bm.verts:
            if not zone[v[origin_layer]]:
                continue
            sides = {f[side_layer] for f in v.link_faces}
            if len(sides) != 1:
                continue
            weights = dict(v[deform].items())
            keep = {g: w for g, w in weights.items() if (g in arm) == (1 in sides)}
            if sum(keep.values()) <= 1e-6:
                # a corner copied onto the other side has no weights for it: borrow its neighbours'
                keep = {}
                for edge in v.link_edges:
                    for g, w in edge.other_vert(v)[deform].items():
                        if (g in arm) == (1 in sides):
                            keep[g] = keep.get(g, 0.0) + w
            total = sum(keep.values())
            if total <= 1e-6:
                continue
            for g in weights:
                del v[deform][g]
            for g, w in keep.items():
                v[deform][g] = w / total
    bm.faces.layers.int.remove(side_layer)
    origin = np.array([v[origin_layer] for v in bm.verts])
    bm.verts.layers.int.remove(origin_layer)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return origin


def _delete_vertices(mesh, indices):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[int(i)] for i in indices], context="VERTS")
    bm.to_mesh(mesh)
    bm.free()


def _follow_head(mesh_obj, indices):
    """Weight these vertices fully to the head bone."""
    if len(indices) == 0:
        return
    indices = indices.tolist()
    for group in mesh_obj.vertex_groups:
        group.remove(indices)
    head = mesh_obj.vertex_groups.get(BONE_NAMES["head"]) or mesh_obj.vertex_groups.new(name=BONE_NAMES["head"])
    head.add(indices, 1.0, "REPLACE")


def fill_unweighted(mesh_obj, max_influences=4):
    """Weight vertices that have no weights, which would stay put while the bones move.

    Weights spread in from the weighted surface around them, with vertices sharing a spot
    counted as joined so separate pieces follow what they touch; pieces touching nothing
    weighted take the nearest weighted vertex's weights. Returns how many were filled.
    """
    mesh = mesh_obj.data
    groups = mesh_obj.vertex_groups
    empty = {v.index for v in mesh.vertices if not any(g.weight > 0 for g in v.groups)}
    filled_count = len(empty)
    if not empty or not groups:
        return 0
    verts = world_vertices(mesh_obj)
    links = {i: set() for i in empty}
    for edge in mesh.edges:
        a, b = edge.vertices
        if a in links:
            links[a].add(b)
        if b in links:
            links[b].add(a)
    _, spot = np.unique(np.round(verts / 1e-6).astype(np.int64), axis=0, return_inverse=True)
    spot = spot.ravel()
    same_spot = {}
    for i in np.nonzero(np.isin(spot, spot[list(empty)]))[0]:
        same_spot.setdefault(spot[i], []).append(int(i))
    for i in empty:
        links[i].update(same_spot[spot[i]])
        links[i].discard(i)

    def weights(i):
        return {g.group: g.weight for g in mesh.vertices[i].groups if g.weight > 0}

    def assign(i, new):
        for g in list(mesh.vertices[i].groups):
            groups[g.group].remove([i])
        strongest = sorted(new.items(), key=lambda item: -item[1])[:max_influences]
        total = sum(w for _, w in strongest)
        for g, w in strongest:
            groups[g].add([i], w / total, "REPLACE")

    while empty:
        found = {}
        for i in empty:
            acc = {}
            for j in links[i]:
                if j not in empty:
                    for g, w in weights(j).items():
                        acc[g] = acc.get(g, 0.0) + w
            if acc:
                found[i] = acc
        if not found:
            break
        for i, new in found.items():
            assign(i, new)
        empty -= found.keys()
    if empty:
        weighted = [v.index for v in mesh.vertices if v.index not in empty]
        if weighted:
            tree = KDTree(len(weighted))
            for k, i in enumerate(weighted):
                tree.insert(verts[i], k)
            tree.balance()
            for i in empty:
                assign(i, weights(weighted[tree.find(verts[i])[1]]))
    return filled_count


def _keep_arms_off_parts(mesh_obj, armature_obj, labels, skip):
    """Take arm weights off whatever doesn't belong to the arm, renormalising what's left.

    Returns a mask of the sleeve vertices whose arm weights were set outright, where a
    sleeve may be cut free of the torso (see _split_sleeves).

    Parts that neither run along nor hug the arm lose arm weights entirely; parts that
    do keep them near the arm and lose them with distance from it.
    """
    verts = world_vertices(mesh_obj)
    bones = armature_obj.data.bones
    matrix = armature_obj.matrix_world
    groups = mesh_obj.vertex_groups
    torso = [(BONE_NAMES[key], (matrix @ bones[BONE_NAMES[key]].head_local).z) for key in TORSO_KEYS]
    cuttable = np.zeros(len(verts), dtype=bool)
    straps = []
    for side in ("left", "right"):
        start = np.array(matrix @ bones[BONE_NAMES[f"{side}_upper_arm"]].head_local)
        end = np.array(matrix @ bones[BONE_NAMES[f"{side}_hand"]].tail_local)
        arm_names = {BONE_NAMES[f"{side}_{key}"] for key in ("upper_arm", "lower_arm", "hand")}
        arm_indices = {groups[name].index for name in arm_names if name in groups}
        if not arm_indices:
            continue
        shaft = end - start
        length = float(np.linalg.norm(shaft))
        along = (verts - start) @ shaft / length**2
        distance = np.linalg.norm(verts - (start + np.clip(along, 0.0, 1.0)[:, None] * shaft), axis=1)
        near = (distance < ARM_REACH * length) & (along > -0.1) & (along < 1.1)
        fade_in, fade_out = ARM_FADE[0] * length, ARM_FADE[1] * length
        faded = np.clip((fade_out - distance) / (fade_out - fade_in), 0.0, 1.0)
        tube_parts, hug_parts = [], []
        for part in np.unique(labels[~skip]):
            members = (labels == part) & ~skip
            on_arm = members & near
            covers = on_arm.sum() > 3 and np.ptp(along[on_arm]) >= ARM_COVERAGE
            hugs = np.mean(distance[members] < ARM_HUG * length) >= HUG_SHARE
            if not hugs and members.any():
                size = float(np.ptp(verts[members], axis=0).max())
                centre = verts[members].mean(axis=0)
                c_along = float((centre - start) @ shaft / length**2)
                c_dist = float(np.linalg.norm(centre - (start + np.clip(c_along, 0.0, 1.0) * shaft)))
                hugs = size <= SMALL_PART * length and c_dist <= SMALL_REACH * length and -0.1 < c_along < 1.1
            keep = faded if covers or hugs else np.zeros(len(verts))
            if covers and not hugs:
                tube_parts.append((members, _sleeve_end(verts[on_arm], along[on_arm], start, shaft)))
                continue
            if hugs:
                hug_parts.append(members)
                continue
            _apply_arm_keep(mesh_obj, groups, verts, torso, arm_indices, members, keep)
        joints_along = [
            float((np.array(matrix @ bones[BONE_NAMES[f"{side}_{key}"]].head_local) - start) @ shaft / length**2)
            for key in ("lower_arm", "hand")
        ]
        names = [BONE_NAMES[f"{side}_{key}"] for key in ("upper_arm", "lower_arm", "hand")]
        # Gloves, cuffs and buckles are rigid on the arm: any torso weight would stretch them.
        for members in hug_parts:
            _set_arm_weights(mesh_obj, groups, verts, torso, arm_indices, names, members,
                             np.ones(len(verts)), along, joints_along)
        hugged = np.zeros(len(verts), dtype=bool)
        for members in hug_parts:
            hugged |= members
        straps.append((arm_indices, names, along, joints_along, hugged))
        if tube_parts:
            sleeve = np.zeros(len(verts), dtype=bool)
            for members, _end in tube_parts:
                sleeve |= members
            tube = _inside_tube(verts, along, distance, sleeve & near, start, shaft)
            for members, sleeve_end in tube_parts:
                share = np.where(along > sleeve_end + SLEEVE_STEP, 0.0, tube)
                region = members & (near | _has_any(mesh_obj, arm_indices, members))
                _set_arm_weights(mesh_obj, groups, verts, torso, arm_indices, names, region, share, along, joints_along)
                cuttable |= region & (along > SHOULDER_BLEND) & (along < sleeve_end + SLEEVE_STEP)
    for arm_indices, names, along, joints_along, hugged in straps:
        _attach_straps(mesh_obj, groups, verts, torso, labels, skip, arm_indices, names, along, joints_along, hugged)
    return cuttable


def _apply_arm_keep(mesh_obj, groups, verts, torso, arm_indices, members, keep):
    """Scale these vertices' arm weights by keep, renormalising; all-arm vertices fall back to the torso."""
    for index in np.nonzero(members & (keep < 1.0))[0]:
        vertex = mesh_obj.data.vertices[int(index)]
        weights = {g.group: g.weight for g in vertex.groups}
        if not arm_indices & weights.keys():
            continue
        kept = {g: w * (keep[index] if g in arm_indices else 1.0) for g, w in weights.items()}
        kept = {g: w for g, w in kept.items() if w > 1e-6}
        if sum(kept.values()) <= 1e-6:
            name = min(torso, key=lambda item: abs(item[1] - verts[index, 2]))[0]
            group = groups.get(name) or groups.new(name=name)
            kept = {group.index: 1.0}
        total = sum(kept.values())
        for g in weights.keys() - kept.keys():
            groups[g].remove([int(index)])
        for g, w in kept.items():
            groups[g].add([int(index)], w / total, "REPLACE")


def _arm_frame(start, shaft):
    axis = shaft / np.linalg.norm(shaft)
    side = np.cross(axis, (0.0, -1.0, 0.0))
    if np.linalg.norm(side) < 1e-6:
        side = np.cross(axis, (0.0, 0.0, 1.0))
    side /= np.linalg.norm(side)
    return axis, side, np.cross(side, axis)


def _inside_tube(verts, along, distance, sample, start, shaft):
    """Per vertex, 1 inside the arm's tube fading to 0 outside it (see ARM_TUBE_PERCENTILE)."""
    _axis, side, up = _arm_frame(start, shaft)
    offset = verts - start
    inward = np.array([-np.sign(start[0]) or -1.0, 0.0, 0.0])
    inward_2d = np.array([inward @ side, inward @ up])
    inward_2d /= max(np.linalg.norm(inward_2d), 1e-9)
    radial = np.stack([offset @ side, offset @ up], axis=1)
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    outer = (radial @ inward_2d) < -0.5  # within 60 degrees of straight away from the torso
    step = np.floor(along / SLEEVE_STEP).astype(int)
    radius = {}
    for s_ in np.unique(step[sample]):
        pick = sample & outer & (step == s_)
        if pick.sum() >= 4:
            radius[s_] = float(np.percentile(distance[pick], ARM_TUBE_PERCENTILE))
    if not radius:
        return np.ones(len(verts))
    known = np.array(sorted(radius))
    length = float(np.linalg.norm(shaft))
    r = np.minimum(np.interp(step, known, [radius[k] for k in known]), ARM_TUBE_MAX * length)
    return np.clip((TUBE_FADE[1] * r - distance) / ((TUBE_FADE[1] - TUBE_FADE[0]) * r), 0.0, 1.0)


def _attach_straps(mesh_obj, groups, verts, torso, labels, skip, arm_indices, names, along, joints_along, hugged):
    """Small parts hanging off a cuff or glove (straps, buckles) follow the arm.

    A small part (see SMALL_PART) within STRAP_GAP of a part that hugs the arm is attached
    to it, even when it sits too far from the arm's axis to count as hugging it itself.
    """
    if not hugged.any():
        return
    tree = KDTree(int(hugged.sum()))
    for i, p in enumerate(verts[hugged]):
        tree.insert(p, i)
    tree.balance()
    length = max(np.ptp(along[hugged]), 1e-6)
    for part in np.unique(labels[~skip & ~hugged]):
        members = (labels == part) & ~skip
        if members.sum() > STRAP_MAX_VERTS or np.ptp(verts[members], axis=0).max() > STRAP_SIZE:
            continue
        gap = min(tree.find(p)[2] for p in verts[members][:: max(1, int(members.sum()) // 30)])
        if gap <= STRAP_GAP:
            _set_arm_weights(mesh_obj, groups, verts, torso, arm_indices, names, members,
                             np.ones(len(verts)), along, joints_along)


def _has_any(mesh_obj, group_indices, mask):
    out = np.zeros(len(mask), dtype=bool)
    for index in np.nonzero(mask)[0]:
        out[index] = any(g.group in group_indices for g in mesh_obj.data.vertices[int(index)].groups)
    return out


def _set_arm_weights(mesh_obj, groups, verts, torso, arm_indices, names, region, share, along, joints_along):
    """Give these vertices `share` of arm weight, split between the arm bones by position."""
    elbow, wrist = joints_along
    ramp = lambda at: np.clip((along - (at - JOINT_BLEND)) / (2 * JOINT_BLEND), 0.0, 1.0)
    to_lower, to_hand = ramp(elbow), ramp(wrist)
    split = np.stack([1 - to_lower, to_lower * (1 - to_hand), to_hand], axis=1)
    blend = np.clip(along / SHOULDER_BLEND, 0.0, 1.0)
    bone_groups = [groups.get(n) or groups.new(name=n) for n in names]
    for index in np.nonzero(region)[0]:
        i = int(index)
        vertex = mesh_obj.data.vertices[i]
        weights = {g.group: g.weight for g in vertex.groups}
        proxy_arm = sum(w for g, w in weights.items() if g in arm_indices)
        arm = blend[i] * share[i] + (1 - blend[i]) * proxy_arm
        others = {g: w for g, w in weights.items() if g not in arm_indices}
        total_others = sum(others.values())
        if arm < 1.0 and total_others <= 1e-6:
            name = min(torso, key=lambda item: abs(item[1] - verts[i, 2]))[0]
            group = groups.get(name) or groups.new(name=name)
            others, total_others = {group.index: 1.0}, 1.0
        new = {g: w / total_others * (1 - arm) for g, w in others.items()} if total_others > 1e-6 else {}
        if blend[i] < 1.0 and proxy_arm > 1e-6:
            # near the shoulder keep the proxy's own split between the arm bones
            for g, w in weights.items():
                if g in arm_indices:
                    new[g] = new.get(g, 0.0) + arm * w / proxy_arm
        else:
            for group, part in zip(bone_groups, split[i]):
                if part * arm > 1e-6:
                    new[group.index] = new.get(group.index, 0.0) + part * arm
        new = {g: w for g, w in new.items() if w > 1e-6}
        for g in weights.keys() - new.keys():
            groups[g].remove([i])
        for g, w in new.items():
            groups[g].add([i], w, "REPLACE")


def _sleeve_end(points, along, start, shaft):
    """How far along the arm (0 shoulder .. 1 hand) a part still wraps round it."""
    axis = shaft / np.linalg.norm(shaft)
    side = np.cross(axis, (0.0, -1.0, 0.0))
    if np.linalg.norm(side) < 1e-6:
        side = np.cross(axis, (0.0, 0.0, 1.0))
    side /= np.linalg.norm(side)
    up = np.cross(side, axis)
    offset = points - start
    angle = np.arctan2(offset @ up, offset @ side)
    sector = ((angle + np.pi) / (2 * np.pi) * SLEEVE_RING).astype(int) % SLEEVE_RING
    step = np.floor(along / SLEEVE_STEP).astype(int)
    end = 0.0
    for s in np.unique(step):
        if len(np.unique(sector[step == s])) >= SLEEVE_SECTORS:
            end = max(end, (s + 1) * SLEEVE_STEP)
    return end


# The head turns about the jaw: its joint goes at the chin, found on the face's front profile
# as the lowest height where the surface is still at least CHIN_FORWARD (m) in front of the neck.
CHIN_FORWARD = 0.012
# Face vertices take full head weight from HEAD_BLEND (m) above the head joint, fading to
# none at the joint.
HEAD_BLEND = 0.02
# The head pivots this far (m) above the chin, near the jaw's hinge, not at the chin itself.
HEAD_ABOVE_CHIN = 0.03


def chin_height(verts, face_mask, centre_x, neck_z, top_z):
    """Height of the chin on the face part's front profile, or None."""
    profile = face_mask & (np.abs(verts[:, 0] - centre_x) < 0.015)
    if profile.sum() < 20:
        return None
    step = 0.004
    fronts = {}
    for z in np.arange(neck_z - 0.02, top_z, step):
        band = profile & (verts[:, 2] >= z) & (verts[:, 2] < z + step)
        if band.any():
            fronts[round(float(z), 4)] = float(verts[band, 1].min())
    if not fronts:
        return None
    heights = sorted(fronts)
    neck_front = np.median([fronts[z] for z in heights[:3]])
    for z in heights:
        if fronts[z] < neck_front - CHIN_FORWARD:
            return z
    return None


def weight_head(mesh_obj, armature_obj, face_mask, labels=None, chin_z=None):
    """Face vertices above the head joint follow the head fully, blending into the neck below.

    With `labels` (part per vertex), separate parts lying wholly above the joint (eye
    highlights, clips, bangs) move whatever body weight they have onto the head; weights for
    hair spring bones are kept.
    """
    groups = mesh_obj.vertex_groups
    head = groups.get(BONE_NAMES["head"]) or groups.new(name=BONE_NAMES["head"])
    head_z = (armature_obj.matrix_world @ armature_obj.data.bones[BONE_NAMES["head"]].head_local).z
    if chin_z is not None:
        head_z = min(head_z, chin_z)  # the face below the pivot, down to the chin, still follows the head
    verts = world_vertices(mesh_obj)
    share = np.clip((verts[:, 2] - head_z) / HEAD_BLEND, 0.0, 1.0)
    for index in np.nonzero(face_mask & (share > 0))[0]:
        i = int(index)
        vertex = mesh_obj.data.vertices[i]
        others = {g.group: g.weight for g in vertex.groups if g.group != head.index}
        total = sum(others.values())
        target = share[i]
        for g, w in others.items():
            groups[g].add([i], (w / total) * (1 - target) if total > 1e-6 else 0.0, "REPLACE")
        head.add([i], target if total > 1e-6 else 1.0, "REPLACE")
    if labels is None:
        return
    lowest = np.full(labels.max() + 1, np.inf)
    np.minimum.at(lowest, labels, verts[:, 2])
    above = (lowest[labels] > head_z) & ~face_mask
    for index in np.nonzero(above)[0]:
        i = int(index)
        vertex = mesh_obj.data.vertices[i]
        body = [(g.group, g.weight) for g in vertex.groups
                if g.group != head.index and not groups[g.group].name.startswith("Hair")]
        moved = sum(w for _, w in body)
        if moved <= 1e-6:
            continue
        for g, _w in body:
            groups[g].remove([i])
        current = next((g.weight for g in vertex.groups if g.group == head.index), 0.0)
        head.add([i], current + moved, "REPLACE")


def make_active(obj):
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
