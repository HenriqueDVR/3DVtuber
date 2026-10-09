"""Build a VRM humanoid armature from detected joints and skin a mesh to it.

Assumes the character stands on z=0 facing -Y.
"""

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

from .landmarks import ARM_HANGS
from .meshdata import pieces, sample_surface, triangles, welded_edges, world_vertices

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

# Two bones this many steps apart in the skeleton (see bone_steps) don't blend at any real
# joint; a surface on both only follows them because two body parts touch in the model. Tripo
# gave model C's right shoulder 18-45% Head where her twin-tail lies on it (Head to
# RightUpperArm: 4). Chest and UpperArm at the armpit (3), neck and shoulder at the collar or
# both thighs at the crotch (2) are real blends.
FAR_BONES = 4

# Where the spine joints sit along hips -> neck.
SPINE_AT, CHEST_AT, UPPER_CHEST_AT = 0.25, 0.55, 0.8

# Resolution of the skinning proxy, in voxels per character height.
PROXY_VOXELS = 160
# Resolutions tried in turn when bone heat fails on the proxy, and the share of proxy
# vertices that must come out weighted for a try to count.
PROXY_VOXEL_TRIES = (PROXY_VOXELS, 140, 120, 180, 110, 150, 130)
PROXY_MIN_WEIGHTED = 0.85
# Leg weights fade out between these heights above the hip joints (fractions of the height).
LEG_FADE = (0.0, 0.04)
# A separate piece with at least this share of its vertices mostly on one arm follows that arm
# all over (see _whole_arm_pieces).
ARM_PIECE = 0.75
# Passes of torso, shoulder, upper arm and thigh weight smoothing over the surface (see _smooth_body_weights).
BODY_SMOOTHING = 12
# The head bone is at least this long (m), even when its joint sits low on the neck.
HEAD_MIN_LENGTH = 0.2

# A separate part keeps arm weights only if it runs along at least this fraction of the arm,
# counting its surface within ARM_REACH arm lengths of the arm's axis (a sleeve does, a coat
# tail or hair lock hanging beside the hand doesn't) ...
ARM_COVERAGE = 0.4
ARM_REACH = 0.25
# ... measured within this tighter radius, because in an A-pose the arm hangs nearly vertical
# and so does a thigh strap or a hip buckle beside the hand: over ARM_REACH those "run along"
# the arm as convincingly as a sleeve does, and one ended up on the hand, cut free, and flung
# out when the arm lifted. A sleeve still counts: it is on the arm, not beside it.
ARM_COVER_REACH = 0.15
# ... or if at least HUG_SHARE of it lies within ARM_HUG arm lengths of the axis (a glove, a cuff).
ARM_HUG = 0.1
HUG_SHARE = 0.6
# ... or if it is small (no longer than SMALL_PART arm lengths) and centred within
# SMALL_REACH arm lengths of the axis (a cuff ring or buckle wider than a glove).
SMALL_PART = 0.25
SMALL_REACH = 0.05
# ... or if it is armour worn on the arm past the shoulder: no longer than ARMOUR_PART arm
# lengths, centred within ARMOUR_REACH of the axis, with ARMOUR_HUG_SHARE of it within
# ARM_HUG (a bracer or elbow guard standing off the sleeve).
ARMOUR_PART = 0.35
ARMOUR_REACH = 0.125
ARMOUR_HUG_SHARE = 0.3
ARMOUR_FROM = 0.3
# A small part only hugs the arm if its centre is at least this far along it (a glove, a cuff);
# gear on the chest lies near the shoulder end of the arm's axis without being on the arm.
HUG_FROM = 0.2
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
# Points sampled over a part's surface for that test (at most, and per triangle).
SLEEVE_SAMPLES = 20000
SLEEVE_SAMPLES_PER_TRIANGLE = 20
# Sleeves get their arm weights set outright rather than trusting the skinning proxy, which
# fuses a sleeve with the torso wherever they touch or intersect in the A-pose. A vertex is on
# the arm inside the arm's tube: its radius at each step along the arm is the ARM_TUBE_PERCENTILE
# distance of the sleeve's surface on the side facing away from the torso, at most ARM_TUBE_MAX arm lengths. Arm weight fades out
# between TUBE_FADE times that radius, and blends into the proxy's weights over the first
# SHOULDER_BLEND of the arm. Between bones it blends over JOINT_BLEND (arm lengths).
ARM_TUBE_PERCENTILE = 85
ARM_TUBE_MAX = 0.2
TUBE_FADE = (1.55, 1.85)
SHOULDER_BLEND = 0.12
JOINT_BLEND = 0.05
# Cloth over the arm follows it unless it faces the torso: fully while the cosine between its
# direction from the arm's axis and the direction to the torso is below the first value, not at
# all above the second. It fades out over OVER_ARM_FADE (arm lengths) before the wrist, or before
# the elbow for cloth reaching below the hips.
OVER_ARM_INWARD = (0.0, 0.4)
# Past the sleeve's fitted tube, cloth on the arm's far side from the torso still follows it,
# fully once the cosine between its direction and the torso's is below minus this.
OUTSIDE_FOLLOW = 0.3
# Sleeve vertices bone heat gave at least this much arm weight keep it (see _keep_arms_off_parts).
HEAT_TRUST = 0.5
OVER_ARM_FADE = 0.15
# ... except cloth lying against the forearm from outside (see _lying_against): it counts from a
# cosine of the first value between its normal and the direction away from the arm, fully at the
# second, and from the first multiple of the sleeve's fitted radius out, fully past the second.
AGAINST_FACING = (-0.1, -0.4)
AGAINST_BEYOND = (1.1, 1.3)
# Rings of edges that verdict spreads over, past the sleeve's radius, to the rest of that cloth.
AGAINST_SPREAD = 8
# Nothing medial of the shoulders is arm: shoulder weight fades out between these fractions of
# the arm joint's distance from the body's centre line, and arm weight between the next two,
# so the chest doesn't fold in when the arms move.
SHOULDER_SPAN = (0.5, 0.9)
ARM_SPAN = (0.6, 1.0)
# Below the shoulder the same goes for anything this far toward the body from the arm's line at
# its height (arm lengths): arm weight is full within the first value and gone past the second,
# so the torso's sides beside a hanging arm stay with the torso.
ARM_INWARD = (0.2, 0.35)
# Cuts between sleeve and torso are stitched shut only within this many arm lengths of the
# shoulder joint (the armpit); further down the sleeve simply parts from the torso.
STITCH_REACH = 0.3
# Rings of faces the cut zone grows by beyond the cuttable vertices.
ZONE_GROW = 2
# Smoothing passes over the arm share before a sleeve is cut free of the torso.
SPLIT_SMOOTHING = 8
# Where the half-share line crosses an edge this close to one of its ends (as a fraction of the
# edge), the cut goes through that end instead of splitting off a sliver.
CUT_SNAP = 0.1
TORSO_KEYS = ("hips", "spine", "chest", "upper_chest")
# With a rigid head, pieces whose mean Head weight is above this go fully onto the head.
MOSTLY_HEAD = 0.5


def bone_steps(armature_obj, names):
    """Steps between every two of the named bones through the skeleton, as a square array: from
    each up to the bone they share (inf with none). Fingers count as their hand."""
    bones = armature_obj.data.bones
    hands = {BONE_NAMES["left_hand"], BONE_NAMES["right_hand"]}
    chains = []
    for name in names:
        chain = [name] + [p.name for p in bones[name].parent_recursive]
        hand = next((k for k, bone in enumerate(chain) if bone in hands), None)
        chains.append(chain if hand is None else chain[hand:])
    steps = np.zeros((len(names), len(names)))
    for i, a in enumerate(chains):
        for j, b in enumerate(chains):
            common = next((x for x in a if x in b), None)
            steps[i, j] = np.inf if common is None else a.index(common) + b.index(common)
    return steps


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


def skin(mesh_obj, armature_obj, parts=None, solver="heat"):
    """Weight the mesh to the armature and deform it with an Armature modifier.

    Automatic (bone heat) weights fail on the self-intersecting surfaces that
    generators produce, so weights are solved on a voxel-remeshed proxy and
    copied back onto the original vertices.

    With `parts` (see parts.find), hair and face inserts are left out of the proxy
    and follow the head, and parts that don't run along an arm lose arm weights, so
    hair and coat tails hanging beside the hands stay put when the arms move. Sleeves
    joined to the torso are then cut free of it, which adds vertices: the returned
    parts cover the new vertex list (None without parts). `solver` "voxel" weights the proxy
    by geodesic voxel binding (voxel_skin.py) instead of bone heat; the rules are the same.
    """
    rigid = None
    if parts is not None:
        rigid = parts.hair | np.isin(parts.labels, [p for p in list(parts.eyes.values()) + [parts.mouth] if p >= 0])

    proxy = _weighted_proxy(mesh_obj, armature_obj, rigid, solver)

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
    # patches the proxy missed are filled now, so the rules below apply to them too
    fill_unweighted(mesh_obj, skip=rigid)
    _keep_legs_below_hips(mesh_obj, armature_obj)

    if parts is None:
        return None
    _follow_head(mesh_obj, np.nonzero(rigid)[0])
    cuttable = _keep_arms_off_parts(mesh_obj, armature_obj, parts.labels, rigid)
    # before the cut, so the sleeve is cut free along the torso's new edge instead of tearing there
    _keep_arms_off_chest(mesh_obj, armature_obj, rigid)
    # before the cut, so a sleeve panel's tip hanging by the hip isn't cut off it
    _whole_arm_pieces(mesh_obj, rigid)
    origin = _split_sleeves(mesh_obj, armature_obj, cuttable)
    _smooth_body_weights(mesh_obj, rigid[origin])
    # the blur spreads thigh weight back up past the hips
    _keep_legs_below_hips(mesh_obj, armature_obj)
    # vertices added by the split belong to the same parts as the ones they were split from
    return parts._replace(labels=parts.labels[origin], hair=parts.hair[origin], face_area=parts.face_area[origin])


def _whole_arm_pieces(mesh_obj, skip):
    """Give a piece that is mostly on one arm (see ARM_PIECE) the arm's weights all over.

    A sleeve panel hanging down beside the hip gets its lowest tip weighted to the hips by bone
    heat, which stays behind when the arm moves. Its body-weighted vertices take their
    arm-weighted neighbours' weights, spreading in over the surface.
    """
    groups = mesh_obj.vertex_groups
    mesh = mesh_obj.data
    # the mesh's own connected pieces: a sleeve panel only touching a coat counts on its own
    labels = pieces(mesh_obj)
    edges = np.array([e.vertices[:] for e in mesh.edges])
    # vertices split along UV seams sit at the same place: link them too
    co = np.round(world_vertices(mesh_obj), 5)
    _, spot = np.unique(co, axis=0, return_inverse=True)
    spot = spot.ravel()
    order = np.argsort(spot, kind="stable")
    same = order[:-1][spot[order[:-1]] == spot[order[1:]]], order[1:][spot[order[:-1]] == spot[order[1:]]]
    edges = np.concatenate([edges, np.stack(same, axis=1)])
    for side in ("left", "right"):
        arm = {groups[BONE_NAMES[f"{side}_{k}"]].index for k in ("upper_arm", "lower_arm", "hand")
               if BONE_NAMES[f"{side}_{k}"] in groups}
        if not arm:
            continue
        share = np.array([sum(g.weight for g in v.groups if g.group in arm) for v in mesh.vertices])
        for part in np.unique(labels[~skip]):
            members = (labels == part) & ~skip
            on_arm = members & (share >= 0.5)
            if on_arm.sum() < ARM_PIECE * members.sum() or (on_arm == members).all():
                continue
            done = on_arm.copy()
            todo = members & ~on_arm
            inside = edges[members[edges[:, 0]] & members[edges[:, 1]]]
            while todo.any():
                grown = False
                for a, b in ((0, 1), (1, 0)):
                    reach = inside[done[inside[:, a]] & todo[inside[:, b]]]
                    for src, dst in reach[:, [a, b]]:
                        if not todo[dst]:
                            continue
                        weights = {g.group: g.weight for g in mesh.vertices[int(src)].groups}
                        for g in list(mesh.vertices[int(dst)].groups):
                            groups[g.group].remove([int(dst)])
                        for g, w in weights.items():
                            groups[g].add([int(dst)], w, "REPLACE")
                        todo[dst] = False
                        done[dst] = True
                        grown = True
                if not grown:
                    break


def _smooth_body_weights(mesh_obj, skip):
    """Blur the torso, shoulder, upper arm and thigh weights over the surface, keeping each vertex's total.

    Bone heat on thin, layered clothing (coat panels, capes) leaves cliffs where neighbouring
    vertices belong wholly to different torso bones, which tear open when the spine bends, and
    where cloth over the shoulder meets the chest. Sleeves are already cut free of the torso, so
    this can't blur across them. Forearm, hand, head and lower-leg weights are left as they are
    (gloves and bracers stay rigid), as are `skip` vertices.
    """
    groups = mesh_obj.vertex_groups
    names = [BONE_NAMES[k] for k in TORSO_KEYS] + [BONE_NAMES[f"{s}_{k}"] for s in ("left", "right")
                                                      for k in ("upper_leg", "shoulder", "upper_arm")]
    body = [groups[n].index for n in names if n in groups]
    column = {g: i for i, g in enumerate(body)}
    mesh = mesh_obj.data
    n = len(mesh.vertices)
    weights = np.zeros((n, len(body)))
    for v in mesh.vertices:
        for g in v.groups:
            if g.group in column:
                weights[v.index, column[g.group]] = g.weight
    total = weights.sum(axis=1)
    # vertices split along UV seams sit at the same place: smooth across them too, but not
    # across a sleeve cut (see _split_sleeves), whose two sides also share places: one side
    # has arm weights, the other none
    co = np.array([v.co for v in mesh.vertices])
    arm = {groups[BONE_NAMES[f"{s}_{k}"]].index for s in ("left", "right")
           for k in ("upper_arm", "lower_arm", "hand") if BONE_NAMES[f"{s}_{k}"] in groups}
    on_arm = np.array([any(g.group in arm and g.weight > 0 for g in v.groups) for v in mesh.vertices])
    key = np.concatenate([np.round(co, 5), on_arm[:, None]], axis=1)
    _, same = np.unique(key, axis=0, return_inverse=True)
    same = same.ravel()
    edges = np.array([e.vertices[:] for e in mesh.edges])
    edges = same[edges]
    edges = edges[edges[:, 0] != edges[:, 1]]
    m = same.max() + 1
    active = np.zeros(m, bool)
    np.logical_or.at(active, same, (total > 1e-6) & ~skip)
    edges = edges[active[edges[:, 0]] & active[edges[:, 1]]]
    shares = np.zeros((m, len(body)))
    counts = np.zeros(m)
    np.add.at(shares, same, np.where(total[:, None] > 1e-6, weights / np.maximum(total, 1e-9)[:, None], 0))
    np.add.at(counts, same, total > 1e-6)
    shares = shares / np.maximum(counts, 1)[:, None]
    degree = np.bincount(edges.ravel(), minlength=m).astype(float)
    for _ in range(BODY_SMOOTHING):
        neighbours = np.zeros_like(shares)
        np.add.at(neighbours, edges[:, 0], shares[edges[:, 1]])
        np.add.at(neighbours, edges[:, 1], shares[edges[:, 0]])
        blended = 0.5 * shares + 0.5 * neighbours / np.maximum(degree, 1)[:, None]
        shares = np.where((degree > 0)[:, None] & active[:, None], blended, shares)
    new = shares[same] * total[:, None]
    for index in np.nonzero((total > 1e-6) & ~skip)[0]:
        for k, g in enumerate(body):
            if new[index, k] > 1e-4:
                groups[g].add([int(index)], float(new[index, k]), "REPLACE")
            elif weights[index, k] > 0:
                groups[g].remove([int(index)])


def _keep_arms_off_chest(mesh_obj, armature_obj, skip):
    """Hand shoulder and arm weight medial of the arm joints back to the torso (see SHOULDER_SPAN).

    Bone heat and weight smoothing spread the shoulder bones over the sternum and the upper
    arm over the front of the chest, which folds the chest in when the arms move. Only the
    torso's height range is touched, and nothing hugging the arm past the shoulder (the inner
    side of a sleeve on an arm hanging close to the body).
    """
    verts = world_vertices(mesh_obj)
    bones = armature_obj.data.bones
    matrix = armature_obj.matrix_world
    groups = mesh_obj.vertex_groups
    torso = [(BONE_NAMES[key], (matrix @ bones[BONE_NAMES[key]].head_local).z) for key in TORSO_KEYS]
    centre_x = float((matrix @ bones[BONE_NAMES["hips"]].head_local).x)
    neck_z = float((matrix @ bones[BONE_NAMES["neck"]].head_local).z)
    body = (verts[:, 2] > torso[0][1]) & (verts[:, 2] < neck_z) & ~np.asarray(skip, dtype=bool)
    arms = {groups[BONE_NAMES[f"{s}_{k}"]].index for s in ("left", "right")
            for k in ("shoulder", "upper_arm", "lower_arm", "hand") if BONE_NAMES[f"{s}_{k}"] in groups}
    for side in ("left", "right"):
        joint = np.array(matrix @ bones[BONE_NAMES[f"{side}_upper_arm"]].head_local)
        span = abs(joint[0] - centre_x)
        out = np.sign(joint[0] - centre_x) * (verts[:, 0] - centre_x) / max(span, 1e-6)
        # arms hanging close to the body put the sleeve's inner side medial of the joint too
        shaft = np.array(matrix @ bones[BONE_NAMES[f"{side}_hand"]].tail_local) - joint
        length = float(np.linalg.norm(shaft))
        along = (verts - joint) @ shaft / length**2
        distance = np.linalg.norm(verts - (joint + np.clip(along, 0.0, 1.0)[:, None] * shaft), axis=1)
        on_arm = (along > SHOULDER_BLEND) & (distance < ARM_HUG * length)
        # how far toward the body from the arm's line at the vertex's height; an arm held out (a
        # T-pose) has no height to go by -- every forearm vertex read as beside it and went to
        # the chest, tearing 36 cm as the arm came down -- so there it is the point along the arm
        if -shaft[2] > ARM_HANGS * length:
            t = np.clip((joint[2] - verts[:, 2]) / max(joint[2] - (joint[2] + shaft[2]), 1e-6), 0.0, 1.0)
        else:
            t = np.clip(along, 0.0, 1.0)
        line_x = joint[0] + t * shaft[0]
        inward = np.sign(joint[0] - centre_x) * (line_x - verts[:, 0]) / length
        beside = _smoothstep((inward - ARM_INWARD[0]) / (ARM_INWARD[1] - ARM_INWARD[0]))
        remove = {}
        for key, (zero, full) in (("shoulder", SHOULDER_SPAN), ("upper_arm", ARM_SPAN),
                                  ("lower_arm", ARM_SPAN), ("hand", ARM_SPAN)):
            group = groups.get(BONE_NAMES[f"{side}_{key}"])
            if group is not None:
                gone = _smoothstep((full - out) / (full - zero))
                if key != "shoulder":
                    gone = np.maximum(gone, beside)
                remove[group.index] = gone
        for index in np.nonzero(body & ~on_arm)[0]:
            i = int(index)
            weights = {g.group: g.weight for g in mesh_obj.data.vertices[i].groups}
            moved = sum(w * remove[g][i] for g, w in weights.items() if g in remove)
            if moved <= 1e-4:
                continue
            new = {g: w * (1.0 - remove[g][i]) if g in remove else w for g, w in weights.items()}
            # the torso (or whatever else, bar the arms, holds the vertex) takes over what the arm gives up
            rest = {g: w for g, w in weights.items() if g not in arms and w > 1e-6}
            if sum(rest.values()) <= 1e-6:
                rest = _torso_weights(groups, torso, verts[i, 2])
            total_rest = sum(rest.values())
            for g, w in rest.items():
                new[g] = new.get(g, 0.0) + moved * w / total_rest
            total = sum(new.values())
            for g in weights.keys() - {g for g, w in new.items() if w > 1e-6}:
                groups[g].remove([i])
            for g, w in new.items():
                if w > 1e-6:
                    groups[g].add([i], w / total, "REPLACE")


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _keep_legs_below_hips(mesh_obj, armature_obj):
    """Fade leg weights out above the hip joints, handing them to the hips.

    Bone heat leaks the thigh bones up coat panels and pouches hanging beside them, so those
    tear away from the waist when the spine bends.
    """
    bones = armature_obj.data.bones
    matrix = armature_obj.matrix_world
    groups = mesh_obj.vertex_groups
    legs = [BONE_NAMES[f"{side}_upper_leg"] for side in ("left", "right")]
    hip_z = float(np.mean([(matrix @ bones[name].head_local).z for name in legs]))
    height = mesh_obj.dimensions.z
    full, zero = hip_z + LEG_FADE[0] * height, hip_z + LEG_FADE[1] * height
    leg_indices = {groups[name].index for name in legs if name in groups}
    torso = [(BONE_NAMES[key], (matrix @ bones[BONE_NAMES[key]].head_local).z) for key in TORSO_KEYS]
    z = world_vertices(mesh_obj)[:, 2]
    for index in np.nonzero(z > full)[0]:
        vertex = mesh_obj.data.vertices[int(index)]
        keep = float(np.clip((zero - z[index]) / (zero - full), 0.0, 1.0))
        moved = 0.0
        for group, weight in [(g.group, g.weight) for g in vertex.groups if g.group in leg_indices]:
            groups[group].add([int(index)], weight * keep, "REPLACE")
            moved += weight * (1 - keep)
        if moved > 0:
            for group, weight in _torso_weights(groups, torso, z[index]).items():
                groups[group].add([int(index)], weight * moved, "ADD")


def _weighted_proxy(mesh_obj, armature_obj, rigid, solver="heat"):
    """A voxel remesh of the mesh (without rigid parts) with bone heat weights, or with
    `solver` "voxel" geodesic voxel weights (voxel_skin.py).

    Blender's heat solve sometimes fails outright on a remesh ("failed to find solution"),
    leaving every vertex unweighted, and whether it does depends on the voxel size. So other
    resolutions are tried until one weights nearly all of the proxy.
    """
    for voxels in PROXY_VOXEL_TRIES:
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
        proxy.data.remesh_voxel_size = mesh_obj.dimensions.z / voxels
        bpy.ops.object.voxel_remesh()

        if solver == "voxel":
            from . import voxel_skin
            voxel_skin.weigh(proxy, armature_obj)
            return proxy
        make_active(armature_obj)
        proxy.select_set(True)
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        weighted = sum(1 for v in proxy.data.vertices if any(g.weight > 0.01 for g in v.groups))
        if weighted >= PROXY_MIN_WEIGHTED * len(proxy.data.vertices):
            return proxy
        print(f"skinning proxy at {voxels} voxels: bone heat weighted only {weighted}/{len(proxy.data.vertices)}, retrying")
        proxy_mesh = proxy.data
        bpy.data.objects.remove(proxy)
        bpy.data.meshes.remove(proxy_mesh)
    raise RuntimeError("bone heat weighting failed at every proxy resolution")


def _split_sleeves(mesh_obj, armature_obj, cuttable):
    """Cut sleeves free where they share a surface with the torso; return each vertex's origin.

    Generators often fuse a sleeve's underside to the jacket's side. However that surface is
    weighted it stretches when the arm lifts, so it is split along the border between faces
    that follow the arm and faces that follow the torso, and each side keeps only its own
    bones. The cut is stitched with zero-area quads so it stretches closed instead of opening a
    hole. Only `cuttable` vertices (sleeve, below the shoulder) are split.
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
    # grown by a couple of rings of faces, so no face at the cut's end keeps mixed weights and
    # stretches into a sliver
    for _ in range(ZONE_GROW):
        grown = zone.copy()
        for f in bm.faces:
            if any(zone[v.index] for v in f.verts):
                for v in f.verts:
                    grown[v.index] = True
        zone = grown
    share_layer = bm.verts.layers.float.new("split_share")
    zone_layer = bm.verts.layers.int.new("split_zone")
    for v in bm.verts:
        v[share_layer] = share[v.index]
        v[zone_layer] = int(zone[v.index])
    _cut_along_half(bm, share_layer, zone_layer, origin_layer, deform)
    for f in bm.faces:
        f[side_layer] = int(np.mean([v[share_layer] for v in f.verts]) >= 0.5)
        if any(v[zone_layer] for v in f.verts) and f[side_layer]:
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
        forearm = {groups[n].index for n in (BONE_NAMES[f"{side}_{key}"] for side in ("left", "right")
                                               for key in ("lower_arm", "hand")) if n in groups}
        bones = armature_obj.data.bones
        shoulders = [(np.array(armature_obj.matrix_world @ bones[BONE_NAMES[f"{side}_upper_arm"]].head_local),
                      float(bones[BONE_NAMES[f"{side}_upper_arm"]].length + bones[BONE_NAMES[f"{side}_lower_arm"]].length))
                     for side in ("left", "right")]
        near_shoulder = lambda co: any(np.linalg.norm(np.array(armature_obj.matrix_world @ co) - head) < STITCH_REACH * length
                                       for head, length in shoulders)
        _stitch_cut(bm, origin_layer, side_layer, deform, forearm, near_shoulder)
    bm.faces.layers.int.remove(side_layer)
    bm.verts.layers.float.remove(share_layer)
    bm.verts.layers.int.remove(zone_layer)
    origin = np.array([v[origin_layer] for v in bm.verts])
    bm.verts.layers.int.remove(origin_layer)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return origin


def _cut_along_half(bm, share_layer, zone_layer, origin_layer, deform):
    """Split the zone's faces where the arm share crosses one half, so the cut runs smoothly
    across them instead of zigzagging along a coarse mesh's edges.

    New vertices sit where the share is exactly half, with weights and UVs interpolated; edges
    crossing within CUT_SNAP of an end are left for that end to carry the cut.
    """
    crossings = []
    for e in bm.edges:
        a, b = e.verts
        if not (a[zone_layer] or b[zone_layer]):
            continue
        sa, sb = a[share_layer] - 0.5, b[share_layer] - 0.5
        if sa * sb < 0:
            t = sa / (sa - sb)
            if CUT_SNAP < t < 1 - CUT_SNAP:
                crossings.append((e, a, b, t))
    on_cut = set()
    for e, a, b, t in crossings:
        wa, wb = dict(a[deform].items()), dict(b[deform].items())
        _edge, v = bmesh.utils.edge_split(e, a, t)
        v[share_layer] = 0.5
        v[zone_layer] = 1
        v[origin_layer] = (a if t < 0.5 else b)[origin_layer]
        for g in list(v[deform].keys()):
            del v[deform][g]
        for g in wa.keys() | wb.keys():
            w = (1 - t) * wa.get(g, 0.0) + t * wb.get(g, 0.0)
            if w > 1e-6:
                v[deform][g] = w
        on_cut.add(v)
    for f in list({f for v in on_cut for f in v.link_faces}):
        corners = [v for v in f.verts if v in on_cut]
        if len(corners) != 2:
            continue
        a, b = corners
        if any(e.other_vert(a) is b for e in a.link_edges):
            continue
        bmesh.utils.face_split(f, a, b)


def _stitch_cut(bm, origin_layer, side_layer, deform, forearm, near_shoulder=lambda co: True):
    """Join each cut edge on the arm side to its twin on the torso side with a quad.

    The quads have no area at rest, so nothing shows; when the arm moves they stretch across
    the opening instead of leaving a hole where the sleeve was cut from the torso. Only that
    strip stretches: the sleeve and the torso each keep their own bones. Cuts beside the
    forearm or hand (`forearm` groups), where an arm hanging close meets the hips, stay open:
    a stitch there would stretch into a long web when the arm lifts. So do cuts further than
    `near_shoulder` allows (see STITCH_REACH): there the cloth parts, showing its lining.
    """

    def on_forearm(vert):
        return sum(w for g, w in vert[deform].items() if g in forearm) > 0.5

    uv = bm.loops.layers.uv.active

    def open_edges(side):
        found = {}
        for e in bm.edges:
            if len(e.link_faces) == 1 and e.link_faces[0][side_layer] == side:
                found[frozenset((e.verts[0][origin_layer], e.verts[1][origin_layer]))] = e
        return found

    torso_edges = open_edges(0)
    for key, arm_edge in open_edges(1).items():
        torso_edge = torso_edges.get(key)
        if (torso_edge is None or len(key) < 2 or any(on_forearm(v) for v in arm_edge.verts)
                or not all(near_shoulder(v.co) for v in arm_edge.verts)):
            continue
        arm_face, torso_face = arm_edge.link_faces[0], torso_edge.link_faces[0]
        # the arm face runs u -> v along the edge; the stitch runs it the other way, so it faces
        # the same way as the cloth on both sides
        loop = next(l for l in arm_face.loops if l.edge == arm_edge)
        u_arm, v_arm = loop.vert, loop.link_loop_next.vert
        twin = {w[origin_layer]: w for w in torso_edge.verts}
        u_torso, v_torso = twin[u_arm[origin_layer]], twin[v_arm[origin_layer]]
        try:
            face = bm.faces.new((v_arm, u_arm, u_torso, v_torso))
        except ValueError:  # already joined
            continue
        face.material_index = arm_face.material_index
        face.smooth = arm_face.smooth
        face[side_layer] = 1
        if uv is not None:
            source = {l.vert: l[uv].uv.copy() for l in arm_face.loops}
            source.update({l.vert: l[uv].uv.copy() for l in torso_face.loops})
            for l in face.loops:
                l[uv].uv = source[l.vert]


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


def fill_unweighted(mesh_obj, max_influences=4, skip=None):
    """Weight vertices that have no weights, which would stay put while the bones move.

    Weights spread in from the weighted surface around them, with vertices sharing a spot
    counted as joined so separate pieces follow what they touch; pieces touching nothing
    weighted take the nearest weighted vertex's weights. `skip` vertices are left alone.
    Returns how many were filled.
    """
    mesh = mesh_obj.data
    groups = mesh_obj.vertex_groups
    empty = {v.index for v in mesh.vertices if not any(g.weight > 0 for g in v.groups)}
    if skip is not None:
        empty -= set(np.nonzero(skip)[0].tolist())
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
        weighted = [v.index for v in mesh.vertices if v.index not in empty and weights(v.index)]
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
    tris = triangles(mesh_obj.data)
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
        # distance from the bones themselves, which bend at the elbow, not the shoulder-hand line
        chain = [start] + [np.array(matrix @ bones[BONE_NAMES[f"{side}_{key}"]].head_local)
                           for key in ("lower_arm", "hand")] + [end]
        distance = _chain_distance(verts, chain)
        near = (distance < ARM_REACH * length) & (along > -0.1) & (along < 1.1)
        fade_in, fade_out = ARM_FADE[0] * length, ARM_FADE[1] * length
        faded = np.clip((fade_out - distance) / (fade_out - fade_in), 0.0, 1.0)
        tube_parts, hug_parts = [], []
        for part in np.unique(labels[~skip]):
            members = (labels == part) & ~skip
            on_arm = members & near
            hugging = members & (distance < ARM_COVER_REACH * length) & (along > -0.1) & (along < 1.1)
            covers = on_arm.sum() > 3 and hugging.any() and np.ptp(along[hugging]) >= ARM_COVERAGE
            hug_share = np.mean(distance[members] < ARM_HUG * length)
            hugs = hug_share >= HUG_SHARE
            if members.any():
                size = float(np.ptp(verts[members], axis=0).max())
                centre = verts[members].mean(axis=0)
                c_along = float((centre - start) @ shaft / length**2)
                c_dist = float(_chain_distance(centre[None, :], chain)[0])
                if not hugs:
                    hugs = size <= SMALL_PART * length and c_dist <= SMALL_REACH * length and -0.1 < c_along < 1.1
                    hugs = hugs or (size <= ARMOUR_PART * length and c_dist <= ARMOUR_REACH * length
                                    and hug_share >= ARMOUR_HUG_SHARE and ARMOUR_FROM < c_along < 1.1)
                # buckles and straps on the chest sit close to where the arm starts, not on it
                hugs = hugs and c_along >= HUG_FROM
            keep = faded if covers or hugs else np.zeros(len(verts))
            if covers and not hugs:
                tube_parts.append((members, _part_sleeve_end(verts, tris, members, start, shaft)))
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
            radius = _tube_radius(verts, along, distance, sleeve & near, start, shaft)
            tube = _inside_tube(verts, along, distance, sleeve & near, start, shaft)
            # the hanging forearm rests beside a jacket's hip skirt: the tube's margin reaches it and
            # bone heat fuses it to the sleeve over the gap, and once cut free it rode the forearms
            # as strips (model A's OBJ); it faces the arm, which no sleeve does
            against = _lying_against(mesh_obj, verts, chain, radius, along, joints_along)
            # Cloth draped over the top, outside or underside of the arm (a mantle, a bell sleeve
            # joined to the robe) follows it even where it doesn't wrap round it: to the wrist, or
            # only along the upper arm for long cloth reaching below the hips (a cape, a coat).
            # the torso is only beside the upper arm: past the elbow the inner side is sleeve too
            away = np.maximum(_away_from_torso(verts, start, shaft),
                              np.clip((along - joints_along[0]) / OVER_ARM_FADE, 0.0, 1.0))
            # a bell sleeve flares past the fitted tube; the part away from the torso still hangs
            # from the arm, or it sticks out when the arm comes in to the body
            outside = np.clip(-_facing_torso(verts, start, shaft) / OUTSIDE_FOLLOW, 0.0, 1.0)
            # bone heat splits a sleeve from what it is joined to about halfway between the bones;
            # where it put a vertex clearly on the arm, that stands
            heat = np.zeros(len(verts))
            for v in mesh_obj.data.vertices:
                heat[v.index] = sum(g.weight for g in v.groups if g.group in arm_indices)
            heat = np.where(heat >= HEAT_TRUST, heat, 0.0)
            hips_z = torso[0][1]
            for members, sleeve_end in tube_parts:
                until = joints_along[1] if verts[members, 2].min() > hips_z else joints_along[0]
                over_arm = np.clip((until - along) / OVER_ARM_FADE, 0.0, 1.0) * away
                share = np.maximum(tube, outside * np.clip((until - along) / OVER_ARM_FADE, 0.0, 1.0))
                share = share * np.where(along > sleeve_end + SLEEVE_STEP, over_arm, 1.0)
                share = np.maximum(share, heat) * (1.0 - against)
                region = members & (near | _has_any(mesh_obj, arm_indices, members))
                _set_arm_weights(mesh_obj, groups, verts, torso, arm_indices, names, region, share, along, joints_along)
                # cut the sleeve free of the torso wherever it follows the arm
                # Cut the sleeve free of the torso only past the elbow, where cuffs and hanging sleeve
                # ends fused to the hips would otherwise be dragged along by the hand. Round the
                # upper arm and armpit a cut opens holes as soon as the arm lifts, so there the
                # weights blend instead and the cloth stretches a little.
                cuttable |= region & (along > joints_along[0] - JOINT_BLEND)
    for arm_indices, names, along, joints_along, hugged in straps:
        _attach_straps(mesh_obj, groups, verts, torso, labels, skip, arm_indices, names, along, joints_along, hugged)
    return cuttable


def _chain_distance(points, chain):
    """Distance from each point to the nearest of the segments joining `chain`'s points in turn."""
    return _chain_nearest(points, chain)[0]


def _chain_nearest(points, chain):
    """(distance, nearest point) from each point to the segments joining `chain`'s points in turn."""
    best = np.full(len(points), np.inf)
    foot = np.zeros_like(points)
    for a, b in zip(chain[:-1], chain[1:]):
        segment = b - a
        t = np.clip((points - a) @ segment / max(segment @ segment, 1e-12), 0.0, 1.0)
        on = a + t[:, None] * segment
        distance = np.linalg.norm(points - on, axis=1)
        closer = distance < best
        best[closer], foot[closer] = distance[closer], on[closer]
    return best, foot


def _torso_weights(groups, torso, z):
    """{group index: weight} for a vertex at height z that only the torso should move.

    Blends between the two torso bones whose middles are above and below it, so parts
    handed back to the torso don't get a step in their weights at each bone boundary.
    """
    heads = [h for _name, h in torso]
    ends = heads[1:] + [heads[-1] + (heads[-1] - heads[-2])]
    middles = [(h + e) / 2 for h, e in zip(heads, ends)]
    t = np.interp(z, middles, np.arange(len(torso)))
    low = int(np.floor(t))
    high = min(low + 1, len(torso) - 1)
    out = {}
    for k, w in ((low, 1 - (t - low)), (high, t - low)):
        if w > 1e-6:
            name = torso[k][0]
            group = groups.get(name) or groups.new(name=name)
            out[group.index] = out.get(group.index, 0.0) + float(w)
    return out


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
            kept = _torso_weights(groups, torso, verts[index, 2])
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


def _away_from_torso(verts, start, shaft):
    """Per vertex, 1 on the arm's top or outer side fading to 0 on the side towards the torso."""
    facing = _facing_torso(verts, start, shaft)
    return np.clip((OVER_ARM_INWARD[1] - facing) / (OVER_ARM_INWARD[1] - OVER_ARM_INWARD[0]), 0.0, 1.0)


def _facing_torso(verts, start, shaft):
    """Per vertex, the cosine between its direction from the arm's axis and the torso's direction."""
    _axis, side, up = _arm_frame(start, shaft)
    offset = verts - start
    inward = np.array([-np.sign(start[0]) or -1.0, 0.0, 0.0])
    inward_2d = np.array([inward @ side, inward @ up])
    inward_2d /= max(np.linalg.norm(inward_2d), 1e-9)
    radial = np.stack([offset @ side, offset @ up], axis=1)
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    return radial @ inward_2d


def _inside_tube(verts, along, distance, sample, start, shaft):
    """Per vertex, 1 inside the arm's tube fading to 0 outside it (see ARM_TUBE_PERCENTILE)."""
    r = _tube_radius(verts, along, distance, sample, start, shaft)
    if r is None:
        return np.ones(len(verts))
    return np.clip((TUBE_FADE[1] * r - distance) / ((TUBE_FADE[1] - TUBE_FADE[0]) * r), 0.0, 1.0)


def _tube_radius(verts, along, distance, sample, start, shaft):
    """Per vertex, the sleeve's fitted radius at its step along the arm, or None without a sleeve
    sample: the ARM_TUBE_PERCENTILE distance of the `sample` surface on the side facing away from
    the torso, where nothing else rests against the arm."""
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
        return None
    known = np.array(sorted(radius))
    length = float(np.linalg.norm(shaft))
    return np.minimum(np.interp(step, known, [radius[k] for k in known]), ARM_TUBE_MAX * length)


def _lying_against(mesh_obj, verts, chain, radius, along, joints_along):
    """Per vertex, 1 for surface past the forearm's fitted sleeve that faces the arm, fading to 0.

    A sleeve faces away from its arm all round, however wide it flares (model B's coat sleeves
    reach 12 cm on their inner side). Cloth the hanging forearm rests beside -- a jacket's hip
    skirt, a coat's front -- faces the arm from outside. Only a hanging forearm: an arm held out
    has nothing resting beside it, and a hand's curled fingers face each other. Off when the
    sleeve's own surface mostly faces the arm (normals pointing inward).
    """
    shaft = chain[-1] - chain[0]
    if radius is None or -shaft[2] <= ARM_HANGS * np.linalg.norm(shaft):
        return np.zeros(len(verts))
    normals = np.empty(len(mesh_obj.data.vertices) * 3)
    mesh_obj.data.vertices.foreach_get("normal", normals)
    normals = normals.reshape(-1, 3) @ np.array(mesh_obj.matrix_world)[:3, :3].T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    distance, foot = _chain_nearest(verts, chain)
    facing = np.einsum("ij,ij->i", normals, (verts - foot) / np.maximum(distance, 1e-9)[:, None])
    elbow, wrist = joints_along
    forearm = (np.clip((along - elbow + JOINT_BLEND) / JOINT_BLEND, 0.0, 1.0)
               * np.clip((wrist + JOINT_BLEND - along) / JOINT_BLEND, 0.0, 1.0))
    fitted = (distance <= radius) & (forearm > 0)
    if not fitted.any() or np.median(facing[fitted]) <= 0:
        return np.zeros(len(verts))
    against = _smoothstep((AGAINST_FACING[0] - facing) / (AGAINST_FACING[0] - AGAINST_FACING[1]))
    beyond = _smoothstep((distance / radius - AGAINST_BEYOND[0]) / (AGAINST_BEYOND[1] - AGAINST_BEYOND[0]))
    against = against * beyond * forearm
    # the rest of that cloth -- its hem, the far side of a panel with some thickness -- faces
    # elsewhere; it is reached over the surface without coming back within the sleeve's radius
    weld, a, b = welded_edges(mesh_obj, verts)
    room = beyond * forearm
    for _ in range(AGAINST_SPREAD):
        spot = np.zeros(len(verts))
        np.maximum.at(spot, weld, against)
        reach = spot.copy()
        np.maximum.at(reach, a, spot[b])
        np.maximum.at(reach, b, spot[a])
        against = np.maximum(against, reach[weld] * room)
    return against


def _attach_straps(mesh_obj, groups, verts, torso, labels, skip, arm_indices, names, along, joints_along, hugged):
    """Small parts hanging off a cuff or glove (straps, buckles) follow the arm.

    A small part (see SMALL_PART) within STRAP_GAP of a part that hugs the arm is attached
    to it, even when it sits too far from the arm's axis to count as hugging it itself, and so
    in turn is one within STRAP_GAP of an attached part (studs along a strap).
    """
    attached = hugged.copy()
    candidates = [part for part in np.unique(labels[~skip & ~hugged])
                  if (members := (labels == part) & ~skip).sum() <= STRAP_MAX_VERTS
                  and np.ptp(verts[members], axis=0).max() <= STRAP_SIZE]
    while attached.any() and candidates:
        tree = KDTree(int(attached.sum()))
        for i, p in enumerate(verts[attached]):
            tree.insert(p, i)
        tree.balance()
        added = []
        for part in candidates:
            members = (labels == part) & ~skip
            gap = min(tree.find(p)[2] for p in verts[members][:: max(1, int(members.sum()) // 30)])
            if gap <= STRAP_GAP:
                _set_arm_weights(mesh_obj, groups, verts, torso, arm_indices, names, members,
                                 np.ones(len(verts)), along, joints_along)
                attached |= members
                added.append(part)
        if not added:
            break
        candidates = [part for part in candidates if part not in added]


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
            others, total_others = _torso_weights(groups, torso, verts[i, 2]), 1.0
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


def _part_sleeve_end(verts, tris, members, start, shaft):
    """_sleeve_end over points spread evenly on the part's surface near the arm.

    Vertices alone are too sparse on a low-poly sleeve to fill the sectors round the arm.
    """
    part = tris[members[tris].all(axis=1)]
    if not len(part):
        return 0.0
    points = sample_surface(verts, part, min(SLEEVE_SAMPLES, SLEEVE_SAMPLES_PER_TRIANGLE * len(part)))
    length = float(np.linalg.norm(shaft))
    along = (points - start) @ shaft / length**2
    distance = np.linalg.norm(points - (start + np.clip(along, 0.0, 1.0)[:, None] * shaft), axis=1)
    near = (distance < ARM_REACH * length) & (along > -0.1) & (along < 1.1)
    return _sleeve_end(points[near], along[near], start, shaft)


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


def make_head_rigid(mesh_obj, armature_obj):
    """For hats, helmets and solid heads (no hair): every head piece moves with the head alone.

    Pieces mostly on the head or centred above the neck go fully onto Head, so a hat brim or
    the bottom of a round head doesn't stay behind when the head turns. Other pieces lose
    their Head weight below the neck to the chest, so collars don't twist with the head.
    Returns the mask of vertices put fully on the head.
    """
    groups = mesh_obj.vertex_groups
    head = groups.get(BONE_NAMES["head"]) or groups.new(name=BONE_NAMES["head"])
    neck = armature_obj.data.bones[BONE_NAMES["neck"]]
    chest = groups.get(neck.parent.name) or groups.new(name=neck.parent.name)
    neck_z = (armature_obj.matrix_world @ neck.head_local).z
    centre_x = (armature_obj.matrix_world @ neck.head_local).x
    # arms held out (a T-pose) reach the neck's height: a sleeve or glove centred there is not headwear
    reach = abs((armature_obj.matrix_world @ armature_obj.data.bones[BONE_NAMES["left_upper_arm"]].head_local).x - centre_x)
    verts = world_vertices(mesh_obj)
    n = len(verts)
    head_w, total = np.zeros(n), np.zeros(n)
    for v in mesh_obj.data.vertices:
        for g in v.groups:
            total[v.index] += g.weight
            if g.group == head.index:
                head_w[v.index] = g.weight
    share = head_w / np.maximum(total, 1e-9)
    labels = pieces(mesh_obj, verts)
    rigid = np.zeros(n, dtype=bool)
    for label in np.unique(labels):
        members = labels == label
        above = verts[members, 2].mean() > neck_z and abs(verts[members, 0].mean() - centre_x) < reach
        # no piece reaching a head's height under the neck is headwear: a body that is one surface
        # with its head (a tester's pirate) has most of its vertices in the head, so its mean is
        # above the neck and most of its weight the head's, and it went on the head whole: arms,
        # coat and legs
        hangs = verts[members, 2].min() < neck_z - (float(verts[:, 2].max()) - neck_z)
        if (share[members].mean() > MOSTLY_HEAD or above) and not hangs:
            rigid |= members
    low = ~rigid & (verts[:, 2] < neck_z) & (head_w > 0)
    for index in np.nonzero(rigid | low)[0]:
        i = int(index)
        vertex = mesh_obj.data.vertices[i]
        if rigid[i]:
            for g in [g.group for g in vertex.groups if g.group != head.index]:
                groups[g].remove([i])
            head.add([i], 1.0, "REPLACE")
        else:
            head.remove([i])
            chest.add([i], float(head_w[i]), "ADD")
    return rigid


def weight_head(mesh_obj, armature_obj, face_mask, labels=None, chin_z=None, behind=None):
    """Face vertices above the head joint follow the head fully, blending into the neck below.

    With `labels` (part per vertex), separate parts lying wholly above the joint (eye
    highlights, clips, bangs) move whatever body weight they have onto the head; weights for
    hair spring bones are kept. With `behind` (m), only the face in front of the head joint
    follows it fully, fading to the weights it has over that distance behind the joint: for a
    face that is one surface with the back of the head and the neck.
    """
    groups = mesh_obj.vertex_groups
    head = groups.get(BONE_NAMES["head"]) or groups.new(name=BONE_NAMES["head"])
    head_z = (armature_obj.matrix_world @ armature_obj.data.bones[BONE_NAMES["head"]].head_local).z
    if chin_z is not None:
        head_z = min(head_z, chin_z)  # the face below the pivot, down to the chin, still follows the head
    verts = world_vertices(mesh_obj)
    share = np.clip((verts[:, 2] - head_z) / HEAD_BLEND, 0.0, 1.0)
    if behind is not None:
        head_y = (armature_obj.matrix_world @ armature_obj.data.bones[BONE_NAMES["head"]].head_local).y
        share *= np.clip(1.0 - (verts[:, 1] - head_y) / behind, 0.0, 1.0)
    for index in np.nonzero(face_mask & (share > 0))[0]:
        i = int(index)
        vertex = mesh_obj.data.vertices[i]
        others = {g.group: g.weight for g in vertex.groups if g.group != head.index}
        total = sum(others.values())
        target = share[i]
        if behind is not None:  # what the head holds already stays: only the rest is handed over
            held = sum(g.weight for g in vertex.groups if g.group == head.index)
            target = held + (1.0 - held) * target
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


# The height (m) about the head joint over which the rest of the head is handed to it.
COLUMN_BLEND = 0.06


def follow_head_column(mesh_obj, armature_obj, chin_z=None, skip=None):
    """Whatever stands above the head joint, between the shoulders, follows the head alone.

    `weight_head` does this for the face part. On a model whose jaw, beard, hair and the back of
    its skull are one surface with the body (a tester's pirate: hat, ears and eyes a piece of their
    own on the head, the rest of the head in the body's piece at 49 to 86% on the head by bone
    heat) the top half of the head turned and the lower half lagged behind it. Vertices from
    half of COLUMN_BLEND above the joint (or the chin, when lower) take the head fully, fading in
    from as far below it; what the head holds already stays. Only within the shoulder joints' width of the neck:
    hands and sleeves held up beside the head are not it. `skip` masks vertices left as they are
    (hair, which gets its own chains). Returns the number of vertices changed.
    """
    groups = mesh_obj.vertex_groups
    head = groups.get(BONE_NAMES["head"]) or groups.new(name=BONE_NAMES["head"])
    world = armature_obj.matrix_world
    bones = armature_obj.data.bones
    joint = world @ bones[BONE_NAMES["head"]].head_local
    head_z = joint.z if chin_z is None else min(joint.z, chin_z)
    reach = abs((world @ bones[BONE_NAMES["left_upper_arm"]].head_local).x - joint.x)
    verts = world_vertices(mesh_obj)
    # hair and a beard hang on past the jaw: handed over across COLUMN_BLEND about the joint, not
    # the face's 2 cm above it (the pirate's hair beside his jaw: 90 edges over 3 cm on a head turn)
    share = np.clip((verts[:, 2] - head_z + COLUMN_BLEND / 2) / COLUMN_BLEND, 0.0, 1.0)
    share[np.abs(verts[:, 0] - joint.x) >= reach] = 0.0
    if skip is not None:
        share[np.asarray(skip, dtype=bool)[:len(share)]] = 0.0
    changed = 0
    for index in np.nonzero(share > 0)[0]:
        i = int(index)
        vertex = mesh_obj.data.vertices[i]
        others = {g.group: g.weight for g in vertex.groups if g.group != head.index}
        total = sum(others.values())
        if total <= 1e-6:
            continue
        held = sum(g.weight for g in vertex.groups if g.group == head.index)
        target = held + total * float(share[i])
        for g, w in others.items():
            groups[g].add([i], w * (1.0 - float(share[i])), "REPLACE")
        head.add([i], target, "REPLACE")
        changed += 1
    return changed


def make_active(obj):
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
