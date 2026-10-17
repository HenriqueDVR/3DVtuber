"""Take the skeleton and skin weights from a rig made elsewhere (Tripo's auto-rig) onto our source.

Tripo's rig of model C placed joints our silhouette reading couldn't (wings touch her arms in
every slice) and weighted a whole body with fingers in one go; what it lacks is everything a
VTuber needs past that, and it hands wings to the arms and tails to the thighs. So the rig is
adopted where it is good and the pipeline does the rest: its bones become our armature (our bone
names, its positions) and its weights move onto our own source mesh, which keeps the texture,
face patch and piece labels the manifest refers to. Everything after skinning then runs as usual.

The rig must be the same generation as the source (same shape; vertex order may differ: Tripo's
FBX welds UV seams, 27 242 vertices against the GLB's 33 715). Its FBX, not its GLB: the GLB
export carries broken inverse bind matrices and every bone imports at NaN.
"""

import re
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import compat, fingers, inpaint, meshdata, normalize
from .rig import BONE_NAMES, FAR_BONES, bone_steps, make_active

# VRoid / Tripo "VRM 1.0 humanoid" names -> ours.
_TORSO = {"C_Hips": "hips", "C_Spine": "spine", "C_Chest": "chest", "C_UpperChest": "upper_chest", "C_Neck": "neck",
          "C_Head": "head"}
_LIMBS = {"Shoulder": "shoulder", "UpperArm": "upper_arm", "LowerArm": "lower_arm", "Hand": "hand",
          "UpperLeg": "upper_leg", "LowerLeg": "lower_leg", "Foot": "foot"}
# Bones of the arm and hand, after a VRM humanoid side prefix (LeftUpperArm, RightIndexProximal...).
_ARM_WORDS = ("Shoulder", "UpperArm", "LowerArm", "Hand", "Thumb", "Index", "Middle", "Ring", "Little")
# keep_arms_off_head: arm weight on the head up to this share is a leak and goes, by this share it
# stays (a piece really riding the arm), and it goes fully this far (m) above the Head joint.
HEAD_LEAK = 0.15
HEAD_KEEP = 0.3
HEAD_FADE = 0.03
# ... and none of it within this distance (m) of a shoulder joint, all of it this much further out:
# model A's scalp squash sat 22 cm from the shoulder, model C's twin-tail on the shoulder 14 cm.
SHOULDER_CLEAR = 0.12
SHOULDER_FADE = 0.06
_FINGER_NAMES = {"Thumb": "thumb", "Index": "index", "Middle": "middle", "Ring": "ring", "Little": "little"}
# Mixamo: mixamorig:Hips, mixamorig:LeftArm, mixamorig:LeftHandIndex1 (some exports drop the prefix).
_MIXAMO = ({"Hips": "hips", "Spine": "spine", "Spine1": "chest", "Spine2": "upper_chest", "Neck": "neck", "Head": "head"},
           {"Shoulder": "shoulder", "Arm": "upper_arm", "ForeArm": "lower_arm", "Hand": "hand",
            "UpLeg": "upper_leg", "Leg": "lower_leg", "Foot": "foot"},
           {"HandThumb": "thumb", "HandIndex": "index", "HandMiddle": "middle", "HandRing": "ring", "HandPinky": "little"})
# Character Creator and AccuRig: CC_Base_Hip, CC_Base_L_Upperarm, CC_Base_L_Index1.
_CREATOR = ({"Hip": "hips", "Waist": "spine", "Spine01": "chest", "Spine02": "upper_chest", "NeckTwist01": "neck", "Head": "head"},
            {"Clavicle": "shoulder", "Upperarm": "upper_arm", "Forearm": "lower_arm", "Hand": "hand",
             "Thigh": "upper_leg", "Calf": "lower_leg", "Foot": "foot"},
            {"Thumb": "thumb", "Index": "index", "Mid": "middle", "Ring": "ring", "Pinky": "little"})
# MMD models as the Cats plugin renames them, which is how most game models are passed round as FBX:
# Hips, Spine, Chest, "Left arm", "Left elbow", "Left wrist", "Left leg", Thumb0_L, IndexFinger1_L.
# A tester's (417 bones, 342 of them weighted) was built as if it had no rig at all.
_CATS = ({"Hips": "hips", "Spine": "spine", "Chest": "chest", "Upper Chest": "upper_chest", "Neck": "neck", "Head": "head"},
         {"shoulder": "shoulder", "arm": "upper_arm", "elbow": "lower_arm", "wrist": "hand",
          "leg": "upper_leg", "knee": "lower_leg", "ankle": "foot"},
         {"Thumb": "thumb", "IndexFinger": "index", "MiddleFinger": "middle", "RingFinger": "ring", "LittleFinger": "little"})
# ... and under their own Japanese names, as mmd_tools imports a .pmx. The legs' weights are on the
# "D" bones beside the legs (左足D), which hang from the hips: each lends to the leg it doubles.
_PMX = ({"下半身": "hips", "上半身": "spine", "上半身2": "chest", "上半身3": "upper_chest", "首": "neck", "頭": "head"},
        {"肩": "shoulder", "腕": "upper_arm", "ひじ": "lower_arm", "手首": "hand", "足": "upper_leg", "ひざ": "lower_leg", "足首": "foot"},
        {"親指": "thumb", "人指": "index", "中指": "middle", "薬指": "ring", "小指": "little"})
_PMX_DOUBLES = ("足", "ひざ", "足首")
KNOWN = ("VRoid/Tripo (J_Bip_*), Mixamo, Character Creator/AccuRig (CC_Base_*), MMD (Cats' English names or Japanese), "
         "VRM humanoid names")
# Bones a humanoid needn't have: one is put between its neighbours, holding no weight (see `read`).
OPTIONAL = ("upper_chest",)
# The source and the rig's mesh must agree this closely (95th percentile, m) once both are normalized.
MATCH = 0.01
# The face follows the head alone in front of the Head joint, fading to the rig's own weights over
# this distance (m) behind it (build.py, rig.weight_head): the nape keeps the rig's bend.
FACE_BEHIND = 0.03
# A vertex counts as on a bone (see reaching_across) holding at least this share of it.
FAR_SHARE = 0.02


def names(found=None):
    """{rig bone name: our bone name} under the naming the rig uses: whichever known one
    (KNOWN) names the most of `found`, the rig's bone names. Without `found`, Tripo's.

    Only Tripo's has been through real builds (model C), and Cats' through a tester's game model;
    the others are read by name alone and tested on a renamed synthetic rig. A bone no table names (a twist or helper bone) lends its
    weights to the nearest named bone up its chain: see `read`.
    """
    if found is None:
        return _vroid()
    found = set(found)
    prefix = next((m.group(0) for m in (re.match(r"mixamorig\d*:", name) for name in found) if m), "")
    return max((_vroid(), _numbered(prefix, "{side}", ("Left", "Right"), *_MIXAMO),
                _numbered("CC_Base_", "{side}_", ("L", "R"), *_CREATOR), _cats(), _pmx(), _humanoid()),
               key=lambda table: len(found & set(table)))


def _vroid():
    # toes and eye helpers go to their parent
    table = {f"J_Bip_{k}": BONE_NAMES[v] for k, v in _TORSO.items()}
    for letter, side in (("L", "left"), ("R", "right")):
        table.update({f"J_Bip_{letter}_{k}": BONE_NAMES[f"{side}_{v}"] for k, v in _LIMBS.items()})
        for vroid, finger in _FINGER_NAMES.items():
            for number, segment in enumerate(fingers._segments(finger), 1):
                table[f"J_Bip_{letter}_{vroid}{number}"] = fingers.bone_name(side, finger, segment)
        table[f"J_Bip_{letter}_ToeBase"] = BONE_NAMES[f"{side}_foot"]
        table[f"J_Adj_{letter}_FaceEye"] = BONE_NAMES["head"]
    table["Root"] = BONE_NAMES["hips"]
    return table


def _numbered(prefix, side_form, sides, torso, limbs, finger_names):
    """A table for rigs naming limbs <prefix><side><part> and finger joints 1-3 from the palm out."""
    table = {prefix + k: BONE_NAMES[v] for k, v in torso.items()}
    for mark, side in zip(sides, ("left", "right")):
        start = prefix + side_form.format(side=mark)
        table.update({start + k: BONE_NAMES[f"{side}_{v}"] for k, v in limbs.items()})
        for stem, finger in finger_names.items():
            for number, segment in enumerate(fingers._segments(finger), 1):
                table[f"{start}{stem}{number}"] = fingers.bone_name(side, finger, segment)
    return table


def _cats():
    table = {k: BONE_NAMES[v] for k, v in _CATS[0].items()}
    for side in ("left", "right"):
        table.update({f"{side.capitalize()} {k}": BONE_NAMES[f"{side}_{v}"] for k, v in _CATS[1].items()})
        for stem, finger in _CATS[2].items():
            # the thumb counts from 0, the others from 1
            for number, segment in enumerate(fingers._segments(finger), 0 if finger == "thumb" else 1):
                table[f"{stem}{number}_{side[0].upper()}"] = fingers.bone_name(side, finger, segment)
    return table


def _pmx():
    table = {k: BONE_NAMES[v] for k, v in _PMX[0].items()}
    for mark, side in (("左", "left"), ("右", "right")):
        table.update({mark + k: BONE_NAMES[f"{side}_{v}"] for k, v in _PMX[1].items()})
        table.update({f"{mark}{k}D": BONE_NAMES[f"{side}_{_PMX[1][k]}"] for k in _PMX_DOUBLES})
        for stem, finger in _PMX[2].items():
            for number, segment in zip("０１２" if finger == "thumb" else "１２３", fingers._segments(finger)):
                table[f"{mark}{stem}{number}"] = fingers.bone_name(side, finger, segment)
    return table


def _lends_only(name, table):
    """Bones that map to one of ours only to hand it their weights: the skeleton's joint is another's."""
    return (name == "Root" or name.startswith("J_Adj") or name.endswith("ToeBase")
            or (name.endswith("D") and name[:-1] in table and name[:1] in "左右"))


def _humanoid():
    # a rig already under VRM's humanoid names, which are ours: UniVRM's, or one this pipeline exported
    table = {name: name for name in BONE_NAMES.values()}
    for side in ("left", "right"):
        for finger in fingers.FINGERS:
            for segment in fingers._segments(finger):
                table[fingers.bone_name(side, finger, segment)] = fingers.bone_name(side, finger, segment)
    return table


def required(table, found, path):
    """Raise ValueError unless the rig has every body bone under names the table knows."""
    have = {table[name] for name in found if name in table}
    missing = sorted(set(BONE_NAMES.values()) - have - {BONE_NAMES[key] for key in OPTIONAL})
    if missing:
        raise ValueError(f"{path}: the rig has no {', '.join(missing)} under any bone naming the app knows ({KNOWN}). "
                         "Clear the rig to let the app place the skeleton itself")


def load(path, height=1.6):
    """Read a rigged file without leaving it in the scene; see `read` for what comes back."""
    before = set(bpy.data.objects)
    if str(path).lower().endswith(".fbx"):
        compat.import_fbx(path)
    else:
        bpy.ops.import_scene.gltf(filepath=str(path))
    imported = set(bpy.data.objects) - before
    try:
        armature = next(o for o in imported if o.type == "ARMATURE")
        meshes = [o for o in imported if o.type == "MESH" and o.find_armature() == armature]
        return read(armature, meshes, path, height)
    finally:
        for obj in list(set(bpy.data.objects) - before):
            bpy.data.objects.remove(obj, do_unlink=True)


def read(armature, meshes, path, height=1.6):
    """A rig and the meshes skinned to it, as the build takes them. Normalizes the meshes.

    Returns (bones, surface, weights): bones as {our name: (head, tail, parent our name)} in the
    normalized frame; surface as (vertices, triangles) of the rig's mesh at rest there; weights as
    an (vertices, bones) array over `order`, the list of our bone names weighted. `path` names the
    rig in errors.
    """
    table = names(bone.name for bone in armature.data.bones)
    required(table, [bone.name for bone in armature.data.bones], path)
    world = np.array(armature.matrix_world)
    raw, lends = {}, {}
    for bone in armature.data.bones:
        # a bone the table doesn't name lends its weights to the nearest one up its chain that is
        # named: Character Creator's forearm twist bones hold half the forearm
        up = bone
        while up is not None and up.name not in table:
            up = up.parent
        lends[bone.name] = table[up.name] if up is not None else BONE_NAMES["hips"]
        ours = table.get(bone.name)
        # Root, the eye helpers and the toes only lend their weights to the bone they map to
        if ours is None or _lends_only(bone.name, table):
            continue
        head, tail = ((world @ np.array([*p, 1.0]))[:3] for p in (bone.head_local, bone.tail_local))
        parent = bone.parent
        while parent is not None and (parent.name not in table or table[parent.name] == ours):
            parent = parent.parent
        # an MMD skeleton hangs the upper body beside the hips (下半身), not from them
        root = None if ours == BONE_NAMES["hips"] else BONE_NAMES["hips"]
        raw[ours] = (head, tail, table[parent.name] if parent is not None else root)
    chest, upper, neck = (BONE_NAMES[key] for key in ("chest", "upper_chest", "neck"))
    if upper not in raw and chest in raw and neck in raw:
        # VRM's upper chest is optional and MMD rigs have none; the steps after this name it, so one
        # stands halfway from chest to neck with no weight of its own
        middle = (raw[chest][0] + raw[neck][0]) / 2
        raw[upper] = (middle, raw[neck][0], chest)
        raw[neck] = (*raw[neck][:2], upper)
    if np.isnan(np.array([h for h, _, _ in raw.values()])).any():
        raise ValueError(f"{path}: the rig's joints don't import (NaN); Tripo's GLB is broken, use its FBX")

    verts, tris = [], []
    for m in meshes:
        offset = sum(len(v) for v in verts)
        verts.append(meshdata.world_vertices(m))
        tris.append(meshdata.triangles(m.data) + offset)
    global FILE_PARTS
    FILE_PARTS = np.repeat(np.arange(len(tris)), [len(t) for t in tris])  # per triangle; see parts_on
    verts, tris = np.concatenate(verts), np.concatenate(tris)
    order = sorted({lends[g.name] for m in meshes for g in m.vertex_groups if g.name in lends})
    column = {name: k for k, name in enumerate(order)}
    weights = np.zeros((len(verts), len(order)))
    row = 0
    for m in meshes:
        index = {g.index: column[lends[g.name]] for g in m.vertex_groups if g.name in lends}
        for v in m.data.vertices:
            for g in v.groups:
                if g.group in index:
                    weights[row + v.index, index[g.group]] += g.weight
        row += len(m.data.vertices)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)

    # the same scale and offset normalize.prepare gives this mesh, for bones and surface alike
    for m in meshes:
        matrix = m.matrix_world.copy()
        m.parent = None
        m.matrix_world = matrix
        m.modifiers.clear()
    prepared = normalize.prepare(meshes, height=height)
    after = meshdata.world_vertices(prepared)
    if not np.allclose(np.ptp(verts, axis=0) / np.ptp(verts[:, 2]), np.ptp(after, axis=0) / np.ptp(after[:, 2]), atol=1e-3):
        raise ValueError(f"{path}: normalizing turned the rig's mesh; can't carry its bones over by scale alone")
    scale = np.ptp(after[:, 2]) / np.ptp(verts[:, 2])
    shift = after.min(axis=0) - scale * verts.min(axis=0)
    bones = {k: (scale * h + shift, scale * t + shift, p) for k, (h, t, p) in raw.items()}
    return bones, (scale * verts + shift, tris), (weights, order)


# Experiment (OC2VRM_PARTS=1, off by default): the mesh object each triangle of the rig's surface came
# in, when the file came in parts (Tripo's segmentation). Set by `read`.
FILE_PARTS = None


def parts_on(mesh_obj, surface):
    """A part number per vertex of `mesh_obj`: that of the rig's triangle nearest each polygon's centre.

    None for a file of one mesh. Assumes the mesh and the rig's surface are the same model in the same
    frame (as transfer_weights does), and that no vertex is shared by two parts (nothing welded yet).
    """
    if FILE_PARTS is None or FILE_PARTS.max() < 1:
        return None
    from mathutils.kdtree import KDTree
    verts, tris = surface
    centres = verts[tris].mean(axis=1)
    tree = KDTree(len(centres))
    for index, centre in enumerate(centres):
        tree.insert(centre, index)
    tree.balance()
    own = meshdata.world_vertices(mesh_obj)
    parts = np.full(len(own), -1, dtype=np.int64)
    for poly in mesh_obj.data.polygons:
        corners = list(poly.vertices)
        parts[corners] = FILE_PARTS[tree.find(own[corners].mean(axis=0))[1]]
    return parts


def parts_of(path, height=1.6):
    """Experiment (OC2VRM_PARTS_FILE): the same model segmented otherwise, read for its parts alone.

    Returns its surface (vertices, triangles) in the normalized frame and sets FILE_PARTS, as `read`
    does for the rig's file; nothing of it stays in the scene. Assumes the same pose as the model built.
    """
    global FILE_PARTS
    before = set(bpy.data.objects)
    normalize.import_model(Path(path))
    meshes = [o for o in set(bpy.data.objects) - before if o.type == "MESH"]
    try:
        bpy.context.view_layer.update()
        verts, tris = [], []
        for m in meshes:
            offset = sum(len(v) for v in verts)
            verts.append(meshdata.world_vertices(m))
            tris.append(meshdata.triangles(m.data) + offset)
        FILE_PARTS = np.repeat(np.arange(len(tris)), [len(t) for t in tris])
        verts, tris = np.concatenate(verts), np.concatenate(tris)
        # scale and offset as normalize.prepare gives this model (see read)
        after = meshdata.world_vertices(normalize.prepare(list(set(bpy.data.objects) - before), height=height))
        scale = np.ptp(after[:, 2]) / np.ptp(verts[:, 2])
        return scale * verts + (after.min(axis=0) - scale * verts.min(axis=0)), tris
    finally:
        for obj in list(set(bpy.data.objects) - before):
            bpy.data.objects.remove(obj, do_unlink=True)


def build_armature(bones, name="Armature"):
    """An armature of the rig's bones under our names, parents before children."""
    data = bpy.data.armatures.new(name)
    armature = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(armature)
    make_active(armature)
    bpy.ops.object.mode_set(mode="EDIT")
    made = {}
    pending = dict(bones)
    while pending:
        ready = [k for k, (_, _, p) in pending.items() if p is None or p in made]
        if not ready:
            raise ValueError(f"bones whose parents are missing: {sorted(pending)}")
        for key in ready:
            head, tail, parent = pending.pop(key)
            bone = data.edit_bones.new(key)
            bone.head, bone.tail = Vector(head.tolist()), Vector(tail.tolist())
            if parent is not None:
                bone.parent = made[parent]
            made[key] = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def transfer_weights(mesh_obj, armature_obj, surface, weights):
    """Weight `mesh_obj` from the rig's mesh: each vertex takes the weights at the nearest point on
    its surface (interpolated over that triangle), and gets the armature.

    Returns (gap, mismatched): the 95th percentile distance to that surface, which says whether
    the rig fits this source at all, and the mask of vertices failing the robust transfer's match
    test (inpaint.matched: too far, or facing another way), whose weights `repair_weights` refills.
    """
    rig_verts, rig_tris = surface
    table, order = weights
    tree = BVHTree.FromPolygons([tuple(v) for v in rig_verts], [tuple(t) for t in rig_tris])
    verts = meshdata.world_vertices(mesh_obj)
    rig_normals = inpaint.vertex_normals(rig_verts, rig_tris)
    groups = {name: (mesh_obj.vertex_groups.get(name) or mesh_obj.vertex_groups.new(name=name)) for name in order}
    gaps = np.zeros(len(verts))
    found_at, found_normals = np.zeros_like(verts), np.zeros_like(verts)
    for i, point in enumerate(verts):
        found, _, face, gaps[i] = tree.find_nearest(Vector(point.tolist()))
        a, b, c = (rig_verts[k] for k in rig_tris[face])
        found_at[i] = found
        w = _barycentric(found_at[i], a, b, c)
        found_normals[i] = w @ rig_normals[rig_tris[face]]
        mixed = w @ table[rig_tris[face]]
        for k in np.flatnonzero(mixed > 1e-4):
            groups[order[k]].add([i], float(mixed[k] / mixed.sum()), "REPLACE")
    if np.percentile(gaps, 95) > MATCH:
        raise ValueError(f"the rig's mesh doesn't fit this source ({np.percentile(gaps, 95):.3f} m apart): "
                         "another generation of the model?")
    found_normals /= np.maximum(np.linalg.norm(found_normals, axis=1, keepdims=True), 1e-30)
    normals = inpaint.vertex_normals(verts, meshdata.triangles(mesh_obj.data))
    ok = inpaint.matched(verts, normals, found_at, found_normals, float(np.linalg.norm(np.ptp(verts, axis=0))))
    world = mesh_obj.matrix_world.copy()
    mesh_obj.parent = armature_obj
    mesh_obj.matrix_world = world
    modifier = mesh_obj.modifiers.new("Armature", "ARMATURE")
    modifier.object = armature_obj
    return float(np.percentile(gaps, 95)), ~ok


def still_fingers(mesh_obj):
    """Hand every finger's weight to its hand: the fingers move nothing. For a character whose hands don't show
    (option `fingers: none`): NerdGirl's are inside her cuffs, 1720 vertices on Tripo's 30 finger bones, and the
    cuffs tore open where Warudo curled them. Returns how many vertices changed."""
    groups = mesh_obj.vertex_groups
    fingers = {g.index: "Left" if g.name.startswith("Left") else "Right" for g in groups
               if g.name.startswith(("Left", "Right")) and g.name.removeprefix("Left").removeprefix("Right").startswith(tuple(_FINGER_NAMES))}
    if not fingers:
        return 0
    hands = {side: groups.get(BONE_NAMES[f"{side.lower()}_hand"]) or groups.new(name=BONE_NAMES[f"{side.lower()}_hand"])
             for side in ("Left", "Right")}
    changed = 0
    for vertex in mesh_obj.data.vertices:
        moved = {"Left": 0.0, "Right": 0.0}
        for g in vertex.groups:
            if g.group in fingers:
                moved[fingers[g.group]] += g.weight
        if not any(moved.values()):
            continue
        for index in fingers:
            groups[index].remove([vertex.index])
        for side, weight in moved.items():
            if weight:
                held = next((g.weight for g in vertex.groups if g.group == hands[side].index), 0.0)
                hands[side].add([vertex.index], held + weight, "REPLACE")
        changed += 1
    return changed


def keep_arms_off_head(mesh_obj, armature_obj):
    """Take the shoulder, arm, hand and finger weights that leaked onto the head off it, handing
    their share to the vertex's other bones (Head when it has none). Returns how many changed.

    Assumes the rest pose has the hands below the head (a T- or A-pose). Tripo's rigs weight the
    scalp and hair to the arms: model A's T-pose FBX 5216 vertices above the head, up to 14%, which
    squashed the back of the head to 27% with the arms raised. Only a leak goes: up to HEAD_LEAK
    of arm weight, fading out by HEAD_KEEP, from the Head joint up over HEAD_FADE, and from
    SHOULDER_CLEAR of a shoulder joint out over SHOULDER_FADE, so no seam opens. Model C's twin-tail
    lies on her shoulder across the head joint's height, 33-46% on the arm beside 9%: cut at the
    head's height it tore 5.8 cm with the arms down, faded by amount and height still 4.7.
    """
    groups = {g.index: g.name for g in mesh_obj.vertex_groups}
    limb = {i for i, name in groups.items() if name.removeprefix("Left").removeprefix("Right").startswith(_ARM_WORDS)
            and name.startswith(("Left", "Right"))}
    if not limb:
        return 0
    head_name = BONE_NAMES["head"]
    head_z = (armature_obj.matrix_world @ armature_obj.data.bones[head_name].head_local).z
    head = mesh_obj.vertex_groups.get(head_name) or mesh_obj.vertex_groups.new(name=head_name)
    verts = meshdata.world_vertices(mesh_obj)
    rise = np.clip((verts[:, 2] - head_z) / HEAD_FADE, 0.0, 1.0)
    shoulders = [np.array(armature_obj.matrix_world @ armature_obj.data.bones[BONE_NAMES[f"{side}_upper_arm"]].head_local)
                 for side in ("left", "right")]
    clear = np.min([np.linalg.norm(verts - s, axis=1) for s in shoulders], axis=0)
    rise *= np.clip((clear - SHOULDER_CLEAR) / SHOULDER_FADE, 0.0, 1.0)
    changed = 0
    for index in np.flatnonzero(rise > 0):
        i = int(index)
        weights = {g.group: g.weight for g in mesh_obj.data.vertices[i].groups if g.weight > 0}
        arm = sum(w for g, w in weights.items() if g in limb)
        total = sum(weights.values())
        if arm <= 0 or total <= 0:
            continue
        share = arm / total
        cut = float(rise[i]) * float(np.clip((HEAD_KEEP - share) / (HEAD_KEEP - HEAD_LEAK), 0.0, 1.0))
        if cut <= 0:
            continue
        kept = {g: w * (1.0 - cut) if g in limb else w for g, w in weights.items()}
        others = sum(w for g, w in kept.items() if g not in limb)
        freed = total - sum(kept.values())
        for g, w in kept.items():
            if g in limb and w <= 1e-6:
                mesh_obj.vertex_groups[g].remove([i])
            elif g in limb:
                mesh_obj.vertex_groups[g].add([i], w / total, "REPLACE")
            else:
                mesh_obj.vertex_groups[g].add([i], (w + freed * w / others) / total if others > 0 else w / total, "REPLACE")
        if others <= 0:
            head.add([i], freed / total, "REPLACE")
        changed += 1
    return changed


def reaching_across(mesh_obj, armature_obj):
    """Vertices weighted (each at least FAR_SHARE) to two bones rig.FAR_BONES or more steps
    apart in the skeleton (rig.bone_steps)."""
    bones = armature_obj.data.bones
    groups = [g for g in mesh_obj.vertex_groups if g.name in bones]
    count = len(mesh_obj.data.vertices)
    if not groups:
        return np.zeros(count, dtype=bool)
    from .loose import _weight_table
    table = _weight_table(mesh_obj, count)[[g.index for g in groups]]
    share = table / np.maximum(table.sum(axis=0, keepdims=True), 1e-12)
    apart = bone_steps(armature_obj, [g.name for g in groups])
    heavy = (share >= FAR_SHARE).T.astype(float)
    return ((heavy @ (apart >= FAR_BONES).astype(float)) * heavy).sum(axis=1) > 0


def repair_weights(mesh_obj, armature_obj, mismatched=None, hair=None):
    """Refill (inpaint.fill_mesh) the adopted weights that can't be right from the ones around them.

    Untrusted: vertices `reaching_across` the skeleton, and `mismatched` ones from
    `transfer_weights`. The mask `hair` is filled only from hair and the rest only from the rest:
    Model C's twin-tail lies on her shoulder, so filling across would hand the hair the arm or
    the shoulder Head again. A stretch
    with nothing trusted on its side (a whole wing on arm and spine, which the appendage role
    replaces anyway) keeps its weights. Returns a report of the counts.
    """
    far = reaching_across(mesh_obj, armature_obj)
    count = len(mesh_obj.data.vertices)
    hair = np.zeros(count, dtype=bool) if hair is None else np.asarray(hair, dtype=bool)
    mismatched = np.zeros(count, dtype=bool) if mismatched is None else np.asarray(mismatched, dtype=bool)
    untrusted = far | mismatched
    report = {"reaching_across": int(far.sum()), "mismatched": int(mismatched.sum()), "filled": 0, "unresolved": 0}
    for part in (hair, ~hair):
        if (untrusted & part).any():
            filled, unresolved = inpaint.fill_mesh(mesh_obj, armature_obj, ~(untrusted & part), ~part)
            report["filled"] += filled
            report["unresolved"] += unresolved
    return report


def joints(bones):
    """The rig's joint positions keyed like landmarks.detect_body."""
    ours = {v: k for k, v in BONE_NAMES.items()}
    return {ours[name]: head for name, (head, _, _) in bones.items() if name in ours}


def _barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-18:
        return np.array([1.0, 0.0, 0.0])
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return np.clip(np.array([1 - v - w, v, w]), 0, 1)
