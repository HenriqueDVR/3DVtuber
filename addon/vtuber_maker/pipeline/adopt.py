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
_FINGER_NAMES = {"Thumb": "thumb", "Index": "index", "Middle": "middle", "Ring": "ring", "Little": "little"}
# The source and the rig's mesh must agree this closely (95th percentile, m) once both are normalized.
MATCH = 0.01
# A vertex counts as on a bone (see reaching_across) holding at least this share of it.
FAR_SHARE = 0.02


def names():
    """{rig bone name: our bone name} for the VRoid-style skeleton; toes and eye helpers go to their parent."""
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


def load(path, height=1.6):
    """Read a rigged file without leaving it in the scene.

    Returns (bones, surface, weights): bones as {our name: (head, tail, parent our name)} in the
    normalized frame; surface as (vertices, triangles) of the rig's mesh at rest there; weights as
    an (vertices, bones) array over `order`, the list of our bone names weighted.
    """
    before = set(bpy.data.objects)
    if str(path).lower().endswith(".fbx"):
        compat.import_fbx(path)
    else:
        bpy.ops.import_scene.gltf(filepath=str(path))
    imported = set(bpy.data.objects) - before
    try:
        armature = next(o for o in imported if o.type == "ARMATURE")
        meshes = [o for o in imported if o.type == "MESH" and o.find_armature() == armature]
        table = names()
        world = np.array(armature.matrix_world)
        raw = {}
        for bone in armature.data.bones:
            ours = table.get(bone.name)
            # Root, the eye helpers and the toes only lend their weights to the bone they map to
            if ours is None or bone.name == "Root" or bone.name.startswith("J_Adj") or bone.name.endswith("ToeBase"):
                continue
            head, tail = ((world @ np.array([*p, 1.0]))[:3] for p in (bone.head_local, bone.tail_local))
            parent = bone.parent
            while parent is not None and (parent.name not in table or table[parent.name] == ours):
                parent = parent.parent
            raw[ours] = (head, tail, table[parent.name] if parent is not None else None)
        if np.isnan(np.array([h for h, _, _ in raw.values()])).any():
            raise ValueError(f"{path}: the rig's joints don't import (NaN); Tripo's GLB is broken, use its FBX")

        verts, tris, groups = [], [], []
        for m in meshes:
            offset = sum(len(v) for v in verts)
            verts.append(meshdata.world_vertices(m))
            tris.append(meshdata.triangles(m.data) + offset)
            groups.append(m)
        verts, tris = np.concatenate(verts), np.concatenate(tris)
        order = sorted({table[g.name] for m in meshes for g in m.vertex_groups if g.name in table})
        column = {name: k for k, name in enumerate(order)}
        weights = np.zeros((len(verts), len(order)))
        row = 0
        for m in meshes:
            index = {g.index: column[table[g.name]] for g in m.vertex_groups if g.name in table}
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
