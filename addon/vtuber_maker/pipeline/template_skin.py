"""Skin a character from a well-rigged template body instead of bone heat and rules.

Our bone-heat skinning grew a rule per defect (sleeves fused to the torso, shoulders reaching the
chest, legs up coat panels) and still broke on a new shape; Tripo's auto-rig weighted model C's whole
body cleanly by fitting what it knows of bodies. This does the conventional version of that: a
template humanoid rigged by hand (pixiv's VRoid sample, samples/reference/, licensed for
modification and redistribution) is bent onto the character's skeleton bone by bone, and each
character vertex takes the weights at the nearest point of that fitted body where the two agree
(close, facing the same way); the rest -- loose cloth, a hat, wide sleeves -- is filled in from
those over the character's own surface (after "Robust Skin Weights Transfer via Weight
Inpainting", Abdrashitov et al. 2023, with a simple diffusion for the solve). Manifest roles and
the later repairs then act on the result as on bone heat's.

Works in any pose the joints describe (A- or T-pose): the template's T-pose is turned bone by bone.
"""

from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import adopt, meshdata
from .rig import BONE_NAMES

TEMPLATE = Path(__file__).resolve().parents[3] / "samples" / "reference" / "VRM1_Constraint_Twist_Sample.vrm"
TEMPLATE_MESHES = ("Body", "Face")
# A character vertex trusts the fitted template's weights within this distance (m, at 1.6 m) ...
TRUST_DISTANCE = 0.04
# ... when the two surfaces face within this angle (cosine) of each other.
TRUST_FACING = 0.5
# Diffusion passes that fill in the vertices left untrusted, over the welded surface.
FILL_PASSES = 400
MAX_INFLUENCES = 4
# A template vertex is body, not costume, when this share of its weight is on body bones.
SKIN_SHARE = 0.9
# Each body bone runs to this joint (our bone names).
_CHILD = {"Hips": "Spine", "Spine": "Chest", "Chest": "UpperChest", "UpperChest": "Neck", "Neck": "Head",
          **{f"{s}{a}": f"{s}{b}" for s in ("Left", "Right") for a, b in
             (("Shoulder", "UpperArm"), ("UpperArm", "LowerArm"), ("LowerArm", "Hand"),
              ("UpperLeg", "LowerLeg"), ("LowerLeg", "Foot"))}}


def load_template(path=TEMPLATE):
    """(bones, (vertices, triangles, normals), (weights, order)) of the template, in its own frame.

    Bones as {our name: (head, tail)}. Weights of bones we don't map (VRoid's J_Sec hair and cloth
    bones, twist helpers) go to their nearest mapped ancestor; fingers fold into the hand (our
    armature gets fingers later, in their own step).
    """
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = set(bpy.data.objects) - before
    try:
        armature = next(o for o in imported if o.type == "ARMATURE")
        table = adopt.names()
        for side in ("left", "right"):
            hand = BONE_NAMES[f"{side}_hand"]
            for name, ours in list(table.items()):
                if ours.startswith(side.capitalize()) and any(f in ours for f in ("Thumb", "Index", "Middle", "Ring", "Little")):
                    table[name] = hand

        def mapped(bone):
            while bone is not None and bone.name not in table:
                bone = bone.parent
            return table[bone.name] if bone is not None else BONE_NAMES["hips"]

        world = np.array(armature.matrix_world)
        bones = {}
        core = set(BONE_NAMES.values())
        toes = {side: (world @ np.array([*armature.data.bones[f"J_Bip_{letter}_ToeBase"].head_local, 1.0]))[:3]
                for letter, side in (("L", "Left"), ("R", "Right")) if f"J_Bip_{letter}_ToeBase" in armature.data.bones}
        for bone in armature.data.bones:
            # the body's own bones: fingers, toes and eye helpers only lend weight to the bone they fold into
            if not bone.name.startswith("J_Bip") or bone.name.endswith("ToeBase") or any(
                    f in bone.name for f in ("Thumb", "Index", "Middle", "Ring", "Little")):
                continue
            ours = table.get(bone.name)
            if ours in core and ours not in bones:
                bones[ours] = tuple((world @ np.array([*p, 1.0]))[:3] for p in (bone.head_local, bone.tail_local))
        # The glTF importer invents bone tails (they pointed the arms into the chest, and the arm's
        # part of the template was turned there), so each bone runs to its child joint instead.
        for name, child in _CHILD.items():
            if name in bones and child in bones:
                bones[name] = (bones[name][0], bones[child][0])
        for hand, lower in (("LeftHand", "LeftLowerArm"), ("RightHand", "RightLowerArm")):
            if hand in bones and lower in bones:
                head = bones[hand][0]
                bones[hand] = (head, head + 0.4 * (head - bones[lower][0]))
        for side, toe in toes.items():
            if f"{side}Foot" in bones:
                bones[f"{side}Foot"] = (bones[f"{side}Foot"][0], toe)
        if "Head" in bones and "Neck" in bones:
            head = bones["Head"][0]
            bones["Head"] = (head, head + np.array([0.0, 0.0, 2.5 * np.linalg.norm(head - bones["Neck"][0])]))

        meshes = [o for o in imported if o.type == "MESH" and any(o.name.startswith(n) or o.data.name.startswith(n)
                                                                   for n in TEMPLATE_MESHES)]
        verts, tris, normals, rows, skin = [], [], [], [], []
        order = sorted(set(table.values()))
        column = {name: k for k, name in enumerate(order)}
        for m in meshes:
            offset = sum(len(v) for v in verts)
            verts.append(meshdata.world_vertices(m))
            tris.append(meshdata.triangles(m.data) + offset)
            n = np.array([v.normal for v in m.data.vertices])
            normals.append(n @ np.array(m.matrix_world)[:3, :3].T)
            index = {g.index: column[mapped(armature.data.bones.get(g.name))] for g in m.vertex_groups
                     if g.name in armature.data.bones}
            w = np.zeros((len(m.data.vertices), len(order)))
            on_body = np.zeros(len(m.data.vertices))
            body_groups = {g.index for g in m.vertex_groups if not g.name.startswith("J_Sec")}
            for v in m.data.vertices:
                for g in v.groups:
                    if g.group in index:
                        w[v.index, index[g.group]] += g.weight
                    if g.group in body_groups:
                        on_body[v.index] += g.weight
            rows.append(w)
            skin.append(on_body >= SKIN_SHARE * np.maximum(w.sum(axis=1), 1e-9))
        weights = np.concatenate(rows)
        # only the body itself: the sample's dress and hair ride VRoid's secondary (J_Sec) bones,
        # and a skirt's Hips weight nearer a character's thigh than its own leg tore model A's knees
        keep = np.concatenate(skin)
        tris = np.concatenate(tris)
        tris = tris[keep[tris].all(axis=1)]
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
        normals = np.concatenate(normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        return bones, (np.concatenate(verts), tris, normals), (weights, order)
    finally:
        for obj in list(set(bpy.data.objects) - before):
            bpy.data.objects.remove(obj, do_unlink=True)


def fit(template, armature_obj):
    """The template's surface bent onto `armature_obj`'s bones: each bone's part of it is turned
    to the bone's direction, stretched along it to its length and moved to its head; across the
    bone it is scaled with the characters' heights. Returns (vertices, normals)."""
    bones, (verts, _, normals), (weights, order) = template
    world = np.array(armature_obj.matrix_world)
    target = {b.name: tuple((world @ np.array([*p, 1.0]))[:3] for p in (b.head_local, b.tail_local))
              for b in armature_obj.data.bones}
    # thickness scales with the body's own size, hips to head: the characters' full heights take in
    # hats and hair
    hips, head = BONE_NAMES["hips"], BONE_NAMES["head"]
    across = (float(np.linalg.norm(target[head][0] - target[hips][0]))
              / max(float(np.linalg.norm(bones[head][0] - bones[hips][0])), 1e-9))
    moved = np.zeros_like(verts)
    turned = np.zeros_like(normals)
    for k, name in enumerate(order):
        w = weights[:, k]
        if not w.any():
            continue
        if name not in bones or name not in target:
            moved += w[:, None] * verts * across
            turned += w[:, None] * normals
            continue
        (h0, t0), (h1, t1) = bones[name], target[name]
        rotation, stretch = _bone_map(t0 - h0, t1 - h1, across)
        local = verts - h0
        moved += w[:, None] * (local @ (rotation @ stretch).T + h1)
        turned += w[:, None] * (normals @ rotation.T)
    turned /= np.maximum(np.linalg.norm(turned, axis=1, keepdims=True), 1e-9)
    return moved, turned


def transfer(mesh_obj, armature_obj, template):
    """Weight `mesh_obj` from the template fitted to `armature_obj`, and give it the armature.

    Returns {"trusted": share of vertices weighted straight from the template, "filled": the rest}.
    """
    _, (verts_t, tris_t, _), (weights_t, order) = template
    fitted, fitted_normals = fit(template, armature_obj)
    tree = BVHTree.FromPolygons([tuple(v) for v in fitted], [tuple(t) for t in tris_t])
    verts = meshdata.world_vertices(mesh_obj)
    normals = np.array([v.normal for v in mesh_obj.data.vertices]) @ np.array(mesh_obj.matrix_world)[:3, :3].T
    height = float(np.ptp(verts[:, 2]))
    weights = np.zeros((len(verts), len(order)))
    trusted = np.zeros(len(verts), dtype=bool)
    for i, point in enumerate(verts):
        found, normal, face, distance = tree.find_nearest(Vector(point.tolist()))
        a, b, c = (fitted[k] for k in tris_t[face])
        w = adopt._barycentric(np.array(found), a, b, c) @ weights_t[tris_t[face]]
        weights[i] = w
        facing = abs(float(np.dot(normals[i], np.array(normal)))) if normal is not None else 0.0
        trusted[i] = distance < TRUST_DISTANCE * height / 1.6 and facing > TRUST_FACING
    weights = _fill(mesh_obj, verts, weights, trusted)
    # keep the strongest few, as apps read at most four per vertex
    strongest = np.argsort(-weights, axis=1)[:, :MAX_INFLUENCES]
    kept = np.zeros_like(weights)
    rows = np.arange(len(weights))[:, None]
    kept[rows, strongest] = weights[rows, strongest]
    kept /= np.maximum(kept.sum(axis=1, keepdims=True), 1e-9)
    groups = {name: mesh_obj.vertex_groups.get(name) or mesh_obj.vertex_groups.new(name=name)
              for name in order if name in armature_obj.data.bones}
    fallback = {name: _nearest_bone(armature_obj, name) for name in order if name not in armature_obj.data.bones}
    for i in range(len(verts)):
        for k in np.flatnonzero(kept[i] > 1e-4):
            name = order[k] if order[k] in groups else fallback[order[k]]
            group = groups.get(name) or mesh_obj.vertex_groups.get(name) or mesh_obj.vertex_groups.new(name=name)
            group.add([i], float(kept[i, k]), "ADD")
    world_matrix = mesh_obj.matrix_world.copy()
    mesh_obj.parent = armature_obj
    mesh_obj.matrix_world = world_matrix
    modifier = mesh_obj.modifiers.new("Armature", "ARMATURE")
    modifier.object = armature_obj
    return {"trusted": round(float(trusted.mean()), 3), "filled": int((~trusted).sum())}


def _fill(mesh_obj, verts, weights, trusted):
    """Untrusted vertices take the average of their neighbours over the welded surface, repeatedly,
    with the trusted ones held; a piece with no trusted vertex keeps its nearest-point weights."""
    weld, a, b = meshdata.welded_edges(mesh_obj, verts)
    held = np.zeros(len(verts), dtype=bool)
    np.logical_or.at(held, weld, trusted)
    values = np.zeros_like(weights)
    np.add.at(values, weld, weights)
    counts = np.bincount(weld, minlength=len(verts))[:, None]
    values = values / np.maximum(counts, 1)
    base = values.copy()
    degree = np.bincount(np.concatenate([a, b]), minlength=len(verts))[:, None]
    reachable = held.copy()
    for _ in range(FILL_PASSES):
        total = np.zeros_like(values)
        np.add.at(total, a, values[b])
        np.add.at(total, b, values[a])
        spread = np.zeros(len(verts), dtype=bool)
        np.logical_or.at(spread, a, reachable[b])
        np.logical_or.at(spread, b, reachable[a])
        averaged = total / np.maximum(degree, 1)
        free = ~held & spread & (degree[:, 0] > 0)
        values[free] = averaged[free]
        reachable |= spread
    values[~reachable] = base[~reachable]
    return values[weld]


def _bone_map(frm, to, across):
    """(rotation, stretch): stretch scales along `frm` to `to`'s length and by `across` sideways,
    rotation turns `frm`'s direction onto `to`'s."""
    lf, lt = float(np.linalg.norm(frm)), float(np.linalg.norm(to))
    if lf < 1e-9 or lt < 1e-9:
        return np.eye(3), np.eye(3) * across
    u, v = frm / lf, to / lt
    stretch = across * (np.eye(3) - np.outer(u, u)) + (lt / lf) * np.outer(u, u)
    axis = np.cross(u, v)
    s, c = float(np.linalg.norm(axis)), float(np.dot(u, v))
    if s < 1e-9:
        return (np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(u, u)), stretch
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K, stretch


def _nearest_bone(armature_obj, name):
    """Ours for a template bone our armature lacks (the toes): its parent's name if we have it."""
    for key, value in BONE_NAMES.items():
        if value == name:
            return value
    return BONE_NAMES["hips"]
