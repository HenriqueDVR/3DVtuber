"""What a step's .blend holds, in numbers the engine compares from step to step (app/contracts.py).

Run by the app before a step's script (app/engine.py): each time the step saves a .blend, the
numbers go to the JSON file named by VTUBER_MAKER_FINGERPRINT. By hand, on an open file:

    blender -b in.blend --python-exit-code 1 -P scripts/_fingerprint.py -- out.json

Read-only. Four bugs of 2026-10-03 each changed one of these numbers and were found by eye in
Warudo instead: the loose step welded the mouth patch (99 vertices to 36), a hand import emptied
the eyes' look keys, the T-pose step fanned an adopted rig's fingers, and eye pieces rode hair
springs.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from bpy.app.handlers import persistent

ENV = "VTUBER_MAKER_FINGERPRINT"
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")
# A shape key moves a vertex when it takes it further than this (metres): float noise is 1e-7
MOVED = 1e-5
# Weight below this is the trace every smoothing pass leaves, not a bone holding the vertex
HELD = 0.05


def fingerprint():
    armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    skinned = [o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers)]
    springs = _spring_bones(armature)
    found = {"meshes": {}, "keys": {}, "face": {}, "fingers": _finger_angles(armature),
             "bones": len(armature.data.bones) if armature else 0, "spring_bones": len(springs)}
    for obj in skinned:
        mesh = obj.data
        used = np.zeros(len(mesh.polygons), dtype=np.int32)
        mesh.polygons.foreach_get("material_index", used)
        names = [slot.material.name if slot.material else "" for slot in obj.material_slots] or [""]
        found["meshes"][obj.name] = {
            "vertices": len(mesh.vertices), "polygons": len(mesh.polygons),
            "materials": {names[i] if i < len(names) else str(i): int(n) for i, n in enumerate(np.bincount(used)) if n}}
        if not mesh.shape_keys:
            continue
        blocks = mesh.shape_keys.key_blocks
        basis = _positions(blocks[0])
        touched = np.zeros(len(basis), dtype=bool)
        keys = found["keys"][obj.name] = {}
        for block in blocks[1:]:
            far = np.linalg.norm(_positions(block) - basis, axis=1)
            moved = far > MOVED
            touched |= moved
            keys[block.name] = {"moved": int(moved.sum()), "max_mm": round(float(far.max()) * 1000, 2)}
        found["face"][obj.name] = _face_weights(obj, np.flatnonzero(touched), springs)
    return found


def _positions(block):
    co = np.zeros(len(block.data) * 3)
    block.data.foreach_get("co", co)
    return co.reshape(-1, 3)


def _spring_bones(armature):
    try:
        springs = armature.data.vrm_addon_extension.spring_bone1.springs
        return {joint.node.bone_name for spring in springs for joint in spring.joints}
    except AttributeError:  # no armature, or the VRM Add-on is off
        return set()


def _face_weights(obj, face, springs):
    """How the vertices that shape keys move are held: the median weight on Head, and how many of
    them a spring bone holds (the face's own pieces must not swing with the hair)."""
    groups = {g.index: g.name for g in obj.vertex_groups}
    head, swinging = [], 0
    for index in face:
        weights = {groups.get(e.group): e.weight for e in obj.data.vertices[index].groups}
        head.append(weights.get("Head", 0.0))
        swinging += sum(w for n, w in weights.items() if n in springs) > HELD
    return {"vertices": int(len(face)), "head": round(float(np.median(head)), 3) if head else None, "on_springs": int(swinging)}


def _finger_angles(armature):
    """Each finger bone's angle from its hand bone, degrees, in the rest pose: raising the arms
    keeps it, relaying the fingers doesn't."""
    angles = {}
    for bone in armature.data.bones if armature else ():
        if not any(f in bone.name for f in FINGERS):
            continue
        hand = next((p for p in bone.parent_recursive if p.name.endswith("Hand")), None)
        if hand is None or bone.length < 1e-6 or hand.length < 1e-6:
            continue
        along = (bone.tail_local - bone.head_local).angle(hand.tail_local - hand.head_local)
        angles[bone.name] = round(math.degrees(along), 1)
    return angles


@persistent
def _on_save(*_):
    try:
        with open(os.environ[ENV], "w", encoding="utf-8") as handle:
            json.dump(fingerprint(), handle)
    except Exception as error:  # the fingerprint must never take the step down with it
        print(f"fingerprint not written: {error!r}")


if os.environ.get(ENV):
    bpy.app.handlers.save_post.append(_on_save)
elif "--" in sys.argv:
    with open(sys.argv[sys.argv.index("--") + 1], "w", encoding="utf-8") as handle:
        json.dump(fingerprint(), handle, indent=1)
