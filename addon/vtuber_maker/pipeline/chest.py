"""Chest bounce: two spring bones on the front of the chest, the way VRoid avatars have them.

Each side gets a bone from inside the chest to its front-most point and a short tip past it, on
a soft spring; the chest surface round that point is blended onto it, fading out over an
ellipsoid. Works on the rest pose (the baked T-pose or the modelled one: only the torso is read).
Leaves alone anything held by the arms, the head or a hair chain (hair hanging over the chest
must swing with its own springs, not bounce with the chest), and every vertex behind the bone.
"""

import numpy as np

from .meshdata import world_vertices

BONES = ("LeftBreast", "RightBreast")
SPRING_NAME = "Chest"
ARM_BONES = {f"{s}{b}" for s in ("Left", "Right") for b in ("UpperArm", "LowerArm", "Hand", "Shoulder")}
# Search for each side's front-most point this far either side of the centre line (m, at 1.6 m tall).
SIDE_RANGE = (0.035, 0.13)
# The bone starts this far behind that point and its tip bone sticks out this far in front.
BONE_DEPTH = 0.07
TIP_LENGTH = 0.03
# Weighted region: an ellipsoid round the front point, full weight in its inner FULL share.
RADIUS = np.array([0.065, 0.06, 0.07])
FULL = 0.35
# A vertex held this much by an arm, the head or a hair chain is left alone.
HELD = 0.1
# Spring settings (VRM 1.0 units): soft, little gravity, so it bounces and settles back.
STIFFNESS = 0.35
DRAG = 0.15
GRAVITY = 0.05
HIT_RADIUS = 0.02
DEFAULT_STRENGTH = 0.8
# How much it bounces (`amount`): 1 is the settings above; more is a softer spring holding more of
# the surface, less a stiffer one holding less.
AMOUNT_RANGE = (0.25, 2.0)


def add(mesh_obj, armature_obj, strength=DEFAULT_STRENGTH, height=1.6, amount=1.0):
    """Add the two chest bones, their weights and their springs; returns {side: vertices weighted}.

    `amount` (AMOUNT_RANGE) scales the bounce: the springs' stiffness is divided by it and the
    weight the bones take multiplied by it (never past 1).

    Assumes the VRM humanoid names (Hips, Chest, UpperChest, Neck) and a body facing -Y. Replaces
    chest bones and springs an earlier run added. Raises ValueError when a side has no chest
    surface between the Chest and Neck joints.
    """
    import bpy
    from mathutils import Vector

    scale = height / 1.6
    amount = float(np.clip(amount, *AMOUNT_RANGE))
    strength = min(strength * amount, 1.0)
    _remove_groups(mesh_obj)
    groups = {g.index: g.name for g in mesh_obj.vertex_groups}
    verts = world_vertices(mesh_obj)
    bones = armature_obj.data.bones
    world = armature_obj.matrix_world
    chest_z = (world @ bones["Chest"].head_local).z
    neck_z = (world @ bones["Neck"].head_local).z
    centre_x = (world @ bones["Hips"].head_local).x
    held_by = lambda name: name in ARM_BONES or name == "Head" or name.startswith("Hair")  # noqa: E731
    held = np.array([any(held_by(groups[g.group]) and g.weight > HELD for g in v.groups if g.group in groups)
                     for v in mesh_obj.data.vertices])
    band = (verts[:, 2] > chest_z) & (verts[:, 2] < neck_z) & ~held

    fronts = {}
    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        off = sign * (verts[:, 0] - centre_x)
        pick = band & (off > SIDE_RANGE[0] * scale) & (off < SIDE_RANGE[1] * scale)
        if not pick.any():
            raise ValueError(f"no chest surface found on the {side.lower()} side")
        fronts[side] = verts[np.flatnonzero(pick)[np.argmin(verts[pick, 1])]]

    view_layer = bpy.context.view_layer
    for obj in view_layer.objects:
        obj.select_set(False)
    view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    edit = armature_obj.data.edit_bones
    inverse = world.inverted()
    for side, front in fronts.items():
        front = Vector(front)
        for name in (f"{side}Breast", f"{side}Breast_end"):
            if name in edit:
                edit.remove(edit[name])
        bone = edit.new(f"{side}Breast")
        bone.head = inverse @ (front + Vector((0.0, BONE_DEPTH * scale, 0.0)))
        bone.tail = inverse @ front
        bone.parent = edit["UpperChest"] if "UpperChest" in edit else edit["Chest"]
        tip = edit.new(f"{side}Breast_end")
        tip.head = bone.tail
        tip.tail = inverse @ (front - Vector((0.0, TIP_LENGTH * scale, 0.0)))
        tip.parent = bone
        tip.use_connect = True
    bpy.ops.object.mode_set(mode="OBJECT")

    weighted = {}
    for side, front in fronts.items():
        name = f"{side}Breast"
        group = mesh_obj.vertex_groups.get(name) or mesh_obj.vertex_groups.new(name=name)
        centre = front + np.array([0.0, 0.5 * BONE_DEPTH * scale, 0.0])
        distance = np.linalg.norm((verts - centre) / (RADIUS * scale), axis=1)
        weight = np.clip((1.0 - distance) / (1.0 - FULL), 0.0, 1.0)
        weight = weight * weight * (3 - 2 * weight) * strength
        weight[held | (verts[:, 1] > centre[1] + 0.5 * BONE_DEPTH * scale)] = 0.0
        indices = np.flatnonzero(weight > 1e-3)
        for index in indices:
            i, w = int(index), float(weight[index])
            for g in list(mesh_obj.data.vertices[i].groups):
                if g.group != group.index:
                    mesh_obj.vertex_groups[g.group].add([i], g.weight * (1.0 - w), "REPLACE")
            group.add([i], w, "REPLACE")
        weighted[side.lower()] = len(indices)

    spring_bone = armature_obj.data.vrm_addon_extension.spring_bone1
    for index in reversed([i for i, s in enumerate(spring_bone.springs) if s.vrm_name == SPRING_NAME]):
        spring_bone.springs.remove(index)
    for side in fronts:
        spring = spring_bone.add_spring()
        spring.vrm_name = SPRING_NAME
        for name in (f"{side}Breast", f"{side}Breast_end"):
            joint = spring.joints.add()
            joint.node.bone_name = name
            joint.stiffness = STIFFNESS / amount
            joint.drag_force = DRAG
            joint.gravity_power = GRAVITY
            joint.hit_radius = HIT_RADIUS
    return weighted


def _remove_groups(mesh_obj):
    """Take an earlier run's chest groups off, handing their share back to the vertex's other bones."""
    old = [g for g in mesh_obj.vertex_groups if g.name in BONES]
    if not old:
        return
    indices = {g.index for g in old}
    touched = [v.index for v in mesh_obj.data.vertices if any(g.group in indices for g in v.groups)]
    for group in old:
        mesh_obj.vertex_groups.remove(group)
    for i in touched:
        rest = [(g.group, g.weight) for g in mesh_obj.data.vertices[i].groups]
        total = sum(w for _, w in rest)
        if total > 1e-6:
            for group, weight in rest:
                mesh_obj.vertex_groups[group].add([i], weight / total, "REPLACE")
