"""User-owned finger guides in armature-local rest space; no mesh reshaping.

Guides are ordinary selectable Blender empties, saved in the blend. Applying them updates
existing named bones (so VRM bindings survive), and redistributes ONLY this hand's weight
budget. Fused surfaces still require manual weight painting or mesh repair.
"""
import json
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import fingers
from .rig import BONE_NAMES, make_active

PREVIEW_KEY = "vtuber_hand_preview"
COLORS = ((1, .5, .15, 1), (.2, .8, 1, 1), (.3, 1, .3, 1), (1, .3, .6, 1), (.7, .4, 1, 1))


def _names(side):
    if side not in ("left", "right"):
        raise ValueError("Choose left or right hand")
    return [BONE_NAMES[f"{side}_hand"]] + [fingers.bone_name(side, f, s)
            for f in fingers.FINGERS for s in fingers._segments(f)]


def require_rest(arm):
    if PREVIEW_KEY in arm:
        raise ValueError("Reset preview before editing or applying guides")
    if any(not np.allclose(b.matrix_basis, np.eye(4), atol=1e-6) for b in arm.pose.bones):
        raise ValueError("Use the rig's rest pose before editing guides (clear pose transforms)")
    if arm.animation_data and arm.animation_data.action:
        raise ValueError("Use an unanimated setup copy; active animation can override the preview")
    if any(b.constraints for b in arm.pose.bones):
        raise ValueError("Use a setup copy without pose constraints")


def guide_objects(arm, side):
    return [o for o in arm.children if o.get("vtuber_hand_side") == side]


def create_guides(arm, side):
    require_rest(arm)
    hand = arm.data.bones.get(_names(side)[0])
    if hand is None:
        raise ValueError(f"Rig has no {side} hand bone")
    existing = guide_objects(arm, side)
    if existing:
        read_guides(arm, side)  # Refuse incomplete sets rather than silently duplicate them.
        return existing
    data = {}
    for i, finger in enumerate(fingers.FINGERS):
        bones = [arm.data.bones.get(fingers.bone_name(side, finger, s)) for s in fingers._segments(finger)]
        if all(bones):
            points = [b.head_local.copy() for b in bones] + [bones[-1].tail_local.copy()]
            bend = points[0] + bones[0].matrix_local.to_3x3().col[2] * hand.length * .4
        else:
            # A visible template, not a claim to have detected the finger.
            sign = 1 if side == "left" else -1
            x = sign * (.32 - i * .15) * hand.length
            start = .3 if finger == "thumb" else .5
            length = .4 if finger == "thumb" else .5
            points = [hand.matrix_local @ Vector((x, (start + t * length) * hand.length, 0))
                      for t in (0, .45, .75, 1)]
            bend = points[0] + hand.matrix_local.to_3x3().col[2] * hand.length * .4
        data[finger] = {"points": [list(p) for p in points], "bend": list(bend),
                        "radius": max(hand.length * .10, .001)}
    return write_guides(arm, side, data)


def reseed_guides(arm):
    """Replace each hand's guides with ones read from its current bones. Returns the sides.

    Guides are rest-space empties; a T-pose bake moves the bones and leaves them behind.
    """
    sides = [side for side in ("left", "right") if guide_objects(arm, side)]
    for side in sides:
        for obj in guide_objects(arm, side):
            bpy.data.objects.remove(obj, do_unlink=True)
        create_guides(arm, side)
    return sides


def validate(data):
    if set(data) != set(fingers.FINGERS):
        raise ValueError("Place guides for all five fingers")
    for name, row in data.items():
        points = np.asarray(row["points"], dtype=float)
        bend = np.asarray(row["bend"], dtype=float)
        radius = float(row["radius"])
        if (points.shape != (4, 3) or bend.shape != (3,) or not np.isfinite(points).all()
                or not np.isfinite(bend).all() or not math.isfinite(radius) or radius <= 0):
            raise ValueError(f"{name}: four finite joints, a bend guide and a positive radius are required")
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.min(lengths) < 1e-5:
            raise ValueError(f"{name}: adjacent joints overlap; move them apart")
        normal = bend - points[0]
        # A constant hinge avoids roll flips on curved fingers. Bend marker sets positive curl.
        hinge = np.cross(points[1] - points[0], normal)
        if np.linalg.norm(hinge) < 1e-8 or any(np.linalg.norm(np.cross(hinge, v)) < 1e-10
                                             for v in np.diff(points, axis=0)):
            raise ValueError(f"{name}: move the bend guide toward the palm, away from the finger axis")


def write_guides(arm, side, data):
    validate(data)
    _names(side)
    existing = {(o.get("finger"), o.get("point")): o for o in guide_objects(arm, side)}
    result = []
    for i, finger in enumerate(fingers.FINGERS):
        row = data[finger]
        for point, position in enumerate(row["points"] + [row["bend"]]):
            obj = existing.get((finger, point))
            if obj is None:
                label = ("Base", "Joint 1", "Joint 2", "Tip", "Bend toward palm")[point]
                obj = bpy.data.objects.new(f"{side.title()} {finger.title()} — {label}", None)
                bpy.context.scene.collection.objects.link(obj)
                obj.parent = arm
                obj["vtuber_hand_side"], obj["finger"], obj["point"] = side, finger, point
            obj.location = position
            obj.empty_display_type = "CONE" if point == 4 else "SPHERE"
            obj.empty_display_size = row["radius"] * .35
            obj.show_in_front = True
            obj.show_name = True
            obj.color = COLORS[i]
            obj.hide_render = True
            obj["radius"] = row["radius"]
            result.append(obj)
    bpy.context.view_layer.update()
    return result


def read_guides(arm, side):
    objects = guide_objects(arm, side)
    data = {}
    for finger in fingers.FINGERS:
        found = {o.get("point"): o for o in objects if o.get("finger") == finger}
        if set(found) != set(range(5)):
            raise ValueError(f"{side} {finger}: guides are incomplete; undo deletion or remove this hand's guides and restart")
        # matrix_local also honors edits made in world space through Blender's transform gizmo.
        data[finger] = {"points": [list(found[i].matrix_local.translation) for i in range(4)],
                        "bend": list(found[4].matrix_local.translation), "radius": float(found[0]["radius"])}
    return data


def mirror_guides(arm, side):
    """Reflect in rig-local X=0. Intended for this pipeline's centered humanoids."""
    require_rest(arm)
    target = "right" if side == "left" else "left"
    if _names(target)[0] not in arm.data.bones:
        raise ValueError("The opposite hand bone is missing")
    data = read_guides(arm, side)
    for row in data.values():
        for p in row["points"] + [row["bend"]]:
            p[0] *= -1
    return write_guides(arm, target, data)


def _influences(coords, row):
    points = np.array(row["points"])
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    starts = np.r_[0, np.cumsum(lengths)]
    projected = np.array([np.clip((coords - p) @ v / (v @ v), 0, 1)
                          for p, v in zip(points, vectors)])
    distances = np.array([np.linalg.norm(coords - (p + t[:, None] * v), axis=1)
                          for p, v, t in zip(points, vectors, projected)])
    owner = distances.argmin(axis=0)
    indices = np.arange(len(coords))
    distance = distances[owner, indices]
    along = starts[owner] + projected[owner, indices] * lengths[owner]
    # Fade into palm at base, and across the user-adjustable tube boundary.
    root = np.clip(along / (lengths[0] * .35), 0, 1)
    radial = np.clip((1 - distance / row["radius"]) * 3, 0, 1)
    influence = root * radial
    shares = np.zeros((3, len(coords)))
    shares[owner, indices] = 1
    for j in (1, 2):
        width = .2 * min(lengths[j - 1], lengths[j])
        band = np.abs(along - starts[j]) < width
        mix = (along[band] - starts[j] + width) / (2 * width)
        shares[:, band] = 0
        shares[j - 1, band], shares[j, band] = 1 - mix, mix
    return influence, shares, distance / row["radius"]


def apply(mesh, arm, side, data):
    validate(data)
    require_rest(arm)
    names = _names(side)
    if names[0] not in arm.data.bones:
        raise ValueError("Missing hand bone")
    if not any(m.type == "ARMATURE" and m.object == arm for m in mesh.modifiers):
        raise ValueError("Choose a mesh already skinned to this rig")
    # Changing bones would also affect any other skinned meshes. Refuse a partial rebuild.
    if any(o != mesh and o.type == "MESH" and any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)
           and any(n in o.vertex_groups for n in names) for o in bpy.context.scene.objects):
        raise ValueError("Multiple hand meshes use this rig; join them in a setup copy before rebuilding")
    weights = np.zeros(len(mesh.data.vertices))
    groups = {g.index for g in mesh.vertex_groups if g.name in names}
    for v in mesh.data.vertices:
        weights[v.index] = sum(g.weight for g in v.groups if g.group in groups)
    selected = np.flatnonzero(weights > 1e-8)
    if not len(selected):
        raise ValueError("No hand/finger weights on the selected mesh")
    matrix = arm.matrix_world.inverted() @ mesh.matrix_world
    coords = np.array([matrix @ v.co for v in mesh.data.vertices])[selected]
    fields = [_influences(coords, data[f]) for f in fingers.FINGERS]
    strengths = np.array([field[0] for field in fields])
    # Nearest centerline dominates instead of lending equal weight to every overlapping tube.
    scores = strengths * np.exp(-8 * np.array([field[2] for field in fields]) ** 2)
    total = scores.sum(axis=0)
    coverage = strengths.max(axis=0)
    allocations = scores / np.maximum(total, 1e-30) * coverage
    if not np.any(coverage > .1):
        raise ValueError("Guides miss the weighted hand; check depth and influence radius")
    make_active(arm)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for finger in fingers.FINGERS:
            row = data[finger]
            points = [Vector(p) for p in row["points"]]
            hinge = (points[1] - points[0]).cross(Vector(row["bend"]) - points[0]).normalized()
            parent = arm.data.edit_bones[names[0]]
            for i, segment in enumerate(fingers._segments(finger)):
                name = fingers.bone_name(side, finger, segment)
                bone = arm.data.edit_bones.get(name) or arm.data.edit_bones.new(name)
                bone.use_connect = False
                bone.head, bone.tail = points[i], points[i + 1]
                bone.align_roll(hinge.cross((bone.tail - bone.head).normalized()))
                bone.parent = parent
                bone.use_connect = i > 0
                parent = bone
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    all_groups = {n: mesh.vertex_groups.get(n) or mesh.vertex_groups.new(name=n) for n in names}
    # Replace zeros too: applying a correction must clear the previous assignment.
    for i, index in enumerate(selected):
        all_groups[names[0]].add([int(index)], float(weights[index] * (1 - coverage[i])), "REPLACE")
        for f, finger in enumerate(fingers.FINGERS):
            for s, segment in enumerate(fingers._segments(finger)):
                value = weights[index] * allocations[f, i] * fields[f][1][s, i]
                all_groups[fingers.bone_name(side, finger, segment)].add([int(index)], float(value), "REPLACE")
    arm[f"vtuber_hand_setup_{side}"] = json.dumps(data)
    if hasattr(arm.data, "vrm_addon_extension"):
        from .vrm_export import bind_fingers
        bind_fingers(arm)
    bpy.context.view_layer.update()
    return len(selected)


def reset_preview(arm):
    if PREVIEW_KEY not in arm:
        return
    saved = json.loads(arm[PREVIEW_KEY])
    for name, state in saved.items():
        bone = arm.pose.bones.get(name)
        if bone:
            bone.rotation_mode = state["mode"]
            bone.matrix_basis = Matrix(state["basis"])
    del arm[PREVIEW_KEY]
    bpy.context.view_layer.update()


def preview(arm, side, pose="fist", amount=1.0):
    if pose not in ("fist", "point", "relaxed", "rest", *fingers.FINGERS):
        raise ValueError("Unknown hand preview")
    reset_preview(arm)
    bones = [arm.pose.bones[n] for n in _names(side)[1:] if n in arm.pose.bones]
    if len(bones) != 15:
        raise ValueError("Apply guides first to build all five fingers")
    arm[PREVIEW_KEY] = json.dumps({b.name: {"mode": b.rotation_mode,
                                "basis": [list(row) for row in b.matrix_basis]} for b in bones})
    for finger in fingers.FINGERS:
        strength = .2 if pose == "relaxed" else 1
        if pose == "rest" or (pose == "point" and finger == "index") or (pose in fingers.FINGERS and finger != pose):
            strength = 0
        for segment in fingers._segments(finger):
            bone = arm.pose.bones[fingers.bone_name(side, finger, segment)]
            bone.rotation_mode = "XYZ"
            bone.rotation_euler = (math.radians(55) * max(0, min(amount, 1)) * strength, 0, 0)
    bpy.context.view_layer.update()
