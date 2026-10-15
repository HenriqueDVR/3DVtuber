"""Put the rigged character into the rest pose VRM asks for: the VRM T-pose.

VRM 1.0 says a model's rest pose must be a T-pose -- standing straight with the upper arms
raised straight out to the sides ("VRM T-pose", specification/VRMC_vrm-1.0/tpose.md). Image-to-3D
characters are modelled in an A-pose, and exporting that as the rest pose leaves every app to
apply tracking rotations against a rest pose 40-70 degrees away from the one it assumes: the arms
shoot out to the sides as soon as arm tracking starts.

So the arms are swung out to horizontal, the deformation is baked into the mesh, and the posed
armature becomes the new rest pose. The character then looks like a T-pose in the file and the
A-pose it was drawn in becomes just another pose an app can put it in.
"""

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from . import finger_guides, hand_setup, meshdata
from . import fingers as finger_rig
from .rig import BONE_NAMES, make_active

# Where shoulder-to-wrist points in the rest pose, for the character's left arm.
T_DIRECTION = (1.0, 0.0, 0.0)
# Arms already this straight out are left alone (radians).
STRAIGHT = np.radians(3)
# How far the thumb sits from the fingers in the rest pose, towards the palm's thumb side
# (VRMC_vrm-1.0/tpose.md definition 1.8), in degrees.
THUMB_OUT = 45.0
# No finger spreads further than this from the hand's axis in the rest pose, in degrees.
FAN_LIMIT = 25.0
# The relaxed rest pose (`relax_fingers`), in degrees: every finger joint bends this much, the
# four fingers fan this far from each other, and the thumb tips this far toward the palm and
# bends this much at its last two joints.
RELAX_CURL = 15.0
RELAX_SPREAD = 7.0
THUMB_DOWN = 20.0
THUMB_CURL = 10.0
# The thumb in `spec_hands`, as on the reference VRoid hand that tracks well in Warudo
# (finger_guides): this far from the arm toward the front, and this far below level, in degrees.
REF_THUMB_OUT = 38.0
REF_THUMB_DOWN = 3.0
# Degrees between neighbouring fingers in `spec_hands`. The reference's slim fingers lie parallel
# 3-6 mm apart; model A's glove fingers laid parallel touched (1 mm), which reads as a bunched hand.
SPEC_SPREAD = 3.0
# `relax_thumb_web`: averaging passes, how far from the wrist the web may start (of the
# wrist-to-knuckle length), and how far from a finger's bones its own surface reaches (of its
# measured radius), which is left alone -- relaxing that too pinched model B's thumb thin.
WEB_PASSES = 15
WEB_FROM = .3
WEB_TUBE = 1.3
# `conform_fingers` traces a finger's centre from rings of its surface this far apart (m),
# from this share of its length out (the web and the palm are nearer the knuckle).
CENTRE_STEP = .004
WEB_SKIP = .15
# `round_fingers` puts a finger's two inner joints this far along it from the knuckle, of its
# length: the reference VRoid hand's (index .43/.70, middle .44/.71, ring and little .46/.72).
REF_JOINTS = (.45, .71)


def to_t_pose(mesh_obj, armature_obj, direction=T_DIRECTION, extra_meshes=(), straighten=True):
    """Swing the arms out to the sides and make that the rest pose. Returns the angles turned.

    Run after skinning and after anything that reads rest positions (hair chains, cloth chains,
    spring colliders): it moves vertices. The mesh keeps its weights; only the rest shape and the
    bones' rest transforms change. `extra_meshes` are any further meshes the armature deforms,
    which are baked too. `straighten` may also name fingers, e.g. ("thumb",), to open only those,
    or be "relaxed" for one even, lightly bent hand (`relax_fingers`), or "spec" for the whole
    hand exactly in the VRM T-pose in world axes (`spec_hands`).
    With `straighten` the fingers are laid out flat along the hand as well,
    for the same reason as the arms: a tracker sends the rotation a straight finger needs, and
    against a rest pose modelled with the hand half closed that comes out twisted.
    """
    meshes = [mesh_obj, *extra_meshes]
    turned = {}
    for side, mirror in (("left", 1.0), ("right", -1.0)):
        upper = BONE_NAMES[f"{side}_upper_arm"]
        hand = BONE_NAMES[f"{side}_hand"]
        if upper not in armature_obj.pose.bones or hand not in armature_obj.pose.bones:
            continue
        want = Vector((mirror * direction[0], direction[1], direction[2])).normalized()
        turned[side] = _point_arm(armature_obj, upper, hand, want)
    if straighten == "spec":
        # Straightening a modelled curl turns finger joints by up to 77 degrees, and a linear
        # blend shrinks a joint by a fifth at that angle (model A's pinched knuckles). So the arms
        # are baked as before and the hands after them on their own, with dual quaternions.
        _bake(meshes, armature_obj, any(turned.values()))
        opened = any(spec_hands(armature_obj).values())
        _bake(meshes, armature_obj, opened, dual=True)
        if opened:
            conform_fingers(meshes, armature_obj)
            round_fingers(meshes, armature_obj)
            for obj in meshes:
                for side in ("left", "right"):
                    relax_thumb_web(obj, armature_obj, side)
        _align_finger_hinges(armature_obj)
        return turned
    if straighten == "relaxed":
        straightened = relax_fingers(armature_obj)
    elif straighten is True:
        straightened = straighten_fingers(armature_obj)
    elif straighten:
        straightened = straighten_fingers(armature_obj, only=tuple(straighten))
    else:
        straightened = {}
    _bake(meshes, armature_obj, any(turned.values()) or any(straightened.values()))
    _align_finger_hinges(armature_obj, own_plane=straighten == "relaxed")
    return turned


def conform_fingers(meshes, armature_obj):
    """Lay each finger's mesh straight and level along its line, like the reference hand's.

    `spec_hands` turns each finger about the knuckle bone the guides placed on the curled hand,
    but the meshes bend where they were modelled to, not where the bones join: opened, model B's
    middle finger still dropped 17-22 degrees to its tip, its ring rose 13, the knuckles spread
    over 26 mm in height (the reference's over 1.2), and the fingers stood "up and down". Moving
    whole segments could not straighten a bend inside one. Here each finger's real centre line
    is traced (the centres of rings of its surface along it) and laid out straight along the
    `spec_hands` direction from its knuckle, the knuckles brought to the hand's common knuckle
    height and evenly apart across it; every vertex keeps its offset from the line and moves as
    far as the finger holds it, so the web and palm blend. The thumb is left as `spec_hands` put
    it: re-aiming model A's (its surface ran 45 degrees out, the target 38) folded its base into a
    crease. Normals turn with the surface, shape keys move alike, and the bones are set down the
    new lines. Returns the largest move (m).
    """
    world = np.array(armature_obj.matrix_world)
    to_rig = np.linalg.inv(world)
    head = lambda n: (world @ np.array([*armature_obj.data.bones[n].head_local, 1.0]))[:3]
    tail = lambda n: (world @ np.array([*armature_obj.data.bones[n].tail_local, 1.0]))[:3]
    finger_of = {finger_rig.bone_name(side, f, s): (side, f) for side in ("left", "right")
                 for f in finger_rig.FINGERS[1:] for s in finger_rig._segments(f)}
    held = []  # per mesh: (object, world vertices, {(side, finger): weight per vertex})
    for obj in meshes:
        names = {g.index: g.name for g in obj.vertex_groups}
        weights = {}
        for vertex in obj.data.vertices:
            for entry in vertex.groups:
                key = finger_of.get(names[entry.group])
                if key:
                    weights.setdefault(key, np.zeros(len(obj.data.vertices)))[vertex.index] += entry.weight
        held.append((obj, meshdata.world_vertices(obj), weights))
    moves = [np.zeros_like(verts) for _, verts, _ in held]
    turns = [np.tile(np.eye(3), (len(verts), 1, 1)) for _, verts, _ in held]
    seats = []
    for side, mirror in (("left", 1.0), ("right", -1.0)):
        hand_name = BONE_NAMES[f"{side}_hand"]
        chains = {f: [finger_rig.bone_name(side, f, s) for s in finger_rig._segments(f)] for f in finger_rig.FINGERS[1:]}
        if hand_name not in armature_obj.data.bones or not all(n in armature_obj.data.bones for c in chains.values() for n in c):
            continue
        palm = finger_guides._hand_frame(head(hand_name), [head(c[0]) for c in chains.values()], side)["palm"]
        out, forward = np.array((mirror, 0.0, 0.0)), np.array((0.0, -1.0, 0.0))
        lines = {}
        for k, (finger, chain) in enumerate(chains.items(), start=1):
            fan = np.radians(SPEC_SPREAD * (2.5 - k))  # as `spec_hands` fans them
            want = out * np.cos(fan) + forward * np.sin(fan)
            points = np.vstack([verts[weights[(side, finger)] > .5] for _, verts, weights in held if (side, finger) in weights])
            line = _centre_line(points, head(chain[0]), tail(chain[-1]))
            if line is not None:
                lines[finger] = (chain, want, line)
        up = -palm
        level = float(np.mean([line[0][0] @ up for _, _, line in lines.values()])) if lines else 0.0
        knuckles = {f: line[0][0] + up * (level - line[0][0] @ up) for f, (_, _, line) in lines.items()}
        if len(knuckles) == 4:
            # evenly apart across the hand, index and little where they are
            first, last = knuckles["index"], knuckles["little"]
            across = (last - first) - up * ((last - first) @ up)
            span = float(np.linalg.norm(across))
            across /= span
            for k, finger in enumerate(("middle", "ring"), start=1):
                knuckles[finger] = knuckles[finger] + across * (k / 3 * span - (knuckles[finger] - first) @ across)
        for finger, (chain, want, (centres, arc)) in lines.items():
            knuckle = knuckles[finger]
            normal = palm - want * (palm @ want)
            normal /= np.linalg.norm(normal)
            target = np.array([want, normal, np.cross(want, normal)])
            for m, (obj, verts, weights) in enumerate(held):
                if (side, finger) not in weights:
                    continue
                weight = np.clip(weights[(side, finger)], 0, 1)
                for i in np.flatnonzero(weight > 1e-4):
                    at, frame, offset = _on_line(verts[i], centres, arc, palm)
                    placed = knuckle + want * at + target.T @ offset
                    moves[m][i] += weight[i] * (placed - verts[i])
                    turns[m][i] = _mix_turn(turns[m][i], target.T @ frame, weight[i])
            # bones down the new line, each joint as far along it as it was along the old one
            joints = [head(chain[0])] + [tail(n) for n in chain]
            seats.append((chain, [knuckle + want * _on_line(j, centres, arc, palm)[0] for j in joints]))
    if not seats:
        return 0.0
    largest = 0.0
    for (obj, _, _), move, turn in zip(held, moves, turns):
        largest = max(largest, float(np.linalg.norm(move, axis=1).max()))
        _move_vertices(obj, move, turn)
    make_active(armature_obj)
    bpy.ops.object.mode_set(mode="EDIT")
    edit = armature_obj.data.edit_bones
    for chain, joints in seats:
        for k, name in enumerate(chain):
            if k == 0:
                edit[name].head = Vector((to_rig @ np.array([*joints[0], 1.0]))[:3])
            edit[name].tail = Vector((to_rig @ np.array([*joints[k + 1], 1.0]))[:3])
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    return largest


def round_fingers(meshes, armature_obj):
    """Make each straightened finger a round, evenly tapering tube, jointed like the reference.

    Opened from a curled, coarse image-to-3D hand, a finger keeps the flats, pinches and bulges it
    was modelled and squeezed with ("stretched or jagged", 2026-09-23), and its knuckle creases
    sit wherever the model put them. Here, along each finger laid straight by `conform_fingers`,
    every point's distance from the finger's axis moves to one smooth radius for its place
    along it -- the finger's own median thickness, tapering evenly toward the tip -- keeping its
    direction around the axis. The fingertip's cap and the web, where the finger meets the
    palm, fade out of it, and skin far off the axis (a neighbour's, the palm's) is left. Shading
    normals become the round tube's. The joints go to the reference hand's proportions
    (`REF_JOINTS`) and each finger's weight is split among its bones again along them, so each
    part bends where its bone does. Leaves the thumb. Returns the smallest gap left between
    neighbouring fingers (m) per side.
    """
    world = np.array(armature_obj.matrix_world)
    to_rig = np.linalg.inv(world)
    head = lambda n: (world @ np.array([*armature_obj.data.bones[n].head_local, 1.0]))[:3]
    tail = lambda n: (world @ np.array([*armature_obj.data.bones[n].tail_local, 1.0]))[:3]
    held = []
    for obj in meshes:
        names = {g.index: g.name for g in obj.vertex_groups}
        weights = np.zeros((len(obj.data.vertices), len(obj.vertex_groups)))
        for vertex in obj.data.vertices:
            for entry in vertex.groups:
                weights[vertex.index, entry.group] = entry.weight
        corners = None
        if obj.data.has_custom_normals:
            corners = np.zeros((len(obj.data.loops), 3))
            obj.data.corner_normals.foreach_get("vector", corners.ravel())
        held.append({"obj": obj, "names": names, "weights": weights, "verts": meshdata.world_vertices(obj),
                     "move": np.zeros((len(obj.data.vertices), 3)), "round": np.zeros(len(obj.data.vertices)),
                     "normal": np.zeros((len(obj.data.vertices), 3)), "corners": corners, "shares": {}})
    seats, gaps = [], {}
    for side in ("left", "right"):
        hand_name = BONE_NAMES[f"{side}_hand"]
        chains = {f: [finger_rig.bone_name(side, f, s) for s in finger_rig._segments(f)] for f in finger_rig.FINGERS[1:]}
        if hand_name not in armature_obj.data.bones or not all(n in armature_obj.data.bones for c in chains.values() for n in c):
            continue
        palm = finger_guides._hand_frame(head(hand_name), [head(c[0]) for c in chains.values()], side)["palm"]
        own = {}
        for finger, chain in chains.items():
            knuckle, tip = head(chain[0]), tail(chain[-1])
            length = float(np.linalg.norm(tip - knuckle))
            along = (tip - knuckle) / length
            n1 = palm - along * (palm @ along)
            n1 /= np.linalg.norm(n1)
            n2 = np.cross(along, n1)
            parts = []
            for mesh in held:
                columns = [k for k, n in mesh["names"].items() if n in chain]
                total = np.clip(mesh["weights"][:, columns].sum(axis=1), 0, 1) if columns else np.zeros(len(mesh["verts"]))
                rel = mesh["verts"] - knuckle
                parts.append((mesh, columns, total, rel @ along, rel @ n1, rel @ n2))
            # the finger's thickness along it, from its own surface, as one smooth taper
            s_own = np.concatenate([s[t > .9] for _, _, t, s, _, _ in parts])
            r_own = np.concatenate([np.hypot(b, c)[t > .9] for _, _, t, _, b, c in parts])
            keep = (s_own > .15 * length) & (s_own < .92 * length)
            s_own, r_own = s_own[keep], r_own[keep]
            where, radii = [], []
            for at in np.arange(.15 * length, .92 * length, .003):
                ring = r_own[np.abs(s_own - at) < .0015]
                if len(ring) >= 4:
                    where.append(at)
                    radii.append(float(np.median(ring)))
            if len(where) < 4:
                continue
            fit = np.polyfit(where, radii, 2)
            low, high = np.percentile(radii, (10, 90))
            radius = lambda s: np.clip(np.polyval(fit, s), low, high)
            slope = np.polyder(fit)
            end = float(radius(.9 * length))
            for mesh, columns, total, s, b, c in parts:
                r = np.hypot(b, c)
                wanted = radius(s)
                amount = (total ** 2 * np.clip((s - .12 * length) / (.13 * length), 0, 1)
                          * np.clip((length - .5 * end - s) / end, 0, 1) * np.clip((2.2 - r / wanted) / .5, 0, 1))
                hit = amount > 1e-4
                if not hit.any():
                    continue
                theta = np.arctan2(c, b)
                radial = np.outer(np.cos(theta), n1) + np.outer(np.sin(theta), n2)
                mesh["move"][hit] += (radial[hit] * ((wanted - r) * amount)[hit, None])
                mesh["round"][hit] = np.maximum(mesh["round"][hit], amount[hit])
                tube = radial - np.outer(np.polyval(slope, s), along)
                mesh["normal"][hit] = tube[hit] / np.linalg.norm(tube[hit], axis=1)[:, None]
                own.setdefault(finger, []).append((mesh, total > .9))
            joints = [knuckle, knuckle + along * REF_JOINTS[0] * length, knuckle + along * REF_JOINTS[1] * length, tip]
            seats.append((chain, joints))
            rig_joints = [(to_rig @ np.array([*j, 1.0]))[:3].tolist() for j in joints]
            for mesh, columns, total, *_ in parts:
                if columns:
                    coords = (to_rig @ np.c_[mesh["verts"], np.ones(len(mesh["verts"]))].T).T[:, :3]
                    shares = hand_setup._influences(coords, {"points": rig_joints, "radius": 1.0})[1]
                    mesh["shares"][tuple(chain)] = (total, shares)
        # how close neighbouring fingers come, after the rounding
        order = [f for f in finger_rig.FINGERS[1:] if f in own]
        closest = np.inf
        for a, b in zip(order, order[1:]):
            pa = np.vstack([(m["verts"] + m["move"])[sel] for m, sel in own[a]])
            pb = np.vstack([(m["verts"] + m["move"])[sel] for m, sel in own[b]])
            if len(pa) and len(pb):
                closest = min(closest, min(float(np.min(np.linalg.norm(pb - p, axis=1))) for p in pa[:: max(1, len(pa) // 300)]))
        gaps[side] = closest
    for mesh in held:
        obj, data = mesh["obj"], mesh["obj"].data
        moved = np.linalg.norm(mesh["move"], axis=1) > 1e-9
        local = mesh["move"] @ np.linalg.inv(np.array(obj.matrix_world)[:3, :3]).T
        for index in np.flatnonzero(moved):
            offset = Vector(local[index])
            data.vertices[int(index)].co += offset
            for key in (data.shape_keys.key_blocks if data.shape_keys else ()):
                key.data[int(index)].co += offset
        if mesh["corners"] is not None and np.any(mesh["round"] > 0):
            loops = np.zeros(len(data.loops), int)
            data.loops.foreach_get("vertex_index", loops)
            amount = mesh["round"][loops][:, None]
            normal = mesh["normal"] @ np.array(obj.matrix_world)[:3, :3]  # world to local: the transpose
            mixed = mesh["corners"] * (1 - amount) + normal[loops] * amount
            data.normals_split_custom_set((mixed / np.maximum(np.linalg.norm(mixed, axis=1), 1e-12)[:, None]).tolist())
        for chain, (total, shares) in mesh["shares"].items():
            groups = [obj.vertex_groups[n] for n in chain]
            for index in np.flatnonzero(total > 0):
                for k, group in enumerate(groups):
                    value = float(total[index] * shares[k, index])
                    if value > 1e-5:
                        group.add([int(index)], value, "REPLACE")
                    else:
                        group.remove([int(index)])
    if seats:
        make_active(armature_obj)
        bpy.ops.object.mode_set(mode="EDIT")
        edit = armature_obj.data.edit_bones
        for chain, joints in seats:
            for k, name in enumerate(chain):
                if k == 0:
                    edit[name].head = Vector((to_rig @ np.array([*joints[0], 1.0]))[:3])
                edit[name].tail = Vector((to_rig @ np.array([*joints[k + 1], 1.0]))[:3])
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.update()
    return gaps


def fit_finger_bones(meshes, armature_obj, side, finger):
    """Lay one finger's bones down the middle of its surface as it now is. Returns the offsets.

    For a finger reshaped by hand in a finished rest pose: model B's right index, bent by the user
    toward the thumb, left its tip 9 mm off its bones (2026-09-23), so it would have swung off
    its own middle when tracked. The finger's centre line is traced (`_centre_line`, as
    `conform_fingers` does), the knuckle stays where it is, and the two inner joints and the tip
    go onto that line at the same shares of its length as before (the reference's, `REF_JOINTS`,
    on a finger `round_fingers` has laid out), the chain keeping its length. The finger's own weight is split among its bones again along the new
    joints, each vertex keeping how much the finger holds it. The mesh is not moved: bones are
    edited in their rest pose. Returns (worst offset before, after) of the joints past the
    knuckle from the traced line, in metres.
    """
    world = np.array(armature_obj.matrix_world)
    to_rig = np.linalg.inv(world)
    chain = [finger_rig.bone_name(side, finger, s) for s in finger_rig._segments(finger)]
    if finger == "thumb":
        chain = chain[1:]  # the metacarpal lies inside the palm, with nothing to trace
    bones = armature_obj.data.bones
    head = lambda n: (world @ np.array([*bones[n].head_local, 1.0]))[:3]
    tail = lambda n: (world @ np.array([*bones[n].tail_local, 1.0]))[:3]
    old = [head(chain[0])] + [tail(n) for n in chain]
    lengths = np.linalg.norm(np.diff(old, axis=0), axis=1)
    length = float(lengths.sum())
    shares = np.cumsum(lengths) / length
    held = []
    for obj in meshes:
        columns = {obj.vertex_groups[n].index: k for k, n in enumerate(chain) if n in obj.vertex_groups}
        if not columns:
            continue
        weights = np.zeros((len(chain), len(obj.data.vertices)))
        for vertex in obj.data.vertices:
            for entry in vertex.groups:
                if entry.group in columns:
                    weights[columns[entry.group], vertex.index] = entry.weight
        held.append((obj, weights, meshdata.world_vertices(obj)))
    points = np.vstack([verts[weights.sum(axis=0) > .5] for _, weights, verts in held])
    line = _centre_line(points, old[0], old[-1])
    if line is None:
        raise ValueError(f"too little of the {side} {finger} to trace")
    centres, arc = line
    hand = BONE_NAMES[f"{side}_hand"]
    palm = finger_guides._hand_frame(head(hand), [head(finger_rig.bone_name(side, f, "proximal"))
                                                  for f in finger_rig.FINGERS[1:]], side)["palm"]
    # the joints past the knuckle; the knuckle stays where it is, inside the palm
    offset = lambda joints: max(float(np.linalg.norm(_on_line(j, centres, arc, palm)[2][1:])) for j in joints[1:])
    before = offset(old)
    start = _on_line(old[0], centres, arc, palm)[0]  # where the knuckle is along the line
    new = [old[0]] + [_at_arc(centres, arc, start + share * length) for share in shares]
    after = offset(new)
    make_active(armature_obj)
    bpy.ops.object.mode_set(mode="EDIT")
    edit = armature_obj.data.edit_bones
    for k, name in enumerate(chain):
        edit[name].tail = Vector((to_rig @ np.array([*new[k + 1], 1.0]))[:3])
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    rig_joints = [(to_rig @ np.array([*j, 1.0]))[:3].tolist() for j in new]
    for obj, weights, verts in held:
        total = weights.sum(axis=0)
        coords = (to_rig @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
        split = hand_setup._influences(coords, {"points": rig_joints, "radius": 1.0})[1]
        for k, name in enumerate(chain):
            group = obj.vertex_groups[name]
            for index in np.flatnonzero(total > 0):
                value = float(total[index] * split[k, index])
                if value > 1e-5:
                    group.add([int(index)], value, "REPLACE")
                else:
                    group.remove([int(index)])
    return before, after


def _at_arc(centres, arc, at):
    """The point `at` along a traced centre line, carried straight on past either end."""
    k = int(np.clip(np.searchsorted(arc, at, side="right") - 1, 0, len(centres) - 2))
    t = (at - arc[k]) / max(arc[k + 1] - arc[k], 1e-12)
    return centres[k] + (centres[k + 1] - centres[k]) * t


def _centre_line(points, knuckle, tip, step=None):
    """(centres, arc lengths) of a finger's surface from its knuckle out, or None.

    Rings of the surface every `step` along the knuckle-to-tip chord, each ring's centre taken
    after dropping points far off it (a neighbour's skin held by this finger), then a cubic
    through them: on coarse meshes single ring centres wander 5 mm either side of the finger's
    axis, and following them put that wander into the straightened finger. A first point is
    added where the line passes the knuckle, so arc 0 is there.
    """
    step = CENTRE_STEP if step is None else step
    chord = tip - knuckle
    length = float(np.linalg.norm(chord))
    chord /= length
    along = (points - knuckle) @ chord
    rings = []
    for at in np.arange(step / 2, length + step, step):
        ring = points[np.abs(along - at) < step / 2]
        if len(ring) >= 4:
            rings.append((at, ring))
    # the finger's own radius, from its middle; near the knuckle the web's skin counts as the
    # finger's too, pulled the rings' centres toward it and bent the fitted line 9 mm off
    middle = [float(np.median(np.linalg.norm(ring - ring.mean(0), axis=1))) for at, ring in rings
              if .3 * length < at < .8 * length]
    if not middle:
        return None
    reach = 1.5 * float(np.median(middle))
    centres, where, counts = [], [], []
    for at, ring in rings:
        if at < WEB_SKIP * length:
            continue
        centre = ring.mean(0)
        for _ in range(2):
            kept = ring[np.linalg.norm(ring - centre, axis=1) < reach]
            if len(kept) < 3:
                break
            centre = kept.mean(0)
        if len(kept) >= 3:
            centres.append(centre)
            where.append(at)
            counts.append(len(kept))
    if len(centres) < 4:
        return None
    where, weights = np.array(where), np.sqrt(counts)
    fits = [np.polyfit(where, np.array(centres)[:, axis], 3, w=weights) for axis in range(3)]
    centres = np.stack([np.polyval(fit, where) for fit in fits], axis=1)
    start = centres[1] - centres[0]
    start /= np.linalg.norm(start)
    centres = np.vstack([centres[0] - start * ((centres[0] - knuckle) @ start), centres])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(centres, axis=0), axis=1))]
    return centres, arc


def _on_line(point, centres, arc, palm):
    """(arc length, frame rows, offset in that frame) of a point against a centre line.

    The frame is the line's direction, toward the palm, and across. Before the first centre
    and past the last the end pieces carry straight on.
    """
    best, found = np.inf, None
    last = len(centres) - 2
    for k in range(last + 1):
        a, span = centres[k], centres[k + 1] - centres[k]
        t = float((point - a) @ span / (span @ span))
        low = -np.inf if k == 0 else 0.0
        high = np.inf if k == last else 1.0
        t = min(max(t, low), high)
        distance = float(np.linalg.norm(point - (a + span * t)))
        if distance < best:
            best, found = distance, (k, t)
    k, t = found
    along = centres[k + 1] - centres[k]
    along = along / np.linalg.norm(along)
    normal = palm - along * (palm @ along)
    normal /= np.linalg.norm(normal)
    frame = np.array([along, normal, np.cross(along, normal)])
    offset = frame @ (point - (centres[k] + (centres[k + 1] - centres[k]) * t))
    return arc[k] + (arc[k + 1] - arc[k]) * t, frame, offset


def _mix_turn(current, turn, weight):
    """Blend a vertex's accumulated turn toward `turn` by `weight` (0..1), kept a rotation."""
    u, _, vt = np.linalg.svd(current * (1 - weight) + turn * weight)
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def _move_vertices(obj, move, turn):
    """Move vertices (world offsets) in the base mesh and every shape key; turn their normals."""
    data = obj.data
    moved = np.linalg.norm(move, axis=1) > 1e-9
    if not moved.any():
        return
    corners = None
    if data.has_custom_normals:
        corners = np.zeros((len(data.loops), 3))
        data.corner_normals.foreach_get("vector", corners.ravel())
    local = move @ np.linalg.inv(np.array(obj.matrix_world)[:3, :3]).T
    for index in np.flatnonzero(moved):
        offset = Vector(local[index])
        data.vertices[int(index)].co += offset
        for key in (data.shape_keys.key_blocks if data.shape_keys else ()):
            key.data[int(index)].co += offset
    if corners is not None:
        loops = np.zeros(len(data.loops), int)
        data.loops.foreach_get("vertex_index", loops)
        corners = np.einsum("lij,lj->li", turn[loops], corners)
        corners /= np.maximum(np.linalg.norm(corners, axis=1), 1e-12)[:, None]
        data.normals_split_custom_set(corners.tolist())


def relax_thumb_web(mesh_obj, armature_obj, side, passes=None):
    """Smooth the skin between the opened thumb and the index finger. Returns vertices moved.

    Image-to-3D hands come with the thumb tucked against the index finger. Opened to the
    reference 38 degrees, the skin joining them stretched into a thick, creased wedge -- in
    Warudo model A's "curve between the index and the thumb" looked wrong, and model B's palm creased
    at the thumb's base. Here that skin (weight shared between the thumb and the rest of the hand,
    away from the wrist and outside each thumb and index segment's own tube) is smoothed with
    its neighbours while everything around it stays put. The smoothing keeps volume (Taubin):
    plain averaging thinned the thumb and the ball under it ("a bit skinny"). Shape keys move
    with it, and its shading normals are smoothed the same way rather than rebuilt from the
    coarse surface, which shaded in facets. Needs the rest pose baked with `spec_hands`.
    """
    passes = WEB_PASSES if passes is None else passes
    names = hand_setup._names(side)
    groups = {mesh_obj.vertex_groups[n].index: n for n in names if n in mesh_obj.vertex_groups}
    thumb = {i for i, n in groups.items() if n in names[1:4]}
    proximal = [finger_rig.bone_name(side, f, "proximal") for f in finger_rig.FINGERS[1:]]
    # the hand's frame is read from all four knuckles: a hand the finger step found only some
    # fingers on (the princess's gloves: thumb, index, middle) stopped the T-pose step
    if not passes or not thumb or any(n not in armature_obj.data.bones for n in [names[0], *proximal]):
        return 0
    verts = meshdata.world_vertices(mesh_obj)
    count = len(verts)
    budget, held, strongest = np.zeros(count), np.zeros(count), np.full(count, -1)
    for vertex in mesh_obj.data.vertices:
        if vertex.groups:
            strongest[vertex.index] = max(vertex.groups, key=lambda g: g.weight).group
        for entry in vertex.groups:
            if entry.group in groups:
                budget[vertex.index] += entry.weight
                held[vertex.index] += entry.weight * (entry.group in thumb)
    if not np.any(budget > .5):
        return 0
    world = armature_obj.matrix_world
    head = lambda n: np.array(world @ armature_obj.data.bones[n].head_local)
    tail = lambda n: np.array(world @ armature_obj.data.bones[n].tail_local)
    wrist = head(names[0])
    frame = finger_guides._hand_frame(wrist, [head(n) for n in proximal], side)
    share = held / np.maximum(budget, 1e-9)
    web = (finger_guides.hand_piece(mesh_obj, side, verts) & (budget > .5) & (share > .02) & (share < .98)
           & ((verts - wrist) @ frame["along"] > WEB_FROM * frame["length"]))
    # each segment's own tube, at its own measured radius: the thumb is thicker at its base
    for finger, segments in (("thumb", finger_rig._segments("thumb")[1:]), ("index", finger_rig._segments("index"))):
        for segment in segments:
            name = finger_rig.bone_name(side, finger, segment)
            group = mesh_obj.vertex_groups.get(name)
            own = strongest == (group.index if group else -2)
            if own.sum() < 5:
                continue
            _, distance = finger_guides._along_chain(verts, [head(name), tail(name)])
            web &= distance > WEB_TUBE * float(np.median(distance[own]))
    weld, edge_a, edge_b = meshdata.welded_edges(mesh_obj, verts)
    region = np.zeros(count, bool)
    region[weld[web]] = True
    keep = edge_a != edge_b
    edge_a, edge_b = edge_a[keep], edge_b[keep]
    degree = np.bincount(np.r_[edge_a, edge_b], minlength=count)
    region &= degree > 0
    position = _taubin(verts, region, edge_a, edge_b, degree, passes)
    moved = np.linalg.norm((position - verts)[weld], axis=1) > 1e-9  # UV-seam twins too
    if not moved.any():
        return 0
    data = mesh_obj.data
    corners = None
    if data.has_custom_normals:
        corners = np.zeros((len(data.loops), 3))
        data.corner_normals.foreach_get("vector", corners.ravel())
    delta = (position - verts)[weld] @ np.linalg.inv(np.array(mesh_obj.matrix_world)[:3, :3]).T
    for index in np.flatnonzero(moved):
        offset = Vector(delta[index])
        data.vertices[int(index)].co += offset
        for key in (data.shape_keys.key_blocks if data.shape_keys else ()):
            key.data[int(index)].co += offset
    if corners is not None:
        loops = np.zeros(len(data.loops), int)
        data.loops.foreach_get("vertex_index", loops)
        pooled = np.zeros((count, 3))
        np.add.at(pooled, weld[loops], corners)  # one normal per point, alike across UV seams
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1), 1e-12)[:, None]
        pooled = _taubin(pooled, region, edge_a, edge_b, degree, passes)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1), 1e-12)[:, None]
        hit = moved[loops]
        corners[hit] = pooled[weld[loops[hit]]]
        data.normals_split_custom_set(corners.tolist())
    return int(moved.sum())


def _taubin(values, region, edge_a, edge_b, degree, passes, shrink=.5, grow=-.53):
    """Smooth per-point `values` over the surface inside `region`, keeping its volume."""
    values = values.copy()
    for _ in range(passes):
        for amount in (shrink, grow):
            summed = np.zeros_like(values)
            np.add.at(summed, edge_a, values[edge_b])
            np.add.at(summed, edge_b, values[edge_a])
            values[region] += amount * (summed[region] / degree[region, None] - values[region])
    return values


def _bake(meshes, armature_obj, needed, dual=False):
    """Bake the current pose into the meshes and make it the rest pose, if anything moved."""
    if not needed:
        return
    for obj in meshes:
        _bake_mesh(obj, armature_obj, dual)
    _apply_rest(armature_obj)


def straighten_fingers(armature_obj, only=None):
    """Open each hand into the rest pose VRM asks for. Returns the angles turned, per bone.

    The spec wants the four fingers extended and parallel to the ground, and the thumb out at 45
    degrees (VRMC_vrm-1.0/tpose.md, definitions 1.7 and 1.8). Each finger keeps the fan of its
    proximal bone, capped and kept from crossing its neighbours (`_fan`), and ALL its segments
    extend in that one direction. Flattening each segment
    separately preserves sideways kinks and depends on the rotations already applied to its
    parents. Pointing every finger the same way, on the other hand, collapses the natural fan.
    `only` limits this to some fingers, e.g. ("thumb",): a thumb modelled tucked along the
    index gives a tracking app no sensible axes, while curled fingers with good bones still
    close cleanly.
    """
    turned = {}
    for side in ("left", "right"):
        hand_name = BONE_NAMES[f"{side}_hand"]
        if hand_name not in armature_obj.pose.bones:
            continue
        hand = armature_obj.pose.bones[hand_name]
        along = (hand.tail - hand.head).normalized()
        across = _across_palm(armature_obj, side, along)
        if across is None:
            continue
        normal = along.cross(across).normalized()  # through the palm
        fan = _fan(armature_obj, side, normal, along, across)
        for finger in finger_rig.FINGERS:
            if only is not None and finger not in only:
                continue
            segments = finger_rig._segments(finger)
            root = armature_obj.pose.bones.get(finger_rig.bone_name(side, finger, segments[0]))
            if root is None:
                continue
            out = np.radians(THUMB_OUT) if finger == "thumb" else fan.get(finger)
            if out is None:
                continue
            want = (along * float(np.cos(out)) + across * float(np.sin(out))).normalized()
            for segment in segments:
                name = finger_rig.bone_name(side, finger, segment)
                if name in armature_obj.pose.bones:
                    turned[name] = _point_bone(armature_obj, name, want)
    return turned


def spec_hands(armature_obj):
    """Put each hand exactly in the VRM T-pose, in world axes. Returns the angles turned.

    VRMC_vrm-1.0/tpose.md 1.6-1.8: the hand level and straight out along the arm, palm to the
    ground, the four fingers along the arm, the thumb level and 45 degrees between out and
    forward. Apps built on UniVRM (Warudo) apply tracking as if the rest hand were exactly
    this. Fingers laid out consistently in the hand's own frame were not enough: model A's palms
    were rolled 16-29 degrees and drooped 9, and in Warudo each finger then opened toward a
    different place and the thumb went wrong. The thumb points the way the reference VRoid hand's
    does (`REF_THUMB_OUT`, `REF_THUMB_DOWN`); the fingers are straight like its, fanned just enough
    (`SPEC_SPREAD`) that thick glove fingers don't touch. The character faces -Y and up is +Z.
    """
    turned = {}
    for side, mirror in (("left", 1.0), ("right", -1.0)):
        hand_name = BONE_NAMES[f"{side}_hand"]
        heads = [armature_obj.pose.bones.get(finger_rig.bone_name(side, f, "proximal"))
                 for f in ("index", "middle", "ring", "little")]
        if hand_name not in armature_obj.pose.bones or not all(heads):
            continue
        hand = armature_obj.pose.bones[hand_name]
        out = Vector((mirror, 0.0, 0.0))
        forward = Vector((0.0, -1.0, 0.0))
        down = Vector((0.0, 0.0, -1.0))
        # the hand's own frame: wrist to knuckles, and toward the palm
        along = (sum((b.head for b in heads), Vector()) / 4 - hand.head).normalized()
        across = heads[0].head - heads[3].head
        across = (across - along * across.dot(along)).normalized()
        palm = along.cross(across).normalized() * mirror
        now = Matrix((along, palm, along.cross(palm))).transposed()
        want = Matrix((out, down, out.cross(down))).transposed()
        turn = want @ now.inverted()
        head = hand.head.copy()
        hand.matrix = Matrix.Translation(head) @ turn.to_4x4() @ Matrix.Translation(-head) @ hand.matrix
        bpy.context.view_layer.update()
        turned[hand_name] = float(np.degrees(turn.to_quaternion().angle))
        side_angle, down_angle = np.radians(REF_THUMB_OUT), np.radians(REF_THUMB_DOWN)
        level = out * float(np.cos(side_angle)) + forward * float(np.sin(side_angle))
        thumb = (level * float(np.cos(down_angle)) + down * float(np.sin(down_angle))).normalized()
        for k, finger in enumerate(finger_rig.FINGERS):
            fan = np.radians(SPEC_SPREAD * (2.5 - k))  # index 1.5 steps toward the thumb, little -1.5
            want = thumb if finger == "thumb" else out * float(np.cos(fan)) + forward * float(np.sin(fan))
            for segment in finger_rig._segments(finger):
                name = finger_rig.bone_name(side, finger, segment)
                if name in armature_obj.pose.bones:
                    turned[name] = _point_bone(armature_obj, name, want)
    return turned


def relax_fingers(armature_obj, curl=RELAX_CURL, spread=RELAX_SPREAD):
    """Pose each hand into one tidy relaxed rest pose. Returns the angles turned, per bone.

    Image-to-3D hands are modelled with every finger bent differently and the fingers squeezed
    together. Kept as the rest pose, that made a tracked hand close unevenly (the same angle
    added to different starting curls) and open with the fingers bunched; straightened flat, the
    mesh showed every crease. Here every finger bends the same `curl` degrees at each joint,
    in its own plane, fanned `spread` degrees from its neighbours about the hand's axis, and the
    thumb takes the VRM rest direction -- out to the side, tipped toward the palm -- with a
    light bend. Needs finger bones that lie along the fingers (`finger_guides`).
    """
    turned = {}
    for side, mirror in (("left", 1.0), ("right", -1.0)):
        hand_name = BONE_NAMES[f"{side}_hand"]
        if hand_name not in armature_obj.pose.bones:
            continue
        hand = armature_obj.pose.bones[hand_name]
        along = (hand.tail - hand.head).normalized()
        across = _across_palm(armature_obj, side, along)
        if across is None:
            continue
        palm = along.cross(across).normalized() * mirror  # the side the fingers close toward
        fans = {f: np.radians(spread * (1.5 - k)) for k, f in enumerate(finger_rig.FINGERS[1:])}
        for finger in finger_rig.FINGERS:
            segments = finger_rig._segments(finger)
            if finger == "thumb":
                out, down, bend = np.radians(THUMB_OUT), np.radians(THUMB_DOWN), np.radians(THUMB_CURL)
                base = (along * float(np.cos(out)) + across * float(np.sin(out))).normalized()
                base = (base * float(np.cos(down)) + palm * float(np.sin(down))).normalized()
                # the thumb folds across the palm toward the little finger
                toward = (palm - across).normalized()
            else:
                out, bend = fans[finger], np.radians(curl)
                base = (along * float(np.cos(out)) + across * float(np.sin(out))).normalized()
                toward = palm
            toward = (toward - base * toward.dot(base)).normalized()
            for k, segment in enumerate(segments):
                name = finger_rig.bone_name(side, finger, segment)
                if name not in armature_obj.pose.bones:
                    continue
                angle = bend * (k + (0 if finger == "thumb" else 1))
                want = (base * float(np.cos(angle)) + toward * float(np.sin(angle))).normalized()
                turned[name] = _point_bone(armature_obj, name, want)
    return turned


def _across_palm(armature_obj, side, along):
    """Across the palm towards the thumb, read off whichever fingers the hand has."""
    heads = {}
    for finger in ("index", "little", "thumb"):
        name = finger_rig.bone_name(side, finger, "metacarpal" if finger == "thumb" else "proximal")
        if name in armature_obj.pose.bones:
            heads[finger] = armature_obj.pose.bones[name].head
    if "index" in heads and "little" in heads:
        # The thumb's metacarpal is below the knuckle row on a cupped hand. Using it first
        # tilts the supposed palm plane into the fingers' curl instead of removing the curl.
        towards = heads["index"] - heads["little"]
    elif "thumb" in heads and "index" in heads:
        towards = heads["thumb"] - heads["index"]
    else:
        return None
    towards -= along * towards.dot(along)
    return towards.normalized() if towards.length > 1e-6 else None


def _align_finger_hinges(armature_obj, own_plane=False):
    """Set a consistent rest-roll AFTER baking, without twisting the mesh to match the roll.

    Positive local-X flexion should move toward the palm on both hands. Minimal rotations
    that straighten a bent chain otherwise retain different rolls on each segment. Changing
    rest roll (not posing a twist) leaves the baked vertices and all shape keys untouched.
    With `own_plane`, a bent finger hinges square to the plane it is bent in rather than to the
    palm's axis: in the relaxed rest pose a fanned index finger otherwise folds 2 degrees
    sideways, and the thumb, which bends across the palm, 4.5.
    """
    hinges = {}
    for side, mirror in (("left", 1.0), ("right", -1.0)):
        hand = armature_obj.pose.bones.get(BONE_NAMES[f"{side}_hand"])
        if hand is None:
            continue
        along = (hand.tail - hand.head).normalized()
        across = _across_palm(armature_obj, side, along)
        if across is not None:
            hinges[side] = -mirror * across
    if not hinges:
        return
    make_active(armature_obj)
    bpy.ops.object.mode_set(mode="EDIT")
    for side, palm_hinge in hinges.items():
        for finger in finger_rig.FINGERS:
            hinge = palm_hinge
            chain = [armature_obj.data.edit_bones.get(finger_rig.bone_name(side, finger, s))
                     for s in finger_rig._segments(finger)]
            if own_plane and all(chain):
                plane = (chain[0].tail - chain[0].head).normalized().cross(
                    (chain[-1].tail - chain[-1].head).normalized())
                if plane.length > np.sin(np.radians(3)):
                    hinge = plane.normalized() * (1 if plane.dot(palm_hinge) >= 0 else -1)
            for segment in finger_rig._segments(finger):
                bone = armature_obj.data.edit_bones.get(finger_rig.bone_name(side, finger, segment))
                if bone is not None:
                    # Keep X across the palm even on a curled segment. Aligning Z to the
                    # palm normal becomes ambiguous/flips when a bone curls through 90 degrees.
                    normal = hinge.cross((bone.tail - bone.head).normalized())
                    if normal.length > 1e-6:
                        bone.align_roll(normal)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()


def _flattened(bone, normal, along):
    """The bone's own direction with the curl taken out: flat in the palm, fanned as it was."""
    direction = (bone.tail - bone.head).normalized()
    flat = direction - normal * direction.dot(normal)
    if flat.length < 1e-4 or flat.normalized().dot(along) <= 0.0:
        return along  # the bone points straight into the palm; nothing of its own is left
    return flat.normalized()


def _fan(armature_obj, side, normal, along, across):
    """Each finger's spread in the palm (radians, toward the thumb), as modelled but untangled.

    The first bone of a curled finger on a fused glove points sideways at random: on model A the
    left ring finger read 36 degrees out while its neighbours read 10 and 14, and straightening
    to those angles split the glove into two V-shaped pairs. So each spread is capped, the
    four are kept in order across the hand -- index furthest toward the thumb, little furthest
    away -- by averaging any neighbours that would otherwise cross, and the fan is centred.
    """
    angles, names = [], []
    for finger in finger_rig.FINGERS[1:]:
        root = armature_obj.pose.bones.get(finger_rig.bone_name(side, finger, "proximal"))
        if root is None:
            continue
        flat = _flattened(root, normal, along)
        angle = float(np.arctan2(flat.dot(across), flat.dot(along)))
        angles.append(float(np.clip(angle, -np.radians(FAN_LIMIT), np.radians(FAN_LIMIT))))
        names.append(finger)
    # pool adjacent violators: a non-increasing sequence nearest the measured one
    blocks = []
    for angle in angles:
        blocks.append([angle, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            value, count = blocks.pop()
            blocks[-1] = [(blocks[-1][0] * blocks[-1][1] + value * count) / (blocks[-1][1] + count),
                          blocks[-1][1] + count]
    ordered = [value for value, count in blocks for _ in range(count)]
    # centred on the hand's own axis: on model A the whole fan leant 14 degrees away from the
    # thumb, which left the thumb splayed 60 degrees from the index instead of 45
    centre = float(np.mean(ordered)) if ordered else 0.0
    return {name: value - centre for name, value in zip(names, ordered)}


def _point_bone(armature_obj, name, want):
    """Rotate one bone about its head so it points along `want`, children following."""
    bone = armature_obj.pose.bones[name]
    now = (bone.tail - bone.head).normalized()
    axis = now.cross(want)
    angle = now.angle(want)
    if axis.length < 1e-6 or angle < STRAIGHT:
        return 0.0
    head = bone.head.copy()
    turn = Matrix.Rotation(angle, 4, axis.normalized())
    bone.matrix = Matrix.Translation(head) @ turn @ Matrix.Translation(-head) @ bone.matrix
    bpy.context.view_layer.update()
    return float(np.degrees(angle))


def _point_arm(armature_obj, upper_name, hand_name, want):
    """Rotate the upper arm about its own head so shoulder-to-wrist points along `want`."""
    upper = armature_obj.pose.bones[upper_name]
    hand = armature_obj.pose.bones[hand_name]
    now = (hand.head - upper.head).normalized()
    axis = now.cross(want)
    angle = now.angle(want)
    if axis.length < 1e-6 or angle < STRAIGHT:
        return 0.0
    head = upper.head.copy()
    turn = Matrix.Rotation(angle, 4, axis.normalized())
    upper.matrix = Matrix.Translation(head) @ turn @ Matrix.Translation(-head) @ upper.matrix
    bpy.context.view_layer.update()
    return float(np.degrees(angle))


def _bake_mesh(mesh_obj, armature_obj, dual=False, moves=None):
    """Move the mesh where the pose puts it, keeping its shape keys.

    Blender won't apply an armature modifier to a mesh that has shape keys, and by this point the
    face has dozens of them, so the skinning is done here: each vertex gets the blend of its
    bones' pose transforms, and that matrix moves the vertex in the base mesh and in every key.
    Imported normals are turned by the same matrix, as a game engine skins them.
    """
    matrices = _vertex_matrices(mesh_obj, armature_obj, dual, moves)
    matrices = _fill_gaps(mesh_obj, matrices)
    normals = _skinned_normals(mesh_obj, matrices)
    coords = [Vector(v.co) for v in mesh_obj.data.vertices]
    moved = [matrices[i] @ coords[i] if matrices[i] is not None else coords[i] for i in range(len(coords))]
    keys = mesh_obj.data.shape_keys.key_blocks if mesh_obj.data.shape_keys else []
    for key in keys:
        for i, point in enumerate(key.data):
            if matrices[i] is not None:
                # a key holds absolute positions, so its own offset from the base mesh is carried
                point.co = matrices[i] @ Vector(point.co)
    for i, vertex in enumerate(mesh_obj.data.vertices):
        vertex.co = moved[i]
    if normals is not None:
        mesh_obj.data.normals_split_custom_set(normals)


def _skinned_normals(mesh_obj, matrices):
    """Each corner's shading normal turned by its vertex's pose matrix, or None if not imported.

    Image-to-3D meshes are coarse (6-9 mm edges on a finger) and look smooth only through the
    normals imported with them. Blender keeps custom normals relative to each corner's own edges,
    which is close enough for small moves, but opening a curled finger or a tucked thumb left them
    pointing the old way: the straightened hands shaded in facets where the modelled ones were
    smooth. Uses the inverse transpose, which is also right for linearly blended matrices.
    """
    data = mesh_obj.data
    if not data.has_custom_normals:
        return None
    corners = np.zeros((len(data.loops), 3))
    data.corner_normals.foreach_get("vector", corners.ravel())
    loops = np.zeros(len(data.loops), int)
    data.loops.foreach_get("vertex_index", loops)
    turns = np.array([np.array(m.to_3x3()) if m is not None else np.eye(3) for m in matrices])
    singular = np.abs(np.linalg.det(turns)) < 1e-9
    turns[singular] = np.eye(3)
    turns = np.linalg.inv(turns).transpose(0, 2, 1)
    turned = np.einsum("lij,lj->li", turns[loops], corners)
    return (turned / np.maximum(np.linalg.norm(turned, axis=1), 1e-12)[:, None]).tolist()


def _fill_gaps(mesh_obj, matrices):
    """Move a vertex with no weights along with its neighbours, not leave it where it was.

    One vertex left behind while the surface around it goes out to the T-pose drags every edge
    it is on into a spike tens of centimetres long.
    """
    missing = [index for index, matrix in enumerate(matrices) if matrix is None]
    if not missing:
        return matrices
    neighbours = {index: [] for index in missing}
    for edge in mesh_obj.data.edges:
        first, second = edge.vertices
        if first in neighbours:
            neighbours[first].append(second)
        if second in neighbours:
            neighbours[second].append(first)
    for index in missing:
        nearby = [matrices[other] for other in neighbours[index] if matrices[other] is not None]
        if not nearby:
            continue
        blended = Matrix(((0.0,) * 4,) * 4)
        for matrix in nearby:
            for row in range(4):
                for column in range(4):
                    blended[row][column] += matrix[row][column] / len(nearby)
        matrices[index] = blended
    return matrices


def _vertex_matrices(mesh_obj, armature_obj, dual=False, moves=None):
    """Per-vertex blend of the bones' pose transforms, in the mesh's own space.

    Linear, or with `dual` a dual-quaternion blend (Blender's "Preserve Volume"), which turns a
    joint between two bones instead of shrinking it. Assumes rigid pose transforms (no scale).
    `moves` (bone name -> armature-space matrix, identity for the rest) replaces the pose.
    """
    into_armature = armature_obj.matrix_world.inverted() @ mesh_obj.matrix_world
    out_of_armature = into_armature.inverted()
    by_group = {}
    for group in mesh_obj.vertex_groups:
        bone = armature_obj.pose.bones.get(group.name)
        if bone is not None:
            by_group[group.index] = (bone.matrix @ bone.bone.matrix_local.inverted() if moves is None
                                     else moves.get(group.name, Matrix.Identity(4)))
    duals = {g: _to_dual(m) for g, m in by_group.items()} if dual else {}
    matrices = []
    for vertex in mesh_obj.data.vertices:
        weighted = [(g.group, g.weight) for g in vertex.groups if g.group in by_group and g.weight > 0.0]
        total = sum(w for _, w in weighted)
        if not weighted or total <= 0.0:
            matrices.append(None)
            continue
        if dual:
            first = duals[max(weighted, key=lambda gw: gw[1])[0]][0]
            blended = np.zeros(8)
            for group, weight in weighted:
                real, dual_part = duals[group]
                sign = 1.0 if real @ first >= 0 else -1.0  # the short way round
                blended += np.r_[real, dual_part] * sign * weight / total
            blended = _from_dual(blended[:4], blended[4:])
        else:
            blended = Matrix(((0,) * 4,) * 4)
            for group, weight in weighted:
                matrix = by_group[group]
                for row in range(4):
                    for column in range(4):
                        blended[row][column] += matrix[row][column] * weight / total
        matrices.append(out_of_armature @ blended @ into_armature)
    return matrices


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2))


def _to_dual(matrix):
    """A rigid 4x4 transform as a unit dual quaternion (real, dual), (w, x, y, z) each."""
    real = np.array(matrix.to_quaternion())
    return real, .5 * _quat_mul(np.r_[0.0, np.array(matrix.translation)], real)


def _from_dual(real, dual):
    norm = np.linalg.norm(real)
    real, dual = real / norm, dual / norm
    matrix = Quaternion(real).to_matrix().to_4x4()
    matrix.translation = 2 * _quat_mul(dual, real * np.array((1, -1, -1, -1)))[1:]
    return matrix


def _apply_rest(armature_obj):
    """Make the current pose the armature's rest pose."""
    make_active(armature_obj)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")
