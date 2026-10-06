"""Spring bone settings for hair, and a check that swinging hair stays out of the body."""

import math
from typing import NamedTuple

import numpy as np

from . import body_volume, meshdata, spring_sim
from .rig import BONE_NAMES, CHEST_AT, SPINE_AT, UPPER_CHEST_AT

# VRM 1.0 spring settings for hair. Stiffness 0-4, gravity power 0-2, drag 0-1, hit radius in metres.
HAIR_STIFFNESS = 1.5
# Joints further down a strand soften towards this at the tip, so long hair trails the head's
# motion instead of swinging out like a stick. Gravity stays light: VRM springs don't keep a
# strand's modelled shape, so strong gravity folds hair that flares out back into the body.
HAIR_TIP_STIFFNESS = 0.5
HAIR_GRAVITY_POWER = 0.1
HAIR_DRAG_FORCE = 0.7
HAIR_HIT_RADIUS = 0.012
# Hanging cloth (capes, coat tails): heavier and slower than hair, and thicker to collide.
CLOTH_STIFFNESS = 1.0
CLOTH_TIP_STIFFNESS = 0.4
CLOTH_GRAVITY_POWER = 0.3
CLOTH_DRAG_FORCE = 0.6
CLOTH_HIT_RADIUS = 0.02
# A pointed hat's tip: firm at the crown so it keeps its shape, loose at the end and lightly
# damped so it bounces after the head moves.
TIP_STIFFNESS = 0.8
TIP_TIP_STIFFNESS = 0.25
TIP_GRAVITY_POWER = 0.05
TIP_DRAG_FORCE = 0.12
TIP_HIT_RADIUS = 0.01
# Wings, tails and big ears (manifest role "appendage"): they hold their shape and sway rather
# than hang. With cloth-like springs (1.2 stiff, 0.05 gravity) model C's wings sagged at rest and her
# 24 chains flailed in the viewer: no gravity, stiff, well damped.
APPENDAGE_STIFFNESS = 3.0
APPENDAGE_TIP_STIFFNESS = 1.5
APPENDAGE_GRAVITY_POWER = 0.0
APPENDAGE_DRAG_FORCE = 0.8
APPENDAGE_HIT_RADIUS = 0.03

# Colliders sit a little inside the fitted body, so hair resting on it isn't pushed away.
COLLIDER_SHRINK = 0.9
# Room left between a collider and the nearest resting hair joint's hit sphere.
REST_CLEARANCE = 0.005
# ... but a collider keeps at least this share of its fitted size: hair modelled lying on the
# body is nudged out a little rather than the body losing its collision.
MIN_FIT_SHARE = 0.75
# Where a capsule spans the torso's width, as fractions of the way from hips to neck.
TORSO_COLLIDER_HEIGHTS = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
# The head collider fills this share of the head's width and depth.
HEAD_FILL = 0.9

# Hair counts as clipping once it is this much deeper inside the body than at rest.
CLIP_TOLERANCE = 0.003
# The check samples every few frames and hair vertices to stay quick.
FRAME_STEP = 4
VERTEX_STEP = 3


class SpringDesign(NamedTuple):
    chains: list  # spring_sim.Chain, with bone names
    colliders: list  # spring_sim.Collider, attached by bone name


class Motion(NamedTuple):
    poses: list  # per frame: {body bone name: 4x4 rest-to-pose transform}
    torso: list  # per frame: 4x4 transform carrying the torso and limbs
    dt: float

    def __call__(self, frame):
        return self.poses[frame]


class ClippingReport(NamedTuple):
    deepest: float
    clipping_fraction: float
    worst_frame: int
    worst_detail: str

    def summary(self):
        return (
            f"hair sank {self.deepest * 1000:.1f} mm into the body at frame {self.worst_frame}; "
            f"{self.clipping_fraction:.1%} of hair samples were inside; worst: {self.worst_detail}"
        )


def design(mesh_obj, armature, chains, joints, cloth_chains=(), tip_chains=(), appendage_chains=(), hair_hits_body=True):
    """Spring chains for the hair, cloth, hat tip and appendage bone chains, plus colliders fitted to the body.

    Without `hair_hits_body` the hair's chains collide with the head alone. Hair that lies on the
    shoulders has nowhere to go when the head tilts onto them: model D's lock was shoved 8.8 cm up
    its chain by the chest's and the shoulder's colliders (1.6 cm with the head's alone) and stood
    off as a ledge. The user's choice per character: long hair needs the body to keep it out of the back.
    """
    bones = armature.data.bones
    springs = []
    kinds = (
        (chains, HAIR_STIFFNESS, HAIR_TIP_STIFFNESS, HAIR_GRAVITY_POWER, HAIR_DRAG_FORCE, HAIR_HIT_RADIUS),
        (cloth_chains, CLOTH_STIFFNESS, CLOTH_TIP_STIFFNESS, CLOTH_GRAVITY_POWER, CLOTH_DRAG_FORCE, CLOTH_HIT_RADIUS),
        (tip_chains, TIP_STIFFNESS, TIP_TIP_STIFFNESS, TIP_GRAVITY_POWER, TIP_DRAG_FORCE, TIP_HIT_RADIUS),
        (appendage_chains, APPENDAGE_STIFFNESS, APPENDAGE_TIP_STIFFNESS, APPENDAGE_GRAVITY_POWER,
         APPENDAGE_DRAG_FORCE, APPENDAGE_HIT_RADIUS),
    )
    for kind, stiffness, tip_stiffness, gravity, drag, hit in kinds:
        for names in kind:
            points = [bones[names[0]].head_local] + [bones[name].tail_local for name in names]
            springs.append(
                spring_sim.Chain(
                    points=np.array([list(point) for point in points]),
                    parent=bones[names[0]].parent.name,
                    stiffness=stiffness,
                    gravity_power=gravity,
                    drag_force=drag,
                    hit_radius=hit,
                    names=names,
                    tip_stiffness=tip_stiffness,
                    collides_with=None if hair_hits_body or kind is not chains else (BONE_NAMES["head"],),
                )
            )
    hair_bones = {name for kind in (chains, cloth_chains, tip_chains, appendage_chains) for names in kind for name in names}
    volume = body_volume.fit(mesh_obj, joints, hair_bones)
    colliders = _body_colliders(volume, joints)
    head = _head_collider(mesh_obj, joints, hair_bones)
    if head is not None:
        colliders[0] = head
    return SpringDesign(springs, _clear_of_resting_hair(colliders, springs))


def _head_collider(mesh_obj, joints, hair_bones):
    """A sphere fitted to the head's own surface (not the hair), or None."""
    head_joint = np.asarray(joints["head"], dtype=float)
    verts = meshdata.world_vertices(mesh_obj)
    names = [group.name for group in mesh_obj.vertex_groups]
    own = np.array([not any(names[g.group] in hair_bones and g.weight > 0.1 for g in v.groups)
                    for v in mesh_obj.data.vertices])
    pick = own & (verts[:, 2] > head_joint[2]) & (np.hypot(*(verts[:, :2] - head_joint[:2]).T) < 0.15)
    if pick.sum() < 50:
        return None
    low, high = verts[pick].min(axis=0), verts[pick].max(axis=0)
    centre = (low + high) / 2
    radius = HEAD_FILL * 0.5 * min(high[0] - low[0], high[1] - low[1])
    return spring_sim.Collider(BONE_NAMES["head"], radius, offset=centre)


def _clear_of_resting_hair(colliders, springs):
    """Shrink colliders that would reach resting hair joints, so hair isn't shoved off the body at rest.

    Hair modelled lying on the body leaves little room, so the fit to the body gives way.
    """
    if not springs:
        return colliders
    joints = np.vstack([chain.points[1:] for chain in springs])  # chain roots never collide
    hit_radii = np.concatenate([np.full(len(chain.points) - 1, chain.hit_radius) for chain in springs])
    kept = []
    for collider in colliders:
        start = collider.offset
        if collider.tail is None:
            distance = np.linalg.norm(joints - start, axis=1)
        else:
            shaft = collider.tail - start
            along = np.clip((joints - start) @ shaft / np.dot(shaft, shaft), 0.0, 1.0)
            distance = np.linalg.norm(joints - (start + along[:, None] * shaft), axis=1)
        radius = max(MIN_FIT_SHARE * collider.radius,
                     min(collider.radius, float(np.min(distance - hit_radii)) - REST_CLEARANCE))
        if radius > REST_CLEARANCE:
            kept.append(spring_sim.Collider(collider.bone, radius, collider.offset, collider.tail))
    return kept


def _body_colliders(volume, joints):
    j = {key: np.asarray(value, dtype=float) for key, value in joints.items()}
    head_length = 2.5 * np.linalg.norm(j["head"] - j["neck"])
    colliders = [
        spring_sim.Collider(
            BONE_NAMES["head"], 0.4 * head_length, offset=j["head"] + np.array([0.0, 0.0, 0.55 * head_length])
        )
    ]
    bottom, top = j["hips"][2], j["neck"][2]
    for fraction in TORSO_COLLIDER_HEIGHTS:
        z = bottom + fraction * (top - bottom)
        cx, cy, rx, ry = volume.torso_ellipse(z)
        half_span = max(rx - ry, 0.0)
        colliders.append(
            spring_sim.Collider(
                _torso_bone(fraction),
                COLLIDER_SHRINK * ry,
                offset=np.array([cx - half_span, cy, z]),
                tail=np.array([cx + half_span, cy, z]) if half_span > 1e-3 else None,
            )
        )
    for side in ("left", "right"):
        # neck to shoulder, as thick as the upper arm
        arm_radius = next((r for (a, _b), (_s, _e, r) in zip(body_volume.LIMBS, volume.capsules)
                           if a == f"{side}_upper_arm"), 0.0)
        if arm_radius > 0.0:
            colliders.append(spring_sim.Collider(BONE_NAMES["upper_chest"], COLLIDER_SHRINK * arm_radius,
                                                 offset=j["neck"], tail=j[f"{side}_upper_arm"]))
    for (start_key, _end_key), (start, end, radius) in zip(body_volume.LIMBS, volume.capsules):
        if radius > 0.0:
            colliders.append(spring_sim.Collider(BONE_NAMES[start_key], COLLIDER_SHRINK * radius, offset=start, tail=end))
    return colliders


def _torso_bone(fraction):
    """The spine bone spanning this fraction of the way from hips to neck."""
    if fraction < SPINE_AT:
        return BONE_NAMES["hips"]
    if fraction < CHEST_AT:
        return BONE_NAMES["spine"]
    if fraction < UPPER_CHEST_AT:
        return BONE_NAMES["chest"]
    return BONE_NAMES["upper_chest"]


def head_turn_and_sway(armature, seconds=4.0, fps=60):
    """A lively idle: the head turns and nods while the whole body sways and twists."""
    bones = armature.data.bones
    hips = np.array(list(bones[BONE_NAMES["hips"]].head_local))
    neck = np.array(list(bones[BONE_NAMES["neck"]].head_local))
    head = np.array(list(bones[BONE_NAMES["head"]].head_local))
    body_bones = [bone.name for bone in bones if not bone.name.startswith("Hair")]

    poses, torsos = [], []
    for frame in range(int(seconds * fps)):
        t = frame / fps
        torso = _translation(0.08 * math.sin(2 * math.pi * 0.6 * t), 0.0, 0.0) @ _rotation_about(
            hips, "Z", math.radians(10) * math.sin(2 * math.pi * 0.5 * t)
        )
        turn = math.radians(35) * math.sin(2 * math.pi * 0.8 * t)
        nod = math.radians(15) * math.sin(2 * math.pi * 0.4 * t)
        neck_pose = torso @ _rotation_about(neck, "Z", 0.5 * turn)
        head_pose = neck_pose @ _rotation_about(head, "Z", 0.5 * turn) @ _rotation_about(head, "X", nod)
        pose = dict.fromkeys(body_bones, torso)
        pose[BONE_NAMES["neck"]] = neck_pose
        pose[BONE_NAMES["head"]] = head_pose
        poses.append(pose)
        torsos.append(torso)
    return Motion(poses, torsos, 1.0 / fps)


def standing_still(armature, seconds=2.0, fps=60):
    """The body held in its rest pose."""
    body_bones = [bone.name for bone in armature.data.bones if not bone.name.startswith("Hair")]
    pose = dict.fromkeys(body_bones, np.eye(4))
    frames = int(seconds * fps)
    return Motion([pose] * frames, [np.eye(4)] * frames, 1.0 / fps)


def clipping_report(mesh_obj, armature, design, motion):
    """How far hanging hair sinks into the body while the springs follow `motion`."""
    verts = meshdata.world_vertices(mesh_obj)
    group_names = [group.name for group in mesh_obj.vertex_groups]
    influences = [[(group_names[g.group], g.weight) for g in v.groups if g.weight > 0] for v in mesh_obj.data.vertices]
    hair_bones = {name: (c, j) for c, chain in enumerate(design.chains) for j, name in enumerate(chain.names)}
    volume = body_volume.fit(mesh_obj, _joints(armature), set(hair_bones))
    hair = [i for i, weights in enumerate(influences) if any(name in hair_bones for name, _ in weights)][::VERTEX_STEP]
    frames = spring_sim.simulate(design.chains, design.colliders, motion, len(motion.poses), motion.dt)

    def in_torso_space(frame_index):
        """Posed hair vertices, carried back into the torso's rest space."""
        frame, pose = frames[frame_index], motion.poses[frame_index]
        posed = np.zeros((len(hair), 3))
        for k, vertex in enumerate(hair):
            rest = verts[vertex]
            for name, weight in influences[vertex]:
                if name in hair_bones:
                    c, j = hair_bones[name]
                    anchor = design.chains[c].points[j]
                    posed[k] += weight * (frame.positions[c][j] + frame.rotations[c][j] @ (rest - anchor))
                else:
                    transform = pose.get(name, pose[BONE_NAMES["head"]])
                    posed[k] += weight * (transform[:3, :3] @ rest + transform[:3, 3])
        inverse = np.linalg.inv(motion.torso[frame_index])
        return posed @ inverse[:3, :3].T + inverse[:3, 3]

    at_rest = np.maximum(volume.depth(in_torso_space(0)), 0.0)
    deepest, worst_frame, worst_sample, clipping, samples = 0.0, 0, None, 0, 0
    for frame_index in range(FRAME_STEP, len(frames), FRAME_STEP):
        extra = volume.depth(in_torso_space(frame_index)) - at_rest
        if extra.size and extra.max() > deepest:
            deepest, worst_frame, worst_sample = float(extra.max()), frame_index, int(extra.argmax())
        clipping += int(np.sum(extra > CLIP_TOLERANCE))
        samples += extra.size

    detail = "none"
    if worst_sample is not None:
        vertex = hair[worst_sample]
        bone = max(influences[vertex], key=lambda item: item[1])[0]
        where = in_torso_space(worst_frame)[worst_sample]
        detail = (
            f"vertex at rest {np.round(verts[vertex], 3).tolist()} moved by {bone} to "
            f"{np.round(where, 3).tolist()} in the torso's rest space"
        )
    return ClippingReport(deepest, clipping / max(samples, 1), worst_frame, detail)


def _joints(armature):
    """Joint positions keyed as landmarks.detect_body names them, read back from the armature."""
    bones = armature.data.bones
    return {key: np.array(list(bones[name].head_local)) for key, name in BONE_NAMES.items() if name in bones}


def _translation(x, y, z):
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, z)
    return matrix


def _rotation_about(pivot, axis, angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.eye(4)
    if axis == "Z":
        rotation[:2, :2] = [[cosine, -sine], [sine, cosine]]
    else:
        rotation[1:3, 1:3] = [[cosine, -sine], [sine, cosine]]
    return _translation(*pivot) @ rotation @ _translation(*(-pivot))
