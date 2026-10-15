"""A small VRM 1.0 spring bone simulation, following three-vrm's VRMSpringBoneJoint.

Used to check hair physics, such as hair swinging through the body, before export.
Positions are world space in metres. Body bones are given per frame as transforms
from their rest pose to their current pose.
"""

from typing import NamedTuple

import numpy as np


class Chain:
    """A strand of spring bones hanging from a body bone.

    points: rest positions of each bone's head followed by the last bone's tail.
    """

    def __init__(
        self,
        points,
        parent,
        stiffness,
        gravity_power,
        drag_force,
        hit_radius,
        gravity_dir=(0.0, 0.0, -1.0),
        names=(),
        tip_stiffness=None,
        collides_with=None,
        groups=(),
    ):
        self.points = np.asarray(points, dtype=float)
        self.parent = parent
        self.stiffness = stiffness
        self.gravity_power = gravity_power
        self.drag_force = drag_force
        self.hit_radius = hit_radius
        self.gravity_dir = np.asarray(gravity_dir, dtype=float)
        self.names = list(names)
        self.tip_stiffness = stiffness if tip_stiffness is None else tip_stiffness
        # the bones whose colliders push this chain; None: every collider
        self.collides_with = None if collides_with is None else tuple(collides_with)
        # colliders of these groups push it too (a Collider's `group`): the skirt's, for hanging sleeves
        self.groups = tuple(groups)


class Collider:
    """A sphere, or a capsule when tail is given, attached to a body bone. Positions are rest pose."""

    def __init__(self, bone, radius, offset, tail=None, group=None):
        self.bone = bone
        # None: the body's, every chain's (as `collides_with` allows); else only chains naming it in `groups`
        self.group = group
        self.radius = radius
        self.offset = np.asarray(offset, dtype=float)
        self.tail = None if tail is None else np.asarray(tail, dtype=float)

    def push_out(self, point, hit_radius, pose):
        """The point moved out of the collider, or None if it isn't touching."""
        transform = pose[self.bone]
        start = transform[:3, :3] @ self.offset + transform[:3, 3]
        closest = start
        if self.tail is not None:
            shaft = transform[:3, :3] @ self.tail + transform[:3, 3] - start
            along = np.dot(point - start, shaft)
            closest = start + shaft * min(max(along / np.dot(shaft, shaft), 0.0), 1.0)
        away = point - closest
        distance = np.linalg.norm(away)
        overlap = self.radius + hit_radius - distance
        if overlap <= 0.0 or distance < 1e-9:
            return None
        return point + away / distance * overlap


class Frame(NamedTuple):
    positions: list  # per chain, (bones + 1, 3) joint positions
    rotations: list  # per chain, (bones, 3, 3) rotation of each bone from its rest pose


# A joint further than this (m) from touching a collider is not put to it; nearer, the collider's own
# sum decides (`push_out`). A million times the rounding of the two sums: the numbers are the very
# ones that asking every collider gave.
NEAR = 1e-6


class _Reach:
    """Where every collider is in one frame's pose, to tell at once which of them a point comes near.

    `push_out` places its collider anew for every point it is asked about: with twenty colliders and
    forty thousand joint steps, 756,000 times in one avatar's spring check, ten of its seventeen
    seconds, for an answer that is nearly always "not touching".
    """

    def __init__(self, colliders, pose):
        count = len(colliders)
        self.starts, self.shafts = np.zeros((count, 3)), np.zeros((count, 3))
        self.squares, self.radii = np.ones(count), np.zeros(count)
        # the ones only `push_out` can answer for: on a bone this pose has no word of (it raises, as
        # it did), or a capsule with no length
        self.ask = np.zeros(count, dtype=bool)
        for i, collider in enumerate(colliders):
            try:
                transform = pose[collider.bone]
            except KeyError:
                self.ask[i] = True
                continue
            self.starts[i] = start = transform[:3, :3] @ collider.offset + transform[:3, 3]
            self.radii[i] = collider.radius
            if collider.tail is not None:
                shaft = transform[:3, :3] @ collider.tail + transform[:3, 3] - start
                square = float(np.dot(shaft, shaft))
                if square > 0.0 and np.isfinite(square):
                    self.shafts[i], self.squares[i] = shaft, square
                else:
                    self.ask[i] = True

    def near(self, point, hit_radius):
        """Which colliders a point with this hit radius touches, or all but touches."""
        away = point - self.starts
        along = np.clip((away * self.shafts).sum(axis=1) / self.squares, 0.0, 1.0)
        away -= self.shafts * along[:, None]
        return (self.radii + hit_radius - np.sqrt((away * away).sum(axis=1)) > -NEAR) | self.ask


def simulate(chains, colliders, body_motion, frames, dt=1 / 60):
    """Step every chain for `frames` frames; body_motion(frame) gives body bone transforms."""
    states = [_ChainState(chain) for chain in chains]
    result = []
    for frame in range(frames):
        pose = body_motion(frame)
        reach = _Reach(colliders, pose) if len(colliders) else None
        steps = [state.step(pose, colliders, dt, reach) for state in states]
        result.append(Frame([positions for positions, _ in steps], [rotations for _, rotations in steps]))
    return result


class _ChainState:
    def __init__(self, chain):
        self.chain = chain
        segments = np.diff(chain.points, axis=0)
        self.lengths = np.linalg.norm(segments, axis=1)
        self.rest_directions = segments / self.lengths[:, None]
        self.tails = chain.points[1:].copy()
        self.previous_tails = chain.points[1:].copy()
        self.mine = None  # which of the colliders push this chain, once they are known

    def step(self, pose, colliders, dt, reach=None):
        """One frame. `reach` (the colliders placed in this pose) only saves asking the colliders a
        joint is nowhere near: with or without it, each is put to the joint in turn and the joint is
        moved by every one it touches, in their order."""
        chain = self.chain
        parent = pose[chain.parent]
        rotation = parent[:3, :3]
        head = rotation @ chain.points[0] + parent[:3, 3]
        positions, rotations = [head], []
        if self.mine is None:
            self.mine = np.array([collider.group in chain.groups if collider.group is not None else
                                  chain.collides_with is None or collider.bone in chain.collides_with
                                  for collider in colliders], dtype=bool)
        for j, length in enumerate(self.lengths):
            axis = rotation @ self.rest_directions[j]
            inertia = (self.tails[j] - self.previous_tails[j]) * (1.0 - chain.drag_force)
            tail = self.tails[j] + inertia + axis * chain.stiffness * dt + chain.gravity_dir * chain.gravity_power * dt
            tail = head + _unit(tail - head) * length
            near = self.mine if reach is None else reach.near(tail, chain.hit_radius) & self.mine
            at = 0
            while near[at:].any():
                at += int(np.argmax(near[at:]))
                pushed = colliders[at].push_out(tail, chain.hit_radius, pose)
                at += 1
                if pushed is not None:
                    tail = head + _unit(pushed - head) * length
                    if reach is not None:  # moved: what it comes near now is asked of where it is now
                        near = reach.near(tail, chain.hit_radius) & self.mine
            self.previous_tails[j] = self.tails[j]
            self.tails[j] = tail
            rotation = _rotation_between(axis, _unit(tail - head)) @ rotation
            rotations.append(rotation)
            positions.append(tail)
            head = tail
        return np.array(positions), np.array(rotations)


def _unit(vector):
    return vector / np.linalg.norm(vector)


def _rotation_between(a, b):
    """Smallest rotation taking unit vector a onto unit vector b."""
    # (written out: np.cross takes three times as long to come to these same three sums)
    cross = np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(a, b))
    if sine < 1e-9:
        if cosine > 0.0:
            return np.eye(3)
        perpendicular = _unit(np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]))
        return 2.0 * np.outer(perpendicular, perpendicular) - np.eye(3)
    skew = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)
