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
    ):
        self.points = np.asarray(points, dtype=float)
        self.parent = parent
        self.stiffness = stiffness
        self.gravity_power = gravity_power
        self.drag_force = drag_force
        self.hit_radius = hit_radius
        self.gravity_dir = np.asarray(gravity_dir, dtype=float)
        self.names = list(names)


class Collider:
    """A sphere, or a capsule when tail is given, attached to a body bone. Positions are rest pose."""

    def __init__(self, bone, radius, offset, tail=None):
        self.bone = bone
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


def simulate(chains, colliders, body_motion, frames, dt=1 / 60):
    """Step every chain for `frames` frames; body_motion(frame) gives body bone transforms."""
    states = [_ChainState(chain) for chain in chains]
    result = []
    for frame in range(frames):
        pose = body_motion(frame)
        steps = [state.step(pose, colliders, dt) for state in states]
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

    def step(self, pose, colliders, dt):
        chain = self.chain
        parent = pose[chain.parent]
        rotation = parent[:3, :3]
        head = rotation @ chain.points[0] + parent[:3, 3]
        positions, rotations = [head], []
        for j, length in enumerate(self.lengths):
            axis = rotation @ self.rest_directions[j]
            inertia = (self.tails[j] - self.previous_tails[j]) * (1.0 - chain.drag_force)
            tail = self.tails[j] + inertia + axis * chain.stiffness * dt + chain.gravity_dir * chain.gravity_power * dt
            tail = head + _unit(tail - head) * length
            for collider in colliders:
                pushed = collider.push_out(tail, chain.hit_radius, pose)
                if pushed is not None:
                    tail = head + _unit(pushed - head) * length
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
    cross = np.cross(a, b)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(a, b))
    if sine < 1e-9:
        if cosine > 0.0:
            return np.eye(3)
        perpendicular = _unit(np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]))
        return 2.0 * np.outer(perpendicular, perpendicular) - np.eye(3)
    skew = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)
