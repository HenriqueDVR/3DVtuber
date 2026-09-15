"""Bone chains for hanging hair, so VRM spring bones can swing it.

Hair is found by its texture color, grown outward from the crown across the
mesh. The part hanging below the head is split into angular sectors around the
head's vertical axis; each sector holding enough hair becomes a chain of bones,
and its vertices are reweighted from the torso onto that chain. Neighbouring
chains share weight across sector borders so the hair doesn't tear.
"""

import math

import bpy
import numpy as np
from mathutils import Vector

from . import texture_space
from .face import HAIR_DISTANCE
from .meshdata import world_vertices
from .rig import BONE_NAMES, make_active

SECTORS = 8
SECTOR_WIDTH = 2 * math.pi / SECTORS
BONES_PER_CHAIN = 4
MIN_SECTOR_VERTICES = 30
MAX_INFLUENCES = 4
# Fractions of the head radius.
HEAD_REACH = 1.3
HANGS_BELOW_CENTER = 0.35
MIN_CHAIN_LENGTH = 0.4
# Fraction of a chain's length over which weight hands over from the head bone.
ROOT_BLEND = 0.12


def build_hair_bones(mesh_obj, armature, joints, features):
    """Add hair bone chains under the head bone and reweight hanging hair onto them.

    Run after skinning. Returns each chain as a list of bone names, root first.
    """
    verts = world_vertices(mesh_obj)
    head = _Head(verts, joints, features)
    hair = _hair_vertices(mesh_obj, verts, features["hair_color"], head)
    hanging = hair[verts[hair, 2] < head.center[2] - HANGS_BELOW_CENTER * head.radius]
    angles = head.angle(verts[hanging])
    sectors = np.floor((angles + math.pi) / SECTOR_WIDTH).astype(int) % SECTORS

    chains = []
    for sector in range(SECTORS):
        points = verts[hanging[sectors == sector]]
        if len(points) < MIN_SECTOR_VERTICES or np.ptp(points[:, 2]) < MIN_CHAIN_LENGTH * head.radius:
            continue
        names = [f"Hair{sector}_{k}" for k in range(BONES_PER_CHAIN)]
        chains.append(_Chain(names, _chain_positions(points), -math.pi + (sector + 0.5) * SECTOR_WIDTH))
    if chains:
        _add_bones(armature, chains)
        _reweight(mesh_obj, verts, hanging, angles, chains)
    return [chain.names for chain in chains]


class _Head:
    def __init__(self, verts, joints, features):
        base = np.asarray(joints["head"], dtype=float)
        eyes = (np.asarray(features["left_eye"]) + np.asarray(features["right_eye"])) / 2
        top = float(verts[:, 2].max())
        self.radius = (top - base[2]) / 2
        self.center = np.array([base[0], eyes[1] + self.radius, (top + base[2]) / 2])

    def angle(self, points):
        """0 in front of the face, positive toward character left (+X), ±pi behind."""
        return np.arctan2(points[:, 0] - self.center[0], -(points[:, 1] - self.center[1]))


class _Chain:
    def __init__(self, names, positions, angle):
        self.names = names
        self.positions = positions
        self.angle = angle
        self.root_z = positions[0][2]
        self.tip_z = positions[-1][2]


def _hair_vertices(mesh_obj, verts, hair_color, head):
    """Indices of hair-colored vertices in connected patches that reach the head.

    Strands in front of the face or shoulders often join the rest of the hair only
    through skin-colored vertices, so each patch is judged on its own: it counts as
    hair if any part of it lies near the head.
    """
    colors = texture_space.vertex_colors(mesh_obj, texture_space.base_color_image(mesh_obj))
    hair_colored = np.linalg.norm(colors - np.asarray(hair_color), axis=1) < HAIR_DISTANCE
    edges = np.empty(len(mesh_obj.data.edges) * 2, dtype=np.int64)
    mesh_obj.data.edges.foreach_get("vertices", edges)
    a, b = edges[0::2], edges[1::2]
    within_hair = hair_colored[a] & hair_colored[b]
    labels = _component_labels(len(verts), a[within_hair], b[within_hair])
    near_head = hair_colored & (np.linalg.norm(verts - head.center, axis=1) < HEAD_REACH * head.radius)
    return np.nonzero(hair_colored & np.isin(labels, labels[near_head]))[0]


def _component_labels(count, a, b):
    """Label each vertex with the smallest vertex index in its connected component."""
    labels = np.arange(count)
    while True:
        updated = labels.copy()
        np.minimum.at(updated, a, labels[b])
        np.minimum.at(updated, b, labels[a])
        updated = updated[updated]
        if np.array_equal(updated, labels):
            return labels
        labels = updated


def _chain_positions(points):
    """BONES_PER_CHAIN + 1 joint positions from the top of the strand to its tip."""
    top, bottom = points[:, 2].max(), points[:, 2].min()
    step = (top - bottom) / BONES_PER_CHAIN
    positions = []
    for k in range(BONES_PER_CHAIN + 1):
        z = top - k * step
        distance = np.abs(points[:, 2] - z)
        band = distance < step / 2
        if not band.any():
            band = distance <= distance.min()
        x, y = points[band, :2].mean(axis=0)
        positions.append(np.array([x, y, z]))
    return positions


def _add_bones(armature, chains):
    make_active(armature)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = armature.data.edit_bones
    for chain in chains:
        parent = bones[BONE_NAMES["head"]]
        for k, name in enumerate(chain.names):
            bone = bones.new(name)
            bone.head = Vector(chain.positions[k].tolist())
            bone.tail = Vector(chain.positions[k + 1].tolist())
            bone.parent = parent
            bone.use_connect = k > 0
            parent = bone
    bpy.ops.object.mode_set(mode="OBJECT")


def _reweight(mesh_obj, verts, hanging, angles, chains):
    head_group = BONE_NAMES["head"]
    new_weights = {}
    for vertex, angle in zip(hanging, angles):
        shares = [(chain, 1.0 - abs(_angle_between(angle, chain.angle)) / SECTOR_WIDTH) for chain in chains]
        shares = [(chain, share) for chain, share in shares if share > 0]
        if not shares:
            continue
        total_share = sum(share for _, share in shares)
        weights = {}
        for chain, share in shares:
            share /= total_share
            along = min(max((chain.root_z - verts[vertex, 2]) / (chain.root_z - chain.tip_z), 0.0), 1.0)
            head_part = max(0.0, 1.0 - along / ROOT_BLEND)
            weights[head_group] = weights.get(head_group, 0.0) + share * head_part
            position = along * BONES_PER_CHAIN - 0.5
            lower = int(min(max(math.floor(position), 0), BONES_PER_CHAIN - 1))
            upper = min(lower + 1, BONES_PER_CHAIN - 1)
            blend = min(max(position - lower, 0.0), 1.0)
            for index, part in ((lower, 1.0 - blend), (upper, blend)):
                name = chain.names[index]
                weights[name] = weights.get(name, 0.0) + share * (1.0 - head_part) * part
        strongest = sorted(weights.items(), key=lambda item: -item[1])[:MAX_INFLUENCES]
        total = sum(weight for _, weight in strongest)
        new_weights[int(vertex)] = {name: weight / total for name, weight in strongest if weight > 0}

    groups = mesh_obj.vertex_groups
    for chain in chains:
        for name in chain.names:
            if name not in groups:
                groups.new(name=name)
    reweighted = list(new_weights)
    for group in groups:
        group.remove(reweighted)
    for vertex, weights in new_weights.items():
        for name, weight in weights.items():
            groups[name].add([vertex], weight, "REPLACE")


def _angle_between(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi
