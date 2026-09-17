"""A hole-free stand-in for the body under the hair: stacked ellipses for the torso, capsules for limbs.

Image-to-3D meshes fuse long hair onto the back, so the skin under it doesn't exist as
geometry. Simple shapes fitted to the visible body give a volume to test swinging hair
against, with the hidden back taken as a mirror of the front.
"""

import numpy as np

from .meshdata import world_vertices
from .rig import BONE_NAMES

TORSO_BANDS = 16
# Joint pairs, as named by landmarks.detect_body, that get a capsule.
LIMBS = (
    ("left_upper_arm", "left_lower_arm"),
    ("left_lower_arm", "left_hand"),
    ("right_upper_arm", "right_lower_arm"),
    ("right_lower_arm", "right_hand"),
    ("left_upper_leg", "left_lower_leg"),
    ("left_lower_leg", "left_foot"),
    ("right_upper_leg", "right_lower_leg"),
    ("right_lower_leg", "right_foot"),
)


class BodyVolume:
    def __init__(self, bands, capsules):
        self.bands = bands  # rows of (z, center x, center y, radius x, radius y), bottom to top
        self.capsules = capsules  # (start, end, radius)

    def torso_ellipse(self, z):
        """(center x, center y, radius x, radius y) of the torso at height z."""
        return tuple(float(np.interp(z, self.bands[:, 0], self.bands[:, i])) for i in range(1, 5))

    def depth(self, points):
        """How far each point lies inside the body, in metres; zero or less outside."""
        points = np.asarray(points, dtype=float)
        deepest = np.full(len(points), -np.inf)
        z = points[:, 2]
        in_torso = (z >= self.bands[0, 0]) & (z <= self.bands[-1, 0])
        if in_torso.any():
            cx, cy, rx, ry = (np.interp(z[in_torso], self.bands[:, 0], self.bands[:, i]) for i in range(1, 5))
            spread = np.hypot((points[in_torso, 0] - cx) / rx, (points[in_torso, 1] - cy) / ry)
            deepest[in_torso] = (1.0 - spread) * np.minimum(rx, ry)
        for start, end, radius in self.capsules:
            shaft = end - start
            along = np.clip((points - start) @ shaft / np.dot(shaft, shaft), 0.0, 1.0)
            distance = np.linalg.norm(points - (start + along[:, None] * shaft), axis=1)
            deepest = np.maximum(deepest, radius - distance)
        return deepest


def fit(mesh_obj, joints, hair_bones):
    """Fit the volume to the mesh's body vertices: everything below the neck not moved by hair or head bones."""
    j = {key: np.asarray(value, dtype=float) for key, value in joints.items()}
    verts = world_vertices(mesh_obj)
    points = verts[_body_vertices(mesh_obj, hair_bones) & (verts[:, 2] < j["neck"][2])]
    return BodyVolume(_torso_bands(points, j), [_capsule(points, j[a], j[b]) for a, b in LIMBS])


def _body_vertices(mesh_obj, hair_bones):
    names = [group.name for group in mesh_obj.vertex_groups]
    head_bones = {BONE_NAMES["head"], BONE_NAMES["neck"]}
    mask = np.ones(len(mesh_obj.data.vertices), dtype=bool)
    for vertex in mesh_obj.data.vertices:
        weights = [(names[g.group], g.weight) for g in vertex.groups if g.weight > 0]
        if weights and (
            any(name in hair_bones for name, _ in weights) or max(weights, key=lambda item: item[1])[0] in head_bones
        ):
            mask[vertex.index] = False
    return mask


def _torso_bands(points, j):
    """Ellipse radii per height band, measured outward from the spine between hips and neck."""
    bottom, top = j["hips"][2], j["neck"][2]
    reach = max(abs(j[f"{side}_upper_arm"][0] - j["hips"][0]) for side in ("left", "right"))
    edges = np.linspace(bottom, top, TORSO_BANDS + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        z = (low + high) / 2
        t = (z - bottom) / (top - bottom)
        cx, cy = (1 - t) * j["hips"][:2] + t * j["neck"][:2]
        band = points[(points[:, 2] >= low) & (points[:, 2] < high) & (np.abs(points[:, 0] - cx) <= reach)]
        if len(band) < 8:
            continue
        rows.append((z, cx, cy, np.percentile(np.abs(band[:, 0] - cx), 95), np.percentile(np.abs(band[:, 1] - cy), 95)))
    return np.array(rows)


def _capsule(points, start, end):
    """A capsule along start-end whose radius is the typical distance of the limb's surface from it."""
    shaft = end - start
    along = (points - start) @ shaft / np.dot(shaft, shaft)
    distance = np.linalg.norm(points - (start + np.clip(along, 0.0, 1.0)[:, None] * shaft), axis=1)
    near = (along > 0.2) & (along < 0.8) & (distance < 0.35 * np.linalg.norm(shaft))
    return start, end, float(np.median(distance[near])) if near.any() else 0.0
