"""Characters made of separate parts, like Tripo's P2 meshes.

Some generators keep hair, eye inserts and a mouth insert as separate pieces of one
mesh instead of fusing everything and painting the face on. Knowing the pieces lets
skinning keep hair off the arms, hair physics follow the real hair, and the face rig
close real eyes.
"""

from typing import NamedTuple

import numpy as np
from mathutils.kdtree import KDTree

from . import meshdata, texture_space
from .face import _dominant_color
from .meshdata import world_vertices

# Vertices closer than this are the same point (meshes split vertices along UV seams).
WELD_TOLERANCE = 1e-6
# An insert fits in this fraction of the head's height.
INSERT_SIZE = 0.3
# A part is hair when at least this share of its vertices is near the hair color.
HAIR_SHARE = 0.5
HAIR_COLOR_DISTANCE = 0.2
# Crown: the top of the model, as a fraction of the head's height, where hair color is sampled.
CROWN = 0.1
# Hair fused into another piece counts when a hair-coloured stretch of it holds this share of the
# vertices (model A's T-pose FBX: its smallest real lock 0.38%; the eyebrows and trims beside the face 0.1%).
FUSED_HAIR = 0.003
# A stretch of it below the shoulders counts when it lies this close (m) to one that reaches above.
FUSED_GAP = 0.02


class MeshParts(NamedTuple):
    labels: np.ndarray  # part id per vertex
    hair: np.ndarray  # True for vertices of hair parts
    face: int  # part holding the eye sockets, or -1
    eyes: dict  # "left"/"right" -> eye insert part; empty when there are no eye inserts
    mouth: int  # mouth insert part, or -1
    hair_color: np.ndarray
    face_area: np.ndarray  # True for the face part's vertices, only its head end when it is the body too

    @property
    def has_eye_inserts(self):
        return bool(self.eyes)

    def fixed(self):
        """Vertices a face shape must not move: everything but the face part and its inserts."""
        movable = self.face_area.copy()
        for part in list(self.eyes.values()) + [self.mouth]:
            movable |= self.labels == part
        return ~movable


def split(verts, edges):
    """Label each vertex with its connected part, joining vertices that share a position."""
    keys = np.round(np.asarray(verts) / WELD_TOLERANCE).astype(np.int64)
    _, weld = np.unique(keys, axis=0, return_inverse=True)
    weld = weld.ravel()
    a, b = weld[edges[:, 0]], weld[edges[:, 1]]
    labels = np.arange(weld.max() + 1)
    while True:
        updated = labels.copy()
        np.minimum.at(updated, a, labels[b])
        np.minimum.at(updated, b, labels[a])
        updated = updated[updated]
        if np.array_equal(updated, labels):
            break
        labels = updated
    return labels[weld]


def _components(count, a, b):
    """Label each of `count` points with the smallest point joined to it by the edges a-b."""
    labels = np.arange(count)
    while True:
        updated = labels.copy()
        np.minimum.at(updated, a, labels[b])
        np.minimum.at(updated, b, labels[a])
        updated = updated[updated]
        if np.array_equal(updated, labels):
            return labels
        labels = updated


def find(mesh_obj, joints):
    """Split the mesh into parts and pick out the hair, the eye and mouth inserts, and the face."""
    verts = world_vertices(mesh_obj)
    edges = np.empty(len(mesh_obj.data.edges) * 2, dtype=np.int64)
    mesh_obj.data.edges.foreach_get("vertices", edges)
    labels = split(verts, edges.reshape(-1, 2))
    ids = np.unique(labels)

    top = float(verts[:, 2].max())
    head_height = top - float(joints["neck"][2])
    head = verts[:, 2] > joints["head"][2]
    head_centre_y = float(np.median(verts[head, 1])) if head.any() else float(joints["head"][1])

    low, high, centre = _boxes(verts, labels, ids)
    inserts = [
        p
        for p in ids
        if (high[p] - low[p]).max() < INSERT_SIZE * head_height
        and centre[p][2] > joints["neck"][2]
        and centre[p][1] < head_centre_y
    ]
    eyes = _eye_pair(inserts, low, high, centre, float(joints["head"][0]), float(joints["head"][2]))
    mouth = _mouth(inserts, eyes, low, high, centre) if eyes else -1

    face = -1
    if eyes:
        others = ~np.isin(labels, inserts)
        candidates = np.nonzero(others)[0]
        nearest = [candidates[np.argmin(np.linalg.norm(verts[candidates] - centre[p], axis=1))] for p in eyes.values()]
        face = int(labels[nearest[0]])

    colors = texture_space.vertex_colors(mesh_obj, texture_space.base_color_image(mesh_obj))
    hair_color = _dominant_color(colors[verts[:, 2] > top - CROWN * head_height])
    hair_colored = np.linalg.norm(colors - hair_color, axis=1) < HAIR_COLOR_DISTANCE
    share = np.bincount(labels, weights=hair_colored, minlength=labels.max() + 1) / np.maximum(
        np.bincount(labels, minlength=labels.max() + 1), 1
    )
    excluded = set(inserts) | {face}
    hair_parts = [p for p in ids if share[p] >= HAIR_SHARE and high[p][2] > joints["head"][2] and p not in excluded]
    hair = np.isin(labels, hair_parts)
    face_area = labels == face
    if face >= 0 and low[face][2] < joints["neck"][2] - head_height:
        # the face is one surface with the whole body (model A's T-pose FBX): only its head end is the
        # face, or the head-top clean-up repaints the up-facing arms and the jaw's chin is a foot.
        # Its hair is usually that same surface too, and swings rather than moving with the face
        hair |= fused_hair(mesh_obj, verts, labels, hair_colored & (labels == face), joints)
        face_area &= (verts[:, 2] > joints["neck"][2]) & ~hair
    return MeshParts(labels, hair, face, eyes, mouth, hair_color, face_area)


def fused_hair(mesh_obj, verts, labels, candidates, joints):
    """Hair modelled as one surface with the body: the hair-coloured stretches of `candidates`
    (hair-coloured vertices of the face part, when it is the whole body too) big enough to be hair
    and reaching above the shoulders, with their holes closed within their piece
    (`roles.close_region`).

    Model A's T-pose FBX has its long back hair and the locks over its shoulders in the body's surface:
    left out, they had no springs, kept the rig's arm weights and floated beside the hands. Only the
    face part is searched: model B's mantle, collar and model B face are her hat's colour, and her face
    is a piece of its own.
    """
    from . import roles
    found = np.zeros(len(verts), dtype=bool)
    if not candidates.any():
        return found
    weld, a, b = meshdata.welded_edges(mesh_obj, verts)
    member = np.zeros(weld.max() + 1, dtype=bool)
    member[weld[candidates]] = True
    keep = member[a] & member[b]
    cluster = _components(len(member), a[keep], b[keep])[weld]
    shoulders = max(float(joints["left_upper_arm"][2]), float(joints["right_upper_arm"][2]))
    ids, counts = np.unique(cluster[candidates], return_counts=True)
    stretches = [candidates & (cluster == ident) for ident, count in zip(ids, counts) if count >= FUSED_HAIR * len(verts)]
    for members in stretches:
        if verts[members, 2].max() > shoulders:
            found |= members
    # the hair's lower stretches, parted from the rest by strands painted in shadow (model A's T-pose:
    # 179 and 171 vertices behind the hips, 25 cm from the swinging hair when they stayed)
    rest = [m for m in stretches if not (m & found).any()]
    while found.any() and rest:
        tree = KDTree(int(found.sum()))
        for i, point in enumerate(verts[found]):
            tree.insert(point, i)
        tree.balance()
        near = [m for m in rest if min(tree.find(p)[2] for p in verts[m]) < FUSED_GAP]
        if not near:
            break
        for members in near:
            found |= members
        rest = [m for m in rest if not (m & found).any()]
    if found.any():
        piece = np.isin(labels, np.unique(labels[found]))
        found = roles.close_region(mesh_obj, piece, found)
    return found


def face_features(mesh_obj, found):
    """Eye and mouth positions and sizes from the inserts, keyed like face.detect_features."""
    verts = world_vertices(mesh_obj)

    def centre_and_size(part):
        points = verts[found.labels == part]
        return points.mean(axis=0), np.array([np.ptp(points[:, 0]), np.ptp(points[:, 2])])

    left, left_size = centre_and_size(found.eyes["left"])
    right, right_size = centre_and_size(found.eyes["right"])
    features = {"left_eye": left, "right_eye": right, "eye_size": (left_size + right_size) / 2}
    spread = abs(left[0] - right[0])
    if found.mouth >= 0:
        mouth, mouth_size = centre_and_size(found.mouth)
        features.update(mouth=mouth, mouth_width=float(mouth_size[0]), mouth_estimated=False)
    else:
        middle = (left + right) / 2
        features.update(
            mouth=middle - np.array([0.0, 0.0, 0.92 * spread]), mouth_width=0.25 * spread, mouth_estimated=True
        )

    colors = texture_space.vertex_colors(mesh_obj, texture_space.base_color_image(mesh_obj))
    middle = (left + right) / 2
    cheeks = (
        found.face_area
        & (np.abs(verts[:, 0] - middle[0]) < spread)
        & (verts[:, 2] < middle[2])
        & (verts[:, 2] > features["mouth"][2])
    )
    features["skin_color"] = _dominant_color(colors[cheeks]) if cheeks.any() else np.ones(3)
    features["hair_color"] = found.hair_color
    return features


def _boxes(verts, labels, ids):
    size = labels.max() + 1
    low = np.full((size, 3), np.inf)
    high = np.full((size, 3), -np.inf)
    np.minimum.at(low, labels, verts)
    np.maximum.at(high, labels, verts)
    counts = np.bincount(labels, minlength=size)
    centre = np.stack([np.bincount(labels, weights=verts[:, k], minlength=size) for k in range(3)], axis=1)
    centre /= np.maximum(counts, 1)[:, None]
    return low, high, centre


def _eye_pair(inserts, low, high, centre, head_x, head_z):
    """The largest pair of inserts mirrored across the head's centre line, above the head joint."""
    candidates = [p for p in inserts if centre[p][2] > head_z]
    best, best_area = {}, 0.0
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            spread = abs(centre[a][0] - centre[b][0])
            size_a, size_b = high[a] - low[a], high[b] - low[b]
            area_a, area_b = size_a[0] * size_a[2], size_b[0] * size_b[2]
            mirrored = abs((centre[a][0] + centre[b][0]) / 2 - head_x) < 0.25 * spread
            level = abs(centre[a][2] - centre[b][2]) < 0.25 * spread
            apart = spread > max(size_a[0], size_b[0])
            alike = max(area_a, area_b) < 3 * min(area_a, area_b)
            if mirrored and level and apart and alike and area_a + area_b > best_area:
                left, right = (a, b) if centre[a][0] > centre[b][0] else (b, a)
                best, best_area = {"left": int(left), "right": int(right)}, area_a + area_b
    return best


def _mouth(inserts, eyes, low, high, centre):
    """The widest insert centred below the eyes."""
    left, right = centre[eyes["left"]], centre[eyes["right"]]
    spread = abs(left[0] - right[0])
    middle = (left + right) / 2
    below = [
        p
        for p in inserts
        if p not in eyes.values()
        and abs(centre[p][0] - middle[0]) < 0.3 * spread
        and 0.3 * spread < middle[2] - centre[p][2] < 2.0 * spread
    ]
    return int(max(below, key=lambda p: high[p][0] - low[p][0])) if below else -1
