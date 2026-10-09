"""Characters made of separate parts, like Tripo's P2 meshes.

Some generators keep hair, eye inserts and a mouth insert as separate pieces of one
mesh instead of fusing everything and painting the face on. Knowing the pieces lets
skinning keep hair off the arms, hair physics follow the real hair, and the face rig
close real eyes.
"""

from typing import NamedTuple

import numpy as np
from mathutils.kdtree import KDTree

from . import meshdata, orient, texture_space
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
# Fused hair taking this share of the body's surface below the hips isn't hair: the skin is its colour.
SKIN_AS_HAIR = 0.3
# Hair shaded in two tones: a piece this big (share of the vertices) hanging from above the head
# joint to below the neck is hair with only this share of it within SHADED_DISTANCE of the crown's
# colour. Model D's hair, dark in front with pale highlights: 19% of the model, 22% within
# HAIR_COLOR_DISTANCE and 33% within SHADED_DISTANCE; left out, it had no springs at all and kept
# the rig's arm weights. No other piece of models A, B and D that hangs so is over 2% of its model.
SHADED_DISTANCE = 0.3
SHADED_SHARE = 0.25
SHADED_SIZE = 0.05


# Ears and horns: a pair standing over the crown, one either side of it, each rising at least
# EAR_RISE of the height above it and holding EAR_MIN vertices. Walked down from its tip in steps
# of EAR_STEP of the height until it runs into the middle of the head or into the other one.
EAR_RISE = 0.015
EAR_MIN = 20
EAR_STEP = 0.004
# The two of a pair are alike: neither over this many times the other's size.
EAR_ALIKE = 3.0
# Where an ear is welded to what it stands beside (the pirate's to his hat's dome, 6 mm under its
# crown) the walk down stops at its tip. It then goes on by colour: over the surface no further in
# colour than this from the tip's, and the ear is that when it stays smaller than EAR_MAX of the
# height (a cat ear of the hair's own colour runs out into the hair, and only its tip is offered).
EAR_COLOUR = 0.2
EAR_MAX = 0.14
# How far behind the frontmost surface at the mouth still counts as the lips (m).
LIP_DEPTH = 0.005


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

    top = orient.head_top(verts, joints["head"][1]) or float(verts[:, 2].max())
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
    hair_color = _dominant_color(colors[(verts[:, 2] > top - CROWN * head_height) & (verts[:, 2] <= top)])
    hair_colored = np.linalg.norm(colors - hair_color, axis=1) < HAIR_COLOR_DISTANCE
    share = np.bincount(labels, weights=hair_colored, minlength=labels.max() + 1) / np.maximum(
        np.bincount(labels, minlength=labels.max() + 1), 1
    )
    excluded = set(inserts) | {face}
    size = np.bincount(labels, minlength=labels.max() + 1)
    shaded = np.bincount(labels, weights=np.linalg.norm(colors - hair_color, axis=1) < SHADED_DISTANCE,
                         minlength=labels.max() + 1) / np.maximum(size, 1)
    hair_parts = [p for p in ids if high[p][2] > joints["head"][2] and p not in excluded
                  and (share[p] >= HAIR_SHARE
                       or (shaded[p] >= SHADED_SHARE and size[p] >= SHADED_SIZE * len(verts) and low[p][2] < joints["neck"][2]))]
    hair = np.isin(labels, hair_parts)
    face_area = labels == face
    if face >= 0 and low[face][2] < joints["neck"][2] - head_height:
        # the face is one surface with the whole body (model A's T-pose FBX): only its head end is the
        # face, or the head-top clean-up repaints the up-facing arms and the jaw's chin is a foot.
        # Its hair is usually that same surface too, and swings rather than moving with the face
        fused = fused_hair(mesh_obj, verts, labels, hair_colored & (labels == face), joints)
        legs = face_area & (verts[:, 2] < joints["hips"][2])
        # a body of the hair's own colour (model D, a slime girl, blue all over): colour can't tell
        # the hair from the skin, and her legs were taken as hair and swung on its chains
        if not legs.any() or fused[legs].mean() < SKIN_AS_HAIR:
            hair |= fused
        face_area &=(verts[:, 2] > joints["neck"][2]) & ~hair
    return MeshParts(labels, hair, face, eyes, mouth, hair_color, face_area)


def with_marked_mouth(found, verts, mouth, width, neck_z, behind):
    """`found` with its face area grown to the surface a marked mouth lies on, for the face rig.

    The face part is the one the eye inserts sit in. A tester's pirate has his eyes, hat and brow in
    one surface and his lips, beard and jaw in the body's: the mouth patch was hung in front of a
    hidden layer of the first, 2 cm under his lips, and jawOpen moved none of the face one sees. The
    frontmost surface in the mouth's window (the one `face_rig._add_mouth_cover` reads the lips in)
    is taken, hair and inserts aside; when most of it is another part, that part's head end joins the
    face, as far back as `behind` (the head joint's depth: the jaw turns all that is under the mouth, and
    took the nape of his neck up 1 cm with it). A face that holds its own lips is returned as it is.
    """
    verts = np.asarray(verts, dtype=float)
    labels, mouth = found.labels, np.asarray(mouth, dtype=float)
    window = ((np.abs(verts[:, 0] - mouth[0]) < 0.6 * width) & (np.abs(verts[:, 2] - mouth[2]) < 0.3 * width)
              & ~found.hair & ~np.isin(labels, list(found.eyes.values()) + [found.mouth]))
    if not window.any():
        return found
    lips = window & (verts[:, 1] < verts[window, 1].min() + LIP_DEPTH)
    if found.face_area[lips].mean() >= 0.5:
        return found
    part = np.bincount(labels[lips & ~found.face_area]).argmax()
    return found._replace(face_area=found.face_area | ((labels == part) & (verts[:, 2] > neck_z)
                                                       & (verts[:, 1] < behind) & ~found.hair))


def ears(mesh_obj, verts, joints, skip=None, colours=None):
    """{"left": mask, "right": mask} of a pair of ears or horns standing over the crown, or {}.

    For a character in the pipeline's frame. They need no piece of their own: a tester's pirate has
    his fox ears, his hat and the top of his face in one surface, so nothing on the Parts page could
    be given a role and they stood still. Each is found from its tip, the surface above the crown
    (read in the strip `orient.head_top` reads it in) to one side of it, and taken down to where it
    runs into the middle of the head or the other one, or to the head joint when it is a piece of its
    own. `skip` masks what is known to be something else (hair). One alone, or two unlike in size,
    are not a pair: a feather in a hat, a bun on one side. Ears are not told from horns.
    With `colours` (per vertex) an ear welded to what is beside it is followed down by its colour.
    """
    verts = np.asarray(verts, dtype=float)
    low, top = float(verts[:, 2].min()), float(verts[:, 2].max())
    head_z = float(joints["head"][2])
    height = top - low
    skip = np.zeros(len(verts), dtype=bool) if skip is None else np.asarray(skip, dtype=bool)
    upper = verts[:, 2] > head_z
    front = verts[:, 1] < float(joints["head"][1]) + orient.TOP_REACH * height  # not a tail held up behind
    if not (upper & front).any():
        return {}
    centre = float(np.median(verts[upper & front, 0]))
    strip = np.abs(verts[:, 0] - centre) < orient.TOP_STRIP * height
    if not (strip & upper & front).any():
        return {}
    crown = float(verts[strip & upper & front, 2].max())
    crown_at = int(np.flatnonzero(strip & upper & front)[np.argmax(verts[strip & upper & front, 2])])
    if float(verts[upper & front & ~skip, 2].max(initial=crown)) - crown < EAR_RISE * height:
        return {}
    weld, a, b = meshdata.welded_edges(mesh_obj, verts)
    tips = upper & front & ~strip & ~skip & (verts[:, 2] > crown)
    found = {}
    for side, sign in (("left", 1), ("right", -1)):
        mine = tips & (sign * (verts[:, 0] - centre) > 0)
        if mine.any():
            found[side] = int(np.flatnonzero(mine)[np.argmax(verts[mine, 2])])
    if len(found) < 2:
        return {}
    masks = {}
    for level in np.arange(crown, head_z, -EAR_STEP * height):
        inside = upper & (verts[:, 2] > level)
        member = np.zeros(len(verts), dtype=bool)
        member[weld[inside]] = True
        keep = member[a] & member[b]
        cluster = _components(len(verts), a[keep], b[keep])[weld]
        # the head is what the crown is part of (an ear's own foot can lie in the strip)
        middle = {int(cluster[crown_at])} if inside[crown_at] else set()
        own = {side: int(cluster[tip]) for side, tip in found.items()}
        done = True
        for side, tip in found.items():
            if side in masks and masks[side][1]:
                continue
            if own[side] in middle or own[side] == own["right" if side == "left" else "left"]:
                masks[side] = (masks.get(side, (None, False))[0], True)  # ran into the head: the last free one stands
                continue
            masks[side] = (inside & (cluster == own[side]), False)
            done = False
        if done:
            break
    pair = {side: mask for side, (mask, _) in masks.items() if mask is not None and mask.sum() >= EAR_MIN
            and float(verts[mask, 2].max()) - crown >= EAR_RISE * height}
    if len(pair) < 2:
        return {}
    if colours is not None:
        other = {"left": "right", "right": "left"}
        grown = {side: _same_colour(verts, colours, weld, a, b, mask, upper & ~skip & ~strip) for side, mask in pair.items()}
        if all(np.ptp(verts[g], axis=0).max() <= EAR_MAX * height and not (g & grown[other[side]]).any()
               for side, g in grown.items()):
            pair = grown
    sizes = sorted(float(np.ptp(verts[mask, 2])) for mask in pair.values())
    return pair if sizes[1] <= EAR_ALIKE * sizes[0] else {}


def _same_colour(verts, colours, weld, a, b, seed, allowed):
    """`seed` grown over edges to the vertices of `allowed` whose colour is near the seed's own."""
    tone = np.median(colours[seed], axis=0)
    like = allowed & (np.linalg.norm(colours - tone, axis=1) < EAR_COLOUR)
    member = np.zeros(len(verts), dtype=bool)
    member[weld[like | seed]] = True
    keep = member[a] & member[b]
    cluster = _components(len(verts), a[keep], b[keep])[weld]
    return (like | seed) & np.isin(cluster, np.unique(cluster[seed]))


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
