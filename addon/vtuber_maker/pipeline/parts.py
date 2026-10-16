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
# ... over a strip this wide either side of the head joint (fraction of the head's height)
CROWN_STRIP = 0.15
# Hair fused into another piece counts when a hair-coloured stretch of it holds this share of the
# vertices (model A's T-pose FBX: its smallest real lock 0.38%; the eyebrows and trims beside the face 0.1%).
FUSED_HAIR = 0.003
# A stretch of it below the shoulders counts when it lies this close (m) to one that reaches above.
FUSED_GAP = 0.02
# Fused hair taking this share of the body's surface below the hips isn't hair: the skin is its colour.
SKIN_AS_HAIR = 0.3
# A piece holds a body when it has vertices below the hips within LEG_REACH (of the model's height)
# of the legs' bones, or within TRUNK_REACH of the line from hips to neck below its middle. In one
# that doesn't, the neck is what lies within NECK_REACH of the line from neck to head.
LEG_REACH = 0.05
TRUNK_REACH = 0.075
NECK_REACH = 0.035
# The face, in distances between the eyes: this far to either side of their middle, above and below
# them. Skin runs on there across an edge whose ends' normals agree to SKIN_TURN (their cosine) and
# that leaves the surface by less than SKIN_STEP (the sine of its angle to it).
FACE_WIDE, FACE_ABOVE, FACE_BELOW = 1.1, 0.5, 1.7
SKIN_TURN = 0.85
SKIN_STEP = 0.4
# Hair shaded in two tones: a piece this big (share of the vertices) hanging from above the head
# joint to below the neck is hair with only this share of it within SHADED_DISTANCE of the crown's
# colour. Model D's hair, dark in front with pale highlights: 19% of the model, 22% within
# HAIR_COLOR_DISTANCE and 33% within SHADED_DISTANCE; left out, it had no springs at all and kept
# the rig's arm weights. No other piece of models A, B and D that hangs so is over 2% of its model.
SHADED_DISTANCE = 0.3
SHADED_SHARE = 0.25
SHADED_SIZE = 0.05
# Hair of one colour drawn in many shades of it: a piece is hair too with SHADES_SHARE of it within
# HAIR_COLOR_DISTANCE of the crown's colour darkened to no less than SHADE_FLOOR of it, and
# SHADES_PLAIN of it within that of the crown's colour itself (a brown hat on fair hair has none).
# The fox girl's pale hair runs from 0.5 to 0.95 in brightness and its commonest colour at the
# crown is the brightest: 19 to 46% of each hair piece was near it (one ear alone was hair, 3% of
# her head) and she had no hair chain at all; with the shades 76 to 93%, her eyebrows' piece 0.
SHADE_FLOOR = 0.7
# Hair read by its own colour: once some is found, a piece of at least SAME_HAIR_SIZE of the vertices
# with HAIR_SHARE of it near the found hair's median colour is hair too. The chibi's crown is mostly
# painted highlights (0.99, 0.72, 0.41), her hair (0.86, 0.46, 0.22): her back hair passed as shades
# of the highlight (0.78), her bangs and front locks didn't (0.67) and never swung.
SAME_HAIR_SIZE = 0.01
SHADES_SHARE = 0.7
SHADES_PLAIN = 0.15
# A piece of the hair's colour that goes all the way round the body below the middle of the trunk is
# clothing, not hair: in a band ROUND_BAND (m) high it has ROUND_MIN vertices in at least ROUND_BODY
# of ROUND_BINS directions from the line hips to neck. A tester's game model (white hair, white
# dress, the dress one piece with the collar above the head joint): 19 088 vertices of dress, arms
# and skirt were hair, 12 of 12 at the waist, and eleven chains hung her skirt from the head (33 cm
# torn on a head turn, the skirt up at her chest on a nod). Hair to the hips goes 6 of 12 at most
# there (OC3, OC3_Tpose); up at the shoulders it goes 10 (OC3), so only below the middle counts.
ROUND_BINS = 12
ROUND_BODY = 10
ROUND_BAND = 0.02
ROUND_MIN = 12
# ... looked for from this far (m) under the hips up.
ROUND_BELOW = 0.2


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
# With the eyes marked, the eye inserts are the mirrored pair nearest the marks, within this many
# head heights of them: by size alone a tester's model took its inner-ear pieces (4.9 x 6.6 cm,
# 11 cm above the eyes' 2.6 x 3.1 cm) and blinked with its ears.
EYES_MARKED = 0.3


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


def split(verts, edges, parts=None):
    """Label each vertex with its connected part, joining vertices that share a position.

    `parts` (experiment, OC2VRM_PARTS): a part number per vertex from the file (adopt.parts_on); vertices
    of different numbers aren't joined, so a border two parts share stays a border.
    """
    keys = np.round(np.asarray(verts) / WELD_TOLERANCE).astype(np.int64)
    if parts is not None:
        parts = np.asarray(parts, dtype=np.int64)
        keys = np.column_stack([keys, parts])
        edges = np.asarray(edges)
        edges = edges[parts[edges[:, 0]] == parts[edges[:, 1]]]  # nor does an edge from one part into the next
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


def _face_piece(verts, labels, inserts, small, eyes):
    """(The piece nearest the first of `eyes` (eye inserts' centres), the vertex of it nearest each eye): no insert, nor another piece an insert's size
    (`small`) when a bigger one is there: the miko's eye-white shells, 3 mm behind the head's median depth,
    weren't inserts, lay nearest her eyes and were taken for the face (49 vertices); her face (2869) stayed a
    quarter on the neck, and her eyes left it on a head tilt."""
    others = ~np.isin(labels, inserts)
    if (others & ~np.isin(labels, small)).any():
        others &= ~np.isin(labels, small)
    candidates = np.nonzero(others)[0]
    nearest = [int(candidates[np.argmin(np.linalg.norm(verts[candidates] - eye, axis=1))]) for eye in eyes]
    return int(labels[nearest[0]]), nearest


def find(mesh_obj, joints, eyes_at=None, file_parts=None):
    """Split the mesh into parts and pick out the hair, the eye and mouth inserts, and the face.

    `eyes_at` is where the eyes were marked, two world (x, z), when they were (face_points.eye_places).
    `file_parts`: see split.
    """
    verts = world_vertices(mesh_obj)
    edges = np.empty(len(mesh_obj.data.edges) * 2, dtype=np.int64)
    mesh_obj.data.edges.foreach_get("vertices", edges)
    labels = split(verts, edges.reshape(-1, 2), file_parts)
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
    eyes = _eye_pair(inserts, low, high, centre, float(joints["head"][0]), float(joints["head"][2]),
                     marked=eyes_at, reach=EYES_MARKED * head_height)
    mouth = _mouth(inserts, eyes, low, high, centre) if eyes else -1

    face = -1
    if eyes:
        small = [p for p in ids if (high[p] - low[p]).max() < INSERT_SIZE * head_height]
        face, nearest = _face_piece(verts, labels, inserts, small, [centre[p] for p in eyes.values()])

    colors = texture_space.vertex_colors(mesh_obj, texture_space.base_color_image(mesh_obj))
    # ... read over the head's middle: the demon girl's horns stood highest, beside her crown, their dark red was taken
    # for the hair's, the horns were hung on chains and her purple hair had none
    middle = np.abs(verts[:, 0] - joints["head"][0]) < CROWN_STRIP * head_height
    crown = float(verts[middle & (verts[:, 2] <= top), 2].max()) if (middle & (verts[:, 2] <= top)).any() else top
    hair_color = _dominant_color(colors[middle & (verts[:, 2] > crown - CROWN * head_height) & (verts[:, 2] <= crown)])
    hair_colored = np.linalg.norm(colors - hair_color, axis=1) < HAIR_COLOR_DISTANCE
    share = np.bincount(labels, weights=hair_colored, minlength=labels.max() + 1) / np.maximum(
        np.bincount(labels, minlength=labels.max() + 1), 1
    )
    excluded = set(inserts) | {face}
    size = np.bincount(labels, minlength=labels.max() + 1)
    shaded = np.bincount(labels, weights=np.linalg.norm(colors - hair_color, axis=1) < SHADED_DISTANCE,
                         minlength=labels.max() + 1) / np.maximum(size, 1)
    shade = np.clip(colors @ hair_color / max(float(hair_color @ hair_color), 1e-12), SHADE_FLOOR, 1.0)
    shades = np.bincount(labels, weights=np.linalg.norm(colors - shade[:, None] * hair_color, axis=1) < HAIR_COLOR_DISTANCE,
                         minlength=labels.max() + 1) / np.maximum(size, 1)
    hair_parts = [p for p in ids if high[p][2] > joints["head"][2] and p not in excluded
                  and (share[p] >= HAIR_SHARE
                       or (shades[p] >= SHADES_SHARE and share[p] >= SHADES_PLAIN)
                       or (shaded[p] >= SHADED_SHARE and size[p] >= SHADED_SIZE * len(verts) and low[p][2] < joints["neck"][2]))]
    worn = [p for p in hair_parts if low[p][2] < (joints["hips"][2] + joints["neck"][2]) / 2
            and goes_round_body(verts[labels == p], joints)]
    if worn:
        print(f"hair: {int(np.isin(labels, worn).sum())} vertices of the hair's colour go round the body: clothing, left off the hair")
        hair_parts = [p for p in hair_parts if p not in worn]
    if hair_parts:
        own = np.median(colors[np.isin(labels, hair_parts)], axis=0)
        near = np.bincount(labels, weights=np.linalg.norm(colors - own, axis=1) < HAIR_COLOR_DISTANCE,
                           minlength=labels.max() + 1) / np.maximum(size, 1)
        more = [p for p in ids if p not in hair_parts and p not in excluded and p not in worn and high[p][2] > joints["head"][2]
                and size[p] >= SAME_HAIR_SIZE * len(verts) and near[p] >= HAIR_SHARE
                and not (low[p][2] < (joints["hips"][2] + joints["neck"][2]) / 2 and goes_round_body(verts[labels == p], joints))]
        hair_parts += more
    hair = np.isin(labels, hair_parts)
    face_area = labels == face
    if face >= 0 and low[face][2] < joints["neck"][2] - head_height:
        # the face is one surface with the whole body (model A's T-pose FBX): only its head end is the
        # face, or the head-top clean-up repaints the up-facing arms and the jaw's chin is a foot.
        # Its hair is usually that same surface too, and swings rather than moving with the face
        fused = fused_hair(mesh_obj, verts, labels, hair_colored & (labels == face), joints)
        # a piece that hangs low and holds no body is a head and its hair (the catgirl's: face, neck,
        # bangs and one twin-tail, the tail down past her hips)
        bodiless = not _holds_body(verts, labels == face, joints)
        if bodiless and fused.any():
            from . import roles
            piece = labels == face
            eye_at = np.array([centre[p] for p in eyes.values()])
            apart = float(np.ptp(eye_at[:, 0]))
            between = (piece & (np.abs(verts[:, 0] - eye_at[:, 0].mean()) < FACE_WIDE * apart)
                       & (verts[:, 2] < eye_at[:, 2].mean() + FACE_ABOVE * apart) & (verts[:, 2] > eye_at[:, 2].mean() - FACE_BELOW * apart))
            # in such a piece a colour is the hair's or the skin's, whichever it is nearer: a lock in
            # shadow is further from the crown's colour than HAIR_COLOR_DISTANCE (the one by her left
            # cheek: 0.24), counted as skin with the cheek it touches, and stood still while the
            # locks round it swung
            if (between & ~hair_colored).any():
                skin_colour = np.median(colors[between & ~hair_colored], axis=0)
                hairish = piece & (np.linalg.norm(colors - hair_color, axis=1) < np.linalg.norm(colors - skin_colour, axis=1))
                fused = fused_hair(mesh_obj, verts, labels, hairish, joints)
            else:
                hairish = piece & hair_colored
            # the skin the eyes sit in is small beside the hair round it: it was closed as a hole in
            # the hair, and the mouth's shapes moved "hair". Its own holes are closed first (lips, a
            # blush and shading of the hair's colour: left to the hair, her chin and lids hung on its
            # chains and tore on a head turn)
            skin = roles.close_region(mesh_obj, piece, _skin_at(mesh_obj, verts, piece & ~hairish, nearest))
            # ... and the neck is skin, of whatever colour (hers is near her hair's pink: on a chain
            # it sagged out over her collar)
            a, b = np.asarray(joints["neck"], dtype=float), np.asarray(joints["head"], dtype=float)
            along = np.clip((verts - a) @ (b - a) / max(float((b - a) @ (b - a)), 1e-12), -1.0, 1.0)
            skin |= piece & (np.linalg.norm(verts - (a + along[:, None] * (b - a)), axis=1) < NECK_REACH * float(np.ptp(verts[:, 2])))
            # ... and so is what runs on smoothly from the skin between brow and chin, though it is the
            # hair's colour: a blush from her nose to her chin joins a lock that touches it, the two
            # were one stretch of "hair", and half her mouth and a cheek swung on a chain
            skin = _smooth_onward(mesh_obj, verts, skin, between)
            # all else of it under the neck is hair, whatever its colour (her tail's inner side is
            # painted dark: left on the head, it tore from the rest 10.7 cm on a head turn)
            fused = (fused | piece & (verts[:, 2] < joints["neck"][2])) & ~skin
            # ... and so are the gaps left in the hair once the skin is known: a stretch of a lock nearer
            # the skin's colour than the hair's (the end of the one by her left cheek, 50 vertices in
            # shadow between the lock and its tip) was neither, stayed on the head and cut the tip off
            # the lock: both stood still. Closed over what isn't skin, so the face isn't taken again
            fused = roles.close_region(mesh_obj, piece & ~skin, fused)
        legs = face_area & (verts[:, 2] < joints["hips"][2])
        # a body of the hair's own colour (model D, a slime girl, blue all over): colour can't tell
        # the hair from the skin, and her legs were taken as hair and swung on its chains. What a
        # bodiless piece has below the hips hangs there: counted as body, the catgirl's tail had her
        # fused hair refused whole (that tail, her bangs and the locks by her face had no springs)
        if bodiless or not legs.any() or fused[legs].mean() < SKIN_AS_HAIR:
            hair |= fused
        face_area &=(verts[:, 2] > joints["neck"][2]) & ~hair
    return MeshParts(labels, hair, face, eyes, mouth, hair_color, face_area)


def goes_round_body(points, joints):
    """Whether `points` (one piece's vertices) ring the body somewhere below the middle of the trunk:
    see ROUND_BODY. Assumes the normalized frame, standing; hair down the back or over both
    shoulders doesn't, a dress, a belt or a skirt does."""
    a, b = np.asarray(joints["hips"], dtype=float), np.asarray(joints["neck"], dtype=float)
    points = points[(points[:, 2] >= a[2] - ROUND_BELOW) & (points[:, 2] < (a[2] + b[2]) / 2)]
    if len(points) < ROUND_MIN:
        return False
    along = np.clip((points[:, 2] - a[2]) / max(float(b[2] - a[2]), 1e-12), 0.0, 1.0)
    axis = a + along[:, None] * (b - a)
    direction = np.floor((np.arctan2(points[:, 0] - axis[:, 0], points[:, 1] - axis[:, 1]) + np.pi)
                         / (2 * np.pi) * ROUND_BINS).astype(np.int64) % ROUND_BINS
    band = np.floor((points[:, 2] - (a[2] - ROUND_BELOW)) / ROUND_BAND).astype(np.int64)
    for k in np.flatnonzero(np.bincount(band) >= ROUND_MIN):
        if len(np.unique(direction[band == k])) >= ROUND_BODY:
            return True
    return False


def _smooth_onward(mesh_obj, verts, skin, allowed):
    """`skin` grown over edges into `allowed` wherever the surface runs on without a crease.

    A vertex joins when its normal is within SKIN_TURN of its skin neighbour's and the edge
    between them lies in that neighbour's surface (SKIN_STEP): a lock standing on the face meets it
    at a wall, a patch of the face painted another colour doesn't.
    """
    weld, a, b = meshdata.welded_edges(mesh_obj, verts)
    normals = np.empty(len(verts) * 3)
    mesh_obj.data.vertices.foreach_get("normal", normals)
    normals = normals.reshape(-1, 3) @ np.array(mesh_obj.matrix_world)[:3, :3].T
    # split along UV seams, a welded vertex has several normals: their mean
    total = np.zeros_like(normals)
    np.add.at(total, weld, normals)
    normals = total[weld] / np.maximum(np.linalg.norm(total[weld], axis=1, keepdims=True), 1e-12)
    a, b = np.concatenate([a, b]), np.concatenate([b, a])
    step = verts[b] - verts[a]
    length = np.maximum(np.linalg.norm(step, axis=1), 1e-12)
    onward = ((normals[a] * normals[b]).sum(axis=1) > SKIN_TURN) & (np.abs((normals[a] * step).sum(axis=1)) / length < SKIN_STEP)
    region = np.zeros(len(verts), dtype=bool)
    region[weld[skin]] = True
    open_ = np.zeros(len(verts), dtype=bool)
    open_[weld[allowed]] = True
    while True:
        reached = np.zeros(len(verts), dtype=bool)
        reached[b[region[a] & onward & open_[b] & ~region[b]]] = True
        if not reached.any():
            return skin | (region[weld] & allowed)
        region |= reached


def _skin_at(mesh_obj, verts, skin, at):
    """The connected stretches of the mask `skin` that hold the vertices `at`, or lie nearest them."""
    weld, a, b = meshdata.welded_edges(mesh_obj, verts)
    if not skin.any():
        return skin
    member = np.zeros(len(verts), dtype=bool)
    member[weld[skin]] = True
    keep = member[a] & member[b]
    cluster = _components(len(verts), a[keep], b[keep])[weld]
    ids = np.flatnonzero(skin)
    seeds = [int(ids[np.argmin(np.linalg.norm(verts[ids] - verts[v], axis=1))]) for v in at]
    return skin & np.isin(cluster, cluster[seeds])


def _holds_body(verts, piece, joints):
    """Whether the mask `piece` holds legs or a trunk: vertices on the legs below the hips (_on_legs),
    or within TRUNK_REACH of the line from hips to neck below its middle. True when the joints name no legs."""
    below = piece & (verts[:, 2] < joints["hips"][2]) & _on_legs(verts, joints)
    if below.any():
        return True
    a, b = np.asarray(joints["hips"], dtype=float), np.asarray(joints["neck"], dtype=float)
    along = np.clip((verts - a) @ (b - a) / max(float((b - a) @ (b - a)), 1e-12), 0.0, 1.0)
    near = np.linalg.norm(verts - (a + along[:, None] * (b - a)), axis=1) < TRUNK_REACH * float(np.ptp(verts[:, 2]))
    return bool((piece & near & (verts[:, 2] < (a[2] + b[2]) / 2)).any())


def _on_legs(verts, joints):
    """True for vertices within LEG_REACH of a leg's bones; all of them when the joints name no legs."""
    names = [(f"{side}_upper_leg", f"{side}_lower_leg", f"{side}_foot") for side in ("left", "right")]
    if not all(name in joints for trio in names for name in trio):
        return np.ones(len(verts), dtype=bool)
    reach = LEG_REACH * float(np.ptp(verts[:, 2]))
    near = np.zeros(len(verts), dtype=bool)
    for hip, knee, foot in names:
        for a, b in ((joints[hip], joints[knee]), (joints[knee], joints[foot])):
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            along = np.clip((verts - a) @ (b - a) / max(float((b - a) @ (b - a)), 1e-12), 0.0, 1.0)
            near |= np.linalg.norm(verts - (a + along[:, None] * (b - a)), axis=1) < reach
    return near


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


def _eye_pair(inserts, low, high, centre, head_x, head_z, marked=None, reach=0.0):
    """The largest pair of inserts mirrored across the head's centre line, above the head joint.

    With the eyes `marked` (two world (x, z)), the pair nearest the marks instead, if one lies
    within `reach` of them (both its pieces together); else the largest, as without marks.
    """
    candidates = [p for p in inserts if centre[p][2] > head_z]
    best, best_area = {}, 0.0
    nearest, nearest_gap = {}, 2.0 * reach
    if marked is not None:
        marked = sorted((np.asarray(m, float) for m in marked), key=lambda m: -m[0])  # the left eye is at +X
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            spread = abs(centre[a][0] - centre[b][0])
            size_a, size_b = high[a] - low[a], high[b] - low[b]
            area_a, area_b = size_a[0] * size_a[2], size_b[0] * size_b[2]
            mirrored = abs((centre[a][0] + centre[b][0]) / 2 - head_x) < 0.25 * spread
            level = abs(centre[a][2] - centre[b][2]) < 0.25 * spread
            apart = spread > max(size_a[0], size_b[0])
            alike = max(area_a, area_b) < 3 * min(area_a, area_b)
            if not (mirrored and level and apart and alike):
                continue
            left, right = (a, b) if centre[a][0] > centre[b][0] else (b, a)
            if area_a + area_b > best_area:
                best, best_area = {"left": int(left), "right": int(right)}, area_a + area_b
            if marked is not None:
                gap = sum(float(np.linalg.norm(centre[p][[0, 2]] - m)) for p, m in zip((left, right), marked))
                if gap < nearest_gap:
                    nearest, nearest_gap = {"left": int(left), "right": int(right)}, gap
    return nearest or best


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
