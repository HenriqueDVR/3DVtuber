"""Eyes that follow face tracking, for faces painted on a face patch (see face_points).

Two ways, chosen per character, both starting from the eye points marked in viewer/setup.html, and
both driven by the eight ARKit eyeLook shape keys (bound to the VRM look expressions):

- Moving irises (default), for eyes with room around the iris (model A). The painted iris moves onto
  its own disc just in front of the face, which slides along the face's shape. The face under it
  has the iris painted out, and a copy of the face around each eye, a little further in front and
  see-through inside the painted opening, keeps the iris under the lids. At rest the three layers
  show exactly the painting.
- Sliding eyes, for a face with no room for a separate iris (a mascot's ball). The whole painted eye
  moves onto a patch that slides a few millimetres over the face, as the free mouth does
  (face_rig), and blinks by squeezing.

Both run on a finished, skinned mesh with its shape keys: every key gets the new vertices.
Images are handled as (rows, columns, channels) arrays with the top row first, like the pixels in
face.json.
"""

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import compat, face_points, face_rig, meshdata
from .rig import BONE_NAMES

SIDE_IDS = face_rig.EYE_INSERT_IDS
IRIS_ATTRIBUTE = "face_rig_iris"  # 1 left, 2 right
WHITE_ATTRIBUTE = "face_rig_eye_white"  # 1 left, 2 right
OVERLAY_ATTRIBUTE = "face_rig_lid_overlay"  # source vertex + 1 on the lid overlay, 0 elsewhere
PATCH_ATTRIBUTE = "face_rig_eye_patch"  # 1 left, 2 right
LID_ATTRIBUTE = "face_rig_eye_lid"  # 1 left, 2 right: the lids the sliding patches blink with
LID_MATERIAL = "EyeLidClosed"
LIDS = False
# The lid: over the eye's box and LID_MARGIN of its width either side, plus as far as the patch
# slides; from LID_ABOVE of the eye's height over its top down to the closing line, which dips
# LID_CURVE of the eye's height in the middle (an anime closed eye); LID_FRONT (m) in front of the
# patch. Its lower LID_LASH is lash colour.
LID_MARGIN = 0.12
LID_ABOVE = 0.12
LID_CURVE = 0.12
LID_FRONT = 0.0006
LID_LASH = 0.1
# The lid's top and sides fade into the face over this share of its height and width.
LID_FADE_TOP = 0.35
LID_FADE_SIDE = 0.15
LID_GRID = (24, 12)
LIDS_MATERIAL, IRIS_MATERIAL, PATCH_MATERIAL = "EyeLids", "EyeIris", "EyePaint"
# Layer depths (m). The iris slides on a smooth track with the eye's white WHITE_GAP behind it;
# the lids are OVERLAY_GAP in front of the face. Within an opening everything else the track
# doesn't clear by IRIS_GAP (TRACK_CLEARS of it, the rest see-through) is hidden behind the white.
IRIS_GAP = 0.0008
WHITE_GAP = 0.0005
TRACK_CLEARS = 95
OVERLAY_GAP = 0.0016
# Where the iris can go, the lids are lifted to at least LID_LIFT in front of its track (m): the
# socket behind them can be far deeper than the track (see track_surface).
LID_LIFT = 0.0016
# The iris's track is fitted to every TRACK_STEP'th pixel of its window; face points further than
# TRACK_OUTLIER times the typical distance from the fit (hair strands) are left out.
TRACK_STEP = 4
TRACK_OUTLIER = 3.0
# Something more than this (m) in front of the iris is in front of the eye (hair), not the face;
# the track clears whatever is less than HAIR_FRONT in front of its fit.
IN_FRONT = 0.003
HAIR_FRONT = 0.006
# What shows is found by looking at every WINDOW_STEP'th pixel of the openings, and every
# SAMPLE_STEP (m) round the eyes.
WINDOW_STEP = 1
SAMPLE_STEP = 0.001
# How far the iris slides at full eyeLook weight: sideways, this share of the white beside it (on
# average over both sides of both eyes: an iris near one corner still slides the same both ways,
# and the lids hide what passes under the corner); up and down, this share of its own height.
SIDE_TRAVEL = 0.55
UP_DOWN_TRAVEL = 0.1
IRIS_GRID = 16
WHITE_GRID = 24
# The eye's white reaches this far (m) past its opening, under the lids.
WHITE_MARGIN = 0.002
# The lid overlay covers the iris's travel and this much more (m) at the full gap. Checks want the
# iris MIN_CLEARANCE (m) from the face in front of it and from the lids behind it, and a sliding
# patch PATCH_CLEARANCE in front of the face.
OVERLAY_MARGIN = 0.003
MIN_CLEARANCE = 0.0001
# The share of an iris's points in one look that may show wrongly before the check fails. A
# blink and a look at once add two moves model B's round face can't follow exactly: up to
# BLINK_STRAY of a sliding patch may dip into it for the moment the eye is shut (4.3% measured).
STRAY = 0.01
BLINK_STRAY = 0.05
PATCH_CLEARANCE = 0.0004
# Sliding eyes: a patch this many columns and rows, laid this far in front of the face (m), sliding
# this share of the eye's width sideways and of its height up and down at full weight.
PATCH_GRID = (40, 30)
# A patch's track (smooth_track) is read every TRACK_CELL (m), TRACK_MARGIN past where it can
# slide; things more than CLEAR_CAP in front of the face (a hat's brim) don't lift it.
TRACK_CELL = 0.001
TRACK_MARGIN = 0.004
CLEAR_CAP = 0.025
# (more than the mouth's 0.8 mm: a blink and a look at once add up, and the ball curves away fast
# at the eyes' outer corners, so the two moves together sank the patch into it by up to ~1 mm)
PATCH_GAP = 0.0015
SLIDE = (0.08, 0.05)
LOOK_KEYS = ("eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
             "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight")

# Finding the painted opening, in eye heights: the marks are clicks, often a few pixels off the
# painted lids (model A's lower curve dips ~20 px into the cheek), so the opening is grown from the
# middle of the marked region over pixels coloured like that middle, within OPENING_REACH of it.
OPENING_REACH = 0.12
OPENING_CORE = 0.18
OPENING_RING = (0.06, 0.3)
# Colours seen outside the marks this close (RGB distance) to the middle's are the eye's own.
SAME_COLOUR = 0.08
# Lashes: pixels darker than this share of the opening's typical brightness. The opening runs up
# to the lashes' lower edge, which is found within LASH_WINDOW of the top curve.
LASH_DARK = 0.4
LASH_WINDOW = (0.25, 0.2)
LASH_GAP = 0.08
# Rows at the top of the opening that stay on the lids (pixels).
LASH_EDGE = 3
# The iris's hidden parts take the colour this many pixels inside its visible top or bottom,
# smoothed over IRIS_SMOOTH passes; the painted-out white is smoothed over WHITE_SMOOTH.
IRIS_CARRY = 3
# Colours carried on are within CARRY_HUE (turns) of the iris's typical hue and this saturated.
CARRY_HUE = 0.125
CARRY_SATURATION = 0.2
IRIS_SMOOTH = 12
WHITE_SMOOTH = 6
# Eye white is at least this share of the opening's typical brightness. The whole white, under the
# iris too, becomes one neutral white at this percentile of the eye's own: model A's white is shaded
# and tinted (lash shadow, a pink lower lid, blue-grey with a pale blue side), and carrying that
# across rows left grey, dark and blue bands. Pixels under WHITE_LINE of it are lines and stay.
WHITE_BRIGHT = 0.85
WHITE_TONE = 90
WHITE_LINE = 0.6
# Depth on that white, as anime eyes are painted: the upper lid's shadow, SHADOW_LID deep at the
# lid and fading over SHADOW_REACH of the opening's height, and SHADOW_CORNER more at the corners.
# Full shadow is the white times SHADOW_TINT, a faint cool grey (a plain grey looked dirty; more
# blue read as a blue eye). The lid's line is evened over SHADOW_EVEN of the width.
SHADOW_LID = 0.9
SHADOW_REACH = 0.3
SHADOW_CORNER = 0.35
SHADOW_TINT = (0.84, 0.85, 0.9)
SHADOW_EVEN = 0.08
# Pixels other than white (the iris's outline) this close to the iris disc go with it: model A's
# ellipse left part of the outline on the white as a dark arc when the iris moved.
OUTLINE_REACH = 6
# Gaps at the top narrower than this (eye heights) are closed: a highlight at the lash line.
NOTCH = 0.07
# The iris: pixels this saturated or this dark inside the opening, in parts over IRIS_THIN
# pixels thick (the lashes' soft lower edge is a thin dark band along the top).
IRIS_SATURATION = 0.35
IRIS_DARK = 0.45
IRIS_THIN = 3
# Sliding eyes: the eye's paint differs from the skin behind it by more than PAINT_STRONG (RGB
# distance) somewhere and PAINT_WEAK everywhere (faint outlines join the lashes they touch).
PAINT_STRONG = 0.2
PAINT_WEAK = 0.06
PAINT_REACH = 0.3
PAINT_RING = 0.25
PAINT_GROW = 2
# The eyes' images are cropped to the polygons using them, CROP_PAD pixels more all round.
CROP_PAD = 4
# The cut edge of an eye's image lies this many pixels out in the skin around it.
FRINGE = 3


# --- images -----------------------------------------------------------------------------------

def image_rows(image):
    """An image's pixels as (rows, columns, 4), top row first."""
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(height, width, 4)[::-1].copy()


def new_image(name, rows):
    """A packed RGBA image holding `rows` (top row first)."""
    height, width = rows.shape[:2]
    image = bpy.data.images.new(name, width, height, alpha=True)
    image.pixels.foreach_set(np.ascontiguousarray(rows[::-1], dtype=np.float32).ravel())
    image.update()
    image.pack()
    return image


def _dilate(mask, steps=1):
    out = mask.copy()
    for _ in range(steps):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _erode(mask, steps=1):
    return ~_dilate(~mask, steps)


def _reach(seeds, allowed):
    """The pixels of `allowed` joined to `seeds` through `allowed` (four neighbours)."""
    current = seeds & allowed
    while True:
        grown = _dilate(current, 4) & allowed
        grown = _dilate(grown) & allowed
        if np.array_equal(grown, current):
            return current
        current = grown


def _fill_holes(mask):
    """The mask with every region it encloses filled."""
    border = np.zeros_like(mask)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    return mask | ~_reach(border, ~mask)


def _kmeans(colours, count, iterations=12):
    """`count` typical colours of an (n, 3) array."""
    colours = np.asarray(colours, dtype=np.float64)
    if len(colours) > 20000:
        colours = colours[np.random.default_rng(0).choice(len(colours), 20000, replace=False)]
    order = np.argsort(colours.sum(axis=1))
    centres = colours[order[np.linspace(0, len(order) - 1, count).astype(int)]].copy()
    for _ in range(iterations):
        nearest = np.argmin(((colours[:, None] - centres[None]) ** 2).sum(axis=2), axis=1)
        for j in range(count):
            if (nearest == j).any():
                centres[j] = colours[nearest == j].mean(axis=0)
    return centres


def _saturation_value(rgb):
    top, bottom = rgb.max(axis=-1), rgb.min(axis=-1)
    return (top - bottom) / np.maximum(top, 1e-6), top


# --- the marked eyes --------------------------------------------------------------------------

def _parabola(a, m, b):
    """y(x) through three points, as viewer/setup.html draws the lid curves."""
    xs, ys = np.array([a[0], m[0], b[0]], float), np.array([a[1], m[1], b[1]], float)
    coeffs = np.linalg.solve(np.stack([xs * xs, xs, np.ones(3)], axis=1), ys)
    return lambda x: coeffs[0] * x * x + coeffs[1] * x + coeffs[2]


class Eye:
    """One marked eye in a face patch image: its crop box and the region between its lid curves.

    `points` are face.json's inner/outer/top/bottom, scaled to the image's pixels (y down).
    """

    def __init__(self, side, points, image_size, grow=(0.45, 1.0, 0.8)):
        self.side = side
        self.points = {k: np.asarray(v, float) for k, v in points.items()}
        p = self.points
        self.width = abs(p["outer"][0] - p["inner"][0])
        self.height = abs(p["bottom"][1] - p["top"][1])
        left, right = sorted((p["inner"][0], p["outer"][0]))
        top = min(p["top"][1], p["inner"][1], p["outer"][1])
        bottom = max(p["bottom"][1], p["inner"][1], p["outer"][1])
        w, h = image_size
        self.box = (max(int(left - grow[0] * self.width), 0), max(int(top - grow[1] * self.height), 0),
                    min(int(right + grow[0] * self.width) + 1, w), min(int(bottom + grow[2] * self.height) + 1, h))
        x0, y0, x1, y1 = self.box
        ys, xs = np.mgrid[y0:y1, x0:x1] + 0.5
        self.upper = _parabola(p["inner"], p["top"], p["outer"])
        self.lower = _parabola(p["inner"], p["bottom"], p["outer"])
        self.marked = (xs >= left) & (xs <= right) & (ys >= self.upper(xs)) & (ys <= self.lower(xs))

    def crop(self, rows):
        x0, y0, x1, y1 = self.box
        return rows[y0:y1, x0:x1]

    def pixels(self, mask):
        """(x, y) image pixel centres of a crop mask's pixels."""
        ys, xs = np.nonzero(mask)
        return np.stack([xs + self.box[0] + 0.5, ys + self.box[1] + 0.5], axis=1)


def marked_eyes(points, image_size):
    """side -> Eye for the face.json points on an image of `image_size` (width, height).

    Sides follow where the eyes are, not what face.json calls them (model B's file has them the
    other way round): the face patch is a front view, so the character's left eye is on the right.
    """
    scale = np.asarray(image_size, float) / np.asarray(points["size"], float)
    eyes = [Eye(key, {k: np.asarray(v) * scale for k, v in points[f"{key}_eye"].items()}, image_size)
            for key in ("left", "right")]
    eyes.sort(key=lambda eye: -(eye.points["inner"][0] + eye.points["outer"][0]))
    eyes[0].side, eyes[1].side = "left", "right"
    return {eye.side: eye for eye in eyes}


def find_opening(rgb, eye):
    """The painted opening (eye white and iris) of one eye's crop, as a crop mask.

    Grown from the middle of the marked region over pixels nearer its colours than those around
    the marks (lashes, skin, lid lines), filled, then run up to the lashes' lower edge column by
    column, so a highlight touching the lash line isn't left out.
    """
    height = eye.height
    core = _erode(eye.marked, max(1, int(OPENING_CORE * height)))
    if core.sum() < 20:
        core = eye.marked
    reach = _dilate(eye.marked, max(2, int(OPENING_REACH * height)))
    inner = _kmeans(rgb[core], 5)
    ring = _dilate(eye.marked, int(OPENING_RING[1] * height)) & ~_dilate(eye.marked, int(OPENING_RING[0] * height))
    outer = _kmeans(rgb[ring], 4)
    outer = outer[np.linalg.norm(outer[:, None] - inner[None], axis=2).min(axis=1) > SAME_COLOUR]
    centres = np.concatenate([inner, outer])
    nearest = np.argmin(((rgb[..., None, :] - centres) ** 2).sum(axis=-1), axis=-1)
    eyeish = reach & (nearest < len(inner))
    opening = _fill_holes(_reach(core & eyeish, eyeish))

    # up to the lashes' lower edge
    _, value = _saturation_value(rgb)
    dark = value < LASH_DARK * np.median(value[opening])
    x0, y0 = eye.box[:2]
    for column in range(rgb.shape[1]):
        rows = np.nonzero(opening[:, column])[0]
        if not len(rows):
            continue
        curve = eye.upper(x0 + column + 0.5) - y0
        lo = max(int(curve - LASH_WINDOW[0] * height), 0)
        hi = min(int(curve + LASH_WINDOW[1] * height), rows[0])
        lash = np.nonzero(dark[lo:hi, column])[0]
        if len(lash) and rows[0] - (lo + lash[-1] + 1) <= LASH_GAP * height:
            opening[lo + lash[-1] + 1:rows[0], column] = True
    closing = max(1, int(NOTCH * height))
    opening = _fill_holes(_erode(_dilate(opening, closing), closing) & reach) | opening
    return opening


def find_iris(rgb, opening):
    """The iris inside an opening, and the ellipse (cx, cy, rx, ry) round it, crop pixels (y down).

    The iris is the saturated or dark part joined to the opening's middle, holes (catchlights)
    filled; thin dark bands (the lashes' soft lower edge) are left out. The ellipse is fitted to
    its edge against the eye white that reaches the corners, below its top fifth (the lids hide
    the top, and catchlights there aren't its edge), dropping the worst-fitting points once.
    """
    saturation, value = _saturation_value(rgb)
    irisish = opening & ((saturation > IRIS_SATURATION) | (value < IRIS_DARK))
    irisish = _dilate(_erode(irisish, IRIS_THIN), IRIS_THIN) & irisish
    ys, xs = np.nonzero(opening)
    middle = np.zeros_like(opening)
    middle[int(np.median(ys)), int(np.median(xs))] = True
    seeds = _dilate(middle, 6) & irisish
    iris = _reach(seeds if seeds.any() else irisish, irisish)
    iris = _fill_holes(iris) & opening | iris
    iy, ix = np.nonzero(iris)
    width = ix.max() - ix.min() + 1
    fallback = (ix.mean() + 0.5, iy.max() + 0.5 - 0.55 * width, width / 2, 0.55 * width)
    white = opening & ~iris
    columns = np.nonzero(opening.any(axis=0))[0]
    sides = np.zeros_like(opening)
    sides[:, :columns[0] + max(2, int(0.1 * len(columns)))] = True
    sides[:, columns[-1] - max(2, int(0.1 * len(columns))):] = True
    sclera = _reach(white & sides, white)
    edge = iris & _dilate(sclera) & ~_dilate(~opening, 2)
    edge[:int(iy.min() + 0.2 * (iy.max() - iy.min()))] = False
    ey, ex = np.nonzero(edge)
    points = np.stack([ex + 0.5, ey + 0.5], axis=1)
    for _ in range(2):
        if len(points) < 12:
            return iris, fallback
        fit = _fit_ellipse(points)
        if fit is None:
            return iris, fallback
        cx, cy, rx, ry = fit
        residual = np.abs(np.hypot((points[:, 0] - cx) / rx, (points[:, 1] - cy) / ry) - 1)
        points = points[residual <= max(2 * np.median(residual), 0.02)]
    if not (0.6 * width < 2 * rx < 1.5 * width and 0.3 * width < ry < 1.5 * width):
        return iris, fallback
    return iris, fit


def _fit_ellipse(points):
    """Axis-aligned ellipse (cx, cy, rx, ry) through (n, 2) points, least squares, or None."""
    mx, my = points.mean(axis=0)
    x, y = points[:, 0] - mx, points[:, 1] - my
    a, b, c, d = np.linalg.lstsq(np.stack([x * x, y * y, x, y], axis=1), np.ones(len(x)), rcond=None)[0]
    if a <= 0 or b <= 0:
        return None
    cx, cy = -c / (2 * a), -d / (2 * b)
    k = 1 + a * cx * cx + b * cy * cy
    return cx + mx, cy + my, float(np.sqrt(k / a)), float(np.sqrt(k / b))


def ellipse_mask(shape, ellipse):
    cx, cy, rx, ry = ellipse
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]] + 0.5
    return ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0


def window(opening):
    """Where the lids are see-through: the opening less its edge pixel and the LASH_EDGE rows at its top.

    The top rows are the lashes' soft lower edge; they stay painted on the lids, so they neither
    ride along on the iris nor leave a gap when it moves down.
    """
    below_lid = np.zeros_like(opening)
    outside = ~opening
    for step in range(1, LASH_EDGE + 1):
        below_lid[step:] |= outside[:-step]
    return _erode(opening, 1) & ~below_lid


def iris_layer(rgb, see_through, iris, ellipse):
    """(RGBA crop of the whole iris, the pixels the face must paint out) for one eye.

    Where the lids are see-through the disc shows the painting. The parts they hide are the
    iris's own colours carried on from its visible top and bottom rows, column by column (the
    nearest iris-coloured pixel, not a catchlight), so looking down shows iris, not a gap or a
    second lash line.
    """
    disc = ellipse_mask(see_through.shape, ellipse) | iris & see_through
    saturation, value = _saturation_value(rgb)
    rest = see_through & ~disc
    white = (saturation < IRIS_SATURATION) & (value > WHITE_BRIGHT * np.median(value[rest])) if rest.any() else False
    outline = see_through & ~white
    disc |= _reach(_dilate(disc) & outline, outline & _dilate(disc, OUTLINE_REACH))
    visible = disc & see_through
    hidden = disc & ~visible
    layer = np.zeros(see_through.shape + (4,), np.float32)
    layer[..., :3] = rgb
    # only the iris's own hue is carried on: a pink catchlight at the top became a streak looking down
    hue = _hue(rgb)
    iris_hue = _circular_median(hue[iris & see_through])
    saturation, _value = _saturation_value(rgb)
    irisish = (np.abs((hue - iris_hue + 0.5) % 1.0 - 0.5) < CARRY_HUE) & (saturation > CARRY_SATURATION)
    for column in np.nonzero(visible.any(axis=0))[0]:
        rows = np.nonzero(visible[:, column])[0]
        top, bottom = rows[0], rows[-1]
        coloured = rows[irisish[rows, column]]
        from_top = coloured[coloured >= top + IRIS_CARRY]
        from_bottom = coloured[coloured <= bottom - IRIS_CARRY]
        above = hidden[:top, column]
        layer[:top, column, :3][above] = rgb[from_top[0] if len(from_top) else min(top + IRIS_CARRY, bottom), column]
        below = hidden[bottom + 1:, column]
        layer[bottom + 1:, column, :3][below] = rgb[from_bottom[-1] if len(from_bottom) else max(bottom - IRIS_CARRY, top), column]
    unset = hidden & ~visible.any(axis=0)[None, :]
    if unset.any():
        layer[..., :3] = face_points.fill_from_border(layer[..., :3], unset, passes=4 * IRIS_SMOOTH, domain=disc)
    layer[..., :3] = _smooth(layer[..., :3], hidden, IRIS_SMOOTH, disc)
    layer[..., 3] = disc
    return layer, _dilate(disc & see_through, 2) & _dilate(see_through, 2)


def _hue(rgb):
    """Hue in turns (0..1) of each pixel of an (..., 3) array."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    top, bottom = rgb.max(axis=-1), rgb.min(axis=-1)
    span = np.maximum(top - bottom, 1e-6)
    hue = np.where(top == r, (g - b) / span, np.where(top == g, 2.0 + (b - r) / span, 4.0 + (r - g) / span))
    return (hue / 6.0) % 1.0


def _circular_median(hues):
    """A typical hue (turns) of many, where they bunch round the circle."""
    angle = 2 * np.pi * hues
    return float(np.arctan2(np.median(np.sin(angle)), np.median(np.cos(angle))) / (2 * np.pi)) % 1.0


def paint_out(rgb, see_through, hole):
    """The crop with the whole eye white, `hole` (under the iris) included, one neutral white.

    The white is the brightest of the eye's own (WHITE_TONE), without its tint: model A's whites are
    painted blue-grey, with a pale blue patch at one side, and any of that carried into the fill
    read as a grey or blue eye. Only dark lines (under WHITE_LINE of the typical brightness) are
    kept. The edge blends into the painting round it over WHITE_SMOOTH passes, so there is no seam
    against the lids.
    """
    rgb = np.array(rgb, dtype=np.float32)
    saturation, value = _saturation_value(rgb)
    rest = see_through & ~hole
    typical = np.median(value[rest]) if rest.any() else float(value.max())
    known = rest & (saturation < IRIS_SATURATION) & (value > WHITE_BRIGHT * typical)
    tone = float(np.percentile(value[known if known.any() else rest], WHITE_TONE))
    whiten = hole | rest & (value >= WHITE_LINE * typical)
    rgb[whiten] = (tone * white_shading(see_through | hole))[whiten]
    return _smooth(rgb, whiten & _dilate(~whiten, WHITE_SMOOTH), WHITE_SMOOTH, np.ones_like(whiten))


def white_shading(opening):
    """(rows, columns, 3) factors shading an eye white: the upper lid's shadow and darker corners.

    Drawn from the opening's shape, not copied from the painting (whose shading carried into the
    fill as bands), and fixed to the face, so the white an iris uncovers matches the rest.
    """
    rows, columns = opening.shape
    have = np.nonzero(opening.any(axis=0))[0]
    if not len(have):
        return np.ones(opening.shape + (3,), np.float32)
    top = np.interp(np.arange(columns), have, np.argmax(opening[:, have], axis=0).astype(float))
    width = max(1, int(SHADOW_EVEN * len(have)))
    top = np.convolve(np.pad(top, width, mode="edge"), np.ones(2 * width + 1) / (2 * width + 1), "valid")
    height = max(np.ptp(np.nonzero(opening.any(axis=1))[0]) + 1, 1)
    ys, xs = np.mgrid[0:rows, 0:columns] + 0.5
    lid = np.exp(-np.maximum(ys - top[None, :], 0.0) / (SHADOW_REACH * height))
    across = (xs - 0.5 * (have[0] + have[-1] + 1)) / max(0.5 * len(have), 1.0)
    shade = np.clip(SHADOW_LID * lid + SHADOW_CORNER * across ** 2, 0.0, 1.0)
    return (1.0 - shade[..., None] * (1.0 - np.array(SHADOW_TINT))).astype(np.float32)


def _smooth(rgb, region, passes, domain):
    """`rgb` with `region` blurred over `passes` rounds, drawing only on pixels in `domain`."""
    rgb = np.array(rgb, dtype=np.float32)
    weight = domain.astype(np.float32)[..., None]
    for _ in range(passes):
        weighted = rgb * weight
        total = weighted + sum(np.roll(weighted, s, a) for a in (0, 1) for s in (1, -1))
        count = weight + sum(np.roll(weight, s, a) for a in (0, 1) for s in (1, -1))
        rgb[region] = (total / np.maximum(count, 1e-6))[region]
    return rgb


def find_eye_paint(rgb, eye):
    """(one eye's whole painting as a crop mask, the skin behind it) for sliding eyes.

    The skin behind is a smooth fit to the skin around the eye (see skin_behind), so a blush or
    shading across the face doesn't count as paint. The painting is what differs from it, joined
    to the marked opening: lashes, iris, eye white and faint outlines, but not separate strokes
    such as brows.
    """
    height = eye.height
    rough = _dilate(eye.marked, max(2, int(PAINT_REACH * height)))
    behind = skin_behind(rgb, _dilate(rough, max(2, int(PAINT_RING * height))) & ~rough)
    difference = np.linalg.norm(rgb - behind, axis=-1)
    strong = difference > PAINT_STRONG
    weak = rough & (difference > PAINT_WEAK)
    seeds = strong & _dilate(eye.marked, max(1, int(0.1 * height)))
    paint = _fill_holes(_reach(seeds, weak | strong & rough))
    return _dilate(paint, PAINT_GROW), behind


def skin_behind(rgb, ring, rounds=3):
    """A smooth colour field over the crop, fitted to the skin pixels in `ring`.

    A quadratic in x and y per channel, refitted `rounds` times without the pixels it fits worst
    (a hat brim or a brow stroke caught in the ring), so it follows the face's shading and
    blush across a hole too big to fill by averaging.
    """
    ys, xs = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]].astype(np.float64)
    scale = max(rgb.shape[:2])
    x, y = xs / scale, ys / scale
    terms = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)
    use = ring.copy()
    for _ in range(rounds):
        coeffs = np.linalg.lstsq(terms[use], rgb[use], rcond=None)[0]
        fitted = terms @ coeffs
        residual = np.linalg.norm(rgb - fitted, axis=-1)
        use = ring & (residual <= max(2.5 * np.median(residual[ring]), 0.02))
    return fitted


def feather(mask, steps=FRINGE):
    """Alpha that is 1 on `mask` and falls to 0 over `steps` pixels outside it."""
    alpha = mask.astype(np.float32)
    grown = mask
    for step in range(1, steps + 1):
        grown_next = _dilate(grown)
        alpha[grown_next & ~grown] = 1.0 - step / (steps + 1)
        grown = grown_next
    return alpha


# --- the face surface --------------------------------------------------------------------------

class FacePatch:
    """The face patch's front mapping and surface, to lay layers onto the painted face.

    Pixels are the patch image's (y down); positions are world space. Depth is read off the
    face patch's own polygons, so hair hanging in front of an eye isn't taken for the face.
    """

    def __init__(self, mesh_obj):
        self.obj = mesh_obj
        self.image = face_points._patch_image(mesh_obj)
        if self.image is None:
            raise face_points.FacePointsError("eye tracking needs a face patch: run scripts/retexture/face_patch.py first")
        self.size = tuple(self.image.size)
        self.au, self.av = face_points._patch_mapping(mesh_obj)
        mesh = mesh_obj.data
        slot = mesh.materials.find(face_points.FACE_PATCH_MATERIAL)
        verts = meshdata.world_vertices(mesh_obj)
        tris = meshdata.triangles(mesh)
        polys = np.empty(len(mesh.loop_triangles), np.int64)
        mesh.loop_triangles.foreach_get("polygon_index", polys)
        materials = np.empty(len(mesh.polygons), np.int64)
        mesh.polygons.foreach_get("material_index", materials)
        self.surface = BVHTree.FromPolygons([Vector(v) for v in verts], tris[materials[polys] == slot].tolist())
        # everything, as seen from the front (the face patch leaves out bits in the model's own texture)
        self.everything = BVHTree.FromPolygons([Vector(v) for v in verts], tris.tolist())

    def world(self, pixels):
        """(n, 2) image pixels -> (n, 2) world (x, z)."""
        pixels = np.asarray(pixels, float)
        u = pixels[:, 0] / self.size[0]
        v = 1.0 - pixels[:, 1] / self.size[1]
        return np.stack([(u - self.au[1]) / self.au[0], (v - self.av[1]) / self.av[0]], axis=1)

    def pixel(self, xz):
        """(n, 2) world (x, z) -> (n, 2) image pixels."""
        u, v = self.uv(xz).T
        return np.stack([u * self.size[0], (1.0 - v) * self.size[1]], axis=1)

    def uv(self, xz):
        xz = np.asarray(xz, float)
        return np.stack([self.au[0] * xz[:, 0] + self.au[1], self.av[0] * xz[:, 1] + self.av[1]], axis=1)

    @property
    def metres_per_pixel(self):
        return 1.0 / (abs(self.au[0]) * self.size[0])

    def depth(self, xz, surface=None):
        """World y of `surface` (the face patch) seen from the front at each (x, z), NaN where none."""
        surface = surface or self.surface
        out = np.full(len(xz), np.nan)
        for i, (x, z) in enumerate(np.asarray(xz, float)):
            hit = surface.ray_cast(Vector((x, -10.0, z)), Vector((0.0, 1.0, 0.0)), 20.0)[0]
            if hit is not None:
                out[i] = hit.y
        return out


def _to_local(mesh_obj, points):
    matrix = np.array(mesh_obj.matrix_world.inverted())
    return np.asarray(points, float) @ matrix[:3, :3].T + matrix[:3, 3]


def _set_int(mesh, name, values):
    attribute = mesh.attributes.get(name) or mesh.attributes.new(name, "INT", "POINT")
    attribute.data.foreach_set("value", np.asarray(values, dtype=np.int32))


def get_int(mesh, name):
    """An integer point attribute's values, zeros when the mesh doesn't have it."""
    values = np.zeros(len(mesh.vertices), dtype=np.int32)
    if name in mesh.attributes:
        mesh.attributes[name].data.foreach_get("value", values)
    return values


def _keys(mesh):
    return list(mesh.shape_keys.key_blocks) if mesh.shape_keys else []


def _key_co(key):
    co = np.empty(len(key.data) * 3)
    key.data.foreach_get("co", co)
    return co.reshape(-1, 3)


def _material_like(mesh, name, image):
    """A copy of the face patch material showing `image`, added to the mesh; returns its slot."""
    material = bpy.data.materials.get(name)
    if material is None:
        material = mesh.materials[mesh.materials.find(face_points.FACE_PATCH_MATERIAL)].copy()
        material.name = name
    for node in material.node_tree.nodes if material.node_tree else ():
        if node.type == "TEX_IMAGE" and node.image is not None:
            node.image = image
    extension = getattr(material, "vrm_addon_extension", None)
    if extension is not None:
        mtoon1 = extension.mtoon1
        mtoon1.pbr_metallic_roughness.base_color_texture.index.source = image
        mtoon1.extensions.vrmc_materials_mtoon.shade_multiply_texture.index.source = image
    if material.name not in mesh.materials:
        mesh.materials.append(material)
    return mesh.materials.find(material.name)


def cut_out(material, blend=False):
    """Let the material's image alpha cut it out (MToon Cutout), or blend it (MToon Transparent)."""
    extension = getattr(material, "vrm_addon_extension", None)
    if extension is not None and extension.mtoon1.enabled:
        extension.mtoon1.alpha_mode = "BLEND" if blend else "MASK"
        extension.mtoon1.alpha_cutoff = 0.5
        return
    compat.set_alpha(material, blend)
    if material.node_tree is not None:
        bsdf = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        image = next((n for n in material.node_tree.nodes if n.type == "TEX_IMAGE"), None)
        if bsdf is not None and image is not None and not bsdf.inputs["Alpha"].links:
            material.node_tree.links.new(image.outputs["Alpha"], bsdf.inputs["Alpha"])


def crop_to_use(mesh, material_name):
    """Crop a material's image to the part its polygons use, moving their UVs to match.

    Returns {"box": (x0, y0, x1, y1), "full": (width, height)}, in the full image's pixels
    (rows bottom first).

    The eyes' images are front views the size of the whole face patch (2048 px square), and
    three of them added 30 MB to model A's VRM; the polygons using each cover a small part of it.
    """
    slot = mesh.materials.find(material_name)
    material = mesh.materials[slot]
    image = next(n.image for n in material.node_tree.nodes if n.type == "TEX_IMAGE" and n.image is not None)
    width, height = image.size
    uv = mesh.uv_layers.active.data
    loops = [li for poly in mesh.polygons if poly.material_index == slot for li in poly.loop_indices]
    if not loops:
        return {"box": (0, 0, width, height), "full": (width, height)}
    coords = np.array([uv[li].uv[:] for li in loops])
    x0, y0 = np.maximum(np.floor(coords.min(axis=0) * (width, height)).astype(int) - CROP_PAD, 0)
    x1, y1 = np.minimum(np.ceil(coords.max(axis=0) * (width, height)).astype(int) + CROP_PAD, (width, height))
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    kept = pixels.reshape(height, width, 4)[y0:y1, x0:x1]  # rows bottom first, like UV v
    cropped = new_image(image.name, kept[::-1])
    for li, (u, v) in zip(loops, coords):
        uv[li].uv = ((u * width - x0) / (x1 - x0), (v * height - y0) / (y1 - y0))
    _material_like(mesh, material_name, cropped)
    name = image.name
    bpy.data.images.remove(image)
    cropped.name = name
    return {"box": (int(x0), int(y0), int(x1), int(y1)), "full": (width, height)}


def _store(image, rows):
    image.pixels.foreach_set(np.ascontiguousarray(rows[::-1], dtype=np.float32).ravel())
    image.update()
    image.pack()


def _filled(values):
    """NaNs (no surface found) replaced by the nearest found value along the array."""
    values = np.asarray(values, float)
    good = ~np.isnan(values)
    if not good.any():
        raise face_points.FacePointsError("no face surface under an eye")
    return np.interp(np.arange(len(values)), np.nonzero(good)[0], values[good])


def _add_grid(mesh_obj, corners_xz, depth_of, uv_of, columns, rows_, slot, group, marks):
    """Add a grid of quads spanning `corners_xz` ((x0, z0), (x1, z1)) at world depth depth_of(xz).

    Its vertices are weighted fully to `group`, sit at their basis position in every shape key
    and are marked in the point attributes `marks` (name -> value). Returns their indices.
    """
    mesh = mesh_obj.data
    (x0, z0), (x1, z1) = corners_xz
    xs, zs = np.meshgrid(np.linspace(x0, x1, columns + 1), np.linspace(z0, z1, rows_ + 1))
    xz = np.stack([xs.ravel(), zs.ravel()], axis=1)
    local = _to_local(mesh_obj, np.stack([xz[:, 0], depth_of(xz), xz[:, 1]], axis=1))
    uvs = uv_of(xz)
    group_index = (mesh_obj.vertex_groups.get(group) or mesh_obj.vertex_groups.new(name=group)).index

    bm = bmesh.new()
    bm.from_mesh(mesh)
    start = len(bm.verts)
    deform = bm.verts.layers.deform.verify()
    uv_layer = bm.loops.layers.uv.active or bm.loops.layers.uv.new()
    new = [bm.verts.new(tuple(p)) for p in local]
    for vert in new:
        vert[deform][group_index] = 1.0
    width = columns + 1
    for j in range(rows_):
        for i in range(columns):
            ids = (j * width + i, j * width + i + 1, (j + 1) * width + i + 1, (j + 1) * width + i)
            quad = bm.faces.new([new[k] for k in ids])
            quad.material_index = slot
            quad.smooth = True
            for loop, k in zip(quad.loops, ids):
                loop[uv_layer].uv = tuple(uvs[k])
    bm.to_mesh(mesh)
    bm.free()

    indices = np.arange(start, start + len(new))
    for key in _keys(mesh):
        key_co = _key_co(key)
        key_co[indices] = local
        key.data.foreach_set("co", key_co.ravel())
    for name, value in marks.items():
        values = get_int(mesh, name)
        values[indices] = value
        _set_int(mesh, name, values)
    mesh.update()
    return indices


# --- moving irises ---------------------------------------------------------------------------

def track_surface(face, pixels):
    """A flat surface for an iris to slide on, over `pixels` (its window): a plane, tilted as the face is.

    Fitted to what shows in the opening, then moved forward until the eye's white (WHITE_GAP behind
    it) clears TRACK_CLEARS percent of it by IRIS_GAP; the rest, bar hair (more than HAIR_FRONT in
    front of the fit), is made see-through (see _see_through_openings). Laying the iris on the face
    itself failed on model A: its opening holds a step of about 1 cm (the socket deepens toward the
    outer corner), edges of the socket slant across it and hair strands are part of the face patch,
    so the iris jumped in depth and a diagonal look (two slides added up) pushed it through the
    lids. It is flat because the lids are lifted in front of it at the corners of their polygons,
    some over 2 cm across: a curved track bulged through the middle of them, a plane can't. The fit
    leaves out points further than TRACK_OUTLIER typical distances from it and is repeated.
    Returns (world y for (n, 2) world x-z points, the spread about the fit in m).
    """
    xz = face.world(pixels[::TRACK_STEP])
    y = face.depth(xz, face.everything)
    xz, y = xz[~np.isnan(y)], y[~np.isnan(y)]

    def terms(points):
        points = np.asarray(points, float)
        return np.stack([np.ones(len(points)), points[:, 0], points[:, 1]], axis=1)

    use = np.ones(len(y), bool)
    for _ in range(3):
        coeffs = np.linalg.lstsq(terms(xz[use]), y[use], rcond=None)[0]
        residual = y - terms(xz) @ coeffs
        use = np.abs(residual) <= max(TRACK_OUTLIER * np.median(np.abs(residual[use])), 0.002)
    shift = float(np.percentile(residual[residual > -HAIR_FRONT], 100 - TRACK_CLEARS)) - WHITE_GAP - IRIS_GAP
    return (lambda points: terms(points) @ coeffs + shift), float(np.std(residual[use]))


def add_moving_irises(mesh_obj, points):
    """Give each marked eye an iris that slides on eyeLook keys, the face under it painted out, lids in front.

    Assumes a finished, skinned mesh with a face patch, the character facing -Y, and face points
    marking both eyes' openings. The iris lies IRIS_GAP in front of the face and follows its shape
    as it slides. A turning disc (eye bones) can't: model A's eyes sit in dished sockets under a rim
    about 2 cm further forward, and any turn swung part of the iris through the lids or into the
    face. Existing vertices, weights and shape keys are left as they were; the face patch image
    changes only under the irises. Returns per side the eye's sizes (m) and iris vertices, the
    overlay's vertices and the see-through window as an image mask.
    """
    face = FacePatch(mesh_obj)
    rows = image_rows(face.image)
    lids = rows.copy()
    iris_rows = np.zeros_like(rows)
    windows = np.zeros(rows.shape[:2], bool)
    found = {}
    for side, eye in marked_eyes(points, face.size).items():
        x0, y0, x1, y1 = eye.box
        rgb = rows[y0:y1, x0:x1, :3].astype(np.float64)
        opening = find_opening(rgb, eye)
        see = window(opening)
        iris, ellipse = find_iris(rgb, opening)
        layer, hole = iris_layer(rgb, see, iris, ellipse)
        rows[y0:y1, x0:x1, :3] = paint_out(rgb, see, hole)
        lids[y0:y1, x0:x1, 3][see] = 0.0
        iris_rows[y0:y1, x0:x1] = layer
        windows[y0:y1, x0:x1] |= see
        found[side] = _measure_eye(face, eye, opening, see, ellipse, layer)

    _store(face.image, rows)
    mesh = mesh_obj.data
    lid_slot = _material_like(mesh, LIDS_MATERIAL, new_image(LIDS_MATERIAL, lids))
    iris_slot = _material_like(mesh, IRIS_MATERIAL, new_image(IRIS_MATERIAL, iris_rows))
    cut_out(mesh.materials[lid_slot])
    cut_out(mesh.materials[iris_slot])

    slide = iris_slide(found)
    for info in found.values():
        info["track"], info["track_spread"] = track_surface(face, info["window_pixels"])
        info["slide"] = slide
    opened = _see_through_openings(mesh_obj, face, found, lid_slot)
    overlay = _add_lid_overlay(mesh_obj, face, found, lid_slot, max(slide))
    patch_slot = mesh.materials.find(face_points.FACE_PATCH_MATERIAL)
    for side, info in found.items():
        pixels = info["window_pixels"]
        margin = WHITE_MARGIN / face.metres_per_pixel
        low, high = pixels.min(axis=0) - margin, pixels.max(axis=0) + margin
        corners = face.world([[low[0], high[1]], [high[0], low[1]]])
        info["white_vertices"] = _add_grid(
            mesh_obj, corners, lambda xz, track=info["track"]: track(xz) + WHITE_GAP, face.uv, WHITE_GRID, WHITE_GRID,
            patch_slot, BONE_NAMES["head"], {WHITE_ATTRIBUTE: SIDE_IDS[side], face_rig.FIXED_ATTRIBUTE: 1})
    for side, info in found.items():
        pixels = info["disc_pixels"]
        low, high = pixels.min(axis=0) - 1.5, pixels.max(axis=0) + 1.5
        corners = face.world([[low[0], high[1]], [high[0], low[1]]])
        info["vertices"] = _add_grid(
            mesh_obj, corners, info["track"], face.uv, IRIS_GRID, IRIS_GRID,
            iris_slot, BONE_NAMES["head"], {IRIS_ATTRIBUTE: SIDE_IDS[side], face_rig.FIXED_ATTRIBUTE: 1})
    _slide_keys(mesh_obj, found, lambda side, xz: found[side]["track"](xz))
    crops = {name: crop_to_use(mesh, name) for name in (LIDS_MATERIAL, IRIS_MATERIAL)}
    return {"eyes": found, "overlay_vertices": overlay, "window": windows, "see_through_polygons": opened,
            "crops": crops}


def _measure_eye(face, eye, opening, see, ellipse, layer):
    """World sizes of one found eye: its opening, its iris and the white beside the iris (m)."""
    x0, y0 = eye.box[:2]
    size = face.metres_per_pixel
    cx, cy, rx, ry = ellipse
    opening_px = eye.pixels(opening)
    see_rows = np.nonzero(see.any(axis=1))[0]
    columns = np.nonzero(see[int(np.clip(cy, see_rows[0], see_rows[-1]))])[0]
    white_left = max(cx - rx - columns[0], 0.0) * size if len(columns) else 0.0
    white_right = max(columns[-1] + 1 - (cx + rx), 0.0) * size if len(columns) else 0.0
    # the image's x runs toward the character's left (+X): the left eye's outer corner is on the right
    outer, inner = (white_right, white_left) if eye.side == "left" else (white_left, white_right)
    return {
        "box": eye.box,
        "ellipse": (float(cx + x0), float(cy + y0), float(rx), float(ry)),
        "opening_size": (float(np.ptp(opening_px[:, 0]) * size), float(np.ptp(opening_px[:, 1]) * size)),
        "iris_size": (float(2 * rx * size), float(2 * ry * size)),
        "white_outer": float(outer),
        "white_inner": float(inner),
        "disc_pixels": eye.pixels(layer[..., 3] > 0.5),
        "window_pixels": eye.pixels(see),
    }


def head_only(mesh_obj):
    """Per vertex, whether it follows the head bone alone: the face, not hair on springs."""
    head = mesh_obj.vertex_groups.get(BONE_NAMES["head"])
    only = np.zeros(len(mesh_obj.data.vertices), bool)
    if head is None:
        return only
    for vertex in mesh_obj.data.vertices:
        on_head = sum(g.weight for g in vertex.groups if g.group == head.index)
        others = sum(g.weight for g in vertex.groups if g.group != head.index)
        only[vertex.index] = on_head >= 0.99 and others <= 0.01
    return only


def _front_polygons(mesh_obj, xz, behind=None, before=None):
    """Indices of the polygons seen first from the front at any of the world points `xz` (n, 2).

    With `behind` (world y per point) only hits no further forward than that count, and with
    `before` only hits further forward than that.
    """
    mesh = mesh_obj.data
    polys = np.empty(len(mesh.loop_triangles), np.int64)
    tris = meshdata.triangles(mesh)
    mesh.loop_triangles.foreach_get("polygon_index", polys)
    scene = BVHTree.FromPolygons([Vector(v) for v in meshdata.world_vertices(mesh_obj)], tris.tolist())
    seen = set()
    for i, (x, z) in enumerate(np.asarray(xz, float)):
        hit, _normal, index, _distance = scene.ray_cast(Vector((x, -10.0, z)), Vector((0.0, 1.0, 0.0)), 20.0)
        if hit is not None and (behind is None or hit.y >= behind[i]) and (before is None or hit.y < before[i]):
            seen.add(int(polys[index]))
    return seen


def _see_through_openings(mesh_obj, face, found, slot):
    """Make what shows in the openings in front of the eyes' white see-through there; returns how many polygons.

    Edges of model A's sockets slant across its openings in front of where the iris and the white go,
    some in the model's own texture. They take the lids' material (slot), front-mapped, so they
    show their painting outside the openings and nothing inside them. Every WINDOW_STEP'th window
    pixel is looked at; hair crossing an opening (on spring bones, or more than HAIR_FRONT in front
    of the iris's track) stays as it is.
    """
    mesh = mesh_obj.data
    verts = meshdata.world_vertices(mesh_obj)
    face_only = head_only(mesh_obj)
    front = set()
    for info in found.values():
        xz = face.world(info["window_pixels"][::WINDOW_STEP])
        track = info["track"](xz)
        front |= _front_polygons(mesh_obj, xz, behind=track - HAIR_FRONT, before=track + WHITE_GAP)
    front = {i for i in front if face_only[list(mesh.polygons[i].vertices)].all()}
    uv = mesh.uv_layers.active.data
    for index in sorted(front):
        poly = mesh.polygons[index]
        poly.material_index = slot
        for li in poly.loop_indices:
            co = verts[mesh.loops[li].vertex_index]
            uv[li].uv = tuple(face.uv([[co[0], co[2]]])[0])
    mesh.update()
    return len(front)


def iris_slide(found):
    """(sideways, up and down) distance an iris slides at full eyeLook weight (m).

    Sideways, SIDE_TRAVEL of the white beside the irises; up and down, UP_DOWN_TRAVEL of the
    smaller iris's height (see those).
    """
    white = np.mean([info[key] for info in found.values() for key in ("white_outer", "white_inner")])
    return (float(SIDE_TRAVEL * white), float(UP_DOWN_TRAVEL * min(info["iris_size"][1] for info in found.values())))


def _add_lid_overlay(mesh_obj, face, found, slot, reach):
    """Copy the face-patch polygons round each eye OVERLAY_GAP toward the viewer: the lids the iris slides under.

    What shows round the eyes (the face patch, and corners in the model's own texture) is copied
    with the lids material, front-mapped, see-through in the openings; what is more than HAIR_FRONT
    in front of the iris's track (hair, the brow's rim) hides the iris by itself and isn't, nor is
    hair on spring bones. The copies cover the iris's whole travel
    (`reach`, m) and OVERLAY_MARGIN more at the full gap, lifted to LID_LIFT in front of the iris's
    track where the socket is deeper; one more ring of polygons eases the gap to nothing, so the
    overlay meets the face without a step. Every shape key moves each copy as it moves the vertex
    it was copied from, so the lids blink with the face. Returns the new vertices' indices.
    """
    mesh = mesh_obj.data
    verts = meshdata.world_vertices(mesh_obj)
    weld, _a, _b = meshdata.welded_edges(mesh_obj, verts)
    patch_slot = mesh.materials.find(face_points.FACE_PATCH_MATERIAL)
    boxes = []
    for info in found.values():
        xz = face.world(np.concatenate([info["disc_pixels"], info["window_pixels"]]))
        boxes.append((xz.min(axis=0) - reach - OVERLAY_MARGIN, xz.max(axis=0) + reach + OVERLAY_MARGIN))
    # whatever shows round the eyes (some corners are in the model's own texture), bar what is
    # well in front of the iris's track and hides it anyway (hair, the brow's rim)
    seen = set()
    for info, (low, high) in zip(found.values(), boxes):
        xz = np.stack(np.meshgrid(np.arange(low[0], high[0], SAMPLE_STEP), np.arange(low[1], high[1], SAMPLE_STEP)), -1).reshape(-1, 2)
        seen |= _front_polygons(mesh_obj, xz, behind=info["track"](xz) - HAIR_FRONT)
    face_only = head_only(mesh_obj)
    seen = {i for i in seen if face_only[list(mesh.polygons[i].vertices)].all()}

    core, touching = set(), {}
    for poly in mesh.polygons:
        if poly.index not in seen:
            continue
        corners = verts[list(poly.vertices)][:, [0, 2]]
        low, high = corners.min(axis=0), corners.max(axis=0)
        if any((low <= b_high).all() and (high >= b_low).all() for b_low, b_high in boxes):
            core.add(poly.index)
        for v in poly.vertices:
            touching.setdefault(int(weld[v]), []).append(poly.index)
    core_welds = {int(weld[v]) for i in core for v in mesh.polygons[i].vertices}
    ring = {i for w in core_welds for i in touching.get(w, ()) if i not in core}
    full = np.isin(weld, list(core_welds))

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    faces = [bm.faces[i] for i in sorted(core | ring)]
    geom = list({v for f in faces for v in f.verts}) + list({e for f in faces for e in f.edges}) + faces
    result = bmesh.ops.duplicate(bm, geom=geom)
    uv_layer = bm.loops.layers.uv.active
    for copy in result["geom"]:
        if isinstance(copy, bmesh.types.BMFace):
            # front-mapped like the face patch, whatever texture the polygon had
            copy.material_index = slot
            for loop in copy.loops:
                world = mesh_obj.matrix_world @ loop.vert.co  # still where its source is
                loop[uv_layer].uv = tuple(face.uv([[world.x, world.z]])[0])
    bm.verts.index_update()
    pairs = sorted((new.index, old.index) for old, new in result["vert_map"].items()
                   if isinstance(old, bmesh.types.BMVert) and new.index >= len(verts))
    bm.to_mesh(mesh)
    bm.free()
    new_indices = np.array([n for n, _ in pairs])
    old_indices = np.array([o for _, o in pairs])

    offset_world = np.zeros((len(old_indices), 3))
    offset_world[:, 1] = -OVERLAY_GAP * full[old_indices]
    # round the iris's travel the lids stay LID_LIFT in front of its track, however deep the socket:
    # every fully offset copy, each to the track of the eye it is nearest
    xz = verts[old_indices][:, [0, 2]]
    middles = np.array([(low + high) / 2 for low, high in boxes])
    nearest = np.argmin(np.linalg.norm(xz[:, None] - middles[None], axis=2), axis=1)
    for eye, info in enumerate(found.values()):
        over = full[old_indices] & (nearest == eye)
        lifted = info["track"](xz[over]) - LID_LIFT
        offset_world[over, 1] = np.minimum(verts[old_indices][over, 1] + offset_world[over, 1], lifted) - verts[old_indices][over, 1]
    offset_local = offset_world @ np.array(mesh_obj.matrix_world.inverted().to_3x3()).T
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    co[new_indices] = co[old_indices] + offset_local
    mesh.vertices.foreach_set("co", co.ravel())
    for key in _keys(mesh):
        key_co = _key_co(key)
        key_co[new_indices] = key_co[old_indices] + offset_local
        key.data.foreach_set("co", key_co.ravel())
    marks = get_int(mesh, OVERLAY_ATTRIBUTE)
    marks[new_indices] = old_indices + 1
    _set_int(mesh, OVERLAY_ATTRIBUTE, marks)
    mesh.update()
    return new_indices


def check_irises(mesh_obj, built):
    """Where the iris shows in each look, alone and two at once, open and with each blink on.

    Rays from the front meet the iris wherever it is painted (its image's alpha): it may show only
    through the see-through part of the lids. Anything opaque in front (the lids' paint, hair) hides
    it; iris showing with no lids in front, or in front of their paint, is a failure, and so is the
    face (not hair) in front of iris the window should show (eyes open). Shape keys add up, so a diagonal look
    is two slides on top of each other. Returns (numbers, failures).
    """
    mesh = mesh_obj.data
    face = FacePatch(mesh_obj)
    world = np.array(mesh_obj.matrix_world)
    keys = mesh.shape_keys.key_blocks
    basis = _key_co(keys[0])
    tris = meshdata.triangles(mesh)
    loops = np.empty(len(mesh.loop_triangles) * 3, np.int64)
    mesh.loop_triangles.foreach_get("loops", loops)
    loops = loops.reshape(-1, 3)
    polys = np.empty(len(mesh.loop_triangles), np.int64)
    mesh.loop_triangles.foreach_get("polygon_index", polys)
    materials = np.empty(len(mesh.polygons), np.int64)
    mesh.polygons.foreach_get("material_index", materials)
    uv = np.empty(len(mesh.loops) * 2)
    mesh.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    iris_vertex = get_int(mesh, IRIS_ATTRIBUTE) > 0
    others = ~iris_vertex[tris].all(axis=1)
    on_face = head_only(mesh_obj)[tris].all(axis=1)  # hair (on springs) may cover the eye
    lid_slot = mesh.materials.find(LIDS_MATERIAL)
    see_through = materials[polys] == lid_slot
    is_lid = (get_int(mesh, OVERLAY_ATTRIBUTE) > 0)[tris].all(axis=1) | see_through
    lid_alpha = image_rows(bpy.data.images[LIDS_MATERIAL])[..., 3]
    iris_alpha = image_rows(bpy.data.images[IRIS_MATERIAL])[..., 3]
    window_mask = built["window"]
    limit = np.array(face.size) - 1
    kept = np.nonzero(others)[0]

    def to_world(co):
        return co @ world[:3, :3].T + world[:3, 3]

    def alpha_at(image_alpha, uvs):
        # the image's own size: the eyes' images are cropped (crop_to_use)
        height, width = image_alpha.shape
        px = np.clip(np.stack([uvs[..., 0] * width, (1 - uvs[..., 1]) * height], axis=-1).astype(int), 0, (width - 1, height - 1))
        return image_alpha[px[..., 1], px[..., 0]]

    looks = [()] + [(d,) for d in ("In", "Out", "Up", "Down")] + [(a, b) for a in ("In", "Out") for b in ("Up", "Down")]
    blinks = [None] + [key.name for key in keys if "blink" in key.name.lower()]
    shown_count, failures = 0, []
    worst = {"over_lids": 0, "sunk": 0}
    for blink in blinks:
        state = to_world(basis if blink is None else _key_co(keys[blink]))
        scene = BVHTree.FromPolygons([Vector(p) for p in state], tris[kept].tolist())
        for side, info in built["eyes"].items():
            ids = info["vertices"]
            n = IRIS_GRID + 1
            grid_uv = uv_of_vertices(mesh, ids)
            # sample the grid's vertices and the middles of its quads
            quads = np.array([[j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i]
                              for j in range(n - 1) for i in range(n - 1)])
            sample_uv = np.concatenate([grid_uv, grid_uv[quads].mean(axis=1)])
            shown = alpha_at(iris_alpha, sample_uv) > 0.5
            for look in looks:
                co = basis[ids].copy()
                for direction in look:
                    co += _key_co(keys[f"eyeLook{direction}{side.title()}"])[ids] - basis[ids]
                points = to_world(co)
                samples = np.concatenate([points, points[quads].mean(axis=1)])[shown]
                where = f"{side} eye looking {'+'.join(look) or 'ahead'}" + (f" with {blink}" if blink else "")
                over_lids = sunk = 0
                for point in samples:
                    seen_lids, hidden, start = False, False, Vector((point[0], -10.0, point[2]))
                    for _ in range(8):
                        hit, _normal, index, _distance = scene.ray_cast(start, Vector((0.0, 1.0, 0.0)), 20.0)
                        if hit is None or hit.y >= point[1] - 1e-6:
                            break
                        t = kept[index]
                        if is_lid[t]:
                            seen_lids = True
                            weights = _barycentric(state[tris[t]], np.array(hit))
                            if not see_through[t] or alpha_at(lid_alpha, (weights @ uv[loops[t]])[None])[0] > 0.5:
                                hidden = True
                                break
                        else:
                            if blink is None and on_face[t] and point[1] - hit.y < IN_FRONT:
                                pixel = np.clip(face.pixel(point[None, [0, 2]]).astype(int), 0, limit)[0]
                                if window_mask[pixel[1], pixel[0]]:
                                    sunk += 1
                            hidden = True
                            break
                        start = hit + Vector((0.0, 1e-5, 0.0))
                    if not hidden and not seen_lids:
                        over_lids += 1
                shown_count += len(samples)
                worst["over_lids"] = max(worst["over_lids"], over_lids)
                worst["sunk"] = max(worst["sunk"], sunk)
                # a point or two where a window's edge pixel meets a polygon's edge isn't a flaw
                if over_lids > STRAY * len(samples):
                    failures.append(f"iris shows over the lids, {where}: {over_lids} of {len(samples)} points")
                if sunk > STRAY * len(samples):
                    failures.append(f"face in front of iris the lids should show, {where}: {sunk} of {len(samples)} points")
    return {"points": shown_count, **worst}, failures


def uv_of_vertices(mesh, indices):
    """The UV of each of `indices` (from any loop using it)."""
    loop_vertex = np.empty(len(mesh.loops), np.int64)
    mesh.loops.foreach_get("vertex_index", loop_vertex)
    uv = np.empty(len(mesh.loops) * 2)
    mesh.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    first = np.full(len(mesh.vertices), -1)
    first[loop_vertex[::-1]] = np.arange(len(loop_vertex))[::-1]
    return uv[first[indices]]


def _barycentric(corners, point):
    a, b, c = corners
    v0, v1, v2 = b - a, c - a, point - a
    d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
    denominator = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    return np.array([1 - v - w, v, w])


# --- sliding eyes ----------------------------------------------------------------------------

def add_sliding_eyes(mesh_obj, points, slide=SLIDE):
    """Move each marked eye's whole painting onto a patch that slides over the face on eyeLook keys.

    Assumes a finished, skinned mesh with a face patch, the character facing -Y, face points
    marking both eyes, and the painted eyes' boxes the build stores (face_rig.store_painted_eyes).
    The painting is erased from the face patch image, filled from the skin around it; the patches
    lie PATCH_GAP in front of the face and follow the head. Adds the eight ARKit eyeLook keys, each
    sliding one eye `slide` (a share of its width sideways, of its height up and down) along the
    face, and rebuilds eyeBlinkLeft/Right to squeeze the patches as the old keys squeezed the
    paint, with the face under them still. Other keys leave the patches alone. Returns what was built.
    """
    face = FacePatch(mesh_obj)
    rows = image_rows(face.image)
    paint_rows = np.zeros_like(rows)
    patches = {}
    for side, eye in marked_eyes(points, face.size).items():
        x0, y0, x1, y1 = eye.box
        rgb = rows[y0:y1, x0:x1, :3].astype(np.float64)
        paint, behind = find_eye_paint(rgb, eye)
        alpha = feather(paint)
        paint_rows[y0:y1, x0:x1, :3] = rgb
        paint_rows[y0:y1, x0:x1, 3] = alpha
        # the fitted skin, nudged to meet the skin round the painting exactly: a slight step left a
        # ghost of the eye where it slid away
        under = alpha > 0
        # drawn only from skin-like pixels round it (faint outline strokes the paint mask missed
        # were averaged in), and all the way across: the blush under model B's eyes stopped at the
        # erase's edge and showed as pale blobs under the blink
        residual = rgb - behind
        clean = np.linalg.norm(residual, axis=-1) <= PAINT_WEAK
        behind = behind + face_points.fill_across(residual, under, domain=clean)
        rows[y0:y1, x0:x1, :3] = rgb + (behind - rgb) * alpha[..., None]
        drawn = eye.pixels(alpha > 0)
        size = face.metres_per_pixel
        top = np.zeros_like(paint)
        top[: max(1, paint.shape[0] // 3)] = True  # rows count down from the eye's top
        dark = rgb[paint & top] if (paint & top).any() else rgb[paint]
        lum = dark @ np.array([0.3, 0.59, 0.11])
        patches[side] = {"pixels": drawn, "slide": (slide[0] * eye.width * size, slide[1] * eye.height * size),
                         "paint_mm2": float(paint.sum() * size * size * 1e6),
                         "skin": np.median(behind[alpha > 0.5], axis=0) if (alpha > 0.5).any() else np.median(behind.reshape(-1, 3), axis=0),
                         "lash": np.median(dark[lum <= np.percentile(lum, 15)], axis=0)}

    _store(face.image, rows)
    mesh = mesh_obj.data
    repainted = _patch_under(mesh_obj, face, patches)
    slot = _material_like(mesh, PATCH_MATERIAL, new_image(PATCH_MATERIAL, paint_rows))
    # a soft edge: the patch's rim is skin that has to fade into the face wherever it slides
    cut_out(mesh.materials[slot], blend=True)
    head = BONE_NAMES["head"]
    for side, info in patches.items():
        low, high = info["pixels"].min(axis=0) - 1.5, info["pixels"].max(axis=0) + 1.5
        corners = face.world([[low[0], high[1]], [high[0], low[1]]])
        reach = max(info["slide"]) + TRACK_MARGIN
        info["track"] = smooth_track(face, corners.min(axis=0) - reach, corners.max(axis=0) + reach, max(info["slide"]), PATCH_GAP)
        info["vertices"] = _add_grid(mesh_obj, corners, info["track"], face.uv, *PATCH_GRID, slot, head,
                                     {PATCH_ATTRIBUTE: SIDE_IDS[side], face_rig.FIXED_ATTRIBUTE: 1})
    _slide_keys(mesh_obj, patches, lambda side, xz: patches[side]["track"](xz))
    # the lid overlay (_lid_blink) is off until it looks right: two attempts on model B left the lower
    # half of the eye showing (it closed to the squeeze's meeting line) and the eye through its fade
    amounts = _lid_blink(mesh_obj, face, patches) if LIDS else _patch_blink(mesh_obj, face, patches)
    crops = {PATCH_MATERIAL: crop_to_use(mesh, PATCH_MATERIAL)}
    return {"patches": patches, "blink": amounts, "repainted_polygons": repainted, "crops": crops}


def _patch_under(mesh_obj, face, patches):
    """Put what shows under the eyes' painting on the face patch, front-mapped; returns how many polygons.

    The build leaves dark polygons in the model's own texture on the face patch's edge
    (face_texture.extend_patch), and on model B some carry the generator's own paint of the
    eyes: it stayed on the face as the eyes slid and blinked. On the face patch they show the
    erased skin. Only polygons at the face's depth are taken (within CLEAR_CAP), not a hat's brim.
    """
    mesh = mesh_obj.data
    slot = mesh.materials.find(face_points.FACE_PATCH_MATERIAL)
    verts = meshdata.world_vertices(mesh_obj)
    under = set()
    for info in patches.values():
        xz = face.world(info["pixels"][::WINDOW_STEP * 2])
        near = _filled(face.depth(xz))
        under |= _front_polygons(mesh_obj, xz, behind=near - CLEAR_CAP)
    uv = mesh.uv_layers.active.data
    moved = 0
    for index in sorted(under):
        poly = mesh.polygons[index]
        if poly.material_index == slot:
            continue
        poly.material_index = slot
        for li in poly.loop_indices:
            co = verts[mesh.loops[li].vertex_index]
            uv[li].uv = tuple(face.uv([[co[0], co[2]]])[0])
        moved += 1
    mesh.update()
    return moved


def smooth_track(face, low, high, radius, gap):
    """A smooth surface `gap` in front of the face over the box `low`..`high` (world x-z): y for (n, 2) points.

    The face is read every TRACK_CELL and, point by point, the most forward of it within `radius`
    is taken, then blurred over the same radius, which never falls behind it: bumps are cleared
    and the result is smooth at the scale of the slides. Laid straight on model B's ball, a
    patch looking two ways at once (two slides added up) sank behind the bumps of the eye sockets.
    Anything more than CLEAR_CAP in front of a robust fit of the face (the hat's brim) is ignored.
    """
    xs = np.arange(low[0], high[0] + TRACK_CELL, TRACK_CELL)
    zs = np.arange(low[1], high[1] + TRACK_CELL, TRACK_CELL)
    gx, gz = np.meshgrid(xs, zs)
    xz = np.stack([gx.ravel(), gz.ravel()], axis=1)
    y = face.depth(xz)
    good = ~np.isnan(y)
    terms = np.stack([np.ones(len(xz)), xz[:, 0], xz[:, 1], xz[:, 0] ** 2, xz[:, 0] * xz[:, 1], xz[:, 1] ** 2], axis=1)
    use = good.copy()
    for _ in range(3):
        coeffs = np.linalg.lstsq(terms[use], y[use], rcond=None)[0]
        residual = y - terms @ coeffs
        use = good & (np.abs(residual) <= max(TRACK_OUTLIER * np.median(np.abs(residual[use])), 0.002))
    fit = terms @ coeffs
    y = np.where(good & (y > fit - CLEAR_CAP), y, fit).reshape(gx.shape)
    steps = max(1, int(round(radius / TRACK_CELL)))
    forward = y.copy()
    padded = np.pad(y, steps, mode="edge")
    for dz in range(-steps, steps + 1):
        for dx in range(-steps, steps + 1):
            forward = np.minimum(forward, padded[steps + dz:steps + dz + y.shape[0], steps + dx:steps + dx + y.shape[1]])
    padded = np.pad(forward, steps, mode="edge")
    blurred = sum(padded[steps + dz:steps + dz + y.shape[0], steps + dx:steps + dx + y.shape[1]]
                  for dz in range(-steps, steps + 1) for dx in range(-steps, steps + 1)) / (2 * steps + 1) ** 2
    track = blurred - gap

    def depth(points):
        points = np.asarray(points, float)
        fx = np.clip((points[:, 0] - xs[0]) / TRACK_CELL, 0, len(xs) - 1.001)
        fz = np.clip((points[:, 1] - zs[0]) / TRACK_CELL, 0, len(zs) - 1.001)
        ix, iz = fx.astype(int), fz.astype(int)
        tx, tz = fx - ix, fz - iz
        return ((track[iz, ix] * (1 - tx) + track[iz, ix + 1] * tx) * (1 - tz)
                + (track[iz + 1, ix] * (1 - tx) + track[iz + 1, ix + 1] * tx) * tz)

    return depth


def _slide_keys(mesh_obj, parts, depth_of):
    """The eight ARKit eyeLook keys: each slides one eye's part (an iris or a patch) by its "slide".

    Moved vertices go to the world depth depth_of(side, xz) at their new place (on a track or
    on the face), so the part follows the surface as it slides.
    """
    mesh = mesh_obj.data
    basis = _key_co(mesh.shape_keys.key_blocks[0])
    rest = meshdata.world_vertices(mesh_obj)
    for side, info in parts.items():
        across, up = info["slide"]
        inward = -1.0 if side == "left" else 1.0  # toward the nose: the left eye sits at +X
        moves = {"In": (inward * across, 0.0), "Out": (-inward * across, 0.0), "Up": (0.0, up), "Down": (0.0, -up)}
        for direction, (dx, dz) in moves.items():
            coords = basis.copy()
            xz = rest[info["vertices"]][:, [0, 2]] + (dx, dz)
            world = np.stack([xz[:, 0], depth_of(side, xz), xz[:, 1]], axis=1)
            coords[info["vertices"]] = _to_local(mesh_obj, world)
            _replace_key(mesh_obj, f"eyeLook{direction}{side.title()}", coords)


def _replace_key(mesh_obj, name, coords):
    keys = mesh_obj.data.shape_keys.key_blocks
    if name in keys:
        keys[name].data.foreach_set("co", np.asarray(coords, dtype=np.float32).ravel())
    else:
        face_rig._add_key(mesh_obj, name, coords)


def _lid_blink(mesh_obj, face, patches):
    """Rebuild eyeBlinkLeft/Right to close a lid over each patch; returns the amounts (1: shut).

    Squeezing the patch, as painted eyes blink, uncovered the skin painted out under it, paler
    than the face round it (model B's closing eyes showed a "hole"); the patch stays, and a lid of the
    skin's colour with the lashes along its edge comes down over it, folded flat at its top edge at
    rest. Warudo drives the ARKit shapes by name, so this is geometry, not a texture swap.
    """
    mesh = mesh_obj.data
    boxes = mesh_obj.get(face_rig.PAINTED_EYES_PROPERTY)
    if boxes is None:
        raise face_points.FacePointsError("sliding eyes need the painted eyes' boxes: build with face points")
    world = np.array(mesh_obj.matrix_world)
    to_world = lambda xz0, z: (world[:3, :3] @ np.array([xz0, 0.0, z]) + world[:3, 3])  # noqa: E731
    rows = np.ones((64, 64, 4))
    skin = np.mean([info["skin"] for info in patches.values()], axis=0)
    lash = np.mean([info["lash"] for info in patches.values()], axis=0)
    rows[..., :3] = skin
    rows[-int(64 * LID_LASH):, :, :3] = lash  # the image's bottom rows: the lid's lower edge
    # its top and sides fade into the face (a flat-coloured rectangle stood out on model B)
    v = np.linspace(1.0, 0.0, 64)[:, None]  # 1 at the image's top row, 0 at the lash
    u = np.abs(np.linspace(-1.0, 1.0, 64))[None, :]
    rows[..., 3] = np.clip((1.0 - v) / LID_FADE_TOP, 0, 1) * np.clip((1.0 - u) / LID_FADE_SIDE, 0, 1)
    rows[-int(64 * LID_LASH):, :, 3] = np.clip((1.0 - u[0]) / LID_FADE_SIDE, 0, 1)
    slot = _material_like(mesh, LID_MATERIAL, new_image(LID_MATERIAL, rows))
    cut_out(mesh.materials[slot], blend=True)
    head = BONE_NAMES["head"]
    amounts = {}
    for side, info in patches.items():
        # the whole painting (lashes and all), not the face points' box, which is smaller
        painted = face.world(info["pixels"])
        (wx0, wz0), (wx1, wz1) = painted.min(axis=0), painted.max(axis=0)
        width, height = wx1 - wx0, wz1 - wz0
        reach_x = LID_MARGIN * width + info["slide"][0]
        line = wz0 + face_rig.MEET_LINE * height
        top = wz1 + LID_ABOVE * height + info["slide"][1]
        corners = np.array([[wx0 - reach_x, line], [wx1 + reach_x, top]])
        depth = lambda xz, info=info: info["track"](xz) - LID_FRONT  # noqa: E731
        span = corners[1] - corners[0]
        uv_of = lambda xz, c=corners, d=span: np.stack([(xz[:, 0] - c[0][0]) / d[0],  # noqa: E731
                                                        (xz[:, 1] - c[0][1]) / d[1]], axis=1)
        ids = _add_grid(mesh_obj, corners, depth, uv_of, *LID_GRID, slot, head,
                        {LID_ATTRIBUTE: SIDE_IDS[side], face_rig.FIXED_ATTRIBUTE: 1})
        keys = mesh.shape_keys.key_blocks
        shut = _key_co(keys[0])[ids]
        # the closing edge dips in the middle; every row keeps its share of the way to it
        columns, rows_ = LID_GRID
        u = np.tile(np.linspace(-1.0, 1.0, columns + 1), rows_ + 1)
        share = np.repeat(np.linspace(0.0, 1.0, rows_ + 1), columns + 1)
        local_line = _to_local(mesh_obj, np.stack([np.zeros(len(ids)), np.zeros(len(ids)),
                                                   np.full(len(ids), line) - LID_CURVE * height * (1 - u * u)], axis=1))[:, 2]
        local_top = _to_local(mesh_obj, np.array([[0.0, 0.0, top]]))[0, 2]
        shut[:, 2] = local_line + share * (local_top - local_line)
        # each vertex at the face's depth where it now is: a round face (a mascot) curves away below
        shut_world = shut @ world[:3, :3].T + world[:3, 3]
        shut_world[:, 1] = depth(shut_world[:, [0, 2]])
        shut = _to_local(mesh_obj, shut_world)
        # at rest: every row on the top row, depth and all: no area, not seen from any side
        folded = np.tile(shut[-(columns + 1):], (rows_ + 1, 1))
        name = f"eyeBlink{side.title()}"
        for key in keys:
            # the blink is the lid alone: the face and the patch stay (the face's own squeeze, left
            # in from face tracking, bent the eye under the lid)
            co = _key_co(keys[0]) if key.name == name else _key_co(key)
            co[ids] = shut if key.name == name else folded
            key.data.foreach_set("co", co.ravel())
        amounts[side] = 1.0
    mesh.update()
    return amounts


def _patch_blink(mesh_obj, face, patches):
    """Rebuild eyeBlinkLeft/Right so they squeeze the patches, not the face; returns the amounts.

    The amount is read back from the old key where it squeezed the painted eye's middle (the
    default face_rig.PAINTED_BLINK without one). Squeezed patch vertices are laid back onto the
    face, which is round on a mascot: moving straight down would sink them into it.
    """
    mesh = mesh_obj.data
    boxes = mesh_obj.get(face_rig.PAINTED_EYES_PROPERTY)
    if boxes is None:
        raise face_points.FacePointsError("sliding eyes need the painted eyes' boxes: build with face points")
    keys = mesh.shape_keys.key_blocks
    basis = _key_co(keys[0])
    world = np.array(mesh_obj.matrix_world)
    amounts = {}
    for side, info in patches.items():
        name = f"eyeBlink{side.title()}"
        box = tuple(boxes[side])
        amounts[side] = _blink_amount(basis, _key_co(keys[name]), box) if name in keys else face_rig.PAINTED_BLINK
        coords = basis.copy()
        squeezed = basis[info["vertices"]].copy()
        squeezed[:, 2] += face_rig.squeeze_shut(squeezed, box, amounts[side])
        world_xz = (squeezed @ world[:3, :3].T + world[:3, 3])[:, [0, 2]]
        depth = info["track"](world_xz)
        coords[info["vertices"]] = _to_local(mesh_obj, np.stack([world_xz[:, 0], depth, world_xz[:, 1]], axis=1))
        _replace_key(mesh_obj, name, coords)
    return amounts


def _blink_amount(basis, key, box):
    """How far a squeeze blink key closes the eye in `box`, from the vertices in its middle."""
    x0, z0, x1, z1 = box
    height = z1 - z0
    line = z0 + face_rig.MEET_LINE * height
    z = basis[:, 2]
    middle = (np.abs(basis[:, 0] - (x0 + x1) / 2) < 0.25 * (x1 - x0)) & (z > z0) & (z < z1) & (np.abs(line - z) > 0.2 * height)
    moved = middle & (np.abs(key[:, 2] - z) > 1e-7)
    if moved.sum() < 3:
        return face_rig.PAINTED_BLINK
    return float(np.median((key[moved, 2] - z[moved]) / (line - z[moved])))


def check_sliding(mesh_obj, built):
    """Where the face hides a sliding patch, in each look (alone and two at once) and each blink.

    Rays from the front meet the patch over its whole area, not just its corners. The face in
    front of a painted point (less than CLEAR_CAP in front: hats and hair may cover the eyes) is a
    failure once more than STRAY of the points in one look, or BLINK_STRAY while blinking too (see
    that). Returns (numbers, failures).
    """
    mesh = mesh_obj.data
    world = np.array(mesh_obj.matrix_world)
    keys = mesh.shape_keys.key_blocks
    basis = _key_co(keys[0])
    tris = meshdata.triangles(mesh)
    patch_vertex = (get_int(mesh, PATCH_ATTRIBUTE) > 0) | (get_int(mesh, LID_ATTRIBUTE) > 0)
    kept = np.nonzero(~patch_vertex[tris].all(axis=1))[0]
    alpha = image_rows(bpy.data.images[PATCH_MATERIAL])[..., 3]
    face = FacePatch(mesh_obj)
    limit = np.array(face.size) - 1
    scene = BVHTree.FromPolygons([Vector(p) for p in basis @ world[:3, :3].T + world[:3, 3]], tris[kept].tolist())
    looks = [()] + [(d,) for d in ("In", "Out", "Up", "Down")] + [(a, b) for a in ("In", "Out") for b in ("Up", "Down")]
    worst, failures, checked = 0, [], 0
    columns, rows_ = PATCH_GRID
    n = columns + 1
    quads = np.array([[j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i]
                      for j in range(rows_) for i in range(columns)])
    for blink in [None] + [key.name for key in keys if key.name.startswith("eyeBlink")]:
        for side, info in built["patches"].items():
            ids = info["vertices"]
            uv = uv_of_vertices(mesh, ids)
            sample_uv = np.concatenate([uv, uv[quads].mean(axis=1)])
            height, width = alpha.shape
            px = np.clip(np.stack([sample_uv[:, 0] * width, (1 - sample_uv[:, 1]) * height], axis=1).astype(int), 0, (width - 1, height - 1))
            painted = alpha[px[:, 1], px[:, 0]] > 0.5
            for look in looks:
                names = [f"eyeLook{d}{side.title()}" for d in look]
                if blink is not None and blink.endswith(side.title()):
                    names.append(blink)
                co = basis[ids].copy()
                for name in names:
                    co += _key_co(keys[name])[ids] - basis[ids]
                points = co @ world[:3, :3].T + world[:3, 3]
                samples = np.concatenate([points, points[quads].mean(axis=1)])[painted]
                hidden = 0
                for point in samples:
                    hit = scene.ray_cast(Vector((point[0], -10.0, point[2])), Vector((0.0, 1.0, 0.0)), 20.0)[0]
                    if hit is not None and point[1] - CLEAR_CAP < hit.y < point[1]:
                        hidden += 1
                checked += len(samples)
                worst = max(worst, hidden)
                if hidden > (STRAY if blink is None else BLINK_STRAY) * len(samples):
                    where = side + " eye looking " + ("+".join(look) or "ahead") + (" with " + blink if blink else "")
                    failures.append(f"face over the eye patch, {where}: {hidden} of {len(samples)} points")
    return {"points": checked, "hidden": worst}, failures
