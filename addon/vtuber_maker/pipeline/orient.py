"""Work out which way an imported character is standing and facing.

The rest of the pipeline expects a standing character with +Z up facing -Y, which is
what Tripo's glTF gives after Blender's import. Other generators and other file formats
don't agree: FBX and OBJ often come in Y-up, and a model can face any way round. Nothing
checked it before, so a model that came in lying down or facing away rigged silently
wrong: legs detected across the body, the face on the back of the head.

Detection is geometric, so it needs no texture:

- **Up** runs along the model's longest side, pointed away from the end that looks like
  feet: the legs split into two clusters there, and a standing character's soles are flat
  and wide while the crown of the head comes to a point.
- **Facing** is the horizontal axis the body does not mirror across (left and right match,
  front and back don't), pointed the way the feet stick out past the hips.

Both are reported with a confidence; a model that doesn't read as a standing body keeps
the pipeline's own frame and the caller warns instead of rigging something wrong.
"""

import numpy as np

from .meshdata import sample_surface

SAMPLES = 120_000
AXES = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

# Axes at least this share of the longest one are candidates for up (a chibi with wide
# arms is barely taller than it is wide), and the legs and soles tell them apart.
LONG_ENOUGH = 0.8
# Height bands, as fractions of the model's extent along a candidate up axis.
LEG_BAND = (0.08, 0.40)
FOOT_BAND = (0.0, 0.08)
HIP_BAND = (0.45, 0.60)
END_SLAB = 0.02
BAND_SLICES = 12
# Two clusters count as legs when the gap between them is at least this share of the
# body's width at that height.
LEG_GAP = 0.04
# Confidence: the feet have to stand out past the hips by this share of the body's depth,
# and the up end has to beat the other one by this much score.
FACING_MARGIN = 0.04
# The two feet: floor-level surface in cells this share of the height, a patch counting as a foot
# from this share of that surface; the ankle is read in the band just above. A pair is two patches
# whose places, widths and lengths differ by no more than this share of the distance between them,
# and the toes must reach past the ankle by this share of the foot's length on both.
FOOT_CELL = 0.012
FOOT_SHARE = 0.05
ANKLE_BAND = (0.08, 0.16)
FEET_ALIKE = 0.5
TOE_MARGIN = 0.08
UP_MARGIN = 0.4
# A body is a mirror image across its upright axis and across no other: the cue that held on every
# model, where legs and soles did not. A tester's pirate in a T-pose (arms as wide as he is tall,
# a coat and a tail between the legs, a head with more surface on top than his soles have below)
# scored legs 0.08 and soles 0 standing, legs 0.50 lying on his side, and was laid on his side.
MIRROR_WEIGHT = 4.0
# The side-on silhouette mirrors less than the front one by at least this much before it names the
# depth axis: 0.12 to 0.53 less on every model measured, with a tail in ten poses or without.
MIRROR_CLEAR = 0.05
# A tail held up behind the back can end above the head. The head's top is looked for no further
# behind the body's middle than this share of the height, and what stands above it by more than
# this share of the body is not the head (hair tips, a hat's point bent back and a thin ahoge on
# every model measured: under 0.005).
TOP_REACH = 0.05
ABOVE_HEAD = 0.04
# Horns, tall ears and antennae stand above the crown too, either side of it: the head's top is read
# no further from the head's middle, left or right, than this share of the height. A tester's demon
# girl had her head joint, and with it the face patch's frame, up among her horns; model A with horns
# 15% of her height added failed as arms at the sides.
TOP_STRIP = 0.025
# The head's middle, left to right, is the median of the surface this far down from the top.
TOP_BAND = 0.25
# The way the file stands counts for this much: importers hand nearly every model over upright, so
# it takes the shape's clear word to turn one. Without it the same pirate, upright in his file,
# read as standing on his head by 0.09 (the soles cue, on hair against boots).
AS_FILED = 0.5


class Orientation:
    """How a model is standing, and the rotation that puts it in the pipeline's frame."""

    def __init__(self, up, facing, up_sure=True, facing_sure=True):
        self.up = np.asarray(up, dtype=float)
        self.facing = np.asarray(facing, dtype=float)
        self.up_sure = up_sure
        self.facing_sure = facing_sure

    @property
    def confident(self):
        return self.up_sure and self.facing_sure

    @property
    def matrix(self):
        """3x3 rotation taking model space to the pipeline's: +Z up, facing -Y, character left +X."""
        return np.array([np.cross(self.up, self.facing), -self.facing, self.up])

    @property
    def is_identity(self):
        return np.allclose(self.matrix, np.eye(3), atol=1e-6)

    def describe(self):
        return f"up {_axis_name(self.up)}, facing {_axis_name(self.facing)}"


def detect(verts, tris):
    """Find which way `verts` is standing and facing. Returns an Orientation."""
    verts = np.asarray(verts, dtype=float)
    tris = np.asarray(tris)
    points = sample_surface(verts, tris, SAMPLES) if len(tris) else verts

    extents = np.ptp(points, axis=0)
    candidates = [AXES[i] * sign for i in range(3) if extents[i] >= LONG_ENOUGH * extents.max()
                  for sign in (1.0, -1.0)]
    scored = sorted(((_standing_score(points, up) + AS_FILED * np.allclose(up, AXES[2]), tuple(up)) for up in candidates), reverse=True)
    best, runner_up = scored[0], scored[1]
    up = np.array(best[1])
    # A second candidate almost as good means the model doesn't read as standing one way up.
    up_sure = best[0] - runner_up[0] > UP_MARGIN

    facing, facing_sure = _facing(points, up)
    return Orientation(up, facing, up_sure, facing_sure)


def head_top(points, middle=None, verts=None):
    """The height of the top of the head when something stands higher behind it (a tail held up
    over the back) or beside its crown (horns, tall ears), else None: the model's top is the head's.

    For a character in the pipeline's frame (+Z up, facing -Y). `middle` is the body's depth, the
    surface's median when not given: pass the head joint's once joints are known. A tail that
    curls forward over the crown is not told from a hat. With `verts` the height is read on them:
    surface samples end a few millimetres under a pointed top, differently on each copy of a
    model, and a rig scaled by one no longer sat on its source scaled by another.
    """
    points = np.asarray(points, dtype=float)
    low, top = float(points[:, 2].min()), float(points[:, 2].max())
    middle = float(np.median(points[:, 1])) if middle is None else float(middle)
    reach = middle + TOP_REACH * (top - low)
    front = points[points[:, 1] < reach]
    if not len(front):
        return None
    # what a tail leaves is the body: its crown lies about the middle of its upper part
    body_top = float(front[:, 2].max())
    centre = float(np.median(front[front[:, 2] > body_top - TOP_BAND * (body_top - low), 0]))
    strip = TOP_STRIP * (body_top - low)
    crown = front[np.abs(front[:, 0] - centre) < strip, 2]
    # ears and hair tips a little over the crown are the head's still: the strip counts only when
    # what stands beside it is as far above it as a tail has to be
    beside = bool(len(crown)) and body_top - float(crown.max()) > ABOVE_HEAD * (float(crown.max()) - low)
    head = float(crown.max()) if beside else body_top
    if top - head <= ABOVE_HEAD * (head - low):
        return None
    if verts is None:
        return head
    verts = np.asarray(verts, dtype=float)
    kept = verts[:, 1] < reach
    return float(verts[kept & (np.abs(verts[:, 0] - centre) < strip) if beside else kept, 2].max())


def _standing_score(points, up):
    """How much the model looks like a body standing along `up`: higher is better."""
    along = points @ up
    low, height = float(along.min()), float(np.ptp(along))
    flat = _plane(points, up)
    width = float(np.ptp(flat, axis=0).max())
    if height <= 0 or width <= 0:
        return 0.0
    # A sole is flat and wide, so a slab at the bottom holds far more surface than one at
    # the crown of the head; legs stand apart above the feet and nothing else does.
    sole = float((along - low < END_SLAB * height).mean())
    crown = float((along - low > (1 - END_SLAB) * height).mean())
    flatness = float(np.clip(np.log10(sole / max(crown, 1e-9)), 0.0, 2.0))
    # A T-pose is about as wide as it is tall, and on a mascot in a robe legs and soles barely
    # told +Z from +X (1.34 against 1.17). A body is a mirror image left to right, never head
    # to foot, so that decides; both ends of one axis score it alike, so it can't flip one.
    # Both read across the body's width: its widest side before, and a tail held up behind an
    # A-pose makes a body deeper than it is wide. Seen from the side, head and tail stood apart
    # like two legs and nothing mirrored: OC3 and Slime with such a tail stood on their heads.
    spread, mirror = _across(flat, along)
    return 2.0 * _legs_share(spread, along, low, height, width) + flatness + MIRROR_WEIGHT * mirror


def _across(flat, along):
    """The body's left-to-right coordinate among the two of `flat`, and how much its silhouette
    matches its mirror image there (1 is exactly): the one that mirrors best, the widest when
    they mirror about alike."""
    shares = [_mirrored(flat[:, i], along) for i in (0, 1)]
    clear = abs(shares[0] - shares[1]) > MIRROR_CLEAR
    i = int(np.argmax(shares)) if clear else int(np.argmax(np.ptp(flat, axis=0)))
    return flat[:, i], shares[i]


def _mirrored(spread, along, cells=40):
    """How much the silhouette across `spread` matches its mirror image: 1 is exactly."""
    side = spread - np.median(spread)
    reach = max(float(np.abs(side).max()), 1e-9)
    grid = np.histogram2d(side, along, bins=cells, range=[[-reach, reach], [along.min(), along.max()]])[0] > 0
    return float((grid & grid[::-1]).sum() / max((grid | grid[::-1]).sum(), 1))


def _plane(points, up):
    """Points projected onto the plane across `up`, as 2D coordinates."""
    u = np.cross(up, AXES[(int(np.argmax(np.abs(up))) + 1) % 3])
    u /= np.linalg.norm(u)
    return np.stack([points @ u, points @ np.cross(up, u)], axis=1)


def _legs_share(spread, along, low, height, width):
    """Share of slices above the feet where the body splits in two across `spread`, like legs."""
    edges = np.linspace(low + LEG_BAND[0] * height, low + LEG_BAND[1] * height, BAND_SLICES + 1)
    # Legs stand either side of the body's middle. Any two clusters counted before: lying on his
    # side, a T-pose pirate's arm and a thin tail beside it were legs in every slice.
    middle, reach = float(np.median(spread)), LEG_GAP * width
    split = 0
    for bottom, top in zip(edges[:-1], edges[1:]):
        band = np.sort(spread[(along >= bottom) & (along < top)])
        if len(band) < 50:
            continue
        gaps = np.diff(band)
        widest = int(np.argmax(gaps))
        if gaps[widest] > reach and band[widest] - reach < middle < band[widest + 1] + reach:
            split += 1
    return split / BAND_SLICES


def _facing(points, up):
    """The horizontal direction the character looks along, and whether the cue was clear.

    Toes stick out in front of the hips, on boots as much as on bare feet, which held on
    every model tried (a Tripo girl, a mascot in a robe, a synthetic fixture). The head is
    no use: hair hangs behind on one character and a hat's brim reaches forward on another.
    """
    flat_axes = [AXES[i] for i in range(3) if not np.isclose(abs(up @ AXES[i]), 1.0)]
    extents = [float(np.ptp(points @ axis)) for axis in flat_axes]
    along = points @ up
    low, height = float(along.min()), float(np.ptp(along))
    # Left and right mirror each other, front and back don't: that names the depth axis. The
    # thinner axis did before ("people are wider than they are deep"), and a tail held out behind
    # an A-pose made the body deeper than wide: it was turned to face sideways. The thinner one
    # still decides when the two silhouettes mirror about alike.
    mirrored = [_mirrored(points @ axis, along) for axis in flat_axes]
    clear = abs(mirrored[0] - mirrored[1]) > MIRROR_CLEAR
    depth_index = int(np.argmin(mirrored)) if clear else int(np.argmin(extents))
    depth_axis = flat_axes[depth_index]
    depth = extents[depth_index]

    # The two feet by themselves first. Everything at foot height counted as "the feet" before: a
    # fox tail hanging to the floor behind a pirate outweighed his boots' toes, his coat pushed the
    # hips forward, and the model was turned to face backwards, marked sure.
    toes = _toes(points, along, low, height, depth_axis, np.cross(up, depth_axis))
    if toes is not None:
        return depth_axis * np.sign(toes), True
    feet = _band_centre(points, along, low, height, FOOT_BAND, depth_axis)
    hips = _band_centre(points, along, low, height, HIP_BAND, depth_axis)
    if feet is None or hips is None:
        return -AXES[1], False
    overhang = feet - hips
    return depth_axis * np.sign(overhang or 1.0), abs(overhang) > FACING_MARGIN * depth


def _toes(points, along, low, height, depth_axis, width_axis):
    """How far the toes reach past the ankles along `depth_axis`, as a share of the foot's length
    (signed), read on the two feet alone; None when the model shows no pair of feet that agree.

    The feet are the two patches of floor-level surface that mirror each other across the body's
    middle: a tail's tip, a cape's hem or a sword's point is a patch of its own with no twin. Each
    foot is measured against the ankle standing on it, so nothing behind or beside the legs counts.
    A robe to the floor is one patch: no pair, and the caller falls back to feet against hips.
    """
    wide, deep = points @ width_axis, points @ depth_axis
    cell = FOOT_CELL * height
    cells = np.stack([np.floor(wide / cell), np.floor(deep / cell)], axis=1).astype(np.int64)
    keys = cells[:, 0] * 100_003 + cells[:, 1]
    foot = (along >= low + FOOT_BAND[0] * height) & (along <= low + FOOT_BAND[1] * height)
    ankle = (along > low + ANKLE_BAND[0] * height) & (along <= low + ANKLE_BAND[1] * height)
    if foot.sum() < 50:
        return None
    patches = _patches(cells[foot])
    middle = float(np.median(wide))
    found = []
    for patch in patches:
        on = foot & np.isin(keys, patch)
        if on.sum() < FOOT_SHARE * foot.sum():
            continue
        span = np.percentile(deep[on], [2, 98])
        found.append({"cells": patch, "offset": float(np.mean(wide[on])) - middle, "width": float(np.ptp(wide[on])),
                      "span": span, "length": float(span[1] - span[0])})
    pairs = [(abs(a["offset"] + b["offset"]) + abs(a["width"] - b["width"]) + abs(a["length"] - b["length"]), a, b)
             for a in found if a["offset"] < 0 for b in found if b["offset"] > 0]
    if not pairs:
        return None
    mismatch, left, right = min(pairs, key=lambda pair: pair[0])
    if mismatch > FEET_ALIKE * (right["offset"] - left["offset"]):
        return None
    reach = []
    for side in (left, right):
        above = ankle & np.isin(keys, side["cells"])
        if above.sum() < 30 or side["length"] <= 0:
            return None
        shin = np.percentile(deep[above], [2, 98])
        reach.append(float(side["span"].mean() - shin.mean()) / side["length"])
    if reach[0] * reach[1] <= 0 or min(map(abs, reach)) < TOE_MARGIN:
        return None
    return float(np.mean(reach))


def _patches(cells):
    """Occupied grid cells grouped into patches that touch (corners count); each a key array."""
    keys = np.unique(cells[:, 0] * 100_003 + cells[:, 1])
    group = {int(k): int(k) for k in keys}

    def root(k):
        while group[k] != k:
            group[k] = group[group[k]]
            k = group[k]
        return k
    for k in group:
        for step in (1, 100_003, 100_004, 100_002):
            if k + step in group:
                group[root(k)] = root(k + step)
    patches = {}
    for k in group:
        patches.setdefault(root(k), []).append(k)
    return [np.array(patch) for patch in patches.values()]


def _band_centre(points, along, low, height, band, axis):
    """The middle of the surface's span along `axis` in a height band, ignoring outliers."""
    picked = points[(along >= low + band[0] * height) & (along <= low + band[1] * height)]
    if len(picked) < 50:
        return None
    values = picked @ axis
    return float(np.percentile(values, 98) + np.percentile(values, 2)) / 2


def _axis_name(vector):
    i = int(np.argmax(np.abs(vector)))
    return f"{'+' if vector[i] > 0 else '-'}{'XYZ'[i]}"
