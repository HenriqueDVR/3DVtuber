"""Estimate humanoid joint positions on an unrigged character mesh.

Expects a standing A-pose character facing -Y with +Z up, which is how
image-to-3D generators export a front-view character sheet. The surface is
resampled evenly and cut into thin horizontal bands; gaps between point
clusters inside a band tell legs, arms and torso apart. Where geometry can't
pin a joint down, typical body proportions fill in.
"""

import numpy as np

from .meshdata import sample_surface

BAND_COUNT = 160
# Enough points that a sparsely triangulated limb still slices into one solid cluster.
SAMPLE_COUNT = 200_000

# Ratios of total height (feet to top of hair).
CLUSTER_GAP = 0.015
HIP_JOINT_ABOVE_CROTCH = 0.045
HIPS_ABOVE_HIP_JOINTS = 0.012
ANKLE_HEIGHT = 0.049
SHOULDER_ABOVE_ARMPIT = 0.035
NECK_ABOVE_SHOULDERS = 0.012
LEG_SEARCH_WIDTH = 0.25
# A leg's centre never sits further out than this from the body's centre line; clusters
# beyond it are a hand, a sleeve or a piece of the costume hanging beside the legs.
LEG_MAX_OFFSET = 0.18
ARM_MIN_OFFSET = 0.12
ARM_AXIS_RADIUS = 0.06
# An arm that stops standing apart below this height is hanging into the body or hair,
# so only the hand is visible and the shoulder falls back to typical proportions.
HANGING_ARM_SEPARATES_BELOW = 0.62
SHOULDER_HEIGHT = 0.78
SHOULDER_OFFSET = 0.1
# Arms out to the sides: the upper body (above ARM_OUT_BAND[0] of the height) spans more than
# ARMS_OUT_WIDTH of the height. Arms are traced inward from the fingertips across X, over
# the heights in ARM_OUT_BAND, down to ARM_OUT_INNER from the centre (clear of any shoulder).
ARMS_OUT_WIDTH = 0.7
ARM_OUT_BAND = (0.5, 0.97)
ARM_OUT_INNER = 0.2
# ... starting on the forearm, this far out: model A's elbow is at 0.19 and hand at 0.33, model C's hand 0.35.
ARM_OUT_START = 0.25
# A traced cross-section taller than this is the arm with something hanging from it (a
# sleeve, a mantle): its top still follows the arm, its bottom doesn't.
ARM_OUT_THICK = 0.1
# A cross-section no thicker than this many times the arm's typical one is the arm alone.
ARM_BARE = 1.3
# The torso's side is read in a band this far under the armpit, and the shoulder joint sits
# SHOULDER_INSET inside it.
TORSO_UNDER_ARMPIT = (0.06, 0.02)
SHOULDER_INSET = 0.0
# Shoulder joints measured 0.068 (model A) and 0.097 (model B) of the height out from the centre. A
# torso side read outside SHOULDER_OUT_RANGE is something else joined to it (a mantle hanging
# from the arms, on a T-pose copy of model B: 0.26), and the shoulder takes SHOULDER_OUT_PRIOR.
SHOULDER_OUT_RANGE = (0.05, 0.11)
SHOULDER_OUT_PRIOR = 0.085
# A level arm nothing could trace: shoulder height and shoulder-to-fingertip length, as shares of
# the height (model A 0.75 and 0.32, model B 0.73 and 0.36).
ARM_PRIOR_HEIGHT = 0.74
ARM_PRIOR_LENGTH = 0.32

# Fractions along shoulder -> fingertip.
ELBOW_ALONG_ARM = 0.43
WRIST_ALONG_ARM = 0.79

# Fraction of neck base -> top of head where the head bone starts.
HEAD_ALONG_NECK_TO_TOP = 0.24

# Typical proportions, as fractions of the model's whole height (feet to the top of the hair),
# measured on the characters this was built on: hips at 0.50, knees halfway down to the ankles.
# They stand in for what clothing hides: a robe, a coat panel or a long skirt joins the legs in
# the silhouette, and the crotch then reads far too low (a mascot in a robe: 0.20 instead of 0.47,
# putting the hips at the ankles) or isn't found at all.
HIP_JOINT_PRIOR = 0.49
LEG_SPREAD_PRIOR = 0.055
CROTCH_RANGE = (0.38, 0.62)


class LandmarkError(RuntimeError):
    """The mesh doesn't look like a standing A-pose humanoid."""


def detect_body(verts, tris, warnings=None):
    """Return VRM humanoid joint positions keyed like "left_upper_arm".

    Character left is +X. Each value is an (x, y, z) array in the input's space.
    Anything clothing hides falls back to typical proportions rather than to a wrong
    reading of the silhouette, and each fallback is appended to `warnings` (a list) so
    callers can tell the user which joints to check or pass by hand.
    """
    warnings = warnings if warnings is not None else []
    points = sample_surface(np.asarray(verts, dtype=float), np.asarray(tris), SAMPLE_COUNT)
    bands = _Bands(points)
    center_x = float(np.median(points[:, 0]))
    height = bands.height

    crotch_z = _hip_height(bands, center_x, warnings)
    hip_z = crotch_z + HIP_JOINT_ABOVE_CROTCH * height
    ankle_z = bands.bottom + ANKLE_HEIGHT * height
    knee_z = (hip_z + ankle_z) / 2

    joints = {}
    for side, name in ((1, "left"), (-1, "right")):
        thigh = _leg_centroid(bands, crotch_z - 0.03 * height, side, center_x, warnings)
        joints[f"{name}_upper_leg"] = np.array([thigh[0], thigh[1], hip_z])
        knee = _leg_centroid(bands, knee_z, side, center_x, warnings)
        joints[f"{name}_lower_leg"] = np.array([knee[0], knee[1], knee_z])
        ankle = _leg_centroid(bands, ankle_z, side, center_x, warnings)
        joints[f"{name}_foot"] = np.array([ankle[0], ankle[1], ankle_z])

    body_y = (joints["left_upper_leg"][1] + joints["right_upper_leg"][1]) / 2
    joints["hips"] = np.array([center_x, body_y, hip_z + HIPS_ABOVE_HIP_JOINTS * height])

    shoulder_heights = []
    # Arms held out to the sides (a T-pose) lie along the bands, fused to the torso in every
    # one of them, so they are sliced across X instead; a model is read that way first when
    # its upper body is about as wide as it is tall.
    upper = points[points[:, 2] > bands.bottom + ARM_OUT_BAND[0] * height]
    out_first = np.ptp(upper[:, 0]) > ARMS_OUT_WIDTH * height
    arms = {}
    for side, name in ((1, "left"), (-1, "right")):
        if not out_first:
            try:
                arms[side] = _arm(bands, points, side, center_x)
            except LandmarkError:
                arms[side] = _arm_out(bands, points, side, center_x, warnings)
            continue
        # A model this wide is held out or has something wide beside the arms (wings): the
        # hanging-arm finder read model C's wing edge as an arm dropping to the hip, so it isn't
        # asked. A side that can't be traced mirrors the other, or both take typical proportions.
        try:
            arms[side] = _arm_out(bands, points, side, center_x, warnings)
        except LandmarkError:
            pass
    for side in (1, -1):
        if side in arms:
            continue
        if -side in arms:
            arms[side] = tuple(np.array([2 * center_x - j[0], j[1], j[2]]) for j in arms[-side])
            warnings.append("one arm couldn't be told apart from what is beside it (wings?): it mirrors the "
                            "other; check both on the Joints page")
        else:
            arms[side] = _arm_prior(bands, side, center_x, body_y)
            warnings.append("the arms couldn't be told apart from what is beside them (wings?): they were "
                            "placed from typical proportions; move them on the Joints page")
    for side, name in ((1, "left"), (-1, "right")):
        shoulder, elbow, wrist = arms[side]
        joints[f"{name}_upper_arm"] = shoulder
        joints[f"{name}_lower_arm"] = elbow
        joints[f"{name}_hand"] = wrist
        shoulder_heights.append(shoulder[2])
    if out_first:
        # arms held out take one depth each from their forearm; a hand turned differently on one
        # side put model A's copy 2 cm apart, and the shoulder bones between them weighted hair beside
        # the head to the chest (9.5 cm tear on a head turn)
        depth = np.mean([joints[f"{n}_upper_arm"][1] for n in ("left", "right")])
        for n in ("left", "right"):
            for k in ("upper_arm", "lower_arm", "hand"):
                joints[f"{n}_{k}"] = np.array([joints[f"{n}_{k}"][0], depth, joints[f"{n}_{k}"][2]])

    neck_z = float(np.mean(shoulder_heights)) + NECK_ABOVE_SHOULDERS * height
    top_z = bands.bottom + height
    joints["neck"] = np.array([center_x, body_y, neck_z])
    joints["head"] = np.array([center_x, body_y, neck_z + HEAD_ALONG_NECK_TO_TOP * (top_z - neck_z)])
    return joints


class _Bands:
    """Vertices grouped into horizontal bands, each split into clusters along X."""

    def __init__(self, verts):
        self.bottom = float(verts[:, 2].min())
        self.height = float(verts[:, 2].max()) - self.bottom
        self.thickness = self.height / BAND_COUNT
        band = np.clip(((verts[:, 2] - self.bottom) / self.thickness).astype(int), 0, BAND_COUNT - 1)
        order = np.argsort(band, kind="stable")
        bounds = np.searchsorted(band[order], np.arange(BAND_COUNT + 1))
        gap = CLUSTER_GAP * self.height
        self.clusters = [_split_along_x(verts[order[bounds[i] : bounds[i + 1]]], gap) for i in range(BAND_COUNT)]

    def z(self, index):
        return self.bottom + (index + 0.5) * self.thickness

    def index(self, z):
        return int(np.clip((z - self.bottom) / self.thickness, 0, BAND_COUNT - 1))


# Limb joint refinement: cross-sections this thick (either side of the joint) and wide are
# circle-fitted in LIMB_SECTORS angular sectors; sectors within LIMB_INWARD radians of the
# body's centre are ignored for arms, which touch the torso in an A-pose.
LIMB_SLICE = 0.012
LIMB_RADIUS = 0.09
LIMB_SECTORS = 16
LIMB_INWARD = np.radians(60)
# A refined joint moving further than this is taken as a failed fit and left where it was.
LIMB_MAX_SHIFT = 0.08
# An arm hangs (rather than being held out) when shoulder-to-wrist drops by more than this share
# of its length: sin 25 degrees, between the steepest T-pose and the flattest A-pose seen.
ARM_HANGS = 0.42
# Knees sit at least this fraction of the leg's length in front of the hip-to-ankle line.
KNEE_BEND = 0.05


def bend_knees(joints):
    """Put each knee in line between hip and ankle, and at least KNEE_BEND of the leg in front.

    Apps bend legs with IK in the direction the rest pose already bends them (crouching when
    the head drops, sitting). A straight or back-bent knee leaves that to chance, and the legs
    fold sideways. Returns a new joints dict.
    """
    joints = {k: np.array(v, dtype=float) for k, v in joints.items()}
    for side in ("left", "right"):
        hip, knee, ankle = (joints[f"{side}_{k}"] for k in ("upper_leg", "lower_leg", "foot"))
        line = ankle - hip
        length = float(np.linalg.norm(line))
        on_line = hip + ((knee - hip) @ line) / (line @ line) * line
        forward = max(-(knee - on_line)[1], KNEE_BEND * length)  # the character faces -Y
        joints[f"{side}_lower_leg"] = on_line + np.array([0.0, -forward, 0.0])
    return joints


def refine_limbs(verts, tris, joints, exclude=None):
    """Move arm and leg joints to the middle of their limb's cross-section.

    Detection finds joints from the silhouette, which puts arm joints near the sleeve's
    outer edge and lets things like a thigh holster pull a leg joint aside. Returns a new
    joints dict; hip joints are made mirror images of each other about the hips.
    `exclude` masks vertices to ignore (hair hanging beside the arms); triangles touching
    them are left out. The surface is sampled evenly, as big flat triangles have few vertices.
    """
    tris = np.asarray(tris)
    if exclude is not None:
        tris = tris[~np.asarray(exclude)[tris].any(axis=1)]
    points = sample_surface(np.asarray(verts, dtype=float), tris, SAMPLE_COUNT)
    joints = {k: np.array(v, dtype=float) for k, v in joints.items()}
    centre_x = float(joints["hips"][0])

    for name, side in (("left", 1.0), ("right", -1.0)):
        shoulder, elbow, wrist = (joints[f"{name}_{k}"] for k in ("upper_arm", "lower_arm", "hand"))
        inward = np.array([-side, 0.0, 0.0])
        new_elbow = _limb_centre(points, elbow, wrist - shoulder, inward)
        new_wrist = _limb_centre(points, wrist, wrist - shoulder, inward)
        # The upper arm is often fused to the torso, so a fitted centre drifts into it. Place the
        # shoulder from the sleeve's outer edge instead: one arm radius (measured at the free
        # elbow) in from that edge, and in line front-to-back with the elbow.
        new_shoulder = shoulder.copy()
        # An arm held out (a T-pose) has no outer edge beside the shoulder, only more arm, and
        # its shoulder already sits on the traced axis: this is for arms hanging down.
        hanging = abs((wrist - shoulder)[2]) > ARM_HANGS * np.linalg.norm(wrist - shoulder)
        elbow_edge = _outer_edge(points, new_elbow, side) if hanging else None
        probe = shoulder + 0.4 * (elbow - shoulder)
        probe_edge = _outer_edge(points, np.array([probe[0], new_elbow[1], probe[2]]), side) if hanging else None
        if elbow_edge is not None and probe_edge is not None:
            radius = abs(elbow_edge - new_elbow[0])
            shift_x = (probe_edge - side * radius) - probe[0]
            if abs(shift_x) < LIMB_MAX_SHIFT:
                new_shoulder[0] += shift_x
                # the joint is the middle of the shoulder's ball, an arm radius under its top
                new_shoulder[2] -= min(radius, LIMB_MAX_SHIFT)
        # Front to back, the upper arm follows the sleeve's outer half: gear worn on the forearm
        # (bracers, gloves held forward) pulls the fitted elbow in front of the upper arm.
        for joint in (new_shoulder, new_elbow) if hanging else ():
            probe_at = joint if joint is new_elbow else shoulder + 0.4 * (elbow - shoulder)
            depth = _sleeve_depth(points, probe_at, elbow - shoulder, side)
            if depth is not None and abs(depth - joint[1]) < LIMB_MAX_SHIFT:
                joint[1] = depth
        new_shoulder[1] = new_elbow[1] = (new_shoulder[1] + new_elbow[1]) / 2
        if not hanging:
            # _arm_out took the depth from the bare forearm; a cross-section here takes in any
            # cloth hanging from the arm (model B's mantle: 5 cm back)
            new_shoulder[1], new_elbow[1], new_wrist[1] = shoulder[1], elbow[1], wrist[1]
        if hanging:
            new_shoulder[0] = _outside_the_body(new_shoulder, new_elbow, new_wrist, side)
        joints[f"{name}_upper_arm"], joints[f"{name}_lower_arm"], joints[f"{name}_hand"] = new_shoulder, new_elbow, new_wrist

        hip, knee, ankle = (joints[f"{name}_{k}"] for k in ("upper_leg", "lower_leg", "foot"))
        joints[f"{name}_lower_leg"] = _limb_centre(points, knee, ankle - hip)
        joints[f"{name}_foot"] = _limb_centre(points, ankle, ankle - hip)
        probe = hip + 0.2 * (knee - hip)
        joints[f"{name}_upper_leg"] = hip + (_limb_centre(points, probe, knee - hip) - probe)

    _mirror(joints, "upper_leg", centre_x)
    # Shoulders too: a fit that lands differently on each side is a bad fit, not a character
    # built lopsided, and mismatched shoulders show up as one arm swinging wrong.
    _mirror(joints, "upper_arm", centre_x, keep_depth=True)
    return joints


def _mirror(joints, key, centre_x, keep_depth=False):
    """Make a left/right joint pair mirror images about the body's centre line."""
    left, right = joints[f"left_{key}"], joints[f"right_{key}"]
    spread = (abs(left[0] - centre_x) + abs(right[0] - centre_x)) / 2
    z = (left[2] + right[2]) / 2
    ys = (left[1], right[1]) if keep_depth else ((left[1] + right[1]) / 2,) * 2
    joints[f"left_{key}"] = np.array([centre_x + spread, ys[0], z])
    joints[f"right_{key}"] = np.array([centre_x - spread, ys[1], z])


def _outside_the_body(shoulder, elbow, wrist, side):
    """Shoulder x, never further in than the arm's own axis reaches at that height.

    Fitting the shoulder from the body's outer edge works while the body is about as wide as
    a person's. On a wide costume (a mascot's robe) the outer edge is the robe's, and taking an
    arm's radius off it buried the joint in the chest, 6 cm inside the sleeve: bone heat then
    spread arm weight over the mantle and the robe tore under the arm. The free part of the
    arm says where the sleeve really runs, so the shoulder is kept on or outside that line.
    """
    shaft = elbow - wrist  # points up the arm, towards the shoulder
    if abs(shaft[2]) < 1e-6:
        return float(shoulder[0])
    along_axis = float(elbow[0] + (shoulder[2] - elbow[2]) / shaft[2] * shaft[0])
    return float(shoulder[0]) if side * (shoulder[0] - along_axis) > 0 else along_axis


def _outer_edge(points, at, side, reach=0.12, band=0.02):
    """x of the body's outermost surface on this side, in a small horizontal band around `at`."""
    pick = (np.abs(points[:, 2] - at[2]) < band / 2) & (np.abs(points[:, 1] - at[1]) < band)         & (side * (points[:, 0] - at[0]) > -reach) & (side * (points[:, 0] - at[0]) < reach)
    if pick.sum() < 5:
        return None
    return float(side * np.percentile(side * points[pick, 0], 98))


def _sleeve_depth(points, at, direction, side):
    """Middle y of the arm's cross-section at `at`, from its outer half only (None if too thin)."""
    axis = direction / np.linalg.norm(direction)
    rel = points - at
    along = rel @ axis
    radial = rel - along[:, None] * axis
    pick = (np.abs(along) < LIMB_SLICE) & (np.linalg.norm(radial, axis=1) < LIMB_RADIUS) & (side * radial[:, 0] > -0.01)
    if pick.sum() < 20:
        return None
    lo, hi = np.percentile(points[pick, 1], [5, 95])
    return float((lo + hi) / 2)


def _limb_centre(points, joint, direction, inward=None):
    """Centre of the limb's cross-section at the joint (the joint itself if the fit fails).

    Detected joints can sit on the limb's surface or outside it, so the fit is repeated
    around each new centre; the side facing the body is only ignored once the centre is
    inside the limb.
    """
    axis = direction / np.linalg.norm(direction)
    u = np.cross(axis, (0.0, 1.0, 0.0))
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(axis, (1.0, 0.0, 0.0))
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    rel = points - joint
    along = rel @ axis
    near = np.abs(along) < LIMB_SLICE
    rel = rel[near] - along[near, None] * axis
    flat_all = np.stack([rel @ u, rel @ v], axis=1)
    towards = None if inward is None else np.arctan2(inward @ v, inward @ u)

    centre = np.zeros(2)
    for attempt in range(4):
        flat = flat_all[np.linalg.norm(flat_all - centre, axis=1) < LIMB_RADIUS] - centre
        angle = np.arctan2(flat[:, 1], flat[:, 0])
        if towards is not None and attempt > 0:
            off = np.abs((angle - towards + np.pi) % (2 * np.pi) - np.pi)
            flat, angle = flat[off > LIMB_INWARD], angle[off > LIMB_INWARD]
        sector = ((angle + np.pi) / (2 * np.pi) * LIMB_SECTORS).astype(int) % LIMB_SECTORS
        samples = []
        for s_ in range(LIMB_SECTORS):
            pick = flat[sector == s_]
            if len(pick):
                radii = np.linalg.norm(pick, axis=1)
                samples.append(pick[np.argsort(radii)[len(radii) // 2]])
        if len(samples) < 5:
            return joint
        samples = np.array(samples)
        for _ in range(2):
            fitted, radius = _fit_circle(samples)
            error = np.abs(np.linalg.norm(samples - fitted, axis=1) - radius)
            keep = error < max(0.5 * radius, 1e-3)
            if keep.sum() < 5 or keep.all():
                break
            samples = samples[keep]
        step = fitted
        centre = centre + step
        if np.linalg.norm(step) < 0.002 and attempt > 0:
            break
    shift = centre[0] * u + centre[1] * v
    if np.linalg.norm(shift) > LIMB_MAX_SHIFT:
        return joint
    return joint + shift


def _fit_circle(points):
    """Least-squares circle through 2D points: (centre, radius)."""
    a = np.column_stack([2 * points, np.ones(len(points))])
    b = (points**2).sum(axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.array([cx, cy]), float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def _runs(points, axis, gap):
    """`points` split wherever they leave a gap wider than `gap` along `axis`."""
    if len(points) == 0:
        return []
    order = np.argsort(points[:, axis])
    return [points[run] for run in np.split(order, np.nonzero(np.diff(points[order, axis]) > gap)[0] + 1)]


def _split_along_x(points, gap):
    if len(points) == 0:
        return []
    order = np.argsort(points[:, 0])
    breaks = np.nonzero(np.diff(points[order, 0]) > gap)[0] + 1
    return [points[run] for run in np.split(order, breaks)]


def _spans(cluster, x):
    return cluster[:, 0].min() <= x <= cluster[:, 0].max()


def _offset(cluster, side, center_x):
    return side * (cluster[:, 0].mean() - center_x)


def _hip_height(bands, center_x, warnings):
    """Where the legs meet, from the silhouette when it is believable and proportions when not.

    Clothing between the legs (a robe's hem, a coat panel, a long skirt) leaves a gap only
    near the floor, so the "crotch" reads at the shins. Anything outside CROTCH_RANGE is
    taken as hidden and replaced by the typical height, which needs saying: leg bones
    placed from proportions are close, not exact.
    """
    prior = bands.bottom + HIP_JOINT_PRIOR * bands.height - HIP_JOINT_ABOVE_CROTCH * bands.height
    try:
        crotch_z = _crotch_height(bands, center_x)
    except LandmarkError:
        warnings.append("the legs never stand apart (a long skirt or a robe?): "
                        "hips and legs placed from typical proportions; check them, or pass --joints")
        return prior
    low, high = (bands.bottom + f * bands.height for f in CROTCH_RANGE)
    if not low <= crotch_z <= high:
        warnings.append(
            f"the legs only stand apart at {(crotch_z - bands.bottom) / bands.height:.0%} of the height, "
            "which is clothing between them, not the crotch: hips and legs placed from typical "
            "proportions; check them, or pass --joints")
        return prior
    return crotch_z


def _crotch_height(bands, center_x):
    """Top of the lowest run of bands where the legs are separate."""
    top = None
    for i, clusters in enumerate(bands.clusters):
        z = bands.z(i)
        if z > bands.bottom + 0.7 * bands.height:
            break
        if not clusters:
            continue
        near = [c for c in clusters if abs(c[:, 0].mean() - center_x) < LEG_SEARCH_WIDTH * bands.height]
        legs_apart = (
            not any(_spans(c, center_x) for c in clusters)
            and any(_offset(c, 1, center_x) > 0 for c in near)
            and any(_offset(c, -1, center_x) > 0 for c in near)
        )
        if legs_apart:
            top = z
        elif top is not None:
            break
    if top is None:
        raise LandmarkError("could not find two separate legs")
    return top


def _leg_centroid(bands, z, side, center_x, warnings=None):
    """XY centre of the leg on one side at height z (the innermost cluster; hands sit further out).

    With the legs joined in the silhouette at that height, the leg is somewhere inside the
    cluster spanning the centre line: it falls back to the typical spread from the middle.
    """
    clusters = bands.clusters[bands.index(z)]
    candidates = [
        c for c in clusters
        if 0 < _offset(c, side, center_x) < LEG_MAX_OFFSET * bands.height and not _spans(c, center_x)
    ]
    if candidates:
        return _middle(min(candidates, key=lambda c: _offset(c, side, center_x)))
    covering = [c for c in clusters if _spans(c, center_x)]
    if not covering:
        raise LandmarkError(f"no body found on side {side} at z={z:.3f}")
    if warnings is not None:
        warnings.append(f"the legs are joined at {(z - bands.bottom) / bands.height:.0%} of the height: "
                        "leg bones there placed from typical proportions")
    body = max(covering, key=len)
    return np.array([center_x + side * LEG_SPREAD_PRIOR * bands.height, _middle(body)[1]])


def _middle(cluster):
    """The middle of a cluster's span in x and y, which a lopsided sample can't pull aside.

    A mean follows the surface: a robe painted mostly on the front has more sampled points
    there and its mean sits in front of the body it hides.
    """
    return np.array([(np.percentile(cluster[:, k], 98) + np.percentile(cluster[:, k], 2)) / 2 for k in (0, 1)])


def _arm(bands, points, side, center_x):
    """Shoulder, elbow and wrist on one side, from the arm's separated cross-sections."""
    height = bands.height
    centroids, armpit_z, misses = [], None, 0
    for i in range(bands.index(bands.bottom + 0.3 * height), bands.index(bands.bottom + 0.85 * height) + 1):
        clusters = bands.clusters[i]
        outer = max(clusters, key=lambda c: _offset(c, side, center_x)) if clusters else None
        separated = (
            outer is not None
            and not _spans(outer, center_x)
            and _offset(outer, side, center_x) > ARM_MIN_OFFSET * height
            and any(_offset(c, side, center_x) < _offset(outer, side, center_x) for c in clusters if c is not outer)
        )
        if separated:
            centroids.append(outer.mean(axis=0))
            armpit_z, misses = bands.z(i), 0
        elif centroids:
            misses += 1
            if misses > 3:
                break
    if len(centroids) < 3:
        raise LandmarkError(f"no arm found on side {side}")

    points = np.array(centroids)
    if armpit_z < bands.bottom + HANGING_ARM_SEPARATES_BELOW * height:
        tip = points[np.argmin(points[:, 2])]
        shoulder = np.array(
            [center_x + side * SHOULDER_OFFSET * height, tip[1], bands.bottom + SHOULDER_HEIGHT * height]
        )
        return shoulder, shoulder + ELBOW_ALONG_ARM * (tip - shoulder), shoulder + WRIST_ALONG_ARM * (tip - shoulder)

    mean = points.mean(axis=0)
    direction = np.linalg.svd(points - mean)[2][0]
    if np.dot(points[np.argmin(points[:, 2])] - mean, direction) < 0:
        direction = -direction

    if abs(direction[2]) > 0.2:
        shoulder_t = (armpit_z + SHOULDER_ABOVE_ARMPIT * height - mean[2]) / direction[2]
    else:
        shoulder_t = (center_x + side * 0.1 * height - mean[0]) / direction[0]
    shoulder = mean + shoulder_t * direction

    relative = points - mean
    along = relative @ direction
    off_axis = np.linalg.norm(relative - np.outer(along, direction), axis=1)
    beyond_shoulder = side * (points[:, 0] - shoulder[0]) > 0
    on_arm = beyond_shoulder & (off_axis < ARM_AXIS_RADIUS * height)
    if not on_arm.any():
        raise LandmarkError(f"no arm found on side {side}")
    tip = mean + along[on_arm].max() * direction

    elbow = shoulder + ELBOW_ALONG_ARM * (tip - shoulder)
    wrist = shoulder + WRIST_ALONG_ARM * (tip - shoulder)
    return shoulder, elbow, wrist


def _arm_prior(bands, side, center_x, depth):
    """A level arm at typical proportions: shoulder, elbow and wrist."""
    height = bands.height
    shoulder = np.array([center_x + side * SHOULDER_OUT_PRIOR * height, depth, bands.bottom + ARM_PRIOR_HEIGHT * height])
    tip = shoulder + np.array([side * ARM_PRIOR_LENGTH * height, 0.0, 0.0])
    return shoulder, shoulder + ELBOW_ALONG_ARM * (tip - shoulder), shoulder + WRIST_ALONG_ARM * (tip - shoulder)


def _arm_out(bands, points, side, center_x, warnings=None):
    """Shoulder, elbow and wrist of an arm held out to the side, traced across X from the forearm.

    Each slice across the arm is cut into clusters by height and depth, and the one continuing
    the arm is followed out to the fingertips and in towards the shoulder. Only the top of a cross-section is trusted: a hanging sleeve or a mantle
    joins the arm's underside. The shoulder is where the torso's side is, just under the armpit.
    """
    height, bottom = bands.height, bands.bottom
    warnings = warnings if warnings is not None else []
    offset = side * (points[:, 0] - center_x)
    zone = points[(offset > ARM_OUT_INNER * height) & (points[:, 2] > bottom + ARM_OUT_BAND[0] * height)
                  & (points[:, 2] < bottom + ARM_OUT_BAND[1] * height)]
    if len(zone) < 100:
        raise LandmarkError(f"no arm found on side {side}")
    reach = side * (zone[:, 0] - center_x)
    gap, step = CLUSTER_GAP * height, bands.thickness

    def clusters(outer):
        """Clusters of the slice ending `outer` out from the centre, split by height and by depth:
        wings or a tail just behind the arm are in the same slice, never at the same depth."""
        cut = zone[(reach > outer - step) & (reach <= outer)]
        return [part for run in _runs(cut, 2, gap) for part in _runs(run, 1, gap) if len(part) >= 5]

    def follows(cluster, last):
        return (abs(cluster[:, 2].max() - last[:, 2].max()) < ARM_OUT_THICK * height / 2
                and abs(_middle(cluster)[1] - _middle(last)[1]) < ARM_OUT_THICK * height / 2)

    # Start on the forearm, clear of the shoulder and of anything hanging from the upper arm, at
    # the typical arm height; traced inward from the outermost points, the trace began on a wing
    # tip reaching past the fingertips (model C) and followed the wing.
    begin = ARM_OUT_START * height
    found = clusters(begin)
    thin = [c for c in found if np.ptp(c[:, 2]) < ARM_OUT_THICK * height] or found
    if not thin:
        raise LandmarkError(f"no arm found on side {side}")
    first = min(thin, key=lambda c: abs(c[:, 2].mean() - (bottom + SHOULDER_HEIGHT * height)))
    traced = {begin: first}
    for direction, limit in ((1, reach.max() + step), (-1, ARM_OUT_INNER * height + step)):
        last, outer = first, begin + direction * step
        while (outer < limit) if direction > 0 else (outer > limit):
            nxt = [c for c in clusters(outer) if follows(c, last)]
            if not nxt:
                break
            last = min(nxt, key=lambda c: abs(c[:, 2].max() - last[:, 2].max()))
            traced[outer] = last
            outer += direction * step
    # outermost first: the forearm and hand lead, as the fits below expect
    order = sorted(traced, reverse=True)
    slices = [(float(traced[o][:, 0].mean()), float(traced[o][:, 2].min()), float(traced[o][:, 2].max()), traced[o])
              for o in order]
    tip_reach = order[0]
    if len(slices) < 5:
        raise LandmarkError(f"no arm found on side {side}")
    thickness = np.array([top - low for _, low, top, _ in slices])
    thin = thickness < ARM_OUT_THICK * height
    if thin.sum() < 3:
        raise LandmarkError(f"no arm found on side {side}")
    radius = float(np.median(thickness[thin])) / 2
    xs = np.array([x for x, _, _, _ in slices])
    zs = np.array([top for _, _, top, _ in slices]) - radius
    ys = np.array([_middle(c[c[:, 2] > top - 2 * radius])[1] for _, _, top, c in slices])
    # Depth is one value, from the forearm and hand (traced first, nothing hangs over them): an
    # arm held out points straight to the side, and where a mantle or sleeve hangs from the upper
    # arm the cloth spreads front to back (model B's pulled the shoulder 9 cm back).
    outer = thickness[: max(3, len(thickness) // 3)]
    bare = thickness < ARM_BARE * float(np.median(outer))
    depth = float(np.median(ys[bare])) if bare.any() else float(np.median(ys))
    fit = lambda values: np.polyfit(xs, values, 1)  # noqa: E731
    z_line, y_line = fit(zs), np.array([0.0, depth])
    on_line = lambda x: np.array([x, np.polyval(y_line, x), np.polyval(z_line, x)])  # noqa: E731

    armpit_z = float(np.polyval(z_line, center_x + side * ARM_OUT_INNER * height)) - radius
    low, high = (armpit_z - k * height for k in TORSO_UNDER_ARMPIT)
    band = points[(points[:, 2] > low) & (points[:, 2] < high)]
    torso = next((c for c in _split_along_x(band, gap) if _spans(c, center_x)), None)
    if torso is None:
        raise LandmarkError(f"no torso found under the arm on side {side}")
    edge = float(side * np.percentile(side * torso[:, 0], 98))
    shoulder_x = edge - side * SHOULDER_INSET * height
    if not SHOULDER_OUT_RANGE[0] * height <= side * (shoulder_x - center_x) <= SHOULDER_OUT_RANGE[1] * height:
        shoulder_x = center_x + side * SHOULDER_OUT_PRIOR * height
        warnings.append("the shoulders were placed from typical proportions (something hangs under the arms "
                        "or joins them to the body): check them on the Joints page")
    shoulder = on_line(shoulder_x)
    tip = on_line(center_x + side * float(tip_reach))
    elbow = shoulder + ELBOW_ALONG_ARM * (tip - shoulder)
    wrist = shoulder + WRIST_ALONG_ARM * (tip - shoulder)
    return shoulder, elbow, wrist
