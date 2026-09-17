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
ARM_MIN_OFFSET = 0.12
ARM_AXIS_RADIUS = 0.06
# An arm that stops standing apart below this height is hanging into the body or hair,
# so only the hand is visible and the shoulder falls back to typical proportions.
HANGING_ARM_SEPARATES_BELOW = 0.62
SHOULDER_HEIGHT = 0.78
SHOULDER_OFFSET = 0.1

# Fractions along shoulder -> fingertip.
ELBOW_ALONG_ARM = 0.43
WRIST_ALONG_ARM = 0.79

# Fraction of neck base -> top of head where the head bone starts.
HEAD_ALONG_NECK_TO_TOP = 0.24


class LandmarkError(RuntimeError):
    """The mesh doesn't look like a standing A-pose humanoid."""


def detect_body(verts, tris):
    """Return VRM humanoid joint positions keyed like "left_upper_arm".

    Character left is +X. Each value is an (x, y, z) array in the input's space.
    """
    points = sample_surface(np.asarray(verts, dtype=float), np.asarray(tris), SAMPLE_COUNT)
    bands = _Bands(points)
    center_x = float(np.median(points[:, 0]))
    height = bands.height

    crotch_z = _crotch_height(bands, center_x)
    hip_z = crotch_z + HIP_JOINT_ABOVE_CROTCH * height
    ankle_z = bands.bottom + ANKLE_HEIGHT * height
    knee_z = (hip_z + ankle_z) / 2

    joints = {}
    for side, name in ((1, "left"), (-1, "right")):
        thigh = _leg_centroid(bands, crotch_z - 0.03 * height, side, center_x)
        joints[f"{name}_upper_leg"] = np.array([thigh[0], thigh[1], hip_z])
        knee = _leg_centroid(bands, knee_z, side, center_x)
        joints[f"{name}_lower_leg"] = np.array([knee[0], knee[1], knee_z])
        ankle = _leg_centroid(bands, ankle_z, side, center_x)
        joints[f"{name}_foot"] = np.array([ankle[0], ankle[1], ankle_z])

    body_y = (joints["left_upper_leg"][1] + joints["right_upper_leg"][1]) / 2
    joints["hips"] = np.array([center_x, body_y, hip_z + HIPS_ABOVE_HIP_JOINTS * height])

    shoulder_heights = []
    for side, name in ((1, "left"), (-1, "right")):
        shoulder, elbow, wrist = _arm(bands, points, side, center_x)
        joints[f"{name}_upper_arm"] = shoulder
        joints[f"{name}_lower_arm"] = elbow
        joints[f"{name}_hand"] = wrist
        shoulder_heights.append(shoulder[2])

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
        elbow_edge = _outer_edge(points, new_elbow, side)
        probe = shoulder + 0.4 * (elbow - shoulder)
        probe_edge = _outer_edge(points, np.array([probe[0], new_elbow[1], probe[2]]), side)
        if elbow_edge is not None and probe_edge is not None:
            radius = abs(elbow_edge - new_elbow[0])
            shift_x = (probe_edge - side * radius) - probe[0]
            if abs(shift_x) < LIMB_MAX_SHIFT:
                new_shoulder[0] += shift_x
                # the joint is the middle of the shoulder's ball, an arm radius under its top
                new_shoulder[2] -= min(radius, LIMB_MAX_SHIFT)
        new_shoulder[1] = new_elbow[1]
        joints[f"{name}_upper_arm"], joints[f"{name}_lower_arm"], joints[f"{name}_hand"] = new_shoulder, new_elbow, new_wrist

        hip, knee, ankle = (joints[f"{name}_{k}"] for k in ("upper_leg", "lower_leg", "foot"))
        joints[f"{name}_lower_leg"] = _limb_centre(points, knee, ankle - hip)
        joints[f"{name}_foot"] = _limb_centre(points, ankle, ankle - hip)
        probe = hip + 0.2 * (knee - hip)
        joints[f"{name}_upper_leg"] = hip + (_limb_centre(points, probe, knee - hip) - probe)

    left, right = joints["left_upper_leg"], joints["right_upper_leg"]
    spread = (abs(left[0] - centre_x) + abs(right[0] - centre_x)) / 2
    y, z = (left[1] + right[1]) / 2, (left[2] + right[2]) / 2
    joints["left_upper_leg"] = np.array([centre_x + spread, y, z])
    joints["right_upper_leg"] = np.array([centre_x - spread, y, z])
    return joints


def _outer_edge(points, at, side, reach=0.12, band=0.02):
    """x of the body's outermost surface on this side, in a small horizontal band around `at`."""
    pick = (np.abs(points[:, 2] - at[2]) < band / 2) & (np.abs(points[:, 1] - at[1]) < band)         & (side * (points[:, 0] - at[0]) > -reach) & (side * (points[:, 0] - at[0]) < reach)
    if pick.sum() < 5:
        return None
    return float(side * np.percentile(side * points[pick, 0], 98))


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


def _leg_centroid(bands, z, side, center_x):
    """XY centre of the leg on one side at height z (the innermost cluster; hands sit further out)."""
    candidates = [
        c for c in bands.clusters[bands.index(z)] if _offset(c, side, center_x) > 0 and not _spans(c, center_x)
    ]
    if not candidates:
        raise LandmarkError(f"no leg found on side {side} at z={z:.3f}")
    leg = min(candidates, key=lambda c: _offset(c, side, center_x))
    return leg[:, :2].mean(axis=0)


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
