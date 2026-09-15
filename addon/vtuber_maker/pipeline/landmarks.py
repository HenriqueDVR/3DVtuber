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
    tip = mean + along[on_arm].max() * direction

    elbow = shoulder + ELBOW_ALONG_ARM * (tip - shoulder)
    wrist = shoulder + WRIST_ALONG_ARM * (tip - shoulder)
    return shoulder, elbow, wrist
