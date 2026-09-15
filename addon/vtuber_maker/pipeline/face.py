"""Find the painted-on eyes and mouth of a generated character.

Image-to-3D generators bake the face into the base-color texture instead of
modelling it. The front of the head is projected into a 2D grid, paint is
whatever differs from both skin and hair color, and the largest symmetric pair
of paint blobs is taken as the eyes.
"""

from collections import deque

import numpy as np

from . import texture_space

# sRGB distance (0..1 per channel) that separates paint from skin or hair.
SKIN_DISTANCE = 0.16
HAIR_DISTANCE = 0.12
MIN_BLOB_CELLS = 6
# Grid cells are this many texel spacings wide, so few cells end up empty.
CELL_TEXELS = 1.5


class FaceNotFound(RuntimeError):
    """No eyes could be located on the head's texture."""


def detect_features(obj, joints):
    """Locate eyes and mouth. Points are world space, sizes in metres.

    Keys: left_eye, right_eye, eye_size (width, height), mouth, mouth_width,
    mouth_estimated, skin_color, hair_color. Character left is +X.
    """
    image = texture_space.base_color_image(obj)
    width, height = image.size
    positions, covered = texture_space.bake_position_map(obj, width, height)
    matrix = np.array(obj.matrix_world)
    points = positions[covered] @ matrix[:3, :3].T + matrix[:3, 3]
    colors = texture_space.image_rgb(image)[covered]

    grid = _FrontGrid(points, colors, joints)
    skin = _dominant_color(grid.colors[grid.lower_face()])
    hair = _dominant_color(grid.colors[grid.crown()])
    paint = (
        grid.filled
        & (np.linalg.norm(grid.colors - skin, axis=2) > SKIN_DISTANCE)
        & (np.linalg.norm(grid.colors - hair, axis=2) > HAIR_DISTANCE)
        & grid.between(skin)
    )
    blobs = [blob for blob in _blobs(paint) if len(blob) >= MIN_BLOB_CELLS]
    left, right = _eye_pair(blobs, grid)

    features = {
        "left_eye": grid.point(left),
        "right_eye": grid.point(right),
        "eye_size": (grid.size(left) + grid.size(right)) / 2,
        "skin_color": skin,
        "hair_color": hair,
    }
    mouth = _mouth(blobs, left, right)
    if mouth is not None:
        features.update(mouth=grid.point(mouth), mouth_width=float(grid.size(mouth)[0]), mouth_estimated=False)
    else:
        features.update(_estimated_mouth(grid, left, right), mouth_estimated=True)
    return features


class _FrontGrid:
    """Color and position of the frontmost texel in each cell, looking at the face along +Y."""

    def __init__(self, points, colors, joints):
        in_head = points[:, 2] > joints["head"][2]
        if not in_head.any():
            raise FaceNotFound("no texture above the head joint")
        front = in_head & (points[:, 1] < points[in_head, 1].mean())
        front_points, front_colors = points[front], colors[front]

        self.center_x = float(joints["head"][0])
        self.bottom = float(joints["head"][2])
        self.top = float(front_points[:, 2].max())
        self.left_x = float(front_points[:, 0].min())
        x_span = float(front_points[:, 0].max()) - self.left_x
        z_span = self.top - self.bottom
        self.cell = CELL_TEXELS * np.sqrt(x_span * z_span / len(front_points))
        shape = (int(z_span / self.cell) + 1, int(x_span / self.cell) + 1)

        rows = np.clip(((front_points[:, 2] - self.bottom) / self.cell).astype(int), 0, shape[0] - 1)
        columns = np.clip(((front_points[:, 0] - self.left_x) / self.cell).astype(int), 0, shape[1] - 1)
        order = np.argsort(front_points[:, 1])
        _, first = np.unique(rows[order] * shape[1] + columns[order], return_index=True)
        pick = order[first]

        self.colors = np.zeros(shape + (3,))
        self.points = np.zeros(shape + (3,))
        self.filled = np.zeros(shape, dtype=bool)
        self.colors[rows[pick], columns[pick]] = front_colors[pick]
        self.points[rows[pick], columns[pick]] = front_points[pick]
        self.filled[rows[pick], columns[pick]] = True

    def column_of(self, x):
        return (x - self.left_x) / self.cell

    def lower_face(self):
        rows, columns = np.indices(self.filled.shape)
        half_width = 0.15 * self.filled.shape[1]
        return (
            self.filled
            & (rows < 0.35 * self.filled.shape[0])
            & (np.abs(columns - self.column_of(self.center_x)) < half_width)
        )

    def crown(self):
        rows = np.indices(self.filled.shape)[0]
        return self.filled & (rows > 0.85 * self.filled.shape[0])

    def between(self, skin):
        """Cells lying between the leftmost and rightmost skin cell of their row."""
        is_skin = self.filled & (np.linalg.norm(self.colors - skin, axis=2) <= SKIN_DISTANCE)
        mask = np.zeros_like(is_skin)
        for row in np.nonzero(is_skin.any(axis=1))[0]:
            columns = np.nonzero(is_skin[row])[0]
            mask[row, columns.min() : columns.max() + 1] = True
        return mask

    def point(self, blob):
        return self.points[blob[:, 0], blob[:, 1]].mean(axis=0)

    def size(self, blob):
        """(width, height) in metres."""
        return np.array([np.ptp(blob[:, 1]) + 1, np.ptp(blob[:, 0]) + 1]) * self.cell


def _dominant_color(colors):
    levels = np.round(colors * 15).astype(int)
    keys = (levels[:, 0] * 16 + levels[:, 1]) * 16 + levels[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    return colors[keys == values[np.argmax(counts)]].mean(axis=0)


def _blobs(mask):
    """8-connected groups of True cells, each an (n, 2) array of (row, column)."""
    seen = np.zeros_like(mask)
    height, width = mask.shape
    blobs = []
    for start in zip(*np.nonzero(mask)):
        if seen[start]:
            continue
        seen[start] = True
        queue, cells = deque([start]), []
        while queue:
            row, column = queue.popleft()
            cells.append((row, column))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    r, c = row + dr, column + dc
                    if 0 <= r < height and 0 <= c < width and mask[r, c] and not seen[r, c]:
                        seen[r, c] = True
                        queue.append((r, c))
        blobs.append(np.array(cells))
    return blobs


def _eye_pair(blobs, grid):
    """Largest pair of blobs mirrored across the head's centre line; returns (left, right)."""
    center = grid.column_of(grid.center_x)
    best, best_area = None, 0
    for i, a in enumerate(blobs):
        for b in blobs[i + 1 :]:
            (a_row, a_column), (b_row, b_column) = a.mean(axis=0), b.mean(axis=0)
            spread = abs(a_column - b_column)
            tallest = max(np.ptp(a[:, 0]), np.ptp(b[:, 0])) + 1
            mirrored = abs((a_column + b_column) / 2 - center) < 0.25 * spread
            level = abs(a_row - b_row) < 0.5 * tallest
            alike = max(len(a), len(b)) < 3 * min(len(a), len(b))
            area = len(a) + len(b)
            if spread > tallest and mirrored and level and alike and area > best_area:
                best, best_area = (a, b), area
    if best is None:
        raise FaceNotFound("no symmetric pair of eyes on the face texture")
    a, b = best
    return (a, b) if a[:, 1].mean() > b[:, 1].mean() else (b, a)


def _mouth(blobs, left, right):
    """Largest blob centred below the eyes, or None for faces without a painted mouth."""
    eye_row = (left[:, 0].mean() + right[:, 0].mean()) / 2
    eye_height = max(np.ptp(left[:, 0]), np.ptp(right[:, 0])) + 1
    center = (left[:, 1].mean() + right[:, 1].mean()) / 2
    spread = abs(left[:, 1].mean() - right[:, 1].mean())
    below = [
        blob
        for blob in blobs
        if blob is not left
        and blob is not right
        and abs(blob[:, 1].mean() - center) < 0.3 * spread
        and blob[:, 0].mean() < eye_row - eye_height
    ]
    return max(below, key=len) if below else None


def _estimated_mouth(grid, left, right):
    """Typical anime placement: centred, about one eye-spacing below the eyes."""
    spread = abs(left[:, 1].mean() - right[:, 1].mean())
    row = (left[:, 0].mean() + right[:, 0].mean()) / 2 - 0.92 * spread
    column = (left[:, 1].mean() + right[:, 1].mean()) / 2
    filled = np.argwhere(grid.filled)
    nearest = filled[np.argmin(np.hypot(filled[:, 0] - row, filled[:, 1] - column))]
    return {"mouth": grid.points[nearest[0], nearest[1]], "mouth_width": 0.25 * spread * grid.cell}
