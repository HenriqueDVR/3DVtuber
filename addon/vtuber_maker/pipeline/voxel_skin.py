"""Skin weights from geodesic distance through the body's volume (geodesic voxel binding).

After Dionne and de Lasa, "Geodesic Voxel Binding for Production Character Meshes" (SCA 2013),
Maya's default bind: the mesh is filled with voxels, each bone's distance to every voxel is
measured through the filled ones only, and a vertex is weighted by how near each bone is that
way. A hand resting on a thigh is near it in space but far from it through the body, so it
doesn't take the thigh. Unlike Blender's bone heat it has no solve that can fail, so no retries.

Used on rig.py's skinning proxy (a voxel remesh, so always closed) in place of bone heat; every
rule rig.skin applies afterwards still runs. NumPy only: the distances spread over the grid as a
wavefront, 26 neighbours per voxel, one bone at a time.
"""

import numpy as np

# Voxels over the character's height. 1.6 m / 200 = 8 mm: finer than the gap between an arm
# at rest and the body on the characters so far, coarse enough to take seconds.
GRID = 200
# Weight falls off as 1 / distance ** FALLOFF.
FALLOFF = 2.0
MAX_INFLUENCES = 4
# Distances below this (m) count as this, so a vertex on the bone isn't infinitely weighted.
NEAREST = 0.005


def _inside(mesh_obj, lower, size, shape):
    """Which cells of the grid lie inside the closed mesh: parity of surface crossings along z."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from . import meshdata
    verts = meshdata.world_vertices(mesh_obj)
    tris = meshdata.triangles(mesh_obj.data)
    tree = BVHTree.FromPolygons([tuple(v) for v in verts], [tuple(t) for t in tris])
    inside = np.zeros(shape, dtype=bool)
    bottom = lower[2] - size
    top_z = lower[2] + (shape[2] + 1) * size
    centres_z = lower[2] + (np.arange(shape[2]) + 0.5) * size
    up = Vector((0.0, 0.0, 1.0))
    for i in range(shape[0]):
        x = lower[0] + (i + 0.5) * size
        for j in range(shape[1]):
            y = lower[1] + (j + 0.5) * size
            hits, z = [], bottom
            while z < top_z:
                found, _, _, _ = tree.ray_cast(Vector((x, y, z)), up, top_z - z)
                if found is None:
                    break
                hits.append(found.z)
                z = found.z + 1e-6
            if len(hits) >= 2:
                crossings = np.searchsorted(np.array(hits), centres_z)
                inside[i, j] = crossings % 2 == 1
    return inside, verts


_STEPS = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)])


def _neighbours(solid):
    """(cells, table, lengths): the solid cells' grid positions, each one's 26 neighbours as
    indices into `cells` (-1 outside the solid) and the length of each step in cells."""
    cells = np.argwhere(solid)
    index = np.full(solid.shape, -1, dtype=np.int64)
    index[tuple(cells.T)] = np.arange(len(cells))
    padded = np.pad(index, 1, constant_values=-1)
    table = np.stack([padded[tuple((cells + 1 + step).T)] for step in _STEPS], axis=1)
    return cells, table, np.linalg.norm(_STEPS, axis=1)


def _segment_distance(points, a, b):
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def _geodesic(table, lengths, start):
    """Distance to every cell through the solid, from `start` (a distance per cell, inf where it
    doesn't start), spread one wavefront at a time."""
    distance = start.copy()
    frontier = np.flatnonzero(np.isfinite(start))
    while len(frontier):
        neighbours = table[frontier]
        reach = distance[frontier, None] + lengths[None, :]
        valid = neighbours >= 0
        targets, values = neighbours[valid], reach[valid]
        best = np.full(len(distance), np.inf)
        np.minimum.at(best, targets, values)
        better = np.flatnonzero(best < distance)
        distance[better] = best[better]
        frontier = better
    return distance


def weigh(proxy, armature_obj):
    """Give the closed mesh `proxy` a vertex group per deforming bone of `armature_obj`, by
    geodesic distance through its volume. Returns the number of vertices weighted."""
    from mathutils.kdtree import KDTree
    bones = [b for b in armature_obj.data.bones if b.use_deform]
    world = np.array(armature_obj.matrix_world)
    segments = [tuple((world @ np.array([*p, 1.0]))[:3] for p in (b.head_local, b.tail_local)) for b in bones]

    from . import meshdata
    verts = meshdata.world_vertices(proxy)
    size = float(np.ptp(verts[:, 2])) / GRID
    lower = verts.min(axis=0) - size
    shape = tuple(np.ceil((verts.max(axis=0) + size - lower) / size).astype(int))
    solid, verts = _inside(proxy, lower, size, shape)
    # the cells the surface passes through count as solid too, so thin parts stay whole
    surface_cells = np.clip(np.floor((verts - lower) / size).astype(int), 0, np.array(shape) - 1)
    solid[tuple(surface_cells.T)] = True
    cells, table, lengths = _neighbours(solid)
    centres = lower + (cells + 0.5) * size

    distances = np.empty((len(cells), len(bones)))
    for k, (head, tail) in enumerate(segments):
        straight = _segment_distance(centres, head, tail)
        # start from the cells nearest the bone: a bone can run outside a thin part
        start = np.where(straight <= straight.min() + size, straight / size, np.inf)
        distances[:, k] = _geodesic(table, lengths, start) * size

    tree = KDTree(len(centres))
    for i, c in enumerate(centres):
        tree.insert(c, i)
    tree.balance()
    nearest = np.array([tree.find(v)[1] for v in verts])
    d = np.maximum(distances[nearest], NEAREST)
    weights = np.where(np.isfinite(d), d ** -FALLOFF, 0.0)
    order = np.argsort(-weights, axis=1)
    keep = np.zeros_like(weights, dtype=bool)
    np.put_along_axis(keep, order[:, :MAX_INFLUENCES], True, axis=1)
    weights = np.where(keep, weights, 0.0)
    total = weights.sum(axis=1, keepdims=True)
    weights = np.where(total > 0, weights / np.maximum(total, 1e-30), 0.0)

    proxy.vertex_groups.clear()
    for k, bone in enumerate(bones):
        group = proxy.vertex_groups.new(name=bone.name)
        for value in np.unique(np.round(weights[:, k], 4)):
            if value <= 0:
                continue
            members = np.flatnonzero(np.round(weights[:, k], 4) == value).tolist()
            group.add(members, float(value), "REPLACE")
    return int((total[:, 0] > 0).sum())
