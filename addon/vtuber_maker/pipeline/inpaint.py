"""Skin weights filled in smoothly where they can't be trusted, from where they can.

A NumPy port of the inpainting in Abdrashitov, Raichstat, Monsen and Hill, "Robust Skin Weights
Transfer via Weight Inpainting" (SIGGRAPH Asia 2023 Technical Communications; MIT reference code
at github.com/rin-23/RobustSkinWeightsTransferCode). The paper copies a body's weights onto a
garment only where the two surfaces match and fills in the rest; here the caller also says which
weights to trust, so the same fill repairs a rig made elsewhere where it is wrong. The fill
minimises trace(Wᵀ Q W), Q = -L + L M⁻¹ L (cotangent Laplacian L, lumped mass M), with the
trusted rows held: a smooth blend over the surface, which never reaches through the air to a
part that merely sits close. Blender's Python has no SciPy, so the system is assembled and
solved here, on the untrusted vertices only.

Differences from the reference code: barycentric instead of Voronoi mass (no visible difference
on the paper's examples, far simpler), and the smoothing afterwards grows by edge rings instead
of by a distance.
"""

import numpy as np

# The paper's match test: the nearest point on the source surface within this share of the
# target's bounding box diagonal, and normals within this angle.
MATCH_DISTANCE = 0.05
MATCH_ANGLE = 30.0
# The paper's smoothing of the filled region and the rings around it, which softens the crease
# where filled weights meet held ones.
SMOOTH_STEPS = 10
SMOOTH_ALPHA = 0.2
SMOOTH_RINGS = 2
# Weights below this after the fill are dropped: the fill spreads faint traces of every bone
# that touches the region's edge.
FLOOR = 1e-3
# Up to this many unknowns the system is solved directly; past it by conjugate gradients.
DENSE = 3000
CG_TOLERANCE = 1e-10
# Triangles with a cotangent past this (an angle under ~0.06 deg) are left out: slivers give
# near-infinite entries that wreck the solve.
COT_LIMIT = 1e3


def weld(verts, tris, tolerance=1e-5, apart=None):
    """(index, verts, tris) with vertices sharing a position merged: `index` maps each original
    vertex to its merged one. Imported meshes split vertices along UV seams, which would cut the
    surface the fill walks over. Vertices in the mask `apart` only merge among themselves."""
    keys = np.round(np.asarray(verts) / tolerance).astype(np.int64)
    if apart is not None:
        keys = np.column_stack([keys, np.asarray(apart, dtype=np.int64)])
    _, first, index = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    index = index.ravel()
    t = index[np.asarray(tris)]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    return index, np.asarray(verts, dtype=float)[first], t


def vertex_normals(verts, tris):
    """Area-weighted unit normals over the welded surface, so a UV seam doesn't give its two
    copies of a vertex different normals."""
    index, wv, wt = weld(verts, tris)
    corners = wv[wt]
    faces = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(wv)
    for k in range(3):
        np.add.at(normals, wt[:, k], faces)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    return normals[index]


def matched(verts, normals, found, found_normals, diagonal):
    """The paper's test of a vertex against its nearest point on the source surface: within
    MATCH_DISTANCE of the target's bounding box `diagonal`, and normals within MATCH_ANGLE."""
    near = np.linalg.norm(found - verts, axis=1) <= MATCH_DISTANCE * diagonal
    cos = (normals * found_normals).sum(axis=1)
    return near & (cos >= np.cos(np.radians(MATCH_ANGLE)))


def laplacian(verts, tris):
    """(rows, cols, values) of the cotangent Laplacian, diagonal included, duplicates summed, sorted
    by row; and each vertex's lumped mass (a third of the area of its triangles)."""
    n = len(verts)
    corners = verts[tris]
    double_area = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    cots = np.stack([((corners[:, (k + 1) % 3] - corners[:, k]) * (corners[:, (k + 2) % 3] - corners[:, k])).sum(axis=1)
                     for k in range(3)], axis=1) / np.maximum(double_area, 1e-30)[:, None]
    # slivers are dropped, not clipped: clipping one angle's cotangent makes the triangle's
    # energy indefinite and the solve diverges
    live = (double_area > 0) & (np.abs(cots) <= COT_LIMIT).all(axis=1)
    t, area, cots = tris[live], double_area[live], cots[live]
    rows, cols, vals = [], [], []
    for k in range(3):
        # the angle at corner k sits opposite the edge between the other two; the reference
        # code's L is twice igl's cotmatrix, so the full cotangent
        b, c = t[:, (k + 1) % 3], t[:, (k + 2) % 3]
        rows += [b, c]
        cols += [c, b]
        vals += [cots[:, k], cots[:, k]]
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    diagonal = -np.bincount(rows, weights=vals, minlength=n)
    rows = np.concatenate([rows, np.arange(n)])
    cols = np.concatenate([cols, np.arange(n)])
    vals = np.concatenate([vals, diagonal])
    key, inverse = np.unique(rows * n + cols, return_inverse=True)
    vals = np.bincount(inverse.ravel(), weights=vals)
    mass = np.bincount(t.ravel(), weights=np.repeat(area / 6.0, 3), minlength=n)
    return key // n, key % n, vals, mass


def fill(verts, tris, weights, known, skip=None):
    """`weights` (vertices, bones) with every row outside the mask `known` replaced by the fill.

    Vertices in the mask `skip` are cut out of the surface first: kept as they are, neither
    filled nor filled from (hair lying on a shoulder mustn't lend the shoulder its weights).
    Returns (weights, unresolved): a vertex whose stretch of untrusted surface touches no trusted
    vertex has nothing to be filled from and keeps its row, marked in `unresolved`. Filled rows
    are clipped to [0, 1], stripped below FLOOR and sum to one.
    """
    weights = np.asarray(weights, dtype=float)
    known = np.asarray(known, dtype=bool)
    skip = np.zeros(len(known), dtype=bool) if skip is None else np.asarray(skip, dtype=bool)
    tris = np.asarray(tris)
    # Skipped vertices never merge with kept ones: where a hair region was cut from the body it
    # sits on, each seam position has a hair copy and a body copy, and merging them would cut
    # the body's copy out too, leaving the rig's weights there (model C's twin-tail seam kept
    # RightUpperArm 0.68 and both thighs).
    index, wv, wt = weld(verts, tris[~skip[tris].any(axis=1)], apart=skip)
    count = len(wv)
    # a merged vertex is trusted only if every copy is
    wknown = np.ones(count, dtype=bool)
    np.logical_and.at(wknown, index, known)
    wskip = np.zeros(count, dtype=bool)
    wskip[index[skip]] = True
    wweights = np.zeros((count, weights.shape[1]))
    wweights[index[known]] = weights[known]

    rows, cols, vals, mass = laplacian(wv, wt)
    off = rows != cols
    free = ~wknown & ~wskip
    unknown = np.flatnonzero(free)
    reached = _reaching(count, rows[off], cols[off], free)
    solve = unknown[reached[unknown]]
    result = wweights.copy()
    if len(solve):
        result[solve] = _solve(count, rows, cols, vals, mass, solve, wweights)
        result = _smooth(result, rows[off], cols[off], solve, wknown)
        filled = np.clip(result[solve], 0.0, 1.0)
        filled[filled < FLOOR] = 0.0
        total = filled.sum(axis=1, keepdims=True)
        result[solve] = np.where(total > 0, filled / np.maximum(total, 1e-12), 0.0)
    done = np.zeros(count, dtype=bool)
    done[solve] = True
    out = weights.copy()
    changed = ~known & ~skip & done[index]
    out[changed] = result[index[changed]]
    return out, ~known & ~skip & ~done[index]


def _reaching(count, a, b, free):
    """Which vertices of `free` are joined, over free vertices, to one that isn't free."""
    labels = np.arange(count)
    ea, eb = a[free[a] & free[b]], b[free[a] & free[b]]
    while True:
        updated = labels.copy()
        np.minimum.at(updated, ea, labels[eb])
        np.minimum.at(updated, eb, labels[ea])
        updated = updated[updated]
        if np.array_equal(updated, labels):
            break
        labels = updated
    touching = np.zeros(count, dtype=bool)
    edge = free[a] & ~free[b]
    touching[labels[a[edge]]] = True
    return touching[labels] & free


def _solve(count, rows, cols, vals, mass, solve, weights):
    """Rows `solve` of the minimiser of trace(Wᵀ Q W) with every other row of `weights` held."""
    local = np.full(count, -1)
    local[solve] = np.arange(len(solve))
    inv_mass = 1.0 / np.maximum(mass, 1e-18)
    # Q's rows for the unknowns: -L, plus L M⁻¹ L as, for each vertex k, every pair of an unknown
    # neighbour i and any neighbour j of k (L is symmetric: L_ik = L_ki)
    mine = local[rows] >= 0
    left = local[cols] >= 0
    lk, li, lv = rows[left], local[cols[left]], vals[left]
    starts = np.searchsorted(rows, np.arange(count + 1))
    width = (starts[1:] - starts[:-1])[lk]
    first = np.repeat(np.arange(len(lk)), width)
    offset = np.arange(len(first)) - np.repeat(np.cumsum(width) - width, width)
    right = starts[lk[first]] + offset
    q_rows = np.concatenate([local[rows[mine]], li[first]])
    q_cols = np.concatenate([cols[mine], cols[right]])
    q_vals = np.concatenate([-vals[mine], lv[first] * vals[right] * inv_mass[lk[first]]])

    inner = local[q_cols] >= 0
    rhs = np.zeros((len(solve), weights.shape[1]))
    np.add.at(rhs, q_rows[~inner], -q_vals[~inner, None] * weights[q_cols[~inner]])
    ar, ac, av = q_rows[inner], local[q_cols[inner]], q_vals[inner]
    m = len(solve)
    if m <= DENSE:
        matrix = np.zeros((m, m))
        np.add.at(matrix, (ar, ac), av)
        return np.linalg.solve(matrix, rhs)
    key, inverse = np.unique(ar * m + ac, return_inverse=True)
    av = np.bincount(inverse.ravel(), weights=av)
    ar, ac = key // m, key % m
    return _conjugate_gradient(ar, ac, av, m, rhs)


def _conjugate_gradient(rows, cols, vals, m, rhs):
    """Solve the symmetric positive definite system (rows, cols, vals sorted by row, every row
    present: each holds its diagonal) for every column of `rhs` at once, Jacobi preconditioned."""
    starts = np.searchsorted(rows, np.arange(m))
    diagonal = np.zeros(m)
    on = rows == cols
    diagonal[rows[on]] = vals[on]

    def apply(x):
        return np.add.reduceat(vals[:, None] * x[cols], starts, axis=0)

    x = np.zeros_like(rhs)
    r = rhs.copy()
    z = r / diagonal[:, None]
    p = z.copy()
    rz = (r * z).sum(axis=0)
    goal = CG_TOLERANCE * np.maximum((rhs * rhs).sum(axis=0), 1e-30)
    for _ in range(20 * m):
        if ((r * r).sum(axis=0) <= goal).all():
            break
        ap = apply(p)
        step = rz / np.where((p * ap).sum(axis=0) > 0, (p * ap).sum(axis=0), 1.0)
        x += step * p
        r -= step * ap
        z = r / diagonal[:, None]
        rz_next = (r * z).sum(axis=0)
        p = z + (rz_next / np.where(rz > 0, rz, 1.0)) * p
        rz = rz_next
    return x


def _smooth(weights, a, b, solve, known):
    """The reference code's smoothing: the filled vertices and SMOOTH_RINGS rings of held ones
    around them each move SMOOTH_ALPHA toward their neighbours' mean, SMOOTH_STEPS times. Unlike
    the reference, only the filled rows are kept: held weights stay exactly as they were."""
    region = np.zeros(len(weights), dtype=bool)
    region[solve] = True
    for _ in range(SMOOTH_RINGS):
        grown = region.copy()
        grown[a[region[b]]] = True
        grown[b[region[a]]] = True
        region = grown
    degree = np.bincount(a, minlength=len(weights)).astype(float)
    moved = region & (degree > 0)
    for _ in range(SMOOTH_STEPS):
        mean = np.zeros_like(weights)
        np.add.at(mean, a, weights[b])
        mean /= np.maximum(degree, 1.0)[:, None]
        weights = np.where(moved[:, None], (1 - SMOOTH_ALPHA) * weights + SMOOTH_ALPHA * mean, weights)
    return weights


def fill_mesh(mesh_obj, armature_obj, trusted, skip=None):
    """Refill the bone weights of `mesh_obj`'s vertices outside the mask `trusted` from the rest
    (see `fill`, also for `skip`).

    Only groups named after bones of `armature_obj` take part. Returns (filled, unresolved)
    vertex counts; unresolved vertices keep their weights.
    """
    from . import meshdata
    from .loose import _weight_table, _write_weights
    count = len(mesh_obj.data.vertices)
    skip = np.zeros(count, dtype=bool) if skip is None else np.asarray(skip, dtype=bool)
    table = _weight_table(mesh_obj, count)
    bones = [g.index for g in mesh_obj.vertex_groups if g.name in armature_obj.data.bones]
    weights, unresolved = fill(meshdata.world_vertices(mesh_obj), meshdata.triangles(mesh_obj.data),
                               table[bones].T, trusted, skip)
    table[bones] = weights.T
    touched = np.flatnonzero(~np.asarray(trusted, dtype=bool) & ~skip & ~unresolved).tolist()
    _write_weights(mesh_obj, table, touched)
    return len(touched), int(unresolved.sum())
