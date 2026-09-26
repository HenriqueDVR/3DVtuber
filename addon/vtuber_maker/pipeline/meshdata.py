"""Mesh data as numpy arrays."""

import numpy as np


def world_vertices(obj):
    coords = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", coords)
    matrix = np.array(obj.matrix_world)
    return coords.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]


def triangles(mesh):
    mesh.calc_loop_triangles()
    indices = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", indices)
    return indices.reshape(-1, 3)


def sample_surface(verts, tris, count, seed=0):
    """Evenly spread random points over the surface, so point density doesn't depend on triangle size."""
    corners = verts[tris]
    edge_a = corners[:, 1] - corners[:, 0]
    edge_b = corners[:, 2] - corners[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(tris), size=count, p=areas / areas.sum())
    u, v = rng.random(count), rng.random(count)
    outside = u + v > 1
    u[outside], v[outside] = 1 - u[outside], 1 - v[outside]
    return corners[chosen, 0] + u[:, None] * edge_a[chosen] + v[:, None] * edge_b[chosen]


def welded_edges(obj, verts, tolerance=1e-5):
    """(weld, a, b): each vertex's representative at its position, and the edges between them.

    Imported meshes split vertices along UV seams, so anything walking the surface has to
    treat vertices sharing a position as one.
    """
    keys = np.round(verts / tolerance).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    weld = first[inverse.ravel()]
    edges = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
    obj.data.edges.foreach_get("vertices", edges)
    return weld, weld[edges[0::2]], weld[edges[1::2]]


def pieces(obj, verts=None):
    """Label each vertex with its separate piece of the mesh (UV seams don't split pieces)."""
    verts = world_vertices(obj) if verts is None else verts
    weld, a, b = welded_edges(obj, verts)
    labels = np.arange(len(verts))
    while True:
        updated = labels.copy()
        np.minimum.at(updated, a, labels[b])
        np.minimum.at(updated, b, labels[a])
        updated = updated[updated]
        if np.array_equal(updated, labels):
            return labels[weld]
        labels = updated
