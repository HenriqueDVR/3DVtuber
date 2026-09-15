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
