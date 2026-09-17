"""Eye and mouth points placed by hand in viewer/setup.html, applied to the face features.

The points are pixels in the face patch image (see scripts/retexture/face_patch.py). That
patch is mapped straight from the front, so its UVs are affine in x and z and a pixel
converts exactly to a point on the face; depth comes from the surface there.
"""

import json

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

FACE_PATCH_MATERIAL = "face_patch"
# Surface hits further than this in front of the feature found without points are hair
# strands crossing the face; the feature's own depth is kept then.
HAIR_IN_FRONT = 0.015


class FacePointsError(ValueError):
    """The points can't be placed on this mesh."""


def load(path):
    with open(path) as f:
        points = json.load(f)
    for key, names in (("right_eye", ("inner", "outer", "top", "bottom")),
                       ("left_eye", ("inner", "outer", "top", "bottom")),
                       ("mouth", ("left", "right", "top", "bottom"))):
        missing = [n for n in names if n not in points.get(key, {})]
        if missing:
            raise FacePointsError(f"{path}: {key} is missing {', '.join(missing)}")
    if not points.get("size"):
        raise FacePointsError(f"{path}: no image size")
    return points


def apply(mesh_obj, features, points):
    """Return features with eyes and mouth taken from the points (world space, like the rest)."""
    to_world = _pixel_to_world(mesh_obj, points["size"])
    surface = BVHTree.FromObject(mesh_obj, bpy.context.evaluated_depsgraph_get())
    matrix = mesh_obj.matrix_world
    inverse = matrix.inverted()
    front = -1.0  # the character faces -Y

    def on_surface(x, z, fallback):
        origin = inverse @ Vector((x, front * 10.0, z))
        direction = (inverse.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()
        hit, *_ = surface.ray_cast(origin, direction)
        y = (matrix @ hit).y if hit is not None else None
        if fallback is not None and (y is None or y < fallback - HAIR_IN_FRONT):
            y = fallback
        return np.array([x, y if y is not None else 0.0, z])

    features = dict(features)
    sizes = []
    for key in ("left_eye", "right_eye"):
        p = {name: to_world(xy) for name, xy in points[key].items()}
        x = (p["inner"][0] + p["outer"][0]) / 2
        z = (p["top"][1] + p["bottom"][1]) / 2
        old = features.get(key)
        features[key] = on_surface(x, z, None if old is None else float(old[1]))
        sizes.append([abs(p["outer"][0] - p["inner"][0]), abs(p["top"][1] - p["bottom"][1])])
    features["eye_size"] = np.mean(sizes, axis=0)

    m = {name: to_world(xy) for name, xy in points["mouth"].items()}
    old = features.get("mouth")
    features["mouth"] = on_surface((m["left"][0] + m["right"][0]) / 2, (m["top"][1] + m["bottom"][1]) / 2,
                                   None if old is None else float(old[1]))
    features["mouth_width"] = float(abs(m["right"][0] - m["left"][0]))
    features["mouth_estimated"] = False
    return features


def _pixel_to_world(mesh_obj, size):
    """(pixel x, pixel y) in the face patch image -> (world x, world z)."""
    mesh = mesh_obj.data
    slot = mesh.materials.find(FACE_PATCH_MATERIAL)
    if slot < 0:
        raise FacePointsError("face points need a face patch: run scripts/retexture/face_patch.py on the model first")
    uv = mesh.uv_layers.active.data
    matrix = np.array(mesh_obj.matrix_world)
    xs, zs, us, vs = [], [], [], []
    for poly in mesh.polygons:
        if poly.material_index != slot:
            continue
        for li in poly.loop_indices:
            co = np.array(mesh.vertices[mesh.loops[li].vertex_index].co)
            world = matrix[:3, :3] @ co + matrix[:3, 3]
            xs.append(world[0])
            zs.append(world[2])
            us.append(uv[li].uv.x)
            vs.append(uv[li].uv.y)
    if len(xs) < 3:
        raise FacePointsError("the face patch has no polygons")
    ones = np.ones(len(xs))
    au = np.linalg.lstsq(np.stack([xs, ones], 1), us, rcond=None)[0]
    av = np.linalg.lstsq(np.stack([zs, ones], 1), vs, rcond=None)[0]
    width, height = size

    def to_world(pixel):
        u, v = pixel[0] / width, 1.0 - pixel[1] / height
        return (u - au[1]) / au[0], (v - av[1]) / av[0]

    return to_world
