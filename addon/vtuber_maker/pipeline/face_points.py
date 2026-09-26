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


# The face's skin colour is read from the face patch in a ring around the mouth, between these
# distances (eye widths): clear of the mouth, short of the eyes.
SKIN_BAND = (0.4, 0.7)
# The narrowest mouth, as a fraction of an eye's width.
MIN_MOUTH_WIDTH = 0.5
# Painting the mouth out covers the marked box grown by ERASE_GROW plus ERASE_MARGIN pixels
# all round (drawn lines spill past the points), filled over ERASE_PASSES smoothing passes.
ERASE_GROW = 1.4
ERASE_MARGIN = 6
ERASE_PASSES = 400


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
    # A mouth marked tiny (a dot, or no mouth drawn at all) opens as a jagged speck on a
    # low-poly face; mouth shapes need room, so it is at least MIN_MOUTH_WIDTH eye widths wide.
    features["mouth_width"] = max(float(abs(m["right"][0] - m["left"][0])),
                                  MIN_MOUTH_WIDTH * float(features["eye_size"][0]))
    features["mouth_estimated"] = False
    skin = _patch_skin(mesh_obj, points)
    if skin is not None:
        features["skin_color"] = skin
    return features


def erase_mouth(mesh_obj, points):
    """Paint the marked mouth out of the face patch image, filling it from the skin around it.

    For a mouth that is its own shape on the face (see face_rig, `free_mouth`): the painted
    one would stay put while that shape slides and curls. Returns whether anything was painted.
    """
    image = _patch_image(mesh_obj)
    if image is None:
        return False
    width, height = image.size
    pixels = np.array(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    scale_x, scale_y = width / points["size"][0], height / points["size"][1]
    m = points["mouth"]
    xs = np.array([m[k][0] for k in m]) * scale_x
    ys = height - np.array([m[k][1] for k in m]) * scale_y  # image pixels count up from the bottom
    cx, cy = xs.mean(), ys.mean()
    half_w = (np.ptp(xs) / 2) * ERASE_GROW + ERASE_MARGIN
    half_h = (np.ptp(ys) / 2) * ERASE_GROW + ERASE_MARGIN
    x0, x1 = int(max(cx - half_w - 3, 0)), int(min(cx + half_w + 4, width))
    y0, y1 = int(max(cy - half_h - 3, 0)), int(min(cy + half_h + 4, height))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    hole = ((xx - cx) / half_w) ** 2 + ((yy - cy) / half_h) ** 2 < 1.0
    pixels[y0:y1, x0:x1, :3] = fill_from_border(pixels[y0:y1, x0:x1, :3], hole)
    image.pixels.foreach_set(pixels.ravel())
    image.update()
    image.pack()
    return True


def fill_from_border(rgb, hole, passes=ERASE_PASSES, domain=None):
    """`rgb` (rows, columns, 3) with `hole` filled smoothly from the pixels around it.

    Repeated averaging of the four neighbours. With `domain`, only pixels in it (and the hole)
    pass colour on, so a fill can draw on one colour next to it (an eye's white) and not another
    (the lashes above it).
    """
    rgb = np.array(rgb, dtype=np.float32)
    known = ~hole if domain is None else ~hole & domain
    if not known.any():
        return rgb
    rgb[hole] = rgb[known].mean(axis=0)
    if domain is None:
        for _ in range(passes):
            blurred = (np.roll(rgb, 1, 0) + np.roll(rgb, -1, 0) + np.roll(rgb, 1, 1) + np.roll(rgb, -1, 1)) / 4
            rgb[hole] = blurred[hole]
        return rgb
    weight = (known | hole).astype(np.float32)[..., None]
    for _ in range(passes):
        weighted = rgb * weight
        total = sum(np.roll(weighted, s, a) for a in (0, 1) for s in (1, -1))
        count = sum(np.roll(weight, s, a) for a in (0, 1) for s in (1, -1))
        blurred = total / np.maximum(count, 1e-6)
        rgb[hole] = blurred[hole]
    return rgb


def fill_across(values, hole, domain=None, passes=60):
    """`values` (rows, columns, channels) with `hole` filled smoothly from what surrounds it, all the way in.

    fill_from_border's averaging only carries colour ~sqrt(passes) pixels into a hole; a 200 px
    erased eye kept a flat mean in the middle, so the blush painted under model B's eyes stopped at
    the hole's edge and showed as pale blobs when the eye squeezed shut. Solved coarse first
    (halving until the hole is small), each level starting from the one below.
    """
    values = np.array(values, dtype=np.float32)
    known = ~hole if domain is None else ~hole & domain
    if not known.any():
        return values
    if min(hole.shape) >= 24:
        # a coarse cell is hole if any of its four pixels is, so every hole pixel starts from a
        # filled cell (sampling every other pixel let one start from a stroke left out of `domain`)
        rows, cols = hole.shape
        padded = np.pad(hole, ((0, rows % 2), (0, cols % 2)))
        coarse_hole = padded.reshape(padded.shape[0] // 2, 2, padded.shape[1] // 2, 2).any(axis=(1, 3))
        coarse = fill_across(values[::2, ::2], coarse_hole, None if domain is None else domain[::2, ::2], passes)
        values[hole] = np.repeat(np.repeat(coarse, 2, 0), 2, 1)[:rows, :cols][hole]
    else:
        values[hole] = values[known].mean(axis=0)
    weight = (known | hole).astype(np.float32)[..., None]
    for _ in range(passes):
        weighted = values * weight
        total = sum(np.roll(weighted, s, a) for a in (0, 1) for s in (1, -1))
        count = sum(np.roll(weight, s, a) for a in (0, 1) for s in (1, -1))
        values[hole] = (total / np.maximum(count, 1e-6))[hole]
    return values


def _patch_image(mesh_obj):
    slot = mesh_obj.data.materials.find(FACE_PATCH_MATERIAL)
    if slot < 0:
        return None
    material = mesh_obj.data.materials[slot]
    if material is None or material.node_tree is None:
        return None
    image = next((n.image for n in material.node_tree.nodes if n.type == "TEX_IMAGE" and n.image), None)
    return image if image is not None and image.size[0] else None


def _patch_skin(mesh_obj, points):
    """Median colour of the face patch image in a band around the marked mouth, or None.

    Parts sample skin through the model's original UVs, which on face-patch polygons point
    into the patch image instead, so they pick up whatever the atlas has there.
    """
    slot = mesh_obj.data.materials.find(FACE_PATCH_MATERIAL)
    if slot < 0:
        return None
    material = mesh_obj.data.materials[slot]
    if material is None or material.node_tree is None:
        return None
    image = next((n.image for n in material.node_tree.nodes if n.type == "TEX_IMAGE" and n.image), None)
    if image is None or not image.size[0]:
        return None
    width, height = image.size
    pixels = np.array(image.pixels[:], dtype=np.float32).reshape(height, width, 4)[..., :3]
    scale_x, scale_y = width / points["size"][0], height / points["size"][1]
    m = points["mouth"]
    cx = (m["left"][0] + m["right"][0]) / 2 * scale_x
    cy = height - (m["top"][1] + m["bottom"][1]) / 2 * scale_y  # image pixels count up from the bottom
    eye_width = abs(points["left_eye"]["outer"][0] - points["left_eye"]["inner"][0]) * scale_x
    ys, xs = np.mgrid[0:height, 0:width]
    distance = np.hypot(xs - cx, ys - cy)
    band = (distance > SKIN_BAND[0] * eye_width) & (distance < SKIN_BAND[1] * eye_width)
    if band.sum() < 20:
        return None
    return np.median(pixels[band], axis=0)


def _world_to_uv(mesh_obj):
    """(world x, world z) -> the face patch's (u, v), from the same front mapping as _pixel_to_world."""
    au, av = _patch_mapping(mesh_obj)
    return lambda x, z: (au[0] * x + au[1], av[0] * z + av[1])


def _patch_mapping(mesh_obj):
    """Least-squares u = a x + b and v = c z + d over the face patch's polygons."""
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
    return au, av


def _pixel_to_world(mesh_obj, size):
    """(pixel x, pixel y) in the face patch image -> (world x, world z)."""
    au, av = _patch_mapping(mesh_obj)
    width, height = size

    def to_world(pixel):
        u, v = pixel[0] / width, 1.0 - pixel[1] / height
        return (u - au[1]) / au[0], (v - av[1]) / av[0]

    return to_world
