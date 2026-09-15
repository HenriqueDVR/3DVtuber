"""Blink and mouth shape keys for faces that are only painted onto the texture.

Blink squashes the upper part of each painted eye down onto its lower lash
line, so the iris compresses into a closed-eye line while the skin above slides
over it. For the mouth, the mesh is cut along the painted mouth line with a dark
cavity behind it, and vowel keys pull the lips apart to show it.

Call cut_mouth and cut_eyelids before skinning (they change topology) and
add_shape_keys after.
"""

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import texture_space
from .face import HAIR_DISTANCE

MOUTH_MATERIAL = "MouthInside"
# How far the cavity's front sits behind the lips, so an open mouth reads as depth.
CAVITY_RECESS = 0.006
SIDE_ATTRIBUTE = "mouth_side"
UPPER_LIP, LOWER_LIP, CAVITY = 1, -1, 2

BLINK_LEFT, BLINK_RIGHT = "blink_left", "blink_right"
# A blink folds a band around each painted eye shut. Heights are fractions of eye
# height from the eye's centre, widths fractions of eye width.
LASH_LINE = 0.45
EYE_TOP = 0.5
# The band closes fully across the painted eye, then eases open over strips toward its sides.
EYELID_CORE = 0.5
EYELID_OUTER = 0.95
EYELID_EASE_STRIPS = 3
# Skin above the eye that slides down over it (less when hair starts closer).
EYELID_STRETCH = 0.35
EYELIDS_PROPERTY = "vtuber_maker_eyelids"

# Vowel -> (how far the lower lip drops, as a fraction of mouth width; horizontal squeeze)
VOWELS = {
    "aa": (0.55, 1.0),
    "ih": (0.2, 1.1),
    "ou": (0.3, 0.75),
    "ee": (0.25, 1.15),
    "oh": (0.45, 0.8),
}


def cut_mouth(obj, features):
    """Split the surface along the painted mouth and add a dark cavity behind it."""
    mesh = obj.data
    if mesh.shape_keys is not None:
        raise ValueError("cut_mouth changes topology; run it before adding shape keys")
    mouth = _to_local(obj, features["mouth"])
    half = features["mouth_width"] / _scale(obj) / 2

    bm = bmesh.new()
    bm.from_mesh(mesh)
    sides = bm.verts.layers.int.get(SIDE_ATTRIBUTE) or bm.verts.layers.int.new(SIDE_ATTRIBUTE)

    def near_mouth(point, reach):
        return abs(point.x - mouth.x) <= reach and abs(point.z - mouth.z) <= reach and abs(point.y - mouth.y) <= reach

    faces = [f for f in bm.faces if near_mouth(f.calc_center_median(), 1.5 * half)]
    geom = list({v for f in faces for v in f.verts}) + list({e for f in faces for e in f.edges}) + faces
    cut = bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6, plane_co=mouth, plane_no=Vector((0.0, 0.0, 1.0)))
    slit = [
        e
        for e in cut["geom_cut"]
        if isinstance(e, bmesh.types.BMEdge) and all(near_mouth(v.co, half) for v in e.verts)
    ]
    bmesh.ops.split_edges(bm, edges=slit)

    for vert in bm.verts:
        if abs(vert.co.z - mouth.z) > 1e-5 or not near_mouth(vert.co, half):
            continue
        above = [f.calc_center_median().z > mouth.z for f in vert.link_faces]
        if above and all(above):
            vert[sides] = UPPER_LIP
        elif above and not any(above):
            vert[sides] = LOWER_LIP

    material = _mouth_material()
    if material.name not in mesh.materials:
        mesh.materials.append(material)
    slot = list(mesh.materials).index(material)
    radius = 1.2 * half
    center = mouth + Vector((0.0, 0.6 * radius + CAVITY_RECESS, 0.0))
    shape = Matrix.Translation(center) @ Matrix.Diagonal((1.0, 0.6, 0.7, 1.0))
    cavity = bmesh.ops.create_uvsphere(
        bm, u_segments=16, v_segments=10, radius=radius, matrix=shape, calc_uvs=True
    )["verts"]
    for vert in cavity:
        vert[sides] = CAVITY
    for face in {f for v in cavity for f in v.link_faces}:
        face.material_index = slot
        face.smooth = True

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def cut_eyelids(obj, features):
    """Cut each eye's eyelid band out of the surface and store it for add_shape_keys.

    The band closes fully across the painted eye and eases open over a few strips at
    its sides, from the lower lash line up to a ceiling below any hair. With edges
    along every border, the fold is smooth across each triangle, so a blink can't
    smear squashed iris into view or flip skin over itself.
    """
    mesh = obj.data
    if mesh.shape_keys is not None:
        raise ValueError("cut_eyelids changes topology; run it before adding shape keys")
    width, height = np.asarray(features["eye_size"]) / _scale(obj)
    rest = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", rest)
    rest = rest.reshape(-1, 3)
    hair_border = _hair_border(obj, features["hair_color"])

    bands = {}
    bm = bmesh.new()
    bm.from_mesh(mesh)
    for side, point in (("left", features["left_eye"]), ("right", features["right_eye"])):
        eye = _to_local(obj, point)
        band = _eyelid_band(rest, hair_border, eye, width, height)
        bands[side] = band
        reach = band["outer"] + width
        low = Vector((band["center"] - reach, eye.y - width, band["bottom"] - height))
        high = Vector((band["center"] + reach, eye.y + width, band["ceiling"] + height))
        cuts = [(Vector((0.0, 0.0, band[key])), Vector((0.0, 0.0, 1.0))) for key in ("bottom", "top", "ceiling")]
        for offset in np.linspace(band["core"], band["outer"], EYELID_EASE_STRIPS + 1):
            for x in (band["center"] - offset, band["center"] + offset):
                cuts.append((Vector((float(x), 0.0, 0.0)), Vector((1.0, 0.0, 0.0))))
        for plane_co, plane_no in cuts:
            faces = [f for f in bm.faces if _inside_box(f.calc_center_median(), low, high)]
            geom = list({v for f in faces for v in f.verts}) + list({e for f in faces for e in f.edges}) + faces
            bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6, plane_co=plane_co, plane_no=plane_no)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj[EYELIDS_PROPERTY] = bands


def add_shape_keys(obj, features):
    """Add blink_left, blink_right and the five VRM vowel keys (aa, ih, ou, ee, oh)."""
    mesh = obj.data
    rest = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", rest)
    rest = rest.reshape(-1, 3)
    sides = np.zeros(len(mesh.vertices), dtype=np.int32)
    if SIDE_ATTRIBUTE in mesh.attributes:
        mesh.attributes[SIDE_ATTRIBUTE].data.foreach_get("value", sides)
    surface = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    scale = _scale(obj)

    if mesh.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)

    bands = obj.get(EYELIDS_PROPERTY)
    if bands is None:
        raise ValueError("run cut_eyelids before add_shape_keys")
    for name, side in ((BLINK_LEFT, "left"), (BLINK_RIGHT, "right")):
        moved = _blink(rest, sides, bands[side].to_dict())
        _add_key(obj, name, _onto_surface(rest, moved, surface))

    mouth = _to_local(obj, features["mouth"])
    mouth_width = features["mouth_width"] / scale
    for name, (drop, squeeze) in VOWELS.items():
        moved = _open_mouth(rest, sides, mouth, mouth_width, drop, squeeze)
        _add_key(obj, name, _onto_surface(rest, moved, surface))


def _blink(rest, sides, band):
    """Fold one eyelid band: the painted eye collapses onto the lash line and the skin above slides down."""
    x, z = rest[:, 0], rest[:, 2]
    bottom, top, ceiling = band["bottom"], band["top"], band["ceiling"]
    closure = np.clip((band["outer"] - np.abs(x - band["center"])) / (band["outer"] - band["core"]), 0.0, 1.0)
    inside = (z > bottom) & (z < ceiling) & (rest[:, 1] < band["front"]) & (sides != CAVITY)
    slid = bottom + (z - top) * (ceiling - bottom) / (ceiling - top)
    folded = np.where(z <= top, bottom, slid)
    moved = rest.copy()
    moved[:, 2] = np.where(inside, z + closure * (folded - z), z)
    return moved


def _eyelid_band(rest, hair_border, eye, width, height):
    """Bounds of the band that folds shut over one eye, stopping below any hair."""
    band = {
        "center": eye.x,
        "core": EYELID_CORE * width,
        "outer": EYELID_OUTER * width,
        "bottom": eye.z - LASH_LINE * height,
        "top": eye.z + EYE_TOP * height,
        "front": eye.y + width,
    }
    ceiling = band["top"] + EYELID_STRETCH * height
    hair_above = (
        hair_border
        & (np.abs(rest[:, 0] - band["center"]) <= band["outer"])
        & (rest[:, 2] > band["top"])
        & (rest[:, 1] < band["front"])
    )
    if hair_above.any():
        ceiling = min(ceiling, float(rest[hair_above, 2].min()) - 1e-4)
    band["ceiling"] = max(ceiling, band["top"] + 1e-4)
    return {key: float(value) for key, value in band.items()}


def _hair_border(obj, hair_color):
    """Hair vertices and those bordering them. Their triangles carry hair texture, so they must stay put."""
    mesh = obj.data
    colors = texture_space.vertex_colors(obj, texture_space.base_color_image(obj))
    hair = np.linalg.norm(colors - np.asarray(hair_color), axis=1) < HAIR_DISTANCE
    edges = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", edges)
    return _grow(hair, edges[0::2], edges[1::2])


def _inside_box(point, low, high):
    return low.x <= point.x <= high.x and low.y <= point.y <= high.y and low.z <= point.z <= high.z


def _grow(mask, a, b):
    grown = mask.copy()
    grown[a[mask[b]]] = True
    grown[b[mask[a]]] = True
    return grown


def _open_mouth(rest, sides, mouth, width, drop, squeeze):
    half = width / 2
    dx = rest[:, 0] - mouth.x
    across = np.clip(1.0 - (dx / (1.3 * half)) ** 2, 0.0, 1.0)
    near = (np.abs(rest[:, 1] - mouth.y) < width) & (sides != CAVITY)

    depth = mouth.z - rest[:, 2]
    lower = near & (sides != UPPER_LIP) & (depth >= 0) & (depth < 1.5 * width)
    lower_weight = across * _fade(depth, 0.3 * width, 1.5 * width) * lower
    upper = near & (sides != LOWER_LIP) & (depth < 0) & (-depth < 0.4 * width)
    upper_weight = across * _fade(-depth, 0.0, 0.4 * width) * upper
    upper_weight[sides == UPPER_LIP] = across[sides == UPPER_LIP]

    moved = rest.copy()
    moved[:, 2] -= drop * width * lower_weight
    moved[:, 2] += 0.15 * drop * width * upper_weight
    lips = np.maximum(lower_weight, upper_weight)
    moved[:, 0] = mouth.x + dx * (1.0 + (squeeze - 1.0) * lips)
    return moved


def _onto_surface(rest, moved, surface):
    """Slide moved vertices back onto the original surface along Y, looking from just in front."""
    result = moved.copy()
    for index in np.nonzero(np.any(np.abs(moved - rest) > 1e-7, axis=1))[0]:
        x, y, z = moved[index]
        hit = surface.ray_cast(Vector((x, rest[index, 1] - 0.02, z)), Vector((0.0, 1.0, 0.0)), 0.05)[0]
        if hit is not None:
            result[index, 1] = hit.y
    return result


def _add_key(obj, name, coords):
    key = obj.shape_key_add(name=name, from_mix=False)
    key.data.foreach_set("co", coords.astype(np.float32).ravel())
    key.value = 0.0


def _fade(distance, full, zero):
    """1 up to `full`, easing to 0 at `zero`."""
    t = np.clip((distance - full) / max(zero - full, 1e-9), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def _mouth_material():
    material = bpy.data.materials.get(MOUTH_MATERIAL) or bpy.data.materials.new(MOUTH_MATERIAL)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.12, 0.02, 0.03, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    return material


def _to_local(obj, point):
    return obj.matrix_world.inverted() @ Vector(tuple(float(c) for c in point))


def _scale(obj):
    return obj.matrix_world.to_scale().x
