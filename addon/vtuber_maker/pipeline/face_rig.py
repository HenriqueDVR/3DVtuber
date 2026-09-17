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
# The mouth-inside patch of a modelled mouth is painted: upper teeth, a dark throat, a tongue.
MOUTH_PATCH_MATERIAL = "MouthInsidePainted"
MOUTH_TEXTURE_SIZE = 256
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
# Vertices face shapes must leave alone (1), e.g. hair parts hanging in front of modelled eyes.
FIXED_ATTRIBUTE = "face_rig_fixed"
# Eye insert vertices: 1 for the left eye, 2 for the right.
EYE_INSERT_ATTRIBUTE = "face_rig_eye_insert"
EYE_INSERT_IDS = {"left": 1, "right": 2}
# A modelled eye's opening, as fractions of its iris insert's width and height: the opening is
# wider than the iris, sits a little toward the outer corner, and hides the iris's top and bottom.
SOCKET_RX = 0.85
SOCKET_RZ = 0.4
SOCKET_OUTWARD = 0.12
# A modelled face isn't built to fold shut, so each eye gets a lid cover: a skin-coloured patch
# folded flat along the top of the opening at rest, which unfolds over the opening to blink.
EYELID_MATERIAL = "EyelidSkin"
LASH_MATERIAL = "EyelidLash"
LID_ATTRIBUTE = "face_rig_lid"  # 1 left lid cover, 2 right
LID_CLOSED_ATTRIBUTE = "face_rig_lid_closed"
LID_COLUMNS, LID_ROWS = 12, 12
# The cover overhangs the opening so its edges tuck behind the rim.
LID_OVERHANG = 1.15
# Depth of the cover in front of the iris, and how much it bulges, in opening half-heights:
# just in front of the rim, so a closed lid hides the open eye's painted lash outline.
LID_DEPTH = 0.95
LID_BULGE = 0.35
# The bottom strip of the cover, as a fraction of its height, is the closed lash line.
LASH_STRIP = 0.08
# A material with this name holds a face texture mapped straight from the front (see
# scripts/retexture/face_patch.py). With a closed-eyes version of that texture, lid covers
# are textured from it instead of flat skin and lash colours.
FACE_PATCH_MATERIAL = "face_patch"
TEXTURED_LID_MATERIAL = "EyelidTextured"

# A modelled mouth opens by unfolding a dark mouth-inside patch over its slit, flat at rest.
MOUTH_COVER_PROPERTY = "vtuber_maker_mouth_cover"
MOUTH_GRID_ATTRIBUTE = "face_rig_mouth_grid"  # (u, v, 1) on mouth patch vertices, zeros elsewhere
MOUTH_COLUMNS, MOUTH_ROWS = 10, 8
# Vowel -> (opening height, patch width), as fractions of the mouth's width.
MOUTH_SHAPES = {
    "aa": (0.75, 1.0),
    "ih": (0.28, 1.1),
    "ou": (0.45, 0.6),
    "ee": (0.33, 1.15),
    "oh": (0.6, 0.78),
}
# The upper lip opens less than the lower one drops.
MOUTH_UPPER_SHARE = 0.3
# How far in front of the lips the patch sits.
MOUTH_COVER_GAP = 0.0004
# Modelled mouths barely move their lips: the mouth-inside patch shows the opening.
MODELLED_MOUTH_OPENING = 0.45

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


def set_eyelids_from_inserts(obj, features, parts, closed_eyes_image=None, lids=True):
    """Eyelid bands for eyes modelled as inserts (see parts.find), for add_shape_keys.

    The face already has real sockets, so nothing is cut. Everything outside the face
    part and its inserts is marked fixed, so blinking and talking never drag the hair.
    `closed_eyes_image` is a closed-eyes version of the face patch texture, if any.
    Without `lids` no lid covers are added and the eyes are left as modelled.
    """
    mesh = obj.data
    width, height = np.asarray(features["eye_size"]) / _scale(obj)
    eyes = {side: _to_local(obj, features[f"{side}_eye"]) for side in ("left", "right")}
    middle_x = (eyes["left"].x + eyes["right"].x) / 2
    bands = {
        side: {
            "socket_x": float(eye.x + np.sign(eye.x - middle_x) * SOCKET_OUTWARD * width),
            "socket_z": float(eye.z),
            "rx": float(SOCKET_RX * width),
            "rz": float(SOCKET_RZ * height),
            "iris_y": float(eye.y),
            "front": float(eye.y + width),
        }
        for side, eye in eyes.items()
    }
    obj[EYELIDS_PROPERTY] = bands

    original_count = len(mesh.vertices)
    rest = np.empty(original_count * 3)
    mesh.vertices.foreach_get("co", rest)
    rest = rest.reshape(-1, 3)
    closed = _add_lid_covers(obj, bands, features["skin_color"], closed_eyes_image) if lids else {}
    mouth_grid = _add_mouth_cover(obj, features, rest, parts)
    added = len(mesh.vertices) - original_count

    fixed = np.concatenate([parts.fixed(), np.zeros(added, dtype=bool)])
    _set_point_attribute(mesh, FIXED_ATTRIBUTE, fixed.astype(np.int32))
    inserts = np.zeros(len(mesh.vertices), dtype=np.int32)
    for side, part in parts.eyes.items():
        inserts[:original_count][parts.labels == part] = EYE_INSERT_IDS[side]
    _set_point_attribute(mesh, EYE_INSERT_ATTRIBUTE, inserts)
    lids = np.zeros(len(mesh.vertices), dtype=np.int32)
    closed_positions = np.empty((len(mesh.vertices), 3))
    mesh.vertices.foreach_get("co", closed_positions.ravel())
    for side, (indices, positions) in closed.items():
        lids[indices] = EYE_INSERT_IDS[side]
        closed_positions[indices] = positions
    _set_point_attribute(mesh, LID_ATTRIBUTE, lids)
    _set_vector_attribute(mesh, LID_CLOSED_ATTRIBUTE, closed_positions)
    grid = np.zeros((len(mesh.vertices), 3))
    indices, coordinates = mouth_grid
    grid[indices] = coordinates
    _set_vector_attribute(mesh, MOUTH_GRID_ATTRIBUTE, grid)


def _add_mouth_cover(obj, features, rest, parts):
    """Add the mouth-inside patch over the slit; return (vertex indices, (u, v, 1) per vertex).

    At rest every vertex lies on the slit, so the patch has no area; add_shape_keys
    spreads it into an opening per vowel. It sits just in front of the lips and follows
    the head bone.
    """
    mesh = obj.data
    mouth = _to_local(obj, features["mouth"])
    width = features["mouth_width"] / _scale(obj)
    # In front of the lips and of the mouth insert, which some models push out past the lips.
    lips = (
        ((parts.labels == parts.face) | (parts.labels == parts.mouth))
        & (np.abs(rest[:, 0] - mouth.x) < 0.6 * width)
        & (np.abs(rest[:, 2] - mouth.z) < 0.3 * width)
    )
    front = float(rest[lips, 1].min()) if lips.any() else mouth.y
    obj[MOUTH_COVER_PROPERTY] = {"x": mouth.x, "y": front - MOUTH_COVER_GAP, "z": mouth.z, "width": width}

    material = _mouth_patch_material()
    if material.name not in mesh.materials:
        mesh.materials.append(material)
    slot = mesh.materials.find(material.name)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    uv_layer = bm.loops.layers.uv.active or bm.loops.layers.uv.new()
    start = len(bm.verts)
    rows, coordinates = [], []
    for j in range(MOUTH_ROWS + 1):
        v = 1.0 - 2.0 * j / MOUTH_ROWS
        row = []
        for i in range(MOUTH_COLUMNS + 1):
            u = -1.0 + 2.0 * i / MOUTH_COLUMNS
            x = mouth.x + 0.5 * width * u * np.sqrt(max(1.0 - v * v, 0.0))
            row.append(bm.verts.new((x, front - MOUTH_COVER_GAP, mouth.z)))
            coordinates.append((u, v, 1.0))
        rows.append(row)
    for j in range(MOUTH_ROWS):
        for i in range(MOUTH_COLUMNS):
            quad = [(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)]
            face = bm.faces.new([rows[r][c] for r, c in quad])
            face.material_index = slot
            face.smooth = True
            for loop, (r, c) in zip(face.loops, quad):
                u, v, _ = coordinates[r * (MOUTH_COLUMNS + 1) + c]
                loop[uv_layer].uv = ((u + 1) / 2, (v + 1) / 2)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    count = (MOUTH_ROWS + 1) * (MOUTH_COLUMNS + 1)
    indices = np.arange(start, start + count)
    head = obj.vertex_groups.get("Head")
    if head is not None:
        head.add(indices.tolist(), 1.0, "REPLACE")
    return indices, np.array(coordinates)


def _mouth_shape(grid, cover, vowel):
    """Mouth patch vertex positions for one vowel (grid holds (u, v, 1) per patch vertex)."""
    height, width_share = MOUTH_SHAPES[vowel]
    width = cover["width"]
    u, v = grid[:, 0], grid[:, 1]
    half_width = 0.5 * width * width_share
    x = cover["x"] + half_width * u * np.sqrt(np.clip(1.0 - v * v, 0.0, 1.0))
    z = cover["z"] + height * width * np.where(v > 0, MOUTH_UPPER_SHARE * v, v) / 2
    return np.stack([x, np.full(len(u), cover["y"]), z], axis=1)


def _add_lid_covers(obj, bands, skin_color, closed_eyes_image=None):
    """Add a folded lid cover per eye; return side -> (vertex indices, closed positions).

    The cover's rows all lie on the top edge of the opening at rest, so it has no area
    and can't be seen; the closed positions spread the rows down to the bottom edge.
    Its vertices follow the head bone. With a face patch and its closed-eyes image, the
    cover shows that image where it lands when closed, so shut lids match the face.
    """
    mesh = obj.data
    patch_uv = _face_patch_uv_map(obj) if closed_eyes_image else None
    textured_slot = None
    if patch_uv is not None:
        textured = _textured_lid_material(mesh, closed_eyes_image)
        if textured.name not in mesh.materials:
            mesh.materials.append(textured)
        textured_slot = mesh.materials.find(textured.name)
    # Texture pixels read through Blender are already linear, like material colours.
    skin = _material(EYELID_MATERIAL, tuple(float(c) for c in skin_color[:3]))
    lash = _material(LASH_MATERIAL, (0.012, 0.006, 0.008))
    for material in (skin, lash):
        if material.name not in mesh.materials:
            mesh.materials.append(material)
    skin_slot, lash_slot = mesh.materials.find(skin.name), mesh.materials.find(lash.name)

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.active or bm.loops.layers.uv.new()
    start = len(bm.verts)
    result = {}
    for side, band in bands.items():
        cx, cz, rx, rz = band["socket_x"], band["socket_z"], band["rx"], band["rz"]
        rows = []
        closed = []
        for j in range(LID_ROWS + 1):
            t = j / LID_ROWS
            row = []
            for i in range(LID_COLUMNS + 1):
                u = -1.0 + 2.0 * i / LID_COLUMNS
                half = LID_OVERHANG * rz * np.sqrt(max(1.0 - u * u, 0.0)) + 0.15 * rz
                x = cx + LID_OVERHANG * rx * u
                v = 1.0 - 2.0 * t  # 1 at the top edge, -1 at the bottom
                y = band["iris_y"] - LID_DEPTH * rz - LID_BULGE * rz * max(1.0 - u * u - v * v, 0.0)
                row.append(bm.verts.new((x, y, cz + half)))
                closed.append((x, y, cz + half * v))
            rows.append(row)
        for j in range(LID_ROWS):
            for i in range(LID_COLUMNS):
                quad = [(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)]
                face = bm.faces.new([rows[r][c] for r, c in quad])
                face.smooth = True
                if textured_slot is None:
                    face.material_index = lash_slot if (j + 1) / LID_ROWS > 1.0 - LASH_STRIP else skin_slot
                    continue
                face.material_index = textured_slot
                for loop, (r, c) in zip(face.loops, quad):
                    x, _, z = closed[r * (LID_COLUMNS + 1) + c]
                    loop[uv_layer].uv = patch_uv(x, z)
        count = (LID_ROWS + 1) * (LID_COLUMNS + 1)
        result[side] = (np.arange(start, start + count), np.array(closed))
        start += count
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    head = obj.vertex_groups.get("Head")
    if head is not None:
        head.add([int(i) for indices, _ in result.values() for i in indices], 1.0, "REPLACE")
    return result


def _face_patch_uv_map(obj):
    """(x, z) in mesh space -> the face patch's UV there, or None without a face patch.

    The patch is an orthographic front projection, so its UVs are affine in x and z;
    fit that map from the patch's own vertices.
    """
    mesh = obj.data
    slot = mesh.materials.find(FACE_PATCH_MATERIAL)
    if slot < 0:
        return None
    uv = mesh.uv_layers.active.data
    xs, zs, us, vs = [], [], [], []
    for poly in mesh.polygons:
        if poly.material_index != slot:
            continue
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            xs.append(co.x)
            zs.append(co.z)
            us.append(uv[li].uv.x)
            vs.append(uv[li].uv.y)
    if len(xs) < 3:
        return None
    ones = np.ones(len(xs))
    a_u = np.linalg.lstsq(np.stack([xs, ones], 1), us, rcond=None)[0]
    a_v = np.linalg.lstsq(np.stack([zs, ones], 1), vs, rcond=None)[0]
    return lambda x, z: (float(a_u[0] * x + a_u[1]), float(a_v[0] * z + a_v[1]))


def _textured_lid_material(mesh, image_path):
    patch = mesh.materials[mesh.materials.find(FACE_PATCH_MATERIAL)]
    material = patch.copy()
    material.name = TEXTURED_LID_MATERIAL
    bsdf = material.node_tree.nodes["Principled BSDF"]
    node = bsdf.inputs["Base Color"].links[0].from_node
    node.image = bpy.data.images.load(str(image_path), check_existing=True)
    return material


def _material(name, color):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    material.diffuse_color = (*color, 1.0)
    return material


def add_shape_keys(obj, features, blink=True):
    """Add blink_left, blink_right and the five VRM vowel keys (aa, ih, ou, ee, oh).

    Without `blink` only the vowel keys are added and nothing moves the eyes.
    """
    mesh = obj.data
    rest = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", rest)
    rest = rest.reshape(-1, 3)
    sides = np.zeros(len(mesh.vertices), dtype=np.int32)
    if SIDE_ATTRIBUTE in mesh.attributes:
        mesh.attributes[SIDE_ATTRIBUTE].data.foreach_get("value", sides)
    fixed = np.zeros(len(mesh.vertices), dtype=np.int32)
    if FIXED_ATTRIBUTE in mesh.attributes:
        mesh.attributes[FIXED_ATTRIBUTE].data.foreach_get("value", fixed)
    free = fixed == 0
    surface = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    scale = _scale(obj)

    if mesh.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)

    bands = obj.get(EYELIDS_PROPERTY)
    if bands is None and blink:
        raise ValueError("run cut_eyelids before add_shape_keys")
    inserts = np.zeros(len(mesh.vertices), dtype=np.int32)
    if EYE_INSERT_ATTRIBUTE in mesh.attributes:
        mesh.attributes[EYE_INSERT_ATTRIBUTE].data.foreach_get("value", inserts)
    for name, side in ((BLINK_LEFT, "left"), (BLINK_RIGHT, "right")) if blink else ():
        band = bands[side].to_dict()
        if "rx" in band:
            # A modelled eye closes by unfolding its lid cover; the iris flattens behind it.
            moved = rest.copy()
            lids = np.zeros(len(mesh.vertices), dtype=np.int32)
            mesh.attributes[LID_ATTRIBUTE].data.foreach_get("value", lids)
            closed = np.empty(len(mesh.vertices) * 3)
            mesh.attributes[LID_CLOSED_ATTRIBUTE].data.foreach_get("vector", closed)
            closed = closed.reshape(-1, 3)
            lid = lids == EYE_INSERT_IDS[side]
            moved[lid] = closed[lid]
            insert = inserts == EYE_INSERT_IDS[side]
            moved[insert, 2] = band["socket_z"] - 0.5 * band["rz"]
            moved[insert, 1] = rest[insert, 1] + 0.5 * band["rz"]
            _add_key(obj, name, moved)
        else:
            moved = _blink(rest, sides, band)
            moved[~free] = rest[~free]
            _add_key(obj, name, _onto_surface(rest, moved, surface))

    mouth = _to_local(obj, features["mouth"])
    mouth_width = features["mouth_width"] / scale
    modelled = EYE_INSERT_ATTRIBUTE in mesh.attributes
    opening = MODELLED_MOUTH_OPENING if modelled else 1.0
    # A modelled mouth's corners stay put, so it opens round instead of as a box.
    corner_reach = 1.0 if modelled else 1.3
    mouth_grid = None
    if MOUTH_GRID_ATTRIBUTE in mesh.attributes:
        mouth_grid = np.empty(len(mesh.vertices) * 3)
        mesh.attributes[MOUTH_GRID_ATTRIBUTE].data.foreach_get("vector", mouth_grid)
        mouth_grid = mouth_grid.reshape(-1, 3)
    for name, (drop, squeeze) in VOWELS.items():
        moved = _open_mouth(rest, sides, mouth, mouth_width, opening * drop, squeeze, corner_reach)
        moved[~free] = rest[~free]
        if mouth_grid is not None:
            patch = mouth_grid[:, 2] > 0.5
            moved[patch] = _mouth_shape(mouth_grid[patch], obj[MOUTH_COVER_PROPERTY].to_dict(), name)
            _add_key(obj, name, moved)
            continue
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


def _set_vector_attribute(mesh, name, values):
    attribute = mesh.attributes.get(name) or mesh.attributes.new(name, "FLOAT_VECTOR", "POINT")
    attribute.data.foreach_set("vector", np.asarray(values, dtype=np.float32).ravel())


def _set_point_attribute(mesh, name, values):
    attribute = mesh.attributes.get(name) or mesh.attributes.new(name, "INT", "POINT")
    attribute.data.foreach_set("value", values)


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


def _open_mouth(rest, sides, mouth, width, drop, squeeze, corner_reach=1.3):
    half = width / 2
    dx = rest[:, 0] - mouth.x
    across = np.clip(1.0 - (dx / (corner_reach * half)) ** 2, 0.0, 1.0)
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


def _mouth_patch_material():
    material = bpy.data.materials.get(MOUTH_PATCH_MATERIAL)
    if material is not None:
        return material
    material = bpy.data.materials.new(MOUTH_PATCH_MATERIAL)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 1.0
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = _mouth_inside_image()
    texture.extension = "EXTEND"
    material.node_tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    return material


def _mouth_inside_image():
    """Paint the inside of an open mouth, u across and v up the opening."""
    n = MOUTH_TEXTURE_SIZE
    v, u = np.mgrid[0:n, 0:n] / (n - 1)
    rim = np.clip(np.hypot((u - 0.5) / 0.5, (v - 0.5) / 0.5), 0, 1)
    throat = np.array([0.10, 0.02, 0.03])
    gum = np.array([0.42, 0.13, 0.15])
    color = throat + (gum - throat) * (rim**3)[..., None]
    teeth = (v > 0.8) & (np.abs(u - 0.5) < 0.42)
    color[teeth] = (0.96, 0.94, 0.91)
    color[teeth & (v < 0.83)] = (0.78, 0.74, 0.74)  # teeth's lower edge in shadow
    tongue = ((u - 0.5) / 0.34) ** 2 + ((v - 0.08) / 0.3) ** 2 < 1
    shade = np.clip(1.2 - v / 0.38, 0.75, 1.0)[..., None]
    color = np.where(tongue[..., None], np.array([0.80, 0.38, 0.42]) * shade, color)
    pixels = np.concatenate([color, np.ones((n, n, 1))], axis=2).astype(np.float32)
    image = bpy.data.images.new(MOUTH_PATCH_MATERIAL, n, n)
    image.pixels.foreach_set(pixels.ravel())
    image.pack()
    return image


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
