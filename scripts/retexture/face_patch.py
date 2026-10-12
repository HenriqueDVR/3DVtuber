"""Give the front of the face its own texture, mapped straight from one view.

Tripo's UV layout spends few texels on the face, so projecting a sharp image
into the shared atlas blurs it. This moves the polygons a view sees head-on
onto a new material whose UVs are that view's screen coordinates, so the
image is used at full resolution.

    blender -b --python face_patch.py -- model.glb|model.fbx views_dir view_name image.png out.glb

A rigged FBX (Tripo's) goes in as it is; the GLB names it as its rig (see matte_materials.py).
"""
import json
import os
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "addon"))
from vtuber_maker.pipeline import meshdata, normalize  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, VIEWS, VIEW, IMAGE, OUT = (os.path.abspath(argv[0]), os.path.abspath(argv[1]), argv[2],
                                os.path.abspath(argv[3]), os.path.abspath(argv[4]))

MIN_FACING = 0.45    # cosine between polygon normal and view direction
BORDER = 0.04        # ignore polygons this close to the image border
# Region of the view (x0, y0, x1, y1 in 0..1, y up) the patch may use: the face, not collar or bows.
REGION = (0.22, 0.18, 0.78, 0.80)
# A polygon is taken by its centre, so one half under a lock of hair reads the hair's pixels for
# that half: the same texels then colour two surfaces. Under the bangs that is skin 1.3-2.3 cm
# behind them, hidden and needed (the eyes are found through the patch). Where what covers it is
# further in front than this share of the height it is the collar or neck behind hanging hair
# (5-9 cm on model A's T-pose model): the hair's orange showed on the clothing, and paint put on the
# hair in the app recoloured the clothing too. Those stay on the model's own texture.
DEEP = 0.025
INSET = 0.15         # how far the samples are pulled in from a polygon's rim
# What is left shares texels still: a lock of hair over the face and the skin behind it read the
# same pixels, so nothing behind the hair could be painted without painting the hair. Patch
# polygons with another patch polygon more than this share of the height behind them (the bangs:
# 1.3 cm and more in front of the face on model A and model C) go on a second material with its own
# copy of the picture. They look the same; the face keeps the first picture to itself. Less than
# this apart are the face's own layers (lids over modelled eyes, under 0.6 cm), which stay together.
LAYER = 0.006
OVER_MATERIAL = "face_patch_over"   # face_points.FACE_PATCH_OVER
LAYER_GRID = 1024
MAX_OVER = 3         # pictures for what lies in front: face_patch_over, face_patch_over2, face_patch_over3

with open(os.path.join(VIEWS, "views.json")) as f:
    meta = json.load(f)
view = next(v for v in meta["views"] if v["name"] == VIEW)
RES, height = meta["resolution"], meta["height"]
M = np.array(view["matrix_world"])
Minv = np.linalg.inv(M)
half = view["ortho_scale"] / 2
cam_axis = M[:3, 2]

def _upstream_texture(node, depth=4):
    """The image texture node feeding `node`, through a few in between: Tripo's baseColorFactor 0.8
    imports as a Mix node between the texture and Base Color."""
    if node.type == "TEX_IMAGE":
        return node
    for socket in node.inputs if depth else ():
        for link in socket.links:
            found = _upstream_texture(link.from_node, depth - 1)
            if found is not None:
                return found
    return None


bpy.ops.wm.read_factory_settings(use_empty=True)
rig = normalize.import_unrigged(Path(SRC))
obj = next(o for o in bpy.context.scene.objects if o.type == "MESH")
mesh = obj.data

zimg = bpy.data.images.load(os.path.join(VIEWS, f"z_{VIEW}.exr"))
zbuf = np.empty(RES * RES * 4, np.float32)
zimg.pixels.foreach_get(zbuf)
zbuf = zbuf.reshape(RES, RES, 4)[..., 0]
tol = height * 0.004


def to_view(p):
    q = Minv[:3, :3] @ np.asarray(p) + Minv[:3, 3]
    return (q[0] / half + 1) / 2, (q[1] / half + 1) / 2, -q[2]


mw = obj.matrix_world
normal_mat = mw.to_3x3().inverted().transposed()
chosen, behind = [], 0
for poly in mesh.polygons:
    n = (normal_mat @ poly.normal).normalized()
    if n.dot(Vector(cam_axis)) < MIN_FACING:
        continue
    corners = [to_view(mw @ mesh.vertices[v].co) for v in poly.vertices]
    cx, cy, cz = to_view(mw @ poly.center)
    if not all(REGION[0] <= x <= REGION[2] and REGION[1] <= y <= REGION[3] for x, y, _ in corners):
        continue
    if not all(BORDER <= x <= 1 - BORDER and BORDER <= y <= 1 - BORDER for x, y, _ in corners):
        continue
    zb = zbuf[min(int(cy * RES), RES - 1), min(int(cx * RES), RES - 1)]
    if zb <= 0 or abs(zb - cz) > tol:
        continue
    # its corners and the middles of its sides, pulled in from the rim
    rim = corners + [tuple((a + b) / 2 for a, b in zip(corners[k], corners[k - 1])) for k in range(len(corners))]
    covered = False
    for x, y, z in rim:
        x, y, z = x + (cx - x) * INSET, y + (cy - y) * INSET, z + (cz - z) * INSET
        front = zbuf[min(int(y * RES), RES - 1), min(int(x * RES), RES - 1)]
        if front > 0 and z - front > height * DEEP:
            covered = True
            break
    if covered:
        behind += 1
        continue
    chosen.append(poly.index)
print(f"face patch polygons: {len(chosen)} ({behind} left on the model's texture: partly behind something well in front)")



def _in(poly):
    """Sample points of a polygon in view space: its corners and the middles of its sides, pulled in, and its centre."""
    corners = [to_view(mw @ mesh.vertices[v].co) for v in poly.vertices]
    cx, cy, cz = to_view(mw @ poly.center)
    rim = corners + [tuple((a + b) / 2 for a, b in zip(corners[k], corners[k - 1])) for k in range(len(corners))]
    return [(x + (cx - x) * INSET, y + (cy - y) * INSET, z + (cz - z) * INSET) for x, y, z in rim] + [(cx, cy, cz)]


G = LAYER_GRID
mesh.calc_loop_triangles()
seen = {}


def depths(polys):
    """The farthest of these polygons under every texel of the picture (0 where there is none)."""
    grid = np.zeros((G, G), np.float32)
    taken = set(polys)
    for tri in mesh.loop_triangles:
        if tri.polygon_index not in taken:
            continue
        for v in tri.vertices:
            if v not in seen:
                seen[v] = to_view(mw @ mesh.vertices[v].co)
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = ((seen[v][0] * G, seen[v][1] * G, seen[v][2]) for v in tri.vertices)
        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if abs(area) < 1e-12:
            continue
        xa, xb = max(0, int(min(x0, x1, x2))), min(G - 1, int(max(x0, x1, x2)))
        ya, yb = max(0, int(min(y0, y1, y2))), min(G - 1, int(max(y0, y1, y2)))
        if xb < xa or yb < ya:
            continue
        px, py = np.meshgrid(np.arange(xa, xb + 1) + 0.5, np.arange(ya, yb + 1) + 0.5)
        w0 = ((x1 - px) * (y2 - py) - (y1 - py) * (x2 - px)) / area
        w1 = ((x2 - px) * (y0 - py) - (y2 - py) * (x0 - px)) / area
        inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
        depth = (z0 * w0 + z1 * w1 + z2 * (1 - w0 - w1)).astype(np.float32)
        part = grid[ya:yb + 1, xa:xb + 1]
        part[inside] = np.maximum(part[inside], depth[inside])
    return grid


farthest = depths(chosen)
over = [i for i in chosen
        if any(farthest[min(int(y * G), G - 1), min(int(x * G), G - 1)] - z > height * LAYER for x, y, z in _in(mesh.polygons[i]))]
# A separate piece in front of the face anywhere is in front of it altogether. Model D's visor lies
# under LAYER from the skin along its rim, and the skin wholly behind it isn't in the patch at all:
# only 36 of its polygons counted as in front, the rest stayed on the face's picture, the skin
# under the rim shared their texels and could not be painted. Not the face's own piece (the one
# with the most polygons in the patch): hair that is one surface with it (model A) is left to the
# test above. In front of the face's piece, that is: an eye insert has layers of its own LAYER
# apart (model A's: 15 polygons each), and on the hair's picture the eye step would not find it.
labels = meshdata.pieces(obj)
piece_of = {i: int(labels[mesh.polygons[i].vertices[0]]) for i in chosen}
sizes = {}
for piece in piece_of.values():
    sizes[piece] = sizes.get(piece, 0) + 1
face_piece = max(sizes, key=sizes.get) if sizes else None
skin = depths([i for i in chosen if piece_of[i] == face_piece])
in_front = {piece_of[i] for i in chosen if piece_of[i] != face_piece
            and any(skin[min(int(y * G), G - 1), min(int(x * G), G - 1)] - z > height * LAYER for x, y, z in _in(mesh.polygons[i]))}
over = [i for i in chosen if i in set(over) or piece_of[i] in in_front]
# ... and in front of each other they need a picture each, for the same reason: with model D's
# visor and hair on one, the visor's outline was in the hair's texels. The piece most in front of
# the others on its picture moves to the next, until none is or the pictures run out.
level = {i: 1 for i in over}
for at in range(1, MAX_OVER):
    while True:
        here = [i for i in over if level[i] == at]
        ahead = {}
        for piece in {piece_of[i] for i in here} - {face_piece}:
            rest = depths([i for i in here if piece_of[i] != piece])
            ahead[piece] = sum(any(rest[min(int(y * G), G - 1), min(int(x * G), G - 1)] - z > height * LAYER
                                   for x, y, z in _in(mesh.polygons[i])) for i in here if piece_of[i] == piece)
        if not ahead or max(ahead.values()) == 0:
            break
        first = max(ahead, key=ahead.get)
        for i in here:
            if piece_of[i] == first:
                level[i] = at + 1
print(f"face patch polygons in front of others (own picture): {len(over)}, "
      f"on {len(set(level.values()))} pictures: {[sum(v == k for v in level.values()) for k in sorted(set(level.values()))]}")

base = mesh.materials[0]
patch = base.copy()
patch.name = "face_patch"
bsdf = patch.node_tree.nodes["Principled BSDF"]
tex_node = _upstream_texture(bsdf.inputs["Base Color"].links[0].from_node)
for link in list(bsdf.inputs["Normal"].links):   # the normal map follows the atlas UVs, not the patch's
    patch.node_tree.links.remove(link)
# ... and so does an alpha read from a node of its own: Blender's FBX import gives a model whose
# texture also cuts it one node for the colour and another for the alpha. The export then wrote the
# atlas's alpha into the face's picture, stretched over it: a tester's flat-colour model (a palette
# 2048 x 512, its unused right part see-through) had the right 40% of the face picture cut away, the
# face half gone on the Face page. The picture is a render and has no holes.
for link in list(bsdf.inputs["Alpha"].links):
    if link.from_node != tex_node:
        patch.node_tree.links.remove(link)
img = bpy.data.images.load(IMAGE)
img.colorspace_settings.name = "sRGB"
tex_node.image = img
tex_node.extension = "EXTEND"
mesh.materials.append(patch)
slot = len(mesh.materials) - 1

uv = mesh.uv_layers.active.data
for i in chosen:
    poly = mesh.polygons[i]
    poly.material_index = slot
    for li in poly.loop_indices:
        x, y, _ = to_view(mw @ mesh.vertices[mesh.loops[li].vertex_index].co)
        uv[li].uv = (x, y)

front_images = {}
for at in sorted(set(level.values())):
    # the same picture under another name: one image used twice would be one set of texels again
    mark = "_over" if at == 1 else f"_over{at}"
    copy = os.path.splitext(IMAGE)[0] + mark + ".png"
    shutil.copyfile(IMAGE, copy)
    front = patch.copy()
    front.name = OVER_MATERIAL if at == 1 else f"{OVER_MATERIAL}{at}"
    front_node = _upstream_texture(front.node_tree.nodes["Principled BSDF"].inputs["Base Color"].links[0].from_node)
    front_images[at] = bpy.data.images.load(copy)
    front_images[at].colorspace_settings.name = "sRGB"
    front_node.image = front_images[at]
    mesh.materials.append(front)
    for i in over:
        if level[i] == at:
            mesh.polygons[i].material_index = len(mesh.materials) - 1

# Where the face lies behind something (the bangs), the picture shows that something: the hair's
# colour sat on the skin under it, and showed whenever the hair swung aside. Those texels of the
# face's picture take the colour of the face around them instead. The hair keeps the picture as it
# was on its own copy; the file the face points are marked on stays as rendered too.
FILL_GROW = 2        # grid texels past the hidden part that are filled too: the hair's soft edge
FILL_STEP = 0.004    # neighbouring grid texels further apart in depth than this share of the height are two surfaces
# Only with --keep, and never inside it: the boxes of the picture (u0,v0,u1,v1, v up) round the eyes
# and the mouth, which the face points give once they are marked. Filled there, the smeared lashes
# and iris made the eye step read a taller iris and refuse it; and without the points there is no
# telling the eyes from the hair, so a first patch is left as rendered and a remade one is filled.
# The fill grows in from the face that shows beside each hidden stretch, whatever it is.
KEEP = [tuple(float(v) for v in argv[i + 1].split(",")) for i, a in enumerate(argv) if a == "--keep"]
front_set = set(over)
face_depth = depths([i for i in chosen if i not in front_set])
rows = np.minimum((np.arange(G) + 0.5) * RES / G, RES - 1).astype(int)
nearest = zbuf[np.ix_(rows, rows)]
hidden = (face_depth > 0) & (nearest > 0) & (face_depth - nearest > height * LAYER)
# ... and all of what a piece in front covers, however near it lies (the skin under a visor's rim)
hidden |= (face_depth > 0) & (depths(over) > 0)
for u0, v0, u1, v1 in KEEP:
    hidden[max(0, int(v0 * G)):int(v1 * G) + 1, max(0, int(u0 * G)):int(u1 * G) + 1] = False


def fill(image, depth, hidden, keep=()):
    """Paint the texels of `image` in `hidden` (a grid mask) from the picture around them, never
    inside the `keep` boxes; returns how many. `depth` is the grid of the picture's own polygons
    (0 where there are none).

    Colour only spreads along one surface: between texels whose depths differ by under FILL_STEP.
    The face's picture also holds the neck under the chin and the inside of a collar behind it;
    filled from whatever was nearest in the picture, model D's neck and the cape's inner back
    behind her hair came out in the pale blue of her face.
    """
    own = depth > 0
    step = height * FILL_STEP
    for _ in range(FILL_GROW):
        grown = hidden.copy()
        grown[1:] |= hidden[:-1]; grown[:-1] |= hidden[1:]; grown[:, 1:] |= hidden[:, :-1]; grown[:, :-1] |= hidden[:, 1:]
        hidden = grown & own
    for u0, v0, u1, v1 in keep:
        hidden[max(0, int(v0 * G)):int(v1 * G) + 1, max(0, int(u0 * G)):int(u1 * G) + 1] = False
    width, tall = image.size
    pixels = np.empty(width * tall * 4, np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(tall, width, 4)
    step_x, step_y = width // G, tall // G
    small = pixels[step_y // 2::step_y, step_x // 2::step_x, :3][:G, :G].copy()
    known = own & ~hidden
    todo = hidden.copy()
    for _ in range(G):
        if not todo.any() or not known.any():
            break
        total = np.zeros_like(small)
        count = np.zeros((G, G), np.float32)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            near = np.roll(known, shift, axis) & (np.abs(np.roll(depth, shift, axis) - depth) < step)
            total += np.roll(small, shift, axis) * near[..., None]
            count += near
        new = todo & (count > 0)
        small[new] = total[new] / count[new][:, None]
        known |= new
        todo &= ~new
    # the fill grew in from the rim in streaks: evened out among itself
    for _ in range(12):
        total = small.copy()
        count = np.ones((G, G), np.float32)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            near = np.roll(own, shift, axis) & (np.abs(np.roll(depth, shift, axis) - depth) < step)
            total += np.roll(small, shift, axis) * near[..., None]
            count += near
        small[hidden] = (total / count[..., None])[hidden]
    big = np.repeat(np.repeat(hidden & ~todo, step_y, 0), step_x, 1)
    colour = np.repeat(np.repeat(small, step_y, 0), step_x, 1)
    area = pixels[:big.shape[0], :big.shape[1], :3]
    area[big] = colour[big]
    image.pixels.foreach_set(pixels.ravel())
    # under the picture's own name, in a folder of its own: paint is kept by the picture's name
    os.makedirs(os.path.join(os.path.dirname(IMAGE), "filled"), exist_ok=True)
    image.filepath_raw = os.path.join(os.path.dirname(IMAGE), "filled", os.path.basename(bpy.path.abspath(image.filepath)))
    image.file_format = "PNG"
    image.save()
    return int(hidden.sum())


if hidden.any() and KEEP:
    count = fill(img, face_depth, hidden, KEEP)
    print(f"face picture: {count} of {int((face_depth > 0).sum())} grid texels of the face lay behind something and were filled from the face around them")
# a picture in front with another in front of it: what that one covers is filled the same way
# (the hair under the visor's rim), with nothing to keep: the eyes and mouth are the face's
for at in sorted(front_images)[:-1]:
    own = depths([i for i in over if level[i] == at])
    covered = (own > 0) & (depths([i for i in over if level[i] > at]) > 0)
    if covered.any():
        print(f"front picture {at}: {fill(front_images[at], own, covered)} grid texels lay behind a piece further in front and were filled")

normalize.export_glb(Path(OUT), rig)
print("wrote", OUT)
