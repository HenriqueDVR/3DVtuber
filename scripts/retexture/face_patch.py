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
from vtuber_maker.pipeline import normalize  # noqa: E402

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


# the farthest patch surface under every texel of the picture
G = LAYER_GRID
farthest = np.zeros((G, G), np.float32)
taken = set(chosen)
mesh.calc_loop_triangles()
seen = {}
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
    part = farthest[ya:yb + 1, xa:xb + 1]
    part[inside] = np.maximum(part[inside], depth[inside])
over = [i for i in chosen
        if any(farthest[min(int(y * G), G - 1), min(int(x * G), G - 1)] - z > height * LAYER for x, y, z in _in(mesh.polygons[i]))]
print(f"face patch polygons in front of others (own picture): {len(over)}")

base = mesh.materials[0]
patch = base.copy()
patch.name = "face_patch"
bsdf = patch.node_tree.nodes["Principled BSDF"]
tex_node = _upstream_texture(bsdf.inputs["Base Color"].links[0].from_node)
for link in list(bsdf.inputs["Normal"].links):   # the normal map follows the atlas UVs, not the patch's
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

if over:
    # the same picture under another name: one image used twice would be one set of texels again
    copy = os.path.splitext(IMAGE)[0] + "_over.png"
    shutil.copyfile(IMAGE, copy)
    front = patch.copy()
    front.name = OVER_MATERIAL
    front_node = _upstream_texture(front.node_tree.nodes["Principled BSDF"].inputs["Base Color"].links[0].from_node)
    front_image = bpy.data.images.load(copy)
    front_image.colorspace_settings.name = "sRGB"
    front_node.image = front_image
    mesh.materials.append(front)
    for i in over:
        mesh.polygons[i].material_index = len(mesh.materials) - 1

normalize.export_glb(Path(OUT), rig)
print("wrote", OUT)
