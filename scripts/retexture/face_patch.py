"""Give the front of the face its own texture, mapped straight from one view.

Tripo's UV layout spends few texels on the face, so projecting a sharp image
into the shared atlas blurs it. This moves the polygons a view sees head-on
onto a new material whose UVs are that view's screen coordinates, so the
image is used at full resolution.

    blender -b --python face_patch.py -- model.glb views_dir view_name image.png out.glb
"""
import json
import os
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
normalize.import_model(Path(SRC))
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
chosen = []
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
    chosen.append(poly.index)
print(f"face patch polygons: {len(chosen)}")

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

bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB")
print("wrote", OUT)
