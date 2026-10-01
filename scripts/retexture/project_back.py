"""Project per-view images back onto a GLB's base colour texture.

Each texel takes a weighted blend of the views that see it: weight favours
views facing the surface head-on, and texels hidden in a view (depth test) or
sitting on a depth discontinuity are skipped. Texels no view covers keep the
original colour.

    blender -b --python project_back.py -- model.glb|model.fbx views_dir image_prefix out.glb [out_texture.png]

`image_prefix` picks which images to project, e.g. "color_" (round-trip test)
or "gen_" (diffusion output), read as <views_dir>/<prefix><view name>.png.
"""
import json
import os
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "addon"))
from vtuber_maker.pipeline import normalize  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, VIEWS, PREFIX, OUT_GLB = os.path.abspath(argv[0]), os.path.abspath(argv[1]), argv[2], os.path.abspath(argv[3])
OUT_TEX = os.path.abspath(argv[4]) if len(argv) > 4 and argv[4] else os.path.splitext(OUT_GLB)[0] + "_basecolor.png"
# Optional comma-separated view names that override others where they see, like the face close-up.
PRIORITY_VIEWS = set(argv[5].split(",")) if len(argv) > 5 else set()

FACING_POWER = 4.0
EDGE_WINDOW = 3          # half-size in pixels of the depth-discontinuity test
BORDER_FADE = 0.06       # fraction of the image border faded out (tiles overlap)
FACE_PRIORITY = 50.0     # the dedicated face close-up overrides wider views where it sees

bpy.ops.wm.read_factory_settings(use_empty=True)
rig = normalize.import_unrigged(Path(SRC))
mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH")
mat = mesh.data.materials[0]
nt = mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
base_node = bsdf.inputs["Base Color"].links[0].from_node
base_img = base_node.image
W, H = base_img.size

with open(os.path.join(VIEWS, "views.json")) as f:
    meta = json.load(f)
RES, height = meta["resolution"], meta["height"]
depth_tol = height * 0.004


def pixels(img, w, h):
    buf = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(buf)
    return buf.reshape(h, w, 4)


def bake_emission(socket_builder, name):
    """Bake an emission shader into a float image in UV space."""
    img = bpy.data.images.new(name, W, H, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    img.generated_color = (0, 0, 0, 0)
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    old = out.inputs["Surface"].links[0].from_socket
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(socket_builder(), em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nt.nodes.active = tex
    bpy.ops.object.bake(type="EMIT", margin=2)
    nt.links.new(old, out.inputs["Surface"])
    nt.nodes.remove(em)
    nt.nodes.remove(tex)
    return pixels(img, W, H)


scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 1
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA"
prefs.get_devices()
for d in prefs.devices:
    d.use = True
scene.cycles.device = "GPU"
bpy.context.view_layer.objects.active = mesh
mesh.select_set(True)

geo = nt.nodes.new("ShaderNodeNewGeometry")
pos = bake_emission(lambda: geo.outputs["Position"], "rt_pos")
nrm = bake_emission(lambda: geo.outputs["Normal"], "rt_nrm")
cover = bake_emission(lambda: nt.nodes.new("ShaderNodeRGB").outputs[0], "rt_cov")
nt.nodes.remove(geo)

valid = (cover[..., 3] > 0.5) & (np.linalg.norm(nrm[..., :3], axis=-1) > 0.5)
P = pos[valid][:, :3].astype(np.float64)
N = nrm[valid][:, :3].astype(np.float64)
N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-8)
print(f"texels covered: {valid.sum()} / {W * H}")

tex_px = pixels(base_img, W, H)
orig = tex_px[valid][:, :3].astype(np.float64)
acc = np.zeros_like(orig)
wsum = np.zeros(len(P))


def local_range(z, r):
    """max - min of z over a (2r+1)^2 window, via separable running filters."""
    pad = np.pad(z, r, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, (2 * r + 1, 2 * r + 1))
    return win.max(axis=(2, 3)) - win.min(axis=(2, 3))


def bilinear(img, x, y):
    x0 = np.clip(np.floor(x).astype(int), 0, img.shape[1] - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, img.shape[0] - 2)
    fx = np.clip(x - x0, 0, 1)[:, None]
    fy = np.clip(y - y0, 0, 1)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


used = 0
for view in meta["views"]:
    path = os.path.join(VIEWS, f"{PREFIX}{view['name']}.png")
    if not os.path.exists(path):
        continue
    used += 1
    M = np.array(view["matrix_world"])
    Minv = np.linalg.inv(M)
    half = view["ortho_scale"] / 2
    cam_axis = M[:3, 2]  # camera +Z points back towards the viewer

    local = P @ Minv[:3, :3].T + Minv[:3, 3]
    px = (local[:, 0] / half + 1) * 0.5 * RES - 0.5
    py = (local[:, 1] / half + 1) * 0.5 * RES - 0.5   # rows stored bottom-up
    depth = -local[:, 2]
    inside = (px >= 0) & (px < RES - 1) & (py >= 0) & (py < RES - 1)

    zimg = bpy.data.images.load(os.path.join(VIEWS, f"z_{view['name']}.exr"))
    z = pixels(zimg, RES, RES)[..., 0]
    bpy.data.images.remove(zimg)
    edges = local_range(z, EDGE_WINDOW) > depth_tol * 4

    ix = np.clip(np.rint(px).astype(int), 0, RES - 1)
    iy = np.clip(np.rint(py).astype(int), 0, RES - 1)
    zb = z[iy, ix]
    visible = inside & (zb > 0) & (np.abs(zb - depth) < depth_tol) & ~edges[iy, ix]

    facing = np.clip(N @ cam_axis, 0, 1) ** FACING_POWER
    bx = np.minimum(px, RES - 1 - px) / (RES * BORDER_FADE)
    by = np.minimum(py, RES - 1 - py) / (RES * BORDER_FADE)
    border = np.clip(np.minimum(bx, by), 0, 1)
    priority = FACE_PRIORITY if view["tile"] == "face" or view["name"] in PRIORITY_VIEWS else 1.0
    w = np.where(visible, facing * border * priority, 0.0)

    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"
    col = pixels(img, RES, RES)[..., :3].astype(np.float64)
    bpy.data.images.remove(img)
    sel = w > 1e-4
    acc[sel] += bilinear(col, px[sel], py[sel]) * w[sel, None]
    wsum[sel] += w[sel]
    print(f"{view['name']}: {sel.sum()} texels")

blend = np.clip(wsum / 0.05, 0, 1)[:, None]
new = np.where(wsum[:, None] > 1e-6, acc / np.maximum(wsum, 1e-6)[:, None], orig)
result = orig * (1 - blend) + new * blend
print(f"projected {used} views, texels reached: {(wsum > 1e-4).sum()}")

out_px = tex_px.copy()
out_px[valid, :3] = result
out_img = bpy.data.images.new("retextured", W, H, alpha=True)
out_img.colorspace_settings.name = "Non-Color"
out_img.pixels.foreach_set(out_px.ravel())
out_img.filepath_raw = OUT_TEX
out_img.file_format = "PNG"
out_img.save()

final = bpy.data.images.load(OUT_TEX)
final.colorspace_settings.name = "sRGB"
base_node.image = final
normalize.export_glb(Path(OUT_GLB), rig)
print("wrote", OUT_GLB)
