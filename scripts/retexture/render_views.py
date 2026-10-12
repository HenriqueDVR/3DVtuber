"""Render a textured GLB from many orthographic views for re-texturing.

For every view this writes an unlit colour render (the current base colour),
a raw view-depth EXR used later for visibility tests, and a normalised depth
PNG (near = white) for ControlNet. Camera data goes to views.json.

    blender -b --python render_views.py -- model.glb|model.fbx out_dir [resolution]
            [--only head_000,torso_180] [--depth-of head_000]

`--only`: draw these views alone (all 34 without it). `--depth-of`: keep the depth, EXR and PNG, of
these alone (of every view drawn without it). The app's face patch asks for the nine views and the one
depth it reads (app/cli.py PATCH_VIEWS): drawn whole for every new character, the other 25 views and
33 depths were read by nothing and took 9 of the face patch's 18 seconds.
"""
import json
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "addon"))
from vtuber_maker.pipeline import normalize  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, OUT = os.path.abspath(argv[0]), os.path.abspath(argv[1])
RES = int(argv[2]) if len(argv) > 2 and not argv[2].startswith("--") else 1024
# --head-at F: the head view centred at this fraction of the model's height (the survey's head).
# Placed by proportion (0.88) it framed the top of whatever is highest: over model C's tall tail the
# face sat at the frame's bottom edge, cut at the mouth.
HEAD_AT = float(argv[argv.index("--head-at") + 1]) if "--head-at" in argv else 0.88
# --body F: the body's share of the model's height, when horns or a tail stand over the head: the head
# and face views are sized by the body, or the face is small in a picture of mostly horns.
BODY = float(argv[argv.index("--body") + 1]) if "--body" in argv else 1.0
ONLY = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None
DEPTH_OF = set(argv[argv.index("--depth-of") + 1].split(",")) if "--depth-of" in argv else None
os.makedirs(OUT, exist_ok=True)

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
normalize.import_unrigged(Path(SRC))  # as face_patch.py imports it, so the views line up
mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH")
orig_mat = mesh.data.materials[0]
base_node = _upstream_texture(orig_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].links[0].from_node)

corners = [mesh.matrix_world @ Vector(c) for c in mesh.bound_box]
lo = Vector([min(c[i] for c in corners) for i in range(3)])
hi = Vector([max(c[i] for c in corners) for i in range(3)])
height = hi.z - lo.z
center = (lo + hi) / 2


def emission_material(name, build):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs[0], out.inputs[0])
    nt.links.new(build(nt), em.inputs["Color"])
    return mat


def color_socket(nt):
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = base_node.image
    return tex.outputs["Color"]


def depth_socket(nt):
    return nt.nodes.new("ShaderNodeCameraData").outputs["View Z Depth"]


color_mat = emission_material("rt_color", color_socket)
depth_mat = emission_material("rt_depth", depth_socket)

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT" if bpy.app.version < (5, 0, 0) else "BLENDER_EEVEE"  # renamed back in 5.0
scene.render.resolution_x = scene.render.resolution_y = RES
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
world = bpy.data.worlds.new("rt_world")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]

cam_data = bpy.data.cameras.new("rt_cam")
cam_data.type = "ORTHO"
cam_data.clip_start = 0.001
cam_data.clip_end = height * 20
cam = bpy.data.objects.new("rt_cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

# (name, height fraction of tile centre, ortho scale as fraction of height)
TILES = [("head", HEAD_AT, 0.26 * BODY), ("torso", 0.64, 0.36), ("hips", 0.40, 0.36), ("legs", 0.15, 0.34)]
YAWS = [0, 45, 90, 135, 180, 225, 270, 315]
distance = height * 3

views = []
for tile, frac, scale in TILES:
    for yaw in YAWS:
        a = math.radians(yaw)
        loc = Vector((center.x + math.sin(a) * distance, center.y - math.cos(a) * distance, lo.z + frac * height))
        views.append((f"{tile}_{yaw:03d}", loc, (math.radians(90), 0, a), scale * height, tile, yaw))
views.append(("face_000", Vector((center.x, center.y - distance, lo.z + (HEAD_AT - 0.01) * height)), (math.radians(90), 0, 0), 0.13 * BODY * height, "face", 0))
views.append(("top", Vector((center.x, center.y + 0.02 * height, hi.z + distance)), (0, 0, 0), 0.3 * height, "top", 0))


def render(path, mat, samples, bg_value, exr):
    mesh.data.materials[0] = mat
    scene.eevee.taa_render_samples = samples
    bg.inputs[0].default_value = (bg_value, bg_value, bg_value, 1)
    settings = scene.render.image_settings
    if exr:
        settings.file_format, settings.color_depth, settings.color_mode = "OPEN_EXR", "32", "RGB"
    else:
        settings.file_format, settings.color_depth, settings.color_mode = "PNG", "8", "RGB"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def save_depth_png(exr_path, png_path):
    img = bpy.data.images.load(exr_path)
    px = np.empty(RES * RES * 4, np.float32)
    img.pixels.foreach_get(px)
    z = px.reshape(RES, RES, 4)[..., 0]
    hit = z > 0
    out = np.zeros_like(z)
    if hit.any():
        near, far = z[hit].min(), z[hit].max()
        out[hit] = 0.15 + 0.85 * (far - z[hit]) / max(far - near, 1e-6)
    rgba = np.stack([out, out, out, np.ones_like(out)], -1)
    png = bpy.data.images.new("rt_depth_png", RES, RES)
    png.colorspace_settings.name = "Non-Color"
    png.pixels.foreach_set(rgba.ravel())
    png.filepath_raw = png_path
    png.file_format = "PNG"
    png.save()
    bpy.data.images.remove(png)
    bpy.data.images.remove(img)


unknown = sorted(((ONLY or set()) | (DEPTH_OF or set())) - {view[0] for view in views})
if unknown:
    sys.exit(f"no such view: {', '.join(unknown)} (there are {', '.join(view[0] for view in views)})")

records = []
for name, loc, rot, ortho, tile, yaw in views:
    if ONLY is not None and name not in ONLY:
        continue
    cam.location = loc
    cam.rotation_euler = rot
    cam_data.ortho_scale = ortho
    bpy.context.view_layer.update()
    render(os.path.join(OUT, f"color_{name}.png"), color_mat, 16, 0.5, False)
    if DEPTH_OF is None or name in DEPTH_OF:
        zpath = os.path.join(OUT, f"z_{name}.exr")
        render(zpath, depth_mat, 1, 0.0, True)
        save_depth_png(zpath, os.path.join(OUT, f"depth_{name}.png"))
    records.append({
        "name": name, "tile": tile, "yaw": yaw, "ortho_scale": ortho,
        "matrix_world": [list(r) for r in cam.matrix_world],
    })

mesh.data.materials[0] = orig_mat
with open(os.path.join(OUT, "views.json"), "w") as f:
    json.dump({"resolution": RES, "height": height, "source": SRC, "views": records}, f, indent=1)
print(f"rendered {len(records)} views")
