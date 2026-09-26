"""Render front, left, back and right views of a textured model for image-to-3D multiview input.

    blender -b -P scripts/render_turnaround.py -- model.glb out_dir [--size 2048]

Views are orthographic, flat-lit (colours straight from the texture) and framed identically,
so they agree with each other exactly, unlike hand-drawn turnaround sheets. Writes
transparent PNGs 1_front, 2_left (the character's left), 3_back, 4_right.
Assumes the character stands on its feet facing -Y, as generators export them.
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

IMPORTERS = {
    ".glb": lambda path: bpy.ops.import_scene.gltf(filepath=path),
    ".gltf": lambda path: bpy.ops.import_scene.gltf(filepath=path),
    ".fbx": lambda path: bpy.ops.import_scene.fbx(filepath=path),
    ".obj": lambda path: bpy.ops.wm.obj_import(filepath=path),
}
MARGIN = 1.04  # frame slightly larger than the model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--size", type=int, default=2048)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    IMPORTERS[args.model.suffix.lower()](str(args.model.resolve()))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector([min(c[i] for c in corners) for i in range(3)])
    hi = Vector([max(c[i] for c in corners) for i in range(3)])
    centre, extent = (lo + hi) / 2, hi - lo

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "TEXTURE"
    scene.view_settings.view_transform = "Standard"
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = scene.render.resolution_y = args.size

    data = bpy.data.cameras.new("turnaround")
    data.type = "ORTHO"
    data.ortho_scale = max(extent) * MARGIN
    camera = bpy.data.objects.new("turnaround", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    distance = max(extent) * 3
    # the character faces -Y, so its left is +X
    views = {
        "1_front": ((0, -distance), 0),
        "2_left": ((distance, 0), math.pi / 2),
        "3_back": ((0, distance), math.pi),
        "4_right": ((-distance, 0), -math.pi / 2),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, ((dx, dy), yaw) in views.items():
        camera.location = Vector((centre.x + dx, centre.y + dy, centre.z))
        camera.rotation_euler = (math.pi / 2, 0, yaw)
        scene.render.filepath = str((args.out_dir / f"{name}.png").resolve())
        bpy.ops.render.render(write_still=True)
        print("rendered", scene.render.filepath)


main()
