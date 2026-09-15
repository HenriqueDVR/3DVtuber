"""Headless preview renders, for checking pipeline output by eye."""

import math

import bpy
import numpy as np
from mathutils import Vector

# View name -> (camera direction from target, camera rotation)
VIEWS = {
    "front": ((0.0, -1.0, 0.0), (math.pi / 2, 0.0, 0.0)),
    "back": ((0.0, 1.0, 0.0), (math.pi / 2, 0.0, math.pi)),
    "side": ((1.0, 0.0, 0.0), (math.pi / 2, 0.0, math.pi / 2)),
}


def render(filepath, view="front", size=768, focus=None):
    """Render visible meshes with flat texture shading from an orthographic camera.

    focus: optional (center, extent) to zoom into a region such as the face.
    """
    scene = bpy.context.scene
    if focus is None:
        meshes = [o for o in scene.objects if o.type == "MESH" and o.visible_get()]
        corners = np.array([list(o.matrix_world @ Vector(c)) for o in meshes for c in o.bound_box])
        low, high = corners.min(axis=0), corners.max(axis=0)
        center = Vector(((low + high) / 2).tolist())
        extent = float((high - low).max()) * 1.1
    else:
        center, extent = Vector(focus[0]), focus[1]

    direction, rotation = VIEWS[view]
    camera_data = bpy.data.cameras.new("preview")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = extent
    camera_data.clip_end = 100.0
    camera = bpy.data.objects.new("preview", camera_data)
    scene.collection.objects.link(camera)
    camera.location = center + Vector(direction) * 10.0
    camera.rotation_euler = rotation

    previous_camera = scene.camera
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "TEXTURE"
    # Keep texture colors as painted, so renders can be checked pixel by pixel.
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.film_transparent = True
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

    scene.camera = previous_camera
    bpy.data.objects.remove(camera)
    bpy.data.cameras.remove(camera_data)
