"""Put an imported character into the shape the rest of the pipeline expects."""

import bpy
import numpy as np
from mathutils import Matrix

from .meshdata import world_vertices
from .rig import make_active


def prepare(objects, height=1.6):
    """Join the imported meshes into one object, `height` metres tall, standing on z=0 over the origin.

    Transforms are baked into the vertices, and non-mesh objects from the import
    (parent empties, cameras, lights) are removed. Returns the joined mesh object.
    """
    meshes = [obj for obj in objects if obj.type == "MESH"]
    if not meshes:
        raise ValueError("the import contains no meshes")
    for obj in meshes:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
        if obj.data.users > 1:
            obj.data = obj.data.copy()
    for obj in objects:
        if obj.type != "MESH":
            bpy.data.objects.remove(obj)

    make_active(meshes[0])
    for obj in meshes[1:]:
        obj.select_set(True)
    if len(meshes) > 1:
        bpy.ops.object.join()
    mesh = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    verts = world_vertices(mesh)
    low, high = verts.min(axis=0), verts.max(axis=0)
    center = (low + high) / 2
    to_origin = Matrix.Translation((-center[0], -center[1], -low[2]))
    mesh.data.transform(Matrix.Scale(height / float(high[2] - low[2]), 4) @ to_origin)
    mesh.data.update()
    return mesh
