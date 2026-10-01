"""Put an imported character into the shape the rest of the pipeline expects."""

from pathlib import Path

import bpy
from mathutils import Matrix

from . import compat, manifest, orient
from .meshdata import triangles, world_vertices
from .rig import make_active

# Custom property left on the mesh: which way the import was standing and facing, and
# whether the detection was sure. Written into the .blend, and read back for the build report.
ORIENTATION_KEY = "vtuber_orientation"

IMPORTERS = {
    ".glb": lambda path: bpy.ops.import_scene.gltf(filepath=path),
    ".gltf": lambda path: bpy.ops.import_scene.gltf(filepath=path),
    ".fbx": compat.import_fbx,
    ".obj": lambda path: bpy.ops.wm.obj_import(filepath=path),
}


def import_model(path):
    """Import a model by its extension; the app accepts any file the build step (make_vrm) takes."""
    importer = IMPORTERS.get(path.suffix.lower())
    if importer is None:
        raise ValueError(f"unsupported model type {path.suffix}; use one of {', '.join(IMPORTERS)}")
    importer(str(path.resolve()))


def import_unrigged(path):
    """Import a model for the texture steps (matte, tint, face patch): any file import_model takes,
    a rig that came in it (Tripo's rigged FBX) taken off again, the meshes left where they stood.

    Those steps edit and write the mesh alone, as GLB; the rig stays in the FBX, which adopt.py
    reads. Returns that file -- the model itself, or the one an earlier step's GLB names (see
    manifest.rig_of) -- for export_glb to name in turn, or None.
    """
    path = Path(path)
    old = set(bpy.data.objects)
    import_model(path)
    imported = [obj for obj in bpy.data.objects if obj not in old]
    if any(obj.type == "ARMATURE" for obj in imported):
        for obj in imported:
            if obj.type != "MESH":
                continue
            world = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = world
            for modifier in [m for m in obj.modifiers if m.type == "ARMATURE"]:
                obj.modifiers.remove(modifier)
            obj.vertex_groups.clear()
        for obj in imported:
            if obj.type == "ARMATURE":
                bpy.data.objects.remove(obj)
    return manifest.rig_of(path)


def export_glb(path, rig=None):
    """Write the scene as a GLB for the texture steps; with `rig`, its glTF scene names that file
    (manifest.RIG_KEY), so the app given this GLB still finds the rig."""
    scene = bpy.context.scene
    if rig is not None:
        scene[manifest.RIG_KEY] = str(Path(rig).resolve())
    elif manifest.RIG_KEY in scene:
        del scene[manifest.RIG_KEY]
    bpy.ops.export_scene.gltf(filepath=str(Path(path).resolve()), export_format="GLB", export_extras=rig is not None)


def prepare(objects, height=1.6, upright=True):
    """Join the imported meshes into one object, `height` metres tall, standing on z=0 over the origin.

    Transforms are baked into the vertices, and non-mesh objects from the import
    (parent empties, cameras, lights, a rig) are removed. A rig that came with the model (Tripo's
    rigged FBX) goes with its weights and modifier: adopt.py reads it from the file on its own.
    With `upright`, the character is
    turned to the pipeline's frame (+Z up, facing -Y) first, so a Y-up FBX or a model
    facing away still rigs: see orient.detect. Pass an orient.Orientation to say which
    way it is standing instead of detecting, or False to leave it as imported.
    Returns the joined mesh object, with what was detected under ORIENTATION_KEY.
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
        rigs = [m for m in obj.modifiers if m.type == "ARMATURE" and m.object in objects]
        if rigs:
            # the rig's J_Bip_* groups would otherwise ride along into our skinning and the export
            for m in rigs:
                obj.modifiers.remove(m)
            obj.vertex_groups.clear()
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

    orientation = _turn_upright(mesh, upright)
    mesh[ORIENTATION_KEY] = {
        "up": [float(v) for v in orientation.up],
        "facing": [float(v) for v in orientation.facing],
        "up_sure": bool(orientation.up_sure),
        "facing_sure": bool(orientation.facing_sure),
        "turned": not orientation.is_identity,
    }

    verts = world_vertices(mesh)
    low, high = verts.min(axis=0), verts.max(axis=0)
    center = (low + high) / 2
    to_origin = Matrix.Translation((-center[0], -center[1], -low[2]))
    mesh.data.transform(Matrix.Scale(height / float(high[2] - low[2]), 4) @ to_origin)
    mesh.data.update()
    return mesh


def _turn_upright(mesh, upright):
    """Rotate the mesh into the pipeline's frame; returns the orientation it was found in."""
    if upright is False:
        return orient.Orientation(orient.AXES[2], -orient.AXES[1])
    orientation = upright if isinstance(upright, orient.Orientation) else orient.detect(
        world_vertices(mesh), triangles(mesh.data)
    )
    if not orientation.is_identity:
        rotation = Matrix([list(row) for row in orientation.matrix]).to_4x4()
        mesh.data.transform(rotation)
        mesh.data.update()
    return orientation


def orientation_of(mesh):
    """What prepare found, as a plain dict (empty when the mesh wasn't prepared here)."""
    stored = mesh.get(ORIENTATION_KEY)
    return {k: (list(v) if hasattr(v, "__len__") and not isinstance(v, str) else v)
            for k, v in (stored.to_dict() if hasattr(stored, "to_dict") else stored or {}).items()}
