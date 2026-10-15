"""Put an imported character into the shape the rest of the pipeline expects."""

import os
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

from . import compat, manifest, orient
from .meshdata import sample_surface, triangles, world_vertices
from .rig import make_active

# Custom property left on the mesh: which way the import was standing and facing, and
# whether the detection was sure. Written into the .blend, and read back for the build report.
ORIENTATION_KEY = "vtuber_orientation"
# Set to "file" for a Blender run (app/engine.py, from the manifest's `facing`): see _turn_upright.
# An environment variable, not an argument: five scripts prepare the model, and one that missed
# the word would build the character facing the other way from the rest.
FACING_AS_FILED = "VTUBER_MAKER_FACING"

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
    meshes = [obj for obj in imported if obj.type == "MESH"]
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
    # One mesh, as the build joins them (prepare): the texture steps read "the mesh", and of a model
    # exported in parts (Tripo's segmentation: 171 objects) the first was a speck of 3 mm, which the
    # views framed and the face patch searched for a face.
    if len(meshes) > 1:
        for obj in meshes:
            if obj.data.users > 1:
                obj.data = obj.data.copy()
        make_active(meshes[0])
        for obj in meshes[1:]:
            obj.select_set(True)
        bpy.ops.object.join()
    return manifest.rig_of(path)


# A picture's alpha counts as none with this share of the corners that use it opaque (kept.OPAQUE_SHARE).
OPAQUE_SHARE = 0.99


def opaque_where_used(objects):
    """Take the alpha off every material whose picture is opaque wherever its polygons use it;
    returns the names of those materials. Call before writing a GLB the app will show.

    An FBX whose texture also cuts it imports with the picture's alpha wired in, and the GLB then
    says BLEND of the whole model. A tester's flat-colour model (a palette, its unused cells clear)
    was drawn in the app without depth: the hair down the back over the face, slivers of what lies
    behind at every edge. Read at the polygons' corners, since the rest of a picture is often empty.
    A picture clear anywhere it is used (lashes, a cut-out hem) keeps its alpha. Materials whose
    alpha comes from no picture are left alone.
    """
    done = []
    for obj in objects:
        data = obj.data if obj.type == "MESH" else None
        if data is None or data.uv_layers.active is None or not len(data.polygons):
            continue
        material, total = np.empty(len(data.polygons), np.int32), np.empty(len(data.polygons), np.int32)
        data.polygons.foreach_get("material_index", material)
        data.polygons.foreach_get("loop_total", total)
        of_loop = np.repeat(material, total)
        uv = np.empty(len(data.loops) * 2, np.float32)
        data.uv_layers.active.data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2)
        for index, slot in enumerate(obj.material_slots):
            tree = slot.material.node_tree if slot.material else None
            bsdf = tree.nodes.get("Principled BSDF") if tree else None
            links = [link for link in bsdf.inputs["Alpha"].links if link.from_node.type == "TEX_IMAGE"] if bsdf else []
            image = links[0].from_node.image if links else None
            at = uv[of_loop == index]
            if image is None or not len(at) or not image.size[0] or slot.material.name in done:
                continue
            width, height = image.size
            if image.channels == 4:
                pixels = np.empty(width * height * 4, np.float32)
                image.pixels.foreach_get(pixels)
                alpha = pixels.reshape(height, width, 4)[..., 3]
                at = at % 1.0
                used = alpha[np.clip((at[:, 1] * height).astype(int), 0, height - 1), np.clip((at[:, 0] * width).astype(int), 0, width - 1)]
                if (used > 0.95).mean() < OPAQUE_SHARE:
                    continue
            for link in links:
                tree.links.remove(link)
            bsdf.inputs["Alpha"].default_value = 1.0
            if hasattr(slot.material, "blend_method"):
                slot.material.blend_method = "OPAQUE"
            done.append(slot.material.name)
    return done


def export_glb(path, rig=None):
    """Write the scene as a GLB for the texture steps; with `rig`, its glTF scene names that file
    (manifest.RIG_KEY), so the app given this GLB still finds the rig."""
    scene = bpy.context.scene
    opaque_where_used(scene.objects)
    if rig is not None:
        scene[manifest.RIG_KEY] = str(Path(rig).resolve())
    elif manifest.RIG_KEY in scene:
        del scene[manifest.RIG_KEY]
    bpy.ops.export_scene.gltf(filepath=str(Path(path).resolve()), export_format="GLB", export_extras=rig is not None)


def prepare(objects, height=1.6, upright=True):
    """Join the imported meshes into one object, `height` metres tall, standing on z=0 over the origin.

    The height is the head's: a tail held up above it (orient.head_top) stands over `height`.

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
    # to the top of the head: a fox tail ending 37 cm above it made the body four fifths its size,
    # and every joint placed by a share of the height (hips, neck, head) sat 25 to 40 cm off
    top = orient.head_top(sample_surface(verts, triangles(mesh.data), orient.SAMPLES), verts=verts) or float(high[2])
    mesh.data.transform(Matrix.Scale(height / float(top - low[2]), 4) @ to_origin)
    mesh.data.update()
    return mesh


def _turn_upright(mesh, upright):
    """Rotate the mesh into the pipeline's frame; returns the orientation it was found in."""
    if upright is False:
        return orient.Orientation(orient.AXES[2], -orient.AXES[1])
    orientation = upright if isinstance(upright, orient.Orientation) else orient.detect(
        world_vertices(mesh), triangles(mesh.data)
    )
    if os.environ.get(FACING_AS_FILED) == "file" and orientation is not upright and np.allclose(orientation.up, orient.AXES[2]):
        # the user's word against the detector's (manifest option `facing`): a model standing
        # upright in its file faces the way the file has it. One lying down has no such way.
        orientation = orient.Orientation(orientation.up, -orient.AXES[1], orientation.up_sure, True)
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
