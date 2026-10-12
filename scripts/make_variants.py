"""Make transformed copies of a model: the same character as other files would hand it over.

    blender -b --factory-startup -P scripts/_startup.py --python-exit-code 1 -P scripts/make_variants.py --
            <model> <folder> [--rigged <rigged.fbx>] [--only name,name]

A tester's model comes turned another way, in centimetres, in several objects, cut along every edge,
decimated, saved by another program in another format, with its arms down or without its picture.
Each copy here is one such change of `<model>` and nothing else, so that what the app makes of it can
be held against what it makes of the model itself (`python -m app inputs`, app/inputs.py): the same
avatar, or a stop that says why not; never a finished file that is quietly worse.

Writes one file a copy into `<folder>` and `variants.json` there: each copy's name, file, what was
done to it and what is expected of the app ("same", or the codes of app/errors.py one of which should
stop it). `--rigged`: a rigged FBX of a character (Tripo's), for the two copies with their arms
lowered, which take a skeleton to make. Nothing of the pipeline's rules is used but the importers.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "addon"))

from vtuber_maker.pipeline import normalize  # noqa: E402

# How far the lowered arms hang below level (degrees), turned at the upper-arm bones. At 86 they lie
# against the body and the survey can't tell them from it (VM-ARMS-AT-SIDES, from 83 on model A's
# rigged T-pose). At 80 they still stand clear and the survey reads them as down (86 by its own
# measure; survey.ARMS_DOWN is 80): the look before a build stops that one (VM-ARMS-DOWN). At 75 it
# reads 74, an A-pose, and builds.
ARMS_LOWERED, ARMS_LOW = 86.0, 80.0
# Upper-arm bones by the names of the rigs the app reads (adopt.py): Tripo / VRoid, Mixamo.
UPPER_ARMS = (("J_Bip_L_UpperArm", "J_Bip_R_UpperArm"), ("mixamorig:LeftArm", "mixamorig:RightArm"))
HANDS = (("J_Bip_L_Hand", "J_Bip_R_Hand"), ("mixamorig:LeftHand", "mixamorig:RightHand"))


def load(path):
    bpy.ops.wm.read_homefile(use_empty=True)
    normalize.import_model(Path(path))


def meshes():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def joined():
    """Every mesh of the scene as one object, transforms baked in; whatever else came with it is removed."""
    objects = meshes()
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    for obj in objects:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
        obj.select_set(True)
    for obj in [o for o in bpy.context.scene.objects if o.type != "MESH"]:
        bpy.data.objects.remove(obj)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return bpy.context.view_layer.objects.active


def triangles(mesh):
    return sum(len(polygon.vertices) - 2 for polygon in mesh.data.polygons)


# ----- the changes


def turn(axis, degrees):
    def change():
        joined().data.transform(Matrix.Rotation(math.radians(degrees), 4, axis))
    return change


def centimetres():
    joined().data.transform(Matrix.Scale(100.0, 4))


def unwelded():
    """Every polygon with corners of its own: what an exporter that splits at every edge writes."""
    mesh = joined()
    bm = bmesh.new()
    bm.from_mesh(mesh.data)
    bmesh.ops.split_edges(bm, edges=bm.edges[:])
    bm.to_mesh(mesh.data)
    bm.free()
    mesh.data.update()


def decimated(count, welded=False):
    """Blender's own decimation to `count` triangles: of the model as it is, cut along every texture
    seam as a generated one comes, or (`welded`) with the vertices at one place merged first, as
    someone who knew to would do it."""
    def change():
        mesh = joined()
        if welded:
            bm = bmesh.new()
            bm.from_mesh(mesh.data)
            bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-5)
            bm.to_mesh(mesh.data)
            bm.free()
            mesh.data.update()
        modifier = mesh.modifiers.new("decimate", "DECIMATE")
        modifier.ratio = min(1.0, count / max(triangles(mesh), 1))
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    return change


def materials_split(own_pictures):
    """The model's surface under three materials by height, all of the one picture (or, with
    `own_pictures`, each of a copy of it): a model exported with a material a part."""
    def change():
        mesh = joined()
        first = mesh.material_slots[0].material
        for index in (1, 2):
            copy = first.copy()
            copy.name = f"{first.name}_part{index}"
            if own_pictures and copy.node_tree:
                for node in copy.node_tree.nodes:
                    if node.type == "TEX_IMAGE" and node.image is not None:
                        node.image = node.image.copy()
            mesh.data.materials.append(copy)
        low = min(v.co.z for v in mesh.data.vertices)
        high = max(v.co.z for v in mesh.data.vertices)
        for polygon in mesh.data.polygons:
            polygon.material_index = min(2, int(3 * (polygon.center.z - low) / max(high - low, 1e-9)))
    return change


def objects_split():
    """The model in four objects, one of them under a turned and scaled parent it is turned back against:
    what a scene of parts saved as it was modelled looks like."""
    mesh = joined()
    low = min(v.co.z for v in mesh.data.vertices)
    high = max(v.co.z for v in mesh.data.vertices)
    kept = []
    # by height, four bands: each object the polygons of its band, cut from its neighbours where they met
    # (by loose part instead, a generated model's thousands of texture patches are thousands of objects)
    for band in range(4):
        part = mesh.copy()
        part.data = mesh.data.copy()
        part.name = f"part_{band}"
        bpy.context.scene.collection.objects.link(part)
        bm = bmesh.new()
        bm.from_mesh(part.data)
        others = [face for face in bm.faces if min(3, int(4 * (face.calc_center_median().z - low) / max(high - low, 1e-9))) != band]
        bmesh.ops.delete(bm, geom=others, context="FACES")
        bm.to_mesh(part.data)
        bm.free()
        kept.append(part)
    bpy.data.objects.remove(mesh)
    parent = bpy.data.objects.new("Parts", None)
    bpy.context.scene.collection.objects.link(parent)
    parent.matrix_world = Matrix.Translation((0.3, -0.2, 0.1)) @ Matrix.Rotation(math.radians(30.0), 4, "Z") @ Matrix.Scale(2.0, 4)
    bpy.context.view_layer.update()
    child = kept[-1]
    world = child.matrix_world.copy()
    child.parent = parent
    child.matrix_world = world


def no_texture():
    """The surface without its pictures: a model exported as geometry alone."""
    mesh = joined()
    for slot in mesh.material_slots:
        tree = slot.material.node_tree if slot.material else None
        for node in [n for n in tree.nodes if n.type == "TEX_IMAGE"] if tree else []:
            tree.nodes.remove(node)


def unchanged():
    joined()


def arms_lowered(rigged, degrees):
    """A rigged character with its arms brought down to `degrees` below level, the skeleton then taken
    off: what a generator hands over when it wasn't asked for a T-pose."""
    def change():
        load(rigged)
        rig = next((obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"), None)
        if rig is None:
            sys.exit(f"{rigged}: no skeleton to lower the arms with")
        arms = next((names for names in UPPER_ARMS if all(n in rig.pose.bones for n in names)), None)
        hands = next((names for names in HANDS if all(n in rig.pose.bones for n in names)), None)
        if arms is None or hands is None:
            sys.exit(f"{rigged}: its arm bones are under no name known here")
        down = Vector((0.0, 0.0, -1.0))
        for arm, hand in zip(arms, hands):
            bone = rig.pose.bones[arm]
            head = rig.matrix_world @ bone.head
            along = (rig.matrix_world @ rig.pose.bones[hand].head - head).normalized()
            # from where it points to `degrees` below level, in the plane it and straight down share
            level = Vector((along.x, along.y, 0.0)).normalized()
            target = (level * math.cos(math.radians(degrees)) + down * math.sin(math.radians(degrees))).normalized()
            turn_to = along.rotation_difference(target).to_matrix().to_4x4()
            world = Matrix.Translation(head) @ turn_to @ Matrix.Translation(-head) @ rig.matrix_world @ bone.matrix
            bone.matrix = rig.matrix_world.inverted() @ world
            bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj in meshes():
            posed = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))
            obj.modifiers.clear()
            obj.vertex_groups.clear()
            obj.data = posed
        joined()
    return change


# ----- the files


def glb(path):
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB")


def fbx(path):
    bpy.ops.export_scene.fbx(filepath=str(path), path_mode="COPY", embed_textures=True, object_types={"MESH", "EMPTY"})


def obj(path):
    # a model read from a GLB holds its pictures inside the file, and an OBJ names files: written out
    # beside it first (left inside, the .mtl named no picture and the copy was one without its texture)
    for image in bpy.data.images:
        if image.size[0] and image.type == "IMAGE" and (image.packed_file or not image.filepath):
            saved = path.parent / f"{Path(image.name).stem}.png"
            image.file_format = "PNG"
            image.save(filepath=str(saved))
            if image.packed_file:
                image.unpack(method="REMOVE")
            image.filepath = str(saved)
    bpy.ops.wm.obj_export(filepath=str(path), path_mode="RELATIVE")


SAME = "same"


def variants(rigged):
    """(name, change, writer, file type, what it is, what is expected of the app)."""
    found = [
        ("same_glb", unchanged, glb, ".glb", "the model saved again by Blender as GLB, nothing changed", SAME),
        ("as_fbx", unchanged, fbx, ".fbx", "saved as FBX, its picture inside", SAME),
        ("as_obj", unchanged, obj, ".obj", "saved as OBJ, with its .mtl and picture beside it", SAME),
        ("turn_x_90", turn("X", 90.0), glb, ".glb", "turned a quarter about X: lying on its face or back", SAME),
        ("turn_x_270", turn("X", -90.0), glb, ".glb", "turned a quarter the other way about X", SAME),
        ("turn_x_180", turn("X", 180.0), glb, ".glb", "turned half about X: standing on its head", SAME),
        ("turn_y_90", turn("Y", 90.0), glb, ".glb", "turned a quarter about Y: lying on its side", SAME),
        ("turn_z_90", turn("Z", 90.0), glb, ".glb", "turned a quarter about Z: upright, facing sideways", SAME),
        ("turn_z_180", turn("Z", 180.0), glb, ".glb", "turned half about Z: upright, facing away", SAME),
        ("centimetres", centimetres, glb, ".glb", "a hundred times the size: modelled in centimetres", SAME),
        ("materials_split", materials_split(False), glb, ".glb", "three materials by height, all of the one picture", SAME),
        ("materials_own_pictures", materials_split(True), glb, ".glb", "three materials by height, each of its own copy of the picture", SAME),
        ("objects_split", objects_split, glb, ".glb", "four objects, one under a turned, scaled parent", SAME),
        ("unwelded", unwelded, glb, ".glb", "every polygon with corners of its own", SAME),
        ("decimated_20k", decimated(20_000), glb, ".glb", "decimated to 20,000 triangles as it is, cut along its texture seams",
         "VM-FRAGMENTS"),
        ("decimated_5k", decimated(5_000), glb, ".glb", "decimated to 5,000 triangles as it is", "VM-FRAGMENTS"),
        ("decimated_20k_welded", decimated(20_000, welded=True), glb, ".glb", "its seams merged, then decimated to 20,000 triangles", SAME),
        ("decimated_5k_welded", decimated(5_000, welded=True), glb, ".glb", "its seams merged, then decimated to 5,000 triangles", SAME),
        ("no_texture", no_texture, glb, ".glb", "its pictures taken off", "VM-NO-TEXTURE"),
    ]
    if rigged:
        found += [
            ("arms_lowered", arms_lowered(rigged, ARMS_LOWERED), glb, ".glb",
             f"its arms brought down to {ARMS_LOWERED:.0f} degrees below level, against its sides", "VM-ARMS-DOWN VM-ARMS-AT-SIDES"),
            ("arms_low", arms_lowered(rigged, ARMS_LOW), glb, ".glb",
             f"its arms brought down to {ARMS_LOW:.0f} degrees below level, clear of its sides", "VM-ARMS-DOWN VM-ARMS-AT-SIDES"),
        ]
    return found


def main():
    parser = argparse.ArgumentParser(description="Make transformed copies of a model.")
    parser.add_argument("model", type=Path)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--rigged", type=Path, help="a rigged FBX, for the two copies with their arms lowered")
    parser.add_argument("--only", help="names of the copies to make, with commas (default: all)")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    folder = args.folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only.split(",")) if args.only else None
    made = []
    for name, change, write, kind, what, expects in variants(args.rigged):
        if wanted is not None and name not in wanted:
            continue
        load(args.model.resolve())
        change()
        # each in a folder of its own: an OBJ comes with its .mtl and pictures, and two copies' would be one file
        target = folder / name / f"{name}{kind}"
        target.parent.mkdir(parents=True, exist_ok=True)
        write(target)
        counted = meshes()
        made.append({"name": name, "file": str(target), "what": what, "expects": expects,
                     "objects": len(counted), "vertices": sum(len(o.data.vertices) for o in counted),
                     "triangles": sum(triangles(o) for o in counted)})
        print(f"made {name}: {made[-1]['objects']} objects, {made[-1]['vertices']} vertices, {made[-1]['triangles']} triangles")
    (folder / "variants.json").write_text(json.dumps({"model": str(args.model.resolve()), "variants": made}, indent=2), encoding="utf-8")
    print("VTUBER_MAKER_REPORT " + json.dumps({"variants": [v["name"] for v in made]}))


if __name__ == "__main__":
    main()
