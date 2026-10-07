"""Take a finished .vrm in as it is, so the later steps can add to it instead of building it again.

A build starts from a bare surface and makes everything. A .vrm already has a skeleton, weights,
expressions, springs and toon materials, and somebody chose each of them: read with the VRM Add-on
they all come into Blender as the add-on's own data, and the export writes them back. Nothing here
normalizes, joins, welds or rescales: the file's meshes stay separate, at the size and in the pose
they were saved in. Only the humanoid bones are renamed to the pipeline's names, which every later
step finds its bones by (renaming a bone renames its vertex group, so weights come along); the
export gives them their own names back.

VRM 1.0 only. A VRM 0.x file reads, but its expressions, springs and look-at are a different
format the export here doesn't write: it is refused rather than exported without them.
"""

import json

import bpy
import numpy as np

from . import fingers, vrm_export
from .rig import BONE_NAMES

# Custom property on the armature of a file taken in as it was: the file's name. A step that has to
# know the body may be several meshes (add_chest_jiggle.py) looks for it.
IMPORTED_KEY = "vtuber_imported_vrm"


class NotImprovable(ValueError):
    """The file can't be taken in as it is (not VRM 1.0, no armature, refused by its licence)."""


def load(path):
    """Import a .vrm into the open scene and hand back (armature, its meshes).

    Raises NotImprovable for a file the add-on won't import or that isn't VRM 1.0. The add-on
    stops at a licence that forbids changes (VRM 0.x metadata); that is not overridden here.
    """
    vrm_export.ensure_vrm_addon()
    before = set(bpy.data.objects)
    if bpy.ops.import_scene.vrm(filepath=str(path)) != {"FINISHED"}:
        raise NotImprovable(f"{path}: the VRM Add-on didn't import it (a licence that forbids changes stops it too)")
    imported = [o for o in bpy.data.objects if o not in before]
    armature = next((o for o in imported if o.type == "ARMATURE"), None)
    if armature is None:
        raise NotImprovable(f"{path}: no skeleton in it")
    extension = armature.data.vrm_addon_extension
    if not str(extension.spec_version).startswith("1"):
        raise NotImprovable(f"{path}: a VRM {extension.spec_version} file; only VRM 1.0 is taken in as it is. "
                            "Convert it to VRM 1.0 first (VRoid Studio and UniVRM both export it)")
    meshes = [o for o in imported if o.type == "MESH" and o.find_armature() == armature]
    if not meshes:
        raise NotImprovable(f"{path}: nothing in it is skinned to the skeleton")
    armature[IMPORTED_KEY] = str(path)
    for mesh in meshes:
        # the pipeline's scripts take the meshes parented to the armature
        if mesh.parent != armature:
            world = mesh.matrix_world.copy()
            mesh.parent = armature
            mesh.matrix_world = world
    origin = look_at_origin(path)
    if origin is not None:
        armature[vrm_export.LOOK_AT_PROPERTY] = origin
    return armature, meshes


def look_at_origin(path):
    """Where the file's eyes look from, as a world point in Blender's axes, or None.

    Read from the file: the VRM Add-on takes lookAt.offsetFromHeadBone in as Blender axes and
    writes it back unconverted (the reference sample's [0.059, 0.061, 0] came back as
    [-0.061, 0.059, 0]), so the export places it itself (vrm_export._place_look_at).
    """
    gltf, _ = vrm_export._read_glb(path)
    vrm = gltf.get("extensions", {}).get("VRMC_vrm", {})
    offset = vrm.get("lookAt", {}).get("offsetFromHeadBone")
    head = vrm.get("humanoid", {}).get("humanBones", {}).get("head", {}).get("node")
    if offset is None or head is None:
        return None
    nodes = gltf["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    world, at = np.eye(4), head
    while at is not None:
        world = vrm_export._node_matrix(nodes[at]) @ world
        at = parent.get(at)
    x, y, z = (world @ np.r_[offset, 1.0])[:3]
    # glTF is Y up with the model facing +Z: its (x, y, z) is Blender's (x, -z, y)
    return [float(x), float(-z), float(y)]


def imported(armature):
    """Whether this armature came from a .vrm taken in as it was (`load`), not from a build."""
    return armature is not None and IMPORTED_KEY in armature


def rename_bones(armature):
    """Give the humanoid bones the pipeline's names (rig.BONE_NAMES, fingers.bone_name); returns
    how many changed. Bones the file doesn't map (hair, skirt, twist helpers) keep their names."""
    human_bones = armature.data.vrm_addon_extension.vrm1.humanoid.human_bones
    ours = dict(BONE_NAMES)
    for side in ("left", "right"):
        for finger in fingers.FINGERS:
            for key, segment in zip(fingers.humanoid_keys(side, finger), fingers._segments(finger)):
                ours[key] = fingers.bone_name(side, finger, segment)
    renamed, was = 0, {}
    for key, name in ours.items():
        bone = getattr(human_bones, key, None)
        current = bone.node.bone_name if bone is not None else ""
        if current and current != name and current in armature.data.bones:
            # a helper already under the name we want would take a ".001" and hide the real bone
            if name in armature.data.bones:
                armature.data.bones[name].name = f"{name}_was"
            armature.data.bones[current].name = name
            was[name] = current
            renamed += 1
    if was:
        armature[vrm_export.BONE_NAMES_PROPERTY] = json.dumps(was)
    return renamed


def holds(armature, meshes):
    """What the file came with, for the report: the numbers a later step is judged against."""
    extension = armature.data.vrm_addon_extension
    presets = extension.vrm1.expressions.preset
    bound = [name for name in ("blink", "aa", "ih", "ou", "ee", "oh", "look_up", "look_down", "look_left", "look_right")
             if len(getattr(presets, name).morph_target_binds)]
    keys = sorted({k.name for m in meshes if m.data.shape_keys for k in m.data.shape_keys.key_blocks[1:]})
    tall = max((m.matrix_world @ v.co).z for m in meshes for v in m.data.vertices)
    low = min((m.matrix_world @ v.co).z for m in meshes for v in m.data.vertices)
    return {"meshes": {m.name: len(m.data.vertices) for m in meshes}, "bones": len(armature.data.bones),
            "height": round(tall - low, 3), "expressions": bound, "custom_expressions": len(extension.vrm1.expressions.custom),
            "shape_keys": len(keys), "springs": len(extension.spring_bone1.springs),
            "colliders": len(extension.spring_bone1.colliders), "look_at": extension.vrm1.look_at.type,
            "materials": len({s.material.name for m in meshes for s in m.material_slots if s.material})}
