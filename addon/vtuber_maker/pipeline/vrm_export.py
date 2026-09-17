"""Fill in the VRM add-on's data on an armature and export a .vrm file."""

import addon_utils
import bpy
from mathutils import Vector

from . import texture_space
from .face_rig import BLINK_LEFT, BLINK_RIGHT, VOWELS
from .hair_physics import HAIR_TIP_STIFFNESS
from .rig import BONE_NAMES

# Module names the VRM Add-on for Blender can have, depending on how it was installed.
LEGACY_ADDON_MODULES = ("io_scene_vrm", "VRM_Addon_for_Blender-release")

# VRM 1.0 preset expression -> face shape keys it turns fully on.
PRESET_SHAPE_KEYS = {
    "blink": (BLINK_LEFT, BLINK_RIGHT),
    "blink_left": (BLINK_LEFT,),
    "blink_right": (BLINK_RIGHT,),
    **{vowel: (vowel,) for vowel in VOWELS},
}

# MToon shadows: the texture darkened with a slight cool tint, like cel-shaded anime art.
SHADE_TINT = (0.78, 0.74, 0.86)
UNTEXTURED_SHADE = 0.75


class VrmAddonMissing(RuntimeError):
    """The VRM Add-on for Blender isn't installed."""


def ensure_vrm_addon():
    """Enable the VRM add-on if it's installed but off (e.g. under --factory-startup)."""
    if "vrm" in dir(bpy.ops.export_scene):
        return
    module = _find_addon_module()
    if module is None:
        raise VrmAddonMissing("Install 'VRM format' from Blender > Preferences > Get Extensions")
    bpy.ops.preferences.addon_enable(module=module)
    # The add-on's one-time setup runs on its first real scene update and deselects
    # everything. Force that update now (an untouched scene skips it), before callers
    # select objects for operators that need a selection.
    for obj in bpy.context.scene.objects:
        obj.update_tag()
    bpy.context.scene.update_tag()
    bpy.context.view_layer.update()


def configure(armature, name, author):
    """Mark the armature as a VRM 1.0 humanoid with the given avatar name and author."""
    extension = armature.data.vrm_addon_extension
    extension.spec_version = "1.0"
    human_bones = extension.vrm1.humanoid.human_bones
    for key, bone_name in BONE_NAMES.items():
        getattr(human_bones, key).node.bone_name = bone_name
    meta = extension.vrm1.meta
    meta.vrm_name = name
    meta.authors.clear()
    meta.authors.add().value = author


def bind_expressions(armature, mesh_obj):
    """Point the VRM preset expressions at the face shape keys on mesh_obj."""
    presets = armature.data.vrm_addon_extension.vrm1.expressions.preset
    for preset, shape_keys in PRESET_SHAPE_KEYS.items():
        expression = getattr(presets, preset)
        expression.morph_target_binds.clear()
        for shape_key in shape_keys:
            if mesh_obj.data.shape_keys is None or shape_key not in mesh_obj.data.shape_keys.key_blocks:
                continue  # e.g. no blink keys when the eyes are kept as modelled
            bind = expression.morph_target_binds.add()
            bind.node.mesh_object_name = mesh_obj.name
            bind.index = shape_key
            bind.weight = 1.0


def bind_spring_bones(armature, design):
    """Export a hair spring design (see hair_physics.design) as VRM springs and body colliders."""
    context = bpy.context
    spring_bone = armature.data.vrm_addon_extension.spring_bone1
    body = spring_bone.add_collider_group()
    body.vrm_name = "Body"
    for shape in design.colliders:
        collider = spring_bone.add_collider(context, armature)
        collider.node.bone_name = shape.bone
        if shape.tail is None:
            collider.shape_type = "Sphere"
            collider.shape.sphere.offset = _in_bone_space(armature, shape.bone, shape.offset)
            collider.shape.sphere.radius = shape.radius
        else:
            collider.shape_type = "Capsule"
            collider.shape.capsule.offset = _in_bone_space(armature, shape.bone, shape.offset)
            collider.shape.capsule.tail = _in_bone_space(armature, shape.bone, shape.tail)
            collider.shape.capsule.radius = shape.radius
        body.add_collider().collider_uuid = collider.uuid

    for chain in design.chains:
        spring = spring_bone.add_spring()
        spring.vrm_name = chain.names[0].rsplit("_", 1)[0]
        for i, bone_name in enumerate(chain.names):
            joint = spring.joints.add()
            joint.node.bone_name = bone_name
            along = i / max(len(chain.names) - 1, 1)
            joint.stiffness = chain.stiffness + (HAIR_TIP_STIFFNESS - chain.stiffness) * along
            joint.gravity_power = chain.gravity_power
            joint.drag_force = chain.drag_force
            joint.hit_radius = chain.hit_radius
        spring.add_collider_group().collider_group_uuid = body.uuid


def use_toon_shading(mesh_obj):
    """Switch every material on the mesh to MToon, shading with a darker tint of its own colors."""
    for slot in mesh_obj.material_slots:
        material = slot.material
        if material is None:
            continue
        image = texture_space.material_base_color_image(material)
        mtoon1 = material.vrm_addon_extension.mtoon1
        mtoon1.enabled = True
        mtoon1.export_shape_key_normals = True
        mtoon = mtoon1.extensions.vrmc_materials_mtoon
        if image is not None:
            mtoon1.pbr_metallic_roughness.base_color_texture.index.source = image
            mtoon.shade_multiply_texture.index.source = image
            mtoon.shade_color_factor = SHADE_TINT
        else:
            base = mtoon1.pbr_metallic_roughness.base_color_factor
            mtoon.shade_color_factor = tuple(UNTEXTURED_SHADE * channel for channel in base[:3])


def export(filepath, armature):
    result = bpy.ops.export_scene.vrm(filepath=str(filepath), armature_object_name=armature.name)
    if result != {"FINISHED"}:
        raise RuntimeError(f"VRM export of {armature.name} to {filepath} returned {result}")


def _in_bone_space(armature, bone_name, point):
    """A rest-pose world point relative to the bone, as VRM collider offsets are."""
    bone = armature.pose.bones[bone_name]
    return (armature.matrix_world @ bone.matrix).inverted() @ Vector(tuple(float(c) for c in point))


def _find_addon_module():
    for module in addon_utils.modules(refresh=False):
        name = module.__name__
        if (name.startswith("bl_ext.") and name.endswith(".vrm")) or name in LEGACY_ADDON_MODULES:
            return name
    return None
