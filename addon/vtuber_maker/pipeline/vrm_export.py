"""Fill in the VRM add-on's data on an armature and export a .vrm file."""

import addon_utils
import bpy
from mathutils import Vector

from . import texture_space
from .face_rig import BLINK_LEFT, BLINK_RIGHT, VOWELS
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
            bind = expression.morph_target_binds.add()
            bind.node.mesh_object_name = mesh_obj.name
            bind.index = shape_key
            bind.weight = 1.0


# VRM 1.0 spring settings for hair. Stiffness 0-4, gravity power 0-2, drag 0-1, hit radius in metres.
HAIR_STIFFNESS = 1.0
HAIR_GRAVITY_POWER = 0.4
HAIR_DRAG_FORCE = 0.4
HAIR_HIT_RADIUS = 0.02


def bind_spring_bones(armature, chains, joints):
    """Make each hair chain a VRM spring that collides with the head, shoulders and upper arms."""
    context = bpy.context
    spring_bone = armature.data.vrm_addon_extension.spring_bone1
    body = spring_bone.add_collider_group()
    body.vrm_name = "Body"

    j = {key: Vector(tuple(float(c) for c in value)) for key, value in joints.items()}
    head_length = 2.5 * (j["head"] - j["neck"]).length
    colliders = [
        _sphere_collider(
            context, armature, "head", j["head"] + Vector((0.0, 0.0, 0.55 * head_length)), 0.4 * head_length
        ),
        _capsule_collider(
            context,
            armature,
            "upper_chest",
            j["left_upper_arm"],
            j["right_upper_arm"],
            0.25 * (j["left_upper_arm"] - j["right_upper_arm"]).length / 2,
        ),
    ]
    for side in ("left", "right"):
        shoulder, elbow = j[f"{side}_upper_arm"], j[f"{side}_lower_arm"]
        colliders.append(
            _capsule_collider(context, armature, f"{side}_upper_arm", shoulder, elbow, 0.15 * (elbow - shoulder).length)
        )
    for collider in colliders:
        body.add_collider().collider_uuid = collider.uuid

    for chain in chains:
        spring = spring_bone.add_spring()
        spring.vrm_name = chain[0].rsplit("_", 1)[0]
        for bone_name in chain:
            joint = spring.joints.add()
            joint.node.bone_name = bone_name
            joint.stiffness = HAIR_STIFFNESS
            joint.gravity_power = HAIR_GRAVITY_POWER
            joint.drag_force = HAIR_DRAG_FORCE
            joint.hit_radius = HAIR_HIT_RADIUS
        spring.add_collider_group().collider_group_uuid = body.uuid


def _sphere_collider(context, armature, bone_key, center, radius):
    spring_bone = armature.data.vrm_addon_extension.spring_bone1
    collider = spring_bone.add_collider(context, armature)
    collider.node.bone_name = BONE_NAMES[bone_key]
    collider.shape_type = "Sphere"
    collider.shape.sphere.offset = _in_bone_space(armature, bone_key, center)
    collider.shape.sphere.radius = radius
    return collider


def _capsule_collider(context, armature, bone_key, start, end, radius):
    spring_bone = armature.data.vrm_addon_extension.spring_bone1
    collider = spring_bone.add_collider(context, armature)
    collider.node.bone_name = BONE_NAMES[bone_key]
    collider.shape_type = "Capsule"
    collider.shape.capsule.offset = _in_bone_space(armature, bone_key, start)
    collider.shape.capsule.tail = _in_bone_space(armature, bone_key, end)
    collider.shape.capsule.radius = radius
    return collider


def _in_bone_space(armature, bone_key, point):
    """A rest-pose world point expressed relative to the bone, as VRM collider offsets are."""
    bone = armature.pose.bones[BONE_NAMES[bone_key]]
    return (armature.matrix_world @ bone.matrix).inverted() @ point


# MToon shadows: the texture darkened with a slight cool tint, like cel-shaded anime art.
SHADE_TINT = (0.78, 0.74, 0.86)
UNTEXTURED_SHADE = 0.75


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


def _find_addon_module():
    for module in addon_utils.modules(refresh=False):
        name = module.__name__
        if (name.startswith("bl_ext.") and name.endswith(".vrm")) or name in LEGACY_ADDON_MODULES:
            return name
    return None
