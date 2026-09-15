"""Build a VRM humanoid armature from detected joints and skin a mesh to it.

Assumes the character stands on z=0 facing -Y.
"""

import bpy
from mathutils import Vector

# VRM humanoid bone key (as named by the VRM add-on) -> Blender bone name.
BONE_NAMES = {
    "hips": "Hips",
    "spine": "Spine",
    "chest": "Chest",
    "upper_chest": "UpperChest",
    "neck": "Neck",
    "head": "Head",
    "left_shoulder": "LeftShoulder",
    "left_upper_arm": "LeftUpperArm",
    "left_lower_arm": "LeftLowerArm",
    "left_hand": "LeftHand",
    "right_shoulder": "RightShoulder",
    "right_upper_arm": "RightUpperArm",
    "right_lower_arm": "RightLowerArm",
    "right_hand": "RightHand",
    "left_upper_leg": "LeftUpperLeg",
    "left_lower_leg": "LeftLowerLeg",
    "left_foot": "LeftFoot",
    "right_upper_leg": "RightUpperLeg",
    "right_lower_leg": "RightLowerLeg",
    "right_foot": "RightFoot",
}

# Where the spine joints sit along hips -> neck.
SPINE_AT, CHEST_AT, UPPER_CHEST_AT = 0.25, 0.55, 0.8

# Resolution of the skinning proxy, in voxels per character height.
PROXY_VOXELS = 160


def build_armature(joints, name="Armature"):
    """Create and return an armature object with one bone per VRM humanoid joint."""
    j = {key: Vector(value) for key, value in joints.items()}
    hips, neck, head = j["hips"], j["neck"], j["head"]
    spine, chest, upper_chest = (hips.lerp(neck, t) for t in (SPINE_AT, CHEST_AT, UPPER_CHEST_AT))

    # bone key -> (head, tail, parent key); parents are listed before their children.
    segments = {
        "hips": (hips, spine, None),
        "spine": (spine, chest, "hips"),
        "chest": (chest, upper_chest, "spine"),
        "upper_chest": (upper_chest, neck, "chest"),
        "neck": (neck, head, "upper_chest"),
        "head": (head, head + Vector((0.0, 0.0, 2.5 * (head - neck).length)), "neck"),
    }
    for side in ("left", "right"):
        upper_arm, lower_arm, hand = j[f"{side}_upper_arm"], j[f"{side}_lower_arm"], j[f"{side}_hand"]
        clavicle = Vector((neck.x + 0.2 * (upper_arm.x - neck.x), upper_arm.y, (upper_arm.z + neck.z) / 2))
        segments[f"{side}_shoulder"] = (clavicle, upper_arm, "upper_chest")
        segments[f"{side}_upper_arm"] = (upper_arm, lower_arm, f"{side}_shoulder")
        segments[f"{side}_lower_arm"] = (lower_arm, hand, f"{side}_upper_arm")
        segments[f"{side}_hand"] = (hand, hand + 0.4 * (hand - lower_arm), f"{side}_lower_arm")

        upper_leg, lower_leg, foot = j[f"{side}_upper_leg"], j[f"{side}_lower_leg"], j[f"{side}_foot"]
        toe = Vector((foot.x, foot.y - 0.35 * (lower_leg - foot).length, foot.z * 0.3))
        segments[f"{side}_upper_leg"] = (upper_leg, lower_leg, "hips")
        segments[f"{side}_lower_leg"] = (lower_leg, foot, f"{side}_upper_leg")
        segments[f"{side}_foot"] = (foot, toe, f"{side}_lower_leg")

    data = bpy.data.armatures.new(name)
    armature = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(armature)
    make_active(armature)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = {}
    for key, (head_point, tail_point, parent) in segments.items():
        bone = data.edit_bones.new(BONE_NAMES[key])
        bone.head, bone.tail = head_point, tail_point
        if parent is not None:
            bone.parent = bones[parent]
            bone.use_connect = (bones[parent].tail - head_point).length < 1e-6
        bones[key] = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def skin(mesh_obj, armature_obj):
    """Weight the mesh to the armature and deform it with an Armature modifier.

    Automatic (bone heat) weights fail on the self-intersecting surfaces that
    generators produce, so weights are solved on a voxel-remeshed proxy and
    copied back onto the original vertices.
    """
    proxy = mesh_obj.copy()
    proxy.data = mesh_obj.data.copy()
    proxy.name = f"{mesh_obj.name}_skin_proxy"
    proxy.modifiers.clear()
    proxy.vertex_groups.clear()
    bpy.context.scene.collection.objects.link(proxy)

    make_active(proxy)
    proxy.data.remesh_voxel_size = mesh_obj.dimensions.z / PROXY_VOXELS
    bpy.ops.object.voxel_remesh()

    make_active(armature_obj)
    proxy.select_set(True)
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")

    make_active(mesh_obj)
    mesh_obj.vertex_groups.clear()
    transfer = mesh_obj.modifiers.new("skin_transfer", "DATA_TRANSFER")
    transfer.object = proxy
    transfer.use_vert_data = True
    transfer.data_types_verts = {"VGROUP_WEIGHTS"}
    transfer.vert_mapping = "POLYINTERP_NEAREST"
    transfer.layers_vgroup_select_src = "ALL"
    transfer.layers_vgroup_select_dst = "NAME"
    bpy.ops.object.datalayout_transfer(modifier=transfer.name)
    bpy.ops.object.modifier_apply(modifier=transfer.name)
    bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
    bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)

    mesh_obj.parent = armature_obj
    mesh_obj.matrix_parent_inverse = armature_obj.matrix_world.inverted()
    deform = mesh_obj.modifiers.new("Armature", "ARMATURE")
    deform.object = armature_obj

    proxy_mesh = proxy.data
    bpy.data.objects.remove(proxy)
    bpy.data.meshes.remove(proxy_mesh)


def make_active(obj):
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
