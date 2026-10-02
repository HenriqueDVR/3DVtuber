"""Fill in the VRM add-on's data on an armature and export a .vrm file."""

import json
import struct

import addon_utils
import bpy
import numpy as np
from mathutils import Vector

from . import fingers, texture_space
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

# VRM look expression -> ARKit eyeLook keys that slide both eyes that way. The character's left
# is +X: looking left moves the left eye out and the right eye in.
LOOK_SHAPE_KEYS = {
    "look_left": ("eyeLookOutLeft", "eyeLookInRight"),
    "look_right": ("eyeLookInLeft", "eyeLookOutRight"),
    "look_up": ("eyeLookUpLeft", "eyeLookUpRight"),
    "look_down": ("eyeLookDownLeft", "eyeLookDownRight"),
}

# Where the eyes look from, kept on the armature for the export (world, Blender axes).
LOOK_AT_PROPERTY = "vtuber_maker_look_at"

# MToon shadows: the texture darkened with a slight cool tint, like cel-shaded anime art.
SHADE_TINT = (0.78, 0.74, 0.86)
UNTEXTURED_SHADE = 0.75


class VrmAddonMissing(RuntimeError):
    """The VRM Add-on for Blender isn't installed."""


class ExportMismatch(RuntimeError):
    """The written .vrm doesn't hold the armature's rest pose."""


# How far an exported joint may sit from its bone's rest head (m); a re-posed finger is off by
# millimetres to centimetres, float32 and the matrix round trip by under a micrometre.
EXPORT_TOLERANCE = 1e-4


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
    bind_fingers(armature)
    meta = extension.vrm1.meta
    meta.vrm_name = name
    meta.authors.clear()
    meta.authors.add().value = author


def bind_look_expressions(armature, mesh_obj):
    """Point the VRM look expressions at the ARKit eyeLook keys that slide both eyes that way."""
    presets = armature.data.vrm_addon_extension.vrm1.expressions.preset
    for preset, shape_keys in LOOK_SHAPE_KEYS.items():
        expression = getattr(presets, preset)
        expression.morph_target_binds.clear()
        for shape_key in shape_keys:
            if mesh_obj.data.shape_keys is None or shape_key not in mesh_obj.data.shape_keys.key_blocks:
                continue
            bind = expression.morph_target_binds.add()
            bind.node.mesh_object_name = mesh_obj.name
            bind.index = shape_key
            bind.weight = 1.0


def set_look_at(armature, kind, ranges, origin=None):
    """Set how the eyes look: "bone" turns eye bones, "expression" drives the look expressions.

    `ranges` maps horizontal_outer/horizontal_inner/vertical_up/vertical_down to (input max
    degrees, output: bone degrees or expression weight). `origin` (world point, Blender axes) is
    where the eyes look from; the export writes it as lookAt.offsetFromHeadBone.
    """
    look_at = armature.data.vrm_addon_extension.vrm1.look_at
    look_at.type = kind
    for name, (input_max, output) in ranges.items():
        range_map = getattr(look_at, f"range_map_{name}")
        range_map.input_max_value = input_max
        range_map.output_scale = output
    if origin is not None:
        armature[LOOK_AT_PROPERTY] = [float(c) for c in origin]
    elif LOOK_AT_PROPERTY in armature:
        del armature[LOOK_AT_PROPERTY]


def bind_fingers(armature):
    """Map whatever finger bones the armature has to the VRM humanoid (they are all optional)."""
    human_bones = armature.data.vrm_addon_extension.vrm1.humanoid.human_bones
    bound = 0
    for side in ("left", "right"):
        for finger in fingers.FINGERS:
            for key, segment in zip(fingers.humanoid_keys(side, finger), fingers._segments(finger)):
                name = fingers.bone_name(side, finger, segment)
                if name in armature.data.bones and hasattr(human_bones, key):
                    getattr(human_bones, key).node.bone_name = name
                    bound += 1
    return bound


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
            joint.stiffness = chain.stiffness + (chain.tip_stiffness - chain.stiffness) * along
            joint.gravity_power = chain.gravity_power
            joint.drag_force = chain.drag_force
            joint.hit_radius = chain.hit_radius
        spring.add_collider_group().collider_group_uuid = body.uuid


def use_toon_shading(mesh_obj):
    """Switch every material on the mesh to double-sided MToon, shading with a darker tint of its own colors."""
    for slot in mesh_obj.material_slots:
        material = slot.material
        if material is None:
            continue
        image = texture_space.material_base_color_image(material)
        mtoon1 = material.vrm_addon_extension.mtoon1
        mtoon1.enabled = True
        mtoon1.export_shape_key_normals = True
        # generators model capes, sleeves and hat brims as single-sided shells: without this
        # their inside is invisible and gaps between layers show straight through the model
        mtoon1.double_sided = True
        mtoon = mtoon1.extensions.vrmc_materials_mtoon
        if image is not None:
            mtoon1.pbr_metallic_roughness.base_color_texture.index.source = image
            mtoon.shade_multiply_texture.index.source = image
            mtoon.shade_color_factor = SHADE_TINT
        else:
            base = mtoon1.pbr_metallic_roughness.base_color_factor
            mtoon.shade_color_factor = tuple(UNTEXTURED_SHADE * channel for channel in base[:3])


def export(filepath, armature):
    # Export the rest pose as it is. The VRM Add-on's default "Auto" pose makes its own T-pose on
    # export, re-posing the fingers and mesh: every hand baked here (curled, relaxed or straight)
    # left the file with the same generic straightened hand, and tracking looked the same in all.
    armature.data.vrm_addon_extension.vrm1.humanoid.pose = "restPositionPose"
    # the Guided Hand Setup empties are editing handles, not part of the avatar
    guides = [o for o in bpy.context.view_layer.objects if "vtuber_hand_side" in o and not o.hide_get()]
    for guide in guides:
        guide.hide_set(True)
    try:
        result = bpy.ops.export_scene.vrm(filepath=str(filepath), armature_object_name=armature.name)
    finally:
        for guide in guides:
            guide.hide_set(False)
    if result != {"FINISHED"}:
        raise RuntimeError(f"VRM export of {armature.name} to {filepath} returned {result}")
    origin = armature.get(LOOK_AT_PROPERTY)
    if origin is not None:
        # glTF is Y up with the model facing +Z: Blender's (x, y, z) is (x, z, -y) there
        origin = (armature.matrix_world @ Vector(tuple(origin)))
        origin = (origin.x, origin.z, -origin.y)
    normalize_bones(filepath, look_at_origin=origin)
    verify(filepath, armature)


def verify(filepath, armature):
    """Check a written .vrm against the armature it came from: every joint unrotated, at its
    bone's rest head. Raises ExportMismatch naming the worst joints.

    The .blend can be right and the file wrong: the add-on's "Auto" pose replaced the baked rest
    pose in every export until 2026-09-22, and a session of finger variants all tracked alike
    before anyone looked in the file. Joints are matched to bones by name; assumes the armature
    is in the rest pose the file was exported with (it is, straight after `export`).
    """
    gltf, _ = _read_glb(filepath)
    nodes = gltf["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    worlds = {}

    def world(i):
        if i not in worlds:
            worlds[i] = (world(parent[i]) if i in parent else np.eye(4)) @ _node_matrix(nodes[i])
        return worlds[i]

    turned, moved = [], []
    for joint in sorted({j for skin in gltf.get("skins", []) for j in skin["joints"]}):
        name = nodes[joint].get("name")
        if not np.allclose(_node_matrix(nodes[joint])[:3, :3], np.eye(3), atol=1e-6):
            turned.append(name)
        bone = armature.data.bones.get(name)
        if bone is None:
            continue
        head = armature.matrix_world @ bone.head_local
        # glTF is Y up with the model facing +Z: Blender's (x, y, z) is (x, z, -y) there
        off = float(np.linalg.norm(world(joint)[:3, 3] - (head.x, head.z, -head.y)))
        if off > EXPORT_TOLERANCE:
            moved.append((off, name))
    if turned or moved:
        moved.sort(reverse=True)
        raise ExportMismatch(
            f"{filepath}: " + "; ".join(filter(None, (
                moved and f"{len(moved)} joints aren't where the rest pose has them (worst: "
                + ", ".join(f"{name} {off * 1000:.1f} mm" for off, name in moved[:3]) + ")",
                turned and f"{len(turned)} joints are rotated ({', '.join(turned[:3])})"))))


def normalize_bones(filepath, look_at_origin=None):
    """Rewrite a .vrm so every joint rests unrotated, its axes the model's own, as VRoid exports.

    The add-on writes Blender's bone frames (Y down the bone, each finger node turned ~120
    degrees). Warudo's MediaPipe finger tracking turns finger bones about their own axes as if
    they were the model's (a VRoid hand, with no node rotations, tracks cleanly), so on ours
    every finger twisted, the middle ones bunched and the little finger splayed (2026-09-22).
    Joints keep their world positions, the skin's inverse bind matrices are rebuilt so the mesh
    is untouched, and spring-bone collider offsets (joint-local) are carried over. Refuses a file
    with node constraints, whose axes are joint-local too.
    """
    gltf, binary = _read_glb(filepath)
    nodes = gltf["nodes"]
    if any("VRMC_node_constraint" in n.get("extensions", {}) for n in nodes):
        raise ValueError("node constraints would need their axes rewritten; not normalizing")
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    old = [None] * len(nodes)

    def world(i):
        if old[i] is None:
            old[i] = (world(parent[i]) if i in parent else np.eye(4)) @ _node_matrix(nodes[i])
        return old[i]

    for i in range(len(nodes)):
        world(i)
    joints = {j for skin in gltf.get("skins", []) for j in skin["joints"]}
    new = [np.eye(4) for _ in nodes]
    for i in range(len(nodes)):
        if i in joints:
            new[i][:3, 3] = old[i][:3, 3]
        else:
            new[i] = old[i]
    for i, node in enumerate(nodes):
        local = (np.linalg.inv(new[parent[i]]) if i in parent else np.eye(4)) @ new[i]
        for key in ("matrix", "translation", "rotation", "scale"):
            node.pop(key, None)
        if not np.allclose(local, np.eye(4), atol=1e-9):
            node["translation"] = local[:3, 3].tolist()
            if not np.allclose(local[:3, :3], np.eye(3), atol=1e-7):
                node["matrix"] = local.T.reshape(-1).tolist()  # non-joints may carry rotation or scale
                del node["translation"]
    for skin in gltf.get("skins", []):
        if "inverseBindMatrices" not in skin:
            continue
        view, offset, count = _accessor_span(gltf, skin["inverseBindMatrices"])
        ibm = np.frombuffer(binary, np.float32, count * 16, offset).reshape(count, 4, 4).transpose(0, 2, 1)
        fixed = [np.linalg.inv(new[j]) @ old[j] @ m for j, m in zip(skin["joints"], ibm)]
        data = np.array(fixed, np.float32).transpose(0, 2, 1).tobytes()
        binary = binary[:offset] + data + binary[offset + len(data):]
    move = lambda i, p: (np.linalg.inv(new[i]) @ old[i] @ np.r_[p, 1.0])[:3].tolist()
    for collider in gltf.get("extensions", {}).get("VRMC_springBone", {}).get("colliders", []):
        for shape in collider["shape"].values():
            for key in ("offset", "tail"):
                if key in shape:
                    shape[key] = move(collider["node"], shape[key])
    _place_look_at(gltf.get("extensions", {}).get("VRMC_vrm"), new, look_at_origin)
    _write_glb(filepath, gltf, binary)


def _place_look_at(vrm, worlds, origin):
    """Set lookAt.offsetFromHeadBone: from the head joint to `origin` (glTF axes).

    Measured in the file, after the joints are normalized, so the head's own frame is the
    model's. The VRM Add-on's preview reads this field in Blender's axes but writes it
    unconverted, so it isn't set through the add-on.
    """
    bones = (vrm or {}).get("humanoid", {}).get("humanBones", {})
    if origin is None or "lookAt" not in (vrm or {}) or "head" not in bones:
        return
    vrm["lookAt"]["offsetFromHeadBone"] = (np.asarray(origin, float) - worlds[bones["head"]["node"]][:3, 3]).tolist()


def _node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    rotation = np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                         (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                         (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * np.array(node.get("scale", (1.0, 1.0, 1.0)))
    matrix[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return matrix


def _accessor_span(gltf, index):
    """(bufferView, byte offset in the binary chunk, element count) of a tightly packed accessor."""
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    if view.get("byteStride", 64) != 64 or "sparse" in accessor:
        raise ValueError("unexpected inverse bind matrix layout")
    return view, view.get("byteOffset", 0) + accessor.get("byteOffset", 0), accessor["count"]


def _read_glb(filepath):
    raw = open(filepath, "rb").read()
    magic, _, _ = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67:
        raise ValueError(f"{filepath} is not a binary glTF")
    length, kind = struct.unpack_from("<II", raw, 12)
    gltf = json.loads(raw[20:20 + length])
    binary = b""
    if len(raw) > 20 + length:
        blength, _ = struct.unpack_from("<II", raw, 20 + length)
        binary = raw[28 + length:28 + length + blength]
    return gltf, binary


def _write_glb(filepath, gltf, binary):
    text = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(text) + (8 + len(binary) if binary else 0)
    with open(filepath, "wb") as out:
        out.write(struct.pack("<III", 0x46546C67, 2, total))
        out.write(struct.pack("<II", len(text), 0x4E4F534A) + text)
        if binary:
            out.write(struct.pack("<II", len(binary), 0x004E4942) + binary)


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
