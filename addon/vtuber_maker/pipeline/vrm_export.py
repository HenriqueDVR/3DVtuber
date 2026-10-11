"""Fill in the VRM add-on's data on an armature and export a .vrm file."""

import json
import os
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

# On the armature of a .vrm taken in as it was (improve.py): {the pipeline's bone name: the name the
# file had}, as JSON. The export gives the joints their own names back.
BONE_NAMES_PROPERTY = "vtuber_maker_bone_names"

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

# Set (to anything) by the app's engine for the steps between the build and its own export step:
# each wrote a .vrm nothing read, 6-9 s and 75-100 MB a step on model A. The .blend is the chain.
SKIP_EXPORT = "VTUBER_MAKER_NO_EXPORT"

# glTF accessor component types as NumPy reads them, and how many numbers an element of each type holds.
COMPONENT_TYPES = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
ELEMENT_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
# Sparse indices are written 32-bit whatever the vertex count: UniVRM 0.115 and earlier read no 8-bit
# ones in VRM 1.0, and an app's UniVRM can't be told from outside.
SPARSE_INDEX_TYPE = 5125
# glTF texture filters. UniVRM reads a LINEAR min filter as "no mipmaps"; a VRoid export carries
# LINEAR_MIPMAP_NEAREST.
GL_LINEAR, GL_LINEAR_MIPMAP_NEAREST = 9729, 9985


def skipped():
    """Whether this run was told to leave the .vrm to a later step."""
    return bool(os.environ.get(SKIP_EXPORT))


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
    uuids = {}
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
        uuids.setdefault(shape.bone, []).append(collider.uuid)

    # a chain that collides with some bones only (hair_physics.design, hair off the body) gets a group of those
    partial = {}
    for chain in design.chains:
        only = getattr(chain, "collides_with", None)
        if only is not None and only not in partial:
            group = spring_bone.add_collider_group()
            group.vrm_name = "+".join(only)
            for bone in only:
                for uuid in uuids.get(bone, []):
                    group.add_collider().collider_uuid = uuid
            partial[only] = group

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
        only = getattr(chain, "collides_with", None)
        spring.add_collider_group().collider_group_uuid = (body if only is None else partial[only]).uuid


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
    if skipped():
        print("export left to a later step:", filepath)
        return
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
    sparse_morph_targets(filepath)
    from . import improve  # it reads this module, so not at the top
    if not improve.imported(armature):
        use_mipmaps(filepath)  # a file taken in as it was keeps the samplers its author gave it
    if armature.get(BONE_NAMES_PROPERTY):
        rename_nodes(filepath, json.loads(armature[BONE_NAMES_PROPERTY]))


def rename_nodes(filepath, names):
    """Rename a written .vrm's nodes ({name now: name wanted}). The humanoid, springs and skins
    name joints by number, so nothing else in the file changes."""
    gltf, binary = _read_glb(filepath)
    for node in gltf["nodes"]:
        node["name"] = names.get(node.get("name"), node.get("name"))
    _write_glb(filepath, gltf, binary)


def verify(filepath, armature):
    """Check a written .vrm against the armature it came from: every joint unrotated, at its
    bone's rest head. Raises ExportMismatch naming the worst joints.

    The .blend can be right and the file wrong: the add-on's "Auto" pose replaced the baked rest
    pose in every export until 2026-09-22, and a session of finger variants all tracked alike
    before anyone looked in the file. Joints are matched to bones by name; assumes the armature
    is in the rest pose the file was exported with (it is, straight after `export`). A joint a
    node constraint is on or reads may rest turned (`normalize_bones` says which and why).
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
    held = constraint_nodes(nodes)
    for joint in sorted({j for skin in gltf.get("skins", []) for j in skin["joints"]}):
        name = nodes[joint].get("name")
        # a joint under one a constraint holds turned is turned back against it, and rests on the
        # model's axes as any other does
        if (joint not in held and not np.allclose(_node_matrix(nodes[joint])[:3, :3], np.eye(3), atol=1e-6)
                and not np.allclose(world(joint)[:3, :3], np.eye(3), atol=1e-5)):
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
    is untouched, and spring-bone collider offsets (joint-local) are carried over.

    Node constraints (a .vrm taken in from elsewhere: twist and aim helpers) name axes in the
    joint's own frame. A roll axis is renamed to the model axis it lies along; where that can't
    say the same thing (an aim joint, whose axis must point at its source at rest; a roll axis off
    every model axis; a rotation copied between two frames that differ) the joints keep the frames
    they had (`_held_by_constraints`).
    """
    gltf, binary = _read_glb(filepath)
    nodes = gltf["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    old = [None] * len(nodes)

    def world(i):
        if old[i] is None:
            old[i] = (world(parent[i]) if i in parent else np.eye(4)) @ _node_matrix(nodes[i])
        return old[i]

    for i in range(len(nodes)):
        world(i)
    joints = {j for skin in gltf.get("skins", []) for j in skin["joints"]}
    joints -= _held_by_constraints(nodes, old, joints)
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


def sparse_morph_targets(filepath):
    """Rewrite a .vrm so each morph target stores only the vertices it moves (glTF sparse
    accessors); returns how many accessors were rewritten.

    The exporter writes every target's position and normal offsets for every vertex of the mesh:
    26 face shapes on a 35 000-vertex body were 21 MB of a 39 MB file, nearly all of it zeros, and
    are 3 MB so (2026-10-11, nine characters: 30-66% of each file, 25-57% smaller after; the fox
    girl's contact sheets the same pixel for pixel and, the user, the same in Warudo). The model is
    untouched: the same accessors with the same counts, min and max, and every target is read back
    from the new data, value for value, before the file is written (ExportMismatch otherwise).

    Indices are 32-bit (SPARSE_INDEX_TYPE); the VRM Add-on's own sparse option, off for the same
    reason, writes the smallest type that fits. A target that moves nothing keeps one entry of
    zeros: a sparse accessor may not be empty. Left as they are: a target that is smaller stored
    whole (it moves most of its vertices), an accessor something other than a morph target reads
    too, and what the exporter doesn't write (interleaved or compressed data, a second buffer).
    Buffer views keep their numbers, so nothing else in the file has to be told: a view that is no
    longer read gives its place to one of the new ones.
    """
    gltf, binary = _read_glb(filepath)
    accessors, views = gltf.get("accessors", []), gltf.get("bufferViews", [])
    if len(gltf.get("buffers", [])) != 1 or any(view.get("buffer", 0) != 0 or "extensions" in view for view in views):
        return 0
    primitives = [p for mesh in gltf.get("meshes", []) for p in mesh.get("primitives", [])]
    targets = {a for p in primitives for target in p.get("targets", []) for a in target.values()}
    targets -= {a for p in primitives for a in (*p.get("attributes", {}).values(), p.get("indices"))}
    targets -= {skin.get("inverseBindMatrices") for skin in gltf.get("skins", [])}
    targets -= {a for animation in gltf.get("animations", []) for sampler in animation.get("samplers", [])
                for a in (sampler.get("input"), sampler.get("output"))}
    whole = {}
    for a in sorted(targets):
        accessor = accessors[a]
        kind, width = COMPONENT_TYPES.get(accessor.get("componentType")), ELEMENT_WIDTHS.get(accessor.get("type"))
        if kind is None or width is None or "bufferView" not in accessor or "sparse" in accessor:
            continue
        view, size = views[accessor["bufferView"]], np.dtype(kind).itemsize * width
        if view.get("byteStride", size) != size:
            continue
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        dense = np.frombuffer(binary, kind, accessor["count"] * width, start).reshape(-1, width)
        rows = np.flatnonzero((dense != 0).any(axis=1)).astype(COMPONENT_TYPES[SPARSE_INDEX_TYPE])
        if not len(rows):
            rows = np.zeros(1, rows.dtype)
        if rows.nbytes + len(rows) * size < dense.nbytes:
            whole[a] = (dense, rows)
    if not whole:
        return 0

    # views only the rewritten targets read are free; whatever else reads a view keeps it
    read = {accessor.get("bufferView") for a, accessor in enumerate(accessors) if a not in whole}
    read |= {part["bufferView"] for accessor in accessors for part in _sparse_parts(accessor)}
    read |= {image.get("bufferView") for image in gltf.get("images", [])}
    free = sorted({accessors[a]["bufferView"] for a in whole} - read, reverse=True)
    fresh = {}

    def place(data):
        index = free.pop() if free else len(views)
        views[index:index + 1] = [{"buffer": 0, "byteLength": len(data)}]
        fresh[index] = data
        return index

    for a, (dense, rows) in whole.items():
        accessor = accessors[a]
        del accessor["bufferView"]
        accessor.pop("byteOffset", None)
        accessor["sparse"] = {"count": len(rows),
                              "indices": {"bufferView": place(rows.tobytes()), "componentType": SPARSE_INDEX_TYPE},
                              "values": {"bufferView": place(dense[rows].tobytes())}}
    packed = bytearray()
    for index, view in enumerate(views):
        start = view.get("byteOffset", 0)
        # where it lay, give or take whole words: what reads it was aligned there
        packed.extend(b"\0" * ((start - len(packed)) % 4))
        view["byteOffset"] = len(packed)
        packed.extend(fresh[index] if index in fresh else binary[start:start + view["byteLength"]])
    packed.extend(b"\0" * (-len(packed) % 4))
    gltf["buffers"][0]["byteLength"] = len(packed)
    packed = bytes(packed)
    changed = [a for a, (dense, _) in whole.items()
               if not np.array_equal(_accessor_array(gltf, packed, a), dense, equal_nan=True)]
    if changed:
        raise ExportMismatch(f"{filepath}: {len(changed)} morph targets don't read back as they were written "
                             f"(accessors {changed[:5]}); the file is left as it was")
    _write_glb(filepath, gltf, packed)
    return len(whole)


def use_mipmaps(filepath):
    """Rewrite a .vrm so its pictures are drawn from mipmaps, as a VRoid export's are; returns how
    many samplers changed.

    The VRM Add-on writes a material's min filter as LINEAR unless told otherwise, and UniVRM reads
    that as "no mipmaps" (TextureSamplerUtil.ImportFilterMode): a 4096 px body picture shown a few
    hundred pixels tall is read a texel in every dozen or more, and thin drawn lines (fur, hair
    strands, gold trim) broke into speckles that crawl as the avatar moves. Seen on the contact
    sheets of the fox girl and the slime, 2026-10-11; with this the fox girl's speckling was gone in
    Warudo and nothing else changed (the user). Close up nothing differs; the pictures take a third
    more memory.

    Only LINEAR changes: NEAREST is somebody's choice (pixel art), and a sampler that names no min
    filter already gets mipmaps. Not called for a .vrm taken in as it was (`export`).
    """
    gltf, binary = _read_glb(filepath)
    plain = [sampler for sampler in gltf.get("samplers", []) if sampler.get("minFilter") == GL_LINEAR]
    for sampler in plain:
        sampler["minFilter"] = GL_LINEAR_MIPMAP_NEAREST
    if plain:
        _write_glb(filepath, gltf, binary)
    return len(plain)


# A roll axis this close (cosine: 5 degrees) to a model axis is renamed to it. VRoid names its
# twist axes by the model axis nearest the limb in the same way, T-posed arms a degree off level.
ROLL_ALONG = 0.9962


def _constraints(nodes):
    """(node, kind, its settings) for every VRMC_node_constraint in the file."""
    for i, node in enumerate(nodes):
        constraint = node.get("extensions", {}).get("VRMC_node_constraint", {}).get("constraint", {})
        for kind in ("roll", "aim", "rotation"):
            if kind in constraint:
                yield i, kind, constraint[kind]


def constraint_nodes(nodes):
    """Nodes a node constraint is on or reads from: the ones that may rest turned in a file."""
    return {n for i, _, c in _constraints(nodes) for n in (i, c.get("source")) if n is not None}


def _held_by_constraints(nodes, worlds, joints):
    """The joints that must keep their rest frame for the file's node constraints to mean what
    they did, with the roll axes of the others renamed to the model's axes (changes `nodes`).

    `worlds`: each node's world matrix as written. Aim: the joint's own axis points at its source
    at rest, so its frame stays. Roll: the axis is the same letter in both joints' frames; when
    both lie along one model axis it takes that axis's name, else both keep their frames. Rotation
    names no axis and copies a turn from one frame into the other: the same thing with both frames
    the model's only when they were equal before.
    """
    def along(i, letter):
        axis = worlds[i][:3, "XYZ".index(letter)]
        axis = axis / np.linalg.norm(axis)
        return int(np.argmax(np.abs(axis))), float(np.max(np.abs(axis)))

    found = list(_constraints(nodes))
    held = {i for i, kind, _ in found if kind == "aim"}
    for i, kind, c in found:
        source = c.get("source")
        if kind == "roll":
            (a, near_a), (b, near_b) = along(i, c["rollAxis"]), along(source, c["rollAxis"])
            if a != b or min(near_a, near_b) < ROLL_ALONG:
                held |= {i, source}
        elif kind == "rotation" and not np.allclose(worlds[i][:3, :3], worlds[source][:3, :3], atol=1e-3):
            held |= {i, source}
    # both ends of a roll or a rotation in the model's frame, or both as they were
    free = lambda n: n in joints and n not in held
    grew = True
    while grew:
        grew = False
        for i, kind, c in found:
            if kind != "aim" and free(i) != free(c["source"]):
                held |= {i, c["source"]}
                grew = True
    for i, kind, c in found:
        if kind == "roll" and free(i):
            c["rollAxis"] = "XYZ"[along(i, c["rollAxis"])[0]]
    return held & joints


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


def _sparse_parts(accessor):
    """Where a sparse accessor keeps its indices and its values (nothing for one stored whole)."""
    sparse = accessor.get("sparse")
    return (sparse["indices"], sparse["values"]) if sparse else ()


def _accessor_array(gltf, binary, index):
    """A tightly packed accessor's elements, one row each, whether it is stored whole or sparse."""
    accessor, views = gltf["accessors"][index], gltf["bufferViews"]
    kind, width, count = COMPONENT_TYPES[accessor["componentType"]], ELEMENT_WIDTHS[accessor["type"]], accessor["count"]
    read = lambda part, numbers_as, numbers: np.frombuffer(  # noqa: E731
        binary, numbers_as, numbers, views[part["bufferView"]].get("byteOffset", 0) + part.get("byteOffset", 0))
    if "bufferView" in accessor:
        array = read(accessor, kind, count * width).reshape(count, width)
    else:
        array = np.zeros((count, width), kind)
    if "sparse" in accessor:
        indices, values = _sparse_parts(accessor)
        array = array.copy()
        rows = read(indices, COMPONENT_TYPES[indices["componentType"]], accessor["sparse"]["count"])
        array[rows] = read(values, kind, len(rows) * width).reshape(-1, width)
    return array


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
