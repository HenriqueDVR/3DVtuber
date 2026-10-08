"""Take a finished .vrm in as it is, so the later steps can add to it instead of building it again.

A build starts from a bare surface and makes everything. A .vrm already has a skeleton, weights,
expressions, springs and toon materials, and somebody chose each of them: read with the VRM Add-on
they all come into Blender as the add-on's own data, and the export writes them back. Nothing here
normalizes, joins, welds or rescales: the file's meshes stay separate, at the size and in the pose
they were saved in. Only the humanoid bones are renamed to the pipeline's names, which every later
step finds its bones by (renaming a bone renames its vertex group, so weights come along); the
export gives them their own names back.

A VRM 0.x file is taken in too and leaves as VRM 1.0 (`convert`). The add-on fills the 1.0 side
of its data as it imports one (bones, expressions under their 1.0 names, springs, colliders, toon
materials), turned to face the 1.0 way; what it leaves out is carried over here: the author's name
and terms, the first-person settings and the eyes' range.
"""

import json

import bpy
import numpy as np

from . import fingers, vrm_export
from .rig import BONE_NAMES

# Custom property on the armature of a file taken in as it was: the file's name. A step that has to
# know the body may be several meshes (add_chest_jiggle.py) looks for it.
IMPORTED_KEY = "vtuber_imported_vrm"


# Custom property on the armature of a file that came as VRM 0.x: its spec version.
CONVERTED_KEY = "vtuber_converted_from"


class NotImprovable(ValueError):
    """The file can't be taken in as it is (no VRM, no armature, nothing skinned)."""


class LicenceUnread(NotImprovable):
    """The file's licence has to be read by the user before it is changed; `.detail` says why."""

    def __init__(self, detail):
        super().__init__(detail)
        self.detail = detail


def load(path, licence_read=False):
    """Import a .vrm into the open scene and hand back (armature, its meshes).

    Raises NotImprovable for a file the add-on won't import. The add-on stops at a VRM 0.x file
    whose licence it can't read as allowing changes (no derivatives, or terms on a web page):
    LicenceUnread then, unless `licence_read` says the user has read the terms and may change the
    avatar. That is theirs to say, never assumed here. A VRM 0.x file comes back converted.
    """
    vrm_export.ensure_vrm_addon()
    before = set(bpy.data.objects)
    if bpy.ops.import_scene.vrm(filepath=str(path)) != {"FINISHED"}:
        why = licence_stop(path)
        if why is None:
            raise NotImprovable(f"{path}: the VRM Add-on didn't import it")
        if not licence_read:
            raise LicenceUnread(why)
        # the add-on's own "Import Anyway", which its dialog offers after showing the same terms
        if bpy.ops.wm.vrm_license_warning(filepath=str(path), import_anyway=True, make_new_texture_folder=True,
                                          set_armature_display_to_wire=True) != {"FINISHED"}:
            raise NotImprovable(f"{path}: the VRM Add-on didn't import it")
    imported = [o for o in bpy.data.objects if o not in before]
    armature = next((o for o in imported if o.type == "ARMATURE"), None)
    if armature is None:
        raise NotImprovable(f"{path}: no skeleton in it")
    extension = armature.data.vrm_addon_extension
    if str(extension.spec_version).startswith("0"):
        convert(armature)
    elif not str(extension.spec_version).startswith("1"):
        raise NotImprovable(f"{path}: a {extension.spec_version} file; VRM 1.0 and 0.x are taken in")
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
    origin = look_at_origin(path) if CONVERTED_KEY not in armature else _look_at_origin_0(armature)
    if origin is not None:
        armature[vrm_export.LOOK_AT_PROPERTY] = origin
    return armature, meshes


def licence_stop(path):
    """Why the VRM Add-on wants a VRM 0.x file's licence confirmed before importing it, as one
    line with the address to read, or None when it asks for nothing (and for a VRM 1.0 file, whose
    terms it doesn't judge). The add-on's own rule, asked of the add-on."""
    from bl_ext.user_default.vrm.importer import license_validation
    gltf, _ = vrm_export._read_glb(path)
    try:
        license_validation.validate_license(gltf, (0, 0) if "VRM" in gltf.get("extensions", {}) else (1, 0))
    except license_validation.LicenseConfirmationRequiredError as error:
        said = {(p.message or "", p.url or "") for p in error.props}
        return "; ".join(f"{message} ({url})" if url else message for message, url in sorted(said))
    return None


# VRM 0.x licence names as the 1.0 terms that say no more than they did:
# (credit, redistribution allowed, modification). "Other" names terms on a web page, which stay
# the terms (other_license_url): the fields beside them take the strictest reading.
LICENCES_0 = {
    "CC0": ("unnecessary", True, "allowModificationRedistribution"),
    "CC_BY": ("required", True, "allowModificationRedistribution"),
    "CC_BY_NC": ("required", True, "allowModificationRedistribution"),
    "CC_BY_SA": ("required", True, "allowModificationRedistribution"),
    "CC_BY_NC_SA": ("required", True, "allowModificationRedistribution"),
    "CC_BY_ND": ("required", True, "prohibited"),
    "CC_BY_NC_ND": ("required", True, "prohibited"),
    "Redistribution_Prohibited": ("required", False, "prohibited"),
    "Other": ("required", False, "prohibited"),
}
USERS_0 = {"OnlyAuthor": "onlyAuthor", "ExplicitlyLicensedPerson": "onlySeparatelyLicensedPerson", "Everyone": "everyone"}
FIRST_PERSON_0 = {"Auto": "auto", "Both": "both", "ThirdPersonOnly": "thirdPersonOnly", "FirstPersonOnly": "firstPersonOnly"}
LOOKS_0 = ("horizontal_inner", "horizontal_outer", "vertical_down", "vertical_up")


def convert(armature):
    """Make a VRM 0.x armature, as the add-on imported it, one the export writes as VRM 1.0.

    Assumes the add-on's import: the 1.0 bones, expressions and springs are already filled from the
    0.x ones. (Its "assign bones automatically" is not asked: on the consortium's own sample it
    left 52 of the 55.) Carried over here, because the add-on leaves them at its defaults: the
    name, author and terms (a converted sample left as "undefined", for its author alone, changes
    forbidden: another's work with its credit and terms replaced), the first-person mesh settings
    and the eyes' range. Terms 0.x can't state (political, hate) stay refused, and "commercial use
    allowed" becomes a person's, not a company's: never looser than the file said.
    """
    extension = armature.data.vrm_addon_extension
    old, new = extension.vrm0, extension.vrm1
    was = str(extension.spec_version)
    meta, terms = new.meta, old.meta
    meta.vrm_name, meta.version = terms.title or armature.name, terms.version
    meta.authors.clear()
    meta.authors.add().value = terms.author or "Unknown"
    meta.contact_information = terms.contact_information
    meta.references.clear()
    if terms.reference:
        meta.references.add().value = terms.reference
    meta.thumbnail_image = terms.texture
    meta.avatar_permission = USERS_0.get(terms.allowed_user_name, "onlyAuthor")
    meta.allow_excessively_violent_usage = terms.violent_ussage_name == "Allow"
    meta.allow_excessively_sexual_usage = terms.sexual_ussage_name == "Allow"
    non_commercial = "_NC" in terms.license_name
    meta.commercial_usage = "personalProfit" if terms.commercial_ussage_name == "Allow" and not non_commercial else "personalNonProfit"
    meta.allow_political_or_religious_usage = meta.allow_antisocial_or_hate_usage = False
    meta.credit_notation, meta.allow_redistribution, meta.modification = LICENCES_0.get(terms.license_name, LICENCES_0["Other"])
    pages = [url for url in dict.fromkeys((terms.other_license_url, terms.other_permission_url)) if url]
    meta.other_license_url = pages[0] if pages else ""
    said = [f"VRM 0.x licence: {terms.license_name}"] + [f"permissions: {url}" for url in pages[1:]]
    meta.third_party_licenses = "; ".join(said)

    first = new.first_person
    first.mesh_annotations.clear()
    for annotation in old.first_person.mesh_annotations:
        if annotation.mesh.mesh_object_name:
            carried = first.mesh_annotations.add()
            carried.node.mesh_object_name = annotation.mesh.mesh_object_name
            carried.type = FIRST_PERSON_0.get(annotation.first_person_flag, "auto")
    for name in LOOKS_0:
        degrees, range_map = getattr(old.first_person, f"look_at_{name}"), getattr(new.look_at, f"range_map_{name}")
        range_map.input_max_value, range_map.output_scale = degrees.x_range, degrees.y_range
    extension.spec_version = "1.0"
    armature[CONVERTED_KEY] = was


def _look_at_origin_0(armature):
    """Where a converted file's eyes look from, as a world point in Blender's axes, or None: the
    head bone and the 0.x first-person offset. The file has it in Unity's axes (x to the
    character's right, y up, z ahead) and the add-on keeps it as (x, z, y)."""
    extension = armature.data.vrm_addon_extension
    head = armature.data.bones.get(extension.vrm1.humanoid.human_bones.head.node.bone_name)
    if head is None:
        return None
    right, ahead, up = extension.vrm0.first_person.first_person_bone_offset
    at = armature.matrix_world @ head.head_local
    return [float(at.x - right), float(at.y - ahead), float(at.z + up)]


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
    return {"converted_from": armature.get(CONVERTED_KEY),
            "meshes": {m.name: len(m.data.vertices) for m in meshes}, "bones": len(armature.data.bones),
            "height": round(tall - low, 3), "expressions": bound, "custom_expressions": len(extension.vrm1.expressions.custom),
            "shape_keys": len(keys), "springs": len(extension.spring_bone1.springs),
            "colliders": len(extension.spring_bone1.colliders), "look_at": extension.vrm1.look_at.type,
            "materials": len({s.material.name for m in meshes for s in m.material_slots if s.material})}
