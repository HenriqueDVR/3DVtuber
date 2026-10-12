"""Take a model that came whole (an MMD model passed round as FBX) in with what it has.

Such a file is a finished character in another program's terms: a skeleton with every vertex
weighted, bone chains for the hair, the skirt and the ribbons that its own program swings, and
face shapes for the blink, the vowels and the brows. A tester's (417 bones, 342 weighted, 117
face shapes) was built as if it were a bare surface: skinned again by bone heat, its white dress
read as hair of the same colour and hung from the head (33 cm torn on a head turn), a face patch
drawn over a face that could already blink. Nothing of that is remade here. The humanoid bones
take the pipeline's names (adopt.py's tables), the chains become VRM springs, the face shapes are
bound to the VRM expressions by their names, and the materials go toon. The meshes are joined
into one, which every later step takes; weights, shapes and UVs stay as they came.

What such a file doesn't have stays missing: no ARKit shapes (the blink, the vowels and four
moods are all a tracker gets), and its twist bones only follow their limb.
"""

import bpy
import numpy as np

from . import adopt, compat, fingers, hair_physics, meshdata, texture_space, vrm_export
from .rig import BONE_NAMES

# On the armature of a model taken in this way: the file's name.
KEPT_KEY = "vtuber_kept_rig"
# A model this far from a person's height (m) was saved in other units and is scaled to the height asked for.
HEIGHTS = (0.5, 2.5)
# A chain swings with at least this many weighted bones: one alone (a flower in the hair, an ear,
# an arm's twist bone) is a fixture and keeps to its parent.
SPRING_BONES = 2
# ... and one of no more than this many is stiff, as an appendage: a chest's two bones on the
# cloth's settings (gravity 0.3) would hang.
STIFF_BONES = 2
# The pipeline's own chains are about this long and its gravity is set for them: a longer one gets
# less, by as much as it is longer.
SWING_JOINTS = 5
# Optional humanoid bones, by the names they go under: Cats', VRM's, MMD's own.
EXTRA_BONES = {"left_eye": ("Eye_L", "LeftEye", "左目"), "right_eye": ("Eye_R", "RightEye", "右目"),
               "left_toes": ("Left toe", "LeftToes", "LeftToeBase", "左つま先"),
               "right_toes": ("Right toe", "RightToes", "RightToeBase", "右つま先")}
# VRM expression -> the face shape that is it, the first of these the model has: MMD's names, then
# the English the Cats plugin gives them. ウィンク closes the left eye smiling, ウィンク２ plainly.
EXPRESSIONS = {
    "blink": ("まばたき", "Blink"),
    "blink_left": ("ウィンク２", "ウィンク", "Wink 2", "Wink"),
    "blink_right": ("ウィンク２右", "ｳｨﾝｸ２右", "ウィンク右", "Wink 2 Right", "Wink Right"),
    "aa": ("あ", "A", "Ah"), "ih": ("い", "I", "Ch"), "ou": ("う", "U"), "ee": ("え", "E"), "oh": ("お", "O", "Oh"),
    "happy": ("笑い", "Smile", "Joy"), "angry": ("怒り", "Angry"), "sad": ("困る", "Sad", "Sorrow"),
    "surprised": ("びっくり", "Surprised"),
}
# Face tracking drives shapes by ARKit's names (Warudo, VSeeFace's iPhone mode, the viewer's webcam
# test), and an MMD model has none: the tester's blinked and spoke in the contact sheets, through
# the VRM expressions, and did neither under a tracker. ARKit name -> the MMD shapes that are it, the
# first the model has; with a side ("L": the character's left, +X; "R") only that half of the shape,
# for one MMD makes of both eyes at once. MMD's 左 and ARKit's Left are both the character's left.
TRACKING = {
    "eyeBlinkLeft": (("ウィンク２", None), ("まばたき", "L"), ("ウィンク", None)),
    "eyeBlinkRight": (("ウィンク２右", None), ("まばたき", "R"), ("ウィンク右", None)),
    "eyeSquintLeft": (("笑い目左", None), ("笑い目", "L")), "eyeSquintRight": (("笑い目右", None), ("笑い目", "R")),
    "eyeWideLeft": (("びっくり左", None), ("びっくり", "L")), "eyeWideRight": (("びっくり右", None), ("びっくり", "R")),
    # the iris shapes a Cats export names in English; each moves both eyes
    "eyeLookUpLeft": (("Up", "L"),), "eyeLookUpRight": (("Up", "R"),),
    "eyeLookDownLeft": (("Down", "L"),), "eyeLookDownRight": (("Down", "R"),),
    "eyeLookOutLeft": (("Left", "L"),), "eyeLookInRight": (("Left", "R"),),
    "eyeLookInLeft": (("Right", "L"),), "eyeLookOutRight": (("Right", "R"),),
    "browDownLeft": (("怒り左", None), ("怒り", "L")), "browDownRight": (("怒り右", None), ("怒り", "R")),
    "browOuterUpLeft": (("上左", None), ("上", "L")), "browOuterUpRight": (("上右", None), ("上", "R")),
    "browInnerUp": (("困る", None),),
    "jawOpen": (("あ", None),), "mouthFunnel": (("お", None),), "mouthPucker": (("う", None),),
    "mouthSmileLeft": (("口角上げ左", None),), "mouthSmileRight": (("口角上げ右", None),),
    "mouthFrownLeft": (("口角下げ左", None),), "mouthFrownRight": (("口角下げ右", None),),
    "mouthStretchLeft": (("口横広げ左", None),), "mouthStretchRight": (("口横広げ右", None),),
    "mouthLeft": (("口左", None),), "mouthRight": (("口右", None),),
    "tongueOut": (("ぺろっ", None),),
}
# How far eye bones turn for a look, as a VRoid file has it: degrees looked -> degrees turned.
LOOK_RANGES = {"horizontal_inner": (90.0, 8.0), "horizontal_outer": (90.0, 12.0),
               "vertical_down": (90.0, 10.0), "vertical_up": (90.0, 10.0)}
# A picture is opaque with this share of the texels its polygons use being so; one soft (neither
# clear nor opaque) over SOFT_SHARE of them is blended, not cut.
OPAQUE_SHARE = 0.99
# The copy of a cut-out material that holds the polygons its picture doesn't cut.
SOLID = "_solid"
# ... found by reading its picture at up to this many points along a polygon's edge.
SAMPLES = 48
SOFT_SHARE = 0.2


class NotKept(ValueError):
    """The file can't be taken in with its own rig (no skeleton, nothing skinned, bones not known)."""


def load(path, height=1.6):
    """Import a rigged FBX into the open scene and hand back (armature, its mesh).

    The meshes skinned to the skeleton are joined into one; whatever else the file holds (lights,
    props skinned to nothing) is dropped. Left standing on Z=0 with its transforms applied, at
    its own size unless that is outside HEIGHTS. Assumes it stands Z up, as Blender's FBX import
    leaves a character; `name_humanoid` turns it to face -Y.
    """
    before = set(bpy.data.objects)
    compat.import_fbx(path)
    imported = [o for o in bpy.data.objects if o not in before]
    armature = next((o for o in imported if o.type == "ARMATURE"), None)
    if armature is None:
        raise NotKept(f"{path}: no skeleton in it")
    meshes = sorted((o for o in imported if o.type == "MESH" and o.find_armature() == armature),
                    key=lambda o: -len(o.data.vertices))
    if not meshes:
        raise NotKept(f"{path}: nothing in it is skinned to the skeleton")
    for obj in imported:
        if obj is not armature and obj not in meshes:
            bpy.data.objects.remove(obj, do_unlink=True)
    names = [obj.name for obj in meshes[1:]]
    if names:
        with bpy.context.temp_override(active_object=meshes[0], object=meshes[0], selected_objects=meshes,
                                       selected_editable_objects=meshes):
            if bpy.ops.object.join() != {"FINISHED"}:
                raise NotKept(f"{path}: its {len(meshes)} meshes didn't join")
        for name in names:
            if name in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    mesh = meshes[0]
    if mesh.parent != armature:
        world = mesh.matrix_world.copy()
        mesh.parent = armature
        mesh.matrix_world = world
    verts = meshdata.world_vertices(mesh)
    tall = float(np.ptp(verts[:, 2]))
    scale = 1.0 if HEIGHTS[0] <= tall <= HEIGHTS[1] else height / max(tall, 1e-9)
    _apply(armature, mesh, scale=scale)
    armature.location.z -= float(meshdata.world_vertices(mesh)[:, 2].min())
    _apply(armature, mesh)
    armature[KEPT_KEY] = str(path)
    return armature, mesh


def _apply(armature, mesh, scale=1.0, turn=0.0):
    """Scale and turn (radians about Z) the armature with its mesh, and apply their transforms."""
    armature.scale = [s * scale for s in armature.scale]
    armature.location = [v * scale for v in armature.location]
    if turn:
        # called on transforms already applied: the turn is all the rotation there is
        armature.rotation_mode = "XYZ"
        armature.rotation_euler = (0.0, 0.0, turn)
    bpy.context.view_layer.update()
    with bpy.context.temp_override(active_object=armature, object=armature, selected_objects=[armature, mesh],
                                   selected_editable_objects=[armature, mesh]):
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def kept(armature):
    """Whether this armature came from a rigged model taken in with its rig (`load`)."""
    return armature is not None and KEPT_KEY in armature


def name_humanoid(armature, mesh, name, author, path=""):
    """Give the humanoid bones the pipeline's names and mark the armature a VRM 1.0 humanoid.

    Which bone is which goes by adopt.py's tables; NotKept when a body bone is missing under all
    of them. A bone that only doubles another (MMD's 左足D) hands that one its weights. An upper
    chest, eyes and toes are bound when the rig has them and left out when it doesn't: none is
    made. The character is turned to face -Y when its left arm is at -X. Returns a report.
    """
    found = [bone.name for bone in armature.data.bones]
    table = adopt.names(found)
    try:
        adopt.required(table, found, path)
    except ValueError as error:
        raise NotKept(str(error)) from None
    lent = [(theirs, table[theirs]) for theirs in found if theirs in table and adopt._lends_only(theirs, table)]
    renamed = 0
    for theirs in found:
        ours = table.get(theirs)
        if ours is None or ours == theirs or adopt._lends_only(theirs, table):
            continue
        if ours in armature.data.bones:
            # a helper already under the name we want would take a ".001" and hide the real bone
            armature.data.bones[ours].name = f"{ours}_was"
        armature.data.bones[theirs].name = ours
        renamed += 1
    for theirs, ours in lent:
        _merge_group(mesh, theirs, ours)

    bones = armature.data.bones
    left, right = (bones[BONE_NAMES[f"{side}_upper_arm"]].head_local.x for side in ("left", "right"))
    turned = left < right
    if turned:
        _apply(armature, mesh, turn=np.pi)

    extension = armature.data.vrm_addon_extension
    extension.spec_version = "1.0"
    human_bones = extension.vrm1.humanoid.human_bones
    # the add-on's own guess by name, made once on a fresh armature: it would fill what is left empty here
    if hasattr(human_bones, "initial_automatic_bone_assignment"):
        human_bones.initial_automatic_bone_assignment = False
    for key, ours in BONE_NAMES.items():
        getattr(human_bones, key).node.bone_name = ours if ours in bones else ""
    vrm_export.bind_fingers(armature)
    extras = {}
    for key, candidates in EXTRA_BONES.items():
        under = BONE_NAMES["head" if key.endswith("eye") else f"{key.split('_')[0]}_foot"]
        bone = next((bones[c] for c in candidates if c in bones and under in [p.name for p in bones[c].parent_recursive]), None)
        if bone is not None and hasattr(human_bones, key):
            getattr(human_bones, key).node.bone_name = bone.name
            extras[key] = bone.name
    meta = extension.vrm1.meta
    meta.vrm_name = name
    meta.authors.clear()
    meta.authors.add().value = author
    return {"naming": _naming(table), "renamed_bones": renamed, "merged": [theirs for theirs, _ in lent],
            "upper_chest": BONE_NAMES["upper_chest"] in bones, "extras": extras, "turned": bool(turned),
            "fingers": sum(fingers.bone_name(side, finger, segment) in bones for side in ("left", "right")
                           for finger in fingers.FINGERS for segment in fingers._segments(finger))}


def _naming(table):
    return "MMD (Japanese)" if "下半身" in table else "MMD (Cats)" if "Left elbow" in table else \
        "Mixamo" if any("mixamorig" in k for k in table) else "Character Creator" if "CC_Base_Hip" in table else \
        "VRoid / Tripo" if "J_Bip_C_Hips" in table else "VRM"


def _merge_group(mesh, source, target):
    """Hand the vertex group `source` to `target`, added to what it holds."""
    groups = mesh.vertex_groups
    if source not in groups:
        return
    index = groups[source].index
    into = groups.get(target) or groups.new(name=target)
    for v in mesh.data.vertices:
        for g in v.groups:
            if g.group == index and g.weight > 0:
                into.add([v.index], g.weight, "ADD")
    groups.remove(groups[source])


def humanoid_bones(armature):
    """The names of the bones the VRM humanoid holds."""
    human_bones = armature.data.vrm_addon_extension.vrm1.humanoid.human_bones
    keys = [*BONE_NAMES, *EXTRA_BONES, *(key for side in ("left", "right") for finger in fingers.FINGERS
                                         for key in fingers.humanoid_keys(side, finger))]
    return {getattr(human_bones, key).node.bone_name for key in keys if hasattr(human_bones, key)} - {""}


def spring_chains(armature, mesh):
    """The rig's own swinging chains, as {"hair": [...], "cloth": [...], "stiff": [...]}, each
    chain a list of bone names from its root out.

    A chain is a run of bones outside the humanoid, each the only child of the one before, with
    SPRING_BONES or more of them weighted; where a bone has several children each starts a chain of
    its own. The FBX's unweighted end bones stay on as the last joint, which a VRM spring never
    turns: without one the last weighted bone would hang stiff. Hair: chains that hang from the
    head. Stiff: of STIFF_BONES weighted bones or fewer. Cloth: the rest.
    """
    bones = armature.data.bones
    human = humanoid_bones(armature)
    weighted = _weighted(mesh)
    chains = {"hair": [], "cloth": [], "stiff": []}

    def walk(bone):
        run = [bone]
        while len(run[-1].children) == 1 and run[-1].children[0].name not in human:
            run.append(run[-1].children[0])
        branches = [child for child in run[-1].children if child.name not in human]
        # unweighted bones before the first weighted one are a mount, not part of what swings
        while run and run[0].name not in weighted:
            run.pop(0)
        real = sum(b.name in weighted for b in run)
        if real >= SPRING_BONES:
            under = next((p.name for p in run[0].parent_recursive if p.name in human), None)
            kind = "hair" if under == BONE_NAMES["head"] else "stiff" if real <= STIFF_BONES else "cloth"
            chains[kind].append([b.name for b in run])
        for child in branches:
            walk(child)

    for bone in bones:
        if bone.name not in human and (bone.parent is None or bone.parent.name in human):
            walk(bone)
    return chains


def _weighted(mesh):
    """The names of the vertex groups that hold any weight."""
    held = set()
    for v in mesh.data.vertices:
        held.update(g.group for g in v.groups if g.weight > 1e-4)
    return {g.name for g in mesh.vertex_groups if g.index in held}


def joints(armature):
    """The humanoid joints keyed like landmarks.detect_body, from the bones' rest heads (world);
    an upper chest the rig doesn't have is read halfway from chest to neck."""
    bones, world = armature.data.bones, armature.matrix_world
    found = {key: np.array(world @ bones[name].head_local) for key, name in BONE_NAMES.items() if name in bones}
    found.setdefault("upper_chest", (found["chest"] + found["neck"]) / 2)
    return found


def bind_springs(armature, mesh, chains, hair_hits_body=True):
    """Make VRM springs of `chains` (`spring_chains`) with colliders fitted to the body, on the
    settings the pipeline gives its own hair, cloth and appendage chains."""
    design = hair_physics.design(mesh, armature, chains["hair"], joints(armature), chains["cloth"],
                                 appendage_chains=chains["stiff"], hair_hits_body=hair_hits_body)
    for chain in design.chains:
        # every joint gives a little to gravity and a long chain adds them up: the tester's twin
        # tails of 20 bones hung 16.5 cm below where they were modelled, standing still
        chain.gravity_power *= min(1.0, SWING_JOINTS / len(chain.names))
    vrm_export.bind_spring_bones(armature, design)
    return design


def bind_expressions(armature, mesh):
    """Bind the VRM expressions to the model's own face shapes by name (EXPRESSIONS); returns
    {expression: the shape bound}. One the model has no shape for stays empty."""
    keys = mesh.data.shape_keys.key_blocks if mesh.data.shape_keys else {}
    presets = armature.data.vrm_addon_extension.vrm1.expressions.preset
    bound = {}
    for preset, candidates in EXPRESSIONS.items():
        expression = getattr(presets, preset)
        expression.morph_target_binds.clear()
        shape = next((c for c in candidates if c in keys), None)
        if shape is None:
            continue
        bind = expression.morph_target_binds.add()
        bind.node.mesh_object_name = mesh.name
        bind.index = shape
        bind.weight = 1.0
        bound[preset] = shape
    # the add-on's own guess by name, made when a file is first opened: it bound the looks to shapes
    # named Up, Down, Left and Right and set the eyes to look by them, over the eye bones (`bind_eyes`)
    expressions = armature.data.vrm_addon_extension.vrm1.expressions
    if hasattr(expressions, "initial_automatic_expression_assignment"):
        expressions.initial_automatic_expression_assignment = False
    for key in list(keys)[1:]:
        key.value = 0.0  # Blender 5 adds a key at 1.0; an import may leave one on
    return bound


def see_through(mesh):
    """Let each material's picture cut it out where the picture is clear, as MMD draws every
    material; returns {material: "CUT", "BLEND" or "OPAQUE"}. Call after `vrm_export.use_toon_shading`.

    The FBX says of no picture that its alpha is see-through, and such a model needs it: the
    tester's skin and dress are one surface drawn twice, each picture clear where the other shows,
    and her eyes' highlights lie on the irises (drawn opaque: blank white eyes). Read where the
    polygons use the picture, since the rest of one is often empty. A picture opaque there stays
    opaque. One that is soft over SOFT_SHARE of it, or whose material is itself half clear (a sheen
    over the hair), is blended.

    The rest are cut ("CUT"), and how was settled by three looks in Warudo. MToon's cutout drew
    the whole body a little see-through there (the hair's sheen behind an arm, the room's lights
    behind the waist), which nothing opaque was. So the polygons whose picture is opaque all over
    move to an opaque copy of the material (`<name>_solid`, "OPAQUE" in the answer): no vertex, UV
    or shape changes. What the picture does cut is blended with depth written (MToon's
    transparent with ZWrite): solid there, but a shade off the opaque skin beside it, so as little
    as can be is left to it.
    """
    data = mesh.data
    if data.uv_layers.active is None:
        return {}
    count = len(data.polygons)
    material, total = np.empty(count, np.int32), np.empty(count, np.int32)
    data.polygons.foreach_get("material_index", material)
    data.polygons.foreach_get("loop_total", total)
    start = np.empty(count, np.int32)
    data.polygons.foreach_get("loop_start", start)
    of_loop = np.repeat(material, total)
    uv = np.empty(len(data.loops) * 2, np.float32)
    data.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    cut = {}
    for index, slot in enumerate(mesh.material_slots):
        loops = of_loop == index
        image = texture_space.material_base_color_image(slot.material) if slot.material else None
        if image is None or not loops.any() or image.channels < 4 or not image.size[0]:
            continue
        width, height = image.size
        pixels = np.empty(width * height * 4, np.float32)
        image.pixels.foreach_get(pixels)
        alpha = pixels.reshape(height, width, 4)[..., 3]
        at = uv[loops] % 1.0
        used = alpha[np.clip((at[:, 1] * height).astype(int), 0, height - 1), np.clip((at[:, 0] * width).astype(int), 0, width - 1)]
        if (used > 0.95).mean() >= OPAQUE_SHARE:
            continue
        mtoon1 = slot.material.vrm_addon_extension.mtoon1
        factor = mtoon1.pbr_metallic_roughness.base_color_factor
        soft = 0.0 < factor[3] < 1.0 or ((used >= 0.05) & (used <= 0.95)).mean() >= SOFT_SHARE
        if not 0.0 < factor[3] < 1.0:
            factor[3] = 1.0  # the import leaves most at alpha 0, which would hide the material whole
        mtoon1.alpha_mode = "BLEND"
        mtoon1.extensions.vrmc_materials_mtoon.transparent_with_z_write = not soft
        cut[slot.material.name] = "BLEND" if soft else "CUT"
        if soft:
            continue
        polygons = np.flatnonzero(material == index)
        solid = polygons[_all_opaque(alpha > 0.95, uv, start[polygons], total[polygons])]
        if len(solid):
            whole = slot.material.copy()
            whole.name = f"{slot.material.name}{SOLID}"
            whole.vrm_addon_extension.mtoon1.alpha_mode = "OPAQUE"
            whole.vrm_addon_extension.mtoon1.extensions.vrmc_materials_mtoon.transparent_with_z_write = False
            data.materials.append(whole)
            material[solid] = len(data.materials) - 1
            cut[whole.name] = "OPAQUE"
    data.polygons.foreach_set("material_index", material)
    data.update()
    return cut


def _all_opaque(opaque, uv, start, total):
    """Which polygons (their first loop and loop count) lie on texels of `opaque` (rows bottom to top)
    alone. Read at points over each polygon no further apart than a texel or so, up to SAMPLES
    along an edge: the rectangle round a polygon's UVs, tried first, left a third of the tester's
    plain skin cut out for a clear texel beside it. One whose UVs leave the picture counts as not."""
    height, width = opaque.shape
    result = np.zeros(len(start), bool)
    size = np.array([width, height], np.float32)
    for corners in np.unique(total):
        if corners < 3:
            continue
        which = np.flatnonzero(total == corners)
        at = uv[start[which][:, None] + np.arange(corners)]            # polygons x corners x 2
        inside = ((at >= 0.0) & (at <= 1.0)).all(axis=(1, 2))
        fine = np.ones(len(which), bool)
        # a fan of triangles from the first corner
        for k in range(1, corners - 1):
            triangle = at[:, [0, k, k + 1]]
            texels = (np.ptp(triangle, axis=1) * size).max(axis=1)
            steps = np.clip(np.ceil(texels), 2, SAMPLES).astype(int)
            for count in np.unique(steps):
                chosen = np.flatnonzero(steps == count)
                i, j = np.meshgrid(np.arange(count + 1), np.arange(count + 1), indexing="ij")
                keep = i + j <= count
                weights = np.stack([i[keep], j[keep], count - i[keep] - j[keep]], axis=1) / count
                points = np.einsum("sk,pkc->psc", weights.astype(np.float32), triangle[chosen])
                x = np.clip((points[..., 0] * width).astype(int), 0, width - 1)
                y = np.clip((points[..., 1] * height).astype(int), 0, height - 1)
                fine[chosen] &= opaque[y, x].all(axis=1)
        result[which] = inside & fine
    return result


def tracking_shapes(mesh):
    """Give the mesh face shapes under ARKit's names, made of its own MMD ones (TRACKING); returns
    {ARKit name: the shape it was made of}. A name the mesh already has is left as it is, one it
    has no shape for stays missing. The model's own shapes all stay.

    Half a shape is cut at X = 0 of the mesh as it stands (facing -Y, so call after
    `name_humanoid`): only shapes of the eyes and brows are halved, which don't cross the middle.
    """
    data = mesh.data
    if data.shape_keys is None:
        return {}
    keys = data.shape_keys.key_blocks
    count = len(data.vertices)

    def read(key):
        co = np.empty(count * 3, np.float32)
        key.data.foreach_get("co", co)
        return co.reshape(-1, 3)

    basis = read(keys[0])
    world_x = meshdata.world_vertices(mesh)[:, 0]
    half = {"L": world_x > 0, "R": world_x < 0}
    made = {}
    for name, sources in TRACKING.items():
        if name in keys:
            continue
        source, side = next(((s, side) for s, side in sources if s in keys), (None, None))
        if source is None:
            continue
        offset = read(keys[source]) - basis
        if side is not None:
            offset[~half[side]] = 0.0
        if not np.abs(offset).max() > 1e-6:
            continue
        key = mesh.shape_key_add(name=name, from_mix=False)
        key.data.foreach_set("co", (basis + offset).ravel())
        key.value = 0.0  # Blender 5 adds a key at 1.0
        made[name] = source
    return made


def bind_eyes(armature):
    """Look with the eye bones when the humanoid has both; returns whether it does."""
    human_bones = armature.data.vrm_addon_extension.vrm1.humanoid.human_bones
    names = [human_bones.left_eye.node.bone_name, human_bones.right_eye.node.bone_name]
    if not all(names):
        return False
    heads = [armature.matrix_world @ armature.data.bones[name].head_local for name in names]
    vrm_export.set_look_at(armature, "bone", LOOK_RANGES, origin=(heads[0] + heads[1]) / 2)
    return True
