"""The whole pipeline: an imported image-to-3D character in, a VRM avatar file out."""

import numpy as np

from . import (adopt, cloth, face, face_points, face_rig, face_texture, hair, hair_physics, landmarks, loose, meshdata,
               normalize, parts, rig, roles, template_skin, vrm_export)


def build_avatar(objects, filepath, name, author, height=1.6, closed_eyes_image=None, face_points_path=None, keep_eyes=False,
                 joints_override=None, rigid_head=False, cloth_physics=False, hat_tip=False, mouth_color=None,
                 free_mouth=False, pieces=None, pieces_reference=None, rig_path=None, skinning="heat"):
    """Rig the character, add face expressions and hair physics, and export VRM 1.0.

    `objects` are the freshly imported objects. Returns a report of what was found.
    Meshes with separate hair and eye inserts (like Tripo P2) are rigged from those parts;
    fused meshes with a painted face are rigged from the texture.
    `closed_eyes_image` textures blink lids on a face patch (see face_rig).
    `face_points_path` is a face.json from viewer/setup.html; its eye and mouth points replace
    the detected ones. With `keep_eyes` nothing changes the eyes: no blink keys or lid covers.
    Joints that clothing hides are placed from typical proportions and listed in the report's
    `warnings`, rather than read wrongly off the silhouette.
    `joints_override` maps joint keys ("hips", "left_upper_leg", ...) to (x, y, z) in the scaled
    avatar's space; those joints replace the detected ones and aren't refined, for when detection
    is fooled (a coat panel hanging between the legs reads as the crotch).
    `rigid_head` is for hats, helmets and solid heads: no hair chains, and every head piece moves
    with the head alone. `cloth_physics` swings hanging capes, coat tails and skirts on spring
    chains; `hat_tip` (with `rigid_head`) swings a pointed hat's tip. `mouth_color` tints the
    inside of the mouth: an RGB triple, or "skin" for the face's own colour. `free_mouth` (with
    face points on a face patch) paints the drawn mouth out and makes the mouth its own shape on
    the face, resting as a short line, so face trackers can slide and curl it. With face
    points on a face patch, the rest of the head is matched to the patch's colour, and the painted
    eyes' boxes are stored for scripts/add_face_tracking.py's blink.
    `pieces` are a character manifest's build-stage pieces (manifest.pieces_for(data, "build")),
    the user's word over detection: hair pieces grow strands, hats go wholly onto the head (a
    "tip" swings), swing_cloth pieces get cloth chains even without `cloth_physics`, and any
    other listed piece is kept off hair and cloth chains. Their points and vertex counts are
    on `pieces_reference` (the manifest's unmodified model) when given, else on this model.
    `rig_path` is a rig of this same model made elsewhere (Tripo's FBX, see adopt.py): its bones
    and weights replace joint detection and skinning; the rest runs as usual. `skinning`
    "template" weights the body from a fitted template (template_skin.py) instead of bone heat;
    "voxel" solves the skinning proxy by geodesic voxel binding (voxel_skin.py), rules unchanged.
    """
    warnings = []
    override = {k: np.asarray(v, dtype=float) for k, v in (joints_override or {}).items()}
    points = face_points.load(face_points_path) if face_points_path else None
    vrm_export.ensure_vrm_addon()
    mesh = normalize.prepare(objects, height=height)
    adopted = adopt.load(rig_path, height) if rig_path else None
    if adopted is not None:
        # the rig's joints stand; the manifest's own overrides still win
        override = {**adopt.joints(adopted[0]), **override}
    joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data), warnings)
    joints.update(override)
    found = parts.find(mesh, joints)
    pieces = pieces or []
    # found on the source as it came in, before any step adds or cuts vertices
    named = {}
    if pieces and pieces_reference:
        with roles.reference(pieces_reference, height=height) as ref:
            named = roles.find(mesh, pieces, ref)
    elif pieces:
        named = roles.find(mesh, pieces)
    if pieces:
        found = found._replace(hair=roles.hair_mask(found.hair, pieces, named))
    joints = landmarks.refine_limbs(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data), joints, exclude=found.hair)
    joints.update(override)
    joints = landmarks.bend_knees(joints)
    modelled_face = found.has_eye_inserts

    if modelled_face:
        features = parts.face_features(mesh, found)
    else:
        features = face.detect_features(mesh, joints)
    if points is not None:
        features = face_points.apply(mesh, features, points)
    if free_mouth and (points is None or face_points._patch_image(mesh) is None):
        raise ValueError("free_mouth needs face points on a face patch (--face face.json)")
    if free_mouth:
        face_points.erase_mouth(mesh, points)
        features = dict(features, free_mouth=True)
    if points is not None and modelled_face and face_points._patch_image(mesh) is not None:
        # a solid head (a mascot) has no detail of its own round the face: the patch covers all of
        # its front, and the rest continues the face smoothly
        if rigid_head:
            face_texture.extend_patch(mesh, found.labels == found.face)
        face_texture.blend_patch_seam(mesh, found.labels == found.face, smooth=rigid_head)
        if not rigid_head:
            face_texture.clean_head_top(mesh, found.labels == found.face)
    if mouth_color is not None:
        features = dict(features, mouth_color=features["skin_color"] if mouth_color == "skin" else mouth_color)
    if not modelled_face:
        face_rig.cut_mouth(mesh, features)  # these two change topology, so before skinning
        if not keep_eyes:
            face_rig.cut_eyelids(mesh, features)
    if modelled_face and adopted is None:
        verts = meshdata.world_vertices(mesh)
        chin = rig.chin_height(verts, found.labels == found.face, float(features["mouth"][0]),
                               float(joints["neck"][2]), float(features["mouth"][2]))
        if chin is not None and chin > joints["neck"][2] + 0.01:
            joints["head"] = np.array([joints["head"][0], joints["head"][1], chin + rig.HEAD_ABOVE_CHIN])
    if adopted is not None:
        armature = adopt.build_armature(adopted[0])
        adopt.transfer_weights(mesh, armature, adopted[1], adopted[2])
    else:
        armature = rig.build_armature(joints)
    if adopted is None and skinning == "template":
        report_skin = template_skin.transfer(mesh, armature, template_skin.load_template())
        print(f"template skinning: {report_skin}")
        if modelled_face:
            rig.weight_head(mesh, armature, found.labels == found.face, found.labels, chin_z=chin)
    elif adopted is None and modelled_face:
        before = found.labels
        found = rig.skin(mesh, armature, found, solver=skinning)
        named = roles.carry(named, before, found.labels, meshdata.world_vertices(mesh))
        rig.weight_head(mesh, armature, found.labels == found.face, found.labels, chin_z=chin)
    elif adopted is None:
        rig.skin(mesh, armature, solver=skinning)
    if modelled_face:
        face_rig.set_eyelids_from_inserts(mesh, features, found, closed_eyes_image, lids=not keep_eyes)
    face_rig.add_shape_keys(mesh, features, blink=not keep_eyes)
    if points is not None:
        face_rig.store_painted_eyes(mesh, features)
    count = len(mesh.data.vertices)
    hats = [p for p in pieces if p["role"] == "hat"]
    tip_chains = []
    if hats and not rigid_head:
        hat = roles.of_role(pieces, named, "hat", count)
        loose.follow_bone(mesh, armature, np.flatnonzero(hat), rig.BONE_NAMES["head"])
        tipped = np.zeros(count, dtype=bool)
        for piece in hats:
            if piece.get("tip"):
                tipped |= roles.pad(named[piece["name"]], count)
        if tipped.any():
            face_mask = roles.pad(found.labels == found.face, count) if modelled_face else None
            tip_chains = cloth.build_tip_chain(mesh, armature, joints, tipped, face_mask)
    if rigid_head:
        chains = []
        head_mask = rig.make_head_rigid(mesh, armature)
        if hat_tip:
            face_mask = None
            if modelled_face:  # face steps add vertices (the mouth interior) after the parts were found
                face_mask = np.zeros(len(mesh.data.vertices), dtype=bool)
                face_mask[:len(found.labels)] = found.labels == found.face
            tip_chains = cloth.build_tip_chain(mesh, armature, joints, head_mask, face_mask)
    else:
        named_hair = roles.of_role(pieces, named, "hair", count)
        hair_mask = found.hair if modelled_face else (named_hair if named_hair.any() else None)
        chains = hair.build_hair_bones(mesh, armature, joints, features, hair_mask)
        head_mask = hair_mask
    swing = roles.of_role(pieces, named, "swing_cloth", count)
    if cloth_physics or swing.any():
        # a listed piece that isn't swinging cloth is the user saying it doesn't swing
        skip = np.zeros(count, dtype=bool) if head_mask is None else roles.pad(head_mask, count)
        for piece in pieces:
            if piece["role"] != "swing_cloth":
                skip |= roles.pad(named[piece["name"]], count)
        skip &= ~swing  # a swinging region inside a listed piece still swings
        cloth_chains = cloth.build_cloth_chains(mesh, armature, joints, skip=skip, force=swing,
                                                detect=cloth_physics)
    else:
        cloth_chains = []
    # wings, tails, big ears: off the limbs onto the core bone they grow from, on a spring chain
    appendage_chains = []
    for piece in pieces:
        if piece["role"] == "appendage":
            appendage_chains += cloth.build_appendage_chain(mesh, armature, joints, roles.pad(named[piece["name"]], count),
                                                            len(appendage_chains), piece.get("bone"))
    # small pieces stuck onto anything a spring swings (ribbons on a twin-tail, feathers on a wing)
    # swing with it; check_springs.py found model C's twin-tail ribbons left in the air
    spring_bones = {name for kind in (chains, cloth_chains, tip_chains, appendage_chains) for names in kind for name in names}
    if spring_bones:
        listed = np.zeros(count, dtype=bool)
        for piece in pieces:
            listed |= roles.pad(named[piece["name"]], count)
        carriers = roles.pad(_share_on(mesh, spring_bones) > 0.5, count)
        cloth.attach_riders(mesh, carriers, listed & ~carriers)
    rig.fill_unweighted(mesh)

    vrm_export.configure(armature, name=name, author=author)
    vrm_export.bind_expressions(armature, mesh)
    vrm_export.bind_spring_bones(armature, hair_physics.design(mesh, armature, chains, joints, cloth_chains, tip_chains,
                                                               appendage_chains))
    vrm_export.use_toon_shading(mesh)
    vrm_export.export(filepath, armature)
    return {
        "height": height,
        "orientation": normalize.orientation_of(mesh),
        # what clothing hid, so a rig that comes out wrong can be traced to a guessed joint
        "warnings": [w for w in dict.fromkeys(warnings) if not override],
        "hair_chains": len(chains),
        "cloth_chains": len(cloth_chains),
        "hat_tip": bool(tip_chains),
        "appendage_chains": len(appendage_chains),
        "mouth_estimated": bool(features["mouth_estimated"]),
        "eye_inserts": modelled_face,
        "face_points": bool(points),
        "free_mouth": bool(free_mouth),
    }


def _share_on(mesh_obj, bone_names):
    """Each vertex's share of weight on `bone_names`."""
    groups = {g.index for g in mesh_obj.vertex_groups if g.name in bone_names}
    share = np.zeros(len(mesh_obj.data.vertices))
    for v in mesh_obj.data.vertices:
        total = sum(g.weight for g in v.groups)
        if total > 0:
            share[v.index] = sum(g.weight for g in v.groups if g.group in groups) / total
    return share
