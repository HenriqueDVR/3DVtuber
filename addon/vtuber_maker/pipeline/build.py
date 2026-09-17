"""The whole pipeline: an imported image-to-3D character in, a VRM avatar file out."""

import numpy as np

from . import face, face_points, face_rig, hair, hair_physics, landmarks, meshdata, normalize, parts, rig, vrm_export


def build_avatar(objects, filepath, name, author, height=1.6, closed_eyes_image=None, face_points_path=None, keep_eyes=False):
    """Rig the character, add face expressions and hair physics, and export VRM 1.0.

    `objects` are the freshly imported objects. Returns a report of what was found.
    Meshes with separate hair and eye inserts (like Tripo P2) are rigged from those parts;
    fused meshes with a painted face are rigged from the texture.
    `closed_eyes_image` textures blink lids on a face patch (see face_rig).
    `face_points_path` is a face.json from viewer/setup.html; its eye and mouth points replace
    the detected ones. With `keep_eyes` nothing changes the eyes: no blink keys or lid covers.
    """
    points = face_points.load(face_points_path) if face_points_path else None
    vrm_export.ensure_vrm_addon()
    mesh = normalize.prepare(objects, height=height)
    joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))
    found = parts.find(mesh, joints)
    joints = landmarks.refine_limbs(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data), joints, exclude=found.hair)
    modelled_face = found.has_eye_inserts

    if modelled_face:
        features = parts.face_features(mesh, found)
    else:
        features = face.detect_features(mesh, joints)
    if points is not None:
        features = face_points.apply(mesh, features, points)
    if not modelled_face:
        face_rig.cut_mouth(mesh, features)  # these two change topology, so before skinning
        if not keep_eyes:
            face_rig.cut_eyelids(mesh, features)
    if modelled_face:
        verts = meshdata.world_vertices(mesh)
        chin = rig.chin_height(verts, found.labels == found.face, float(features["mouth"][0]),
                               float(joints["neck"][2]), float(features["mouth"][2]))
        if chin is not None and chin > joints["neck"][2] + 0.01:
            joints["head"] = np.array([joints["head"][0], joints["head"][1], chin + rig.HEAD_ABOVE_CHIN])
    armature = rig.build_armature(joints)
    if modelled_face:
        found = rig.skin(mesh, armature, found)
        rig.weight_head(mesh, armature, found.labels == found.face, found.labels, chin_z=chin)
    else:
        rig.skin(mesh, armature)
    if modelled_face:
        face_rig.set_eyelids_from_inserts(mesh, features, found, closed_eyes_image, lids=not keep_eyes)
    face_rig.add_shape_keys(mesh, features, blink=not keep_eyes)
    chains = hair.build_hair_bones(mesh, armature, joints, features, found.hair if modelled_face else None)
    rig.fill_unweighted(mesh)

    vrm_export.configure(armature, name=name, author=author)
    vrm_export.bind_expressions(armature, mesh)
    vrm_export.bind_spring_bones(armature, hair_physics.design(mesh, armature, chains, joints))
    vrm_export.use_toon_shading(mesh)
    vrm_export.export(filepath, armature)
    return {
        "height": height,
        "hair_chains": len(chains),
        "mouth_estimated": bool(features["mouth_estimated"]),
        "eye_inserts": modelled_face,
        "face_points": bool(points),
    }
