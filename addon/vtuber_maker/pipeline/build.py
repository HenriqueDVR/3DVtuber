"""The whole pipeline: an imported image-to-3D character in, a VRM avatar file out."""

from . import face, face_rig, hair, landmarks, meshdata, normalize, rig, vrm_export


def build_avatar(objects, filepath, name, author, height=1.6):
    """Rig the character, add face expressions and hair physics, and export VRM 1.0.

    `objects` are the freshly imported objects. Returns a report of what was found.
    """
    vrm_export.ensure_vrm_addon()
    mesh = normalize.prepare(objects, height=height)
    joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))
    features = face.detect_features(mesh, joints)

    face_rig.cut_mouth(mesh, features)  # these two change topology, so before skinning
    face_rig.cut_eyelids(mesh, features)
    armature = rig.build_armature(joints)
    rig.skin(mesh, armature)
    face_rig.add_shape_keys(mesh, features)
    chains = hair.build_hair_bones(mesh, armature, joints, features)

    vrm_export.configure(armature, name=name, author=author)
    vrm_export.bind_expressions(armature, mesh)
    vrm_export.bind_spring_bones(armature, chains, joints)
    vrm_export.use_toon_shading(mesh)
    vrm_export.export(filepath, armature)
    return {
        "height": height,
        "hair_chains": len(chains),
        "mouth_estimated": bool(features["mouth_estimated"]),
    }
