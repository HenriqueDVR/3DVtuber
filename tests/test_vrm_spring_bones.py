"""Exporting hair bone chains as VRM 1.0 spring bones that collide with the body."""

import json
import struct
import unittest

from support import FIXTURE_OUT, load_fixture
from vtuber_maker.pipeline import face, hair, landmarks, meshdata, rig, vrm_export


def read_glb_json(path):
    data = path.read_bytes()
    chunk_length, _chunk_type = struct.unpack_from("<II", data, 12)
    return json.loads(data[20 : 20 + chunk_length])


class ExportHairSpringBones(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture()
        vrm_export.ensure_vrm_addon()
        joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))
        features = face.detect_features(mesh, joints)
        armature = rig.build_armature(joints)
        rig.skin(mesh, armature)
        cls.chains = hair.build_hair_bones(mesh, armature, joints, features)
        vrm_export.configure(armature, name="Synthetic", author="Tests")
        vrm_export.bind_spring_bones(armature, cls.chains, joints)

        path = FIXTURE_OUT / "synthetic_character_hair.vrm"
        path.unlink(missing_ok=True)
        vrm_export.export(path, armature)
        cls.gltf = read_glb_json(path)
        cls.spring_bone = cls.gltf["extensions"]["VRMC_springBone"]

    def node_name(self, index):
        return self.gltf["nodes"][index]["name"]

    def test_declares_the_spring_bone_extension(self):
        self.assertIn("VRMC_springBone", self.gltf["extensionsUsed"])

    def test_every_hair_chain_becomes_a_spring_in_bone_order(self):
        springs = {tuple(self.node_name(joint["node"]) for joint in spring["joints"]) for spring in self.spring_bone["springs"]}
        self.assertEqual(springs, {tuple(chain) for chain in self.chains})

    def test_every_spring_collides_with_the_head(self):
        head_colliders = {
            index
            for index, collider in enumerate(self.spring_bone["colliders"])
            if self.node_name(collider["node"]) == rig.BONE_NAMES["head"] and "sphere" in collider["shape"]
        }
        self.assertTrue(head_colliders, "no sphere collider on the head bone")
        for spring in self.spring_bone["springs"]:
            with self.subTest(spring=spring.get("name")):
                groups = spring.get("colliderGroups", [])
                colliders = {c for g in groups for c in self.spring_bone["colliderGroups"][g]["colliders"]}
                self.assertTrue(head_colliders & colliders)


if __name__ == "__main__":
    unittest.main()
