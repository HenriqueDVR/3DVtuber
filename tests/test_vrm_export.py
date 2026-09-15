"""Exporting the rigged character as a VRM 1.0 file."""

import json
import struct
import unittest

from support import FIXTURE_OUT, load_fixture
from vtuber_maker.pipeline import landmarks, meshdata, rig, vrm_export

REQUIRED_VRM_BONES = (
    "hips",
    "spine",
    "head",
    "left_upper_leg",
    "left_lower_leg",
    "left_foot",
    "right_upper_leg",
    "right_lower_leg",
    "right_foot",
    "left_upper_arm",
    "left_lower_arm",
    "left_hand",
    "right_upper_arm",
    "right_lower_arm",
    "right_hand",
)


def read_glb_json(path):
    data = path.read_bytes()
    chunk_length, _chunk_type = struct.unpack_from("<II", data, 12)
    return json.loads(data[20 : 20 + chunk_length])


def camel_case(snake):
    first, *rest = snake.split("_")
    return first + "".join(word.capitalize() for word in rest)


class ExportSyntheticCharacter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture()
        vrm_export.ensure_vrm_addon()
        joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))
        armature = rig.build_armature(joints)
        rig.skin(mesh, armature)
        vrm_export.configure(armature, name="Synthetic", author="Tests")
        cls.path = FIXTURE_OUT / "synthetic_character.vrm"
        cls.path.unlink(missing_ok=True)
        vrm_export.export(cls.path, armature)
        cls.gltf = read_glb_json(cls.path)

    def test_declares_the_vrm_1_extension(self):
        self.assertIn("VRMC_vrm", self.gltf["extensionsUsed"])

    def test_maps_every_required_human_bone_to_its_node(self):
        human_bones = self.gltf["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"]
        for key in REQUIRED_VRM_BONES:
            with self.subTest(bone=key):
                node = human_bones[camel_case(key)]["node"]
                self.assertEqual(self.gltf["nodes"][node]["name"], rig.BONE_NAMES[key])

    def test_writes_avatar_name_and_author(self):
        meta = self.gltf["extensions"]["VRMC_vrm"]["meta"]
        self.assertEqual(meta["name"], "Synthetic")
        self.assertEqual(meta["authors"], ["Tests"])

    def test_keeps_the_mesh_skinned(self):
        self.assertTrue(any("skin" in node and "mesh" in node for node in self.gltf["nodes"]))


if __name__ == "__main__":
    unittest.main()
