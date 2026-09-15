"""Turning an imported generator mesh into a finished VRM in one call."""

import json
import struct
import unittest

from support import FIXTURE_OUT, load_fixture
from vtuber_maker.pipeline import build, vrm_export

REQUIRED_HUMAN_BONES = {
    "hips",
    "spine",
    "head",
    "leftUpperLeg",
    "leftLowerLeg",
    "leftFoot",
    "rightUpperLeg",
    "rightLowerLeg",
    "rightFoot",
    "leftUpperArm",
    "leftLowerArm",
    "leftHand",
    "rightUpperArm",
    "rightLowerArm",
    "rightHand",
}


def read_glb_json(path):
    data = path.read_bytes()
    chunk_length, _chunk_type = struct.unpack_from("<II", data, 12)
    return json.loads(data[20 : 20 + chunk_length])


class BuildAvatarFromSyntheticCharacter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture()
        mesh.scale = (0.5, 0.5, 0.5)  # generators rarely export at real-world scale
        vrm_export.ensure_vrm_addon()
        path = FIXTURE_OUT / "synthetic_character_built.vrm"
        path.unlink(missing_ok=True)
        cls.report = build.build_avatar([mesh], path, name="Synthetic", author="Tests")
        cls.gltf = read_glb_json(path)
        cls.vrm = cls.gltf["extensions"]["VRMC_vrm"]

    def test_writes_a_named_vrm_1_avatar(self):
        self.assertIn("VRMC_vrm", self.gltf["extensionsUsed"])
        self.assertEqual(self.vrm["meta"]["name"], "Synthetic")

    def test_rigs_every_required_human_bone(self):
        self.assertLessEqual(REQUIRED_HUMAN_BONES, set(self.vrm["humanoid"]["humanBones"]))

    def test_binds_blink_and_mouth_expressions(self):
        preset = self.vrm["expressions"]["preset"]
        for expression in ("blink", "blinkLeft", "blinkRight", "aa", "ih", "ou", "ee", "oh"):
            with self.subTest(expression=expression):
                self.assertTrue(preset[expression].get("morphTargetBinds"))

    def test_uses_toon_shading(self):
        for material in self.gltf["materials"]:
            with self.subTest(material=material.get("name")):
                self.assertIn("VRMC_materials_mtoon", material.get("extensions", {}))

    def test_gives_hanging_hair_spring_bones(self):
        self.assertGreaterEqual(len(self.gltf["extensions"]["VRMC_springBone"]["springs"]), 4)

    def test_reports_what_it_found(self):
        self.assertGreaterEqual(self.report["hair_chains"], 4)
        self.assertFalse(self.report["mouth_estimated"])
        self.assertAlmostEqual(self.report["height"], 1.6, places=3)


if __name__ == "__main__":
    unittest.main()
