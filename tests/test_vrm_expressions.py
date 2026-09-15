"""Exporting blink and mouth shape keys as VRM 1.0 expressions."""

import json
import struct
import unittest

from support import FIXTURE_OUT, load_fixture
from vtuber_maker.pipeline import face, face_rig, landmarks, meshdata, rig, vrm_export


def read_glb_json(path):
    data = path.read_bytes()
    chunk_length, _chunk_type = struct.unpack_from("<II", data, 12)
    return json.loads(data[20 : 20 + chunk_length])


class ExportFaceExpressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture()
        vrm_export.ensure_vrm_addon()
        joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))
        features = face.detect_features(mesh, joints)
        face_rig.cut_mouth(mesh, features)
        face_rig.cut_eyelids(mesh, features)
        armature = rig.build_armature(joints)
        rig.skin(mesh, armature)
        face_rig.add_shape_keys(mesh, features)
        vrm_export.configure(armature, name="Synthetic", author="Tests")
        vrm_export.bind_expressions(armature, mesh)

        path = FIXTURE_OUT / "synthetic_character_face.vrm"
        path.unlink(missing_ok=True)
        vrm_export.export(path, armature)
        cls.gltf = read_glb_json(path)
        cls.mesh_node = next(i for i, node in enumerate(cls.gltf["nodes"]) if "mesh" in node and "skin" in node)
        mesh_index = cls.gltf["nodes"][cls.mesh_node]["mesh"]
        cls.target_names = cls.gltf["meshes"][mesh_index]["extras"]["targetNames"]

    def bound_shape_keys(self, expression):
        preset = self.gltf["extensions"]["VRMC_vrm"]["expressions"]["preset"]
        binds = preset.get(expression, {}).get("morphTargetBinds", [])
        return sorted((bind["node"], self.target_names[bind["index"]]) for bind in binds)

    def test_blink_closes_both_eyes(self):
        self.assertEqual(
            self.bound_shape_keys("blink"), [(self.mesh_node, "blink_left"), (self.mesh_node, "blink_right")]
        )

    def test_each_eye_can_blink_on_its_own(self):
        self.assertEqual(self.bound_shape_keys("blinkLeft"), [(self.mesh_node, "blink_left")])
        self.assertEqual(self.bound_shape_keys("blinkRight"), [(self.mesh_node, "blink_right")])

    def test_vowels_drive_their_mouth_shapes(self):
        for vowel in ("aa", "ih", "ou", "ee", "oh"):
            with self.subTest(vowel=vowel):
                self.assertEqual(self.bound_shape_keys(vowel), [(self.mesh_node, vowel)])


if __name__ == "__main__":
    unittest.main()
