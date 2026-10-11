"""Exporting with MToon toon shading, the usual look for anime VTuber avatars."""

import json
import struct
import unittest

from support import FIXTURE_OUT, load_fixture
from vtuber_maker.pipeline import landmarks, meshdata, rig, vrm_export


def read_glb_json(path):
    data = path.read_bytes()
    chunk_length, _chunk_type = struct.unpack_from("<II", data, 12)
    return json.loads(data[20 : 20 + chunk_length])


class ExportToonMaterials(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture()
        vrm_export.ensure_vrm_addon()
        joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))
        armature = rig.build_armature(joints)
        rig.skin(mesh, armature)
        vrm_export.configure(armature, name="Synthetic", author="Tests")
        vrm_export.use_toon_shading(mesh)

        path = FIXTURE_OUT / "synthetic_character_toon.vrm"
        path.unlink(missing_ok=True)
        vrm_export.export(path, armature)
        cls.gltf = read_glb_json(path)

    def test_every_material_uses_mtoon(self):
        for material in self.gltf["materials"]:
            with self.subTest(material=material.get("name")):
                self.assertIn("VRMC_materials_mtoon", material.get("extensions", {}))

    def test_keeps_the_painted_texture_as_the_lit_color(self):
        self.assertIn("baseColorTexture", self.gltf["materials"][0]["pbrMetallicRoughness"])

    def test_shades_with_a_darker_tint_of_the_texture(self):
        mtoon = self.gltf["materials"][0]["extensions"]["VRMC_materials_mtoon"]
        self.assertIn("shadeMultiplyTexture", mtoon)
        shade = mtoon["shadeColorFactor"]
        self.assertTrue(all(0.0 < channel < 1.0 for channel in shade), shade)

    def test_its_pictures_are_drawn_from_mipmaps(self):
        # LINEAR (9729), as the add-on writes it, is "no mipmaps" to UniVRM: thin lines speckled in Warudo
        self.assertTrue(self.gltf["samplers"])
        self.assertEqual({sampler["minFilter"] for sampler in self.gltf["samplers"]}, {9985})


if __name__ == "__main__":
    unittest.main()
