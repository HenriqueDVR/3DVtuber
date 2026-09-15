"""Preparing an imported character: one mesh, real-world height, standing on the floor."""

import unittest

import bpy
import numpy as np

from support import load_fixture
from vtuber_maker.pipeline import meshdata, normalize


class PrepareScaledAndOffsetCharacter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture()
        mesh.scale = (0.37, 0.37, 0.37)
        mesh.location = (3.0, -2.0, 5.0)
        # Generators sometimes export accessories as separate objects.
        accessory = mesh.copy()
        accessory.data = mesh.data.copy()
        accessory.location = (3.2, -2.0, 5.1)
        bpy.context.scene.collection.objects.link(accessory)
        cls.vertex_count = len(mesh.data.vertices) + len(accessory.data.vertices)

        cls.result = normalize.prepare([mesh, accessory], height=1.6)
        cls.verts = meshdata.world_vertices(cls.result)

    def test_joins_all_parts_into_one_mesh(self):
        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        self.assertEqual(meshes, [self.result])
        self.assertEqual(len(self.result.data.vertices), self.vertex_count)

    def test_scales_to_the_requested_height(self):
        self.assertAlmostEqual(float(np.ptp(self.verts[:, 2])), 1.6, places=3)

    def test_stands_on_the_floor(self):
        self.assertAlmostEqual(float(self.verts[:, 2].min()), 0.0, places=4)

    def test_is_centred_over_the_origin(self):
        center = (self.verts.min(axis=0) + self.verts.max(axis=0)) / 2
        self.assertAlmostEqual(float(center[0]), 0.0, places=3)
        self.assertAlmostEqual(float(center[1]), 0.0, places=3)

    def test_bakes_the_transform_into_the_mesh(self):
        self.assertTrue(np.allclose(np.array(self.result.matrix_world), np.eye(4)))


if __name__ == "__main__":
    unittest.main()
