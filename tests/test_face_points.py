"""Eye and mouth points placed on the face patch image, turned into face features."""

import json
import tempfile
import unittest
from pathlib import Path

import bmesh
import bpy
import numpy as np

from vtuber_maker.pipeline import face_points

SIZE = 1000
# The patch image spans x -0.1..0.1 and z 1.3..1.5 on a flat face at y = -0.1.
X0, X1, Z0, Z1, FACE_Y = -0.1, 0.1, 1.3, 1.5, -0.1


def pixel(x, z):
    return [(x - X0) / (X1 - X0) * SIZE, (1 - (z - Z0) / (Z1 - Z0)) * SIZE]


def make_face():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new("face")
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new()
    n = 8
    grid = [[bm.verts.new((X0 + (X1 - X0) * i / n, FACE_Y, Z0 + (Z1 - Z0) * j / n)) for i in range(n + 1)] for j in range(n + 1)]
    for j in range(n):
        for i in range(n):
            face = bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
            for loop in face.loops:
                co = loop.vert.co
                loop[uv].uv = ((co.x - X0) / (X1 - X0), (co.z - Z0) / (Z1 - Z0))
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(bpy.data.materials.new(face_points.FACE_PATCH_MATERIAL))
    obj = bpy.data.objects.new("face", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


class FacePoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obj = make_face()
        cls.points = {
            "size": [SIZE, SIZE],
            # the character's right eye is on the image's left, at -x
            "right_eye": {"inner": pixel(-0.02, 1.42), "outer": pixel(-0.06, 1.42),
                          "top": pixel(-0.04, 1.43), "bottom": pixel(-0.04, 1.41)},
            "left_eye": {"inner": pixel(0.02, 1.42), "outer": pixel(0.06, 1.42),
                         "top": pixel(0.04, 1.43), "bottom": pixel(0.04, 1.41)},
            "mouth": {"left": pixel(-0.015, 1.35), "right": pixel(0.015, 1.35),
                      "top": pixel(0.0, 1.355), "bottom": pixel(0.0, 1.345)},
        }
        cls.features = face_points.apply(cls.obj, {}, cls.points)

    def test_eyes_land_on_the_face_at_their_points(self):
        np.testing.assert_allclose(self.features["left_eye"], (0.04, FACE_Y, 1.42), atol=1e-4)
        np.testing.assert_allclose(self.features["right_eye"], (-0.04, FACE_Y, 1.42), atol=1e-4)
        np.testing.assert_allclose(self.features["eye_size"], (0.04, 0.02), atol=1e-4)

    def test_mouth_centre_and_width(self):
        np.testing.assert_allclose(self.features["mouth"], (0.0, FACE_Y, 1.35), atol=1e-4)
        self.assertAlmostEqual(self.features["mouth_width"], 0.03, delta=1e-4)
        self.assertFalse(self.features["mouth_estimated"])

    def test_a_hair_strand_in_front_keeps_the_known_depth(self):
        # the surface (y = -0.1) is only 1 cm in front of -0.09: that's the face, so it's used
        features = face_points.apply(self.obj, {"left_eye": np.array([0.04, -0.09, 1.42])}, self.points)
        self.assertAlmostEqual(features["left_eye"][1], FACE_Y, delta=1e-4)
        # 5 cm in front of -0.05: that's a strand of hair, so the known depth stays
        features = face_points.apply(self.obj, {"left_eye": np.array([0.04, -0.05, 1.42])}, self.points)
        self.assertAlmostEqual(features["left_eye"][1], -0.05, delta=1e-4)

    def test_load_rejects_a_missing_point(self):
        broken = json.loads(json.dumps(self.points))
        del broken["mouth"]["top"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "face.json"
            path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(face_points.FacePointsError, "mouth is missing top"):
                face_points.load(path)

    def test_needs_a_face_patch(self):
        self.obj.data.materials[0].name = "other"
        try:
            with self.assertRaises(face_points.FacePointsError):
                face_points.apply(self.obj, {}, self.points)
        finally:
            self.obj.data.materials[0].name = face_points.FACE_PATCH_MATERIAL


if __name__ == "__main__":
    unittest.main()
