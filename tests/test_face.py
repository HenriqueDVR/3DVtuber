"""Finding painted-on eyes and mouth on a generated character's texture."""

import unittest

import numpy as np

from support import load_fixture
from vtuber_maker.pipeline import face, landmarks, meshdata

# Eyes on the fixture are about 3.4 cm wide.
POSITION_TOLERANCE = 0.01


class DetectFaceOnSyntheticCharacter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mesh, cls.truth = load_fixture()
        joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))
        cls.features = face.detect_features(mesh, joints)

    def assert_near_truth(self, name):
        found = np.asarray(self.features[name], dtype=float)
        expected = np.asarray(self.truth["face"][name], dtype=float)
        error = float(np.linalg.norm(found - expected))
        self.assertLess(
            error,
            POSITION_TOLERANCE,
            f"{name}: found {found.round(3)}, expected {expected.round(3)}, off by {error:.3f} m",
        )

    def test_finds_left_eye(self):
        self.assert_near_truth("left_eye")

    def test_finds_right_eye(self):
        self.assert_near_truth("right_eye")

    def test_finds_mouth(self):
        self.assert_near_truth("mouth")

    def test_left_eye_is_on_character_left(self):
        self.assertGreater(self.features["left_eye"][0], self.features["right_eye"][0])

    def test_measures_eye_size_within_a_third(self):
        width, height = self.features["eye_size"]
        expected_width, expected_height = self.truth["face"]["eye_size"]
        self.assertAlmostEqual(width, expected_width, delta=expected_width / 3)
        self.assertAlmostEqual(height, expected_height, delta=expected_height / 3)

    def test_measures_mouth_width_within_a_third(self):
        expected = self.truth["face"]["mouth_width"]
        self.assertAlmostEqual(self.features["mouth_width"], expected, delta=expected / 3)


if __name__ == "__main__":
    unittest.main()
