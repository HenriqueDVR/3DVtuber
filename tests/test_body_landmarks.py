"""Finding humanoid joints on a fused, unrigged character mesh."""

import unittest

import numpy as np

from support import load_fixture
from vtuber_maker.pipeline import landmarks, meshdata

# About 3% of the fixture's 1.64 m height.
TOLERANCE = 0.05


class DetectBodyOnSyntheticCharacter(unittest.TestCase):
    FIXTURE = "synthetic_character"

    @classmethod
    def setUpClass(cls):
        mesh, cls.truth = load_fixture(cls.FIXTURE)
        cls.joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))

    def assert_joint_near_truth(self, name):
        found = np.asarray(self.joints[name], dtype=float)
        expected = np.asarray(self.truth["joints"][name], dtype=float)
        error = float(np.linalg.norm(found - expected))
        self.assertLess(
            error, TOLERANCE, f"{name}: found {found.round(3)}, expected {expected.round(3)}, off by {error:.3f} m"
        )

    def test_finds_hips(self):
        self.assert_joint_near_truth("hips")

    def test_finds_left_leg_joints(self):
        for name in ("left_upper_leg", "left_lower_leg", "left_foot"):
            with self.subTest(joint=name):
                self.assert_joint_near_truth(name)

    def test_finds_right_leg_joints(self):
        for name in ("right_upper_leg", "right_lower_leg", "right_foot"):
            with self.subTest(joint=name):
                self.assert_joint_near_truth(name)

    def test_finds_left_arm_joints(self):
        for name in ("left_upper_arm", "left_lower_arm", "left_hand"):
            with self.subTest(joint=name):
                self.assert_joint_near_truth(name)

    def test_finds_right_arm_joints(self):
        for name in ("right_upper_arm", "right_lower_arm", "right_hand"):
            with self.subTest(joint=name):
                self.assert_joint_near_truth(name)

    def test_finds_neck_and_head(self):
        for name in ("neck", "head"):
            with self.subTest(joint=name):
                self.assert_joint_near_truth(name)

    def test_character_left_is_positive_x(self):
        self.assertGreater(self.joints["left_hand"][0], 0.0)
        self.assertLess(self.joints["right_hand"][0], 0.0)


class DetectBodyWithArmsHangingIntoHair(DetectBodyOnSyntheticCharacter):
    """Arms close to the body with long hair fused behind them, so only the hands stand apart."""

    FIXTURE = "synthetic_character_arms_down"


if __name__ == "__main__":
    unittest.main()
