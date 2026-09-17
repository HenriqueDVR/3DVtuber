"""A hole-free model of the body under the hair, for checking where swinging hair ends up."""

import unittest

import numpy as np

from support import load_fixture
from vtuber_maker.pipeline import body_volume, face, hair, landmarks, meshdata, rig


class BodyVolumeOfSyntheticCharacter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mesh, cls.truth = load_fixture()
        joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))
        features = face.detect_features(mesh, joints)
        armature = rig.build_armature(joints)
        rig.skin(mesh, armature)
        chains = hair.build_hair_bones(mesh, armature, joints, features)
        cls.volume = body_volume.fit(mesh, joints, hair_bones={name for chain in chains for name in chain})

    def depth(self, point):
        return float(self.volume.depth(np.array([point], dtype=float))[0])

    def midpoint(self, a, b):
        return (np.array(self.truth["joints"][a]) + np.array(self.truth["joints"][b])) / 2

    def test_chest_is_inside(self):
        self.assertGreater(self.depth([0.0, 0.0, 1.15]), 0.05)

    def test_back_hidden_under_the_hair_is_inside(self):
        self.assertGreater(self.depth([0.0, 0.07, 1.2]), 0.0)

    def test_hair_hanging_behind_the_back_is_outside(self):
        self.assertLessEqual(self.depth([0.0, 0.16, 1.2]), 0.0)

    def test_gap_between_arm_and_torso_is_outside(self):
        self.assertLessEqual(self.depth([0.25, 0.0, 1.05]), 0.0)

    def test_upper_arm_is_inside(self):
        self.assertGreater(self.depth(self.midpoint("left_upper_arm", "left_lower_arm")), 0.01)

    def test_thigh_is_inside(self):
        self.assertGreater(self.depth(self.midpoint("left_upper_leg", "left_lower_leg")), 0.02)


if __name__ == "__main__":
    unittest.main()
