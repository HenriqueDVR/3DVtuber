"""Draggable markers for the joints and face points a rig is built from."""

import unittest

import bpy
import numpy as np
from mathutils import Vector

from support import load_fixture
from vtuber_maker.pipeline import face, landmarks, markers, meshdata


class MarkersOnSyntheticCharacter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh, _ = load_fixture()
        cls.joints = landmarks.detect_body(meshdata.world_vertices(cls.mesh), meshdata.triangles(cls.mesh.data))
        cls.features = face.detect_features(cls.mesh, cls.joints)

    def setUp(self):
        self.mesh.location = (0.0, 0.0, 0.0)
        markers.place(self.mesh, self.joints, self.features)

    def marker(self, key):
        return markers.collection().objects[markers.marker_name(key)]

    def test_places_a_marker_for_every_joint_and_face_point(self):
        names = {obj.name for obj in markers.collection().objects}
        self.assertEqual(names, {markers.marker_name(key) for key in [*self.joints, *markers.FACE_POINTS]})

    def test_untouched_markers_give_back_what_detection_found(self):
        joints, features = markers.read()
        for key, value in self.joints.items():
            with self.subTest(joint=key):
                np.testing.assert_allclose(joints[key], value, atol=1e-5)
        for key in markers.FACE_POINTS:
            with self.subTest(face_point=key):
                np.testing.assert_allclose(features[key], self.features[key], atol=1e-5)
        np.testing.assert_allclose(features["eye_size"], self.features["eye_size"], atol=1e-6)
        self.assertAlmostEqual(features["mouth_width"], self.features["mouth_width"], places=6)

    def test_keeps_what_markers_cannot_show(self):
        _, features = markers.read()
        np.testing.assert_allclose(features["skin_color"], self.features["skin_color"], atol=1e-6)
        np.testing.assert_allclose(features["hair_color"], self.features["hair_color"], atol=1e-6)
        self.assertEqual(features["mouth_estimated"], self.features["mouth_estimated"])

    def test_a_dragged_marker_moves_its_point(self):
        self.marker("left_lower_arm").location.z += 0.05
        joints, _ = markers.read()
        self.assertAlmostEqual(joints["left_lower_arm"][2], self.joints["left_lower_arm"][2] + 0.05, places=5)

    def test_scaling_eye_markers_resizes_the_eyes(self):
        for key in ("left_eye", "right_eye"):
            self.marker(key).scale.x *= 1.5
        _, features = markers.read()
        self.assertAlmostEqual(features["eye_size"][0], 1.5 * self.features["eye_size"][0], places=5)
        self.assertAlmostEqual(features["eye_size"][1], self.features["eye_size"][1], places=5)

    def test_scaling_the_mouth_marker_widens_the_mouth(self):
        self.marker("mouth").scale.x *= 2.0
        _, features = markers.read()
        self.assertAlmostEqual(features["mouth_width"], 2.0 * self.features["mouth_width"], places=5)

    def test_markers_move_with_the_character(self):
        self.mesh.location.x += 1.0
        joints, features = markers.read()
        np.testing.assert_allclose(joints["hips"], self.joints["hips"] + [1.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(features["mouth"], self.features["mouth"] + [1.0, 0.0, 0.0], atol=1e-5)
        self.assertEqual(markers.owner(), self.mesh)

    def test_mirroring_copies_one_side_onto_the_other(self):
        self.marker("left_lower_leg").location += Vector((0.02, 0.0, -0.03))
        self.marker("left_eye").scale.z *= 1.4
        markers.mirror("left")
        joints, _ = markers.read()
        center = np.mean([joints[key][0] for key in markers.CENTER_JOINTS])
        left = joints["left_lower_leg"]
        np.testing.assert_allclose(joints["right_lower_leg"], [2 * center - left[0], left[1], left[2]], atol=1e-5)
        np.testing.assert_allclose(self.marker("right_eye").scale, self.marker("left_eye").scale, atol=1e-6)

    def test_reading_names_a_deleted_marker(self):
        bpy.data.objects.remove(self.marker("mouth"))
        with self.assertRaisesRegex(markers.MarkersMissing, "Mouth"):
            markers.read()

    def test_placing_again_replaces_the_markers(self):
        markers.place(self.mesh, self.joints, self.features)
        self.assertEqual(len(markers.collection().objects), len(markers.LABELS))
        self.assertEqual([c.name for c in bpy.data.collections if c.name.startswith(markers.COLLECTION)], [markers.COLLECTION])
        self.assertTrue(all("." not in obj.name for obj in markers.collection().objects))


if __name__ == "__main__":
    unittest.main()
