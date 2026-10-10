"""Reading a model before rigging it: what it is made of, and how it should be built."""

import unittest

import numpy as np

from support import load_fixture
from vtuber_maker.pipeline import normalize, parts, survey


def surveyed(name):
    mesh, truth = load_fixture(name)
    return survey.survey(normalize.prepare([mesh], height=truth["height"])), truth


class SurveyASimpleCharacter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.truth = surveyed("synthetic_character_parts")

    def test_measures_the_mesh(self):
        self.assertGreater(self.report["triangles"], 1000)
        self.assertEqual(self.report["size"][2], 1.64)
        self.assertIsNotNone(self.report["texture"])

    def test_reads_the_generator_s_separate_parts(self):
        self.assertTrue(self.report["parts"]["read"])
        self.assertTrue(self.report["parts"]["eye_inserts"])

    def test_hanging_hair_is_not_a_solid_head(self):
        self.assertGreater(self.report["head"]["hair_hang"], survey.HAIR_HANG)
        self.assertFalse(self.report["options"]["rigid_head"])

    def test_no_hair_of_it_lies_over_the_face(self):
        self.assertEqual(self.report["head"]["hair_over_face"], 0.0)

    def test_bare_legs_ask_for_no_cloth_chains(self):
        self.assertFalse(self.report["options"]["cloth_physics"])

    def test_says_nothing_alarming_about_a_model_in_scope(self):
        self.assertEqual([n for n in self.report["notes"] if "typical proportions" in n], [])


class ShortHairIsNotASolidHead(unittest.TestCase):
    """The head's numbers as measured on real models: only the choice is tested."""

    def options(self, share, hang, over_face=0.0):
        report = {"head": {"hair_share": share, "hair_hang": hang, "hair_over_face": over_face}, "cloth": {"share": 0.0}, "face_colour": None}
        return survey._options(report, None)

    def test_short_hair_on_a_head_that_shows_its_face_is_hair(self):
        self.assertFalse(self.options(0.095, 0.0)["rigid_head"])  # the plain boy

    def test_headwear_read_as_hair_all_over_the_head_stays_solid(self):
        self.assertTrue(self.options(0.839, 0.3)["rigid_head"])  # the witch hat
        self.assertTrue(self.options(0.794, 0.18)["rigid_head"])

    def test_hair_that_comes_down_over_the_face_is_no_headwear(self):
        self.assertFalse(self.options(0.744, 0.34, 0.047)["rigid_head"])  # the fox girl's short hair, a witch hat's numbers but for her bangs
        self.assertTrue(self.options(0.744, 0.34, 0.0)["rigid_head"])
        self.assertTrue(self.options(0.744, 0.34, survey.OVER_FACE / 2)["rigid_head"])

    def test_a_head_with_no_hair_found_stays_solid(self):
        self.assertTrue(self.options(0.0, 0.0)["rigid_head"])  # the pirate under his hat

    def test_hanging_hair_was_never_solid(self):
        self.assertFalse(self.options(0.344, 1.91)["rigid_head"])
        self.assertFalse(self.options(0.673, 0.62)["rigid_head"])


class HairOverTheFace(unittest.TestCase):
    """Two eye inserts and ten hair vertices: how many of them lie over the face."""

    EYES = [(0.04, -0.15, 1.40), (0.05, -0.15, 1.41), (-0.04, -0.15, 1.40), (-0.05, -0.15, 1.41)]

    def share(self, *hair):
        hair = list(hair) + [(0.0, 0.0, 1.55)] * (10 - len(hair))  # the rest on the crown
        verts = np.array(self.EYES + hair, dtype=float)
        labels = np.array([1, 1, 2, 2] + [3] * 10)
        found = parts.MeshParts(labels, labels == 3, -1, {"left": 1, "right": 2}, -1, np.zeros(3), np.zeros(14, bool))
        return survey._hair_over_face(verts, found)

    def test_bangs_between_the_eyes_below_their_middle_count(self):
        self.assertEqual(self.share((0.0, -0.16, 1.39), (0.02, -0.17, 1.35)), 0.2)

    def test_a_brim_over_the_brow_does_not(self):
        self.assertEqual(self.share((0.0, -0.25, 1.45), (0.03, -0.2, 1.42)), 0.0)

    def test_nor_locks_beside_the_face_or_hair_behind_the_head(self):
        self.assertEqual(self.share((0.09, -0.12, 1.30), (0.0, 0.05, 1.30)), 0.0)

    def test_without_eye_inserts_nothing_is_read(self):
        found = parts.MeshParts(np.zeros(3, int), np.ones(3, bool), -1, {}, -1, np.zeros(3), np.zeros(3, bool))
        self.assertEqual(survey._hair_over_face(np.zeros((3, 3)), found), 0.0)


class SurveyACharacterInARobe(unittest.TestCase):
    """The second real character's shape: a robe joining the legs, a body wider than its arms."""

    @classmethod
    def setUpClass(cls):
        cls.report, cls.truth = surveyed("synthetic_character_robe")

    def test_asks_for_a_rigid_head_and_cloth_chains(self):
        self.assertTrue(self.report["options"]["rigid_head"])
        self.assertTrue(self.report["options"]["cloth_physics"])

    def test_warns_that_the_legs_are_hidden(self):
        self.assertTrue([n for n in self.report["notes"] if "typical proportions" in n], self.report["notes"])

    def test_places_the_hips_from_proportions_instead_of_the_hem(self):
        hips = self.report["joints"]["hips"][2]
        self.assertAlmostEqual(hips, self.truth["joints"]["hips"][2], delta=0.08)

    def test_keeps_the_shoulders_outside_the_wide_body(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                found = abs(self.report["joints"][f"{side}_upper_arm"][0])
                self.assertAlmostEqual(found, abs(self.truth["joints"][f"{side}_upper_arm"][0]), delta=0.05)


if __name__ == "__main__":
    unittest.main()
