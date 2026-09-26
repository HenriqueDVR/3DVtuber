"""Reading a model before rigging it: what it is made of, and how it should be built."""

import unittest

from support import load_fixture
from vtuber_maker.pipeline import normalize, survey


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

    def test_bare_legs_ask_for_no_cloth_chains(self):
        self.assertFalse(self.report["options"]["cloth_physics"])

    def test_says_nothing_alarming_about_a_model_in_scope(self):
        self.assertEqual([n for n in self.report["notes"] if "typical proportions" in n], [])


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
