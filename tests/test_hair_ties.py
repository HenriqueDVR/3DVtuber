"""Which pieces scripts/fix_hair_weights.py takes for ties in the hair, away from Blender's scene."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from mathutils.kdtree import KDTree

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fix_hair_weights.py"


def load():
    spec = importlib.util.spec_from_file_location("fix_hair_weights", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # the fix itself only runs as a script
    return module


def tree_of(points):
    tree = KDTree(len(points))
    for k, p in enumerate(points):
        tree.insert(p, k)
    tree.balance()
    return tree


class TiesSitInTheHair(unittest.TestCase):
    """The miko's red waist bow, a tie's colour on her clothes, swung with her front locks."""

    @classmethod
    def setUpClass(cls):
        cls.fix = load()
        # a strand hanging beside the body, and the body's front
        cls.strand = np.array([[0.26, 0.0, z] for z in np.linspace(0.9, 1.4, 50)])
        cls.body = np.array([[x, -0.1, z] for x in np.linspace(-0.15, 0.15, 15) for z in np.linspace(0.8, 1.3, 25)])

    def test_a_tie_round_a_strand_is_in_the_hair(self):
        ring = np.array([[0.26 + 0.01 * np.cos(a), 0.01 * np.sin(a), 1.0] for a in np.linspace(0, 6.2, 12)])
        share = self.fix.in_the_hair(ring, tree_of(self.strand), self.body)
        self.assertGreaterEqual(share, self.fix.TIE_IN_HAIR)

    def test_a_bow_on_the_clothes_is_not(self):
        bow = np.array([[x, -0.115, z] for x in np.linspace(-0.04, 0.04, 5) for z in np.linspace(0.98, 1.04, 4)])
        share = self.fix.in_the_hair(bow, tree_of(self.strand), self.body)
        self.assertLess(share, self.fix.TIE_IN_HAIR)


if __name__ == "__main__":
    unittest.main()
