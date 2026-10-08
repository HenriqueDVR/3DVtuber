"""Finding which way an imported character stands and faces, whatever the exporter did."""

import unittest

import numpy as np
from mathutils import Matrix

from support import load_fixture
from vtuber_maker.pipeline import meshdata, normalize, orient

# Rotations an exporter can leave a character in, as (axis, degrees).
TURNS = {
    "Y-up": ("X", -90),
    "Y-up the other way": ("X", 90),
    "facing away": ("Z", 180),
    "facing left": ("Z", 90),
    "facing right": ("Z", -90),
    "lying down": ("Y", 90),
}


def turned(verts, axis, degrees):
    rotation = np.array(Matrix.Rotation(np.radians(degrees), 3, axis))
    return verts @ rotation.T


class DetectOrientation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture("synthetic_character_parts")
        cls.verts = meshdata.world_vertices(mesh)
        cls.tris = meshdata.triangles(mesh.data)

    def test_leaves_a_model_already_in_the_pipeline_frame_alone(self):
        found = orient.detect(self.verts, self.tris)
        self.assertTrue(found.is_identity, found.describe())
        self.assertTrue(found.confident)

    def test_turns_every_exporter_orientation_back(self):
        for name, (axis, degrees) in TURNS.items():
            with self.subTest(turn=name):
                verts = turned(self.verts, axis, degrees)
                found = orient.detect(verts, self.tris)
                self.assertTrue(found.confident, f"{name}: unsure, found {found.describe()}")
                back = verts @ found.matrix.T
                error = float(np.abs(back - self.verts).max())
                self.assertLess(error, 1e-6, f"{name}: found {found.describe()}, off by {error:.4f} m")

    def test_character_left_stays_on_positive_x(self):
        verts = turned(self.verts, "Z", 180)
        back = verts @ orient.detect(verts, self.tris).matrix.T
        # the hand furthest left in the fixture must come back on the left
        self.assertGreater(back[np.argmax(self.verts[:, 0])][0], 0.0)


def with_tail(verts, tris):
    """The fixture with a tail behind it: a block from the small of the back down to the floor,
    longer behind the heels than the toes are in front of them."""
    low, high = np.array([-0.08, 0.10, 0.0]), np.array([0.14, 0.50, 0.75])
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return np.concatenate([verts, corners]), np.concatenate([tris, faces + len(verts)])


class ATailDoesNotTurnTheModelRound(unittest.TestCase):
    """A tester's pirate had a fox tail hanging to the floor behind him: at foot height it reached
    further back than his boots reached forward, and the model was built facing backwards."""

    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture("synthetic_character_parts")
        cls.verts, cls.tris = with_tail(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))

    def test_the_feet_alone_say_which_way_it_faces(self):
        found = orient.detect(self.verts, self.tris)
        self.assertTrue(found.is_identity, found.describe())
        self.assertTrue(found.confident)

    def test_everything_at_foot_height_would_have_turned_it(self):
        # the rule before: the tail counted among "the feet"
        real, orient._toes = orient._toes, lambda *args: None
        try:
            self.assertFalse(orient.detect(self.verts, self.tris).is_identity)
        finally:
            orient._toes = real

    def test_turned_any_way_it_comes_back_with_its_tail_behind(self):
        for name, (axis, degrees) in TURNS.items():
            with self.subTest(turn=name):
                verts = turned(self.verts, axis, degrees)
                back = verts @ orient.detect(verts, self.tris).matrix.T
                self.assertLess(float(np.abs(back - self.verts).max()), 1e-6, name)


class DetectOrientationOfATPose(unittest.TestCase):
    """A T-pose is about as wide as it is tall, so legs and soles alone barely told up from sideways."""

    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture("synthetic_character_tpose")
        cls.verts = meshdata.world_vertices(mesh)
        cls.tris = meshdata.triangles(mesh.data)

    def test_stands_up_with_confidence(self):
        found = orient.detect(self.verts, self.tris)
        self.assertTrue(found.is_identity, found.describe())
        self.assertTrue(found.up_sure)

    def test_turns_a_lying_t_pose_back(self):
        for name in ("Y-up", "lying down"):
            with self.subTest(turn=name):
                verts = turned(self.verts, *TURNS[name])
                back = verts @ orient.detect(verts, self.tris).matrix.T
                self.assertLess(float(np.abs(back - self.verts).max()), 1e-6)


class PrepareTurnsTheModelUpright(unittest.TestCase):
    def prepared(self, axis, degrees):
        mesh, _ = load_fixture("synthetic_character_parts")
        mesh.data.transform(Matrix.Rotation(np.radians(degrees), 4, axis))
        return normalize.prepare([mesh], height=1.6)

    def test_a_y_up_import_ends_up_standing(self):
        mesh = self.prepared("X", -90)
        verts = meshdata.world_vertices(mesh)
        self.assertAlmostEqual(float(np.ptp(verts[:, 2])), 1.6, places=3)
        self.assertGreater(float(np.ptp(verts[:, 0])), float(np.ptp(verts[:, 1])))
        self.assertTrue(normalize.orientation_of(mesh)["turned"])

    def test_a_model_facing_away_is_turned_round(self):
        mesh = self.prepared("Z", 180)
        verts = meshdata.world_vertices(mesh)
        # the feet stick out in front of the hips, which is -Y in the pipeline's frame
        feet = verts[verts[:, 2] < 0.1 * 1.6]
        hips = verts[(verts[:, 2] > 0.72) & (verts[:, 2] < 0.96)]
        self.assertLess(float(np.median(feet[:, 1])), float(np.median(hips[:, 1])))

    def test_told_to_trust_the_file_it_leaves_the_facing_as_filed(self):
        import os
        os.environ[normalize.FACING_AS_FILED] = "file"
        try:
            found = normalize.orientation_of(self.prepared("Z", 180))
            self.assertEqual((found["facing"], found["turned"], found["facing_sure"]), ([0.0, -1.0, 0.0], False, True))
            # a model lying down has no "as filed" to keep: it is still stood up
            self.assertTrue(normalize.orientation_of(self.prepared("X", -90))["turned"])
        finally:
            del os.environ[normalize.FACING_AS_FILED]

    def test_records_what_it_found(self):
        mesh = self.prepared("X", 0)
        found = normalize.orientation_of(mesh)
        self.assertEqual([round(v) for v in found["up"]], [0, 0, 1])
        self.assertEqual([round(v) for v in found["facing"]], [0, -1, 0])
        self.assertFalse(found["turned"])
        self.assertTrue(found["up_sure"] and found["facing_sure"])


if __name__ == "__main__":
    unittest.main()
