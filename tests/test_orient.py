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


def box(low, high):
    """Vertices and triangles of a block."""
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return corners, faces


def with_raised_tail(verts, tris, top):
    """The fixture with a tail held up behind it: out from the small of the back, then up to `top`."""
    for low, high in (((-0.07, 0.08, 0.88), (0.07, 0.62, 1.0)), ((-0.07, 0.48, 1.0), (0.07, 0.62, top))):
        corners, faces = box(low, high)
        verts, tris = np.concatenate([verts, corners]), np.concatenate([tris, faces + len(verts)])
    return verts, tris


class ARaisedTailDoesNotTurnTheModel(unittest.TestCase):
    """A tail held up behind an A-pose makes the body deeper than it is wide. The thinner axis was
    taken for its depth, the side view for its front: such a model faced sideways or stood on its head."""

    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture("synthetic_character_parts")
        cls.plain = meshdata.world_vertices(mesh), meshdata.triangles(mesh.data)
        cls.height = float(cls.plain[0][:, 2].max())
        cls.tails = {"to the shoulders": 0.8 * cls.height, "over the head": 1.25 * cls.height}

    def test_the_tail_makes_it_deeper_than_wide(self):
        verts, _ = with_raised_tail(*self.plain, self.tails["to the shoulders"])
        self.assertGreater(float(np.ptp(verts[:, 1])), float(np.ptp(verts[:, 0])))

    def test_it_stands_and_faces_as_it_did(self):
        for name, top in self.tails.items():
            with self.subTest(tail=name):
                found = orient.detect(*with_raised_tail(*self.plain, top))
                self.assertTrue(found.is_identity, found.describe())
                self.assertTrue(found.confident)

    def test_turned_any_way_it_comes_back(self):
        for name, top in self.tails.items():
            verts, tris = with_raised_tail(*self.plain, top)
            for turn, (axis, degrees) in TURNS.items():
                with self.subTest(tail=name, turn=turn):
                    there = turned(verts, axis, degrees)
                    back = there @ orient.detect(there, tris).matrix.T
                    self.assertLess(float(np.abs(back - verts).max()), 1e-6)

    def test_the_head_top_is_found_under_a_tail_that_ends_above_it(self):
        points = meshdata.sample_surface(*with_raised_tail(*self.plain, self.tails["over the head"]), orient.SAMPLES)
        self.assertAlmostEqual(orient.head_top(points), self.height, delta=0.01)

    def test_no_head_top_is_named_when_the_head_is_the_top(self):
        self.assertIsNone(orient.head_top(meshdata.sample_surface(*self.plain, orient.SAMPLES)))
        lower = with_raised_tail(*self.plain, self.tails["to the shoulders"])
        self.assertIsNone(orient.head_top(meshdata.sample_surface(*lower, orient.SAMPLES)))


class ATailAboveTheHeadIsNotTheBodysHeight(unittest.TestCase):
    """A fox tail ending 37 cm above the head made the body four fifths its size and put the hips,
    neck and head 25 to 40 cm off; A-pose models read as arms at the sides."""

    def prepared(self, tail):
        import bmesh
        mesh, _ = load_fixture("synthetic_character_parts")
        if tail:
            verts, tris = meshdata.world_vertices(mesh), meshdata.triangles(mesh.data)
            added, faces = with_raised_tail(verts[:0], tris[:0], 1.25 * float(verts[:, 2].max()))
            bm = bmesh.new()
            bm.from_mesh(mesh.data)
            new = [bm.verts.new(v) for v in added]
            for face in faces:
                bm.faces.new([new[i] for i in face])
            bm.to_mesh(mesh.data)
            bm.free()
        mesh = normalize.prepare([mesh], height=1.6)
        return meshdata.world_vertices(mesh), meshdata.triangles(mesh.data)

    def test_the_body_is_as_tall_and_its_joints_are_where_they_were(self):
        from vtuber_maker.pipeline import landmarks
        plain = landmarks.detect_body(*self.prepared(tail=False))
        verts, tris = self.prepared(tail=True)
        self.assertAlmostEqual(float(verts[:, 2].max()), 1.25 * 1.6, delta=0.02)
        warnings = []
        joints = landmarks.detect_body(verts, tris, warnings)
        self.assertTrue(any("stands higher" in w for w in warnings), warnings)
        # the tail widens the box the model is centred by: joints compared from the hips
        for key in ("head", "neck", "left_upper_arm", "right_hand", "left_upper_leg", "right_foot"):
            with self.subTest(joint=key):
                moved = (joints[key] - joints["hips"]) - (plain[key] - plain["hips"])
                self.assertLess(float(np.linalg.norm(moved)), 0.03, key)
        self.assertAlmostEqual(float(joints["hips"][2]), float(plain["hips"][2]), delta=0.03)


def with_horns(verts, tris, share):
    """The fixture with two horns: blocks either side of the crown, ending `share` of its height above it."""
    top, low = float(verts[:, 2].max()), float(verts[:, 2].min())
    height = top - low
    crown = verts[verts[:, 2] > top - 0.05 * height]
    x, y = float(np.median(crown[:, 0])), float(np.median(crown[:, 1]))
    for sign in (-1, 1):
        inner, outer = sorted((x + sign * 0.035 * height, x + sign * 0.06 * height))
        corners, faces = box((inner, y - 0.012 * height, top - 0.04 * height), (outer, y + 0.012 * height, top + share * height))
        verts, tris = np.concatenate([verts, corners]), np.concatenate([tris, faces + len(verts)])
    return verts, tris


class HornsAreNotTheBodysHeight(unittest.TestCase):
    """A tester's demon girl: her horns were the model's top, so every joint placed by a share of the
    height sat too high and the face patch was framed among the horns. Model A with horns 15% of
    her height added failed as arms at the sides."""

    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture("synthetic_character_parts")
        cls.plain = meshdata.world_vertices(mesh), meshdata.triangles(mesh.data)
        cls.height = float(cls.plain[0][:, 2].max())

    def test_the_head_top_is_found_between_horns(self):
        horned = with_horns(*self.plain, 0.15)
        self.assertAlmostEqual(float(horned[0][:, 2].max()), 1.15 * self.height, delta=0.01)
        points = meshdata.sample_surface(*horned, orient.SAMPLES)
        self.assertAlmostEqual(orient.head_top(points), self.height, delta=0.01)
        self.assertAlmostEqual(orient.head_top(points, verts=horned[0]), self.height, delta=0.01)

    def test_small_ears_are_still_the_head(self):
        # under ABOVE_HEAD of the body: the model's top is the head's, as it was
        self.assertIsNone(orient.head_top(meshdata.sample_surface(*with_horns(*self.plain, 0.03), orient.SAMPLES)))

    def test_it_stands_and_faces_as_it_did(self):
        found = orient.detect(*with_horns(*self.plain, 0.15))
        self.assertTrue(found.is_identity, found.describe())

    def test_the_body_is_as_tall_and_its_joints_are_where_they_were(self):
        import bmesh
        from vtuber_maker.pipeline import landmarks

        def prepared(horns):
            mesh, _ = load_fixture("synthetic_character_parts")
            if horns:
                verts, tris = meshdata.world_vertices(mesh), meshdata.triangles(mesh.data)
                added, faces = with_horns(verts, tris, 0.15)
                bm = bmesh.new()
                bm.from_mesh(mesh.data)
                new = [bm.verts.new(v) for v in added[len(verts):]]
                for face in faces[len(tris):]:
                    bm.faces.new([new[i - len(verts)] for i in face])
                bm.to_mesh(mesh.data)
                bm.free()
            mesh = normalize.prepare([mesh], height=1.6)
            return meshdata.world_vertices(mesh), meshdata.triangles(mesh.data)

        plain = landmarks.detect_body(*prepared(False))
        verts, tris = prepared(True)
        self.assertAlmostEqual(float(verts[:, 2].max()), 1.15 * 1.6, delta=0.02)
        warnings = []
        joints = landmarks.detect_body(verts, tris, warnings)
        self.assertTrue(any("stands higher" in w for w in warnings), warnings)
        for key in ("hips", "head", "neck", "left_upper_arm", "right_hand", "left_upper_leg", "right_foot"):
            with self.subTest(joint=key):
                self.assertLess(float(np.linalg.norm(joints[key] - plain[key])), 0.03, key)


class ATailBehindTheLegsDoesNotHideThem(unittest.TestCase):
    """From the front, a tail hanging to the floor behind the legs fills the gap between them: the
    crotch was not found and hips, knees and ankles went to typical proportions, 15 to 33 cm from
    where they sat without the tail, at the depth of legs and tail together."""

    @classmethod
    def setUpClass(cls):
        from vtuber_maker.pipeline import landmarks
        mesh, _ = load_fixture("synthetic_character_parts")
        cls.landmarks = landmarks
        cls.plain = meshdata.world_vertices(mesh), meshdata.triangles(mesh.data)
        cls.tailed = with_tail(*cls.plain)
        cls.legs = [f"{side}_{part}" for side in ("left", "right") for part in ("upper_leg", "lower_leg", "foot")]

    def test_the_legs_are_found_where_they_are_without_it(self):
        plain = self.landmarks.detect_body(*self.plain)
        warnings = []
        joints = self.landmarks.detect_body(*self.tailed, warnings)
        self.assertFalse([w for w in warnings if "typical proportions" in w], warnings)
        for key in self.legs + ["hips"]:
            with self.subTest(joint=key):
                self.assertLess(float(np.linalg.norm(joints[key] - plain[key])), 0.03)

    def test_read_from_the_front_alone_they_went_to_proportions(self):
        real, self.landmarks._leg_pair = self.landmarks._leg_pair, lambda *args: None
        try:
            warnings = []
            self.landmarks.detect_body(*self.tailed, warnings)
            self.assertTrue([w for w in warnings if "typical proportions" in w])
        finally:
            self.landmarks._leg_pair = real

    def test_legs_nothing_hides_are_read_as_before(self):
        real = self.landmarks._leg_pair
        after = self.landmarks.detect_body(*self.plain)
        self.landmarks._leg_pair = lambda *args: None
        try:
            before = self.landmarks.detect_body(*self.plain)
        finally:
            self.landmarks._leg_pair = real
        for key in before:
            self.assertTrue(np.array_equal(before[key], after[key]), key)


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
