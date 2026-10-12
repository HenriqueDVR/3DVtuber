"""What the rig check judges: scripts/check_avatar.py's limits, away from Blender's scene."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_avatar.py"


def load():
    spec = importlib.util.spec_from_file_location("check_avatar", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # the check itself only runs as a script
    return module


class ACreaseIsNotAFold(unittest.TestCase):
    """An armpit closing as the arm comes down squashes edges on any rig, a VRoid one too."""

    @classmethod
    def setUpClass(cls):
        cls.check = load()

    def setUp(self):
        # six edges, by how much of each end the bones that move hold: three in a joint's blend
        # (the reference's armpit: 0.31 and 0.48, 0.34 and 0.81), then a cape cut into bands (one
        # end wholly on the moving side), a limb's weights reaching onto the body (one end not on
        # it at all), and an edge nothing moves
        self.share = np.array([(0.31, 0.48), (0.34, 0.81), (0.17, 0.5), (1.0, 0.58), (0.24, 0.0), (0.0, 0.0)])
        self.middle = np.arange(18, dtype=float).reshape(6, 3) / 10

    def folds(self, squashed, share=None, middle=None):
        return self.check.folds_outside_the_blend(self.share if share is None else share,
                                                  self.middle if middle is None else middle, np.array(squashed))

    def test_an_edge_shared_at_both_ends_is_a_joints_crease(self):
        crease, folded, places = self.folds([0.08, 0.1, 0.2, 1, 1, 1])
        self.assertEqual(crease.tolist(), [True, True, True, False, False, False])
        self.assertEqual((folded.tolist(), places), ([], 0))

    def test_an_edge_with_an_end_wholly_on_one_side_folds(self):
        _, folded, places = self.folds([1, 1, 1, 0.05, 0.3, 0.4])
        self.assertEqual((folded.tolist(), places), ([3, 4], 2))  # 0.4 is over the limit of 0.35

    def test_copies_of_one_edge_along_a_seam_are_one_place(self):
        # a file read back has an edge for each copy of a vertex along a texture seam
        share, middle = np.concatenate([self.share, self.share[3:5]]), np.concatenate([self.middle, self.middle[3:5]])
        _, folded, places = self.folds([1, 1, 1, 0.05, 0.3, 1, 0.05, 0.3], share, middle)
        self.assertEqual((len(folded), places), (4, 2))

    def test_a_crease_however_deep_fails_nothing(self):
        # model B, for three weeks: 8% with the arms at its sides, the armpit closing
        measured = {"unweighted": 0, "worst": {"stretch_cm": 4.3, "stretch_pose": "arms forward",
                                               "squash": 0.078, "squash_pose": "arms at sides"}}
        self.assertEqual(self.check.failures(measured), [])

    def test_two_or_three_edges_where_the_weights_step_fail_nothing(self):
        # the rim of model B's hood in a chest twist: nothing to see
        measured = {"unweighted": 0, "worst": {"squash": 0.159, "squash_pose": "chest twist",
                                               "folded_edges": 3, "fold": 0.159, "fold_pose": "chest twist"}}
        self.assertEqual(self.check.failures(measured), [])

    def test_a_run_of_edges_folding_where_the_weights_step_fails(self):
        # model D's cape, banded between two bones 25 cm from her spine: a kink in a side bend
        measured = {"unweighted": 0, "worst": {"squash": 0.047, "squash_pose": "spine side bend",
                                               "folded_edges": 14, "fold": 0.047, "fold_pose": "spine side bend"}}
        failed = self.check.failures(measured)
        self.assertEqual(len(failed), 1)
        self.assertRegex(failed[0], r"^14 edges fold to under 35% of their length with the spine side bend \(the worst to 5%\)")
        # a .vrm taken in is told the finding without the guess at its cause (engine.verdict): up to the colon
        self.assertIn(": ", failed[0])

    def test_stretch_is_judged_as_before(self):
        measured = {"unweighted": 0, "worst": {"stretch_cm": 6.5, "stretch_pose": "head turn", "squash": 0.9, "squash_pose": "leg lift"}}
        self.assertRegex(self.check.failures(measured)[0], r"^an edge stretches 6\.5 cm with the head turn")


CLEAN = {"unweighted": 0, "worst": {}}


class TheArmsOwnSurface(unittest.TestCase):
    """Whether a raised arm takes its surface along is read on the arm's own surface, not on a cape beside it."""

    @classmethod
    def setUpClass(cls):
        cls.check = load()

    def test_pieces_are_joined_by_edges_and_across_the_copies_of_a_vertex(self):
        # two triangles sharing an edge only by position (a texture seam), and a third apart
        rest = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (5, 5, 5), (6, 5, 5), (5, 6, 5)], dtype=float)
        edges = np.array([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (6, 7), (7, 8), (8, 6)])
        piece = self.check.surface_pieces(rest, edges)
        self.assertEqual(len(set(piece[:6])), 1)
        self.assertEqual(len(set(piece[6:])), 1)
        self.assertNotEqual(piece[0], piece[6])

    @staticmethod
    def all_round(count):
        return np.linspace(0, 2 * np.pi, count, endpoint=False)

    def test_a_cape_lying_on_the_arm_is_not_the_arms_surface(self):
        # model D: bare arms that go up whole, under a cape hung from the torso that stays down: 49% of
        # everything within 7 cm of the bone moved, and the line said her sleeve was on another bone. The
        # cape lies on the arm, as close to the bone as its skin, over its top alone
        off = np.array([0.035] * 40 + [0.03] * 42)
        moved = np.array([True] * 40 + [False] * 42)
        piece = np.array([0] * 40 + [1] * 42)
        about = np.concatenate([self.all_round(40), np.linspace(0.3, 2.6, 42)])
        own, everything = self.check.arm_follows(off, moved, piece, about)
        self.assertEqual(own, 1.0)
        self.assertLess(everything, 0.5)

    def test_a_body_in_one_piece_that_stays_behind_is_still_caught(self):
        # the pirate, his whole body on the head bone: only hands and cuffs went up
        off = np.array([0.03] * 40 + [0.06] * 40)
        piece, about = np.zeros(80, dtype=int), np.concatenate([self.all_round(40), self.all_round(40)])
        self.assertEqual(self.check.arm_follows(off, np.zeros(80, dtype=bool), piece, about), (0.0, 0.0))
        half = np.array([True] * 40 + [False] * 40)
        self.assertEqual(self.check.arm_follows(off, half, piece, about), (0.5, 0.5))

    def test_the_arms_own_surface_staying_behind_fails_whatever_goes_up_round_it(self):
        # the arm inside stays, a sleeve further out goes up: the arm is what is asked about
        off = np.array([0.03] * 40 + [0.06] * 40)
        moved = np.array([False] * 40 + [True] * 40)
        piece, about = np.array([0] * 40 + [1] * 40), np.concatenate([self.all_round(40), self.all_round(40)])
        own, everything = self.check.arm_follows(off, moved, piece, about)
        self.assertEqual((own, everything), (0.0, 0.5))
        self.assertTrue(any("upper arm" in line for line in self.check.failures(
            {"unweighted": 0, "worst": {}, "arm_follows": {"Left": own, "Right": 1.0}})))

    def test_a_few_vertices_of_a_trinket_are_not_the_arm(self):
        # an armband of a dozen vertices right on the bone, the arm round it left behind
        off = np.array([0.01] * 12 + [0.04] * 60)
        moved = np.array([True] * 12 + [False] * 60)
        piece, about = np.array([1] * 12 + [0] * 60), np.concatenate([self.all_round(12), self.all_round(60)])
        self.assertEqual(self.check.arm_follows(off, moved, piece, about)[0], 0.0)

    def test_with_nothing_all_round_the_bone_everything_is_read(self):
        # two half shells, neither going round: no piece is the arm's own, so all of it is asked to follow
        off = np.full(80, 0.04)
        moved = np.array([True] * 40 + [False] * 40)
        piece, about = np.array([0] * 40 + [1] * 40), np.concatenate([np.linspace(0, 3, 40), np.linspace(3.2, 6.2, 40)])
        self.assertEqual(self.check.arm_follows(off, moved, piece, about), (0.5, 0.5))

class ASeamThatOpens(unittest.TestCase):
    """The copies of a vertex along a texture seam, each with its own weights: model A's hair and
    model B's cape tore in Warudo (2026-10-07) while every pose's stretch measured as before."""

    @classmethod
    def setUpClass(cls):
        cls.check = load()

    def test_copies_are_the_vertices_at_another_vertex_place(self):
        rest = np.array([(0, 0, 1), (0.1, 0, 1), (0, 0, 1), (0.1, 0, 1 + 4e-6), (0, 0, 1), (0.3, 0, 1)], dtype=float)
        copy, twin = self.check.seam_copies(rest)
        self.assertEqual((copy.tolist(), twin.tolist()), ([2, 3, 4], [0, 1, 0]))

    def test_a_mesh_without_copies_has_none(self):
        copy, twin = self.check.seam_copies(np.arange(12, dtype=float).reshape(4, 3))
        self.assertEqual((len(copy), len(twin)), (0, 0))

    def test_which_seams_the_build_keeps_closed(self):
        # vertex groups: 0 Hair3_0, 1 Hair4_0, 2 Spine, 3 Hips, 4 LeftUpperArm, 5 RightUpperLeg; -1 no weights
        swings = np.array([True, True, False, False, False, False, False])
        on_arm = np.array([False, False, False, False, True, False, False])
        lead = np.array([2, 0, 0, 0, 2, 2, 4, -1])
        twin = np.array([2, 1, 5, 4, 4, 3, 4, 2])
        kept = self.check.seams_kept(lead, twin, swings, on_arm)
        # one bone; two chains (model A's hair); a chain and the thigh under it (model B's cape); then the
        # sides of a cut, meant to part: a chain from an arm, an arm from the spine, spine from hips;
        # an arm's own seam; and a vertex with no weights, which the check fails by itself
        self.assertEqual(kept.tolist(), [True, True, True, False, False, False, True, False])

    def test_a_place_that_comes_apart_fails(self):
        measured = dict(CLEAN, seams={"copies": 74888, "kept": 74888, "torn": 65, "torn_cm": 3.9, "pose": "swing together"})
        self.assertRegex(self.check.failures(measured)[0], r"^65 places where the surface is joined at rest come apart with "
                                                           r"the swing together \(up to 3\.9 cm\): a tear")

    def test_seams_that_stay_closed_fail_nothing(self):
        self.assertEqual(self.check.failures(dict(CLEAN, seams={"copies": 8000, "kept": 7990, "torn": 0, "torn_cm": 0.0})), [])
        # model D's builds of early October: a place or two, under 2 cm, and nothing was seen
        self.assertEqual(self.check.failures(dict(CLEAN, seams={"copies": 62, "kept": 28, "torn": 2, "torn_cm": 1.9,
                                                                "pose": "swing together"})), [])


class HairOnTheHeadWrongly(unittest.TestCase):
    """Hair the head bone holds alone far below the shoulders (a bit of model A's back hair, standing
    in the air), and hair over the head joint that the neck holds part of (model D's ledge)."""

    @classmethod
    def setUpClass(cls):
        cls.check = load()

    def findings(self, rest, on_head_bone, on_head):
        return self.check.hair_findings(np.array(rest, dtype=float), np.array(on_head_bone, dtype=float),
                                        np.array(on_head, dtype=float), shoulders=1.3, head=1.4)

    def test_what_the_head_bone_holds_below_the_shoulders_hangs(self):
        # a lock to the waist on the head bone, its copy along a seam, a chin above the shoulders, a lock on a chain
        found = self.findings([(0, 0, 0.9), (0, 0, 0.9), (0, -0.1, 1.33), (0.1, 0, 0.9)], on_head_bone=[1, 1, 1, 0], on_head=[1, 1, 1, 1])
        self.assertEqual((found["hangs_on_head"], found["hangs_cm"]), (1, 40.0))

    def test_the_skin_of_the_neck_and_chest_is_not_hair(self):
        # held by the neck and the chest, not by the head bone alone
        found = self.findings([(0, 0, 1.2), (0, 0, 1.1)], on_head_bone=[0.3, 0], on_head=[0.3, 0])
        self.assertEqual(found["hangs_on_head"], 0)

    def test_a_rig_without_arms_has_no_shoulders_to_hang_below(self):
        found = self.check.hair_findings(np.array([(0, 0, 0.9)]), np.ones(1), np.ones(1), None, 1.4)
        self.assertEqual(found["hangs_on_head"], 0)

    def test_what_is_shared_with_the_body_over_the_head_joint_is_counted(self):
        # over the joint: hair 20% and 34% on the neck, hair on the head alone, a tail held over the head on
        # the hips; and the neck's own skin right at the joint
        found = self.findings([(0, 0.1, 1.5), (0.02, 0.1, 1.5), (0, 0, 1.6), (0, 0.2, 1.7), (0, 0.05, 1.41)],
                              on_head_bone=[0.8, 0.66, 1, 0, 0.5], on_head=[0.8, 0.66, 1, 0, 0.5])
        self.assertEqual((found["shared_above"], found["shared_most"]), (2, 0.34))

    def test_limits(self):
        hair = {"hangs_on_head": 94, "hangs_cm": 25.3, "shared_above": 1578, "shared_most": 0.34, "ledge_edges": 429, "ledge_mm": 17.5}
        failed = self.check.failures(dict(CLEAN, hair=hair))
        self.assertEqual(len(failed), 2)
        self.assertRegex(failed[0], r"^94 vertices hang from the head bone to 25 cm below the shoulders: ")
        self.assertRegex(failed[1], r"^429 edges above the head joint grow over 5 mm when head and neck tilt together "
                                    r"\(the worst 18 mm\): ")
        # model D once mended (30 edges), a short-haired head the adopted rig shares with the neck all over (34)
        for edges in (30, 34):
            self.assertEqual(self.check.failures(dict(CLEAN, hair=dict(hair, hangs_on_head=0, ledge_edges=edges))), [])


class TheWhiteOfAnEye(unittest.TestCase):
    """What lies beside an iris, where the white shows: the fox girl's was her model's own socket,
    3 cm back, and only Warudo's side view showed it."""

    @classmethod
    def setUpClass(cls):
        cls.check = load()

    def setUp(self):
        # an iris disc 12 mm across at y = -0.1 that the look keys take 3 mm; skin beside it they don't move
        turn = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        ring = np.stack([0.03 + 0.006 * np.cos(turn), np.full(48, -0.1), 1.4 + 0.006 * np.sin(turn)], axis=1)
        self.basis = np.concatenate([ring, [(0.03, -0.1, 1.4)], ring + (0.2, 0, 0)])
        self.far = np.concatenate([np.full(49, 0.003), np.zeros(48)])

    def eye(self, depth_at):
        return self.check.iris_depths(self.basis, self.far, lambda origin: depth_at(np.asarray(origin)))

    def test_a_white_right_behind_the_iris_is_read_shallow(self):
        eye = self.eye(lambda origin: 0.004)
        self.assertEqual((eye["vertices"], eye["rays"]), (49, 24))
        self.assertEqual((eye["beside_mm"], eye["behind_mm"]), (4.0, 4.0))
        self.assertEqual(self.check.failures(dict(CLEAN, face={"eyes": {"Left": eye, "Right": eye}})), [])

    def test_a_socket_at_the_sides_is_read_deep_though_the_lids_lie_right_above_and_below(self):
        # deep where the ray passes left or right of the disc (10 of the 24), the lids' own skin above and below
        eye = self.eye(lambda origin: 0.004 if abs(origin[2] - 1.4) > 0.005 else 0.03)
        self.assertEqual((eye["beside_mm"], eye["beside_median_mm"]), (30.0, 4.0))
        failed = self.check.failures(dict(CLEAN, face={"eyes": {"Left": eye}}))
        self.assertRegex(failed[0], r"^the surface beside the left iris lies 30 mm behind it: ")

    def test_a_ray_that_meets_nothing_has_gone_under_a_lid(self):
        # an eye set back behind the lids' opening (model A's): past its sides nothing lies behind at all
        eye = self.eye(lambda origin: None if abs(origin[0] - 0.03) > 0.005 else 0.001)
        self.assertEqual((eye["beside_mm"], eye["rays"]), (1.0, 10))

    def test_too_few_rays_that_meet_anything_measure_nothing(self):
        self.assertIsNone(self.eye(lambda origin: None))

    def test_an_eye_that_does_not_look_about_is_not_measured(self):
        self.assertIsNone(self.check.iris_depths(self.basis, np.zeros(len(self.basis)), lambda origin: 0.03))

    def test_the_limit_leaves_a_patch_standing_off_a_round_face(self):
        # model B's sliding eye patches stand 11 to 13 mm off her face at their outer rim: nothing to see
        eye = {"vertices": 1218, "behind_mm": 4.9, "beside_mm": 13.2, "beside_median_mm": 6.9, "rays": 24}
        self.assertEqual(self.check.failures(dict(CLEAN, face={"eyes": {"Right": eye}})), [])
        # nor is the catgirl's white, 16 mm down her sockets, told from them by its depth; the fox girl's at 29 is
        self.assertEqual(self.check.failures(dict(CLEAN, face={"eyes": {"Right": dict(eye, beside_mm=16.2)}})), [])
        self.assertEqual(len(self.check.failures(dict(CLEAN, face={"eyes": {"Right": dict(eye, beside_mm=29.3)}}))), 1)


class TheDrawnMouthInsideOut(unittest.TestCase):
    """Shape keys add up: jaw, pucker and funnel each narrow the drawn mouth, together past nothing."""

    @classmethod
    def setUpClass(cls):
        cls.check = load()

    def setUp(self):
        # a patch of two triangles 20 mm wide, a line when closed; `aa` opens it 10 mm
        self.basis = np.array([(-0.01, 0, 1.3), (0.01, 0, 1.3), (0.01, 0, 1.3), (-0.01, 0, 1.3)])
        self.triangles = np.array([(0, 1, 2), (0, 2, 3)])
        self.opened = self.basis + np.array([(0, 0, 0.005), (0, 0, 0.005), (0, 0, -0.005), (0, 0, -0.005)])

    def narrowed(self, by, opened=0.004):
        """A key that takes `by` of the closed width off the mouth and opens it a little."""
        return self.basis * (1 - by, 1, 1) + np.array([(0, 0, opened / 2)] * 2 + [(0, 0, -opened / 2)] * 2)

    def turned(self, pucker, funnel, triangles=None):
        shapes = {"aa": self.opened, "jawOpen": self.opened, "mouthPucker": self.narrowed(pucker), "mouthFunnel": self.narrowed(funnel)}
        return self.check.mouth_turned(self.basis, shapes, self.triangles if triangles is None else triangles)

    def test_shapes_that_leave_it_some_width_between_them_turn_nothing(self):
        found = self.turned(0.4, 0.4)
        self.assertEqual((found["mixes"], found["of"], found["share"]), (0, 27, 0.0))
        self.assertEqual(self.check.failures(dict(CLEAN, face={"mouth_turned": found})), [])

    def test_shapes_that_narrow_it_past_nothing_turn_it(self):
        # 0.7 and 0.6 of the width: both at full leave -0.3 of it, the mouth out the other side
        found = self.turned(0.7, 0.6)
        self.assertGreater(found["mixes"], 0)
        self.assertEqual(found["share"], 1.0)
        self.assertEqual((found["mix"].get("mouthPucker"), found["mix"].get("mouthFunnel")), (1.0, 1.0))
        failed = self.check.failures(dict(CLEAN, face={"mouth_turned": found}))
        self.assertRegex(failed[0], r"^the drawn mouth turns inside out in \d+ of 27 mixes of its shapes \(100% of it facing backwards with ")

    def test_a_patch_with_sides_of_its_own_has_no_front_to_turn_from(self):
        # model B's mouth, from before the free mouth: part of it faces away in every shape
        found = self.turned(0.7, 0.6, np.array([(0, 1, 2), (0, 2, 3), (0, 3, 2)]))
        self.assertEqual((found["mixes"], found["of"]), (0, 0))

    def test_a_mouth_without_a_drawn_patch_is_not_measured(self):
        found = self.check.mouth_turned(self.basis, {"aa": self.opened}, self.triangles[:0])
        self.assertEqual((found["mixes"], found["of"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
