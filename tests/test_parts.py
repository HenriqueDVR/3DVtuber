"""Characters built from separate parts, like Tripo's P2 meshes: hair, eye inserts and a mouth insert."""

import math
import unittest
from unittest import mock

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from support import FIXTURE_OUT, load_fixture
from test_build_avatar import read_glb_json
from vtuber_maker.pipeline import build, face_rig, landmarks, meshdata, parts, rig, texture_space, vrm_export

FIXTURE = "synthetic_character_parts"


def evaluated_world_vertices(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    coords = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", coords)
    evaluated.to_mesh_clear()
    matrix = np.array(obj.matrix_world)
    return coords.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]


class PartsFixture(unittest.TestCase):
    @classmethod
    def load(cls):
        cls.mesh, cls.truth = load_fixture(FIXTURE)
        cls.verts = meshdata.world_vertices(cls.mesh)
        cls.joints = landmarks.detect_body(cls.verts, meshdata.triangles(cls.mesh.data))
        cls.found = parts.find(cls.mesh, cls.joints)

    def near(self, point, radius=0.03):
        return np.linalg.norm(self.verts - np.asarray(point), axis=1) < radius

    def part_centre(self, part):
        return self.verts[self.found.labels == part].mean(axis=0)


class FindParts(PartsFixture):
    @classmethod
    def setUpClass(cls):
        cls.load()

    def test_finds_both_eye_inserts(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                error = np.linalg.norm(self.part_centre(self.found.eyes[side]) - self.truth["parts"][f"{side}_eye"])
                self.assertLess(error, 0.005)

    def test_finds_the_mouth_insert(self):
        error = np.linalg.norm(self.part_centre(self.found.mouth) - self.truth["parts"]["mouth"])
        self.assertLess(error, 0.005)

    def test_hair_covers_hanging_locks_but_not_the_face_or_hands(self):
        for tip in ("left_long_tip", "right_long_tip", "back_tip"):
            with self.subTest(tip=tip):
                # The back tip lies close to the torso's back too, so check the nearest vertex only.
                nearest = np.argmin(np.linalg.norm(self.verts - np.asarray(self.truth["hair"][tip]), axis=1))
                self.assertTrue(self.found.hair[nearest])
        for point in (self.truth["parts"]["left_eye"], self.truth["joints"]["left_hand"]):
            with self.subTest(point=point):
                self.assertFalse(self.found.hair[self.near(point, 0.02)].any())

    def test_face_features_come_from_the_inserts(self):
        features = parts.face_features(self.mesh, self.found)
        self.assertFalse(features["mouth_estimated"])
        for key in ("left_eye", "right_eye", "mouth"):
            with self.subTest(key=key):
                self.assertLess(np.linalg.norm(features[key] - self.truth["parts"][key]), 0.005)
        np.testing.assert_allclose(features["eye_size"], self.truth["parts"]["eye_size"], rtol=0.25)
        self.assertAlmostEqual(features["mouth_width"], self.truth["parts"]["mouth_width"], delta=0.006)


class FaceFusedWithBody(PartsFixture):
    """A face that is one surface with the whole body (model A's T-pose FBX): only its head end is the face."""

    @classmethod
    def setUpClass(cls):
        cls.load()
        # the fixture's face is already one surface with the torso; tie it to a foot as well
        face = np.flatnonzero(cls.found.labels == cls.found.face)
        foot = int(np.argmin(cls.verts[:, 2]))
        mesh = cls.mesh.data
        # and a long lock of hair to the face, as model A's T-pose FBX has its back hair in its body's surface
        cls.lock = int(np.argmin(np.linalg.norm(cls.verts - np.asarray(cls.truth["hair"]["left_long_tip"]), axis=1)))
        lock_part = np.flatnonzero(cls.found.labels == cls.found.labels[cls.lock])
        # through bmesh: `mesh.edges.add` crashes Blender 5.2.2 itself (access violation filling the
        # new edges' attributes), which took the whole suite down there
        joined = bmesh.new()
        joined.from_mesh(mesh)
        joined.verts.ensure_lookup_table()
        for a, b in ((int(face[0]), foot), (int(face[0]), int(lock_part[np.argmax(cls.verts[lock_part, 2])]))):
            joined.edges.new((joined.verts[a], joined.verts[b]))
        joined.to_mesh(mesh)
        joined.free()
        mesh.update()
        cls.fused = parts.find(cls.mesh, cls.joints)
        cls.foot = foot

    def test_the_face_part_reaches_the_feet(self):
        self.assertTrue((self.fused.labels == self.fused.face)[self.foot])

    def test_face_area_stops_at_the_neck(self):
        self.assertGreater(self.verts[self.fused.face_area, 2].min(), self.joints["neck"][2])

    def test_face_area_keeps_the_face_round_the_eyes(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                # the fringe by the eyes is hair, tied into the face's surface with the lock
                around = self.near(self.truth["parts"][f"{side}_eye"], 0.04) & (self.fused.labels == self.fused.face) & ~self.fused.hair
                self.assertTrue(around.any())
                self.assertTrue(self.fused.face_area[around].all())

    def test_face_shapes_leave_the_body_fixed(self):
        self.assertTrue(self.fused.fixed()[self.foot])

    def test_hair_in_the_fused_surface_is_still_hair(self):
        self.assertTrue((self.fused.labels == self.fused.face)[self.lock])
        self.assertTrue(self.fused.hair[self.lock])
        np.testing.assert_array_equal(self.fused.hair, self.found.hair)
        self.assertFalse(self.fused.face_area[self.fused.hair].any())

    def test_the_body_and_the_face_are_not_hair(self):
        self.assertFalse(self.fused.hair[self.foot])
        for side in ("left", "right"):
            self.assertFalse(self.fused.hair[self.near(self.truth["parts"][f"{side}_eye"], 0.02)].any())


class HairShadedInTwoTones(PartsFixture):
    """Hair dark in front with pale highlights (model D): well under half of it is the crown's colour."""

    @classmethod
    def setUpClass(cls):
        cls.load()
        tip = int(np.argmin(np.linalg.norm(cls.verts - np.asarray(cls.truth["hair"]["back_tip"]), axis=1)))
        cls.piece = cls.found.labels == cls.found.labels[tip]
        cls.colours = texture_space.vertex_colors(cls.mesh, texture_space.base_color_image(cls.mesh))

    def found_with(self, share):
        """The parts with only the top `share` of the hair piece left its own colour."""
        colours = self.colours.copy()
        members = np.flatnonzero(self.piece)
        down = members[np.argsort(-self.verts[members, 2])]
        colours[down[int(share * len(down)):], :3] = 1.0 - self.found.hair_color[:3]
        with mock.patch.object(texture_space, "vertex_colors", return_value=colours):
            return parts.find(self.mesh, self.joints)

    def test_the_piece_is_big_and_hangs_past_the_neck(self):
        self.assertGreater(self.piece.mean(), parts.SHADED_SIZE)
        self.assertLess(self.verts[self.piece, 2].min(), self.joints["neck"][2])
        self.assertTrue(self.found.hair[self.piece].all())

    def test_a_third_of_it_in_the_hair_colour_is_still_hair(self):
        self.assertTrue(self.found_with(0.33).hair[self.piece].all())

    def test_a_tenth_of_it_in_the_hair_colour_is_not(self):
        self.assertFalse(self.found_with(0.1).hair[self.piece].any())


class HairDrawnInShades(PartsFixture):
    """Pale hair drawn in many shades of one colour (the fox girl): its commonest colour at the crown
    is the brightest, and most of every piece is further from it than HAIR_COLOR_DISTANCE."""

    PALE = np.array([0.93, 0.91, 0.86])

    @classmethod
    def setUpClass(cls):
        cls.load()
        cls.colours = texture_space.vertex_colors(cls.mesh, texture_space.base_color_image(cls.mesh))
        # the pieces that are hair and nothing else: the fringe in the face's own surface is read by `fused_hair`
        cls.pieces = [p for p in np.unique(cls.found.labels[cls.found.hair]) if cls.found.hair[cls.found.labels == p].all()]
        cls.whole = np.isin(cls.found.labels, cls.pieces)

    def found_with(self, darkest, lightest):
        """The parts with the top fifth of each hair piece pale and the rest of it from `darkest` to `lightest` of that."""
        colours = self.colours.copy()
        for piece in self.pieces:
            members = np.flatnonzero(self.found.labels == piece)
            down = members[np.argsort(-self.verts[members, 2])]
            shade = np.ones(len(down))
            shade[len(down) // 5:] = np.linspace(lightest, darkest, len(down) - len(down) // 5)
            colours[down, :3] = self.PALE * shade[:, None]
        with mock.patch.object(texture_space, "vertex_colors", return_value=colours):
            return parts.find(self.mesh, self.joints)

    def test_the_shades_are_too_far_from_the_crown_for_the_plain_rule(self):
        self.assertGreater(np.linalg.norm(self.PALE) * (1 - 0.8), parts.HAIR_COLOR_DISTANCE)

    def test_hair_in_shades_of_the_crown_is_hair(self):
        self.assertTrue(self.whole.any())
        self.assertTrue(self.found_with(0.72, 0.8).hair[self.whole].all())

    def test_a_colour_darker_than_any_shade_of_it_is_not(self):
        self.assertFalse(self.found_with(0.3, 0.4).hair[self.whole].any())


class SkinParts(PartsFixture):
    @classmethod
    def setUpClass(cls):
        cls.load()
        cls.armature = rig.build_armature(cls.joints)
        # skinning can cut sleeves free of the torso, adding vertices
        cls.found = rig.skin(cls.mesh, cls.armature, cls.found)
        cls.verts = meshdata.world_vertices(cls.mesh)

    def tearDown(self):
        for bone in self.armature.pose.bones:
            bone.rotation_mode = "XYZ"
            bone.rotation_euler = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()

    def displacement_after_bending(self, bone_key, degrees):
        bone = self.armature.pose.bones[rig.BONE_NAMES[bone_key]]
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = (math.radians(degrees), 0.0, 0.0)
        bpy.context.view_layer.update()
        return np.linalg.norm(evaluated_world_vertices(self.mesh) - self.verts, axis=1)

    def test_bending_the_elbow_leaves_the_hair_beside_the_arm(self):
        moved = self.displacement_after_bending("left_lower_arm", 60)
        self.assertGreater(moved[self.near(self.truth["joints"]["left_hand"])].mean(), 0.1)
        self.assertLess(moved[self.near(self.truth["hair"]["left_long_tip"], 0.02)].max(), 0.002)

    def test_a_separate_glove_follows_the_hand(self):
        front = np.asarray(self.truth["parts"]["left_glove_front"])
        glove = self.found.labels == self.found.labels[np.argmin(np.linalg.norm(self.verts - front, axis=1))]
        moved = self.displacement_after_bending("left_lower_arm", 60)
        self.assertGreater(moved[glove].mean(), 0.05)

    def test_raising_the_arm_leaves_the_hair_beside_it(self):
        moved = self.displacement_after_bending("left_upper_arm", 60)
        self.assertLess(moved[self.near(self.truth["hair"]["left_long_tip"], 0.02)].max(), 0.002)

    def test_raising_the_arms_leaves_the_middle_of_the_chest(self):
        # arm and shoulder weight spread onto the chest caves it in as the arms move
        hips, neck = self.joints["hips"], self.joints["neck"]
        chest = ((np.abs(self.verts[:, 0] - hips[0]) < 0.03)
                 & (self.verts[:, 2] > hips[2] + 0.3 * (neck[2] - hips[2]))
                 & (self.verts[:, 2] < neck[2] - 0.03) & ~self.found.hair)
        self.assertTrue(chest.any(), "fixture should have a chest")
        for bone in ("left_upper_arm", "left_shoulder"):
            with self.subTest(bone=bone):
                moved = self.displacement_after_bending(bone, 45)
                self.assertLess(moved[chest].max(), 0.003)
                self.tearDown()

    def test_hair_turns_with_the_head(self):
        moved = self.displacement_after_bending("head", 30)
        self.assertGreater(moved[self.near(self.truth["hair"]["left_long_tip"], 0.02)].mean(), 0.05)

    def test_every_vertex_weights_sum_to_one(self):
        totals = np.array([sum(g.weight for g in v.groups) for v in self.mesh.data.vertices])
        bad = int(np.sum(~np.isclose(totals, 1.0, atol=0.02)))
        self.assertEqual(bad, 0, f"{bad} of {len(totals)} vertices have weights not summing to 1")


class FaceShapesFromInserts(PartsFixture):
    """The fixture mesh has an identity transform, so object space is world space."""

    @classmethod
    def setUpClass(cls):
        cls.load()
        cls.features = parts.face_features(cls.mesh, cls.found)
        face_rig.set_eyelids_from_inserts(cls.mesh, cls.features, cls.found)
        face_rig.add_shape_keys(cls.mesh, cls.features)
        cls.rest = evaluated_world_vertices(cls.mesh)

    def tearDown(self):
        for block in self.mesh.data.shape_keys.key_blocks:
            block.value = 0.0

    def displacement_with(self, key):
        """Displacement of the original vertices (the lid covers added after them are left out)."""
        self.mesh.data.shape_keys.key_blocks[key].value = 1.0
        count = len(self.found.labels)
        return evaluated_world_vertices(self.mesh)[:count] - self.rest[:count]

    def test_blink_left_squashes_the_left_eye_shut(self):
        iris = self.found.labels == self.found.eyes["left"]
        rest = self.rest[: len(iris)]
        closed = rest + self.displacement_with("blink_left")
        self.assertLess(np.ptp(closed[iris, 2]), 0.25 * np.ptp(rest[iris, 2]))

    def test_closed_eye_shows_the_lid_cover_from_the_front(self):
        eye = self.features["left_eye"]
        self.assertNotEqual(self.front_material(eye[0], eye[2]), face_rig.EYELID_MATERIAL)
        self.mesh.data.shape_keys.key_blocks["blink_left"].value = 1.0
        self.assertEqual(self.front_material(eye[0], eye[2]), face_rig.EYELID_MATERIAL)

    def test_aa_shows_the_inside_of_the_mouth(self):
        mouth = self.features["mouth"]
        x, z = mouth[0], mouth[2] - 0.2 * self.features["mouth_width"]
        self.assertNotEqual(self.front_material(x, z), face_rig.MOUTH_PATCH_MATERIAL)
        self.mesh.data.shape_keys.key_blocks["aa"].value = 1.0
        self.assertEqual(self.front_material(x, z), face_rig.MOUTH_PATCH_MATERIAL)

    def front_material(self, x, z):
        tree = BVHTree.FromObject(self.mesh, bpy.context.evaluated_depsgraph_get())
        _hit, _normal, polygon, _distance = tree.ray_cast(Vector((x, -5.0, z)), Vector((0.0, 1.0, 0.0)))
        return self.mesh.material_slots[self.mesh.data.polygons[polygon].material_index].material.name

    def test_blink_left_leaves_the_right_eye_and_the_hair_alone(self):
        moved = np.abs(self.displacement_with("blink_left"))
        self.assertLess(moved[self.found.labels == self.found.eyes["right"]].max(), 1e-4)
        self.assertLess(moved[self.found.hair].max(), 1e-4)

    def test_aa_opens_the_mouth_without_moving_the_face(self):
        # the mouth opens as its own patch; bending a coarse face round it dents the face
        moved = np.abs(self.displacement_with("aa"))
        self.assertLess(moved[self.found.labels == self.found.face].max(), 1e-6)

    def test_open_mouth_is_round_at_the_corners(self):
        self.mesh.data.shape_keys.key_blocks["aa"].value = 1.0
        mouth, width = self.features["mouth"], self.features["mouth_width"]
        half_width, half_height, drop = (share * width for share in face_rig.MOUTH_SHAPES["aa"])
        middle_z = mouth[2] - drop
        # well inside the side of the opening, but not at its top or bottom corner
        self.assertEqual(self.front_material(mouth[0] + 0.85 * half_width, middle_z), face_rig.MOUTH_PATCH_MATERIAL)
        self.assertNotEqual(self.front_material(mouth[0] + 0.85 * half_width, middle_z - 0.9 * half_height),
                            face_rig.MOUTH_PATCH_MATERIAL)


class BuildAvatarFromParts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture(FIXTURE)
        vrm_export.ensure_vrm_addon()
        path = FIXTURE_OUT / "synthetic_character_parts.vrm"
        path.unlink(missing_ok=True)
        cls.report = build.build_avatar([mesh], path, name="Parts", author="Tests")
        cls.gltf = read_glb_json(path)

    def test_uses_the_inserts_for_the_face(self):
        self.assertFalse(self.report["mouth_estimated"])
        self.assertTrue(self.report["eye_inserts"])

    def test_gives_the_hair_spring_bones(self):
        self.assertGreaterEqual(self.report["hair_chains"], 4)
        self.assertGreaterEqual(len(self.gltf["extensions"]["VRMC_springBone"]["springs"]), 4)

    def test_binds_blink_and_mouth_expressions(self):
        preset = self.gltf["extensions"]["VRMC_vrm"]["expressions"]["preset"]
        for expression in ("blink", "blinkLeft", "blinkRight", "aa"):
            with self.subTest(expression=expression):
                self.assertTrue(preset[expression].get("morphTargetBinds"))


if __name__ == "__main__":
    unittest.main()


class EyePairAtTheMarks(unittest.TestCase):
    """A tester's head, as measured: inner-ear pieces bigger than the eyes and 11 cm above them."""

    CENTRE = {1: np.array([0.09, -0.141, 1.51]), 2: np.array([-0.09, -0.141, 1.51]),      # inner ears
              3: np.array([0.04, -0.164, 1.394]), 4: np.array([-0.04, -0.164, 1.394])}    # eyes
    SIZE = {1: np.array([0.049, 0.034, 0.066]), 2: np.array([0.049, 0.034, 0.066]),
            3: np.array([0.026, 0.009, 0.031]), 4: np.array([0.026, 0.009, 0.031])}

    def pair(self, **marks):
        low = {p: self.CENTRE[p] - self.SIZE[p] / 2 for p in self.CENTRE}
        high = {p: self.CENTRE[p] + self.SIZE[p] / 2 for p in self.CENTRE}
        return parts._eye_pair(list(self.CENTRE), low, high, self.CENTRE, 0.0, 1.3, **marks)

    def test_by_size_alone_the_ears_win(self):
        self.assertEqual(self.pair(), {"left": 1, "right": 2})

    def test_marked_eyes_pick_the_pair_under_the_marks(self):
        marked = [(-0.04, 1.40), (0.04, 1.40)]  # either order: the left eye is the one at +X
        self.assertEqual(self.pair(marked=marked, reach=0.3 * 0.25), {"left": 3, "right": 4})

    def test_marks_far_from_every_pair_change_nothing(self):
        self.assertEqual(self.pair(marked=[(0.04, 1.75), (-0.04, 1.75)], reach=0.3 * 0.25), {"left": 1, "right": 2})


class FaceTrackingOnAPaintedFace(unittest.TestCase):
    """One surface with the eyes and mouth painted on: no eye inserts, nothing fixed, no mouth cover
    (a tester's generated model). The step used to die reading them; the blink is still built."""

    @classmethod
    def setUpClass(cls):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import add_face_tracking
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=40, y_subdivisions=40, size=0.2, rotation=(math.pi / 2, 0, 0))
        cls.obj = bpy.context.object
        bpy.ops.object.transform_apply(rotation=True)
        cls.obj.shape_key_add(name="Basis").value = 0.0
        cls.obj[face_rig.PAINTED_EYES_PROPERTY] = {"left": [0.02, 0.0, 0.06, 0.03], "right": [-0.06, 0.0, -0.02, 0.03]}
        cls.shapes = add_face_tracking.FaceShapes(cls.obj)
        cls.shapes.add_blink("", face_rig.PAINTED_BLINK)

    def moved(self, name):
        keys = self.obj.data.shape_keys.key_blocks
        co, basis = np.empty(len(self.obj.data.vertices) * 3), np.empty(len(self.obj.data.vertices) * 3)
        keys[name].data.foreach_get("co", co)
        keys["Basis"].data.foreach_get("co", basis)
        basis = basis.reshape(-1, 3)
        return basis, np.abs(co.reshape(-1, 3) - basis).max(axis=1) > 1e-6

    def test_it_has_no_mouth_to_shape(self):
        self.assertIsNone(self.shapes.mouth)
        self.assertTrue(self.shapes.painted)

    def test_each_blink_squeezes_its_own_eye(self):
        for name, sign in (("eyeBlinkLeft", 1), ("eyeBlinkRight", -1)):
            with self.subTest(name=name):
                basis, moved = self.moved(name)
                self.assertGreater(moved.sum(), 10)
                self.assertTrue((sign * basis[moved, 0] > 0).all())


class APieceWithNoBody(unittest.TestCase):
    """A head piece with a twin-tail hanging past the hips holds no body; one with a trunk or legs does."""

    JOINTS = {"hips": np.array([0.0, 0.0, 0.9]), "neck": np.array([0.0, 0.0, 1.3]),
              "left_upper_leg": np.array([0.1, 0.0, 0.85]), "left_lower_leg": np.array([0.1, 0.0, 0.5]),
              "left_foot": np.array([0.1, 0.0, 0.1]), "right_upper_leg": np.array([-0.1, 0.0, 0.85]),
              "right_lower_leg": np.array([-0.1, 0.0, 0.5]), "right_foot": np.array([-0.1, 0.0, 0.1])}

    @classmethod
    def setUpClass(cls):
        z = np.linspace(0.0, 1.6, 161)
        head = np.stack([np.zeros(20), np.zeros(20), np.linspace(1.3, 1.6, 20)], axis=1)
        cls.tail = np.stack([np.full(60, -0.3), np.full(60, 0.1), np.linspace(0.7, 1.4, 60)], axis=1)  # the catgirl's, as measured
        cls.trunk = np.stack([np.full(40, 0.05), np.zeros(40), np.linspace(0.9, 1.3, 40)], axis=1)
        cls.leg = np.stack([np.full(40, 0.12), np.zeros(40), np.linspace(0.1, 0.85, 40)], axis=1)
        cls.verts = np.concatenate([head, cls.tail, cls.trunk, cls.leg, np.stack([np.zeros(len(z)), np.ones(len(z)), z], axis=1)])
        cls.head, cls.n = len(head), len(cls.verts)

    def piece(self, *spans):
        mask = np.zeros(self.n, dtype=bool)
        for start, count in spans:
            mask[start:start + count] = True
        return mask

    def test_a_head_and_its_tail_hold_no_body(self):
        self.assertFalse(parts._holds_body(self.verts, self.piece((0, self.head + 60)), self.JOINTS))

    def test_a_trunk_is_a_body(self):
        self.assertTrue(parts._holds_body(self.verts, self.piece((0, self.head), (self.head + 60, 40)), self.JOINTS))

    def test_legs_are_a_body(self):
        self.assertTrue(parts._holds_body(self.verts, self.piece((0, self.head), (self.head + 100, 40)), self.JOINTS))

    def test_without_named_legs_anything_below_the_hips_is(self):
        joints = {"hips": self.JOINTS["hips"], "neck": self.JOINTS["neck"]}
        self.assertTrue(parts._holds_body(self.verts, self.piece((0, self.head + 60)), joints))
