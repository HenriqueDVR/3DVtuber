"""Characters built from separate parts, like Tripo's P2 meshes: hair, eye inserts and a mouth insert."""

import math
import unittest

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from support import FIXTURE_OUT, load_fixture
from test_build_avatar import read_glb_json
from vtuber_maker.pipeline import build, face_rig, landmarks, meshdata, parts, rig, vrm_export

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
