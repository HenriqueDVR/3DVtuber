"""Building a VRM humanoid armature and skinning the fused mesh to it."""

import math
import unittest

import bpy
import numpy as np

from support import load_fixture
from vtuber_maker.pipeline import landmarks, meshdata, rig

REQUIRED_VRM_BONES = (
    "hips",
    "spine",
    "head",
    "left_upper_leg",
    "left_lower_leg",
    "left_foot",
    "right_upper_leg",
    "right_lower_leg",
    "right_foot",
    "left_upper_arm",
    "left_lower_arm",
    "left_hand",
    "right_upper_arm",
    "right_lower_arm",
    "right_hand",
)


def posed_world_vertices(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    coords = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", coords)
    evaluated.to_mesh_clear()
    matrix = np.array(obj.matrix_world)
    return coords.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]


class SkinnedSyntheticCharacter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh, cls.truth = load_fixture()
        joints = landmarks.detect_body(meshdata.world_vertices(cls.mesh), meshdata.triangles(cls.mesh.data))
        cls.armature = rig.build_armature(joints)
        rig.skin(cls.mesh, cls.armature)
        cls.rest = meshdata.world_vertices(cls.mesh)

    def tearDown(self):
        for bone in self.armature.pose.bones:
            bone.rotation_mode = "XYZ"
            bone.rotation_euler = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()

    def near(self, point, radius=0.03):
        return np.linalg.norm(self.rest - np.asarray(point), axis=1) < radius

    def displacement_after_bending(self, bone_key, degrees):
        bone = self.armature.pose.bones[rig.BONE_NAMES[bone_key]]
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = (math.radians(degrees), 0.0, 0.0)
        bpy.context.view_layer.update()
        return np.linalg.norm(posed_world_vertices(self.mesh) - self.rest, axis=1)

    def test_has_every_required_vrm_bone(self):
        names = set(self.armature.data.bones.keys())
        for key in REQUIRED_VRM_BONES:
            with self.subTest(bone=key):
                self.assertIn(rig.BONE_NAMES[key], names)

    def test_hand_chain_hangs_from_the_spine(self):
        chain = []
        bone = self.armature.data.bones[rig.BONE_NAMES["left_hand"]]
        while bone:
            chain.append(bone.name)
            bone = bone.parent
        self.assertEqual(chain[-1], rig.BONE_NAMES["hips"])
        self.assertIn(rig.BONE_NAMES["left_upper_arm"], chain)
        self.assertIn(rig.BONE_NAMES["spine"], chain)

    def test_every_vertex_weights_sum_to_one(self):
        totals = np.array([sum(g.weight for g in v.groups) for v in self.mesh.data.vertices])
        bad = int(np.sum(~np.isclose(totals, 1.0, atol=0.02)))
        self.assertEqual(bad, 0, f"{bad} of {len(totals)} vertices have weights not summing to 1")

    def test_bending_left_elbow_moves_only_the_left_hand(self):
        moved = self.displacement_after_bending("left_lower_arm", 60)
        self.assertGreater(moved[self.near(self.truth["joints"]["left_hand"])].mean(), 0.1)
        self.assertLess(moved[self.near(self.truth["joints"]["right_hand"])].max(), 0.002)
        self.assertLess(moved[self.near(self.truth["face"]["left_eye"])].max(), 0.002)

    def test_nodding_moves_the_face_but_not_the_hands(self):
        moved = self.displacement_after_bending("head", 30)
        self.assertGreater(moved[self.near(self.truth["face"]["left_eye"])].mean(), 0.02)
        self.assertLess(moved[self.near(self.truth["joints"]["left_hand"])].max(), 0.002)


class FillUnweighted(unittest.TestCase):
    def setUp(self):
        verts = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),  # weighted to Neck at y=0, Head at y=1
            (0, 2, 0), (1, 2, 0),  # unweighted, joined to the Head row by edges
            (0, 2, 0), (1, 2, 0), (0, 3, 0), (1, 3, 0),  # unweighted separate piece sharing that row's spot
            (5, 5, 5), (6, 5, 5), (5, 6, 5),  # unweighted piece touching nothing
        ]
        faces = [(0, 1, 3, 2), (2, 3, 5, 4), (6, 7, 9, 8), (10, 11, 12)]
        mesh = bpy.data.meshes.new("fill_unweighted")
        mesh.from_pydata(verts, [], faces)
        self.obj = bpy.data.objects.new("fill_unweighted", mesh)
        bpy.context.scene.collection.objects.link(self.obj)
        self.obj.vertex_groups.new(name="Neck").add([0, 1], 1.0, "REPLACE")
        self.obj.vertex_groups.new(name="Head").add([2, 3], 1.0, "REPLACE")

    def tearDown(self):
        mesh = self.obj.data
        bpy.data.objects.remove(self.obj)
        bpy.data.meshes.remove(mesh)

    def weights(self, index):
        return {self.obj.vertex_groups[g.group].name: round(g.weight, 3) for g in self.obj.data.vertices[index].groups}

    def test_every_vertex_ends_up_weighted(self):
        self.assertEqual(rig.fill_unweighted(self.obj), 9)
        for v in self.obj.data.vertices:
            self.assertAlmostEqual(sum(g.weight for g in v.groups), 1.0, places=3)

    def test_takes_weights_from_the_surface_around_it(self):
        rig.fill_unweighted(self.obj)
        self.assertEqual(self.weights(4), {"Head": 1.0})

    def test_separate_piece_follows_what_it_touches(self):
        rig.fill_unweighted(self.obj)
        self.assertEqual(self.weights(8), {"Head": 1.0})

    def test_isolated_piece_takes_the_nearest_weights(self):
        rig.fill_unweighted(self.obj)
        self.assertEqual(self.weights(10), {"Head": 1.0})


if __name__ == "__main__":
    unittest.main()
