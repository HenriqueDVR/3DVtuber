"""Bone chains for hair that hangs below the head, ready for spring bone physics."""

import math
import unittest

import bpy
import numpy as np

from support import load_fixture
from vtuber_maker.pipeline import face, hair, landmarks, meshdata, rig


def posed_world_vertices(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    coords = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", coords)
    evaluated.to_mesh_clear()
    matrix = np.array(obj.matrix_world)
    return coords.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]


class HairBonesOnSyntheticCharacter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh, cls.truth = load_fixture()
        joints = landmarks.detect_body(meshdata.world_vertices(cls.mesh), meshdata.triangles(cls.mesh.data))
        features = face.detect_features(cls.mesh, joints)
        cls.armature = rig.build_armature(joints)
        rig.skin(cls.mesh, cls.armature)
        cls.chains = hair.build_hair_bones(cls.mesh, cls.armature, joints, features)
        cls.rest = meshdata.world_vertices(cls.mesh)

    def tearDown(self):
        for bone in self.armature.pose.bones:
            bone.rotation_mode = "XYZ"
            bone.rotation_euler = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()

    def near(self, point, radius=0.03):
        return np.linalg.norm(self.rest - np.asarray(point), axis=1) < radius

    def chains_rooted(self, where):
        return [chain for chain in self.chains if where(np.array(self.armature.data.bones[chain[0]].head_local))]

    def displacement_after_swinging(self, chains, degrees):
        for chain in chains:
            bone = self.armature.pose.bones[chain[0]]
            bone.rotation_mode = "XYZ"
            bone.rotation_euler = (math.radians(degrees), 0.0, 0.0)
        bpy.context.view_layer.update()
        return np.linalg.norm(posed_world_vertices(self.mesh) - self.rest, axis=1)

    def test_splits_hanging_hair_into_several_chains(self):
        self.assertGreaterEqual(len(self.chains), 4)

    def test_every_chain_hangs_down_from_the_head(self):
        for chain in self.chains:
            with self.subTest(root=chain[0]):
                bones = [self.armature.data.bones[name] for name in chain]
                self.assertEqual(bones[0].parent.name, rig.BONE_NAMES["head"])
                for parent, child in zip(bones, bones[1:]):
                    self.assertEqual(child.parent.name, parent.name)
                self.assertGreater(bones[0].head_local.z, bones[-1].tail_local.z)

    def test_swinging_back_hair_moves_its_tips_but_not_the_face(self):
        back = self.chains_rooted(lambda root: root[1] > 0.02)
        moved = self.displacement_after_swinging(back, 30)
        self.assertGreater(moved[self.near(self.truth["hair"]["back_tip"])].mean(), 0.03)
        self.assertLess(moved[self.near(self.truth["face"]["left_eye"])].max(), 1e-4)

    def test_swinging_left_lock_leaves_right_lock_alone(self):
        left = self.chains_rooted(lambda root: root[0] > 0.06 and root[1] < -0.02)
        roots = {chain[0]: np.round(self.armature.data.bones[chain[0]].head_local, 3).tolist() for chain in self.chains}
        self.assertTrue(left, f"no chain found for the left side lock; chain roots: {roots}")
        moved = self.displacement_after_swinging(left, 30)
        self.assertGreater(moved[self.near(self.truth["hair"]["left_lock_tip"], 0.02)].mean(), 0.02)
        self.assertLess(moved[self.near(self.truth["hair"]["right_lock_tip"], 0.02)].max(), 1e-4)

    def test_hanging_hair_no_longer_follows_the_chest(self):
        torso = {
            self.mesh.vertex_groups[rig.BONE_NAMES[key]].index
            for key in ("spine", "chest", "upper_chest")
            if rig.BONE_NAMES[key] in self.mesh.vertex_groups
        }
        tip = np.nonzero(self.near(self.truth["hair"]["back_tip"]))[0]
        worst = max(
            (g.weight for i in tip for g in self.mesh.data.vertices[i].groups if g.group in torso), default=0.0
        )
        self.assertLess(worst, 0.01)


if __name__ == "__main__":
    unittest.main()
