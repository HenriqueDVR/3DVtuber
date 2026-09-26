"""Wings, tails and big ears (manifest role "appendage"): on the core bone they grow from, swinging on a chain."""

import math
import unittest

import bpy
import numpy as np

from support import load_fixture
from vtuber_maker.pipeline import cloth, landmarks, meshdata, rig


class AppendageOnSyntheticCharacter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh, _ = load_fixture("synthetic_character_parts")
        verts = meshdata.world_vertices(cls.mesh)
        cls.joints = landmarks.detect_body(verts, meshdata.triangles(cls.mesh.data))
        cls.armature = rig.build_armature(cls.joints)
        rig.skin(cls.mesh, cls.armature)
        verts = meshdata.world_vertices(cls.mesh)
        labels = meshdata.pieces(cls.mesh, verts)
        # stands in for a tail: the biggest separate piece hanging behind the body, not the body itself
        body = np.bincount(labels).argmax()
        behind = [l for l in np.unique(labels) if l != body and verts[labels == l, 1].mean() > verts[:, 1].mean()]
        cls.label = max(behind, key=lambda l: (labels == l).sum())
        cls.members = labels == cls.label
        cls.names = cloth.build_appendage_chain(cls.mesh, cls.armature, cls.joints, cls.members, 0)

    def weights(self, index):
        groups = self.mesh.vertex_groups
        return {groups[g.group].name: g.weight for g in self.mesh.data.vertices[int(index)].groups}

    def test_a_chain_hangs_from_a_core_bone(self):
        self.assertTrue(self.names)
        parent = self.armature.data.bones[self.names[0][0]].parent.name
        self.assertIn(parent, [rig.BONE_NAMES[k] for k in cloth.CORE_BONES])

    def test_no_limb_moves_it(self):
        limbs = {rig.BONE_NAMES[f"{s}_{k}"] for s in ("left", "right")
                 for k in ("upper_arm", "lower_arm", "hand", "upper_leg", "lower_leg", "foot")}
        on_limbs = [i for i in np.flatnonzero(self.members) if limbs & self.weights(i).keys()]
        self.assertEqual(on_limbs, [])

    def test_its_far_end_rides_the_chain(self):
        chain = {name for names in self.names for name in names}
        far = [i for i in np.flatnonzero(self.members) if sum(w for n, w in self.weights(i).items() if n in chain) > 0.99]
        self.assertGreater(len(far), 0.5 * self.members.sum())

    def test_raising_an_arm_leaves_it_alone(self):
        bone = self.armature.pose.bones[rig.BONE_NAMES["left_upper_arm"]]
        bone.rotation_mode = "XYZ"
        before = meshdata.world_vertices(self.mesh)
        bone.rotation_euler = (0.0, math.radians(60), 0.0)
        bpy.context.view_layer.update()
        evaluated = self.mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
        posed = np.array([v.co for v in evaluated.to_mesh().vertices]) @ np.array(self.mesh.matrix_world)[:3, :3].T
        evaluated.to_mesh_clear()
        bone.rotation_euler = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        moved = np.linalg.norm(posed[self.members] + np.array(self.mesh.matrix_world)[:3, 3] - before[self.members], axis=1)
        self.assertLess(moved.max(), 1e-4)


if __name__ == "__main__":
    unittest.main()
