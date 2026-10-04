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


class RidersBesideTheFace(unittest.TestCase):
    """A fringe on a spring hangs in front of the face; an eye modelled as its own piece lies on the face."""

    def build(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)

        def grid(xs, zs, y):
            start = len(verts)
            verts.extend((x, y, z) for z in zs for x in xs)
            faces.extend((start + j * len(xs) + i, start + j * len(xs) + i + 1, start + (j + 1) * len(xs) + i + 1,
                          start + (j + 1) * len(xs) + i) for j in range(len(zs) - 1) for i in range(len(xs) - 1))
            return range(start, len(verts))

        verts, faces = [], []
        fringe = grid(np.arange(-0.2, 0.201, 0.02), np.arange(0.0, 1.601, 0.02), -0.03)
        face = grid(np.arange(-0.15, 0.151, 0.02), np.arange(1.2, 1.501, 0.02), 0.0)
        eye = grid([0.04, 0.06], [1.34, 1.36], -0.003)
        ribbon = grid([-0.06, -0.04], [1.34, 1.36], -0.04)
        data = bpy.data.meshes.new("riders")
        data.from_pydata(verts, [], faces)
        obj = bpy.data.objects.new("riders", data)
        bpy.context.scene.collection.objects.link(obj)
        obj.vertex_groups.new(name="Hair0_0").add(list(fringe), 1.0, "REPLACE")
        obj.vertex_groups.new(name="Head").add(list(face) + list(eye) + list(ribbon), 1.0, "REPLACE")
        mask = lambda members: np.isin(np.arange(len(verts)), list(members))  # noqa: E731
        return obj, mask(fringe), mask(face), eye, ribbon

    def bones(self, obj, members):
        names = {g.index: g.name for g in obj.vertex_groups}
        return {names[g.group] for i in members for g in obj.data.vertices[i].groups if g.weight > 0}

    def test_a_piece_on_the_face_stays_and_one_on_the_fringe_rides(self):
        obj, fringe, face, eye, ribbon = self.build()
        moved = cloth.attach_riders(obj, fringe, np.zeros(len(fringe), bool), anchors=face)
        self.assertEqual(moved, len(ribbon))
        self.assertEqual(self.bones(obj, eye), {"Head"})
        self.assertEqual(self.bones(obj, ribbon), {"Hair0_0"})

    def test_without_the_face_named_the_eye_rides_the_fringe(self):
        # what model A's T-pose FBX did: her eyes swung out of her face on the fringe's chains
        obj, fringe, _face, eye, _ribbon = self.build()
        cloth.attach_riders(obj, fringe, np.zeros(len(fringe), bool))
        self.assertEqual(self.bones(obj, eye), {"Hair0_0"})


if __name__ == "__main__":
    unittest.main()
