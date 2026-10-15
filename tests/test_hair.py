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

    def chains_moving(self, point, radius=0.02):
        """Names of the hair chains that move the vertices near a point."""
        chain_of = {name: chain[0].rsplit("_", 1)[0] for chain in self.chains for name in chain}
        found = set()
        for index in np.nonzero(self.near(point, radius))[0]:
            for group in self.mesh.data.vertices[index].groups:
                name = self.mesh.vertex_groups[group.group].name
                if name in chain_of and group.weight > 0.01:
                    found.add(chain_of[name])
        return found

    def test_each_side_lock_hangs_on_a_single_chain(self):
        # A strand blended across two chains gets averaged into the body when those chains
        # swing to opposite sides of the shoulder.
        for tip in ("left_lock_tip", "right_lock_tip"):
            with self.subTest(tip=tip):
                self.assertEqual(len(self.chains_moving(self.truth["hair"][tip])), 1)

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


def _sheet(x0, x1, z0, z1, y, nx, nz, offset):
    """A flat grid of quads in the x-z plane at depth y; vertex indices start at `offset`."""
    xs, zs = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(z0, z1, nz), indexing="ij")
    verts = np.stack([xs.ravel(), np.full(xs.size, y), zs.ravel()], axis=1)
    index = np.arange(nx * nz).reshape(nx, nz) + offset
    faces = np.stack([index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(),
                      index[:-1, 1:].ravel()], axis=1)
    return verts, faces


class StrayHairBits(unittest.TestCase):
    """A strand, a bit too small for a chain lying on it, and one lying well away from it."""

    @classmethod
    def setUpClass(cls):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        strand, strand_faces = _sheet(0.10, 0.12, 1.0, 1.6, 0.0, 3, 25, 0)
        near, near_faces = _sheet(0.10, 0.11, 1.19, 1.21, -0.005, 3, 3, len(strand))
        far, far_faces = _sheet(0.25, 0.26, 1.19, 1.21, 0.0, 3, 3, len(strand) + len(near))
        verts = np.vstack([strand, near, far])
        data = bpy.data.meshes.new("hair")
        data.from_pydata(verts.tolist(), [], np.vstack([strand_faces, near_faces, far_faces]).tolist())
        cls.mesh = bpy.data.objects.new("hair", data)
        bpy.context.scene.collection.objects.link(cls.mesh)
        joints = {key: np.array(p, float) for key, p in (("hips", (0, 0, 0.9)), ("head", (0, 0, 1.4)))}
        armature_data = bpy.data.armatures.new("Armature")
        cls.armature = bpy.data.objects.new("Armature", armature_data)
        bpy.context.scene.collection.objects.link(cls.armature)
        rig.make_active(cls.armature)
        bpy.ops.object.mode_set(mode="EDIT")
        for key in ("hips", "head"):
            bone = armature_data.edit_bones.new(rig.BONE_NAMES[key])
            bone.head, bone.tail = joints[key].tolist(), (joints[key] + [0, 0, 0.1]).tolist()
        bpy.ops.object.mode_set(mode="OBJECT")
        body = cls.mesh.vertex_groups.new(name=rig.BONE_NAMES["hips"])
        body.add(list(range(len(verts))), 1.0, "REPLACE")
        cls.mesh.vertex_groups.new(name=rig.BONE_NAMES["head"])
        features ={"left_eye": np.array([0.03, -0.1, 1.5]), "right_eye": np.array([-0.03, -0.1, 1.5])}
        cls.chains = hair.build_hair_bones(cls.mesh, cls.armature, joints, features, np.ones(len(verts), dtype=bool))
        cls.near = range(len(strand), len(strand) + len(near))
        cls.far = range(len(strand) + len(near), len(verts))

    def groups_of(self, index):
        names = {g.index: g.name for g in self.mesh.vertex_groups}
        return {names[g.group]: g.weight for g in self.mesh.data.vertices[index].groups if g.weight > 0}

    def test_only_the_strand_gets_a_chain(self):
        self.assertEqual(len(self.chains), 1)

    def test_a_bit_lying_on_the_strand_moves_with_it(self):
        for index in self.near:
            weights = self.groups_of(index)
            self.assertNotIn(rig.BONE_NAMES["hips"], weights)
            self.assertTrue(all(name.startswith("Hair") or name == rig.BONE_NAMES["head"] for name in weights))

    def test_a_bit_away_from_the_strand_keeps_its_weights(self):
        for index in self.far:
            self.assertEqual(self.groups_of(index), {rig.BONE_NAMES["hips"]: 1.0})


class BangsOverTheForehead(unittest.TestCase):
    """A fringe ending at the brows swings; the same strand behind the head stays on it."""

    @classmethod
    def setUpClass(cls):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        # head joint 1.4, top 1.6: radius 0.1, centre 1.5; in front the hair hangs from 1.535, behind from 1.465
        front, front_faces = _sheet(-0.02, 0.02, 1.44, 1.6, -0.12, 3, 17, 0)
        back, back_faces = _sheet(-0.02, 0.02, 1.44, 1.6, 0.12, 3, 17, len(front))
        verts = np.vstack([front, back])
        data = bpy.data.meshes.new("hair")
        data.from_pydata(verts.tolist(), [], np.vstack([front_faces, back_faces]).tolist())
        cls.mesh = bpy.data.objects.new("hair", data)
        bpy.context.scene.collection.objects.link(cls.mesh)
        joints = {key: np.array(p, float) for key, p in (("hips", (0, 0, 0.9)), ("head", (0, 0, 1.4)))}
        armature_data = bpy.data.armatures.new("Armature")
        cls.armature = bpy.data.objects.new("Armature", armature_data)
        bpy.context.scene.collection.objects.link(cls.armature)
        rig.make_active(cls.armature)
        bpy.ops.object.mode_set(mode="EDIT")
        for key in ("hips", "head"):
            bone = armature_data.edit_bones.new(rig.BONE_NAMES[key])
            bone.head, bone.tail = joints[key].tolist(), (joints[key] + [0, 0, 0.1]).tolist()
        bpy.ops.object.mode_set(mode="OBJECT")
        cls.mesh.vertex_groups.new(name=rig.BONE_NAMES["head"]).add(list(range(len(verts))), 1.0, "REPLACE")
        features = {"left_eye": np.array([0.03, -0.1, 1.5]), "right_eye": np.array([-0.03, -0.1, 1.5])}
        cls.chains = hair.build_hair_bones(cls.mesh, cls.armature, joints, features, np.ones(len(verts), dtype=bool))

    def test_the_fringe_alone_gets_a_chain_from_above_the_brows(self):
        self.assertEqual(len(self.chains), 1)
        root = self.armature.data.bones[self.chains[0][0]].head_local
        self.assertLess(root.y, 0)
        self.assertAlmostEqual(root.z, 1.5 + hair.FRONT_HANGS_ABOVE * 0.1, delta=0.011)

    def test_it_is_marked_to_spring_stiffer(self):
        self.assertTrue(self.armature.data.bones[self.chains[0][0]].get(hair.FRONT_HUNG))


if __name__ == "__main__":
    unittest.main()
