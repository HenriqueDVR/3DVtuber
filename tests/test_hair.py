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


class ATuftBesideTheHead(unittest.TestCase):
    """The plain boy's spiky side tufts, within the front's angle: by angle alone they hung from the front's
    line on chains and read stretched in Warudo. A short bang over the face still swings."""

    def chains(self, y):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        verts, faces = _sheet(0.04, 0.06, 1.47, 1.6, y, 7, 27, 0)
        data = bpy.data.meshes.new("hair")
        data.from_pydata(verts.tolist(), [], faces.tolist())
        mesh = bpy.data.objects.new("hair", data)
        bpy.context.scene.collection.objects.link(mesh)
        joints = {key: np.array(p, float) for key, p in (("hips", (0, 0, 0.9)), ("head", (0, 0, 1.4)))}
        armature_data = bpy.data.armatures.new("Armature")
        armature = bpy.data.objects.new("Armature", armature_data)
        bpy.context.scene.collection.objects.link(armature)
        rig.make_active(armature)
        bpy.ops.object.mode_set(mode="EDIT")
        for key in ("hips", "head"):
            bone = armature_data.edit_bones.new(rig.BONE_NAMES[key])
            bone.head, bone.tail = joints[key].tolist(), (joints[key] + [0, 0, 0.1]).tolist()
        bpy.ops.object.mode_set(mode="OBJECT")
        mesh.vertex_groups.new(name=rig.BONE_NAMES["head"]).add(list(range(len(verts))), 1.0, "REPLACE")
        # head joint 1.4, top 1.6: radius 0.1; the eyes 1 cm in front of y -0.09, the centre at y 0
        features = {"left_eye": np.array([0.03, -0.1, 1.5]), "right_eye": np.array([-0.03, -0.1, 1.5])}
        return hair.build_hair_bones(mesh, armature, joints, features, np.ones(len(verts), dtype=bool))

    def test_a_tuft_level_with_the_middle_of_the_head_stays(self):
        self.assertEqual(self.chains(-0.03), [])

    def test_a_short_one_rooted_behind_the_eyes_stays(self):
        # 6.5 cm from the front's line, 0.65 head radii, 3 cm behind the eyes: the boy's, his face shallow
        self.assertEqual(self.chains(-0.07), [])

    def test_a_short_bang_level_with_the_eyes_swings(self):
        # the catgirl's bangs: 0.7 head radii, level with her eyes
        self.assertEqual(len(self.chains(-0.1)), 1)


class BangsJoinedToLongLocks(unittest.TestCase):
    """The miko's fringe is one surface with the long locks hanging past her chin either side of her face:
    grouped with them by angle it lay in their roots' hand-over, 60% on the head, and never swung."""

    WINDOW = True

    @classmethod
    def setUpClass(cls):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        verts, faces = _sheet(-0.08, 0.08, 1.0, 1.6, -0.12, 17, 31, 0)
        middle = verts[faces].mean(axis=1)
        if cls.WINDOW:
            faces = faces[~((np.abs(middle[:, 0]) < 0.045) & (middle[:, 2] < 1.44))]  # the face's window
        used = np.unique(faces)
        verts, faces = verts[used], np.searchsorted(used, faces)
        data = bpy.data.meshes.new("hair")
        data.from_pydata(verts.tolist(), [], faces.tolist())
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
        features = {"left_eye": np.array([0.03, -0.1, 1.5]), "right_eye": np.array([-0.03, -0.1, 1.5]),
                    "eye_size": np.array([0.02, 0.01])}
        cls.chains = hair.build_hair_bones(cls.mesh, cls.armature, joints, features, np.ones(len(verts), dtype=bool))
        cls.verts = verts

    def test_the_fringe_swings_on_a_short_chain_of_its_own(self):
        bones = self.armature.data.bones
        short = [c for c in self.chains if min(bones[n].tail_local.z for n in c) > 1.4]
        self.assertTrue(short, "no chain ends above the jaw")
        fringe = np.flatnonzero((np.abs(self.verts[:, 0]) < 0.03) & (self.verts[:, 2] < 1.5))
        names = {g.index: g.name for g in self.mesh.vertex_groups}
        own = {n for c in short for n in c}
        on_bangs = [sum(g.weight for g in self.mesh.data.vertices[int(i)].groups if names[g.group] in own) for i in fringe]
        self.assertGreater(float(np.mean(on_bangs)), 0.5)

    def test_the_cut_between_fringe_and_locks_does_not_open(self):
        # weighted apart, the cut hung each side from its own chain with no bone between: NerdGirl's fringe went to
        # shards over her face as her head swung (330 such edges, 29 before the bangs rule)
        names = {g.index: g.name for g in self.mesh.vertex_groups}
        chain_of = {n: chain[0] for chain in self.chains for n in chain}

        def chains(vertex):
            return {chain_of[names[g.group]] for g in self.mesh.data.vertices[vertex].groups
                    if g.weight > 1e-4 and names[g.group] in chain_of}
        apart = [e for e in self.mesh.data.edges
                 if chains(e.vertices[0]) and chains(e.vertices[1]) and not chains(e.vertices[0]) & chains(e.vertices[1])]
        self.assertEqual(len(apart), 0)



class ALockDownAcrossTheFace(BangsJoinedToLongLocks):
    """NerdGirl's side-swept fringe goes on past her chin over the face: cut at the jaw into bangs on a short chain
    and a lock on one rooted at her jaw, it folded to shards between them as her head swung."""

    WINDOW = False

    def test_the_fringe_swings_on_a_short_chain_of_its_own(self):
        bones = self.armature.data.bones
        self.assertEqual([c for c in self.chains if min(bones[n].tail_local.z for n in c) > 1.4], [], "a chain ends above the jaw")


if __name__ == "__main__":
    unittest.main()
