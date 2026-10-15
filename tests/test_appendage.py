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


class AWingOnTwoChains(unittest.TestCase):
    """The demon girl's wings: one surface forking into a top edge and a bottom edge, a chain along each.
    With each vertex wholly on its own chain the membrane sheared along the line between them and tore."""

    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture("synthetic_character_parts")  # for a whole skeleton
        verts = meshdata.world_vertices(mesh)
        cls.joints = landmarks.detect_body(verts, meshdata.triangles(mesh.data))
        cls.armature = rig.build_armature(cls.joints)
        bone = cls.armature.data.bones[rig.BONE_NAMES["chest"]]
        chest = np.array(cls.armature.matrix_world @ bone.head_local)
        # a fan behind the chest, from its root out to the left, 100 degrees wide
        rings, spokes = 12, 21
        points = [chest + [0.0, 0.15, 0.0]]
        for r in range(1, rings + 1):
            for k in range(spokes):
                a = math.radians(-50 + 100 * k / (spokes - 1))
                reach = 0.3 + 0.15 * abs(math.degrees(a)) / 50  # the two edges reach furthest, as a wing's do
                points.append(chest + [reach * r / rings * math.cos(a), 0.15, reach * r / rings * math.sin(a)])
        faces = [(0, 1 + k, 2 + k) for k in range(spokes - 1)]
        for r in range(rings - 1):
            for k in range(spokes - 1):
                i = 1 + r * spokes + k
                faces.append((i, i + 1, i + spokes + 1, i + spokes))
        data = bpy.data.meshes.new("wing")
        data.from_pydata([list(p) for p in points], [], faces)
        cls.mesh = bpy.data.objects.new("wing", data)
        bpy.context.scene.collection.objects.link(cls.mesh)
        cls.points = np.array(points)
        cls.names = cloth.build_appendage_chain(cls.mesh, cls.armature, cls.joints, np.ones(len(points), dtype=bool), 0)

    def share(self, index, chain):
        groups = self.mesh.vertex_groups
        return sum(g.weight for g in self.mesh.data.vertices[int(index)].groups if groups[g.group].name in chain)

    def test_it_forks_into_two_chains(self):
        self.assertEqual(len(self.names), 2)

    def test_the_middle_of_the_membrane_hangs_on_both(self):
        rim = len(self.points) - 21  # the outer ring's first vertex
        middle = rim + 10
        self.assertGreater(min(self.share(middle, set(c)) for c in self.names), 0.2)

    def test_its_edges_hang_on_their_own_chain(self):
        rim = len(self.points) - 21
        for edge in (rim, rim + 20):
            self.assertGreater(max(self.share(edge, set(c)) for c in self.names), 0.95)


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

    def test_without_the_face_named_the_eye_stays_on_what_it_lies_on(self):
        # model A's T-pose FBX: her eyes swung out of her face on the fringe's chains. Until 2026-10-08
        # only naming the face kept them (this test pinned the eye riding without it); now a piece
        # rides only when what lies under it swings
        obj, fringe, _face, eye, ribbon = self.build()
        cloth.attach_riders(obj, fringe, np.zeros(len(fringe), bool))
        self.assertEqual(self.bones(obj, eye), {"Head"})
        self.assertEqual(self.bones(obj, ribbon), {"Hair0_0"})


class RidersUnderALock(unittest.TestCase):
    """A lock on a spring hangs down over the chest; shirt panels and trims are pieces of their own under it."""

    def build(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)

        def grid(xs, zs, y):
            start = len(verts)
            verts.extend((x, y, z) for z in zs for x in xs)
            faces.extend((start + j * len(xs) + i, start + j * len(xs) + i + 1, start + (j + 1) * len(xs) + i + 1,
                          start + (j + 1) * len(xs) + i) for j in range(len(zs) - 1) for i in range(len(xs) - 1))
            return list(range(start, len(verts)))

        verts, faces = [], []
        chest = grid(np.arange(-0.2, 0.201, 0.02), np.arange(1.0, 1.301, 0.02), 0.0)
        lock = grid(np.arange(-0.08, -0.039, 0.01), np.arange(1.0, 1.601, 0.02), -0.025)
        panel = grid(np.arange(-0.1, 0.001, 0.02), np.arange(1.1, 1.201, 0.02), -0.004)
        trim = grid([-0.07, -0.06], [1.14, 1.15], -0.008)        # on the panel, under the lock
        ribbon = grid([-0.07, -0.05], [1.5, 1.52], -0.03)        # on the lock, above the chest
        button = grid([-0.065, -0.055], [1.505, 1.515], -0.033)  # on the ribbon
        data = bpy.data.meshes.new("lock")
        data.from_pydata(verts, [], faces)
        obj = bpy.data.objects.new("lock", data)
        bpy.context.scene.collection.objects.link(obj)
        obj.vertex_groups.new(name="Hair0_0").add(lock, 1.0, "REPLACE")
        obj.vertex_groups.new(name="Chest").add(chest + panel + trim, 1.0, "REPLACE")
        obj.vertex_groups.new(name="Head").add(ribbon + button, 1.0, "REPLACE")
        return obj, np.isin(np.arange(len(verts)), lock), panel, trim, ribbon, button

    def bones(self, obj, members):
        names = {g.index: g.name for g in obj.vertex_groups}
        return {names[g.group] for i in members for g in obj.data.vertices[i].groups if g.weight > 0}

    def test_what_lies_on_the_chest_stays_on_it(self):
        obj, lock, panel, trim, _ribbon, _button = self.build()
        verts = meshdata.world_vertices(obj)
        # near enough to the lock to have ridden it by nearness alone
        for piece in (panel, trim):
            gaps = [np.linalg.norm(verts[lock] - verts[v], axis=1).min() for v in piece]
            self.assertLess(float(np.median(gaps)), cloth.RIDER_GAP)
        cloth.attach_riders(obj, lock, np.zeros(len(lock), bool))
        self.assertEqual(self.bones(obj, panel), {"Chest"})
        self.assertEqual(self.bones(obj, trim), {"Chest"})

    def test_a_piece_on_a_piece_that_rides_rides_too(self):
        obj, lock, _panel, _trim, ribbon, button = self.build()
        moved = cloth.attach_riders(obj, lock, np.zeros(len(lock), bool))
        self.assertEqual(self.bones(obj, ribbon), {"Hair0_0"})
        self.assertEqual(self.bones(obj, button), {"Hair0_0"})
        self.assertEqual(moved, len(ribbon) + len(button))

    def test_two_pieces_lying_on_each_other_ride_what_is_under_both(self):
        # model B: two beads at the end of her hat's tip, one inside the other, each read the other
        # as what it lay on and both stayed on the head
        obj, lock, _panel, _trim, ribbon, button = self.build()
        for index in ribbon + button:  # 7 mm from each other, 11 mm off the lock
            obj.data.vertices[index].co.y = -0.036
        cloth.attach_riders(obj, lock, np.zeros(len(lock), bool))
        self.assertEqual(self.bones(obj, ribbon), {"Hair0_0"})
        self.assertEqual(self.bones(obj, button), {"Hair0_0"})


if __name__ == "__main__":
    unittest.main()
