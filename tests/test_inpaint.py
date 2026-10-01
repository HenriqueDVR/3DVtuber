"""Filling untrusted skin weights from trusted ones (inpaint.py), and repairing adopted weights with it."""

import unittest

import bpy
import numpy as np

from vtuber_maker.pipeline import adopt, inpaint
from vtuber_maker.pipeline.rig import BONE_NAMES


def grid(nx, ny, x0=0.0, width=1.0):
    """A flat nx by ny grid of vertices over [x0, x0 + width] x [0, 1], two triangles per cell."""
    xs, ys = np.meshgrid(np.linspace(x0, x0 + width, nx), np.linspace(0, 1, ny), indexing="ij")
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    index = np.arange(nx * ny).reshape(nx, ny)
    a, b, c, d = index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    return verts, np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])


def two_bones(verts, low=0.25, high=0.75):
    """Bone 0 left of `low`, bone 1 right of `high`, the middle untrusted and full of junk."""
    x = verts[:, 0]
    known = (x < low) | (x > high)
    weights = np.zeros((len(verts), 2))
    weights[x < low, 0] = 1
    weights[x > high, 1] = 1
    weights[~known] = [0.9, 0.1]
    return weights, known


class Fill(unittest.TestCase):
    def setUp(self):
        self.verts, self.tris = grid(41, 11)
        self.weights, self.known = two_bones(self.verts)
        self.dense = inpaint.DENSE

    def tearDown(self):
        inpaint.DENSE = self.dense

    def test_blends_smoothly_between_held_weights(self):
        out, unresolved = inpaint.fill(self.verts, self.tris, self.weights, self.known)
        self.assertFalse(unresolved.any())
        np.testing.assert_array_equal(out[self.known], self.weights[self.known])
        np.testing.assert_allclose(out.sum(axis=1), 1.0)
        self.assertGreaterEqual(out.min(), 0.0)
        middle = out[np.abs(self.verts[:, 1] - 0.5) < 1e-9, 0]
        self.assertTrue((np.diff(middle) <= 1e-9).all(), "bone 0 falls off monotonically")
        self.assertAlmostEqual(float(middle[20]), 0.5, places=2)

    def test_conjugate_gradients_agree_with_the_direct_solve(self):
        direct, _ = inpaint.fill(self.verts, self.tris, self.weights, self.known)
        inpaint.DENSE = 10
        iterative, _ = inpaint.fill(self.verts, self.tris, self.weights, self.known)
        np.testing.assert_allclose(iterative, direct, atol=1e-3)

    def test_a_piece_with_nothing_trusted_keeps_its_weights(self):
        island, island_tris = grid(5, 5, x0=3.0)
        verts = np.vstack([self.verts, island])
        tris = np.vstack([self.tris, island_tris + len(self.verts)])
        weights = np.vstack([self.weights, np.tile([0.4, 0.6], (len(island), 1))])
        known = np.concatenate([self.known, np.zeros(len(island), dtype=bool)])
        out, unresolved = inpaint.fill(verts, tris, weights, known)
        np.testing.assert_array_equal(unresolved, np.arange(len(verts)) >= len(self.verts))
        np.testing.assert_array_equal(out[len(self.verts):], weights[len(self.verts):])
        alone, _ = inpaint.fill(self.verts, self.tris, self.weights, self.known)
        np.testing.assert_allclose(out[:len(self.verts)], alone)

    def test_a_uv_seam_does_not_cut_the_surface(self):
        # split the grid down its middle column, as an imported UV seam does
        whole, _ = inpaint.fill(self.verts, self.tris, self.weights, self.known)
        column = np.flatnonzero(np.abs(self.verts[:, 0] - 0.5) < 1e-9)
        copies = len(self.verts) + np.arange(len(column))
        remap = np.arange(len(self.verts))
        remap[column] = copies
        right = self.verts[self.tris][:, :, 0].mean(axis=1) > 0.5
        tris = self.tris.copy()
        tris[right] = remap[tris[right]]
        verts = np.vstack([self.verts, self.verts[column]])
        weights = np.vstack([self.weights, self.weights[column]])
        known = np.concatenate([self.known, self.known[column]])
        split, _ = inpaint.fill(verts, tris, weights, known)
        np.testing.assert_allclose(split[:len(self.verts)], whole, atol=1e-9)
        np.testing.assert_allclose(split[copies], whole[column], atol=1e-9)

    def test_a_kept_copy_on_a_skipped_seam_is_still_filled(self):
        # a hair region cut from the body: each seam position has a hair copy and a body copy
        column = np.flatnonzero(np.abs(self.verts[:, 0] - 0.5) < 1e-9)
        copies = len(self.verts) + np.arange(len(column))
        remap = np.arange(len(self.verts))
        remap[column] = copies
        right = self.verts[self.tris][:, :, 0].mean(axis=1) > 0.5
        tris = self.tris.copy()
        tris[right] = remap[tris[right]]
        verts = np.vstack([self.verts, self.verts[column]])
        weights = np.vstack([self.weights, np.tile([0.9, 0.1], (len(column), 1))])
        x = verts[:, 0]
        skip = np.arange(len(verts)) < len(self.verts)
        skip &= x < 0.51
        known = x > 0.76
        out, unresolved = inpaint.fill(verts, tris, weights, known, skip)
        self.assertFalse(unresolved[copies].any())
        np.testing.assert_allclose(out[copies], [[0.0, 1.0]] * len(copies), atol=1e-9)

    def test_skipped_vertices_neither_fill_nor_are_filled(self):
        x = self.verts[:, 0]   # grid columns every 2.5 cm; the cuts fall between them
        known = x < 0.19
        skip = (x > 0.39) & (x < 0.61)
        out, unresolved = inpaint.fill(self.verts, self.tris, self.weights, known, skip)
        np.testing.assert_array_equal(out[skip], self.weights[skip])
        between = (x > 0.19) & (x < 0.39)
        np.testing.assert_allclose(out[between], [[1.0, 0.0]] * int(between.sum()), atol=1e-9)
        # past the cut nothing trusted is reachable
        np.testing.assert_array_equal(unresolved, x > 0.61)


class Match(unittest.TestCase):
    def test_distance_and_facing(self):
        verts, tris = grid(5, 5)
        normals = inpaint.vertex_normals(verts, tris)
        np.testing.assert_allclose(np.abs(normals[:, 2]), 1.0)
        found = verts.copy()
        found[0] += [0, 0, 0.2]            # too far: 5% of the 1.41 m diagonal is 7 cm
        turned = normals.copy()
        turned[1] = np.array([0, np.sin(0.7), np.cos(0.7)]) * np.sign(normals[1, 2])   # 40 degrees off
        ok = inpaint.matched(verts, normals, found, turned, float(np.linalg.norm(np.ptp(verts, axis=0))))
        self.assertFalse(ok[0])
        self.assertFalse(ok[1])
        self.assertTrue(ok[2:].all())


class RepairAdopted(unittest.TestCase):
    """A strip from the right arm over the shoulder, one patch of it also on Head (hair touching it)."""

    @classmethod
    def setUpClass(cls):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        up = np.array([0, 0, 0.1])
        chain = [("hips", None, (0, 0, 0.9)), ("spine", "hips", (0, 0, 1.0)), ("chest", "spine", (0, 0, 1.1)),
                 ("upper_chest", "chest", (0, 0, 1.2)), ("neck", "upper_chest", (0, 0, 1.3)),
                 ("head", "neck", (0, 0, 1.4)), ("right_shoulder", "upper_chest", (-0.05, 0, 1.3)),
                 ("right_upper_arm", "right_shoulder", (-0.15, 0, 1.3))]
        bones = {}
        for key, parent, head in chain:
            bones[BONE_NAMES[key]] = (np.array(head, float), np.array(head, float) + up,
                                      BONE_NAMES[parent] if parent else None)
        cls.armature = adopt.build_armature(bones)
        verts, tris = grid(41, 11, x0=-1.0)
        mesh = bpy.data.meshes.new("strip")
        mesh.from_pydata(verts.tolist(), [], tris.tolist())
        cls.mesh = bpy.data.objects.new("strip", mesh)
        bpy.context.scene.collection.objects.link(cls.mesh)
        x = verts[:, 0]
        arm, chest, head = (cls.mesh.vertex_groups.new(name=BONE_NAMES[k]) for k in
                            ("right_upper_arm", "upper_chest", "head"))
        cls.x = x
        # a lock of hair lying on the shoulder: its end on arm and head, the rest on head and chest
        cls.hair = (x > -0.46) & (x < -0.24)      # grid columns every 2.5 cm; the cuts fall between them
        cls.mixed = ((x > -0.61) & (x < -0.39)) | (cls.hair & (x < -0.36))
        for i in range(len(verts)):
            share = float(np.clip((x[i] + 1.0) / 1.0, 0, 1))  # arm on the left, upper chest on the right
            arm.add([i], 0.0 if cls.hair[i] and not cls.mixed[i] else 1 - share, "REPLACE")
            chest.add([i], share, "REPLACE")
            if cls.mixed[i] or cls.hair[i]:
                head.add([i], 0.4, "REPLACE")
        cls.before = {g.name: [cls._weight(i, g.index) for i in range(len(verts))] for g in cls.mesh.vertex_groups}
        cls.far = adopt.reaching_across(cls.mesh, cls.armature)
        cls.report = adopt.repair_weights(cls.mesh, cls.armature, hair=cls.hair)

    @classmethod
    def _weight(cls, i, group):
        return next((g.weight for g in cls.mesh.data.vertices[i].groups if g.group == group), 0.0)

    def test_finds_head_on_the_arm_but_not_the_armpit_blend(self):
        np.testing.assert_array_equal(self.far, self.mixed)
        self.assertEqual(self.report["reaching_across"], int(self.mixed.sum()))
        self.assertEqual(self.report["unresolved"], 0)

    def test_the_shoulder_refills_without_head(self):
        head = self.mesh.vertex_groups[BONE_NAMES["head"]].index
        for i in np.flatnonzero(self.mixed & ~self.hair):
            self.assertLess(self._weight(i, head), inpaint.FLOOR)
            total = sum(g.weight for g in self.mesh.data.vertices[i].groups)
            self.assertAlmostEqual(total, 1.0, places=5)

    def test_the_hair_refills_from_hair_without_the_arm(self):
        arm = self.mesh.vertex_groups[BONE_NAMES["right_upper_arm"]].index
        head = self.mesh.vertex_groups[BONE_NAMES["head"]].index
        for i in np.flatnonzero(self.mixed & self.hair):
            self.assertLess(self._weight(i, arm), inpaint.FLOOR)
            self.assertGreater(self._weight(i, head), 0.2)

    def test_leaves_everything_trusted_alone(self):
        for group in self.mesh.vertex_groups:
            for i in np.flatnonzero(~self.mixed):
                self.assertAlmostEqual(self._weight(i, group.index), self.before[group.name][i], places=6)


if __name__ == "__main__":
    unittest.main()
