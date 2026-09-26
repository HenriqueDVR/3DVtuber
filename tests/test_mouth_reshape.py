"""Re-hanging a tracked mouth patch's vowels from the mouth line (face_rig.reshape_mouth)."""

import unittest

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

from vtuber_maker.pipeline import face_rig

WIDTH = 0.024
COVER = {"x": 0.0, "y": -0.0008, "z": 1.3, "width": WIDTH}
BULGE = 0.0025  # how far forward "aa" pushes the lip below the mouth (m)


def co(key):
    values = np.empty(len(key.data) * 3)
    key.data.foreach_get("co", values)
    return values.reshape(-1, 3)


class ReshapeMouth(unittest.TestCase):
    """A flat face (y = 0, facing -Y) with a closed mouth patch, "aa" opened above the line as model A's was."""

    ARGS = {}

    @classmethod
    def setUpClass(cls):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mesh = bpy.data.meshes.new("face")
        obj = bpy.data.objects.new("face", mesh)
        bpy.context.scene.collection.objects.link(obj)
        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=40, y_segments=40, size=0.04)
        bmesh.ops.rotate(bm, verts=bm.verts, matrix=Matrix.Rotation(np.pi / 2, 3, "X"))
        bmesh.ops.translate(bm, verts=bm.verts, vec=(0.0, 0.0, COVER["z"]))
        bm.to_mesh(mesh)
        bm.free()
        face_count = len(mesh.vertices)
        obj[face_rig.MOUTH_COVER_PROPERTY] = COVER
        # the patch: a (u, v) grid, all on the mouth line at rest
        bm = bmesh.new()
        bm.from_mesh(mesh)
        grid, rows = [], []
        for j in range(face_rig.MOUTH_ROWS + 1):
            v = 1.0 - 2.0 * j / face_rig.MOUTH_ROWS
            row = []
            for i in range(face_rig.MOUTH_COLUMNS + 1):
                u = -1.0 + 2.0 * i / face_rig.MOUTH_COLUMNS
                row.append(bm.verts.new((0.5 * WIDTH * u * np.sqrt(max(1 - v * v, 0)), COVER["y"], COVER["z"])))
                grid.append((u, v, 1.0))
            rows.append(row)
        for j in range(face_rig.MOUTH_ROWS):
            for i in range(face_rig.MOUTH_COLUMNS):
                bm.faces.new([rows[j][i], rows[j + 1][i], rows[j + 1][i + 1], rows[j][i + 1]])
        bm.to_mesh(mesh)
        bm.free()
        attribute = mesh.attributes.new(face_rig.MOUTH_GRID_ATTRIBUTE, "FLOAT_VECTOR", "POINT")
        values = np.zeros((len(mesh.vertices), 3))
        values[face_count:] = grid
        attribute.data.foreach_set("vector", values.ravel())
        cls.patch = np.arange(len(mesh.vertices)) >= face_count

        obj.shape_key_add(name="Basis")
        basis = co(mesh.shape_keys.reference_key)
        opened = basis.copy()
        opened[cls.patch] = face_rig._mouth_shape(np.array(grid), COVER, (0.42, 0.24, 0.13))
        # the vowel drops the lower lip, bulging it forward under the patch
        lip = ~cls.patch & (np.abs(basis[:, 0]) < 0.4 * WIDTH) & (basis[:, 2] < COVER["z"] - 0.1 * WIDTH) \
            & (basis[:, 2] > COVER["z"] - 0.5 * WIDTH)
        opened[lip, 1] -= BULGE
        obj.shape_key_add(name="aa").data.foreach_set("co", opened.astype(np.float32).ravel())
        slid = basis.copy()
        slid[cls.patch, 0] += 0.3 * WIDTH
        obj.shape_key_add(name="mouthLeft").data.foreach_set("co", slid.astype(np.float32).ravel())
        cls.before = {key.name: co(key) for key in mesh.shape_keys.key_blocks}
        cls.lip = lip
        cls.result = face_rig.reshape_mouth(obj, **cls.ARGS)
        cls.keys = mesh.shape_keys.key_blocks

    def test_open_mouth_hangs_from_the_line(self):
        aa = co(self.keys["aa"])[self.patch]
        self.assertGreater(self.before["aa"][self.patch, 2].max() - COVER["z"], 0.1 * WIDTH)
        self.assertAlmostEqual(aa[:, 2].max(), COVER["z"] + face_rig.MOUTH_TOP_LIFT * WIDTH, places=6)

    def test_vowel_keeps_its_size(self):
        was, now = self.before["aa"][self.patch], co(self.keys["aa"])[self.patch]
        scale = self.ARGS.get("scale", 1.0)
        np.testing.assert_allclose(np.ptp(now[:, [0, 2]], axis=0), scale * np.ptp(was[:, [0, 2]], axis=0), atol=1e-6)

    def test_closed_line_is_narrower(self):
        rest = co(self.keys["Basis"])[self.patch]
        aa_width = self.ARGS.get("scale", 1.0) * np.ptp(self.before["aa"][self.patch, 0])
        self.assertAlmostEqual(np.ptp(rest[:, 0]), self.ARGS.get("rest_share", face_rig.MOUTH_REST_SHARE) * aa_width, places=6)
        self.assertAlmostEqual(np.ptp(rest[:, 2]), self.ARGS.get("rest_height", 0.0) * WIDTH, places=6)
        self.assertAlmostEqual(rest[:, 2].max(), COVER["z"] + face_rig.MOUTH_TOP_LIFT * WIDTH, places=6)

    def test_patch_stays_in_front_of_the_dropped_lip(self):
        # every patch vertex whose neighbouring quads overlap the bulge, not only those right over it
        aa = co(self.keys["aa"])
        patch = aa[self.patch]
        cell = (np.ptp(patch[:, 0]) / face_rig.MOUTH_COLUMNS, np.ptp(patch[:, 2]) / face_rig.MOUTH_ROWS)
        lip = aa[self.lip]
        over = [(np.abs(lip[:, 0] - x) <= cell[0]) & (np.abs(lip[:, 2] - z) <= cell[1]) for x, _y, z in patch]
        touching = np.array([hit.any() for hit in over])
        self.assertGreater(touching.sum(), 10)
        self.assertLess(patch[touching, 1].max(), -BULGE)

    def test_other_keys_keep_their_patch_move(self):
        delta = co(self.keys["mouthLeft"]) - co(self.keys["Basis"])
        np.testing.assert_allclose(delta[self.patch], (self.before["mouthLeft"] - self.before["Basis"])[self.patch], atol=1e-6)

    def test_face_is_left_alone(self):
        for name in ("Basis", "aa", "mouthLeft"):
            with self.subTest(key=name):
                np.testing.assert_allclose(co(self.keys[name])[~self.patch], self.before[name][~self.patch], atol=1e-7)


class ReshapeFreeMouth(ReshapeMouth):
    """Bigger openings and a drawn closed mouth, as fix_mouth.py --free --scale makes them."""

    ARGS = {"scale": 1.3, "rest_share": 0.6, "rest_height": 0.07}


if __name__ == "__main__":
    unittest.main()
