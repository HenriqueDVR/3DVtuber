"""Re-hanging a tracked mouth patch's vowels from the mouth line (face_rig.reshape_mouth)."""

import sys
import unittest
from pathlib import Path

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


class MouthFixture:
    """A flat face (y = 0, facing -Y) with a closed mouth patch and its openings, put through reshape_mouth."""

    ARGS = {}
    # vowel -> (half width, half height, drop), mouth widths: the openings the mesh comes with
    OPENINGS = {"aa": (0.42, 0.24, 0.13)}

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
        # the vowel drops the lower lip, bulging it forward under the patch
        lip = ~cls.patch & (np.abs(basis[:, 0]) < 0.4 * WIDTH) & (basis[:, 2] < COVER["z"] - 0.1 * WIDTH) \
            & (basis[:, 2] > COVER["z"] - 0.5 * WIDTH)
        for name, shape in cls.OPENINGS.items():
            opened = basis.copy()
            opened[cls.patch] = face_rig._mouth_shape(np.array(grid), COVER, shape)
            if name == "aa":
                opened[lip, 1] -= BULGE
            obj.shape_key_add(name=name).data.foreach_set("co", opened.astype(np.float32).ravel())
        slid = basis.copy()
        slid[cls.patch, 0] += 0.3 * WIDTH
        obj.shape_key_add(name="mouthLeft").data.foreach_set("co", slid.astype(np.float32).ravel())
        cls.before = {key.name: co(key) for key in mesh.shape_keys.key_blocks}
        cls.lip = lip
        cls.result = face_rig.reshape_mouth(obj, **cls.ARGS)
        cls.keys = mesh.shape_keys.key_blocks


class ReshapeMouth(MouthFixture, unittest.TestCase):
    """ "aa" opened above the line, as model A's was."""

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


class TrackerShapesOnASmallFreeMouth(MouthFixture, unittest.TestCase):
    """The tracker's mouth shapes (add_face_tracking.py), rebuilt on a free mouth drawn small, as
    fix_mouth.py --free --scale 0.7 leaves it: a closed line 0.8 mouth widths wide, every opening
    narrower than it. Shape keys add up, and jaw, pucker and funnel together narrowed the patch past
    nothing: it came out the other side facing backwards (the user, 2026-10-08: "the mouth inverts")."""

    OPENINGS = dict(face_rig.MOUTH_SHAPES)
    ARGS = {"scale": 0.7, "rest_share": 0.8 / 0.7, "rest_height": 0.07}
    TOGETHER = ({"mouthPucker": 1}, {"mouthFunnel": 1}, {"mouthPucker": 1, "mouthFunnel": 1},
                {"mouthPucker": 1, "mouthFunnel": 1, "jawOpen": 1}, {"mouthPucker": 1, "jawOpen": 0.3})

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import add_face_tracking
        obj = bpy.data.objects["face"]
        for name in (face_rig.EYE_INSERT_ATTRIBUTE, face_rig.FIXED_ATTRIBUTE):
            obj.data.attributes.new(name, "INT", "POINT")
        cls.tracking = add_face_tracking
        add_face_tracking.FaceShapes(obj).add_mouth(0.008)
        cls.keys = obj.data.shape_keys.key_blocks
        obj.data.calc_loop_triangles()
        triangles = np.array([t.vertices[:] for t in obj.data.loop_triangles])
        cls.triangles = triangles[cls.patch[triangles].all(axis=1)]

    def shown(self, values):
        basis = co(self.keys["Basis"])
        return basis + sum(value * (co(self.keys[name]) - basis) for name, value in values.items())

    def facing(self, points):
        a, b, c = (points[self.triangles[:, i]] for i in range(3))
        return (b[:, 0] - a[:, 0]) * (c[:, 2] - a[:, 2]) - (b[:, 2] - a[:, 2]) * (c[:, 0] - a[:, 0])

    def test_the_closed_line_is_the_wide_one(self):
        # 0.8 mouth widths, whatever the openings' size
        self.assertAlmostEqual(np.ptp(co(self.keys["Basis"])[self.patch, 0]), 0.8 * WIDTH, places=6)

    def test_no_part_of_the_mouth_turns_its_back(self):
        front = np.sign(self.facing(self.shown({"aa": 1})))
        for values in self.TOGETHER:
            with self.subTest(shapes=values):
                area = self.facing(self.shown(values))
                turned = (np.sign(area) != front) & (np.abs(area) > 1e-12)
                self.assertEqual(int(turned.sum()), 0, f"{int(turned.sum())} of {len(area)} triangles face backwards")

    def test_the_mouth_stays_open_wide_enough_to_see(self):
        narrowest = min(np.ptp(co(self.keys[name])[self.patch, 0]) for name in ("aa", "ou", "oh"))
        everything = self.shown({"mouthPucker": 1, "mouthFunnel": 1, "jawOpen": 1})[self.patch]
        self.assertGreaterEqual(np.ptp(everything[:, 0]), self.tracking.PATCH_FLOOR * narrowest - 1e-6)

    def test_the_pucker_alone_is_a_small_upright_mouth(self):
        # scaled down with the jaw and the funnel it was a flat line, 25 by 9 mm on model D where it had
        # been 8 by 9 (the user, 2026-10-09: the small upright pucker was the better one, its bug aside)
        pucker, ou = (self.shown({name: 1})[self.patch] for name in ("mouthPucker", "ou"))
        self.assertAlmostEqual(np.ptp(pucker[:, 0]), np.ptp(ou[:, 0]), places=6)
        self.assertGreater(np.ptp(pucker[:, 2]), 0.6 * np.ptp(ou[:, 2]))

    def test_the_mouth_comes_forward_with_the_puckered_skin(self):
        # the corners' pull moves the skin forward; a patch left behind it showed skin through the mouth
        pucker = co(self.keys["mouthPucker"]) - co(self.keys["Basis"])
        self.assertLess(pucker[~self.patch, 1].min(), -1e-4)
        self.assertLess(pucker[self.patch, 1].min(), -1e-4)

    def test_no_shape_puts_skin_in_front_of_the_mouth(self):
        # smiles and frowns moved skin and patch by one bump: skin stood 5 mm through the princess's mouth
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree
        mesh = bpy.data.objects["face"].data
        polygons = [p.vertices[:] for p in mesh.polygons if not self.patch[list(p.vertices)].all()]
        for key in self.keys:
            with self.subTest(key=key.name):
                points = co(key)
                face = BVHTree.FromPolygons([Vector(v) for v in points], polygons)
                for x, y, z in points[self.patch]:
                    hit = face.ray_cast(Vector((x, y - 0.03, z)), Vector((0.0, 1.0, 0.0)), 0.06)[0]
                    if hit is not None:
                        self.assertLessEqual(y, hit.y - face_rig.MOUTH_SURFACE_GAP + 1e-6)


class TrackerShapesOnABigFreeMouth(TrackerShapesOnASmallFreeMouth):
    """Drawn big (--scale 1.3, as model A's), the openings never narrow the closed line to nothing:
    each keeps its whole shape."""

    ARGS = {"scale": 1.3, "rest_share": 0.8 / 1.3, "rest_height": 0.07}

    def test_pucker_is_as_narrow_as_its_vowel_and_the_others_keep_their_shape(self):
        # the pucker: the whole vowel's width at its share of the height; nothing else gives way to it here
        basis = co(self.keys["Basis"])
        pucker, ou = co(self.keys["mouthPucker"]) - basis, co(self.keys["ou"]) - basis
        narrower = self.patch & (np.abs(co(self.keys["ou"])[:, 0]) < np.abs(basis[:, 0]))
        np.testing.assert_allclose(pucker[narrower, 0], ou[narrower, 0], atol=1e-6)
        np.testing.assert_allclose(pucker[self.patch, 2], 0.7 * ou[self.patch, 2], atol=1e-6)
        funnel, oh = co(self.keys["mouthFunnel"]) - basis, co(self.keys["oh"]) - basis
        np.testing.assert_allclose(funnel[self.patch, 0], oh[self.patch, 0], atol=1e-6)


class AnUpperLipOverTheMouth(unittest.TestCase):
    """The catgirl's upper lip stands 10 mm in front of her drawn mouth; in a frown it came down into its
    line and the mouth was pushed out in front of it, floating in profile."""

    def test_the_mouth_stays_behind_an_overhang(self):
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree
        quad = lambda y: [(-0.01, y, -0.01), (0.01, y, -0.01), (0.01, y, 0.01), (-0.01, y, 0.01)]
        surface = BVHTree.FromPolygons([Vector(v) for v in quad(0.0) + quad(-0.01)], [(0, 1, 2, 3), (4, 5, 6, 7)])
        point = np.array([[0.0, -face_rig.MOUTH_SURFACE_GAP, 0.0]])
        # skin behind the mouth at rest counts, the lip that stood 10 mm in front of it doesn't
        kept = face_rig._in_front_of_face(point, surface, (1e-4, 1e-4), skip=(np.array([0.0, -0.01]), point[:, 1]))
        self.assertAlmostEqual(kept[0, 1], -face_rig.MOUTH_SURFACE_GAP, places=6)
        pushed = face_rig._in_front_of_face(point, surface, (1e-4, 1e-4))
        self.assertAlmostEqual(pushed[0, 1], -0.01 - face_rig.MOUTH_SURFACE_GAP, places=6)


class ALipComingDownOverTheMouth(unittest.TestCase):
    """The miko's upper lip, 10 mm in front of her drawn mouth, came down over it in a frown: skin through the
    mouth in Warudo. The lip stops at the mouth's edge; the mouth stays where it is."""

    def test_the_lip_stops_short_of_the_mouth(self):
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree
        bpy.ops.wm.read_factory_settings(use_empty=True)
        quad = lambda x0, x1, z0, z1, y: [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
        verts = quad(-0.05, 0.05, -0.05, 0.05, 0.0)                      # the face
        verts += quad(-0.01, 0.01, -0.002, 0.002, -face_rig.MOUTH_SURFACE_GAP)  # the drawn mouth
        verts += quad(-0.015, 0.015, 0.003, 0.01, -0.01)                  # the lip standing out above it
        mesh = bpy.data.meshes.new("face")
        mesh.from_pydata(verts, [], [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)])
        obj = bpy.data.objects.new("face", mesh)
        bpy.context.scene.collection.objects.link(obj)
        grid = np.zeros((12, 3))
        grid[4:8, 2] = 1.0
        mesh.attributes.new(face_rig.MOUTH_GRID_ATTRIBUTE, "FLOAT_VECTOR", "POINT").data.foreach_set("vector", grid.ravel())
        obj.shape_key_add(name="Basis")
        frown = obj.shape_key_add(name="mouthFrownLeft")
        for i in range(8, 12):
            frown.data[i].co.z -= 0.008  # down over the mouth
        self.assertGreater(face_rig.lip_off_mouth(obj), 0)
        moved = np.array([v.co for v in frown.data])
        face = BVHTree.FromPolygons([Vector(v) for v in moved], [(0, 1, 2, 3), (8, 9, 10, 11)])
        for x, y, z in moved[4:8]:
            hit = face.ray_cast(Vector((x, y - 0.03, z)), Vector((0.0, 1.0, 0.0)), 0.06)[0]
            self.assertTrue(hit is None or hit.y > y, "skin in front of the mouth")
        np.testing.assert_allclose(moved[4:8], np.array(verts[4:8]), atol=1e-7)  # the mouth stays


class AnUpperLipEasedBehindTheMouth(unittest.TestCase):
    """The miko's upper lip stood 3.5 mm proud of her drawn mouth's top edge: from a little below it hid the mouth."""

    def test_the_lip_over_the_mouth_ends_behind_it_and_the_nose_is_left(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        quad = lambda x0, x1, z0, z1, y: [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
        verts = quad(-0.01, 0.01, -0.002, 0.002, 0.0)          # the drawn mouth, 2 cm wide
        verts += quad(-0.01, 0.01, 0.0025, 0.003, -0.0035)     # the lip just over it, standing out
        verts += quad(-0.01, 0.01, 0.02, 0.021, -0.008)        # the nose, well above
        mesh = bpy.data.meshes.new("face")
        mesh.from_pydata(verts, [], [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)])
        obj = bpy.data.objects.new("face", mesh)
        bpy.context.scene.collection.objects.link(obj)
        grid = np.zeros((12, 3))
        grid[:4, 2] = 1.0
        mesh.attributes.new(face_rig.MOUTH_GRID_ATTRIBUTE, "FLOAT_VECTOR", "POINT").data.foreach_set("vector", grid.ravel())
        obj.shape_key_add(name="Basis")
        self.assertGreater(face_rig.flatten_lip(obj), 0.0)
        y = np.array([v.co.y for v in mesh.vertices])
        self.assertTrue((y[4:8] > 0.0).all(), "the lip still stands in front of the mouth")
        np.testing.assert_allclose(y[8:], -0.008)  # the nose
        np.testing.assert_allclose(y[:4], 0.0)     # the mouth


class AnInsertThatIsNoEye(unittest.TestCase):
    """The lid blink is sized from the eye insert. A girl in glasses had a scrap of 4 x 5.5 mm found for it
    in an eye marked 39 x 31 mm: her blink moved 1.3 mm. Model B's inserts, a quarter of her marked eyes'
    height, make the blink that was seen and kept."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import add_face_tracking
        cls.tracking = add_face_tracking

    @staticmethod
    def box(width, height):
        return np.array([[0.0, 0.0, 0.0], [width, 0.0, height]])

    def test_a_scrap_gives_way_to_the_marks(self):
        self.assertTrue(self.tracking.is_scrap(self.box(.0039, .0055), (.0229, 1.3902, .0624, 1.4214)))

    def test_a_small_eye_insert_does_not(self):
        self.assertFalse(self.tracking.is_scrap(self.box(.0274, .0133), (.0308, 1.3024, .0887, 1.3507)))

    def test_nor_an_insert_as_tall_as_the_eye(self):
        self.assertFalse(self.tracking.is_scrap(self.box(.0234, .0305), (.0173, 1.37, .0581, 1.3999)))


if __name__ == "__main__":
    unittest.main()
