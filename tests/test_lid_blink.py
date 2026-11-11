"""The lid blink: a lash strip and a lid brought down over a still, tracked eye (pipeline/lid_blink.py)."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from vtuber_maker.pipeline import eyes, lid_blink, vrm_export
from test_eyes import EYES, FACE_Y, IRIS, LASH, PUPIL, face_json, make_face, painted_face, pixel_world


def blinked(only=None):
    """The synthetic face with moving irises and the lid blink: (object, vertex count before it, keys before it, report)."""
    obj = make_face()[0]
    eyes.add_moving_irises(obj, face_json(), only=only)
    count = len(obj.data.vertices)
    before = {k.name: eyes._key_co(k) for k in obj.data.shape_keys.key_blocks}
    return obj, count, before, lid_blink.add_lid_blink(obj, face_json(), only=only)


def sheet_tree(obj, co, mark):
    """The sheets marked `mark` (1 strip, 2 lid; 0 both) with their vertices at `co`, as a BVH tree."""
    mesh = obj.data
    marks = eyes.get_int(mesh, lid_blink.ATTRIBUTE)
    polygons = [list(p.vertices) for p in mesh.polygons if marks[p.vertices[0]] > 0 and mark in (0, marks[p.vertices[0]])]
    return BVHTree.FromPolygons([Vector(v) for v in co], polygons)


def opening_points(side):
    """World (x, z) of points well inside one painted eye's opening."""
    eye = EYES[side]
    left, right = sorted((eye["inner"], eye["outer"]))
    xs = np.linspace(left + 12, right - 12, 9)
    ys = np.linspace(eye["y"] - 8, eye["y"] + 12, 5)
    return np.array([pixel_world(x, y) for x in xs for y in ys])


class LidBlink(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obj, cls.count, cls.before, cls.report = blinked()
        cls.keys = cls.obj.data.shape_keys.key_blocks
        cls.basis = eyes._key_co(cls.keys[0])
        cls.marks = eyes.get_int(cls.obj.data, lid_blink.ATTRIBUTE)

    def test_a_strip_and_a_lid_for_each_eye(self):
        self.assertEqual(sorted(k for k in self.report if k != "images"), ["left", "right"])
        new = self.marks[self.count:]
        self.assertTrue((new > 0).all())
        self.assertFalse(self.marks[:self.count].any())
        for mark in (1, 2):
            sides = np.sign(self.basis[self.count:][new == mark, 0])
            self.assertEqual(sorted(set(sides.tolist())), [-1.0, 1.0], mark)

    def test_only_the_sheets_move_in_a_blink_and_only_there(self):
        for name, before in self.before.items():
            after = eyes._key_co(self.keys[name])
            if name.startswith("eyeBlink"):
                # nothing of the face, iris or white moves: the old squeeze is gone
                np.testing.assert_allclose(after[:self.count], self.basis[:self.count], atol=1e-7, err_msg=name)
            else:
                np.testing.assert_allclose(after[:self.count], before, atol=1e-7, err_msg=name)
                np.testing.assert_allclose(after[self.count:], self.basis[self.count:], atol=1e-7, err_msg=name)
        for side in ("left", "right"):
            moved = np.linalg.norm(eyes._key_co(self.keys[f"eyeBlink{side.title()}"]) - self.basis, axis=1) > 1e-6
            mine = np.zeros(len(self.basis), bool)
            mine[self.count:] = np.sign(self.basis[self.count:, 0]) == (1 if side == "left" else -1)
            self.assertTrue(moved.any(), side)
            self.assertFalse((moved & ~mine).any(), side)

    def test_at_rest_the_opening_is_clear(self):
        # (of the lid: the strip's cut-out edge lies TUCK into the opening, under the lashes' soft edge)
        tree = sheet_tree(self.obj, self.basis, 2)
        for side in EYES:
            hits = [tree.ray_cast(Vector((x, -1.0, z)), Vector((0, 1, 0)), 2.0)[0] for x, z in opening_points(side)]
            self.assertEqual(sum(h is not None for h in hits), 0, side)

    def test_shut_the_lid_covers_the_opening_in_front_of_the_eye(self):
        shut = self.basis.copy()
        for side in ("left", "right"):
            co = eyes._key_co(self.keys[f"eyeBlink{side.title()}"])
            shut[self.count:] = np.where(self.basis[self.count:, :1] * (1 if side == "left" else -1) > 0, co[self.count:], shut[self.count:])
        tree = sheet_tree(self.obj, shut, 2)
        for side in EYES:
            hits = [tree.ray_cast(Vector((x, -1.0, z)), Vector((0, 1, 0)), 2.0)[0] for x, z in opening_points(side)]
            self.assertTrue(all(h is not None for h in hits), side)
            # in front of the face and of the lids layer laid over it, never behind
            self.assertLess(max(h.y for h in hits), FACE_Y - eyes.OVERLAY_GAP, side)
            self.assertGreater(min(h.y for h in hits), FACE_Y - 0.01, side)

    def test_the_lid_shows_skin_with_no_eye_on_it(self):
        rows = eyes.image_rows(bpy.data.images[lid_blink.LID_MATERIAL])
        rgb = rows[..., :3]
        for colour in (IRIS, PUPIL, LASH):
            self.assertEqual(int((np.abs(rgb - colour).max(axis=-1) < 0.1).sum()), 0, colour)

    def test_the_strip_carries_the_lashes_and_nothing_else(self):
        rows = eyes.image_rows(bpy.data.images[lid_blink.LASH_MATERIAL])
        shown = rows[..., 3] > 0.5
        self.assertGreater(shown.sum(), 100)
        np.testing.assert_allclose(np.median(rows[shown, :3], axis=0), LASH, atol=0.05)

    def test_each_eye_closes_its_whole_height(self):
        for side in EYES:
            r = self.report[side]
            self.assertGreater(r["middle_mm"], r["opening_mm"])
            self.assertGreaterEqual(r["least_clear_mm"], -0.05, side)


class TheLashesRead(unittest.TestCase):
    def test_the_dark_band_over_the_opening_and_no_skin(self):
        rows, openings, _ = painted_face()
        points = face_json()
        for side, eye in eyes.marked_eyes(points, (rows.shape[1], rows.shape[0])).items():
            x0, y0, x1, y1 = eye.box
            rgb = rows[y0:y1, x0:x1, :3].astype(np.float64)
            opening = openings[side][y0:y1, x0:x1]
            lash, _ = lid_blink._lash_mask(rgb, opening, eye)
            painted = np.abs(rgb - LASH).max(axis=-1) < 0.05
            self.assertGreater((lash & painted).sum(), 0.8 * painted.sum(), side)
            self.assertFalse((lash & ~eyes._dilate(painted, 1)).any(), side)


class OneEyeShown(unittest.TestCase):
    def test_the_covered_eye_gets_no_sheets(self):
        obj, count, before, report = blinked(only="left")
        self.assertNotIn("right", report)
        basis = eyes._key_co(obj.data.shape_keys.key_blocks[0])
        self.assertTrue((basis[count:, 0] > 0).all())


class Refused(unittest.TestCase):
    def test_without_tracked_eyes_nothing_changes(self):
        obj = make_face()[0]  # a painted face with no EyeLids layer
        count, materials = len(obj.data.vertices), len(obj.data.materials)
        with self.assertRaises(lid_blink.LidBlinkRefused):
            lid_blink.add_lid_blink(obj, face_json())
        self.assertEqual((len(obj.data.vertices), len(obj.data.materials)), (count, materials))


class SheetNormalsInTheFile(unittest.TestCase):
    def test_only_the_sheets_targets_lose_their_normals(self):
        gltf = {"asset": {"version": "2.0"}, "materials": [{"name": "Face"}, {"name": lid_blink.LID_MATERIAL}],
                "meshes": [{"primitives": [{"material": 0, "targets": [{"POSITION": 0, "NORMAL": 1}]},
                                           {"material": 1, "targets": [{"POSITION": 0, "NORMAL": 1}, {"POSITION": 0}]}]}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "t.vrm"
            vrm_export._write_glb(path, gltf, b"\0" * 8)
            self.assertEqual(vrm_export.still_sheet_normals(path), 1)
            text = path.read_bytes()
            size = struct.unpack_from("<I", text, 12)[0]
            got = json.loads(text[20:20 + size])
        self.assertEqual(got["meshes"][0]["primitives"][0]["targets"], [{"POSITION": 0, "NORMAL": 1}])
        self.assertEqual(got["meshes"][0]["primitives"][1]["targets"], [{"POSITION": 0}, {"POSITION": 0}])


if __name__ == "__main__":
    unittest.main()
