"""Blink and mouth shape keys on a face whose features are only painted on."""

import unittest

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from support import FIXTURE_OUT, load_fixture
from vtuber_maker.pipeline import face, face_rig, landmarks, meshdata, preview, texture_space

SHAPE_KEYS = ("blink_left", "blink_right", "aa", "ih", "ou", "ee", "oh")


def evaluated_vertices(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    coords = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", coords)
    evaluated.to_mesh_clear()
    return coords.reshape(-1, 3)


def front_material(obj, x, z):
    """Material of the first surface a ray hits looking at the character from the front (object space)."""
    tree = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    _hit, _normal, polygon, _distance = tree.ray_cast(Vector((x, -5.0, z)), Vector((0.0, 1.0, 0.0)))
    return obj.material_slots[obj.data.polygons[polygon].material_index].material.name


class FaceRigOnSyntheticCharacter(unittest.TestCase):
    """The fixture mesh has an identity transform, so object space is world space."""

    @classmethod
    def setUpClass(cls):
        cls.mesh, _ = load_fixture()
        joints = landmarks.detect_body(meshdata.world_vertices(cls.mesh), meshdata.triangles(cls.mesh.data))
        cls.features = face.detect_features(cls.mesh, joints)
        face_rig.cut_mouth(cls.mesh, cls.features)
        face_rig.cut_eyelids(cls.mesh, cls.features)
        face_rig.add_shape_keys(cls.mesh, cls.features)
        cls.rest = evaluated_vertices(cls.mesh)

    def tearDown(self):
        for block in self.mesh.data.shape_keys.key_blocks:
            block.value = 0.0

    def displacement_with(self, key):
        self.mesh.data.shape_keys.key_blocks[key].value = 1.0
        return evaluated_vertices(self.mesh) - self.rest

    def near(self, point, radius):
        return np.linalg.norm(self.rest - np.asarray(point), axis=1) < radius

    def test_creates_blink_and_vowel_keys(self):
        names = self.mesh.data.shape_keys.key_blocks.keys()
        for key in SHAPE_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, names)

    def test_blink_left_pulls_the_upper_eye_down(self):
        eye = np.asarray(self.features["left_eye"])
        width, height = self.features["eye_size"]
        upper_eye = (
            (np.abs(self.rest[:, 0] - eye[0]) < 0.3 * width)
            & (self.rest[:, 2] > eye[2])
            & (self.rest[:, 2] < eye[2] + 0.4 * height)
            & (self.rest[:, 1] < eye[1] + 0.01)
        )
        moved = self.displacement_with("blink_left")
        self.assertGreater(-moved[upper_eye, 2].mean(), 0.35 * height)

    def test_blink_left_closes_the_whole_upper_eye(self):
        eye = np.asarray(self.features["left_eye"])
        width, height = self.features["eye_size"]
        upper_eye = (
            (np.abs(self.rest[:, 0] - eye[0]) < 0.4 * width)
            & (self.rest[:, 2] > eye[2])
            & (self.rest[:, 2] < eye[2] + 0.45 * height)
            & (self.rest[:, 1] < eye[1] + 0.01)
        )
        closed = self.rest + self.displacement_with("blink_left")
        self.assertLess(float(closed[upper_eye, 2].max()), eye[2] - 0.3 * height)

    def test_closed_eye_shows_no_paint_above_the_lash_line(self):
        # Checks the rendered result: triangles straddling the lash line can smear
        # squashed iris into view even when every vertex has moved.
        eye = np.asarray(self.features["left_eye"], dtype=float)
        width, height = self.features["eye_size"]
        extent, size = 4 * width, 256

        def paint_fraction():
            path = str(FIXTURE_OUT / "test_face_rig_left_eye.png")
            preview.render(path, "front", size=size, focus=(tuple(eye), extent))
            image = bpy.data.images.load(path)
            pixels = np.array(image.pixels[:], dtype=np.float32).reshape(size, size, 4)[..., :3]
            bpy.data.images.remove(image)

            def to_pixel(offset):
                return int(round((offset / extent + 0.5) * size))

            region = pixels[
                to_pixel(-0.35 * height) : to_pixel(0.45 * height),
                to_pixel(-0.4 * width) : to_pixel(0.4 * width),
            ]
            return float(np.mean(np.linalg.norm(region - self.features["skin_color"], axis=2) > 0.2))

        self.assertGreater(paint_fraction(), 0.2, "the open eye should show paint in this region")
        self.mesh.data.shape_keys.key_blocks["blink_left"].value = 1.0
        self.assertLess(paint_fraction(), 0.02)

    def test_blinking_does_not_fold_the_skin_over_itself(self):
        # Folded triangles face backwards and render as dark slivers under lighting.
        # Seen from the front, facing is the sign of a triangle's normal along Y; triangles
        # folded flat onto the lash line are edge-on and invisible, so they don't count.
        tris = meshdata.triangles(self.mesh.data)
        nearby = tris[self.near(self.features["left_eye"], 0.06)[tris].any(axis=1)]
        closed = self.rest + self.displacement_with("blink_left")

        def facing(points):
            corners = points[nearby]
            return np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 1]

        before, after = facing(self.rest), facing(closed)
        visible = (np.abs(before) > 2e-7) & (np.abs(after) > 2e-7)
        flipped = visible & (np.sign(before) != np.sign(after))
        self.assertEqual(int(flipped.sum()), 0, f"{int(flipped.sum())} of {len(nearby)} triangles flipped")

    def test_blink_left_leaves_the_right_eye_alone(self):
        moved = self.displacement_with("blink_left")
        self.assertLess(np.abs(moved[self.near(self.features["right_eye"], 0.03)]).max(), 1e-4)

    def test_blinking_keeps_the_eyelid_on_the_face_surface(self):
        rest_surface = BVHTree.FromObject(self.mesh, bpy.context.evaluated_depsgraph_get())
        moved = self.displacement_with("blink_left")
        closed = self.rest + moved
        shifted = np.linalg.norm(moved, axis=1) > 1e-4
        gaps = [rest_surface.find_nearest(Vector(point))[3] for point in closed[shifted]]
        self.assertLess(max(gaps), 0.002)

    def test_blinking_does_not_drag_the_hairline(self):
        # Any vertex sharing an edge with hair carries hair texture on its triangles;
        # moving it smears the bangs down over the forehead.
        colors = texture_space.vertex_colors(self.mesh, texture_space.base_color_image(self.mesh))
        hair = np.linalg.norm(colors - self.features["hair_color"], axis=1) < 0.12
        edges = np.empty(len(self.mesh.data.edges) * 2, dtype=np.int64)
        self.mesh.data.edges.foreach_get("vertices", edges)
        a, b = edges[0::2], edges[1::2]
        touching_hair = hair.copy()
        touching_hair[a[hair[b]]] = True
        touching_hair[b[hair[a]]] = True
        hairline = touching_hair & self.near(self.features["left_eye"], 0.05)
        self.assertTrue(hairline.any(), "fixture should have bangs near the eye")
        moved = self.displacement_with("blink_left")
        self.assertLess(np.abs(moved[hairline]).max(), 1e-4)

    def test_aa_opens_the_mouth_to_show_inside(self):
        mouth = self.features["mouth"]
        # Just below the cut, so with the mouth closed the ray lands on the lower lip.
        x, z = mouth[0], mouth[2] - 0.001
        self.assertNotEqual(front_material(self.mesh, x, z), face_rig.MOUTH_MATERIAL)
        self.mesh.data.shape_keys.key_blocks["aa"].value = 1.0
        self.assertEqual(front_material(self.mesh, x, z), face_rig.MOUTH_MATERIAL)

    def test_aa_leaves_the_eyes_alone(self):
        moved = self.displacement_with("aa")
        self.assertLess(np.abs(moved[self.near(self.features["left_eye"], 0.02)]).max(), 1e-4)


if __name__ == "__main__":
    unittest.main()
