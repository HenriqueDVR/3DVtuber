"""Eyes that follow face tracking: a moving iris under see-through lids, or a whole eye sliding."""

import unittest

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from vtuber_maker.pipeline import eyes, face_points, face_rig

SIZE = 500
# The patch image spans x -0.1..0.1 and z 1.3..1.5 on a flat face at y = -0.1, like test_face_points.
X0, X1, Z0, Z1, FACE_Y = -0.1, 0.1, 1.3, 1.5, -0.1
SKIN, WHITE, IRIS, PUPIL, LASH, LID = ((0.95, 0.82, 0.74), (0.92, 0.94, 0.97), (0.15, 0.35, 0.85),
                                        (0.05, 0.08, 0.3), (0.05, 0.04, 0.05), (0.8, 0.5, 0.45))
# Each eye's painting in image pixels: inner and outer corner x, corner y, top and bottom of the
# opening, and the iris's centre and radii (its top hides under the lashes, its bottom under the lid).
EYES = {"left": dict(inner=300, outer=400, y=100, top=75, bottom=125, iris=(350, 100, 18, 30)),
        "right": dict(inner=200, outer=100, y=100, top=75, bottom=125, iris=(150, 100, 18, 30))}


def pixel_world(px, py):
    return X0 + (X1 - X0) * px / SIZE, Z1 - (Z1 - Z0) * py / SIZE


def lid_curves(eye):
    xs = np.array([eye["inner"], (eye["inner"] + eye["outer"]) / 2, eye["outer"]], float)
    upper = np.polyfit(xs, [eye["y"], eye["top"], eye["y"]], 2)
    lower = np.polyfit(xs, [eye["y"], eye["bottom"], eye["y"]], 2)
    return np.poly1d(upper), np.poly1d(lower)


def painted_face():
    """(rows, opening masks, iris masks): the face patch image, top row first, and where each eye's parts are."""
    ys, xs = np.mgrid[0:SIZE, 0:SIZE] + 0.5
    rows = np.ones((SIZE, SIZE, 4), np.float32)
    rows[..., :3] = SKIN
    openings, irises = {}, {}
    for side, eye in EYES.items():
        upper, lower = lid_curves(eye)
        left, right = sorted((eye["inner"], eye["outer"]))
        across = (xs >= left) & (xs <= right)
        opening = across & (ys >= upper(xs)) & (ys <= lower(xs))
        rows[opening, :3] = WHITE
        cx, cy, rx, ry = eye["iris"]
        iris = opening & (((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1)
        rows[iris, :3] = IRIS
        rows[iris & (((xs - cx) / 6) ** 2 + ((ys - cy) / 10) ** 2 <= 1), :3] = PUPIL
        rows[iris & (np.hypot(xs - cx + 6, ys - 84) <= 3), :3] = WHITE  # a catchlight near the top
        rows[(xs >= left - 4) & (xs <= right + 4) & (ys < upper(xs)) & (ys >= upper(xs) - 6), :3] = LASH
        rows[across & (ys > lower(xs)) & (ys <= lower(xs) + 2), :3] = LID
        openings[side], irises[side] = opening, iris
    return rows, openings, irises


def face_json(swap_names=False):
    """face.json points, marked like a person would: the lower curve a few pixels low."""
    points = {"size": [SIZE, SIZE], "mouth": {"left": [220, 300], "right": [280, 300], "top": [250, 295], "bottom": [250, 305]}}
    for side, eye in EYES.items():
        middle = (eye["inner"] + eye["outer"]) / 2
        name = {"left": "right", "right": "left"}[side] if swap_names else side
        points[f"{name}_eye"] = {"inner": [eye["inner"], eye["y"]], "outer": [eye["outer"], eye["y"]],
                                 "top": [middle, eye["top"]], "bottom": [middle, eye["bottom"] + 3]}
    return points


def make_face():
    """A flat face on a front-mapped face patch, weighted to Head, with blink keys; the patch image's rows."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new("face")
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new()
    n = 40
    grid = [[bm.verts.new((X0 + (X1 - X0) * i / n, FACE_Y, Z0 + (Z1 - Z0) * j / n)) for i in range(n + 1)] for j in range(n + 1)]
    for j in range(n):
        for i in range(n):
            quad = bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
            for loop in quad.loops:
                co = loop.vert.co
                loop[uv].uv = ((co.x - X0) / (X1 - X0), (co.z - Z0) / (Z1 - Z0))
    bm.to_mesh(mesh)
    bm.free()
    rows, openings, irises = painted_face()
    material = bpy.data.materials.new(face_points.FACE_PATCH_MATERIAL)
    material.use_nodes = True
    texture = material.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = eyes.new_image("face_patch_image", rows)
    material.node_tree.links.new(texture.outputs["Color"], material.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    mesh.materials.append(material)
    obj = bpy.data.objects.new("face", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.vertex_groups.new(name="Head").add(list(range(len(mesh.vertices))), 1.0, "REPLACE")
    obj.shape_key_add(name="Basis", from_mix=False)
    co = np.array([v.co for v in mesh.vertices])
    for side, eye in EYES.items():
        cx, cz = pixel_world(eye["iris"][0], eye["y"])
        near = (np.abs(co[:, 0] - cx) < 0.03) & (co[:, 2] > cz) & (co[:, 2] < cz + 0.02)
        moved = co.copy()
        moved[near, 2] -= 0.002
        face_rig._add_key(obj, f"eyeBlink{side.title()}", moved)
    return obj, rows, openings, irises


def full_rows(built, name):
    """A material's image, cropped to what its polygons use (eyes.crop_to_use), back at the face patch's size."""
    x0, y0, x1, y1 = built["crops"][name]["box"]
    width, height = built["crops"][name]["full"]
    rows = np.zeros((height, width, 4), np.float32)
    rows[height - y1:height - y0, x0:x1] = eyes.image_rows(bpy.data.images[name])
    return rows


def iris_coloured(rows):
    return (np.abs(rows[..., :3] - IRIS).max(axis=-1) < 0.1) | (np.abs(rows[..., :3] - PUPIL).max(axis=-1) < 0.1)


class FindingTheEye(unittest.TestCase):
    """The painted opening and iris, found from the marks and the painting."""

    @classmethod
    def setUpClass(cls):
        cls.rows, cls.openings, cls.irises = painted_face()
        cls.eyes = eyes.marked_eyes(face_json(), (SIZE, SIZE))

    def crop(self, side):
        eye = self.eyes[side]
        x0, y0, x1, y1 = eye.box
        return eye, self.rows[y0:y1, x0:x1, :3].astype(np.float64)

    def test_the_opening_follows_the_paint_not_the_marks(self):
        for side in EYES:
            eye, rgb = self.crop(side)
            opening = eyes.find_opening(rgb, eye)
            x0, y0, x1, y1 = eye.box
            truth = self.openings[side][y0:y1, x0:x1]
            # the marked lower curve is 3 px low; the found opening stops at the painted lid
            self.assertLess((opening & ~truth).sum(), 0.03 * truth.sum(), side)
            self.assertGreater((opening & truth).sum(), 0.95 * truth.sum(), side)

    def test_the_iris_ellipse_is_fitted_below_the_lashes(self):
        for side in EYES:
            eye, rgb = self.crop(side)
            opening = eyes.find_opening(rgb, eye)
            _iris, (cx, cy, rx, ry) = eyes.find_iris(rgb, opening)
            x0, y0 = eye.box[:2]
            true_cx, true_cy, true_rx, true_ry = EYES[side]["iris"]
            self.assertAlmostEqual(cx + x0, true_cx, delta=2, msg=side)
            self.assertAlmostEqual(rx, true_rx, delta=2, msg=side)
            self.assertAlmostEqual(cy + y0, true_cy, delta=4, msg=side)

    def test_a_shaded_white_is_repainted_smooth_and_the_outline_goes_with_the_iris(self):
        # model A: shade under the lashes, a pink lower lid, a pale blue side and a dark outline a few
        # pixels outside the iris left grey, pink and dark bands where the iris had been.
        eye, rgb = self.crop("left")
        x0, y0, x1, y1 = eye.box
        ys, xs = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]] + 0.5
        white = self.openings["left"][y0:y1, x0:x1] & ~self.irises["left"][y0:y1, x0:x1]
        top, bottom = EYES["left"]["top"] - y0, EYES["left"]["bottom"] - y0
        rgb[white & (ys < top + 10)] = (0.6, 0.62, 0.7)
        rgb[white & (ys > bottom - 12)] = (0.9, 0.8, 0.82)
        rgb[white & (xs > EYES["left"]["outer"] - x0 - 20)] = (0.78, 0.9, 1.0)
        cx, cy, rx, ry = EYES["left"]["iris"]
        ring = np.hypot((xs - cx + x0) / (rx + 3), (ys - cy + y0) / (ry + 3))
        outline = white & (ring <= 1) & (ring > rx / (rx + 3))
        rgb[outline] = (0.3, 0.3, 0.45)
        found = eyes.find_opening(rgb, eye)
        see = eyes.window(found)
        iris, ellipse = eyes.find_iris(rgb, found)
        layer, hole = eyes.iris_layer(rgb, see, iris, ellipse)
        self.assertGreater((layer[..., 3][outline & see] > 0).mean(), 0.9)
        painted = eyes.paint_out(rgb, see, hole)
        inside = (see | hole) & ~eyes._dilate(~(see | hole), eyes.WHITE_SMOOTH + 1)
        filled = painted[inside]
        self.assertGreater(float(filled.min()), 0.7)
        self.assertLess(float((filled.max(axis=-1) - filled.min(axis=-1)).max()), 0.1)  # white, not pink or blue
        # shaded, but smoothly: no band or edge from the painting's shading
        for axis in (0, 1):
            step = np.abs(np.diff(painted, axis=axis)).max(axis=-1)
            both = inside[1:] & inside[:-1] if axis == 0 else inside[:, 1:] & inside[:, :-1]
            self.assertLess(float(step[both].max()), 0.02)
        self.assertGreater(float(np.ptp(filled.mean(axis=-1))), 0.03)

    def test_sides_follow_where_the_eyes_are_not_what_the_file_says(self):
        swapped = eyes.marked_eyes(face_json(swap_names=True), (SIZE, SIZE))
        for side in EYES:
            self.assertEqual(swapped[side].box, self.eyes[side].box)


class MovingIrises(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obj, cls.rows, cls.openings, cls.irises = make_face()
        mesh = cls.obj.data
        cls.count = len(mesh.vertices)
        cls.keys_before = {k.name: eyes._key_co(k) for k in mesh.shape_keys.key_blocks}
        cls.built = eyes.add_moving_irises(cls.obj, face_json())

    def test_the_iris_is_painted_out_where_the_lids_are_see_through(self):
        face = eyes.image_rows(face_points._patch_image(self.obj))
        window = self.built["window"]
        self.assertGreater(window.sum(), 0.8 * sum(o.sum() for o in self.openings.values()))
        self.assertEqual(int((iris_coloured(face) & window).sum()), 0)

    def test_the_lids_are_see_through_only_in_the_opening(self):
        lids = full_rows(self.built, eyes.LIDS_MATERIAL)
        for side, eye in EYES.items():
            cx, cy = eye["iris"][:2]
            middle = (eye["inner"] + eye["outer"]) // 2
            self.assertEqual(lids[cy, cx, 3], 0.0, side)
            self.assertEqual(lids[eye["top"] - 3, middle, 3], 1.0, side)  # the lashes stay on the lids
            np.testing.assert_allclose(lids[eye["top"] - 3, middle, :3], LASH, atol=0.02)

    def test_the_eyes_images_are_cropped_to_what_they_show(self):
        for name in (eyes.LIDS_MATERIAL, eyes.IRIS_MATERIAL):
            width, height = bpy.data.images[name].size
            self.assertLess(width * height, 0.5 * SIZE * SIZE, name)

    def test_the_iris_layer_carries_the_whole_iris(self):
        layer = full_rows(self.built, eyes.IRIS_MATERIAL)
        for side, eye in EYES.items():
            cx, cy = eye["iris"][:2]
            self.assertEqual(layer[cy + 20, cx, 3], 1.0, side)  # shown by the opening
            hidden = eye["top"] + 3  # under the lashes' soft edge, which stays on the lids
            self.assertEqual(layer[hidden, cx, 3], 1.0, side)
            self.assertTrue(iris_coloured(layer[..., :3])[hidden, cx], side)
            # the iris's colour carried on past the catchlight (round cx - 6), not the catchlight
            self.assertTrue(iris_coloured(layer[..., :3])[hidden, cx - 4], side)

    def test_every_look_keeps_the_iris_under_the_lids(self):
        numbers, failures = eyes.check_irises(self.obj, self.built)
        self.assertEqual(failures, [])
        self.assertEqual(numbers["over_lids"], 0)

    def test_look_keys_slide_one_iris_along_its_track(self):
        mesh = self.obj.data
        keys = mesh.shape_keys.key_blocks
        basis = eyes._key_co(keys[0])
        for side, info in self.built["eyes"].items():
            across, up = info["slide"]
            ids = info["vertices"]
            inward = -1 if side == "left" else 1
            for name, expected in ((f"eyeLookIn{side.title()}", (inward * across, 0.0)),
                                   (f"eyeLookOut{side.title()}", (-inward * across, 0.0)),
                                   (f"eyeLookUp{side.title()}", (0.0, up))):
                delta = eyes._key_co(keys[name]) - basis
                np.testing.assert_allclose(delta[ids][:, [0, 2]], np.tile(expected, (len(ids), 1)), atol=1e-6, err_msg=name)
                others = np.setdiff1d(np.arange(len(basis)), ids)
                self.assertLess(np.abs(delta[others]).max(), 1e-7, name)

    def test_the_iris_and_white_stay_put_while_the_lids_blink(self):
        # (a painted face without eye inserts: with inserts they squeeze with them, see BlinkWithAModelledEye)
        mesh = self.obj.data
        keys = mesh.shape_keys.key_blocks
        basis = eyes._key_co(keys[0])
        source = eyes.get_int(mesh, eyes.OVERLAY_ATTRIBUTE) - 1
        copies = np.nonzero(source >= 0)[0]
        still = (eyes.get_int(mesh, eyes.IRIS_ATTRIBUTE) > 0) | (eyes.get_int(mesh, eyes.WHITE_ATTRIBUTE) > 0)
        for name in ("eyeBlinkLeft", "eyeBlinkRight"):
            delta = eyes._key_co(keys[name]) - basis
            self.assertLess(np.abs(delta[still]).max(), 1e-7, name)
            np.testing.assert_allclose(delta[copies], delta[source[copies]], atol=1e-7, err_msg=name)

    def test_the_face_it_started_with_is_untouched(self):
        mesh = self.obj.data
        for name, before in self.keys_before.items():
            after = eyes._key_co(mesh.shape_keys.key_blocks[name])[:self.count]
            np.testing.assert_allclose(after, before, atol=1e-7, err_msg=name)


class BlinkWithAModelledEye(unittest.TestCase):
    """An eye modelled as its own piece is squeezed by the blink: the iris and white on it squeeze too."""

    def test_the_iris_and_white_take_the_move_of_the_insert_under_them(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        # an insert of 3 x 3 vertices 2 cm square, then one "iris" and one "white" vertex in front of its middle row and top row
        points = [(x, 0.0, z) for z in (0.0, 0.01, 0.02) for x in (0.0, 0.01, 0.02)] + [(0.01, -0.001, 0.01), (0.01, -0.0005, 0.02)]
        data = bpy.data.meshes.new("eye")
        data.from_pydata(points, [], [(0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7)])
        obj = bpy.data.objects.new("eye", data)
        bpy.context.scene.collection.objects.link(obj)
        eyes._set_int(data, face_rig.EYE_INSERT_ATTRIBUTE, [eyes.SIDE_IDS["left"]] * 9 + [0, 0])
        obj.shape_key_add(name="Basis")
        blink, look = obj.shape_key_add(name="eyeBlinkLeft"), obj.shape_key_add(name="eyeLookUpLeft")
        for key in (blink, look):
            key.value = 0.0
            for i in range(9):  # squeezed toward the bottom row: the higher, the further down
                key.data[i].co.z -= 0.5 * points[i][2]
        eyes._blink_with_inserts(obj, {"left": {"vertices": np.array([9]), "white_vertices": np.array([10])}})
        self.assertAlmostEqual(blink.data[9].co.z, 0.005, places=5)
        self.assertAlmostEqual(blink.data[10].co.z, 0.01, places=5)
        self.assertAlmostEqual(blink.data[9].co.y, -0.001, places=6)
        # keys that aren't blinks leave the layers alone
        self.assertAlmostEqual(look.data[9].co.z, 0.01, places=6)


class TheSocketBehindAnIris(unittest.TestCase):
    """How far behind an iris disc the model's own eye white lies (the fox girl's: 28 mm, the inside of
    her head showing beside her irises from the side)."""

    def depth(self, back):
        """A socket's floor `back` metres behind a disc standing at y = -0.1, 4 cm square."""
        corners = [(-0.02, -0.1 + back, 1.38), (0.02, -0.1 + back, 1.38), (0.02, -0.1 + back, 1.42), (-0.02, -0.1 + back, 1.42)]
        scene = BVHTree.FromPolygons([Vector(c) for c in corners], [(0, 1, 2, 3)])
        flat = lambda xz: np.full(len(xz), -0.1)  # noqa: E731
        return eyes._socket_depth(scene, (np.array([-0.015, 1.385]), np.array([0.015, 1.415])), flat, lambda xz: xz)

    def test_a_pit_is_measured(self):
        self.assertAlmostEqual(self.depth(0.028), 0.028, places=5)

    def test_what_stands_in_front_of_the_disc_is_no_socket(self):
        self.assertEqual(self.depth(-0.005), 0.0)


class OneEyeCovered(unittest.TestCase):
    """One eye under hair or an eyepatch (manifest `eye_shown`): the other alone is worked on and blinks."""

    POINTS = {"size": [100, 100],
              "left_eye": {"outer": [80, 50], "inner": [60, 50], "top": [70, 45], "bottom": [70, 55]},
              "right_eye": {"outer": [20, 50], "inner": [40, 50], "top": [30, 45], "bottom": [30, 55]}}

    def test_both_show_unless_one_is_named(self):
        self.assertEqual(sorted(eyes.shown_eyes(self.POINTS, (100, 100))), ["left", "right"])
        self.assertEqual(eyes.shown_eyes(self.POINTS, (100, 100)).keys(), eyes.marked_eyes(self.POINTS, (100, 100)).keys())

    def test_the_one_named_is_the_character_s_own_side(self):
        for side, column in (("left", 70), ("right", 30)):  # a front view: her left eye is on the right of the picture
            shown = eyes.shown_eyes(self.POINTS, (100, 100), side)
            self.assertEqual(list(shown), [side])
            self.assertAlmostEqual((shown[side].points["inner"][0] + shown[side].points["outer"][0]) / 2, column)

    def test_the_covered_eye_s_blink_is_emptied_and_nothing_else(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        data = bpy.data.meshes.new("face")
        data.from_pydata([(0, 0, 0), (0.01, 0, 0), (0, 0, 0.01)], [], [(0, 1, 2)])
        obj = bpy.data.objects.new("face", data)
        bpy.context.scene.collection.objects.link(obj)
        obj.shape_key_add(name="Basis")
        keys = {name: obj.shape_key_add(name=name) for name in ("eyeBlinkLeft", "eyeBlinkRight", "eyeLookUpRight", "aa")}
        for key in keys.values():
            key.value = 0.0
            key.data[2].co.z = 0.0
        self.assertEqual(eyes.still_blink(obj, "right"), ["eyeBlinkRight"])
        self.assertAlmostEqual(keys["eyeBlinkRight"].data[2].co.z, 0.01, places=6)
        for name in ("eyeBlinkLeft", "eyeLookUpRight", "aa"):
            self.assertAlmostEqual(keys[name].data[2].co.z, 0.0, places=6)


class SlidingEyes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obj, cls.rows, cls.openings, cls.irises = make_face()
        points = face_json()
        features = face_points.apply(cls.obj, {}, points)
        face_rig.store_painted_eyes(cls.obj, features)
        # the face tracking blink squeezes the painted eyes, as scripts/add_face_tracking.py does
        keys = cls.obj.data.shape_keys.key_blocks
        basis = eyes._key_co(keys[0])
        for side, box in cls.obj[face_rig.PAINTED_EYES_PROPERTY].items():
            squeezed = basis.copy()
            squeezed[:, 2] += face_rig.squeeze_shut(basis, tuple(box), 0.7)
            keys[f"eyeBlink{side.title()}"].data.foreach_set("co", squeezed.astype(np.float32).ravel())
        cls.built = eyes.add_sliding_eyes(cls.obj, points)

    def test_the_blink_keeps_the_amount_it_had(self):
        for side in EYES:
            self.assertAlmostEqual(self.built["blink"][side], 0.7, places=3)

    def test_the_painting_moves_off_the_face_onto_the_patches(self):
        face = eyes.image_rows(face_points._patch_image(self.obj))
        paint = full_rows(self.built, eyes.PATCH_MATERIAL)
        for side, eye in EYES.items():
            cx, cy = eye["iris"][:2]
            np.testing.assert_allclose(face[cy, cx, :3], SKIN, atol=0.03, err_msg=side)
            np.testing.assert_allclose(paint[cy + 20, cx, :3], IRIS, atol=0.02, err_msg=side)
            self.assertEqual(paint[cy + 20, cx, 3], 1.0, side)
            self.assertEqual(paint[eye["top"] - 3, cx, 3], 1.0, side)  # the lashes go with it

    def test_look_keys_slide_the_patch_along_the_face(self):
        keys = self.obj.data.shape_keys.key_blocks
        basis = eyes._key_co(keys[0])
        for side, info in self.built["patches"].items():
            across, _up = info["slide"]
            outward = 1 if side == "left" else -1
            delta = eyes._key_co(keys[f"eyeLookOut{side.title()}"]) - basis
            np.testing.assert_allclose(delta[info["vertices"], 0], outward * across, atol=1e-7)
            np.testing.assert_allclose(delta[info["vertices"], 1], 0.0, atol=1e-6)  # the face is flat

    def test_the_blink_squeezes_only_the_patch(self):
        mesh = self.obj.data
        keys = mesh.shape_keys.key_blocks
        basis = eyes._key_co(keys[0])
        patch = eyes.get_int(mesh, eyes.PATCH_ATTRIBUTE)
        for side, ident in eyes.SIDE_IDS.items():
            delta = eyes._key_co(keys[f"eyeBlink{side.title()}"]) - basis
            self.assertGreater(np.abs(delta[patch == ident, 2]).max(), 0.005, side)
            self.assertLess(np.abs(delta[patch != ident]).max(), 1e-7, side)

    def test_nothing_hides_the_patches_in_any_look(self):
        _numbers, failures = eyes.check_sliding(self.obj, self.built)
        self.assertEqual(failures, [])


class ErasedEyeFill(unittest.TestCase):
    def test_fill_carries_the_skin_round_it_into_the_middle_of_a_big_hole(self):
        # a blush ramp across the face with a 200 px eye erased from it: the middle has to follow the
        # ramp (plain averaging left it flat, and the blush stopped at the hole's edge)
        ys, xs = np.mgrid[0:300, 0:300].astype(np.float32)
        values = np.stack([xs / 300, ys / 300, np.full_like(xs, .5)], axis=-1)
        hole = np.zeros((300, 300), bool)
        hole[50:250, 50:250] = True
        filled = face_points.fill_across(np.where(hole[..., None], 0.0, values), hole)
        self.assertLess(np.abs(filled[hole] - values[hole]).max(), .05)
        flat = face_points.fill_from_border(np.where(hole[..., None], 0.0, values), hole, passes=300)
        self.assertGreater(np.abs(flat[hole] - values[hole]).max(), .2)
        # a stroke the paint mask missed, outside the domain, doesn't bleed in
        stroke = np.zeros_like(hole)
        stroke[40:50, 50:250] = True
        dirty = np.where(stroke[..., None], 1.0, np.where(hole[..., None], 0.0, values))
        clean = face_points.fill_across(dirty, hole, domain=~stroke)
        self.assertLess(np.abs(clean[hole][:, 2] - .5).max(), .02)  # flat .5 in blue; the stroke is 1


class SqueezeShut(unittest.TestCase):
    def test_the_lid_and_the_bottom_meet_and_the_skin_beyond_stays(self):
        box = (0.0, 0.0, 0.04, 0.02)
        line = face_rig.MEET_LINE * 0.02
        points = np.array([[0.02, 0.0, 0.02], [0.02, 0.0, 0.0], [0.02, 0.0, 0.1], [0.2, 0.0, 0.01]])
        move = face_rig.squeeze_shut(points, box, 1.0)
        self.assertAlmostEqual(points[0, 2] + move[0], line)
        self.assertAlmostEqual(points[1, 2] + move[1], line)
        self.assertEqual(move[2], 0.0)  # far above the lid stroke
        self.assertEqual(move[3], 0.0)  # far to the side


class ModelledSlide(unittest.TestCase):
    def test_an_iris_off_centre_stays_in_its_socket(self):
        # the miko's: 6.3 mm beside her right iris inward, 16.5 out; from the mean it slid 6.1 mm and left it
        parts = {"left": {"white_beside": (0.0093, 0.0121), "iris_size": (0.0197, 0.0197)},
                 "right": {"white_beside": (0.0063, 0.0165), "iris_size": (0.0197, 0.0197)}}
        across, up = eyes.modelled_slide(parts)
        self.assertAlmostEqual(across, eyes.SIDE_ROOM_MOST * 0.0063, places=6)
        self.assertAlmostEqual(up, eyes.UP_DOWN_TRAVEL * 0.0197, places=6)
        # centred irises slide as before
        even = {side: {"white_beside": (0.01, 0.01), "iris_size": (0.02, 0.02)} for side in ("left", "right")}
        self.assertAlmostEqual(eyes.modelled_slide(even)[0], eyes.SIDE_TRAVEL * 0.01, places=6)


class FaceOverTheDisc(unittest.TestCase):
    """Where the face stands against a modelled iris's disc: in front (it keeps its drawing), about level
    (the disc takes the drawing and comes in front of it: the miko's pink lower iris on her lower lid)."""

    def classes(self, face_y):
        from types import SimpleNamespace
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree
        plane = BVHTree.FromPolygons([Vector(v) for v in ((-1, face_y, -1), (1, face_y, -1), (1, face_y, 1), (-1, face_y, 1))], [(0, 1, 2, 3)])
        face = SimpleNamespace(world=lambda pixels: np.asarray(pixels, float) * 0.001)
        eye = SimpleNamespace(box=(0, 0, 4, 4))
        curve = lambda places: np.zeros(len(places))  # noqa: E731 -- the disc at y = 0
        near = np.ones((4, 4), bool)
        return (bool(eyes._in_front(face, eye, near, curve, plane).all()),
                bool(eyes._in_front(face, eye, near, curve, plane, level=True).all()))

    def test_in_front_level_and_behind(self):
        self.assertEqual(self.classes(-0.003), (True, False))   # well in front: the face's own
        self.assertEqual(self.classes(-0.0007), (True, True))   # just in front: the disc takes it, and comes forward
        self.assertEqual(self.classes(0.0002), (False, True))   # level, a little behind
        self.assertEqual(self.classes(0.003), (False, False))   # behind: the disc shows anyway


class LeftoversInTheWhite(unittest.TestCase):
    """The demon girl's white kept her iris's rim drawn past the circle read and flecks standing alone in it: both
    read as iris leaking into her white. The lids' lines, touching the eye's edge, stay."""

    def test_rim_and_flecks_go_lines_from_the_edge_stay(self):
        ys, xs = np.mgrid[0:60, 0:100] + 0.5
        see = (np.abs(xs - 50) < 45) & (np.abs(ys - 30) < 25)
        rgb = np.full((60, 100, 3), 0.95)
        rgb[~see] = (0.9, 0.75, 0.7)
        hole = np.hypot(xs - 50, ys - 30) < 12
        rgb[hole] = (0.6, 0.2, 0.6)
        rim = (np.hypot(xs - 50, ys - 30) >= 12) & (np.hypot(xs - 50, ys - 30) < 14)
        rgb[rim] = (0.3, 0.3, 0.4)          # the iris's rim, past the circle read
        rgb[30:32, 85:87] = (0.2, 0.2, 0.2)  # a fleck alone in the white
        line = see & (ys > 52) & (np.abs(xs - 20) < 8)
        rgb[(ys > 52) & (ys < 56) & (np.abs(xs - 20) < 8)] = (0.1, 0.1, 0.1)  # a lid's line from the edge
        painted = eyes.paint_out(rgb, see, hole)
        self.assertGreater(float(painted[rim & see].mean()), 0.7)
        self.assertGreater(float(painted[30:32, 85:87].mean()), 0.7)
        self.assertLess(float(painted[line].mean()), 0.4)


class TheWhitePastTheFace(unittest.TestCase):
    """The miko's eye white was painted past her face's outline, and the sheet and the disc's copy stood flat there,
    a skin panel before her side hair from 40 degrees; her bangs were the first thing in front."""

    class Picture:  # 100 x 100 px over x, z in -0.05 .. 0.05
        def pixel(self, xz):
            return (np.asarray(xz, float) + 0.05) * 1000.0

        def world(self, pixels):
            return np.asarray(pixels, float) / 1000.0 - 0.05

    def test_past_the_face_but_not_a_hole_in_it(self):
        # a face from x = -0.03 to 0.02 with a socket hole round the middle
        quads = [((-0.03, -0.03), (0.02, -0.005)), ((-0.03, 0.005), (0.02, 0.03)),
                 ((-0.03, -0.005), (-0.005, 0.005)), ((0.005, -0.005), (0.02, 0.005))]
        verts, faces = [], []
        for (x0, z0), (x1, z1) in quads:
            faces.append(tuple(range(len(verts), len(verts) + 4)))
            verts += [(x0, 0.0, z0), (x1, 0.0, z0), (x1, 0.0, z1), (x0, 0.0, z1)]
        face_tree = BVHTree.FromPolygons([Vector(v) for v in verts], faces)
        past = eyes._past_face(face_tree, self.Picture(), np.array([-0.04, -0.04]), np.array([0.04, 0.04]), (100, 100))
        at = lambda x, z: bool(past[int((z + 0.05) * 1000), int((x + 0.05) * 1000)])  # noqa: E731
        self.assertTrue(at(0.03, 0.0))    # past the face's edge
        self.assertFalse(at(0.0, 0.0))    # the socket's hole, which the face encloses
        self.assertFalse(at(-0.01, 0.02)) # the face itself

    def test_tucked_behind_what_lies_behind_not_the_bangs_in_front(self):
        quads = [(0, 1, 2, 3), (4, 5, 6, 7)]  # bangs 30 mm in front, side hair 20 mm behind
        verts = [Vector(v) for y in (-0.03, 0.02) for v in ((-0.05, y, -0.05), (0.05, y, -0.05), (0.05, y, 0.05), (-0.05, y, 0.05))]
        scene = BVHTree.FromPolygons(verts, quads)
        depth = eyes._tucked_behind(np.array([0.0, 0.0]), np.array([np.nan, 0.004]), scene, [(0.0, 0.0), (0.01, 0.0)],
                                    np.array([True, True]))
        self.assertAlmostEqual(depth[0], 0.02 + eyes.BACKING_GAP, places=6)
        self.assertAlmostEqual(depth[1], 0.004 + eyes.BACKING_GAP, places=6)


class TheLidEasedOffTheDisc(unittest.TestCase):
    """The knight's upper lids stood in front of his discs with the top of his drawn irises mapped onto them: that
    half stayed put as the disc looked aside. Lid geometry painted in the iris's colours goes behind the disc."""

    def test_only_the_lid_painted_with_the_iris_goes_and_never_the_disc(self):
        from types import SimpleNamespace
        # two quads 5 mm in front of the disc over the upper half of a 40 px crop (1 px = 1 mm): left, right
        verts = [(0.0, -0.005, 0.0), (0.02, -0.005, 0.0), (0.02, -0.005, 0.02), (0.0, -0.005, 0.02),
                 (0.02, -0.005, 0.0), (0.04, -0.005, 0.0), (0.04, -0.005, 0.02), (0.02, -0.005, 0.02)]
        quads = [(0, 1, 2, 3), (4, 5, 6, 7)]
        scene = BVHTree.FromPolygons([Vector(v) for v in verts], quads)
        face = SimpleNamespace(world=lambda pixels: np.asarray(pixels, float) * 0.001)
        eye = SimpleNamespace(box=(0, 0, 40, 40))
        yy, xx = np.mgrid[0:40, 0:40] + 0.5
        rgb = np.full((40, 40, 3), 0.95)
        circle = (20.0, 20.0, 10.0, 10.0)
        rgb[eyes._in_oval(xx, yy, circle)] = (0.2, 0.3, 0.9)  # a blue iris
        rgb[(yy < 20) & (xx < 20)] = (0.5, 0.5, 0.5)           # its upper left drawn grey: lid shading, not iris
        over = yy < 20
        curve = lambda places: np.zeros(len(places))  # noqa: E731 -- the disc at y = 0
        moves = eyes._ease_off_disc(face, eye, rgb, circle, over, curve, scene, np.arange(2), np.array(quads), [5])
        self.assertEqual(set(moves), {4, 6, 7})  # the right quad, but its vertex of the disc


class TheWhiteTuckedBehindTheFace(unittest.TestCase):
    """The miko's eye-white sheet stood 12.5 mm out of her face past the eye she was drawn with."""

    def test_only_outside_the_painted_eye_is_the_face_read(self):
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree

        class Picture:  # 100 x 100 px over x, z in -0.05 .. 0.05
            def pixel(self, xz):
                return (np.asarray(xz) + 0.05) * 1000.0

        face = BVHTree.FromPolygons([Vector(v) for v in ((-0.1, 0.02, -0.1), (0.1, 0.02, -0.1), (0.1, 0.02, 0.1),
                                                          (-0.1, 0.02, 0.1))], [(0, 1, 2, 3)])
        inside = np.zeros((100, 100), bool)
        inside[40:60, 40:60] = True  # the painted eye round the middle
        depth = eyes._face_outside(face, [(0.0, 0.0), (0.03, 0.03)], Picture(), inside)
        self.assertTrue(np.isnan(depth[0]))
        self.assertAlmostEqual(depth[1], 0.02, places=6)


class MovingIrisReading(unittest.TestCase):
    def test_an_iris_read_as_wide_as_its_eye_takes_the_other_mirrored(self):
        # the demon girl's left iris read 45 mm wide in an opening of 46 (a strand of fringe across it)
        opening = np.zeros((40, 100), bool)
        opening[10:30, 10:90] = True
        good = (40.0, 20.0, 12.0, 15.0)  # 8 px left of its opening's middle (50, 20)
        reads = {"left": (None, None, opening, opening.copy(), (50.0, 20.0, 39.0, 15.0)),
                 "right": (None, None, opening, opening.copy(), good)}
        eyes._mirror_misread(reads)
        self.assertEqual(reads["right"][4], good)
        self.assertEqual(reads["left"][4], (60.0, 20.0, 12.0, 15.0))
        self.assertLess(reads["left"][3].sum(), opening.sum())
        # both read plausibly: nothing changes
        reads = {side: (None, None, opening, opening.copy(), good) for side in ("left", "right")}
        eyes._mirror_misread(reads)
        self.assertEqual(reads["left"][4], good)

    def test_both_irises_read_as_wide_as_their_eyes_are_no_iris(self):
        # the plain boy's: 43 mm in openings of 43; the eye step refuses, his own iris discs move instead
        opening = np.zeros((40, 100), bool)
        opening[10:30, 10:90] = True
        wide = (50.0, 20.0, 39.0, 15.0)
        reads = {side: (None, None, opening, opening.copy(), wide) for side in ("left", "right")}
        self.assertTrue(eyes._mirror_misread(reads))
        self.assertEqual(reads["left"][4], wide)
        reads = {side: (None, None, opening, opening.copy(), (40.0, 20.0, 12.0, 15.0)) for side in ("left", "right")}
        self.assertFalse(eyes._mirror_misread(reads))

    def test_a_moving_iris_touching_its_corner_still_slides(self):
        # the plain boy's: no white on the inner side, 2.5 mm out; under the lid overlay it still looks round
        found = {"left": {"white_outer": 0.0025, "white_inner": 0.0, "iris_size": (0.043, 0.048)},
                 "right": {"white_outer": 0.0027, "white_inner": 0.0, "iris_size": (0.044, 0.048)}}
        across, _up = eyes.iris_slide(found)
        self.assertAlmostEqual(across, eyes.SIDE_TRAVEL * 0.0013, places=6)


if __name__ == "__main__":
    unittest.main()


class AWholeIrisFromThePartThatShows(unittest.TestCase):
    """A modelled iris disc, taller than the iris drawn on it, its top under the lid and its
    bottom skin-coloured behind the lower eye line (the plain boy's: 21 by 29 mm)."""

    RADIUS, CENTRE = 40, (60, 70)  # the drawn iris; the disc is 80 wide, 110 tall, from row 10

    @classmethod
    def setUpClass(cls):
        yy, xx = np.mgrid[0:130, 0:120] + 0.5
        cls.distance = np.hypot(xx - cls.CENTRE[0], yy - cls.CENTRE[1])
        cls.outline = ((xx - 60) / 40) ** 2 + ((yy - 65) / 55) ** 2 <= 1.0
        rgb = np.empty((130, 120, 3))
        rgb[:] = WHITE
        rgb[cls.distance < cls.RADIUS] = IRIS
        rgb[cls.distance < 0.9 * cls.RADIUS] = (0.6, 0.4, 0.2)
        rgb[cls.distance < 0.4 * cls.RADIUS] = PUPIL
        rgb[(np.hypot(xx - 48, yy - 58) < 6)] = (1.0, 1.0, 1.0)  # a catchlight
        cls.rgb = rgb.copy()
        rgb[yy < 50] = LASH      # the lid hides the top
        rgb[yy > 95] = SKIN      # behind the lower eye line the disc is skin-coloured
        cls.painted = rgb
        cls.known = cls.outline & (yy >= 50) & (yy <= 95)
        cls.layer, cls.circle = eyes.complete_iris(rgb, cls.outline, cls.known)

    def test_the_middle_is_found_where_the_drawing_is_round(self):
        cx, cy, radius, tall = self.circle
        self.assertAlmostEqual(cx, self.CENTRE[0], delta=1.0)
        self.assertAlmostEqual(cy, self.CENTRE[1], delta=3.0)
        self.assertAlmostEqual(radius, self.RADIUS, delta=1.0)
        self.assertAlmostEqual(tall, self.RADIUS, delta=1.5)  # a circle is read as one

    def test_what_shows_is_left_as_painted(self):
        shown = self.known & (self.distance < self.RADIUS - 3)
        np.testing.assert_allclose(self.layer[..., :3][shown], self.painted[shown], atol=1e-6)

    def test_the_hidden_top_and_bottom_are_iris_not_lash_or_skin(self):
        hidden = self.outline & ~self.known & (self.distance < 0.85 * self.RADIUS) & (self.distance > 0.5 * self.RADIUS)
        self.assertGreater(hidden.sum(), 200)
        self.assertLess(np.abs(self.layer[..., :3][hidden] - (0.6, 0.4, 0.2)).max(), 0.12)

    def test_past_the_iris_the_disc_is_see_through(self):
        self.assertTrue((self.layer[..., 3][self.distance > self.RADIUS + 4] == 0).all())
        self.assertTrue((self.layer[..., 3][self.distance < self.RADIUS - 4] == 1).all())


class AnIrisDrawnPaleBelow(unittest.TestCase):
    """A disc that is iris to its edge, deep violet above and lilac below (the fox girl's)."""

    RADIUS, CENTRE = 40, (60, 65)
    VIOLET, LILAC, PINK = (0.35, 0.25, 0.6), (0.72, 0.62, 0.85), (0.9, 0.72, 0.74)

    @classmethod
    def setUpClass(cls):
        cls.yy, xx = np.mgrid[0:130, 0:120] + 0.5
        cls.outline = np.hypot(xx - cls.CENTRE[0], cls.yy - cls.CENTRE[1]) <= cls.RADIUS
        cls.known = cls.outline & (cls.yy >= 28)

    def circle(self, below):
        rgb = np.empty((130, 120, 3))
        rgb[:] = WHITE
        rgb[self.outline] = self.VIOLET
        rgb[self.outline & (self.yy > 45)] = below
        return eyes.drawn_circle(rgb, self.outline, self.known)

    def test_the_lilac_is_too_pale_for_the_plain_rule(self):
        saturation, value = eyes._saturation_value(np.array(self.LILAC))
        self.assertLess(saturation, eyes.IRIS_SATURATION)
        self.assertGreater(value, eyes.IRIS_DARK)

    def test_the_iris_is_the_whole_disc(self):
        cx, cy, radius, tall = self.circle(self.LILAC)
        self.assertAlmostEqual(radius, self.RADIUS, delta=1.5)
        self.assertAlmostEqual(cy, self.CENTRE[1], delta=3.0)

    def test_a_pale_colour_of_another_hue_is_no_iris(self):
        self.assertLess(self.circle(self.PINK)[2], 0.9 * self.RADIUS)

    def test_nor_is_white(self):
        self.assertLess(self.circle(WHITE)[2], 0.9 * self.RADIUS)


class AnIrisDrawnAsAnOval(unittest.TestCase):
    """An iris taller than wide, its top under the lid and its lower end cut by the lower eye line
    (the user: an iris is an oval as often as a circle, and a circle left white under it)."""

    WIDE, TALL, CENTRE = 36, 45, (60, 70)

    @classmethod
    def setUpClass(cls):
        yy, xx = np.mgrid[0:140, 0:120] + 0.5
        cls.inside = np.hypot((xx - cls.CENTRE[0]) / cls.WIDE, (yy - cls.CENTRE[1]) / cls.TALL)
        cls.outline = np.hypot((xx - 60) / 44, (yy - 70) / 58) <= 1.0
        rgb = np.empty((140, 120, 3))
        rgb[:] = WHITE
        rgb[cls.inside < 1.0] = IRIS
        rgb[cls.inside < 0.4] = PUPIL
        cls.known = cls.outline & (yy >= 50) & (yy <= 108)
        cls.rgb = rgb
        cls.oval = eyes.drawn_circle(rgb, cls.outline, cls.known)
        cls.layer, cls.kept = eyes.complete_iris(rgb, cls.outline, cls.known, cls.oval)

    def test_its_width_and_its_height_are_read(self):
        cx, cy, wide, tall = self.oval
        self.assertAlmostEqual(cx, self.CENTRE[0], delta=1.0)
        self.assertAlmostEqual(cy, self.CENTRE[1], delta=3.0)
        self.assertAlmostEqual(wide, self.WIDE, delta=1.0)
        self.assertAlmostEqual(tall, self.TALL, delta=2.5)

    def test_its_picture_is_the_oval_not_a_circle(self):
        self.assertEqual(self.kept, self.oval)
        self.assertTrue((self.layer[..., 3][self.outline & (self.inside < 0.93)] == 1).all())
        self.assertTrue((self.layer[..., 3][self.inside > 1.08] == 0).all())
        # under the circle of its width: still iris
        below = self.outline & (self.inside < 0.93) & (np.hypot(*(np.mgrid[0:140, 0:120] + 0.5 - np.array(self.CENTRE[::-1])[:, None, None])) > self.WIDE + 2)
        self.assertGreater(below.sum(), 100)

    def test_it_is_never_flatter_than_a_circle_nor_taller_than_the_limit(self):
        yy, xx = np.mgrid[0:140, 0:120] + 0.5
        for tall, wanted in ((20, 36), (80, eyes.OVAL_TALL * 36)):
            rgb = np.empty((140, 120, 3))
            rgb[:] = WHITE
            rgb[np.hypot((xx - 60) / 36, (yy - 70) / tall) < 1.0] = IRIS
            with self.subTest(tall=tall):
                self.assertAlmostEqual(eyes.drawn_circle(rgb, np.ones((140, 120), bool), yy >= 50)[3], wanted, delta=1.5)

    def test_both_grow_under_the_lid_in_their_own_shape(self):
        lid = np.zeros((140, 120), bool)
        lid[:20] = True  # the lid's edge 5 rows over the oval's top
        cx, cy, wide, tall = eyes.tucked({"left": ((60.0, 70.0, 36.0, 45.0), lid, self.outline)}, (36.0, 45.0))["left"]
        self.assertAlmostEqual(cy + tall, 115.0, delta=1e-6)  # the lowest point stays
        self.assertLess(cy - tall, 20.0)
        self.assertAlmostEqual(tall / wide, 45 / 36, delta=1e-6)


class TheIrisAsTheUserAsksForIt(unittest.TestCase):
    """The Face page's shape and size, over what was read (manifest `iris_shape`, `iris_size`)."""

    READ = (60.0, 70.0, 36.0, 45.0)

    def test_left_alone_it_is_as_read(self):
        self.assertEqual(eyes.shaped(self.READ), self.READ)
        self.assertIsNone(eyes.shaped(None, "circle", 1.2))

    def test_a_circle_is_as_tall_as_wide_on_the_same_lower_edge(self):
        cx, cy, wide, tall = eyes.shaped(self.READ, "circle")
        self.assertEqual((cx, wide, tall), (60.0, 36.0, 36.0))
        self.assertAlmostEqual(cy + tall, 115.0)

    def test_an_oval_is_taller_than_wide_whatever_was_read(self):
        self.assertEqual(eyes.shaped(self.READ, "oval"), self.READ)  # tall enough already
        cx, cy, wide, tall = eyes.shaped((60.0, 70.0, 36.0, 36.0), "oval")
        self.assertAlmostEqual(tall, eyes.OVAL_ASKED * 36.0)
        self.assertAlmostEqual(cy + tall, 106.0)

    def test_the_size_is_a_factor_on_both(self):
        cx, cy, wide, tall = eyes.shaped(self.READ, None, 1.2)
        self.assertAlmostEqual(wide, 43.2)
        self.assertAlmostEqual(tall, 54.0)
        self.assertAlmostEqual(cy + tall, 115.0)


class WhiteWhereTheIrisWas(unittest.TestCase):
    """The eye white is shaded from the lid down; the fill follows it row by row."""

    @classmethod
    def setUpClass(cls):
        yy, xx = np.mgrid[0:60, 0:140] + 0.5
        shade = (0.6 + 0.4 * yy / 60)[..., None]
        cls.clean = shade * np.array(WHITE)
        cls.hole = np.hypot(xx - 70, yy - 30) < 22
        rgb = cls.clean.copy()
        rgb[cls.hole] = PUPIL
        rgb[np.hypot(xx - 70, yy - 30) < 25] *= 0.5  # a dark rim just outside the hole
        cls.white = ~(np.hypot(xx - 70, yy - 30) < 26)
        cls.filled = eyes.whiten(rgb, cls.hole, cls.white)

    def test_the_fill_is_the_white_beside_it_row_by_row(self):
        self.assertLess(np.abs(self.filled[self.hole] - self.clean[self.hole]).max(), 0.05)

    def test_nothing_else_is_touched(self):
        far = ~self.hole
        np.testing.assert_allclose(self.filled[far], np.where(self.white[..., None], self.clean, self.filled)[far], atol=1e-6)


class OneWhiteBehindAModelledIris(unittest.TestCase):
    """The whole eye white is repainted, not only where the iris was: what the generator drew on it goes."""

    @classmethod
    def setUpClass(cls):
        yy, xx = np.mgrid[0:90, 0:160] + 0.5
        eye = ((xx - 80) / 60) ** 2 + ((yy - 45) / 25) ** 2 < 1
        rgb = np.empty(eye.shape + (3,))
        rgb[:] = SKIN
        rgb[eye] = WHITE
        rgb[eye & (yy > 55)] = (0.9, 0.7, 0.75)                           # a pink band along the lower lid
        rgb[eyes._dilate(eye, 4) & ~eye & (yy < 45)] = LASH
        cls.lash = eye & (yy < 24)                                        # the lashes' soft lower edge, inside the opening
        rgb[cls.lash] = LASH
        cls.hole = np.hypot(xx - 80, yy - 45) < 18
        cls.rim = (np.hypot(xx - 80, yy - 45) < 22) & ~cls.hole & eye     # the iris's rim, left beside the hole
        rgb[cls.rim] = PUPIL
        cls.strokes = eye & (xx > 118) & (xx < 124) & (yy > 30) & (yy < 52)  # a dark stroke on the white, clear of its edge
        rgb[cls.strokes] = LASH
        cls.corner = eye & (xx > 133) & (yy < 45)                         # the outline, on from the lashes round the outer corner
        rgb[cls.corner] = LASH
        rgb[cls.hole] = WHITE                                             # as eyes.whiten leaves it
        cls.eye, cls.rgb = eye, rgb
        saturation, value = eyes._saturation_value(rgb)
        white = eye & (saturation < 0.1) & (value > 0.8)
        cls.out, cls.painted = eyes.even_white(rgb, eye & ~cls.strokes, cls.hole, white)
        cls.inside = eyes._erode(eye, 8) & (yy > 34) & ~eyes._dilate(cls.corner, eyes.WHITE_SMOOTH + 1)  # clear of the soft edges

    def test_the_white_is_one_neutral_tone(self):
        saturation, value = eyes._saturation_value(self.out)
        self.assertLess(saturation[self.inside].max(), 0.12)
        self.assertGreater(value[self.inside].min(), 0.85)

    def test_the_rim_and_strokes_left_on_it_are_gone(self):
        for mask in (self.rim, self.strokes):
            self.assertTrue(self.painted[mask & self.inside].all())

    def test_the_lashes_lower_edge_stays(self):
        kept = self.lash & ~self.hole
        self.assertGreater(kept.sum(), 20)
        self.assertFalse(self.painted[kept].any())
        np.testing.assert_allclose(self.out[kept], self.rgb[kept], atol=1e-6)

    def test_the_outline_round_the_corner_stays(self):
        kept = self.corner & ~self.painted
        self.assertGreater(kept.sum(), 0.8 * self.corner.sum())
        np.testing.assert_allclose(self.out[kept], self.rgb[kept], atol=1e-6)

    def test_the_skin_round_the_eye_is_as_it_was(self):
        far = ~eyes._dilate(self.eye, 6)
        np.testing.assert_allclose(self.out[far], self.rgb[far], atol=1e-6)
