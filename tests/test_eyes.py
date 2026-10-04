"""Eyes that follow face tracking: a moving iris under see-through lids, or a whole eye sliding."""

import unittest

import bmesh
import bpy
import numpy as np

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


if __name__ == "__main__":
    unittest.main()
