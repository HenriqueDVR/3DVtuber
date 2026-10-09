"""A character whose body, head, coat and legs are one surface (a tester's pirate), and ears with no piece of their own."""
import unittest

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

from vtuber_maker.pipeline import cloth, face_texture, meshdata, parts, rig, roles


def armature():
    joints = {"hips": (0, 0, .8), "neck": (0, 0, 1.3), "head": (0, 0, 1.4)}
    for side, sign in (("left", 1), ("right", -1)):
        joints.update({f"{side}_upper_arm": (sign * .15, 0, 1.25), f"{side}_lower_arm": (sign * .4, 0, 1.25),
                       f"{side}_hand": (sign * .65, 0, 1.25), f"{side}_upper_leg": (sign * .07, 0, .75),
                       f"{side}_lower_leg": (sign * .07, 0, .4), f"{side}_foot": (sign * .07, 0, .08)})
    return rig.build_armature(joints), {k: np.array(v, dtype=float) for k, v in joints.items()}


def strip(points, faces, xs, zs, y):
    """A sheet in the plane y; returns its vertex indices, rows from the first of `zs`."""
    start = len(points)
    points.extend((x, y, z) for z in zs for x in xs)
    faces.extend((start + j * len(xs) + i, start + j * len(xs) + i + 1, start + (j + 1) * len(xs) + i + 1,
                  start + (j + 1) * len(xs) + i) for j in range(len(zs) - 1) for i in range(len(xs) - 1))
    return list(range(start, len(points)))


def mesh(points, faces, weights, arm):
    data = bpy.data.meshes.new("Fused")
    data.from_pydata(points, [], faces)
    obj = bpy.data.objects.new("Fused", data)
    bpy.context.scene.collection.objects.link(obj)
    for name, members in weights.items():
        for index, weight in members:
            group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
            group.add([index], weight, "REPLACE")
    obj.modifiers.new("Armature", "ARMATURE").object = arm
    bpy.context.view_layer.update()
    return obj


def weights_of(obj, index):
    return {obj.vertex_groups[g.group].name: round(g.weight, 3) for g in obj.data.vertices[index].groups if g.weight > 1e-6}


class FusedBody(unittest.TestCase):
    def setUp(self):
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        self.arm, self.joints = armature()

    def test_legs_in_one_surface_with_a_coat_are_not_swung_with_it(self):
        # the pirate: trousers and boots went onto his coat's chains, 100% at the thighs
        points, faces = [], []
        zs = list(np.arange(.82, .09, -.04))
        left = strip(points, faces, [.05, .07, .09], zs, 0.0)
        right = strip(points, faces, [-.09, -.07, -.05], zs, 0.0)
        coat = strip(points, faces, list(np.arange(-.3, .301, .03)), zs, .25)
        # joined only above the hips: a yoke from each top row to one point at the waist
        apex = len(points)
        points.append((0, .1, .9))
        faces += [(apex, part[0], part[1]) for part in (left, right, coat)]
        obj = mesh(points, faces, {"Hips": [(i, 1.0) for i in range(len(points))]}, self.arm)
        chains = cloth.build_cloth_chains(obj, self.arm, self.joints)
        self.assertTrue(chains)
        on_chain = lambda i: any(name.startswith("Cloth") for name in weights_of(obj, i))  # noqa: E731
        below = lambda part: [i for i in part if points[i][2] < .7]  # noqa: E731
        self.assertFalse([i for i in below(left) + below(right) if on_chain(i)])
        self.assertTrue(all(on_chain(i) for i in below(coat)))

    def test_a_coat_by_itself_swings_as_before(self):
        points, faces = [], []
        coat = strip(points, faces, list(np.arange(-.3, .301, .03)), list(np.arange(.82, .09, -.04)), .25)
        obj = mesh(points, faces, {"Hips": [(i, 1.0) for i in range(len(points))]}, self.arm)
        self.assertTrue(cloth.build_cloth_chains(obj, self.arm, self.joints))
        self.assertTrue(all(any(n.startswith("Cloth") for n in weights_of(obj, i)) for i in coat if points[i][2] < .7))

    def test_the_rest_of_the_head_follows_the_head(self):
        # the pirate's beard, jaw and hair, in his body's piece: 49 to 86% on the head by bone heat
        points, faces = [], []
        column = strip(points, faces, [-.08, 0, .08], list(np.arange(1.2, 1.601, .02)), -.1)
        hand = strip(points, faces, [.6, .62], [1.5, 1.52], -.1)      # held up beside the head
        hair = strip(points, faces, [-.02, .02], [1.5, 1.52], .1)
        obj = mesh(points, faces, {"Head": [(i, .5) for i in column + hair], "Neck": [(i, .5) for i in column + hair],
                                   "LeftHand": [(i, 1.0) for i in hand]}, self.arm)
        skip = np.zeros(len(points), dtype=bool)
        skip[hair] = True
        changed = rig.follow_head_column(obj, self.arm, skip=skip)
        self.assertGreater(changed, 0)
        for i in column:
            z = points[i][2]
            if z > 1.4 + rig.COLUMN_BLEND / 2 + 1e-6:
                self.assertEqual(weights_of(obj, i), {"Head": 1.0}, z)
            elif z < 1.4 - rig.COLUMN_BLEND / 2 - 1e-6:
                self.assertEqual(weights_of(obj, i), {"Head": .5, "Neck": .5}, z)
        self.assertEqual(weights_of(obj, hand[0]), {"LeftHand": 1.0})
        self.assertEqual(weights_of(obj, hair[0]), {"Head": .5, "Neck": .5})

    def test_a_body_in_one_surface_with_its_head_is_not_headwear(self):
        # the pirate again: most of his body's vertices are his head's, so the piece read as mostly
        # on the head and all of it went there: arms, coat and legs stood still
        points, faces = [], []
        head = strip(points, faces, list(np.arange(-.08, .081, .01)), list(np.arange(1.32, 1.601, .01)), -.1)
        body = strip(points, faces, [-.1, 0, .1], list(np.arange(.3, 1.301, .1)), -.1)
        faces.append((head[0], head[1], body[-1]))   # one surface
        hat = strip(points, faces, [-.1, 0, .1], [1.62, 1.66], -.1)
        obj = mesh(points, faces, {"Head": [(i, 1.0) for i in head + hat], "Hips": [(i, 1.0) for i in body]}, self.arm)
        rigid = rig.make_head_rigid(obj, self.arm)
        self.assertFalse(rigid[body].any())
        self.assertEqual(weights_of(obj, body[0]), {"Hips": 1.0})
        self.assertTrue(rigid[hat].all())

    def test_the_skeleton_stands_on_the_line_the_body_mirrors_about(self):
        from vtuber_maker.pipeline import landmarks
        rng = np.random.default_rng(0)
        torso = np.column_stack([rng.uniform(-.15, .15, 6000), np.zeros(6000), rng.uniform(.8, 1.4, 6000)])
        arms = np.column_stack([rng.uniform(-.8, .8, 3000), np.zeros(3000), rng.uniform(1.25, 1.33, 3000)])
        legs = np.concatenate([np.column_stack([s * rng.uniform(.04, .14, 2000), np.zeros(2000), rng.uniform(0, .8, 2000)])
                               for s in (-1, 1)])
        body = np.concatenate([torso, arms, legs])
        self.assertAlmostEqual(landmarks.centre_line(body), float(np.median(body[:, 0])), places=6)
        # a tail swept out to one side, as dense as the pirate's: the median goes with it, the line doesn't
        tail = np.column_stack([rng.uniform(-.45, -.1, 3500), np.zeros(3500), rng.uniform(0, .6, 3500)])
        tailed = np.concatenate([body, tail])
        self.assertLess(float(np.median(tailed[:, 0])), -.025)
        self.assertLess(abs(landmarks.centre_line(tailed)), .012)

    def test_a_part_marked_on_the_source_is_found_by_place_after_a_cut_renumbers(self):
        before = np.array([1, 1, 1, 1])
        was = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        now = was[::-1].copy()
        carried = roles.carry({"ear": np.array([False, False, True, True])}, before, before, now, was)
        self.assertEqual(carried["ear"].tolist(), [True, True, False, False])


class Ears(unittest.TestCase):
    """Ears or horns standing over the crown, in whatever piece."""

    def setUp(self):
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        self.joints = {"head": np.array([0, 0, 1.4]), "neck": np.array([0, 0, 1.3])}

    def head(self, ears, fused=False):
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=16, radius=.1, matrix=Matrix.Translation((0, 0, 1.5)))
        bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=.3, radius2=.3, depth=1.3,
                              matrix=Matrix.Translation((0, 0, .65)))  # a body under it: the model's height
        for x in ears:
            # a cone from inside the head's side up past its crown
            bmesh.ops.create_cone(bm, cap_ends=False, segments=10, radius1=.03, radius2=.002, depth=.12,
                                  matrix=Matrix.Translation((x, 0, 1.6)))
        if fused:
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=.012)
        data = bpy.data.meshes.new("Head")
        bm.to_mesh(data)
        bm.free()
        obj = bpy.data.objects.new("Head", data)
        bpy.context.scene.collection.objects.link(obj)
        return obj, meshdata.world_vertices(obj)

    def test_a_pair_is_found_one_on_each_side(self):
        obj, verts = self.head([-.07, .07])
        found = parts.ears(obj, verts, self.joints)
        self.assertEqual(set(found), {"left", "right"})
        self.assertGreater(float(verts[found["left"], 0].min()), 0)   # character left is +X
        self.assertLess(float(verts[found["right"], 0].max()), 0)
        for mask in found.values():
            self.assertGreater(float(np.ptp(verts[mask, 2])), .1)     # the whole cone, not its tip
            self.assertGreater(float(verts[mask, 2].max()), 1.65)

    def test_one_alone_is_not_a_pair(self):
        obj, verts = self.head([.07])
        self.assertEqual(parts.ears(obj, verts, self.joints), {})

    def test_a_head_without_any_has_none(self):
        obj, verts = self.head([])
        self.assertEqual(parts.ears(obj, verts, self.joints), {})

    def test_what_is_named_as_something_else_is_left_out(self):
        obj, verts = self.head([-.07, .07])
        self.assertEqual(parts.ears(obj, verts, self.joints, skip=verts[:, 2] > 1.601), {})

    def test_an_ear_swings_on_one_chain_from_its_foot(self):
        obj, verts = self.head([-.07, .07])
        arm, joints = armature()
        obj.modifiers.new("Armature", "ARMATURE").object = arm
        obj.vertex_groups.new(name="Head").add(list(range(len(verts))), 1.0, "REPLACE")
        mask = parts.ears(obj, verts, self.joints)["left"]
        names = cloth.build_appendage_chain(obj, arm, joints, mask, 0, "Head")
        self.assertEqual(len(names), 1)
        bones = arm.data.bones
        root, tip = bones[names[0][0]].head_local, bones[names[0][-1]].tail_local
        self.assertLess(root.z, float(verts[mask, 2].min()) + .03)
        self.assertGreater(tip.z, float(verts[mask, 2].max()) - .03)


class LipsOnAnotherSurfaceThanTheEyes(unittest.TestCase):
    """The pirate: eyes and brow in the hat's surface, lips and jaw in the body's."""

    def found(self, lips_part):
        # part 0: the brow, with a layer of it running down behind the mouth; part 1: the lower face, its
        # nape and its chest; part 2: hair in front of the chin
        brow = [(x, -.10, z) for x in (-.03, 0, .03) for z in (1.50, 1.52)] + [(x, -.08, 1.44) for x in (-.01, 0, .01)]
        lower = ([(x, -.10, 1.44) for x in (-.02, -.01, 0, .01, .02)] + [(0, -.10, 1.40), (0, .08, 1.40), (0, -.10, 1.20)])
        hair = [(0, -.11, 1.40)]
        verts = np.array(brow + lower + hair, dtype=float)
        labels = np.array([0] * len(brow) + [lips_part] * len(lower) + [2] * len(hair))
        hair_mask = labels == 2
        return verts, parts.MeshParts(labels, hair_mask, 0, {}, -1, np.zeros(3), labels == 0)

    def grown(self, lips_part=1):
        verts, found = self.found(lips_part)
        return verts, found, parts.with_marked_mouth(found, verts, (0, -.10, 1.44), .04, 1.30, 0.0)

    def test_the_surface_the_lips_are_on_joins_the_face(self):
        verts, found, grown = self.grown()
        lips = (found.labels == 1) & (verts[:, 2] > 1.43)
        self.assertFalse(found.face_area[lips].any())
        self.assertTrue(grown.face_area[lips].all())
        self.assertTrue(grown.face_area[found.face_area].all())

    def test_only_its_head_end_in_front_and_never_hair(self):
        verts, found, grown = self.grown()
        added = grown.face_area & ~found.face_area
        self.assertGreater(float(verts[added, 2].min()), 1.30)   # not the chest
        self.assertLess(float(verts[added, 1].max()), 0.0)       # not the nape
        self.assertFalse(grown.face_area[found.hair].any())

    def test_a_face_that_holds_its_own_lips_is_left_as_it_is(self):
        verts, found, grown = self.grown(lips_part=0)
        self.assertIs(grown, found)


class AHeadCalledSolidForItsHat(unittest.TestCase):
    """Only a head of one colour round its face patch is repainted to continue the patch."""

    def test_one_colour_round_the_patch_is_a_solid_head(self):
        rng = np.random.default_rng(0)
        edge = np.array([.55, .85, .95]) + rng.normal(0, .03, (40, 3))
        self.assertLess(face_texture._edge_spread(edge), face_texture.EDGE_AGREES)

    def test_skin_hair_and_hat_round_the_patch_are_not(self):
        edge = np.array([[.80, .55, .40]] * 14 + [[.25, .15, .10]] * 14 + [[.45, .30, .10]] * 14)
        self.assertGreater(face_texture._edge_spread(edge), face_texture.EDGE_AGREES)


if __name__ == "__main__":
    unittest.main()
