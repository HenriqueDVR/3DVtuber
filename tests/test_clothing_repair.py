"""Clothing repairs keep skirt seams on the body and shoulder coverings on the arms."""
import unittest

import bpy
import numpy as np
from mathutils import Matrix, Vector

from vtuber_maker.pipeline import loose, rig, tpose
from test_fingers import deformed


class ClothingRepair(unittest.TestCase):
    def setUp(self):
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        joints = {"hips": (0, 0, .8), "neck": (0, 0, 1.3), "head": (0, 0, 1.4)}
        for side, sign in (("left", 1), ("right", -1)):
            joints.update({f"{side}_upper_arm": (sign * .15, 0, 1.2),
                           f"{side}_lower_arm": (sign * .23, 0, 1.0),
                           f"{side}_hand": (sign * .30, 0, .8),
                           f"{side}_upper_leg": (sign * .07, 0, .75),
                           f"{side}_lower_leg": (sign * .07, 0, .4),
                           f"{side}_foot": (sign * .07, 0, .08)})
        self.arm = rig.build_armature(joints)

    def mesh(self, points, weights, faces=()):
        data = bpy.data.meshes.new("Clothing")
        data.from_pydata(points, [], faces)
        obj = bpy.data.objects.new("Clothing", data)
        bpy.context.scene.collection.objects.link(obj)
        for i, row in enumerate(weights):
            for name, w in row.items():
                group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
                group.add([i], w, "REPLACE")
        obj.modifiers.new("Armature", "ARMATURE").object = self.arm
        obj.shape_key_add(name="Basis")
        key = obj.shape_key_add(name="Expression")
        key.data[0].co.y += .001
        key.value = 0.0  # 5.x adds new keys at 1.0; the repairs must carry an unused key along
        bpy.context.view_layer.update()
        return obj

    def test_torn_seam_is_cut_between_faces_so_no_edge_joins_the_two_limbs(self):
        # a strip fused across the thigh and the forearm, as a glove resting on a skirt is: splitting
        # the edges that cross the join once left every one of them joining the two sides
        points = [(x * .02, 0, z * .02 + .9) for z in range(3) for x in range(6)]
        weights = [{"LeftUpperLeg": 1} if x < 3 else {"LeftLowerArm": 1} for z in range(3) for x in range(6)]
        faces = [(z * 6 + x, z * 6 + x + 1, (z + 1) * 6 + x + 1, (z + 1) * 6 + x) for z in range(2) for x in range(5)]
        obj = self.mesh(points, weights, faces)
        self.assertGreater(loose.split_torn_seams(obj, self.arm), 0)
        leader = loose._leading_bones(obj, len(obj.data.vertices))
        names = {g.index: g.name for g in obj.vertex_groups}
        for edge in obj.data.edges:
            self.assertEqual(len({names[leader[i]] for i in edge.vertices}), 1)
        for face in obj.data.polygons:
            self.assertEqual(len({names[leader[i]] for i in face.vertices}), 1)
        for vertex in obj.data.vertices:
            self.assertAlmostEqual(sum(g.weight for g in vertex.groups), 1, places=6)
        self.assertEqual(len(obj.data.polygons), 10)

    def test_bones_a_limb_joint_apart_are_torn_even_on_one_chain(self):
        # the hips are the forearm's ancestor, but the shoulder and elbow lie between them: a
        # hand resting on a skirt moves 40 cm away from it when the arm lifts
        torn = lambda *names: loose._torn(self.arm, list(names))
        self.assertTrue(torn("LeftLowerArm", "Hips"))
        self.assertTrue(torn("LeftHand", "Spine"))
        self.assertTrue(torn("LeftLowerLeg", "Hips"))
        self.assertTrue(torn("LeftLowerArm", "RightLowerArm"))
        self.assertFalse(torn("LeftUpperArm", "Chest"))  # the armpit: one joint, the skin stretches
        self.assertFalse(torn("LeftUpperArm", "LeftLowerArm", "LeftHand"))
        self.assertFalse(torn("Hips", "LeftUpperLeg", "LeftLowerLeg"))  # a panel down to the knee
        self.assertFalse(torn("Hips", "LeftUpperLeg", "RightUpperLeg"))  # a skirt over both thighs
        self.assertFalse(torn("Spine", "Hips", "Chest"))

    def grid(self, x0, z0, columns, rows, step=.01):
        return [(x0 + i * step, -.03, z0 + j * step) for j in range(rows) for i in range(columns)]

    def quads(self, first, columns, rows):
        return [(first + j * columns + i, first + j * columns + i + 1, first + (j + 1) * columns + i + 1,
                 first + (j + 1) * columns + i) for j in range(rows - 1) for i in range(columns - 1)]

    def test_glove_resting_on_skirt_is_not_welded_to_it(self):
        # The sleeve cut leaves the glove and the skirt as two surfaces meeting along a seam;
        # joining them there drew a 3 mm edge out to 77 cm when the arms rose (model A, A-pose).
        # A sleeve against the chest at the armpit still joins: one joint apart, it stretches.
        skirt, glove = self.grid(.26, .78, 3, 3), self.grid(.28, .78, 3, 3)
        chest, sleeve = self.grid(.10, 1.12, 3, 3), self.grid(.12, 1.12, 3, 3)
        points = skirt + glove + chest + sleeve
        weights = ([{"Hips": 1}] * 9 + [{"LeftLowerArm": 1}] * 9 + [{"Chest": 1}] * 9
                   + [{"LeftUpperArm": .6, "Chest": .4}] * 9)
        faces = [f for first in (0, 9, 18, 27) for f in self.quads(first, 3, 3)]
        obj = self.mesh(points, weights, faces)
        loose.join_touching_surfaces(obj, self.arm)
        self.assertEqual(len(obj.data.vertices), 36 - 3)  # only the armpit seam joined
        edges = np.array([e.vertices[:] for e in obj.data.edges])
        before = deformed(obj)
        tpose._point_arm(self.arm, "LeftUpperArm", "LeftHand", Vector((.5, 0, 1)).normalized())
        after = deformed(obj)
        growth = np.linalg.norm(after[edges[:, 0]] - after[edges[:, 1]], axis=1) - np.linalg.norm(before[edges[:, 0]] - before[edges[:, 1]], axis=1)
        self.assertLess(growth[(edges < 18).all(axis=1)].max(), .01)  # skirt and glove, not the armpit

    def test_a_patch_folded_on_a_line_until_a_shape_key_opens_it_is_not_welded(self):
        # the mouth patch: every row on the mouth's line at rest, spread into an opening by its keys
        chest, sleeve = self.grid(.10, 1.12, 3, 3), self.grid(.12, 1.12, 3, 3)
        folded = [(x, -.03, 1.3) for _row in range(3) for x in (0.0, .01, .02)]
        obj = self.mesh(chest + sleeve + folded, [{"Chest": 1}] * 9 + [{"LeftUpperArm": .6, "Chest": .4}] * 9 + [{"Head": 1}] * 9,
                        [f for first in (0, 9, 18) for f in self.quads(first, 3, 3)])
        opened = obj.shape_key_add(name="aa")
        for row in range(3):
            for column in range(3):
                opened.data[18 + row * 3 + column].co.z = 1.3 - .005 * row
        opened.value = 0.0
        loose.join_touching_surfaces(obj, self.arm)
        self.assertEqual(len(obj.data.vertices), 27 - 3)  # the armpit seam joined, the patch whole
        heights = sorted({round(d.co.z, 4) for d in obj.data.shape_keys.key_blocks["aa"].data[-9:]})
        self.assertEqual(heights, [1.29, 1.295, 1.3])

    def test_far_side_of_a_cut_is_not_a_stray_patch_on_the_arm(self):
        # A thigh strip meeting the glove only along a cut: walking the surface across the cut
        # read it as a patch on the arm and handed it the forearm's weights (17 vertices on model A).
        glove, thigh = self.grid(.28, .76, 4, 4), self.grid(.25, .76, 4, 2)
        obj = self.mesh(glove + thigh, [{"LeftLowerArm": 1}] * 16 + [{"LeftUpperLeg": 1}] * 8,
                        self.quads(0, 4, 4) + self.quads(16, 4, 2))
        self.assertEqual(loose.fix_weight_islands(obj, self.arm), [])
        for i in range(24):
            self.assertEqual({obj.vertex_groups[g.group].name: g.weight for g in obj.data.vertices[i].groups},
                             {"LeftLowerArm" if i < 16 else "LeftUpperLeg": 1})

    def test_small_piece_torn_between_forearm_and_hips_rides_the_bone_holding_most(self):
        # a cuff trim bone heat gave partly to the hips it rests on: putting it on the ancestor
        # both hang from (the hips) would leave it behind when the arm lifts
        body = self.grid(-.2, .3, 20, 10)
        trim = [(.30, -.04, .82), (.31, -.04, .82), (.30, -.04, .83)]
        obj = self.mesh(body + trim, [{"Hips": 1}] * 200 + [{"LeftLowerArm": .7, "Hips": .3}] * 3,
                        self.quads(0, 20, 10) + [(200, 201, 202)])
        loose.rigidify_props(obj, self.arm)
        for i in (200, 201, 202):
            self.assertEqual({obj.vertex_groups[g.group].name: g.weight for g in obj.data.vertices[i].groups},
                             {"LeftLowerArm": 1})

    def test_the_tidy_up_guard_measures_a_tear_and_puts_the_pose_back(self):
        obj = self.mesh([(.30, -.03, .80), (.29, -.03, .80), (.29, -.03, .81)],
                        [{"LeftLowerArm": 1}, {"Hips": 1}, {"Hips": 1}], [(0, 1, 2)])
        worst = loose.worst_stretch([obj], self.arm)
        self.assertGreater(worst["arms raised"], loose.TEARS)
        self.assertLess(worst["head turn"], 1e-6)
        self.assertIn("arms raised", [pose for pose, _, _ in loose.tore({"arms raised": .01}, worst)])
        self.assertEqual(loose.tore(worst, worst), [])
        for bone in self.arm.pose.bones:
            self.assertEqual(bone.matrix_basis, Matrix.Identity(4))

    def test_a_hanging_hand_reaches_what_lies_inside_its_forearm(self):
        # the hip under each wrist, and the outside of the right forearm
        points = np.array([(-.14, -.03, .90), (.14, -.03, .90), (-.33, 0, .85)])
        reach = loose.hand_reach(self.arm, points)
        self.assertGreater(reach[0], .99)
        self.assertGreater(reach[1], .99)
        self.assertEqual(reach[2], 0)
        # arms held out reach nothing: a T-pose model's hands rest on no skirt
        level = {"hips": (0, 0, .8), "neck": (0, 0, 1.3), "head": (0, 0, 1.4)}
        for side, sign in (("left", 1), ("right", -1)):
            level.update({f"{side}_upper_arm": (sign * .15, 0, 1.2), f"{side}_lower_arm": (sign * .37, 0, 1.2),
                          f"{side}_hand": (sign * .58, 0, 1.2), f"{side}_upper_leg": (sign * .07, 0, .75),
                          f"{side}_lower_leg": (sign * .07, 0, .4), f"{side}_foot": (sign * .07, 0, .08)})
        self.assertEqual(float(loose.hand_reach(rig.build_armature(level), points).max()), 0.0)

    def test_cut_skirt_copies_rejoin_without_dragging_sleeve_or_hand(self):
        points = [(-.14, -.03, .90), (-.14, -.03, .90), (-.16, -.03, .92),
                  (-.16, -.03, .92), (-.29, 0, .90), (-.30, 0, .78), (.14, -.03, .90)]
        obj = self.mesh(points, [{"Hips": 1}, {"RightLowerArm": 1}, {"Hips": 1},
                                {"RightLowerArm": 1}, {"RightLowerArm": 1}, {"RightHand": 1},
                                {"LeftLowerArm": 1}], [(0, 2, 6), (1, 3, 4)])
        before = [np.array([p.co[:] for p in key.data]) for key in obj.data.shape_keys.key_blocks]
        faces = [tuple(p.vertices) for p in obj.data.polygons]
        loose.restore_skirt_weights(obj, self.arm, np.array([True, True, True, True, True, False, True]))
        for side, sign in (("Left", 1), ("Right", -1)):
            tpose._point_arm(self.arm, side + "UpperArm", side + "Hand", Vector((sign, 0, 0)))
        moved = deformed(obj)
        np.testing.assert_allclose(moved[[0, 1, 2, 3, 6]], np.array(points)[[0, 1, 2, 3, 6]], atol=1e-6)
        self.assertGreater(np.linalg.norm(moved[4] - points[4]), .1)
        self.assertGreater(np.linalg.norm(moved[5] - points[5]), .1)
        self.assertEqual(faces, [tuple(p.vertices) for p in obj.data.polygons])
        for old, key in zip(before, obj.data.shape_keys.key_blocks):
            np.testing.assert_array_equal(old, [p.co[:] for p in key.data])
        self.assert_normalized(obj)

    def test_separates_fused_skirt_sleeve_contact_before_baking(self):
        points = [(-.14, -.03, .90), (-.14, -.03, .92), (-.29, -.03, .90),
                  (-.29, -.03, .92)]
        obj = self.mesh(points, [{"Hips": 1}, {"Hips": 1}, {"RightLowerArm": 1},
                                {"RightLowerArm": 1}], [(0, 2, 3, 1)])
        # The same fused contact that a weight-only repair stretches into a ribbon.
        origin = loose.separate_skirt_contact(obj, self.arm, np.ones(4, dtype=bool))
        self.assertEqual(len(origin), len(obj.data.vertices))
        edges = np.array([e.vertices[:] for e in obj.data.edges])
        before = deformed(obj)
        tpose._point_arm(self.arm, "RightUpperArm", "RightHand", Vector((-1, 0, 0)))
        after = deformed(obj)
        growth = np.linalg.norm(after[edges[:, 0]] - after[edges[:, 1]], axis=1) - np.linalg.norm(before[edges[:, 0]] - before[edges[:, 1]], axis=1)
        self.assertLess(growth.max(), .01)
        self.assertEqual(set(obj.data.shape_keys.key_blocks.keys()), {"Basis", "Expression"})
        self.assert_normalized(obj)

    def test_mantle_follows_both_upper_arms_but_not_collar_or_lower_cape(self):
        points = [(-.25, -.02, 1.15), (.25, -.02, 1.15), (0, 0, 1.25),
                  (.23, .05, .60), (.30, 0, .78), (.25, -.02, 1.15)]
        obj = self.mesh(points, [{"Chest": 1}] * 3 + [{"Hips": 1}, {"LeftHand": 1}, {"Head": 1}])
        loose.follow_shoulder_cloth(obj, self.arm, np.array([True, True, True, True, False, False]))
        for side, sign in (("Left", 1), ("Right", -1)):
            tpose._point_arm(self.arm, side + "UpperArm", side + "Hand", Vector((sign, 0, 0)))
        moved = deformed(obj)
        for i in (0, 1):
            self.assertGreater(np.linalg.norm(moved[i] - points[i]), .06)
        np.testing.assert_allclose(moved[[2, 3, 5]], np.array(points)[[2, 3, 5]], atol=1e-6)
        self.assertGreater(np.linalg.norm(moved[4] - points[4]), .1)
        self.assert_normalized(obj)

    def test_on_arms_held_out_the_mantle_over_the_arm_follows_it_wholly(self):
        # a T-pose copy of model B: "above the elbow" was read by height, and an arm held out has its
        # elbow at the shoulder's height: what lay 6 cm over the arm took half its weight and stood
        # out as a wing at each shoulder once the arms came down
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        joints = {"hips": (0, 0, .8), "neck": (0, 0, 1.3), "head": (0, 0, 1.4)}
        for side, sign in (("left", 1), ("right", -1)):
            joints.update({f"{side}_upper_arm": (sign * .15, 0, 1.25), f"{side}_lower_arm": (sign * .40, 0, 1.25),
                           f"{side}_hand": (sign * .62, 0, 1.25), f"{side}_upper_leg": (sign * .07, 0, .75),
                           f"{side}_lower_leg": (sign * .07, 0, .4), f"{side}_foot": (sign * .07, 0, .08)})
        self.arm = rig.build_armature(joints)
        # over the left upper arm, just under it, far under it (the hem), the collar, past the elbow;
        # and the arm's own surface where the first lies, to say where the arm carries it
        points = [(.26, 0, 1.31), (.26, 0, 1.22), (.26, 0, 1.11), (0, 0, 1.33), (.46, 0, 1.31), (.26, 0, 1.31)]
        obj = self.mesh(points, [{"Chest": 1}] * 5 + [{"LeftUpperArm": 1}])
        loose.follow_shoulder_cloth(obj, self.arm, np.array([True] * 5 + [False]))
        tpose._point_arm(self.arm, "LeftUpperArm", "LeftHand", Vector((.25, 0, -1)))
        moved = deformed(obj)
        np.testing.assert_allclose(moved[0], moved[5], atol=1e-4)
        self.assertGreater(np.linalg.norm(moved[1] - points[1]), .06)
        np.testing.assert_allclose(moved[[2, 3, 4]], np.array(points)[[2, 3, 4]], atol=1e-6)
        self.assert_normalized(obj)

    def test_hanging_sleeve_rides_upper_arm_and_keeps_other_groups(self):
        # A wide sleeve hanging beside the ribs, skinned only to the torso.
        points = [(.16, -.05, 1.15), (.25, -.05, .95), (.30, .02, .82)]
        obj = self.mesh(points, [{"Spine": .5, "Chest": .5}] * 3)
        obj.vertex_groups.new(name="Mask").add([0, 1, 2], .7, "REPLACE")
        loose.follow_bone(obj, self.arm, [0, 1, 2], "LeftUpperArm")
        tpose._point_arm(self.arm, "LeftUpperArm", "LeftHand", Vector((.5, 0, 1)).normalized())
        bone = self.arm.pose.bones["LeftUpperArm"]
        expected = [bone.matrix @ bone.bone.matrix_local.inverted() @ Vector(p) for p in points]
        np.testing.assert_allclose(deformed(obj), expected, atol=1e-6)
        self.assertAlmostEqual(obj.vertex_groups["Mask"].weight(1), .7, places=6)

    def test_detail_borrows_anchor_skinning_without_changing_anchor(self):
        obj = self.mesh([(-.28, -.04, .85), (-.281, -.04, .85), (-.282, -.04, .85)],
                        [{"Hips": 1}, {"RightLowerArm": .8, "RightHand": .2}, {"Hips": 1}])
        loose.attach_clothing_detail(obj, self.arm, [0, 2], [1])
        tpose._point_arm(self.arm, "RightUpperArm", "RightHand", Vector((-1, 0, 0)))
        moved = deformed(obj)
        self.assertLess(np.linalg.norm(moved[0] - moved[1]), .002)
        self.assertLess(np.linalg.norm(moved[2] - moved[1]), .002)
        self.assert_normalized(obj)

    def test_inner_forearm_connected_to_upper_sleeve_is_not_skirt(self):
        # Sleeve and skirt touch at vertex 1/3 but have separate skinning/topology.
        points = [(-.20, -.03, 1.10), (-.24, -.03, .92), (-.27, -.03, .93),
                  (-.24, -.03, .92), (-.14, -.03, .89), (-.12, -.03, .88)]
        obj = self.mesh(points, [{"RightUpperArm": 1}, {"RightLowerArm": 1},
                                {"RightLowerArm": 1}, {"Hips": 1}, {"Hips": 1}, {"Hips": 1}],
                        [(0, 1, 2), (3, 4, 5)])
        loose.restore_skirt_weights(obj, self.arm, np.ones(len(points), dtype=bool))
        tpose._point_arm(self.arm, "RightUpperArm", "RightHand", Vector((-1, 0, 0)))
        moved = deformed(obj)
        bone = self.arm.pose.bones["RightLowerArm"]
        expected = bone.matrix @ bone.bone.matrix_local.inverted() @ Vector(points[1])
        np.testing.assert_allclose(moved[1], expected, atol=1e-6)
        np.testing.assert_allclose(moved[[3, 4, 5]], np.array(points)[[3, 4, 5]], atol=1e-6)

    def test_history_transfer_restores_whole_selected_covering_not_just_outer_tip(self):
        points = [(.16, 0, 1.18), (.24, 0, 1.12), (.25, .02, .6)]
        target = self.mesh(points, [{"Chest": 1}, {"LeftUpperArm": 1}, {"Hips": 1}])
        donor = self.mesh(points, [{"LeftUpperArm": .8, "Chest": .2},
                                  {"LeftUpperArm": 1}, {"Chest": 1}])
        loose.transfer_clothing_weights(target, self.arm, donor, [0, 1], [0, 1])
        tpose._point_arm(self.arm, "LeftUpperArm", "LeftHand", Vector((1, 0, 0)))
        np.testing.assert_allclose(deformed(target)[:2], deformed(donor)[:2], atol=1e-6)
        np.testing.assert_allclose(deformed(target)[2], points[2], atol=1e-6)
        self.assert_normalized(target)

    def test_history_transfer_rejects_unmatched_geometry_before_editing(self):
        target = self.mesh([(.2, 0, 1.1)], [{"Chest": 1}])
        donor = self.mesh([(.5, 0, 1.1)], [{"LeftUpperArm": 1}])
        with self.assertRaises(ValueError):
            loose.transfer_clothing_weights(target, self.arm, donor, [0], [0])
        self.assertEqual(target.vertex_groups["Chest"].weight(0), 1)

    def assert_normalized(self, obj):
        for vertex in obj.data.vertices:
            self.assertAlmostEqual(sum(g.weight for g in vertex.groups), 1, places=6)


if __name__ == "__main__":
    unittest.main()
