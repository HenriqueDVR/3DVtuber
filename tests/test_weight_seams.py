"""Closing shoulder weight seams must not weld touching hands to clothing."""
import unittest

import bpy
import numpy as np
from mathutils import Vector

from vtuber_maker.pipeline import loose, rig, tpose
from test_fingers import deformed


class ShoulderWeightSeams(unittest.TestCase):
    def setUp(self):
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        data = bpy.data.armatures.new("Rig")
        self.arm = bpy.data.objects.new("Rig", data)
        bpy.context.scene.collection.objects.link(self.arm)
        rig.make_active(self.arm)
        bpy.ops.object.mode_set(mode="EDIT")
        chest = data.edit_bones.new("Chest")
        chest.head, chest.tail = (0, 0, .8), (0, 0, 1.1)
        upper = data.edit_bones.new("LeftUpperArm")
        upper.head, upper.tail, upper.parent = (.15, 0, 1.1), (.2, 0, .9), chest
        lower = data.edit_bones.new("LeftLowerArm")
        lower.head, lower.tail, lower.parent = upper.tail, (.25, 0, .7), upper
        hand = data.edit_bones.new("LeftHand")
        hand.head, hand.tail, hand.parent = lower.tail, (.26, 0, .65), lower
        bpy.ops.object.mode_set(mode="OBJECT")
        # Two surfaces touching at the shoulder, plus another contact down at the wrist.
        vertices = [(.14, 0, 1.05), (.14, 0, 1.05), (.15, .02, 1.04), (.15, .02, 1.04),
                    (.09, 0, 1.08), (.20, 0, 1.03), (.25, 0, .7), (.25, 0, .7)]
        data = bpy.data.meshes.new("Surface")
        data.from_pydata(vertices, [], [(0, 2, 4), (1, 5, 3)])
        self.mesh = bpy.data.objects.new("Surface", data)
        bpy.context.scene.collection.objects.link(self.mesh)
        chest_group = self.mesh.vertex_groups.new(name="Chest")
        chest_group.add([0, 2, 4, 6], 1, "REPLACE")
        arm_group = self.mesh.vertex_groups.new(name="LeftUpperArm")
        arm_group.add([1, 3, 5, 7], 1, "REPLACE")
        self.mesh.modifiers.new("Armature", "ARMATURE").object = self.arm
        self.mesh.shape_key_add(name="Basis")
        key = self.mesh.shape_key_add(name="Expression")
        key.data[4].co.z += .005
        uv = data.uv_layers.new(name="UVMap")
        for i, loop in enumerate(uv.data):
            loop.uv = (i / 10, .4)
        bpy.context.view_layer.update()

    def test_coincident_shoulder_seams_stay_closed_when_arm_moves(self):
        changed = loose.synchronize_shoulder_seams(self.mesh, self.arm)
        self.assertEqual(changed, 4)
        tpose._point_arm(self.arm, "LeftUpperArm", "LeftHand", Vector((1, 0, 0)))
        coords = deformed(self.mesh)
        np.testing.assert_allclose(coords[[0, 2]], coords[[1, 3]], atol=1e-6)
        # The unrelated surfaces at the hand must remain free to part.
        self.assertGreater(np.linalg.norm(coords[6] - coords[7]), .1)

    def test_preserves_geometry_topology_uvs_and_every_shape_key(self):
        mesh = self.mesh.data
        before = [np.array([p.co[:] for p in key.data]) for key in mesh.shape_keys.key_blocks]
        faces = [tuple(p.vertices) for p in mesh.polygons]
        uv = np.array([p.uv[:] for p in mesh.uv_layers.active.data])
        loose.synchronize_shoulder_seams(self.mesh, self.arm)
        self.assertEqual(faces, [tuple(p.vertices) for p in mesh.polygons])
        np.testing.assert_array_equal(uv, [p.uv[:] for p in mesh.uv_layers.active.data])
        for old, key in zip(before, mesh.shape_keys.key_blocks):
            np.testing.assert_array_equal(old, [p.co[:] for p in key.data])
        for vertex in mesh.vertices:
            self.assertAlmostEqual(sum(g.weight for g in vertex.groups), 1, places=6)
        self.assertEqual(loose.synchronize_shoulder_seams(self.mesh, self.arm), 0)


class HairCutFromTheShoulder(unittest.TestCase):
    """A twin-tail cut free of the arm it lies on: the copies at the cut stay on their own bones."""

    def setUp(self):
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        data = bpy.data.armatures.new("Rig")
        self.arm = bpy.data.objects.new("Rig", data)
        bpy.context.scene.collection.objects.link(self.arm)
        rig.make_active(self.arm)
        bpy.ops.object.mode_set(mode="EDIT")
        chest = data.edit_bones.new("UpperChest")
        chest.head, chest.tail = (0, 0, .8), (0, 0, 1.1)
        shoulder = data.edit_bones.new("LeftShoulder")
        shoulder.head, shoulder.tail, shoulder.parent = (.02, 0, 1.1), (.15, 0, 1.1), chest
        upper = data.edit_bones.new("LeftUpperArm")
        upper.head, upper.tail, upper.parent = (.15, 0, 1.1), (.2, 0, .9), shoulder
        lower = data.edit_bones.new("LeftLowerArm")
        lower.head, lower.tail, lower.parent = upper.tail, (.25, 0, .7), upper
        neck = data.edit_bones.new("Neck")
        neck.head, neck.tail, neck.parent = (0, 0, 1.1), (0, 0, 1.2), chest
        head = data.edit_bones.new("Head")
        head.head, head.tail, head.parent = (0, 0, 1.2), (0, 0, 1.4), neck
        strand = data.edit_bones.new("Hair0_0")
        strand.head, strand.tail, strand.parent = (.1, 0, 1.2), (.14, 0, 1.0), head
        bpy.ops.object.mode_set(mode="OBJECT")
        # the two copies of each cut point, the hair side first
        vertices = [(.14, 0, 1.05), (.14, 0, 1.05), (.15, .02, 1.04), (.15, .02, 1.04), (.12, 0, 1.1), (.2, 0, 1.03)]
        data = bpy.data.meshes.new("Surface")
        data.from_pydata(vertices, [], [(0, 2, 4), (1, 5, 3)])
        self.mesh = bpy.data.objects.new("Surface", data)
        bpy.context.scene.collection.objects.link(self.mesh)
        self.mesh.vertex_groups.new(name="Hair0_0").add([0, 2, 4], 1, "REPLACE")
        self.mesh.vertex_groups.new(name="LeftUpperArm").add([1, 3, 5], 1, "REPLACE")
        self.mesh.modifiers.new("Armature", "ARMATURE").object = self.arm
        bpy.context.view_layer.update()

    def test_a_collar_still_closes(self):
        # neck and shoulder copies (2 steps apart) belong together
        names = {g.name: g for g in self.mesh.vertex_groups}
        neck = self.mesh.vertex_groups.new(name="Neck")
        names["Hair0_0"].remove([0])
        neck.add([0], 1, "REPLACE")
        names["LeftUpperArm"].remove([1])
        self.mesh.vertex_groups.new(name="LeftShoulder").add([1], 1, "REPLACE")
        self.assertEqual(loose.synchronize_shoulder_seams(self.mesh, self.arm), 2)

    def test_the_cut_is_not_averaged_shut(self):
        self.assertEqual(loose.synchronize_shoulder_seams(self.mesh, self.arm), 0)
        names = {g.index: g.name for g in self.mesh.vertex_groups}
        for index, bone in ((0, "Hair0_0"), (1, "LeftUpperArm"), (2, "Hair0_0"), (3, "LeftUpperArm")):
            weights = {names[g.group]: g.weight for g in self.mesh.data.vertices[index].groups if g.weight > 0}
            self.assertEqual(weights, {bone: 1.0})


class ChainsAreNotTornFromWhatTheyHangOn(unittest.TestCase):
    """Hair and cloth chains: no slit beside them and no seam that opens, only stretch."""

    def setUp(self):
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        data = bpy.data.armatures.new("Rig")
        self.arm = bpy.data.objects.new("Rig", data)
        bpy.context.scene.collection.objects.link(self.arm)
        rig.make_active(self.arm)
        bpy.ops.object.mode_set(mode="EDIT")
        hips = data.edit_bones.new("Hips")
        hips.head, hips.tail = (0, 0, .8), (0, 0, .9)
        spine = data.edit_bones.new("Spine")
        spine.head, spine.tail, spine.parent = (0, 0, .9), (0, 0, 1.1), hips
        leg = data.edit_bones.new("RightUpperLeg")
        leg.head, leg.tail, leg.parent = (-.07, 0, .78), (-.07, 0, .4), hips
        upper = data.edit_bones.new("LeftUpperArm")
        upper.head, upper.tail, upper.parent = (.15, 0, 1.1), (.2, 0, .9), spine
        lower = data.edit_bones.new("LeftLowerArm")
        lower.head, lower.tail, lower.parent = upper.tail, (.25, 0, .7), upper
        for chain, x in (("Cloth0", -.1), ("Cloth1", -.06)):
            top = data.edit_bones.new(f"{chain}_0")
            top.head, top.tail, top.parent = (x, .1, .8), (x, .1, .6), hips
            low = data.edit_bones.new(f"{chain}_1")
            low.head, low.tail, low.parent = top.tail, (x, .1, .4), top
        bpy.ops.object.mode_set(mode="OBJECT")

    def strip(self, left, right):
        """Three rows of six vertices, the left three columns on `left`, the right three on `right`."""
        points = [(x * .02 - .1, .1, .7 - z * .02) for z in range(3) for x in range(6)]
        faces = [(z * 6 + x, z * 6 + x + 1, (z + 1) * 6 + x + 1, (z + 1) * 6 + x) for z in range(2) for x in range(5)]
        data = bpy.data.meshes.new("Cape")
        data.from_pydata(points, [], faces)
        obj = bpy.data.objects.new("Cape", data)
        bpy.context.scene.collection.objects.link(obj)
        for name, columns in ((left, range(3)), (right, range(3, 6))):
            group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
            group.add([z * 6 + x for z in range(3) for x in columns], 1, "REPLACE")
        obj.modifiers.new("Armature", "ARMATURE").object = self.arm
        bpy.context.view_layer.update()
        return obj

    def test_a_cape_is_not_cut_from_the_thigh_its_top_lies_on(self):
        # model B's cape opened along its top as the body bent, where part of it was on a thigh
        obj = self.strip("Cloth0_0", "RightUpperLeg")
        self.assertEqual(loose.split_torn_seams(obj, self.arm), 0)
        self.assertEqual(len(obj.data.vertices), 18)

    def test_a_cape_is_still_cut_from_an_arm_resting_on_it(self):
        self.assertGreater(loose.split_torn_seams(self.strip("Cloth0_0", "LeftLowerArm"), self.arm), 0)

    def test_a_thigh_is_still_cut_from_an_arm_resting_on_it(self):
        self.assertGreater(loose.split_torn_seams(self.strip("RightUpperLeg", "LeftLowerArm"), self.arm), 0)

    def test_a_patch_of_one_chain_among_another_is_no_stray(self):
        obj = self.strip("Cloth0_0", "Cloth0_0")
        obj.vertex_groups["Cloth0_0"].remove([7, 8])
        obj.vertex_groups.new(name="Cloth1_0").add([7, 8], 1, "REPLACE")
        self.assertEqual(loose.fix_weight_islands(obj, self.arm), [])
        # a thigh and the same chain do read as strays on each other
        obj.vertex_groups["Cloth1_0"].remove([7, 8])
        obj.vertex_groups.new(name="RightUpperLeg").add([7, 8], 1, "REPLACE")
        self.assertTrue(loose.fix_weight_islands(obj, self.arm))

    def seam(self, first, second):
        """Two triangles meeting along an edge, each with its own copies of the edge's two ends."""
        vertices = [(-.1, .1, .7), (-.1, .1, .6), (-.14, .1, .65), (-.1, .1, .7), (-.1, .1, .6), (-.06, .1, .65)]
        data = bpy.data.meshes.new("Seam")
        data.from_pydata(vertices, [], [(0, 1, 2), (3, 5, 4)])
        obj = bpy.data.objects.new("Seam", data)
        bpy.context.scene.collection.objects.link(obj)
        for name, indices in ((first, [0, 1, 2]), (second, [3, 4, 5])):
            group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
            group.add(indices, 1, "REPLACE")
        return obj

    def weights(self, obj, index):
        names = {g.index: g.name for g in obj.vertex_groups}
        return {names[g.group]: round(g.weight, 3) for g in obj.data.vertices[index].groups if g.weight > 0}

    def test_copies_along_a_texture_seam_between_two_chains_move_as_one(self):
        # model A's hair: a copy on each chain at one spot, 22.6 cm apart with one chain swung
        obj = self.seam("Cloth0_0", "Cloth1_0")
        self.assertEqual(loose.share_strand_seams(obj), 4)
        for index in (0, 1, 3, 4):
            self.assertEqual(self.weights(obj, index), {"Cloth0_0": 0.5, "Cloth1_0": 0.5})
        self.assertEqual(self.weights(obj, 2), {"Cloth0_0": 1.0})
        self.assertEqual(self.weights(obj, 5), {"Cloth1_0": 1.0})
        self.assertEqual(loose.share_strand_seams(obj), 0)
        self.assertEqual(len(obj.data.vertices), 6)

    def test_copies_on_one_bone_left_a_little_apart_are_made_one_before_the_bake(self):
        # model A's lower back: a twelfth of an upper arm on one copy, none on its twin
        obj = self.seam("Spine", "Spine")
        obj.vertex_groups.new(name="LeftUpperArm").add([0, 1], 0.12, "REPLACE")
        obj.vertex_groups["Spine"].add([0, 1], 0.88, "REPLACE")
        self.assertEqual(loose.share_seam_copies(obj), 4)
        for index in (0, 1, 3, 4):
            self.assertEqual(self.weights(obj, index), {"Spine": 0.94, "LeftUpperArm": 0.06})
        self.assertEqual(loose.share_seam_copies(obj), 0)
        # the two sides of a cut are led by different bones and stay as they are
        cut = self.seam("RightUpperLeg", "LeftLowerArm")
        self.assertEqual(loose.share_seam_copies(cut), 0)
        self.assertEqual(self.weights(cut, 0), {"RightUpperLeg": 1.0})

    def test_a_bake_that_parts_copies_on_one_bone_is_counted(self):
        # model A's lower back, 2026-10-07: the bake moved one copy of 30 seam vertices and not its twin
        obj = self.seam("Spine", "Spine")
        copy, twin = loose.seam_copies_on_one_bone(obj)
        self.assertEqual((copy.tolist(), twin.tolist()), ([3, 4], [0, 1]))
        self.assertEqual(loose.parted_copies(obj, copy, twin), 0)
        obj.data.vertices[3].co.z += 0.004
        self.assertEqual(loose.parted_copies(obj, copy, twin), 1)
        # the two sides of a cut are meant to part in the bake: not among the seams that must stay closed
        cut = self.seam("RightUpperLeg", "LeftLowerArm")
        self.assertEqual(len(loose.seam_copies_on_one_bone(cut)[0]), 0)

    def test_a_copy_on_the_body_keeps_its_own_weights(self):
        # the other side of a cut that is meant to part (a sleeve cut free of the skirt)
        obj = self.seam("Cloth0_0", "LeftLowerArm")
        self.assertEqual(loose.share_strand_seams(obj), 0)
        self.assertEqual(self.weights(obj, 0), {"Cloth0_0": 1.0})
        self.assertEqual(self.weights(obj, 3), {"LeftLowerArm": 1.0})

    def bit(self, weights):
        """A triangle of its own beside a body of 200 vertices on the hips."""
        body = [(x * .02 - .2, 0, z * .02 + .8) for z in range(10) for x in range(20)]
        faces = [(z * 20 + x, z * 20 + x + 1, (z + 1) * 20 + x + 1, (z + 1) * 20 + x) for z in range(9) for x in range(19)]
        data = bpy.data.meshes.new("Body")
        data.from_pydata(body + [(-.08, .1, .6), (-.07, .1, .6), (-.08, .1, .61)], [], faces + [(200, 201, 202)])
        obj = bpy.data.objects.new("Body", data)
        bpy.context.scene.collection.objects.link(obj)
        obj.vertex_groups.new(name="Hips").add(list(range(200)), 1, "REPLACE")
        for name, weight in weights.items():
            obj.vertex_groups.new(name=name).add([200, 201, 202], weight, "REPLACE")
        return obj

    def test_a_bit_on_two_chains_of_its_strand_keeps_swinging_with_both(self):
        # model A: a 180-vertex bit of her back hair on Hair2 and Hair3 was put on the head, the bone
        # both hang from, and stood still while the hair round it swung 30 cm away
        obj = self.bit({"Cloth0_1": .5, "Cloth1_1": .5})
        self.assertEqual(loose.rigidify_props(obj, self.arm), [])
        self.assertEqual(self.weights(obj, 200), {"Cloth0_1": .5, "Cloth1_1": .5})

    def test_a_bit_on_a_chain_and_an_arm_still_gets_one_bone(self):
        obj = self.bit({"Cloth0_1": .5, "LeftLowerArm": .5})
        self.assertEqual(len(loose.rigidify_props(obj, self.arm)), 1)
        self.assertEqual(len(self.weights(obj, 200)), 1)


if __name__ == "__main__":
    unittest.main()
