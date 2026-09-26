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


if __name__ == "__main__":
    unittest.main()
