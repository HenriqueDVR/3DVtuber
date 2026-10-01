"""Geodesic voxel weights (voxel_skin.py): distance through the solid, not through the air."""

import unittest

import bmesh
import bpy
import numpy as np

from vtuber_maker.pipeline import meshdata, rig, voxel_skin


def _box(bm, low, high):
    made = bmesh.ops.create_cube(bm, size=1.0)
    centre, scale = (np.array(low) + high) / 2, np.array(high) - low
    for v in made["verts"]:
        v.co = centre + np.array(v.co) * scale


def _bars(boxes, bones):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bm = bmesh.new()
    for low, high in boxes:
        _box(bm, low, high)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=8, use_grid_fill=True)
    data = bpy.data.meshes.new("bars")
    bm.to_mesh(data)
    bm.free()
    mesh = bpy.data.objects.new("bars", data)
    bpy.context.scene.collection.objects.link(mesh)
    armature = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
    bpy.context.scene.collection.objects.link(armature)
    rig.make_active(armature)
    bpy.ops.object.mode_set(mode="EDIT")
    made = {}
    for name, head, tail, parent in bones:
        bone = armature.data.edit_bones.new(name)
        bone.head, bone.tail = head, tail
        if parent:
            bone.parent = made[parent]
        made[name] = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    voxel_skin.weigh(mesh, armature)
    return mesh


def _weights(mesh, name):
    group = mesh.vertex_groups[name].index
    return np.array([next((g.weight for g in v.groups if g.group == group), 0.0) for v in mesh.data.vertices])


class SeparateParts(unittest.TestCase):
    """A hand resting 3 cm from a thigh: near through the air, unreachable through the body."""

    def test_neither_takes_the_other_bone(self):
        mesh = _bars([((0, 0, 0), (0.1, 0.1, 1.0)), ((0.13, 0, 0.3), (0.23, 0.1, 0.7))],
                     [("Thigh", (0.05, 0.05, 0.9), (0.05, 0.05, 0.1), None),
                      ("Hand", (0.18, 0.05, 0.65), (0.18, 0.05, 0.35), None)])
        x = meshdata.world_vertices(mesh)[:, 0]
        thigh, hand = _weights(mesh, "Thigh"), _weights(mesh, "Hand")
        np.testing.assert_allclose(thigh[x < 0.115], 1.0, atol=1e-3)
        np.testing.assert_allclose(hand[x > 0.115], 1.0, atol=1e-3)


class Joint(unittest.TestCase):
    """One bar, two bones meeting halfway: each end on its own bone, a blend at the joint."""

    @classmethod
    def setUpClass(cls):
        cls.mesh = _bars([((0, 0, 0), (0.1, 0.1, 1.0))],
                         [("Upper", (0.05, 0.05, 0.95), (0.05, 0.05, 0.5), None),
                          ("Lower", (0.05, 0.05, 0.5), (0.05, 0.05, 0.05), "Upper")])
        cls.z = meshdata.world_vertices(cls.mesh)[:, 2]

    def test_each_end_follows_its_bone(self):
        self.assertGreater(_weights(self.mesh, "Upper")[self.z > 0.8].min(), 0.9)
        self.assertGreater(_weights(self.mesh, "Lower")[self.z < 0.2].min(), 0.9)

    def test_blends_at_the_joint_and_sums_to_one(self):
        upper, lower = _weights(self.mesh, "Upper"), _weights(self.mesh, "Lower")
        joint = np.abs(self.z - 0.5) < 0.03
        self.assertTrue(((upper[joint] > 0.2) & (lower[joint] > 0.2)).all())
        np.testing.assert_allclose(upper + lower, 1.0, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
