"""Small holes in the surface: cracks where two vertices stand apart that should be one point."""

import unittest

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

from vtuber_maker.pipeline import loose, meshdata

CELL = 0.01
# How far apart the crack's two vertices stand (model A's: 0.6 to 0.9 mm).
APART = 0.0008


def sheet_with_a_crack():
    """An 8 x 8 sheet of 1 cm squares facing -Y, cracked open at one vertex, with a shape key.

    Two edges in line are split at the vertex between them, and one of its two copies is moved
    APART sideways: a ring of four open edges round two slivers, as on model A's shirt.
    Returns (object, the moved vertex's index).
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new("sheet")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=8, y_segments=8, size=4 * CELL)
    bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(np.pi / 2, 3, "X"))
    middle = min(bm.verts, key=lambda v: v.co.length)
    edges = [e for e in middle.link_edges if abs(e.other_vert(middle).co.x - middle.co.x) < 1e-6]
    before = set(bm.verts)
    bmesh.ops.split_edges(bm, edges=edges, verts=[middle], use_verts=True)
    copy = next(v for v in bm.verts if v not in before)
    copy.co.x += APART
    bm.verts.index_update()
    moved = copy.index
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("sheet", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.shape_key_add(name="Basis", from_mix=False)
    key = obj.shape_key_add(name="up", from_mix=False)
    key.value = 0.0
    for point in key.data:
        point.co.z += 0.003
    return obj, moved


def key_coords(obj, name):
    data = obj.data.shape_keys.key_blocks[name].data
    coords = np.empty(len(data) * 3)
    data.foreach_get("co", coords)
    return coords.reshape(-1, 3)


class SmallHoles(unittest.TestCase):
    def setUp(self):
        self.obj, self.moved = sheet_with_a_crack()

    def test_the_sheets_own_border_is_too_large_to_be_a_hole(self):
        # 8 cm across, over the size: only the crack counts, until the size is raised
        self.assertEqual(len(loose.find_small_holes(self.obj)), 1)
        loose.close_small_holes(self.obj)
        self.assertEqual(loose.find_small_holes(self.obj), [])
        self.assertEqual(len(loose.find_small_holes(self.obj, size=1.0)), 1)

    def test_finds_the_crack_as_one_ring_of_four_edges(self):
        holes = loose.find_small_holes(self.obj)
        self.assertEqual([len(hole) for hole in holes], [4])

    def test_a_box_keeps_to_one_place(self):
        self.assertEqual(len(loose.find_small_holes(self.obj, low=(-0.02, -1, -0.02), high=(0.02, 1, 0.02))), 1)
        self.assertEqual(loose.find_small_holes(self.obj, low=(0.02, -1, 0.02), high=(0.04, 1, 0.04)), [])

    def test_closing_moves_only_the_cracks_vertex_and_every_shape_key_with_it(self):
        before = meshdata.world_vertices(self.obj)
        lift = key_coords(self.obj, "up") - key_coords(self.obj, "Basis")
        self.assertEqual(loose.close_small_holes(self.obj), 1)
        after = meshdata.world_vertices(self.obj)
        step = np.linalg.norm(after - before, axis=1)
        self.assertEqual(np.nonzero(step > 0)[0].tolist(), [self.moved])
        self.assertAlmostEqual(step[self.moved], APART, places=6)
        self.assertEqual(len(after), len(before))
        np.testing.assert_allclose(key_coords(self.obj, "Basis"), after, atol=1e-7)
        np.testing.assert_allclose(key_coords(self.obj, "up") - key_coords(self.obj, "Basis"), lift, atol=1e-7)

    def test_a_hole_with_real_area_is_found_and_left_alone(self):
        bm = bmesh.new()
        bm.from_mesh(self.obj.data)
        corner = max(bm.faces, key=lambda f: f.calc_center_median().x + f.calc_center_median().z)
        inner = next(f for f in bm.faces if abs(f.calc_center_median().x - corner.calc_center_median().x + CELL) < 1e-6
                     and abs(f.calc_center_median().z - corner.calc_center_median().z + CELL) < 1e-6)
        bmesh.ops.delete(bm, geom=[inner], context="FACES_ONLY")
        bm.to_mesh(self.obj.data)
        bm.free()
        self.assertEqual(sorted(len(hole) for hole in loose.find_small_holes(self.obj)), [4, 4])
        self.assertEqual(loose.close_small_holes(self.obj), 1)
        self.assertEqual([len(hole) for hole in loose.find_small_holes(self.obj)], [4])


if __name__ == "__main__":
    unittest.main()
