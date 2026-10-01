"""Preparing an imported character: one mesh, real-world height, standing on the floor."""

import tempfile
import unittest
from pathlib import Path

import bpy
import numpy as np

from support import load_fixture
from vtuber_maker.pipeline import manifest, meshdata, normalize


class PrepareScaledAndOffsetCharacter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture()
        mesh.scale = (0.37, 0.37, 0.37)
        mesh.location = (3.0, -2.0, 5.0)
        # Generators sometimes export accessories as separate objects.
        accessory = mesh.copy()
        accessory.data = mesh.data.copy()
        accessory.location = (3.2, -2.0, 5.1)
        bpy.context.scene.collection.objects.link(accessory)
        cls.vertex_count = len(mesh.data.vertices) + len(accessory.data.vertices)

        cls.result = normalize.prepare([mesh, accessory], height=1.6)
        cls.verts = meshdata.world_vertices(cls.result)

    def test_joins_all_parts_into_one_mesh(self):
        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        self.assertEqual(meshes, [self.result])
        self.assertEqual(len(self.result.data.vertices), self.vertex_count)

    def test_scales_to_the_requested_height(self):
        self.assertAlmostEqual(float(np.ptp(self.verts[:, 2])), 1.6, places=3)

    def test_stands_on_the_floor(self):
        self.assertAlmostEqual(float(self.verts[:, 2].min()), 0.0, places=4)

    def test_is_centred_over_the_origin(self):
        center = (self.verts.min(axis=0) + self.verts.max(axis=0)) / 2
        self.assertAlmostEqual(float(center[0]), 0.0, places=3)
        self.assertAlmostEqual(float(center[1]), 0.0, places=3)

    def test_bakes_the_transform_into_the_mesh(self):
        self.assertTrue(np.allclose(np.array(self.result.matrix_world), np.eye(4)))


class PrepareRiggedImport(unittest.TestCase):
    """A rigged FBX (Tripo's) imports as a mesh bound to an armature; only the mesh is kept."""

    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture()
        armature = bpy.data.objects.new("Armature", bpy.data.armatures.new("Armature"))
        bpy.context.scene.collection.objects.link(armature)
        mesh.parent = armature
        mesh.vertex_groups.new(name="J_Bip_C_Hips").add([0, 1, 2], 1.0, "REPLACE")
        mesh.modifiers.new("Armature", "ARMATURE").object = armature
        cls.result = normalize.prepare([armature, mesh], height=1.6)

    def test_drops_the_rig_and_its_weights(self):
        self.assertEqual([o.type for o in bpy.context.scene.objects], ["MESH"])
        self.assertEqual(len(self.result.vertex_groups), 0)
        self.assertEqual(len(self.result.modifiers), 0)


class TextureStepsOnARiggedFbx(unittest.TestCase):
    """The texture steps (matte, tint, face patch) take Tripo's rigged FBX and write a GLB of the
    mesh alone that still names the FBX as its rig, as a GLB the app is given does."""

    def test_the_mesh_comes_off_the_rig_in_place_and_the_glb_names_the_rig(self):
        mesh, _ = load_fixture()
        armature = bpy.data.objects.new("Armature", bpy.data.armatures.new("Armature"))
        bpy.context.scene.collection.objects.link(armature)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="EDIT")
        bone = armature.data.edit_bones.new("J_Bip_C_Hips")
        bone.head, bone.tail = (0, 0, .9), (0, 0, 1.1)
        bpy.ops.object.mode_set(mode="OBJECT")
        armature.scale = (.5, .5, .5)
        mesh.parent = armature
        mesh.vertex_groups.new(name="J_Bip_C_Hips").add(list(range(len(mesh.data.vertices))), 1.0, "REPLACE")
        mesh.modifiers.new("Armature", "ARMATURE").object = armature
        bpy.context.view_layer.update()
        expected = meshdata.world_vertices(mesh)
        with tempfile.TemporaryDirectory() as folder:
            fbx, glb = Path(folder) / "rigged.fbx", Path(folder) / "matte.glb"
            bpy.ops.export_scene.fbx(filepath=str(fbx), add_leaf_bones=False)
            bpy.ops.wm.read_factory_settings(use_empty=True)
            self.assertEqual(normalize.import_unrigged(fbx), fbx)
            meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
            self.assertEqual([o.type for o in bpy.context.scene.objects if o.type == "ARMATURE"], [])
            self.assertEqual([(len(o.modifiers), len(o.vertex_groups), o.parent) for o in meshes], [(0, 0, None)])
            np.testing.assert_allclose(meshdata.world_vertices(meshes[0]), expected, atol=1e-4)
            normalize.export_glb(glb, fbx)
            bpy.ops.wm.read_factory_settings(use_empty=True)
            self.assertEqual(manifest.rig_of(glb), fbx)
            self.assertEqual(normalize.import_unrigged(glb), fbx)  # the next step names it again


if __name__ == "__main__":
    unittest.main()
