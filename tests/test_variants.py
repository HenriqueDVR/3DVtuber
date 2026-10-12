"""scripts/make_variants.py: each copy is the model with the one change its name says, and nothing else."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import bpy
import numpy as np

from support import FIXTURE_OUT

REPO = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("make_variants", REPO / "scripts" / "make_variants.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CopiesOfAModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = _script()
        cls.model = FIXTURE_OUT / "synthetic_character_tpose.glb"
        cls.folder = tempfile.TemporaryDirectory()
        cls.made = {}
        wanted = ("same_glb", "as_fbx", "as_obj", "turn_x_90", "turn_z_180", "centimetres", "materials_split", "objects_split",
                  "unwelded", "no_texture")
        for name, change, write, kind, _, _ in cls.tool.variants(None):
            if name not in wanted:
                continue
            cls.tool.load(cls.model)
            change()
            target = Path(cls.folder.name) / name / f"{name}{kind}"
            target.parent.mkdir(parents=True)
            write(target)
            cls.made[name] = target
        cls.plain = cls.read(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    @classmethod
    def read(cls, path):
        """What a file holds once imported as the pipeline imports it: its mesh objects, and all their
        vertices in the world."""
        cls.tool.load(path)
        objects = cls.tool.meshes()
        bpy.context.view_layer.update()
        verts = np.concatenate([np.array([obj.matrix_world @ v.co for v in obj.data.vertices]) for obj in objects])
        triangles = sum(cls.tool.triangles(obj) for obj in objects)
        materials = {slot.material.name for obj in objects for slot in obj.material_slots if slot.material}
        pictures = sum(1 for obj in objects for slot in obj.material_slots if slot.material and slot.material.node_tree
                       for node in slot.material.node_tree.nodes if node.type == "TEX_IMAGE" and node.image)
        return {"objects": objects, "verts": verts, "size": np.ptp(verts, axis=0), "triangles": triangles,
                "materials": materials, "pictures": pictures}

    def test_saved_again_it_is_the_same_surface_in_every_format(self):
        for name in ("same_glb", "as_fbx", "as_obj"):
            again = self.read(self.made[name])
            self.assertEqual(again["triangles"], self.plain["triangles"], name)
            # (as the pipeline's own importers hand each format over: standing as it stood)
            np.testing.assert_allclose(again["size"], self.plain["size"], rtol=1e-3, err_msg=name)
        self.assertTrue(self.made["as_obj"].with_suffix(".mtl").exists())

    def test_turned_it_lies_or_faces_as_its_name_says(self):
        lying = self.read(self.made["turn_x_90"])
        np.testing.assert_allclose(sorted(lying["size"]), sorted(self.plain["size"]), rtol=1e-3)
        # its height now runs front to back
        self.assertEqual(int(np.argmax(lying["size"])), 1)
        away = self.read(self.made["turn_z_180"])
        np.testing.assert_allclose(away["size"], self.plain["size"], rtol=1e-3)
        # what was in front is behind: the surface's middle, front to back, changed sign
        self.assertLess(float(np.median(away["verts"][:, 1])) * float(np.median(self.plain["verts"][:, 1])), 1e-9)

    def test_in_centimetres_it_is_a_hundred_times_the_size(self):
        np.testing.assert_allclose(self.read(self.made["centimetres"])["size"], self.plain["size"] * 100.0, rtol=1e-3)

    def test_split_it_is_the_same_surface_in_more_materials_or_objects(self):
        split = self.read(self.made["materials_split"])
        self.assertEqual((len(split["materials"]), split["triangles"]), (3, self.plain["triangles"]))
        parts = self.read(self.made["objects_split"])
        self.assertEqual((len(parts["objects"]), parts["triangles"]), (4, self.plain["triangles"]))
        np.testing.assert_allclose(parts["size"], self.plain["size"], rtol=1e-3)
        # one of them hangs under a parent that is turned and scaled, and still stands where it stood
        self.assertTrue(any(obj.parent is not None for obj in parts["objects"]))

    def test_unwelded_every_polygon_has_corners_of_its_own(self):
        cut = self.read(self.made["unwelded"])
        self.assertEqual(cut["triangles"], self.plain["triangles"])
        self.assertGreater(len(cut["verts"]), len(self.plain["verts"]))
        np.testing.assert_allclose(cut["size"], self.plain["size"], rtol=1e-3)

    def test_without_its_pictures_it_has_none(self):
        self.assertEqual(self.read(self.made["no_texture"])["pictures"], 0)
        self.assertEqual(self.read(self.made["no_texture"])["triangles"], self.plain["triangles"])

    def test_every_copy_says_what_is_expected_of_the_app(self):
        listed = self.tool.variants(Path("rigged.fbx"))
        self.assertEqual(len({name for name, *_ in listed}), len(listed))
        for name, _, _, kind, what, expects in listed:
            self.assertIn(kind, (".glb", ".fbx", ".obj"), name)
            self.assertTrue(what and (expects == "same" or all(code.startswith("VM-") for code in expects.split())), name)
        # the two that take a skeleton to make: arms against the sides, and down but clear of them
        self.assertLessEqual({"arms_lowered", "arms_low"}, {name for name, *_ in listed})
        self.assertFalse({"arms_lowered", "arms_low"} & {name for name, *_ in self.tool.variants(None)})
        self.assertGreater(self.tool.ARMS_LOWERED, self.tool.ARMS_LOW)


if __name__ == "__main__":
    unittest.main()
