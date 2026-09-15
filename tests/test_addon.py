"""The Blender add-on: a sidebar button that turns the selected character into a VRM."""

import unittest

import bpy

import vtuber_maker
from support import FIXTURE_OUT, load_fixture
from vtuber_maker.pipeline import vrm_export


class MakeVrmFromTheSidebar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_fixture()
        vrm_export.ensure_vrm_addon()
        vtuber_maker.register()
        for obj in bpy.context.scene.objects:
            obj.select_set(True)
        cls.path = FIXTURE_OUT / "addon_operator.vrm"
        cls.path.unlink(missing_ok=True)
        cls.result = bpy.ops.vtuber_maker.make_vrm(filepath=str(cls.path), avatar_name="Synthetic", author="Tests")

    @classmethod
    def tearDownClass(cls):
        vtuber_maker.unregister()

    def test_operator_finishes(self):
        self.assertEqual(self.result, {"FINISHED"})

    def test_saves_the_vrm_file(self):
        self.assertGreater(self.path.stat().st_size, 0)

    def test_adds_a_panel_to_the_3d_view_sidebar(self):
        panel = getattr(bpy.types, "VTUBER_MAKER_PT_panel", None)
        self.assertIsNotNone(panel)
        self.assertEqual((panel.bl_space_type, panel.bl_region_type), ("VIEW_3D", "UI"))


if __name__ == "__main__":
    unittest.main()
