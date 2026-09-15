"""Enabling the VRM add-on in the middle of a session."""

import unittest

import bpy

from support import load_fixture
from vtuber_maker.pipeline import vrm_export


class EnableVrmAddonMidSession(unittest.TestCase):
    def test_keeps_the_selection_for_the_next_operator(self):
        # The add-on's first scene update after enabling deselects everything, which made
        # operators that act on the selection fail with "context is incorrect".
        mesh, _ = load_fixture()
        vrm_export.ensure_vrm_addon()
        mesh.select_set(True)
        bpy.context.view_layer.update()
        self.assertEqual([obj.name for obj in bpy.context.view_layer.objects.selected], [mesh.name])


if __name__ == "__main__":
    unittest.main()
