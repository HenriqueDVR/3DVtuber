"""What the eye step reads off a face before it moves the eyes: scripts/add_eye_tracking.py."""

import importlib.util
import unittest
from pathlib import Path

import bpy

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "add_eye_tracking.py"


def load():
    spec = importlib.util.spec_from_file_location("add_eye_tracking", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # the step itself only runs as a script
    return module


class TheLidsBlink(unittest.TestCase):
    """The demon girl's lid blink moved 2.8 mm in openings 24 mm tall: her eyes stayed open."""

    def test_the_smaller_eye_s_blink_is_read(self):
        step = load()
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mesh = bpy.data.meshes.new("face")
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 0, 1)], [], [(0, 1, 2)])
        obj = bpy.data.objects.new("face", mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.shape_key_add(name="Basis")
        for name, drop in (("eyeBlinkLeft", 0.0028), ("eyeBlinkRight", 0.007)):
            key = obj.shape_key_add(name=name)
            key.data[2].co.z -= drop
        self.assertAlmostEqual(step.lid_travel(obj), 0.0028, places=6)
        self.assertLess(step.lid_travel(obj), step.BLINK_LEAST * 0.024)


class ASqueezeBlink(unittest.TestCase):
    """The knight's drawn eyes squeezed shut to a slit: the iris left in it is the blink, not an iris over the lids."""

    def test_only_looks_with_a_blink_are_let_through(self):
        step = load()
        failures = ["iris shows over the lids, left eye looking ahead with eyeBlinkLeft: 142 of 368 points",
                    "iris shows over the lids, right eye looking In+Up: 8 of 377 points"]
        self.assertEqual(step.squeezed(failures), failures[1:])


if __name__ == "__main__":
    unittest.main()
