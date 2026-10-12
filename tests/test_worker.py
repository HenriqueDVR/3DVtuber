"""The Blender a run keeps for its steps (scripts/_worker.py, app/engine.py `_Worker`) runs each step as a
Blender of the step's own would have: the same file, arguments and names, its output in the step's log,
its way of ending as an exit code, and nothing of one step left for the next."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app import engine  # noqa: E402

# What a step sees of where it runs, written to the file named after "--".
SEES = """
import json, os, sys
import bpy
out = sys.argv[sys.argv.index("--") + 1]
seen = {"objects": sorted(o.name for o in bpy.data.objects), "file": os.path.basename(bpy.data.filepath),
        "arguments": sys.argv[sys.argv.index("--") + 2:], "name": __name__, "script": os.path.basename(__file__),
        "main": sys.modules["__main__"].__dict__ is globals(), "blender": os.path.basename(sys.argv[0]),
        "said": os.environ.get("VTUBER_MAKER_TEST_SAID"), "left": os.environ.get("VTUBER_MAKER_TEST_LEFT"),
        "earlier": "earlier" in globals(), "handlers": len(bpy.app.handlers.save_post),
        "paths": sum(p.endswith("left_on_the_path") for p in sys.path), "pid": os.getpid()}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(seen, handle)
print("a line of the step's own")
"""
# A step that leaves all it can behind: a switch in the environment, a handler, a path, a name.
LEAVES = """
import os, sys
import bpy
from bpy.app.handlers import persistent
os.environ["VTUBER_MAKER_TEST_LEFT"] = "1"
sys.path.insert(0, "left_on_the_path")
earlier = True

@persistent
def left(*_):
    pass
bpy.app.handlers.save_post.append(left)
bpy.data.objects.new("left in the file", None)
"""


class KeptBlender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        for name in ("one", "two"):
            bpy.ops.wm.read_homefile(use_empty=True)
            bpy.context.scene.collection.objects.link(bpy.data.objects.new(name, None))
            bpy.ops.wm.save_as_mainfile(filepath=str(cls.root / f"{name}.blend"), copy=True)
        for name, text in (("sees.py", SEES), ("leaves.py", LEAVES), ("ends_3.py", "import sys\nprint('before it ends')\nsys.exit(3)\n"),
                           ("ends_in_words.py", "import sys\nsys.exit('nothing to do here')\n"), ("ends_well.py", "import sys\nsys.exit()\n"),
                           ("raises.py", "raise ValueError('the step went wrong')\n")):
            (cls.root / name).write_text(text, encoding="utf-8")
        cls.worker = engine._Worker(bpy.app.binary_path, cls.root / "run")

    @classmethod
    def tearDownClass(cls):
        cls.worker.close()
        cls.folder.cleanup()

    def step(self, script, blend="one", *arguments, extra=(), kept=True):
        """(exit code, the step's log, what sees.py wrote) of one step, by the kept Blender or one of its own."""
        self.steps = getattr(self, "steps", 0) + 1
        log, out = self.root / "run" / f"{self._testMethodName}_{self.steps}.log", self.root / f"{self._testMethodName}_{self.steps}.json"
        code = engine._blender(bpy.app.binary_path, self.root / f"{blend}.blend", [str(self.root / script), out, *arguments], log,
                               list(extra), **({"worker": self.worker} if kept else {}))
        return code, log.read_text(encoding="utf-8", errors="replace"), json.loads(out.read_text(encoding="utf-8")) if out.exists() else None

    def test_a_step_sees_what_it_would_in_a_blender_of_its_own(self):
        said = ["--python-expr", "import os; os.environ['VTUBER_MAKER_TEST_SAID'] = 'yes'"]
        code, log, kept = self.step("sees.py", "one", "--amount", "0.5", extra=said)
        alone_code, alone_log, alone = self.step("sees.py", "one", "--amount", "0.5", extra=said, kept=False)
        self.assertEqual((code, alone_code), (0, 0), log + alone_log)
        self.assertEqual({k: v for k, v in kept.items() if k != "pid"}, {k: v for k, v in alone.items() if k != "pid"})
        self.assertEqual((kept["objects"], kept["file"], kept["arguments"]), (["one"], "one.blend", ["--amount", "0.5"]))
        self.assertEqual((kept["name"], kept["script"], kept["main"], kept["said"]), ("__main__", "sees.py", True, "yes"))
        # its log starts with the command line a Blender of its own would have had, and holds what it said
        self.assertTrue(log.splitlines()[0].endswith("--amount 0.5") and "--factory-startup" in log.splitlines()[0], log)
        self.assertIn("a line of the step's own", log)
        self.assertNotIn("a line of the step's own", (self.root / "run" / "worker.log").read_text(encoding="utf-8", errors="replace"))

    def test_nothing_of_one_step_is_left_for_the_next(self):
        _, _, before = self.step("sees.py", "one")
        code, log, _ = self.step("leaves.py", "one", extra=["--python-expr", "import os; os.environ['VTUBER_MAKER_TEST_SAID'] = 'once'"])
        self.assertEqual(code, 0, log)
        _, _, after = self.step("sees.py", "two")
        self.assertEqual(after["pid"], before["pid"])  # the same Blender
        self.assertEqual((after["objects"], after["file"]), (["two"], "two.blend"))  # its own file, opened afresh
        self.assertEqual((after["said"], after["left"], after["earlier"], after["paths"]), (None, None, False, 0))
        self.assertEqual(after["handlers"], before["handlers"])

    def test_a_step_ends_with_the_code_its_own_blender_would_have(self):
        for script, code, said in (("ends_3.py", 3, "before it ends"), ("ends_in_words.py", 1, "nothing to do here"),
                                   ("ends_well.py", 0, ""), ("raises.py", 1, "ValueError: the step went wrong")):
            got, log, _ = self.step(script)
            self.assertEqual(got, code, (script, log))
            self.assertIn(said, log)
        # ... and the Blender is there for the next step all the same
        code, _, seen = self.step("sees.py", "two")
        self.assertEqual((code, seen["objects"]), (0, ["two"]))
        self.assertIsNone(self.worker.process.poll())

    def test_the_numbers_beside_a_steps_blend_are_written_as_it_saves(self):
        # as the engine asks for them: the switch in the environment, then scripts/_fingerprint.py, then the step
        prints, saved = self.root / "saved_fingerprint.json", self.root / "saved.blend"
        (self.root / "saves.py").write_text(f"import bpy\nbpy.ops.wm.save_as_mainfile(filepath={str(saved)!r}, copy=True)\n", encoding="utf-8")
        extra = ["--python-expr", f"import os; os.environ['VTUBER_MAKER_FINGERPRINT'] = {str(prints)!r}", "-P", str(engine.FINGERPRINT)]
        code, log, _ = self.step("saves.py", "one", extra=extra)
        self.assertEqual(code, 0, log)
        self.assertTrue(saved.exists())
        self.assertIn("meshes", json.loads(prints.read_text(encoding="utf-8")))
        # the handler that wrote them is gone with the step: the next step's save writes none
        prints.unlink()
        saved.unlink()
        code, log, _ = self.step("saves.py", "two")
        self.assertEqual((code, saved.exists(), prints.exists()), (0, True, False), log)


if __name__ == "__main__":
    unittest.main()
