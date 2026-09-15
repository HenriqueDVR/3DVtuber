"""Run the test suite inside Blender.

    blender -b --factory-startup -P tests/run_tests.py [-- <file pattern>]

Builds the synthetic character fixture first if it is missing.
"""

import subprocess
import sys
import unittest
from pathlib import Path

import bpy

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
FIXTURE_OUT = TESTS / "fixtures" / "out"

sys.path.insert(0, str(REPO / "addon"))
sys.path.insert(0, str(TESTS))


def ensure_fixture():
    if (FIXTURE_OUT / "synthetic_character.glb").exists():
        return
    script = TESTS / "fixtures" / "make_synthetic_character.py"
    subprocess.run(
        [bpy.app.binary_path, "-b", "--factory-startup", "-P", str(script), "--", str(FIXTURE_OUT)],
        check=True,
    )


def main():
    ensure_fixture()
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    pattern = args[0] if args else "test_*.py"
    suite = unittest.defaultTestLoader.discover(str(TESTS), pattern=pattern, top_level_dir=str(TESTS))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


main()
