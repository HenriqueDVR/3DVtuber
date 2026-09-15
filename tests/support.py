"""Loading the synthetic character fixture into a clean scene."""

import json
from pathlib import Path

import bpy

FIXTURE_OUT = Path(__file__).resolve().parent / "fixtures" / "out"


def load_fixture(name="synthetic_character"):
    """Import the fixture GLB into an empty scene; return (mesh object, truth dict)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(FIXTURE_OUT / f"{name}.glb"))
    mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH")
    truth = json.loads((FIXTURE_OUT / f"{name}.truth.json").read_text())
    return mesh, truth
