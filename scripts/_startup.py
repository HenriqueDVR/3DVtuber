"""Run by the app before a step's file loads (app/engine.py): turns the VRM Add-on on.

The app starts Blender with --factory-startup, so none of the user's own add-ons or
preferences load: Rokoko loaded in every step, and anyone's add-on could slow or break one.
Factory settings switch the VRM Add-on off too, and a .blend's VRM data (humanoid bones,
springs, MToon) only reads once it is on, so it has to be on before the file opens.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import vrm_export  # noqa: E402

try:
    vrm_export.ensure_vrm_addon()
except vrm_export.VrmAddonMissing:
    # the survey doesn't need it; a step that does says so itself (VM-NO-VRM-ADDON)
    print("VRM Add-on not installed")
