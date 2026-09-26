"""Real-window smoke check (not a background unittest): Blender -P this_file.py.

Opens an isolated synthetic scene, exercises viewport focus/views/guide drawing, then quits.
This verifies UI execution, not aesthetic quality or human usability.
"""
import sys
import traceback
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))
sys.path.insert(0, str(ROOT / "tests"))
import vtuber_maker
from vtuber_maker import hand_setup_ui
from vtuber_maker.pipeline import hand_setup
from test_fingers import build_hand
from test_hand_setup import guides

state = {"draws": 0, "error": None}


def draw():
    try:
        hand_setup_ui._draw_guides()
        state["draws"] += 1
    except Exception:
        state["error"] = traceback.format_exc()


def finish():
    try:
        assert not state["error"], state["error"]
        assert state["draws"] > 0, "GPU guide callback did not draw"
        with bpy.context.temp_override(window=state["window"], area=state["area"], region=state["region"]):
            assert bpy.ops.vtuber_maker.hand_action(action="APPLY") == {"FINISHED"}
            assert bpy.ops.vtuber_maker.hand_action(action="FIST") == {"FINISHED"}
            assert bpy.ops.vtuber_maker.hand_action(action="RESET") == {"FINISHED"}
        print("PASS VIEWPORT: focus, palm/side cameras, move gizmo, GPU guides, apply, preview, reset", flush=True)
    except Exception:
        traceback.print_exc()
        print("FAIL VIEWPORT", flush=True)
    finally:
        bpy.types.SpaceView3D.draw_handler_remove(state["handle"], "WINDOW")
        vtuber_maker.unregister()
        bpy.ops.wm.quit_blender()


def start():
    try:
        vtuber_maker.register()
        mesh, arm = build_hand()
        settings = bpy.context.scene.vtuber_hand_setup
        settings.mesh, settings.armature = mesh, arm
        window = bpy.context.window_manager.windows[0]
        area = next(a for a in window.screen.areas if a.type == "VIEW_3D")
        region = next(r for r in area.regions if r.type == "WINDOW")
        state.update(window=window, area=area, region=region)
        with bpy.context.temp_override(window=window, area=area, region=region):
            area.spaces.active.show_region_ui = True
            assert bpy.ops.vtuber_maker.hand_action(action="START") == {"FINISHED"}
            hand_setup.write_guides(arm, "left", guides())
            for action in ("PALM", "SIDE", "SELECT"):
                assert bpy.ops.vtuber_maker.hand_action(action=action, point=3) == {"FINISHED"}
        state["handle"] = bpy.types.SpaceView3D.draw_handler_add(draw, (), "WINDOW", "POST_VIEW")
        area.tag_redraw()
        bpy.app.timers.register(finish, first_interval=3)
    except Exception:
        traceback.print_exc()
        print("FAIL VIEWPORT", flush=True)
        bpy.ops.wm.quit_blender()


bpy.app.timers.register(start, first_interval=2)
