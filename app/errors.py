"""Known ways a step fails, each with a code, what it means and what to do.

A failed step's log is matched against these (a script may also name its code itself with a
`VTUBER_MAKER_ERROR {"code": ..., "detail": ...}` line), so the app and agents get the fix from
docs/agent/troubleshooting.md instead of a traceback's last lines. Codes are stable: the
troubleshooting doc and the GUI refer to them.
"""

import json
import re

# (code, steps it applies to or None for any, pattern in the log, what happened, what to do,
#  manifest options that fix it or None)
KNOWN = [
    ("VM-OOM", None, r"Malloc returns null|STOPPED by the engine after \d+ s: out of memory",
     "Blender ran out of memory",
     "close other programs and rerun; a model over ~100k triangles is worth decimating first", None),
    ("VM-DISK-FULL", None, r"STOPPED by the engine after \d+ s: disk full|No space left on device|not enough space on the disk",
     "the drive the build writes to is full",
     "free up space (Characters page: it removes old runs' model files and keeps their reports), then rerun", None),
    ("VM-TIMEOUT", None, r"STOPPED by the engine after \d+ s$",
     "the step ran past the engine's time limit and was stopped",
     "rerun the step alone; if it hangs again, send a bundle", None),
    # the operator's own "could not be found", not its name: vrm_check's command line, echoed at
    # the top of its log, names import_scene.vrm, and every failed check read as a missing add-on
    ("VM-NO-VRM-ADDON", None, r"_scene\.vrm\" error, could not be found|VrmAddonMissing|has no attribute 'vrm'",
     "the VRM Add-on for Blender isn't installed in this Blender",
     "install \"VRM format\" in it (Preferences > Get Extensions); the app turns it on by itself", None),
    ("VM-EXPORT-POSE", None, r"ExportMismatch",
     "the exported .vrm doesn't hold the rest pose that was built: the VRM Add-on re-posed it on export",
     "use a VRM Add-on version the app was tested with and rerun the step; if it repeats, send a bundle", None),
    ("VM-TRIPO-MIX", None, r"'ShaderNodeMix' object has no attribute 'image'",
     "the model's material multiplies its colour by a grey factor (newer Tripo exports)",
     "run the matte step on the model first (scripts/matte_materials.py)", None),
    ("VM-NOT-HUMANOID", ("survey", "build"), r"doesn't read as a standing humanoid|LandmarkError",
     "joint detection found no standing body",
     "check the model stands upright, alone and unposed in the viewer; or mark the joints on the Joints page", None),
    ("VM-ORIENT-UNSURE", ("survey",), r"UNSURE: check the model stands up",
     "the model's orientation couldn't be decided",
     "turn it upright, facing you, in Blender and export it again", None),
    ("VM-NO-HAIR-BONES", ("hair",), r"need at least one array to stack",
     "no hair bones to reweight to (a rigid head)",
     "rerun: current builds skip hair weights without hair bones", None),
    ("VM-EYES-REFUSED", ("eyes",), r"^FAILED:",
     "the eye step's own check refused the moving eyes (iris over the lids, or hidden by the face)",
     "for a face with little eye white, set eyes to slide; read the eye close-ups in the step's log folder",
     {"eyes": "slide"}),
    ("VM-UNSUPPORTED-FILE", ("survey", "build"), r"Unsupported model type",
     "the model file type can't be read",
     "export it as FBX, GLB/glTF or OBJ", None),
    ("VM-BLENDER-OLD", None, r"needs Blender 4\.2 or later",
     "this Blender is older than 4.2, which the pipeline needs",
     "install Blender 4.2 LTS or later and point the app at it ($BLENDER or --blender)", None),
    # raised by the engine's own check of a step's output, never matched in a log
    ("VM-BAD-OUTPUT", (), None,
     "the step exited cleanly but its output file is missing, truncated or not a valid VRM 1.0",
     "rerun the step; if it repeats, the step's script has a bug: send a bundle", None),
    ("VM-STEP-DAMAGE", (), None,
     "the step changed something that isn't its own: another step's shape keys, the mesh, or the finger bones",
     "the step's script has a bug: send a bundle (the reason names what changed)", None),
]

TAG = "VTUBER_MAKER_ERROR"


def classify(step, output):
    """{"code", "what", "fix", "options"} for a failed step's log, or None when it isn't a known one."""
    for line in output.splitlines():
        if line.startswith(TAG):
            try:
                named = json.loads(line[len(TAG):].strip())
            except ValueError:
                break
            known = describe(named.get("code"))
            return {**(known or {"code": named.get("code"), "what": named.get("detail", ""), "fix": "", "options": None}),
                    "detail": named.get("detail", "")}
    for code, steps, pattern, what, fix, options in KNOWN:
        if pattern and (steps is None or step in steps) and re.search(pattern, output, re.MULTILINE):
            return {"code": code, "what": what, "fix": fix, "options": options}
    return None


def describe(code):
    for known, _, _, what, fix, options in KNOWN:
        if known == code:
            return {"code": code, "what": what, "fix": fix, "options": options}
    return None
