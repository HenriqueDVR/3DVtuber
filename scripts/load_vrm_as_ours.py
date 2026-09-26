"""Import a .vrm and rename its humanoid bones to the pipeline's names, for our checks and renders.

    blender -b --python-exit-code 1 -P scripts/load_vrm_as_ours.py -P scripts/check_avatar.py -- model.vrm [check args]

The first argument after "--" is the .vrm; it is removed from sys.argv so the next script sees
only its own. For VRMs rigged elsewhere (Tripo's J_Bip_* names): check_avatar.py, render_pose.py
and check_movement.py find bones by the pipeline's names (rig.BONE_NAMES). Renaming a bone
renames its vertex group, so weights come along. Changes nothing on disk.
"""

import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import fingers, vrm_export  # noqa: E402
from vtuber_maker.pipeline.rig import BONE_NAMES  # noqa: E402

split = sys.argv.index("--")
path = sys.argv.pop(split + 1)
vrm_export.ensure_vrm_addon()
bpy.ops.wm.read_homefile(use_empty=True)
assert bpy.ops.import_scene.vrm(filepath=str(Path(path).resolve())) == {"FINISHED"}
armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
human_bones = armature.data.vrm_addon_extension.vrm1.humanoid.human_bones
ours = dict(BONE_NAMES)
for side in ("left", "right"):
    for finger in fingers.FINGERS:
        for key, segment in zip(fingers.humanoid_keys(side, finger), fingers._segments(finger)):
            ours[key] = fingers.bone_name(side, finger, segment)
renamed = 0
for key, name in ours.items():
    bone = getattr(human_bones, key, None)
    current = bone.node.bone_name if bone is not None else ""
    if current and current != name and current in armature.data.bones:
        armature.data.bones[current].name = name
        renamed += 1
# check_avatar.py takes the mesh parented to the armature
for obj in bpy.data.objects:
    if obj.type == "MESH" and obj.find_armature() == armature and obj.parent != armature:
        world = obj.matrix_world.copy()
        obj.parent = armature
        obj.matrix_world = world
print(f"loaded {path}: renamed {renamed} bones to the pipeline's names")
