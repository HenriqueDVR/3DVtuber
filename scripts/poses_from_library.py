"""Take single poses out of animation files and write them down for `render_portraits.py`.

    blender -b --factory-startup -P scripts/poses_from_library.py -- library.glb poses.json Idle_Loop:0=idle Dance_Loop:12 ...
    blender -b --factory-startup -P scripts/poses_from_library.py -- folder_of_fbx poses.json --frames 5
    blender -b --factory-startup -P scripts/poses_from_library.py -- folder_of_fbx poses.json Waving.fbx:20=setup ...

Reads two kinds of skeleton, told apart by their bone names: Quaternius's Universal Animation
Library (one .glb holding many actions; CC0) and Mixamo's (.fbx, one animation a file, downloaded
"without skin"; Adobe's terms let anyone use them and nobody pass the files on, so those stay on
the user's own disk, as do poses taken from them). Both are bound in a T-pose facing -Y once in
Blender.

Each `Action:frame` (or `File.fbx:frame`; `=name` to call it something else) becomes, per humanoid
bone, the turn that frame gives the bone away from its rest, as a quaternion in world space, plus
how far the hips moved in hip heights and which way each limb bone points at rest. With those, any
rig with the VRM bone names can be put in the pose whatever pose it was bound in. `--frames N`
takes N frames spread over every file instead, to choose among. The poses are numbers, a few
kilobytes each: the animation files are not kept.
"""
import json
import sys
from pathlib import Path

import bpy

SIDES = {"l": "Left", "r": "Right"}
FINGERS = (("Index", "Index"), ("Middle", "Middle"), ("Ring", "Ring"), ("Pinky", "Little"))
JOINTS = ((1, "Proximal"), (2, "Intermediate"), (3, "Distal"))
# Quaternius: lower case, the side at the end ("upperarm_l")
QUATERNIUS = {
    "pelvis": "Hips", "spine_01": "Spine", "spine_02": "Chest", "spine_03": "UpperChest", "neck_01": "Neck", "Head": "Head",
    **{f"{theirs}_{side}": f"{SIDES[side]}{ours}" for side in SIDES for theirs, ours in {
        "clavicle": "Shoulder", "upperarm": "UpperArm", "lowerarm": "LowerArm", "hand": "Hand", "thigh": "UpperLeg", "calf": "LowerLeg",
        "foot": "Foot", "ball": "Toes", "thumb_01": "ThumbMetacarpal", "thumb_02": "ThumbProximal", "thumb_03": "ThumbDistal",
        **{f"{finger.lower()}_0{n}": f"{ours}{part}" for finger, ours in FINGERS for n, part in JOINTS}}.items()}}
# Mixamo: "mixamorig:LeftForeArm", read with whatever stands before the colon taken off
MIXAMO = {
    "Hips": "Hips", "Spine": "Spine", "Spine1": "Chest", "Spine2": "UpperChest", "Neck": "Neck", "Head": "Head",
    **{f"{side}{theirs}": f"{side}{ours}" for side in SIDES.values() for theirs, ours in {
        "Shoulder": "Shoulder", "Arm": "UpperArm", "ForeArm": "LowerArm", "Hand": "Hand", "UpLeg": "UpperLeg", "Leg": "LowerLeg",
        "Foot": "Foot", "ToeBase": "Toes", "HandThumb1": "ThumbMetacarpal", "HandThumb2": "ThumbProximal", "HandThumb3": "ThumbDistal",
        **{f"Hand{finger}{n}": f"{ours}{part}" for finger, ours in FINGERS for n, part in JOINTS}}.items()}}
# The bone whose head a limb bone points at: its direction at rest is what a rig bound in another
# pose is lined up with.
POINTS_AT = {"Shoulder": "UpperArm", "UpperArm": "LowerArm", "LowerArm": "Hand", "Hand": "MiddleProximal", "UpperLeg": "LowerLeg",
             "LowerLeg": "Foot", "ThumbMetacarpal": "ThumbProximal", "ThumbProximal": "ThumbDistal",
             **{f"{finger}{part}": f"{finger}{after}" for finger in ("Index", "Middle", "Ring", "Little")
                for part, after in (("Proximal", "Intermediate"), ("Intermediate", "Distal"))}}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    source, out, rest = Path(argv[0]), Path(argv[1]), argv[2:]
    spread = int(rest[rest.index("--frames") + 1]) if "--frames" in rest else 0
    wanted = [item for item in rest if ":" in item]
    poses = {}
    files = sorted(source.glob("*.fbx")) if source.is_dir() else [source]
    for path in files:
        armature = load(path)
        actions = {a.name: a for a in bpy.data.actions}
        if path.suffix.lower() == ".fbx":  # one animation a file: named for the file
            actions = {path.name: armature.animation_data.action} if armature.animation_data and armature.animation_data.action else {}
        todo = []
        for item in wanted:
            clip, _, name = item.partition("=")
            action, _, frame = clip.rpartition(":")
            if action in actions:
                todo.append((action, int(frame), name or f"{Path(action).stem}_{int(frame)}"))
        if spread:
            for action, clip in actions.items():
                first, last = (int(v) for v in clip.frame_range)
                for k in range(spread):
                    frame = first + round((last - first) * (k + 0.5) / spread)
                    todo.append((action, frame, f"{Path(action).stem}_{frame}"))
        for action, frame, name in todo:
            poses[name] = read(armature, actions[action], frame, f"{action} frame {frame}")
            print(f"POSE {name}")
    credit = ("Universal Animation Library by Quaternius, CC0 1.0 (https://quaternius.com)" if not source.is_dir()
              else "the user's own animation files: not for passing on")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": credit, "poses": poses}, indent=1), encoding="utf-8")


def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if path.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        bpy.ops.import_scene.gltf(filepath=str(path))
    return next(o for o in bpy.data.objects if o.type == "ARMATURE")


def read(armature, action, frame, said):
    """One frame as a pose: the bones' turns from rest and the rest's directions, in world space (an
    imported FBX's armature lies on its side and is a hundredth the size: its own space is no use)."""
    names = {}
    for bone in armature.data.bones:
        ours = QUATERNIUS.get(bone.name) or MIXAMO.get(bone.name.rpartition(":")[2] if ":" in bone.name else "")
        if ours and ours not in names.values():
            names[bone.name] = ours
    back = {ours: theirs for theirs, ours in names.items()}
    world = armature.matrix_world
    at_rest = {theirs: world @ armature.data.bones[theirs].matrix_local for theirs in names}
    rest = {}
    for ours, theirs in back.items():
        part = ours.removeprefix("Left").removeprefix("Right")
        target = back.get(ours[:len(ours) - len(part)] + POINTS_AT[part]) if part in POINTS_AT else None
        if target:
            rest[ours] = [round(v, 5) for v in (at_rest[target].translation - at_rest[theirs].translation).normalized()]
    armature.animation_data_create()
    armature.animation_data.action = action
    # Blender 4.4 and later keep an action's curves in a slot
    if hasattr(armature.animation_data, "action_slot") and getattr(action, "slots", None):
        armature.animation_data.action_slot = action.slots[0]
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    bones = {}
    for theirs, ours in names.items():
        turn = (world @ armature.pose.bones[theirs].matrix).to_quaternion() @ at_rest[theirs].to_quaternion().inverted()
        bones[ours] = [round(v, 5) for v in turn]
    hips = back["Hips"]
    moved = ((world @ armature.pose.bones[hips].matrix).translation - at_rest[hips].translation) / at_rest[hips].translation.z
    # up and down is the pose (a crouch, a jump); far sideways is the animation walking off, not kept
    moved.x, moved.y = max(-0.15, min(0.15, moved.x)), max(-0.15, min(0.15, moved.y))
    return {"from": said, "hips": [round(v, 5) for v in moved], "rest": rest, "bones": bones}


main()
