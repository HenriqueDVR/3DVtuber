"""Render a rigged avatar in the poses the app's menus show it in.

    blender -b in.blend -P scripts/render_portraits.py -- out_dir [--size 1400]

`render_pose.py` shows the poses the rig check measures, each framed to fit, for judging a build.
These are pictures of the character, one for each place the app shows it:
character select (standing, arms by its sides), and a gesture for each card of the main menu that
says what the card is for: a wave (Set up), an arm flexed (Build), a thumb up (Check).
Every one is framed alike, by the character's own height and the place of its hips, and
`portraits.json` says where the face and the floor came out in each picture and how wide the face
is, so the app can put any character in any pose on the same spot at the same size, and cut a
close-up of the face for its roster.

Assumes a humanoid armature with the VRM bone names, standing upright and facing -Y. The poses are
made here, of directions and angles in that frame (shoulder to elbow along this, the palm facing
that, these fingers curled), so the pose the rig was bound in, T or A, makes no difference. Poses
nothing it does not find (a rig without finger bones keeps flat hands), and saves nothing but the
pictures: the .blend is left as it was.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

# The four pictures, named for where the app shows them. In the character's frame: +X its left, -Y its
# front, +Z up.
#   aim    each bone pointed from its own head at its child's along a direction. Arms stay clear of the
#          body: nothing here knows where the clothes are.
#   sway   degrees the hips tip, the spine tipping back to keep the head over the feet: weight on one leg.
#   roll   a forearm turned about its own length until the palm ("palm") or the thumb's side of the hand
#          ("thumb") faces a direction.
#   hands  each hand's shape: "relaxed", "fist", "point" (index out) or "thumb" (thumb out).
#   turn   a bone turned about a world axis, in degrees.
#   yaw    how far round to the character's left the camera stands, in degrees.
CHILD = {"UpperArm": "LowerArm", "LowerArm": "Hand", "UpperLeg": "LowerLeg", "LowerLeg": "Foot",
         "ThumbMetacarpal": "ThumbProximal", "ThumbProximal": "ThumbDistal"}
# the arm that is not making the gesture: elbow out, hand down by the hip (a hand's width off it)
HIP_LEFT = {"LeftUpperArm": (0.75, 0.15, -0.65), "LeftLowerArm": (-0.40, -0.45, -0.80)}
HIP_RIGHT = {"RightUpperArm": (-0.75, 0.15, -0.65), "RightLowerArm": (0.40, -0.45, -0.80)}
POSES = {
    # standing as anyone stands, arms down by the sides: character select
    "stand": {"yaw": 0, "aim": {
        "LeftUpperArm": (0.27, 0.02, -1), "LeftLowerArm": (0.20, -0.14, -1),
        "RightUpperArm": (-0.27, 0.02, -1), "RightLowerArm": (-0.20, -0.14, -1)},
        "hands": {"Left": "relaxed", "Right": "relaxed"}},
    # Set up: a wave, open hand up. Hello, let's begin
    "setup": {"yaw": 10, "sway": 4, "aim": {
        "RightUpperArm": (-0.80, -0.10, -0.40), "RightLowerArm": (-0.28, -0.12, 1),
        "LeftUpperArm": (0.27, 0.02, -1), "LeftLowerArm": (0.20, -0.14, -1)},
        "roll": {"RightLowerArm": ("palm", (0, -1, 0))}, "hands": {"Left": "relaxed"},
        "turn": [("Head", "Y", -6)]},
    # Build: an arm flexed, fist up
    "build": {"yaw": -12, "sway": -4, "aim": {
        "RightUpperArm": (-1, 0.05, 0.15), "RightLowerArm": (0.45, -0.10, 1), **HIP_LEFT,
        "LeftUpperLeg": (0.10, -0.12, -1), "LeftLowerLeg": (0.06, 0.10, -1)},
        "roll": {"RightLowerArm": ("palm", (1, 0.2, -0.2))}, "hands": {"Right": "fist", "Left": "fist"},
        "turn": [("Head", "Z", -10), ("Head", "Y", -4)]},
    # Check: a thumb up with the other hand, held out toward whoever is looking
    "check": {"yaw": -10, "sway": -4, "aim": {
        "LeftUpperArm": (0.55, -0.70, -0.45), "LeftLowerArm": (0.35, -0.85, 0.45), **HIP_RIGHT,
        "LeftUpperLeg": (0.10, -0.12, -1), "LeftLowerLeg": (0.06, 0.10, -1)},
        "roll": {"LeftLowerArm": ("thumb", (0, 0, 1))}, "hands": {"Left": "thumb", "Right": "relaxed"},
        "turn": [("Head", "Y", -5)]},
}
# How far each joint of a finger bends for a hand's shape, knuckle to tip, in degrees.
CURL = {"relaxed": (14, 20, 12), "fist": (78, 95, 62), "open": (0, 0, 0)}
# Poses taken from animation files instead (poses_from_library.py), for whoever wants their own: a file
# in tools/poses, which is kept out of the repository (most such files may not be passed on). A pose in
# it named for a picture ("setup") takes that picture's place.
LIBRARY = Path(__file__).resolve().parents[1] / "tools" / "poses" / "poses.json"
# The picture is this many character heights across, and its middle this far up the character: room
# over the head for a raised arm, a little under the feet.
SPAN, MIDDLE = 1.5625, 0.62
# The face, for the close-up a roster slot shows: as wide as what the head bone holds is just above its
# joint (the jaw and cheeks, under any hat's brim), its middle half that width above the joint. Not from
# the top of the model: a tall hat or a crown is not the head. In character heights when it can't be measured.
FACE_WIDTH, FACE_BAND = 0.15, (0.02, 0.10)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--size", type=int, default=1400)
    parser.add_argument("--library", type=Path, default=LIBRARY, help="poses taken from an animation library (poses_from_library.py)")
    parser.add_argument("--sheet", action="store_true", help="every pose in the library, from the front: for choosing among them")
    args = parser.parse_args(argv)
    library = json.loads(args.library.read_text(encoding="utf-8")) if args.library.exists() else {"poses": {}}
    todo = {name: {"yaw": 0, "library": name} for name in library["poses"]} if args.sheet else POSES
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    reset(armature)
    low, high = bounds()
    height, floor = high.z - low.z, low.z
    world = armature.matrix_world
    hips = world @ armature.data.bones["Hips"].head_local if "Hips" in armature.data.bones else (low + high) / 2
    centre = Vector((hips.x, hips.y, floor + MIDDLE * height))
    span = SPAN * height
    face = face_width(armature, height)
    report = {"blend": bpy.data.filepath, "size": args.size, "height": 1 / SPAN, "face": round(face / span, 4), "poses": {}}
    for name, pose in todo.items():
        reset(armature)
        # a pose in the file under the picture's own name ("setup") comes first: the user's own choice
        chosen = library["poses"].get(name) or library["poses"].get(pose.get("library"))
        if chosen:
            strike(armature, chosen, chosen["rest"])
        else:
            make(armature, pose)
        bpy.context.view_layer.update()
        yaw = math.radians(pose["yaw"])
        right = Vector((math.cos(yaw), math.sin(yaw), 0.0))
        render(out_dir / f"{name}.png", centre + Vector((math.sin(yaw), -math.cos(yaw), 0.0)) * 10.0, (math.pi / 2, 0.0, yaw), span, args.size)
        head = head_point(armature, face)
        report["poses"][name] = {"file": f"{name}.png", "floor": round(0.5 - (floor - centre.z) / span, 4),
                                 "head": [round(0.5 + (head - centre).dot(right) / span, 4), round(0.5 - (head.z - centre.z) / span, 4)]}
        print(f"PORTRAIT {name} -> {out_dir / f'{name}.png'}")
    reset(armature)
    (out_dir / "portraits.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


# The bone a limb bone points at, as poses_from_library.py has it: a rig bound in another pose than the
# library's is lined up with the library's rest by these directions.
POINTS_AT = {"Shoulder": "UpperArm", "UpperArm": "LowerArm", "LowerArm": "Hand", "Hand": "MiddleProximal", "UpperLeg": "LowerLeg",
             "LowerLeg": "Foot", "ThumbMetacarpal": "ThumbProximal", "ThumbProximal": "ThumbDistal",
             **{f"{finger}{part}": f"{finger}{after}" for finger in ("Index", "Middle", "Ring", "Little")
                for part, after in (("Proximal", "Intermediate"), ("Intermediate", "Distal"))}}


def strike(armature, pose, rest):
    """Put the rig in a pose from the library: each bone turned from its rest as the library's was from
    its own, after lining the bone's rest direction up with the library's (a T-pose), so an A-pose rig
    lands in the same pose. Bones the pose doesn't name (hair, chest, ahoge) ride their parents.
    Every bone's place is worked out from its parent's, with no scene update in between: a pose of
    fifty bones on a heavy mesh was a minute of updates otherwise."""
    bones, posed, lined = armature.pose.bones, {}, {}
    for bone in bones:  # parents come before their children
        local, parent = bone.bone.matrix_local, bone.parent
        carried = posed[parent.name] @ parent.bone.matrix_local.inverted() @ local if parent else local.copy()
        wanted = carried
        if bone.name in pose["bones"]:
            side = "Left" if bone.name.startswith("Left") else "Right" if bone.name.startswith("Right") else ""
            part, line = bone.name[len(side):], Quaternion()
            at = armature.data.bones.get(side + POINTS_AT[part]) if part in POINTS_AT else None
            if at is not None and bone.name in rest:
                line = (at.head_local - bone.bone.head_local).normalized().rotation_difference(Vector(rest[bone.name]))
            elif parent is not None and parent.name in lined and part.endswith("Distal"):
                line = lined[parent.name]  # a fingertip has nothing to point at: as the joint before it
            lined[bone.name] = line
            wanted = (Quaternion(pose["bones"][bone.name]) @ line @ local.to_quaternion()).to_matrix().to_4x4()
            wanted.translation = carried.translation
            if bone.name == "Hips":  # a crouch or a jump: the hips' own move, in this rig's hip heights
                wanted.translation = local.translation + Vector(pose["hips"]) * bone.bone.head_local.z
        posed[bone.name] = wanted
        bone.matrix_basis = carried.inverted() @ wanted
    bpy.context.view_layer.update()


def reset(armature):
    for bone in armature.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def aim(armature, name, direction):
    """Swing a bone about its head so it points at its child along `direction`."""
    side, part = ("Left", name[4:]) if name.startswith("Left") else ("Right", name[5:])
    bones = armature.pose.bones
    if name not in bones or f"{side}{CHILD[part]}" not in bones:
        return
    bone, child = bones[name], bones[f"{side}{CHILD[part]}"]
    now, want = (child.head - bone.head).normalized(), Vector(direction).normalized()
    axis = now.cross(want)
    if axis.length > 1e-6:
        spin(bone, axis.normalized(), now.angle(want))


def make(armature, pose):
    """Put the rig in one of the poses above: the hips' sway, the limbs, the forearms' roll, the hands, the head."""
    if pose.get("sway"):
        turn(armature, "Hips", "Y", pose["sway"])
        turn(armature, "Spine", "Y", -pose["sway"] * 1.4)
        turn(armature, "Neck", "Y", pose["sway"] * 0.4)
    for bone, direction in pose.get("aim", {}).items():
        aim(armature, bone, direction)
    for bone, (what, facing) in pose.get("roll", {}).items():
        roll(armature, bone, what, facing)
    for side, shape in pose.get("hands", {}).items():
        hand(armature, side, shape)
    for bone, axis, degrees in pose.get("turn", []):
        turn(armature, bone, axis, degrees)


def spin(bone, axis, angle):
    """Turn a pose bone about an axis through its head, in the armature's space."""
    head = bone.head.copy()
    bone.matrix = Matrix.Translation(head) @ Matrix.Rotation(angle, 4, axis) @ Matrix.Translation(-head) @ bone.matrix
    bpy.context.view_layer.update()


def turn(armature, name, axis, degrees):
    if name in armature.pose.bones:
        spin(armature.pose.bones[name], axis, math.radians(degrees))


def palm(armature, side):
    """The hand as it is now posed: the way its palm faces, the way to its thumb's side, the way its
    fingers run. None for a rig without finger bones. Worked out on the rest pose, where a palm faces
    the floor (T) or the leg (A), and carried by the hand bone to where the pose has it."""
    rest, bones = armature.data.bones, armature.pose.bones
    need = [f"{side}{n}" for n in ("Hand", "IndexProximal", "MiddleProximal", "LittleProximal")]
    if any(n not in rest for n in need):
        return None
    wrist, index, middle, little = (rest[n].head_local for n in need)
    along, across = (middle - wrist).normalized(), (index - little).normalized()
    out = along.cross(across).normalized()
    if out.dot(Vector((-0.5 if side == "Left" else 0.5, 0.0, -1.0))) < 0:
        out = -out
    carried = bones[need[0]].matrix.to_quaternion() @ rest[need[0]].matrix_local.to_quaternion().inverted()
    return carried @ out, carried @ across, carried @ along


def roll(armature, name, what, facing):
    """Turn a forearm about its own length until the palm, or the thumb's side of the hand, faces `facing`
    as nearly as it can."""
    side = "Left" if name.startswith("Left") else "Right"
    frame, bones = palm(armature, side), armature.pose.bones
    if frame is None or name not in bones or f"{side}Hand" not in bones:
        return
    length = (bones[f"{side}Hand"].head - bones[name].head).normalized()
    flat = lambda v: (v - length * v.dot(length))  # noqa: E731: what is left of a direction across the forearm
    now, want = flat(frame[0] if what == "palm" else frame[1]), flat(Vector(facing).normalized())
    if now.length < 1e-4 or want.length < 1e-4:
        return
    angle = now.angle(want) * (1 if now.cross(want).dot(length) > 0 else -1)
    spin(bones[name], length, angle)


def hand(armature, side, shape):
    """Shape a hand: fingers curled toward the palm joint by joint, the thumb folded over them or held out."""
    frame, bones = palm(armature, side), armature.pose.bones
    if frame is None:
        return
    out, thumbward, along = frame
    for finger in ("Index", "Middle", "Ring", "Little"):
        bends = CURL["open" if shape == "point" and finger == "Index" else "fist" if shape in ("point", "thumb") else shape]
        first = bones.get(f"{side}{finger}Proximal")
        after = bones.get(f"{side}{finger}Intermediate")
        if first is None or after is None:
            continue
        # about the line across the finger, in the palm: turning that way takes the tip toward the palm
        axis = (after.head - first.head).normalized().cross(out).normalized()
        for joint, bend in zip(("Proximal", "Intermediate", "Distal"), bends):
            if bend and f"{side}{finger}{joint}" in bones:
                spin(bones[f"{side}{finger}{joint}"], axis, math.radians(bend))
    if shape == "thumb":
        # straight out to its own side, the fist's top once the forearm is rolled
        aim(armature, f"{side}ThumbMetacarpal", thumbward * 0.8 + along * 0.5 - out * 0.1)
        aim(armature, f"{side}ThumbProximal", thumbward + along * 0.1)
    elif shape in ("fist", "point"):
        # folded across the front of the curled fingers
        aim(armature, f"{side}ThumbMetacarpal", along * 0.6 + thumbward * 0.4 + out * 0.6)
        aim(armature, f"{side}ThumbProximal", along * 0.5 - thumbward * 0.5 + out * 0.7)


def face_width(armature, height):
    """How wide the face is, in the model's units: the spread, side to side, of what the head bone holds
    in a band just above its joint. Measured at rest. The usual share of the height when the model has
    no such bone or too little on it to tell."""
    if "Head" not in armature.data.bones:
        return FACE_WIDTH * height
    joint = armature.matrix_world @ armature.data.bones["Head"].head_local
    low, high = joint.z + FACE_BAND[0] * height, joint.z + FACE_BAND[1] * height
    across = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or not obj.visible_get() or "Head" not in obj.vertex_groups:
            continue
        group, world = obj.vertex_groups["Head"].index, obj.matrix_world
        for vertex in obj.data.vertices:
            if any(g.group == group and g.weight > 0.5 for g in vertex.groups):
                point = world @ vertex.co
                if low < point.z < high:
                    across.append(point.x)
    if len(across) < 50:
        return FACE_WIDTH * height
    across.sort()
    wide = across[int(len(across) * 0.97)] - across[int(len(across) * 0.03)]
    return min(max(wide, 0.09 * height), 0.32 * height)


def head_point(armature, face):
    """The middle of the face, where the pose has taken it, in world space."""
    bones = armature.pose.bones
    if "Head" not in bones:
        low, high = bounds()
        return Vector(((low.x + high.x) / 2, (low.y + high.y) / 2, low.z + 0.9 * (high.z - low.z)))
    rest = armature.data.bones["Head"]
    return armature.matrix_world @ (bones["Head"].matrix @ rest.matrix_local.inverted() @ (rest.head_local + Vector((0.0, 0.0, 0.5 * face))))


def bounds():
    """The lowest and highest corner of the meshes as they stand, from the evaluated (deformed) mesh."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    low, high = Vector((1e9, 1e9, 1e9)), Vector((-1e9, -1e9, -1e9))
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or not obj.visible_get():
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        for vertex in mesh.vertices:
            point = evaluated.matrix_world @ vertex.co
            low = Vector((min(low[i], point[i]) for i in range(3)))
            high = Vector((max(high[i], point[i]) for i in range(3)))
        evaluated.to_mesh_clear()
    return low, high


def render(filepath, location, rotation, span, size):
    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("portrait_view")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = span
    camera_data.clip_end = 100.0
    camera = bpy.data.objects.new("portrait_view", camera_data)
    scene.collection.objects.link(camera)
    camera.location = location
    camera.rotation_euler = rotation

    previous = scene.camera
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "TEXTURE"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = str(filepath)
    bpy.ops.render.render(write_still=True)

    scene.camera = previous
    bpy.data.objects.remove(camera)
    bpy.data.cameras.remove(camera_data)


main()
