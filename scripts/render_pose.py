"""Render a rigged avatar from four sides in the pose you ask for.

    blender -b in.blend -P scripts/render_pose.py -- out_dir [--pose t-pose] [--size 1024]

`check_avatar.py` says how far things move wrong; this shows what that looks like. The poses
are the ones VTubing apps actually use: `t-pose` (what every VRM app starts from), `rest` as
modelled, `hang`, `forward` and `raised`. Views are flat-lit and orthographic, so two builds
of the same character can be put side by side and compared pixel by pixel.

Writes `<pose>_front.png`, `_left.png`, `_back.png`, `_right.png` (left is the character's).
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

# Where both arms point (shoulder to wrist, for the left arm; the right is mirrored),
# the same set check_avatar.py measures.
POSES = {
    "rest": None,
    "t-pose": (1, 0, 0),
    "hang": (0.25, 0, -1),
    "forward": (0.3, -1, -0.3),
    "raised": (0.5, 0, 1),
}
# View name -> (camera direction from the model, camera rotation).
VIEWS = {
    "front": ((0.0, -1.0, 0.0), (math.pi / 2, 0.0, 0.0)),
    "left": ((1.0, 0.0, 0.0), (math.pi / 2, 0.0, math.pi / 2)),
    "back": ((0.0, 1.0, 0.0), (math.pi / 2, 0.0, math.pi)),
    "right": ((-1.0, 0.0, 0.0), (math.pi / 2, 0.0, -math.pi / 2)),
}
MARGIN = 1.08


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--pose", choices=sorted(POSES), default="t-pose")
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    direction = POSES[args.pose]
    if direction is not None:
        point_arms(armature, direction)
    bpy.context.view_layer.update()

    centre, extent = bounds()
    for view, (offset, rotation) in VIEWS.items():
        render(out_dir / f"{args.pose}_{view}.png", centre, extent, offset, rotation, args.size)
        print(f"RENDER {args.pose} {view} -> {out_dir / f'{args.pose}_{view}.png'}")


def point_arms(armature, direction):
    """Swing both upper arms so shoulder-to-wrist points along `direction` (mirrored on the right)."""
    for side, mirror in (("Left", 1.0), ("Right", -1.0)):
        if f"{side}UpperArm" not in armature.pose.bones:
            continue
        upper, hand = armature.pose.bones[f"{side}UpperArm"], armature.pose.bones[f"{side}Hand"]
        now = (hand.head - upper.head).normalized()
        want = Vector((mirror * direction[0], direction[1], direction[2])).normalized()
        axis = now.cross(want)
        if axis.length < 1e-6:
            continue
        head = upper.head.copy()
        turn = Matrix.Rotation(now.angle(want), 4, axis.normalized())
        upper.matrix = Matrix.Translation(head) @ turn @ Matrix.Translation(-head) @ upper.matrix
        bpy.context.view_layer.update()


def bounds():
    """Centre and size of the posed meshes, from the evaluated (deformed) mesh."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    low = Vector((1e9, 1e9, 1e9))
    high = Vector((-1e9, -1e9, -1e9))
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
    return (low + high) / 2, float(max(high - low)) * MARGIN


def render(filepath, centre, extent, offset, rotation, size):
    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("pose_view")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = extent
    camera_data.clip_end = 100.0
    camera = bpy.data.objects.new("pose_view", camera_data)
    scene.collection.objects.link(camera)
    camera.location = centre + Vector(offset) * 10.0
    camera.rotation_euler = rotation

    previous = scene.camera
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "TEXTURE"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = str(filepath)
    bpy.ops.render.render(write_still=True)

    scene.camera = previous
    bpy.data.objects.remove(camera)
    bpy.data.cameras.remove(camera_data)


main()
