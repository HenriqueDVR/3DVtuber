"""Render a rigged avatar .blend posed, with the inside of every surface painted red.

    blender -b in.blend -P scripts/render_inside.py -- out_dir [pose,pose,...]

Anything red is the back of a surface: the inside of a cape or sleeve, or the inside of the
body showing through a gap. VTubing apps draw single-sided materials only from the front, so
without double-sided materials those spots are see-through. Poses are the arm poses
check_avatar.py measures (rest, sides, tpose, forward, raised); each is shot from the front,
back, left, three-quarter, below and above, plus a chest close-up. Flat textures, no MToon,
so shading doesn't hide anything. Nothing is saved to the .blend.
"""
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import texture_space  # noqa: E402

# Where both arms point (shoulder to wrist, for the left arm; the right is mirrored).
POSES = {"rest": None, "sides": (0.25, 0, -1), "tpose": (1, 0, 0), "forward": (0.3, -1, -0.3), "raised": (0.5, 0, 1)}
# Camera direction from the body's middle.
VIEWS = {"front": (0, -1, 0), "back": (0, 1, 0), "left": (1, 0, 0), "q34": (0.7, -0.7, 0),
         "below": (0, -0.7, -0.7), "above": (0, -0.5, 0.87)}
RESOLUTION = 900


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    out = Path(argv[0]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    only = set(argv[1].split(",")) if len(argv) > 1 else set(POSES)

    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    obj = next(o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm)
    for other in bpy.data.objects:
        if other.type == "EMPTY":
            other.hide_render = True
    for slot in obj.material_slots:
        slot.material = inside_red(slot.material)

    scene = setup_scene()
    camera = scene.camera
    height = max((obj.matrix_world @ Vector(c)).z for c in obj.bound_box)
    middle = Vector((0, 0, 0.6 * height))
    chest = arm.matrix_world @ arm.pose.bones["UpperChest"].head if "UpperChest" in arm.pose.bones else middle
    for pose, direction in POSES.items():
        if pose not in only:
            continue
        reset(arm)
        if direction:
            point_arms(arm, direction)
        shots = [(view, middle, Vector(d), 1.2 * height) for view, d in VIEWS.items()]
        shots.append(("chest", chest, Vector((0, -1, 0.1)), 0.5 * height))
        for view, target, direction_to_camera, size in shots:
            camera.location = target + 3 * direction_to_camera.normalized()
            camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
            camera.data.ortho_scale = size
            scene.render.filepath = str(out / f"{pose}_{view}.png")
            bpy.ops.render.render(write_still=True)
    print("rendered to", out)


def inside_red(material):
    """A plain copy of the material: its colour texture on the front, flat red on the back."""
    image = None
    if material is not None:
        try:
            image = texture_space.material_base_color_image(material)
        except Exception:  # noqa: BLE001 - odd node setups just render untextured
            image = None
        if image is None:
            image = material.vrm_addon_extension.mtoon1.pbr_metallic_roughness.base_color_texture.index.source
    new = bpy.data.materials.new((material.name if material else "none") + "_inside")
    new.use_nodes = True
    nodes, links = new.node_tree.nodes, new.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    front = nodes.new("ShaderNodeBsdfDiffuse")
    if image is not None:
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = image
        links.new(texture.outputs["Color"], front.inputs["Color"])
    elif material is not None:
        front.inputs["Color"].default_value = material.diffuse_color
    back = nodes.new("ShaderNodeEmission")
    back.inputs["Color"].default_value = (1, 0, 0, 1)
    geometry = nodes.new("ShaderNodeNewGeometry")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(geometry.outputs["Backfacing"], mix.inputs["Fac"])
    links.new(front.outputs[0], mix.inputs[1])
    links.new(back.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])
    return new


def setup_scene():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = scene.render.resolution_y = RESOLUTION
    scene.view_settings.view_transform = "Standard"
    world = scene.world or bpy.data.worlds.new("inside")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.9, 0.9, 0.9, 1)
    background.inputs["Strength"].default_value = 1.6
    sun = bpy.data.objects.new("inside_sun", bpy.data.lights.new("inside_sun", "SUN"))
    sun.rotation_euler = (0.6, 0.2, 0.3)
    sun.data.energy = 2.5
    scene.collection.objects.link(sun)
    camera = bpy.data.objects.new("inside_camera", bpy.data.cameras.new("inside_camera"))
    camera.data.type = "ORTHO"
    scene.collection.objects.link(camera)
    scene.camera = camera
    return scene


def reset(arm):
    for bone in arm.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def turn(arm, name, axis, angle):
    """Rotate a bone about a world axis through its head."""
    bone = arm.pose.bones[name]
    head = bone.head.copy()
    bone.matrix = Matrix.Translation(head) @ Matrix.Rotation(angle, 4, axis) @ Matrix.Translation(-head) @ bone.matrix
    bpy.context.view_layer.update()


def point_arms(arm, direction):
    for side, mirror in (("Left", 1.0), ("Right", -1.0)):
        upper, hand = arm.pose.bones[f"{side}UpperArm"], arm.pose.bones[f"{side}Hand"]
        now = (hand.head - upper.head).normalized()
        want = Vector((mirror * direction[0], direction[1], direction[2])).normalized()
        axis = now.cross(want)
        if axis.length > 1e-6:
            turn(arm, f"{side}UpperArm", axis.normalized(), now.angle(want))


main()
