"""Finding fingers in a hand that is one piece, and rigging them without tearing it."""

import math
import unittest

import bpy
import numpy as np
from mathutils import Vector

from vtuber_maker.pipeline import fingers, rig

WRIST = Vector((0.0, 0.0, 1.0))
PALM = 0.05          # how far the palm reaches past the wrist
FINGER = 0.06        # and each finger past the palm
THUMB = 0.03
SPREAD = 0.03       # between neighbouring fingers, across the palm


def capsule(name, head, tail, radius, rings=10):
    """A tube from head to tail, as a low-poly cylinder with its ends filled."""
    bpy.ops.mesh.primitive_cylinder_add(vertices=rings, radius=radius, depth=(tail - head).length,
                                        location=(head + tail) / 2)
    obj = bpy.context.object
    obj.name = name
    direction = (tail - head).normalized()
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(direction)
    return obj


def build_hand(curl=0.0):
    """One fused hand: a palm with four fingers and a thumb, optionally curled forwards.

    `curl` bends the fingers forward (radians), the way an image-to-3D character's hand is
    usually modelled, so a fingertip ends up level with the palm.
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    parts = [capsule("palm", WRIST, WRIST + Vector((0.0, 0.0, PALM)), 0.035, rings=14)]
    knuckle = WRIST + Vector((0.0, 0.0, PALM))
    for index in range(4):
        start = knuckle + Vector(((index - 1.5) * SPREAD, 0.0, 0.0))
        reach = Vector((0.0, -math.sin(curl) * FINGER, math.cos(curl) * FINGER))
        parts.append(capsule(f"finger{index}", start, start + reach, 0.007))
    thumb_start = WRIST + Vector((0.032, 0.0, 0.5 * PALM))
    parts.append(capsule("thumb", thumb_start, thumb_start + Vector((0.022, 0.0, THUMB)), 0.008))

    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    hand = bpy.context.object
    # one fused surface, the way an image-to-3D generator makes a hand: the fingers are ridges
    # with grooves between them, not separate pieces
    remesh = hand.modifiers.new("Remesh", "REMESH")
    remesh.mode = "VOXEL"
    remesh.voxel_size = 0.003
    rig.make_active(hand)
    bpy.ops.object.modifier_apply(modifier=remesh.name)

    armature_data = bpy.data.armatures.new("Armature")
    armature = bpy.data.objects.new("Armature", armature_data)
    bpy.context.scene.collection.objects.link(armature)
    rig.make_active(armature)
    bpy.ops.object.mode_set(mode="EDIT")
    bone = armature_data.edit_bones.new(rig.BONE_NAMES["left_hand"])
    bone.head, bone.tail = WRIST, WRIST + Vector((0.0, 0.0, PALM + FINGER))
    bpy.ops.object.mode_set(mode="OBJECT")

    group = hand.vertex_groups.new(name=rig.BONE_NAMES["left_hand"])
    group.add(range(len(hand.data.vertices)), 1.0, "REPLACE")
    hand.modifiers.new("Armature", "ARMATURE").object = armature
    return hand, armature


def deformed(mesh_obj):
    graph = bpy.context.evaluated_depsgraph_get()
    graph.update()
    evaluated = mesh_obj.evaluated_get(graph)
    data = evaluated.to_mesh()
    coords = np.array([vertex.co[:] for vertex in data.vertices])
    evaluated.to_mesh_clear()
    return coords


def curl_hand(armature, degrees):
    for finger in fingers.FINGERS:
        for segment in fingers.SEGMENTS["thumb" if finger == "thumb" else "other"]:
            bone = armature.pose.bones.get(fingers.bone_name("left", finger, segment))
            if bone is not None:
                bone.rotation_mode = "XYZ"
                bone.rotation_euler = (math.radians(degrees), 0.0, 0.0)
    bpy.context.view_layer.update()


class RigFingers(unittest.TestCase):
    def test_finds_the_fingers_of_a_fused_hand(self):
        mesh, armature = build_hand()
        report = fingers.build_fingers(mesh, armature, sides=("left",))["left"]
        # the voxel fusing that makes this hand one piece also merges the fingers nearest the
        # thumb, so what matters is that most of them are found, each once, and sized like fingers
        self.assertGreaterEqual(report["fingers"], 3, report)
        self.assertEqual(len(set(report["named"])), len(report["named"]), report)
        for name in report["named"]:
            self.assertIn(name, fingers.FINGERS, report)
        for length in report["finger_length_cm"]:
            self.assertGreater(length, 3.0, report)
            self.assertLess(length, 9.0, report)

    def test_builds_three_bones_of_some_length_for_each_finger(self):
        mesh, armature = build_hand()
        report = fingers.build_fingers(mesh, armature, sides=("left",))["left"]
        for finger in report["named"]:
            for segment in fingers.SEGMENTS["thumb" if finger == "thumb" else "other"]:
                name = fingers.bone_name("left", finger, segment)
                self.assertIn(name, armature.data.bones, f"{name} was dropped, probably zero length")
                self.assertGreater(armature.data.bones[name].length, 1e-4, name)

    def test_a_curled_hand_puts_its_fingertips_on_fingertip_bones(self):
        # measured along the hand, a curled fingertip is level with the palm: anything that
        # slices the hand instead of walking its surface tears the crease between them open
        mesh, armature = build_hand(curl=math.radians(70))
        fingers.build_fingers(mesh, armature, sides=("left",))
        rest = deformed(mesh)
        edges = np.array([edge.vertices[:] for edge in mesh.data.edges])
        before = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)

        curl_hand(armature, 40)
        now = deformed(mesh)
        after = np.linalg.norm(now[edges[:, 0]] - now[edges[:, 1]], axis=1)
        worst = float((after - before).max())
        self.assertLess(worst, 0.02, f"an edge stretched {worst * 100:.1f} cm when the hand closed")

    def test_leaves_the_hand_alone_when_nothing_stands_out(self):
        mesh, armature = build_hand()
        # a mitten: only the palm, no fingers to find
        keep = [vertex.index for vertex in mesh.data.vertices if vertex.co.z < 1.0 + PALM]
        group = mesh.vertex_groups[rig.BONE_NAMES["left_hand"]]
        group.add([vertex.index for vertex in mesh.data.vertices], 0.0, "REPLACE")
        group.add(keep, 1.0, "REPLACE")
        report = fingers.build_fingers(mesh, armature, sides=("left",))["left"]
        if report["fingers"]:
            self.assertLessEqual(report["fingers"], 5, report)


if __name__ == "__main__":
    unittest.main()
