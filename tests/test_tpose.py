"""T-pose conversion must open curved fingers, not bake their sideways kinks/twist."""
import math
import unittest

import bpy
import numpy as np
from mathutils import Matrix, Vector

from vtuber_maker.pipeline import fingers, rig, tpose
from test_fingers import deformed


def curved_hand(side="left"):
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    mirror = 1 if side == "left" else -1
    data = bpy.data.armatures.new("HandRig")
    arm = bpy.data.objects.new("HandRig", data)
    bpy.context.scene.collection.objects.link(arm)
    rig.make_active(arm)
    bpy.ops.object.mode_set(mode="EDIT")
    hand = data.edit_bones.new(rig.BONE_NAMES[f"{side}_hand"])
    hand.head, hand.tail = (0, 0, 1), (mirror * .05, 0, 1)
    for i, finger in enumerate(fingers.FINGERS):
        parent = hand
        # The thumb sits below/behind the knuckles; it must NOT define the palm plane.
        head = Vector((mirror * (.025 if i == 0 else .05), (i - 2) * .02,
                       .98 if i == 0 else 1.0))
        for j, segment in enumerate(fingers._segments(finger)):
            bone = data.edit_bones.new(fingers.bone_name(side, finger, segment))
            # Curved fingers with small sideways bends and differing roll from path detection.
            direction = Vector((mirror, (-.12 if i < 3 else .12) + j * .2, -.4 - j * .4)).normalized()
            bone.head, bone.tail = head, head + direction * .018
            bone.parent, bone.use_connect = parent, j > 0
            bone.roll = .3 * j
            head, parent = bone.tail.copy(), bone
    bpy.ops.object.mode_set(mode="OBJECT")
    vertices, edges, owners = [], [], []
    for bone in data.bones:
        for at in (0, .5, 1):
            centre = bone.head_local.lerp(bone.tail_local, at)
            vertices.extend([centre + Vector((0, 0, -.002)), centre + Vector((0, 0, .002))])
            owners.extend([bone.name] * 2)
            edges.append((len(vertices) - 2, len(vertices) - 1))
    mesh_data = bpy.data.meshes.new("hand")
    mesh_data.from_pydata(vertices, edges, [])
    mesh = bpy.data.objects.new("hand", mesh_data)
    bpy.context.scene.collection.objects.link(mesh)
    for name in set(owners):
        group = mesh.vertex_groups.new(name=name)
        group.add([i for i, owner in enumerate(owners) if owner == name], 1, "REPLACE")
    mesh.modifiers.new("Armature", "ARMATURE").object = arm
    mesh.shape_key_add(name="Basis")
    key = mesh.shape_key_add(name="Tracking")
    for point in key.data:
        point.co.z += .001
    bpy.context.view_layer.update()
    return mesh, arm


class TPoseFingers(unittest.TestCase):
    def test_cli_exposes_preserving_mode_without_running_on_import(self):
        import runpy
        from pathlib import Path
        script = Path(__file__).resolve().parents[1] / "scripts" / "to_tpose.py"
        namespace = runpy.run_path(str(script), run_name="tpose_cli_test")
        args = namespace["parse_args"](["out.blend", "out.vrm", "--keep-finger-pose"])
        self.assertTrue(args.keep_finger_pose)

    def test_preserving_mode_moves_curved_hand_rigidly_with_arm(self):
        mesh, arm = curved_hand()
        rig.make_active(arm)
        bpy.ops.object.mode_set(mode="EDIT")
        upper = arm.data.edit_bones.new("LeftUpperArm")
        upper.head, upper.tail = (-.1, 0, 1.2), (0, 0, 1)
        arm.data.edit_bones["LeftHand"].parent = upper
        bpy.ops.object.mode_set(mode="OBJECT")
        before = deformed(mesh)
        original_keys = {k.name: np.array([p.co[:] for p in k.data]) for k in mesh.data.shape_keys.key_blocks}
        tpose._point_arm(arm, "LeftUpperArm", "LeftHand", Vector((1, 0, 0)))
        transform = arm.pose.bones["LeftHand"].matrix @ arm.data.bones["LeftHand"].matrix_local.inverted()
        rotation, translation = np.array(transform)[:3, :3], np.array(transform)[:3, 3]
        for bone in arm.pose.bones:
            bone.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()
        tpose.to_t_pose(mesh, arm, straighten=False)
        np.testing.assert_allclose(deformed(mesh), before @ rotation.T + translation, atol=2e-6)
        for key in mesh.data.shape_keys.key_blocks:
            np.testing.assert_allclose([p.co[:] for p in key.data],
                                       original_keys[key.name] @ rotation.T + translation, atol=2e-6)
        # A rigid arm raise must not change any distances within the curved hand.
        np.testing.assert_allclose(np.linalg.norm(before[1:] - before[:-1], axis=1),
                                   np.linalg.norm(deformed(mesh)[1:] - deformed(mesh)[:-1], axis=1), atol=2e-6)

    def test_curved_finger_segments_share_a_hinge_without_rotating_the_mesh(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                mesh, arm = curved_hand(side)
                before = deformed(mesh)
                tpose.to_t_pose(mesh, arm, straighten=False)
                np.testing.assert_allclose(deformed(mesh), before, atol=2e-6)
                along = (arm.pose.bones[side.capitalize() + "Hand"].tail -
                         arm.pose.bones[side.capitalize() + "Hand"].head).normalized()
                across = tpose._across_palm(arm, side, along)
                hinge = across * (-1 if side == "left" else 1)
                for finger in fingers.FINGERS:
                    for segment in fingers._segments(finger):
                        bone = arm.pose.bones[fingers.bone_name(side, finger, segment)]
                        tangent = (bone.tail - bone.head).normalized()
                        expected = (hinge - tangent * hinge.dot(tangent)).normalized()
                        self.assertGreater(bone.matrix.to_3x3().col[0].dot(expected), .9999, bone.name)

    def test_palm_plane_uses_knuckles_not_lower_thumb_base(self):
        _, arm = curved_hand()
        across = tpose._across_palm(arm, "left", Vector((1, 0, 0)))
        self.assertLess(abs(across.z), 1e-5)
        self.assertLess(across.y, -.99)

    def test_all_segments_extend_in_one_direction_preserving_each_finger_fan(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                mesh, arm = curved_hand(side)
                directions = {}
                for finger in fingers.FINGERS[1:]:
                    bone = arm.pose.bones[fingers.bone_name(side, finger, "proximal")]
                    v = bone.tail - bone.head
                    v.z = 0
                    directions[finger] = v.normalized()
                tpose.to_t_pose(mesh, arm)
                for finger, want in directions.items():
                    for segment in fingers._segments(finger):
                        bone = arm.data.bones[fingers.bone_name(side, finger, segment)]
                        got = (bone.tail_local - bone.head_local).normalized()
                        self.assertGreater(got.dot(want), .999, bone.name)

    def test_baked_fingers_have_consistent_hinges_and_curl_toward_palm(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                mesh, arm = curved_hand(side)
                tpose.to_t_pose(mesh, arm)
                for finger in fingers.FINGERS[1:]:
                    chain = [arm.pose.bones[fingers.bone_name(side, finger, s)]
                             for s in fingers._segments(finger)]
                    for bone in chain:
                        self.assertLess(abs(bone.matrix.to_3x3().col[0].z), 1e-4, bone.name)
                    before = chain[-1].tail.copy()
                    for bone in chain:
                        bone.rotation_mode = "XYZ"
                        bone.rotation_euler.x = math.radians(20)
                    bpy.context.view_layer.update()
                    self.assertLess(chain[-1].tail.z, before.z - .01)
                    for bone in chain:
                        bone.matrix_basis = Matrix.Identity(4)
                    bpy.context.view_layer.update()

    def test_bake_matches_blender_with_shape_keys_and_is_idempotent(self):
        mesh, arm = curved_hand()
        tpose.straighten_fingers(arm)
        key = mesh.data.shape_keys.key_blocks["Tracking"]
        key.value = 0
        expected_basis = deformed(mesh)
        key.value = 1
        expected_key = deformed(mesh)
        key.value = 0
        for bone in arm.pose.bones:
            bone.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()
        tpose.to_t_pose(mesh, arm)
        np.testing.assert_allclose(deformed(mesh), expected_basis, atol=2e-6)
        key.value = 1
        np.testing.assert_allclose(deformed(mesh), expected_key, atol=2e-6)
        key.value = 0
        tpose.to_t_pose(mesh, arm)
        np.testing.assert_allclose(deformed(mesh), expected_basis, atol=2e-6)

    def test_spec_hands_bake_turns_joints_like_blenders_preserve_volume(self):
        mesh, arm = curved_hand()
        # every vertex half on the hand bone, so straightening blends two different rotations
        mesh.vertex_groups[rig.BONE_NAMES["left_hand"]].add(range(len(mesh.data.vertices)), .5, "REPLACE")
        modifier = mesh.modifiers["Armature"]
        modifier.use_deform_preserve_volume = True
        tpose.spec_hands(arm)
        expected = deformed(mesh)
        modifier.use_deform_preserve_volume = False
        linear = deformed(mesh)
        self.assertGreater(float(np.abs(linear - expected).max()), 1e-3, "the blends should differ here")
        tpose._bake([mesh], arm, True, dual=True)
        np.testing.assert_allclose(deformed(mesh), expected, atol=2e-6)

    def test_straightening_turns_imported_normals_with_the_mesh(self):
        from test_finger_guides import build
        from vtuber_maker.pipeline import finger_guides
        mesh, arm, _ = build(math.radians(35))
        finger_guides.rebuild(mesh, arm, sides=("left",))
        data = mesh.data
        # imported meshes carry their own shading normals
        data.normals_split_custom_set_from_vertices([v.normal[:] for v in data.vertices])
        before = np.zeros((len(data.loops), 3))
        data.corner_normals.foreach_get("vector", before.ravel())
        name = fingers.bone_name("left", "index", "distal")
        group = mesh.vertex_groups[name].index
        rigid = np.array([len(v.groups) == 1 and v.groups[0].group == group and v.groups[0].weight > .999
                          for v in data.vertices])
        tpose.spec_hands(arm)
        bone = arm.pose.bones[name]
        turn = np.array((bone.matrix @ bone.bone.matrix_local.inverted()).to_3x3())
        tpose._bake([mesh], arm, True, dual=True)
        after = np.zeros((len(data.loops), 3))
        data.corner_normals.foreach_get("vector", after.ravel())
        loops = np.array([loop.vertex_index for loop in data.loops])
        hit = rigid[loops]
        self.assertGreater(int(hit.sum()), 20)
        cosines = np.einsum("ij,ij->i", before[hit] @ turn.T, after[hit])
        self.assertLess(float(np.degrees(np.arccos(np.clip(cosines.min(), -1, 1)))), 1.0)

    def test_spec_thumb_points_like_the_reference_hand(self):
        _, arm = curved_hand()
        tpose.spec_hands(arm)
        for segment in fingers._segments("thumb"):
            bone = arm.pose.bones[fingers.bone_name("left", "thumb", segment)]
            direction = (bone.tail - bone.head).normalized()
            self.assertAlmostEqual(math.degrees(math.asin(-direction.z)), tpose.REF_THUMB_DOWN, places=3)
            self.assertAlmostEqual(math.degrees(math.atan2(-direction.y, direction.x)), tpose.REF_THUMB_OUT, places=3)


if __name__ == "__main__":
    unittest.main()
