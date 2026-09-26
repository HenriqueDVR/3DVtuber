"""Manual guides must drive real deformation, without accumulating rigs or losing weights."""
import json
import unittest

import bpy
import numpy as np
from mathutils import Matrix, Vector

from vtuber_maker.pipeline import fingers, hand_setup
from test_fingers import build_hand, deformed, WRIST, PALM, FINGER, SPREAD


def guides():
    result = {}
    for i, name in enumerate(("little", "ring", "middle", "index")):
        base = WRIST + Vector(((i - 1.5) * SPREAD, 0, PALM))
        result[name] = {"points": [list(base + Vector((0, 0, FINGER * t)))
                                   for t in (0, .45, .75, 1)],
                        "bend": list(base + Vector((0, -.04, 0))), "radius": .012}
    base = WRIST + Vector((.032, 0, .5 * PALM))
    result["thumb"] = {"points": [list(base + Vector((.022 * t, 0, .03 * t)))
                                  for t in (0, .45, .75, 1)],
                       "bend": list(base + Vector((0, -.04, 0))), "radius": .012}
    return result


class ManualHandSetup(unittest.TestCase):
    def test_build_preview_restore_and_repeat_preserve_mesh_and_weight_budget(self):
        mesh, arm = build_hand()
        mesh.shape_key_add(name="Basis")
        key = mesh.shape_key_add(name="Expression")
        key.data[0].co.y += .001
        original = [np.array([v.co[:] for v in k.data]) for k in mesh.data.shape_keys.key_blocks]
        rest = deformed(mesh)
        hand_setup.apply(mesh, arm, "left", guides())
        self.assertEqual(len(arm.data.bones), 16)
        np.testing.assert_allclose(deformed(mesh), rest, atol=1e-6)
        weights = [[g.weight for g in v.groups] for v in mesh.data.vertices]
        for row in weights:
            self.assertAlmostEqual(sum(row), 1, places=6)
        hand_setup.preview(arm, "left", "fist", .7)
        self.assertGreater(np.linalg.norm(deformed(mesh) - rest, axis=1).max(), .02)
        hand_setup.reset_preview(arm)
        np.testing.assert_allclose(deformed(mesh), rest, atol=1e-6)
        hand_setup.apply(mesh, arm, "left", guides())
        self.assertEqual(len(arm.data.bones), 16)
        for before, vertex in zip(weights, mesh.data.vertices):
            np.testing.assert_allclose(before, [g.weight for g in vertex.groups], atol=1e-6)
        for before, k in zip(original, mesh.data.shape_keys.key_blocks):
            np.testing.assert_array_equal(before, [v.co[:] for v in k.data])

    def test_selected_finger_preview_and_restoration_of_existing_pose(self):
        mesh, arm = build_hand()
        hand_setup.apply(mesh, arm, "left", guides())
        bone = arm.pose.bones[fingers.bone_name("left", "index", "proximal")]
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = Vector((0, 1, 0)).rotation_difference(Vector((.1, 1, 0)))
        before = bone.matrix_basis.copy()
        hand_setup.preview(arm, "left", "index", .5)
        self.assertGreater(abs(bone.rotation_euler.x), .1)
        other = arm.pose.bones[fingers.bone_name("left", "ring", "proximal")]
        np.testing.assert_allclose(other.matrix_basis, Matrix.Identity(4), atol=1e-6)
        hand_setup.reset_preview(arm)
        self.assertEqual(bone.rotation_mode, "QUATERNION")
        np.testing.assert_allclose(bone.matrix_basis, before, atol=1e-6)

    def test_invalid_guide_rejected_before_mutation(self):
        mesh, arm = build_hand()
        for field, value in (("radius", 0), ("bend", [float("nan"), 0, 0]),
                             ("points", [[0, 0, 0]] * 4)):
            data = guides()
            data["index"][field] = value
            with self.assertRaises(ValueError):
                hand_setup.apply(mesh, arm, "left", data)
            self.assertEqual(len(arm.data.bones), 1)
            self.assertEqual(len(mesh.vertex_groups), 1)

    def test_armature_transform_and_other_weights_are_respected(self):
        mesh, arm = build_hand()
        transform = Matrix.Translation((.2, -.1, .3)) @ Matrix.Rotation(.4, 4, "Y")
        arm.matrix_world = transform
        mesh.matrix_world = transform @ mesh.matrix_world
        bpy.context.view_layer.update()
        mesh.vertex_groups["LeftHand"].add(range(len(mesh.data.vertices)), .75, "REPLACE")
        other = mesh.vertex_groups.new(name="Unrelated")
        other.add(range(len(mesh.data.vertices)), .25, "REPLACE")
        hand_setup.apply(mesh, arm, "left", guides())
        bone = arm.data.bones[fingers.bone_name("left", "index", "proximal")]
        np.testing.assert_allclose(bone.head_local, guides()["index"]["points"][0], atol=1e-6)
        for v in mesh.data.vertices:
            self.assertAlmostEqual(sum(g.weight for g in v.groups), 1, places=6)
            self.assertAlmostEqual(other.weight(v.index), .25, places=6)

    def test_guides_persist_and_do_not_duplicate_on_resume(self):
        mesh, arm = build_hand()
        objects = hand_setup.create_guides(arm, "left")
        self.assertEqual(len(objects), 25)
        obj = objects[0]
        original_x = obj.location.x
        obj.location.x += .004
        bpy.context.view_layer.update()
        data = hand_setup.read_guides(arm, "left")
        self.assertAlmostEqual(data[obj["finger"]]["points"][obj["point"]][0], original_x + .004, places=6)
        self.assertEqual(len(hand_setup.create_guides(arm, "left")), 25)
        self.assertEqual(data, hand_setup.read_guides(arm, "left"))
        # Serializable data, not transient numpy/mathutils values.
        self.assertEqual(data, json.loads(json.dumps(data)))

    def test_saved_blend_retains_guides_and_can_restore_saved_preview(self):
        import tempfile
        from pathlib import Path
        mesh, arm = build_hand()
        hand_setup.write_guides(arm, "left", guides())
        hand_setup.apply(mesh, arm, "left", guides())
        data = hand_setup.read_guides(arm, "left")
        rest = deformed(mesh)
        names = mesh.name, arm.name
        hand_setup.preview(arm, "left", "point", .6)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "setup.blend")
            bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
            bpy.ops.wm.open_mainfile(filepath=path)
            mesh, arm = (bpy.data.objects[n] for n in names)
            self.assertEqual(data, hand_setup.read_guides(arm, "left"))
            hand_setup.reset_preview(arm)
            np.testing.assert_allclose(deformed(mesh), rest, atol=1e-6)

    def test_preview_combines_with_wrist_motion_and_expression(self):
        mesh, arm = build_hand()
        hand_setup.apply(mesh, arm, "left", guides())
        mesh.shape_key_add(name="Basis")
        key = mesh.shape_key_add(name="Expression")
        key.data[0].co.y += .001
        key.value = .6
        wrist = arm.pose.bones["LeftHand"]
        wrist.rotation_mode = "XYZ"
        wrist.rotation_euler.z = .3
        before = deformed(mesh)
        hand_setup.preview(arm, "left", "point", .6)
        self.assertGreater(np.linalg.norm(deformed(mesh) - before, axis=1).max(), .02)
        self.assertAlmostEqual(wrist.rotation_euler.z, .3, places=6)
        self.assertAlmostEqual(key.value, .6, places=6)
        hand_setup.reset_preview(arm)
        np.testing.assert_allclose(deformed(mesh), before, atol=1e-6)

    def test_mirror_uses_rig_space_and_requires_opposite_hand(self):
        mesh, arm = build_hand()
        hand_setup.create_guides(arm, "left")
        with self.assertRaises(ValueError):
            hand_setup.mirror_guides(arm, "left")
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode="EDIT")
        bone = arm.data.edit_bones.new("RightHand")
        bone.head, bone.tail = (0, 0, 1), (0, 0, 1.11)
        bpy.ops.object.mode_set(mode="OBJECT")
        hand_setup.mirror_guides(arm, "left")
        left, right = hand_setup.read_guides(arm, "left"), hand_setup.read_guides(arm, "right")
        for name in fingers.FINGERS:
            expected = np.array(left[name]["points"]) * [-1, 1, 1]
            np.testing.assert_allclose(right[name]["points"], expected, atol=1e-6)

    def test_registered_wizard_operators_use_actual_mesh(self):
        import vtuber_maker
        vtuber_maker.register()
        try:
            mesh, arm = build_hand()
            settings = bpy.context.scene.vtuber_hand_setup
            settings.mesh, settings.armature = mesh, arm
            self.assertEqual(bpy.ops.vtuber_maker.hand_action(action="START"), {"FINISHED"})
            hand_setup.write_guides(arm, "left", guides())
            self.assertEqual(bpy.ops.vtuber_maker.hand_action(action="APPLY"), {"FINISHED"})
            rest = deformed(mesh)
            self.assertEqual(bpy.ops.vtuber_maker.hand_action(action="FIST"), {"FINISHED"})
            self.assertGreater(np.linalg.norm(deformed(mesh) - rest, axis=1).max(), .02)
            self.assertEqual(bpy.ops.vtuber_maker.hand_action(action="RESET"), {"FINISHED"})
            np.testing.assert_allclose(deformed(mesh), rest, atol=1e-6)
        finally:
            vtuber_maker.unregister()

    def test_refuses_non_rest_pose_before_rebuilding(self):
        mesh, arm = build_hand()
        arm.pose.bones["LeftHand"].rotation_mode = "XYZ"
        arm.pose.bones["LeftHand"].rotation_euler.x = .2
        with self.assertRaises(ValueError):
            hand_setup.apply(mesh, arm, "left", guides())
        self.assertEqual(len(arm.data.bones), 1)


if __name__ == "__main__":
    unittest.main()
