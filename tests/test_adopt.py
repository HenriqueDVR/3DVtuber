"""Taking a rig made elsewhere: which bone is which under each tool's names, and where the
weights of bones we have no name for go."""
import unittest

import bpy
import numpy as np

from support import load_fixture
from vtuber_maker.pipeline import adopt, fingers, landmarks, meshdata, rig
from vtuber_maker.pipeline.rig import BONE_NAMES

# How three other tools name the bones our fixture rig calls Hips, LeftUpperArm, ...
MIXAMO = {"hips": "Hips", "spine": "Spine", "chest": "Spine1", "upper_chest": "Spine2", "neck": "Neck", "head": "Head",
          "shoulder": "Shoulder", "upper_arm": "Arm", "lower_arm": "ForeArm", "hand": "Hand",
          "upper_leg": "UpLeg", "lower_leg": "Leg", "foot": "Foot"}
CREATOR = {"hips": "Hip", "spine": "Waist", "chest": "Spine01", "upper_chest": "Spine02", "neck": "NeckTwist01", "head": "Head",
           "shoulder": "Clavicle", "upper_arm": "Upperarm", "lower_arm": "Forearm", "hand": "Hand",
           "upper_leg": "Thigh", "lower_leg": "Calf", "foot": "Foot"}


def theirs(key, style):
    """Our bone key ("left_upper_arm") as Mixamo or Character Creator names that bone."""
    side, _, part = key.partition("_") if key.startswith(("left_", "right_")) else ("", "", key)
    if style == "mixamo":
        return f"mixamorig:{side.capitalize()}{MIXAMO[part]}"
    return f"CC_Base_{side[0].upper() + '_' if side else ''}{CREATOR[part]}"


class NamesOfOtherRigs(unittest.TestCase):
    def test_each_tools_names_map_to_ours(self):
        for style in ("mixamo", "creator"):
            found = [theirs(key, style) for key in BONE_NAMES]
            table = adopt.names(found)
            with self.subTest(style=style):
                self.assertEqual({theirs(key, style): ours for key, ours in BONE_NAMES.items()},
                                 {name: table[name] for name in found})

    def test_fingers_count_from_the_palm(self):
        table = adopt.names(["mixamorig:Hips", "mixamorig:LeftHandIndex1", "mixamorig:LeftHandThumb3", "mixamorig:RightHandPinky2"])
        self.assertEqual(table["mixamorig:LeftHandIndex1"], fingers.bone_name("left", "index", "proximal"))
        self.assertEqual(table["mixamorig:LeftHandThumb3"], fingers.bone_name("left", "thumb", "distal"))
        self.assertEqual(table["mixamorig:RightHandPinky2"], fingers.bone_name("right", "little", "intermediate"))
        table = adopt.names(["CC_Base_Hip", "CC_Base_R_Mid1", "CC_Base_L_Thumb1"])
        self.assertEqual(table["CC_Base_R_Mid1"], fingers.bone_name("right", "middle", "proximal"))
        self.assertEqual(table["CC_Base_L_Thumb1"], fingers.bone_name("left", "thumb", "metacarpal"))

    def test_a_rig_under_our_own_names_and_tripos_still_read(self):
        ours = list(BONE_NAMES.values())
        self.assertEqual(adopt.names(ours)["LeftUpperArm"], "LeftUpperArm")
        vroid = adopt.names(["J_Bip_C_Hips", "J_Bip_L_UpperArm", "J_Bip_R_Index2"])
        self.assertEqual((vroid["J_Bip_C_Hips"], vroid["J_Bip_L_UpperArm"]), ("Hips", "LeftUpperArm"))
        self.assertEqual(adopt.names(), adopt.names(["J_Bip_C_Hips"]))  # no names given: Tripo's, as before

    def test_names_no_tool_uses_are_refused_by_name(self):
        with self.assertRaisesRegex(ValueError, "LeftUpperArm.*Mixamo"):
            adopt.required(adopt.names(["Bone", "Bone.001"]), ["Bone", "Bone.001"], "rig.fbx")


class ReadMixamoRig(unittest.TestCase):
    """The fixture character rigged by us, its bones renamed as Mixamo names them, plus a twist
    bone under the forearm holding half the forearm's weight (as Character Creator rigs do)."""

    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture()
        joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))
        armature = rig.build_armature(joints)
        rig.skin(mesh, armature)
        cls.before = cls.weights(mesh)
        cls.heads = {b.name: np.array(armature.matrix_world @ b.head_local) for b in armature.data.bones}
        rig.make_active(armature)
        bpy.ops.object.mode_set(mode="EDIT")
        forearm = armature.data.edit_bones[BONE_NAMES["left_lower_arm"]]
        twist = armature.data.edit_bones.new("ForeArmTwist")
        twist.head, twist.tail, twist.parent = (forearm.head + forearm.tail) / 2, forearm.tail, forearm
        bpy.ops.object.mode_set(mode="OBJECT")
        group, moved = mesh.vertex_groups[BONE_NAMES["left_lower_arm"]], mesh.vertex_groups.new(name="ForeArmTwist")
        for v in mesh.data.vertices:
            for g in v.groups:
                if g.group == group.index and g.weight > 0:
                    moved.add([v.index], g.weight / 2, "REPLACE")
                    group.add([v.index], g.weight / 2, "REPLACE")
        for key, ours in BONE_NAMES.items():   # renaming a bone renames its vertex group with it
            armature.data.bones[ours].name = theirs(key, "mixamo")
        cls.bones, cls.surface, (cls.table, cls.order) = adopt.read(armature, [mesh], "fixture")

    @staticmethod
    def weights(mesh):
        names = {g.index: g.name for g in mesh.vertex_groups}
        table = {name: np.zeros(len(mesh.data.vertices)) for name in names.values()}
        for v in mesh.data.vertices:
            for g in v.groups:
                table[names[g.group]][v.index] = g.weight
        return table

    def test_the_skeleton_comes_out_under_our_names_with_our_parents(self):
        self.assertEqual(set(self.bones), set(BONE_NAMES.values()))
        self.assertEqual(self.bones["LeftLowerArm"][2], "LeftUpperArm")
        self.assertIsNone(self.bones["Hips"][2])

    def test_joints_keep_their_places_relative_to_each_other(self):
        # the reader rescales and moves the rig as normalize.prepare does the mesh: shape is kept
        ours = np.array([self.heads[name] for name in BONE_NAMES.values()])
        read = np.array([self.bones[name][0] for name in BONE_NAMES.values()])
        scale = np.ptp(read[:, 2]) / np.ptp(ours[:, 2])
        np.testing.assert_allclose(read - read[0], scale * (ours - ours[0]), atol=1e-4)

    def test_a_twist_bones_weights_go_to_the_bone_it_hangs_from(self):
        self.assertNotIn("ForeArmTwist", self.order)
        column = self.order.index("LeftLowerArm")
        total = sum(self.before.values())
        np.testing.assert_allclose(self.table[:, column], self.before["LeftLowerArm"] / np.maximum(total, 1e-9), atol=1e-5)
        np.testing.assert_allclose(self.table.sum(axis=1)[total > 0], 1.0, atol=1e-6)


class ArmsOffTheHead(unittest.TestCase):
    """An adopted rig's arm weights on the scalp: Tripo's leak (up to 14% on model A's T-pose) goes,
    a piece really riding the arm across the head's height (model C's twin-tail, 46%) stays."""

    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture()
        joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))
        cls.armature = rig.build_armature(joints)
        rig.skin(mesh, cls.armature)
        cls.verts = meshdata.world_vertices(mesh)
        cls.head_z = (cls.armature.matrix_world @ cls.armature.data.bones["Head"].head_local).z
        above = np.flatnonzero(cls.verts[:, 2] > cls.head_z - 0.05)
        cls.riding = cls.verts[:, 0] < 0
        # the right side of the head rides the arm by half, the rest leaks a tenth to it
        arm = mesh.vertex_groups.get("LeftUpperArm") or mesh.vertex_groups.new(name="LeftUpperArm")
        for i in above:
            share = 0.5 if cls.riding[i] else 0.1
            for g in mesh.data.vertices[int(i)].groups:
                mesh.vertex_groups[g.group].add([int(i)], g.weight * (1 - share), "REPLACE")
            arm.add([int(i)], share, "REPLACE")
        cls.before = ReadMixamoRig.weights(mesh)
        cls.changed = adopt.keep_arms_off_head(mesh, cls.armature)
        cls.after = ReadMixamoRig.weights(mesh)
        cls.arm_after = sum(w for n, w in cls.after.items() if n.removeprefix("Left").removeprefix("Right").startswith(adopt._ARM_WORDS))
        cls.arm_before = sum(w for n, w in cls.before.items() if n.removeprefix("Left").removeprefix("Right").startswith(adopt._ARM_WORDS))

    def test_a_leak_on_the_head_goes(self):
        # the fixture's own skinning gives a few of them more of the arm: those count as riding it
        clear = (self.verts[:, 2] > self.head_z + adopt.HEAD_FADE) & ~self.riding & (self.arm_before <= adopt.HEAD_LEAK + 1e-6)
        self.assertGreater(int(clear.sum()), 100)
        self.assertLess(float(self.arm_after[clear].max()), 1e-6)

    def test_a_piece_riding_the_arm_keeps_it(self):
        riding = (self.verts[:, 2] > self.head_z) & self.riding
        np.testing.assert_allclose(self.arm_after[riding], self.arm_before[riding], atol=1e-6)

    def test_it_fades_in_above_the_head_joint(self):
        below = self.verts[:, 2] <= self.head_z
        np.testing.assert_allclose(self.arm_after[below], self.arm_before[below], atol=1e-6)
        band = (self.verts[:, 2] > self.head_z) & (self.verts[:, 2] < self.head_z + adopt.HEAD_FADE / 2) & ~self.riding
        if band.any():
            self.assertTrue((self.arm_after[band] > 1e-6).all())

    def test_weights_still_sum_to_one(self):
        total = sum(self.after.values())
        changed = np.abs(self.arm_after - self.arm_before) > 1e-6
        self.assertTrue(changed.any())
        np.testing.assert_allclose(total[changed], 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
