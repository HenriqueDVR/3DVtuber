"""Building a VRM humanoid armature and skinning the fused mesh to it."""

import math
import unittest

import bpy
import numpy as np

from support import load_fixture
from vtuber_maker.pipeline import landmarks, meshdata, rig

REQUIRED_VRM_BONES = (
    "hips",
    "spine",
    "head",
    "left_upper_leg",
    "left_lower_leg",
    "left_foot",
    "right_upper_leg",
    "right_lower_leg",
    "right_foot",
    "left_upper_arm",
    "left_lower_arm",
    "left_hand",
    "right_upper_arm",
    "right_lower_arm",
    "right_hand",
)


def posed_world_vertices(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    coords = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", coords)
    evaluated.to_mesh_clear()
    matrix = np.array(obj.matrix_world)
    return coords.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]


class SkinnedSyntheticCharacter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh, cls.truth = load_fixture()
        joints = landmarks.detect_body(meshdata.world_vertices(cls.mesh), meshdata.triangles(cls.mesh.data))
        cls.armature = rig.build_armature(joints)
        rig.skin(cls.mesh, cls.armature)
        cls.rest = meshdata.world_vertices(cls.mesh)

    def tearDown(self):
        for bone in self.armature.pose.bones:
            bone.rotation_mode = "XYZ"
            bone.rotation_euler = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()

    def near(self, point, radius=0.03):
        return np.linalg.norm(self.rest - np.asarray(point), axis=1) < radius

    def displacement_after_bending(self, bone_key, degrees):
        bone = self.armature.pose.bones[rig.BONE_NAMES[bone_key]]
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = (math.radians(degrees), 0.0, 0.0)
        bpy.context.view_layer.update()
        return np.linalg.norm(posed_world_vertices(self.mesh) - self.rest, axis=1)

    def test_has_every_required_vrm_bone(self):
        names = set(self.armature.data.bones.keys())
        for key in REQUIRED_VRM_BONES:
            with self.subTest(bone=key):
                self.assertIn(rig.BONE_NAMES[key], names)

    def test_hand_chain_hangs_from_the_spine(self):
        chain = []
        bone = self.armature.data.bones[rig.BONE_NAMES["left_hand"]]
        while bone:
            chain.append(bone.name)
            bone = bone.parent
        self.assertEqual(chain[-1], rig.BONE_NAMES["hips"])
        self.assertIn(rig.BONE_NAMES["left_upper_arm"], chain)
        self.assertIn(rig.BONE_NAMES["spine"], chain)

    def test_every_vertex_weights_sum_to_one(self):
        totals = np.array([sum(g.weight for g in v.groups) for v in self.mesh.data.vertices])
        bad = int(np.sum(~np.isclose(totals, 1.0, atol=0.02)))
        self.assertEqual(bad, 0, f"{bad} of {len(totals)} vertices have weights not summing to 1")

    def test_bending_left_elbow_moves_only_the_left_hand(self):
        moved = self.displacement_after_bending("left_lower_arm", 60)
        self.assertGreater(moved[self.near(self.truth["joints"]["left_hand"])].mean(), 0.1)
        self.assertLess(moved[self.near(self.truth["joints"]["right_hand"])].max(), 0.002)
        self.assertLess(moved[self.near(self.truth["face"]["left_eye"])].max(), 0.002)

    def test_nodding_moves_the_face_but_not_the_hands(self):
        moved = self.displacement_after_bending("head", 30)
        self.assertGreater(moved[self.near(self.truth["face"]["left_eye"])].mean(), 0.02)
        self.assertLess(moved[self.near(self.truth["joints"]["left_hand"])].max(), 0.002)


class SkinnedTPoseCharacter(unittest.TestCase):
    """Arm rules written for hanging arms, applied to arms held out."""

    @classmethod
    def setUpClass(cls):
        cls.mesh, cls.truth = load_fixture("synthetic_character_tpose")
        joints = landmarks.detect_body(meshdata.world_vertices(cls.mesh), meshdata.triangles(cls.mesh.data))
        cls.armature = rig.build_armature(joints)
        rig.skin(cls.mesh, cls.armature)
        cls.rest = meshdata.world_vertices(cls.mesh)

    def test_forearms_carry_no_torso_weight(self):
        # _keep_arms_off_chest read every forearm vertex of a level arm as beside it, and the arm
        # tore 36 cm from the chest when it came down (a T-pose copy of model A)
        torso = {self.mesh.vertex_groups[rig.BONE_NAMES[k]].index for k in rig.TORSO_KEYS
                 if rig.BONE_NAMES[k] in self.mesh.vertex_groups}
        for side in ("left", "right"):
            with self.subTest(side=side):
                elbow = np.asarray(self.truth["joints"][f"{side}_lower_arm"])
                wrist = np.asarray(self.truth["joints"][f"{side}_hand"])
                forearm = np.linalg.norm(self.rest - (elbow + wrist) / 2, axis=1) < 0.05
                weight = [sum(g.weight for g in self.mesh.data.vertices[i].groups if g.group in torso)
                          for i in np.flatnonzero(forearm)]
                self.assertGreater(len(weight), 10)
                self.assertLess(max(weight), 0.05)


class ClothBesideTheForearm(unittest.TestCase):
    """A jacket's sleeve and, beside the hanging forearm, a panel of its hip skirt that bone heat fused
    to the arm across the gap: cut free with the sleeve, it rode the forearm as a strip (model A's OBJ)."""

    def setUp(self):
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        joints = {"hips": (0, 0, .8), "neck": (0, 0, 1.3), "head": (0, 0, 1.4)}
        for side, sign in (("left", 1), ("right", -1)):
            joints.update({f"{side}_upper_arm": (sign * .15, 0, 1.2), f"{side}_lower_arm": (sign * .23, 0, 1.0),
                           f"{side}_hand": (sign * .30, 0, .8), f"{side}_upper_leg": (sign * .07, 0, .75),
                           f"{side}_lower_leg": (sign * .07, 0, .4), f"{side}_foot": (sign * .07, 0, .08)})
        self.arm = rig.build_armature(joints)

    def jacket(self, panel_faces_arm):
        """A jacket's left sleeve, its back, and a panel of its hip skirt 6-11 cm medial of the hanging
        forearm, as one part; weighted as a fused skinning proxy leaves them, the panel on the
        forearm. Returns (object, panel mask, sleeve mask)."""
        start, elbow, wrist = np.array([.15, 0, 1.2]), np.array([.23, 0, 1.0]), np.array([.30, 0, .8])
        axis = (wrist - start) / np.linalg.norm(wrist - start)
        side = np.cross(axis, (0, -1, 0))
        side /= np.linalg.norm(side)
        up = np.cross(side, axis)
        points, bones, faces, facing = [], [], [], []
        rings, around = 24, 24
        for i in range(rings):
            centre = start + (wrist - start) * (i + .5) / rings
            for k in range(around):
                radial = math.cos(2 * math.pi * k / around) * side + math.sin(2 * math.pi * k / around) * up
                points.append(centre + (.036 + .006 * (k % 2)) * radial)
                bones.append("LeftUpperArm" if centre[2] > elbow[2] else "LeftLowerArm")
                facing.append(radial)
        faces += [(i * around + k, i * around + (k + 1) % around, (i + 1) * around + (k + 1) % around,
                   (i + 1) * around + k) for i in range(rings - 1) for k in range(around)]

        def sheet(corner, across, down, columns, rows, bone, normal):
            first = len(points)
            for j in range(rows):
                for i in range(columns):
                    points.append(np.asarray(corner) + np.asarray(across) * i / (columns - 1)
                                  + np.asarray(down) * j / (rows - 1))
                    bones.append(bone)
                    facing.append(np.asarray(normal, dtype=float))
            faces.extend((first + j * columns + i, first + j * columns + i + 1, first + (j + 1) * columns + i + 1,
                          first + (j + 1) * columns + i) for j in range(rows - 1) for i in range(columns - 1))
            return first

        panel_from = sheet((.18, -.05, .82), (0, .1, 0), (0, 0, .14), 10, 14, "LeftLowerArm",
                           (1, 0, 0) if panel_faces_arm else (-1, 0, 0))
        back_from = sheet((-.1, .08, .8), (.15, 0, 0), (0, 0, .35), 25, 25, "Spine", (0, 1, 0))
        points, facing = np.array(points), np.array(facing)
        oriented = []
        for face in faces:
            corners = points[list(face)]
            normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
            oriented.append(face if normal @ facing[list(face)].sum(axis=0) > 0 else face[::-1])
        data = bpy.data.meshes.new("Jacket")
        data.from_pydata([tuple(p) for p in points], [], oriented)
        data.update()
        obj = bpy.data.objects.new("Jacket", data)
        bpy.context.scene.collection.objects.link(obj)
        for name in set(bones):
            obj.vertex_groups.new(name=name).add([i for i, bone in enumerate(bones) if bone == name], 1.0, "REPLACE")
        index = np.arange(len(points))
        return obj, (index >= panel_from) & (index < back_from), index < panel_from

    def arm_share(self, obj):
        arm = {obj.vertex_groups[n].index for n in ("LeftUpperArm", "LeftLowerArm", "LeftHand") if n in obj.vertex_groups}
        return np.array([sum(g.weight for g in v.groups if g.group in arm) for v in obj.data.vertices])

    def test_cloth_facing_the_forearm_from_outside_goes_to_the_torso(self):
        obj, panel, sleeve = self.jacket(panel_faces_arm=True)
        count = len(obj.data.vertices)
        rig._keep_arms_off_parts(obj, self.arm, np.zeros(count, dtype=int), np.zeros(count, dtype=bool))
        share = self.arm_share(obj)
        self.assertLess(share[panel].max(), .01)
        self.assertGreater(share[sleeve].min(), .99)

    def test_a_sleeve_flaring_away_from_the_forearm_stays_on_it(self):
        # model B's coat sleeves reach 12 cm on their inner side, facing away from the arm
        obj, panel, sleeve = self.jacket(panel_faces_arm=False)
        count = len(obj.data.vertices)
        rig._keep_arms_off_parts(obj, self.arm, np.zeros(count, dtype=int), np.zeros(count, dtype=bool))
        self.assertGreater(self.arm_share(obj)[panel | sleeve].min(), .99)


class ArmWeightHandedBack(unittest.TestCase):
    """The back of a collar between the shoulders, 8 cm under the neck joint, on the shoulder bones as
    bone heat leaves it: who takes over what the shoulders give up (rig._keep_arms_off_chest)."""

    def setUp(self):
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        joints = {"hips": (0, 0, .8), "neck": (0, 0, 1.3), "head": (0, 0, 1.4)}
        for side, sign in (("left", 1), ("right", -1)):
            joints.update({f"{side}_upper_arm": (sign * .15, 0, 1.2), f"{side}_lower_arm": (sign * .23, 0, 1.0),
                           f"{side}_hand": (sign * .30, 0, .8), f"{side}_upper_leg": (sign * .07, 0, .75),
                           f"{side}_lower_leg": (sign * .07, 0, .4), f"{side}_foot": (sign * .07, 0, .08)})
        self.arm = rig.build_armature(joints)

    def handed_back(self, *weights):
        """A row of collar vertices, one per {bone: weight}, after the rule: [{bone: weight}]."""
        data = bpy.data.meshes.new("Collar")
        data.from_pydata([(-.03 + .06 * i / max(len(weights) - 1, 1), .05, 1.22) for i in range(len(weights))],
                         [(i, i + 1) for i in range(len(weights) - 1)], [])
        data.update()
        obj = bpy.data.objects.new("Collar", data)
        bpy.context.scene.collection.objects.link(obj)
        for i, held in enumerate(weights):
            for bone, weight in held.items():
                (obj.vertex_groups.get(bone) or obj.vertex_groups.new(name=bone)).add([i], weight, "REPLACE")
        rig._keep_arms_off_chest(obj, self.arm, np.zeros(len(weights), dtype=bool))
        names = {g.index: g.name for g in obj.vertex_groups}
        return [{names[g.group]: g.weight for g in v.groups if g.weight > 1e-6} for v in obj.data.vertices]

    def torso(self, held):
        return sum(held.get(rig.BONE_NAMES[k], 0.0) for k in rig.TORSO_KEYS)

    def test_a_trace_of_the_head_does_not_take_all_the_shoulders_give_up(self):
        # model A's collar under the back hair: Head 0.003 beside shoulders 0.99 became Head 1.0, its
        # neighbours with no trace went to the chest, and the edges between stretched 6.5 cm as the head turned
        trace, none = self.handed_back({"LeftShoulder": .55, "RightShoulder": .44, "Head": .01},
                                       {"LeftShoulder": .6, "RightShoulder": .4})
        # each side's shoulder gives up in turn, so the trace is at most doubled twice
        self.assertLess(trace.get("Head", 0.0), .04 + 1e-6)
        self.assertGreater(self.torso(trace), .96 - 1e-6)
        self.assertAlmostEqual(self.torso(none), 1.0, places=5)
        for held in (trace, none):
            self.assertFalse({"LeftShoulder", "RightShoulder"} & set(held))
            self.assertAlmostEqual(sum(held.values()), 1.0, places=5)

    def test_a_torso_bone_that_holds_the_vertex_takes_it_all(self):
        held, = self.handed_back({"Chest": .05, "LeftShoulder": .95})
        self.assertAlmostEqual(held["Chest"], 1.0, places=5)

    def test_a_head_that_holds_half_takes_the_other_half(self):
        # hair lying on a shoulder stays the head's
        held, = self.handed_back({"Head": .5, "RightShoulder": .5})
        self.assertAlmostEqual(held["Head"], 1.0, places=5)


class FillUnweighted(unittest.TestCase):
    def setUp(self):
        verts = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),  # weighted to Neck at y=0, Head at y=1
            (0, 2, 0), (1, 2, 0),  # unweighted, joined to the Head row by edges
            (0, 2, 0), (1, 2, 0), (0, 3, 0), (1, 3, 0),  # unweighted separate piece sharing that row's spot
            (5, 5, 5), (6, 5, 5), (5, 6, 5),  # unweighted piece touching nothing
        ]
        faces = [(0, 1, 3, 2), (2, 3, 5, 4), (6, 7, 9, 8), (10, 11, 12)]
        mesh = bpy.data.meshes.new("fill_unweighted")
        mesh.from_pydata(verts, [], faces)
        self.obj = bpy.data.objects.new("fill_unweighted", mesh)
        bpy.context.scene.collection.objects.link(self.obj)
        self.obj.vertex_groups.new(name="Neck").add([0, 1], 1.0, "REPLACE")
        self.obj.vertex_groups.new(name="Head").add([2, 3], 1.0, "REPLACE")

    def tearDown(self):
        mesh = self.obj.data
        bpy.data.objects.remove(self.obj)
        bpy.data.meshes.remove(mesh)

    def weights(self, index):
        return {self.obj.vertex_groups[g.group].name: round(g.weight, 3) for g in self.obj.data.vertices[index].groups}

    def test_every_vertex_ends_up_weighted(self):
        self.assertEqual(rig.fill_unweighted(self.obj), 9)
        for v in self.obj.data.vertices:
            self.assertAlmostEqual(sum(g.weight for g in v.groups), 1.0, places=3)

    def test_takes_weights_from_the_surface_around_it(self):
        rig.fill_unweighted(self.obj)
        self.assertEqual(self.weights(4), {"Head": 1.0})

    def test_separate_piece_follows_what_it_touches(self):
        rig.fill_unweighted(self.obj)
        self.assertEqual(self.weights(8), {"Head": 1.0})

    def test_isolated_piece_takes_the_nearest_weights(self):
        rig.fill_unweighted(self.obj)
        self.assertEqual(self.weights(10), {"Head": 1.0})


class FourBonesAVertex(unittest.TestCase):
    """A .vrm holds four bones a vertex; the .blend is left on the same four, or it isn't what ships."""

    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        data = bpy.data.armatures.new("bones")
        self.armature = bpy.data.objects.new("bones", data)
        bpy.context.scene.collection.objects.link(self.armature)
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="EDIT")
        for i, name in enumerate("ABCDEF"):
            bone = data.edit_bones.new(name)
            bone.head, bone.tail = (i, 0, 0), (i, 0, 1)
        bpy.ops.object.mode_set(mode="OBJECT")
        mesh = bpy.data.meshes.new("skin")
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        self.obj = bpy.data.objects.new("skin", mesh)
        bpy.context.scene.collection.objects.link(self.obj)
        groups = {name: self.obj.vertex_groups.new(name=name) for name in ("A", "B", "C", "D", "E", "F", "marker")}
        for name, weight in zip("ABCDEF", (0.30, 0.25, 0.20, 0.15, 0.06, 0.04)):  # six bones
            groups[name].add([0], weight, "REPLACE")
        groups["marker"].add([0], 1.0, "REPLACE")  # not a bone: a flag some step reads
        for name, weight in zip("ABCD", (0.4, 0.3, 0.2, 0.1)):  # four already
            groups[name].add([1], weight, "REPLACE")
        for name, weight in zip("ABCDE", (0.4, 0.3, 0.1, 0.1, 0.1)):  # the fourth and fifth alike
            groups[name].add([2], weight, "REPLACE")

    def weights(self, index):
        return {self.obj.vertex_groups[g.group].name: round(g.weight, 3) for g in self.obj.data.vertices[index].groups}

    def test_a_vertex_on_six_bones_keeps_its_four_strongest(self):
        self.assertEqual(rig.limit_influences(self.obj, self.armature), 2)
        self.assertEqual(set(self.weights(0)) - {"marker"}, set("ABCD"))

    def test_the_four_share_what_all_six_held(self):
        rig.limit_influences(self.obj, self.armature)
        held = {self.obj.vertex_groups[g.group].name: g.weight for g in self.obj.data.vertices[0].groups}
        self.assertAlmostEqual(sum(held[name] for name in "ABCD"), 1.0, places=5)
        # in the proportions they had: the exporter scales its four the same way
        self.assertAlmostEqual(held["A"] / held["D"], 2.0, places=5)

    def test_a_group_that_is_no_bone_is_left_alone(self):
        rig.limit_influences(self.obj, self.armature)
        self.assertEqual(self.weights(0)["marker"], 1.0)

    def test_a_vertex_on_four_bones_is_untouched(self):
        rig.limit_influences(self.obj, self.armature)
        self.assertEqual(self.weights(1), {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})

    def test_equal_weights_keep_the_first_listed_as_the_exporter_does(self):
        rig.limit_influences(self.obj, self.armature)
        self.assertEqual(set(self.weights(2)), set("ABCD"))


if __name__ == "__main__":
    unittest.main()
