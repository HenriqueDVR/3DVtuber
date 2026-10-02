"""Exporting the rigged character as a VRM 1.0 file."""

import json
import struct
import unittest

import numpy as np

from support import FIXTURE_OUT, load_fixture
from vtuber_maker.pipeline import landmarks, meshdata, rig, vrm_export

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


def read_glb_json(path):
    data = path.read_bytes()
    chunk_length, _chunk_type = struct.unpack_from("<II", data, 12)
    return json.loads(data[20 : 20 + chunk_length])


def camel_case(snake):
    first, *rest = snake.split("_")
    return first + "".join(word.capitalize() for word in rest)


class ExportSyntheticCharacter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture()
        vrm_export.ensure_vrm_addon()
        joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))
        armature = rig.build_armature(joints)
        rig.skin(mesh, armature)
        vrm_export.configure(armature, name="Synthetic", author="Tests")
        cls.path = FIXTURE_OUT / "synthetic_character.vrm"
        cls.path.unlink(missing_ok=True)
        vrm_export.export(cls.path, armature)
        cls.armature = armature
        cls.gltf = read_glb_json(cls.path)

    def test_the_file_holds_the_rest_pose_it_was_built_from(self):
        vrm_export.verify(self.path, self.armature)  # as export itself just did: no error

    def test_a_file_whose_joints_moved_is_refused(self):
        # what the add-on's automatic pose did to every export: the .blend right, the file not
        gltf, binary = vrm_export._read_glb(self.path)
        hand = rig.BONE_NAMES["left_hand"]
        node = next(n for n in gltf["nodes"] if n.get("name") == hand)
        node["translation"] = [node["translation"][0] + 0.02, *node["translation"][1:]]
        moved = glb(FIXTURE_OUT / "synthetic_moved.vrm", gltf, binary)
        with self.assertRaisesRegex(vrm_export.ExportMismatch, rf"joints aren't where the rest pose has them \(worst: {hand} 20\.0 mm"):
            vrm_export.verify(moved, self.armature)

    def test_a_file_whose_joints_are_turned_is_refused(self):
        gltf, binary = vrm_export._read_glb(self.path)
        hips = rig.BONE_NAMES["hips"]
        next(n for n in gltf["nodes"] if n.get("name") == hips)["rotation"] = [0.0, 0.0, 0.7071068, 0.7071068]
        turned = glb(FIXTURE_OUT / "synthetic_turned.vrm", gltf, binary)
        with self.assertRaisesRegex(vrm_export.ExportMismatch, rf"1 joints are rotated \({hips}\)"):
            vrm_export.verify(turned, self.armature)

    def test_declares_the_vrm_1_extension(self):
        self.assertIn("VRMC_vrm", self.gltf["extensionsUsed"])

    def test_maps_every_required_human_bone_to_its_node(self):
        human_bones = self.gltf["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"]
        for key in REQUIRED_VRM_BONES:
            with self.subTest(bone=key):
                node = human_bones[camel_case(key)]["node"]
                self.assertEqual(self.gltf["nodes"][node]["name"], rig.BONE_NAMES[key])

    def test_writes_avatar_name_and_author(self):
        meta = self.gltf["extensions"]["VRMC_vrm"]["meta"]
        self.assertEqual(meta["name"], "Synthetic")
        self.assertEqual(meta["authors"], ["Tests"])

    def test_keeps_the_mesh_skinned(self):
        self.assertTrue(any("skin" in node and "mesh" in node for node in self.gltf["nodes"]))

    def test_every_joint_rests_unrotated_like_a_vroid_export(self):
        for joint in self.gltf["skins"][0]["joints"]:
            node = self.gltf["nodes"][joint]
            with self.subTest(joint=node["name"]):
                self.assertNotIn("matrix", node)
                np.testing.assert_allclose(node.get("rotation", (0, 0, 0, 1)), (0, 0, 0, 1), atol=1e-7)


class ExportLookAt(unittest.TestCase):
    """Eyes that move on the ARKit eyeLook keys (eyes.py) look by expression from between the eyes."""

    ORIGIN = (0.0, -0.08, 1.45)  # Blender axes

    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture()
        vrm_export.ensure_vrm_addon()
        joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))
        armature = rig.build_armature(joints)
        rig.skin(mesh, armature)
        mesh.shape_key_add(name="Basis", from_mix=False)
        for names in vrm_export.LOOK_SHAPE_KEYS.values():
            for name in names:
                key = mesh.shape_key_add(name=name, from_mix=False)
                key.data[0].co.z += 0.001
        vrm_export.configure(armature, name="Synthetic", author="Tests")
        vrm_export.bind_look_expressions(armature, mesh)
        ranges = {name: (90.0, 1.0) for name in ("horizontal_outer", "horizontal_inner", "vertical_up", "vertical_down")}
        vrm_export.set_look_at(armature, "expression", ranges, origin=cls.ORIGIN)
        cls.head = np.array(armature.matrix_world @ armature.data.bones[rig.BONE_NAMES["head"]].head_local)
        cls.path = FIXTURE_OUT / "synthetic_look_at.vrm"
        cls.path.unlink(missing_ok=True)
        vrm_export.export(cls.path, armature)
        cls.gltf = read_glb_json(cls.path)

    def test_looks_by_expression_from_between_the_eyes(self):
        look_at = self.gltf["extensions"]["VRMC_vrm"]["lookAt"]
        self.assertEqual(look_at["type"], "expression")
        # glTF is Y up with the model facing +Z: Blender's (x, y, z) is (x, z, -y)
        offset = np.subtract(self.ORIGIN, self.head)
        np.testing.assert_allclose(look_at["offsetFromHeadBone"], (offset[0], offset[2], -offset[1]), atol=1e-5)
        self.assertEqual(look_at["rangeMapHorizontalOuter"], {"inputMaxValue": 90.0, "outputScale": 1.0})

    def test_looking_left_moves_the_left_eye_out_and_the_right_eye_in(self):
        preset = self.gltf["extensions"]["VRMC_vrm"]["expressions"]["preset"]
        mesh = next(m for m in self.gltf["meshes"] if m.get("extras", {}).get("targetNames"))
        names = mesh["extras"]["targetNames"]
        for expression, keys in (("lookLeft", {"eyeLookOutLeft", "eyeLookInRight"}),
                                 ("lookUp", {"eyeLookUpLeft", "eyeLookUpRight"})):
            bound = {names[bind["index"]] for bind in preset[expression]["morphTargetBinds"]}
            self.assertEqual(bound, keys, expression)


def glb(path, gltf, binary):
    vrm_export._write_glb(path, gltf, binary)
    return path


class NormalizeBones(unittest.TestCase):
    """A small skinned file with turned joints: positions, skinning at rest and colliders survive."""

    def build(self, path):
        turn = lambda angle, axis: {"x": (np.sin(angle / 2), 0, 0), "z": (0, 0, np.sin(angle / 2))}[axis] + (np.cos(angle / 2),)
        nodes = [{"name": "root", "translation": [0, 1, 0], "rotation": list(turn(.7, "z")), "children": [1]},
                 {"name": "child", "translation": [0, .3, 0], "rotation": list(turn(-1.2, "x")), "children": [2]},
                 {"name": "prop", "translation": [.1, 0, 0], "rotation": list(turn(.4, "x"))},
                 {"name": "body", "mesh": 0, "skin": 0}]
        gltf = {"asset": {"version": "2.0"}, "nodes": nodes, "scenes": [{"nodes": [0, 3]}],
                "meshes": [{"primitives": []}], "skins": [{"joints": [0, 1], "inverseBindMatrices": 0}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "count": 2, "type": "MAT4"}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 128}],
                "buffers": [{"byteLength": 128}],
                "extensions": {"VRMC_springBone": {"colliders": [
                    {"node": 1, "shape": {"capsule": {"offset": [.01, .02, .03], "tail": [0, .1, 0], "radius": .05}}}]}}}
        self.worlds = self.world_matrices(gltf)
        ibm = np.array([np.linalg.inv(self.worlds[j]) for j in (0, 1)], np.float32).transpose(0, 2, 1).tobytes()
        return glb(path, gltf, ibm), gltf

    @staticmethod
    def world_matrices(gltf):
        nodes, out = gltf["nodes"], {}
        parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
        def world(i):
            local = vrm_export._node_matrix(nodes[i])
            return world(parent[i]) @ local if i in parent else local
        return {i: world(i) for i in range(len(nodes))}

    def test_joints_lose_their_turn_and_nothing_moves(self):
        path, before = self.build(FIXTURE_OUT / "normalize_bones.glb")
        vrm_export.normalize_bones(path)
        after, binary = vrm_export._read_glb(path)
        worlds = self.world_matrices(after)
        ibm = np.frombuffer(binary, np.float32, 32).reshape(2, 4, 4).transpose(0, 2, 1)
        for j in (0, 1):
            np.testing.assert_allclose(worlds[j][:3, :3], np.eye(3), atol=1e-7)
            np.testing.assert_allclose(worlds[j][:3, 3], self.worlds[j][:3, 3], atol=1e-7)
            # skinning at rest is unchanged: joint world times inverse bind is still identity
            np.testing.assert_allclose(worlds[j] @ ibm[j], np.eye(4), atol=1e-6)
        # a non-joint child keeps its world transform
        np.testing.assert_allclose(worlds[2], self.worlds[2], atol=1e-7)
        shape = after["extensions"]["VRMC_springBone"]["colliders"][0]["shape"]["capsule"]
        was = before["extensions"]["VRMC_springBone"]["colliders"][0]["shape"]["capsule"]
        for key in ("offset", "tail"):
            np.testing.assert_allclose(worlds[1] @ np.r_[shape[key], 1], self.worlds[1] @ np.r_[was[key], 1], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
