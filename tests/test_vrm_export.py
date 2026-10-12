"""Exporting the rigged character as a VRM 1.0 file."""

import json
import struct
import unittest
from unittest import mock

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

    def test_a_file_whose_vertices_moved_is_refused(self):
        # every joint where the rest pose has it and the surface 2 cm off: the other half of what the
        # add-on's automatic pose did, which a look at the joints alone lets through
        gltf, binary = vrm_export._read_glb(self.path)
        accessor = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
        start = gltf["bufferViews"][accessor["bufferView"]].get("byteOffset", 0) + accessor.get("byteOffset", 0)
        moved = np.frombuffer(binary, "<f4", accessor["count"] * 3, start).reshape(-1, 3) + np.float32((0.02, 0, 0))
        binary = binary[:start] + moved.astype("<f4").tobytes() + binary[start + moved.nbytes:]
        shifted = glb(FIXTURE_OUT / "synthetic_shifted.vrm", gltf, binary)
        with self.assertRaisesRegex(vrm_export.ExportMismatch, r"vertices aren't on the mesh as it rests \(worst: 20\.0 mm\)"):
            vrm_export.verify(shifted, self.armature)

    def test_declares_the_vrm_1_extension(self):
        self.assertIn("VRMC_vrm", self.gltf["extensionsUsed"])

    def test_the_addon_guesses_no_vrm_0_humanoid_on_the_way(self):
        # asked to export, it searched every way the bones could make a body for a VRM 0.x mapping that
        # no VRM 1.0 file holds: 22 of an export's 29 s on the fox girl's skeleton
        humanoid = self.armature.data.vrm_addon_extension.vrm0.humanoid
        self.assertFalse(humanoid.initial_automatic_bone_assignment)
        self.assertEqual([bone.node.bone_name for bone in humanoid.human_bones if bone.node.bone_name], [])

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

    def test_each_look_is_stored_as_the_vertices_it_moves(self):
        gltf, binary = vrm_export._read_glb(self.path)
        looks = [target["POSITION"] for mesh in gltf["meshes"] for p in mesh["primitives"] for target in p.get("targets", [])]
        self.assertGreaterEqual(len(looks), 8)
        for a in looks:
            accessor = gltf["accessors"][a]
            self.assertNotIn("bufferView", accessor)
            # 32-bit though a byte would do: UniVRM 0.115 and earlier read no 8-bit indices in VRM 1.0
            self.assertEqual(accessor["sparse"]["indices"]["componentType"], 5125)
            offsets = vrm_export._accessor_array(gltf, binary, a)
            moved = offsets[offsets.any(axis=1)]
            self.assertEqual(len(moved), accessor["sparse"]["count"])
            # the one vertex each key lifts by a millimetre (Blender's +Z is glTF's +Y)
            np.testing.assert_allclose(moved, [(0.0, 0.001, 0.0)] * len(moved), atol=1e-7)


def glb(path, gltf, binary):
    vrm_export._write_glb(path, gltf, binary)
    return path


class SparseMorphTargets(unittest.TestCase):
    """A small mesh with its face shapes as the exporter writes them, every offset of every vertex:
    "blink" moves two vertices and turns the normal of one, "aa" moves nothing, "puff" moves them
    all, and "base" is the mesh's own positions used again. Its picture lies among them in the file."""

    VERTICES = 60
    PICTURE = b"\x89PNG, or 25 bytes of one"

    def build(self, path):
        n = self.VERTICES
        rng = np.random.default_rng(7)
        shape = lambda rows=(): np.zeros((n, 3), np.float32) if not len(rows) else self.offsets(rows, rng)  # noqa: E731
        self.arrays = {"positions": rng.random((n, 3), np.float32), "indices": np.arange(n, dtype=np.uint16).reshape(-1, 1),
                       "blink": shape((1, 5)), "blink_normals": shape((1,)), "aa": shape(), "aa_normals": shape(),
                       "puff": np.full((n, 3), 0.01, np.float32), "puff_normals": shape()}
        order = ("positions", "indices", "blink", None, "blink_normals", "aa", "aa_normals", "puff", "puff_normals")
        binary, views = bytearray(), []
        for name in order:
            binary.extend(b"\0" * (-len(binary) % 4))
            data = self.PICTURE if name is None else self.arrays[name].tobytes()
            views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
            binary.extend(data)
        vec3 = lambda view, **more: {"bufferView": view, "componentType": 5126, "count": n, "type": "VEC3", **more}  # noqa: E731
        bounds = lambda name: {"min": self.arrays[name].min(0).tolist(), "max": self.arrays[name].max(0).tolist()}  # noqa: E731
        accessors = [vec3(0, **bounds("positions")), {"bufferView": 1, "componentType": 5123, "count": n, "type": "SCALAR"},
                     vec3(2, **bounds("blink")), vec3(4), vec3(5, **bounds("aa")), vec3(6), vec3(7, **bounds("puff")), vec3(8)]
        self.names = ("positions", "indices", "blink", "blink_normals", "aa", "aa_normals", "puff", "puff_normals")
        targets = [{"POSITION": 2, "NORMAL": 3}, {"POSITION": 4, "NORMAL": 5}, {"POSITION": 6, "NORMAL": 7}, {"POSITION": 0}]
        gltf = {"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "nodes": [{"name": "face", "mesh": 0}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "targets": targets}],
                            "extras": {"targetNames": ["blink", "aa", "puff", "base"]}}],
                "images": [{"bufferView": 3, "mimeType": "image/png"}],
                "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(binary)}]}
        return glb(path, gltf, bytes(binary)), gltf

    def offsets(self, rows, rng):
        out = np.zeros((self.VERTICES, 3), np.float32)
        out[list(rows)] = rng.random((len(rows), 3), np.float32) + 0.5
        return out

    def setUp(self):
        self.path, self.before = self.build(FIXTURE_OUT / "sparse_morph_targets.glb")
        self.size = self.path.stat().st_size
        self.rewritten = vrm_export.sparse_morph_targets(self.path)
        self.gltf, self.binary = vrm_export._read_glb(self.path)
        self.accessor = dict(zip(self.names, self.gltf["accessors"]))

    def read(self, name):
        return vrm_export._accessor_array(self.gltf, self.binary, self.names.index(name))

    def test_a_shape_is_stored_as_the_vertices_it_moves(self):
        self.assertEqual(self.rewritten, 5)
        for name, moved in (("blink", 2), ("blink_normals", 1)):
            self.assertNotIn("bufferView", self.accessor[name])
            self.assertEqual(self.accessor[name]["sparse"]["count"], moved)

    def test_every_shape_reads_as_it_did(self):
        for name, was in self.arrays.items():
            with self.subTest(name):
                np.testing.assert_array_equal(self.read(name), was)

    def test_indices_are_32_bit_whatever_the_vertex_count(self):
        # 60 vertices fit a byte, and UniVRM 0.115 and earlier can't read a byte here (VRM 1.0)
        kinds = {a["sparse"]["indices"]["componentType"] for a in self.gltf["accessors"] if "sparse" in a}
        self.assertEqual(kinds, {5125})

    def test_a_shape_that_moves_nothing_keeps_one_entry_of_zeros(self):
        for name in ("aa", "aa_normals", "puff_normals"):
            self.assertEqual(self.accessor[name]["sparse"]["count"], 1, name)  # an empty sparse accessor isn't allowed
            self.assertFalse(self.read(name).any(), name)

    def test_a_shape_that_moves_every_vertex_stays_whole(self):
        self.assertIn("bufferView", self.accessor["puff"])
        self.assertNotIn("sparse", self.accessor["puff"])

    def test_what_the_mesh_itself_reads_stays_whole(self):
        # "base" is a morph target and the mesh's positions at once: an app may not take those sparse
        for name in ("positions", "indices"):
            self.assertIn("bufferView", self.accessor[name])
            self.assertNotIn("sparse", self.accessor[name])

    def test_counts_and_bounds_are_untouched(self):
        kept = lambda accessor: {k: v for k, v in accessor.items() if k not in ("bufferView", "byteOffset", "sparse")}  # noqa: E731
        self.assertEqual([kept(a) for a in self.gltf["accessors"]], [kept(a) for a in self.before["accessors"]])

    def test_the_picture_is_where_the_file_says(self):
        # buffer views keep their numbers: nothing that names one has to be told
        self.assertEqual(self.gltf["images"], self.before["images"])
        view = self.gltf["bufferViews"][3]
        self.assertEqual(self.binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]], self.PICTURE)

    def test_the_file_is_smaller_and_nothing_overlaps(self):
        self.assertLess(self.path.stat().st_size, self.size)
        self.assertEqual(self.gltf["buffers"][0]["byteLength"], len(self.binary))
        spans = sorted((v["byteOffset"], v["byteOffset"] + v["byteLength"]) for v in self.gltf["bufferViews"])
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), spans)
        self.assertLessEqual(spans[-1][1], len(self.binary))
        self.assertTrue(all(start % 4 == 0 for start, _ in spans))

    def test_a_second_run_finds_nothing_to_do(self):
        once = self.path.read_bytes()
        self.assertEqual(vrm_export.sparse_morph_targets(self.path), 0)
        self.assertEqual(self.path.read_bytes(), once)

    def test_a_file_that_would_read_back_differently_is_left_as_it_was(self):
        path, _ = self.build(FIXTURE_OUT / "sparse_morph_targets_refused.glb")
        whole, wrong = path.read_bytes(), lambda gltf, binary, index: np.ones((self.VERTICES, 3), np.float32)
        with mock.patch.object(vrm_export, "_accessor_array", wrong):
            with self.assertRaisesRegex(vrm_export.ExportMismatch, "5 morph targets don't read back as they were written"):
                vrm_export.sparse_morph_targets(path)
        self.assertEqual(path.read_bytes(), whole)


class Mipmaps(unittest.TestCase):
    """UniVRM reads a LINEAR min filter as "no mipmaps": thin drawn lines speckled in Warudo."""

    def build(self, name, filters):
        samplers = [{"magFilter": 9729, **({} if kind is None else {"minFilter": kind})} for kind in filters]
        return glb(FIXTURE_OUT / name, {"asset": {"version": "2.0"}, "samplers": samplers}, b"")

    def test_a_linear_sampler_gets_mipmaps_as_a_vroid_file_has(self):
        path = self.build("mipmaps.glb", (9729, 9728, 9987, None))
        self.assertEqual(vrm_export.use_mipmaps(path), 1)
        after = [sampler.get("minFilter") for sampler in vrm_export._read_glb(path)[0]["samplers"]]
        # NEAREST is somebody's choice (pixel art); the last two had mipmaps already
        self.assertEqual(after, [9985, 9728, 9987, None])

    def test_a_file_with_nothing_to_change_is_not_written_again(self):
        path = self.build("mipmaps_already.glb", (9985,))
        before = path.read_bytes()
        self.assertEqual(vrm_export.use_mipmaps(path), 0)
        self.assertEqual(path.read_bytes(), before)


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


class NormalizeConstrainedBones(unittest.TestCase):
    """Node constraints (a .vrm from elsewhere: twist and aim helpers) name axes in a joint's own frame."""

    QUARTER = [0.0, 0.0, 0.7071068, 0.7071068]  # 90 degrees about Z: the joint's X is the model's Y

    def build(self, name, twist_turn, aim_turn=(0.0, 0.0, 0.2, 0.98)):
        constraint = lambda kind, **settings: {"VRMC_node_constraint": {"constraint": {kind: {"weight": 1, **settings}}}}  # noqa: E731
        nodes = [{"name": "arm", "translation": [0, 1, 0], "rotation": self.QUARTER, "children": [1, 2, 3]},
                 {"name": "twist", "translation": [.1, 0, 0], "rotation": list(twist_turn),
                  "extensions": constraint("roll", source=0, rollAxis="X")},
                 {"name": "aim", "translation": [.2, 0, 0], "rotation": list(aim_turn), "children": [4],
                  "extensions": constraint("aim", source=0, aimAxis="PositiveX")},
                 {"name": "plain", "translation": [.3, 0, 0], "rotation": [0.0, 0.3, 0.0, 0.954]},
                 {"name": "under_aim", "translation": [.05, 0, 0]},
                 {"name": "body", "mesh": 0, "skin": 0}]
        gltf = {"asset": {"version": "2.0"}, "nodes": nodes, "scenes": [{"nodes": [0, 5]}],
                "meshes": [{"primitives": []}], "skins": [{"joints": [0, 1, 2, 3, 4], "inverseBindMatrices": 0}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "count": 5, "type": "MAT4"}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 320}], "buffers": [{"byteLength": 320}]}
        self.worlds = NormalizeBones.world_matrices(gltf)
        ibm = np.array([np.linalg.inv(self.worlds[j]) for j in range(5)], np.float32).transpose(0, 2, 1).tobytes()
        path = glb(FIXTURE_OUT / name, gltf, ibm)
        vrm_export.normalize_bones(path)
        after, _ = vrm_export._read_glb(path)
        return after, NormalizeBones.world_matrices(after)

    def test_a_roll_axis_takes_the_name_of_the_model_axis_it_lies_along(self):
        after, worlds = self.build("normalize_roll.glb", twist_turn=(0, 0, 0, 1))
        roll = after["nodes"][1]["extensions"]["VRMC_node_constraint"]["constraint"]["roll"]
        self.assertEqual(roll["rollAxis"], "Y")
        for j in (0, 1, 3):
            np.testing.assert_allclose(worlds[j][:3, :3], np.eye(3), atol=1e-7)
            np.testing.assert_allclose(worlds[j][:3, 3], self.worlds[j][:3, 3], atol=1e-7)

    def test_an_aim_joint_keeps_its_frame_and_what_hangs_from_it_is_straightened(self):
        after, worlds = self.build("normalize_aim.glb", twist_turn=(0, 0, 0, 1))
        self.assertEqual(after["nodes"][2]["extensions"]["VRMC_node_constraint"]["constraint"]["aim"]["aimAxis"], "PositiveX")
        np.testing.assert_allclose(worlds[2], self.worlds[2], atol=1e-7)
        np.testing.assert_allclose(worlds[4][:3, :3], np.eye(3), atol=1e-7)
        np.testing.assert_allclose(worlds[4][:3, 3], self.worlds[4][:3, 3], atol=1e-7)

    def test_a_roll_axis_off_every_model_axis_leaves_both_joints_as_they_were(self):
        after, worlds = self.build("normalize_roll_off.glb", twist_turn=(0.0, 0.0, 0.2588190, 0.9659258))  # 30 degrees
        self.assertEqual(after["nodes"][1]["extensions"]["VRMC_node_constraint"]["constraint"]["roll"]["rollAxis"], "X")
        for j in (0, 1):
            np.testing.assert_allclose(worlds[j], self.worlds[j], atol=1e-7)
        np.testing.assert_allclose(worlds[3][:3, :3], np.eye(3), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
