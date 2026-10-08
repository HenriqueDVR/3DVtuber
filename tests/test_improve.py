"""Taking a finished .vrm in as it is (pipeline/improve.py) and writing it out again."""

import unittest

import bpy
import numpy as np

from support import FIXTURE_OUT, load_fixture
from vtuber_maker.pipeline import improve, landmarks, meshdata, rig, vrm_export

# glTF axes: 6 cm above the head joint and 7 cm in front of it
LOOK_FROM = [0.0, 0.06, 0.07]


def another_tools_file(path):
    """A small avatar exported here, then given what a file from elsewhere has: its own bone
    names, a hat as a second mesh, and a look-at origin."""
    mesh, _ = load_fixture()
    vrm_export.ensure_vrm_addon()
    joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))
    armature = rig.build_armature(joints)
    rig.skin(mesh, armature)
    mesh.shape_key_add(name="Basis", from_mix=False)
    mesh.shape_key_add(name="aa", from_mix=False).data[0].co.z += 0.002
    hat = bpy.data.objects.new("Hat", bpy.data.meshes.new("Hat"))
    head = armature.matrix_world @ armature.data.bones[rig.BONE_NAMES["head"]].head_local
    hat.data.from_pydata([(head.x + x, head.y + y, head.z + 0.3) for x, y in ((0, 0), (0.05, 0), (0, 0.05))], [], [(0, 1, 2)])
    bpy.context.scene.collection.objects.link(hat)
    hat.parent = armature
    hat.vertex_groups.new(name=rig.BONE_NAMES["head"]).add([0, 1, 2], 1.0, "REPLACE")
    hat.modifiers.new("Armature", "ARMATURE").object = armature
    vrm_export.configure(armature, name="Elsewhere", author="Tests")
    vrm_export.bind_expressions(armature, mesh)
    path.unlink(missing_ok=True)
    vrm_export.export(path, armature)
    vrm_export.rename_nodes(path, {name: f"J_{name}" for name in rig.BONE_NAMES.values()})
    gltf, binary = vrm_export._read_glb(path)
    gltf["extensions"]["VRMC_vrm"].setdefault("lookAt", {})["offsetFromHeadBone"] = LOOK_FROM
    vrm_export._write_glb(path, gltf, binary)
    return gltf


def joint_positions(gltf):
    nodes = gltf["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}

    def world(i):
        local = vrm_export._node_matrix(nodes[i])
        return world(parent[i]) @ local if i in parent else local
    return {nodes[j]["name"]: world(j)[:3, 3] for skin in gltf["skins"] for j in skin["joints"]}


class TakeAVrmIn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = FIXTURE_OUT / "improve_source.vrm"
        cls.before = another_tools_file(cls.source)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        cls.armature, cls.meshes = improve.load(cls.source)
        cls.holds = improve.holds(cls.armature, cls.meshes)
        cls.renamed = improve.rename_bones(cls.armature)
        # read now: a later test empties the scene
        body = max(cls.meshes, key=lambda m: len(m.data.vertices))
        cls.bones, cls.groups = {b.name for b in cls.armature.data.bones}, {g.name for g in body.vertex_groups}
        cls.taken_in = improve.imported(cls.armature)
        cls.path = FIXTURE_OUT / "improve_out.vrm"
        cls.path.unlink(missing_ok=True)
        vrm_export.export(cls.path, cls.armature)
        cls.after, _ = vrm_export._read_glb(cls.path)

    def test_the_bones_take_the_pipelines_names_and_the_weights_come_along(self):
        self.assertEqual(self.renamed, len(rig.BONE_NAMES))
        self.assertTrue(self.taken_in)
        for name in rig.BONE_NAMES.values():
            self.assertIn(name, self.bones)
            self.assertIn(name, self.groups)

    def test_the_meshes_stay_apart(self):
        self.assertEqual(len(self.holds["meshes"]), 2)
        self.assertEqual(min(self.holds["meshes"].values()), 3)
        self.assertEqual(len(self.after["meshes"]), len(self.before["meshes"]))

    def test_the_file_written_has_its_own_bone_names_back(self):
        was, now = joint_positions(self.before), joint_positions(self.after)
        self.assertEqual(set(was), set(now))
        self.assertIn(f"J_{rig.BONE_NAMES['hips']}", now)
        for name, position in was.items():
            np.testing.assert_allclose(now[name], position, atol=1e-5, err_msg=name)

    def test_what_the_file_had_is_all_there(self):
        was, now = (g["extensions"]["VRMC_vrm"] for g in (self.before, self.after))
        self.assertEqual(set(now["humanoid"]["humanBones"]), set(was["humanoid"]["humanBones"]))
        self.assertEqual(self.holds["expressions"], ["aa"])
        self.assertEqual(len(now["expressions"]["preset"]["aa"]["morphTargetBinds"]), 1)
        self.assertEqual(now["meta"]["name"], "Elsewhere")

    def test_the_eyes_look_from_where_they_did(self):
        # the VRM Add-on hands this back with its axes swapped: the export places it itself
        np.testing.assert_allclose(self.after["extensions"]["VRMC_vrm"]["lookAt"]["offsetFromHeadBone"], LOOK_FROM, atol=1e-5)

    def test_a_file_that_is_no_avatar_is_refused(self):
        plain = FIXTURE_OUT / "improve_plain.vrm"
        vrm_export._write_glb(plain, {"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "nodes": [{"name": "nothing"}]}, b"")
        bpy.ops.wm.read_factory_settings(use_empty=True)
        with self.assertRaises(improve.NotImprovable):
            improve.load(plain)


def a_vrm_0_file(path):
    """The same small avatar written as VRM 0.x by the add-on's own exporter, under a licence that
    forbids changed versions, with the settings 0.x keeps elsewhere than 1.0 does."""
    mesh, _ = load_fixture()
    vrm_export.ensure_vrm_addon()
    joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))
    armature = rig.build_armature(joints)
    rig.skin(mesh, armature)
    mesh.shape_key_add(name="Basis", from_mix=False)
    mesh.shape_key_add(name="aa", from_mix=False).data[0].co.z += 0.002
    vrm_export.configure(armature, name="Elsewhere", author="Tests")
    extension = armature.data.vrm_addon_extension
    extension.spec_version = "0.0"
    bpy.ops.vrm.assign_vrm0_humanoid_human_bones_automatically(armature_object_name=armature.name)
    old = extension.vrm0
    ours = {name: bone.node.bone_name for name, bone in extension.vrm1.humanoid.human_bones.human_bone_name_to_human_bone().items()}
    for bone in old.humanoid.human_bones:
        if not bone.node.bone_name and ours.get(bone.bone):
            bone.node.bone_name = ours[bone.bone]
    meta = old.meta
    meta.title, meta.author, meta.version = "Old Avatar", "Somebody Else", "2"
    meta.allowed_user_name, meta.commercial_ussage_name, meta.violent_ussage_name = "Everyone", "Allow", "Allow"
    meta.license_name = "CC_BY_ND"
    group = old.blend_shape_master.blend_shape_groups.add()
    group.name, group.preset_name = "A", "a"
    bind = group.binds.add()
    bind.mesh.mesh_object_name, bind.index, bind.weight = mesh.name, "aa", 1.0
    first = old.first_person
    first.first_person_bone.bone_name = rig.BONE_NAMES["head"]
    first.first_person_bone_offset = (0.0, LOOK_FROM[2], LOOK_FROM[1])  # the add-on's order: (x, z, y) of the file's
    first.look_at_horizontal_outer.x_range, first.look_at_horizontal_outer.y_range = 25.0, 7.0
    seen = first.mesh_annotations.add()
    seen.mesh.mesh_object_name, seen.first_person_flag = mesh.name, "ThirdPersonOnly"
    path.unlink(missing_ok=True)
    if bpy.ops.export_scene.vrm(filepath=str(path), armature_object_name=armature.name) != {"FINISHED"}:
        raise RuntimeError("the add-on didn't write the VRM 0.x fixture")
    return vrm_export._read_glb(path)[0]


class TakeAVrm0In(unittest.TestCase):
    """A VRM 0.x file leaves as VRM 1.0 with what it had, its author's name and terms among it."""

    @classmethod
    def setUpClass(cls):
        cls.source = FIXTURE_OUT / "improve_source_0.vrm"
        cls.before = a_vrm_0_file(cls.source)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        try:
            improve.load(cls.source)
            cls.refused = None
        except improve.LicenceUnread as error:
            cls.refused = error.detail
        cls.left_behind = len(bpy.data.objects)
        cls.armature, cls.meshes = improve.load(cls.source, licence_read=True)
        cls.holds = improve.holds(cls.armature, cls.meshes)
        improve.rename_bones(cls.armature)
        cls.path = FIXTURE_OUT / "improve_out_0.vrm"
        cls.path.unlink(missing_ok=True)
        vrm_export.export(cls.path, cls.armature)
        cls.after, _ = vrm_export._read_glb(cls.path)

    def test_the_fixture_is_a_vrm_0_file(self):
        self.assertIn("VRM", self.before["extensions"])
        self.assertNotIn("VRMC_vrm", self.before["extensions"])

    def test_a_licence_that_forbids_changes_is_the_users_to_read(self):
        self.assertIn("not allowed to be edited", self.refused or "")
        self.assertEqual(self.left_behind, 0)

    def test_it_leaves_as_vrm_1(self):
        self.assertEqual(self.holds["converted_from"], "0.0")
        self.assertNotIn("VRM", self.after["extensions"])
        self.assertEqual(self.after["extensions"]["VRMC_vrm"]["specVersion"], "1.0")

    def test_the_authors_name_and_terms_come_along_no_looser(self):
        meta = self.after["extensions"]["VRMC_vrm"]["meta"]
        self.assertEqual((meta["name"], meta["authors"], meta["version"]), ("Old Avatar", ["Somebody Else"], "2"))
        self.assertEqual((meta["avatarPermission"], meta["commercialUsage"]), ("everyone", "personalProfit"))
        self.assertTrue(meta["allowExcessivelyViolentUsage"])
        self.assertFalse(meta["allowExcessivelySexualUsage"])
        # no derivatives stays no derivatives; what 0.x can't say stays refused
        self.assertEqual((meta["modification"], meta["creditNotation"], meta["allowRedistribution"]), ("prohibited", "required", True))
        self.assertFalse(meta["allowPoliticalOrReligiousUsage"] or meta["allowAntisocialOrHateUsage"])
        self.assertIn("CC_BY_ND", meta["thirdPartyLicenses"])

    def test_bones_and_the_mouth_shape_are_all_there(self):
        was = {b["bone"] for b in self.before["extensions"]["VRM"]["humanoid"]["humanBones"]}
        now = self.after["extensions"]["VRMC_vrm"]
        self.assertEqual(set(now["humanoid"]["humanBones"]), was)
        self.assertEqual(self.holds["expressions"], ["aa"])
        self.assertEqual(len(now["expressions"]["preset"]["aa"]["morphTargetBinds"]), 1)

    def test_first_person_and_the_eyes_range_and_place_come_along(self):
        now = self.after["extensions"]["VRMC_vrm"]
        self.assertEqual([a["type"] for a in now["firstPerson"]["meshAnnotations"]], ["thirdPersonOnly"])
        self.assertEqual(now["lookAt"]["rangeMapHorizontalOuter"], {"inputMaxValue": 25.0, "outputScale": 7.0})
        np.testing.assert_allclose(now["lookAt"]["offsetFromHeadBone"], LOOK_FROM, atol=1e-5)

    def test_joints_stay_where_they_were_seen_from_the_front(self):
        # 0.x faces the other way in its file: turned about the upright axis, x and z change sign
        was, now = joint_positions(self.before), joint_positions(self.after)
        self.assertEqual(len(was), len(now))
        for name, position in was.items():
            np.testing.assert_allclose(now[name], position * [-1, 1, -1], atol=1e-5, err_msg=name)


if __name__ == "__main__":
    unittest.main()
