"""A model that came whole, taken in with its own rig: which bones are the humanoid's, which
chains swing, and which face shapes are the VRM expressions."""
import unittest

import bpy
import numpy as np

from support import load_fixture
from test_adopt import CATS
from vtuber_maker.pipeline import kept, landmarks, meshdata, rig, vrm_export
from vtuber_maker.pipeline.rig import BONE_NAMES

# Bones added to the fixture's rig, as an MMD model has them: (name, parent, weighted)
EXTRA = [("TailROOT", "Head", True), ("Tail_1", "TailROOT", True), ("Tail_2", "Tail_1", True), ("Tail_3", "Tail_2", True),
         ("Tail_3_end", "Tail_3", False), ("Side_1", "TailROOT", True), ("Side_2", "Side_1", True), ("Side_2_end", "Side_2", False),
         ("Flower", "Head", True), ("Flower_end", "Flower", False),
         ("Dress_1", "Hips", True), ("Dress_2", "Dress_1", True), ("Dress_3", "Dress_2", True), ("Dress_3_end", "Dress_3", False),
         ("Bust_1", "Chest", True), ("Bust_2", "Bust_1", True), ("Bust_2_end", "Bust_2", False),
         ("zArmTwist_L", "LeftUpperArm", True), ("zArmTwist_L_end", "zArmTwist_L", False),
         ("Eye_L", "Head", True), ("Eye_R", "Head", True), ("Left toe", "LeftFoot", True), ("Mount", "Hips", False),
         ("Ribbon_1", "Mount", True), ("Ribbon_2", "Ribbon_1", True), ("Ribbon_3", "Ribbon_2", True)]


class KeptRig(unittest.TestCase):
    """The fixture character rigged by us, renamed as the Cats plugin names an MMD model, without
    an upper chest, with hair, skirt, chest and ornament bones and four MMD face shapes."""

    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture()
        vrm_export.ensure_vrm_addon()
        joints = landmarks.detect_body(meshdata.world_vertices(mesh), meshdata.triangles(mesh.data))
        armature = rig.build_armature(joints)
        rig.skin(mesh, armature)
        rig.make_active(armature)
        bpy.ops.object.mode_set(mode="EDIT")
        bones = armature.data.edit_bones
        for child in list(bones["UpperChest"].children):
            child.parent = bones["Chest"]
        bones.remove(bones["UpperChest"])
        for name, parent, _ in EXTRA:
            bone = bones.new(name)
            bone.parent = bones[parent]
            bone.head = bones[parent].tail if parent in BONE_NAMES.values() else bones[parent].tail
            bone.tail = bone.head + (bones[parent].tail - bones[parent].head).normalized() * 0.03
        bpy.ops.object.mode_set(mode="OBJECT")
        chest, upper = mesh.vertex_groups["Chest"], mesh.vertex_groups["UpperChest"]
        for v in mesh.data.vertices:
            moved = sum(g.weight for g in v.groups if g.group == upper.index)
            if moved > 0:
                chest.add([v.index], moved, "ADD")
        mesh.vertex_groups.remove(upper)
        for index, (name, _, weighted) in enumerate(EXTRA):
            group = mesh.vertex_groups.new(name=name)
            if weighted:
                group.add([index], 0.5, "REPLACE")
        mesh.shape_key_add(name="Basis")
        top = np.argsort(meshdata.world_vertices(mesh)[:, 2])[-40:]
        for name in ("まばたき", "ウィンク", "ウィンク２", "ウィンク右", "あ", "お", "怒り", "じと目"):
            key = mesh.shape_key_add(name=name)
            key.value = 0.0
            for index in top:
                key.data[index].co.y -= 0.01
        for key, ours in BONE_NAMES.items():
            if key == "upper_chest":
                continue
            side, _, part = key.partition("_") if key.startswith(("left_", "right_")) else ("", "", key)
            armature.data.bones[ours].name = f"{side.capitalize()} {CATS[part]}" if side else CATS[part]
        cls.armature, cls.mesh = armature, mesh
        cls.report = kept.name_humanoid(armature, mesh, "Kept", "Tester", path="fixture.fbx")
        cls.chains = kept.spring_chains(armature, mesh)
        cls.bound = kept.bind_expressions(armature, mesh)
        cls.eyes = kept.bind_eyes(armature)
        cls.human = armature.data.vrm_addon_extension.vrm1.humanoid.human_bones

    def test_the_humanoid_bones_take_our_names_and_keep_their_weights(self):
        bones = self.armature.data.bones
        for key, ours in BONE_NAMES.items():
            if key != "upper_chest":
                self.assertIn(ours, bones)
                self.assertEqual(getattr(self.human, key).node.bone_name, ours)
                self.assertIn(ours, self.mesh.vertex_groups)
        self.assertEqual(self.report["naming"], "MMD (Cats)")
        self.assertFalse(self.report["turned"])

    def test_no_upper_chest_is_made_for_a_rig_without_one(self):
        self.assertNotIn("UpperChest", self.armature.data.bones)
        self.assertEqual(self.human.upper_chest.node.bone_name, "")
        self.assertFalse(self.report["upper_chest"])
        # ... and the spring design, which names one for a collider, reads it between chest and neck
        joints = kept.joints(self.armature)
        np.testing.assert_allclose(joints["upper_chest"], (joints["chest"] + joints["neck"]) / 2)

    def test_eyes_and_toes_are_bound_where_the_rig_has_them(self):
        self.assertEqual(self.report["extras"], {"left_eye": "Eye_L", "right_eye": "Eye_R", "left_toes": "Left toe"})
        self.assertTrue(self.eyes)
        self.assertEqual(self.armature.data.vrm_addon_extension.vrm1.look_at.type, "bone")

    def test_chains_swing_and_fixtures_dont(self):
        # the hair's chains start where the root branches; the end bone stays as the last joint
        self.assertEqual(sorted(self.chains["hair"]), [["Side_1", "Side_2", "Side_2_end"], ["Tail_1", "Tail_2", "Tail_3", "Tail_3_end"]])
        # a skirt's chain, and one hung on an unweighted mount, which is no part of it
        self.assertEqual(sorted(self.chains["cloth"]), [["Dress_1", "Dress_2", "Dress_3", "Dress_3_end"], ["Ribbon_1", "Ribbon_2", "Ribbon_3"]])
        self.assertEqual(self.chains["stiff"], [["Bust_1", "Bust_2", "Bust_2_end"]])
        swung = {name for kind in self.chains.values() for chain in kind for name in chain}
        self.assertFalse(swung & {"Flower", "zArmTwist_L", "Eye_L", "Left toe", "TailROOT", "Mount"})

    def test_face_shapes_are_bound_by_their_names(self):
        self.assertEqual(self.bound, {"blink": "まばたき", "blink_left": "ウィンク２", "blink_right": "ウィンク右",
                                      "aa": "あ", "oh": "お", "angry": "怒り"})
        presets = self.armature.data.vrm_addon_extension.vrm1.expressions.preset
        self.assertEqual([b.index for b in presets.blink.morph_target_binds], ["まばたき"])
        self.assertEqual(len(presets.ih.morph_target_binds), 0)

    def test_tracking_gets_shapes_under_arkit_names(self):
        made = kept.tracking_shapes(self.mesh)
        self.assertEqual(made, {"eyeBlinkLeft": "ウィンク２", "eyeBlinkRight": "まばたき", "browDownLeft": "怒り",
                                "browDownRight": "怒り", "jawOpen": "あ", "mouthFunnel": "お"})
        keys = self.mesh.data.shape_keys.key_blocks
        co = lambda name: np.array([p.co[:] for p in keys[name].data])
        x = meshdata.world_vertices(self.mesh)[:, 0]
        moved = np.linalg.norm(co("まばたき") - co("Basis"), axis=1) > 1e-6
        self.assertTrue(moved[x > 0].any() and moved[x < 0].any())
        # the right eye's blink is the right half of the blink of both, and the model's own shape stays
        half = np.linalg.norm(co("eyeBlinkRight") - co("Basis"), axis=1) > 1e-6
        np.testing.assert_array_equal(half, moved & (x < 0))
        np.testing.assert_allclose(co("jawOpen"), co("あ"), atol=1e-6)
        self.assertTrue(all(keys[name].value == 0.0 for name in made))
        # ... and asked again, what is there is left alone
        self.assertEqual(kept.tracking_shapes(self.mesh), {})

    def test_the_springs_are_made_of_the_chains(self):
        kept.bind_springs(self.armature, self.mesh, self.chains)
        springs = self.armature.data.vrm_addon_extension.spring_bone1.springs
        self.assertEqual(len(springs), 5)
        joints = {tuple(j.node.bone_name for j in s.joints) for s in springs}
        self.assertIn(("Tail_1", "Tail_2", "Tail_3", "Tail_3_end"), joints)
        colliders = self.armature.data.vrm_addon_extension.spring_bone1.colliders
        self.assertTrue(all(c.node.bone_name in self.armature.data.bones for c in colliders))


class SeeThrough(unittest.TestCase):
    """Four squares: an eye, its highlights lying on it, a blush drawn soft, and a sheen whose
    material is itself half clear."""

    @classmethod
    def setUpClass(cls):
        vrm_export.ensure_vrm_addon()
        corners = np.array([(0, 0, 0), (0.02, 0, 0), (0.02, 0, 0.02), (0, 0, 0.02)], float)
        made = (("Eye", (1.0, 1.0), 0.0), ("Eye+", (0.0, 1.0), 0.0), ("Blush", (0.5, 0.5), 0.0), ("Sheen", (0.0, 1.0), 0.6))
        data = bpy.data.meshes.new("SeeThrough")
        data.from_pydata(np.concatenate([corners + (0.1 * k, 0, 0) for k in range(len(made))]).tolist(), [],
                         [tuple(range(4 * k, 4 * k + 4)) for k in range(len(made))])
        uv = data.uv_layers.new(name="UVMap")
        for loop, at in zip(uv.data, [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)] * len(made)):
            loop.uv = at
        for index, (name, alpha, factor) in enumerate(made):
            image = bpy.data.images.new(name, 8, 8, alpha=True)
            pixels = np.ones((8, 8, 4), np.float32)
            pixels[:4, :, 3], pixels[4:, :, 3] = alpha
            image.pixels.foreach_set(pixels.ravel())
            material = bpy.data.materials.new(name)
            material.use_nodes = True
            node = material.node_tree.nodes.new("ShaderNodeTexImage")
            node.image = image
            bsdf = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            material.node_tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
            data.materials.append(material)
            data.polygons[index].material_index = index
        cls.mesh = bpy.data.objects.new("SeeThrough", data)
        bpy.context.scene.collection.objects.link(cls.mesh)
        vrm_export.use_toon_shading(cls.mesh)
        for name, _, factor in made:
            # as the import leaves them: alpha 0 on most, a sheen's own on it
            bpy.data.materials[name].vrm_addon_extension.mtoon1.pbr_metallic_roughness.base_color_factor[3] = factor
        cls.cut = kept.see_through(cls.mesh)

    def mtoon(self, name):
        return bpy.data.materials[name].vrm_addon_extension.mtoon1

    def test_a_picture_clear_in_places_cuts_its_material_out(self):
        self.assertEqual(self.cut, {"Eye+": "CUT", "Blush": "BLEND", "Sheen": "BLEND"})
        # cut by blending with depth written: Warudo drew MToon's cutout a little see-through
        self.assertEqual(self.mtoon("Eye+").alpha_mode, "BLEND")
        self.assertTrue(self.mtoon("Eye+").extensions.vrmc_materials_mtoon.transparent_with_z_write)
        self.assertFalse(self.mtoon("Blush").extensions.vrmc_materials_mtoon.transparent_with_z_write)
        # ... and isn't hidden whole by the alpha 0 the import left on its colour
        self.assertEqual(self.mtoon("Eye+").pbr_metallic_roughness.base_color_factor[3], 1.0)

    def test_an_opaque_picture_stays_opaque(self):
        self.assertEqual(self.mtoon("Eye").alpha_mode, "OPAQUE")

    def test_a_material_half_clear_keeps_its_own_alpha(self):
        self.assertAlmostEqual(self.mtoon("Sheen").pbr_metallic_roughness.base_color_factor[3], 0.6, places=5)


class CutAndSolid(unittest.TestCase):
    """Three squares on one material: two on opaque parts of its picture, one across the clear quarter."""

    @classmethod
    def setUpClass(cls):
        vrm_export.ensure_vrm_addon()
        corners = np.array([(0, 0, 0), (0.02, 0, 0), (0.02, 0, 0.02), (0, 0, 0.02)], float)
        data = bpy.data.meshes.new("CutAndSolid")
        data.from_pydata(np.concatenate([corners, corners + (0.1, 0, 0), corners + (0.2, 0, 0)]).tolist(), [],
                         [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)])
        uv = data.uv_layers.new(name="UVMap")
        # the third lies against the clear quarter, a triangle of it apart: no texel of it is clear
        for loop, at in zip(uv.data, [(0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9), (0.1, 0.1), (0.4, 0.1), (0.4, 0.9), (0.1, 0.9),
                                      (0.52, 0.02), (0.98, 0.02), (0.98, 0.48), (0.52, 0.48)]):
            loop.uv = at
        image = bpy.data.images.new("Dress", 32, 32, alpha=True)
        pixels = np.ones((32, 32, 4), np.float32)
        pixels[:16, :16, 3] = 0.0  # clear at the bottom left
        image.pixels.foreach_set(pixels.ravel())
        material = bpy.data.materials.new("Dress")
        material.use_nodes = True
        node = material.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        bsdf = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        material.node_tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
        data.materials.append(material)
        cls.mesh = bpy.data.objects.new("CutAndSolid", data)
        bpy.context.scene.collection.objects.link(cls.mesh)
        vrm_export.use_toon_shading(cls.mesh)
        cls.before = meshdata.world_vertices(cls.mesh).copy()
        cls.cut = kept.see_through(cls.mesh)

    def test_only_the_polygons_the_picture_cuts_stay_cut_out(self):
        self.assertEqual(self.cut, {"Dress": "CUT", "Dress_solid": "OPAQUE"})
        names = [self.mesh.material_slots[p.material_index].material.name for p in self.mesh.data.polygons]
        self.assertEqual(names, ["Dress_solid", "Dress", "Dress_solid"])
        solid = bpy.data.materials["Dress_solid"].vrm_addon_extension.mtoon1
        self.assertEqual(solid.alpha_mode, "OPAQUE")
        self.assertEqual(solid.pbr_metallic_roughness.base_color_texture.index.source.name, "Dress")
        self.assertTrue(bpy.data.materials["Dress"].vrm_addon_extension.mtoon1.extensions.vrmc_materials_mtoon.transparent_with_z_write)
        self.assertFalse(solid.extensions.vrmc_materials_mtoon.transparent_with_z_write)

    def test_the_surface_is_as_it_was(self):
        np.testing.assert_array_equal(meshdata.world_vertices(self.mesh), self.before)
        self.assertEqual(len(self.mesh.data.polygons), 3)


class NotAHumanoid(unittest.TestCase):
    def test_a_rig_under_names_no_table_knows_is_refused(self):
        data = bpy.data.armatures.new("Odd")
        armature = bpy.data.objects.new("Odd", data)
        bpy.context.scene.collection.objects.link(armature)
        rig.make_active(armature)
        bpy.ops.object.mode_set(mode="EDIT")
        bone = data.edit_bones.new("Bone")
        bone.head, bone.tail = (0, 0, 0), (0, 0, 1)
        bpy.ops.object.mode_set(mode="OBJECT")
        with self.assertRaisesRegex(kept.NotKept, "LeftUpperArm"):
            kept.name_humanoid(armature, None, "Odd", "Tester", path="odd.fbx")
