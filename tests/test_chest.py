"""Chest bounce: two spring bones on the front of the chest (pipeline/chest.py)."""

import unittest

import numpy as np

from support import FIXTURE_OUT, load_fixture
from test_vrm_spring_bones import read_glb_json
from vtuber_maker.pipeline import chest, face, hair, hair_physics, landmarks, meshdata, rig, vrm_export


class ChestBounce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh, _ = load_fixture()
        vrm_export.ensure_vrm_addon()
        cls.joints = landmarks.detect_body(meshdata.world_vertices(cls.mesh), meshdata.triangles(cls.mesh.data))
        features = face.detect_features(cls.mesh, cls.joints)
        cls.armature = rig.build_armature(cls.joints)
        rig.skin(cls.mesh, cls.armature)
        chains = hair.build_hair_bones(cls.mesh, cls.armature, cls.joints, features)
        vrm_export.configure(cls.armature, name="Synthetic", author="Tests")
        vrm_export.bind_spring_bones(cls.armature, hair_physics.design(cls.mesh, cls.armature, chains, cls.joints))
        cls.before = cls.weights()
        cls.weighted = chest.add(cls.mesh, cls.armature)
        # a second run replaces the first instead of adding to it
        cls.weighted = chest.add(cls.mesh, cls.armature)
        cls.after = cls.weights()
        cls.verts = meshdata.world_vertices(cls.mesh)
        path = FIXTURE_OUT / "synthetic_character_chest.vrm"
        path.unlink(missing_ok=True)
        vrm_export.export(path, cls.armature)
        cls.gltf = read_glb_json(path)

    @classmethod
    def weights(cls):
        names = {g.index: g.name for g in cls.mesh.vertex_groups}
        return [{names[g.group]: g.weight for g in v.groups if g.weight > 0} for v in cls.mesh.data.vertices]

    def chest_share(self):
        return np.array([sum(w for n, w in v.items() if n in chest.BONES) for v in self.after])

    def test_weights_both_sides_of_the_chest_front(self):
        self.assertGreater(self.weighted["left"], 10)
        self.assertGreater(self.weighted["right"], 10)
        share = self.chest_share()
        held = share > 0
        chest_z = self.joints["chest"][2] if "chest" in self.joints else self.joints["hips"][2]
        self.assertGreater(self.verts[held, 2].min(), chest_z - 0.1)
        self.assertLess(self.verts[held, 2].max(), self.joints["neck"][2] + 0.1)
        # the front of the body, not its back
        self.assertLess(np.median(self.verts[held, 1]), np.median(self.verts[:, 1]))

    def test_leaves_arms_head_and_hair_alone(self):
        for before, after in zip(self.before, self.after):
            if any(n in chest.ARM_BONES or n == "Head" or n.startswith("Hair") for n, w in before.items() if w > chest.HELD):
                self.assertFalse(set(after) & set(chest.BONES))

    def test_weights_still_sum_to_one(self):
        pairs = [(sum(b.values()), sum(a.values())) for b, a in zip(self.before, self.after) if set(a) & set(chest.BONES)]
        before, after = np.array(pairs).T
        np.testing.assert_allclose(after[np.abs(before - 1) < 1e-3], 1.0, atol=1e-3)

    def test_exports_one_spring_per_side(self):
        springs = [s for s in self.gltf["extensions"]["VRMC_springBone"]["springs"] if s.get("name") == chest.SPRING_NAME]
        self.assertEqual(len(springs), 2)
        names = {self.gltf["nodes"][j["node"]]["name"] for s in springs for j in s["joints"]}
        self.assertEqual(names, {"LeftBreast", "LeftBreast_end", "RightBreast", "RightBreast_end"})

    def test_amount_softens_the_springs_and_holds_more_of_the_chest(self):
        stiffness = lambda: {j.stiffness for s in self.armature.data.vrm_addon_extension.spring_bone1.springs  # noqa: E731
                             if s.vrm_name == chest.SPRING_NAME for j in s.joints}
        usual, usual_share = stiffness(), self.chest_share().sum()
        try:
            chest.add(self.mesh, self.armature, amount=2.0)
            self.after = self.weights()
            self.assertEqual(stiffness(), {next(iter(usual)) / 2.0})
            self.assertGreater(self.chest_share().sum(), 1.1 * usual_share)
            self.assertLessEqual(self.chest_share().max(), 1.0 + 1e-6)
            chest.add(self.mesh, self.armature, amount=0.5)
            self.after = self.weights()
            self.assertEqual(stiffness(), {next(iter(usual)) * 2.0})
            self.assertLess(self.chest_share().sum(), 0.9 * usual_share)
        finally:
            chest.add(self.mesh, self.armature)
            type(self).after = self.weights()
            del self.after


if __name__ == "__main__":
    unittest.main()
