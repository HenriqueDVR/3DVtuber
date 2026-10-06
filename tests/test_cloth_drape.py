"""A cape an adopted rig weighted to the arms, on a body that is one surface from feet to head."""

import unittest

import bpy
import numpy as np

from vtuber_maker.pipeline import cloth, hair, rig

from test_hair import _sheet

JOINTS = {key: np.array(p, float) for key, p in {
    "hips": (0, 0, 0.9), "neck": (0, 0, 1.3), "head": (0, 0, 1.38),
    "left_upper_arm": (0.15, 0, 1.25), "left_lower_arm": (0.40, 0, 1.25), "left_hand": (0.62, 0, 1.25),
    "right_upper_arm": (-0.15, 0, 1.25), "right_lower_arm": (-0.40, 0, 1.25), "right_hand": (-0.62, 0, 1.25),
    "left_upper_leg": (0.09, 0, 0.87), "left_lower_leg": (0.09, 0, 0.45), "left_foot": (0.09, 0, 0.08),
    "right_upper_leg": (-0.09, 0, 0.87), "right_lower_leg": (-0.09, 0, 0.45), "right_foot": (-0.09, 0, 0.08),
}.items()}


def _tube(centre, radius, z0, z1, around, up, offset):
    """An open tube round a vertical line; vertex indices start at `offset`."""
    angles = np.linspace(0, 2 * np.pi, around, endpoint=False)
    zs = np.linspace(z0, z1, up)
    verts = np.array([[centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a), z] for z in zs for a in angles])
    index = np.arange(around * up).reshape(up, around) + offset
    faces = np.stack([index[:-1].ravel(), np.roll(index[:-1], -1, axis=1).ravel(),
                      np.roll(index[1:], -1, axis=1).ravel(), index[1:].ravel()], axis=1)
    return verts, faces


class CapeOnTheArms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        # the body: two legs joined to a wide trunk and arms by a bridge, so it is one piece whose
        # middle lies far from the leg bones
        parts, faces, offset = [], [], 0
        for shape in (_tube((0.09, 0), 0.06, 0.1, 0.9, 12, 17, 0), _tube((-0.09, 0), 0.06, 0.1, 0.9, 12, 17, 0),
                      _sheet(-0.62, 0.62, 0.9, 1.55, 0.0, 40, 30, 0)):
            parts.append(shape[0])
            faces.append(shape[1] + offset)
            offset += len(shape[0])
        cls.body = np.arange(offset)
        # each leg's top ring touches the trunk's bottom row: joined by a triangle
        for start in (0, 12 * 17):
            top = start + 12 * 16
            near = 2 * 12 * 17 + int(np.argmin(np.linalg.norm(parts[2] - parts[0 if start == 0 else 1][-12], axis=1)))
            faces.append(np.array([[top, top + 1, near, near]]))
        cape, cape_faces = _sheet(-0.45, 0.45, 0.3, 1.3, 0.2, 19, 21, offset)
        cls.cape = np.arange(offset, offset + len(cape))
        verts = np.vstack(parts + [cape])
        data = bpy.data.meshes.new("body")
        data.from_pydata(verts.tolist(), [], [[int(i) for i in dict.fromkeys(f)] for f in np.vstack(faces + [cape_faces])])
        cls.mesh = bpy.data.objects.new("body", data)
        bpy.context.scene.collection.objects.link(cls.mesh)
        cls.armature = rig.build_armature(JOINTS)
        groups = {name: cls.mesh.vertex_groups.new(name=name) for name in rig.BONE_NAMES.values()}
        for i in cls.body:
            x, _, z = verts[i]
            side = "left" if x > 0 else "right"
            key = (f"{side}_lower_leg" if z < 0.45 else f"{side}_upper_leg" if z < 0.9 else
                   f"{side}_upper_arm" if abs(x) > 0.2 and z < 1.32 else "head" if z > 1.38 else "chest")
            groups[rig.BONE_NAMES[key]].add([int(i)], 1.0, "REPLACE")
        for i in cls.cape:
            side = "left" if verts[i][0] > 0 else "right"
            groups[rig.BONE_NAMES[f"{side}_upper_arm"]].add([int(i)], 0.8, "REPLACE")
            groups[rig.BONE_NAMES[f"{side}_shoulder"]].add([int(i)], 0.2, "REPLACE")
        cls.verts = verts
        cls.before = cls.weights(cls.body)
        cls.changed = cloth.hang_from_torso(cls.mesh, cls.armature, JOINTS)
        cls.chains = cloth.build_cloth_chains(cls.mesh, cls.armature, JOINTS)

    @classmethod
    def weights(cls, indices):
        names = {g.index: g.name for g in cls.mesh.vertex_groups}
        return [{names[g.group]: round(g.weight, 4) for g in cls.mesh.data.vertices[int(i)].groups if g.weight > 0}
                for i in indices]

    def test_the_cape_leaves_the_arms(self):
        self.assertEqual(self.changed, len(self.cape))
        for weights in self.weights(self.cape):
            self.assertFalse([name for name in weights if name.startswith(("Left", "Right"))], weights)
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)

    def test_it_hangs_from_the_chest_and_swings_below_the_hips(self):
        top = self.weights(self.cape[self.verts[self.cape, 2] > 1.25])
        self.assertTrue(all(set(w) <= {"UpperChest", "Chest"} for w in top))
        hem = self.weights(self.cape[self.verts[self.cape, 2] < 0.5])
        self.assertTrue(all(any(name.startswith("Cloth") for name in w) for w in hem))
        self.assertGreaterEqual(len(self.chains), 1)

    def test_the_body_keeps_its_arms_and_its_legs(self):
        # one surface with arms and a head: most of it is far from the leg bones, and its legs
        # once went onto cloth chains
        self.assertEqual(self.weights(self.body), self.before)


class HairOnTheArms(unittest.TestCase):
    def test_hair_on_the_head_gives_up_its_arm_weight(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        verts, faces = _sheet(-0.1, 0.1, 1.5, 1.6, 0.0, 5, 5, 0)
        data = bpy.data.meshes.new("hair")
        data.from_pydata(verts.tolist(), [], faces.tolist())
        mesh = bpy.data.objects.new("hair", data)
        bpy.context.scene.collection.objects.link(mesh)
        armature = rig.build_armature(JOINTS)
        for name, weight in (("Head", 0.5), ("LeftUpperArm", 0.3), ("RightUpperArm", 0.2)):
            mesh.vertex_groups.new(name=name).add(list(range(len(verts))), weight, "REPLACE")
        mesh.vertex_groups["Head"].remove([0])
        features = {"left_eye": np.array([0.03, -0.1, 1.5]), "right_eye": np.array([-0.03, -0.1, 1.5])}
        hair.build_hair_bones(mesh, armature, JOINTS, features, np.ones(len(verts), dtype=bool), off_arms=True)
        names = {g.index: g.name for g in mesh.vertex_groups}
        for v in mesh.data.vertices:
            self.assertEqual({names[g.group]: round(g.weight, 4) for g in v.groups}, {"Head": 1.0})


if __name__ == "__main__":
    unittest.main()
