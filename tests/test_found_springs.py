"""What a default build swings by itself: wings, a cape and a thin tail on the back (roles.found_behind),
a cape that ends above the hips, and sleeves hanging deep under the arms (cloth.build_sleeve_chains)."""

import unittest
from types import SimpleNamespace

import bpy
import numpy as np

from vtuber_maker.pipeline import cloth, rig, roles

JOINTS = {key: np.array(p, float) for key, p in {
    "hips": (0, 0, 0.8), "spine": (0, 0, 0.95), "chest": (0, 0, 1.1), "upper_chest": (0, 0, 1.22),
    "neck": (0, 0, 1.35), "head": (0, 0, 1.42),
    "left_upper_arm": (0.15, 0, 1.3), "left_lower_arm": (0.38, 0, 1.3), "left_hand": (0.6, 0, 1.3),
    "right_upper_arm": (-0.15, 0, 1.3), "right_lower_arm": (-0.38, 0, 1.3), "right_hand": (-0.6, 0, 1.3),
    "left_upper_leg": (0.1, 0, 0.8), "left_lower_leg": (0.1, 0, 0.45), "left_foot": (0.1, 0, 0.05),
    "right_upper_leg": (-0.1, 0, 0.8), "right_lower_leg": (-0.1, 0, 0.45), "right_foot": (-0.1, 0, 0.05),
}.items()}


def _sheet(x0, x1, z0, z1, y, nx, nz):
    xs, zs = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(z0, z1, nz), indexing="ij")
    verts = np.stack([xs.ravel(), np.full(xs.size, y), zs.ravel()], axis=1)
    index = np.arange(nx * nz).reshape(nx, nz)
    faces = np.stack([index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()], axis=1)
    return verts, faces


def _tube(x, y, z0, z1, radius, rings=40, around=8):
    angle = np.linspace(0, 2 * np.pi, around, endpoint=False)
    zs = np.linspace(z1, z0, rings)
    verts = np.array([(x + radius * np.cos(a), y + radius * np.sin(a), z) for z in zs for a in angle])
    faces = [(r * around + k, r * around + (k + 1) % around, (r + 1) * around + (k + 1) % around, (r + 1) * around + k)
             for r in range(rings - 1) for k in range(around)]
    return verts, np.array(faces)


def _mesh(parts):
    """One mesh object of separate pieces [(verts, faces)]; returns it and each piece's vertex range."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    verts, faces, ranges = [], [], []
    for v, f in parts:
        start = sum(len(p) for p in verts)
        verts.append(v)
        faces.append(f + start)
        ranges.append(range(start, start + len(v)))
    data = bpy.data.meshes.new("body")
    data.from_pydata(np.vstack(verts).tolist(), [], np.vstack(faces).tolist())
    obj = bpy.data.objects.new("body", data)
    bpy.context.scene.collection.objects.link(obj)
    return obj, ranges


def _body():
    return _sheet(-0.15, 0.15, 0.0, 1.6, 0.0, 31, 161)


class FoundBehind(unittest.TestCase):
    def found(self, mesh):
        n = len(mesh.data.vertices)
        return roles.found_behind(mesh, JOINTS, SimpleNamespace(hair=np.zeros(n, bool), face_area=np.zeros(n, bool)), 1.6)

    def test_a_wing_to_one_side_is_an_appendage(self):
        mesh, ranges = _mesh([_body(), _sheet(0.05, 0.45, 1.0, 1.4, 0.2, 41, 41)])
        found = self.found(mesh)
        self.assertEqual([role for role, _ in found], ["appendage"])
        self.assertEqual(set(np.flatnonzero(found[0][1])), set(ranges[1]))

    def test_a_cape_across_the_back_is_cloth(self):
        mesh, _ = _mesh([_body(), _sheet(-0.27, 0.27, 0.95, 1.4, 0.2, 41, 41)])
        self.assertEqual([role for role, _ in self.found(mesh)], ["swing_cloth"])

    def test_hair_that_grows_from_the_head_is_left_to_the_hair(self):
        mesh, ranges = _mesh([_body(), _sheet(-0.1, 0.1, 0.9, 1.6, 0.1, 21, 71)])
        n = len(mesh.data.vertices)
        self.assertEqual([role for role, _ in self.found(mesh)], ["appendage"])  # not hair, it would swing
        hair = np.zeros(n, bool)
        hair[list(ranges[1])] = True
        found = roles.found_behind(mesh, JOINTS, SimpleNamespace(hair=hair, face_area=np.zeros(n, bool)), 1.6)
        self.assertEqual(found, [])

    def test_a_thin_tail_is_found_and_a_thick_one_is_not(self):
        mesh, ranges = _mesh([_body(), _tube(0.0, 0.25, 0.3, 0.85, 0.01)])
        found = self.found(mesh)
        self.assertEqual([role for role, _ in found], ["appendage"])
        self.assertTrue(set(np.flatnonzero(found[0][1])) <= set(ranges[1]))
        mesh, _ = _mesh([_body(), _tube(0.0, 0.3, 0.3, 0.85, 0.1)])
        self.assertEqual(self.found(mesh), [])
        n = len(mesh.data.vertices)
        self.assertEqual(roles.found_behind(mesh, JOINTS, SimpleNamespace(hair=np.zeros(n, bool), face_area=np.zeros(n, bool)),
                                            1.6, tails=False), [])


def _rigged(parts, groups):
    """`_mesh`, an armature on JOINTS and each piece's vertices wholly on its bone in `groups`."""
    mesh, ranges = _mesh(parts)
    armature = rig.build_armature(JOINTS)
    for bone, part in zip(groups, ranges):
        group = mesh.vertex_groups.get(rig.BONE_NAMES[bone]) or mesh.vertex_groups.new(name=rig.BONE_NAMES[bone])
        group.add(list(part), 1.0, "REPLACE")
    return mesh, armature, ranges


def _on(mesh, index, prefix):
    names = {g.index: g.name for g in mesh.vertex_groups}
    return sum(g.weight for g in mesh.data.vertices[index].groups if names[g.group].startswith(prefix))


class CapeAboveTheHips(unittest.TestCase):
    def test_it_hangs_from_its_own_top_on_the_torso(self):
        mesh, armature, ranges = _rigged([_body(), _sheet(-0.27, 0.27, 0.95, 1.4, 0.2, 41, 41)], ["hips", "upper_chest"])
        force = np.zeros(len(mesh.data.vertices), bool)
        force[list(ranges[1])] = True
        chains = cloth.build_cloth_chains(mesh, armature, JOINTS, force=force, detect=False)
        self.assertGreater(len(chains), 1)
        self.assertEqual({armature.data.bones[names[0]].parent.name for names in chains}, {rig.BONE_NAMES["upper_chest"]})
        hem = [i for i in ranges[1] if mesh.data.vertices[i].co.z < 1.0]
        self.assertGreater(min(_on(mesh, i, "Cloth") for i in hem), 0.99)


class HangingSleeves(unittest.TestCase):
    def test_a_deep_sleeve_hangs_on_chains_from_the_arm(self):
        mesh, armature, ranges = _rigged([_body(), _sheet(0.3, 0.55, 0.7, 1.25, 0.0, 26, 56)], ["hips", "left_lower_arm"])
        chains = cloth.build_sleeve_chains(mesh, armature, JOINTS)
        self.assertTrue(chains)  # one or two a sleeve: SLEEVE_SPAN
        arm = {rig.BONE_NAMES["left_upper_arm"], rig.BONE_NAMES["left_lower_arm"]}
        self.assertTrue({armature.data.bones[names[0]].parent.name for names in chains} <= arm)
        hem = [i for i in ranges[1] if mesh.data.vertices[i].co.z < 0.8]
        self.assertGreater(min(_on(mesh, i, "Sleeve") for i in hem), 0.99)
        self.assertEqual(max(_on(mesh, i, "Sleeve") for i in ranges[0]), 0)

    def test_a_fitted_sleeve_gets_none(self):
        mesh, armature, _ = _rigged([_body(), _sheet(0.3, 0.55, 1.15, 1.25, 0.0, 26, 11)], ["hips", "left_lower_arm"])
        self.assertEqual(cloth.build_sleeve_chains(mesh, armature, JOINTS), [])


class SkirtColliders(unittest.TestCase):
    """The skirt's colliders push the sleeves that name their group, and never the skirt's own chains."""

    def test_only_a_chain_naming_the_group_is_pushed(self):
        from vtuber_maker.pipeline import spring_sim
        points = np.array([[0.0, 0.0, 1.0 - 0.1 * k] for k in range(5)])
        skirt = spring_sim.Collider("Hips", 0.05, offset=[0.01, 0.0, 0.8], group="Skirt")
        still = lambda frame: {"Hips": np.eye(4)}  # noqa: E731
        moved = {}
        for groups in ((), ("Skirt",)):
            chain = spring_sim.Chain(points, "Hips", stiffness=1.0, gravity_power=0.0, drag_force=0.5, hit_radius=0.01,
                                     groups=groups)
            last = spring_sim.simulate([chain], [skirt], still, 30)[-1]
            moved[groups] = float(np.linalg.norm(last.positions[0] - points, axis=1).max())
        self.assertLess(moved[()], 1e-6)
        self.assertGreater(moved[("Skirt",)], 0.01)


if __name__ == "__main__":
    unittest.main()
