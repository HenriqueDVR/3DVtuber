"""The character manifest: pieces named by a point and a size, rebuilt with the roles the user gave them."""
import json
import tempfile
import unittest
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

from support import FIXTURE_OUT, load_fixture
from test_fingers import deformed
from vtuber_maker.pipeline import build, landmarks, manifest, meshdata, normalize, parts, rig, roles, tpose, vrm_export


def piece(name, point, count, role, **extra):
    return {"name": name, "point": list(point), "count": count, "role": role, **extra}


class Manifest(unittest.TestCase):
    def good(self):
        data = manifest.new("model.glb", "Test")
        data["pieces"] = [piece("skirt", (0, 0, .8), 10, "fitted_skirt"),
                          piece("cuff", (.2, 0, .9), 4, "body"),
                          piece("trim", (.21, 0, .9), 2, "detail", anchor="cuff"),
                          piece("sleeve", (.2, 0, 1), 5, "follow_bone", bone="LeftUpperArm"),
                          piece("tail", (-.2, 0, 1.3), 6, "hair", region=[0, 3, 5])]
        return data

    def test_a_complete_manifest_passes(self):
        manifest.validate(self.good())

    def test_names_every_problem_at_once(self):
        data = self.good()
        data["pieces"] += [piece("sleeve", (0, 0, 0), 0, "cape"),
                           piece("prop", (0, 0), 3, "follow_bone"),
                           piece("pin", (0, 0, 0), 3, "detail", anchor="nowhere"),
                           piece("bow", (0, 0, 1), 3, "body", region=[1, -2])]
        data["options"]["eyes"] = "googly"
        data["joints"]["tail"] = [0, 0, 0]
        with self.assertRaises(ValueError) as raised:
            manifest.validate(data)
        message = str(raised.exception)
        for problem in ("used twice", "count must", "role must", "point must", "needs a bone", "needs an anchor",
                        "eyes must", "unknown joint", "region must"):
            self.assertIn(problem, message)

    def test_round_trips_and_resolves_paths_beside_itself(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "character.json"
            data = self.good()
            data["face"] = "face.json"
            manifest.save(data, path)
            loaded = manifest.load(path)
            self.assertEqual(loaded["pieces"], data["pieces"])
            self.assertEqual(manifest.path_of(loaded, "face"), (Path(folder) / "face.json").resolve())
            self.assertIsNone(manifest.path_of(loaded, "reference"))
            self.assertNotIn("_path", json.loads(path.read_text()))

    def test_options_missing_from_the_file_take_their_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "character.json"
            path.write_text(json.dumps({"version": 1, "name": "A", "source": "a.glb", "options": {"eyes": "slide"}}))
            loaded = manifest.load(path)
            self.assertEqual(loaded["options"]["eyes"], "slide")
            self.assertEqual(loaded["options"]["fingers"], manifest.OPTIONS["fingers"])

    def test_stages_split_the_roles(self):
        data = self.good()
        self.assertEqual([p["name"] for p in manifest.pieces_for(data, "build")], ["cuff", "tail"])
        self.assertEqual([p["name"] for p in manifest.pieces_for(data, "pieces")], ["skirt", "trim", "sleeve"])

    def test_joint_keys_are_the_rigs(self):
        self.assertEqual(set(manifest.JOINTS), set(rig.BONE_NAMES) - {"left_shoulder", "right_shoulder"})


class Roles(unittest.TestCase):
    def setUp(self):
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        joints = {"hips": (0, 0, .8), "neck": (0, 0, 1.3), "head": (0, 0, 1.4)}
        for side, sign in (("left", 1), ("right", -1)):
            joints.update({f"{side}_upper_arm": (sign * .15, 0, 1.2),
                           f"{side}_lower_arm": (sign * .23, 0, 1.0),
                           f"{side}_hand": (sign * .30, 0, .8),
                           f"{side}_upper_leg": (sign * .07, 0, .75),
                           f"{side}_lower_leg": (sign * .07, 0, .4),
                           f"{side}_foot": (sign * .07, 0, .08)})
        self.arm = rig.build_armature(joints)

    def mesh(self, points, weights, faces):
        data = bpy.data.meshes.new("Clothing")
        data.from_pydata(points, [], faces)
        obj = bpy.data.objects.new("Clothing", data)
        bpy.context.scene.collection.objects.link(obj)
        for i, row in enumerate(weights):
            for name, w in row.items():
                group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
                group.add([i], w, "REPLACE")
        obj.modifiers.new("Armature", "ARMATURE").object = self.arm
        bpy.context.view_layer.update()
        return obj

    def sleeve_and_hand(self):
        # a sleeve hanging by the ribs, skinned to the chest, and a hand beside it
        points = [(.16, -.05, 1.15), (.25, -.05, .95), (.30, .02, .82),
                  (.30, 0, .78), (.31, 0, .76), (.30, .01, .74)]
        return self.mesh(points, [{"Chest": 1}] * 3 + [{"LeftHand": 1}] * 3, [(0, 1, 2), (3, 4, 5)])

    def test_finds_a_piece_by_its_point_and_refuses_another_size(self):
        obj = self.sleeve_and_hand()
        masks = roles.find(obj, [piece("sleeve", (.25, -.05, .95), 3, "follow_bone", bone="LeftUpperArm")])
        np.testing.assert_array_equal(masks["sleeve"], [True] * 3 + [False] * 3)
        with self.assertRaisesRegex(ValueError, "3 vertices"):
            roles.find(obj, [piece("sleeve", (.25, -.05, .95), 4, "body")])
        with self.assertRaisesRegex(ValueError, "within 1 mm"):
            roles.find(obj, [piece("sleeve", (.25, -.05, .90), 3, "body")])

    def test_named_sleeve_rides_the_arm_and_the_hand_is_left_alone(self):
        obj = self.sleeve_and_hand()
        pieces = [piece("sleeve", (.16, -.05, 1.15), 3, "follow_bone", bone="LeftUpperArm")]
        hands = roles.hands_snapshot(obj)
        report, origin = roles.apply(obj, self.arm, pieces, roles.find(obj, pieces))
        self.assertEqual(report["sleeve"], 3)
        self.assertEqual(roles.hands_snapshot(obj, origin), hands)
        rest = meshdata.world_vertices(obj)
        tpose._point_arm(self.arm, "LeftUpperArm", "LeftHand", Vector((1, 0, 0)))
        bone = self.arm.pose.bones["LeftUpperArm"]
        expected = [bone.matrix @ bone.bone.matrix_local.inverted() @ Vector(p) for p in rest[:3]]
        np.testing.assert_allclose(deformed(obj)[:3], expected, atol=1e-6)

    def test_detail_rides_its_named_anchor(self):
        obj = self.mesh([(-.28, -.04, .85), (-.281, -.04, .85), (-.279, -.04, .851), (-.282, -.04, .86)],
                        [{"Hips": 1}, {"RightLowerArm": .8, "RightHand": .2}, {"RightLowerArm": 1}, {"Hips": 1}],
                        [(1, 2, 3)])
        obj.data.vertices[0].co.z = .84  # the trim is a lone vertex beside the cuff triangle
        pieces = [piece("cuff", (-.281, -.04, .85), 3, "body"), piece("trim", (-.28, -.04, .84), 1, "detail",
                                                                      anchor="cuff")]
        rest = meshdata.world_vertices(obj)
        roles.apply(obj, self.arm, pieces[1:], roles.find(obj, pieces))
        tpose._point_arm(self.arm, "RightUpperArm", "RightHand", Vector((-1, 0, 0)))
        moved = deformed(obj)
        self.assertGreater(np.linalg.norm(moved[0] - rest[0]), .05)
        self.assertLess(abs(np.linalg.norm(moved[0] - moved[1]) - np.linalg.norm(rest[0] - rest[1])), .001)

    def test_a_picked_region_is_closed_over_its_own_surface(self):
        # a 12 x 12 grid: the region is its left half with two vertices missed inside it and one
        # at its edge, as a box pick behind the surface leaves them; the right half stays out
        n = 12
        points = [(x * .01, 0, z * .01) for z in range(n) for x in range(n)]
        faces = [(z * n + x, z * n + x + 1, (z + 1) * n + x + 1, (z + 1) * n + x) for z in range(n - 1) for x in range(n - 1)]
        grid = self.mesh(points, [{"Hips": 1}] * n * n, faces)
        left = np.array([x < n // 2 for z in range(n) for x in range(n)])
        picked = left.copy()
        missed = [3 * n + 2, 7 * n + 3, 5 * n + 5]
        picked[missed] = False
        closed = roles.close_region(grid, np.ones(n * n, bool), picked)
        self.assertTrue(closed[missed].all())
        self.assertFalse(closed[~left].any())
        # nothing outside the piece is taken
        self.assertFalse(roles.close_region(grid, left & ~np.isin(np.arange(n * n), [3 * n + 2]), picked)[3 * n + 2])

    def test_a_vertex_cut_into_a_piece_stays_on_it_though_another_piece_is_nearer(self):
        # the reference: a long skirt edge, and a sleeve vertex 6 mm from its middle (and, far
        # off, the rest of a model, so the few new vertices are the exception they are on a real one)
        rest = [(1 + i * .01, 0, 0) for i in range(40)]
        ref = self.mesh([(0, 0, .8), (.04, 0, .8), (.02, .03, .8), (.02, -.006, .8), (.03, -.02, .8), (.01, -.02, .8),
                         *rest], [{"Hips": 1}] * 46, [(0, 1, 2), (3, 4, 5)])
        # the rig: the same, plus a vertex a cut added halfway along the skirt edge
        rig = self.mesh([(0, 0, .8), (.04, 0, .8), (.02, .03, .8), (.02, -.006, .8), (.03, -.02, .8), (.01, -.02, .8),
                         *rest, (.02, 0, .8)], [{"Hips": 1}] * 47, [(0, 46, 2), (46, 1, 2), (3, 4, 5)])
        masks = roles.find(rig, [piece("skirt", (0, 0, .8), 3, "fitted_skirt")], ref)
        np.testing.assert_array_equal(np.flatnonzero(masks["skirt"]), [0, 1, 2, 46])

    def test_masks_follow_parts_through_a_cut(self):
        before = np.array([0, 0, 1, 1, 2])
        after = np.array([0, 0, 1, 1, 2, 1, 0])
        carried = roles.carry({"a": before == 1}, before, after)
        np.testing.assert_array_equal(carried["a"], after == 1)
        with self.assertRaisesRegex(ValueError, "whole part"):
            roles.carry({"b": np.array([True, False, False, False, False])}, before, after)

    def test_listed_pieces_overrule_detected_hair(self):
        hair = np.array([True, True, False, False])
        pieces = [piece("hat", (0, 0, 0), 1, "hat"), piece("lock", (0, 0, 0), 1, "hair")]
        masks = {"hat": np.array([True, False, False, False]), "lock": np.array([False, False, True, False])}
        np.testing.assert_array_equal(roles.hair_mask(hair, pieces, masks), [False, True, True, False])


class BuildWithPieces(unittest.TestCase):
    """A hat among hair pieces: the manifest puts it wholly on the head, not on strands."""

    @classmethod
    def setUpClass(cls):
        mesh, _ = load_fixture("synthetic_character_parts")
        mesh = normalize.prepare([mesh])
        verts = meshdata.world_vertices(mesh)
        found = parts.find(mesh, landmarks.detect_body(verts, meshdata.triangles(mesh.data)))
        labels = meshdata.pieces(mesh, verts)
        hair_ids = np.unique(labels[found.hair])
        # the biggest hair piece becomes the hat
        hat_id = max(hair_ids, key=lambda i: np.count_nonzero(labels == i))
        cls.hat_points = verts[labels == hat_id]
        top = int(np.argmax(np.where(labels == hat_id, verts[:, 2], -np.inf)))
        pieces = [piece("hat", verts[top], int(np.count_nonzero(labels == hat_id)), "hat")]

        mesh, _ = load_fixture("synthetic_character_parts")
        vrm_export.ensure_vrm_addon()
        path = FIXTURE_OUT / "synthetic_character_hat_built.vrm"
        path.unlink(missing_ok=True)
        cls.report = build.build_avatar([mesh], path, name="Hat", author="Tests", pieces=pieces)
        cls.mesh = next(o for o in bpy.data.objects if o.type == "MESH")

    def test_the_hat_is_wholly_on_the_head(self):
        tree = KDTree(len(self.hat_points))
        for i, point in enumerate(self.hat_points):
            tree.insert(point, i)
        tree.balance()
        on_hat = [i for i, p in enumerate(meshdata.world_vertices(self.mesh)) if tree.find(p)[2] < 1e-5]
        self.assertGreaterEqual(len(on_hat), len(self.hat_points))
        groups = self.mesh.vertex_groups
        for index in on_hat:
            weights = {groups[g.group].name: g.weight for g in self.mesh.data.vertices[int(index)].groups if g.weight > 1e-6}
            self.assertEqual(set(weights), {"Head"}, weights)


if __name__ == "__main__":
    unittest.main()
