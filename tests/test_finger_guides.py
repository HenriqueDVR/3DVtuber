"""Finger bones laid down the middle of each finger, the thumb found, and a clean curl after baking."""
import json
import math
import unittest

import bpy
import numpy as np
from mathutils import Matrix, Vector

from vtuber_maker.pipeline import finger_guides, fingers, rig, tpose
from test_fingers import deformed

WRIST = Vector((0.0, 0.0, 1.0))
PALM = .08
# index, middle, ring, little: across the palm (x, thumb side +x) and length
FINGERS = ((.036, .075), (.012, .085), (-.012, .08), (-.036, .065))
RADIUS = .009
THUMB_BASE = Vector((.03, 0.0, 1.02))
THUMB = Vector((.7, -.3, .65)).normalized() * .075


def segment(head, tail, radius):
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius, depth=(tail - head).length,
                                        location=(head + tail) / 2)
    obj = bpy.context.object
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(tail - head)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=tail, segments=12, ring_count=8)
    return [obj, bpy.context.object]


def finger_line(knuckle, length, curl):
    """Three phalanges from a knuckle, each bending `curl` radians further toward the palm (-y)."""
    points, direction, angle = [knuckle], Vector((0, 0, 1)), 0.0
    for share in (.45, .30, .25):
        points.append(points[-1] + direction * share * length)
        angle += curl
        direction = Vector((0, -math.sin(angle), math.cos(angle)))
    return points


def build(curl=0.0, mitten=False):
    """A hand past the wrist at z=1, fingers up +z, palm facing -y, thumb on the +x side."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.mesh.primitive_cube_add(size=1, location=WRIST + Vector((0, 0, PALM / 2)))
    palm = bpy.context.object
    palm.scale = (.1, .03, PALM)
    parts, lines = [palm], []
    for x, length in ([] if mitten else FINGERS):
        line = finger_line(WRIST + Vector((x, 0, PALM - .01)), length, curl)
        lines.append(line)
        for a, b in zip(line, line[1:]):
            parts += segment(a, b, RADIUS)
    if not mitten:
        tip = THUMB_BASE + THUMB
        lines.insert(0, [THUMB_BASE, tip])
        parts += segment(THUMB_BASE, tip, RADIUS * 1.15)
    bpy.ops.object.select_all(action="DESELECT")
    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active = palm
    bpy.ops.object.join()
    hand = bpy.context.object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    remesh = hand.modifiers.new("Remesh", "REMESH")
    remesh.mode, remesh.voxel_size = "VOXEL", .003
    rig.make_active(hand)
    bpy.ops.object.modifier_apply(modifier=remesh.name)

    data = bpy.data.armatures.new("Armature")
    arm = bpy.data.objects.new("Armature", data)
    bpy.context.scene.collection.objects.link(arm)
    rig.make_active(arm)
    bpy.ops.object.mode_set(mode="EDIT")
    bone = data.edit_bones.new(rig.BONE_NAMES["left_hand"])
    bone.head, bone.tail = WRIST, WRIST + Vector((0, 0, PALM))
    bpy.ops.object.mode_set(mode="OBJECT")
    group = hand.vertex_groups.new(name=rig.BONE_NAMES["left_hand"])
    group.add(range(len(hand.data.vertices)), 1.0, "REPLACE")
    hand.modifiers.new("Armature", "ARMATURE").object = arm
    bpy.context.view_layer.update()
    return hand, arm, lines


def distance_to_line(point, line):
    point = np.array(point)
    best = np.inf
    for a, b in zip(line, line[1:]):
        a, b = np.array(a), np.array(b)
        t = np.clip((point - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
        best = min(best, float(np.linalg.norm(point - (a + t * (b - a)))))
    return best


class FingerGuides(unittest.TestCase):
    def check_placement(self, curl):
        mesh, arm, lines = build(curl)
        data, report, _ = finger_guides.guides(mesh, arm, "left")
        for name, line in zip(fingers.FINGERS, lines):
            points = data[name]["points"]
            # the tip is the fingertip, not a bump on the knuckles or a neighbour's tip
            self.assertLess(np.linalg.norm(np.array(points[-1]) - np.array(line[-1])), RADIUS * 1.6,
                            (name, report[name]))
            # the joints outside the palm lie inside the finger, near its middle (the tip is a
            # point on the fingertip's surface, checked above)
            for point in points[1:-1]:
                self.assertLess(distance_to_line(point, line), RADIUS * .9, (name, point, report[name]))
        return mesh, arm

    def test_straight_hand_finds_every_finger_and_the_thumb(self):
        self.check_placement(0.0)

    def test_curled_hand_bones_follow_the_curl(self):
        self.check_placement(math.radians(35))

    def test_rebuilt_and_baked_hand_curls_toward_the_palm_without_tearing(self):
        mesh, arm, _ = build(math.radians(35))
        finger_guides.rebuild(mesh, arm, sides=("left",))
        tpose.to_t_pose(mesh, arm)
        rest = deformed(mesh)
        edges = np.array([e.vertices[:] for e in mesh.data.edges])
        before = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
        hand = arm.pose.bones[rig.BONE_NAMES["left_hand"]]
        along = (hand.tail - hand.head).normalized()
        across = tpose._across_palm(arm, "left", along)
        palm = along.cross(across)
        tips = {f: arm.pose.bones[fingers.bone_name("left", f, fingers._segments(f)[-1])].tail.copy()
                for f in fingers.FINGERS}
        for finger in fingers.FINGERS:
            for segment in fingers._segments(finger):
                bone = arm.pose.bones[fingers.bone_name("left", finger, segment)]
                bone.rotation_mode = "XYZ"
                bone.rotation_euler = (math.radians(35), 0, 0)
        bpy.context.view_layer.update()
        for finger in fingers.FINGERS[1:]:
            moved = arm.pose.bones[fingers.bone_name("left", finger, "distal")].tail - tips[finger]
            self.assertGreater(moved.dot(palm), .02, finger)
            self.assertLess(abs(moved.normalized().dot(across)), .15, f"{finger} curls sideways")
        after = deformed(mesh)
        grown = np.linalg.norm(after[edges[:, 0]] - after[edges[:, 1]], axis=1) - before
        self.assertLess(float(grown.max()), .02, f"an edge stretched {grown.max() * 100:.1f} cm")
        for bone in arm.pose.bones:
            bone.matrix_basis = Matrix.Identity(4)

    def test_weights_are_laid_out_like_the_reference_hand(self):
        mesh, arm, _ = build(math.radians(35))
        finger_guides.rebuild(mesh, arm, sides=("left",))
        data = json.loads(arm["vtuber_hand_setup_left"])
        count = len(mesh.data.vertices)
        weight = {g.name: np.zeros(count) for g in mesh.vertex_groups}
        names = {g.index: g.name for g in mesh.vertex_groups}
        for vertex in mesh.data.vertices:
            for entry in vertex.groups:
                weight[names[entry.group]][vertex.index] = entry.weight
        coords = np.array([v.co for v in mesh.data.vertices])
        wrist = np.array(WRIST)
        frame = finger_guides._hand_frame(wrist, [data[f]["points"][0] for f in fingers.FINGERS[1:]], "left")
        # the thumb's first joint sits where the reference has it
        root = np.array(data["thumb"]["points"][0]) - wrist
        found = [root @ frame[a] / frame["length"] for a in ("along", "across", "palm")]
        np.testing.assert_allclose(found, finger_guides.THUMB_ROOT, atol=1e-6)
        # the ball of the thumb rides its first bone, on the thumb's side of the hand
        ball = weight[fingers.bone_name("left", "thumb", "metacarpal")] > .5
        self.assertGreater(int(ball.sum()), 20)
        self.assertGreater(float((coords[ball, 0] > 0).mean()), .95)
        for finger in fingers.FINGERS[1:]:
            rows = np.array([weight[fingers.bone_name("left", finger, s)] for s in fingers._segments(finger)])
            total = rows.sum(axis=0)
            # most of a finger is wholly on one of its bones; only the joints blend
            held = total > .9
            share = (rows[:, held].max(axis=0) / total[held] >= .95).mean()
            self.assertGreater(share, .5, finger)
            # and none of it reaches back into the palm past the fade behind the knuckle
            points = np.array(data[finger]["points"])
            along, _ = finger_guides._along_chain(coords, points, behind=True)
            reach = -finger_guides.KNUCKLE_FADE * frame["length"]
            self.assertGreaterEqual(float(along[total > 1e-4].min()), reach - 1e-9, finger)
            # nor into the palm under a curled finger's first bone
            first = (total > 1e-4) & (along < np.linalg.norm(points[1] - points[0]))
            self.assertGreaterEqual(float(((coords[first] - points[0]) @ frame["along"]).min()), reach - 1e-9, finger)

    def test_thumb_web_relaxes_and_nothing_else_moves(self):
        mesh, arm, _ = build(math.radians(35))
        finger_guides.rebuild(mesh, arm, sides=("left",))
        tpose.to_t_pose(mesh, arm, straighten="spec")
        before = np.array([v.co for v in mesh.data.vertices])
        self.assertGreater(tpose.relax_thumb_web(mesh, arm, "left"), 0)
        after = np.array([v.co for v in mesh.data.vertices])
        moved = np.linalg.norm(after - before, axis=1) > 1e-7
        names = {g.index: g.name for g in mesh.vertex_groups}
        strongest = [names[max(v.groups, key=lambda g: g.weight).group] for v in mesh.data.vertices]
        spared = {fingers.bone_name("left", f, s) for f in ("middle", "ring", "little") for s in fingers._segments(f)}
        spared |= {fingers.bone_name("left", "thumb", "distal"), fingers.bone_name("left", "index", "distal")}
        self.assertFalse(any(strongest[i] in spared for i in np.flatnonzero(moved)))
        wrist = np.array(arm.data.bones[rig.BONE_NAMES["left_hand"]].head_local)
        knuckles = [np.array(arm.data.bones[fingers.bone_name("left", f, "proximal")].head_local)
                    for f in fingers.FINGERS[1:]]
        frame = finger_guides._hand_frame(wrist, knuckles, "left")
        self.assertGreater(float(((before[moved] - wrist) @ frame["along"]).min()),
                           tpose.WEB_FROM * frame["length"] - 1e-9)
        # the moved skin ends up smoother: closer to the average of its neighbours
        edges = np.array([e.vertices[:] for e in mesh.data.edges])
        def roughness(co):
            summed, degree = np.zeros_like(co), np.bincount(edges.ravel(), minlength=len(co))
            np.add.at(summed, edges[:, 0], co[edges[:, 1]])
            np.add.at(summed, edges[:, 1], co[edges[:, 0]])
            return float(np.linalg.norm(summed[moved] / degree[moved, None] - co[moved], axis=1).mean())
        self.assertLess(roughness(after), roughness(before))

    def test_opened_fingers_lie_level_with_their_knuckles_at_one_height(self):
        mesh, arm, _ = build(math.radians(35))
        finger_guides.rebuild(mesh, arm, sides=("left",))
        # a knuckle placed a centimetre off, as the guides did on the real curled hands
        rig.make_active(arm)
        bpy.ops.object.mode_set(mode="EDIT")
        arm.data.edit_bones[fingers.bone_name("left", "ring", "proximal")].head.y += .01
        bpy.ops.object.mode_set(mode="OBJECT")
        tpose.to_t_pose(mesh, arm, straighten="spec")
        heights = []
        for finger in fingers.FINGERS[1:]:
            chain = [arm.data.bones[fingers.bone_name("left", finger, s)] for s in fingers._segments(finger)]
            joints = [chain[0].head_local.z] + [b.tail_local.z for b in chain]
            self.assertLess(max(joints) - min(joints), 5e-4, f"{finger} bones not level")
            heights.append(chain[0].head_local.z)
            # and the finger itself lies level: the centres of rings of its surface along it sit at
            # the knuckle's height, within a third of the 9 mm finger radius (from half way out:
            # nearer the knuckle this hand's joint spheres and web swing ring centres 6 mm either
            # way whether or not a knuckle was misplaced)
            names = {g.index: g.name for g in mesh.vertex_groups}
            ours = {b.name for b in chain}
            pts = np.array([v.co for v in mesh.data.vertices
                            if sum(g.weight for g in v.groups if names[g.group] in ours) > .5])
            head, tip = np.array(chain[0].head_local), np.array(chain[-1].tail_local)
            length = float(np.linalg.norm(tip - head))
            along = (pts - head) @ (tip - head) / length
            for share in (.5, .6, .75, .85, .95):
                ring = pts[np.abs(along - share * length) < .004]
                self.assertGreater(len(ring), 3)
                self.assertLess(abs(float(ring[:, 2].mean()) - head[2]), RADIUS / 3, f"{finger} at {share}")
        self.assertLess(max(heights) - min(heights), 5e-4, "knuckles at different heights")

    def test_opened_fingers_are_round_and_jointed_like_the_reference(self):
        mesh, arm, _ = build(math.radians(35))
        finger_guides.rebuild(mesh, arm, sides=("left",))
        tpose.to_t_pose(mesh, arm, straighten="spec")
        names = {g.index: g.name for g in mesh.vertex_groups}
        for finger in fingers.FINGERS[1:]:
            chain = [arm.data.bones[fingers.bone_name("left", finger, s)] for s in fingers._segments(finger)]
            head, tip = np.array(chain[0].head_local), np.array(chain[-1].tail_local)
            length = float(np.linalg.norm(tip - head))
            along = (tip - head) / length
            # joints where the reference hand has them
            for joint, share in zip((chain[1].head_local, chain[2].head_local), tpose.REF_JOINTS):
                self.assertAlmostEqual(float((np.array(joint) - head) @ along) / length, share, places=4)
            ours = {b.name for b in chain}
            held = [(np.array(v.co), {names[g.group]: g.weight for g in v.groups}) for v in mesh.data.vertices]
            pts = np.array([p for p, w in held if sum(w.get(n, 0) for n in ours) > .9])
            s = (pts - head) @ along
            r = np.linalg.norm((pts - head) - np.outer(s, along), axis=1)
            # round: past the web each ring of the finger's own tube is one distance from the axis
            # (this hand's web reaches 40% out; web skin is left as it is)
            for share in (.5, .7):
                ring = r[np.abs(s - share * length) < .0015]
                ring = ring[ring < 1.5 * np.median(ring)] if len(ring) else ring
                if len(ring) >= 5:
                    self.assertLess(float(np.std(ring) / np.mean(ring)), .05, f"{finger} at {share}")
            # and the middle of each segment is wholly its own bone's
            joints = [0, tpose.REF_JOINTS[0], tpose.REF_JOINTS[1], 1]
            for k, bone in enumerate(chain):
                middle = (joints[k] + joints[k + 1]) / 2 * length
                for p, w in held:
                    if abs((p - head) @ along - middle) < .002 and sum(w.get(n, 0) for n in ours) > .99:
                        self.assertGreater(w.get(bone.name, 0), .95, bone.name)

    def test_bones_follow_a_finger_reshaped_by_hand(self):
        mesh, arm, _ = build(math.radians(35))
        finger_guides.rebuild(mesh, arm, sides=("left",))
        tpose.to_t_pose(mesh, arm, straighten="spec")
        # names, not bone references: those go stale when the armature leaves edit mode
        chain = [fingers.bone_name("left", "middle", s) for s in fingers._segments("middle")]
        bones = lambda: arm.data.bones
        head, tip = np.array(bones()[chain[0]].head_local), np.array(bones()[chain[-1]].tail_local)
        length = float(np.linalg.norm(tip - head))
        along = (tip - head) / length
        # bend the outer half of the finger 1 cm sideways toward the thumb (the front, -y, on
        # this left hand held out along +x), as a person sculpting might
        names = {g.index: g.name for g in mesh.vertex_groups}
        for vertex in mesh.data.vertices:
            held = sum(g.weight for g in vertex.groups if names[g.group] in chain)
            s = (np.array(vertex.co) - head) @ along / length
            if held > .5 and s > .4:
                vertex.co.y -= .01 * min((s - .4) / .6, 1.0)
        rest = np.array([v.co for v in mesh.data.vertices])
        before, after = tpose.fit_finger_bones([mesh], arm, "left", "middle")
        self.assertGreater(before, .005)
        self.assertLess(after, .002)
        # the mesh stays where the person put it
        np.testing.assert_allclose(np.array([v.co for v in mesh.data.vertices]), rest, atol=1e-9)
        # the knuckle stays, the tip bone now leans with the finger, and the chain kept its length
        new = [np.array(bones()[chain[0]].head_local)] + [np.array(bones()[n].tail_local) for n in chain]
        np.testing.assert_allclose(new[0], head, atol=1e-9)
        self.assertLess(new[-1][1] - tip[1], -.007)
        # (within 2%: this hand's knuckle sits 10 mm off the finger's line inside its box palm,
        # which lengthens the first bone by about 1 mm; on model B it sits 0.8 mm off)
        self.assertLess(abs(sum(float(np.linalg.norm(b - a)) for a, b in zip(new, new[1:])) - length), .02 * length)

    def test_refuses_a_mitten_rather_than_guessing(self):
        mesh, arm, _ = build(mitten=True)
        with self.assertRaises(ValueError):
            finger_guides.guides(mesh, arm, "left")


class UntangledFan(unittest.TestCase):
    def test_crossing_fingers_are_kept_in_order_and_centred(self):
        from test_tpose import curved_hand
        _, arm = curved_hand()
        rig.make_active(arm)
        bpy.ops.object.mode_set(mode="EDIT")
        # the ring finger's first bone points far toward the little finger, past it
        ring = arm.data.edit_bones[fingers.bone_name("left", "ring", "proximal")]
        ring.tail = ring.head + Vector((1, .9, -.4)).normalized() * .018
        bpy.ops.object.mode_set(mode="OBJECT")
        hand = arm.pose.bones[rig.BONE_NAMES["left_hand"]]
        along = (hand.tail - hand.head).normalized()
        across = tpose._across_palm(arm, "left", along)
        fan = tpose._fan(arm, "left", along.cross(across).normalized(), along, across)
        order = [fan[f] for f in ("index", "middle", "ring", "little")]
        self.assertEqual(order, sorted(order, reverse=True))
        self.assertAlmostEqual(sum(order), 0.0, places=6)
        self.assertLessEqual(max(abs(a) for a in order), math.radians(tpose.FAN_LIMIT) * 2)


if __name__ == "__main__":
    unittest.main()
