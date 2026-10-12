"""A small VRM 1.0 spring bone simulation, used to check hair physics before export."""

import unittest

import numpy as np

from vtuber_maker.pipeline import spring_sim

SIDEWAYS = [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0], [0.3, 0.0, 1.0]]
HANGING = [[0.0, 0.0, 1.0], [0.0, 0.0, 0.9], [0.0, 0.0, 0.8], [0.0, 0.0, 0.7]]


def resting(_frame):
    return {"Head": np.eye(4)}


def strand(points, **settings):
    values = {"stiffness": 1.0, "gravity_power": 0.0, "drag_force": 0.4, "hit_radius": 0.01}
    values.update(settings)
    return spring_sim.Chain(points=np.array(points, dtype=float), parent="Head", **values)


def distance_to_segment(points, start, end):
    along = np.clip((points - start) @ (end - start) / np.dot(end - start, end - start), 0.0, 1.0)
    return np.linalg.norm(points - (start + along[:, None] * (end - start)), axis=1)


class SpringSimulation(unittest.TestCase):
    def test_gravity_pulls_a_loose_sideways_strand_down(self):
        chain = strand(SIDEWAYS, stiffness=0.0, gravity_power=1.0)
        tip = spring_sim.simulate([chain], [], resting, frames=300)[-1].positions[0][-1]
        self.assertLess(tip[2], 0.8)

    def test_stiffness_keeps_the_rest_shape_without_gravity(self):
        final = spring_sim.simulate([strand(SIDEWAYS)], [], resting, frames=120)[-1].positions[0]
        np.testing.assert_allclose(final, SIDEWAYS, atol=1e-3)

    def test_bones_never_stretch(self):
        chain = strand(SIDEWAYS, stiffness=0.2, gravity_power=2.0)
        for frame in spring_sim.simulate([chain], [], resting, frames=60):
            lengths = np.linalg.norm(np.diff(frame.positions[0], axis=0), axis=1)
            np.testing.assert_allclose(lengths, 0.1, atol=1e-6)

    def test_strand_follows_its_parent_bone(self):
        def carried(_frame):
            moved = np.eye(4)
            moved[0, 3] = 0.5
            return {"Head": moved}

        final = spring_sim.simulate([strand(HANGING)], [], carried, frames=240)[-1].positions[0]
        np.testing.assert_allclose(final, np.array(HANGING) + [0.5, 0.0, 0.0], atol=5e-3)

    # Like three-vrm, a joint pushed out of a collider is pulled back to its bone length,
    # which can leave it a few millimetres inside; hence the tolerance.

    def test_sphere_collider_pushes_a_joint_out(self):
        chain = strand(HANGING, stiffness=0.0, gravity_power=1.0, hit_radius=0.01)
        ball = spring_sim.Collider(bone="Head", radius=0.05, offset=np.array([0.003, 0.0, 0.8]))
        free = spring_sim.simulate([chain], [], resting, frames=120)[-1].positions[0][2]
        held = spring_sim.simulate([chain], [ball], resting, frames=120)[-1].positions[0][2]
        self.assertLess(np.linalg.norm(free - ball.offset), 0.01)
        self.assertGreater(np.linalg.norm(held - ball.offset), 0.05 + 0.01 - 0.01)

    def test_capsule_collider_pushes_a_joint_out(self):
        chain = strand(HANGING, stiffness=0.0, gravity_power=1.0, hit_radius=0.01)
        start, end = np.array([-0.1, 0.005, 0.8]), np.array([0.1, 0.005, 0.8])
        capsule = spring_sim.Collider(bone="Head", radius=0.04, offset=start, tail=end)
        free = spring_sim.simulate([chain], [], resting, frames=120)[-1].positions[0][2:3]
        held = spring_sim.simulate([chain], [capsule], resting, frames=120)[-1].positions[0][2:3]
        self.assertLess(distance_to_segment(free, start, end)[0], 0.01)
        self.assertGreater(distance_to_segment(held, start, end)[0], 0.04 + 0.01 - 0.01)

    def test_a_chain_is_pushed_only_by_the_bones_it_collides_with(self):
        # hair lying on the shoulders, set to collide with the head alone (hair_physics.design)
        ball = spring_sim.Collider(bone="Chest", radius=0.05, offset=np.array([0.003, 0.0, 0.8]))
        pose = lambda _frame: {"Head": np.eye(4), "Chest": np.eye(4)}  # noqa: E731
        settings = {"stiffness": 0.0, "gravity_power": 1.0, "hit_radius": 0.01}
        free = spring_sim.simulate([strand(HANGING, **settings)], [], pose, frames=120)[-1].positions[0]
        head_only = spring_sim.simulate([strand(HANGING, collides_with=("Head",), **settings)], [ball], pose, frames=120)[-1].positions[0]
        chest_too = spring_sim.simulate([strand(HANGING, collides_with=("Head", "Chest"), **settings)], [ball], pose, frames=120)[-1].positions[0]
        np.testing.assert_allclose(head_only, free)
        self.assertGreater(np.linalg.norm(chest_too[2] - ball.offset), 0.05)


def every_collider_asked(chains, colliders, body_motion, frames, dt=1 / 60):
    """The simulation as it was first written: every collider put to every joint, each placing itself
    anew (`push_out`), the turn between two directions through np.cross. What `simulate` must still
    give, to the last digit."""
    def rotation_between(a, b):
        cross = np.cross(a, b)
        sine = np.linalg.norm(cross)
        cosine = float(np.dot(a, b))
        if sine < 1e-9:
            if cosine > 0.0:
                return np.eye(3)
            perpendicular = spring_sim._unit(np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]))
            return 2.0 * np.outer(perpendicular, perpendicular) - np.eye(3)
        skew = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]])
        return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)

    states = [(chain, np.linalg.norm(np.diff(chain.points, axis=0), axis=1), chain.points[1:].copy(), chain.points[1:].copy())
              for chain in chains]
    result, pushes = [], 0
    for frame in range(frames):
        pose = body_motion(frame)
        positions_now, rotations_now = [], []
        for chain, lengths, tails, previous in states:
            directions = np.diff(chain.points, axis=0) / lengths[:, None]
            rotation = pose[chain.parent][:3, :3]
            head = rotation @ chain.points[0] + pose[chain.parent][:3, 3]
            positions, rotations = [head], []
            for j, length in enumerate(lengths):
                axis = rotation @ directions[j]
                inertia = (tails[j] - previous[j]) * (1.0 - chain.drag_force)
                tail = tails[j] + inertia + axis * chain.stiffness * dt + chain.gravity_dir * chain.gravity_power * dt
                tail = head + spring_sim._unit(tail - head) * length
                for collider in colliders:
                    if chain.collides_with is not None and collider.bone not in chain.collides_with:
                        continue
                    pushed = collider.push_out(tail, chain.hit_radius, pose)
                    if pushed is not None:
                        pushes += 1
                        tail = head + spring_sim._unit(pushed - head) * length
                previous[j] = tails[j]
                tails[j] = tail
                rotation = rotation_between(axis, spring_sim._unit(tail - head)) @ rotation
                rotations.append(rotation)
                positions.append(tail)
                head = tail
            positions_now.append(np.array(positions))
            rotations_now.append(np.array(rotations))
        result.append(spring_sim.Frame(positions_now, rotations_now))
    return result, pushes


class TheSameNumbersSooner(unittest.TestCase):
    """`simulate` only asks the colliders a joint comes near (756,000 askings in one avatar's spring check,
    ten of its seventeen seconds, nearly all answered "not touching"). What it gives must not move by a digit:
    the spring check's limits were set on these numbers."""

    def swaying(self, frame):
        t = frame / 60
        lean, turn = 0.3 * np.sin(2.4 * t), 0.6 * np.sin(3.1 * t)
        body, head = np.eye(4), np.eye(4)
        body[:3, :3] = [[np.cos(lean), 0, np.sin(lean)], [0, 1, 0], [-np.sin(lean), 0, np.cos(lean)]]
        body[:3, 3] = (0.05 * np.sin(4.0 * t), 0.0, 0.0)
        head[:3, :3] = [[np.cos(turn), -np.sin(turn), 0], [np.sin(turn), np.cos(turn), 0], [0, 0, 1]]
        return {"Head": body @ head, "Chest": body, "Hips": body}

    def avatar(self):
        rng = np.random.default_rng(7)
        chains = []
        for k in range(8):
            # strands round the head, hanging down past the chest: some lie on a collider from the start
            x, y = 0.09 * np.cos(k * 0.8), 0.09 * np.sin(k * 0.8)
            points = [[x + rng.normal(0, 0.004), y + rng.normal(0, 0.004), 1.5 - 0.07 * j] for j in range(6)]
            chains.append(strand(points, stiffness=0.4 + 0.1 * k, gravity_power=0.6, hit_radius=0.012 + 0.002 * (k % 3),
                                 collides_with=None if k % 3 else ("Head", "Chest")))
        colliders = [spring_sim.Collider("Head", 0.085, [0.0, 0.0, 1.47]),
                     spring_sim.Collider("Chest", 0.07, [-0.09, 0.0, 1.3], [0.09, 0.0, 1.3]),
                     spring_sim.Collider("Chest", 0.08, [0.0, 0.01, 1.25], [0.0, 0.01, 1.1]),
                     spring_sim.Collider("Hips", 0.09, [0.0, 0.0, 1.0])]
        return chains, colliders

    def test_every_frame_is_the_one_asking_every_collider_gave(self):
        chains, colliders = self.avatar()
        slow, pushes = every_collider_asked(chains, colliders, self.swaying, frames=150)
        fast = spring_sim.simulate(chains, colliders, self.swaying, frames=150)
        self.assertGreater(pushes, 500, "the strands have to meet the colliders for this to say anything")
        for was, now in zip(slow, fast):
            for a, b in zip(was.positions + was.rotations, now.positions + now.rotations):
                np.testing.assert_array_equal(a, b)

    def test_a_collider_on_a_bone_the_pose_lacks_still_says_so(self):
        chains, colliders = self.avatar()
        colliders.append(spring_sim.Collider("LeftHand", 0.05, [0.4, 0.0, 1.0]))
        with self.assertRaises(KeyError):
            spring_sim.simulate(chains, colliders, self.swaying, frames=2)
        # ... but only when a chain is put to it
        only_head = [strand(HANGING, collides_with=("Head",))]
        self.assertEqual(len(spring_sim.simulate(only_head, colliders, self.swaying, frames=2)), 2)

    def test_the_colliders_a_joint_comes_near_are_the_ones_it_touches_and_a_hair_more(self):
        _chains, colliders = self.avatar()
        pose = self.swaying(20)
        reach = spring_sim._Reach(colliders, pose)
        touched = passed_over = 0
        rng = np.random.default_rng(3)
        # points round each collider where the pose has carried it
        for point in (reach.starts[:, None, :] + rng.uniform(-0.15, 0.15, (len(colliders), 150, 3))).reshape(-1, 3):
            touching = np.array([c.push_out(point, 0.01, pose) is not None for c in colliders])
            near = reach.near(point, 0.01)
            self.assertTrue(near[touching].all())
            touched, passed_over = touched + int(touching.sum()), passed_over + int((~near).sum())
        self.assertGreater(touched, 50)
        self.assertGreater(passed_over, 1500)  # ... and most askings are saved

    def test_a_capsule_with_no_length_is_left_to_answer_for_itself(self):
        # what it answers is not a number (nothing divided by nothing): the same not-a-number as before
        chains, colliders = self.avatar()
        colliders.insert(1, spring_sim.Collider("Chest", 0.05, [0.05, 0.0, 1.32], [0.05, 0.0, 1.32]))
        self.assertEqual(spring_sim._Reach(colliders, self.swaying(0)).ask.tolist(), [False, True, False, False, False])
        with np.errstate(invalid="ignore", divide="ignore"):
            slow, _ = every_collider_asked(chains, colliders, self.swaying, frames=20)
            fast = spring_sim.simulate(chains, colliders, self.swaying, frames=20)
        for was, now in zip(slow, fast):
            for a, b in zip(was.positions + was.rotations, now.positions + now.rotations):
                np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
