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


if __name__ == "__main__":
    unittest.main()
