"""Run the avatar's spring bones the way VTubing apps do, and list what goes wrong while they swing.

    blender -b rigged.blend -P scripts/check_springs.py -- --json springs.json [--reference source.glb]

Blender renders and check_avatar.py never run spring physics, so a build could pass them and still
fall apart in Warudo: model C's wings sagged at rest and her feather decorations flew off as she
swayed. This simulates the springs the file ships (the VRM springs and colliders stored on the
armature, with spring_sim, which follows three-vrm) while the body stands still and then turns its
head and sways, poses the mesh from them, and reports:

- chains that droop from their modelled shape at rest (gravity beating stiffness),
- pieces that don't swing but lie on one that does, and are left in the air as it swings (a
  decoration its wing leaves behind); a swinging piece's own tearing is check_avatar.py's,
- how deep swinging surfaces sink into the body (hair_physics.clipping_report).

With --reference (the manifest's source) pieces carry the labels scripts/list_pieces.py gives
them. Changes nothing.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix
from mathutils.kdtree import KDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "addon"))

from vtuber_maker.pipeline import hair_physics, meshdata, roles, spring_sim  # noqa: E402

# A chain droops when a vertex it carries sits this far (m) from its modelled place after standing:
# wings, tails and a hat's tip should hold their shape; hair and cloth settle under gravity (3 and
# 5 cm on model C, which look right).
SAG = {"Appendage": 0.03, "HatTip": 0.03}
SAG_SETTLING = 0.08
# Touching: a vertex of another piece within CONTACT (m); coming away: SEPARATE further apart than at
# rest, for at least MIN_LEFT vertices and LEFT_SHARE of the piece's touching ones (as check_movement.py).
CONTACT = 0.015
SEPARATE = 0.04
MIN_LEFT = 8
LEFT_SHARE = 0.3
# A piece lies on a swinging one when at least this share of its touching vertices touch one.
LIES_ON = 0.5
# Every this many frames is measured, over at most this many vertices of the swinging parts.
FRAME_STEP = 6
MAX_VERTICES = 40000


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--height", type=float, default=1.6)
    args = parser.parse_args(argv)

    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = max((o for o in bpy.data.objects if o.type == "MESH" and o.find_armature() == arm), key=lambda o: len(o.data.vertices))
    for bone in arm.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    design = shipped_design(arm)
    if not design.chains:
        return report(args.json, {"chains": 0, "sagging": [], "coming_away": [], "clipping": None})
    rest = meshdata.world_vertices(mesh)
    labels = None
    if args.reference:
        try:
            with roles.reference(args.reference.resolve(), height=args.height) as source:
                _, labels, _, _ = roles.source_labels(mesh, source)
        except ValueError:
            pass  # a baked T-pose no longer sits on its source: the build's own pieces then
    if labels is None:
        labels = meshdata.pieces(mesh, rest)

    names = [g.name for g in mesh.vertex_groups]
    influences = [[(names[g.group], g.weight) for g in v.groups if g.weight > 0] for v in mesh.data.vertices]
    on_chain = {name: (c, j) for c, chain in enumerate(design.chains) for j, name in enumerate(chain.names)}
    swinging = np.array([any(n in on_chain for n, _ in w) for w in influences])
    partner = _partners(rest, labels, CONTACT * args.height / 1.6)
    # what can come away: swinging vertices, and anything touching them
    touched = np.zeros(len(rest), dtype=bool)
    touched[partner[(partner >= 0) & swinging]] = True
    watch = np.flatnonzero(swinging | touched | ((partner >= 0) & swinging[np.maximum(partner, 0)]))
    if len(watch) > MAX_VERTICES:
        watch = np.sort(np.random.default_rng(0).choice(watch, MAX_VERTICES, replace=False))

    # every (watched vertex, influence) pair, split by whether a chain joint or a body bone moves it
    joints = [(c, j) for c, chain in enumerate(design.chains) for j in range(len(chain.names))]
    joint_index = {pair: k for k, pair in enumerate(joints)}
    anchors = np.array([design.chains[c].points[j] for c, j in joints])
    body_names = sorted({n for v in watch for n, _ in influences[v] if n not in on_chain})
    body_index = {n: k for k, n in enumerate(body_names)}
    cv, cj, cw, bv, bb, bw = [], [], [], [], [], []
    for k, vertex in enumerate(watch):
        for name, weight in influences[vertex]:
            if name in on_chain:
                cv.append(k), cj.append(joint_index[on_chain[name]]), cw.append(weight)
            else:
                bv.append(k), bb.append(body_index[name]), bw.append(weight)
    cv, cj, cw, bv, bb, bw = (np.array(x) for x in (cv, cj, cw, bv, bb, bw))
    watched_rest = rest[watch]

    def posed(frames, motion, index):
        frame, pose = frames[index], motion.poses[index]
        out = np.zeros((len(watch), 3))
        if len(cv):
            positions = np.concatenate([p[:-1] for p in frame.positions])
            rotations = np.concatenate(frame.rotations)
            moved = positions[cj] + np.einsum("nij,nj->ni", rotations[cj], watched_rest[cv] - anchors[cj])
            np.add.at(out, cv, cw[:, None] * moved)
        if len(bv):
            transforms = np.array([pose.get(n, np.eye(4)) for n in body_names])
            moved = np.einsum("nij,nj->ni", transforms[bb][:, :3, :3], watched_rest[bv]) + transforms[bb][:, :3, 3]
            np.add.at(out, bv, bw[:, None] * moved)
        return out

    still = hair_physics.standing_still(arm)
    frames = spring_sim.simulate(design.chains, design.colliders, still, len(still.poses), still.dt)
    settled = posed(frames, still, len(frames) - 1)
    drift = np.linalg.norm(settled - rest[watch], axis=1)
    sagging = []
    for c, chain in enumerate(design.chains):
        carried = np.array([any(n in chain.names for n, _ in influences[v]) for v in watch])
        limit = SAG.get(chain.names[0].rsplit("_", 1)[0].rstrip("0123456789"), SAG_SETTLING)
        if carried.any() and drift[carried].max() > limit * args.height / 1.6:
            sagging.append({"chain": chain.names[0].rsplit("_", 1)[0], "sag_cm": round(float(drift[carried].max()) * 100, 1)})

    sway = hair_physics.head_turn_and_sway(arm)
    frames = spring_sim.simulate(design.chains, design.colliders, sway, len(sway.poses), sway.dt)
    slot = {v: k for k, v in enumerate(watch)}
    pairs = [(slot[v], slot[int(partner[v])]) for v in watch if partner[v] >= 0 and int(partner[v]) in slot]
    a = np.array([p[0] for p in pairs], dtype=int)
    b = np.array([p[1] for p in pairs], dtype=int)
    base = np.linalg.norm(rest[watch][a] - rest[watch][b], axis=1)
    widest = np.zeros(len(a))
    for index in range(FRAME_STEP, len(frames), FRAME_STEP):
        now = posed(frames, sway, index)
        widest = np.maximum(widest, np.linalg.norm(now[a] - now[b], axis=1) - base)
    coming_away = []
    for label in np.unique(labels[watch[a]]):
        # cloth swinging off the legs is what it is for: what matters is a piece that doesn't swing
        # left in the air by one that does (a decoration its wing leaves behind)
        if swinging[labels == label].any():
            continue
        # nor is what lies on the body with hair or a cape hanging over it: that swings off it by
        # design. Model A's shirt under her front locks, and her whole body (13,652 vertices, "left
        # 18.8 cm behind"), stood in every report, and the list was no longer read
        touching = (labels == label) & (partner >= 0)
        if swinging[partner[touching]].mean() < LIES_ON:
            continue
        mine = (labels[watch[a]] == label) & swinging[watch[b]]
        left = widest[mine] > SEPARATE * args.height / 1.6
        if mine.any() and left.sum() >= max(MIN_LEFT, LEFT_SHARE * mine.sum()):
            points = rest[labels == label]
            coming_away.append({"label": int(label), "count": int((labels == label).sum()),
                                "share": round(float(left.mean()), 2), "gap_cm": round(float(widest[mine].max()) * 100, 1),
                                "point": [round(float(v), 4) for v in points[np.argmin(np.linalg.norm(points - points.mean(0), axis=1))]],
                                "swings": bool(swinging[labels == label].any())})
    coming_away.sort(key=lambda r: -r["gap_cm"])
    clipping = hair_physics.clipping_report(mesh, arm, design, sway, frames)  # (the sway as it was run above)
    report(args.json, {"chains": len(design.chains), "sagging": sagging, "coming_away": coming_away,
                       "clipping": {"deepest_cm": round(clipping.deepest * 100, 1),
                                    "share": round(clipping.clipping_fraction, 4), "where": clipping.worst_detail}})


def shipped_design(arm):
    """The springs and colliders stored on the armature (what the .vrm carries), for spring_sim."""
    spring_bone = arm.data.vrm_addon_extension.spring_bone1
    world = arm.matrix_world
    bones = arm.data.bones
    chains = []
    bone_of = {c.uuid: c.node.bone_name for c in spring_bone.colliders}
    groups = {g.uuid: [bone_of[c.collider_uuid] for c in g.colliders if c.collider_uuid in bone_of]
              for g in spring_bone.collider_groups}
    for spring in spring_bone.springs:
        names = [j.node.bone_name for j in spring.joints if j.node.bone_name in bones]
        if not names:
            continue
        joints = list(spring.joints)
        points = [world @ bones[names[0]].head_local] + [world @ bones[n].tail_local for n in names]
        chains.append(spring_sim.Chain(np.array([list(p) for p in points]), bones[names[0]].parent.name,
                                       joints[0].stiffness, joints[0].gravity_power, joints[0].drag_force,
                                       joints[0].hit_radius, names=names, tip_stiffness=joints[-1].stiffness,
                                       collides_with=_collides_with(spring, groups, len(bone_of))))
    colliders = []
    for c in spring_bone.colliders:
        bone = c.node.bone_name
        if bone not in arm.pose.bones:
            continue
        frame = world @ arm.pose.bones[bone].matrix
        if c.shape_type == "Sphere":
            colliders.append(spring_sim.Collider(bone, c.shape.sphere.radius, list(frame @ c.shape.sphere.offset)))
        else:
            colliders.append(spring_sim.Collider(bone, c.shape.capsule.radius, list(frame @ c.shape.capsule.offset),
                                                 list(frame @ c.shape.capsule.tail)))
    return hair_physics.SpringDesign(chains, colliders)


def _collides_with(spring, groups, colliders):
    """The bones whose colliders the spring's groups hold, or None when they hold them all."""
    bones = [bone for g in spring.collider_groups for bone in groups.get(g.collider_group_uuid, [])]
    return None if len(bones) >= colliders else tuple(sorted(set(bones)))


def _partners(rest, labels, reach):
    tree = KDTree(len(rest))
    for i, p in enumerate(rest):
        tree.insert(p, i)
    tree.balance()
    partner = np.full(len(rest), -1)
    for i, p in enumerate(rest):
        best = None
        for _, j, distance in tree.find_range(p, reach):
            if labels[j] != labels[i] and (best is None or distance < best):
                partner[i], best = j, distance
    return partner


def report(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for s in data["sagging"]:
        print(f"SPRINGS {s['chain']} droops {s['sag_cm']} cm standing still")
    for r in data["coming_away"]:
        print(f"SPRINGS piece {r['label']} ({r['count']} vertices) comes away {r['gap_cm']} cm while swaying "
              f"({r['share']:.0%} of where it touches)")
    if data.get("clipping"):
        print(f"SPRINGS deepest into the body: {data['clipping']['deepest_cm']} cm")
    print(f"VTUBER_MAKER_REPORT {json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in data.items() if k != 'clipping'})}")


main()
