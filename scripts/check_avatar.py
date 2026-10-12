"""Find rigging bugs on a rigged avatar .blend by moving bones and measuring the mesh.

    blender -b in.blend -P scripts/check_avatar.py [-- --swing 0.7] [--json report.json]

Reports:
- vertices with no weights (they stay put while the body moves)
- edges that stretch when the head turns (parts not following the head)
- edges that stretch when the arms go to a T-pose, hang, reach forward or rise, and when the
  spine and chest twist and bend or a leg lifts (sleeves left behind, torn clothing)
- edges that shrink to under half their length in those poses (a chest caving in because the
  arm bones reach onto it, cloth folding into itself); judged where the weights step from one
  bone to the next, where a run of them is a fold: shared between the bones that move and the
  ones that stay, they are a joint's own crease closing
- edges that stretch when each hair, cloth or hat tip chain swings alone (strips poking out,
  torn strands), with the weights on both ends of the worst edge
- copies of a vertex along a texture seam that come apart in a pose or when every chain swings
  (a tear in the hair or a cape), hair hanging from the head bone with no springs, hair above its
  springs that the neck holds part of (a ledge when the head tilts)
- on the face: what lies behind and beside each iris (an eye white deep in the socket) and
  whether the drawn mouth turns inside out in mixes of the tracker's shapes
Blender renders never run spring physics, so this is how to see hair problems that only
show in VTubing apps. Nothing is saved but the optional --json report, which holds the same
numbers for a build to judge itself by (see scripts/make_avatar.py).
"""
import argparse
import json
import sys

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

HEAD_TURNS = [(0.5, 0, 0), (-0.5, 0, 0), (0, 0, 0.6), (0, 0, -0.6), (0, 0.4, 0)]
# Where both arms are pointed (shoulder to wrist, for the left arm; the right is mirrored):
# the T-pose every VRM app starts from, hanging at the sides, reaching forward, raised high.
ARM_POSES = {"T-pose": (1, 0, 0), "at sides": (0.25, 0, -1), "forward": (0.3, -1, -0.3), "raised": (0.5, 0, 1)}
# Body bones turned both ways about a world axis (radians).
BODY_POSES = {
    "spine twist": ("Spine", (0, 0, 1), 0.5),
    "spine bend": ("Spine", (1, 0, 0), 0.4),
    "spine side bend": ("Spine", (0, 1, 0), 0.35),
    "chest twist": ("Chest", (0, 0, 1), 0.4),
    "leg lift": ("LeftUpperLeg", (1, 0, 0), 0.8),
}
STRETCH = 0.03  # m; edges growing more than this are reported
# Edges at least SQUASH_MIN long at rest that shrink below SQUASH of their length are reported
# for arm and body poses: cloth folding into itself, like a chest caving in as the arms move.
SQUASH = 0.5
# Only edges this long at rest (m) count. Shorter ones are the blend across a shoulder or a
# cut, where a fraction of a millimetre of fold reads as a 5% squash and means nothing.
SQUASH_MIN = 0.02
# A joint that bends closes the crease beside it: the armpit as an arm comes down, the fold of the
# hip as a leg lifts. Edges across it shrink to almost nothing on any rig, and nothing is wrong.
# With the arms at its sides the VRoid-made reference, which tracks well in Warudo, has 36 edges
# under half their length (the worst 15%), model B 27 (8%) and the approved model A 35 (10%): on
# all three the renders show an armpit closing and nothing else (2026-10-11). For three weeks that
# line alone kept model B "not ready". What such an edge has on every one of them: both its ends
# are shared between the bones the pose moves and the ones that stay (the reference: 0.17 to 0.83
# of each end on the moving bones; ours 0.11 to 0.87). So an edge with both ends shared between
# SHARED[0] and SHARED[1] is in the joint's blend, its crease closing: measured, not judged.
SHARED = (0.05, 0.95)
# An edge with an end wholly on one side (0.97 to 1.0, or 0.0 to 0.03, on what was measured) is
# where the weights step, and there a run of edges under the limit is a fold. Model D's cape, cut
# into bands between Spine and Hips 25 cm from her spine, kinks along 14 edges in a side bend (seen
# in the render); the rim of model B's hood, rigid on the neck against a chest that turns less,
# loses 3 edges in a twist and nothing shows.
FOLD_EDGES = 5
SHOW = 3
# Close-ups of the worst spots (--spots) cut away what lies more than this (m) in front of them.
SPOT_CLIP = 0.08
# The ring drawn round a spot in its close-ups (m, its outer radius): a fifth of the picture's width.
SPOT_RING = 0.03
# A build is worth looking at by hand when it is over one of these.
LIMITS = {"unweighted": 0, "stretch_cm": 6.0, "squash": 0.35, "fold_edges": FOLD_EDGES, "arm_follows": 0.5,
          "mouth_on_head": 0.5, "torn": 4, "mouth_turned": 0.25, "beside_iris_mm": 20.0, "hangs_on_head": 50,
          "ledge_edges": 100, "finger_slits": 4}
# A shape key moves a vertex when it takes it further than this (m), as scripts/_fingerprint.py counts
KEY_MOVED = 1e-5
# The key the mouth is placed by, the surface within MOUTH_NEAR (m) of it seen from the front, and
# how deep behind the frontmost of that surface still counts as the face
MOUTH_KEY = "aa"
MOUTH_NEAR = 0.02
MOUTH_DEPTH = 0.03
# An arm that moves has to take the surface round it along: of the vertices within ARM_NEAR (m) of
# the middle of an upper arm's bone, the share that moves ARM_MOVED (m) or more when the arms are
# raised. A tester's pirate, his whole body put on the head bone: every arm pose measured "0 edges
# stretch", nothing over a limit, and only his hands and cuffs went up.
ARM_NEAR = 0.07
ARM_MOVED = 0.03
# ... read on the arm's own surface: of the pieces of the surface with at least ARM_OWN vertices there
# that go round the bone (in at least ARM_ROUND of ARM_SECTORS directions about it), the one lying
# closest to it. Model D's bare arms go up whole under a cape that hangs from her torso and stays
# down, as a cape should: of everything within ARM_NEAR 49% moved, and the line said her sleeve was on
# another bone. Her cape lies on her arms, as close to the bone as their skin, and covers their top
# alone. A body in one piece is read whole, as before.
ARM_OWN = 20
ARM_SECTORS = 8
ARM_ROUND = 7
# Vertices this close (m) are at one place: the copies an import makes along a texture seam, and
# the ones a file read back has for each normal (scripts/_fingerprint.py counts places the same way)
PLACE = 1e-5
# Copies of a vertex this far apart (m) in a pose have opened the surface there. Model A built from
# her manifest on 2026-10-07 (tears in her hair in Warudo): 65 places with every chain swung
# together, 3.9 cm the worst; model B the same morning (tears in her cape): 24, 6.9 cm; both rebuilt
# after loose.share_strand_seams and the cut kept to the arms: 0. Model D's builds of that week: 1 or
# 2 places under 2 cm, and nothing was seen (LIMITS "torn").
TORN = 0.01
ARM_BONE = ("Shoulder", "Arm", "Hand", "Thumb", "Index", "Middle", "Ring", "Little")
# Within one hand no cut is meant. Model D's build of 2026-10-05 (the user in Warudo: "hands slit like
# the hair and cape"; split_torn_seams cut between two fingers, 60 edges) with its fingers curled and
# spread in turn (FINGER_CURL about the knuckles' line, FINGER_SPREAD about the hand's normal, radians
# for index, middle, ring and little, both ways): 77 places in her two hands where the copies of a
# vertex end over FINGER_TORN (m) apart, the worst 9 mm; model A's T-pose build of 10-04, cut by the
# same rule: 36. Every build since the fix shares the weights there and reads 0, to the last digit
# (ten read here, 36 exported files by a probe; LIMITS "finger_slits"). Under TORN: a slit a finger
# wide shows.
HAND_BONE = ("Hand", "Thumb", "Index", "Middle", "Ring", "Little")
FINGER_CURL = (0.9, -0.2, 0.9, -0.2)
FINGER_SPREAD = (0.25, 0.08, -0.1, -0.3)
FINGER_TORN = 0.002
# The tracker's shapes that narrow the drawn mouth, and the vowels: shape keys add up, and jaw,
# pucker and funnel together took the patch past nothing and out the other side, facing backwards
# (the user, 2026-10-08: "the mouth inverts"; 18 of 111 mixes on model D, the pucker alone on model A).
# Those builds: 74 to 100% of the patch turned with pucker and funnel at full. A mix counts from a
# quarter of the patch (LIMITS "mouth_turned"): the pirate's, all three shapes at full leaving a
# sliver, turns 14% of it and nothing shows.
MOUTH_PATCH = "MouthInside"
MOUTH_NARROWING = ("jawOpen", "mouthPucker", "mouthFunnel")
MOUTH_STEPS = (0.0, 0.5, 1.0)
FLAT = 0.98
VOWELS = ("aa", "ih", "ou", "ee", "oh")
# The keys an eye looks about with; the vertices one of them takes at least IRIS_SHARE of its
# furthest move are the iris. IRIS_ROUND (m) outside its outline is beside it, where the white shows.
# What lies there, a quarter of the way round (mm behind the iris; LIMITS "beside_iris_mm"): the fox
# girl's socket 29, and 4 with the white on a surface of its own; model A 2 to 5. Between them, and
# not told apart by depth: the catgirl's socket before that fix (16), the pirate's eye (15), model
# B's sliding patches standing off her round face (11 to 13, nothing to see from the side).
LOOK_KEYS = ("eyeLookIn", "eyeLookOut", "eyeLookUp", "eyeLookDown")
IRIS_SHARE = 0.5
IRIS_ROUND = 0.002
IRIS_RAYS = 24
# A ray that meets nothing this far (m) into the head has left through the back of it
EYE_DEEP = 0.1
# What stands this far (m) over the head joint is the head's own, or its hair's or hat's, and keeps
# its shape whatever the neck does. A tracking app tilts and nods the head with the neck: turned
# together here (radians), an edge up there that grows more than LEDGE (m) is hair the neck or the
# chest holds part of. Model D, 12 and 18 degrees: 165 edges over 5 mm, the worst 15 mm, in a line
# across her hair where its springs start (the probe of 2026-10-06, the tilt alone; this check, with
# the nod: 429 and 17.5 mm). Turning the head bone alone shows none of it.
HEAD_CLEAR = 0.03
NECK_TILT, HEAD_TILT = 0.21, 0.31
LEDGE = 0.005
OFF_HEAD = 0.1
# Only edges the head and what hangs from it hold at least this much of, at both ends: a tail held
# over the head that the neck and shoulders hold has troubles of its own, and isn't hair. Model D
# with the ledge: 429 edges; mended, 30; a short-haired head the adopted rig shares with the neck
# all over (the plain boy): 34; the catgirl: 26 (LIMITS "ledge_edges").
MOSTLY_HEAD = 0.5
# A vertex the head bone holds by this much, further below the shoulder joints than HANGS (m), is
# hair that turns with the head instead of swinging: on model A's builds of 2026-10-07 the bit of
# back hair the tidy-up had put on the head, 307 places down to 19 cm under her shoulders; none on
# any build since (LIMITS "hangs_on_head")
HEAD_ALONE = 0.9
HANGS = 0.03
# Every chain swung at once, as springs swing them: the pose's name in the report
TOGETHER = "swing together"


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--swing", type=float, default=0.7, help="hair chain swing in radians")
    parser.add_argument("--json", help="write the same measurements here, for a build to judge itself by")
    parser.add_argument("--spots", help="render close-ups of the worst stretch and squash into this folder")
    args = parser.parse_args(argv)
    measured = {"poses": {}, "worst": {}}

    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh_obj = next(o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm)
    mesh = mesh_obj.data
    names = {g.index: g.name for g in mesh_obj.vertex_groups}
    matrix = np.array(mesh_obj.matrix_world)
    rest = np.array([v.co for v in mesh.vertices]) @ matrix[:3, :3].T + matrix[:3, 3]
    edges = np.array([e.vertices[:] for e in mesh.edges])
    rest_length = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    middle = (rest[edges[:, 0]] + rest[edges[:, 1]]) / 2

    empty = [v.index for v in mesh.vertices if not any(g.weight > 0 for g in v.groups)]
    print(f"CHECK unweighted vertices: {len(empty)}" + (f" (e.g. at {rest[empty[0]].round(3)})" if empty else ""))
    measured["unweighted"] = len(empty)
    # a .vrm holds four bones a vertex (rig.limit_influences): with more, this .blend is not quite the rig that ships
    bones = {g.index for g in mesh_obj.vertex_groups if g.name in arm.data.bones}
    held = [(v.index, g.group, g.weight) for v in mesh.vertices for g in v.groups if g.group in bones and g.weight > 0]
    held_vertex, held_bone, held_weight = (np.array(column) for column in zip(*held)) if held else (np.zeros(0, int),) * 3
    crowded = int((np.bincount(held_vertex, minlength=len(rest)) > 4).sum())
    print(f"CHECK bones: {crowded} vertices on more than four bones" + (": the file keeps their strongest four" if crowded else ""))
    measured["over_four_bones"] = crowded
    # each vertex's strongest bone, and the seams the build keeps closed (seams_kept)
    lead, order = np.full(len(rest), -1), np.argsort(held_weight, kind="stable")
    lead[held_vertex[order]] = held_bone[order]
    springs = spring_bones(arm)
    swings = np.array([names.get(g) in springs for g in range(len(mesh_obj.vertex_groups))] + [False])
    on_arm = np.array([any(part in names.get(g, "") for part in ARM_BONE) for g in range(len(mesh_obj.vertex_groups))] + [False])
    copy, twin = seam_copies(rest)
    kept = seams_kept(lead[copy], lead[twin], swings, on_arm)
    seams = measured["seams"] = {"copies": int(len(copy)), "kept": int(kept.sum()), "torn": 0, "torn_cm": 0.0}

    def weights(i):
        top = sorted(mesh.vertices[i].groups, key=lambda g: -g.weight)[:3]
        return {names[g.group]: round(g.weight, 2) for g in top}

    def reset():
        for bone in arm.pose.bones:
            bone.rotation_mode = "XYZ"
            bone.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()

    squashed = np.ones(len(edges))
    resting = {bone.name: np.array(bone.matrix_local) for bone in arm.data.bones}

    def moving_share():
        """How much of each vertex the bones hold that the pose now on the rig moves: the turned
        ones, what hangs from them, and helpers a constraint turns along with them."""
        moved = {name for name, at_rest in resting.items() if np.abs(np.array(arm.pose.bones[name].matrix) - at_rest).max() > 1e-5}
        on = np.array([names.get(g) in moved for g in range(len(mesh_obj.vertex_groups))])
        total = np.bincount(held_vertex, weights=held_weight, minlength=len(rest))
        return np.bincount(held_vertex, weights=held_weight * on[held_bone], minlength=len(rest)) / np.maximum(total, 1e-12)

    posed_by = {}  # label -> the poses measured, to show the worst of them again

    def stretch(rotations):
        stretch.last = rotations
        worst = np.zeros(len(edges))
        squashed[:] = 1.0
        stretch.apart = np.zeros(len(copy))
        for pose in rotations:
            reset()
            if callable(pose):
                pose()
            else:
                for name, rotation in pose.items():
                    arm.pose.bones[name].rotation_euler = Euler(rotation)
            bpy.context.view_layer.update()
            evaluated = mesh_obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            posed_mesh = evaluated.to_mesh()
            posed = np.array([v.co for v in posed_mesh.vertices]) @ matrix[:3, :3].T + matrix[:3, 3]
            evaluated.to_mesh_clear()
            stretch.posed = posed
            stretch.share = moving_share()
            length = np.linalg.norm(posed[edges[:, 0]] - posed[edges[:, 1]], axis=1)
            np.maximum(stretch.apart, np.linalg.norm(posed[copy] - posed[twin], axis=1), out=stretch.apart)
            worst = np.maximum(worst, length - rest_length)
            np.minimum(squashed, np.where(rest_length > SQUASH_MIN, length / np.maximum(rest_length, 1e-9), 1.0), out=squashed)
        reset()
        return worst

    def turn(name, axis, angle):
        """Rotate a bone about a world axis through its head (bone-local axes twist a limb instead)."""
        bone = arm.pose.bones[name]
        head = bone.head.copy()
        bone.matrix = Matrix.Translation(head) @ Matrix.Rotation(angle, 4, axis) @ Matrix.Translation(-head) @ bone.matrix
        bpy.context.view_layer.update()

    def point_arms(direction):
        """Swing both upper arms so shoulder-to-wrist points along direction (mirrored for the right)."""
        def pose():
            for side, mirror in (("Left", 1.0), ("Right", -1.0)):
                if f"{side}UpperArm" not in arm.pose.bones:
                    continue
                upper, hand = arm.pose.bones[f"{side}UpperArm"], arm.pose.bones[f"{side}Hand"]
                now = (hand.head - upper.head).normalized()
                want = Vector((mirror * direction[0], direction[1], direction[2])).normalized()
                axis = now.cross(want)
                if axis.length > 1e-6:
                    turn(f"{side}UpperArm", axis.normalized(), now.angle(want))
        return pose

    def finger_poses():
        """The fingers of both hands curled, then spread, each against its neighbours and both ways
        (FINGER_CURL, FINGER_SPREAD); none for a rig without them."""
        hands = {}
        for side in ("Left", "Right"):
            joints = [f"{side}Hand"] + [f"{side}{finger}Proximal" for finger in ("Index", "Middle", "Ring", "Little")]
            if all(name in arm.data.bones for name in joints):
                at = [np.array(arm.data.bones[name].head_local) for name in joints]
                hands[side] = (joints[1:], hand_axes(at[0], at[1], at[2], at[4]))

        def pose(angles, which, way):
            for joints, axes in hands.values():
                for name, angle in zip(joints, angles):
                    turn(name, Vector(axes[which]), way * angle)
        return [lambda angles=angles, which=which, way=way: pose(angles, which, way)
                for angles, which in ((FINGER_CURL, 0), (FINGER_SPREAD, 1)) for way in (1.0, -1.0)] if hands else []

    def report(label, worst):
        # edges with no length at rest are the stitches across sleeve cuts, made to stretch
        bad = np.nonzero((worst > STRETCH) & (rest_length > 1e-6))[0]
        worst = np.where(rest_length > 1e-6, worst, 0.0)
        print(f"CHECK {label}: {len(bad)} edges stretch over {STRETCH * 100:.0f} cm, worst {worst.max() * 100:.1f} cm")
        for e in bad[np.argsort(-worst[bad])][:SHOW]:
            a, b = edges[e]
            print(f"    +{worst[e] * 100:.1f} cm  {rest[a].round(3)} {weights(a)}  <->  {rest[b].round(3)} {weights(b)}")
        pose = measured["poses"].setdefault(label, {})
        posed_by[label] = stretch.last
        pose["stretched_edges"] = int(len(bad))
        pose["stretch_cm"] = round(float(worst.max()) * 100, 1)
        pose["stretch_edge"] = int(np.argmax(worst))
        # A chain swinging on its own always stretches the edges to the chain beside it, which
        # stays still; real springs move neighbours together. So swings are measured, not judged.
        if not label.startswith("swing") and pose["stretch_cm"] > measured["worst"].get("stretch_cm", 0.0):
            measured["worst"] = dict(measured["worst"], stretch_cm=pose["stretch_cm"], stretch_pose=label)
        # A seam the build keeps closed must stay closed in every pose and with all the chains swinging
        # as springs swing them, together. One chain alone parts it from what its neighbour holds.
        if label == TOGETHER or not label.startswith("swing"):
            torn = kept & (stretch.apart > TORN)
            pose["torn"] = int(len(np.unique(twin[torn])))
            if pose["torn"]:
                pose["torn_cm"] = round(float(stretch.apart[torn].max()) * 100, 1)
                worst_copy = int(np.flatnonzero(torn)[np.argmax(stretch.apart[torn])])
                a, b = int(copy[worst_copy]), int(twin[worst_copy])
                print(f"    {pose['torn']} places joined at rest come apart, the worst {pose['torn_cm']} cm:  "
                      f"{rest[a].round(3)} {weights(a)}  |  {weights(b)}")
                if pose["torn"] > seams["torn"]:
                    seams.update(torn=pose["torn"], torn_cm=pose["torn_cm"], pose=label, at=[round(float(v), 3) for v in rest[a]],
                                 weights=[weights(a), weights(b)])

    def report_squash(label):
        bad = np.nonzero(squashed < SQUASH)[0]
        crease, folded, folds = folds_outside_the_blend(stretch.share[edges], middle, squashed)
        print(f"CHECK {label} squash: {len(bad)} edges shrink below {SQUASH:.0%}, worst to {squashed.min():.0%}; "
              f"{folds} under {LIMITS['squash']:.0%} where the weights step")
        for e in bad[np.argsort(squashed[bad])][:SHOW]:
            a, b = edges[e]
            print(f"    {squashed[e]:.0%}{' (a crease: both ends shared with what moves)' if crease[e] else ''}  "
                  f"{rest[a].round(3)} {weights(a)}  <->  {rest[b].round(3)} {weights(b)}")
        pose = measured["poses"].setdefault(label, {})
        pose["squashed_edges"] = int(len(bad))
        pose["squash"] = round(float(squashed.min()), 3)
        pose["squash_edge"] = int(np.argmin(squashed))
        pose["folded_edges"] = folds
        if folds:
            pose["fold"] = round(float(squashed[folded].min()), 3)
            pose["fold_edge"] = int(folded[np.argmin(squashed[folded])])
        if pose["squash"] < measured["worst"].get("squash", 1.0):
            measured["worst"] = dict(measured["worst"], squash=pose["squash"], squash_pose=label)
        if folds > measured["worst"].get("folded_edges", 0):
            measured["worst"] = dict(measured["worst"], folded_edges=folds, fold=pose["fold"], fold_pose=label)

    piece = None
    if "Head" in arm.pose.bones:
        report("head turn", stretch([{"Head": r} for r in HEAD_TURNS]))
    if "LeftUpperArm" in arm.pose.bones:
        for label, direction in ARM_POSES.items():
            ends = {side: (np.array(arm.matrix_world @ arm.pose.bones[f"{side}UpperArm"].head),
                           np.array(arm.matrix_world @ arm.pose.bones[f"{side}LowerArm"].head))
                    for side in ("Left", "Right") if f"{side}LowerArm" in arm.pose.bones}
            report(f"arms {label}", stretch([point_arms(direction)]))
            report_squash(f"arms {label}")
            if label == "raised":
                moved = np.linalg.norm(stretch.posed - rest, axis=1)
                for side, (start, end) in ends.items():
                    along = (rest - start) @ (end - start) / max(float((end - start) @ (end - start)), 1e-9)
                    off = np.linalg.norm(rest - (start + np.clip(along, 0, 1)[:, None] * (end - start)), axis=1)
                    near = (along > 0.4) & (along < 0.9) & (off < ARM_NEAR)
                    if near.sum() >= ARM_OWN:
                        if piece is None:
                            piece = surface_pieces(rest, edges)
                        axis = (end - start) / max(float(np.linalg.norm(end - start)), 1e-9)
                        across = np.cross(axis, (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (1.0, 0.0, 0.0))
                        across /= np.linalg.norm(across)
                        out = rest[near] - (start + np.clip(along[near], 0, 1)[:, None] * (end - start))
                        about = np.arctan2(out @ np.cross(axis, across), out @ across)
                        share, everything = arm_follows(off[near], moved[near] > ARM_MOVED, piece[near], about)
                        measured.setdefault("arm_follows", {})[side] = round(share, 2)
                        measured.setdefault("arm_surface", {})[side] = round(everything, 2)
                        print(f"CHECK {side.lower()} upper arm: {share:.0%} of its own surface moves with it"
                              + (f" ({everything:.0%} of everything round it: what else lies there stays)" if everything < share - 0.005 else ""))
    for label, (bone, axis, angle) in BODY_POSES.items():
        if bone in arm.pose.bones:
            report(label, stretch([lambda: turn(bone, Vector(axis), angle), lambda: turn(bone, Vector(axis), -angle)]))
            report_squash(label)
    chains = sorted({b.name.rsplit("_", 1)[0] for b in arm.data.bones
                     if b.name.endswith("_0") and b.parent and b.parent.name in ("Head", "Hips")})
    s = args.swing
    for chain in chains:
        bones = [b.name for b in arm.data.bones if b.name.rsplit("_", 1)[0] == chain]
        poses = [{name: r for name in bones[:1]} for r in [(s, 0, 0), (-s, 0, 0), (0, 0, s), (0, 0, -s)]]
        report(f"swing {chain}", stretch(poses))
    if len(chains) > 1:
        roots = [f"{chain}_0" for chain in chains]
        report(TOGETHER, stretch([{name: r for name in roots} for r in [(s, 0, 0), (-s, 0, 0), (0, 0, s), (0, 0, -s)]]))
    # not one of the judged poses either: no app holds the fingers so, and the webs between them stretch
    hand = hand_of([names.get(g, "") for g in range(len(mesh_obj.vertex_groups))] + [""])
    in_hand = (hand[lead[copy]] > 0) & (hand[lead[copy]] == hand[lead[twin]])
    fingers = finger_poses()
    if fingers and in_hand.any():
        stretch(fingers)
        slit = in_hand & (stretch.apart > FINGER_TORN)
        seams["fingers"], seams["fingers_mm"] = int(len(np.unique(twin[slit]))), round(float(stretch.apart[in_hand].max()) * 1000, 1)
    print(f"CHECK seams: {seams['copies']} copies of a vertex at another's place, {seams['kept']} of them the build keeps "
          f"closed; {seams['torn']} places come apart over {TORN * 100:.0f} cm"
          + (f" ({seams['pose']}, the worst {seams['torn_cm']} cm)" if seams["torn"] else "")
          + (f"; {seams['fingers']} places in the hands part over {FINGER_TORN * 1000:.0f} mm as the fingers curl and spread"
             f" (the worst {seams['fingers_mm']} mm)" if "fingers" in seams else ""))

    hair, on_head = hair_on_the_head(arm, mesh_obj, rest)
    measured["hair"] = hair
    if "Head" in arm.pose.bones and "Neck" in arm.pose.bones:
        # not one of the judged poses: the neck's own skin stretches as it should
        crown = float((arm.matrix_world @ arm.pose.bones["Head"].head).z) + HEAD_CLEAR
        both = [lambda axis=axis, way=way: (turn("Neck", Vector(axis), way * NECK_TILT), turn("Head", Vector(axis), way * HEAD_TILT))
                for axis in ((0, 1, 0), (1, 0, 0)) for way in (1, -1)]
        its_own = (rest[edges][:, :, 2] > crown).all(axis=1) & (on_head[edges] >= MOSTLY_HEAD).all(axis=1)
        grown = np.where(its_own & (rest_length > 1e-6), stretch(both), 0.0)
        hair["ledge_edges"], hair["ledge_mm"] = places(middle[grown > LEDGE]), round(float(grown.max()) * 1000, 1)
        if hair["ledge_edges"]:
            a, b = edges[int(np.argmax(grown))]
            hair["ledge_at"] = [round(float(v), 3) for v in middle[int(np.argmax(grown))]]
            hair["ledge_weights"] = [weights(a), weights(b)]
    print(f"CHECK hair: {hair['hangs_on_head']} vertices hang from the head bone below the shoulders"
          + (f" (down to {hair['hangs_cm']} cm under them)" if hair["hangs_on_head"] else "")
          + f"; {hair.get('ledge_edges', 0)} edges above the head joint grow over {LEDGE * 1000:.0f} mm when head and neck tilt together"
          + (f" (the worst {hair['ledge_mm']} mm, {hair['ledge_weights'][0]} | {hair['ledge_weights'][1]})" if hair.get("ledge_edges") else "")
          + f"; {hair['shared_above']} vertices up there are shared between the head and the body")

    face = face_shapes(arm, mesh_obj)
    if face:
        measured["face"] = face
        if "mouth_on_head" in face:
            print(f"CHECK mouth: the surface under it follows the head by {face['mouth_on_head']:.0%}")
        for side, eye in face.get("eyes", {}).items():
            print(f"CHECK {side.lower()} eye: behind the iris {eye['behind_mm']} mm, beside it {eye['beside_mm']} mm "
                  f"({eye['vertices']} vertices look about)")
        if "mouth_turned" in face:
            turned = face["mouth_turned"]
            print(f"CHECK drawn mouth: {turned['mixes']} of {turned['of']} mixes of its shapes turn part of it backwards"
                  + (f", the worst {turned['share']:.0%} of it with {turned['mix']}" if turned["mixes"] else ""))

    measured["failed"] = failures(measured)
    if args.spots:
        measured["spots"] = render_spots(args.spots, measured, posed_by, arm, mesh_obj, edges, rest, reset, weights)
    for line in measured["failed"]:
        print(f"CHECK FAIL {line}")
    print("CHECK VERDICT " + ("look at this one by hand" if measured["failed"] else "nothing over the limits"))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(measured, handle, indent=2)


def render_spots(folder, measured, posed_by, arm, mesh_obj, edges, rest, reset, weights):
    """Close-ups of the worst stretch and squash, at rest and in the pose that did it; returns them.

    A textured, flat-lit orthographic view from the front and from the side the spot is on, the
    camera and two red markers on the edge's ends staying put between rest and posed, so the pair
    shows what the numbers mean. Only the rig check's worst two, not every pose.

    A ring is drawn round the spot in every picture, and one more picture shows the whole figure in
    the pose with the ring on it: the two 6 mm dots alone were not found on the close-ups (the user,
    2026-10-07), and a close-up of black cloth did not say where on the body it was.
    """
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    worst = measured["worst"]
    wanted = []
    if "stretch_pose" in worst:
        label = worst["stretch_pose"]
        wanted.append(("stretch", label, measured["poses"][label]["stretch_edge"], f"stretches {worst['stretch_cm']:.1f} cm"))
    if worst.get("folded_edges", 0) >= LIMITS["fold_edges"]:
        # the fold that fails the check, not the crease of a joint that only measures worse
        label = worst["fold_pose"]
        wanted.append(("squash", label, measured["poses"][label]["fold_edge"], f"folds to {worst['fold']:.0%}"))
    elif "squash_pose" in worst and worst["squash"] < 1.0:
        label = worst["squash_pose"]
        wanted.append(("squash", label, measured["poses"][label]["squash_edge"], f"shrinks to {worst['squash']:.0%}"))
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.resolution_x = scene.render.resolution_y = 480
    scene.render.film_transparent = False
    camera = bpy.data.objects.new("SpotCamera", bpy.data.cameras.new("SpotCamera"))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 0.3
    # a section: whatever lies more than CLIP in front of the spot (a mantle over a squashed
    # shoulder) is cut away, or the view shows only the cloth covering it
    camera.data.clip_start, camera.data.clip_end = 2 - SPOT_CLIP, 2 + 1.0
    scene.collection.objects.link(camera)
    scene.camera = camera
    red = bpy.data.materials.new("SpotMarker")
    red.diffuse_color = (1, 0.05, 0.05, 1)
    markers = []
    for _ in range(2):
        marker = bpy.data.objects.new("SpotMarker", bpy.data.meshes.new("SpotMarker"))
        import bmesh
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.006)
        bm.to_mesh(marker.data)
        bm.free()
        marker.data.materials.append(red)
        scene.collection.objects.link(marker)
        markers.append(marker)
    # flat, facing the camera and held in front of whatever covers the spot
    pink = bpy.data.materials.new("SpotRing")
    pink.diffuse_color = (1.0, 0.0, 0.75, 1)  # reads on skin, black cloth, white and orange hair alike
    ring = bpy.data.objects.new("SpotRing", bpy.data.meshes.new("SpotRing"))
    turn = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    circle = np.stack([np.cos(turn), np.sin(turn), np.zeros(64)], axis=1)
    ring.data.from_pydata([*(circle * SPOT_RING).tolist(), *(circle * SPOT_RING * 0.8).tolist()], [],
                          [(i, (i + 1) % 64, 64 + (i + 1) % 64, 64 + i) for i in range(64)])
    ring.data.materials.append(pink)
    scene.collection.objects.link(ring)
    matrix = np.array(mesh_obj.matrix_world)
    spots = []
    tall = float(rest[:, 2].max() - rest[:, 2].min())

    def posed_points():
        bpy.context.view_layer.update()
        evaluated = mesh_obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        data = evaluated.to_mesh()
        points = np.array([v.co for v in data.vertices]) @ matrix[:3, :3].T + matrix[:3, 3]
        evaluated.to_mesh_clear()
        return points

    def apply(pose):
        reset()
        if callable(pose):
            pose()
        else:
            for name, rotation in pose.items():
                arm.pose.bones[name].rotation_euler = Euler(rotation)
        bpy.context.view_layer.update()

    for kind, label, edge, what in wanted:
        a, b = edges[edge]
        best, chosen = None, None
        for pose in posed_by[label]:
            apply(pose)
            points = posed_points()
            length = float(np.linalg.norm(points[a] - points[b]))
            if best is None or (length > best if kind == "stretch" else length < best):
                best, chosen = length, pose
        centre = (rest[a] + rest[b]) / 2
        files = {}
        for state in ("rest", "posed"):
            if state == "posed":
                apply(chosen)
            else:
                reset()
            points = posed_points() if state == "posed" else rest
            for marker, i in zip(markers, (a, b)):
                marker.location = Vector(points[i])
            focus = (points[a] + points[b]) / 2 if state == "posed" else centre
            # from the side the spot is on: front or back, left or right
            facing = ("front", (0, -1, 0)) if centre[1] <= 0 else ("back", (0, 1, 0))
            for view, direction in (facing, ("side", (float(np.sign(centre[0]) or 1), 0, 0))):
                camera.location = Vector(focus) + Vector(direction) * 2
                camera.rotation_euler = (-Vector(direction)).to_track_quat("-Z", "Z").to_euler()
                ring.rotation_euler, ring.scale = camera.rotation_euler, (1, 1, 1)
                ring.location = Vector(focus) + Vector(direction) * (SPOT_CLIP - 0.005)
                path = folder / f"{kind}_{view}_{state}.png"
                scene.render.filepath = str(path)
                bpy.ops.render.render(write_still=True)
                files[f"{'side' if view == 'side' else 'face'}_{state}"] = str(path)
            if state == "posed":
                # the whole figure in the pose, from the spot's side, nothing cut away: where on the body it is
                direction = Vector(facing[1])
                middle = Vector((0, float(focus[1]), float(rest[:, 2].min()) + tall / 2))
                camera.data.ortho_scale, camera.data.clip_start = tall * 1.25, 0.1
                camera.location = middle + direction * 2
                camera.rotation_euler = (-direction).to_track_quat("-Z", "Z").to_euler()
                ring.rotation_euler, ring.scale = camera.rotation_euler, (tall * 1.6,) * 3
                ring.location = Vector(focus) + direction * 1.5
                path = folder / f"{kind}_where.png"
                scene.render.filepath = str(path)
                bpy.ops.render.render(write_still=True)
                files["where"] = str(path)
                camera.data.ortho_scale, camera.data.clip_start = 0.3, 2 - SPOT_CLIP
        reset()
        spots.append({"kind": kind, "pose": label, "what": what, "at": [round(float(v), 3) for v in centre],
                      "weights": [weights(a), weights(b)], "files": files})
        print(f"CHECK SPOT {kind} ({label}, {what}) at {np.round(centre, 3)}: {folder}")
    for marker in (*markers, ring):
        bpy.data.objects.remove(marker, do_unlink=True)
    bpy.data.objects.remove(camera, do_unlink=True)
    return spots


def face_shapes(arm, mesh_obj):
    """Where each shape key works and how far ({"keys": {name: {"moved", "max_mm", "at"}}}, `at` the
    middle of what it moves, world metres, weighted by how far), and `mouth_on_head`: how much of
    the surface under the mouth the head bone and its children hold. Read-only, in the rest pose.

    Model D's mouth points were saved again 30 cm down her picture: the mouth opened on her throat
    for two days of rebuilds while every pose measured as before. Height can't tell (a mascot's
    mouth sits 6 cm under its head joint, hers on the throat 7): the weights under it can, 1.00 to
    0.75 on five good characters and 0.03 on her throat.
    """
    mesh = mesh_obj.data
    if not mesh.shape_keys or len(mesh.shape_keys.key_blocks) < 2:
        return None
    matrix = np.array(mesh_obj.matrix_world)
    blocks = mesh.shape_keys.key_blocks

    def positions(block):
        co = np.zeros(len(block.data) * 3)
        block.data.foreach_get("co", co)
        return co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]

    basis = positions(blocks[0])
    found, mouth = {"keys": {}}, None
    for block in blocks[1:]:
        far = np.linalg.norm(positions(block) - basis, axis=1)
        moved = far > KEY_MOVED
        if not moved.any():
            found["keys"][block.name] = {"moved": 0, "max_mm": 0.0}
            continue
        at = (basis[moved] * far[moved, None]).sum(0) / far[moved].sum()
        found["keys"][block.name] = {"moved": int(moved.sum()), "max_mm": round(float(far.max()) * 1000, 2),
                                     "at": [round(float(v), 4) for v in at]}
        if block.name == MOUTH_KEY:
            mouth = (at, moved)
    if mouth is not None and "Head" in arm.data.bones:
        at, moved = mouth
        beside = (np.hypot(basis[:, 0] - at[0], basis[:, 2] - at[2]) < MOUTH_NEAR) & ~moved
        # the frontmost layer alone: behind it are the back of the head and the hair
        beside &= basis[:, 1] < (basis[beside, 1].min() + MOUTH_DEPTH if beside.any() else 0.0)
        if beside.any():
            head = {"Head", *(b.name for b in arm.data.bones["Head"].children_recursive)}
            groups = {g.index for g in mesh_obj.vertex_groups if g.name in head}
            held = [sum(g.weight for g in mesh.vertices[i].groups if g.group in groups) for i in np.flatnonzero(beside)]
            found["mouth_on_head"] = round(float(np.mean(held)), 2)

    mesh.calc_loop_triangles()
    triangles = np.zeros(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", triangles)
    triangles = triangles.reshape(-1, 3)
    shapes = {block.name: positions(block) for block in blocks[1:]
              if block.name in (*MOUTH_NARROWING, *VOWELS) or block.name.startswith(LOOK_KEYS)}
    looks = {side: np.max([np.linalg.norm(shapes[key + side] - basis, axis=1) for key in LOOK_KEYS if key + side in shapes]
                          or [np.zeros(len(basis))], axis=0) for side in ("Left", "Right")}
    if any(far.max() >= KEY_MOVED for far in looks.values()):
        from mathutils.bvhtree import BVHTree
        own = np.zeros(len(basis), dtype=bool)
        for far in looks.values():
            own |= (far >= KEY_MOVED) & (far >= IRIS_SHARE * far.max())
        tree = BVHTree.FromPolygons(basis.tolist(), triangles[~own[triangles].all(axis=1)].tolist())

        def cast(origin):
            hit = tree.ray_cast(Vector(origin), Vector((0.0, 1.0, 0.0)), EYE_DEEP)
            return None if hit[0] is None else hit[3]

        eyes = {side: iris_depths(basis, far, cast) for side, far in looks.items()}
        found["eyes"] = {side: eye for side, eye in eyes.items() if eye}
    drawn = [i for i, slot in enumerate(mesh_obj.material_slots) if slot.material and slot.material.name.startswith(MOUTH_PATCH)]
    if drawn:
        used = np.zeros(len(mesh.polygons), dtype=np.int64)
        mesh.polygons.foreach_get("material_index", used)
        of_polygon = np.zeros(len(mesh.loop_triangles), dtype=np.int64)
        mesh.loop_triangles.foreach_get("polygon_index", of_polygon)
        found["mouth_turned"] = mouth_turned(basis, shapes, triangles[np.isin(used[of_polygon], drawn)])
        found["mouth_turned"]["material"] = [mesh_obj.material_slots[i].material.name for i in drawn]
    return found


def spring_bones(arm):
    """The names of the bones springs move: the VRM springs kept on the armature; in a file without
    them, the bones named as a chain's are (Hair3_0, Cloth1_2)."""
    try:
        found = {joint.node.bone_name for spring in arm.data.vrm_addon_extension.spring_bone1.springs for joint in spring.joints}
    except AttributeError:  # the VRM Add-on is off
        found = set()
    return found or {bone.name for bone in arm.data.bones if "_" in bone.name and bone.name.rsplit("_", 1)[1].isdigit()}


def surface_pieces(rest, edges):
    """The piece of the surface each vertex is in, as a number: vertices joined by edges, the copies
    of a vertex at one place (PLACE; a texture seam, a file read back) counted as that one place."""
    _, where = np.unique(np.round(np.asarray(rest) / PLACE).astype(np.int64), axis=0, return_inverse=True)
    where = where.ravel()
    parent = np.arange(int(where.max()) + 1 if len(where) else 0)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in where[np.asarray(edges, dtype=np.int64).reshape(-1, 2)]:
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b
    return np.array([find(i) for i in where], dtype=np.int64)


def arm_follows(off, moved, piece, about):
    """(own, everything): of the surface round an upper arm, the share that moves with it when it is
    raised; on the arm's own surface, and on all of it. `off`: each vertex's distance from the bone;
    `moved`: whether it went along; `piece`: the piece of the surface it is in (surface_pieces);
    `about`: the direction it lies in from the bone, as an angle round it.

    The arm's own is, of the pieces with at least ARM_OWN vertices there that go round the bone
    (ARM_ROUND of ARM_SECTORS), the one lying closest to it (its vertices' median distance): what
    hangs beside or over the arm from something else is not asked to follow. With no such piece,
    everything.
    """
    off, moved, piece = np.asarray(off, dtype=float), np.asarray(moved, dtype=bool), np.asarray(piece)
    sector = np.floor((np.asarray(about, dtype=float) % (2 * np.pi)) / (2 * np.pi / ARM_SECTORS)).astype(int) % ARM_SECTORS
    everything = float(moved.mean()) if len(moved) else 0.0
    labels, counts = np.unique(piece, return_counts=True)
    round_it = [label for label in labels[counts >= ARM_OWN] if len(np.unique(sector[piece == label])) >= ARM_ROUND]
    if not round_it:
        return everything, everything
    own = min(round_it, key=lambda label: float(np.median(off[piece == label])))
    return float(moved[piece == own].mean()), everything


def places(points):
    """How many places the points are at (PLACE): a file read back has a vertex for each side of a
    texture seam and each normal, so one place counts once."""
    return len(np.unique(np.round(np.asarray(points) / PLACE).astype(np.int64), axis=0)) if len(points) else 0


def seam_copies(rest):
    """(copy, twin): every vertex that lies where an earlier one does (PLACE), and that first one."""
    _, where = np.unique(np.round(rest / PLACE).astype(np.int64), axis=0, return_inverse=True)
    where = where.ravel()
    first = np.full(int(where.max()) + 1 if len(where) else 0, len(rest), dtype=np.int64)
    np.minimum.at(first, where, np.arange(len(rest)))
    copy = np.flatnonzero(first[where] != np.arange(len(rest)))
    return copy, first[where[copy]]


def seams_kept(lead, lead_twin, swings, on_arm):
    """Which copies of a vertex the build keeps with their twin (a bool for each), by the bone that
    leads each of the two (`lead`, `lead_twin`: vertex group numbers, -1 for none; `swings`,
    `on_arm`: for each group, and a last False for none).

    Kept: both led by one bone (loose.share_seam_copies), both by spring chains
    (loose.share_strand_seams), or one by a chain and the other by the body it hangs from
    (split_torn_seams cuts a chain from an arm only). Not kept: an arm against anything else and
    two bones of the body: the two sides of a cut, a sleeve cut free of the skirt it rests on, are
    meant to part.
    """
    weighted = (lead >= 0) & (lead_twin >= 0)
    strand = swings[lead] & swings[lead_twin]
    hem = (swings[lead] != swings[lead_twin]) & ~on_arm[lead] & ~on_arm[lead_twin]
    return weighted & ((lead == lead_twin) | strand | hem)


def hand_of(names):
    """For each bone name, which hand it is of: 1 the left's hand bone and fingers, 2 the right's, 0
    neither (the arm above the wrist is not the hand: a cuff is cut from it by design)."""
    return np.array([next((number for number, side in ((1, "Left"), (2, "Right"))
                           if name.startswith(side) and any(part in name for part in HAND_BONE)), 0) for name in names])


def hand_axes(wrist, index, middle, little):
    """(across, normal) of a hand in any pose, from its wrist and three knuckles: the line the
    fingers curl about and the one they spread about."""
    along, flat = middle - wrist, index - little
    normal = np.cross(along, flat)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    across = np.cross(normal, along)
    return across / max(float(np.linalg.norm(across)), 1e-12), normal


def hair_on_the_head(arm, mesh_obj, rest):
    """(hair_findings, how much of each vertex the head bone and all that hangs from it hold) for a
    rigged mesh, its weights read in the rest pose; nothing found for a rig without a head bone."""
    bones = arm.data.bones
    names = {g.index: g.name for g in mesh_obj.vertex_groups}
    held = [(v.index, g.group, g.weight) for v in mesh_obj.data.vertices for g in v.groups if names[g.group] in bones and g.weight > 0]
    if "Head" not in bones or not held:
        return hair_findings(rest, np.zeros(len(rest)), np.ones(len(rest)), None, 0.0), np.ones(len(rest))
    vertex, group, weight = (np.array(column) for column in zip(*held))
    total = np.bincount(vertex, weights=weight, minlength=len(rest))

    def share(wanted):
        of = np.array([names.get(g) in wanted for g in range(len(mesh_obj.vertex_groups))])
        return np.bincount(vertex, weights=weight * of[group], minlength=len(rest)) / np.maximum(total, 1e-12)

    world = np.array(arm.matrix_world)
    height = lambda name: float((world @ np.array([*bones[name].head_local, 1.0]))[2])  # noqa: E731
    arms = [height(name) for name in ("LeftUpperArm", "RightUpperArm") if name in bones]
    family = {"Head", *(b.name for b in bones["Head"].children_recursive)}
    # a vertex with no weights is the check's own failure, not hair
    on_head = np.where(total > 0, share(family), 1.0)
    return hair_findings(rest, share({"Head"}), on_head, min(arms) if arms else None, height("Head")), on_head


def hair_findings(rest, on_head_bone, on_head, shoulders, head):
    """{"hangs_on_head", "hangs_cm", "shared_above", "shared_most"}: two ways hair is on the head
    wrongly, from each vertex's place (`rest`) and how much of it the head bone holds alone
    (`on_head_bone`) and with all that hangs from it (`on_head`: its hair's chains too).
    `shoulders`, `head`: the heights of the lower shoulder joint and of the head joint. Counts are
    of places.

    Hanging from the head bone alone, below the shoulders: a bit of model A's back hair (180
    vertices, 30 cm down) was put on the head by the tidy-up and stood in the air while the hair
    round it swung.

    Above the head joint, shared between the head and the body: model D's hair came 20% on the neck
    from the adopted rig and her chains hang from the head, so a tilt shared by neck and head
    parted them along the line where the springs start (291 edges, 15 mm the worst; three readings
    missed it, two of them poses that turned the head bone alone). The fox girl's first build found
    none of her pale hair, built no chain and said "ready": her hair stayed on the adopted rig's
    head, neck and chest (0.52, 0.26, 0.15 on average) and lagged the head instead of swinging.
    What a bone of the body holds almost alone up there (a tail held over the head, a staff) is
    not hair.
    """
    found = {"hangs_on_head": 0, "hangs_cm": 0.0, "shared_above": 0, "shared_most": 0.0}
    if shoulders is not None:
        hangs = (on_head_bone >= HEAD_ALONE) & (rest[:, 2] < shoulders - HANGS)
        found["hangs_on_head"] = places(rest[hangs])
        if hangs.any():
            found["hangs_cm"] = round(float(shoulders - rest[hangs, 2].min()) * 100, 1)
    above = (on_head > OFF_HEAD) & (on_head < 1.0 - OFF_HEAD) & (rest[:, 2] > head + HEAD_CLEAR)
    found["shared_above"] = places(rest[above])
    if above.any():
        found["shared_most"] = round(float(1.0 - on_head[above].min()), 2)
    return found


def mouth_turned(basis, shapes, triangles):
    """{"mixes", "of", "share", "mix"}: in how many mixes of the tracker's mouth shapes part of the
    drawn mouth (`triangles`) faces backwards, the largest share of its area so turned and the mix
    that does it. `shapes`: {key: positions}; seen from the front (x across, z up).

    A mix that leaves the mouth a sliver (under a twentieth of its widest area) isn't counted: there
    is nothing left of it to see turned.
    """
    moves = {name: shapes[name] - basis for name in (*MOUTH_NARROWING, *VOWELS) if name in shapes}
    found = {"mixes": 0, "of": 0, "share": 0.0}
    if not moves or not len(triangles):
        return found

    def facing(points):
        a, b, c = (points[triangles[:, i]] for i in range(3))
        return (b[:, 0] - a[:, 0]) * (c[:, 2] - a[:, 2]) - (b[:, 2] - a[:, 2]) * (c[:, 0] - a[:, 0])

    # which way it faces is read where it is open widest: closed, a drawn mouth is a line with no area
    open_wide = max((facing(basis + move) for move in moves.values()), key=lambda area: np.abs(area).sum())
    front, widest = np.sign(open_wide.sum()), float(np.abs(open_wide).sum())
    # a patch with sides of its own (model B's, from before the free mouth: a sixth of it faces away
    # in every shape, and nothing shows) has no one front to turn from
    if widest <= 0.0 or abs(float(open_wide.sum())) < FLAT * widest:
        return found
    narrowing = [name for name in MOUTH_NARROWING if name in moves]
    mixes = [{name: MOUTH_STEPS[i] for name, i in zip(narrowing, steps) if i}
             for steps in np.ndindex(*(len(MOUTH_STEPS),) * len(narrowing)) if any(steps)]
    mixes += [{name: 1.0} for name in VOWELS if name in moves]
    found["of"] = len(mixes)
    for mix in mixes:
        area = facing(basis + sum(value * moves[name] for name, value in mix.items()))
        if np.abs(area).sum() < 0.05 * widest:
            continue
        share = float(np.abs(area[np.sign(area) == -front]).sum() / np.abs(area).sum())
        if share > LIMITS["mouth_turned"]:
            found["mixes"] += 1
        if share > found["share"]:
            found.update(share=round(share, 3), mix=mix)
    return found


def iris_depths(basis, far, cast):
    """{"vertices", "behind_mm", "beside_mm", "beside_median_mm"} for one eye: how far into the
    head the next surface lies behind its iris (the median) and beside it (the depth a quarter of
    the rays round it pass, and the median), or None for an eye that doesn't look about.
    `far`: how far the eye's look keys take each vertex at most (m); `cast(origin)`: the depth
    (m) at which a ray into the head (+y) from there meets the surface, the iris's own triangles
    left out, or None.

    Beside it is where the eye's white shows: the fox girl's was painted on her model's own socket,
    3 cm behind the iris disc. Every front view and the contact sheets looked right; from the side
    in Warudo the iris floated in the opening and one looked into her head.
    """
    if not len(far) or far.max() < KEY_MOVED:
        return None
    iris = np.flatnonzero(far >= IRIS_SHARE * far.max())
    points = basis[iris]

    def depth(origin):
        met = cast(origin)
        return EYE_DEEP if met is None else min(float(met), EYE_DEEP)

    behind = [depth(p + (0.0, 1e-4, 0.0)) for p in points[::max(1, len(points) // 200)]]
    centre = points.mean(axis=0)
    across = np.stack([points[:, 0] - centre[0], points[:, 2] - centre[2]], axis=1)
    beside = []
    for turn in np.linspace(0, 2 * np.pi, IRIS_RAYS, endpoint=False):
        way = np.array([np.cos(turn), np.sin(turn)])
        out = across @ way
        # the iris's rim on this side: the vertices furthest out within a twelfth of a turn
        rim = out > np.cos(np.pi / 12) * np.linalg.norm(across, axis=1)
        if not rim.any():
            continue
        reach = float(out[rim].max())
        level = float(np.median(points[rim & (out > 0.7 * reach), 1]))
        at = centre[[0, 2]] + (reach + IRIS_ROUND) * way
        # from the iris's own depth: the frame in front of it is a window where it matters, cut out of
        # its picture. A ray that meets nothing has gone under a lid with no white behind it at all
        # (model A's upper lids, 4 to 9 mm in front of her irises): nothing to see there either.
        met = cast(np.array([at[0], level + 1e-4, at[1]]))
        if met is not None:
            beside.append(min(float(met), EYE_DEEP))
    if len(beside) < IRIS_RAYS // 4:
        return None
    # A quarter of the way round is enough to see into: above and below, the lids' own skin lies
    # right at the iris whatever is behind it (the fox girl: 13 of 24 rays under 7 mm, 11 at 27 to 31)
    return {"vertices": int(len(iris)), "behind_mm": round(float(np.median(behind)) * 1000, 1),
            "beside_mm": round(float(np.percentile(beside, 75)) * 1000, 1),
            "beside_median_mm": round(float(np.median(beside)) * 1000, 1), "rays": len(beside)}


def folds_outside_the_blend(share, middle, squashed):
    """(crease, folded, how many): which edges have both ends shared between the bones the pose
    moves and the ones that stay (`share`: per edge, how much of each end the moving bones hold;
    SHARED), which of the others are under the squash limit, and how many places that is. A
    crease is a joint's own, closing; a file read back has an edge for each copy of a vertex
    along a texture seam, so one place counts once.
    """
    crease = ((share >= SHARED[0]) & (share <= SHARED[1])).all(axis=1)
    folded = np.nonzero((squashed < LIMITS["squash"]) & ~crease)[0]
    places = len(np.unique(np.round(middle[folded] / 1e-4).astype(np.int64), axis=0)) if len(folded) else 0
    return crease, folded, places


def failures(measured):
    """Which measurements are over the limits, in plain language."""
    failed = []
    if measured["unweighted"] > LIMITS["unweighted"]:
        failed.append(f"{measured['unweighted']} vertices have no weights: they stay put while the body moves")
    worst = measured["worst"]
    if worst.get("stretch_cm", 0.0) > LIMITS["stretch_cm"]:
        failed.append(f"an edge stretches {worst['stretch_cm']:.1f} cm with the {worst['stretch_pose']}: "
                      "something is weighted to the wrong bone")
    for side, share in measured.get("arm_follows", {}).items():
        if share < LIMITS["arm_follows"]:
            failed.append(f"the {side.lower()} upper arm takes only {share:.0%} of its own surface along when it is "
                          "raised: the sleeve is weighted to another bone")
    on_head = measured.get("face", {}).get("mouth_on_head")
    if on_head is not None and on_head < LIMITS["mouth_on_head"]:
        failed.append(f"the head bone holds only {on_head:.0%} of the surface under the mouth: the mouth isn't on "
                      "the face (look at the face points)")
    if worst.get("folded_edges", 0) >= LIMITS["fold_edges"]:
        failed.append(f"{worst['folded_edges']} edges fold to under {LIMITS['squash']:.0%} of their length with the "
                      f"{worst['fold_pose']} (the worst to {worst['fold']:.0%}), where the weights step from one bone to "
                      "the next: they change too fast there")
    seams = measured.get("seams", {})
    if seams.get("torn", 0) > LIMITS["torn"]:
        failed.append(f"{seams['torn']} places where the surface is joined at rest come apart with the {seams['pose']} "
                      f"(up to {seams['torn_cm']:.1f} cm): a tear, the copies of a vertex along a texture seam held by "
                      "different bones")
    if seams.get("fingers", 0) > LIMITS["finger_slits"]:
        failed.append(f"{seams['fingers']} places in the hands come apart as the fingers curl and spread (up to "
                      f"{seams['fingers_mm']:.0f} mm): a slit between two fingers, the copies of a vertex along a texture "
                      "seam held by different fingers")
    hair = measured.get("hair", {})
    if hair.get("hangs_on_head", 0) > LIMITS["hangs_on_head"]:
        failed.append(f"{hair['hangs_on_head']} vertices hang from the head bone to {hair['hangs_cm']:.0f} cm below the "
                      "shoulders: hair no spring holds, it turns with the head and stands still while the hair round it swings")
    if hair.get("ledge_edges", 0) > LIMITS["ledge_edges"]:
        failed.append(f"{hair['ledge_edges']} edges above the head joint grow over {LEDGE * 1000:.0f} mm when head and neck "
                      f"tilt together (the worst {hair['ledge_mm']:.0f} mm): the neck or the body holds part of what is the "
                      "head's, hair most often; it lags the head and steps where its springs start")
    for side, eye in measured.get("face", {}).get("eyes", {}).items():
        if eye["beside_mm"] > LIMITS["beside_iris_mm"]:
            failed.append(f"the surface beside the {side.lower()} iris lies {eye['beside_mm']:.0f} mm behind it: the eye's "
                          "white is deep in the socket, from the side the iris floats and the inside of the head shows")
    turned = measured.get("face", {}).get("mouth_turned", {})
    if turned.get("mixes"):
        failed.append(f"the drawn mouth turns inside out in {turned['mixes']} of {turned['of']} mixes of its shapes "
                      f"({turned['share']:.0%} of it facing backwards with {_mix(turned['mix'])}): the shapes narrow it "
                      "past nothing between them")
    return failed


def _mix(values):
    return ", ".join(f"{name} {value:g}" for name, value in values.items())


if __name__ == "__main__":
    main()
