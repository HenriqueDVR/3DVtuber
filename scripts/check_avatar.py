"""Find rigging bugs on a rigged avatar .blend by moving bones and measuring the mesh.

    blender -b in.blend -P scripts/check_avatar.py [-- --swing 0.7] [--json report.json]

Reports:
- vertices with no weights (they stay put while the body moves)
- edges that stretch when the head turns (parts not following the head)
- edges that stretch when the arms go to a T-pose, hang, reach forward or rise, and when the
  spine and chest twist and bend or a leg lifts (sleeves left behind, torn clothing)
- edges that shrink to under half their length in those poses (a chest caving in because the
  arm bones reach onto it, cloth folding into itself)
- edges that stretch when each hair, cloth or hat tip chain swings alone (strips poking out,
  torn strands), with the weights on both ends of the worst edge
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
SHOW = 3
# Close-ups of the worst spots (--spots) cut away what lies more than this (m) in front of them.
SPOT_CLIP = 0.08
# The ring drawn round a spot in its close-ups (m, its outer radius): a fifth of the picture's width.
SPOT_RING = 0.03
# A build is worth looking at by hand when it is over one of these.
LIMITS = {"unweighted": 0, "stretch_cm": 6.0, "squash": 0.35}


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

    empty = [v.index for v in mesh.vertices if not any(g.weight > 0 for g in v.groups)]
    print(f"CHECK unweighted vertices: {len(empty)}" + (f" (e.g. at {rest[empty[0]].round(3)})" if empty else ""))
    measured["unweighted"] = len(empty)

    def weights(i):
        top = sorted(mesh.vertices[i].groups, key=lambda g: -g.weight)[:3]
        return {names[g.group]: round(g.weight, 2) for g in top}

    def reset():
        for bone in arm.pose.bones:
            bone.rotation_mode = "XYZ"
            bone.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()

    squashed = np.ones(len(edges))

    posed_by = {}  # label -> the poses measured, to show the worst of them again

    def stretch(rotations):
        stretch.last = rotations
        worst = np.zeros(len(edges))
        squashed[:] = 1.0
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
            length = np.linalg.norm(posed[edges[:, 0]] - posed[edges[:, 1]], axis=1)
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

    def report_squash(label):
        bad = np.nonzero(squashed < SQUASH)[0]
        print(f"CHECK {label} squash: {len(bad)} edges shrink below {SQUASH:.0%}, worst to {squashed.min():.0%}")
        for e in bad[np.argsort(squashed[bad])][:SHOW]:
            a, b = edges[e]
            print(f"    {squashed[e]:.0%}  {rest[a].round(3)} {weights(a)}  <->  {rest[b].round(3)} {weights(b)}")
        pose = measured["poses"].setdefault(label, {})
        pose["squashed_edges"] = int(len(bad))
        pose["squash"] = round(float(squashed.min()), 3)
        pose["squash_edge"] = int(np.argmin(squashed))
        if pose["squash"] < measured["worst"].get("squash", 1.0):
            measured["worst"] = dict(measured["worst"], squash=pose["squash"], squash_pose=label)

    if "Head" in arm.pose.bones:
        report("head turn", stretch([{"Head": r} for r in HEAD_TURNS]))
    if "LeftUpperArm" in arm.pose.bones:
        for label, direction in ARM_POSES.items():
            report(f"arms {label}", stretch([point_arms(direction)]))
            report_squash(f"arms {label}")
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
    if "squash_pose" in worst and worst["squash"] < 1.0:
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


def failures(measured):
    """Which measurements are over the limits, in plain language."""
    failed = []
    if measured["unweighted"] > LIMITS["unweighted"]:
        failed.append(f"{measured['unweighted']} vertices have no weights: they stay put while the body moves")
    worst = measured["worst"]
    if worst.get("stretch_cm", 0.0) > LIMITS["stretch_cm"]:
        failed.append(f"an edge stretches {worst['stretch_cm']:.1f} cm with the {worst['stretch_pose']}: "
                      "something is weighted to the wrong bone")
    if worst.get("squash", 1.0) < LIMITS["squash"]:
        failed.append(f"an edge shrinks to {worst['squash']:.0%} with the {worst['squash_pose']}: "
                      "a limb's weights reach onto the body it folds into")
    return failed


main()
