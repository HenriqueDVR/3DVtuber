"""Give the ahoge (the strand standing up on top of the head) a spring, then export VRM.

    blender -b in.blend -P scripts/add_ahoge_spring.py -- out.blend out.vrm

Finds the topmost small separate piece on the head, adds a short bone chain from its base
to its top under the head bone, weights it by height and adds a VRM spring. When that piece has
a twin across the head (a pair of ears), the twin gets the same: one ear swinging beside a still
one is worse than neither. The original file is left alone. Needs the VRM Add-on for Blender.
"""
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import parts, vrm_export  # noqa: E402

MAX_SIZE = 0.15          # the ahoge fits in this (m)
BONES = 2
BASE_BLEND = 0.2         # share of its height (from the base) blending from the head
STIFFNESS, DRAG, GRAVITY = 1.2, 0.35, 0.05
EAR_STIFFNESS, EAR_DRAG = 2.5, 0.5
TWIN = 0.02              # a twin's box is the piece's own, mirrored, within this (m)


def main():
    out_blend, out_vrm = sys.argv[sys.argv.index("--") + 1:]
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh_obj = next(o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm)
    mesh = mesh_obj.data
    matrix = np.array(mesh_obj.matrix_world)
    verts = np.array([v.co for v in mesh.vertices]) @ matrix[:3, :3].T + matrix[:3, 3]
    labels = parts.split(verts, np.array([e.vertices[:] for e in mesh.edges]))

    top = labels[int(np.argmax(verts[:, 2]))]
    members = np.nonzero(labels == top)[0]
    if np.ptp(verts[members], axis=0).max() > MAX_SIZE:
        sys.exit(f"the topmost piece is {np.ptp(verts[members], axis=0).max():.2f} m across, too big for an ahoge")
    # the catgirl's ears are two pieces: the taller by a hair was the ahoge, and one ear swung
    middle = float((arm.matrix_world @ arm.data.bones["Head"].head_local).x)
    low, high = verts[members].min(axis=0), verts[members].max(axis=0)
    mirrored = (np.array([2 * middle - high[0], low[1], low[2]]), np.array([2 * middle - low[0], high[1], high[2]]))
    for label in np.unique(labels):
        twin = np.nonzero(labels == label)[0]
        if label != top and np.abs(verts[twin].min(axis=0) - mirrored[0]).max() < TWIN                 and np.abs(verts[twin].max(axis=0) - mirrored[1]).max() < TWIN:
            add_chain(arm, mesh_obj, verts, members, "Ahoge", ear=True)
            add_chain(arm, mesh_obj, verts, twin, "Ahoge2", ear=True)
            break
    else:
        add_chain(arm, mesh_obj, verts, members, "Ahoge")

    bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(out_blend), copy=True)
    if not vrm_export.skipped():
        print("exported", bpy.ops.export_scene.vrm(filepath=bpy.path.abspath(out_vrm), armature_object_name=arm.name))


def add_chain(arm, mesh_obj, verts, members, name, ear=False):
    """A chain of BONES bones named `name`_k under the head for the piece `members`, and its spring.

    An `ear` is a shell, not a strand: walked along its surface, the way from its foot leads over
    the tip and down its other side, and the chain went up the ear and back down into the head
    (the catgirl's: second bone 3 cm lower than it began, the whole ear on it). Its chain is the
    straight line from the middle of its foot to its tip, and it swings stiffer, with no weight.
    """
    mesh = mesh_obj.data
    points = verts[members]
    # distance along the strand from its base, walking the mesh, so a curl isn't cut across
    low = points[:, 2].min()
    local = {int(v): k for k, v in enumerate(members)}
    neighbours = [[] for _ in members]
    for a, b in (e.vertices[:] for e in mesh.edges):
        if a in local and b in local:
            d = float(np.linalg.norm(verts[a] - verts[b]))
            neighbours[local[a]].append((local[b], d))
            neighbours[local[b]].append((local[a], d))
    # the mesh is split along UV seams: join vertices that share a position
    keys = np.round(points / 1e-6).astype(np.int64)
    _, weld = np.unique(keys, axis=0, return_inverse=True)
    weld = weld.ravel()
    by_weld = {}
    for k, w in enumerate(weld):
        by_weld.setdefault(int(w), []).append(k)
    for same in by_weld.values():
        for a in same:
            for b in same:
                if a != b:
                    neighbours[a].append((b, 0.0))
    import heapq
    along = np.full(len(members), np.inf)
    queue = []
    for k in np.nonzero(points[:, 2] < low + 0.1 * np.ptp(points[:, 2]))[0]:
        along[k] = 0.0
        queue.append((0.0, int(k)))
    heapq.heapify(queue)
    while queue:
        d, k = heapq.heappop(queue)
        if d > along[k]:
            continue
        for j, step in neighbours[k]:
            if d + step < along[j]:
                along[j] = d + step
                heapq.heappush(queue, (d + step, j))
    along[~np.isfinite(along)] = 0.0
    t_all = along / max(along.max(), 1e-6)
    joints = []
    for k in range(BONES + 1):
        band = np.abs(t_all - k / BONES) < 0.08
        joints.append(points[band].mean(axis=0))
    if ear:
        foot = points[points[:, 2] < low + 0.15 * np.ptp(points[:, 2])].mean(axis=0)
        peak = points[int(np.argmax(points[:, 2]))]
        t_all = np.clip((points - foot) @ (peak - foot) / float((peak - foot) @ (peak - foot)), 0.0, 1.0)
        along = t_all * float(np.linalg.norm(peak - foot))
        joints = [foot + (peak - foot) * k / BONES for k in range(BONES + 1)]
    base, tip = joints[0], joints[-1]
    print(f"{name.lower()}: {len(members)} verts, strand {along.max():.3f} m, base {base.round(3)}, tip {tip.round(3)}")

    names = [f"{name}_{k}" for k in range(BONES)] + [f"{name}_{BONES}"]
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    edit = arm.data.edit_bones
    inverse = arm.matrix_world.inverted()
    for name in names:
        if name in edit:
            edit.remove(edit[name])
    parent = edit["Head"]
    for k, name in enumerate(names):
        bone = edit.new(name)
        head = joints[k]
        tail = joints[k + 1] if k < BONES else joints[-1] + (joints[-1] - joints[-2]) * 0.5
        bone.head, bone.tail = inverse @ Vector(head), inverse @ Vector(tail)
        bone.parent = parent
        bone.use_connect = k > 0
        parent = bone
    bpy.ops.object.mode_set(mode="OBJECT")

    groups = [mesh_obj.vertex_groups.get(n) or mesh_obj.vertex_groups.new(name=n) for n in names[:BONES]]
    head_group = mesh_obj.vertex_groups["Head"]
    for i, t in zip(members, t_all):
        for g in list(mesh.vertices[int(i)].groups):
            mesh_obj.vertex_groups[g.group].remove([int(i)])
        head_w = max(0.0, 1.0 - t / BASE_BLEND)
        pos = t * BONES
        lower = min(int(pos), BONES - 1)
        frac = pos - lower
        shares = {lower: 1.0 - frac}
        if lower + 1 < BONES:
            shares[lower + 1] = frac
        else:
            shares[lower] = 1.0  # past the last joint: all on the last bone
        if head_w > 1e-3:
            head_group.add([int(i)], head_w, "REPLACE")
        for b, share in shares.items():
            if share * (1 - head_w) > 1e-3:
                groups[b].add([int(i)], share * (1 - head_w), "REPLACE")

    spring_bone = arm.data.vrm_addon_extension.spring_bone1
    for index in reversed(range(len(spring_bone.springs))):
        if spring_bone.springs[index].vrm_name == name:
            spring_bone.springs.remove(index)
    spring = spring_bone.add_spring()
    spring.vrm_name = name
    for name in names:
        joint = spring.joints.add()
        joint.node.bone_name = name
        joint.stiffness, joint.drag_force, joint.gravity_power, joint.hit_radius = \
            (EAR_STIFFNESS, EAR_DRAG, 0.0, 0.01) if ear else (STIFFNESS, DRAG, GRAVITY, 0.01)


main()
