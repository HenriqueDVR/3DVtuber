"""Fix hair that hangs still or detached on a rigged avatar .blend, then export VRM.

    blender -b in.blend -P scripts/fix_hair_weights.py -- out.blend out.vrm

0. Vertices with no weights at all (they stay put when the head moves) take the weights of
   the weighted surface around them.
1. Hair-coloured separate pieces weighted to body bones (ties, beads sitting in the hair)
   take the weights of the nearest hair, so they move with it.
2. Hair vertices far from the bone that moves them are reweighted to their nearest hair
   bones, blended and smoothed along the surface, so strands follow the bone beside them.
3. Hidden hair faces joining strands whose roots are far apart (a closed hair shell's
   underside, inside the head) are deleted, as are any joining two strands inside the skull
   or behind the neck: when the strands swing apart those faces stretch into long strips
   poking out of the hair.
The original file is left alone. Needs the VRM Add-on for Blender.
"""
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils.kdtree import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import parts, rig, texture_space  # noqa: E402

BODY_BONES = {"Hips", "Spine", "Chest", "UpperChest", "LeftShoulder", "RightShoulder",
              "LeftUpperArm", "RightUpperArm", "LeftLowerArm", "RightLowerArm", "LeftHand", "RightHand"}
# Reweight a hair vertex when its bone is further than FAR (m) and FAR_RATIO times the nearest one.
FAR = 0.06
FAR_RATIO = 1.5
SMOOTHING = 4
# Faces whose corners follow hair chains with roots further apart than this (m) are removed,
# if they lie within HIDDEN_RADIUS (m) of the head's vertical axis. Below the crown (more than
# BELOW_CROWN m under the hair roots) that axis runs inside the skull and behind the neck, so
# any faces there within HIDDEN_RADIUS_BELOW_CROWN joining two chains go, even neighbouring ones.
BRIDGE_GAP = 0.10
HIDDEN_RADIUS = 0.06
BELOW_CROWN = 0.06
HIDDEN_RADIUS_BELOW_CROWN = 0.08
MAX_TIE_VERTS = 2000


def hair_color_share(mesh, image):
    w, h = image.size
    px = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    uv = mesh.uv_layers.active.data
    hits = np.zeros(len(mesh.vertices))
    count = np.zeros(len(mesh.vertices))
    for poly in mesh.polygons:
        if poly.material_index != 0:
            continue
        for li in poly.loop_indices:
            u, v = uv[li].uv
            c = px[int(v * h) % h, int(u * w) % w, :3]
            sat = (c.max() - c.min()) / max(c.max(), 1e-6)
            vi = mesh.loops[li].vertex_index
            hits[vi] += float(sat > 0.55 and c[0] > 0.5 and c[0] > c[2] + 0.3)
            count[vi] += 1
    return hits / np.maximum(count, 1)


def segment_distance(points, head, tail):
    s = tail - head
    a = np.clip((points - head) @ s / (s @ s), 0.0, 1.0)
    return np.linalg.norm(points - (head + a[:, None] * s), axis=1)


def main():
    out_blend, out_vrm = sys.argv[sys.argv.index("--") + 1:]
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh_obj = next(o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm)
    mesh = mesh_obj.data
    groups = mesh_obj.vertex_groups
    names = {g.index: g.name for g in groups}
    matrix = np.array(mesh_obj.matrix_world)
    verts = np.array([v.co for v in mesh.vertices]) @ matrix[:3, :3].T + matrix[:3, 3]
    edges = np.array([e.vertices[:] for e in mesh.edges])
    labels = parts.split(verts, edges)

    def weights(i):
        return {names[g.group]: g.weight for g in mesh.vertices[i].groups}

    def set_weights(i, new):
        for g in list(mesh.vertices[i].groups):
            groups[g.group].remove([i])
        total = sum(new.values())
        for name, w in new.items():
            if w / total > 1e-4:
                (groups.get(name) or groups.new(name=name)).add([i], w / total, "REPLACE")

    # 0. unweighted vertices stay behind when the head moves
    print(f"weighted {rig.fill_unweighted(mesh_obj)} vertices that had no weights")

    hair_w = np.array([sum(w for n, w in weights(i).items() if n.startswith("Hair")) for i in range(len(verts))])
    hair_idx = np.nonzero(hair_w > 0.5)[0]
    if not len(hair_idx) or not any(b.name.startswith("Hair") for b in arm.data.bones):
        # a solid head (--rigid-head) has no hair bones: filling unweighted vertices above is
        # still worth doing, the rest of this script has nothing to work on
        print("no hair bones: nothing to reweight")
        export(arm, out_blend, out_vrm)
        return
    tree = KDTree(len(hair_idx))
    for k, i in enumerate(hair_idx):
        tree.insert(verts[i], k)
    tree.balance()

    # 1. ties and beads in the hair
    image = texture_space.material_base_color_image(mesh.materials[0])
    orange = hair_color_share(mesh, image)
    fixed_parts = 0
    for part in np.unique(labels):
        members = np.nonzero(labels == part)[0]
        if len(members) > MAX_TIE_VERTS or hair_w[members].mean() > 0.05 or orange[members].mean() < 0.8:
            continue
        body = np.mean([max(weights(i), key=weights(i).get) in BODY_BONES for i in members[:: max(1, len(members) // 20)]])
        if body < 0.5:
            continue
        for i in members:
            _co, k, _d = tree.find(verts[i])
            set_weights(int(i), weights(int(hair_idx[k])))
        fixed_parts += 1
        print(f"tie part: {len(members)} verts at {verts[members].mean(0).round(3)} now follows the hair")

    # 2. hair vertices far from their bone
    bones = [b for b in arm.data.bones if b.name.startswith("Hair")]
    heads = [np.array(arm.matrix_world @ b.head_local) for b in bones]
    tails = [np.array(arm.matrix_world @ b.tail_local) for b in bones]
    pts = verts[hair_idx]
    dist = np.stack([segment_distance(pts, h, t) for h, t in zip(heads, tails)], axis=1)
    index = {b.name: n for n, b in enumerate(bones)}
    moved = {}
    for k, i in enumerate(hair_idx):
        w = weights(int(i))
        hair = {n: x for n, x in w.items() if n in index}
        dominant = max(hair, key=hair.get)
        current = dist[k, index[dominant]]
        nearest = dist[k].min()
        if current > FAR and current > FAR_RATIO * nearest:
            two = np.argsort(dist[k])[:2]
            inv = 1.0 / np.maximum(dist[k, two], 0.005) ** 2
            moved[int(i)] = {bones[b].name: float(x) for b, x in zip(two, inv / inv.sum())}
    # smooth the new weights along the surface, mixing in unchanged neighbours
    neighbours = {i: [] for i in moved}
    for a, b in edges:
        if a in neighbours:
            neighbours[a].append(b)
        if b in neighbours:
            neighbours[b].append(a)
    current = {i: dict(moved[i]) for i in moved}
    for _ in range(SMOOTHING):
        nxt = {}
        for i, w in current.items():
            acc = dict(w)
            for j in neighbours[i]:
                other = current.get(j) or {n: x for n, x in weights(j).items() if n in index}
                for n, x in other.items():
                    acc[n] = acc.get(n, 0.0) + x
            total = sum(acc.values())
            nxt[i] = {n: x / total for n, x in acc.items()}
        current = nxt
    for i, w in current.items():
        top = dict(sorted(w.items(), key=lambda kv: -kv[1])[:4])
        set_weights(i, top)
    print(f"reweighted {len(current)} hair vertices that were far from their bone; fixed {fixed_parts} tie parts")

    # 3. hidden faces bridging far-apart strands
    roots = {b.name.split("_")[0]: np.array(arm.matrix_world @ b.head_local) for b in bones if b.name.endswith("_0")}
    head_axis = np.mean([r[:2] for r in roots.values()], axis=0)  # middle of the hair roots
    crown = min(r[2] for r in roots.values())
    dominant = {}
    for i in hair_idx:
        chains = {}
        for n, w in weights(int(i)).items():
            if n.startswith("Hair"):
                chains[n.split("_")[0]] = chains.get(n.split("_")[0], 0.0) + w
        dominant[int(i)] = max(chains, key=chains.get)
    bridging = []
    for poly in mesh.polygons:
        corner = [dominant[v] for v in poly.vertices if v in dominant]
        if len(set(corner)) < 2:
            continue
        centre = verts[list(poly.vertices)].mean(axis=0)
        radius = np.linalg.norm(centre[:2] - head_axis)
        if centre[2] < crown - BELOW_CROWN and radius < HIDDEN_RADIUS_BELOW_CROWN:
            bridging.append(poly.index)
            continue
        if radius > HIDDEN_RADIUS:
            continue
        gap = max(np.linalg.norm(roots[a] - roots[b]) for a in set(corner) for b in set(corner))
        if gap > BRIDGE_GAP:
            bridging.append(poly.index)
    if bridging:
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.faces[i] for i in bridging], context="FACES_ONLY")
        bmesh.ops.delete(bm, geom=[e for e in bm.edges if not e.link_faces], context="EDGES")
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
    print(f"removed {len(bridging)} hidden hair faces joining far-apart strands")

    export(arm, out_blend, out_vrm)


def export(arm, out_blend, out_vrm):
    bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(out_blend), copy=True)
    print("exported", bpy.ops.export_scene.vrm(filepath=bpy.path.abspath(out_vrm), armature_object_name=arm.name))


main()
