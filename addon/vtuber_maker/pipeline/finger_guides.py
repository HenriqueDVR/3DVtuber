"""Finger guides read off the hand's surface, then applied through the hand-setup editor.

`fingers.build_fingers` names chains from how far their tips reach, and on the real models that
went wrong where it matters: model B's thumb got no bones at all and its index chain started at the
thumb tip; other chains stopped half way up a finger or zigzagged across it. Bones placed like
that twist a tracked finger however the weights are painted.

This finds each finger as a tube on the surface and lays its bones down the middle of it:

* fingertips are points furthest from the wrist, over the surface, whose surface narrows into a
  tube of steady width behind them -- knuckle and palm bumps are such points too, but their
  rings of surface widen at once;
* a finger's centreline is the centre of each ring of surface at a growing distance from its tip,
  up to the web, where the ring swells into the palm;
* knuckles sit back from the webs along the hand, joints at standard proportions of the finger;
* the thumb is the digit that leaves the hand nearest the wrist, its metacarpal runs from beside
  the wrist, and it curls toward the palm and the little finger.

The result is the same guide data the Guided Hand Setup panel edits, written as its guide
empties, so a person can still nudge a joint in Blender and apply again. Read on the modelled
A-pose, before the T-pose bake; coarse image-to-3D meshes (6-9 mm edges on 1 cm fingers) are
expected. Refuses a hand where five digits can't be told apart rather than guessing.
"""

import heapq

import numpy as np

from . import fingers, hand_setup, meshdata
from .rig import BONE_NAMES

# A fingertip's surface is a tube while its rings stay this much thinner than the thinnest found.
TUBE = 1.3
# ... the thumb is thicker than the fingers, so it may be up to this much thinner.
THUMB_TUBE = 1.8
# Two tips of different fingers are at least this far apart over the surface (m); a second bump
# on the same coarse finger is not.
TIPS_APART = .05
# Along a finger, the ring of surface is the palm once it swells past this multiple of the
# finger's own width.
WEB = 1.45
# Rings are this far apart along the finger (m).
RING_STEP = .004
# Of a finger's length from its knuckle, the visible part starts here (the web) and its joints
# sit here -- the proportions `fingers.JOINTS_AT` also uses.
WEB_AT = .15
JOINTS_AT = (.45, .75)
# Of the visible thumb from its web, its interphalangeal joint sits here.
THUMB_IP = .55
# Guide influence radius, as a multiple of the finger's own ring radius.
REACH = 1.9
# Past the web, the surface becomes wholly the finger's over this distance (m).
OWN_FADE = .015
# Passes of weight averaging over the hand's surface once the guides are applied.
SMOOTHING = 3
# The layout below is read off a VRoid hand that tracks well in Warudo (VRM1_Constraint_Twist_Sample
# from the three-vrm repo, measured 2026-09-22), in units of the wrist-to-knuckle length.
# The thumb's first joint: level with the wrist, toward the thumb and toward the palm.
THUMB_ROOT = (.02, .25, .15)
# The thumb's first bone carries the ball of the thumb: surface within this distance of it is
# wholly the bone's, fading out by the second (the reference moves 47 vertices this way; ours
# moved none, so a tracked thumb swung from a base that stayed put).
THENAR = (.2, .35)
# Passes of averaging over the surface that soften the edge of the ball of the thumb.
BALL_SMOOTHING = 2
# ... which fades in over this much of the hand from the wrist, and between these distances
# from the hand's centre line toward the thumb.
BALL_FROM = .15
BALL_ACROSS = (.1, .25)
# Finger weight ends this far behind the knuckle (the reference: none behind it). Averaging had
# carried it a quarter of the hand back into the palm, which then bent with the finger.
KNUCKLE_FADE = .1


def _dijkstra(adj, sources, count, limit=np.inf):
    distance = np.full(count, np.inf)
    heap = [(0.0, int(s)) for s in sources]
    for s in sources:
        distance[s] = 0.0
    heapq.heapify(heap)
    while heap:
        d, u = heapq.heappop(heap)
        if d > distance[u] or d > limit:
            continue
        for v, w in adj[u]:
            if d + w < distance[v]:
                distance[v] = d + w
                heapq.heappush(heap, (d + w, v))
    return distance


def hand_piece(mesh_obj, side, verts=None):
    """The separate piece of mesh carrying most of this hand's weight: the glove or bare hand.

    Cuffs and straps beside the wrist are pieces of their own; finger weight belongs on none.
    """
    names = hand_setup._names(side)
    ours = {mesh_obj.vertex_groups[n].index for n in names if n in mesh_obj.vertex_groups}
    weight = np.zeros(len(mesh_obj.data.vertices))
    for vertex in mesh_obj.data.vertices:
        weight[vertex.index] = sum(g.weight for g in vertex.groups if g.group in ours)
    labels = meshdata.pieces(mesh_obj, verts)
    return labels == int(np.argmax(np.bincount(labels, weights=weight)))


def _surface(mesh_obj, armature_obj, side, verts, piece):
    """The hand piece past the wrist, as a welded graph for walking over."""
    weld, edge_a, edge_b = meshdata.welded_edges(mesh_obj, verts)
    bone = armature_obj.data.bones[BONE_NAMES[f"{side}_hand"]]
    wrist = np.array(armature_obj.matrix_world @ bone.head_local)
    along = np.array(armature_obj.matrix_world @ bone.tail_local) - wrist
    along /= np.linalg.norm(along)
    depth = (verts - wrist) @ along
    reps = np.unique(weld[piece & (depth > -.005)])
    inside = np.zeros(len(verts), bool)
    inside[reps] = True
    a, b = weld[edge_a], weld[edge_b]
    keep = inside[a] & inside[b] & (a != b)
    a, b = a[keep], b[keep]
    adj = {int(i): [] for i in reps}
    for i, j, length in zip(a.tolist(), b.tolist(), np.linalg.norm(verts[a] - verts[b], axis=1).tolist()):
        adj[i].append((j, length))
        adj[j].append((i, length))
    return reps, adj, wrist, depth, weld


def _ring(verts, members, walk, at, band):
    """Centre and median radius of the surface `at` a distance from a tip, or (None, None)."""
    ring = members[np.abs(walk[members] - at) < band]
    if len(ring) < 4:
        return None, None
    centre = verts[ring].mean(0)
    return centre, float(np.median(np.linalg.norm(verts[ring] - centre, axis=1)))


def _tips(verts, adj, reached, from_wrist, span, band, count):
    """Four finger tips and the thumb's, as vertex indices (thumb last), or raise."""
    candidates = []
    for i in reached[np.argsort(-from_wrist[reached])]:
        if from_wrist[i] < .3 * span or len(candidates) >= 15:
            break
        if any(np.linalg.norm(verts[i] - verts[t]) < .015 for t, _ in candidates):
            continue
        local = _dijkstra(adj, [int(i)], count, limit=.035)
        if from_wrist[reached[local[reached] < .012]].max() > from_wrist[i] + 1e-9:
            continue  # something nearby reaches further: not a tip
        around = reached[np.isfinite(local[reached])]
        radii = [r for r in (_ring(verts, around, local, at, band)[1] for at in np.arange(.01, .031, .004))
                 if r is not None]
        if len(radii) >= 3:
            candidates.append((int(i), float(np.median(radii))))
    if not candidates:
        raise ValueError("no fingertips stand out from the hand's surface")
    thinnest = min(r for _, r in candidates)
    walked = {}

    def apart(i, chosen):
        if i not in walked:
            walked[i] = _dijkstra(adj, [i], count, limit=TIPS_APART)
        return all(walked[i][t] >= TIPS_APART for t in chosen)

    tips = []
    for i, r in candidates:
        if len(tips) < 4 and r <= TUBE * thinnest and apart(i, tips):
            tips.append(i)
    rest = [(r, i) for i, r in candidates if i not in tips and r <= THUMB_TUBE * thinnest and apart(i, tips)]
    if len(tips) < 4 or not rest:
        raise ValueError(f"found {len(tips) + bool(rest)} digits standing out from the hand, need 5")
    return tips + [min(rest)[1]]


def _chain(verts, members, walk, tip, span, band):
    """Centreline from the tip to the web, the finger's ring radius, and its visible length."""
    centres, rows = [verts[tip]], []
    for at in np.arange(RING_STEP, span, RING_STEP):
        centre, radius = _ring(verts, members, walk, at, band)
        if centre is not None:
            centres.append(centre)
            rows.append((at, radius))
    rows = np.array(rows)
    plateau = float(np.median(rows[(rows[:, 0] >= .01) & (rows[:, 0] <= .03), 1]))
    smooth = np.array([np.median(rows[max(0, j - 1):j + 2, 1]) for j in range(len(rows))])
    wide = np.flatnonzero((rows[:, 0] > .025) & (smooth > WEB * plateau))
    stop = int(wide[0]) if len(wide) else len(rows)
    line = np.array(centres[:stop + 1])
    # end on the centreline at the tip's depth, not on the tip vertex off to one side: the far
    # side of the fingertip otherwise falls outside the bone's reach and stays on the hand
    if len(line) >= 3:
        ahead = line[1] - line[2]
        ahead /= np.linalg.norm(ahead)
        line[0] = line[1] + ahead * max(float((line[0] - line[1]) @ ahead), 0.0)
    return line[::-1], plateau  # web -> tip


def _hand_frame(wrist, knuckles, side):
    """Unit axes along the hand, across it toward the thumb and toward the palm, and its length.

    `knuckles` run index to little. The same frame the reference hand was measured in.
    """
    knuckles = np.asarray(knuckles)
    along = knuckles.mean(0) - wrist
    length = float(np.linalg.norm(along))
    along /= length
    across = knuckles[0] - knuckles[-1]
    across -= along * (across @ along)
    across /= np.linalg.norm(across)
    palm = (1 if side == "left" else -1) * np.cross(along, across)
    return {"along": along, "across": across, "palm": palm, "length": length}


def _along_chain(coords, points, behind=False):
    """Each vertex's distance along a guide chain, on the segment nearest to it.

    With `behind`, a vertex behind the chain's base reads negative instead of 0.
    """
    points = np.asarray(points)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    starts = np.r_[0, np.cumsum(lengths)]
    best, along = np.full(len(coords), np.inf), np.zeros(len(coords))
    for k, (start, a, b) in enumerate(zip(starts, points, points[1:])):
        t = np.clip((coords - a) @ (b - a) / ((b - a) @ (b - a)), -np.inf if behind and k == 0 else 0, 1)
        distance = np.linalg.norm(coords - (a + t[:, None] * (b - a)), axis=1)
        closer = distance < best
        best[closer], along[closer] = distance[closer], start + t[closer] * lengths[k]
    return along, best


def _at(line, arc, s):
    """The point `s` along a polyline measured by `arc`."""
    s = min(max(s, 0.0), arc[-1])
    k = min(int(np.searchsorted(arc, s, side="right") - 1), len(line) - 2)
    t = (s - arc[k]) / max(arc[k + 1] - arc[k], 1e-9)
    return line[k] + t * (line[k + 1] - line[k])


def guides(mesh_obj, armature_obj, side, verts=None, piece=None):
    """Guide data for `hand_setup.apply` (armature space), plus a report per finger."""
    verts = meshdata.world_vertices(mesh_obj) if verts is None else verts
    piece = hand_piece(mesh_obj, side, verts) if piece is None else piece
    reps, adj, wrist, depth, weld = _surface(mesh_obj, armature_obj, side, verts, piece)
    count = len(verts)
    from_wrist = _dijkstra(adj, reps[depth[reps] < depth[reps].min() + .01], count)
    reached = reps[np.isfinite(from_wrist[reps])]
    span = float(from_wrist[reached].max())
    # image-to-3D hands are coarse: a ring of surface needs a band wider than one edge
    band = max(.002, .6 * float(np.median([w for u in adj for _, w in adj[u]])))
    tips = _tips(verts, adj, reached, from_wrist, span, band, count)

    walks = np.array([_dijkstra(adj, [t], count) for t in tips])
    owner = np.argmin(walks, axis=0)
    chains = []
    for k, tip in enumerate(tips):
        members = reached[(owner[reached] == k) & np.isfinite(walks[k][reached])]
        line, radius = _chain(verts, members, walks[k], tip, span, band)
        if len(line) < 3:
            raise ValueError("a digit is too short to lay three bones along")
        surface = np.zeros(count, bool)
        surface[members] = True
        chains.append({"line": line, "radius": radius, "surface": surface[weld]})
    bone = armature_obj.data.bones[BONE_NAMES[f"{side}_hand"]]
    along = np.array(armature_obj.matrix_world @ bone.tail_local) - wrist
    along /= np.linalg.norm(along)
    # the thumb leaves the hand nearest the wrist; the other four leave at the knuckle row
    thumb = chains.pop(int(np.argmin([(c["line"][0] - wrist) @ along for c in chains])))
    webs = np.array([c["line"][0] for c in chains])
    across = thumb["line"][0] - webs.mean(0)
    across -= along * (across @ along)
    across /= np.linalg.norm(across)
    chains.sort(key=lambda c: -(c["line"][0] @ across))  # index first, from the thumb side
    palm = (1 if side == "left" else -1) * np.cross(along, across)
    # knuckles sit back from the webs along the hand itself: a curled finger's own direction at
    # the web points out through the back of the hand
    hand_axis = webs.mean(0) - wrist
    hand_axis /= np.linalg.norm(hand_axis)
    for chain in chains:
        visible = float(np.linalg.norm(np.diff(chain["line"], axis=0), axis=1).sum())
        chain["knuckle"] = chain["line"][0] - hand_axis * WEB_AT * visible / (1 - WEB_AT)
    frame = _hand_frame(wrist, [c["knuckle"] for c in chains], side)

    to_rig = np.array(armature_obj.matrix_world.inverted())
    local = lambda p: (to_rig[:3, :3] @ p + to_rig[:3, 3]).tolist()
    data, report = {}, {}
    for name, chain in zip(fingers.FINGERS, [thumb] + chains):
        line = chain["line"]
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
        visible = float(arc[-1])
        if name == "thumb":
            tangent = line[min(3, len(line) - 1)] - line[0]
            mcp = line[0] - tangent / np.linalg.norm(tangent) * .1 * visible
            cmc = wrist + frame["length"] * sum(k * frame[a] for k, a in zip(THUMB_ROOT, ("along", "across", "palm")))
            points = [cmc, mcp, _at(line, arc, THUMB_IP * visible), line[-1]]
            toward = (webs.mean(0) + wrist) / 2 - mcp
            bend_dir = palm + toward / np.linalg.norm(toward)
            bend = mcp + bend_dir / np.linalg.norm(bend_dir) * .03
            web = float(np.linalg.norm(mcp - cmc) + np.linalg.norm(line[0] - mcp))
        else:
            total = visible / (1 - WEB_AT)
            knuckle = chain["knuckle"]
            full, farc = np.vstack([knuckle, line]), np.r_[0, arc + WEB_AT * total]
            points = [knuckle] + [_at(full, farc, j * total) for j in JOINTS_AT] + [line[-1]]
            bend = knuckle + palm * .03
            web = WEB_AT * total
        chain["web"] = web  # how far along its chain, from the base, the finger leaves the hand
        data[name] = {"points": [local(p) for p in points], "bend": local(bend),
                      "radius": max(chain["radius"] * REACH, .004)}
        report[name] = {"visible_cm": round(visible * 100, 1), "radius_cm": round(chain["radius"] * 100, 2),
                        "web_depth_cm": round(float((line[0] - wrist) @ along) * 100, 1)}
    surfaces = {name: (chain["surface"], chain["web"]) for name, chain in zip(fingers.FINGERS, [thumb] + chains)}
    return data, report, surfaces


def claim_hand_weight(mesh_obj, armature_obj, side, data, piece):
    """Inside a finger's tube on the hand piece, forearm or body weight passes to the hand bone.

    `hand_setup.apply` only redistributes what the hand already holds. A finger still partly on
    the forearm keeps that part when the finger curls, and stretches.
    """
    names = hand_setup._names(side)
    hand = mesh_obj.vertex_groups[names[0]]
    ours = {mesh_obj.vertex_groups[n].index for n in names if n in mesh_obj.vertex_groups}
    matrix = armature_obj.matrix_world.inverted() @ mesh_obj.matrix_world
    coords = np.array([matrix @ v.co for v in mesh_obj.data.vertices])
    inside = np.max([hand_setup._influences(coords, data[f])[0] for f in fingers.FINGERS], axis=0)
    changed = 0
    for index in np.flatnonzero((inside > .5) & piece):
        vertex = mesh_obj.data.vertices[int(index)]
        own = sum(g.weight for g in vertex.groups if g.group in ours)
        if own <= 0 or own > .999:
            continue
        held = next((g.weight for g in vertex.groups if g.group == hand.index), 0.0)
        for entry in list(vertex.groups):
            bone = armature_obj.data.bones.get(mesh_obj.vertex_groups[entry.group].name)
            if entry.group not in ours and bone is not None and bone.use_deform:
                mesh_obj.vertex_groups[entry.group].remove([int(index)])
        hand.add([int(index)], held + 1 - own, "REPLACE")
        changed += 1
    return changed


def fold_back(mesh_obj, side, piece):
    """Finger weight that landed off the hand piece (a cuff inside the thumb's tube) goes to the hand."""
    names = hand_setup._names(side)
    hand = mesh_obj.vertex_groups[names[0]]
    finger_groups = [mesh_obj.vertex_groups[n] for n in names[1:] if n in mesh_obj.vertex_groups]
    indices = {g.index for g in finger_groups}
    moved = 0
    for index in np.flatnonzero(~piece):
        vertex = mesh_obj.data.vertices[int(index)]
        extra = sum(g.weight for g in vertex.groups if g.group in indices)
        if extra <= 0:
            continue
        held = next((g.weight for g in vertex.groups if g.group == hand.index), 0.0)
        for group in finger_groups:
            group.remove([int(index)])
        hand.add([int(index)], held + extra, "REPLACE")
        moved += 1
    return moved


def own_finger_surface(mesh_obj, armature_obj, side, data, surfaces):
    """Past its web a finger is all finger: hand weight left there moves onto its bones.

    The guide tubes fade at their edge, and a bulging joint on a coarse mesh pokes out of the
    tube; on a test hand that left a third of a knuckle on the hand bone, and it stretched 2 cm
    when the finger curled. The finger's share fades in past the web, measured along its bones:
    taking it all at once there made a hard seam that stretched 3 cm, and surface distance from
    the tip puts the web at different depths on the two sides of a curled finger.
    """
    names = hand_setup._names(side)
    hand = mesh_obj.vertex_groups[names[0]]
    matrix = armature_obj.matrix_world.inverted() @ mesh_obj.matrix_world
    coords = np.array([matrix @ v.co for v in mesh_obj.data.vertices])
    moved = 0
    for finger, (members, web) in surfaces.items():
        along, _ = _along_chain(coords, data[finger]["points"])
        surface = np.where(members, np.clip((along - web) / OWN_FADE, 0.0, 1.0), 0.0)
        groups = [mesh_obj.vertex_groups.get(fingers.bone_name(side, finger, s)) for s in fingers._segments(finger)]
        indices = {g.index: g for g in groups if g is not None}
        for index in np.flatnonzero(surface > 0):
            vertex = mesh_obj.data.vertices[int(index)]
            held = next((g.weight for g in vertex.groups if g.group == hand.index), 0.0)
            own = {g.group: g.weight for g in vertex.groups if g.group in indices and g.weight > 0}
            total = sum(own.values())
            if held <= 0 or total <= 0:
                continue
            passed = held * float(surface[index])
            for group, weight in own.items():
                indices[group].add([int(index)], weight * (total + passed) / total, "REPLACE")
            hand.add([int(index)], held - passed, "REPLACE")
            moved += 1
    return moved


def smooth(mesh_obj, side, piece, passes=None):
    """Average the hand's weights over its surface, keeping each vertex's hand share.

    The guide tubes end sharply at webs and knuckles; on model A that stretched edges 2.4 cm with
    the fingers curled 35 degrees, where the smoothed weights of `fingers.build_fingers` gave
    0.5 cm. Neighbouring fused fingers blend a little across their web, which keeps it whole.
    """
    passes = SMOOTHING if passes is None else passes
    names = hand_setup._names(side)
    groups = [mesh_obj.vertex_groups[n] for n in names if n in mesh_obj.vertex_groups]
    count = len(mesh_obj.data.vertices)
    weights = np.zeros((len(groups), count))
    row = {g.index: k for k, g in enumerate(groups)}
    for vertex in mesh_obj.data.vertices:
        for entry in vertex.groups:
            if entry.group in row:
                weights[row[entry.group], vertex.index] = entry.weight
    budget = weights.sum(axis=0)
    region = piece & (budget > 1e-6)
    weld, edge_a, edge_b = meshdata.welded_edges(mesh_obj, meshdata.world_vertices(mesh_obj))
    inside = region[edge_a] & region[edge_b] & (edge_a != edge_b)
    edge_a, edge_b = edge_a[inside], edge_b[inside]
    # shares of the hand budget, so smoothing never moves weight onto or off the forearm
    shares = np.where(budget > 1e-6, weights / np.maximum(budget, 1e-12), 0.0)[:, weld]
    neighbours = np.bincount(np.r_[edge_a, edge_b], minlength=count)
    for _ in range(passes):
        summed = shares.copy()
        for k in range(len(groups)):
            np.add.at(summed[k], edge_a, shares[k][edge_b])
            np.add.at(summed[k], edge_b, shares[k][edge_a])
        shares = np.where(neighbours > 0, summed / (neighbours + 1), shares)
    shares = shares[:, weld]
    shares /= np.maximum(shares.sum(axis=0), 1e-12)
    for k, group in enumerate(groups):
        for index in np.flatnonzero(region):
            value = float(budget[index] * shares[k, index])
            if value > 1e-5:
                group.add([int(index)], value, "REPLACE")
            else:
                group.remove([int(index)])
    return int(region.sum())


def reference_weights(mesh_obj, armature_obj, side, data, piece):
    """Reshape the averaged hand weights the way the reference VRoid hand has them.

    `smooth` blends every group over the surface. That is right between a finger and the palm and
    between neighbouring fingers, but it also blurred each finger's joints over about 2 cm (8% of
    Model B's finger vertices sat wholly on one bone, 53% on the reference) and carried finger
    weight a quarter of the hand back into the palm. Here each finger keeps its averaged total,
    split among its bones by where the vertex lies along the chain (the joint bands of
    `hand_setup._influences`); finger weight ends at the knuckle; and the thumb's first bone takes
    the ball of the thumb from the hand bone. Reads the modelled pose the guides were placed in;
    weight only moves between this hand's bones, never onto or off the forearm.
    """
    names = hand_setup._names(side)
    groups = [mesh_obj.vertex_groups.get(n) or mesh_obj.vertex_groups.new(name=n) for n in names]
    count = len(mesh_obj.data.vertices)
    weights = np.zeros((len(groups), count))
    row = {g.index: k for k, g in enumerate(groups)}
    for vertex in mesh_obj.data.vertices:
        for entry in vertex.groups:
            if entry.group in row:
                weights[row[entry.group], vertex.index] = entry.weight
    region = piece & (weights.sum(axis=0) > 1e-6)
    matrix = armature_obj.matrix_world.inverted() @ mesh_obj.matrix_world
    coords = np.array([matrix @ v.co for v in mesh_obj.data.vertices])
    wrist = np.array(armature_obj.data.bones[names[0]].head_local)
    frame = _hand_frame(wrist, [data[f]["points"][0] for f in fingers.FINGERS[1:]], side)
    length = frame["length"]
    for k, finger in enumerate(fingers.FINGERS):
        rows = slice(1 + 3 * k, 4 + 3 * k)
        total = weights[rows].sum(axis=0)
        weights[rows] = total * hand_setup._influences(coords, data[finger])[1]
        if finger != "thumb":
            points = np.array(data[finger]["points"])
            along, _ = _along_chain(coords, points, behind=True)
            # a curled finger's first bone points down into the palm, so the palm under the
            # knuckle reads as ahead of it along the bone: measure along the hand there too
            first = along < np.linalg.norm(points[1] - points[0])
            behind = np.where(first, np.minimum(along, (coords - points[0]) @ frame["along"]), along)
            keep = np.clip(1 + behind / (KNUCKLE_FADE * length), 0, 1)
            weights[0] += weights[rows].sum(axis=0) * (1 - keep)
            weights[rows] *= keep
    cmc, mcp = (np.array(p) for p in data["thumb"]["points"][:2])
    _, distance = _along_chain(coords, [cmc, mcp])
    near, far = (r * length for r in THENAR)
    ball = np.clip((far - distance) / (far - near), 0, 1)
    # not the back of the hand: on the reference it stays with the hand bone
    ball *= np.clip(1 + ((coords - wrist) @ frame["palm"]) / (.1 * length), 0, 1)
    # nor the wrist: model A's glove ring there turned with the thumb; nor the middle of the palm,
    # which on the reference stays with the hand bone up to a quarter of the hand from its centre
    ball *= np.clip(((coords - wrist) @ frame["along"]) / (BALL_FROM * length), 0, 1)
    ball *= np.clip(((coords - wrist) @ frame["across"] / length - BALL_ACROSS[0]) / (BALL_ACROSS[1] - BALL_ACROSS[0]), 0, 1)
    # a fade shorter than one edge of a coarse mesh folds the palm when the thumb turns
    # (model B's palm notched at the thumb's base): soften it over the surface
    weld, edge_a, edge_b = meshdata.welded_edges(mesh_obj, meshdata.world_vertices(mesh_obj))
    inside = region[edge_a] & region[edge_b] & (edge_a != edge_b)
    edge_a, edge_b = edge_a[inside], edge_b[inside]
    ball = ball[weld]
    neighbours = np.bincount(np.r_[edge_a, edge_b], minlength=count)
    for _ in range(BALL_SMOOTHING):
        summed = ball.copy()
        np.add.at(summed, edge_a, ball[edge_b])
        np.add.at(summed, edge_b, ball[edge_a])
        ball = np.where(neighbours > 0, summed / (neighbours + 1), ball)
    ball = ball[weld]
    taken = weights[0] * ball
    weights[0] -= taken
    weights[1] += taken
    for k, group in enumerate(groups):
        for index in np.flatnonzero(region):
            value = float(weights[k, index])
            if value > 1e-5:
                group.add([int(index)], value, "REPLACE")
            else:
                group.remove([int(index)])
    return {"vertices": int(region.sum()), "thumb_ball": int((taken[region] > .3).sum())}


def rebuild(mesh_obj, armature_obj, sides=("left", "right")):
    """Find, place and apply finger guides on both hands. Returns a report per side.

    Leaves the guides in the file as Guided Hand Setup empties, so they can be adjusted and
    applied again by hand. Touches only this hand's weights, plus forearm weight inside a finger.
    """
    verts = meshdata.world_vertices(mesh_obj)
    result = {}
    for side in sides:
        piece = hand_piece(mesh_obj, side, verts)
        data, report, surfaces = guides(mesh_obj, armature_obj, side, verts, piece)
        claimed = claim_hand_weight(mesh_obj, armature_obj, side, data, piece)
        hand_setup.write_guides(armature_obj, side, data)
        applied = hand_setup.apply(mesh_obj, armature_obj, side, data)
        folded = fold_back(mesh_obj, side, piece)
        owned = own_finger_surface(mesh_obj, armature_obj, side, data, surfaces)
        smoothed = smooth(mesh_obj, side, piece)
        reshaped = reference_weights(mesh_obj, armature_obj, side, data, piece)
        result[side] = {"fingers": report, "claimed_from_arm": claimed, "weighted": applied,
                        "folded_off_hand": folded, "hand_weight_moved_onto_fingers": owned,
                        "smoothed": smoothed, "reshaped": reshaped}
    return result
