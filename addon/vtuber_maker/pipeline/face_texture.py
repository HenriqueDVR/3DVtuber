"""Clean-ups of a face patch's texture (see scripts/retexture/face_patch.py).

A face patch is a sharper texture for the front of the face, mapped straight from the front.
It is drawn separately from the model's own texture, so its colours don't quite meet the rest
of the head: `blend_patch_seam` matches the rest of the head to it. Generators
also paint whatever sits over a head (a hat brim's shadow) onto its top: `clean_head_top`
paints those dark smudges out.
"""

import numpy as np

from . import face_points, meshdata, texture_space

# The rest of the head is matched to the patch across its edge within this share of the
# head's size, fading out beyond; the change also bleeds this many texels past the surface.
SEAM_BLEND = 0.35
SEAM_BLEED = 3
# With extend_patch, head polygons facing the front more than this (cosine) take the face patch.
EXTEND_FACING = 0.55
# ... unless their part of the image is darker than this share of the typical head polygon's.
EXTEND_DARK = 0.75
# Sampling boxes (pixels) either side of the seam.
SEAM_SAMPLE = 3
# Smooth mode: the colour field comes from patch points within SMOOTH_EDGE (share of the head's
# size) of its edge and no darker than SMOOTH_DARK of its typical brightness (not the eyes),
# blended by inverse distance to the SMOOTH_POWER, so it meets the patch without a step.
SMOOTH_EDGE = 0.03
SMOOTH_DARK = 0.8
SMOOTH_POWER = 4
# Across the seam, colours further apart than this (0-1 RGB distance) belong to something
# else meeting the face (a collar): they're left out.
SEAM_MAX_DIFFERENCE = 0.35

# Head top: surface facing up more than this (normal z) is cleaned; texels darker than
# TOP_DARK of the head's typical brightness there are repainted.
TOP_FACING = 0.25
TOP_DARK = 0.7
TOP_GROW = 2



def extend_patch(mesh_obj, head, min_facing=EXTEND_FACING):
    """Put every front-facing polygon of the `head` vertices on the face patch; returns how many moved.

    The patch image is a straight-on view of the whole head, mapped from the front, but the patch
    step only textures part of it: on a coarse mesh that leaves the model's own texture on whole
    polygons inside the face, and their straight edges show. Polygons facing the front more than
    `min_facing` (cosine) get the patch with the same front projection instead, unless the image
    is dark there (see EXTEND_DARK).
    """
    mesh = mesh_obj.data
    slot = mesh.materials.find(face_points.FACE_PATCH_MATERIAL)
    if slot < 0:
        return 0
    to_uv = face_points._world_to_uv(mesh_obj)
    pixels = texture_space.image_rgb(face_points._patch_image(mesh_obj))
    head = np.asarray(head, dtype=bool)
    world = np.array(mesh_obj.matrix_world)
    rotation = np.array(mesh_obj.matrix_world.to_3x3())
    uv = mesh.uv_layers.active.data
    moved = 0
    candidates = []
    over = face_points.over_slots(mesh)
    for poly in mesh.polygons:
        if poly.material_index == slot or poly.material_index in over or not head[list(poly.vertices)].all():
            continue
        facing = -(rotation @ np.array(poly.normal))[1] / max(np.linalg.norm(rotation @ np.array(poly.normal)), 1e-9)
        if facing < min_facing:
            continue
        corners = [world[:3, :3] @ np.array(mesh.vertices[mesh.loops[li].vertex_index].co) + world[:3, 3]
                   for li in poly.loop_indices]
        corner_uvs = [to_uv(co[0], co[2]) for co in corners]
        candidates.append((poly, corner_uvs))
    # the image shows whatever is in front of the head there too (a collar's edge under the chin):
    # polygons whose share of it is much darker than the head's colour keep the smooth fill instead
    brightness = [min(_sample(pixels, *p).max() for p in _spread(uvs)) for _poly, uvs in candidates]
    typical = np.median(brightness) if brightness else 0.0
    for (poly, corner_uvs), light in zip(candidates, brightness):
        if light < EXTEND_DARK * typical:
            continue
        poly.material_index = slot
        for li, value in zip(poly.loop_indices, corner_uvs):
            uv[li].uv = value
        moved += 1
    mesh.update()
    return moved


def blend_patch_seam(mesh_obj, head, smooth=False):
    """Match the texture on the rest of the `head` vertices' surface to the face patch across its edge.

    The patch is left as drawn. The rest of the head is shifted by the colour difference at the
    nearest stretch of the patch's edge, fading out over SEAM_BLEND of the head's size, so its
    own detail (ears, a painted neck) stays. With `smooth` (a solid head like a mascot's, with no
    detail of its own) it becomes a smooth continuation of the patch's edge colours everywhere,
    which also clears stripes and smudges the generator painted on it. Returns the texels changed.
    """
    mesh = mesh_obj.data
    slot = mesh.materials.find(face_points.FACE_PATCH_MATERIAL)
    patch_image = face_points._patch_image(mesh_obj)
    if slot < 0 or patch_image is None:
        return 0
    head = np.asarray(head, dtype=bool)
    verts = meshdata.world_vertices(mesh_obj)
    weld, _a, _b = meshdata.welded_edges(mesh_obj, verts)
    uv = mesh.uv_layers.active.data
    inside, outside = {}, {}
    rest = []
    for poly in mesh.polygons:
        if poly.material_index == slot:
            for li in poly.loop_indices:
                inside.setdefault(int(weld[mesh.loops[li].vertex_index]), []).append(tuple(uv[li].uv))
            continue
        if not head[list(poly.vertices)].all():
            continue
        image = texture_space.material_base_color_image(mesh.materials[poly.material_index])
        if image is None:
            continue
        rest.append((poly, image))
        for li in poly.loop_indices:
            outside.setdefault(int(weld[mesh.loops[li].vertex_index]), []).append((image, tuple(uv[li].uv)))
    border = inside.keys() & outside.keys()
    seam = _outer_loop(mesh, weld, border, verts)
    print("SMOOTHDBG border", len(border), "loop", len(seam))
    if not seam:
        return 0
    rest = _outside_patch(rest, weld, seam)
    patch_pixels = texture_space.image_rgb(patch_image)
    cache = {}
    points, patch_colours, deltas = [], [], []
    for key in seam:
        here = np.mean([_sample(patch_pixels, *p) for p in inside[key]], axis=0)
        there = np.mean([_sample(cache.setdefault(img.name, texture_space.image_rgb(img)), *p)
                         for img, p in outside[key]], axis=0)
        points.append(verts[key])
        patch_colours.append(here)
        if np.linalg.norm(there - here) <= SEAM_MAX_DIFFERENCE:
            deltas.append((len(points) - 1, here - there))
    points, patch_colours = np.array(points), np.array(patch_colours)
    head_size = float(np.ptp(verts[head], axis=0).max())
    # per head vertex off the patch: its correction, blended from the seam by inverse distance
    targets = np.nonzero(head)[0]
    distance = np.linalg.norm(verts[targets][:, None, :] - points[None, :, :], axis=2)
    weight = 1.0 / np.maximum(distance, 1e-4) ** 2
    per_vertex = np.zeros((len(verts), 3))
    if smooth:
        return _smooth_head(mesh, patch_pixels, verts, weld, head, inside, rest, points, head_size)
    if deltas:
        index = np.array([i for i, _ in deltas])
        diff = np.array([d for _, d in deltas])
        w = weight[:, index]
        fade = np.clip(1.0 - distance[:, index].min(axis=1) / (SEAM_BLEND * head_size), 0.0, 1.0)
        per_vertex[targets] = (w @ diff) / w.sum(axis=1)[:, None] * (fade * fade * (3 - 2 * fade))[:, None]
    else:
        return 0
    per_vertex = per_vertex[weld]  # UV-seam copies share their correction
    changed = 0
    by_image = {}
    for poly, image in rest:
        by_image.setdefault(image.name, (image, []))[1].append(poly)
    for image, polys in by_image.values():
        width, height = image.size
        rgba = np.array(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
        values, covered = _rasterise(mesh, polys, per_vertex, width, height)
        values, covered = _bleed(values, covered, SEAM_BLEED)
        rgba[covered, :3] = np.clip(rgba[covered, :3] + values[covered], 0.0, 1.0)
        _store(image, rgba)
        changed += int(covered.sum())
    return changed


def _outside_patch(rest, weld, seam):
    """The (polygon, image) pairs reachable from the patch's outer edge without crossing the patch.

    Surface enclosed by the patch (eye sockets painted in the model's own texture) is left out.
    """
    by_vertex = {}
    for index, (poly, _image) in enumerate(rest):
        for v in poly.vertices:
            by_vertex.setdefault(int(weld[v]), []).append(index)
    reached = set()
    frontier = [i for v in seam for i in by_vertex.get(v, [])]
    while frontier:
        index = frontier.pop()
        if index in reached:
            continue
        reached.add(index)
        for v in rest[index][0].vertices:
            frontier.extend(i for i in by_vertex[int(weld[v])] if i not in reached)
    return [rest[i] for i in sorted(reached)]


def _outer_loop(mesh, weld, border, verts):
    """The border vertices on the patch's outer edge: the widest connected run of them.

    The patch also borders holes inside it (eye sockets holding separate eyes); those aren't seams.
    """
    border = set(border)
    parent = {v: v for v in border}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for edge in mesh.edges:
        a, b = int(weld[edge.vertices[0]]), int(weld[edge.vertices[1]])
        if a in border and b in border:
            parent[find(a)] = find(b)
    groups = {}
    for v in border:
        groups.setdefault(find(v), []).append(v)
    return sorted(max(groups.values(), key=lambda g: float(np.ptp(verts[g], axis=0).max()))) if groups else []


def _smooth_head(mesh, patch_pixels, verts, weld, head, inside, rest, seam_points, head_size):
    """blend_patch_seam's `smooth` mode: the head off the patch takes a smooth colour field spread
    from the patch's colours along its edge (not the eyes)."""
    keys = np.array(sorted(inside))
    colours = np.array([np.mean([_sample(patch_pixels, *p) for p in inside[k]], axis=0) for k in keys])
    to_seam = np.linalg.norm(verts[keys][:, None, :] - seam_points[None, :, :], axis=2).min(axis=1)
    brightness = colours.max(axis=1)
    sources = (to_seam <= SMOOTH_EDGE * head_size) & (brightness > SMOOTH_DARK * np.median(brightness))
    if sources.sum() < 3:
        return 0
    src_points, src_colours = verts[keys[sources]], colours[sources]
    targets = np.nonzero(head)[0]
    distance = np.linalg.norm(verts[targets][:, None, :] - src_points[None, :, :], axis=2)
    weight = 1.0 / np.maximum(distance, 1e-4) ** SMOOTH_POWER
    field = np.zeros((len(verts), 3))
    field[targets] = (weight @ src_colours) / weight.sum(axis=1)[:, None]
    field = field[weld]
    changed = 0
    by_image = {}
    for poly, image in rest:
        by_image.setdefault(image.name, (image, []))[1].append(poly)
    for image, polys in by_image.values():
        width, height = image.size
        rgba = np.array(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
        values, covered = _rasterise(mesh, polys, field, width, height)
        values, covered = _bleed(values, covered, SEAM_BLEED)
        rgba[covered, :3] = values[covered]
        _store(image, rgba)
        changed += int(covered.sum())
    return changed


def clean_head_top(mesh_obj, head):
    """Repaint dark texels on the up-facing, non-patch surface of the `head` vertices. Returns the texels changed."""
    mesh = mesh_obj.data
    slot = mesh.materials.find(face_points.FACE_PATCH_MATERIAL)
    normals = np.array([v.normal for v in mesh.vertices])
    world = np.array(mesh_obj.matrix_world.to_3x3())
    up = (normals @ world.T)[:, 2]
    head = np.asarray(head, dtype=bool)
    changed = 0
    by_image = {}
    for poly in mesh.polygons:
        if poly.material_index == slot or not head[list(poly.vertices)].all():
            continue
        image = texture_space.material_base_color_image(mesh.materials[poly.material_index])
        if image is not None:
            by_image.setdefault(image.name, (image, []))[1].append(poly)
    for image, polys in by_image.values():
        width, height = image.size
        rgba = np.array(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
        facing, covered = _rasterise(mesh, polys, up, width, height)
        top = covered & (facing > TOP_FACING)
        if top.sum() < 20:
            continue
        value = rgba[..., :3].max(axis=2)
        typical = np.median(value[top])
        dark = top & (value < TOP_DARK * typical)
        dark = _grow(dark, TOP_GROW) & covered
        if not dark.any():
            continue
        good = top & ~dark
        fill = np.median(rgba[good][:, :3], axis=0) if good.any() else rgba[top][:, :3].mean(axis=0)
        rgba[dark, :3] = fill
        _store(image, rgba)
        changed += int(dark.sum())
    return changed


def _spread(uvs, steps=4):
    """Points spread over a polygon's UV area: corners, edge points and points toward the middle."""
    uvs = np.asarray(uvs, dtype=float)
    middle = uvs.mean(axis=0)
    points = [middle]
    for k in range(len(uvs)):
        a, b = uvs[k], uvs[(k + 1) % len(uvs)]
        for t in np.linspace(0.0, 1.0, steps, endpoint=False):
            edge = a + t * (b - a)
            points += [edge, (edge + middle) / 2]
    return [tuple(p) for p in points]


def _sample(pixels, u, v):
    height, width = pixels.shape[:2]
    x, y = int(u % 1.0 * width), int(v % 1.0 * height)
    box = pixels[max(y - SEAM_SAMPLE, 0):y + SEAM_SAMPLE + 1, max(x - SEAM_SAMPLE, 0):x + SEAM_SAMPLE + 1]
    return box.reshape(-1, 3).mean(axis=0)


def _rasterise(mesh, polys, per_vertex, width, height):
    """Per texel: per_vertex (a value or a vector per vertex) over the polygons' UVs, and which texels they cover."""
    uv = mesh.uv_layers.active.data
    per_vertex = np.asarray(per_vertex, dtype=np.float32)
    vector = per_vertex.ndim == 2
    if vector:
        value = np.zeros((height, width, per_vertex.shape[1]), np.float32)
    else:
        value = np.full((height, width), -np.inf, np.float32)
    covered = np.zeros((height, width), bool)
    for poly in polys:
        loops = list(poly.loop_indices)
        for k in range(1, len(loops) - 1):
            tri = (loops[0], loops[k], loops[k + 1])
            pts = np.array([[uv[l].uv.x % 1.0 * width, uv[l].uv.y % 1.0 * height] for l in tri])
            vals = [per_vertex[mesh.loops[l].vertex_index] for l in tri]
            (x0, y0), (x1, y1) = np.floor(pts.min(0)).astype(int), np.ceil(pts.max(0)).astype(int) + 1
            xs, ys = np.meshgrid(np.arange(max(x0, 0), min(x1, width)), np.arange(max(y0, 0), min(y1, height)))
            (ax, ay), (bx, by), (cx, cy) = pts
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-12:
                continue
            l1 = ((by - cy) * (xs + 0.5 - cx) + (cx - bx) * (ys + 0.5 - cy)) / den
            l2 = ((cy - ay) * (xs + 0.5 - cx) + (ax - cx) * (ys + 0.5 - cy)) / den
            l3 = 1 - l1 - l2
            inside = (l1 > -0.02) & (l2 > -0.02) & (l3 > -0.02)
            rows, cols = ys[inside], xs[inside]
            if vector:
                value[rows, cols] = l1[inside, None] * vals[0] + l2[inside, None] * vals[1] + l3[inside, None] * vals[2]
            else:
                value[rows, cols] = np.maximum(value[rows, cols], (l1 * vals[0] + l2 * vals[1] + l3 * vals[2])[inside])
            covered[rows, cols] = True
    return value, covered


def _bleed(values, covered, radius):
    """Spread covered texels' values a few texels out, so filtering at UV island edges doesn't show old colours."""
    values, covered = values.copy(), covered.copy()
    for _ in range(radius):
        total = np.zeros_like(values)
        count = np.zeros(covered.shape, np.float32)
        for axis, step in ((0, 1), (0, -1), (1, 1), (1, -1)):
            total += np.roll(values * covered[..., None], step, axis=axis)
            count += np.roll(covered, step, axis=axis)
        grow = ~covered & (count > 0)
        values[grow] = total[grow] / count[grow][:, None]
        covered |= grow
    return values, covered


def _grow(mask, radius):
    out = mask.copy()
    for step in range(1, radius + 1):
        for axis in (0, 1):
            out |= np.roll(mask, step, axis=axis) | np.roll(mask, -step, axis=axis)
    return out


def _store(image, rgba):
    image.pixels.foreach_set(np.asarray(rgba, dtype=np.float32).ravel())
    image.update()
    image.pack()
