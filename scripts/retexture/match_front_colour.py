"""Tint a part's sides and back to match the colour its front has in the texture.

    blender -b -P scripts/retexture/match_front_colour.py -- in.fbx|in.glb out.glb --at X,Y,Z [--underside] [--close]

Tripo paints what the front view shows and guesses the rest, so a coloured part (a mascot's
head, dyed skin) can fade to near white round the sides and back. This finds the separate
piece of the mesh nearest the point `--at` (in the model's own space, as imported), measures the
hue, saturation and brightness of its front-facing texture, and shifts the rest of the piece
towards them. The shift fades in as the surface turns away from the front, so the front and the
face stay as painted, and it keeps the shading: each texel moves by the difference between the
front's and the sides' typical colour. Dark texels (lines, eyes) are left alone.

`--underside` also repaints the part's downward-facing surface with the front's colour, dark
texels included: Tripo paints whatever sits under a part (a collar under a round head) onto it.
`--close` caps holes in the lower part of the piece (a round head left open at the bottom)
with faces in that colour, so tilting the head doesn't show its inside.

Tripo's rigged FBX goes in as it is; the GLB names it as its rig (see matte_materials.py).
"""
import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "addon"))

from vtuber_maker.pipeline import meshdata, normalize, texture_space  # noqa: E402

# Facing is the cosine between a vertex normal and the front (-Y): at or above FRONT the texel
# keeps its colour; at or below SIDE it gets the full shift.
FRONT, SIDE = 0.85, 0.4
# Texels darker than this are lines and eyes, not the part's colour.
MIN_VALUE = 0.5
# The front's typical colour comes from texels at least this saturated (not highlights).
MIN_FRONT_SATURATION = 0.15
# Texels round each triangle get the shift too, so UV seams don't show (pixels).
BLEED = 3
# With --underside, surface facing further down than the first value (cosine with straight
# down) starts being repainted, fully past the second.
DOWN = (0.2, 0.5)
# With --close, holes whose centre is in this lowest fraction of the piece's height are capped.
CLOSE_BELOW = 0.35


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("out")
    parser.add_argument("--at", required=True, help="a point on the part, X,Y,Z in the model's space")
    parser.add_argument("--underside", action="store_true", help="repaint the downward-facing surface too")
    parser.add_argument("--close", action="store_true", help="cap holes low on the part")
    args = parser.parse_args(argv)
    at = Vector([float(c) for c in args.at.split(",")])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    rig = normalize.import_unrigged(Path(args.src))
    mesh_obj = next(o for o in bpy.context.scene.objects if o.type == "MESH")
    image = texture_space.material_base_color_image(mesh_obj.data.materials[0])
    width, height = image.size

    if args.close:
        print(f"capped {_close_low_holes(mesh_obj, at)} holes")
    verts = meshdata.world_vertices(mesh_obj)
    labels = meshdata.pieces(mesh_obj, verts)
    part = labels == labels[int(np.argmin(np.linalg.norm(verts - np.array(at), axis=1)))]
    normals = np.array([mesh_obj.matrix_world.to_3x3() @ v.normal for v in mesh_obj.data.vertices])
    # vertices split along UV seams each see only their own faces: average them back together
    weld, _a, _b = meshdata.welded_edges(mesh_obj, verts)
    summed = np.zeros_like(normals)
    np.add.at(summed, weld, normals)
    normals = summed[weld]
    unit = normals / np.maximum(np.linalg.norm(normals, axis=1), 1e-9)[:, None]
    facing = -unit[:, 1]
    shift = np.clip((FRONT - facing) / (FRONT - SIDE), 0.0, 1.0)
    shift = shift * shift * (3 - 2 * shift)
    down = np.clip((-unit[:, 2] - DOWN[0]) / (DOWN[1] - DOWN[0]), 0.0, 1.0) if args.underside else np.zeros(len(verts))
    print(f"part: {int(part.sum())} vertices")

    weight, covered, front = _rasterise(mesh_obj, part, shift, facing, width, height)
    weight, covered = _grow(weight, BLEED), _grow(covered, BLEED) > 0
    under = _grow(_rasterise(mesh_obj, part, down, facing, width, height)[0], BLEED) if args.underside else None

    pixels = np.array(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    hsv = _rgb_to_hsv(pixels[..., :3])
    value, saturation = hsv[..., 2], hsv[..., 1]
    front_texels = front & (value > MIN_VALUE) & (saturation > MIN_FRONT_SATURATION)
    side_texels = covered & (weight > 0.9) & (value > MIN_VALUE)
    if front_texels.sum() < 50 or side_texels.sum() < 50:
        sys.exit("the part has too little front or side texture to compare")
    target, current = np.median(hsv[front_texels], axis=0), np.median(hsv[side_texels], axis=0)
    print(f"front hsv {target.round(3)}, sides hsv {current.round(3)}")

    w = np.where(covered & (value > MIN_VALUE), weight, 0.0)[..., None]
    moved = hsv + w * np.array([0.0, target[1] - current[1], target[2] - current[2]])
    moved[..., 0] = hsv[..., 0] + w[..., 0] * _hue_difference(hsv[..., 0], target[0])
    moved[..., 0] %= 1.0
    moved[..., 1:] = np.clip(moved[..., 1:], 0.0, 1.0)
    pixels[..., :3] = _hsv_to_rgb(moved)
    if under is not None:
        u = np.where(covered, under, 0.0)[..., None]
        pixels[..., :3] = (1 - u) * pixels[..., :3] + u * _hsv_to_rgb(target[None, None, :])
    image.pixels[:] = pixels.ravel()
    image.pack()
    print(f"changed {int((w[..., 0] > 0).sum())} texels")
    normalize.export_glb(Path(args.out), rig)
    print("wrote", args.out)


def _close_low_holes(mesh_obj, at):
    """Cap holes in the lower part of the piece nearest `at`; returns how many were capped.

    Vertices split along UV seams are welded first (seams aren't holes). Each cap takes the UV
    of the most downward-facing face, which --underside repaints.
    """
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.transform(mesh_obj.matrix_world)
    bm.verts.ensure_lookup_table()
    seed = min(bm.verts, key=lambda v: (v.co - at).length)
    part, stack = {seed}, [seed]
    while stack:
        v = stack.pop()
        for e in v.link_edges:
            w = e.other_vert(v)
            if w not in part:
                part.add(w)
                stack.append(w)
    # vertices at the same place on the other side of a seam are part of the piece too
    positions = {tuple(round(c, 6) for c in v.co) for v in part}
    part |= {v for v in bm.verts if tuple(round(c, 6) for c in v.co) in positions}
    bmesh.ops.remove_doubles(bm, verts=list(part), dist=1e-6)
    bm.verts.ensure_lookup_table()
    part = [v for v in bm.verts if v.is_valid and tuple(round(c, 6) for c in v.co) in positions]
    low, high = min(v.co.z for v in part), max(v.co.z for v in part)
    part_faces = {f for v in part for f in v.link_faces}
    bottom = min(part_faces, key=lambda f: f.normal.z)
    uv = bm.loops.layers.uv.active
    bottom_uv = sum((l[uv].uv for l in bottom.loops), Vector((0, 0))) / len(bottom.loops)
    boundary = [e for e in bm.edges if e.is_boundary and e.verts[0] in set(part)]
    capped = 0
    for loop_edges in _edge_loops(boundary):
        centre_z = sum(e.verts[0].co.z for e in loop_edges) / len(loop_edges)
        if centre_z > low + CLOSE_BELOW * (high - low):
            continue
        made = bmesh.ops.holes_fill(bm, edges=loop_edges, sides=0)["faces"]
        made = bmesh.ops.triangulate(bm, faces=made)["faces"]
        for f in made:
            f.material_index = bottom.material_index
            f.smooth = True
            for l in f.loops:
                l[uv].uv = bottom_uv
        capped += 1
    bm.normal_update()
    bm.transform(mesh_obj.matrix_world.inverted())
    bm.to_mesh(mesh_obj.data)
    bm.free()
    mesh_obj.data.update()
    return capped


def _edge_loops(edges):
    """Group boundary edges into connected loops."""
    remaining, loops = set(edges), []
    while remaining:
        start = remaining.pop()
        loop, stack = [start], [start]
        while stack:
            e = stack.pop()
            for v in e.verts:
                for other in v.link_edges:
                    if other in remaining:
                        remaining.discard(other)
                        loop.append(other)
                        stack.append(other)
        loops.append(loop)
    return loops


def _rasterise(mesh_obj, part, shift, facing, width, height):
    """Per texel: the part's shift (interpolated across triangles), coverage, and front-facing texels."""
    mesh = mesh_obj.data
    mesh.calc_loop_triangles()
    uv = mesh.uv_layers.active.data
    weight = np.zeros((height, width), np.float32)
    covered = np.zeros((height, width), np.float32)
    front = np.zeros((height, width), bool)
    for tri in mesh.loop_triangles:
        corners = list(tri.vertices)
        if not part[corners].all():
            continue
        points = np.array([[uv[l].uv.x % 1.0 * width, uv[l].uv.y % 1.0 * height] for l in tri.loops])
        values = shift[corners]
        is_front = facing[corners].mean() > FRONT
        (x0, y0), (x1, y1) = np.floor(points.min(0)).astype(int), np.ceil(points.max(0)).astype(int) + 1
        xs, ys = np.meshgrid(np.arange(max(x0, 0), min(x1, width)), np.arange(max(y0, 0), min(y1, height)))
        (ax, ay), (bx, by), (cx, cy) = points
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        l1 = ((by - cy) * (xs + 0.5 - cx) + (cx - bx) * (ys + 0.5 - cy)) / den
        l2 = ((cy - ay) * (xs + 0.5 - cx) + (ax - cx) * (ys + 0.5 - cy)) / den
        l3 = 1 - l1 - l2
        inside = (l1 > -0.02) & (l2 > -0.02) & (l3 > -0.02)
        rows, cols = ys[inside], xs[inside]
        value = (l1 * values[0] + l2 * values[1] + l3 * values[2])[inside]
        weight[rows, cols] = np.maximum(weight[rows, cols], value)
        covered[rows, cols] = 1.0
        if is_front:
            front[rows, cols] = True
    return weight, covered, front


def _grow(image, radius):
    """Max filter over a (2 radius + 1) square, done as two passes."""
    out = image
    for axis in (0, 1):
        stacked = out.copy()
        for step in range(1, radius + 1):
            stacked = np.maximum(stacked, np.roll(out, step, axis=axis))
            stacked = np.maximum(stacked, np.roll(out, -step, axis=axis))
        out = stacked
    return out


def _hue_difference(hue, target):
    return (target - hue + 0.5) % 1.0 - 0.5


def _rgb_to_hsv(rgb):
    high, low = rgb.max(-1), rgb.min(-1)
    spread = high - low
    safe = np.where(spread > 1e-9, spread, 1.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.where(high == r, ((g - b) / safe) % 6, np.where(high == g, (b - r) / safe + 2, (r - g) / safe + 4)) / 6
    hue = np.where(spread > 1e-9, hue, 0.0)
    saturation = np.where(high > 1e-9, spread / np.where(high > 1e-9, high, 1.0), 0.0)
    return np.stack([hue % 1.0, saturation, high], -1)


def _hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0] * 6, hsv[..., 1], hsv[..., 2]
    sector = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    choices = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    out = np.zeros(hsv.shape, np.float32)
    for k, (r, g, b) in enumerate(choices):
        pick = sector == k
        out[pick] = np.stack([r[pick], g[pick], b[pick]], -1)
    return out


main()
