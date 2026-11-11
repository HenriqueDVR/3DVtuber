"""A blink that brings a lid down over a still eye, as hand-made anime faces do.

For tracked eyes only: a moving iris or modelled discs, under an EyeLids layer with a see-through
opening in front of the iris (eyes.add_moving_irises, eyes.add_modelled_irises). Per eye:
- the upper lashes are cut from the face picture (dark pixels joined to the opening's top edge) onto
  a strip of their own, which lies over them at rest;
- a lid, showing the face picture with the eye filled smoothly from the skin round it, lies folded
  to nothing under the strip at rest;
- eyeBlinkLeft/Right are rebuilt: nothing of the face, iris or white moves; the strip comes down
  (its ends CORNER of the way the middle goes) and the lid unrolls behind it over the opening.

Prototyped as out/_lid_proto/lid_blink.py and passed by the user in Warudo on OC3 and CatGirl
(2026-11-10). Seen on seven characters, it stays experimental (manifest option `lid_blink`, off): the squeeze
blink, the eye squeezed toward its lash line, is the default (the user, 2026-11-11).
Painted eyes that slide, a visor, a blink alone keep the squeeze (no EyeLids layer). Everything is
read and placed before the mesh is touched: an eye refused (LidBlinkRefused) leaves the blend as it was.
"""
import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import eyes, face_points, face_rig, fingers, meshdata, rig

DARK = 0.50          # a lash pixel is darker than this share of the skin's brightness round the eye
LASH_UP = 0.40       # lashes reach no further above the opening than this share of the eye's height
LASH_SIDE = 0.25     # nor further past its ends than this share of its width
CORNER = 0.35        # the lash line's ends come down this share of the way its middle does
OVER = 6             # px the closed lid passes the opening's lower edge
TUCK = 2             # px the lid's edges lie under the lashes
FILL_GROW = 4        # px round what is no skin that is filled too: its soft edge left ghosts of the eye on the lid
COLUMNS = 64
LID_ROWS = 24
STRIP_ROWS = 6
LID_GAP = 0.0003     # m in front of the face's front
STRIP_GAP = 0.0006
HAIR_FRONT = 0.006   # what stands this far before the lids of a painted eye is hair, not face
SOCKET_FRONT = 0.018 # ... and this far before the frame of a modelled one, which lies deep in its socket under the lid's rim
HAIR_CLEAR = 0.08    # hair drawn nearer the eye than this share of its width is covered by the lid like the eye's own lines
CLEAR = 0.00015      # m the sheets keep before the face all the way down
SKIN_TOL = 0.10
# Skin in shade, blush: darker than the skin's own tone by one share in every channel, give or take SHADE_SPREAD,
# no darker than SHADE_DARK of it. It stays on the lid as drawn and the fill starts from it: filled from the
# lit skin alone the lid was one flat pale tone with an edge all round (the user, 2026-11-10)
SHADE_SPREAD = 0.25
SHADE_DARK = 0.6
LOCK_FRONT = 0.003   # m before the lids' fitted surface: what of their layer stands further off is a lock of hair
FACE_BACK = 0.03     # m behind the eye's lids: what lies further back is no face
ON_FACE = 0.85       # share of a column's way down that has face under it, for the sheets to reach that far
LASH_MATERIAL = "EyeLash"
LID_MATERIAL = "EyeLidSkin"
SHEET_MATERIALS = (LASH_MATERIAL, LID_MATERIAL)
ATTRIBUTE = "face_rig_lid_blink"  # 1 lash strip, 2 lid


class LidBlinkRefused(Exception):
    """An eye the lid blink can't be laid over: nothing was changed."""


def has_lids(mesh):
    """True when the mesh's eyes are tracked under an EyeLids layer, which the lid blink needs."""
    return mesh.materials.find(eyes.LIDS_MATERIAL) >= 0


def add_lid_blink(mesh_obj, points, only=None, corner=CORNER):
    """Lay a lash strip and a folded lid over each shown eye and rebuild eyeBlinkLeft/Right to bring them down.

    Assumes the eye step has run (an EyeLids layer, the face patch, blink keys) and the model faces -Y in
    the modelled pose. `only` ("left"/"right"): the other eye is covered and gets nothing. Touches only the
    two blink keys and the new sheets (head-weighted, in every key at their rest place, face_rig fixed):
    the face, iris, white and their pictures stay as they are. Returns side -> numbers (mm, pixels).
    Raises LidBlinkRefused, with the blend untouched, when an eye can't be read or has no face under it.
    """
    mesh = mesh_obj.data
    if not has_lids(mesh):
        raise LidBlinkRefused("no EyeLids layer: the lid blink needs tracked eyes (moving iris or modelled)")
    keys = mesh.shape_keys.key_blocks if mesh.shape_keys else {}
    face = eyes.FacePatch(mesh_obj)
    shown = eyes.shown_eyes(points, face.size, only)
    missing = [f"eyeBlink{side.title()}" for side in shown if f"eyeBlink{side.title()}" not in keys]
    if missing or not shown:
        raise LidBlinkRefused(f"no blink keys to rebuild ({', '.join(missing) or 'no eye shown'})")
    rows = eyes.image_rows(face.image)
    lid_rows = rows.copy()
    lash_rows = rows.copy()
    lash_rows[..., 3] = 0.0
    lids_alpha = _lids_alpha(mesh, mesh_obj, face)
    size = face.metres_per_pixel
    tree = _face_tree(mesh_obj)
    plans, report = {}, {}
    # where the face picture shows the face itself: skin, the eyes' openings and lashes. The rest (hair drawn
    # across it, brows) says nothing of where the face's surface lies
    shows_face = np.ones(rows.shape[:2], bool)
    for side, eye in shown.items():
        x0, y0, x1, y1 = eye.box
        rgb = rows[y0:y1, x0:x1, :3].astype(np.float64)
        opening = eyes.find_opening(rgb, eye)
        # ... and whatever the lids layer lets the iris show through next to it
        ys, xs = np.mgrid[y0:y1, x0:x1]
        seen = lids_alpha(xs + 0.5, ys + 0.5) < 0.5
        opening |= eyes._reach(opening, opening | seen) & eyes._dilate(opening, max(2, int(0.15 * eye.height)))
        opening = eyes._fill_holes(opening)
        if opening.sum() < 20:
            raise LidBlinkRefused(f"no opening read in the {side} eye")
        lash, skin_luma = _lash_mask(rgb, opening, eye)
        # the lid shows the face as drawn with the eye taken out: shut, it lies where its picture was drawn, so skin,
        # shade and the hair drawn across it run on over its edge unbroken
        filled, hole, hair, skin = _skin_fill(rgb, opening, lash, skin_luma, eye)
        lid_rows[y0:y1, x0:x1, :3] = filled
        shows_face[y0:y1, x0:x1] = skin | opening | lash
        # the strip's edge soft and its colour carried past it: cut out hard, thin lashes broke up when drawn small
        lash_rows[y0:y1, x0:x1, :3] = face_points.fill_across(rgb, ~lash)
        lash_rows[y0:y1, x0:x1, 3] = eyes.feather(lash, 3)
        plans[side] = _plan(opening, lash, corner)
        report[side] = {"opening_px": int(opening.sum()), "lash_px": int(lash.sum()), "filled_px": int(hole.sum()),
                        "opening_mm": 1000 * plans[side]["height"] * size,
                        "middle_mm": 1000 * plans[side]["d"].max() * size, "ends_mm": 1000 * plans[side]["d"].min() * size}

    sheets = {}
    painted = eyes.get_int(mesh, eyes.OVERLAY_ATTRIBUTE).any()
    trusted = lambda xz: shows_face[tuple(np.clip(face.pixel(xz)[:, ::-1].astype(int), 0, np.array(shows_face.shape) - 1).T)]  # noqa: E731
    for side, p in plans.items():
        eye = shown[side]
        x0, y0 = eye.box[:2]
        px = lambda c, r: face.world(np.stack([np.asarray(c, float) + x0 + 0.5, np.asarray(r, float) + y0 + 0.5], axis=1))  # noqa: E731
        low = px([p["cols"][0] - 20], [p["bottom"] + p["d"].max() + 20])[0]
        high = px([p["cols"][-1] + 20], [p["top"] - 20])[0]
        front = _Front(tree, np.minimum(low, high), np.maximum(low, high), HAIR_FRONT if painted else SOCKET_FRONT, trusted)
        sheets[side] = _sheets(front, p, px, side)
        report[side].update(sheets[side]["report"])

    # nothing was changed until here
    lash_slot = eyes._material_like(mesh, LASH_MATERIAL, eyes.new_image(LASH_MATERIAL, lash_rows))
    eyes.cut_out(mesh.materials[lash_slot])
    lid_slot = eyes._material_like(mesh, LID_MATERIAL, eyes.new_image(LID_MATERIAL, lid_rows))
    eyes.cut_out(mesh.materials[lid_slot])
    normals = {}
    for side, s in sheets.items():
        strip = _add_sheet(mesh_obj, s["strip"][0], face.uv(s["strip_uv"]), s["strip_shape"], lash_slot, 1)
        lid = _add_sheet(mesh_obj, s["lid"][0], face.uv(s["lid_uv"]), s["lid_shape"], lid_slot, 2)
        normals.update(zip(strip.tolist(), s["strip_normals"]))
        normals.update(zip(lid.tolist(), s["lid_normals"]))
        s["ids"] = (strip, lid)
    basis = eyes._key_co(keys[0])
    for side, s in sheets.items():
        co = basis.copy()
        for ids, name in zip(s["ids"], ("strip", "lid")):
            co[ids] = eyes._to_local(mesh_obj, s[name][1])
        keys[f"eyeBlink{side.title()}"].data.foreach_set("co", co.ravel())
    mesh.update()
    _set_normals(mesh_obj, normals)
    for name in SHEET_MATERIALS:
        crop = eyes.crop_to_use(mesh, name)
        report.setdefault("images", {})[name] = (crop["box"][2] - crop["box"][0], crop["box"][3] - crop["box"][1])
    return report


def _sheets(front, p, px, side):
    """World points of one eye's strip and lid, at rest and shut, with their UVs' world (x, z) and normals."""
    # the lash strip: a band round the lashes, each column coming down by its d
    # ... as far to either side as the face goes: past its edge the sheets hung in the air beside the head, into the
    # hair when seen from the side (the user, 2026-11-10; OC3's lashes end at x 76 mm, where the next surface
    # behind is 70 mm back)
    every = np.arange(p["cols"][0] - 3, p["cols"][-1] + 4, dtype=float)
    top_ = np.interp(every, p["cols"], p["strip_top"]) - TUCK
    bottom_ = np.interp(every, p["cols"], p["lid_foot"]) + np.interp(every, p["cols"], p["d"])
    way = top_[None] + np.linspace(0.0, 1.0, 12)[:, None] * (bottom_ - top_)[None]
    on = front.on_face(px(np.broadcast_to(every[None], way.shape).ravel(), way.ravel())).reshape(way.shape).mean(axis=0) >= ON_FACE
    first = last = len(every) // 2
    if not on[first]:
        raise LidBlinkRefused(f"no face under the middle of the {side} eye")
    while first > 0 and on[first - 1]:
        first -= 1
    while last < len(every) - 1 and on[last + 1]:
        last += 1
    cols = np.linspace(every[first], every[last], COLUMNS + 1)
    upper = np.interp(cols, p["cols"], p["strip_top"]) - TUCK
    lower = np.interp(cols, p["cols"], p["strip_bottom"]) + TUCK
    d = np.interp(cols, p["cols"], p["d"])
    # an end cut short of the lashes' tip stays where it is, and the lashes come down more the further from it:
    # moved whole, the strip broke off from the tip left on the face. Never less than shuts the eye
    ramp = np.ones(len(cols))
    ease = np.linspace(0.0, 1.0, len(cols) // 5)
    ease = ease * ease * (3 - 2 * ease)
    if first > 0:
        ramp[:len(ease)] = ease
    if last < len(every) - 1:
        ramp[-len(ease):] = ease[::-1]
    d = np.maximum(d * ramp, np.interp(cols, p["cols"], p["shuts"]))
    share = np.linspace(0.0, 1.0, STRIP_ROWS + 1)[:, None]          # 0 bottom row, 1 top
    r = lower[None] + share * (upper - lower)[None]
    c = np.broadcast_to(cols[None], r.shape)
    strip_rest, strip_shut = px(c.ravel(), r.ravel()), px(c.ravel(), (r + d[None]).ravel())

    # the lid: behind the strip, from over the lashes to their foot at rest (whatever is left of them on the
    # face stays hidden), and down to where they land when shut
    foot = np.interp(cols, p["cols"], p["lid_foot"])
    share = np.linspace(1.0, 0.0, LID_ROWS + 1)[:, None]            # the bottom row goes all the way
    r_rest = upper[None] + share * (foot - upper)[None]
    r_shut = upper[None] + share * (foot + d - TUCK - upper)[None]
    c = np.broadcast_to(cols[None], r_shut.shape)
    lid_rest, lid_shut = px(c.ravel(), r_rest.ravel()), px(c.ravel(), r_shut.ravel())

    # shut, the lid follows the face, or where the face falls back behind it (an eye in its socket under the lid's
    # rim) runs straight from the lashes' foot to where they land: a lid over an eye, not a lining of its socket.
    # Only there: as the hull of the whole way down it bridged the hollow a flat painted eye lies in too, and
    # stood millimetres off the face, "a layer of skin on top of the eyes" (the user, 2026-11-10)
    width = COLUMNS + 1
    depth = front(lid_shut).reshape(-1, width)
    run = (r_shut - foot[None]) / np.maximum(r_shut[0] - foot, 1e-6)[None]     # 0 at the foot, 1 where they land
    at_foot = front(px(cols, foot))
    straight = at_foot[None] + np.clip(run, 0.0, 1.0) * (depth[0] - at_foot)[None]
    depth = np.where(run > 0, np.minimum(depth, straight), depth)
    lid_shut_at = np.stack([lid_shut[:, 0], depth.ravel() - LID_GAP, lid_shut[:, 1]], axis=1)
    # ... and the lashes ride on it: each strip row as deep as the lid is at its height
    rows_shut = (r + d[None])
    on_lid = np.stack([np.interp(rows_shut[:, i], r_shut[::-1, i], depth[::-1, i]) for i in range(width)], axis=1)
    below = rows_shut > r_shut[0][None]
    on_lid = np.where(below, np.minimum(front(strip_shut).reshape(-1, width), depth[0][None]), on_lid)
    strip_shut_at = np.stack([strip_shut[:, 0], on_lid.ravel() - STRIP_GAP, strip_shut[:, 1]], axis=1)
    places = {"lid": [_at(front, lid_rest, LID_GAP), lid_shut_at],
              "strip": [_at(front, strip_rest, STRIP_GAP), strip_shut_at]}
    # a straight way from rest to shut cuts behind what bulges between: a column comes forward by as much, at both ends
    push = {name: _behind(front, *ends).reshape(-1, width).max(axis=0) for name, ends in places.items()}
    push["strip"] = np.maximum(push["strip"], push["lid"])
    report = {"cut_px": (first, len(every) - 1 - last)}
    for name, ends in places.items():
        # ... evenly: column by column the sheet was corrugated, and shaded in stripes (CatGirl's lid over her socket)
        wide = _envelope(push[name], 6, True)
        wide = np.convolve(np.pad(wide, 3, mode="edge"), np.ones(7) / 7, mode="valid")
        for points_ in ends:
            points_[:, 1] -= np.tile(wide, len(points_) // width)
        report[f"{name}_forward_mm"] = 1000 * float(wide.max())
    least = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        for name in ("strip", "lid"):
            begin, closed = places[name]
            now = begin + t * (closed - begin)
            least.append(float(np.min(front.raw(now[:, [0, 2]]) - now[:, 1])))
    report["least_clear_mm"] = 1000 * min(least)
    return {"strip": places["strip"], "lid": places["lid"], "strip_uv": strip_rest, "lid_uv": lid_shut,
            "strip_shape": r.shape, "lid_shape": r_shut.shape,
            # lit as the face is where it lies shut: its own normals, from a band a few millimetres tall at rest, aren't the eye's
            "lid_normals": _sheet_normals(places["lid"][1].reshape(-1, width, 3), front.normal(lid_shut).reshape(-1, width, 3)),
            "strip_normals": front.normal(strip_rest), "report": report}


def _lash_mask(rgb, opening, eye):
    """The upper lashes of one eye's crop: dark pixels joined to the opening's top edge; and the skin's brightness."""
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    ring = eyes._dilate(opening, int(0.5 * eye.height)) & ~eyes._dilate(opening, 4)
    bright = luma[ring]
    skin = float(np.median(bright[bright >= np.median(bright)]))
    dark = (luma < DARK * skin) & ~opening
    ys, xs = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
    columns = np.nonzero(opening.any(axis=0))[0]
    first = np.array([np.nonzero(opening[:, c])[0][0] for c in columns])
    last = np.array([np.nonzero(opening[:, c])[0][-1] for c in columns])
    every = np.arange(rgb.shape[1])
    top = np.interp(every, columns, first)
    middle = np.interp(every, columns, (first + last) / 2)
    reach = LASH_SIDE * (columns[-1] - columns[0])
    allowed = ((ys < middle[None]) & (ys > top[None] - LASH_UP * eye.height)
               & (xs > columns[0] - reach) & (xs < columns[-1] + reach))
    seeds = eyes._dilate(opening, 3) & dark & allowed
    lash = eyes._reach(seeds, dark & allowed)
    lash = eyes._dilate(lash, 1) & ~opening & allowed
    return lash, skin


def _skin_fill(rgb, opening, lash, skin_luma, eye):
    """The crop with the eye and its lines taken out, filled smoothly from the skin round them: (picture, what was
    filled, hair drawn across the face, skin as drawn). Hair and skin, in light or shade, stay as they are."""
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    dark = luma < 0.8 * skin_luma
    away = ~eyes._dilate(opening | lash, 4) & ~eyes._dilate(dark, 2)
    fitted = eyes.skin_behind(rgb, away)
    lit = away & (np.abs(rgb - fitted).max(axis=-1) < SKIN_TOL)
    share = rgb / np.maximum(fitted, 1e-3)
    shaded = ((np.ptp(share, axis=-1) < SHADE_SPREAD) & (share.mean(axis=-1) > SHADE_DARK) & (share.max(axis=-1) < 1.08)
              & ~eyes._dilate(opening | lash, 3) & ~eyes._dilate(luma < DARK * skin_luma, 2))
    skin = lit | shaded
    # hair, as against the eye's own lines: what is no skin and runs to the crop's edge without coming near the eye
    border = np.zeros_like(skin)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    near = eyes._dilate(opening | lash, max(3, int(HAIR_CLEAR * eye.width)))
    hair = eyes._reach(border, ~skin & ~near)
    # ... and nearer the eye, what is drawn in the hair's colours rather than the lashes' or the skin's (a lock over
    # the eye's corner): where hair and lashes are one colour nothing is, and the lid covers it
    if hair.sum() > 50 and lash.sum() > 50:
        centres = np.concatenate([eyes._kmeans(rgb[hair], 4), eyes._kmeans(rgb[lash], 3)])
        gap = np.linalg.norm(rgb[..., None, :] - centres, axis=-1)
        like_hair = (gap[..., :4].min(axis=-1) < gap[..., 4:].min(axis=-1)) & (gap[..., :4].min(axis=-1) < np.linalg.norm(rgb - fitted, axis=-1))
        hair |= eyes._reach(eyes._dilate(hair, 2), like_hair & ~skin & near & ~opening & ~lash)
    hole = eyes._dilate(~skin & ~hair, FILL_GROW) & ~hair
    filled = face_points.fill_across(rgb, hole, skin & ~hole).astype(np.float64)
    return filled, hole, hair, skin


def _envelope(values, reach, highest):
    """Each value's lowest (or highest) neighbour within `reach` columns, NaNs skipped."""
    out = np.empty(len(values))
    pick = np.nanmax if highest else np.nanmin
    for i in range(len(values)):
        window = values[max(i - reach, 0):i + reach + 1]
        out[i] = pick(window) if np.isfinite(window).any() else np.nan
    return out


def _bridged(values):
    index = np.arange(len(values))
    good = np.isfinite(values)
    return np.interp(index, index[good], values[good])


def _plan(opening, lash, corner):
    """Per crop column from the lashes' first to their last: where the strip and lid lie and how far they come down (px)."""
    both = opening | lash
    cols = np.nonzero(both.any(axis=0))[0]
    cols = np.arange(cols[0], cols[-1] + 1)
    first = lambda mask: np.array([np.nonzero(mask[:, c])[0][0] if mask[:, c].any() else np.nan for c in cols])  # noqa: E731
    last = lambda mask: np.array([np.nonzero(mask[:, c])[0][-1] + 1 if mask[:, c].any() else np.nan for c in cols])  # noqa: E731
    reach = max(2, len(cols) // COLUMNS)
    o_top, o_bottom = _envelope(first(opening), reach, False), _envelope(last(opening), reach, True)
    is_open = np.isfinite(o_top)
    height = np.where(is_open, o_bottom - o_top, 0.0)
    full = float(height.max())
    inside = cols[is_open]
    u = np.clip((cols - (inside[0] + inside[-1]) / 2) / max((inside[-1] - inside[0]) / 2, 1.0), -1.0, 1.0)
    d = np.maximum(full * (corner + (1 - corner) * (1 - u * u)), height) + OVER
    kernel = np.ones(2 * reach + 1) / (2 * reach + 1)
    d = np.maximum(np.convolve(np.pad(d, reach, mode="edge"), kernel, mode="valid"), height + OVER)
    strip_top = _bridged(_envelope(np.fmin(first(lash), first(opening)), reach, False))
    strip_bottom = _bridged(_envelope(np.fmax(last(lash), first(opening)), reach, True))
    foot = _bridged(np.where(is_open, o_top, _envelope(last(lash), reach, True)))
    return {"cols": cols, "d": d, "shuts": np.where(is_open, height + OVER, 0.0), "height": full, "lid_foot": foot,
            "strip_top": strip_top, "strip_bottom": strip_bottom, "top": float(np.nanmin(strip_top)),
            "bottom": float(np.nanmax(o_bottom))}


def _lids_alpha(mesh, mesh_obj, face):
    """Alpha of the EyeLids picture at face-picture pixels (x, y): 1 off it. Placed by its polygons' UVs, as cropped."""
    slot = mesh.materials.find(eyes.LIDS_MATERIAL)
    image = next(n.image for n in mesh.materials[slot].node_tree.nodes if n.type == "TEX_IMAGE" and n.image is not None)
    verts = meshdata.world_vertices(mesh_obj)
    uv = mesh.uv_layers.active.data
    own, whole = [], []
    for poly in mesh.polygons:
        if poly.material_index == slot:
            for li in poly.loop_indices:
                own.append(uv[li].uv[:])
                whole.append(verts[mesh.loops[li].vertex_index][[0, 2]])
    if not own:
        raise LidBlinkRefused("the EyeLids layer has no polygons")
    own, whole = np.array(own), face.uv(np.array(whole))
    fit = [np.polyfit(whole[:, k], own[:, k], 1) for k in (0, 1)]
    rows = eyes.image_rows(image)
    height, width = rows.shape[:2]

    def alpha(x, y):
        c = np.floor(np.polyval(fit[0], np.asarray(x, float) / face.size[0]) * width).astype(int)
        r = np.floor((1.0 - np.polyval(fit[1], 1.0 - np.asarray(y, float) / face.size[1])) * height).astype(int)
        ok = (c >= 0) & (c < width) & (r >= 0) & (r < height)
        out = np.ones(np.shape(c))
        out[ok] = rows[r[ok], c[ok], 3]
        return out
    return alpha


def _face_tree(mesh_obj):
    """(the face as seen from the front without hair on spring bones or laid over the face, the EyeLids layer)."""
    mesh = mesh_obj.data
    tris = meshdata.triangles(mesh)
    polygon_of = np.empty(len(mesh.loop_triangles), np.int64)
    mesh.loop_triangles.foreach_get("polygon_index", polygon_of)
    materials = np.empty(len(mesh.polygons), np.int64)
    mesh.polygons.foreach_get("material_index", materials)
    lids_slot = mesh.materials.find(eyes.LIDS_MATERIAL)
    over = mesh.materials.find("face_patch_over")
    humanoid = set(rig.BONE_NAMES.values()) | {fingers.bone_name(side, f, seg) for side in ("left", "right")
                                                for f in fingers.FINGERS for seg in fingers._segments(f)}
    bones = mesh_obj.parent.data.bones if mesh_obj.parent is not None and mesh_obj.parent.type == "ARMATURE" else {}
    chain = {g.index for g in mesh_obj.vertex_groups if g.name not in humanoid and g.name in bones}
    swung = np.array([sum(g.weight for g in v.groups if g.group in chain) > 0.1 for v in mesh.vertices]) if chain \
        else np.zeros(len(mesh.vertices), bool)
    keep = (materials[polygon_of] != over) & ~swung[tris].any(axis=1)
    lids = materials[polygon_of] == lids_slot
    verts = [Vector(v) for v in meshdata.world_vertices(mesh_obj)]
    return BVHTree.FromPolygons(verts, tris[keep].tolist()), BVHTree.FromPolygons(verts, tris[lids].tolist())


class _Front:
    """The face's front over a box of (x, z): sampled, each sample taking its neighbourhood's nearest, smoothed.

    Never behind the surface it was read from, so a sheet laid a gap in front of it clears the lids' rims.
    """

    def __init__(self, trees, low, high, hair_front, trusted, step=0.0005, reach=3):
        everything, lids = trees
        self.low, self.step = np.asarray(low, float), step
        xs = np.arange(low[0], high[0] + step, step)
        zs = np.arange(low[1], high[1] + step, step)
        depth = np.full((len(zs), len(xs)), np.nan)
        layer = np.full(depth.shape, np.nan)
        for j, z in enumerate(zs):
            for i, x in enumerate(xs):
                hit = everything.ray_cast(Vector((x, -10.0, z)), Vector((0.0, 1.0, 0.0)), 20.0)[0]
                if hit is not None:
                    depth[j, i] = hit.y
                hit = lids.ray_cast(Vector((x, -10.0, z)), Vector((0.0, 1.0, 0.0)), 20.0)[0]
                if hit is not None:
                    layer[j, i] = hit.y
        # (not by the lids' middle depth: the lids copied round a painted eye take in a lock of hair lying close, and
        # by their nearest the fringe 2 cm before an eye passed for face; nor by one depth for them all: the face is
        # round, the lids of OC3's eye lie from 14 mm before their middle depth to 15 behind it, and the nearer third
        # of them was read as hair: the lid sank behind them and the lashes it should cover showed over it. By a
        # curved surface through them, fitted past what stands off it)
        gx, gz = np.meshgrid((xs - xs.mean()) / 0.03, (zs - zs.mean()) / 0.03)
        terms = np.stack([np.ones_like(gx), gx, gz, gx * gx, gx * gz, gz * gz], axis=-1)
        source = layer if np.isfinite(layer).sum() > 50 else depth
        use = np.isfinite(source)
        if use.sum() < 6:
            raise LidBlinkRefused("no face found in front of an eye")
        for _ in range(6):
            fit = np.linalg.lstsq(terms[use], source[use], rcond=None)[0]
            off_fit = source - terms @ fit
            spread = max(0.0015, 2.5 * float(np.nanmedian(np.abs(off_fit[use]))))
            use = np.isfinite(source) & (np.abs(off_fit) < spread)
        nearest_own = terms @ fit
        hair = depth < nearest_own - hair_front
        # no face: nothing there, or the next thing behind where the face has ended (the side of the head, the back
        # hair's inside). Read as face it pulled the sheets' ends centimetres back and the rule against cutting
        # through the face threw them as far forward
        self.off = np.isnan(depth) | (depth > nearest_own + FACE_BACK)
        depth[self.off] = np.nan
        gx, gz = np.meshgrid(xs, zs)
        drawn = ~trusted(np.stack([gx.ravel(), gz.ravel()], axis=1)).reshape(depth.shape) & np.isfinite(depth)
        depth[hair | drawn] = np.nan
        # ... and there the face is where the skin round it is: carried in from its edge
        while np.isnan(depth).any():
            padded = np.pad(depth, 1, constant_values=np.nan)
            around = np.stack([padded[1:-1, :-2], padded[1:-1, 2:], padded[:-2, 1:-1], padded[2:, 1:-1]])
            known = np.isfinite(around)
            if not known.any():
                raise LidBlinkRefused("no face found in front of an eye")
            mean = np.where(known, around, 0.0).sum(axis=0) / np.maximum(known.sum(axis=0), 1)
            depth = np.where(np.isnan(depth) & known.any(axis=0), mean, depth)
        # never behind the lids themselves where hair or a brow is drawn over them: carried in from the skin beside
        # it the front lay under a few of their lashes, and those showed through the shut lid as a dark sliver.
        # (What of that layer stands off its surface is a lock of hair taken in with it, and stays in front)
        own_lids = np.isfinite(layer) & (layer > nearest_own - LOCK_FRONT)
        depth[own_lids] = np.minimum(depth, layer)[own_lids]
        self.surface = depth
        padded = np.pad(depth, reach, mode="edge")
        size = 2 * reach + 1
        self.grid = np.stack([padded[a:a + depth.shape[0], b:b + depth.shape[1]] for a in range(size) for b in range(size)]).min(axis=0)
        padded = np.pad(self.grid, reach, mode="edge")
        self.grid = np.stack([padded[a:a + depth.shape[0], b:b + depth.shape[1]] for a in range(size) for b in range(size)]).mean(axis=0)
        gz, gx = np.gradient(self.grid, step)
        normal = np.stack([gx, -np.ones_like(gx), gz], axis=-1)
        self.normals = normal / np.linalg.norm(normal, axis=-1, keepdims=True)

    def _sample(self, field, xz):
        at_ = (np.asarray(xz, float) - self.low) / self.step
        i = np.clip(at_[:, 0], 0, field.shape[1] - 1.001)
        j = np.clip(at_[:, 1], 0, field.shape[0] - 1.001)
        i0, j0 = i.astype(int), j.astype(int)
        fi, fj = (i - i0), (j - j0)
        if field.ndim == 3:
            fi, fj = fi[:, None], fj[:, None]
        return ((1 - fj) * ((1 - fi) * field[j0, i0] + fi * field[j0, i0 + 1])
                + fj * ((1 - fi) * field[j0 + 1, i0] + fi * field[j0 + 1, i0 + 1]))

    def __call__(self, xz):
        return self._sample(self.grid, xz)

    def on_face(self, xz):
        at_ = np.rint((np.asarray(xz, float) - self.low) / self.step).astype(int)
        inside = (at_ >= 0).all(axis=1) & (at_[:, 0] < self.off.shape[1]) & (at_[:, 1] < self.off.shape[0])
        out = np.zeros(len(at_), bool)
        out[inside] = ~self.off[at_[inside, 1], at_[inside, 0]]
        return out

    def raw(self, xz):
        return self._sample(self.surface, xz)

    def normal(self, xz):
        n = self._sample(self.normals, xz)
        return n / np.linalg.norm(n, axis=1, keepdims=True)


def _behind(front, rest, shut):
    """How far (m) each point comes behind the face's front, less CLEAR, on its straight way from `rest` to `shut`."""
    worst = np.zeros(len(rest))
    for t in np.linspace(0.0, 1.0, 11):
        now = rest + t * (shut - rest)
        worst = np.maximum(worst, now[:, 1] - (front.raw(now[:, [0, 2]]) - CLEAR))
    return worst


def _at(front, xz, gap):
    """World points at (x, z), `gap` in front of the face's front."""
    return np.stack([xz[:, 0], front(xz) - gap, xz[:, 1]], axis=1)


def _add_sheet(mesh_obj, points, uvs, shape, slot, mark):
    """A sheet of quads through world `points`, (rows, columns) of them, the first row lowest; returns their indices."""
    mesh = mesh_obj.data
    local = eyes._to_local(mesh_obj, points)
    rows_, width = shape
    if points[1, 0] < points[0, 0]:
        raise RuntimeError("the face picture runs against world x: the lid blink's sheet would face backwards")
    group = (mesh_obj.vertex_groups.get(eyes.BONE_NAMES["head"]) or mesh_obj.vertex_groups.new(name=eyes.BONE_NAMES["head"])).index
    bm = bmesh.new()
    bm.from_mesh(mesh)
    start = len(bm.verts)
    deform = bm.verts.layers.deform.verify()
    uv_layer = bm.loops.layers.uv.active
    new = [bm.verts.new(tuple(p)) for p in local]
    for vert in new:
        vert[deform][group] = 1.0
    for j in range(rows_ - 1):
        for i in range(width - 1):
            ids = (j * width + i, j * width + i + 1, (j + 1) * width + i + 1, (j + 1) * width + i)
            quad = bm.faces.new([new[k] for k in ids])
            quad.material_index = slot
            quad.smooth = True
            for loop, k in zip(quad.loops, ids):
                loop[uv_layer].uv = tuple(uvs[k])
    bm.to_mesh(mesh)
    bm.free()
    indices = np.arange(start, start + len(new))
    for key in eyes._keys(mesh):
        co = eyes._key_co(key)
        co[indices] = local
        key.data.foreach_set("co", co.ravel())
    for name, value in ((ATTRIBUTE, mark), (face_rig.FIXED_ATTRIBUTE, 1)):
        values = eyes.get_int(mesh, name)
        values[indices] = value
        eyes._set_int(mesh, name, values)
    mesh.update()
    return indices


def _sheet_normals(grid, face_normals):
    """A shut lid's normals, per point of its (rows, columns) grid: its own shape's, smoothed, where it spans a socket
    and the face's under it point into the hollow; half the face's elsewhere, which carry the face's own roundness."""
    across, up = np.gradient(grid, axis=1), np.gradient(grid, axis=0)
    own = np.cross(across, up)
    own *= np.sign(-own[..., 1:2] + 1e-12)
    own /= np.maximum(np.linalg.norm(own, axis=-1, keepdims=True), 1e-12)
    mixed = own + face_normals
    padded = np.pad(mixed, ((2, 2), (2, 2), (0, 0)), mode="edge")
    mixed = np.stack([padded[a:a + grid.shape[0], b:b + grid.shape[1]] for a in range(5) for b in range(5)]).mean(axis=0)
    return (mixed / np.linalg.norm(mixed, axis=-1, keepdims=True)).reshape(-1, 3)


def _set_normals(mesh_obj, normals):
    """Give the new vertices the face's normals: a lid folded to nothing at rest has none of its own.

    The export takes the sheets' normals out of their face shapes (vrm_export.still_sheet_normals): Blender reckons a
    shape's normals from its geometry, and from a band folded under the lashes they turned the shut lid in stripes."""
    mesh = mesh_obj.data
    turn = np.array(mesh_obj.matrix_world.inverted().to_3x3())
    corner = np.empty(len(mesh.loops) * 3, np.float32)
    mesh.corner_normals.foreach_get("vector", corner)
    corner = corner.reshape(-1, 3)
    vertex = np.empty(len(mesh.loops), np.int64)
    mesh.loops.foreach_get("vertex_index", vertex)
    ids = np.array(sorted(normals))
    table = np.zeros((len(mesh.vertices), 3))
    table[ids] = np.array([normals[i] for i in ids]) @ turn.T
    mine = np.isin(vertex, ids)
    corner[mine] = table[vertex[mine]]
    mesh.normals_split_custom_set(corner.tolist())
