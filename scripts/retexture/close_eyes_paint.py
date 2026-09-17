"""Close the eyes in a face image using the image's own paint, no diffusion.

Each eye's opening and lash band are filled with skin (smooth fill from the
surrounding pixels), then the eye's own upper lash line is lifted out, flipped
so the lashes hang down, and laid along the closed-lid line. Lashes, colours and
line weight stay the model's.

Eyes are given as four points each in image pixels: inner corner, outer corner,
top of the opening (where the lashes meet the white) and bottom of the opening.

    python close_eyes_paint.py face.png eyes.json out.png [--only left|right]
"""
import argparse
import json

import numpy as np
from PIL import Image

LASH_ABOVE = 0.5     # lash band height above the top curve, in eye heights
FILL_ABOVE = 0.38    # skin fill reaches this far above the top curve (covers the open lashes)
LASH_BELOW = 0.06    # and below it
FILL_BELOW = 0.22    # skin fill reaches this far under the bottom curve
CORNER_PAD = 0.08    # skin fill reaches past each corner, in eye widths
SAG = 0.55           # closed line: 0 = straight between corners, 1 = on the bottom curve
LASH_SCALE = 0.75    # flipped lashes are this tall relative to the open ones
DARK = (0.55, 0.3)   # luminance: fully lash at or below the second value, none above the first


def curve(p0, p1, p2):
    return np.polyfit([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], 2)


def classify(img):
    """Masks of hair-coloured and skin-like pixels."""
    rgb = img[..., :3]
    mx, mn = rgb.max(2), rgb.min(2)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    hair = (sat > 0.6) & (lum > 0.4) & (rgb[..., 0] > 0.7) & (rgb[..., 0] > rgb[..., 2] + 0.4)
    skin = ~hair & (lum > 0.62) & (sat < 0.45) & (rgb[..., 0] >= rgb[..., 2])
    return hair, skin


def dilate(mask, r):
    out = mask.copy()
    for _ in range(r):
        out = out | np.roll(out, 1, 0) | np.roll(out, -1, 0) | np.roll(out, 1, 1) | np.roll(out, -1, 1)
    return out


def smooth_fill(img, mask, iterations=1500):
    """Fill masked pixels with a harmonic blend of their surroundings."""
    out = img.copy()
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - 2, 1), min(ys.max() + 3, img.shape[0] - 1)
    x0, x1 = max(xs.min() - 2, 1), min(xs.max() + 3, img.shape[1] - 1)
    region = out[y0 - 1:y1 + 1, x0 - 1:x1 + 1]
    m = mask[y0 - 1:y1 + 1, x0 - 1:x1 + 1]
    # start from the mean border colour so it converges quickly
    border = ~m & np.pad(m, 1)[2:, 1:-1] | ~m & np.pad(m, 1)[:-2, 1:-1] | ~m & np.pad(m, 1)[1:-1, 2:] | ~m & np.pad(m, 1)[1:-1, :-2]
    region[m] = region[border].mean(0)
    for _ in range(iterations):
        avg = (np.roll(region, 1, 0) + np.roll(region, -1, 0) + np.roll(region, 1, 1) + np.roll(region, -1, 1)) / 4
        region[m] = avg[m]
    return out


def close_eye(img, pts):
    h, w = img.shape[:2]
    inner, outer, top_p, bot_p = (np.array(pts[k], float) for k in ("inner", "outer", "top", "bottom"))
    xl, xr = sorted((inner[0], outer[0]))
    top, bot = curve(inner, top_p, outer), curve(inner, bot_p, outer)
    eye_h = bot_p[1] - top_p[1]
    eye_w = xr - xl
    line = lambda x: inner[1] + (x - inner[0]) * (outer[1] - inner[1]) / (outer[0] - inner[0])

    xs = np.arange(int(xl - CORNER_PAD * eye_w), int(xr + CORNER_PAD * eye_w) + 1)
    xc = np.clip(xs, xl, xr)
    t_y, b_y = np.polyval(top, xc), np.polyval(bot, xc)
    # corners taper: pad columns get a band that shrinks to nothing
    taper = np.clip(1 - np.maximum(xl - xs, xs - xr).clip(0) / (CORNER_PAD * eye_w + 1e-6), 0, 1)

    source = img.copy()
    mask = np.zeros((h, w), bool)
    for x, ty, by, k in zip(xs, t_y, b_y, taper):
        mid = (ty + by) / 2
        y_top = mid - (mid - (ty - FILL_ABOVE * eye_h)) * k
        y_bot = mid + (by + FILL_BELOW * eye_h - mid) * k
        mask[int(y_top):int(np.ceil(y_bot)) + 1, x] = True
    hair, skin = classify(source)
    # only skin may seed the fill: anything else next to the hole is filled too
    near = dilate(mask, 30)
    fill = mask | (near & ~skin)
    img = smooth_fill(img, fill)
    # hair strands painted across the eye stay on top
    keep = fill & hair
    img[keep] = source[keep]

    # lashes: take the band around the top curve, flip it, lay it along the closed line
    lum = source[..., :3] @ np.array([0.299, 0.587, 0.114])
    closed_y = (1 - SAG) * line(xc) + SAG * b_y
    for x, ty, cy, k in zip(xs, t_y, closed_y, taper):
        if k <= 0:
            continue
        above = int(LASH_ABOVE * eye_h * k)
        below = int(LASH_BELOW * eye_h)
        for dy in range(-below, above + 1):
            sy = int(round(ty - dy))            # source: dy pixels above the top curve
            ty_dst = cy + dy * LASH_SCALE       # destination: that far below the closed line
            dst = int(round(ty_dst))
            if not (0 <= sy < h and 0 <= dst < h):
                continue
            a = np.clip((DARK[0] - lum[sy, x]) / (DARK[0] - DARK[1]), 0, 1) * k
            img[dst, x, :3] = img[dst, x, :3] * (1 - a) + source[sy, x, :3] * a
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("face")
    ap.add_argument("eyes", help="face.json from viewer/setup.html (or {left: {...}, right: {...}})")
    ap.add_argument("out")
    ap.add_argument("--only", choices=["left", "right"])
    args = ap.parse_args()
    img = np.asarray(Image.open(args.face).convert("RGB")).astype(float) / 255
    with open(args.eyes) as f:
        eyes = json.load(f)
    if "right_eye" in eyes:  # face.json from viewer/setup.html
        eyes = {"right": eyes["right_eye"], "left": eyes["left_eye"]}
    for side, pts in eyes.items():
        if args.only and side != args.only:
            continue
        img = close_eye(img, pts)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
