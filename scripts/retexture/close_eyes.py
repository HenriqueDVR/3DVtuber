"""Paint a closed-eyes version of a face image, for textured blink lids.

Finds the two irises (saturated blue-ish blobs; override with --eyes), inpaints an
ellipse around each through ComfyUI, and pastes the result back with a feathered
edge so everything outside the eyes stays pixel-identical.

Run with ComfyUI's embedded Python:
    python close_eyes.py face.png out.png [--eyes x1,y1,x2,y2] [--seed 3]
"""
import argparse
import json
import os
import time
import urllib.request
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SERVER = "http://127.0.0.1:8188"
COMFY = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "ComfyUI_windows_portable", "ComfyUI")
CHECKPOINT = "wai_illustrious_v170.safetensors"
CROP = 1024
# Mask ellipse around an iris, in iris-widths: half-width, half-height, and upward shift (lashes).
EYE_RX, EYE_RY, EYE_UP = 1.55, 1.15, 0.2

POSITIVE = ("masterpiece, best quality, amazing quality, anime coloring, 1girl, orange hair, closed eyes, "
            "eyelashes, calm, flat color, cel shading, clean lineart, face close-up")
NEGATIVE = ("open eyes, blue eyes, iris, pupils, lowres, worst quality, blurry, jpeg artifacts, realistic, "
            "3d render, shiny skin, text, watermark, tears, glasses")


def iris_boxes(img):
    a = np.asarray(img.convert("RGB")).astype(float) / 255
    sat = (a.max(2) - a.min(2)) / np.maximum(a.max(2), 1e-6)
    blue = (a[..., 2] > 0.55) & (a[..., 2] > a[..., 0] + 0.25) & (sat > 0.45)
    h, w = blue.shape
    blue[: int(h * 0.35)] = False
    blue[int(h * 0.75):] = False
    ys, xs = np.nonzero(blue)
    boxes = []
    for sel in (xs < w / 2, xs >= w / 2):
        x0, x1 = np.percentile(xs[sel], [2, 98])
        y0, y1 = np.percentile(ys[sel], [2, 98])
        boxes.append(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0))
    return boxes


def post(payload):
    req = urllib.request.Request(SERVER + "/prompt", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["prompt_id"]


def wait_for(prompt_id):
    while True:
        hist = json.load(urllib.request.urlopen(f"{SERVER}/history/{prompt_id}"))
        if prompt_id in hist:
            entry = hist[prompt_id]
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError(json.dumps(entry["status"], indent=1))
            for node in entry["outputs"].values():
                for img in node.get("images", []):
                    return os.path.join(COMFY, "output", img.get("subfolder", ""), img["filename"])
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("face")
    ap.add_argument("out")
    ap.add_argument("--eyes", default="", help="x1,y1,x2,y2 iris centres in image pixels")
    ap.add_argument("--iris", type=float, default=0, help="iris width in pixels (with --eyes)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--denoise", type=float, default=1.0)
    args = ap.parse_args()

    face = Image.open(args.face).convert("RGB")
    if args.eyes:
        v = [float(c) for c in args.eyes.split(",")]
        boxes = [(v[0], v[1], args.iris), (v[2], v[3], args.iris)]
    else:
        boxes = iris_boxes(face)
    print("irises:", [(round(x), round(y), round(s)) for x, y, s in boxes])

    cx = int(sum(b[0] for b in boxes) / 2) - CROP // 2
    cy = int(sum(b[1] for b in boxes) / 2) - CROP // 2
    cx = min(max(cx, 0), face.width - CROP)
    cy = min(max(cy, 0), face.height - CROP)

    mask = Image.new("L", face.size, 0)
    draw = ImageDraw.Draw(mask)
    for x, y, size in boxes:
        rx, ry = EYE_RX * size, EYE_RY * size
        y -= EYE_UP * size
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=255)

    input_dir = os.path.join(COMFY, "input", "rt")
    os.makedirs(input_dir, exist_ok=True)
    box = (cx, cy, cx + CROP, cy + CROP)
    face.crop(box).save(os.path.join(input_dir, "eyes_crop.png"))
    mask.crop(box).filter(ImageFilter.GaussianBlur(6)).save(os.path.join(input_dir, "eyes_mask.png"))

    wf = {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": POSITIVE, "clip": ["ckpt", 1]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["ckpt", 1]}},
        "img": {"class_type": "LoadImage", "inputs": {"image": "rt/eyes_crop.png"}},
        "mask": {"class_type": "LoadImageMask", "inputs": {"image": "rt/eyes_mask.png", "channel": "red"}},
        "enc": {"class_type": "VAEEncode", "inputs": {"pixels": ["img", 0], "vae": ["ckpt", 2]}},
        "noise_mask": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["enc", 0], "mask": ["mask", 0]}},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["ckpt", 0], "seed": args.seed, "steps": 28, "cfg": 6.0, "sampler_name": "euler_ancestral",
            "scheduler": "normal", "positive": ["pos", 0], "negative": ["neg", 0],
            "latent_image": ["noise_mask", 0], "denoise": args.denoise}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["ckpt", 2]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": "rt/eyes_closed"}},
    }
    result = Image.open(wait_for(post({"prompt": wf, "client_id": str(uuid.uuid4())}))).convert("RGB")

    out = face.copy()
    feather = mask.filter(ImageFilter.GaussianBlur(10))
    patch = Image.new("RGB", face.size)
    patch.paste(result, box[:2])
    out = Image.composite(patch, out, feather)
    out.save(args.out)
    mask.save(os.path.splitext(args.out)[0] + "_mask.png")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
