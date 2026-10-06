"""Refine rendered views with SDXL img2img through a running ComfyUI server.

Each color_<view>.png is re-drawn at low denoise, guided by ControlNet depth
(depth_<view>.png) and an IP-Adapter reference crop from the character sheet.
Results are written as gen_<view>.png next to the inputs.

Run with ComfyUI's embedded Python (it has Pillow):
    python comfy_img2img.py views_dir sheet.png --prompt "silver hair, short hair, red eyes, white coat"
        [--crops crops.json] [--only head_000,torso_000] [--denoise 0.4] [--seed 7]

--prompt says what the character looks like (hair, eyes, outfit). --crops names the regions of the
character sheet to take as references, {"face": [left, top, right, bottom], "front": ..., "side": ...,
"back": ...}; without it the whole sheet is the reference for every view.
"""
import argparse
import json
import os
import shutil
import time
import urllib.request
import uuid

from PIL import Image

SERVER = "http://127.0.0.1:8188"
COMFY = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "ComfyUI_windows_portable", "ComfyUI")

CHECKPOINT = "wai_illustrious_v170.safetensors"
CONTROLNET = "cn_union_promax.safetensors"
IPADAPTER = "ip-adapter-plus_sdxl_vit-h.safetensors"
CLIP_VISION = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"

# The references a view can ask for (reference_for); each is a region of the character sheet.
REFERENCES = ("face", "front", "side", "back")

# The style every view is drawn in; the character's own look comes from --prompt.
POSITIVE = (
    "masterpiece, best quality, amazing quality, anime coloring, solo, {character}, flat color, cel shading, "
    "clean lineart, sharp details, grey background"
)
NEGATIVE = (
    "lowres, worst quality, low quality, blurry, jpeg artifacts, realistic, photorealistic, 3d render, "
    "shiny skin, specular highlights, glossy, dramatic lighting, shadow, text, watermark, signature, "
    "extra limbs, deformed"
)


def reference_for(view):
    yaw, tile = view["yaw"], view["tile"]
    if tile == "top" or 135 <= yaw <= 225:
        return "back"
    if yaw in (90, 270):
        return "side"
    return "face" if tile in ("head", "face") else "front"


def view_hint(view):
    yaw, tile = view["yaw"], view["tile"]
    parts = []
    if tile == "top":
        parts.append("from above, top of head, hair")
    elif 135 <= yaw <= 225:
        parts.append("from behind, back view")
    elif yaw in (90, 270):
        parts.append("from side, profile")
    elif yaw in (45, 315):
        parts.append("three quarter view")
    else:
        parts.append("front view")
    parts.append({"head": "portrait, face focus", "torso": "upper body", "hips": "cowboy shot",
                  "legs": "legs focus, boots", "face": "close-up, face focus, detailed eyes"}.get(tile, ""))
    return ", ".join(p for p in parts if p)


CHARACTER = "1girl"  # replaced by --prompt


def workflow(color, depth, ref, seed, denoise, prefix, hint, use_ipadapter, cn_type, steps, extra_neg, ipa_weight, cn_strength=0.6):
    wf = {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": f"{POSITIVE.format(character=CHARACTER)}, {hint}", "clip": ["ckpt", 1]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": f"{NEGATIVE}, {extra_neg}" if extra_neg else NEGATIVE, "clip": ["ckpt", 1]}},
        "color": {"class_type": "LoadImage", "inputs": {"image": color}},
        "depth": {"class_type": "LoadImage", "inputs": {"image": depth}},
        "latent": {"class_type": "VAEEncode", "inputs": {"pixels": ["color", 0], "vae": ["ckpt", 2]}},
        "cn": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROLNET}},
        "cn_type": {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["cn", 0], "type": cn_type}},
        "cn_apply": {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["pos", 0], "negative": ["neg", 0], "control_net": ["cn_type", 0],
            "image": ["color" if cn_type == "tile" else "depth", 0], "strength": cn_strength, "start_percent": 0.0, "end_percent": 0.8,
            "vae": ["ckpt", 2]}},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["ckpt", 0], "seed": seed, "steps": steps, "cfg": 5.5, "sampler_name": "dpmpp_2m",
            "scheduler": "karras", "positive": ["cn_apply", 0], "negative": ["cn_apply", 1],
            "latent_image": ["latent", 0], "denoise": denoise}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["ckpt", 2]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }
    if use_ipadapter:
        wf.update({
            "ipa": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": IPADAPTER}},
            "clipv": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP_VISION}},
            "ref": {"class_type": "LoadImage", "inputs": {"image": ref}},
            "ipa_apply": {"class_type": "IPAdapterAdvanced", "inputs": {
                "model": ["ckpt", 0], "ipadapter": ["ipa", 0], "image": ["ref", 0], "weight": ipa_weight,
                "weight_type": "linear", "combine_embeds": "concat", "start_at": 0.0, "end_at": 0.8,
                "embeds_scaling": "V only", "clip_vision": ["clipv", 0]}},
        })
        wf["sample"]["inputs"]["model"] = ["ipa_apply", 0]
    return wf


def post(path, payload):
    req = urllib.request.Request(SERVER + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


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
    ap.add_argument("views")
    ap.add_argument("sheet")
    ap.add_argument("--only", default="")
    ap.add_argument("--denoise", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-ipadapter", action="store_true")
    ap.add_argument("--cn-type", default="depth", choices=["depth", "tile"])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--prefix", default="gen_")
    ap.add_argument("--extra-prompt", default="")
    ap.add_argument("--ref-image", default="", help="override the sheet crop: path[:left,top,right,bottom]")
    ap.add_argument("--ipa-weight", type=float, default=0.45)
    ap.add_argument("--cn-strength", type=float, default=0.6)
    ap.add_argument("--input-prefix", default="color_", help="image to start from, e.g. clay_ for untextured renders")
    ap.add_argument("--extra-negative", default="")
    ap.add_argument("--prompt", required=True, help="what the character looks like: hair, eyes, outfit")
    ap.add_argument("--crops", default="", help="JSON file: the sheet's regions to use as references")
    args = ap.parse_args()
    global CHARACTER
    CHARACTER = args.prompt

    views_dir = os.path.abspath(args.views)
    with open(os.path.join(views_dir, "views.json")) as f:
        views = json.load(f)["views"]
    if args.only:
        wanted = set(args.only.split(","))
        views = [v for v in views if v["name"] in wanted]

    input_dir = os.path.join(COMFY, "input", "rt")
    os.makedirs(input_dir, exist_ok=True)
    sheet = Image.open(args.sheet).convert("RGB")
    crops = {}
    if args.crops:
        with open(args.crops) as f:
            crops = json.load(f)
    for key in REFERENCES:
        (sheet.crop(tuple(crops[key])) if key in crops else sheet).save(os.path.join(input_dir, f"ref_{key}.png"))
    if args.ref_image:
        path, _, box = args.ref_image.partition(":")
        ref = Image.open(path).convert("RGB")
        if box:
            ref = ref.crop(tuple(int(v) for v in box.split(",")))
        ref.save(os.path.join(input_dir, "ref_custom.png"))

    for i, view in enumerate(views, 1):
        name = view["name"]
        shutil.copy(os.path.join(views_dir, f"{args.input_prefix}{name}.png"), os.path.join(input_dir, f"color_{name}.png"))
        shutil.copy(os.path.join(views_dir, f"depth_{name}.png"), os.path.join(input_dir, f"depth_{name}.png"))
        wf = workflow(f"rt/color_{name}.png", f"rt/depth_{name}.png", "rt/ref_custom.png" if args.ref_image else f"rt/ref_{reference_for(view)}.png",
                      args.seed, args.denoise, f"rt/{args.prefix}{name}", f"{view_hint(view)}, {args.extra_prompt}" if args.extra_prompt else view_hint(view),
                      not args.no_ipadapter, args.cn_type, args.steps, args.extra_negative, args.ipa_weight, args.cn_strength)
        start = time.time()
        prompt_id = post("/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})["prompt_id"]
        result = wait_for(prompt_id)
        shutil.copy(result, os.path.join(views_dir, f"{args.prefix}{name}.png"))
        print(f"[{i}/{len(views)}] {name} ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
