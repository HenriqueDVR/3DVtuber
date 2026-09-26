# Face retexture (optional branch of AVATAR_GUIDE step 5)

Redraws the face with Stable Diffusion and gives it its own high-resolution texture: sharper
than the generator's face. The recipe does the face only; redrawing the body makes little
difference.

Needs an NVIDIA GPU (tested on a 6 GB laptop RTX 3060, ~1–2 min per view) and ~20 GB of disk.

## Setup · Agent (ask before each download)

Everything lives in `tools/` (never committed). Confirm current file names and sizes on the source
pages before downloading; these are the ones that worked.

| What | Source | Put at | Size |
|---|---|---|---|
| ComfyUI portable for Windows/NVIDIA | GitHub `comfyanonymous/ComfyUI` releases | `tools/ComfyUI_windows_portable/` | ~2 GB |
| ComfyUI_IPAdapter_plus custom node | GitHub `cubiq/ComfyUI_IPAdapter_plus` | `ComfyUI/custom_nodes/` | small |
| WAI-Illustrious-SDXL v17 checkpoint | Hugging Face `LyliaEngine/waiIllustriousSDXL_v170` | `ComfyUI/models/checkpoints/wai_illustrious_v170.safetensors` | 6.9 GB |
| ControlNet Union SDXL promax | Hugging Face `xinsir/controlnet-union-sdxl-1.0` (promax file) | `ComfyUI/models/controlnet/cn_union_promax.safetensors` | 2.5 GB |
| IP-Adapter Plus SDXL ViT-H | Hugging Face `h94/IP-Adapter`, `sdxl_models/` | `ComfyUI/models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors` | 0.85 GB |
| CLIP-ViT-H image encoder | Hugging Face `h94/IP-Adapter`, `models/image_encoder/model.safetensors` | `ComfyUI/models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | 2.5 GB |

The base Illustrious model draws loose, painterly brushwork; the WAI fine-tune gives clean cel
shading. Use the fine-tune.

Start ComfyUI with `tools/ComfyUI_windows_portable/run_nvidia_gpu.bat` (or as a background
process) and wait until `http://127.0.0.1:8188/system_stats` answers. Stop it afterwards to free
the GPU.

**Checkpoint:** ComfyUI answers on port 8188 and lists the four model files.

## Adapt the prompt · Agent

`scripts/retexture/comfy_img2img.py` holds a sample character's description in `POSITIVE`
and crops of its sheet in `SHEET_CROPS`. Rewrite `POSITIVE` for this character (hair colour and
length, eye colour, outfit pieces, keeping the quality and "flat color, cel shading, clean
lineart" tags). The sheet argument is required by the script even without IP-Adapter; pass the
front view image.

## Recipe · Agent

`$py` is `tools/ComfyUI_windows_portable/python_embeded/python.exe`.

1. Render views of the matte model (34 orthographic views with depth):
   ```powershell
   & $blender -b --python scripts/retexture/render_views.py -- samples/my_oc_matte.glb tools/work/views 1024
   ```
2. Redraw the front and three-quarter head views:
   ```powershell
   & $py scripts/retexture/comfy_img2img.py tools/work/views samples/front.png --only head_000,head_045,head_315 --cn-type tile --denoise 0.45 --steps 24 --no-ipadapter --prefix face45_ --extra-prompt "closed mouth, neutral expression, calm" --extra-negative "pout, frown, pursed lips, angry"
   ```
   Show the person `tools/work/views/face45_head_000.png` next to `color_head_000.png`.
   Strength (`--denoise`) is the main dial: 0.3 barely changes anything, 0.45 was the favourite,
   0.55 sharper lashes, 0.65+ changes the expression and the eyes stop matching the mesh.
   Try 2–3 strengths on `head_000` only and let the person choose.
3. Project the redrawn views onto the model, so the sides of the head blend in:
   ```powershell
   & $blender -b --python scripts/retexture/project_back.py -- samples/my_oc_matte.glb tools/work/views face45_ samples/my_oc_face45.glb
   ```
4. Upscale the chosen front view for the patch:
   ```powershell
   & $py -c "from PIL import Image; Image.open('tools/work/views/face45_head_000.png').resize((2048, 2048), Image.LANCZOS).save('tools/work/face_patch_head_000.png')"
   ```
5. Give the front of the face its own texture from that image (Tripo's atlas only spends ~350 px
   on the face):
   ```powershell
   & $blender -b --python scripts/retexture/face_patch.py -- samples/my_oc_face45.glb tools/work/views head_000 tools/work/face_patch_head_000.png samples/my_oc_face.glb
   ```
6. Render a face close-up and a 3/4 view of `samples/my_oc_face.glb` (`scripts/render_turnaround.py`,
   or `scripts/render_vrm.py` after the build) and show the person.

**Checkpoint:** the person approves the face on the model from the front and the side.

## Hand fixes · Person

Small flaws (a stray strand painted on the forehead, a smudge beside an eye) are faster to paint
than to prompt away. Edit `tools/work/face_patch_head_000.png` in any paint program without
resizing or moving anything, then rerun step 5.
