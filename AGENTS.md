# AGENTS.md

Guidance for an AI coding agent (Claude Code or similar) working in this repository.

## What this is

OC2VRM: a Blender add-on (4.2 and later) plus command-line scripts and a guided desktop app
that turn an image-to-3D humanoid model (Tripo P2 and similar) into a **VRM 1.0** avatar: body
rig, blink and mouth expressions, hair and cloth spring bones, optional fingers, MToon materials,
T-pose bake. Targets Warudo and VNyan. VRM 0.x (VSeeFace) is not exported.

Everything is Python and plain JavaScript; no build step beyond `blender --command extension build`
for the add-on zip.

## Read these before changing anything

| File | What it holds |
|---|---|
| `README.md` | Quick start. |
| `AVATAR_GUIDE.md` | Step by step, for walking a person from their character art to Warudo. |
| `docs/agent/app.md` | Driving a build with `python -m app`: every command, the steps, piece roles, texture paint, what to do when something looks wrong. |
| `docs/agent/troubleshooting.md` | Symptom, measured cause, fix, for nearly every defect seen so far. Check here before inventing a diagnosis. |
| `docs/agent/retexture.md` | The optional face redraw with Stable Diffusion. |
| `docs/HAND_SETUP.md` | The guided hand and finger editor. |

## Environment

- Any Blender **4.2 or later** with the VRM Add-on ("VRM format" in Get Extensions). Tested: 4.2,
  4.5 LTS, 5.2 LTS. A file saved by a newer Blender can't be opened by an older one. Version
  differences go through `pipeline/compat.py`. 5.x adds new shape keys at value 1.0 (4.2: 0):
  always set `key.value`.
- The app runs `--blender`, else `$BLENDER`, else the one chosen on its Setup page, else the
  installed one. `python -m app setup` says what it found and what is missing.
- Almost every script needs Blender's bundled Python (`bpy`, `bmesh`, `mathutils`, NumPy). Ordinary
  `python` cannot exercise deformation. The app (`app/`) and `scripts/make_avatar.py` are plain
  Python and launch Blender themselves; the app needs Pillow.
- Always pass `--python-exit-code 1`. Without it a script exception can still exit 0.
- Run Blender **one at a time**: out of memory it does not exit but spins (`Malloc returns null`
  in its log), and two at once have hung a machine. The app's engine holds a lock for this.
- The engine starts every Blender with `--factory-startup` and `scripts/_startup.py` before the
  step's file, so the user's own add-ons never load and the VRM Add-on is on before the `.blend`
  opens. By hand, do the same, or leave `--factory-startup` out: with it alone the add-on is off
  and export fails.
- In PowerShell, redirecting Blender's output prints a spurious `NativeCommandError`; trust
  `$LASTEXITCODE` and the log.
- `samples/`, `out/`, `tools/` and `dist/` hold the user's private art, builds and downloads:
  never commit or publish them, and ask before any download or install.

## Code map

- `addon/vtuber_maker/pipeline/`: the implementation. Coordinate frame and preflight
  (`normalize.py`, `orient.py`, `survey.py`), joints and surface pieces (`landmarks.py`,
  `parts.py`, `meshdata.py`), skinning (`rig.py`, `voxel_skin.py`, `adopt.py` for a rig made
  elsewhere), loose-part and clothing repairs (`loose.py`), fingers and T-pose (`fingers.py`,
  `finger_guides.py`, `hand_setup.py`, `tpose.py`), the face (`face_points.py`, `face_rig.py`,
  `face_texture.py`, `eyes.py`), physics (`hair.py`, `hair_physics.py`, `cloth.py`,
  `spring_sim.py`), `vrm_export.py`, the character manifest (`manifest.py`, `roles.py`), and
  `improve.py` (a finished `.vrm` taken in as it is and added to, instead of a model built).
- `scripts/`: Blender entry points (`make_vrm.py`, `to_tpose.py`, `apply_pieces.py`,
  `check_avatar.py`, `render_pose.py`, the `fix_*`/`add_*`/`check_*` tools, `retexture/`).
- `app/`: plain Python. `engine.py` runs a character's build chain from its manifest, one Blender
  run per step; `contracts.py` (what each step may change); `cli.py` (`python -m app`, JSON out);
  `server.py` and `gui/` (the guided app, one ES module per page in `gui/js/`); `paint.py` and
  `gui/js/paint.js` (texture paint); `sheet.py` (contact sheets of the exported file);
  `vendor.py` (pinned copies of three.js and three-vrm in `app/gui/vendor/`: never edit those).
- `tests/`: `unittest` run inside Blender against synthetic fixtures; `tests/test_app.py` is plain
  Python.
- `viewer/`: the VRM viewer, the face-point marker, the contact-sheet page.

Coordinate frame after `normalize.prepare`: 1.6 m tall, standing on Z=0, facing **-Y**, character
left **+X**. Piece roles and clothing recipes are written in the **modelled pose**, never a baked
T-pose.

## Commands

```bash
pip install -r requirements.txt                    # Pillow, for the app
python -m app.server                               # the guided app (or OC2VRM.bat)
python -m app run out/MyOC/character.json          # the whole build from a manifest
python -m app regress out/MyOC/character.json      # rebuild in a scratch folder and compare with its good build
python -m unittest tests.test_app                  # the app's tests, plain Python, seconds
blender -b --factory-startup --python-exit-code 1 -P tests/run_tests.py                      # the suite, minutes
blender -b --factory-startup --python-exit-code 1 -P tests/run_tests.py -- test_manifest.py  # one module
```

The suite has **two expected failures** (the synthetic hair/body collision tests in
`test_hair_physics.py`); anything else failing is new. The `Paint` tests of `test_app.py` skip
inside Blender, whose Python has no Pillow. Never weaken, skip or delete a test to make a change
pass. If a number moves, say so.

## Working rules

- **Measure before changing rules.** `scripts/check_avatar.py` reports unweighted vertices and
  edge stretch and squash per pose. It does not exit nonzero for quality warnings: read `failed`.
- **Numbers are not visual proof.** Render the posed model (`python -m app render --pose raised`)
  and look at the pictures, before and after. Blender renders run no spring physics.
- **Look at the contact sheets** (`python -m app sheet`): the exported file drawn with three-vrm,
  closer to a tracking app than Blender's renders, not equal to it.
- **Verify the exported file, not just the `.blend`.** The exporter can change the result.
- **Find the piece that actually stays behind before fixing it.** Pose the arms, then list the
  surface pieces near an arm at rest that barely move. Most character-specific fixes are a piece
  given the right role in the manifest, not code.
- **After a pipeline change, run `regress`** on every character you have, and say what moved.
- **Prefer the conventional VTuber solution** over an elaborate new system. Stop after two failed
  attempts at the same visual problem and ask the user.
- **Each step reads a `.blend` and writes a new one.** Feed the step's recorded output into the
  next (`state.json`), never "the newest file".
- **Preserve** hand curvature, shape keys, UVs, materials and springs unless changing them was
  asked for. Never fix a floating piece by deleting geometry.
- **Do not reset, clean or commit the working tree as cleanup**; the user decides what leaves it.
- UV seams split vertices on imported meshes: anything that walks the surface must first join
  vertices at the same place.
- A generated texture is hundreds of small patches packed edge to edge, and the patch next door
  in the picture is another part of the model. Work on the surface in 3D, not along the texture.
- A picture mapped from one view is shared by everything in line with it: check texel sharing by
  depth before blaming whatever painted it.
- Rotate arms about **world** axes when testing poses; about its own axis an upper arm twists.
- `mesh.edges.add(n)` crashes Blender 5.2: add edges through bmesh.

## The test models in comments

Comments and docs cite the models a rule was measured on by letter, not by name (the art is
private): **Model A**, a long-haired character with modelled eyes, a fitted skirt and wide
sleeves, also as a single-surface T-pose file with Tripo's rig; **Model B**, a chibi mascot with
a solid head, painted eyes, a mantle and hanging coat sleeves; **Model C**, a character with
twin-tails and a tail. Keep to the letters when adding a comment.

## Style

Match the surrounding code: NumPy-vectorised, terse, a docstring on each pipeline function saying
what it assumes about the pose and what it refuses to touch. Comments explain *why* a rule exists
(usually the defect it fixed), not what the line does.
