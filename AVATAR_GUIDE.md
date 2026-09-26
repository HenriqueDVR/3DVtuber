# VTuber avatar guide (for AI coding agents)

Give this file to an AI coding agent (Claude Code or similar) opened in this repository. It
walks a person from an original character (OC) to a 3D VRM avatar streaming in Warudo.

## How you work, agent

You are the **guide** for a person who may never have used Blender. Run the steps below in
order. Each step says who does it:

- **Agent** steps: run them yourself, then show the person the result (a render or the numbers).
- **Person** steps: happen in apps you can't drive (Tripo, Warudo, Blender's weight paint, a
  paint program). Give a **hand-off**: the exact clicks and settings, what file to put where,
  and what to tell you when done. Then wait.

Every step ends at a **checkpoint**, a condition you confirm before moving on. When a checkpoint
fails, stay on that step: reach for `docs/agent/troubleshooting.md`, which maps symptoms to causes
and fixes.

Be candid about limits. Some results need hand work (texture touch-ups, weight painting), and
the pipeline is an early prototype tested on a handful of characters. When something looks wrong,
measure it (`scripts/check_avatar.py`, renders) before changing rules, and stop after two
failed attempts at the same visual problem to ask the person how they want to proceed.

Ask before every download, install, purchase or account action, stating name, source and size.

Explain each step in a sentence or two of plain language before running it.

## Environment

- Windows, commands for PowerShell. Blender is usually at
  `C:\Program Files\Blender Foundation\Blender 4.2\blender.exe`; find it if not. Commands
  below use `$blender = "<that path>"`. Blender scripts take their own arguments after `--`.
- Blender 4.2+ with the **VRM Add-on for Blender** installed (Preferences → Get Extensions → "VRM
  format"). `python -m app setup` (the app's Setup page) says whether both are there and which
  Blender runs. Scripts run by hand that export VRM need the add-on on, so run them without
  `--factory-startup`; the app starts Blender clean and switches the add-on on itself.
- The browser viewer needs a local server from the repo root:
  `python -m http.server 8321 --bind 127.0.0.1` (inside the app it is the Review page).
- Tests: `blender -b --factory-startup --python-exit-code 1 -P tests/run_tests.py`.
- Every fix script reads a `.blend` and writes a **new** `.blend` and `.vrm`, leaving its input
  alone. Keep a chain of named files (`my_oc_built`, `my_oc_fixed`, `my_oc_jiggle`, …) and always
  feed the newest into the next script.
- Keep the person's art and models in `samples/` and outputs in `out/`; never commit either.

## Steps

**Steps 6 and 7 through the app (preferred).** `python -m app.server`, then on Characters: New
character, pick the model (the FBX of step 3, or the GLB steps 4-5 made from it), **Create the
character**. It surveys the model, lists its pieces and makes the face patch in one go, and lands
on the Face page: the person marks the twelve points there (or presses **Place them for me** and drags any
point the detector put wrong; it refuses faces it can't read), then **Save and build**. The Build
page ends with a verdict, what is in the file, and what is left with a button for each; "Watch
it" beside a failed check opens the viewer in that pose. As an agent, the same is `python -m app
init`, `prepare`, `face`, `run` (`docs/agent/app.md`). The by-hand commands under steps 6 and 7
are for pinning something down, and for the older one-shot build.

### 1. Scope check · Person + Agent

Ask for the OC art. Compare it with what this pipeline handles well:
- humanoid, standing in a **T-pose** (A-pose also works), arms clear of body and hair, gap between the legs
- medium hair hanging free of the arms (long hair works, with more fixing)
- close-fitting outfit; coats with tails and sleeves joined to the body need extra hand work
- few loose accessories (straps, ties, clips each cause weight bugs)
- a hat, helmet or solid head instead of hair, and a cape or coat tails: supported with build
  options (step 7); a panel hanging between the legs fools joint detection (fixable, step 7)
- creatures, wings, tails, big dresses, props: out of scope

**Checkpoint:** the person knows which parts of their design are likely to need extra work, and
has either simplified the design or accepted that.

### 2. Front view art · Person

One clean full-body front view in a **T-pose** on a plain background (or ask Tripo for a T-pose when
generating). A T-pose builds better than an A-pose: the arms stand clear of the body, so sleeves
don't fuse with the ribs, and the bake to the VRM rest pose barely moves them. An A-pose still works. A single consistent view beats
a hand-drawn turnaround: drawn side and back views disagree with each other and the generator
averages the disagreement into a worse model.

**Checkpoint:** a front view image in `samples/`.

### 3. Generate the model in Tripo · Person (Agent assists)

Tripo (tripo3d.ai, paid plan for streaming use; free models are public and non-commercial).
Hand-off: generate a **P2** model from the **front view** alone, with Tripo's own texture. Then
press **Rig**, pick "VRM 1.0 humanoid", and download it as **FBX**. That one file holds the model,
its texture and Tripo's skeleton and weights, which usually bend better than the app's own; the
app finds the rig in it by itself. Never Tripo's rigged GLB: its bind
matrices are broken and every bone imports at NaN. An unrigged GLB still builds, rigged by us.
P2 gives clean topology with separate parts (hair, clothes, eye inserts, mouth insert, sculpted
eye sockets).

Don't route it through an HD model or multiview: HD fuses hair and clothes into the body, its
texture smears when baked or projected onto P2, and multiview from drawn or rendered views came
out worse than the front view alone.

Steps 4-6 take this FBX as it is and write GLB copies of the mesh. Each copy names the FBX as
its rig, so the app, given the last copy as the model, still finds the rig: keep the FBX where
it is (or move it along, beside the copy). No second, unrigged download is needed.

**Checkpoint:** `samples/my_oc_p2.fbx` exists; agent renders it
(`scripts/render_turnaround.py -- samples/my_oc_p2.fbx samples/my_oc_views`) and the person
agrees the design reads as their OC.

### 4. Matte materials · Agent

```powershell
& $blender -b -P scripts/matte_materials.py -- samples/my_oc_p2.fbx samples/my_oc_matte.glb
```

**Checkpoint:** output exists; materials report metallic 0, roughness 0.9, base colour and normal
map still linked (straight into the shader, with no grey multiply node).

If a coloured part (skin, a mascot's body) fades to near white round the sides or back in the
renders, tint it from its front, giving a point on the part in the model's own space:

```powershell
& $blender -b -P scripts/retexture/match_front_colour.py -- samples/my_oc_matte.glb samples/my_oc_tinted.glb --at 0,-0.09,0.83
```

For a round head or a helmet sitting on a collar, add `--underside --close`: Tripo paints the
collar onto the part's underside and leaves it open at the bottom, which shows when the head tilts.

Render the head views again (`scripts/retexture/render_views.py`) to show the result.

### 5. Face texture · optional branch · Agent + Person

Tripo's P2 face is usually soft because its texture layout gives the face few pixels. If the
person wants a sharper face, follow `docs/agent/retexture.md` (needs an NVIDIA GPU with 6 GB+
and ~20 GB of downloads). Otherwise continue with the matte model.

**Checkpoint:** the person picked a face they're happy with, as a `.glb` in `samples/`.

### 6. Mark the face · Person (only after step 5)

The points are read in the face patch's image space, so this step needs a face patch. Without
the step 5 redraw, make one from a plain render of the face (any Python with Pillow):

```powershell
& $blender -b --python scripts/retexture/render_views.py -- samples/my_oc_matte.glb tools/work/views 1024
python -c "from PIL import Image; Image.open('tools/work/views/color_head_000.png').convert('RGB').resize((2048, 2048), Image.LANCZOS).save('tools/work/face_patch_head_000.png')"
& $blender -b --python scripts/retexture/face_patch.py -- samples/my_oc_matte.glb tools/work/views head_000 tools/work/face_patch_head_000.png samples/my_oc_face.glb
```

A character with no drawn mouth still needs 4 mouth points: put a small box where a mouth would
be. It sets the head's pivot height, and talking opens a painted mouth there, at least half an
eye wide however small the box.

Agent starts the viewer server and opens
`http://localhost:8321/viewer/setup.html?image=/tools/work/face_patch_head_000.png`.

Hand-off: click 4 points per eye (inner corner, outer corner, top and bottom of the opening, where
lashes meet the white) and 4 on the mouth (left, right, top, bottom of the lip line). Zoom with
the scroll wheel, drag to adjust, then Save. `face.json` lands in Downloads.

**Checkpoint:** `face.json` exists with both eyes and the mouth.

### 7. Build · Agent

**Survey the model first** and tell the person what it says; it decides most of the options
below and names what will need hand work:

```powershell
& $blender -b -P scripts/preflight.py -- samples/my_oc_face.glb
```

**Then build.** One command runs the survey, the build, the hair fixes, the ahoge spring, face
tracking and the check:

```powershell
python scripts/make_avatar.py samples/my_oc_face.glb --face "$env:USERPROFILE/Downloads/face.json" --name MyOC --author "Their name"
```

It picks `--rigid-head`, `--cloth`, `--mouth-colour skin` and `--free-mouth` itself, writes a
`.blend` per step into `out/`, and ends with the check's verdict. It rigs the model itself:
Tripo's rig from step 3 is only adopted through a character manifest (`python -m app init
samples/my_oc_face.glb --name MyOC`, then `python -m app run`; see `docs/agent/app.md`), which
finds it in the FBX or through the GLB steps 4-6 made from it. Pass `--joints` when detection
needs correcting and `--extra "--hat-tip"` for anything it can't decide. Steps 7 to 9 below are
what it runs; do them one at a time when something needs pinning down.

```powershell
& $blender -b -P scripts/make_vrm.py -- samples/my_oc_face.glb out/my_oc_built.vrm --name MyOC --author "Their name" --face "$env:USERPROFILE\Downloads\face.json" --keep-eyes --blend out/my_oc_built.blend
```

`--keep-eyes` leaves the eyes as modelled (no blink); automatic blink is unsolved, see "Known
limits". Add options from the scope check:

- `--rigid-head`: a hat, helmet or solid head instead of hair (no hair chains, head pieces rigid)
- `--hat-tip`: with `--rigid-head`, a pointed hat's tip swings on a stiff spring
- `--cloth`: a cape, coat tails or skirt hanging below the hips swings on springs
- `--mouth-colour skin`: a non-human face (a mascot, a blue demon); the mouth inside takes the
  face's colour instead of red and pink
- `--joints samples/my_oc_joints.json`: joints detection got wrong (see below)

Then check it:

```powershell
& $blender -b out/my_oc_built.blend -P scripts/check_avatar.py
& $blender -b -P scripts/render_vrm.py -- out/my_oc_built.vrm out/shots_built
```

The build reports what it had to guess. A `BUILD WARNING` about "typical proportions" means
clothing hid the legs and the hips and leg bones are estimates within a few centimetres: look at
them against the mesh in the `.blend` before accepting them. To place them exactly, measure knee
height (knee pads, the narrowest point) and ankle height, write `{"hips": [x, y, z], "left_upper_leg": [...],
"right_upper_leg": [...], "left_lower_leg": [...], "right_lower_leg": [...], "left_foot": [...],
"right_foot": [...]}` in the built avatar's space (hip joint about as far above the knee as the
knee is above the ankle, hips a little above that), and rebuild with `--joints`.

**Checkpoint:** the build report line `VTUBER_MAKER_REPORT` shows the chains you expect (hair,
`cloth_chains`, `hat_tip`) and `"face_points": true` when `--face` was passed; the build didn't
stop with "bone heat weighting failed at every proxy resolution"; `check_avatar.py` shows 0
unweighted vertices, no head-turn stretch over 3 cm, and at most a few dozen edges over 3 cm in
the arm and body poses; the renders look like the model. Swinging one hair chain alone stretches edges shared with the
neighbouring chain; that part of the report is expected, unless the edge sits inside the head
or face.

### 8. Weight check in Blender · Person (Agent guides)

Hand-off, in Blender, opening `out/my_oc_built.blend` (not the VRM, so VRM settings stay attached):

1. Click the armature → Pose Mode (Ctrl+Tab). Select `LeftUpperArm`, press R, then Y, type
   `-60`, Enter. Bend `LeftLowerArm` the same way. Turn `Head` with R, Z.
2. Look for anything left behind or stretched: underarms, hips, cuffs, straps.
3. To fix: Object Mode, click armature, Shift+click mesh, Weight Paint mode. Ctrl+click a bone.
   Brush weight 0, strength 1, paint where things shouldn't follow that bone. Blur brush along
   borders. Turn on X symmetry.
4. For a few vertices: Edit Mode, select, N → Item → Vertex Weights, remove the wrong groups.
5. Alt+R clears the pose. Save as `out/my_oc_fixed.blend`.

Offer the automatic fixes first when the check points at a known cause (troubleshooting file).

**Checkpoint:** arms and head move without stretching in Blender, file saved under a new name.

### 9. Physics and fixes · Agent

Run in this order, each on the previous output. Skip what doesn't apply.

| Script | When |
|---|---|
| `add_chest_jiggle.py -- out.blend out.vrm [--amount 1.0] [--strength 0.8]` | person wants chest jiggle (in the app: Options, "Chest bounce" and its amount slider) |
| `rebuild_hair_physics.py -- out.blend out.vrm` | after adding jiggle (adds chest colliders); keeps cloth and hat tip springs |
| `fix_hair_weights.py -- out.blend out.vrm` | always: fills unweighted vertices, hair ties, far strands, hidden faces |
| `add_ahoge_spring.py -- out.blend out.vrm` | a strand stands up on top of the head |
| `tune_avatar.py -- out.blend out.vrm [--mouth 1.35] [--head-up 0.03] [--chest-stiffness 0.35] [--chest-drag 0.15]` | mouth too small, head pivots too low, chest jiggle too stiff or floaty |

```powershell
& $blender -b out/my_oc_fixed.blend -P scripts/fix_hair_weights.py -- "$PWD\out\my_oc_hair.blend" "$PWD\out\my_oc_hair.vrm"
```

Rerun `check_avatar.py` on the last `.blend`.

**Checkpoint:** 0 unweighted vertices, no head-turn stretch, and the viewer's
"Swing head and body" (`http://localhost:8321/viewer/index.html?model=/out/<file>.vrm`) shows
hair and chest moving with nothing left floating.

### 10. Warudo · Person (Agent guides)

1. Install Warudo (free, Steam).
2. Menu → Open Data Folder → copy the final `.vrm` into `Characters`.
3. Onboarding Assistant → pick the character → MediaPipe for face tracking. Pose tracking only if
   arms will be on screen.
4. Test: turn and tilt the head, talk, lean quickly, move arms if tracked.

Tuning lives in Warudo, not the model:
- jitter → raise tracker smoothing, light the face evenly from the front, 720p/30 fps+, steady camera
- small movements too large → lower head movement/rotation multipliers, calibrate in normal seat
- body turning against the head → Warudo's body movement from face tracking; turn it down
- spring feel → adjust in Warudo's character physics settings while watching

**Checkpoint:** the person streams a test scene and lists anything still wrong. Take each item
to `docs/agent/troubleshooting.md`.

## Known limits

- **Blink and face tracking** need the face points. The blink is a squeeze of the eye, not drawn
  lids.
- **Eye movement:** a moving iris for modelled eyes, sliding eye patches for painted ones. In
  Warudo set eye movement to blendshapes.
- **Mouth:** a generated mouth is a flat plate; talking opens a painted interior, and the lips
  barely move. The Face page's "redraw the talking mouth shapes" gives cleaner shapes.
- **Fingers** track, but less well than a hand-made avatar's.
- **Underarm:** cloth joined between the arm and torso stretches with the arms raised high.
  A T-pose model avoids most of it.
- **Hair clipping:** VRM springs collide only with spheres and capsules; a little hair still dips
  into the body while swinging.
- **Joints under clothing:** a robe or a panel between the legs hides the crotch, so the hips and
  leg bones are placed from typical proportions and the build says so. Check them on the Joints
  page.
- **VRM 0.x** isn't exported, so VSeeFace isn't supported. Use Warudo or VNyan.
- **A new design can bring new problems.** Measure before fixing: `python -m app status`, the
  contact sheets, and `python -m app regress` to see that a change left other characters alone.
