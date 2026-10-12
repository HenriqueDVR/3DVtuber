# Driving a build with `python -m app` (for agents)

A character is one **manifest** (`character.json`, schema in
`addon/vtuber_maker/pipeline/manifest.py`): everything the user decided, from joints and piece
roles to face points and eye and mouth modes. The engine (`app/engine.py`) runs the build chain
from it, one Blender run per step, and records what each step produced in `<out>/state.json`.
The desktop app and you use the same commands.

Every command prints one JSON object on stdout; `run` streams progress on stderr as JSON lines.
Exit code 1 means the command failed (`{"error": ...}`); a build whose checks found problems
still exits 0: read `check_failed` and `verdict`.

## Commands

```bash
python -m app setup                                      # Blender, VRM Add-on, disk, memory: what is missing and what to do
python -m app init samples/model.fbx --name MyOC         # writes out/MyOC/character.json
python -m app prepare out/MyOC/character.json            # survey + piece listing + face patch, whichever is missing
python -m app face out/MyOC/character.json face.json     # keep face points with the character
python -m app detector                                   # is the face detector installed? --install puts it in tools/face_detector
python -m app face-guess out/MyOC/character.json --save  # face points placed by a detector, to check and nudge
python -m app run out/MyOC/character.json                # the whole chain
python -m app run out/MyOC/character.json --from build   # or --to STEP, --only STEP
python -m app status out/MyOC/character.json             # options, plan, what each step made, the verdict

python -m app pieces out/MyOC/character.json             # every piece, with a suggested role and why
python -m app piece out/MyOC/character.json hat --label 12 --role hat --tip
python -m app piece out/MyOC/character.json appendage_38 --rename "left wing"
python -m app joint out/MyOC/character.json hips 0 0 0.9 # override a detected joint (--clear to undo)
python -m app set out/MyOC/character.json eyes '"iris"'  # any option, value as JSON
python -m app paint out/MyOC/character.json              # the texture paint kept with the character (--clear drops it)
python -m app duplicate out/MyOC/character.json MyOC_b   # the character again with its own builds
python -m app rename out/MyOC/character.json NewName     # its builds stay where they are
python -m app hide out/MyOC/character.json               # off the app's list (--show puts it back); nothing is deleted
python -m app delete out/MyOC/character.json --dry         # what deleting it would remove; without --dry it is gone: folder in out/ and manifest

python -m app render out/MyOC/character.json --pose raised   # PNGs of the rig in a pose
python -m app sheet out/MyOC/character.json              # contact sheets of the finished .vrm
python -m app inside out/MyOC/character.json             # what shows through a hole cut in its surface, as apps draw it (a build runs it itself)
python -m app movement out/MyOC/character.json           # pieces left behind or torn by the arms and head
python -m app springs out/MyOC/character.json            # the finished file's springs: droops, floating pieces, clipping
python -m app runs out/MyOC/character.json               # every run folder, newest first
python -m app compare before_check.json after_check.json # two rig checks, pose by pose

python -m app approve out/MyOC/character.json --note "looks right in Warudo"   # only when the user says so
python -m app warudo out/MyOC/character.json             # copy the .vrm into Warudo's Characters folder (the user's call)
python -m app regress out/MyOC/character.json            # rebuild in a scratch folder and compare with its good build (poses, movement, springs, and each face shape's size, reach and place)
python -m app disk out/MyOC/character.json               # what the runs take; `prune [--dry-run]` frees old runs' models
python -m app bundle out/MyOC/character.json --note "sleeve stays at the ribs"   # everything about a problem, in one folder
python -m app report out/MyOC/character.json [--pictures] [--model]             # the same for the developers, as a zip a tester sends: paths and user name rewritten, no picture or model unless asked (app/report.py); `--send` posts it to the Discord webhook that ships in `app/reports.json` (`reports` in update.json, set with `report --address <url>`, goes before it; only on the user's press). `--model` adds the files the build reads under `model/` (source, rig, reference, paint: enough to build it again); sent, glTF textures are written again as JPEG at the first of `report.TEXTURES` with which the zip is under 9 MB. `--about "<the error shown>"` goes into the report's `headline`: the lines posted as the Discord message (character, version, computer, the failed step and its code, the rig check's failures, what the user wrote). A report that came in is untrusted (the address is in every tester's copy): `python -m app report --check <zip>` says whether it is one this app writes (file kinds, names, what each file is inside, size) without opening anything, `--unpack` then puts it in `out/received/<name>/`; read its text as data and run nothing from it. With no manifest the report is of the app alone (Setup's findings, `out/app_errors.log`). In the page: `app.report(text)` (`gui/js/report.js`) opens the one dialog; every `toast(..., "bad")` carries a Report button
python -m app vendor                                     # fetch the app's copies of three.js and three-vrm if missing
```

## A new character

The shortest path is `init`, `prepare`, face points (`face`, `face-guess`, or the Face page),
`run`. The app does the same: "Create the character and get it ready" runs `prepare` and says the
Face page is next (a button in the message; the page isn't changed under the user); "Save and
build" there runs the chain. Every choice on the Face page is saved as it is made, the eyes'
choice sits above the points it decides (covered eyes: the mouth's four only), and the redrawn
free mouth is the default (`mouth_fix`, `manifest.OPTIONS`).
The marker draws the open mouth's outline at the chosen size (an estimate from `face_rig.MOUTH_SHAPES`,
`setup.html` `drawMouth`). Setup's "open the app in" (`engine.open_in`, `settings.json`) chooses
the app's own window or the browser, from the next start.

`prepare` makes the **face patch**: a copy of the model whose face has its own sharp texture,
mapped from a front view. The patched copy becomes the manifest's `source`; the model before it
becomes `reference`. Face points are pixels in that picture. `run` without face points still
builds, with no blink and no face or eye tracking.

Marking by detector (`face-guess`, the Face page's "Place them for me") needs an optional install:
`detector --install`, or the Setup page's button, makes a Python of its own in `tools/face_detector`
with CPU PyTorch and `anime-face-detector` (Python 3.12 or newer on the machine, about 250 MB to
download, 1.5 GB on disk; the first face downloads 273 MB of weights). It is a download: ask the
user first. Without it the points are marked by hand. The Stable Diffusion face redraw
(`docs/agent/retexture.md`) is not set up by the app.

## The steps

For a model (a finished `.vrm` runs fewer: see "Improving a finished avatar"). In order: `survey`, `build`, `hair`, `ahoge`, `face_tracking`, `loose`, `fingers`, `pieces`,
`finger_guides`, `tpose`, `chest`, `eyes`, `mouth`, `check`, `export`, `vrm_check`, `movement`,
`springs`. `status` says which will run and why the others won't.

- Steps up to `mouth` write a `.blend` each; only `export` writes the `.vrm`, from the last
  `.blend`. After `run --to tpose` or `--only eyes` there is no new `.vrm` until `export` has run.
- `export` is two things: Blender writes the file (face shapes stored sparse, pictures asking for
  mipmaps: `vrm_export.export`), then the app's own Python makes its pictures no bigger than the
  avatar uses (`app/slim.py`: the face patch cut to the rectangle its triangles read, a 4096 px
  normal map halved). `steps[export].report.pictures` says what was cut or halved, or `left` and
  why (no Pillow, nothing to do). A `.vrm` exported by hand with `scripts/export_vrm.py` gets the
  first only, and a `.vrm` taken in as it is keeps its own pictures.
- Each run gets its own folder under `<out>/runs/`; nothing is overwritten. A step reads the file
  the step before it last produced (from `state.json`), never the newest file on disk. Rerunning
  a step marks every later step `stale`.
- A step that stops the chain keeps its last good output; the failure is kept beside it as
  `last_failure`, with an error code (`docs/agent/troubleshooting.md`, "Error codes").
- One run at a time: the engine holds `out/.blender.lock` for it, and no Blender of a run outlives
  it. Don't start Blender yourself while a run holds it. Inside a run (since 2026-10-12, SETUP.md
  "One Blender for a build's steps; the last steps side by side"; the files are the same byte for
  byte either way):
  - the steps that start from a `.blend` are told to **one Blender the run keeps**
    (`engine._Worker`, `scripts/_worker.py`) instead of each starting its own. A step's output is in
    its own log as before; the kept Blender's own is `<run>/worker.log`. Off: `"one_blender": false`
    in `out/settings.json`, or `OC2VRM_ONE_BLENDER=0`.
  - the **last steps** (check, export, vrm_check, movement, springs, the portraits) run **side by
    side**, each in a Blender of its own, when the computer has room (`Project._room`: the memory
    still to be had against what each step's Blender took in this character's last build, `peak_mb`
    in `state.json`; two processor cores idle; four at most). A step that runs short of memory or
    fails in company is run again by itself, and the run goes on one at a time. Never more than N:
    `python -m app setup --blenders N` (`auto` to undo; `"blenders"` in `out/settings.json`,
    `OC2VRM_BLENDERS`); 1 is one at a time. Setup's "Speed" line says what there is room for now.
  - events: a step run side by side says `running` as it starts, `ahead` as it is over, and
    `done` or `failed` when the chain takes it, in the build's order.
- Every Blender the engine starts runs with `--factory-startup` and `scripts/_startup.py`, so none
  of the user's own add-ons load and the VRM Add-on is on. To run a chain script by hand the same
  way: `blender -b --factory-startup -P scripts/_startup.py in.blend --python-exit-code 1 -P scripts/<step>.py -- ...`.
- Which Blender runs: `--blender`, `$BLENDER`, the one chosen on the Setup page (`setup --use
  PATH`), PATH, then the best installed one.
- A run warns when memory or disk is short, doesn't start a step with under 1 GB free where it
  writes, and stops a Blender that runs out of either.

Every step's output is checked where it is made. A `.vrm` must have VRM 1.0's required humanoid
bones on joints the mesh is skinned to, and is read back and compared with the armature. Each
`.blend` gets a fingerprint (`app/contracts.py`): a step that changes what another step owns
(a shape key, geometry, finger angles, bones) or opens the surface without cutting it (border
edges, `open_edges`) fails with `VM-STEP-DAMAGE`. A new step, or one that starts cutting geometry or
reshaping keys, goes into `contracts.GEOMETRY` / `KEYS` / `FINGERS`. The T-pose step, whose bake
parts every sleeve cut's two sides, counts its own slits instead: copies of a vertex led by one
bone that lie apart after the bake (`loose.parted_copies`) stop it.

The rig check (`check`, and `vrm_check` on the exported file) judges, beside stretch and folds in its
poses: seams that open (copies of a vertex over 1 cm apart in a pose or with every chain swung
together: a tear in hair or a cape), hair hanging from the head bone alone, what is the head's
growing when head and neck tilt together (a ledge across the hair), a surface over 20 mm behind
beside an iris (the eye's white deep in its socket) and the drawn mouth turning inside out in mixes
of the tracker's shapes. Each has its limit and the build it came from in `scripts/check_avatar.py`
(`LIMITS`); `python -m app compare` and `regress` say when one of them moved.

The `vrm_check` step ends with one more, by the app's own Python and a browser without a window
(`app/inside.py`, `viewer/inside.html`): the exported file drawn by three-vrm from 21 cameras round the
face, and counted where the first surface on a ray is one its picture cuts a hole in (alphaMode MASK: the
eye's frame). What shows through over 12 mm behind the hole, over 60 mm2 of it in the worst camera, is an
issue in the verdict ("as apps draw it: ..."): an eye white deep in its socket, which rays can't tell
from a blended patch standing off the face. `steps[vrm_check].report.seen` has the numbers (`deep_mm2`,
the camera, and `hair_inside_mm2`: the hair's inside seen beside the neck, measured and not judged), or
`left` and why when the file couldn't be drawn (no Edge or Chrome): then nothing is held against the
build. Its pictures are in `<run>/15_vrm_check_seen/`.

## Piece roles

A generated model is made of separate surface pieces. Most are fine on automatic; a role says how
a piece should move when detection gets it wrong.

| Role | Stage | What it does |
|---|---|---|
| `body` | build | ordinary skinning; listing a piece keeps it off hair and cloth chains |
| `hair` | build | spring strands |
| `hat` | build | wholly on the head; `"tip": true` swings a pointed tip |
| `swing_cloth` | build | cloth spring chains (cape, coat tail, loose skirt) |
| `appendage` | build | wing, tail, big ear: wholly on the core bone it grows from (`bone`, default nearest), one spring chain per end; small pieces stuck on it ride along |
| `fitted_skirt` | pieces | stays on the torso, cut free of the hands resting on it |
| `shoulder_cloth` | pieces | mantle: blended onto the upper arms, collar and hem stay |
| `follow_bone` | pieces | wholly on `bone` (hanging sleeve, bracelet, prop) |
| `detail` | pieces | rides `anchor` (cuff trim, button) |

A piece is named by a surface point and its vertex count in the source model (normalized: 1.6 m
tall, facing -Y, +X = character left). `pieces` lists both for every piece; `--label` copies them.
A piece given another role leaves what the old role was told (`bone`, `anchor`, `tip`) unless the same
call says it again: a hanging sleeve's arm bone, kept, hung the tail it was then called from the arm.
Under the same role they stay as they are; the page takes a hat's tip off again with `"tip": false`.

**`auto_pieces`** (option, off by default; Options page "Fix clothing the arms leave behind"): the
`pieces` step first runs the movement check on the model as it stands (`<run>/07_pieces_movement.json`)
and gives every piece it flags the role it suggests, if that is `follow_bone`, `shoulder_cloth` or
`fitted_skirt` (`engine.found_pieces`: what "Apply this fix" on the Movement page writes). Pieces
named in the manifest keep their role. The roles found are kept beside the step
(`07_pieces_manifest.json`, pieces marked `found`), not written into the character's manifest; the
step measures the rig before and after (`loose.worst_stretch`) and a tear, or any failure with the
found roles, runs it again without them (a warning on the step). Roles the `build` step acts on
(hair, hat, swing_cloth, appendage) are never given this way.

A part fused into a bigger piece (a twin-tail modelled as one surface with the body) takes a role
through `region`: the vertex numbers of that part (`piece NAME --label L --role hair --region
part.json`, or painted on the Parts page). Take all of the part, ribbons and hidden side included:
whatever is left out keeps the piece's weights and tears from it.

## Rig source and input pose

Recommend Tripo's rigged **FBX** ("VRM 1.0 humanoid") as the model itself, generated in a
**T-pose**. `init` finds its skeleton and sets `rig`; its bones and weights are adopted
(`pipeline/adopt.py`) and the finger steps are skipped. Tripo's rigged GLB has broken bind
matrices: don't use it. A separate rig file for an unrigged model works too (`init --rig`).
Mixamo and Character Creator / AccuRig skeletons are read when named as `rig`, but only tested on
synthetic rigs. Without a rig the build places and skins a skeleton itself; an A-pose still builds.

Eye tracking (`eyes`) and mouth tracking (`face_tracking`) are separate: either, both or neither.
`eyes: "hidden"` is for eyes behind a visor, glasses or a mask: the face points are then the mouth's
four alone (a `face.json` without eye points means the same), and there is no blink.

`hair_hits_body: false` makes the hair's springs collide with the head alone, for hair that rests on the
shoulders (it is lifted up its chains when the head tilts onto them); long hair wants the default.

With `eyes` left to the survey, modelled eyes get a moving iris and painted eyes sliding patches.
When the eye step refuses an iris (it would show over the lids), the engine reruns it with sliding
eyes and remembers that for the character.

The user's word over what the eye step reads (Face page, "Fine-tune the eyes"; all take effect from
the `eyes` step on, `eye_shown` from `face_tracking`):

| Option | Values | What it does |
|---|---|---|
| `eye_shown` | null, `"left"`, `"right"` (the character's own side) | the other eye is covered (hair, an eyepatch): left as modelled and painted, no blink, no look keys. Both eyes are still marked, the covered one where it would be |
| `iris_shape` | null (as drawn), `"circle"`, `"oval"` | modelled iris discs only: the shape painted, lowest point kept |
| `iris_size` | 0.7 to 1.4 | modelled iris discs only: a factor on the size read; never wider than the disc |
| `look_travel` | 0.5 to 1.5 | a factor on how far the eyes look, every eye mode |
| `blink_amount` | 0.5 to 1.4 | a factor on the blink keys (`eyes.scale_blinks`); needs the eye step to run |

A modelled iris is read as an oval (`eyes.drawn_circle`: centre, half width, half height), never
flatter than a circle. A changed option can make the eye check refuse the step (an iris showing
outside the eye): the log's `FAILED:` lines say in which look.

## Improving a finished avatar (.vrm)

A character whose `source` is a `.vrm` is **taken in as it is** (`manifest.improves`,
`pipeline/improve.py`, `scripts/import_vrm.py`): its skeleton, weights, expressions, springs,
colliders, node constraints, toon materials and metadata all stay, its meshes stay apart, and
nothing is rigged again. VRM 1.0 or 0.x; what is written is always VRM 1.0.

**A VRM 0.x file** is converted on the way in (`improve.convert`). The VRM Add-on fills the 1.0
bones, expressions, springs and colliders as it imports one and turns the model to face the 1.0
way; the name, author and terms, the first-person mesh settings and the eyes' range it leaves at
its defaults, and those are carried over here. Terms never come out looser than the file had them:
what 0.x can't state (political, hate use) stays refused, "commercial use allowed" becomes a
person's, a licence named "Other" keeps its address and the strictest fields beside it.

**Its licence.** The add-on won't import a VRM 0.x file whose licence forbids changed versions
(`CC_BY_ND`, `CC_BY_NC_ND`) or is a web page it can't read, until a person has looked. `init`
answers with `licence` (`engine.licence`: whose avatar, the pages, `confirm`, `why`); the build
stops with `VM-LICENCE` and the address until the user has said they read the terms and may change
the avatar: option `licence_read` (the page asks when the character is made, and on Options).
That is the user's to say: never set it for them. A VRM 1.0 file's terms the add-on doesn't judge.

```bash
python -m app init avatar.vrm --name MyAvatar     # no prepare: there is nothing to get ready
python -m app set out/MyAvatar/character.json chest true   # optional
python -m app run out/MyAvatar/character.json     # about a minute; the file given is not changed
```

The steps that run: `build` (here: read the file), `chest` when asked for, `check`, `export`,
`vrm_check`, `springs`. Every step that would make what the file already has is skipped, with the
reason in `status` (`engine.KEPT`). The Parts, Joints, Face, Paint and Movement pages have nothing
to do for such a character. Only the options `licence_read`, `chest` and `chest_amount` change
anything in it (`manifest.VRM_OPTIONS`): `set` on another (`skinning`, `tpose`, `fingers`...)
answers with a `note`, and the run warns of them at `build`.

What the build does to the file:

- **Joints straightened.** Every joint is written unrotated, on the model's axes, as VRoid
  exports them: tracking apps turn a bone about its own axes, and Warudo twists the fingers of a
  file whose joints rest turned (any file saved from Blender by the VRM Add-on's own menu). Node
  constraints are carried: a roll axis is renamed to the model axis it lies along; an aim joint
  keeps its frame, since its axis has to point at its source at rest.
- **Chest bounce** (`chest`, `chest_amount`): two spring bones, on the mesh the chest is on.
- **Checked like a build**: the rig check and the spring check on all its meshes together, and
  contact sheets.

`status` says what the file came with (`source_holds`), what it gained (`gained`) and the
verdict. Anything the file had that the export lacks (a bone, a face shape, a spring, a material,
the blink) is an **issue**. What the rig check finds is a *look*: the skinning is the file's own
and no step here remakes it.

Not done to such a file: new face shapes, eye tracking, hair springs for a file without them, a
T-pose bake, finger work. The humanoid bones carry the pipeline's names inside Blender and get
their own names back in the exported file.

## A model that came rigged (MMD as FBX)

A character whose `source` is an FBX with an MMD rig (bones as the Cats plugin names them, or in
Japanese: `manifest.rig_kind` says "MMD") is **taken in with its own rig** (`manifest.keeps_rig`,
`pipeline/kept.py`, `scripts/import_rigged.py`): skeleton, weights, face shapes and materials stay,
its bone chains become springs, its face shapes are bound to the VRM expressions by name, its eye
bones look. `prepare` answers `kept_rig: true`. Steps that run: `build` (the import), `tpose`,
`check`, `export`, `vrm_check`, `springs`; `status` gives the reason for each of the others
(`engine.KEPT_RIG`). Options read: `keep_rig`, `height` (only for a model saved in other units),
`tpose`, `repair_shoulder_seams`, `hair_hits_body`; any other set option is listed as `inert`.

`keep_rig: false` builds the same file as a bare surface, as before. The build step's report says
what was found: `naming`, `hair_chains`, `cloth_chains`, `stiff_chains`, `expressions`,
`eye_bones`, `look_at`, `tracking_shapes` (the ARKit-named face shapes made of its own, for trackers),
`see_through` (per material: `CUT`, cut out by its picture, drawn blended with depth written; `BLEND`, a soft layer;
`OPAQUE`, the `_solid` copy that holds the polygons the picture doesn't cut).
"MODEL NOT TAKEN IN" in its log: a body bone is missing under every table of names.

## Texture paint

The Paint page paints on the source model: brush, eraser, soften, clone, fill, colours picked from
the model, shade / light / recolour modes, both sides at once. The brush is measured on the
surface in 3D and spreads over the surface it is on, so it keeps its size, crosses texture seams
and leaves alone what lies behind.

**Isolate** (H) shows one part alone (Shift+click adds another; "Show all" or H brings the rest
back): the brush then reaches what another part covers, and only the shown parts take paint.

The view takes the page; **Expand** (or F) gives it the whole window. Undo and redo are Ctrl+Z and
Ctrl+Y. Paint that isn't saved yet is kept as a draft a few seconds after each stroke
(`<out>/paint/draft/`) and comes back when the page is opened again.

Paint is kept as one transparent PNG per texture, in the manifest's `paint` folder; the generated
texture is never changed. `survey` and `build` read a copy of the source with the paint laid over
it, so saving paint marks `build` onward stale. The source must be a `.glb`, which it is once the
face patch exists.

The face patch has two pictures: the face, and hair lying over the face on its own copy, so each
can be painted without the other. A character whose patch was made by an older version shares one
picture between them (`paint` reports `old_patch`, and the Paint page offers the remake): remake
the patch, on the Paint or the Face page; the face points stay valid and saved paint is kept. A
patch remade once the face points exist also fills the skin hidden under hair, outside the marked
eyes and mouth, from the face beside it.

## Looking at a build

- **`status`**: the verdict (`ready`, `issues`, `looks`), what the finished file holds (`ships`:
  bones, fingers, blink, vowels, face-tracking shapes, look-at, springs), and `since_approved`:
  what got worse or better than the build the user marked good.
- **Contact sheets** (`sheet`, the Review page): the exported file drawn with three-vrm in a
  headless Edge or Chrome: the face blinking, looking and talking; the body in the check's poses
  from four sides; neck, hands and shoulders close up. The closest thing to a tracking app that
  runs here. Read the PNGs.
- **Renders** (`render --pose raised`, also `t-pose`, `forward`, `hang`, `rest`): the rig posed in
  Blender. No spring physics.
- **The check's close-ups** (`steps[check].report.spots`): the worst stretch and squash, at rest
  and in the pose that did it.
- **The viewer** (`viewer/index.html?model=/out/<...>.vrm`, the Review page): springs running,
  the check's poses (`&pose=arms%20raised`), skin weights by bone (`&weights=LeftUpperArm`), bones,
  springs and colliders, and a webcam face-tracking test.
- **Notes from the tracking app** are kept per build in `<out>/feedback/`, with what was ticked
  and a verdict; they are in every `bundle`.

Marking a build good (`approve`) and copying it into Warudo (`warudo`) are the user's calls.

## When something looks wrong

1. `status` and the step's log (`steps[].log`). A failed step names its log and its error code.
2. Look: contact sheets, `render --pose raised`. A clean check doesn't mean it looks right.
3. `movement` lists the pieces that stay behind or tear when an arm or the head moves, each with
   the role that usually fixes it.
4. Is it a *which piece* problem? Most character-specific fixes are: a sleeve left at the ribs, a
   hat that should be rigid, a cape that should swing. Give the piece a role, rerun from `build`
   or `pieces`. That is a manifest change, not code.
5. Is it a wrong joint? Override it with `joint`, rerun from `build`.
6. Check `docs/agent/troubleshooting.md` for the symptom.
7. Only when no role or option covers it is it a pipeline change: follow `AGENTS.md`'s working
   rules, and run `regress` on every character you have before and after.
8. Stop after two failed attempts at the same visual problem and ask the user.

A new kind of piece that keeps coming up deserves a new role: add it to `manifest.ROLES`, the
repair to `pipeline/loose.py`, the dispatch to `pipeline/roles.py`, a test to
`tests/test_manifest.py`.

## What the app can't decide

Whether the hands, the face art and the spring feel are good enough, and whether it looks right in
the tracking app. Ask the user for a screenshot per pose.

## The server

`python -m app.server` serves the page on `127.0.0.1` only. A GET only reads; everything that
starts a run or writes a file is a POST with the page's own header. A request a browser marks as
coming from another site (`Sec-Fetch-Site`) is refused, but a link to the first page; every answer
says `nosniff`, same-origin use only and no framing by another site (`server.GUARDS`); the viewer
opens by address only a model this server hands out. A program that is no browser (an agent's
`urlopen`, curl) sends no such mark and is answered as before. A second start shows the app
already running. The page's three.js and three-vrm are the copies in `app/gui/vendor/`, pinned by
SHA-256 (`app/vendor.py`), so the app needs no internet connection.

## First start

A computer that lacks something a build needs opens on **Setup** (`main.js`; its check rows are
`gui/js/start.js`); one that has it all opens on Setup once, its first start (`welcomed` false), and on Characters after. The rows are drawn from `engine.setup()`: a check that a button can fix carries
`action` (`install_blender`, Windows and Linux x64: `engine.install_blender` downloads the build in
`engine.BLENDER_BUILDS` from blender.org into `tools/blender`, checks its pinned SHA-256, unpacks it
as a portable copy, chooses it and installs the add-on in it; `get_blender` elsewhere (macOS) opens
blender.org; `install_addon` has
the chosen Blender install the VRM Add-on from its extensions site, `engine.addon_command`;
`install:<package>` pip-installs one of `engine.PACKAGES`; `vendor` keeps the pages' libraries).
Each install is a download: it runs only when the user presses its button and confirms. The app's
own downloads go through `app/trust.py` (`open_url`): a certificate Python refuses on Windows for an
expired intermediate kept in Windows' store is checked once more against the roots alone. `welcome`
(POST) remembers the page was finished (`out/settings.json`, `welcomed`). `OC2VRM.bat` offers
Python (winget) and Pillow before the app starts; `oc2vrm.sh` (Linux, macOS; `OC2VRM.command` for
Finder) says how to get Python and offers Pillow. A new Blender LTS means new `BLENDER_ZIP` and
`BLENDER_TAR` (address, size and the hash from blender.org's `.sha256` file beside them). The
add-on is looked for in each system's own Blender settings folder (`engine.blender_settings`).
**macOS and Linux have not been run**: only what the tests can show on Windows is checked. For people trying the app:
`docs/TESTERS.md`.

## Updates and the testers' copy

The repository is private, so the app updates itself from GitHub with a key (`app/update.py`).
`update.json` beside the app (never committed; `update.example.json` shows it): `repo`, `ref` (the
branch testers follow) and `token`, a fine-grained key the owner makes with read-only access to
this one repository's contents.

```bash
python -m app pack [--at origin/testers]   # dist/OC2VRM-<commit>.zip (--at: the released commit, when main has gone on since): the committed files, their version, update.json, music.* if there
python -m app update               # is a newer commit published on that branch?
python -m app update --apply       # get it (the Setup page's Update button)
```

An update downloads the published commit's zip (about 8 MB) and lays it over the app's own files.
`out/`, `tools/`, `samples/`, `dist/`, `update.json`, the settings and a `music.*` are never written
or removed; replaced files go to `out/update/backup_<old>/`; a file the last version brought and
this one doesn't is removed. `.oc2vrm_version.json` says which commit a packed copy is. A running
launcher isn't written over: its new copy waits as `<name>.new` and the launcher takes it when the
app restarts (the server exits with code 75 and `OC2VRM.bat` / `oc2vrm.sh` start it again). A git
checkout is updated with `git pull --ff-only` instead. Publishing to testers is moving the branch
named in `ref`: `git push origin main:testers`.

**An update trusts the branch**: nothing is checked beyond the connection to GitHub, so whoever can
push to the branch named in `ref` runs code on every tester's computer at their next Update. Not
changed yet; SETUP.md "Security review of what testers run" has what would close it and what the
shipped key allows.

## The look

Two looks, chosen on Setup ("Look") and kept in the browser (`localStorage`; `?look=classic` or
`?look=arcade` in the address overrides it for one visit):

- **Arcade** (the default): a fighting game's menus. Characters is a character select (a band in
  the character's colour, a roster), opening one shows a main menu (Set up / Build / Check), a
  card opens its pages as a list, and the page list along the left becomes a trail along the top.
  `app/gui/arcade.css` (every rule under `html.arcade`) and `app/gui/js/look.js`.
- **Classic**: `app.css` alone, the app as it was, the page list down the left side.

The pages are the same in both: `look.js` adds its own screens (`#menu`, `#pages`, the band on
Build) and reads what the pages already drew (the sidebar's badges, the Characters cards, the run
bar), called from three hooks (`app.shown`, `app.navDrawn`, `app.homeDrawn`). A new page needs an
entry in `look.js`'s `GROUPS` and `TITLES` to be reachable in the arcade look. "Less motion" on
Setup stops the moving background and the effects. The arcade look has sound (`js/sfx.js`:
made from oscillators, no audio files; the sliders on Setup, all the way left is off): the menus,
ordinary buttons, a note for each build step that finishes, a fanfare for "Avatar ready", and
music (low and muffled behind a work page): `music.mp3`, `.ogg` or `.wav` in the repository's
folder if there is one (the server's `/music`; the user's own file, not part of the repository),
else a loop the page makes itself. The top bar has a switch for each.

Build in the arcade look shows the band and the steps as tiles; the step table, logs, history,
result and rig check are the same elements, folded under "Details" (a tile opens its step
there). The tiles and the band read a running build from the page's Progress lines (`#events`).
The viewer (`viewer/index.html`) dresses the same when the app embeds it in this look.

The pictures are the character posed: `scripts/render_portraits.py` draws it standing with its arms by
its sides (`stand`, for character select) and in a gesture for each menu card: a wave
(`setup`), an arm flexed (`build`), a thumb up (`check`). The poses are made in the script (`POSES`: directions for
the limbs, the palm's facing, each hand's shape), so they work on a rig bound in a T or an A.
`<out>/portraits/`: four PNGs and
`portraits.json`; about five seconds. A build draws them once the `build` step has made the rig
and again at its end from the finished one; `python -m app portraits <manifest>` or the buttons on
character select draw them for a character built before. Every picture is framed alike and the
JSON says where the head and floor are in it, so `look.js` `art()` puts any character in any pose
on the same spot at the same size, head on the middle line. `summary()` hands them out as
`portraits`. A character without them shows a plain still render (`portrait`, `stills`), measured
from the picture; never a T-pose, in either case. Before any build there is `first_look.png`
(`engine.FIRST_PICTURE`): the model as it came, from the front, drawn when its pieces are listed
(`list_pieces.py --picture`, so `prepare` makes it), which may well be a T-pose.

The fonts (Anton, Chakra Petch) come from Google Fonts; offline it falls back to Impact
and Segoe UI.
