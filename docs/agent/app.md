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

python -m app render out/MyOC/character.json --pose raised   # PNGs of the rig in a pose
python -m app sheet out/MyOC/character.json              # contact sheets of the finished .vrm
python -m app movement out/MyOC/character.json           # pieces left behind or torn by the arms and head
python -m app springs out/MyOC/character.json            # the finished file's springs: droops, floating pieces, clipping
python -m app runs out/MyOC/character.json               # every run folder, newest first
python -m app compare before_check.json after_check.json # two rig checks, pose by pose

python -m app approve out/MyOC/character.json --note "looks right in Warudo"   # only when the user says so
python -m app warudo out/MyOC/character.json             # copy the .vrm into Warudo's Characters folder (the user's call)
python -m app regress out/MyOC/character.json            # rebuild in a scratch folder and compare with its good build
python -m app disk out/MyOC/character.json               # what the runs take; `prune [--dry-run]` frees old runs' models
python -m app bundle out/MyOC/character.json --note "sleeve stays at the ribs"   # everything about a problem, in one folder
python -m app vendor                                     # fetch the app's copies of three.js and three-vrm if missing
```

## A new character

The shortest path is `init`, `prepare`, face points (`face`, `face-guess`, or the Face page),
`run`. The app does the same: "Create the character and get it ready" runs `prepare` and lands on
the Face page; "Save and build" there runs the chain.

`prepare` makes the **face patch**: a copy of the model whose face has its own sharp texture,
mapped from a front view. The patched copy becomes the manifest's `source`; the model before it
becomes `reference`. Face points are pixels in that picture. `run` without face points still
builds, with no blink and no face or eye tracking.

## The steps

In order: `survey`, `build`, `hair`, `ahoge`, `face_tracking`, `loose`, `fingers`, `pieces`,
`finger_guides`, `tpose`, `chest`, `eyes`, `mouth`, `check`, `export`, `vrm_check`, `movement`,
`springs`. `status` says which will run and why the others won't.

- Steps up to `mouth` write a `.blend` each; only `export` writes the `.vrm`, from the last
  `.blend`. After `run --to tpose` or `--only eyes` there is no new `.vrm` until `export` has run.
- Each run gets its own folder under `<out>/runs/`; nothing is overwritten. A step reads the file
  the step before it last produced (from `state.json`), never the newest file on disk. Rerunning
  a step marks every later step `stale`.
- A step that stops the chain keeps its last good output; the failure is kept beside it as
  `last_failure`, with an error code (`docs/agent/troubleshooting.md`, "Error codes").
- Only one Blender runs at a time: the engine holds `out/.blender.lock`. Don't start Blender
  yourself while a run holds it.
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
(a shape key, geometry, finger angles, bones) fails with `VM-STEP-DAMAGE`. A new step, or one that
starts cutting geometry or reshaping keys, goes into `contracts.GEOMETRY` / `KEYS` / `FINGERS`.

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

With `eyes` left to the survey, modelled eyes get a moving iris and painted eyes sliding patches.
When the eye step refuses an iris (it would show over the lids), the engine reruns it with sliding
eyes and remembers that for the character.

## Texture paint

The Paint page paints on the source model: brush, eraser, soften, clone, fill, colours picked from
the model, shade / light / recolour modes, both sides at once. The brush is measured on the
surface in 3D and spreads over the surface it is on, so it keeps its size, crosses texture seams
and leaves alone what lies behind.

Paint is kept as one transparent PNG per texture, in the manifest's `paint` folder; the generated
texture is never changed. `survey` and `build` read a copy of the source with the paint laid over
it, so saving paint marks `build` onward stale. The source must be a `.glb`, which it is once the
face patch exists.

The face patch has two pictures: the face, and hair lying over the face on its own copy, so each
can be painted without the other. A character whose patch was made by an older version shares one
picture between them: remake the patch on the Face page (the face points stay valid).

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
starts a run or writes a file is a POST with the page's own header. A second start shows the app
already running. The page's three.js and three-vrm are the copies in `app/gui/vendor/`, pinned by
SHA-256 (`app/vendor.py`), so the app needs no internet connection.
