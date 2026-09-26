# Troubleshooting

Symptoms, their causes and fixes. Before changing anything, **measure**:

```bash
python -m app status out/MyOC/character.json     # the verdict, each step's result and error code
python -m app sheet out/MyOC/character.json      # contact sheets of the finished file
blender -b out/<a step's>.blend -P scripts/check_avatar.py
```

`check_avatar.py` reports vertices with no weights, and edges that stretch or squash when the head
turns, the arms go to the T-pose, hang, reach forward and rise, the spine and chest twist and
bend, a leg lifts, and each hair, cloth or hat chain swings alone, with the bone weights at both
ends of the worst edges. Blender renders run no spring physics: hair problems that show only in
a tracking app show up here, in `python -m app springs`, or in the viewer.

Reading a swing report: edges between two neighbouring hair chains stretch in a single-chain swing
because the neighbour stays still; real springs move neighbours together, so that is expected.
Suspicious edges have one end weighted to `Head`/`Neck` or to nothing, or sit inside the head.

"Rebuild" below means rerun from the `build` step: `python -m app run <manifest> --from build`.

## Hair

| Symptom | Cause | Fix |
|---|---|---|
| A strand partly stays still when the head moves | vertices with no weights (`unweighted vertices` > 0) | the `hair` step fills them; rerun from `hair` |
| Thin straight strips poke out of the hair in the tracking app only | hidden hair faces inside the head joining two strands | the `hair` step deletes them |
| Hair ties or beads float beside swinging hair | separate hair-coloured pieces weighted to chest or shoulder bones | the `hair` step; or give the piece the role `hair` |
| Long hair has no bounce, is stretched by the arms, or floats beside the hands | the hair is one surface with the body, so it wasn't found as hair | give it the role `hair`; hair inside a bigger piece by painting it as a part of the piece (Parts page) |
| Hair swings in front of the body when looking up, or flings violently | spring settings | tune stiffness and drag in the tracking app's physics settings |
| Hair clips into the body | VRM colliders are only spheres and capsules on bones | a few percent of clipping is normal; tune in the tracking app |
| The strand on top of the head doesn't bounce | no bones on it | option `ahoge` on (default) |

## Body and clothing

| Symptom | Cause | Fix |
|---|---|---|
| A sleeve, mantle or bracelet stays behind when the arm moves | the piece isn't on the arm's bones | run `movement`; give the piece the role it suggests (`follow_bone`, `shoulder_cloth`) |
| A skirt or coat stretches toward the hands | the hands rest on it in the modelled pose | role `fitted_skirt` on that piece |
| Hat brim flaps like hair, the bottom of a round head stays behind | headwear found as hair | option `rigid_head`; `hat_tip` for a pointed hat |
| Cape or coat tails hang stiff | no cloth springs | option `cloth`, or role `swing_cloth` on the piece |
| A wing, tail or big ear bends with the body | found as body | role `appendage` |
| `Hips` near the knees or ankles, legs don't bend | a robe, long skirt or hanging panel hides the legs | the build falls back to typical proportions and says so; place the joints on the Joints page |
| Sleeves fused to the ribs, a web between arm and torso | arms too close to the body in the model | generate the model in a T-pose |
| Chest caves in or folds when the arms move | arm weights on the chest | use Tripo's rig (the rigged FBX), or check the shoulder joints on the Joints page |
| Whole limbs move with the hands | skinning failed on this mesh | rebuild; if it repeats, try option `skinning: "voxel"` |
| Hands come out misshapen | finger detection on curled or touching fingers | option `fingers: "none"` for mitten hands; or the guided hand setup (`docs/HAND_SETUP.md`) |
| Arms sink into the body with the arms down | the tracking app's idle pose lowers the arms further than the model allows | pick an idle pose with the arms further out |
| The body turns against the head, or the avatar shakes when still | tracking settings, not the rig | lower body movement from face tracking; raise tracker smoothing; even lighting |
| No chest bounce | off by default | option `chest`, and `chest_amount` |

## Face

| Symptom | Cause | Fix |
|---|---|---|
| No blink, no face tracking, eyes don't move | no face points | mark them on the Face page, rebuild |
| Eyes don't follow gaze in the tracking app | the app is turning eye bones the avatar doesn't have | in Warudo set eye movement to blendshapes |
| The eye step fails: "iris shows over the lids" | the drawn iris is as tall as the eye's opening | `eyes: "slide"` (the app switches by itself when eyes are on automatic) |
| A character keeps building sliding eyes | the fallback is remembered | set `eyes: "iris"` explicitly |
| Mouth barely opens, looks smeared, or the face dents when talking | the face bends round the marked mouth | Face page: "redraw the talking mouth shapes", and "free mouth" for simple or chibi faces |
| Mouth too small | size of the drawn mouth | Face page: mouth "size ×" |
| Mouth inside is red on a non-human face | default mouth colours | option `mouth_colour: "skin"` |
| Face soft or blurry | the generator's texture gives the face few pixels | the face patch (made by "Get it ready"); or the redraw in `docs/agent/retexture.md` |
| Painting the hair changes the face behind it | a face patch made by an older version | remake the face patch (Face page) |
| A colour is wrong, a smudge, a seam | the generator's texture | paint it on the Paint page |

## The model going in

| Symptom | Cause | Fix |
|---|---|---|
| The avatar lies down, or the face is on the back of the head | the file's axes | detected and corrected automatically; if the survey says the orientation is unsure, turn the model upright in Blender and re-export |
| The survey says "placed from typical proportions" | clothing hides the crotch or the legs | check the joints on the Joints page |
| Every bone imports broken from a Tripo rigged GLB | its bind matrices are wrong | export the rigged model as FBX |
| Model looks shiny in ordinary viewers | the generator's metallic map | `scripts/matte_materials.py`; the exported avatar is toon-shaded either way |

## Error codes

A failed step's log is matched against `app/errors.py`; the Build page and `status`
(`steps[].known`, or `steps[].last_failure.known`) show the code, what happened and what to do.

| Code | Meaning | Fix |
|---|---|---|
| `VM-OOM` | Blender ran out of memory | close other programs; decimate a model over ~100k triangles; free space on the system drive |
| `VM-DISK-FULL` | under 1 GB free where the step writes | `python -m app prune <manifest>`, or "Free up space" on the Characters page |
| `VM-TIMEOUT` | a step ran too long and was stopped | rerun it alone |
| `VM-BLENDER-OLD` | Blender older than 4.2 | choose a newer one on the Setup page |
| `VM-NO-VRM-ADDON` | the VRM Add-on isn't installed in the Blender the app runs | install "VRM format" in that Blender (Get Extensions) |
| `VM-EXPORT-POSE` | the written `.vrm`'s joints aren't where the rig's are | use a tested VRM Add-on version; rerun |
| `VM-TRIPO-MIX` | `'ShaderNodeMix' object has no attribute 'image'` | run `scripts/matte_materials.py` on the model first |
| `VM-NOT-HUMANOID` | joint detection found no standing body | check the model; place joints on the Joints page |
| `VM-ORIENT-UNSURE` | which way the model stands is undecided | turn it upright in Blender, re-export |
| `VM-EYES-REFUSED` | the eye step's own check refused the iris | `eyes: "slide"` |
| `VM-UNSUPPORTED-FILE` | the model type isn't readable | FBX, GLB/glTF or OBJ |
| `VM-BAD-OUTPUT` | a step ended without a usable file | rerun; a repeat is a bug |
| `VM-STEP-DAMAGE` | a step changed something that belongs to another step (`app/contracts.py`) | the reason names what changed; a bug in that step's script |

## Commands failing

| Symptom | Fix |
|---|---|
| A command fails in PowerShell | quoted program paths need `&` in front |
| `export_scene.vrm` missing when running a script by hand | the VRM Add-on is off: start Blender with `-P scripts/_startup.py`, or without `--factory-startup` |
| A glTF exporter traceback prints during export but it says FINISHED | harmless; check the `.vrm` loads in the viewer |
| Blender hangs with two runs at once | run one Blender at a time |
