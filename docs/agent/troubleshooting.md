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
| Dark or white bits of the shirt slide about on the chest when the head turns or the hair swings | clothing pieces lying under a front lock were taken for riders of the lock (within 3.5 cm of it), and the tidy-up put those two chains pulled onto the head (model A: 12 pieces, 14 to 22 cm in `springs.json` `coming_away`) | fixed 2026-10-08 (`cloth.attach_riders`: a piece rides only when what it lies on swings); rebuild from `build`. The Movement test names such a piece ("turns with the head") |
| A bit of hair hangs still in the air while the hair round it swings | a small hair piece on two chains of its strand was put on the head by the tidy-up (`loose.rigidify_props`; model A: 180 vertices, 30 cm) | fixed 2026-10-08: such a bit keeps both chains; rebuild from `loose` |
| Long hair has no bounce, is stretched by the arms, or floats beside the hands | the hair is one surface with the body, so it wasn't found as hair | give it the role `hair`; hair inside a bigger piece by painting it as a part of the piece (Parts page) |
| A ledge across the hair where its springs start, when the head tilts or nods | on an adopted rig the hair above the chains was still part on the neck (model D: 20% up to the crown) while the chains hang from the head | `hair._on_the_head` puts hair that doesn't hang on the head alone; rerun from `build` |
| Hair resting on the shoulders is lifted up its chain when the head tilts onto them | the chest's and shoulders' colliders shove it (model D: 8.8 cm in the spring check; 1.6 cm with the head's collider alone) | option `hair_hits_body: false` (Options: "Hair collides with the body" off); rerun from `build` |
| A sliver of faces stands off a joint when it bends, like a tear | a crack in the generated surface: open edges whose rims are under a millimetre apart, each with its own weights | the `loose` step closes them (`close_small_holes`) before joining surfaces; rerun from `loose` |
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
| A cape folds into the body or is squeezed when the arms come down | the adopted rig weighted the whole cape to the arms it is draped over | the build takes hanging cloth off the arms by itself (`cloth.hang_from_torso`); it needs option `cloth` on (automatic when the survey finds the cape) or role `swing_cloth`. `shoulder_cloth` is for a short mantle, not a cape |
| Legs, arms or boots swing, droop or tear on a character that is one colour all over (a slime) | the body read as hair: it is the hair's colour and one surface with the head | fixed in `parts.find` (fused hair that would take the legs is refused); name the real hair pieces `hair` |
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
| The eyes can't be marked: a visor, glasses or a mask covers them | the twelve points include the eyes | Face page, Eye tracking: "eyes are covered" (`eyes: "hidden"`): only the mouth's four points are marked, the mouth tracks, no blink and no eye movement |
| Only eye tracking, or only mouth tracking, is wanted | they are separate choices | Face page: Eye tracking (`eyes`) and Mouth tracking (`face_tracking`) |
| Eyes don't follow gaze in the tracking app | the app is turning eye bones the avatar doesn't have | in Warudo set eye movement to blendshapes |
| The eye step fails: "iris shows over the lids" | the drawn iris is as tall as the eye's opening | `eyes: "slide"` (the app switches by itself when eyes are on automatic) |
| A character keeps building sliding eyes | the fallback is remembered | set `eyes: "iris"` explicitly |
| Mouth barely opens, looks smeared, or the face dents when talking | the face bends round the marked mouth | Face page: "redraw the talking mouth shapes", and "free mouth" for simple or chibi faces |
| With the webcam only the drawn mouth moves, the face round it stays; the mouth sits under the surface or slides off | the lips are on another surface than the eyes (a tester's pirate: eyes with the hat, lips with the body), so the face shapes could not move them and the patch was hung on a hidden layer | fixed 2026-10-09 (`parts.with_marked_mouth`). A build from before: run it again from `build` |
| A hat (or ears, a helmet) loses its colour once face points are marked: flat blocks of one lighter colour | no hair found above it, so the head counts as solid and was repainted to continue the face patch, as a mascot's is | fixed 2026-10-09 (`face_texture.extend_if_solid`: only a head of one colour round the patch). The build log says "face patch edge: ...". A build from before: run it again from `build` |
| `build` fails: "the face patch has no polygons" (a painted face, no modelled eyes, with face points) | reading the painted face bakes positions with the materials taken off (`texture_space.bake_position_map`); every polygon came back on the first material, the patch and the hair's picture empty | fixed 2026-10-09 (the polygons' materials are put back). Run it again from `build` |
| Mouth too small | size of the drawn mouth | Face page: mouth "size ×" |
| The drawn mouth "inverts", breaks into slivers or vanishes on a small "o" (webcam tracking), skin shows through it | shape keys add up: jaw, pucker and funnel together narrowed the patch past nothing and it came out facing backwards (18 of 111 mixes on model D at size 0.7, pucker alone on model A); the corners' pull meant for the skin also moved the patch | fixed 2026-10-08 (`add_face_tracking.py` `narrowing_kept`, the pull kept off the patch's width; `MOUTH_SURFACE_GAP` 1.2 mm). A build from before: run it again from `mouth` |
| Mouth inside is red on a non-human face | default mouth colours | option `mouth_colour: "skin"` |
| Face soft or blurry | the generator's texture gives the face few pixels | the face patch (made by "Get it ready"); or the redraw in `docs/agent/retexture.md` |
| Painting the hair changes the face behind it | a face patch made by an older version | remake the face patch (Face page) |
| A colour is wrong, a smudge, a seam | the generator's texture | paint it on the Paint page |

## The model going in

| Symptom | Cause | Fix |
|---|---|---|
| The avatar lies down, or the face is on the back of the head | the file's axes | detected and corrected automatically; if the survey says the orientation is unsure, turn the model upright in Blender and re-export |
| A character with a tail faces sideways, stands on its head, or fails with "arms don't stand clear" / joints far off | a tail held out or up behind made the body deeper than wide (the thinner axis was taken for depth, the side view's head and tail for legs); one ending above the head was taken for the body's height | fixed 2026-10-08 (`orient._across`, `orient.head_top`): width is the axis the silhouette mirrors across, height is the head's. Legs behind which a tail, cape or coat hangs are read from above as well (`landmarks._leg_pair`): knees and ankles land on the legs; the crotch can still be hidden (hips from proportions, said in the notes). A thin tail lying along an A-pose arm can still read as arms at the sides: T-pose input |
| The face patch picture shows horns (or ears, a tail) and the face sits at its lower edge or out of it; or a horned character fails with "arms don't stand clear" | what stands over the head was taken for the model's top: joints by shares of the height sat too high, and the head view was framed by a share of the body on a model taller than its body | fixed 2026-10-08 (`orient.head_top` reads the crown between horns; `cli._frame`, `render_views.py --body`): remake the patch on the Face page. Horns leaning in over the crown are not told from a hat: set the head on the Joints page |
| The head turns from halfway up: hat and eyes go, jaw, beard and hair lag | the lower head is one surface with the body and only part on the head by bone heat | fixed 2026-10-08 (`rig.follow_head_column`); rebuild from `build` |
| Legs swing with a coat or cape | body, coat and legs are one piece, and the piece was judged cloth whole | fixed 2026-10-08 (`cloth.build_cloth_chains` judges each hanging stretch); rebuild from `build` |
| One arm moves the chest; the skeleton stands beside the body's middle | a sword, tail or bag on one side pulled the centre line (the surface's median) | fixed 2026-10-08 (`landmarks.centre_line`); or set the joints on the Joints page |
| Arms raised: only hands and cuffs go up; the rig check says nothing stretches | the whole body was taken for headwear on a solid head and put on the head bone | fixed 2026-10-08 (`rig.make_head_rigid`); the rig check now fails it (`arm_follows`) |
| Ears don't move, and no piece on the Parts page is the ears | they are one surface with a hat or the head | List pieces again: `ear_left`/`ear_right` are offered under the pieces (`parts.ears`); use only for ears, not horns |
| A part painted on the Parts page moves other vertices (a sleeve, the coat) | built on our own rig, the sleeve cut renumbered the vertices | fixed 2026-10-08 (`roles.carry` by place); rebuild from `build` |
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
| `VM-LICENCE` | a VRM 0.x avatar whose licence forbids changes or is a web page the VRM Add-on can't read | the user reads the terms, then sets `licence_read` (Options: "The avatar's licence") |
| `VM-ARMS-AT-SIDES` | both arm finders failed: the arms hang against the body or inside the clothing's outline (a tester's pirate, arms down in a long coat) | generate in a T-pose |
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
