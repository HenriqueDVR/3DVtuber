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
| From the side, with the head turned or tilted, the dark inside of the back hair shows between the hair and the neck | the hair shell's rim round the neck is open (it is on every long-haired build made here); model A's copy `OC3_v3.vrm` had it and `OC3_v10.vrm`, its rim capped by hand, hadn't | measured in every build, judged in none (`steps[vrm_check].report.seen.hair_inside_mm2`: 4,911 mm2 on v3, 167 on v10, 3,900 to 5,700 on today's builds of her). No cap in the build yet |
| Rig check: "an edge stretches N cm with the head turn" at the back of the collar, under the hair (model A: 6.5 cm, 14 vertices on `Head` 1.0 below the neck joint) | `rig._keep_arms_off_chest` handed what the shoulders gave up to the vertex's other holders in their own proportions: a trace of the head (0.003) took all of it, the neighbour without a trace went to the chest, and the blur that follows smooths torso weights only | fixed 2026-10-11 (`rig.TAKEOVER_GAIN`: a bone that is not the torso's gains at most what it held; the torso takes the rest by height); rebuild from `build`. Model A: 1.8 cm, nothing over the limits |
| A rigged game or MMD model (FBX) built as a bare surface: skirt flying up on a nod, an edge 30 cm long on a head turn, "hair" two thirds of the model | its bones were under names no table knew (Cats' `Left elbow`), so its rig was thrown away and its white dress read as hair | MMD names are known now and such a model keeps its rig (`kept.py`); built bare all the same, a piece of the hair's colour ringing the body is left off the hair (`parts.goes_round_body`) |
| A model taken in with its rig: eyes blank white, hair grey | every material exported opaque; the highlight and sheen layers hid what they lie on | `kept.see_through`: picture alpha cuts or blends, as MMD draws it |
| A model taken in with its rig, in Warudo: the body a little see-through (hair or the room faint behind skin and dress), the face not | its skin and dress were on MToon cutout, which Warudo drew see-through on this model (cause not known: the file and UniVRM's shader say solid) | `kept.see_through`: polygons whose picture is opaque all over go to an opaque copy of the material; the ones a cut edge runs through are blended with depth written. Those are solid but a shade lighter in shadow: a known limit |
| Same: `lookAt` is "expression" in the file though eye bones were bound | the VRM Add-on's guess of expressions by name, made when the `.blend` is first opened | `kept.bind_expressions` switches `initial_automatic_expression_assignment` off |
| Same: long tails or a skirt hang well below their modelled shape standing still | gravity adds up over a chain of many joints | `kept.SWING_JOINTS` |
| A tail (or anything that is not hair) held up behind the head is dragged sideways when the head turns: rig check "an edge stretches N cm with the head turn", `Head` 0.8 to 1.0 on vertices half a metre behind the head (TailTest_up_behind: 21.7 cm) | `rig.follow_head_column` gave the head whatever stands above the head joint between the shoulders, bounded in width and not in depth | fixed 2026-10-11 (`rig.COLUMN_DEPTH`: only within about twice the shoulder's distance before and behind the joint); rebuild from `build`. A tail closer than that still goes to the head: give it the role `appendage` on Parts |
| Hair swings in front of the body when looking up, or flings violently | spring settings | tune stiffness and drag in the tracking app's physics settings |
| Hair clips into the body | VRM colliders are only spheres and capsules on bones | a few percent of clipping is normal; tune in the tracking app |
| The strand on top of the head doesn't bounce | no bones on it | option `ahoge` on (default) |
| The rig check says "N places where the surface is joined at rest come apart" (in Warudo: tears in the hair or a cape as it swings) | the copies of a vertex along a texture seam hold different weights: two hair chains (model A, 2026-10-07: 65 places, 3.9 cm), a cape's chain and the thigh under it (model B: 24 places, 6.9 cm). A single chain's swing never showed it: its neighbour stretches by design | the `loose` step gives such copies one set of weights since 2026-10-07: rebuild from `loose` (both read 0 after). The check swings every chain together (`swing together`) and counts copies over 1 cm apart; a build from before fails it rightly |
| The rig check says "N vertices hang from the head bone to X cm below the shoulders" | hair no spring holds: a bit on two chains of its strand that the tidy-up put on the head (model A's builds of 2026-10-07: 307 places, 19 cm down), or hair that got no chain | fixed 2026-10-08 for the first: rebuild from `loose`; hair that got no chain: role `hair` (Parts page) |
| The rig check says "N edges above the head joint grow over 5 mm when head and neck tilt together" | what is the head's is partly on the neck or the body: hair above its springs on an adopted rig (model D: 429 edges, 17.5 mm, the ledge the user saw; 30 once mended), a head our own rig shared with neck and shoulders (the two tail tests' builds of 2026-10-08: 143 and 164; rebuilt 2026-10-11: 0). Turning the head bone alone shows none of it | rebuild from `build` (`hair._on_the_head`, adopted rigs); a head that is one solid piece: Options, solid head (`rigid_head`) |

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
| The rig check said "an edge shrinks to 8% with the arms at sides" (or the leg lift) on a build that looks right | the crease beside a joint closing: the armpit as the arm comes down, the hip as the leg lifts. The VRoid-made reference measures 36 edges under half their length there, the worst 15%; both ends of every such edge are shared between the bones that move and the ones that stay | corrected 2026-10-11 (`check_avatar.py` `SHARED`, `FOLD_EDGES`): such edges are measured, not judged; a run of 5 where an end is wholly on one bone fails ("N edges fold ... where the weights step"). A build from before keeps its old line until checked again (`--from check`) |
| Pale fragments of a skirt (or of the thighs' skin) ride the wrists when the arms rise, and no check says a word | an A-pose model whose hands rest on the skirt: the rig step's sleeve cut frees the patch under each wrist together with the sleeve, so no contact is left behind and no edge stretches (model A built without her skirt named: the rig check's worst stretch 1.5 cm, the patch 52 cm off with the arms out) | the movement check now says "a hand hanging against it takes a patch of it along" (64 vertices, 104 cm) and suggests `fitted_skirt`: "Apply this fix" on the Movement page, or option `auto_pieces` |
| `tpose` stops: "N copies of a vertex along a texture seam, led by one bone, lie apart after the bake" | the copies' weights differ a little and the bake, turning the arms, moves one and not its twin: a slit (model A's lower back, 2026-10-07: 49 copies, to 29 mm, under her hair at rest) | option `repair_shoulder_seams` (on by default) gives such copies one set of weights before the bake; with it on, the step that left them apart has a bug: send a bundle |
| The rig check says "N edges fold ... where the weights step from one bone to the next", or an edge of a cape stretches with a spine twist | hanging cloth an adopted rig had on the arms is put on the torso bones at its height (`cloth.hang_from_torso`); the body's own handover from one bone to the next (model D: Hips to Spine in 7 cm) is too short 25 to 29 cm from the spine: 6.6 cm of stretch in a twist, 14 edges folded in a side bend, a step in the cape's outline | fixed 2026-10-11 (`cloth.HANG_BLEND`: the handover is half as long as the cloth is far from the spine); rebuild from `build`. Model D: 2.6 cm, none folded; to the eye the step is smaller, not gone |
| The rig check said "the upper arm takes only N% of the surface round it along when it is raised" on a character whose arms go up whole | a cape (or hair, or cloth) lying on or beside the arm was counted as the arm's surface (model D: 49%) | the check corrected 2026-10-11: it reads the arm's own surface, the piece that goes round the bone (`check_avatar.arm_follows`). A body in one piece is read as before; a sleeve left behind as a piece of its own is the movement step's to say ("stays behind") |
| On a T-pose input a mantle (`shoulder_cloth`) stands out as a wing at each shoulder when the arms come down, and the Movement test lists nothing | `loose.follow_shoulder_cloth` read "above the elbow" by height, and an arm held out has its elbow at the shoulder's height: the mantle took a third of the arm's weight; the Movement test's limit for draped cloth was 0.3 and it read 0.33 | blend changed 2026-10-11 (held-out arms: before the elbow along the arm, fading out under the arm's line) and the test's limit moved to 0.5; rebuild from `pieces`. **Not finished**: on the T-pose copy of model B the raised arms then stretch the mantle 8.2 cm and its back hem has a small step (SETUP.md "Clothing on T-pose input") |
| A pose in Blender (a render, the rig check of a `.blend`) stood up to 2 cm from the same pose of the exported file | the `.blend` held more than four bones on some vertices (the tidy-up's averaging, the blur across a sleeve cut: 1,675 of model B's 29,292); glTF holds four and the export keeps the strongest. At rest the two matched to 0.003 mm | fixed 2026-10-11 (`rig.limit_influences`, after the bake and after the chest bones): what ships is unchanged, the `.blend` is now that rig. `CHECK bones:` in the check's log says how many vertices still hold more; rebuild from `tpose` |

## Face

| Symptom | Cause | Fix |
|---|---|---|
| No blink, no face tracking, eyes don't move | no face points | mark them on the Face page, rebuild |
| Face page: the picture to mark shows only part of the face, cut along a straight edge (a tester's flat-colour FBX, a palette 2048 x 512) | Blender's FBX import wires a texture that also cuts the model as two nodes, colour and alpha; the patch replaced only the colour's, and the export wrote the atlas's alpha into the face picture (its log: "More than one shader node tex image used for a texture") | fixed 2026-10-11 (`face_patch.py` drops an alpha from a node of its own). "Remake the face patch". Too small to mark: "Bigger picture", or drag the box's bottom corner |
| In the app's 3D views the hair down the back shows over the face, slivers of what lies behind at every edge (the same FBX) | the palette's alpha is wired in, so the GLB says BLEND of the whole model and the viewer draws it without depth | fixed 2026-10-12 (`normalize.opaque_where_used`, before every GLB the app shows: alpha comes off a picture opaque wherever its polygons use it). Remake the face patch; the Parts page's model on its next list |
| The eyes can't be marked: a visor, glasses or a mask covers them | the twelve points include the eyes | Face page, Eye tracking: "eyes are covered" (`eyes: "hidden"`): only the mouth's four points are marked, the mouth tracks, no blink and no eye movement |
| Only eye tracking, or only mouth tracking, is wanted | they are separate choices | Face page: Eye tracking (`eyes`) and Mouth tracking (`face_tracking`) |
| Eyes don't follow gaze in the tracking app | the app is turning eye bones the avatar doesn't have | in Warudo set eye movement to blendshapes |
| The eye step fails: "iris shows over the lids" | the drawn iris is as tall as the eye's opening | `eyes: "slide"` (the app switches by itself when eyes are on automatic) |
| A character keeps building sliding eyes | the fallback is remembered | set `eyes: "iris"` explicitly |
| In the tracking app the eye white is far back in the head: the iris floats in the opening and the inside of the head shows from the side | the white beside a modelled iris was painted on the model's own socket, 3 cm behind the disc on the fox girl (catgirl 1.6, boy 0.6); front views and the contact sheets look right | fixed 2026-10-10 (`eyes.add_modelled_irises`: the white on a surface of its own right behind the iris); rebuild from `eyes`. The eye step's log says how deep the socket lay |
| The iris is small, with white where its lower part was | a drawn iris that is pale below (or an oval) was read as a small circle | fixed 2026-10-10 (pale pixels of the iris's hue count; ovals are read); rebuild from `eyes`. Still wrong: Face page, "Fine-tune the eyes" (`iris_shape`, `iris_size`) |
| Short hair has no springs and the whole head moves as one | the survey took the hair for headwear (it covers most of the head and hangs little), or read the face as hair before the eyes were marked | mark the face, then build from the start (`survey` reads the marks); else Options: solid head off (`rigid_head: false`) |
| Pale or shaded hair has no springs | under half of each hair piece was near the crown's commonest colour | fixed 2026-10-10 (`parts.find`: shades of the crown's colour count); rebuild from `build`; else give the pieces the role `hair` |
| One eye is under hair or an eyepatch | both eyes get an iris, a white and a blink | Face page, "Fine-tune the eyes": say which eye shows (`eye_shown`); mark both all the same |
| Mouth barely opens, looks smeared, or the face dents when talking | the face bends round the marked mouth | Face page: "redraw the talking mouth shapes", and "free mouth" for simple or chibi faces |
| With the webcam only the drawn mouth moves, the face round it stays; the mouth sits under the surface or slides off | the lips are on another surface than the eyes (a tester's pirate: eyes with the hat, lips with the body), so the face shapes could not move them and the patch was hung on a hidden layer | fixed 2026-10-09 (`parts.with_marked_mouth`). A build from before: run it again from `build` |
| A hat (or ears, a helmet) loses its colour once face points are marked: flat blocks of one lighter colour | no hair found above it, so the head counts as solid and was repainted to continue the face patch, as a mascot's is | fixed 2026-10-09 (`face_texture.extend_if_solid`: only a head of one colour round the patch). The build log says "face patch edge: ...". A build from before: run it again from `build` |
| `build` fails: "the face patch has no polygons" (a painted face, no modelled eyes, with face points) | reading the painted face bakes positions with the materials taken off (`texture_space.bake_position_map`); every polygon came back on the first material, the patch and the hair's picture empty | fixed 2026-10-09 (the polygons' materials are put back). Run it again from `build` |
| Mouth too small | size of the drawn mouth | Face page: mouth "size ×" |
| The drawn mouth "inverts", breaks into slivers or vanishes on a small "o" (webcam tracking), skin shows through it | shape keys add up: jaw, pucker and funnel together narrowed the patch past nothing and it came out facing backwards (18 of 111 mixes on model D at size 0.7, pucker alone on model A); the corners' pull meant for the skin also moved the patch | fixed 2026-10-08 (`add_face_tracking.py` `narrowing_kept`, the pull kept off the patch's width; `MOUTH_SURFACE_GAP` 1.2 mm). A build from before: run it again from `mouth` |
| The rig check says "the surface beside the left iris lies N mm behind it" | the eye's white is the model's own socket, far behind a modelled iris (the fox girl: 29 mm; 4 with the white on a surface of its own). Only over 20 mm: the catgirl's 16 is not told from a sliding patch standing off a round face (model B: 11 to 13, nothing to see) | rebuild from `eyes` (fixed 2026-10-10) |
| The verdict says "as apps draw it: N mm2 of what shows through the opening in EyeLids lies over 12 mm behind it" | the eye's white is far behind its frame: the frame is an opaque sheet with a window cut by its picture's alpha, and from the side one looks past the iris into the socket (the fox girl: 544 mm2 in the worst of 21 cameras; the catgirl, 16 mm down, which the rig check's rays could not tell from a blended patch standing off a round face: 408; every mended or other build: 12 at most) | rebuild from `eyes` (fixed 2026-10-10). `python -m app inside <manifest>` draws it again; the pictures are `inside_worst.png` and `inside_face.png` beside the report (`<run>/15_vrm_check_seen/`) |
| The verdict says "the avatar wasn't drawn to see what shows through it" | no Edge or Chrome, or a browser whose graphics can't be read back as numbers (no float render target) | nothing is wrong with the avatar: that one check was left out. `VTUBER_MAKER_BROWSER` names a browser to use |
| The rig check says "the drawn mouth turns inside out in N of 31 mixes of its shapes" | a build from before 2026-10-08: jaw, pucker and funnel narrow the patch past nothing between them (model D: 19 mixes, all of it turned; model A's copy v13: 7 mixes) | rebuild from `mouth` (0 of 31 after). A patch with sides of its own (model B's, from before the free mouth) isn't judged |
| Mouth inside is red on a non-human face | default mouth colours | option `mouth_colour: "skin"` |
| Face soft or blurry | the generator's texture gives the face few pixels | the face patch (made by "Get it ready"); or the redraw in `docs/agent/retexture.md` |
| Painting the hair changes the face behind it | a face patch made by an older version | remake the face patch (Face page) |
| A colour is wrong, a smudge, a seam | the generator's texture | paint it on the Paint page |

## The finished file

| Symptom | Cause | Fix |
|---|---|---|
| Thin drawn lines (fur, hair strands, trim) break into dots that crawl when the avatar is small on screen; close up they are fine | the file's pictures have no mipmaps: the VRM Add-on writes the min filter as LINEAR, which UniVRM reads as "none" | fixed 2026-10-11 (`vrm_export.use_mipmaps`); rerun an older build from `export`. A `.vrm` taken in as it is keeps its own filters |
| The `.vrm` is 40 to 100 MB | every face shape was stored for every vertex of the body: 30-66% of the file | fixed 2026-10-11 (`vrm_export.sparse_morph_targets`, 25-57% smaller, the model the same value for value); rerun from `export` |

## The model going in

| Symptom | Cause | Fix |
|---|---|---|
| The avatar lies down, or the face is on the back of the head | the file's axes | detected and corrected automatically; if the survey says the orientation is unsure, turn the model upright in Blender and re-export |
| A character with a tail faces sideways, stands on its head, or fails with "arms don't stand clear" / joints far off | a tail held out or up behind made the body deeper than wide (the thinner axis was taken for depth, the side view's head and tail for legs); one ending above the head was taken for the body's height | fixed 2026-10-08 (`orient._across`, `orient.head_top`): width is the axis the silhouette mirrors across, height is the head's. Legs behind which a tail, cape or coat hangs are read from above as well (`landmarks._leg_pair`): knees and ankles land on the legs; the crotch can still be hidden (hips from proportions, said in the notes). A thin tail lying along an A-pose arm can still read as arms at the sides: T-pose input |
| The face patch picture shows horns (or ears, a tail) and the face sits at its lower edge or out of it; or a horned character fails with "arms don't stand clear" | what stands over the head was taken for the model's top: joints by shares of the height sat too high, and the head view was framed by a share of the body on a model taller than its body | fixed 2026-10-08 (`orient.head_top` reads the crown between horns; `cli._frame`, `render_views.py --body`): remake the patch on the Face page. Horns leaning in over the crown are not told from a hat: set the head on the Joints page |
| Making the face patch again once the face is marked takes a minute or two; a new character's first steps (`prepare`) take half a minute to over a minute | `face_patch.py` `fill` ran every pass of its loop whenever one hidden texel had no neighbour on its own surface to be filled from (40 to 90 s of nothing changing); and `render_views.py` drew 34 views with depths of which the app reads nine and one depth (9 s) | fixed 2026-10-12: the loop stops at the first pass that fills nothing, and `cli.PATCH_VIEWS` asks for the views that are read. The patched model is the same file byte for byte on all 23 characters here that have one (a remade patch: 92 to 17 s on model A's T-pose build, 44 to 6 on the fox girl) |
| The head turns from halfway up: hat and eyes go, jaw, beard and hair lag | the lower head is one surface with the body and only part on the head by bone heat | fixed 2026-10-08 (`rig.follow_head_column`); rebuild from `build` |
| Legs swing with a coat or cape | body, coat and legs are one piece, and the piece was judged cloth whole | fixed 2026-10-08 (`cloth.build_cloth_chains` judges each hanging stretch); rebuild from `build` |
| One arm moves the chest; the skeleton stands beside the body's middle | a sword, tail or bag on one side pulled the centre line (the surface's median) | fixed 2026-10-08 (`landmarks.centre_line`); or set the joints on the Joints page |
| Arms raised: only hands and cuffs go up; the rig check says nothing stretches | the whole body was taken for headwear on a solid head and put on the head bone | fixed 2026-10-08 (`rig.make_head_rigid`); the rig check now fails it (`arm_follows`) |
| The mouth opens on the throat or chin, the drawn smile stays on the face; a visor or collar tears | the face points' mouth was marked off the face (model D: saved again 300 px down the picture) | mark the mouth again on the Face page and rebuild from `build`; the rig check fails it (`mouth_on_head`: the head bone holds under half of the surface under the mouth) and `regress` says `face shapes: ... changed` |
| Ears don't move, and no piece on the Parts page is the ears | they are one surface with a hat or the head | List pieces again: `ear_left`/`ear_right` are offered under the pieces (`parts.ears`); use only for ears, not horns |
| A part painted on the Parts page moves other vertices (a sleeve, the coat) | built on our own rig, the sleeve cut renumbered the vertices | fixed 2026-10-08 (`roles.carry` by place); rebuild from `build` |
| The survey says "placed from typical proportions" | clothing hides the crotch or the legs | check the joints on the Joints page |
| Every bone imports broken from a Tripo rigged GLB | its bind matrices are wrong | export the rigged model as FBX |
| A model decimated in Blender builds with seams torn, hair drooping 24 cm and no finger guides, or stops in the T-pose step ("Matrix.invert(ed): matrix does not have an inverse") | a generated model is cut along every texture seam; the decimator doesn't close the cuts and each patch shrinks away from its neighbours (model A's stand-in at 20,000 triangles: 1,740 pieces of 163) | `VM-FRAGMENTS` stops it before the build since 2026-10-12. Use the generator's low-poly export, or merge vertices by distance before decimating |
| The same model saved again (by another program, with a material a part, in several objects, turned) builds a little otherwise: joints a few millimetres to 1.6 cm elsewhere, a hair chain more or less; **without face points** the mouth shapes can land on another small piece (rig check: "the head bone holds only 24% of the surface under the mouth") | joints are read from 120,000 random points of the surface drawn by triangle number (`meshdata.sample_surface`, `landmarks.detect_body`): another order is another draw, and even the rounding of a file saved again (0.1 micrometres) moved them 4 mm on model A's stand-in. The pieces taken for the mouth and the hair follow the joints | not mended: it would move every character's joints (SETUP.md "Preflight, coded failures, inputs handed over otherwise, the tester's computer", Part 3). Mark the face, which places the mouth by the marks; compare two builds only when they are of the identical file. `python -m app inputs <manifest>` shows it for a model |
| An OBJ builds with no face and one flat colour; the Model page says no texture | its `.mtl` names no picture (`map_Kd`), or the picture isn't beside it | `VM-NO-TEXTURE` before the build. Export it with its texture, the picture in the same folder as the `.obj` and `.mtl` |
| Model looks shiny in ordinary viewers | the generator's metallic map | `scripts/matte_materials.py`; the exported avatar is toon-shaded either way |

## Error codes

A failed step's log is matched against `app/errors.py`; the Build page and `status`
(`steps[].failure`, or `steps[].last_failure.failure`) show the code, what happened and what to do.
The same table holds what the engine refuses before a step starts, and the lines of the look before a
build (`python -m app preflight`, `docs/agent/app.md` "Before a build: preflight"): the second table.

| Code | Meaning | Fix |
|---|---|---|
| `VM-OOM` | Blender ran out of memory | close other programs; decimate a model over ~100k triangles; free space on the system drive |
| `VM-DISK-FULL` | under 1 GB free where the step writes | `python -m app prune <manifest>`, or "Free up space" on the Characters page |
| `VM-TIMEOUT` | a step ran too long and was stopped | rerun it alone |
| `VM-BLENDER-OLD` | Blender older than 4.2 | choose a newer one on the Setup page |
| `VM-NO-VRM-ADDON` | the VRM Add-on isn't installed in the Blender the app runs | install "VRM format" in that Blender (Get Extensions) |
| `VM-EXPORT-POSE` | the written `.vrm`'s joints aren't where the rig's are, or its vertices aren't on the mesh as it rests (the add-on's "Auto" pose moved both: joints 63 to 81 mm, vertices to 98 mm on the September files) | use a tested VRM Add-on version; rerun |
| `VM-TRIPO-MIX` | `'ShaderNodeMix' object has no attribute 'image'` | run `scripts/matte_materials.py` on the model first |
| `VM-LICENCE` | a VRM 0.x avatar whose licence forbids changes or is a web page the VRM Add-on can't read | the user reads the terms, then sets `licence_read` (Options: "The avatar's licence") |
| `VM-ARMS-AT-SIDES` | both arm finders failed: the arms hang against the body or inside the clothing's outline (a tester's pirate, arms down in a long coat) | generate in a T-pose |
| `VM-NOT-HUMANOID` | joint detection found no standing body | check the model; place joints on the Joints page |
| `VM-ORIENT-UNSURE` | which way the model stands is undecided | turn it upright in Blender, re-export |
| `VM-EYES-REFUSED` | the eye step's own check refused the iris | `eyes: "slide"` |
| `VM-EYES-REFUSED` with "face over the eye patch" | a sliding patch on a model whose eyes are modelled pieces (the fox girl) | `eyes: "modelled"`, or leave eyes automatic |
| `VM-UNSUPPORTED-FILE` | the model type isn't readable | FBX, GLB/glTF or OBJ |
| `VM-BAD-OUTPUT` | a step ended without a usable file | rerun; a repeat is a bug |
| `VM-STEP-DAMAGE` | a step changed something that belongs to another step (`app/contracts.py`) | the reason names what changed; a bug in that step's script |
| (no code) | a failure the app has no name for: `failure.detail` is the log's last telling line | send a report (the Report button); the step's log is in it |
| `VM-BUSY` | another build holds the one-at-a-time lock, or a job is running in the app | wait for it or stop it; a lock left by a process that died is taken over by itself |
| `VM-NO-SURVEY` | a step was asked for before the model was surveyed | Build everything, or `run --only survey` |
| `VM-NOTHING-BUILT` | a step was asked for before any step left a `.blend` | build from the start |
| `VM-NO-EXPORT` | no exported `.vrm` yet (to check, to mark good, to send to Warudo) | build to the end, or `run --from export` |
| `VM-STATE-UNREADABLE` | `state.json` and its backup can't be read | move `state.json` aside: the builds stay, the record starts afresh |
| `VM-WARUDO-FILE` / `VM-NO-WARUDO` | Warudo's folder holds another file under the avatar's name / Warudo wasn't found | rename or remove that file / give its Characters folder |

Before a build (`preflight`): **stop** lines are asked about on the Build page, **warn** lines only said.

| Code | Kind | Said when | Fix |
|---|---|---|---|
| `VM-NO-BLENDER`, `VM-BLENDER-OLD`, `VM-NO-VRM-ADDON` | stop | no Blender found or it doesn't start; older than 4.2; no VRM Add-on in it | Setup page |
| `VM-NO-MODEL`, `VM-UNSUPPORTED-FILE` | stop | the model file isn't where the character says, or isn't FBX, GLB/glTF, OBJ or VRM | put it back, or name it again (Model page) |
| `VM-FILE-GONE` | stop / warn | the rig, the reference model or the face points file is gone (stop); the paint folder is gone (warn: the build uses the unpainted texture) | put it back, or take it off the character |
| `VM-RIG-GLB` | stop | the rig file is a GLB whose bind matrices no skeleton can have (Tripo's rigged GLB, as model E's came) | export the rigged model as FBX |
| `VM-LICENCE` | stop | a VRM 0.x avatar whose licence must be read first, `licence_read` not set | the user reads the terms and says so on Options; never set for them |
| `VM-NO-TEXTURE` | stop | the survey found no colour texture | export the model with its texture |
| `VM-ARMS-DOWN` | stop | the survey read the arms over 80 degrees below level | generate in a T-pose |
| `VM-HEAVY-MODEL` / `VM-COARSE-MODEL` | stop / warn | over 300,000 triangles / under 3,000 | the low-poly export, about 50,000 |
| `VM-FRAGMENTS` | stop | the survey counts over 1,000 separate surface pieces (the characters here have 25 to 239; model A's stand-in decimated in Blender as it came has 1,740) | the generator's own low-poly export; in Blender, merge vertices by distance before decimating |
| `VM-ORIENT-UNSURE` | stop, warn once built | the survey couldn't tell which way up the model stands or which way it faces | check the picture on the Model page; turn it in Blender, or option `facing: "file"` |
| `VM-PARTS-NOT-READ` | warn | one fused surface, no pieces to read | nothing, if it looks right built |
| `VM-NO-FACE-POINTS` | warn | the eyes and mouth aren't marked | Face page |
| `VM-FACE-FILE` | stop | the face points can't be read, or lack a point | mark them again |
| `VM-FACE-MARKS` | stop | marks that can't be a face: outside the picture, both eyes on one eye, one far above the other, the mouth off to a side, at the eyes' height, or over 1.15 eye distances below them (on the chin or throat) | drag them on the Face page. **Not caught**: a mouth marked low when the eyes aren't marked (model D's case); the rig check says it after the build |
| `VM-FACE-REFRAMED` | stop | the face patch was made again framed otherwise than the picture the marks were made on (marks saved since this was kept carry `frame`) | mark them again |
| `VM-ROLE-ON-FACE` | stop | a hair, cloth or appendage role on the piece the eyes are marked on, or on an eye or mouth piece (needs a listing made with the marks) | take the role off (the line's button) |
| `VM-HAIR-AS-HAT` | stop | half or more of what the listing reads as hair is named a hat, and the user didn't ask for a solid head | take the hat roles off |
| `VM-LISTING-STALE` | warn | head roles given from a listing made before the eyes were marked, on a character not yet built to its end with these marks (or built with hats and no hair chain) | List the parts again |
| `VM-PIECE-GONE` | stop | a named piece's vertex count is no piece's of the model as listed | take the role off and give it again |
| `VM-EYES-UNMARKED` | warn | an eye mode is chosen and the eyes aren't marked | mark the eyes |
| `VM-EYES-FALLBACK` | warn | the eyes slide by a fallback an earlier build remembered, and the face points were saved again since | "Try the moving iris again" |
| a step's own code, "failed this way on the last build" | stop | the step failed with a code its inputs decide, and the manifest, model, rig, face points and app are what they were (`run.json` beside its log) | the failure's own fix, with its option |
| `VM-BLEND-NEWER` | stop | a run from the middle would open a `.blend` saved by a newer Blender than the one chosen | build from the start, or choose that Blender |
| `VM-DISK-FULL` / `VM-DISK-LOW` | stop / stop or warn | under 1 GB where the build writes / not room for what this build writes over that floor (stop), under 3 GB (warn) | free up space (Characters page) |
| `VM-MEMORY-LOW` | stop / warn | less memory to be had than this character's heaviest step took last time (2.3 GB taken for one never built) / under 2 GB | close other programs |
| `VM-SYSTEM-DISK-LOW` | warn | the system drive, when it is another than the build's, has under 4 GB | free some space on it |
| `VM-PATH-LONG` | stop | the character's folder is so deep that the longest file a build writes there (60 characters under it: a rig check's close-up) would pass Windows' 260 | move the app nearer the top of the drive, or a shorter name |
| `VM-ONEDRIVE` | warn | the character's output folder lies inside a OneDrive folder (`%OneDrive%`, `%OneDriveCommercial%`, `%OneDriveConsumer%`) | move the app out of OneDrive |
| `VM-MODEL-ONLINE` | warn | the model file carries OneDrive's "online only" mark | "Always keep on this device", or copy it out |
| `VM-NO-BROWSER` | warn | no Edge or Chrome: the exported file isn't drawn to see through it, no contact sheets | nothing is wrong with the avatar; install one, or `VTUBER_MAKER_BROWSER` |

## The tester's computer

| Symptom | Measured cause | Fix |
|---|---|---|
| The app was closed (or ended in Task Manager) in the middle of a build, and the computer stayed slow for minutes; the next build crawled or ran out of memory | Windows ends no child with its parent: the build's Blender ran on, holding two to three gigabytes, beside the next build (which rightly took the lock a dead process left) | fixed 2026-10-12 (`engine.dies_with_us`: every Blender is in a Windows job that ends with the app). Tested by killing an app that had tied a child and one that hadn't |
| After such a kill, "another build is running (pid N ...)" | the lock file names a process | it is taken over by itself when that process is gone (`engine._Lock`); only a lock whose process is alive refuses. Nothing to delete by hand |
| A character won't open: "state.json is unreadable and has no usable backup" (`VM-STATE-UNREADABLE`) | the record of its builds was cut short twice (a crash or a full drive while it and its backup were written) | the page offers to set it aside: both files are renamed `...unreadable-<when>`, the builds stay, the next build starts the record afresh (`engine.set_state_aside`). One cut short once is read from its backup without a word |
| The drive filled in the middle of a step | the step's Blender is stopped under 50 MB free (`VM-DISK-FULL`) | what it had half written (its `.blend`, Blender's `.blend@`, a `.vrm`) is removed, so the room it took comes back; free up space on the Characters page and run from that step. A write of the app's own refused for want of room (state, manifest, face points) is said as `VM-DISK-FULL` too, not as a traceback |
| A folder with spaces or with letters of another alphabet in its name (`C:\Users\José María\...`, a model `キャラ 1.glb`) | none: every path goes to Blender as one whole argument and every file of the app's is read and written as UTF-8 | builds as any other (SETUP.md "The tester's computer": a character built from such a folder, the same file byte for byte) |
| Saves fail in a folder very deep on the disk | Windows gives a path 260 characters, and Blender saves a `.blend` the old way whatever the long-path switch says (5.2.2: saved at 258 characters, not at 259) | `VM-PATH-LONG` before the build: a character's folder may be 199 characters deep |
| A build in a deep folder takes five minutes longer, and its file check says "the browser never asked for the page, started 3 times" (`seen.left`) | the browser's profile was made inside the run's folder; Edge writes files up to 120 characters below its profile and none past 260: with less room it drops files without a word, and from about 250 characters of profile it never asks for its page | fixed 2026-10-12 (`sheet.profile_for`: the profile lies beside the sheets when that leaves it room, else in the app's own `out` folder, else in the temp folder). A fox girl in a folder 199 characters deep: the look back in 21 s with her own numbers |

## The app's window

| Symptom | Measured cause | Fix |
|---|---|---|
| The window goes blank or closes by itself; the console flashed "Failed to delete user data folder" and was gone | the window's browser (WebView2) crashed; that line is pywebview unable to remove its profile afterwards, not the cause. Four times on one computer (2026-10-07 to 10-10): an access violation in `RTSSHooks64.dll`, RivaTuner's overlay hook, each some ten seconds after the window opened. Nothing of the app's was at fault: no entry in `out/app_errors.log`, the request under the click (`POST /api/piece`) only writes the manifest | the app now finds the crash by its dump (`server._window_crash`), writes the module in `out/app_errors.log` (`window`), keeps the last two dumps in `out/crashes/` and goes on in the browser. For the user: exclude `msedgewebview2.exe` in RivaTuner (application detection level: None), or "open the app in: browser" |

## Commands failing

| Symptom | Fix |
|---|---|
| A command fails in PowerShell | quoted program paths need `&` in front |
| `export_scene.vrm` missing when running a script by hand | the VRM Add-on is off: start Blender with `-P scripts/_startup.py`, or without `--factory-startup` |
| A glTF exporter traceback prints during export but it says FINISHED | harmless; check the `.vrm` loads in the viewer |
| Blender hangs with two runs at once | by hand, run one Blender at a time (the engine's own side-by-side steps ask for room first, `engine._room`) |
| A step fails or comes out otherwise in a build, and not when its script is run by hand | the run's kept Blender (`scripts/_worker.py`) or the last steps side by side: build once with `"one_blender": false, "blenders": 1` in `out/settings.json` (or `OC2VRM_ONE_BLENDER=0`, `OC2VRM_BLENDERS=1`). If that mends it, it is a bug: keep both runs' folders (`worker.log` is in them) |
| The computer crawls while a build does its last steps | they run side by side when there is room: `python -m app setup --blenders 1` for one at a time (`--blenders auto` to undo) |
| A build stood five minutes at the exported file's check, then went on | the browser that draws the file never asked for its page (2026-10-12, once). It is now ended after 25 s and started again (`sheet.NEVER_ASKED`); a report that still says "wasn't drawn" says which it was |
