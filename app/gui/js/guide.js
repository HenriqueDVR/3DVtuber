// "What to check here": on every step page, the few things a person should look at before going on, automatic
// step or not. The user, 2026-11-10: the app should go "step by step what a human should be looking at and making
// sure it's good even on automatic steps". Each line is a defect a build has shipped with while its checks passed.
import { $ } from "./core.js";

const GUIDE = {
  model: ["The model stands upright and faces you; if it lies down or shows its back, turn it here before anything else.",
    "Arms are held away from the body (a T-pose is best). Arms at the sides weld sleeves to the ribs.",
    "The rig line says whose skeleton is used. A rigged FBX from Tripo is the most reliable input."],
  parts: ["Hair is coloured as hair. Pale or two-tone hair is the most often missed: it then doesn't swing and tears on a head turn.",
    "The face is not part of the hair, a hat or a hood.",
    "Capes, tails, wings and long sleeves have a role. With none they are skinned like skin and stretch.",
    "Small trims and buttons ride the piece they sit on."],
  joints: ["Shoulders, elbows and wrists sit inside the arms, not in a sleeve or a cuff.",
    "Hips and knees sit inside the legs; under a long skirt they are a guess, so look closely.",
    "The neck joint is at the base of the neck, not in the collar or the hair."],
  face: ["The face picture shows the whole face from the front, not cut off or off-centre.",
    "Each eye has four dots: both corners, top and bottom of the opening, lashes left outside.",
    "The mouth's four dots are on the drawn mouth, not on the nose's shadow or the chin.",
    "The 3D view beside the picture shows the dots on the head: none may float or sit on hair.",
    "Eyes hidden behind glasses, a visor or hair: say so in the eye choice, don't mark a guess."],
  options: ["Leave what you are unsure of on automatic: the build decides from the model.",
    "Rigid head for a helmet or a hat that covers the hair. Cloth for capes and loose skirts.",
    "Change one thing at a time and rebuild from the step it names: two changes at once can't be told apart."],
  build: ["Read the card above the steps first: what is known to go wrong, found without building.",
    "A step marked failed was left out, or done a plainer way: the avatar still works, without that part. Its row says why.",
    "“Ready” means the checks passed, not that it looks right. Go on to Review."],
  movement: ["A piece listed as left behind or torn: look at the picture of that pose, then give the piece a role on Parts.",
    "Hair that droops standing still has chains too soft or too long for it.",
    "A cape or mantle that stretches with raised arms is expected; one that stays at the ribs is not."],
  review: ["Turn the head left and right: no tear in the hair, no ledge across it, no gap under the back hair.",
    "Raise both arms: sleeves and capes go up with them, nothing stays at the hips.",
    "Blink and look around: the eyes close fully, and nothing shows through them from the side.",
    "Open the mouth: the lips stay on the face and the inside is not skin-coloured.",
    "Look from the side and from behind as well as the front. Then try it in your tracking app: that is the real test."],
};

for (const [page, lines] of Object.entries(GUIDE)) {
  // (the Model page has two heads, a new character's and the open one's: the second is the one with something to check)
  const heads = $(page)?.querySelectorAll(".page-head");
  const head = heads?.[heads.length - 1];
  if (!head) continue;
  head.insertAdjacentHTML("beforeend", `<details class="more guide"><summary>What to check here</summary><ul>${lines.map((l) => `<li>${l}</li>`).join("")}</ul></details>`);
}
