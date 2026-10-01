// Options: the avatar-wide switches, each saved as it changes.
import { $, S, app, call, esc, say, refreshManifest, refreshStatus } from "./core.js";
import { pushUndo, saved } from "./ui.js";

// [key, title, what it does, [[value, label, what choosing it means]]]. "auto" is null: the survey decides.
const AUTO = (what) => [null, "auto", what];
const OPTION_UI = [
  ["Head and face"],
  ["rigid_head", "Solid head", "Moves the whole head as one solid block instead of bending it: for hats, helmets, masks, or a head that is one hard shape (a mascot's).",
    [AUTO("the survey decides: on when it finds a hat or a solid head"), [true, "on", "The head and everything on it turn together and never bend. Hair on it doesn't sway."],
      [false, "off", "The head bends normally and hair gets its own springs."]]],
  ["hat_tip", "Swinging hat tip", "Only with a solid head: the tip of a pointed hat (a witch hat) swings with physics.",
    [[true, "on", "The tip bounces when the head moves; the rest of the hat stays solid."], [false, "off", "The whole hat stays solid."]]],
  ["face_tracking", "Face tracking shapes", "Adds the 52 ARKit face shapes that iPhone and webcam face tracking drive (brows, cheeks, jaw, mouth). Needs the face points (Face page).",
    [[true, "on", "The face follows your expressions in Warudo/VNyan."], [false, "off", "Only the basic blink and the five vowel mouths."]]],
  ["mouth_colour", "Inside of the mouth", "The colour painted inside the mouth when it opens.",
    [AUTO("picked from the model's texture"), ["skin", "skin", "Skin-coloured: for simple faces where a dark mouth reads as a hole."]]],
  ["Hair and clothes"],
  ["cloth", "Cloth physics", "Pieces hanging free below the hips (skirt, coat tails, cape) swing with spring physics.",
    [AUTO("the survey decides: on when much of the model hangs free"), [true, "on", "They sway and trail as you move."],
      [false, "off", "They bend with the legs like skin: no swinging."]]],
  ["ahoge", "Ahoge spring", "The single hair strand sticking up from the top of the head, if the model has one.",
    [[true, "on", "It bounces when the head moves."], [false, "off", "It stays stiff."]]],
  ["loose_pieces", "Repair torn straps", "Straps, belts and chains that cross between two limbs are weighted so they don't tear apart when the limbs move apart.",
    [[true, "on (recommended)", "Repaired."], [false, "off", "Left as rigged: they can stretch into long spikes."]]],
  ["repair_shoulder_seams", "Repair shoulder seams", "Closes gaps that open at the shoulders between the body and a sleeve when the arms go up.",
    [[true, "on (recommended)", "Repaired."], [false, "off", "Left as rigged."]]],
  ["Body and hands"],
  ["skinning", "Body weights", "How the skin follows the bones when the character moves. With Tripo's rig (Model page) the rig's own weights are used instead.",
    [["heat", "computed (recommended)", "Worked out from the model's shape, with the app's rules for arms and body."],
      ["template", "copied from VRoid (experimental)", "Copies a VRoid body's weights onto this one. Measured worse on model A (the arms stretched far more)."],
      ["voxel", "through the body (experimental)", "Weights by distance through the inside of the body, as Maya's geodesic voxel bind does, with the same rules for arms and body. On model A and model B it looked the same as computed; model A's head turn stretched less, its armpit squashed a little more."]]],
  ["fingers", "Fingers", "Finger bones, so hand tracking can open and close the hands.",
    [["guides", "found on the surface (recommended)", "Finds each finger on the mesh and lays the bones along it, the way model A and model B were done."],
      ["auto", "quick guess", "Older, faster detection; less exact on curled or touching fingers."],
      ["none", "none", "No finger bones: each hand moves as one piece. For mittens, or if the fingers come out broken."]]],
  ["tpose", "Final pose", "VRM apps expect the avatar saved standing in a T-pose; this is how the model gets there.",
    [["spec", "T-pose, flat hands (recommended)", "Arms straight out and fingers straight, like a VRoid avatar: finger tracking in Warudo works best from this."],
      ["plain", "T-pose, hands as modelled", "Arms straight out; the fingers keep their modelled curl."],
      ["none", "no T-pose", "Keeps the modelled pose. No final .vrm is made: for testing the rig only."]]],
];

async function setOption(key, value) {
  await call("set", { key, value });
  await refreshManifest(); await refreshStatus(); app.renderAll();
}

export function renderOptions() {
  if (!S.manifest) { $("option-grid").innerHTML = `<p class="msg">Open or create a character first.</p>`; return; }
  const resolved = S.status?.options || {};
  $("option-grid").innerHTML = OPTION_UI.map((row) => {
    if (row.length === 1) return `<h3>${row[0]}</h3>`;
    const [key, title, what, choices] = row, v = S.manifest.options[key] ?? null;
    return `<div class="opt"><b>${title}</b><p class="help">${what}</p>
      <select data-option="${key}" aria-label="${esc(title)}">${choices.map(([val, label]) =>
        `<option value='${JSON.stringify(val)}' ${JSON.stringify(val) === JSON.stringify(v) ? "selected" : ""}>${label}</option>`).join("")}</select>
      <p class="choice" data-choice="${key}"></p></div>`;
  }).join("");
  const explain = (sel) => {
    const [key, , , choices] = OPTION_UI.find((r) => r[0] === sel.dataset.option);
    const val = JSON.parse(sel.value), [, , text] = choices.find(([c]) => JSON.stringify(c) === sel.value);
    // what "auto" came to for this model, when the survey has run
    const picked = val === null && key in resolved ? choices.find(([c]) => c !== null && JSON.stringify(c) === JSON.stringify(resolved[key])) : null;
    document.querySelector(`[data-choice="${key}"]`).textContent = text + (picked ? `. For this model: ${picked[1]}.` : "");
  };
  document.querySelectorAll("[data-option]").forEach((sel) => {
    explain(sel);
    sel.onchange = async () => {
      const key = sel.dataset.option, before = S.manifest.options[key] ?? null, title = OPTION_UI.find((r) => r[0] === key)[1];
      explain(sel);
      try {
        await setOption(key, JSON.parse(sel.value));
        pushUndo(title, () => setOption(key, before));
        say("options-msg", "");
        saved(`${title} saved. Rebuild for it to count.`);
      } catch (e) { say("options-msg", e.message, true); }
    };
  });
}
