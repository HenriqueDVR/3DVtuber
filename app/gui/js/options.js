// Options: the avatar-wide switches, each saved as it changes.
import { $, S, app, call, esc, say, refreshManifest, refreshStatus } from "./core.js";
import { pushUndo, saved } from "./ui.js";

// [key, title, what it does, [[value, label, what choosing it means]]]. "auto" is null: the survey decides.
// A slider instead of choices: {min, max, step, fallback, low, high} (the words under its two ends).
const AUTO = (what) => [null, "auto", what];
const OPTION_UI = [
  ["Head and face"],
  ["rigid_head", "Solid head", "Moves the whole head as one solid block instead of bending it: for hats, helmets, masks, or a head that is one hard shape (a mascot's).",
    [AUTO("the survey decides: on when it finds a hat or a solid head"), [true, "on", "The head and everything on it turn together and never bend. Hair on it doesn't sway."],
      [false, "off", "The head bends normally and hair gets its own springs."]]],
  ["hat_tip", "Swinging hat tip", "Only with a solid head: the tip of a pointed hat (a witch hat) swings with physics.",
    [[true, "on", "The tip bounces when the head moves; the rest of the hat stays solid."], [false, "off", "The whole hat stays solid."]]],
  ["face_tracking", "Mouth tracking and blink", "Adds the ARKit-named face shapes that iPhone and webcam face tracking drive: the jaw, the mouth's corners and a blink (no brows or cheeks yet). Needs the face points (Face page).",
    [[true, "on", "The mouth and blink follow your face in your tracking app."], [false, "off", "Only the five vowel mouths: no blink unless eye tracking adds one."]]],
  ["lid_blink", "How the eyes blink", "For eyes that look around with an iris of their own (a moving iris or the model's iris pieces). Painted eyes that slide, and eyes that only blink, always squeeze.",
    [[false, "the eye squeezes (recommended)", "The whole eye squeezes toward its lash line."],
      [true, "a lid comes down (experimental)", "The upper lashes come down over the still eye and a lid of skin closes behind them, as hand-drawn anime faces blink. Rough where hair is drawn over the lashes in their colour."]]],
  ["mouth_colour", "Inside of the mouth", "The colour painted inside the mouth when it opens.",
    [AUTO("picked from the model's texture"), ["skin", "skin", "Skin-coloured: for simple faces where a dark mouth reads as a hole."]]],
  ["Hair and clothes"],
  ["cloth", "Cloth physics", "Pieces hanging free below the hips (skirt, coat tails, cape) swing with spring physics.",
    [AUTO("the survey decides: on when much of the model hangs free"), [true, "on", "They sway and trail as you move."],
      [false, "off", "They bend with the legs like skin: no swinging."]]],
  ["ahoge", "Ahoge spring", "The single hair strand sticking up from the top of the head, if the model has one.",
    [[true, "on", "It bounces when the head moves."], [false, "off", "It stays stiff."]]],
  ["hair_hits_body", "Hair collides with the body", "Whether the hair's springs are pushed away by the chest, shoulders and arms, or only by the head.",
    [[true, "on", "Long hair stays out of the back and off the arms."],
      [false, "off (head only)", "For hair that rests on the shoulders: it no longer lifts or breaks into a ledge when the head tilts. It can sink a little into a shoulder."]]],
  ["chest", "Chest bounce", "Two spring bones on the front of the chest, as VRoid avatars have: it bounces and settles when the body moves. Hair hanging over the chest keeps its own springs.",
    [[false, "off", "The chest moves with the body only."], [true, "on", "It bounces in your tracking app, as much as the slider below says."]]],
  ["chest_amount", "Chest bounce amount", "How much the chest bounces when chest bounce is on: softer springs that move more of the chest, or stiffer ones that move less.",
    { min: 0.25, max: 2, step: 0.05, fallback: 0.5, low: "barely", high: "a lot" }],
  ["loose_pieces", "Repair torn straps", "Straps, belts and chains that cross between two limbs are weighted so they don't tear apart when the limbs move apart.",
    [[true, "on (recommended)", "Repaired."], [false, "off", "Left as rigged: they can stretch into long spikes."]]],
  ["repair_shoulder_seams", "Repair shoulder seams", "Closes gaps that open at the shoulders between the body and a sleeve when the arms go up.",
    [[true, "on (recommended)", "Repaired."], [false, "off", "Left as rigged."]]],
  ["auto_pieces", "Fix clothing the arms leave behind", "During the build, the pose test's own fixes are applied: a hanging sleeve follows its arm, a mantle moves with the shoulders, a skirt the hands rest on stays on the body. The same fixes as “Apply this fix” on the Movement page, without the clicks. Pieces you gave a role on the Parts page keep it.",
    [[false, "off", "Only the roles you gave are used."], [true, "on (new: being tried)", "Applied by itself; a fix that tears the rig is taken back."]]],
  ["A finished avatar"],
  ["licence_read", "The avatar's licence", "An avatar made by someone else comes with terms. The app changes the file (it straightens joints, converts VRM 0.x to 1.0, adds what you choose), so where it can't read the terms as allowing that, it asks you. The address of the terms is on the Build page when the build stops for them.",
    [[false, "not confirmed", "A file whose terms have to be read is not taken in."], [true, "I have read them and may change it", "The terms allow changing the avatar, or it is your own."]]],
  ["Body and hands"],
  ["facing", "Front and back", "Which side of the model is its front. The app finds it from the feet; a model it gets wrong is built with the face on the back of the head. After changing this, run again from the survey.",
    [AUTO("found from which way the toes point"), ["file", "as the file has it", "The model faces the way it was exported: for one the app turned around."]]],
  ["skinning", "Body weights", "How the skin follows the bones when the character moves. With Tripo's rig (Model page) the rig's own weights are used instead.",
    [["heat", "computed (recommended)", "Worked out from the model's shape, with the app's rules for arms and body."],
      ["template", "copied from VRoid (experimental)", "Copies a VRoid body's weights onto this one. Measured worse on a test model (the arms stretched far more)."],
      ["voxel", "through the body (experimental)", "Weights by distance through the inside of the body, as Maya's geodesic voxel bind does, with the same rules for arms and body. On two test models it looked the same as computed; on one the head turn stretched less, its armpit squashed a little more."]]],
  ["fingers", "Fingers", "Finger bones, so hand tracking can open and close the hands.",
    [["guides", "found on the surface (recommended)", "Finds each finger on the mesh and lays the bones along it."],
      ["auto", "quick guess", "Older, faster detection; less exact on curled or touching fingers."],
      ["none", "none", "No finger bones: each hand moves as one piece. For mittens, or if the fingers come out broken."]]],
  ["tpose", "Final pose", "VRM apps expect the avatar saved standing in a T-pose; this is how the model gets there.",
    [["spec", "T-pose, flat hands (recommended)", "Arms straight out and fingers straight, like a VRoid avatar: finger tracking works best from this."],
      ["plain", "T-pose, hands as modelled", "Arms straight out; the fingers keep their modelled curl."],
      ["none", "no T-pose", "Keeps the modelled pose. No final .vrm is made: for testing the rig only."]]],
];

async function setOption(key, value) {
  await call("set", { key, value });
  await refreshManifest(); await refreshStatus(); app.renderAll();
}

// The options that count for a character taken in from a finished .vrm (engine.KEPT skips the rest).
const IMPROVES = ["chest", "chest_amount", "licence_read"];

// Each option's default (pipeline/manifest.py OPTIONS): a tile whose value differs says "changed".
const DEFAULTS = { rigid_head: null, hat_tip: false, face_tracking: true, lid_blink: false, mouth_colour: null, cloth: null, ahoge: true, hair_hits_body: true,
  chest: false, chest_amount: 0.5, loose_pieces: true, repair_shoulder_seams: true, auto_pieces: false, licence_read: false, facing: null,
  skinning: "heat", fingers: "guides", tpose: "spec" };
// One line each, said plainly; the whole of what the option does is a click away (the "?").
const SHORT = { rigid_head: "Hats, helmets, masks: the head moves as one block.", hat_tip: "A pointed hat's tip bounces.",
  face_tracking: "Your mouth and blinks drive the avatar's.", lid_blink: "The eye squeezes, or (experimental) a lid comes down over it.", mouth_colour: "The colour inside the open mouth.",
  cloth: "Skirts, coat tails and capes swing.", ahoge: "The single strand on top of the head bounces.",
  hair_hits_body: "Long hair stays off the back and arms.", chest: "The chest bounces as the body moves.",
  loose_pieces: "Straps and belts between limbs don't tear.", repair_shoulder_seams: "No gaps at the shoulders when the arms go up.",
  auto_pieces: "Sleeves and mantles the arms leave behind follow them.", licence_read: "The avatar's terms let you change it.",
  facing: "Which side of the model is its front.", skinning: "How the skin follows the bones.", fingers: "Finger bones for hand tracking.",
  tpose: "The pose the avatar is saved in." };
// The page in groups: what most people look at open, the rest folded with a word on when to open it.
const GROUPS = [
  { title: "What moves", keys: ["cloth", "ahoge", "hair_hits_body", "chest"] },
  { title: "Face and head", note: "mouth tracking, the blink, the mouth's colour, solid heads and hats", keys: ["face_tracking", "lid_blink", "mouth_colour", "rigid_head", "hat_tip"], folded: true },
  { title: "Repairs", note: "leave these on unless something looks wrong after a build", keys: ["loose_pieces", "repair_shoulder_seams", "auto_pieces"], folded: true },
  { title: "Advanced", note: "body weights, fingers, the saved pose, which way the model faces", keys: ["skinning", "fingers", "tpose", "facing"], folded: true },
  { title: "A finished avatar", keys: ["licence_read"], improving: true },
];
const row = (key) => OPTION_UI.find((r) => r[0] === key);
const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
const folded = {};

export function renderOptions() {
  if (!S.manifest) { $("option-grid").innerHTML = `<p class="msg">Open or create a character first.</p>`; return; }
  const resolved = S.status?.options || {}, o = S.manifest.options, improving = !!S.status?.improving;
  const value = (key) => o[key] ?? (typeof DEFAULTS[key] === "number" ? DEFAULTS[key] : null);
  // what "auto" came to for this model, once the survey has run
  const picked = (key, choices) => value(key) === null && key in resolved ? choices.find(([c]) => c !== null && same(c, resolved[key])) : null;
  // a swinging hat tip only means something on a solid head; the bounce's amount only with the bounce on
  const shown = (key) => key === "hat_tip" ? (value("rigid_head") ?? resolved.rigid_head) === true : true;
  const tile = (key) => {
    const [, title, what, choices] = row(key), v = value(key), changed = !same(v, DEFAULTS[key]);
    const auto = picked(key, choices);
    const amount = key === "chest" && v === true ? (() => {
      const [, , , range] = row("chest_amount"), a = o.chest_amount ?? range.fallback;
      return `<label class="amount">how much <input type="range" data-slider="chest_amount" aria-label="chest bounce amount" min="${range.min}" max="${range.max}" step="${range.step}" value="${a}">
        <b data-amount="chest_amount">${(+a).toFixed(2)}</b></label>`;
    })() : "";
    return `<div class="otile ${changed ? "changed" : ""}" data-key="${key}">
      <div class="ohead"><b>${esc(title)}</b>${changed ? `<span class="chip warn">changed</span>` : ""}<button class="why" data-why="${key}" title="what this does" aria-expanded="false">?</button></div>
      <p class="short">${esc(SHORT[key] || "")}</p>
      <div class="seg" role="group" aria-label="${esc(title)}">${choices.map(([c, label, text]) =>
        `<button data-option="${key}" data-value='${esc(JSON.stringify(c))}' class="${same(c, v) ? "on" : ""}" title="${esc(text)}">${esc(label)}</button>`).join("")}</div>
      ${amount}
      ${auto ? `<p class="choice"><b>For this model: ${esc(auto[1])}.</b></p>` : ""}
      <p class="long" hidden>${esc(what)}</p></div>`;
  };
  const changedCount = Object.keys(DEFAULTS).filter((k) => k !== "chest_amount" && !same(value(k), DEFAULTS[k])).length;
  const groups = GROUPS.filter((g) => improving ? g.keys.some((k) => IMPROVES.includes(k)) : !g.improving);
  $("option-grid").innerHTML = `<p class="olead">${improving ? "This character is a finished avatar (.vrm): its head, hair, hands, face and pose stay as they are, so only what can be added to it is here."
      : "Most models need nothing here: the defaults suit them, and <b>auto</b> lets the survey decide."}
      ${changedCount ? `<span class="chip warn">${changedCount} changed</span>` : `<span class="chip done">all defaults</span>`}</p>`
    + groups.map((g) => {
      const keys = g.keys.filter((k) => (!improving || IMPROVES.includes(k)) && shown(k));
      // (a folded group with a changed option in it opens: what was changed is never out of sight)
      const open = folded[g.title] ?? (!g.folded || keys.some((k) => !same(value(k), DEFAULTS[k])));
      const changedHere = keys.filter((k) => !same(value(k), DEFAULTS[k])).length;
      return `<details class="ogroup" data-group="${esc(g.title)}" ${open ? "open" : ""}><summary>${esc(g.title)}${g.note ? ` <span>${esc(g.note)}</span>` : ""}
        ${changedHere ? `<span class="chip warn">${changedHere} changed</span>` : ""}</summary><div class="ogrid">${keys.map(tile).join("")}</div></details>`;
    }).join("");
  document.querySelectorAll(".ogroup").forEach((d) => d.ontoggle = () => { folded[d.dataset.group] = d.open; });
  document.querySelectorAll("[data-why]").forEach((b) => b.onclick = () => {
    const long = b.closest(".otile").querySelector(".long");
    long.hidden = !long.hidden; b.setAttribute("aria-expanded", String(!long.hidden)); b.classList.toggle("on", !long.hidden);
  });
  document.querySelectorAll("[data-slider]").forEach((slider) => {
    const key = slider.dataset.slider, [, title, , range] = row(key);
    slider.oninput = () => { document.querySelector(`[data-amount="${key}"]`).textContent = (+slider.value).toFixed(2); };
    slider.onchange = async () => {
      const before = S.manifest.options[key] ?? range.fallback;
      try {
        await setOption(key, +slider.value);
        pushUndo(title, () => setOption(key, before));
        say("options-msg", "");
        saved(`${title} saved. Rebuild for it to count.`);
      } catch (e) { say("options-msg", e.message, true); }
    };
  });
  document.querySelectorAll("button[data-option]").forEach((b) => b.onclick = async () => {
    const key = b.dataset.option, next = JSON.parse(b.dataset.value), before = S.manifest.options[key] ?? null, title = row(key)[1];
    if (same(next, value(key))) return;
    try {
      await setOption(key, next);
      pushUndo(title, () => setOption(key, before));
      say("options-msg", "");
      saved(`${title} saved. Rebuild for it to count.`);
    } catch (e) { say("options-msg", e.message, true); }
  });
}
