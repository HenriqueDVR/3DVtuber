// Review: still renders (next to the approved build's), the spring viewer, and notes from Warudo.
import { $, S, app, call, esc, lastVrm } from "./core.js";
import { toast } from "./ui.js";
import { job } from "./jobs.js";

// ---------- still renders
const POSE_NAME = { raised: "arms raised", "t-pose": "T-pose", forward: "arms forward", hang: "arms down", rest: "rest" };
const approvedRun = () => {
  const vrm = S.status?.approved?.vrm;
  return vrm ? vrm.replaceAll("\\", "/").split("/").slice(-2)[0] : null;
};
const currentRun = () => {
  const latest = S.status?.latest_blend;
  return latest ? latest.replaceAll("\\", "/").split("/").slice(-2)[0] : null;
};
const pictures = (urls, label) => urls.filter(Boolean).map((u) => `<img src="${u}?t=${Date.now()}" alt="${esc(label)}: ${esc(u.split("/").pop().replace(".png", "").replace("_", " "))}">`).join("");

export async function loadRenders() {
  S.renders = S.manifestPath ? (await call("renders")).renders : [];
  showRenders();
}
function showRenders() {
  const pose = $("pose").value, name = POSE_NAME[pose];
  const mine = S.renders.find((r) => r.pose === pose && (!r.run || r.run === currentRun())) || S.renders.find((r) => r.pose === pose);
  const good = approvedRun();
  $("renders-label").textContent = mine ? `${good && mine.run === good ? "The build marked good" : "This build"}, ${name}` : `No render in ${name} yet: press Render.`;
  $("renders").innerHTML = mine ? pictures(mine.urls, name) : "";
  // side by side with the build the user said was good, when there is one and it isn't this one
  const theirs = good && S.renders.find((r) => r.run === good && r.pose === pose);
  const box = $("approved-renders-box");
  box.hidden = !good || mine?.run === good;
  if (box.hidden) return;
  $("approved-renders-label").innerHTML = theirs ? `The build marked good, ${name}`
    : `The build marked good has no render in ${name}. <button class="act alt small" id="render-approved" data-job>Render it</button>`;
  $("approved-renders").innerHTML = theirs ? pictures(theirs.urls, `marked good, ${name}`) : "";
  $("render-approved")?.addEventListener("click", () => job("render", { pose, run: good }));
  if (!$("runbar").hidden && $("render-approved")) $("render-approved").disabled = true;
}
$("pose").onchange = showRenders;
$("render").onclick = () => job("render", { pose: $("pose").value });
app.jobDone = (state) => {
  if (!state.result?.urls) return;
  if (state.result.pose && $("pose").value !== state.result.pose) $("pose").value = state.result.pose;
};

// ---------- the viewer, springs running
// `show`: what the viewer opens on, e.g. { pose: "arms raised" } or { weights: "LeftUpperArm" }
export async function openViewer(url, what, show = {}) {
  if (!url) {
    const vrm = lastVrm();
    url = vrm && (await call("url", { path: vrm })).url;
  }
  if (!url) return toast("No finished .vrm yet: build first.", "bad");
  const extra = Object.entries(show).map(([key, value]) => `&${key}=${encodeURIComponent(value)}`).join("");
  $("viewer").hidden = false; $("viewer").src = `/viewer/index.html?model=${encodeURIComponent(url)}${extra}`;
  $("face-test").hidden = false;
  $("viewer").scrollIntoView({ behavior: "smooth", block: "nearest" });
  if (what) toast(`Viewing the ${what}.`, "info");
}
app.openViewer = openViewer;
$("viewer-open").onclick = () => openViewer();

// A blink as trackers send it: shut and open in about 0.15 s every few seconds, which is why a
// partial squeeze reads as a blink. Talking and looking cycle the mouth and gaze presets.
let faceTest = null;
function expression(name, value) {
  const slider = $("viewer").contentDocument?.querySelector(`input[data-expression="${name}"]`);
  if (!slider) return false;
  slider.value = value; slider.dispatchEvent(new Event("input"));
  return true;
}
const FACE = {
  blink: (t) => ({ blink: Math.max(0, 1 - Math.abs((t % 3) - 0.15) / 0.075) }),
  talk: (t) => ({ aa: Math.max(0, Math.sin(t * 9)) * 0.8, oh: Math.max(0, Math.sin(t * 9 + 2)) * 0.5 }),
  look: (t) => { const k = ["lookLeft", "lookUp", "lookRight", "lookDown"][Math.floor(t / 1.2) % 4]; return { lookLeft: 0, lookRight: 0, lookUp: 0, lookDown: 0, [k]: 1 }; },
};
document.querySelectorAll("[data-face]").forEach((b) => b.onclick = () => {
  clearInterval(faceTest);
  for (const name of ["blink", "aa", "oh", "lookLeft", "lookRight", "lookUp", "lookDown"]) expression(name, 0);
  faceTest = null; $("face-test-msg").textContent = "";
  document.querySelectorAll("[data-face]").forEach((x) => x.classList.toggle("on", x === b && b.dataset.face !== "stop"));
  if (b.dataset.face === "stop") return;
  // a timer, not animation frames: those stop while the window is behind another
  const start = performance.now(), drive = FACE[b.dataset.face];
  const step = () => {
    const missing = Object.entries(drive((performance.now() - start) / 1000)).filter(([n, v]) => !expression(n, v)).map(([n]) => n);
    $("face-test-msg").textContent = missing.length ? `this file has no ${missing.join(", ")}` : "";
    $("face-test-msg").classList.toggle("bad", !!missing.length);
  };
  step();
  faceTest = setInterval(step, 20);
});

// ---------- Warudo feedback: a note and screenshots per build, listed newest first
export async function loadFeedback() {
  S.feedback = S.manifestPath ? (await call("feedback_list")).feedback : [];
  $("fb-list").innerHTML = S.feedback.map((f) => `<div class="feedback">
    <span class="msg">${esc(f.when.replace("T", " "))} · ${esc(f.vrm.split(/[\\/]/).slice(-2).join("/") || "no .vrm yet")}</span>
    <p style="margin:2px 0;white-space:pre-wrap">${esc(f.note)}</p>
    ${f.urls.filter(Boolean).map((u) => `<img src="${u}" alt="Warudo screenshot">`).join("")}</div>`).join("");
  app.renderNav();
}
$("fb-save").onclick = async () => {
  const read = (file) => new Promise((ok, fail) => { const r = new FileReader(); r.onload = () => ok(r.result); r.onerror = fail; r.readAsDataURL(file); });
  // the checklist goes with the note: which poses were tried and looked right
  const boxes = [...$("fb-checks").querySelectorAll("label")];
  const ticked = boxes.filter((l) => l.querySelector("input").checked).map((l) => l.textContent.trim());
  const note = $("fb-note").value.trim() + (ticked.length ? `${$("fb-note").value.trim() ? "\n\n" : ""}Looked right: ${ticked.join("; ")}.` : "");
  try {
    const images = await Promise.all([...$("fb-shots").files].map(read));
    await call("feedback", { note, images });
    $("fb-note").value = ""; $("fb-shots").value = ""; boxes.forEach((l) => l.querySelector("input").checked = false);
    $("fb-msg").textContent = ""; toast("Note saved with this build.");
    loadFeedback();
  } catch (e) { $("fb-msg").textContent = e.message; $("fb-msg").classList.add("bad"); }
};

// ---------- Stuck?
$("bundle").onclick = async () => {
  try {
    const made = await call("bundle", { note: $("note").value });
    $("bundle-out").hidden = false;
    $("bundle-out").textContent = `Gathered in ${made.bundle}\n\nIn Claude Code, in this repository, say:\n\n` +
      `Read docs/agent/app.md, then look at the problem bundle in ${made.bundle} and fix it with python -m app.`;
  } catch (e) { $("bundle-out").hidden = false; $("bundle-out").textContent = e.message; }
};
