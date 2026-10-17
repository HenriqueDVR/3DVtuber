// Review: the spring viewer (open with the page), where the file goes, contact sheets and still renders.
import { $, S, app, call, esc, lastVrm, say } from "./core.js";
import { toast } from "./ui.js";
import { reportDialog } from "./report.js";
import { job } from "./jobs.js";
import { onLeave, onShow } from "./nav.js";

// ---------- contact sheets: the exported file as apps draw it, beside the approved build's
const sheetPictures = (urls, label) => urls.filter(Boolean).map((u) =>
  `<img src="${u}?t=${Date.now()}" alt="${esc(label)}: ${esc(u.split("/").pop().replace("sheet_", "").replace(".png", ""))}">`).join("");
const vrmRun = () => { const vrm = lastVrm(); return vrm ? vrm.replaceAll("\\", "/").split("/").slice(-2)[0] : null; };
let sheets = [];
export async function loadSheets() {
  sheets = S.manifestPath ? (await call("sheets")).sheets : [];
  showSheets();
}
function showSheets() {
  const run = vrmRun(), good = approvedRun();
  const mine = sheets.find((s) => s.run === run);
  $("sheets-label").textContent = !run ? "No finished .vrm yet: build first." : !mine ? "No sheets of this build yet: press Draw the sheets."
    // what the shots asked for and the file lacks: said, or those shots read as a face that didn't move
    : `${run === good ? "The build marked good" : "This build"}.${mine.report?.missing?.length ? ` Not on this avatar: ${mine.report.missing.join(", ")}.` : ""}`;
  $("sheets").innerHTML = mine ? sheetPictures(mine.urls, "this build") : "";
  const theirs = good && sheets.find((s) => s.run === good);
  const box = $("approved-sheets-box");
  box.hidden = !good || !run || run === good;
  if (box.hidden) return;
  $("approved-sheets-label").innerHTML = theirs ? "The build marked good."
    : `The build marked good has no sheets. <button class="act alt small" id="sheet-approved" data-job>Draw them</button>`;
  $("approved-sheets").innerHTML = theirs ? sheetPictures(theirs.urls, "marked good") : "";
  $("sheet-approved")?.addEventListener("click", () => job("sheet", { run: good }));
  if (!$("runbar").hidden && $("sheet-approved")) $("sheet-approved").disabled = true;
}
$("sheet-run").onclick = () => { say("sheet-msg", ""); job("sheet", {}); };

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
// The page opens on the avatar: the last finished .vrm, loaded once per build. One asked for elsewhere (a pose
// from the Build page, an older run) counts as shown, so it isn't swapped back under the user.
let shownVrm = null;
export async function openViewer(url, what, show = {}) {
  if (what !== false) shownVrm = lastVrm();
  if (!url) {
    const vrm = lastVrm();
    url = vrm && (await call("url", { path: vrm })).url;
  }
  $("viewer-empty").hidden = !!url; $("viewer").hidden = !url;
  if (!url) return what !== false && toast("No finished .vrm yet: build first.", "bad");
  const extra = Object.entries(show).map(([key, value]) => `&${key}=${encodeURIComponent(value)}`).join("");
  $("viewer").src = `/viewer/index.html?model=${encodeURIComponent(url)}${extra}`;
  if (what) toast(`Viewing the ${what}.`, "info");
}
app.openViewer = openViewer;
// on the page, and when the status comes (a first load lands on the page before it) or a build finishes on it
onShow.review = () => {
  const vrm = lastVrm();
  if (!vrm) { $("viewer-empty").hidden = false; $("viewer").hidden = true; return; }
  if (vrm !== shownVrm) { shownVrm = vrm; openViewer(null, false); }
};
app.statusShown = () => { if (location.hash === "#review") onShow.review(); };
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

// Leaving the page lets go of the webcam (the viewer's face tracking test) and stops the face test:
// out of sight they ran on, the camera's light with them.
onLeave.review = () => {
  try { const frame = $("viewer").contentWindow; if (frame?.viewer?.webcam.on) frame.document.getElementById("webcam").click(); } catch {}
  document.querySelector('[data-face="stop"]')?.click();
};

// The notes from tracking apps once written here (cli.save_feedback) are kept with their builds; the page no
// longer asks for them. Nothing to load.
export async function loadFeedback() { S.feedback = []; }

// ---------- Stuck?
$("bundle").onclick = async () => {
  try {
    const made = await call("bundle", { note: $("note").value });
    $("bundle-out").hidden = false;
    $("bundle-out").textContent = `Gathered in ${made.bundle}\n\nIn Claude Code, in this repository, say:\n\n` +
      `Read docs/agent/app.md, then look at the problem bundle in ${made.bundle} and fix it with python -m app.`;
  } catch (e) { $("bundle-out").hidden = false; $("bundle-out").textContent = e.message; }
};

// the report itself is one dialog for the whole app (report.js); what was written here goes with it
$("report-open").onclick = () => { $("rd-note").value = $("note").value; reportDialog(); };
