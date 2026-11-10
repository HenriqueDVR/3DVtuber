// Quick build: the automatic mode's one screen. Model to avatar in four stages, each done by the app and each
// saying what a person should look at; beside every automatic step, the way to do it by hand. The user,
// 2026-11-10: the default should be "quickly test it out and see what happens", and the face detector "fails if
// there's not a clearly shown face", so marking by hand is always one click away and its guess is never built on
// unseen. The pages behind it (Parts, Face, Options, ...) are the advanced mode: the same character, the same build.
import { $, S, app, esc, sentence, tookLast } from "./core.js";
import { nextRun, onShow } from "./nav.js";
import { toast } from "./ui.js";

const MODE = "oc2vrm-mode";
export const automatic = () => { try { return localStorage.getItem(MODE) !== "advanced"; } catch { return true; } };
// where a character opens: its one screen, or the menu of every page
app.landing = () => automatic() ? "quick" : "menu";

// Per character, for this visit: the face left out on purpose, a detector's guess no one has looked at yet,
// and what the detector last said.
const skipped = new Set(), unseen = new Set(), said = {};
let busy = false;

const detector = () => !S.setup?.checks.some((c) => c.name === "Auto face marking" && c.state !== "ok");
const step = (name) => S.status?.steps.find((s) => s.step === name);

// Where this character stands: which stage is next.
function where() {
  const st = S.status, kept = !!st?.improving, face = app.face?.has() || {};
  // (a character built before has been read: its face picture loads a few seconds after the page, and the
  // first stage stood at "next" over a finished avatar meanwhile)
  const ready = kept || step("build")?.status === "done" || (step("survey")?.status === "done" && !!S.listing && !!face.patch);
  const marked = kept || !!S.manifest?.face;
  const faced = marked ? !unseen.has(S.manifestPath) : skipped.has(S.manifestPath);
  const built = step("vrm_check")?.status === "done" && !nextRun(st);
  return { kept, ready, marked, faced, built, face, next: !ready ? "ready" : !faced ? "face" : !built ? "build" : "result" };
}

// What the finished file is, in one word: from what the build itself left out (engine.PLAINER, SOFT).
function tier() {
  const st = S.status, v = st.verdict || {}, ships = st.ships || {};
  const plain = step("build")?.report?.plainer, without = (v.issues || []).filter(([, t]) => /^built without /.test(t)).map(([, t]) => t.slice(14));
  if (plain || !ships.blink) return ["Body only", "warn", plain ? "The full build failed on this model, so it was made the plain way: the body moves, the face and hair don't."
    : "The body moves. The face doesn't blink: it has no face points."];
  if (without.length) return ["Basic", "warn", `It works, without ${without.join(", ")}: ${without.length > 1 ? "those steps" : "that step"} failed on this model and ${without.length > 1 ? "were" : "was"} left out.`];
  if (!v.ready) return ["Needs a look", "warn", "It was built whole, and a check found something to fix."];
  return ["Full", "good", v.looks?.length ? "Every check passed; a few things are worth a look." : "Every check passed."];
}

const button = (id, text, kind = "alt") => `<button class="act ${kind} small" data-quick="${id}" data-job>${text}</button>`;
const stage = (n, title, state, kind, body) => `<div class="card qstage ${kind}"><div class="title"><span class="qn">${n}</span> ${title} <span class="chip ${kind}">${state}</span></div>${body}</div>`;
const look = (lines) => `<p class="help"><b>Look at:</b> ${lines}</p>`;

export function renderQuick() {
  const host = $("quick-stages");
  if (!host) return;
  if (!S.manifest || !S.status) { host.innerHTML = `<p class="msg">Open or create a character first.</p>`; $("quick-go").disabled = true; return; }
  const w = where(), path = S.manifestPath, at = (name) => w.next === name;
  const state = (name, done) => done ? ["done", "done"] : at(name) ? ["next", "warn"] : ["waiting", ""];
  const cards = [];

  { // 1: the model read
    const [text, kind] = state("ready", w.ready);
    cards.push(stage(1, "Read the model", text, kind, w.kept ? `<p class="help">A finished avatar or a model that came rigged: it is taken in as it is.</p>`
      : w.ready ? look("is it standing upright, facing you, arms clear of the body? If not, the rest goes wrong.") + `<div class="row">${button("go:model", "See the model")}${button("go:parts", "Name its parts by hand")}</div>`
      : `<p class="help">The app measures the model, lists its pieces and makes a sharp picture of the face. About a minute.</p>`));
  }
  { // 2: the face
    const [text, kind] = w.kept ? ["kept", "done"] : w.marked && w.faced ? ["marked", "done"] : skipped.has(path) && !w.marked ? ["left out", "warn"] : state("face", false);
    let body;
    if (w.kept) body = `<p class="help">The file's own face stays.</p>`;
    else if (!w.ready) body = `<p class="help">Twelve points on the eyes and mouth tell the build where the face is: without them the avatar can't blink or look around.</p>`;
    else if (w.marked) body = `<canvas id="quick-face" width="520" height="300"></canvas>`
      + look("four dots on each eye (both corners, top, bottom) and four round the mouth, each on the right spot. A dot on a brow, the nose or the hair gives a broken face.")
      + `<div class="row">${unseen.has(path) ? button("seen", "Looks right", "") : ""}${button("go:face", unseen.has(path) ? "Fix them by hand" : "Change them by hand")}</div>`;
    else body = `<p class="help">${detector() ? "The app can look for the face itself. It needs a face shown clearly from the front, and is often wrong on covered eyes, masks and unusual faces: you see its guess before anything is built."
        : `The face detector isn't installed (optional, 1.5 GB: <a href="#setup">Setup</a> gets it). Marking by hand takes a minute and is more exact.`}</p>`
      + (said[path] ? `<p class="msg bad">${esc(sentence(said[path]))} Mark the points by hand instead.</p>` : "")
      + `<div class="row">${detector() ? button("guess", "Find the face for me", said[path] ? "alt" : "") : ""}${button("go:face", "Mark it by hand", detector() && !said[path] ? "alt" : "")}${button("skip", skipped.has(path) ? "Left out: body only" : "Build without a face")}</div>`;
    cards.push(stage(2, "Find the face", text, kind, body));
  }
  { // 3: the build
    const [text, kind] = state("build", w.built), took = tookLast(S.status);
    // (a step no build does without, failed every way it was tried: the others are left out and said under 4)
    const failed = !w.built && S.status.steps.find((s) => s.runs && s.status === "failed" && s.failure && ["survey", "build", "tpose", "export", "vrm_check"].includes(s.step));
    cards.push(stage(3, "Build the avatar", text, kind, (w.built ? `<p class="help">Built${took ? ` in about ${took}` : ""}.</p>`
      : `<p class="help">Rig, hair, face, hands, checks, the .vrm file${took ? `: about ${took} last time` : ": a few minutes"}. A step that fails on this model is left out or done a plainer way, and the result says so.</p>`)
      + (failed ? `<p class="msg bad">${esc(sentence(failed.failure.what))} ${esc(sentence(failed.failure.fix || ""))}</p>` : "")
      + `<div class="row">${button("go:options", "Change options first")}${button("go:build", "See every step")}</div>`));
  }
  { // 4: what came out
    if (!w.built) cards.push(stage(4, "Check the result", "waiting", "", `<p class="help">The avatar moving, and what the build's own checks found.</p>`));
    else {
      const [name, kind, why] = tier(), v = S.status.verdict || {};
      const lines = [...(v.issues || []).map(([page, t]) => [page, t, "failed"]), ...(v.looks || []).map(([page, t]) => [page, t, ""])];
      cards.push(stage(4, "Check the result", name, kind, `<p class="help">${esc(why)}</p>`
        + look("turn the head, raise the arms, blink, open the mouth. Hair or clothes tearing, a face that doesn't move, hands that twist: each has a page that mends it.")
        + `<div class="row">${button("go:review", "See it move", "")}${button("go:movement", "What moves with what")}</div>`
        + (lines.length ? `<div class="qlines">${lines.slice(0, 8).map(([page, t, k]) => `<div class="todo"><span class="${k}">${esc(sentence(t))}</span>${button(`go:${page === "review" ? "build" : page}`, "Open")}</div>`).join("")}`
          + (lines.length > 8 ? `<p class="msg">and ${lines.length - 8} more on the Build page.</p>` : "") + `</div>` : "")));
    }
  }
  host.innerHTML = cards.join("");
  const go = $("quick-go");
  go.disabled = busy || !$("runbar").hidden;
  go.textContent = { ready: "Go: read the model", face: w.marked ? "The face looks right: build" : detector() && !said[path] ? "Find the face" : "Mark the face by hand",
    build: nextRun(S.status) ? "Build what changed" : "Build the avatar", result: "See it move" }[w.next];
  const [chip, chipKind] = w.next === "result" ? tier() : [`step ${["ready", "face", "build"].indexOf(w.next) + 1} of 4`, "warn"];
  $("quick-chip").textContent = chip; $("quick-chip").className = `chip ${chipKind}`;
  if (w.marked && w.ready) drawFace(w.face);
  host.querySelectorAll("[data-quick]").forEach((b) => b.onclick = () => act(b.dataset.quick));
}
app.renderQuick = renderQuick;

// The face picture with the points on it, drawn close: what the detector (or a person) marked, to be seen before a build.
function drawFace(face) {
  const canvas = $("quick-face"), p = face.points;
  if (!canvas || !face.picture || !p?.size) { if (canvas) canvas.hidden = true; return; }
  const dots = ["right_eye", "left_eye", "mouth"].flatMap((key) => Object.values(p[key] || {}).map((xy) => [key, ...xy]));
  if (!dots.length) { canvas.hidden = true; return; }
  const image = new Image();
  image.onload = () => {
    const xs = dots.map((d) => d[1]), ys = dots.map((d) => d[2]), scale = image.width / p.size[0];
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const wide = Math.max(Math.max(...xs) - Math.min(...xs), (Math.max(...ys) - Math.min(...ys)) * canvas.width / canvas.height) * 1.7, tall = wide * canvas.height / canvas.width;
    const g = canvas.getContext("2d"), k = canvas.width / wide;
    g.fillStyle = "#111"; g.fillRect(0, 0, canvas.width, canvas.height);
    g.drawImage(image, (cx - wide / 2) * scale, (cy - tall / 2) * scale, wide * scale, tall * scale, 0, 0, canvas.width, canvas.height);
    for (const [key, x, y] of dots) {
      g.beginPath(); g.arc((x - cx + wide / 2) * k, (y - cy + tall / 2) * k, 5, 0, 7);
      g.fillStyle = key === "mouth" ? "#ff5d8f" : "#3df2d8"; g.fill(); g.lineWidth = 2; g.strokeStyle = "#000"; g.stroke();
    }
  };
  image.src = face.picture;
}

async function act(what) {
  const path = S.manifestPath;
  if (what.startsWith("go:")) return app.go(what.slice(3));
  if (what === "seen") { unseen.delete(path); renderQuick(); return; }
  if (what === "skip") { skipped.has(path) ? skipped.delete(path) : skipped.add(path); renderQuick(); return; }
  if (what === "guess") return guess();
}

async function guess() {
  const path = S.manifestPath;
  busy = true; renderQuick();
  $("quick-msg").textContent = "Looking for the face…";
  const found = await app.face.guess();
  busy = false; $("quick-msg").textContent = "";
  if (found.error !== undefined) said[path] = found.error || "The detector was stopped";
  else { delete said[path]; unseen.add(path); toast(`The detector placed the points (${Math.round(found.sure * 100)}% sure of its least certain one). Look at them before building.`, "info", { ms: 8000 }); }
  renderQuick();
}

// The big button: the next thing, and on through whatever needs no person. It stops where one should look:
// at a detector's guess, at a face it couldn't find, at the result.
async function go() {
  if (!S.manifest) return;
  const ended = (state) => !state || state.error || state.cancelled;
  let w = where();
  if (w.next === "ready") { if (ended(await app.job("prepare", {}))) return; w = where(); }
  if (w.next === "face") {
    if (w.marked) unseen.delete(S.manifestPath);
    else if (detector() && !said[S.manifestPath]) return guess();
    else return app.go("face");
    renderQuick(); w = where();
  }
  if (w.next === "build") {
    const next = nextRun(S.status);
    if (ended(await app.job("run", next ? { from: next.from } : {}))) return renderQuick();
    return renderQuick();
  }
  if (w.next === "result") app.go("review");
}
app.quickGo = go;
$("quick-go").onclick = go;
$("quick-all").onclick = () => app.go("menu");
onShow.quick = renderQuick;

// The mode's switch, in the top bar beside the other switches: where a character opens. Nothing is hidden in
// either mode and nothing is lost by changing it.
function modeButton() {
  const bar = document.querySelector(".tbs");
  if (!bar || $("a-mode")) return;
  bar.insertAdjacentHTML("afterbegin", `<button id="a-mode"></button>`);
  const show = () => {
    $("a-mode").textContent = automatic() ? "Mode: automatic" : "Mode: advanced";
    $("a-mode").title = automatic() ? "a character opens on Quick build: one screen, the app does each step. Click for advanced: every page, every choice"
      : "a character opens on the menu of every page. Click for automatic: one screen, the app does each step";
  };
  $("a-mode").onclick = () => {
    try { localStorage.setItem(MODE, automatic() ? "advanced" : "automatic"); } catch {}
    show();
    toast(automatic() ? "Automatic: characters open on Quick build." : "Advanced: characters open on the menu of every page.", "info");
    if (S.manifest && ["quick", "menu"].includes(location.hash.slice(1))) app.go(app.landing());
  };
  show();
}
modeButton();
