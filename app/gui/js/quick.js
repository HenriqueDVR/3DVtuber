// Quick build: the automatic mode's one screen. Model to avatar, one thing at a time: what the app is doing or
// what it needs from the person, in a sentence or two, with one button. The user, 2026-11-10: the default should be
// "quickly test it out and see what happens", a new character landed on "a random page with a ton of information",
// and people "who don't know what they're doing" should get through it, while those who do can "refine and tinker".
// So: one card for now, a waiting view that says what comes next, and everything else folded under "Refine it
// yourself". The face detector "fails if there's not a clearly shown face": marking by hand is always one click
// away, and its guess is never built on unseen.
import { $, S, app, esc, sentence, served, tookLast } from "./core.js";
import { nextRun, onShow, TITLES } from "./nav.js";
import { toast } from "./ui.js";

const MODE = "oc2vrm-mode";
export const automatic = () => { try { return localStorage.getItem(MODE) !== "advanced"; } catch { return true; } };
// where a character opens: the main menu, with the steps short in the drawer at the side (renderSide). A new character goes
// to this page instead, which says what the app is doing while it reads the model (home.js)
app.landing = () => "menu";

// Per character, for this visit: the face left out on purpose, a detector's guess no one has looked at yet,
// and what the detector last said. `doing`: what this page started and is waiting for.
const skipped = new Set(), unseen = new Set(), said = {};
let doing = null;

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

// What the finished file is, in a word: from what the build itself left out (engine.PLAINER, SOFT).
function tier() {
  const st = S.status, v = st.verdict || {}, ships = st.ships || {};
  const plain = step("build")?.report?.plainer, without = (v.issues || []).filter(([, t]) => /^built without /.test(t)).map(([, t]) => t.slice(14));
  if (plain || !ships.blink) return ["Body only", "warn", plain ? "The full build failed on this model, so it was made the plain way: the body moves, the face and hair don't."
    : "The body moves. The face doesn't blink: it has no face points."];
  if (without.length) return ["Basic", "warn", `It works, without ${without.join(", ")}: ${without.length > 1 ? "those steps" : "that step"} failed on this model and ${without.length > 1 ? "were" : "was"} left out.`];
  if (!v.ready) return ["Needs a look", "warn", "It was built whole, and a check found something to fix."];
  return ["Full", "good", v.looks?.length ? "Every check passed; a few things are worth a look." : "Every check passed."];
}

const button = (id, text, kind = "alt") => `<button class="act ${kind}" data-quick="${id}" data-job>${text}</button>`;
const STAGES = [["ready", "Read the model"], ["face", "Check the face"], ["build", "Build"], ["result", "See the result"]];
// What the app is doing while the person waits: [headline, what it is and how long, what they will do after]
const WAITING = {
  prepare: ["Reading your model…", "The app is measuring it, listing its parts and taking a picture of the face. About a minute.",
    "you check where the eyes and mouth are."],
  guess: ["Looking for the face…", "A few seconds.", "you see where it thinks the eyes and mouth are, and say if that is right."],
  build: ["Building your avatar…", "Skeleton, hair, face, hands, then the checks and the file.", "you see it move, with anything worth a look pointed out."],
};
// The pages behind this one, for whoever wants to change something themselves: [page, what it is for]
const REFINE = [["model", "Turn the model round, or change whose skeleton is used"], ["parts", "Say what is hair, a cape, a tail, a hat: what swings and what stays"],
  ["joints", "Move a shoulder, elbow, hip or knee the app placed wrong"], ["face", "Move the face dots; choose how the eyes and mouth move"],
  ["paint", "Paint over the model's texture"], ["options", "Switches for the whole avatar: rigid head, cloth, fingers, chest"],
  ["build", "Every build step, its log, and running from any one of them"], ["movement", "What moves with the body, and what was left behind"],
  ["review", "The avatar moving, pose by pose; pictures; sending it to your tracking app"]];

export function renderQuick() {
  const now = $("quick-now");
  if (!now) return;
  if (!S.manifest || !S.status) {
    now.innerHTML = `<h3>No character open</h3><p>Pick one on Character select, or start a new one from a model.</p><div class="row">${button("go:model", "New character", "")}${button("go:home", "Character select")}</div>`;
    $("quick-strip").innerHTML = ""; $("quick-more").hidden = true; $("quick-chip").textContent = "";
    hook(); return;
  }
  if (!automatic()) return overview();
  const w = where(), path = S.manifestPath, at = STAGES.findIndex(([name]) => name === w.next), took = tookLast(S.status);
  $("quick-strip").innerHTML = STAGES.map(([, title], i) => `<li class="${i < at || (i === at && w.next === "result") ? "done" : i === at ? "on" : ""}"><b>${i + 1}</b>${title}</li>`).join("");
  $("quick-more").hidden = false;
  $("quick-refine").innerHTML = REFINE.filter(([page]) => !document.querySelector(`#steps button[data-go="${page}"]`)?.classList.contains("locked"))
    .map(([page, text]) => `<div class="todo"><span>${text}</span><button class="act alt small" data-quick="go:${page}">${page[0].toUpperCase() + page.slice(1)}</button></div>`).join("")
    + `<div class="todo"><span>All of them as a menu. To always open characters there, switch the mode in the top bar to advanced.</span><button class="act alt small" data-quick="go:menu">All pages</button></div>`;

  let html;
  if (doing) {
    const [head, what, after] = WAITING[doing];
    html = `<div class="qwait"><i></i><h3>${head}</h3></div><p>${what}${doing === "build" && took ? ` About ${took} last time.` : doing === "build" ? " A few minutes." : ""}
      The bar at the top shows how far it is. Nothing to do but wait.</p><p class="qafter"><b>After this:</b> ${after}</p>`;
  } else if (w.next === "ready") {
    html = `<h3>First, the app reads your model</h3><p>It measures it, lists its parts and takes a picture of the face. About a minute, and nothing for you to do.</p>
      <div class="row">${button("go", "Start", "")}</div>`;
  } else if (w.next === "face" && w.marked) {
    html = `<h3>Are the dots in the right places?</h3><canvas id="quick-face" width="520" height="300"></canvas>
      <p>Each eye should have four dots: both corners, top and bottom. The mouth has four round it. A dot on a brow, the nose or the hair gives a broken face.</p>
      <div class="row">${button("go", "Yes, build it", "")}${button("go:face", "No, I'll fix them")}</div>`;
  } else if (w.next === "face") {
    const auto = detector() && !said[path];
    html = `<h3>${said[path] ? "The app couldn't find the face" : "Next: where is the face?"}</h3>`
      + (said[path] ? `<p class="msg bad">${esc(sentence(said[path]))}</p><p>That happens when the face isn't shown clearly from the front. Marking it yourself takes about a minute: you click the corners of each eye and the mouth on a picture.</p>`
        : `<p>The avatar needs to know where the eyes and mouth are, so it can blink and talk. ${auto ? "The app can look for them, and you check its guess. Or mark them yourself: about a minute, and more exact."
          : `You click the corners of each eye and the mouth on a picture: about a minute.${detector() ? "" : ` (The app can also look for them itself once the optional face detector is installed: <a href="#setup">Setup</a>.)`}`}</p>`)
      + `<div class="row">${auto ? button("guess", "Find it for me", "") : ""}${button("go:face", auto ? "I'll mark it myself" : "Mark the face", auto ? "alt" : "")}${button("skip", "Skip: no face")}</div>
      <p class="help">Skipping builds a body that moves, with a face that doesn't blink or talk.</p>`;
  } else if (w.next === "build") {
    // (a step no build does without, failed every way it was tried: the others are left out and said in the result)
    const failed = S.status.steps.find((s) => s.runs && s.status === "failed" && s.failure && ["survey", "build", "tpose", "export", "vrm_check"].includes(s.step));
    html = failed ? `<h3>The build stopped</h3><p class="msg bad">${esc(sentence(failed.failure.what))}</p><p>${esc(sentence(failed.failure.fix || "Try again, or report it with the flag in the top bar"))}</p>
        <div class="row">${button("go", "Try again", "")}${button("go:build", "See what happened")}</div>`
      : `<h3>${nextRun(S.status) ? "Something changed: build again" : "Ready to build"}</h3><p>${took ? `About ${took}.` : "A few minutes."} If a part of the build fails on this model, it is left out or done a simpler way, and you are told which.</p>
        <div class="row">${button("go", nextRun(S.status) ? "Build what changed" : "Build the avatar", "")}</div>
        ${!w.marked ? `<p class="help">No face marked: the body will move, the face won't. ${`<a href="#face">Mark it first</a>`}.</p>`
          : w.kept ? "" : `<p class="help">The face as it is marked (<a href="#face">change it</a>):</p><canvas id="quick-face" class="small" width="520" height="300"></canvas>`}`;
  } else {
    const [name, kind, why] = tier(), v = S.status.verdict || {};
    const lines = [...(v.issues || []).map(([page, t]) => [page, t, "failed"]), ...(v.looks || []).map(([page, t]) => [page, t, ""])];
    html = `<h3>${kind === "good" ? "Your avatar is ready" : "Your avatar is built"} <span class="chip ${kind}">${name}</span></h3><p>${esc(why)}</p>
      <p>Now look at it move: turn the head, raise the arms, blink, open the mouth. Then try it in your tracking app.</p>
      <div class="row">${button("go", "See it move", "")}</div>`
      + (lines.length ? `<div class="qlines"><div class="title">${lines.some((l) => l[2]) ? "To fix" : "Worth a look"}</div>${lines.slice(0, 6).map(([page, t, k]) => `<div class="todo"><span class="${k}">${esc(sentence(t))}</span><button class="act alt small" data-quick="go:${page === "review" ? "build" : page}">Open</button></div>`).join("")}`
        + (lines.length > 6 ? `<p class="msg">and ${lines.length - 6} more on the Build page.</p>` : "") + `</div>` : "");
  }
  now.innerHTML = html;
  now.classList.toggle("waiting", !!doing);
  const [chip, chipKind] = w.next === "result" ? tier() : [`${at + 1} of 4`, "warn"];
  $("quick-chip").textContent = chip; $("quick-chip").className = `chip ${chipKind}`;
  if ($("quick-face")) drawFace(w.face);
  hook();
}
app.renderQuick = () => { renderQuick(); renderSide(); };

// The guide: your steps in a drawer at the window's right, on every page of an open character. The user,
// 2026-11-10: the steps sat on the main menu alone, so a step that sent you to the Face page left them behind and
// "you can't really come back... and can get lost"; they wanted them "to the side", a guide you "pop out and in to
// hide". Then: it "could look much better and be more modern... it should feel good to use it and be informative".
// So: a timeline. How far along (a bar that fills), each step a node on a line (done ticked, the one due pulsing,
// a locked one saying why), the step due opened with what it is and its button, the page shown marked "you are
// here". Folded in, a slim rail of the same dots. Automatic: the four stages, every one leading to the one screen.
// Advanced: the pages. Drawn again only when something in it changed, so its animations play when they mean
// something: the rows rise in when it opens, a tick pops when a step gets done, the bar slides.
const GUIDE = "oc2vrm-guide", BARE = ["home", "setup", "help"];
const tucked = () => { try { return localStorage.getItem(GUIDE) === "in"; } catch { return false; } };
const STAGE_TEXT = { ready: "It measures the model, lists its parts and takes a picture of the face. About a minute.",
  face: "Where the eyes and mouth are, so the avatar can blink and talk.",
  build: "Skeleton, hair, face and hands, then the checks and the file. A few minutes.",
  result: "Watch it move: turn the head, raise the arms, blink. Then try it in your tracking app." };
const TICK = `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.5l3.2 3.2L13 5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const LOCK = `<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="3.5" y="7" width="9" height="6.5" rx="1" fill="currentColor"/><path d="M5.5 7V5.2a2.5 2.5 0 015 0V7" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>`;
let drawn = "", states = {}, shownFor = null;

function drawer() {
  let g = $("guide");
  if (g) return g;
  $("main").insertAdjacentHTML("afterend", `<aside id="guide" aria-label="Your steps">
    <button id="guide-rail" title="show your steps"><span class="lbl">Steps</span><span class="dots"></span><b class="count"></b></button>
    <div id="guide-body"><header class="g-head"><div><small id="g-mode"></small><h4 id="g-name"></h4></div>
      <button class="tuck" data-tuck title="fold your steps away (the rail at the right brings them back)" aria-label="fold your steps away">
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3l5 5-5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></button></header>
      <div class="g-prog"><div class="g-bar"><i id="g-fill"></i></div><span id="g-count"></span></div>
      <div id="g-where"></div><ol class="g-steps" id="g-steps"></ol><div id="g-foot"></div></div></aside>`);
  g = $("guide");
  const flip = () => {
    try { localStorage.setItem(GUIDE, tucked() ? "out" : "in"); } catch {}
    document.body.classList.toggle("guide-in", tucked());
    if (!tucked()) enter();
  };
  $("guide-rail").onclick = flip;
  g.addEventListener("click", (e) => {
    if (e.target.closest("[data-tuck]")) return flip();
    const b = e.target.closest("[data-side]");
    if (b && !b.disabled) app.go(b.dataset.side);
  });
  return g;
}
// the rows rise in one after another: when the drawer comes out, or another character is opened
function enter() {
  const list = $("g-steps");
  list.classList.remove("enter"); void list.offsetWidth; list.classList.add("enter");
  setTimeout(() => list.classList.remove("enter"), 900);
}

// What the guide says, as data: each step with how it stands, the one due, and how far along.
function plan(here) {
  const nav = (page) => document.querySelector(`#steps button[data-go="${page}"]`);
  const done = (page) => !!nav(page)?.querySelector(".n.done"), locked = (page) => nav(page)?.classList.contains("locked") ? nav(page).title : "";
  if (automatic()) {
    const w = where(), at = STAGES.findIndex(([name]) => name === w.next), over = w.next === "result" && w.built;
    const label = doing ? WAITING[doing][0] : { ready: "Read the model", face: w.marked ? "Check the face dots" : "Find the face",
      build: nextRun(S.status) ? "Build what changed" : "Build the avatar", result: "See it move" }[w.next];
    const items = STAGES.map(([name, title], i) => ({ key: name, title, text: STAGE_TEXT[name], side: name === "result" && w.built ? "review" : "quick",
      state: i < at || over ? "done" : i === at ? "now" : "todo", note: i < at || over ? "done" : i === at ? (doing ? "working" : "next") : "" }));
    const now = items.find((it) => it.state === "now") || items[items.length - 1];
    return { items, now, label, to: over ? "review" : "quick", done: items.filter((it) => it.state === "done").length, busy: !!doing, foot: ["quick", "Open Quick build"] };
  }
  const first = PLAN.map(([page]) => page).find((page) => !done(page) && !locked(page) && !["joints", "options"].includes(page)) || "review";
  const items = PLAN.map(([page, text, note]) => {
    const why = locked(page), badge = document.querySelector(`[data-badge="${page}"]`)?.textContent || "";
    return { key: page, title: TITLES[page], text, side: page, why, state: done(page) ? "done" : why ? "locked" : page === first ? "now" : "todo",
      note: done(page) ? "done" : why ? "locked" : badge || note };
  });
  return { items, now: items.find((it) => it.key === first), label: TITLES[first], to: first, done: items.filter((it) => it.state === "done").length, foot: ["quick", "Overview: what each step is for"] };
}

function renderSide() {
  drawer();
  const here = location.hash.slice(1) || "home", on = !!(S.manifest && S.status) && !BARE.includes(here);
  document.body.classList.toggle("guide", on);
  document.body.classList.toggle("guide-in", tucked());
  if (!on) { drawn = ""; return; }
  const p = plan(here), page = here === "paint" ? "face" : here, total = p.items.length;
  const name = S.projects?.find((x) => x.manifest === S.manifestPath)?.name || S.manifest.name || "";
  // where you are: a page off the guide's own path (automatic: any page but the one screen) says what it is for,
  // with the way back
  const about = REFINE.find(([q]) => q === here)?.[1] || (here === "paint" ? "Paint over the model's texture" : "");
  const away = automatic() && here !== "quick" && here !== "menu" && about;
  const shape = JSON.stringify([p, here, name, automatic(), tucked()]);
  if (shape === drawn) return;
  drawn = shape;
  if (shownFor !== S.manifestPath) { shownFor = S.manifestPath; states = {}; if (!tucked()) enter(); }

  $("g-mode").textContent = `Your steps · ${automatic() ? "automatic" : "advanced"}`;
  $("g-name").textContent = name;
  $("g-fill").style.width = `${Math.round(100 * p.done / total)}%`;
  $("g-count").textContent = p.done === total ? "all done" : `${p.done} of ${total}`;
  html($("g-where"), away ? `<div class="g-here"><span class="pin">You are here</span><b>${esc(TITLES[here] || here)}</b><p>${esc(about)}.</p>
    <button class="g-back" data-side="${p.to}">← ${p.to === "review" ? "See it move" : "Back to Quick build"}</button></div>` : "");
  html($("g-steps"), p.items.map((it, i) => {
    const open = it === p.now || (!automatic() && it.key === page), here_ = !automatic() && it.key === page;
    const just = states[it.key] && states[it.key] !== "done" && it.state === "done";
    const go = it === p.now && !away ? `<button class="act g-go" data-side="${p.to}">${p.busy ? `<i class="spin"></i>${esc(p.label)}` : `${it.state === "done" ? "" : "Next: "}${esc(p.label)}`} →</button>` : "";
    return `<li class="s-${it.state}${open ? " open" : ""}${here_ ? " here" : ""}${just ? " just" : ""}" style="--i:${i}">
      <button class="g-row" data-side="${it.side}" ${it.why ? `disabled title="${esc(it.why)}"` : `title="${esc(it.text)}"`}>
        <span class="node">${it.state === "done" ? TICK : it.state === "locked" ? LOCK : i + 1}</span>
        <span class="t">${esc(it.title)}</span>${here_ ? `<span class="pin">here</span>` : ""}<em>${esc(it.note)}</em></button>
      <div class="g-more"><div><p>${esc(it.why ? `Locked: ${it.why}.` : it.text)}</p>${go}</div></div></li>`;
  }).join(""));
  states = Object.fromEntries(p.items.map((it) => [it.key, it.state]));
  html($("g-foot"), `<button class="g-all" data-side="${p.foot[0]}">${esc(p.foot[1])}</button>`);
  // folded in: the same steps as dots on the rail
  $("guide-rail").querySelector(".dots").innerHTML = p.items.map((it) => `<i class="s-${it.state}"></i>`).join("");
  $("guide-rail").querySelector(".count").textContent = `${p.done}/${total}`;
}
const html = (el, s) => { if (el.innerHTML !== s) el.innerHTML = s; };
app.renderSide = renderSide;

// The advanced mode's version of this page: the same waiting view while the model is read, then the steps as a
// short list, each with what the person does there and how it stands, and the next one as the main button. The
// user, 2026-11-10: a new character in advanced mode was thrown at "step 1 of 8", a page of survey numbers, with
// no word of what was going on or what came next.
const PLAN = [["parts", "Say what is hair, a cape, a tail or a hat, so it swings instead of stretching.", "worth a look"],
  ["joints", "See that shoulders, elbows, hips and knees sit inside the limbs.", "usually fine as found"],
  ["face", "Mark the eyes and mouth on a picture, and choose how they move.", "needed to blink and talk"],
  ["options", "Switches for the whole avatar. Automatic is a good start.", "optional"],
  ["build", "The app runs every step and writes the avatar file. A few minutes.", ""],
  ["movement", "After a build: what was left behind when the body moves.", "after building"],
  ["review", "After a build: the avatar moving, pose by pose, and sending it to your tracking app.", "after building"]];
function overview() {
  const now = $("quick-now"), w = where();
  $("quick-strip").innerHTML = ""; $("quick-more").hidden = true;
  const nav = (page) => document.querySelector(`#steps button[data-go="${page}"]`);
  const done = (page) => !!nav(page)?.querySelector(".n.done"), locked = (page) => nav(page)?.classList.contains("locked") ? nav(page).title : "";
  const plan = () => `<div class="qlines">${PLAN.map(([page, text, note], i) => {
    const why = locked(page), badge = document.querySelector(`[data-badge="${page}"]`)?.textContent || "";
    return `<div class="todo"><span><b>${i + 1} · ${TITLES[page]}</b> <em class="chip ${done(page) ? "done" : ""}">${done(page) ? "done" : esc(badge || note)}</em><br>${text}</span>
      <button class="act alt small" data-quick="go:${page}" ${why ? `disabled title="${esc(why)}"` : ""}>Open</button></div>`;
  }).join("")}</div>`;
  if (doing) {
    const [head, what] = WAITING[doing];
    now.innerHTML = `<div class="qwait"><i></i><h3>${head}</h3></div><p>${what} The bar at the top shows how far it is. Nothing to do but wait.</p>
      <p class="qafter"><b>After this:</b> you go through the steps below at your own pace. Only the face is needed; the rest is there to make it better.</p>${plan()}`;
  } else if (!w.ready) {
    now.innerHTML = `<h3>First, the app reads your model</h3><p>It measures it, lists its parts and takes a picture of the face. About a minute, and nothing for you to do.
      After that the steps below open up.</p><div class="row">${button("prepare", "Start", "")}</div>${plan()}`;
  } else {
    const next = PLAN.map(([page]) => page).find((page) => !done(page) && !locked(page) && !["joints", "options"].includes(page)) || "review";
    now.innerHTML = `<h3>${w.built ? "Built. Look it over, or change something and build again" : "Your steps"}</h3>
      <p>${w.built ? "Each page below changes one thing; the app then says what to rebuild." : "Go through them in order, at your own pace. Only the face is needed; the rest is there to make it better."}</p>
      <div class="row">${button(`go:${next}`, `Next: ${TITLES[next]}`, "")}${button("go:menu", "Main menu")}</div>${plan()}`;
  }
  now.classList.toggle("waiting", !!doing);
  $("quick-chip").textContent = "advanced"; $("quick-chip").className = "chip";
  hook();
}
function hook() {
  $("quick").querySelectorAll("[data-quick]").forEach((b) => {
    b.onclick = () => act(b.dataset.quick);
    if (b.hasAttribute("data-job") && !b.dataset.quick.startsWith("go:")) b.disabled = !!doing || !$("runbar").hidden;
  });
}

// The face picture with the points on it, drawn close: what the detector (or a person) marked, to be seen before a build.
// The picture is the Face page's when it has it loaded, else read from the character's folder: the page asked "are
// the dots in the right places?" over no picture at all when the Face page's copy wasn't there yet (the user,
// 2026-11-10). With no picture to be had it says so, and sends to the Face page, where the dots can be seen.
function drawFace(face) {
  const canvas = $("quick-face"), p = face.points;
  if (!canvas) return;
  const dots = ["right_eye", "left_eye", "mouth"].flatMap((key) => Object.values(p?.[key] || {}).map((xy) => [key, ...xy]));
  const none = () => { canvas.outerHTML = `<p class="msg bad" id="quick-face">The picture of the face couldn't be shown here. Open the Face page to see the dots before you build.</p>`; };
  if (!dots.length || !p?.size) return none();
  const from = [face.picture, served(`${S.status.out}/face/face_patch_head_000.png`),
    S.manifest.source && served(`${S.status.out}/${S.manifest.source}`.replace(/[^\\/]+$/, "face_patch_head_000.png"))].filter(Boolean);
  const image = new Image();
  image.onerror = () => { if (from.length) image.src = from.shift(); else if (canvas.isConnected) none(); };
  image.onload = () => {
    if (!canvas.isConnected) return;
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
  image.src = from.shift();
}

function act(what) {
  const path = S.manifestPath;
  if (what.startsWith("go:")) return app.go(what.slice(3));
  if (what === "skip") { skipped.add(path); renderQuick(); return; }
  if (what === "guess") return guess();
  if (what === "prepare") return prepare();
  if (what === "go") return go();
}

// One thing this page started, with the waiting view up while it runs.
async function wait(what, work) {
  doing = what; app.renderQuick();
  try { return await work(); } finally { doing = null; app.renderQuick(); }
}

async function guess() {
  const path = S.manifestPath;
  const found = await wait("guess", () => app.face.guess());
  if (found.error !== undefined) said[path] = found.error || "The search was stopped";
  else { delete said[path]; unseen.add(path); }
  renderQuick();
}

// The main button: the next thing, and on through whatever needs no person. It stops where one should look:
// at a detector's guess, at a face it couldn't find, at the result.
async function go() {
  if (!S.manifest || doing) return;
  const ended = (state) => !state || state.error || state.cancelled;
  let w = where();
  if (w.next === "ready") { if (ended(await wait("prepare", () => app.job("prepare", {})))) return; w = where(); }
  if (w.next === "face") {
    if (w.marked) unseen.delete(S.manifestPath);
    else if (detector() && !said[S.manifestPath]) return guess();
    else return renderQuick();
    w = where();
  }
  if (w.next === "build") { await wait("build", () => app.job("run", nextRun(S.status) ? { from: nextRun(S.status).from } : {})); return; }
  if (w.next === "result") app.go("review");
}
// Advanced mode after a character is made: the model read, with the waiting view up, and then the steps.
const prepare = () => doing ? null : wait("prepare", () => app.job("prepare", {}));
app.quickGo = () => automatic() ? go() : prepare();
onShow.quick = renderQuick;

// The mode's switch, in the top bar beside the other switches: where a character opens, and how much the New
// character page asks. Nothing is lost by changing it.
function modeButton() {
  const bar = document.querySelector(".tbs");
  const show = () => {
    document.body.classList.toggle("mode-auto", automatic());
    // the page is the one screen in one mode and the list of steps in the other
    TITLES.quick = automatic() ? "Quick build" : "Overview";
    const head = $("quick").querySelector(".page-head");
    head.querySelector(".kicker").textContent = automatic() ? "Automatic" : "Advanced";
    head.querySelector("h2").firstChild.textContent = `${TITLES.quick} `;
    head.querySelector(".lead").textContent = automatic() ? "One thing at a time: the app does the work and tells you when it needs you."
      : "What the app is doing, and the steps that are yours, in order.";
    document.querySelector('#steps button[data-go="quick"] .t').textContent = TITLES.quick;
    app.renderQuick();
    // the same choice where a character is made (the user, 2026-11-10: it belongs there, not only in the top bar)
    document.querySelectorAll("#new-mode button").forEach((b) => {
      const on = (b.dataset.mode === "automatic") === automatic();
      b.classList.toggle("on", on); b.setAttribute("aria-checked", on);
    });
    if (!$("a-mode")) return;
    $("a-mode").textContent = automatic() ? "Mode: automatic" : "Mode: advanced";
    $("a-mode").title = automatic() ? "a character opens on Quick build: one thing at a time, the app does each step. Click for advanced: the steps as a list, every page and choice yours"
      : "a character opens on its list of steps, yours to go through page by page. Click for automatic: one thing at a time, the app does each step";
  };
  if (bar && !$("a-mode")) {
    bar.insertAdjacentHTML("afterbegin", `<button id="a-mode"></button>`);
    $("a-mode").onclick = () => {
      try { localStorage.setItem(MODE, automatic() ? "advanced" : "automatic"); } catch {}
      show();
      toast(automatic() ? "Automatic: the app does each step and tells you when it needs you." : "Advanced: the steps are yours, page by page.", "info");
      if (S.manifest && ["quick", "menu"].includes(location.hash.slice(1))) app.go(app.landing());
    };
  }
  document.querySelectorAll("#new-mode button").forEach((b) => b.onclick = () => {
    try { localStorage.setItem(MODE, b.dataset.mode); } catch {}
    show();
  });
  show();
}
modeButton();
