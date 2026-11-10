// The arcade look (arcade.css): the stage behind the pages, character select, the main menu and its
// lists, the build band, sparks, sound. It adds its own elements and reads what the pages already
// drew (the sidebar's badges, the Characters cards, the run bar and its events), so the pages know
// nothing of it. It is the app's one look: the plain one it had before (app.css alone, none of this
// run) could be chosen on Setup until 2026-10-12.
import { $, S, app, esc, lastVrm, stepName, tookLast } from "./core.js";
import { arm, mood, musicOn, musicVolume, play, setMusic, setVolume, soundOn, switchMusic, switchSound, volume } from "./sfx.js";
import { ORDER, after, before, skipped } from "./nav.js";

const root = document.documentElement;
const MOTION = "vtuber-maker-motion";

// Setup's switches
$("look-still").checked = root.classList.contains("still");
$("look-still").onchange = () => {
  try { localStorage.setItem(MOTION, $("look-still").checked ? "off" : "on"); } catch {}
  root.classList.toggle("still", $("look-still").checked);
};
$("look-sound").value = volume();
$("look-sound").oninput = () => { setVolume(+$("look-sound").value); tuneButton(); };
$("look-sound").onchange = () => play("select");
$("look-music").value = musicVolume();
$("look-music").oninput = () => { setMusic(+$("look-music").value); tuneButton(); };
// the top bar's two switches say which way they stand
function tuneButton() {
  for (const [id, is, name] of [["a-music", musicOn(), "♪ Music"], ["a-sfx", soundOn(), "♫ Sounds"]]) {
    const b = $(id);
    if (b) { b.classList.toggle("off", !is); b.textContent = `${name} ${is ? "on" : "off"}`; b.setAttribute("aria-pressed", is); }
  }
}
const reduced = matchMedia("(prefers-reduced-motion: reduce)"), still = () => root.classList.contains("still") || reduced.matches;

const A = (id) => $(`a-${id}`);
const GROUPS = { setup: ["model", "parts", "joints", "face", "paint", "options"], check: ["movement", "review"] };
const CARDS = [["Set up", "setup", "#2a5cff"], ["Build", "build", "#ff3d8b"], ["Check", "check", "#ffb224"]];
const TITLES = { home: "Character select", model: "Model", parts: "Parts", joints: "Joints", face: "Face", paint: "Paint", options: "Options",
  build: "Build", movement: "Movement", review: "Review", help: "Stuck?", setup: "Setup" };
const SAYS = { paint: "Fix the texture on the model itself: a smudge by an eye, a seam of the wrong colour." };
const NEW = "#39404f";   // the band behind "?": no character, no colour
let page = "home", opened = null, opening = null, sel = 0, card = 0, item = 0, group = "setup", colour = "#2a5cff";
let roster = null, cheer = null, ran = { was: null }, wasRunning = false, share = 0, settledWas = 0, flash = 0;

// ---------- what the pages already know, read from where they drew it
const tab = (name) => document.querySelector(`#steps button[data-go="${name}"]`);
const done = (name) => !!tab(name)?.querySelector(".n.done");
const badge = (name) => tab(name)?.querySelector(".badge")?.textContent || "";
const why = (name) => tab(name)?.title || "";
// a badge that asks for something (yellow or red in the sidebar), not one that only reports
const asks = (name) => /warn|bad/.test(tab(name)?.querySelector(".badge")?.className || "");
// the open character; while one is being opened, the one picked
const open = () => opening || S.projects.find((p) => p.manifest === S.manifestPath);
// whether the badges are the shown character's yet (they are the last one's until it has loaded)
const known = () => !opening || (S.manifestPath === opening.manifest && !!S.manifest);
const kick = (el, c) => { el.classList.remove(c); void el.offsetWidth; el.classList.add(c); };
const at = (el, fx = .5, fy = .5) => { const r = el.getBoundingClientRect(); return [r.left + r.width * fx, r.top + r.height * fy]; };
// redrawn only when it reads differently: a redraw replays every animation in it
const html = (el, text) => { if (el._html === text) return false; el._html = text; el.innerHTML = text; return true; };
const put = (el, text) => { if (el.textContent === text) return false; el.textContent = text; return true; };

// A character's render cut to one frame for every character and pose, and the colour it is mostly
// drawn in. Renders fit their camera to the pose (arms out: a small figure; arms down: a big one), so
// each is laid into a square 1.3 figures high: feet on its floor, the body's axis down its middle, the
// top of the head 21% down. arcade.css places and sizes the pictures by those numbers.
const TALL = 1.3, ART = new Map();
function art(url, meta) {
  if (!url) return Promise.resolve(null);
  if (!ART.has(url)) ART.set(url, new Promise((ready) => {
    const img = new Image();
    img.onerror = () => ready(null);
    img.onload = () => {
      try {
        const k = Math.min(1, 2048 / Math.max(img.naturalWidth, img.naturalHeight)), w = Math.round(img.naturalWidth * k), h = Math.round(img.naturalHeight * k);
        const c = Object.assign(document.createElement("canvas"), { width: w, height: h }), g = c.getContext("2d", { willReadFrequently: true });
        g.drawImage(img, 0, 0, w, h);
        const d = g.getImageData(0, 0, w, h).data, bins = new Float32Array(24 * 4);
        let y0 = h, y1 = -1, sx = 0, n = 0;
        for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {
          const i = (y * w + x) * 4;
          if (d[i + 3] < 40) continue;
          if (y < y0) y0 = y; if (y > y1) y1 = y; sx += x; n += 1;
          // the colour: the fullest hue among the saturated pixels (skin and greys are left out)
          const r = d[i], gr = d[i + 1], b = d[i + 2], mx = Math.max(r, gr, b), mn = Math.min(r, gr, b), s = mx ? (mx - mn) / mx : 0;
          if (s < .45 || mx < 90) continue;
          const bin = Math.floor(hue(r, gr, b) / 15) % 24 * 4, wt = s * mx / 255;
          bins[bin] += wt; bins[bin + 1] += r * wt; bins[bin + 2] += gr * wt; bins[bin + 3] += b * wt;
        }
        if (!n) return ready(null);
        let best = 0;
        for (let bin = 4; bin < bins.length; bin += 4) if (bins[bin] > bins[best]) best = bin;
        const tone = bins[best] > 20 ? `hsl(${Math.round(hue(bins[best + 1], bins[best + 2], bins[best + 3]))} 85% 55%)` : null;
        // the head's top: the first row with anything on the body's axis. Raised arms reach higher than the
        // head, off the axis, and must not count as height
        const axis = sx / n, strip = Math.max(2, Math.round((y1 - y0) * .06));
        let top = y0;
        scan: for (let y = y0; y < y1; y++) for (let x = Math.max(0, Math.round(axis - strip)); x <= Math.min(w - 1, axis + strip); x++)
          if (d[(y * w + x) * 4 + 3] >= 40) { top = y; break scan; }
        if (y1 - top < (y1 - y0) * .5) top = y0;
        // at the render's own size: the band shows the figure about as tall as the render made it, and a
        // picture shrunk here and stretched again there was blurred twice
        // A posed picture (render_portraits.py) says how tall the character is in it and where its head and
        // floor are: the head goes on the middle line, whatever the arms and the hair do. A plain render is measured.
        const tall = meta ? meta.height * h : y1 - top, middle = meta ? meta.head[0] * w : axis, ground = meta ? meta.floor * h : y1;
        const side = Math.round(tall * TALL), cut = Object.assign(document.createElement("canvas"), { width: side, height: side });
        cut.getContext("2d").drawImage(c, Math.round(side / 2 - middle), Math.round(side * .98 - ground));
        // the roster's close-up: a square on the face, a little wider than it (hair, ears, the brim of a hat), the
        // face above its middle: the slot's name lies over the bottom of it
        let face = null;
        if (meta?.face) {
            const wide = meta.face * w * 1.7, close = Object.assign(document.createElement("canvas"), { width: 168, height: 168 });
            close.getContext("2d").drawImage(c, meta.head[0] * w - wide / 2, meta.head[1] * h - wide * .44, wide, wide, 0, 0, 168, 168);
            face = close.toDataURL("image/png");
        }
        cut.toBlob((blob) => ready({ src: URL.createObjectURL(blob), colour: tone, face }));
      } catch { ready(null); }
    };
    img.src = url;
  }));
  return ART.get(url);
}
function hue(r, g, b) {
  const mx = Math.max(r, g, b), d = mx - Math.min(r, g, b);
  if (!d) return 0;
  const h = mx === r ? (g - b) / d % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return (h * 60 + 360) % 360;
}
// no render yet: a colour from the name, the same every time
const named = (name) => `hsl(${[...name].reduce((n, ch) => (n * 31 + ch.charCodeAt(0)) % 360, 7)} 80% 55%)`;
// A character's picture in one of its drawn poses ("stand", and one for each menu card); for a character
// built before the app drew them, its plain render from the front.
const pic = (p, pose = "stand") => { const drawn = p?.portraits?.[pose]; return drawn ? art(drawn.url, drawn) : art(p?.portrait); };
const toneOf = async (p) => (await pic(p))?.colour || (p ? named(p.name) : NEW);
// The three menu cards' pictures: a pose drawn for each (render_portraits.py). Before those are drawn: the plain renders there are, never a T-pose (the rig's rest, not a
// picture of anyone), else the one pose from the front, the side and the back.
function poses(p) {
  if (p?.portraits?.setup) return ["setup", "build", "check"].map((pose) => pic(p, pose));
  const stills = p?.stills || {}, main = (p?.portrait || "").split("/").pop().split("?")[0].replace(/_front\.png$/, "");
  const pick = (names, used) => names.map((n) => stills[n]).find((u) => u && !used.includes(u)) || p?.portrait;
  const setup = pick(["rest_front", `${main}_front`], []);
  const build = pick(["raised_front", "forward_front", `${main}_right`, `${main}_left`], [setup]);
  return [setup, build, pick(["hang_front", "rest_front", `${main}_back`, `${main}_left`, `${main}_right`], [setup, build])].map((u) => art(u));
}

// ---------- the stage and the sparks; filled in by stage()
let burst = () => {}, tint = () => {};

function start() {
  document.body.insertAdjacentHTML("afterbegin", `<canvas id="a-bg"></canvas><div class="ghost" aria-hidden="true"><div><span id="a-ghost-a"></span></div><div><span id="a-ghost-b"></span></div></div>
    <canvas id="a-fx"></canvas><div class="wipe" id="a-wipe"></div>`);
  document.body.insertAdjacentHTML("beforeend", `<footer id="a-hints"></footer>`);
  $("steps").insertAdjacentHTML("beforebegin", `<div id="a-crumbs"></div>`);
  document.querySelector("nav .foot").insertAdjacentHTML("beforebegin", `<div class="tbs"><button id="a-bug" title="something went wrong or looks off: tell the developers (nothing is sent until you press Send)"><i aria-hidden="true">⚑</i>Report a bug</button><button id="a-music" title="the music, on or off; its loudness is on Setup"></button><button id="a-sfx" title="the menus' and buttons' sounds, on or off; their loudness is on Setup"></button><button data-to="help">Stuck?</button><button data-to="setup">Setup</button></div>`);
  $("home").insertAdjacentHTML("afterbegin", `<div class="hero"><h2 class="atitle mid">Character select<span id="a-count"></span></h2>
    <div class="band" id="a-band"><div class="who"><span class="tag" id="a-tag"></span><h3 id="a-name"></h3><p id="a-note"></p><div class="stats" id="a-stats"></div><button class="act small" id="a-draw" data-job hidden>Draw its poses</button></div></div>
    <div class="pop" id="a-pop"><img id="a-art" alt="" hidden><div class="mystery" id="a-mystery" hidden>?</div></div><div class="roster" id="a-roster"></div>
    <button class="more left" id="a-drawall" data-job hidden></button><button class="more" id="a-more">disk space · rebuild all ▼</button></div>`);
  $("model").insertAdjacentHTML("beforebegin", `<section class="page" id="menu"><h2 class="atitle mid">Main menu<span>pick what to do</span></h2><div class="mcards" id="a-cards"></div></section>
    <section class="page" id="pages"><h2 class="atitle"><span id="a-group"></span></h2><div class="alist" id="a-list"></div>
      <div class="preview"><div class="frame"><img id="a-pv" alt="" hidden></div><h4 id="a-pv-title"></h4><p id="a-pv-text"></p><p class="why" id="a-pv-why"></p></div></section>`);
  // the title's own text is the gradient; its small note is a child, so the title is a text node first
  A("group").parentElement.prepend(document.createTextNode(""));
  // Build: the band and the steps as tiles; the tables, logs and history fold away under them
  const head = $("build").querySelector(".page-head"), stack = $("build").querySelector(".stack"), tables = stack.querySelector(".split");
  head.insertAdjacentHTML("afterend", `<div class="bband" id="a-bband"><div class="band"><div class="fill" id="a-fill"></div><img id="a-build-art" alt="" hidden>
    <div class="who"><span class="tag" id="a-s-tag"></span><h3 id="a-s-name"></h3></div></div><div class="left" id="a-left"></div><div class="slam">Avatar ready</div></div>`);
  stack.insertAdjacentHTML("afterbegin", `<div class="rounds" id="a-rounds"></div>`);
  stack.insertBefore($("run-all").closest(".row"), $("todo").closest(".card"));
  tables.insertAdjacentHTML("beforebegin", `<details class="tech" id="a-tech"><summary>Details <span>each step's log and notes · history · what is in the file · the rig check</span></summary></details>`);
  A("tech").append(tables);

  stage(); arm();
  app.shown = shown; app.navDrawn = navDrawn; app.homeDrawn = hero;
  document.querySelector(".tbs").onclick = (e) => { const b = e.target.closest("button[data-to]"); if (b) { play("select"); app.go(b.dataset.to); } };
  A("bug").onclick = () => { play("select"); app.report(); };
  A("music").onclick = () => { switchMusic(); $("look-music").value = musicVolume(); tuneButton(); play("select"); };
  A("sfx").onclick = () => { switchSound(); $("look-sound").value = volume(); tuneButton(); play("select"); };
  tuneButton();
  // "Avatar ready" stays up until it is clicked, the page is left or another build starts
  A("bband").querySelector(".slam").onclick = () => A("bband").classList.remove("built");
  A("crumbs").onclick = (e) => { const b = e.target.closest("button"); if (b) { play("back"); app.go(b.dataset.to); } };
  // the step bar: back, the steps, next
  A("hints").onclick = (e) => { const b = e.target.closest("button[data-to]"); if (b && !b.disabled) { play(b.classList.contains("back") ? "back" : "select"); app.go(b.dataset.to); } };
  A("more").onclick = () => $("main").scrollTo({ top: document.querySelector(".hero").offsetHeight, behavior: still() ? "auto" : "smooth" });
  A("draw").onclick = () => draw([S.projects[sel]]);
  A("drawall").onclick = () => draw(undrawn());
  A("roster").onpointerover = (e) => { const s = e.target.closest(".slot"); if (s && +s.dataset.i !== sel) show(+s.dataset.i); };
  A("roster").onclick = (e) => { const s = e.target.closest(".slot"); if (s) choose(+s.dataset.i); };
  A("cards").onpointerover = (e) => { const m = e.target.closest(".mcard"); if (m) light(+m.dataset.i); };
  A("cards").onclick = (e) => { const m = e.target.closest(".mcard"); if (m) enter(+m.dataset.i); };
  A("list").onpointerover = (e) => { const b = e.target.closest("button"); if (b && +b.dataset.i !== item) { item = +b.dataset.i; play("move"); list(); } };
  A("list").onclick = (e) => { const b = e.target.closest("button"); if (b) { play("select"); app.go(GROUPS[group][+b.dataset.i]); } };
  // a tile opens its step's row under Details: the log, the notes, running from it
  A("rounds").onclick = (e) => {
    const r = e.target.closest("[data-step]"), row = r && document.querySelector(`#step-rows tr[data-step="${r.dataset.step}"]`);
    if (!row) return;
    A("tech").open = true;
    if (!row.classList.contains("sel")) row.click();
    $("step-rows").closest(".card").scrollIntoView({ behavior: still() ? "auto" : "smooth", block: "start" });
  };
  addEventListener("keydown", keys);
  addEventListener("resize", bars); bars();
  setInterval(watch, 300);
}

// the two bars' height, for the screens that fill what is left of the window
function bars() { root.style.setProperty("--bars", `${document.querySelector("nav").offsetHeight + A("hints").offsetHeight}px`); }

// ---------- a page shows: the slash, the trail, the hints, the rolling name
function trail(name) {
  if (name === "home") return ["home"];
  if (!S.manifest && !opening) return ["home", name];
  if (name === "menu") return ["home", "menu"];
  if (name === "pages" || GROUPS.setup.includes(name) || GROUPS.check.includes(name)) return ["home", "menu", "pages", ...(name === "pages" ? [] : [name])];
  if (name === "build") return ["home", "menu", "build"];
  return ["home", name];
}
const label = (name) => name === "menu" ? open()?.name || "Main menu" : name === "pages" ? CARDS.find((c) => c[1] === group)[0]
  : name === "model" && !S.manifest ? "New character" : TITLES[name] || name;

function shown(name) {
  const fresh = name !== page;
  if (fresh) { kick(A("wipe"), "sw"); play("whoosh"); }
  page = name;
  mood(["home", "menu", "pages"].includes(name) ? "menu" : "work");
  for (const g of ["setup", "check"]) if (GROUPS[g].includes(name)) { group = g; item = GROUPS[g].indexOf(name); }
  // what drops or slides in does so when its screen comes up, not each time it is redrawn
  if (fresh && name !== "build") A("bband").classList.remove("built");
  if (fresh && $(name)) { $(name).classList.add("fresh"); setTimeout(() => $(name).classList.remove("fresh"), 900); }
  chrome();
  if (name === "home") hero(true);
  if (name === "menu") menu();
  if (name === "pages") list();
  if (name === "build") band();
}
function chrome() {
  html(A("crumbs"), trail(page).map((t) => `<button data-to="${t}" class="${t === page ? "on" : ""}">${esc(label(t))}</button>`).join(""));
  document.querySelectorAll(".tbs button").forEach((b) => {
    b.classList.toggle("on", b.dataset.to === page);
    if (b.dataset.to === "setup") b.classList.toggle("bad", !!S.setup && !S.setup.ready);
  });
  const k = (a, b) => `<span><kbd>${a}</kbd>${b}</span>`, menus = ["home", "menu", "pages"].includes(page), step = $(page)?.dataset.step, path = open()?.manifest;
  const where = `<span class="path">${path && page !== "home" ? esc(path.replaceAll("/", "\\").split("\\").slice(-3).join("\\")) + " · saved as you go" : ""}</span>`;
  html(A("hints"), S.manifest && (ORDER.includes(page) || page === "paint") ? stepBar(where)
    : (menus ? k(page === "pages" ? "↑ ↓" : "← →", "choose") + k("Enter", "select") + (page === "home" ? "" : k("Esc", "back"))
      : `<span>${step ? `step ${step} of 8 · ` : ""}${esc(label(page))}</span>`) + `<span class="sp"></span>${where}`);
  put(A("ghost-a"), `${label(page)} · `.repeat(8));
  put(A("ghost-b"), `${page === "home" || !open() ? "OC2VRM" : open().name} · `.repeat(12));
  bars();
}

// The step bar along the bottom of a step page: the way back, every step (done ones ticked, the one shown lit, a
// locked one saying why), and the way on, there without scrolling to a page's end. Next glows once this page is done.
// A nudge (app.nudge): something finished and says where to go on; the Next button takes it, its label pointing
// there, bouncing, with the reason in a bubble over it, instead of a message to read and a button elsewhere.
let nudge = null;
app.nudge = (to, text) => { nudge = { from: page, to, text }; play("select"); chrome(); };
function stepBar(where) {
  const nav = (n) => document.querySelector(`#steps button[data-go="${n}"]`);
  const done = (n) => !!nav(n)?.querySelector(".n.done");
  if (nudge && nudge.from !== page) nudge = null;
  const prev = before(page), next = nudge ? nudge.to : after(page);
  const steps = ORDER.filter((n) => !skipped(n)).map((n, i) => {
    const why = nav(n)?.classList.contains("locked") ? nav(n).title : "";
    return `<li><button data-to="${n}" class="${n === page || (page === "paint" && n === "face") ? "on" : ""} ${done(n) ? "done" : ""}" title="${esc(why ? `${TITLES[n]}: ${why}` : `step ${i + 1}: ${TITLES[n]}`)}"${why ? " disabled" : ""}>`
      + `<i>${done(n) ? "✓" : i + 1}</i><span>${esc(TITLES[n])}</span></button></li>`;
  }).join("");
  return `<button class="back" data-to="${prev || "menu"}">← ${esc(prev ? TITLES[prev] : "Menu")}</button><ol class="steps">${steps}</ol>`
    + `<span class="sp"></span>${where}`
    + `<span class="go">${nudge ? `<span class="bubble">${esc(nudge.text)}</span>` : ""}`
    + `<button class="next ${done(page) ? "ready" : ""} ${nudge ? "nudge" : ""}" data-to="${next || "menu"}">${next ? `Next: ${esc(TITLES[next])}` : "Main menu"} →</button></span>`;
}

// the pages redrew their badges: the open character, its colour, and whichever menu is up
async function navDrawn() {
  chrome();
  if (page === "menu") menu();
  if (page === "pages") list();
  band();
  const p = open(), tone = p ? await toneOf(p) : "#2a5cff";
  if (p?.manifest !== open()?.manifest) return;
  colour = tone;
  root.style.setProperty("--c", colour);
  if (page !== "home") tint(colour);
}

// ---------- character select
// The roster is built when the list of characters changes, and the band animates when the cursor
// moves or the page comes up: the list is read again after every run and every character opened, and
// drawing it all again each time replayed every animation, twice a click.
function hero(force) {
  if (opening) return;
  const ps = S.projects, now = JSON.stringify(ps.map((p) => [p.manifest, p.portrait, p.portraits?.stand?.url]));
  const first = now !== roster;
  if (first) {
    roster = now;
    A("count").textContent = `${ps.length} character${ps.length === 1 ? "" : "s"}`;
    A("roster").style.setProperty("--cols", Math.min(ps.length + 1, 10));
    A("roster").innerHTML = ps.map((p, i) => `<button class="slot" data-i="${i}" style="--n:${Math.min(i, 12)}" aria-label="${esc(p.name)}"><b>${esc(p.name[0] || "?")}</b><small>${esc(p.name)}</small></button>`).join("")
      + `<button class="slot new" data-i="${ps.length}" style="--n:${Math.min(ps.length, 12)}" aria-label="new character">?<small>new</small></button>`;
    ps.forEach(async (p, i) => {
      const a = await pic(p), tone = await toneOf(p), s = A("roster").children[i];
      if (S.projects[i]?.manifest !== p.manifest || !s) return;
      s.style.setProperty("--c", tone);
      if (a) { s.style.setProperty("--f", `url(${a.face || a.src})`); s.classList.add("pic"); s.classList.toggle("cut", !!a.face); }
    });
  }
  const todo = undrawn().length;
  A("drawall").hidden = !todo;
  put(A("drawall"), `draw poses for ${todo} character${todo === 1 ? "" : "s"}`);
  // coming to the page, or another character opened: the cursor starts on the open one
  const i = ps.findIndex((p) => p.manifest === S.manifestPath);
  if (i >= 0 && (force === true || S.manifestPath !== opened)) sel = i;
  opened = S.manifestPath;
  sel = Math.max(0, Math.min(sel, ps.length));
  fill(sel, first || force === true);
}
// the band for whoever the cursor is on; `lively` plays its entrance
async function fill(i, lively) {
  sel = i;
  const p = S.projects[i], b = A("band");
  [...A("roster").children].forEach((s, n) => s.classList.toggle("on", n === i));
  if (p) {
    const made = [...$("cards").querySelectorAll("[data-manifest]")].find((c) => c.dataset.manifest === p.manifest);
    const notes = [...(made?.querySelectorAll(".chip") || [])].map((c) => c.textContent);
    put(A("tag"), p.manifest.replaceAll("\\", "/").split("/").slice(-3, -1).join(" / "));
    put(A("name"), p.name); put(A("note"), notes.join(" · "));
    const when = p.finished ? new Date(p.finished).toLocaleDateString([], { day: "numeric", month: "short" }) : "never";
    html(A("stats"), `<div><b>${p.approved ? "good" : p.built ? "built" : p.surveyed ? "started" : "new"}</b><span>so far</span></div>
      <div><b>${esc(when)}</b><span>last run</span></div><div><b>${p.stale?.length || 0}</b><span>steps to rebuild</span></div>`);
  } else {
    put(A("tag"), "from a Tripo model"); put(A("name"), "New character"); put(A("note"), "Who's next? A rigged .fbx from Tripo works best."); html(A("stats"), "");
  }
  // built before the app drew a character's poses: one click draws them (one Blender run, under a minute)
  A("draw").hidden = !(p && p.built && !p.portraits?.stand);
  // long names shrink to stay on the band
  A("name").style.fontSize = `min(15vh, 150px, ${(104 / Math.max(3, A("name").textContent.length)).toFixed(1)}vw)`;
  const a = await pic(p), tone = await toneOf(p);
  if (sel !== i) return;
  b.style.setProperty("--c", tone);
  A("art").hidden = !a; if (a) A("art").src = a.src;
  A("mystery").hidden = !!p;
  if (lively) for (const el of [b, A("pop")]) kick(el, "in");
  if (page === "home") tint(tone);
}
// Characters built before the app drew their poses, and drawing them: one Blender run each, one after another.
const undrawn = () => S.projects.filter((p) => p.built && !p.portraits?.stand);
async function draw(list) {
  for (const p of list) {
    const ended = p && await app.job("portraits", { manifest: p.manifest });
    if (!ended || ended.error || ended.cancelled) break;
  }
  // (with a character open, the run's end has read the list again already)
  if (!S.manifestPath) await app.loadProjects(null);
}
// the cursor moved: sparks from the slot it landed on, so they travel with it
function show(i) {
  play("move"); fill(i, true);
  const slot = A("roster").children[i];
  if (slot) burst(...at(slot, .5, .1), "#3df2d8", 8, .6, -Math.PI / 2, 1.6);
}
// The menu comes up at once under the picked character; its pages' states follow when it has
// loaded (a second or so: the model's pieces are read).
async function choose(i) {
  const p = S.projects[i];
  play("select");
  if (!p) { await app.openProject(null); return app.go("model"); }
  if (p.manifest === S.manifestPath && S.manifest) { card = upNext(); return app.go(app.landing?.() || "menu"); }
  opening = p; card = 0;
  let moved = false;
  const watchCard = () => { moved = true; };
  A("cards").addEventListener("pointerover", watchCard); addEventListener("keydown", watchCard);
  // (automatic mode opens a character on its one screen, quick.js; the menu is a button away)
  app.go(app.landing?.() || "menu");
  try { await app.openProject(p.manifest); }
  finally { opening = null; A("cards").removeEventListener("pointerover", watchCard); removeEventListener("keydown", watchCard); }
  if (page === "menu") { if (!moved) card = upNext(); menu(); }
  if (page === "home") hero();
  chrome();
}

// ---------- main menu
// the card holding the first page that still has something to do
function upNext() {
  const first = ["model", "parts", "face", "build", "movement"].find((n) => !done(n));
  return !first ? 2 : first === "build" ? 1 : GROUPS.setup.includes(first) ? 0 : 2;
}
async function menu() {
  const p = open(), sure = known(), next = sure ? upNext() : -1;
  const tags = (names) => names.map((n) => `<em class="${sure && done(n) ? "done" : ""}">${TITLES[n]}</em>`).join("");
  const says = (i) => !sure ? "" : i === 1 ? badge("build") || "model to .vrm" : i === 2 ? badge("movement") : badge("parts");
  // The cards are made once and written into after: the menu is drawn again when the picked character
  // has loaded (its badges, "next up"), and cards made anew then dropped in a second time, pictures gone
  // for a moment: a glitch a second after every character picked.
  if (A("cards").children.length !== CARDS.length) A("cards").innerHTML = CARDS.map((c, i) => `<button class="mcard" data-i="${i}" style="--n:${i};--m:${c[2]}"><div class="un"><span class="no">0${i + 1}</span><span class="next" hidden>next up</span>
    <img alt="" hidden><div class="mystery" hidden>?</div><div class="cap"><b>${c[0]}</b><span class="pgs"></span><span class="says"></span></div></div></button>`).join("");
  [...A("cards").children].forEach((m, n) => {
    m.classList.toggle("on", n === card);
    m.querySelector(".next").hidden = n !== next;
    html(m.querySelector(".pgs"), tags(GROUPS[CARDS[n][1]] || ["build"]));
    put(m.querySelector(".says"), says(n));
  });
  const arts = await Promise.all(poses(p));
  if (p?.manifest !== open()?.manifest) return;
  A("cards").querySelectorAll("img").forEach((img, i) => {
    const a = arts[i];
    img.hidden = !a; img.nextElementSibling.hidden = !!a;
    if (a && img.src !== a.src) img.src = a.src;
  });
}
function light(i) {
  if (i === card) return;
  card = i; play("move");
  [...A("cards").children].forEach((m, n) => m.classList.toggle("on", n === i));
  burst(...at(A("cards").children[i], .5, .08), CARDS[i][2], 14, .9, -Math.PI / 2, 2);
}
function enter(i) {
  play("select");
  if (CARDS[i][1] === "build") return app.go("build");
  if (group !== CARDS[i][1]) { group = CARDS[i][1]; item = Math.max(0, GROUPS[group].findIndex((n) => !done(n) && !why(n))); }
  app.go("pages");
}

// ---------- a group's pages as a list, with what each is for beside it
async function list() {
  const names = GROUPS[group], sure = known();
  item = Math.max(0, Math.min(item, names.length - 1));
  put(A("group").parentElement.firstChild, label("pages"));
  put(A("group"), `${names.length} pages`);
  const state = (n) => !sure ? "" : why(n) ? "locked" : done(n) ? "done" : badge(n);
  html(A("list"), names.map((n, i) => `<button class="${i === item ? "on" : ""} ${sure && done(n) ? "done" : sure && asks(n) ? "todo" : ""}" data-i="${i}">${TITLES[n]}<small>${esc(state(n))}</small></button>`).join(""));
  // the picture's frame is art()'s: the head starts 21% down the square, the feet stand at 98%
  const name = names[item], zoom = { face: ["-101%", "560%"], parts: ["-27%", "128%"], joints: ["-27%", "128%"], movement: ["-27%", "128%"] }[name] || ["-47%", "250%"];
  put(A("pv-title"), TITLES[name]);
  put(A("pv-text"), SAYS[name] || document.querySelector(`#${name} .lead`)?.textContent.replace(/\s+/g, " ").trim() || "");
  put(A("pv-why"), sure && why(name) ? `Not yet: ${why(name)}.` : "");
  const a = await pic(open()), img = A("pv");
  img.hidden = !a; if (a) { if (img.src !== a.src) img.src = a.src; img.style.top = zoom[0]; img.style.height = zoom[1]; }
}

// ---------- build: the steps as tiles and the band filling as they finish
// A step that has nothing more to give: done, built without (failed), nothing to do on this model, skipped.
const settled = (state) => ["done", "failed", "skip"].includes(state);
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
// The steps that run on this character, each "done", "failed", "skip", "run" or "wait". While a build
// runs the run's own reports count (the Build page's Progress lines: "hair done (6 s)"), and every step
// after the last one reported waits, whatever it was before.
function steps() {
  const list = (S.status?.steps || []).filter((s) => s.runs).map((s) => ({ step: s.step, seconds: s.seconds,
    state: s.status === "done" ? "done" : s.status === "failed" ? "failed" : /^(nothing|skipped)/.test(s.status) ? "skip" : "wait" }));
  if ($("runbar").hidden) return list;
  const said = {};
  for (const line of $("events").textContent.split("\n")) {
    const [step, word] = line.split(" "), state = (word || "").replace(":", "");
    if (["running", "done", "failed", "cancelled", "skipped", "nothing"].includes(state)) said[step] = state;
  }
  let reached = -1;
  list.forEach((s, i) => { if (said[s.step]) reached = i; });
  list.forEach((s, i) => {
    const now = said[s.step];
    if (now) s.state = now === "running" ? "run" : now === "done" ? "done" : now === "failed" ? "failed" : now === "cancelled" ? "wait" : "skip";
    else if (i > reached && reached >= 0) s.state = "wait";
  });
  return list;
}
async function band() {
  const st = S.status, running = !$("runbar").hidden, list = steps();
  const over = list.filter((s) => settled(s.state)).length, left = list.length - over, failed = list.filter((s) => s.state === "failed");
  share = list.length ? Math.min(100, 100 * over / list.length) : 0;
  const words = { done: "ok", failed: "left out", skip: "nothing to do", run: "now", wait: "" };
  html(A("rounds"), list.map((s, i) => `<button class="round ${s.state}" data-step="${s.step}" style="--n:${i}" title="open this step under Details"><i>${String(i + 1).padStart(2, "0")}</i>${esc(stepName(s.step))}
    <time>${words[s.state]}${s.seconds && s.state === "done" ? ` ${clock(s.seconds)}` : ""}</time></button>`).join(""));
  let name, tag, time = "";
  if (running) {
    const now = list.findIndex((s) => s.state === "run");
    name = now >= 0 ? stepName(list[now].step) : $("run-step").textContent.split(" · ")[0] || $("run-label").textContent || "Working";
    tag = now >= 0 ? `step ${now + 1} of ${list.length}` : $("run-label").textContent;
    time = $("run-time").textContent.split(" · ")[0];
    // each finished step is heard, higher as the build goes
    if (over > settledWas) play("step", share);
  } else {
    name = !S.manifest ? "No character" : left ? (over > 1 ? `${left} to build` : "Ready?") : st?.verdict?.ready ? "Ready to use" : "Built";
    tag = !S.manifest ? "open or create a character first" : failed.length ? `built without ${failed.map((s) => stepName(s.step)).join(", ")}` : `${list.length} steps · ${tookLast(st) ? `about ${tookLast(st)} the last time` : "a few minutes"}`;
  }
  settledWas = over;
  put(A("s-name"), name); put(A("s-tag"), tag);
  A("fill").style.setProperty("--p", share);
  if (put(A("left"), time) && time) kick(A("left"), "tick");
  const a = await pic(open()), img = A("build-art");
  img.hidden = !a; if (a && img.src !== a.src) img.src = a.src;
  // A build ended and left an avatar file with nothing still to build: say so, loudly, once, when the Build
  // page is next looked at. Not before the character has been read again (`cheer.was` is the status the
  // run began on): until then the steps are the last build's, and a stopped run would be cheered for them.
  if (!running && cheer && st && st !== cheer.was && page === "build") {
    cheer = null;
    if (left || !lastVrm()) return;
    put(A("bband").querySelector(".slam"), st.verdict?.ready ? "Avatar ready" : "Avatar built");
    kick(A("bband"), "built"); flash = 1; play("ready");
    burst(...at(A("bband")), colour, 90, 2); burst(...at(A("bband")), "#3df2d8", 40, 1.5);
  }
}
// where the fill's leading edge is on the screen: the band is turned 2 degrees, so the edge's box is not its place
function edge() {
  const b = A("fill").parentElement, r = b.getBoundingClientRect(), w = b.offsetWidth, h = b.offsetHeight, turn = -2 * Math.PI / 180;
  const x = -w / 2 + w * share / 100, y = (Math.random() - .5) * h * .9;
  return [r.left + r.width / 2 + x * Math.cos(turn) - y * Math.sin(turn), r.top + r.height / 2 + x * Math.sin(turn) + y * Math.cos(turn)];
}
function watch() {
  const running = !$("runbar").hidden;
  root.classList.toggle("running", running);
  // The status is noted when the run starts, not when this notices it has ended: the character can be read
  // again within the same third of a second, and the cheer then waited for a change that had already come.
  // Only a build is cheered (the run bar says "Build", "Build from eyes"): a render or a sheet is a run too.
  if (wasRunning && !running) cheer = /^build/i.test($("run-label").textContent) ? ran : null;
  if (running && !wasRunning) { ran = { was: S.status }; cheer = null; A("bband").classList.remove("built"); play("start"); }
  wasRunning = running;
  if (page === "build" && (running || cheer)) band();
}

// ---------- keys, on the menu screens only: the work pages have their own
function keys(e) {
  if (!["home", "menu", "pages"].includes(page) || e.ctrlKey || e.altKey || e.metaKey) return;
  const focus = document.activeElement;
  if (document.querySelector("dialog[open]") || !$("lightbox").hidden || /^(INPUT|TEXTAREA|SELECT)$/.test(focus?.tagName)) return;
  const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
  if (!d && e.key !== "Enter" && e.key !== "Escape") return;
  // a focused button takes Enter itself
  if (e.key === "Enter" && focus?.tagName === "BUTTON") return;
  if (d) focus?.blur?.();
  if (e.key === "Escape") { const t = trail(page); if (t.length > 1) { play("back"); app.go(t[t.length - 2]); } }
  else if (page === "home") { const n = S.projects.length + 1; if (d) show((sel + d + n) % n); else choose(sel); }
  else if (page === "menu") { if (d) light((card + d + 3) % 3); else enter(card); }
  else { const n = GROUPS[group].length; if (d) { item = (item + d + n) % n; play("move"); list(); } else { play("select"); app.go(GROUPS[group][item]); } }
  e.preventDefault();
}

// ---------- the stage behind everything and the sparks in front. The field swells under the cursor;
// its far end takes the character's colour. It rests on Paint (the brush wants every frame) and in a hidden window.
function stage() {
  const bg = A("bg").getContext("2d"), fx = A("fx").getContext("2d"), bits = [];
  let W = 0, H = 0, mx = -9999, my = -9999, field = "#2a5cff", last = 0, clear = true;
  const streaks = Array.from({ length: 30 }, () => ({ x: Math.random(), y: Math.random(), l: 90 + Math.random() * 280, v: .15 + Math.random() * .85, w: 1 + Math.random() * 2.5 }));
  tint = (c) => { field = bg.createLinearGradient(0, H, W, 0); field.addColorStop(0, "#2a5cff"); field.addColorStop(.55, "#3df2d8"); field.addColorStop(1, c); if (still()) draw(0); };
  const fit = () => {
    const d = Math.min(devicePixelRatio || 1, 1.5); W = innerWidth; H = innerHeight;
    for (const c of [bg, fx]) { c.canvas.width = W * d; c.canvas.height = H * d; c.setTransform(d, 0, 0, d, 0, 0); }
    tint(colour); draw(last);
  };
  function draw(t) {
    bg.clearRect(0, 0, W, H);
    const cx = W / 2, cy = H / 2, step = 30, work = !["home", "menu", "pages"].includes(page);
    const gx = W * (.5 + .24 * Math.sin(t * .23)), gy = H * (.6 + .14 * Math.cos(t * .31)), glow = bg.createRadialGradient(gx, gy, 0, gx, gy, Math.max(W, H) * .6);
    glow.addColorStop(0, `rgba(42,92,255,${work ? .14 : .22})`); glow.addColorStop(1, "rgba(42,92,255,0)"); bg.fillStyle = glow; bg.fillRect(0, 0, W, H);
    // halftone: diamonds on a lattice leaning like the look's slanted edges, sized by two crossing waves; smaller
    // toward the middle, where the text is, and fainter behind a work page
    bg.fillStyle = field; bg.globalAlpha = work ? .16 : .3; bg.beginPath();
    for (let y = 0; y < H + step; y += step) for (let x = -(y * .25) % step; x < W + step; x += step) {
      const v = .5 * Math.sin(x * .0042 + y * .0075 - t * .7) + .5 * Math.sin(x * .0105 - y * .0052 + t * .43) + 1.2 * Math.exp(-((x - mx) ** 2 + (y - my) ** 2) / 24000);
      const r = Math.min(1.25, v) * step * .4 * (.25 + .75 * Math.min(1, Math.hypot((x - cx) / cx, (y - cy) / cy)));
      if (r > .8) { bg.moveTo(x, y - r); bg.lineTo(x + r, y); bg.lineTo(x, y + r); bg.lineTo(x - r, y); }
    }
    bg.fill(); bg.globalAlpha = 1;
    bg.save(); bg.translate(cx, cy); bg.rotate(-3 * Math.PI / 180); bg.translate(-cx, -cy);
    for (const s of streaks) {
      const x = (s.x + t * s.v * .2) % 1 * (W + 800) - 400, y = s.y * H * 1.2 - H * .1, line = bg.createLinearGradient(x - s.l, 0, x, 0);
      line.addColorStop(0, "rgba(61,242,216,0)"); line.addColorStop(1, `rgba(200,255,246,${(.08 + s.v * .34) * (work ? .5 : 1)})`); bg.fillStyle = line; bg.fillRect(x - s.l, y, s.l, s.w);
    }
    bg.restore();
  }
  // sparks: slanted shards and short streaks thrown from (x, y), toward `ang` within `spread`
  burst = (x, y, col, n = 12, pow = 1, ang = 0, spread = 6.283) => {
    if (still() || !last || document.hidden) return;
    for (let i = 0; i < n; i++) {
      const a = ang + (Math.random() - .5) * spread, v = (120 + Math.random() * 420) * pow;
      bits.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 0, max: .3 + Math.random() * .5, s: 2 + Math.random() * 7, col: Math.random() < .3 ? "#fff" : col, line: Math.random() < .45 });
    }
  };
  function sparks(dt) {
    if (!bits.length && flash <= 0) { if (!clear) { fx.clearRect(0, 0, W, H); clear = true; } return; }
    clear = false;
    fx.clearRect(0, 0, W, H); fx.globalCompositeOperation = "lighter"; fx.lineWidth = 2;
    for (let i = bits.length; i--;) {
      const b = bits[i]; b.life += dt;
      if (b.life > b.max) { bits.splice(i, 1); continue; }
      b.vx *= 1 - 2.2 * dt; b.vy = b.vy * (1 - 2.2 * dt) + 420 * dt; b.x += b.vx * dt; b.y += b.vy * dt;
      const k = 1 - b.life / b.max, s = b.s * k; fx.globalAlpha = k; fx.fillStyle = fx.strokeStyle = b.col; fx.beginPath();
      if (b.line) { fx.moveTo(b.x, b.y); fx.lineTo(b.x - b.vx * .06, b.y - b.vy * .06); fx.stroke(); }
      else { fx.moveTo(b.x - s * .75, b.y - s * .5); fx.lineTo(b.x + s * 1.25, b.y - s * .5); fx.lineTo(b.x + s * .75, b.y + s * .5); fx.lineTo(b.x - s * 1.25, b.y + s * .5); fx.fill(); }
    }
    fx.globalCompositeOperation = "source-over";
    if (flash > 0) { fx.globalAlpha = flash * .6; fx.fillStyle = "#fff"; fx.fillRect(0, 0, W, H); flash -= dt * 3.5; }
    fx.globalAlpha = 1;
  }
  function frame(ms) {
    requestAnimationFrame(frame);
    // (two frames at one instant move nothing: not a sixtieth of a second each)
    const t = ms / 1000, dt = last ? Math.min(.05, Math.max(0, t - last)) : .016;
    last = t;
    if (still() || document.hidden) return;
    if (page !== "paint") draw(t);
    // the fill's leading edge burns while a build runs
    if (page === "build" && root.classList.contains("running") && share > 0 && share < 100 && Math.random() < .7) {
      const [x, y] = edge();
      if (x > 0 && x < W && y > 0 && y < H) burst(x, y, Math.random() < .5 ? "#fff" : colour, 1, .7, 0, 1.6);
    }
    sparks(dt);
  }
  addEventListener("resize", fit); fit();
  addEventListener("pointermove", (e) => {
    mx = e.clientX; my = e.clientY;
    // the character leans against the cursor on character select; a work page is left alone (every move would restyle it)
    if (page === "home") { root.style.setProperty("--mx", (mx / W * 2 - 1).toFixed(2)); root.style.setProperty("--my", (my / H * 2 - 1).toFixed(2)); }
  });
  addEventListener("pointerdown", (e) => {
    if (!e.target.closest?.("button")) return;
    burst(e.clientX, e.clientY, "#3df2d8", 14, .8);
    // the menus' own buttons have their own sounds
    if (!e.target.closest("#a-roster, #a-cards, #a-list, #a-crumbs, .tbs")) play("click");
  });
  requestAnimationFrame(frame);
}

start();
