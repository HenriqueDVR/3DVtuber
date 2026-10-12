// Parts: the model's pieces in 3D and in a list, each given the role that says how it moves.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { $, S, app, call, esc, hex, say, refreshManifest, ROLES, ROLE_INFO, ROLE_COLOUR, ROLE_STAGE, CORE, BONES, roleName, namedFor } from "./core.js";
import { makeView } from "./views.js";
import { ask, keyable, pushUndo, saved, toast } from "./ui.js";
import { job } from "./jobs.js";

export const partsView = makeView($("parts-view"));
let partMeshes = [], selected = null, hovered = null, region = new Set(), topo = null;
let filter = "all", sort = { key: "count", dir: "desc" }, isolate = false, xray = false;
const SPECK = 50;

// The listed model, loaded once per listing; the Joints and Movement views load the same file.
export async function loadParts(url) {
  partsView.clear(); partMeshes = [];
  if (!url) return;
  const gltf = await new GLTFLoader().loadAsync(url + `?t=${Date.now()}`);
  partsView.model = gltf.scene; partsView.scene.add(gltf.scene);
  gltf.scene.traverse((o) => { if (o.isMesh) { partMeshes.push(o); o.material = o.material.clone(); o.material.vertexColors = true; } });
  topo = null; region = new Set();
  colourPieces();
  partsView.frame();
  return gltf;
}

function suggestionOf(label) {
  const row = S.listing?.pieces.find((p) => p.label === label);
  return row && row.suggested !== "body" ? row.suggested : "";
}
function roleOf(label) {
  const row = S.listing?.pieces.find((p) => p.label === label);
  return row ? namedFor(row)?.role || "" : "";
}

// Colours by role (faded where only suggested), the picked piece lit, the painted part yellow
// (it was white, on a picked piece lit nearly white: the paint was there and couldn't be seen).
const PAINTED = [1, 0.82, 0.1, 1];
// Isolate fades every other piece nearly out; X-ray makes the whole model see-through.
export function colourPieces() {
  const focus = hovered ?? selected;
  for (const mesh of partMeshes) {
    const attr = mesh.geometry.attributes._piece;
    if (!attr) continue;
    const numbers = mesh.geometry.attributes._vertex;
    const colours = new Float32Array(attr.count * 4), c = new THREE.Color();
    const cache = new Map();
    for (let i = 0; i < attr.count; i++) {
      const label = Math.round(attr.getX(i));
      if (!cache.has(label)) {
        const own = roleOf(label);
        c.setHex(ROLE_COLOUR[own || suggestionOf(label)] ?? 0x9a9a9a);
        if (!own) c.lerp(new THREE.Color(0x9a9a9a), 0.6);
        if (label === focus) c.lerp(new THREE.Color(0xffffff), 0.55);
        const alpha = isolate && selected != null && label !== selected && label !== hovered ? 0.05 : xray ? (label === focus ? 0.9 : 0.3) : 1;
        cache.set(label, [...c.toArray(), alpha]);
      }
      const painted = region.size && numbers && topo && region.has(topo.rep.get(Math.round(numbers.getX(i))));
      colours.set(painted ? PAINTED : cache.get(label), i * 4);
    }
    mesh.geometry.setAttribute("color", new THREE.BufferAttribute(colours, 4));
    const see = xray || (isolate && selected != null);
    Object.assign(mesh.material, { map: null, transparent: see, depthWrite: !see, needsUpdate: true });
  }
}

partsView.host.querySelectorAll("[data-tool]").forEach((b) => b.onclick = () => {
  if (b.dataset.tool === "isolate") isolate = !isolate; else xray = !xray;
  b.classList.toggle("on", b.dataset.tool === "isolate" ? isolate : xray);
  if (isolate && selected == null) toast("Pick a piece to isolate it", "info");
  colourPieces();
});

// ---------- part of a piece
// The surface of the listed model: vertices sharing a position in one piece are one (UV seams
// split them), `region` holds those, and the source vertex numbers are what the manifest gets.
function buildTopo() {
  const rep = new Map(), members = new Map(), nbrs = new Map(), piece = new Map(), pos = new Map(), byKey = new Map();
  const v = new THREE.Vector3();
  for (const mesh of partMeshes) {
    const g = mesh.geometry, nums = g.attributes._vertex, pa = g.attributes._piece, p = g.attributes.position;
    if (!nums || !pa) continue;
    const local = new Int32Array(p.count);
    for (let i = 0; i < p.count; i++) {
      v.fromBufferAttribute(p, i).applyMatrix4(mesh.matrixWorld);
      const label = Math.round(pa.getX(i)), n = Math.round(nums.getX(i));
      const key = `${label}:${Math.round(v.x * 1e5)},${Math.round(v.y * 1e5)},${Math.round(v.z * 1e5)}`;
      let r = byKey.get(key);
      if (r === undefined) { r = n; byKey.set(key, r); members.set(r, new Set()); nbrs.set(r, new Set()); piece.set(r, label); pos.set(r, v.clone()); }
      if (!rep.has(n)) rep.set(n, r);
      members.get(r).add(n); local[i] = r;
    }
    const tri = g.index ? g.index.array : Array.from({ length: p.count }, (_, i) => i);
    for (let t = 0; t + 2 < tri.length; t += 3) {
      const a = local[tri[t]], b = local[tri[t + 1]], c = local[tri[t + 2]];
      for (const [x, y] of [[a, b], [b, c], [c, a]]) if (x !== y) { nbrs.get(x).add(y); nbrs.get(y).add(x); }
    }
  }
  topo = { rep, members, nbrs, piece, pos };
}
const inPiece = (r) => topo.piece.get(r) === selected;
function regionNumbers() { const out = []; for (const r of region) out.push(...topo.members.get(r)); return out.sort((a, b) => a - b); }

// what the pick looks like as a part: one connected bit, no gaps, a clean edge
function regionReport() {
  const seen = new Set(); let bits = 0;
  for (const r of region) {
    if (seen.has(r)) continue;
    bits++; const stack = [r]; seen.add(r);
    while (stack.length) for (const o of topo.nbrs.get(stack.pop())) if (region.has(o) && !seen.has(o)) { seen.add(o); stack.push(o); }
  }
  let border = 0; const edge = new Set();
  for (const r of region) for (const o of topo.nbrs.get(r)) if (!region.has(o) && inPiece(o)) { border++; edge.add(o); }
  const gaps = outsideParts().filter((c) => c.length < GAP_SHARE * region.size).reduce((n, c) => n + c.length, 0);
  return { bits, border, gaps, rim: edge.size };
}
// the piece's vertices outside the part, as connected groups
function outsideParts() {
  const seen = new Set(), parts = [];
  for (const [r, label] of topo.piece) {
    if (label !== selected || region.has(r) || seen.has(r)) continue;
    const part = [r]; seen.add(r);
    for (let k = 0; k < part.length; k++) for (const o of topo.nbrs.get(part[k])) if (inPiece(o) && !region.has(o) && !seen.has(o)) { seen.add(o); part.push(o); }
    parts.push(part);
  }
  return parts;
}
// the same rule the build applies (roles.close_region): gaps under a quarter of the part are its own
const GAP_SHARE = 0.25;
function fillGaps() {
  for (const part of outsideParts()) if (part.length < GAP_SHARE * region.size) part.forEach((r) => region.add(r));
  for (let pass = 0; pass < 3; pass++) {
    const add = [];
    for (const [r, label] of topo.piece) {
      if (label !== selected || region.has(r)) continue;
      const around = [...topo.nbrs.get(r)].filter(inPiece);
      if (around.length && 2 * around.filter((o) => region.has(o)).length >= around.length) add.push(r);
    }
    if (!add.length) break;
    add.forEach((r) => region.add(r));
  }
}
function grow() { const add = []; for (const r of region) for (const o of topo.nbrs.get(r)) if (inPiece(o)) add.push(o); add.forEach((r) => region.add(r)); }
function shrink() { const drop = [...region].filter((r) => [...topo.nbrs.get(r)].some((o) => !region.has(o))); drop.forEach((r) => region.delete(r)); }

const box = document.createElement("div");
box.className = "box";
$("parts-view").appendChild(box);
let dragFrom = null, painting = null, tool = null;
// The tool chosen on the card: with one on, a plain drag paints and the right button turns the model.
// It was Alt+drag and Shift+drag alone, written in a closed card: nobody found it, Alt opens the
// window's menu in the desktop app, and with no piece picked first a drag did nothing at all.
const TOOL_SAYS = { brush: "Brush: drag over the part", erase: "Eraser: drag over what to take out", box: "Box: drag a box round the part" };
function setTool(name) {
  tool = tool === name ? null : name;
  $("region-tools").querySelectorAll("[data-region-tool]").forEach((b) => b.classList.toggle("on", b.dataset.regionTool === tool));
  $("region-tool-msg").textContent = tool ? `${TOOL_SAYS[tool]} · right-drag turns the model` : "no tool: dragging turns the model";
  partsView.orbit.mouseButtons.RIGHT = tool ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN;
  canvas.style.cursor = tool ? "crosshair" : "";
}
function brushAt(e) {
  const hit = partsView.pick(e, partMeshes);
  if (!hit) return;
  const radius = +$("region-brush").value / 100;
  const touched = [];
  for (const [r, label] of topo.piece) if (label === selected && topo.pos.get(r).distanceTo(hit.point) <= radius) touched.push(r);
  // a brush smaller than the triangle under it touches none of its corners (a low-poly lock: edges of
  // 5 cm under a 2 cm brush, and nothing was painted): the triangle that was hit is taken whole
  const numbers = hit.object.geometry.attributes._vertex, pieces = hit.object.geometry.attributes._piece;
  if (numbers && pieces && hit.face) for (const i of [hit.face.a, hit.face.b, hit.face.c]) {
    if (Math.round(pieces.getX(i)) === selected) touched.push(topo.rep.get(Math.round(numbers.getX(i))));
  }
  for (const r of touched) if (painting.erase) region.delete(r); else region.add(r);
  colourPieces();
}
const canvas = partsView.renderer.domElement;
$("region-tools").onclick = (e) => { if (e.target.dataset.regionTool) setTool(e.target.dataset.regionTool); };
canvas.addEventListener("contextmenu", (e) => { if (tool) e.preventDefault(); });
canvas.addEventListener("pointerdown", (e) => {
  const mode = e.altKey ? (e.ctrlKey ? "erase" : "brush") : e.shiftKey ? "box" : e.button === 0 ? tool : null;
  if (!mode) return;
  // the piece under the cursor is the one to work on, unless a part of another is already begun
  const under = partsView.pick(e, partMeshes), attr = under?.object.geometry.attributes._piece;
  const label = attr ? Math.round(attr.getX(under.face.a)) : null;
  if (label != null && label !== selected && (selected == null || !region.size)) selectPiece(label);
  if (selected == null) { toast("Drag on the model: start on the piece the part belongs to", "info"); return; }
  if (!topo) buildTopo();
  $("region-card").open = true;
  e.preventDefault(); e.stopPropagation(); partsView.orbit.enabled = false;
  if (mode !== "box") { painting = { erase: mode === "erase" }; brushAt(e); return; }
  dragFrom = { x: e.offsetX, y: e.offsetY, add: e.ctrlKey || (!e.shiftKey && tool === "box") };
}, true);
canvas.addEventListener("pointermove", (e) => {
  if (painting) { brushAt(e); return; }
  if (!dragFrom) return;
  const [x0, x1] = [dragFrom.x, e.offsetX].sort((a, b) => a - b), [y0, y1] = [dragFrom.y, e.offsetY].sort((a, b) => a - b);
  Object.assign(box.style, { display: "block", left: `${x0}px`, top: `${y0}px`, width: `${x1 - x0}px`, height: `${y1 - y0}px` });
});
canvas.addEventListener("pointerup", (e) => {
  if (painting) { painting = null; partsView.orbit.enabled = true; showRegion(); return; }
  if (!dragFrom) return;
  const r = canvas.getBoundingClientRect();
  const [x0, x1] = [dragFrom.x, e.offsetX].sort((a, b) => a - b), [y0, y1] = [dragFrom.y, e.offsetY].sort((a, b) => a - b);
  if (!dragFrom.add) region = new Set();
  const v = new THREE.Vector3();
  for (const [rep, label] of topo.piece) {
    if (label !== selected) continue;
    v.copy(topo.pos.get(rep)).project(partsView.camera);
    const sx = (v.x + 1) / 2 * r.width, sy = (1 - v.y) / 2 * r.height;
    if (sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1) region.add(rep);
  }
  dragFrom = null; box.style.display = "none"; partsView.orbit.enabled = true;
  showRegion();
});
function showRegion() {
  $("region-save").disabled = !region.size;
  $("region-warn").textContent = "";
  if (!region.size) { $("region-count").textContent = "no part chosen"; colourPieces(); return; }
  const n = regionNumbers().length, rep = regionReport(), warn = [];
  $("region-count").textContent = `${n.toLocaleString()} vertices chosen in piece #${selected}`;
  if (rep.bits > 1) warn.push(`it is in ${rep.bits} separate bits: a part is one connected piece, so paint the bits together (or Grow)`);
  if (rep.gaps) warn.push(`${rep.gaps} vertices inside it are left out: press Fill gaps`);
  // a clean cut crosses the part once, at its root; a ragged pick borders the rest all over
  if (rep.rim > 3 * Math.sqrt(region.size)) warn.push(`its edge meets the rest of the piece at ${rep.rim} vertices, a ragged edge for its size: every one of them can tear`);
  $("region-warn").innerHTML = warn.length ? `<span class="bad">Check the pick:</span> ${warn.join("; ")}.` : `<span class="ok">One connected part with a clean edge.</span>`;
  colourPieces();
}
$("region-brush").oninput = () => { $("region-brush-cm").textContent = `${$("region-brush").value} cm`; };
$("region-grow").onclick = () => { if (topo && region.size) { grow(); showRegion(); } };
$("region-shrink").onclick = () => { if (topo && region.size) { shrink(); showRegion(); } };
$("region-fill").onclick = () => { if (topo && region.size) { fillGaps(); showRegion(); } };
$("region-role").innerHTML = ROLES.filter(Boolean).map((r) => `<option value="${r}" title="${ROLE_INFO[r][1]}">${roleName(r)}</option>`).join("");
$("region-role").value = "hair";  // what a part is nearly always painted for (it opened on "body", which changes nothing)
$("role-legend").innerHTML = ROLES.map((r) => `<p class="help" style="margin:2px 0"><span class="swatch" style="background:${hex(ROLE_COLOUR[r])}"></span>
  <b style="color:var(--text)">${roleName(r)}</b>: ${ROLE_INFO[r][1]}</p>`).join("");
$("region-clear").onclick = () => { region = new Set(); showRegion(); };
$("region-card").addEventListener("toggle", () => {
  if (!$("region-card").open && tool) setTool(tool);
  if ($("region-card").open) $("region-card").scrollIntoView({ block: "nearest", behavior: "smooth" });
});
$("region-save").onclick = async () => {
  const name = $("region-name").value.trim() || `${$("region-role").value}_part_${selected}`;
  const rep = regionReport();
  if ((rep.bits > 1 || rep.gaps) && !await ask("Save this part anyway?", { text: $("region-warn").textContent, ok: "Save anyway" })) return;
  try {
    await call("piece", { name, label: selected, role: $("region-role").value, region: regionNumbers() });
    pushUndo(`part ${name}`, async () => { await call("piece", { name, remove: true }); await changed(); });
    region = new Set(); showRegion(); await changed();
    saved(`Saved ${name}. Rebuild from “build” (or “pieces” for a clothing role) for it to count.`);
  } catch (e) { say("pieces-msg", e.message, true); }
};

// ---------- picking
// a click picks a piece; letting go after dragging to orbit doesn't (it picked whatever was under the cursor)
let partsDown = null;
canvas.addEventListener("pointerdown", (e) => { partsDown = { x: e.clientX, y: e.clientY }; });
canvas.addEventListener("click", (event) => {
  if (event.shiftKey || event.altKey || tool) return;
  if (partsDown && Math.hypot(event.clientX - partsDown.x, event.clientY - partsDown.y) > 4) return;
  const hit = partsView.pick(event, partMeshes);
  const attr = hit?.object.geometry.attributes._piece;
  if (!attr) return;
  selectPiece(Math.round(attr.getX(hit.face.a)));
});
function selectPiece(label) {
  if (label !== selected && region.size) { region = new Set(); showRegion(); }
  selected = label;
  const row = S.listing?.pieces.find((r) => r.label === label);
  // a piece hidden by the filter comes into the list when picked in the view
  if (row && !visible(row)) { filter = "all"; $("hide-specks").checked = row.count >= SPECK && $("hide-specks").checked; paintFilter(); }
  renderPieces(); colourPieces();
  const tr = document.querySelector(`#piece-rows tr[data-label="${label}"]`);
  if (tr) tr.scrollIntoView({ block: "nearest" }); else say("pieces-msg", "a speck too small to list");
}

$("pieces-run").onclick = () => job("list_pieces", {});

// ---------- the list
const hasSuggestion = (row) => row.suggested !== "body" && !namedFor(row);
function visible(row) {
  if (row.label === selected) return true;
  if ($("hide-specks").checked && row.count < SPECK) return false;
  return filter === "all" || (filter === "suggested" ? hasSuggestion(row) : !!namedFor(row));
}
function paintFilter() { $("piece-filter").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.filter === filter)); }
$("piece-filter").onclick = (e) => { if (e.target.dataset.filter) { filter = e.target.dataset.filter; paintFilter(); renderPieces(); } };
$("hide-specks").onchange = () => renderPieces();
document.querySelectorAll("#piece-list th.sort").forEach((th) => th.onclick = () => {
  sort = { key: th.dataset.sort, dir: sort.key === th.dataset.sort && sort.dir === "desc" ? "asc" : "desc" };
  document.querySelectorAll("#piece-list th.sort").forEach((t) => t.dataset.dir = t === th ? sort.dir : "");
  renderPieces();
});
document.querySelectorAll("#piece-list th.sort").forEach(keyable);
function sorted(rows) {
  const key = { count: (r) => r.count, name: (r) => (namedFor(r)?.name || r.where || "").toLowerCase(), suggested: (r) => r.suggested }[sort.key];
  return [...rows].sort((a, b) => (key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0) * (sort.dir === "asc" ? 1 : -1));
}

// the details a role needs, when it's first given: follow_bone the upper arm on its side, detail an anchor
function withDefaults(role, row, named) {
  return role === "follow_bone" ? { role, bone: row.max[0] > 0 ? "LeftUpperArm" : "RightUpperArm" }
    : role === "detail" ? { role, anchor: S.manifest.pieces.find((p) => p.name !== named?.name)?.name } : { role };
}
async function changed() { await refreshManifest(); renderPieces(); colourPieces(); app.renderNav(); app.refreshStatus?.(); }

// Gives a listed piece a role (or takes it away); undo puts the piece back as it was.
async function setRole(row, patch, { quiet = false } = {}) {
  const named = namedFor(row), before = named ? { ...named } : null;
  const name = named?.name || `${patch.role || named?.role}_${row.label}`;
  if (patch.role === "") { if (named) await call("piece", { name: named.name, remove: true }); }
  else await call("piece", { name, label: row.label, ...patch });
  const restore = async () => {
    // (the tip said either way: left unsaid, a tip ticked since stayed on after the undo)
    if (before) await call("piece", { name: before.name, label: row.label, role: before.role, bone: before.bone, anchor: before.anchor, tip: !!before.tip });
    else if (patch.role !== "") await call("piece", { name, remove: true });
  };
  if (!quiet) {
    pushUndo(`${roleName(patch.role)} on ${name}`, async () => { await restore(); await changed(); });
    saved(patch.role === "" ? `${name}: back to automatic.` : `${name}: ${roleName(patch.role)}. Rebuild from “${ROLE_STAGE(patch.role)}” for it to count.`);
  }
  return restore;
}

export function renderPieces() {
  const body = $("piece-rows");
  // suggestions made before the eyes were marked read the face by size: a tester accepted a face as hair
  $("accept-all").disabled = S.listingStale || !S.listing?.pieces.some(hasSuggestion);
  if (S.listing && S.listingStale) say("pieces-msg", "These suggestions were made before the face was marked and may take the face for hair. Press List pieces to make them again.", true);
  if (!S.listing) { body.innerHTML = `<tr><td colspan="6" class="msg">Not listed yet: press List pieces (one Blender run, about a minute).</td></tr>`; return; }
  const rows = sorted(S.listing.pieces.filter(visible));
  const hidden = S.listing.pieces.length - rows.length;
  // questions first: a row waiting for an answer is lost under hundreds of pieces
  body.innerHTML = foundRows() + rows.map((row) => {
    const named = namedFor(row) || {};
    const role = named.role || "";
    const others = S.manifest.pieces.filter((p) => p.name !== named.name).map((p) => p.name);
    return `<tr data-label="${row.label}" class="${row.label === selected ? "sel" : ""}">
      <td><span class="swatch" style="background:${hex(ROLE_COLOUR[role] ?? 0x9a9a9a)}"></span></td>
      <td>${esc(named.name || row.where || `piece ${row.label}`)}${named.name ? ` <button class="act alt small" data-rename title="rename">✎</button>` : ""}
        <span class="sub">${row.where && named.name ? esc(row.where) : `#${row.label}`}</span></td><td>${row.count.toLocaleString()}</td>
      <td title="${esc(row.why)}">${roleName(row.suggested === "body" ? "body" : row.suggested)}</td>
      <td><select data-role aria-label="role">${ROLES.map((r) => `<option ${r === role ? "selected" : ""} value="${r}" title="${ROLE_INFO[r][1]}">${roleName(r)}</option>`).join("")}</select>
        ${role === "follow_bone" ? `<select data-bone aria-label="bone">${BONES.map((b) => `<option ${b === named.bone ? "selected" : ""}>${b}</option>`).join("")}</select>` : ""}
        ${role === "appendage" ? `<select data-core title="the bone it grows from">${CORE.map((b) => `<option ${b === (named.bone || "") ? "selected" : ""} value="${b}">${b || "grows from: nearest"}</option>`).join("")}</select>` : ""}
        ${role === "detail" ? `<select data-anchor aria-label="rides on">${others.map((n) => `<option ${n === named.anchor ? "selected" : ""}>${esc(n)}</option>`).join("")}</select>` : ""}
        ${role === "hat" ? `<label class="inline"><input type="checkbox" data-tip ${named.tip ? "checked" : ""}> tip swings</label>` : ""}</td>
      <td>${hasSuggestion(row) ? `<button class="act alt small" data-accept title="${ROLE_INFO[row.suggested]?.[1] ?? ""}">use: ${roleName(row.suggested)}</button>` : ""}</td></tr>`;
  }).join("") + regionRows()
    + (hidden ? `<tr><td colspan="6" class="msg">${hidden} more hidden by the filter.</td></tr>` : "");
  body.querySelectorAll("tr[data-region]").forEach((tr) => {
    const name = tr.dataset.region, piece = S.manifest.pieces.find((p) => p.name === name);
    tr.querySelector("[data-rename]").onclick = async () => {
      const to = await ask("Rename the part", { input: true, value: name, ok: "Rename" });
      if (!to || to === name) return;
      try { await call("piece", { name, rename: to }); pushUndo(`rename ${to}`, async () => { await call("piece", { name: to, rename: name }); await changed(); }); await changed(); }
      catch (e) { say("pieces-msg", e.message, true); }
    };
    tr.querySelector("[data-remove]").onclick = async () => {
      if (!await ask(`Remove the part ${name}?`, { ok: "Remove", danger: true })) return;
      const whole = wholeOf(piece);
      try {
        await call("piece", { name, remove: true });
        if (whole) pushUndo(`remove ${name}`, async () => { await call("piece", { name, label: whole.label, role: piece.role, region: piece.region }); await changed(); });
        await changed(); saved(`Removed ${name}.`);
      } catch (e) { say("pieces-msg", e.message, true); }
    };
  });
  body.querySelectorAll("tr[data-found]").forEach((tr) => {
    const part = foundParts().find((p) => p.name === tr.dataset.found);
    tr.querySelectorAll("[data-use-found]").forEach((button) => button.onclick = async () => {
      const [role, what] = FOUND_AS[button.dataset.useFound];
      try {
        await call("piece", { name: part.name, label: part.label, role, region: part.region, ...(role === "appendage" ? { bone: part.bone } : {}) });
        pushUndo(`${what} on ${part.name}`, async () => { await call("piece", { name: part.name, remove: true }); await changed(); });
        await changed(); saved(`${part.name}: ${what}. Rebuild from “${ROLE_STAGE(role)}” for it to count.`);
      } catch (e) { say("pieces-msg", e.message, true); }
    });
  });
  body.querySelectorAll("tr[data-label]").forEach((tr) => {
    const label = +tr.dataset.label, row = S.listing.pieces.find((r) => r.label === label);
    const named = namedFor(row);
    keyable(tr);
    tr.onclick = (e) => { if (!["SELECT", "INPUT", "BUTTON", "OPTION", "LABEL"].includes(e.target.tagName)) selectPiece(label); };
    tr.onmouseenter = () => { hovered = label; colourPieces(); };
    tr.onmouseleave = () => { hovered = null; colourPieces(); };
    const save = async (patch) => {
      try { await setRole(row, patch); await changed(); } catch (e) { say("pieces-msg", e.message, true); }
    };
    tr.querySelector("[data-role]").onchange = (e) => save(withDefaults(e.target.value, row, named));
    tr.querySelector("[data-bone]")?.addEventListener("change", (e) => save({ role: "follow_bone", bone: e.target.value }));
    tr.querySelector("[data-core]")?.addEventListener("change", (e) => save({ role: "appendage", bone: e.target.value || "auto" }));
    tr.querySelector("[data-anchor]")?.addEventListener("change", (e) => save({ role: "detail", anchor: e.target.value }));
    tr.querySelector("[data-tip]")?.addEventListener("change", (e) => save({ role: "hat", tip: e.target.checked }));
    tr.querySelector("[data-accept]")?.addEventListener("click", () => save(withDefaults(row.suggested, row, named)));
    tr.querySelector("[data-rename]")?.addEventListener("click", async () => {
      const to = await ask("Rename the piece", { text: "e.g. left wing. Details riding on it follow the new name.", input: true, value: named.name, ok: "Rename" });
      if (!to || to === named.name) return;
      try { await call("piece", { name: named.name, rename: to }); pushUndo(`rename ${to}`, async () => { await call("piece", { name: to, rename: named.name }); await changed(); }); await changed(); }
      catch (e) { say("pieces-msg", e.message, true); }
    });
  });
}
const wholeOf = (p) => S.listing?.pieces.find((r) => r.count === p.count && Math.hypot(...p.point.map((v, i) => v - r.point[i])) < 1e-6);
// parts of pieces (a twin-tail fused into the body) are listed after the pieces, each on its own row
// Parts the listing found inside pieces (ears or horns over the crown, with no piece of their own):
// asked about in a row each until answered; then it is a saved part like a painted one. Shape doesn't
// tell an ear from a horn, and one swings while the other must not: the user says which.
const FOUND_AS = { ear: ["appendage", "ear (swings)"], horn: ["hat", "horn (rigid on head)"] };
const foundParts = () => (S.listing?.parts || []).filter((p) => !S.manifest.pieces.some((m) => m.name === p.name));
function foundRows() {
  return foundParts().map((p) => `<tr data-found="${esc(p.name)}"><td><span class="swatch" style="background:${hex(0x9a9a9a)}"></span></td>
      <td>${esc(p.name)}<span class="sub">found in #${p.label}: ${esc(p.where)}</span></td>
      <td>${p.count.toLocaleString()}</td><td title="${esc(p.why)}">ear or horn?</td><td><span class="sub">${esc(p.why)}</span></td>
      <td><button class="act alt small" data-use-found="ear" title="an animal ear: gets a spring and sways when the head moves">ear: swings</button>
        <button class="act alt small" data-use-found="horn" title="a horn: ${ROLE_INFO.hat[1]}">horn: stays still</button></td></tr>`).join("");
}

function regionRows() {
  return S.manifest.pieces.filter((p) => p.region?.length).map((p) => {
    const whole = wholeOf(p);
    return `<tr data-region="${esc(p.name)}"><td><span class="swatch" style="background:${hex(ROLE_COLOUR[p.role] ?? 0x9a9a9a)}"></span></td>
      <td>${esc(p.name)} <button class="act alt small" data-rename title="rename">✎</button><span class="sub">part of ${whole ? `#${whole.label}` : "a piece"}</span></td>
      <td>${p.region.length.toLocaleString()}</td><td></td><td>${esc(roleName(p.role))}</td><td><button class="act alt small" data-remove>remove</button></td></tr>`;
  }).join("");
}

$("accept-all").onclick = async () => {
  const rows = S.listing.pieces.filter(hasSuggestion);
  const list = rows.slice(0, 12).map((r) => `• #${r.label} (${r.count.toLocaleString()} vertices): ${roleName(r.suggested)}`).join("\n");
  if (!await ask(`Accept ${rows.length} suggestion${rows.length > 1 ? "s" : ""}?`, { text: list + (rows.length > 12 ? `\n… and ${rows.length - 12} more` : ""), ok: "Accept all" })) return;
  const restores = [];
  try {
    for (const row of rows) restores.push(await setRole(row, withDefaults(row.suggested, row, null), { quiet: true }));
  } catch (e) { say("pieces-msg", e.message, true); }
  pushUndo(`${restores.length} suggestions`, async () => { for (const r of restores.reverse()) await r(); await changed(); });
  await changed();
  saved(`Accepted ${restores.length} suggestions. Rebuild from “build” for them to count.`);
};

// Movement's "Apply this fix" gives the role through here, so it's undoable the same way.
app.setRole = setRole;
app.piecesChanged = changed;
