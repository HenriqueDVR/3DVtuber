// Paint: fix the texture on the model itself. What is drawn is kept as one transparent overlay per
// picture of the source model (app/paint.py), laid over it at the build step; the generated
// texture is never changed, so the eraser always gets it back.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { $, S, app, call, esc, say, refreshManifest, refreshStatus } from "./core.js";
import { makeView } from "./views.js";
import { ask, pushUndo, redo, toast, undo } from "./ui.js";
import { job } from "./jobs.js";
import { onShow } from "./nav.js";

const view = makeView($("paint-view"));
export const paintView = view;
// left paints (on the model), right orbits, middle pans: a paint page where a plain drag turned the
// model would be the wrong way round
view.orbit.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.ROTATE };
const ring = new THREE.Mesh(new THREE.RingGeometry(0.94, 1, 48), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true, opacity: 0.9, side: THREE.DoubleSide }));
ring.renderOrder = 10; ring.visible = false; view.scene.add(ring);

// Brush sizes are in cm on a 1.6 m character, whatever size the model file is.
const HEIGHT_CM = 160;
// How far past a triangle's edge paint goes, in texels: without it filtering draws the old colour as a line along UV seams.
const PAD = 1;
// The picture is drawn and sent to the graphics card in squares this wide. A generated model's
// texture is hundreds of small patches all over the picture: one stroke touches texels far apart,
// and redrawing the rectangle around them all was the whole 4096 px picture, every frame.
const TILE = 64;
const TOOLS = {
  brush: "Brush: paints the colour. Alt+click picks a colour from the model.",
  erase: "Eraser: takes paint off; the generated texture shows again.",
  soften: "Soften: blurs what is under the brush, for a hard edge or a speckled patch. A bigger brush blurs wider; stroke again for more.",
  clone: "Clone: Ctrl+click the spot to copy from, then paint elsewhere on the same picture (it copies along the texture, so stay within one patch of it).",
  fill: "Fill: click a patch to fill everything of a similar colour joined to it (it stops at the texture's seams: click again past one).",
  pick: "Pick colour: click the model to take its colour.",
  isolate: "Isolate: click a part to see only that part, so nothing is in the way of the brush; Shift+click adds another. The tool you had comes back for painting. “Show all” (or H) brings the rest back.",
};
const MODES = { normal: "as it is", multiply: "shade (darkens)", screen: "light (brightens)", colour: "recolour (keeps the shading and lines)" };

let P = null;       // the open character's model and pictures, or null
let key = null;     // what P was loaded for: the manifest and its source
let tool = "brush", stroke = null, cloneFrom = null, lastScreen = null, pending = null, loading = false;
// the stroke's brush, read once when it starts: the texel loop reads plain numbers
const B = { tool: "brush", mode: "normal", r: 0, g: 0, b: 0, flow: 1, size: 1, hard: 0.5, mirror: false, soft: 1, dx: 0, dy: 0 };
const T = [0, 0, 0];  // a colour handed back without making an array per texel
const rgb = (text) => [1, 3, 5].map((i) => parseInt(text.slice(i, i + 2), 16));
const hexOf = (c) => `#${c.map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")}`;
function readBrush(as = tool) {
  const c = rgb($("paint-colour").value);
  Object.assign(B, { tool: as, mode: $("paint-mode").value, r: c[0], g: c[1], b: c[2], flow: +$("paint-flow").value, size: +$("paint-size").value,
    hard: +$("paint-hard").value, mirror: $("paint-mirror").checked, soft: 1, dx: 0, dy: 0 });
}

// ---------- loading: the source model, its base-colour pictures as pixels, and the paint kept so far
export function resetPaint() {
  const now = S.manifestPath ? `${S.manifestPath}|${S.manifest?.source}` : null;
  if (now === key) return;  // the same character read again (a run ended): unsaved paint stays
  key = now; P = null; cloneFrom = null;
  view.clear(); ring.visible = false;
  showState();
  if ($("paint").classList.contains("on")) load();
}
onShow.paint = () => { if (!P) load(); };

async function load() {
  if (loading || !S.manifestPath) return;
  loading = true;
  const mine = key;
  say("paint-msg", "Loading the model…");
  try {
    const state = await call("paint");
    if (!state.paintable) { say("paint-msg", `Nothing to paint on yet: ${state.why}.`, true); return; }
    const gltf = await new GLTFLoader().loadAsync(state.model);
    if (mine !== key) return;
    gltf.scene.updateMatrixWorld(true);
    const p = { layers: new Map(), meshes: [], total: 0, grid: null, links: null, island: null, piece: null, hidden: null, shown: null };
    const json = gltf.parser.json;
    gltf.scene.traverse((mesh) => {
      const map = mesh.isMesh && mesh.material?.map, g = mesh.geometry;
      if (!map || !g.attributes.uv) return;
      const image = json.textures[gltf.parser.associations.get(map)?.textures]?.source;
      const picture = state.pictures.find((x) => x.index === image);
      if (!picture) return;
      if (!p.layers.has(image)) p.layers.set(image, layerOf(picture, map));
      const L = p.layers.get(image);
      // positions in the scene's own space, once: every dab measures distances there
      const count = g.attributes.position.count, pos = new Float32Array(count * 3), uv = new Float32Array(count * 2), v = new THREE.Vector3();
      for (let i = 0; i < count; i++) {
        v.fromBufferAttribute(g.attributes.position, i).applyMatrix4(mesh.matrixWorld);
        pos[i * 3] = v.x; pos[i * 3 + 1] = v.y; pos[i * 3 + 2] = v.z;
        uv[i * 2] = g.attributes.uv.getX(i); uv[i * 2 + 1] = g.attributes.uv.getY(i);
      }
      const idx = g.index ? g.index.array : Uint32Array.from({ length: count }, (_, i) => i);
      p.meshes.push({ mesh, L, pos, uv, idx, base: p.total, count: idx.length / 3, side: mesh.material.side, drawn: g.index });
      p.total += idx.length / 3;
      mesh.userData.lit = mesh.material.clone(); mesh.userData.lit.map = L.texture;
      mesh.userData.flat = new THREE.MeshBasicMaterial({ map: L.texture, side: mesh.material.side });
    });
    if (!p.meshes.length) { say("paint-msg", "Nothing to paint on: the model's texture isn't on a surface the page can read.", true); return; }
    let restored = 0;
    for (const L of p.layers.values()) {
      // paint the window closed on, or a crash took, before it was saved
      if (L.draftUrl) { await readOverlay(L, L.draftUrl); L.changed = true; restored++; }
      else if (L.url) await readOverlay(L, L.url);
    }
    const box = new THREE.Box3().setFromObject(gltf.scene);
    p.min = box.min.toArray(); p.max = box.max.toArray(); p.height = box.max.y - box.min.y; p.middle = (box.min.x + box.max.x) / 2;
    p.stamp = new Uint32Array(p.total); p.dabs = 0; p.rayStamp = new Uint32Array(p.total); p.rays = 0;
    p.seen = new Uint32Array(p.total * 2); p.strokes = 0;
    P = p;
    view.clear(); view.model = gltf.scene; view.scene.add(gltf.scene); view.frame();
    $("paint-picture").innerHTML = [...p.layers.values()].map((L) => `<option value="${L.index}">${esc(L.name)} (${L.W} × ${L.H})</option>`).join("");
    showLit(); for (const L of p.layers.values()) { flushAll(L); view.renderer.initTexture(L.texture); }
    preview();
    say("paint-msg", restored ? "Brought back paint that wasn't saved last time. Save to keep it."
      : state.orphans?.length ? `Paint kept for a picture this model no longer has (${state.orphans.join(", ")}) is left alone and not used.` : "");
    $("paint-old").hidden = !state.old_patch;
    showState(); showIsolated();
    // worked out now, a moment after the model shows, not in the middle of the first stroke
    setTimeout(() => { if (P === p) { grid(); links(); for (const L of p.layers.values()) owners(L); } }, 100);
  } catch (e) { say("paint-msg", e.message, true); }
  finally { loading = false; }
}

function layerOf(picture, map) {
  const image = map.image, W = image.width, H = image.height;
  const canvas = Object.assign(document.createElement("canvas"), { width: W, height: H });
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0);
  const base = ctx.getImageData(0, 0, W, H).data;
  const texture = new THREE.CanvasTexture(canvas);
  // the picture as the model's own texture reads it: same way up, same colours, same wrapping. No
  // smaller copies of it: they would be made again from the whole picture after every dab
  Object.assign(texture, { flipY: map.flipY, colorSpace: map.colorSpace, wrapS: map.wrapS, wrapT: map.wrapT, channel: map.channel,
    generateMipmaps: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
  const tx = Math.ceil(W / TILE), ty = Math.ceil(H / TILE);
  return { index: picture.index, name: picture.name, file: picture.overlay, url: picture.url, draftUrl: picture.draft_url, drafted: false, W, H, base, canvas, ctx, texture,
    over: new Uint8ClampedArray(W * H * 4), before: new Uint8ClampedArray(W * H * 4), mask: new Uint8Array(W * H),
    tx, drawn: new Uint8Array(tx * ty), draw: [], touched: new Uint8Array(tx * ty), touch: [], changed: false, own: null };
}
async function readOverlay(L, url) {
  const bitmap = await createImageBitmap(await (await fetch(`${url}&t=${Date.now()}`)).blob());
  const c = Object.assign(document.createElement("canvas"), { width: L.W, height: L.H });
  const ctx = c.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, L.W, L.H);
  L.over.set(ctx.getImageData(0, 0, L.W, L.H).data);
}

// ---------- geometry: which triangles are near a point, what a ray hits, and which piece each belongs to
function meshOf(t) { for (const m of P.meshes) if (t < m.base + m.count) return m; return null; }
const cellKey = (x, y, z) => (x + 8) + (y + 8) * 512 + (z + 8) * 262144;
function grid() {
  if (P.grid) return P.grid;
  const cell = P.height / 64, cells = new Map(), [mx, my, mz] = P.min;
  for (const m of P.meshes) for (let t = 0; t < m.count; t++) {
    let lx = 1e9, ly = 1e9, lz = 1e9, hx = -1e9, hy = -1e9, hz = -1e9;
    for (let k = 0; k < 3; k++) {
      const o = m.idx[t * 3 + k] * 3, x = m.pos[o], y = m.pos[o + 1], z = m.pos[o + 2];
      if (x < lx) lx = x; if (x > hx) hx = x; if (y < ly) ly = y; if (y > hy) hy = y; if (z < lz) lz = z; if (z > hz) hz = z;
    }
    for (let x = Math.floor((lx - mx) / cell); x <= Math.floor((hx - mx) / cell); x++)
      for (let y = Math.floor((ly - my) / cell); y <= Math.floor((hy - my) / cell); y++)
        for (let z = Math.floor((lz - mz) / cell); z <= Math.floor((hz - mz) / cell); z++) {
          const id = cellKey(x, y, z);
          let list = cells.get(id); if (!list) cells.set(id, list = []);
          list.push(m.base + t);
        }
  }
  return P.grid = { cell, cells };
}
// The nearest triangle a ray hits, found cell by cell along it: three's own ray test tries every
// triangle of the model on every mouse move, which is what made a stroke stutter.
function cast(o, d) {
  const { cell, cells } = grid();
  let t0 = 0, t1 = 1e9;
  for (let a = 0; a < 3; a++) {
    const lo = (P.min[a] - cell - o[a]) / d[a], hi = (P.max[a] + cell - o[a]) / d[a];
    if (d[a] === 0) { if (o[a] < P.min[a] - cell || o[a] > P.max[a] + cell) return null; continue; }
    t0 = Math.max(t0, Math.min(lo, hi)); t1 = Math.min(t1, Math.max(lo, hi));
  }
  if (t0 > t1) return null;
  const c = [0, 0, 0], step = [0, 0, 0], next = [0, 0, 0], delta = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const at = o[a] + d[a] * t0 - P.min[a];
    c[a] = Math.floor(at / cell);
    step[a] = d[a] > 0 ? 1 : -1;
    delta[a] = d[a] ? cell / Math.abs(d[a]) : 1e30;
    next[a] = d[a] ? t0 + ((d[a] > 0 ? (c[a] + 1) * cell - at : at - c[a] * cell)) / Math.abs(d[a]) : 1e30;
  }
  const stampNo = ++P.rays;
  let best = null;
  for (let guard = 0; guard < 4096; guard++) {
    const list = cells.get(cellKey(c[0], c[1], c[2]));
    if (list) for (const t of list) {
      if (P.rayStamp[t] === stampNo) continue;
      P.rayStamp[t] = stampNo;
      if (P.hidden && P.hidden[t]) continue;  // a part put out of the way (isolate)
      const m = meshOf(t), q = (t - m.base) * 3, ia = m.idx[q] * 3, ib = m.idx[q + 1] * 3, ic = m.idx[q + 2] * 3, p = m.pos;
      const e1x = p[ib] - p[ia], e1y = p[ib + 1] - p[ia + 1], e1z = p[ib + 2] - p[ia + 2], e2x = p[ic] - p[ia], e2y = p[ic + 1] - p[ia + 1], e2z = p[ic + 2] - p[ia + 2];
      const hx = d[1] * e2z - d[2] * e2y, hy = d[2] * e2x - d[0] * e2z, hz = d[0] * e2y - d[1] * e2x, det = e1x * hx + e1y * hy + e1z * hz;
      if (det > -1e-14 && det < 1e-14) continue;
      // a face the view doesn't draw (the far wall's inside, seen through a gap) isn't there to paint on
      if (m.side === THREE.FrontSide ? det < 0 : m.side === THREE.BackSide && det > 0) continue;
      const sx = o[0] - p[ia], sy = o[1] - p[ia + 1], sz = o[2] - p[ia + 2], u = (sx * hx + sy * hy + sz * hz) / det;
      if (u < 0 || u > 1) continue;
      const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x, v = (d[0] * qx + d[1] * qy + d[2] * qz) / det;
      if (v < 0 || u + v > 1) continue;
      const dist = (e2x * qx + e2y * qy + e2z * qz) / det;
      if (dist > 1e-9 && (!best || dist < best.dist)) best = { dist, tri: t, u, v, m };
    }
    const a = next[0] < next[1] ? (next[0] < next[2] ? 0 : 2) : (next[1] < next[2] ? 1 : 2);
    if (best && best.dist <= next[a]) break;  // nothing in a later cell can be nearer
    if (next[a] > t1) break;
    c[a] += step[a]; next[a] += delta[a];
  }
  return best;
}
// The surface as it hangs together: for every place a vertex sits (UV seams split one place into
// several vertices), the triangles that meet there. {at: each triangle's three places, first/list:
// the triangles at each place}
function links() {
  if (P.links) return P.links;
  const ids = new Map(), at = new Int32Array(P.total * 3), scale = 1e5 / P.height, [mx, my, mz] = P.min;
  let places = 0;
  for (const m of P.meshes) {
    const of = new Int32Array(m.pos.length / 3);
    for (let i = 0; i < of.length; i++) {
      // the same place is the same three numbers exactly, so rounding never splits it
      const k = Math.round((m.pos[i * 3] - mx) * scale) + Math.round((m.pos[i * 3 + 1] - my) * scale) * 262144 + Math.round((m.pos[i * 3 + 2] - mz) * scale) * 68719476736;
      let id = ids.get(k); if (id === undefined) { id = places++; ids.set(k, id); }
      of[i] = id;
    }
    for (let q = 0; q < m.count * 3; q++) at[m.base * 3 + q] = of[m.idx[q]];
  }
  const first = new Int32Array(places + 1);
  for (let q = 0; q < at.length; q++) first[at[q] + 1]++;
  for (let i = 0; i < places; i++) first[i + 1] += first[i];
  const list = new Int32Array(at.length), next = first.slice(0, places);
  for (let q = 0; q < at.length; q++) list[next[at[q]]++] = (q / 3) | 0;
  return P.links = { at, first, list };
}
// The model's separate parts (the Parts page's pieces): triangles joined through the places their
// corners sit, whatever the texture's seams.
function pieces() {
  if (P.piece) return P.piece;
  const { at, first } = links(), parent = new Int32Array(first.length - 1).map((_, i) => i);
  const find = (a) => { while (parent[a] !== a) a = parent[a] = parent[parent[a]]; return a; };
  for (let t = 0; t < P.total; t++) { const a = find(at[t * 3]); parent[find(at[t * 3 + 1])] = a; parent[find(at[t * 3 + 2])] = a; }
  const piece = new Int32Array(P.total);
  for (let t = 0; t < P.total; t++) piece[t] = find(at[t * 3]);
  return P.piece = piece;
}
// Only these parts are drawn, and only they are there for the brush and for what counts as in
// the way: the inside of a collar or the neck behind hair is painted with the hair gone. null: all.
function isolate(shown) {
  if (!P) return;
  const piece = shown && pieces();
  P.shown = shown && shown.size ? shown : null;
  P.hidden = P.shown ? Uint8Array.from(piece, (id) => (P.shown.has(id) ? 0 : 1)) : null;
  for (const m of P.meshes) {
    if (!P.hidden) { m.mesh.geometry.setIndex(m.drawn); continue; }
    const kept = [];
    for (let t = 0; t < m.count; t++) if (!P.hidden[m.base + t]) kept.push(m.idx[t * 3], m.idx[t * 3 + 1], m.idx[t * 3 + 2]);
    m.mesh.geometry.setIndex(new THREE.BufferAttribute(Uint32Array.from(kept), 1));
  }
  P.strokes++;  // what the eye sees has changed
  showIsolated();
}
function showIsolated() {
  $("paint-all").hidden = !P?.shown;
  $("paint-all").textContent = P?.shown ? `Show all (${P.shown.size} part${P.shown.size > 1 ? "s" : ""} shown · H)` : "";
}
// Texture patches: triangles that share a vertex of the file share its place in the picture too
// (a vertex on a UV seam is two vertices), so they are one patch of the texture.
function islands() {
  if (P.island) return P.island;
  const island = new Int32Array(P.total);
  let offset = 0;
  for (const m of P.meshes) {
    const parent = new Int32Array(m.pos.length / 3).map((_, i) => i);
    const find = (a) => { while (parent[a] !== a) a = parent[a] = parent[parent[a]]; return a; };
    for (let t = 0; t < m.count; t++) { const a = find(m.idx[t * 3]); parent[find(m.idx[t * 3 + 1])] = a; parent[find(m.idx[t * 3 + 2])] = a; }
    for (let t = 0; t < m.count; t++) island[m.base + t] = offset + find(m.idx[t * 3]);
    offset += parent.length;
  }
  return P.island = island;
}
// Which patch of the texture each texel belongs to (-1: none). A texel under a triangle's edge
// lies partly under the next patch of the picture too, and that patch is somewhere else on the
// model: painting it from here put specks of paint on parts never touched. A texel is its patch's
// alone; the ring of unused texels around a patch (which filtering reads) goes to the one patch it touches.
function owners(L) {
  if (L.own) return L.own;
  const W = L.W, H = L.H, own = new Int32Array(W * H).fill(-1), island = islands();
  for (const m of P.meshes) if (m.L === L) for (let t = 0; t < m.count; t++) {
    const a = m.idx[t * 3] * 2, b = m.idx[t * 3 + 1] * 2, c = m.idx[t * 3 + 2] * 2, uv = m.uv, id = island[m.base + t];
    const fu = Math.floor(Math.min(uv[a], uv[b], uv[c])), fv = Math.floor(Math.min(uv[a + 1], uv[b + 1], uv[c + 1]));
    const x0 = (uv[a] - fu) * W, y0 = (uv[a + 1] - fv) * H, x1 = (uv[b] - fu) * W, y1 = (uv[b + 1] - fv) * H, x2 = (uv[c] - fu) * W, y2 = (uv[c + 1] - fv) * H;
    const area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    if (Math.abs(area) < 1e-9) continue;
    const xa = Math.max(0, Math.floor(Math.min(x0, x1, x2))), xb = Math.min(W - 1, Math.ceil(Math.max(x0, x1, x2)));
    const ya = Math.max(0, Math.floor(Math.min(y0, y1, y2))), yb = Math.min(H - 1, Math.ceil(Math.max(y0, y1, y2)));
    for (let y = ya; y <= yb; y++) for (let x = xa; x <= xb; x++) {
      const px = x + 0.5, py = y + 0.5;
      const w0 = ((x1 - px) * (y2 - py) - (y1 - py) * (x2 - px)) / area, w1 = ((x2 - px) * (y0 - py) - (y2 - py) * (x0 - px)) / area;
      if (w0 >= 0 && w1 >= 0 && w0 + w1 <= 1 && own[y * W + x] < 0) own[y * W + x] = id;
    }
  }
  const ring = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    if (own[i] >= 0) continue;
    // between two patches it is neither's: painted from one, the other would show a line of it
    let near = -1, shared = false;
    for (const n of [x > 0 ? i - 1 : -1, x < W - 1 ? i + 1 : -1, y > 0 ? i - W : -1, y < H - 1 ? i + W : -1]) {
      if (n < 0 || own[n] < 0) continue;
      if (near >= 0 && own[n] !== near) shared = true;
      near = own[n];
    }
    if (near >= 0 && !shared) ring.push(i, near);
  }
  for (let k = 0; k < ring.length; k += 2) own[ring[k]] = ring[k + 1];
  return L.own = own;
}

// ---------- one texel: what the stroke's tool makes of it at coverage m (1..255)
function shownTo(L, x, y, start) {
  // the colour a texel shows (paint over the picture); `start`: as it was when the stroke began
  const i = y * L.W + x, j = i * 4, o = start && L.mask[i] ? L.before : L.over, a = o[j + 3] / 255;
  T[0] = L.base[j] * (1 - a) + o[j] * a; T[1] = L.base[j + 1] * (1 - a) + o[j + 1] * a; T[2] = L.base[j + 2] * (1 - a) + o[j + 2] * a;
}
function texel(L, x, y, m) {
  const i = y * L.W + x;
  if (m <= L.mask[i]) return;  // a stroke covers a spot once: going over it again in one drag doesn't pile up
  const j = i * 4, o = L.over, was = L.before;
  if (!L.mask[i]) { was[j] = o[j]; was[j + 1] = o[j + 1]; was[j + 2] = o[j + 2]; was[j + 3] = o[j + 3]; }
  L.mask[i] = m;
  const a = (m / 255) * B.flow, wa = was[j + 3] / 255;
  if (B.tool === "erase") o[j + 3] = was[j + 3] * (1 - a);
  else {
    let r, g, b;
    if (B.tool === "soften") { r = SOFT.r; g = SOFT.g; b = SOFT.b; }  // worked out over the surface, see soften()
    else if (B.tool === "clone") {
      shownTo(L, Math.min(L.W - 1, Math.max(0, x + B.dx)), Math.min(L.H - 1, Math.max(0, y + B.dy)), true);
      r = T[0]; g = T[1]; b = T[2];
    } else {
      const ur = L.base[j] * (1 - wa) + was[j] * wa, ug = L.base[j + 1] * (1 - wa) + was[j + 1] * wa, ub = L.base[j + 2] * (1 - wa) + was[j + 2] * wa;
      if (B.mode === "multiply") { r = ur * B.r / 255; g = ug * B.g / 255; b = ub * B.b / 255; }
      else if (B.mode === "screen") { r = 255 - (255 - ur) * (255 - B.r) / 255; g = 255 - (255 - ug) * (255 - B.g) / 255; b = 255 - (255 - ub) * (255 - B.b) / 255; }
      else if (B.mode === "colour") {
        // the brush's hue on the picture's own light and dark: a recolour that keeps drawn lines and shading
        const d = 0.3 * (ur - B.r) + 0.59 * (ug - B.g) + 0.11 * (ub - B.b);
        r = B.r + d; g = B.g + d; b = B.b + d;
        const l = 0.3 * r + 0.59 * g + 0.11 * b, lo = Math.min(r, g, b), hi = Math.max(r, g, b);
        if (lo < 0) { const s = l / (l - lo); r = l + (r - l) * s; g = l + (g - l) * s; b = l + (b - l) * s; }
        else if (hi > 255) { const s = (255 - l) / (hi - l); r = l + (r - l) * s; g = l + (g - l) * s; b = l + (b - l) * s; }
      } else { r = B.r; g = B.g; b = B.b; }
    }
    const out = a + wa * (1 - a), keep = wa * (1 - a);
    if (out > 0) { o[j] = (r * a + was[j] * keep) / out; o[j + 1] = (g * a + was[j + 1] * keep) / out; o[j + 2] = (b * a + was[j + 2] * keep) / out; }
    o[j + 3] = out * 255;
  }
  const t = (y >> 6) * L.tx + (x >> 6);
  if (!L.touched[t]) { L.touched[t] = 1; L.touch.push(t); stroke.layers.add(L); }
  if (!L.drawn[t]) { L.drawn[t] = 1; L.draw.push(t); }
}

// ---------- a dab: every texel within the brush's reach of a point on the surface, by distance in 3D.
// Measured on the model, not in the texture: the brush keeps its size where the texture is stretched,
// and both sides of a UV seam are painted.
const D = { cx: 0, cy: 0, cz: 0, ex: 0, ey: 0, ez: 0, radius: 0 };
function reach(t) {
  const m = meshOf(t), q = (t - m.base) * 3, p = m.pos, a = m.idx[q] * 3, b = m.idx[q + 1] * 3, c = m.idx[q + 2] * 3, r = D.radius;
  return !(Math.min(p[a], p[b], p[c]) > D.cx + r || Math.max(p[a], p[b], p[c]) < D.cx - r || Math.min(p[a + 1], p[b + 1], p[c + 1]) > D.cy + r
    || Math.max(p[a + 1], p[b + 1], p[c + 1]) < D.cy - r || Math.min(p[a + 2], p[b + 2], p[c + 2]) > D.cz + r || Math.max(p[a + 2], p[b + 2], p[c + 2]) < D.cz - r);
}
// Whether the eye sees a triangle's middle, once a stroke (the camera stands still while painting).
function seen(t, eye, pass) {
  const mark = P.seen[t * 2 + pass], no = P.strokes * 2;
  if (mark >= no) return mark === no + 1;
  const m = meshOf(t), q = (t - m.base) * 3, p = m.pos, a = m.idx[q] * 3, b = m.idx[q + 1] * 3, c = m.idx[q + 2] * 3;
  const dx = (p[a] + p[b] + p[c]) / 3 - eye[0], dy = (p[a + 1] + p[b + 1] + p[c + 1]) / 3 - eye[1], dz = (p[a + 2] + p[b + 2] + p[c + 2]) / 3 - eye[2], far = Math.hypot(dx, dy, dz);
  const hit = cast(eye, [dx / far, dy / far, dz / far]);
  const yes = !hit || hit.tri === t || hit.dist > far - P.height * 0.002;
  P.seen[t * 2 + pass] = no + (yes ? 1 : 0);
  return yes;
}
function paintTri(t) {
  if (P.hidden && P.hidden[t]) return;
  const { cx, cy, cz, radius } = D, r2 = radius * radius, hard = B.hard;
  const m = meshOf(t), L = m.L, q = (t - m.base) * 3, ia = m.idx[q], ib = m.idx[q + 1], ic = m.idx[q + 2], p = m.pos;
  const ax = p[ia * 3], ay = p[ia * 3 + 1], az = p[ia * 3 + 2], bx = p[ib * 3], by = p[ib * 3 + 1], bz = p[ib * 3 + 2], kx = p[ic * 3], ky = p[ic * 3 + 1], kz = p[ic * 3 + 2];
  // facing away from the eye: the inside of a sleeve, the far side of a thin strand
  const nx = (by - ay) * (kz - az) - (bz - az) * (ky - ay), ny = (bz - az) * (kx - ax) - (bx - ax) * (kz - az), nz = (bx - ax) * (ky - ay) - (by - ay) * (kx - ax);
  const nl = Math.hypot(nx, ny, nz);
  if (!nl || (nx * D.ex + ny * D.ey + nz * D.ez) / nl < -0.15) return;
  const W = L.W, H = L.H, uv = m.uv, own = owners(L), mine = islands()[t];
  const u0 = uv[ia * 2], v0 = uv[ia * 2 + 1], u1 = uv[ib * 2], v1 = uv[ib * 2 + 1], u2 = uv[ic * 2], v2 = uv[ic * 2 + 1];
  const fu = Math.floor(Math.min(u0, u1, u2)), fv = Math.floor(Math.min(v0, v1, v2));
  const x0 = (u0 - fu) * W, y0 = (v0 - fv) * H, x1 = (u1 - fu) * W, y1 = (v1 - fv) * H, x2 = (u2 - fu) * W, y2 = (v2 - fv) * H;
  const area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
  if (Math.abs(area) < 1e-9) return;
  // how far outside each edge a texel may lie, as a share of the triangle
  const t0 = -PAD * Math.hypot(x2 - x1, y2 - y1) / Math.abs(area), t1 = -PAD * Math.hypot(x0 - x2, y0 - y2) / Math.abs(area), t2 = -PAD * Math.hypot(x1 - x0, y1 - y0) / Math.abs(area);
  const xa = Math.max(0, Math.floor(Math.min(x0, x1, x2) - PAD)), xb = Math.min(W - 1, Math.ceil(Math.max(x0, x1, x2) + PAD));
  const ya = Math.max(0, Math.floor(Math.min(y0, y1, y2) - PAD)), yb = Math.min(H - 1, Math.ceil(Math.max(y0, y1, y2) + PAD));
  for (let y = ya; y <= yb; y++) for (let x = xa; x <= xb; x++) {
    if (own[y * W + x] !== mine) continue;  // another patch's texel, or nobody's: not this triangle's to paint
    const px = x + 0.5, py = y + 0.5;
    let w0 = ((x1 - px) * (y2 - py) - (y1 - py) * (x2 - px)) / area, w1 = ((x2 - px) * (y0 - py) - (y2 - py) * (x0 - px)) / area, w2 = 1 - w0 - w1;
    if (w0 < t0 || w1 < t1 || w2 < t2) continue;
    if (w0 < 0) w0 = 0; if (w1 < 0) w1 = 0; if (w2 < 0) w2 = 0;
    const s = w0 + w1 + w2; w0 /= s; w1 /= s; w2 /= s;
    const dx = ax * w0 + bx * w1 + kx * w2 - cx, dy = ay * w0 + by * w1 + ky * w2 - cy, dz = az * w0 + bz * w1 + kz * w2 - cz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 >= r2) continue;
    const far = Math.sqrt(d2) / radius, edge = far <= hard ? 0 : (far - hard) / (1 - hard);
    const f = Math.round((1 - edge * edge * (3 - 2 * edge)) * 255);
    if (f <= 0) continue;
    if (B.tool === "soften") SOFT.todo.push(L, x, y, f, dx, dy, dz);  // blurred once the whole dab is gathered
    else texel(L, x, y, f);
  }
}
// `seeds`: triangles the brush is on. The dab then spreads from them over the surface, triangle to
// touching triangle, as far as the brush reaches: a sphere around the point also took in whatever
// lay behind it (painting a strand of hair recoloured the clothing under it). Without seeds (the
// surface option off) it is everything within reach that the eye sees.
// Returns the triangles it went over, for the next dab of the stroke to start from.
function dab(point, toEye, seeds, eye, pass) {
  SOFT.todo.length = 0;
  const over = spread(point, toEye, seeds, eye, pass);
  if (B.tool === "soften") soften();
  return over;
}
// Soften: every texel under the brush takes the average colour of the surface around it, measured
// on the model. Blurring along the texture stopped at each patch of it, and a generated texture is
// patches of a triangle or two: a patch darker than its neighbours stayed a hard-edged square
// however often it was gone over.
const SOFT = { r: 0, g: 0, b: 0, todo: [], cells: new Float32Array(32768 * 7), used: [] };
function soften() {
  const todo = SOFT.todo, cells = SOFT.cells, used = SOFT.used, reach = D.radius * 0.3, s2 = 2 * (reach * 0.8) ** 2;
  const cellOf = (dx, dy, dz) => (Math.floor(dx / reach) + 16) + (Math.floor(dy / reach) + 16) * 32 + (Math.floor(dz / reach) + 16) * 1024;
  for (const c of used) cells.fill(0, c * 7, c * 7 + 7);
  used.length = 0;
  // the colours under the brush, gathered in small cubes of space
  for (let k = 0; k < todo.length; k += 7) {
    const c = cellOf(todo[k + 4], todo[k + 5], todo[k + 6]) * 7;
    shownTo(todo[k], todo[k + 1], todo[k + 2], false);
    if (!cells[c + 3]) used.push(c / 7);
    cells[c] += T[0]; cells[c + 1] += T[1]; cells[c + 2] += T[2]; cells[c + 3] += 1;
    cells[c + 4] += todo[k + 4]; cells[c + 5] += todo[k + 5]; cells[c + 6] += todo[k + 6];
  }
  for (let k = 0; k < todo.length; k += 7) {
    const dx = todo[k + 4], dy = todo[k + 5], dz = todo[k + 6], home = cellOf(dx, dy, dz);
    let r = 0, g = 0, b = 0, w = 0;
    for (let a = -1; a <= 1; a++) for (let e = -1; e <= 1; e++) for (let i = -1; i <= 1; i++) {
      const c = (home + a + e * 32 + i * 1024) * 7, n = cells[c + 3];
      if (!n) continue;
      // each cube counts by how many texels it holds and how near their middle is
      const far = (cells[c + 4] / n - dx) ** 2 + (cells[c + 5] / n - dy) ** 2 + (cells[c + 6] / n - dz) ** 2, weight = Math.exp(-far / s2);
      r += cells[c] * weight; g += cells[c + 1] * weight; b += cells[c + 2] * weight; w += weight * n;
    }
    SOFT.r = r / w; SOFT.g = g / w; SOFT.b = b / w;
    texel(todo[k], todo[k + 1], todo[k + 2], todo[k + 3]);
  }
  todo.length = 0;
}
function spread(point, toEye, seeds, eye, pass) {
  Object.assign(D, { cx: point[0], cy: point[1], cz: point[2], ex: toEye[0], ey: toEye[1], ez: toEye[2], radius: B.size / HEIGHT_CM * P.height });
  const stampNo = ++P.dabs, over = [];
  if (seeds) {
    const { at, first, list } = links(), queue = [];
    for (const t of seeds) if (P.stamp[t] !== stampNo) { P.stamp[t] = stampNo; queue.push(t); }
    for (let k = 0; k < queue.length; k++) {
      const t = queue[k];
      if (!reach(t)) continue;
      over.push(t); paintTri(t);
      for (let c = 0; c < 3; c++) for (let n = first[at[t * 3 + c]]; n < first[at[t * 3 + c] + 1]; n++) {
        const other = list[n];
        if (P.stamp[other] !== stampNo) { P.stamp[other] = stampNo; queue.push(other); }
      }
    }
    return over;
  }
  const { cell, cells } = grid(), r = D.radius, lo = [0, 1, 2].map((a) => Math.floor((point[a] - r - P.min[a]) / cell)), hi = [0, 1, 2].map((a) => Math.floor((point[a] + r - P.min[a]) / cell));
  for (let gx = lo[0]; gx <= hi[0]; gx++) for (let gy = lo[1]; gy <= hi[1]; gy++) for (let gz = lo[2]; gz <= hi[2]; gz++) {
    const found = cells.get(cellKey(gx, gy, gz));
    if (found) for (const t of found) {
      if (P.stamp[t] === stampNo) continue;
      P.stamp[t] = stampNo;
      if (reach(t) && seen(t, eye, pass)) paintTri(t);
    }
  }
  return null;
}
// One dab of the stroke, and its twin on the other side of the body when asked for.
function dabs(point, hit) {
  const surface = stroke.surface;
  stroke.front = dab(point, hit.toEye, surface ? [hit.tri, ...(stroke.front || [])] : null, hit.eye, 0);
  if (!B.mirror) return;
  const flip = (v) => [2 * P.middle - v[0], v[1], v[2]], eye = flip(hit.eye), at = flip(point), toEye = [-hit.toEye[0], hit.toEye[1], hit.toEye[2]];
  // what the mirrored brush is on: the surface the mirrored eye sees there
  const twin = surface ? cast(eye, [-toEye[0], -toEye[1], -toEye[2]]) : null;
  if (surface && !twin && !stroke.twinFront?.length) return;
  stroke.twinFront = dab(at, toEye, surface ? [...(twin ? [twin.tri] : []), ...(stroke.twinFront || [])] : null, eye, 1);
}

// ---------- fill: the part of one texture patch joined to the clicked texel in a similar colour
function fill(L, sx, sy) {
  const own = owners(L), W = L.W, H = L.H, limit = +$("paint-tolerance").value, patch = own[sy * W + sx];
  shownTo(L, sx, sy, false);
  const sr = T[0], sg = T[1], sb = T[2];
  const seen = new Uint8Array(W * H), stack = [sy * W + sx], found = [];
  const alike = (i) => { shownTo(L, i % W, (i / W) | 0, false); return Math.max(Math.abs(T[0] - sr), Math.abs(T[1] - sg), Math.abs(T[2] - sb)) <= limit; };
  seen[stack[0]] = 1;
  while (stack.length) {
    const i = stack.pop(), x = i % W, y = (i / W) | 0;
    found.push(i);
    for (const n of [x > 0 ? i - 1 : -1, x < W - 1 ? i + 1 : -1, y > 0 ? i - W : -1, y < H - 1 ? i + W : -1])
      if (n >= 0 && !seen[n] && own[n] === patch && alike(n)) { seen[n] = 1; stack.push(n); }
  }
  for (const i of found) texel(L, i % W, (i / W) | 0, 255);
  return found.length;
}

// ---------- strokes, undo, and showing what changed
function begin(as = tool) { readBrush(as); P.strokes++; stroke = { layers: new Set(), last: null, surface: $("paint-piece").checked, front: null, twinFront: null }; }
// each tile's rows inside the picture: [x0, y0, width, height]
const tileBox = (L, t) => { const x0 = (t % L.tx) * TILE, y0 = Math.floor(t / L.tx) * TILE; return [x0, y0, Math.min(TILE, L.W - x0), Math.min(TILE, L.H - y0)]; };
function end() {
  if (!stroke) return;
  const mine = P, records = [], stroke_layers = [...stroke.layers];
  for (const L of stroke.layers) {
    for (const t of L.touch) {
      const [x0, y0, w, h] = tileBox(L, t), was = new Uint8ClampedArray(w * h * 4);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = (y0 + y) * L.W + x0 + x, from = L.mask[i] ? L.before : L.over, j = i * 4, k = (y * w + x) * 4;
        was[k] = from[j]; was[k + 1] = from[j + 1]; was[k + 2] = from[j + 2]; was[k + 3] = from[j + 3];
        L.mask[i] = 0;
      }
      records.push({ L, t, was });
      L.touched[t] = 0;
    }
    L.touch = []; L.changed = true;
  }
  stroke = null;
  flushTiles(); preview();
  if (!records.length) return;
  // each call swaps the stroke's tiles with what they were: the same call undoes, then redoes
  const swap = async () => {
    if (P !== mine) throw new Error("that paint is no longer open");
    for (const record of records) {
      const { L, t, was } = record, [x0, y0, w, h] = tileBox(L, t), now = new Uint8ClampedArray(w * h * 4);
      for (let y = 0; y < h; y++) {
        const at = ((y0 + y) * L.W + x0) * 4;
        now.set(L.over.subarray(at, at + w * 4), y * w * 4);
        L.over.set(was.subarray(y * w * 4, (y + 1) * w * 4), at);
      }
      record.was = now;
      if (!L.drawn[t]) { L.drawn[t] = 1; L.draw.push(t); }
      L.changed = L.drafted = true;
    }
    flushTiles(); preview(); showState(); draftSoon();
    return swap;
  };
  pushUndo("paint", swap);
  for (const L of stroke_layers) L.drafted = true;
  showState(); draftSoon();
}
// paint over picture (or over grey, to show only the paint), for a stretch of texels
function compose(L, x0, y0, w, h, d, stride) {
  const only = $("paint-only").checked, base = L.base, over = L.over;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const j = ((y0 + y) * L.W + x0 + x) * 4, k = (y * stride + x) * 4, a = over[j + 3] / 255;
    if (only) { const g = 60 + (0.3 * base[j] + 0.59 * base[j + 1] + 0.11 * base[j + 2]) * 0.25; d[k] = g * (1 - a) + over[j] * a; d[k + 1] = g * (1 - a) + over[j + 1] * a; d[k + 2] = g * (1 - a) + over[j + 2] * a; }
    else { d[k] = base[j] * (1 - a) + over[j] * a; d[k + 1] = base[j + 1] * (1 - a) + over[j + 1] * a; d[k + 2] = base[j + 2] * (1 - a) + over[j + 2] * a; }
    d[k + 3] = 255;
  }
}
// The tiles a stroke changed, into the canvas and straight into the texture on the graphics card:
// a few small squares, not the picture again. Sent with the graphics calls themselves: three's own
// copy asks the card for five settings before every square, and each question waits for an answer.
const tiles = new Map();  // "w,h" -> an ImageData that size, reused
function flushTiles() {
  if (!P) return false;
  const gl = view.renderer.getContext();
  let drew = false;
  for (const L of P.layers.values()) {
    if (!L.draw.length) continue;
    const handle = view.renderer.properties.get(L.texture).__webglTexture;
    if (!handle) { flushAll(L); for (const t of L.draw) L.drawn[t] = 0; L.draw = []; drew = true; continue; }  // not on the card yet
    view.renderer.state.bindTexture(gl.TEXTURE_2D, handle);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, L.texture.flipY);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    for (const t of L.draw) {
      const [x0, y0, w, h] = tileBox(L, t), id = `${w},${h}`;
      let image = tiles.get(id); if (!image) tiles.set(id, image = new ImageData(w, h));
      compose(L, x0, y0, w, h, image.data, w);
      L.ctx.putImageData(image, x0, y0);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, x0, L.texture.flipY ? L.H - y0 - h : y0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, image.data);
      L.drawn[t] = 0;
    }
    L.draw = []; drew = true;
  }
  return drew;
}
// the whole picture again: when it is first shown, and when "show only the paint" changes
function flushAll(L) {
  const band = 256;
  for (let y0 = 0; y0 < L.H; y0 += band) {
    const h = Math.min(band, L.H - y0), image = new ImageData(L.W, h);
    compose(L, 0, y0, L.W, h, image.data, L.W);
    L.ctx.putImageData(image, 0, y0);
  }
  L.texture.needsUpdate = true;
}
let previewed = 0;
function tick() {
  requestAnimationFrame(tick);
  if (pending) { const e = pending; pending = null; move(e); }
  // the small copy of the picture beside the view is the whole picture scaled down: a few times a second is plenty
  if (flushTiles() && performance.now() - previewed > 300) preview();
}
requestAnimationFrame(tick);
function preview() {
  const L = P?.layers.get(+$("paint-picture").value), c = $("paint-preview");
  if (!L) return;
  previewed = performance.now();
  const h = Math.round(512 * L.H / L.W);
  if (c.width !== 512 || c.height !== h) { c.width = 512; c.height = h; }
  c.getContext("2d").drawImage(L.canvas, 0, 0, c.width, c.height);
}
function showLit() {
  if (!P) return;
  for (const m of P.meshes) m.mesh.material = $("paint-lit").checked ? m.mesh.userData.lit : m.mesh.userData.flat;
}
function showState() {
  const unsaved = P && [...P.layers.values()].some((L) => L.changed);
  $("paint-state").textContent = !P ? "" : unsaved ? "not saved" : [...P.layers.values()].some(hasPaint) ? "saved" : "no paint";
  $("paint-state").className = `chip ${unsaved ? "warn" : "good"}`;
  $("paint-save").disabled = $("paint-build").disabled = !unsaved;
  $("paint-clear").disabled = !P;
}
const hasPaint = (L) => { for (let i = 3; i < L.over.length; i += 4) if (L.over[i]) return true; return false; };

// ---------- the pointer: left on the model paints, anything else is the camera's
const caster = new THREE.Raycaster();
function hitAt(e) {
  if (!P) return null;
  const r = view.renderer.domElement.getBoundingClientRect();
  caster.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), view.camera);
  const o = caster.ray.origin, d = caster.ray.direction, hit = cast([o.x, o.y, o.z], [d.x, d.y, d.z]);
  if (!hit) return null;
  const m = hit.m, L = m.L, q = (hit.tri - m.base) * 3, ia = m.idx[q], ib = m.idx[q + 1], ic = m.idx[q + 2], w = 1 - hit.u - hit.v;
  let u = m.uv[ia * 2] * w + m.uv[ib * 2] * hit.u + m.uv[ic * 2] * hit.v, v = m.uv[ia * 2 + 1] * w + m.uv[ib * 2 + 1] * hit.u + m.uv[ic * 2 + 1] * hit.v;
  u -= Math.floor(u); v -= Math.floor(v);
  const p = m.pos, a = new THREE.Vector3(p[ia * 3], p[ia * 3 + 1], p[ia * 3 + 2]);
  const normal = new THREE.Vector3(p[ib * 3], p[ib * 3 + 1], p[ib * 3 + 2]).sub(a).cross(new THREE.Vector3(p[ic * 3], p[ic * 3 + 1], p[ic * 3 + 2]).sub(a)).normalize();
  if (normal.dot(d) > 0) normal.negate();  // toward the eye, whichever way the triangle winds
  return { point: [o.x + d.x * hit.dist, o.y + d.y * hit.dist, o.z + d.z * hit.dist], toEye: [-d.x, -d.y, -d.z], eye: [o.x, o.y, o.z], L,
    x: Math.min(L.W - 1, Math.floor(u * L.W)), y: Math.min(L.H - 1, Math.floor(v * L.H)), tri: hit.tri, normal };
}
function pickColour(hit) {
  shownTo(hit.L, hit.x, hit.y, false);
  $("paint-colour").value = hexOf(T);
  remember();
}
function step(hit) {
  // dabs a third of the brush apart, in a straight line from the last one (half for the blur,
  // whose dabs cost several times a brush's and overlap smoothly anyway)
  const from = stroke.last, gap = B.size / HEIGHT_CM * P.height / (B.tool === "soften" ? 2 : 3);
  const far = from ? Math.hypot(hit.point[0] - from[0], hit.point[1] - from[1], hit.point[2] - from[2]) : 0, n = Math.max(1, Math.ceil(far / gap));
  for (let k = 1; k <= n; k++) dabs(from ? hit.point.map((v, a) => from[a] + (v - from[a]) * k / n) : hit.point, hit);
  stroke.last = hit.point;
}
function move(e) {
  const hit = hitAt(e);
  ring.visible = !!hit && tool !== "pick" && tool !== "fill" && tool !== "isolate";
  if (hit) {
    ring.position.fromArray(hit.point).addScaledVector(hit.normal, P.height * 0.0015);
    ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), hit.normal);
    ring.scale.setScalar(+$("paint-size").value / HEIGHT_CM * P.height);
  }
  if (stroke && hit && e.buttons & 1) { step(hit); lastScreen = [e.clientX, e.clientY]; }
}
$("paint-view").addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || !P || e.target.closest(".cams, .tools")) return;
  const hit = hitAt(e);
  if (!hit) return;  // off the model: the drag turns it
  view.orbit.enabled = false;
  try { e.target.setPointerCapture(e.pointerId); } catch {}  // the stroke goes on past the view's edge
  if (e.altKey || tool === "pick") { pickColour(hit); return; }
  if (tool === "isolate") {
    const id = pieces()[hit.tri];
    isolate(e.shiftKey && P.shown ? new Set([...P.shown, id]) : new Set([id]));
    if (!e.shiftKey) setTool(toolBefore);
    return;
  }
  if (tool === "clone" && e.ctrlKey) { cloneFrom = hit; say("paint-msg", "Copying from there. Now paint where it should go."); return; }
  if (tool === "clone" && !cloneFrom) { say("paint-msg", "Ctrl+click the spot to copy from first.", true); return; }
  if (tool === "clone" && cloneFrom.L !== hit.L) { say("paint-msg", "That spot is on another picture than the one copied from: clone works within one.", true); return; }
  begin();
  if (tool === "clone") { B.dx = cloneFrom.x - hit.x; B.dy = cloneFrom.y - hit.y; }
  if (tool === "fill") { const n = fill(hit.L, hit.x, hit.y); end(); say("paint-msg", `Filled ${n.toLocaleString()} texels.`); return; }
  if (e.shiftKey && lastScreen) {
    // a straight line from where the last stroke ended, followed over the surface
    const [fx, fy] = lastScreen, n = Math.max(2, Math.ceil(Math.hypot(e.clientX - fx, e.clientY - fy) / 6));
    for (let k = 0; k <= n; k++) { const at = hitAt({ clientX: fx + (e.clientX - fx) * k / n, clientY: fy + (e.clientY - fy) * k / n }); if (at) step(at); else stroke.last = stroke.front = stroke.twinFront = null; }
  } else step(hit);
  lastScreen = [e.clientX, e.clientY];
  remember();
}, true);
// at most one move a frame's worth of time: the mouse reports faster than the screen draws
let moved = 0;
$("paint-view").addEventListener("pointermove", (e) => {
  if (performance.now() - moved < 12) { pending = e; return; }
  moved = performance.now(); pending = null; move(e);
});
addEventListener("pointerup", () => { if (!view.orbit.enabled) { view.orbit.enabled = true; end(); } });
$("paint-view").addEventListener("pointerleave", () => { ring.visible = false; });
$("paint-view").addEventListener("contextmenu", (e) => e.preventDefault());

// ---------- the view over the whole window
function expand(on) {
  $("paint").classList.toggle("full", on);
  $("paint-expand").textContent = on ? "Back (Esc)" : "Expand";
  $("paint-expand").classList.toggle("on", on);
  view.resize();
}
$("paint-expand").onclick = () => expand(!$("paint").classList.contains("full"));
$("paint-all").onclick = () => isolate(null);

// ---------- the tools
let toolBefore = "brush";
function setTool(name) {
  if (name === "isolate" && tool !== "isolate") toolBefore = tool;
  tool = name;
  document.querySelectorAll("#paint-tools button").forEach((b) => b.classList.toggle("on", b.dataset.tool === name));
  $("paint-tool-help").textContent = TOOLS[name];
  $("paint-fill-row").hidden = name !== "fill";
  $("paint-mode-row").hidden = !["brush", "fill"].includes(name);
}
$("paint-tools").innerHTML = Object.keys(TOOLS).map((t) => `<button data-tool="${t}">${TOOLS[t].split(":")[0]}</button>`).join("");
$("paint-tools").onclick = (e) => { if (e.target.dataset.tool) setTool(e.target.dataset.tool); };
$("paint-mode").innerHTML = Object.entries(MODES).map(([k, t]) => `<option value="${k}">${t}</option>`).join("");
const label = () => {
  $("paint-size-text").textContent = `${(+$("paint-size").value).toFixed(1)} cm`;
  $("paint-hard-text").textContent = +$("paint-hard").value > 0.8 ? "hard edge" : +$("paint-hard").value < 0.3 ? "soft edge" : "";
  $("paint-flow-text").textContent = `${Math.round(+$("paint-flow").value * 100)}%`;
};
for (const id of ["paint-size", "paint-hard", "paint-flow"]) $(id).oninput = label;
$("paint-lit").onchange = showLit;
$("paint-only").onchange = () => { if (P) { for (const L of P.layers.values()) flushAll(L); preview(); } };
$("paint-picture").onchange = preview;
$("paint-undo").onclick = () => undo();
$("paint-redo").onclick = () => redo();
// the colours used lately, a click away
let recent = [];
function remember() {
  const c = $("paint-colour").value;
  recent = [c, ...recent.filter((x) => x !== c)].slice(0, 10);
  $("paint-recent").innerHTML = recent.map((x) => `<button style="background:${x}" data-colour="${x}" title="${x}" aria-label="colour ${x}"></button>`).join("");
}
$("paint-recent").onclick = (e) => { if (e.target.dataset.colour) $("paint-colour").value = e.target.dataset.colour; };
$("paint-colour").onchange = remember;
addEventListener("keydown", (e) => {
  if (!$("paint").classList.contains("on") || ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName) && document.activeElement.type !== "range") return;
  const keys = { b: "brush", e: "erase", s: "soften", c: "clone", g: "fill", i: "pick" };
  if (!e.ctrlKey && !e.metaKey && keys[e.key.toLowerCase()]) setTool(keys[e.key.toLowerCase()]);
  if (!e.ctrlKey && !e.metaKey && e.key.toLowerCase() === "f") expand(!$("paint").classList.contains("full"));
  if (e.key === "Escape" && $("paint").classList.contains("full") && !$("ask").open && $("lightbox").hidden) expand(false);
  if (!e.ctrlKey && !e.metaKey && e.key.toLowerCase() === "h") { if (P?.shown) isolate(null); else setTool("isolate"); }
  if (e.key === "[" || e.key === "]") { $("paint-size").value = Math.max(0.2, +$("paint-size").value * (e.key === "[" ? 0.8 : 1.25)); label(); }
});

// ---------- drafts: unsaved paint kept on disk once the brush has rested, so a closed window loses nothing
let draftTimer = null;
function draftSoon() { clearTimeout(draftTimer); draftTimer = setTimeout(draft, 6000); }
async function draft() {
  if (!P || stroke) return draftSoon();
  const todo = [...P.layers.values()].filter((L) => L.drafted);
  if (!todo.length) return;
  for (const L of todo) L.drafted = false;
  try { await call("paint_draft", { images: Object.fromEntries(todo.map((L) => [L.file, pngOf(L)])) }); }
  catch { for (const L of todo) L.drafted = true; }  // tried again after the next stroke
}

// ---------- an old face patch: hair and the face behind it share one picture there
$("paint-remake").onclick = async () => {
  if ([...(P?.layers.values() || [])].some((L) => L.changed) && !await save()) return;
  const state = await job("face_patch", {});
  if (state && !state.error && !state.cancelled) { key = null; resetPaint(); toast("Face patch remade: the hair over the face now has its own picture."); }
};

// ---------- saving: each picture's paint as a PNG, kept with the character
function pngOf(L) {
  if (!hasPaint(L)) return null;
  const c = Object.assign(document.createElement("canvas"), { width: L.W, height: L.H });
  c.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(L.over), L.W, L.H), 0, 0);
  return c.toDataURL("image/png");
}
async function save() {
  if (!P) return false;
  const changed = [...P.layers.values()].filter((L) => L.changed);
  say("paint-msg", "Saving…");
  try {
    await call("paint_save", { images: Object.fromEntries(changed.map((L) => [L.file, pngOf(L)])) });
    for (const L of changed) L.changed = L.drafted = false;
    clearTimeout(draftTimer);
    await refreshManifest(); await refreshStatus(); app.renderAll();
    say("paint-msg", "Saved. It goes onto the avatar at the next build (from “build”).");
    showState();
    return true;
  } catch (e) { say("paint-msg", e.message, true); return false; }
}
$("paint-save").onclick = save;
$("paint-build").onclick = async () => {
  if (!await save()) return;
  const built = S.status?.steps.some((s) => s.step === "build" && ["done", "stale", "failed"].includes(s.status));
  app.go("build"); job("run", built ? { from: "build" } : {});
};
$("paint-clear").onclick = async () => {
  if (!P || !await ask("Clear all paint", { text: "Takes every bit of paint off this character: the generated texture shows again. Ctrl+Z brings it back until you leave the page.", ok: "Clear", danger: true })) return;
  // through a stroke of the eraser, so one undo brings it all back
  begin("erase"); B.flow = 1;
  for (const L of P.layers.values()) for (let i = 0; i < L.W * L.H; i++) if (L.over[i * 4 + 3]) texel(L, i % L.W, (i / L.W) | 0, 255);
  end();
};

setTool("brush"); label(); remember();
