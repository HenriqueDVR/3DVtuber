// Face: the face patch, the twelve marked points (in the embedded marker) and how the face moves.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { $, S, app, call, say, refreshManifest, refreshStatus } from "./core.js";
import { makeView } from "./views.js";
import { saved, toast } from "./ui.js";
import { job } from "./jobs.js";
import { onShow } from "./nav.js";

// The marker (viewer/setup.html, embedded) marks pixels on the face patch texture. Here the same
// pixels are found on the 3D model through the patch's UVs, so a point on the wrong feature shows.
const faceView = makeView($("face-view"));
const faceDots = new THREE.Group(); faceView.scene.add(faceDots);
const FEATURE_COLOUR = { right_eye: 0x4ad7ff, left_eye: 0x7dff6b, mouth: 0xff6bb5 };
let facePatch = null, faceImage = null, facePoints = null, markerReady = false, patchFit = null;
// (a new address each time: a reloaded page was handed the frame it had before, an older marker)
$("marker").src = `/viewer/setup.html?embed=1&t=${Date.now()}`;

addEventListener("message", async (e) => {
  if (e.origin !== location.origin) return;
  const { type, data } = e.data || {};
  if (type === "face-marker-ready") { markerReady = true; sendFaceImage(); fitMarker(); }
  if (type === "face-points-draft") {
    facePoints = data; showFacePoints(); showPointsState();
    // saved as soon as all twelve are placed: a separate save inside the marker was easy to miss,
    // and "Save face options" next to it saved only the options
    // Only on the Face page: the marker also answers every time it is handed the points again (each
    // time the character is read: after any job, on any page), and "Face points saved." came up on Movement.
    clearTimeout(pointsTimer);
    if (e.data.complete && $("face").classList.contains("on")) pointsTimer = setTimeout(() => savePoints().then((wrote) => wrote === "saved" && toast("Face points saved.")).catch((err) => say("face-msg", err.message, true)), 600);
  }
});
let pointsTimer = null, pointsSaved = null;
// Eyes covered by a visor, glasses or a mask aren't marked: the mouth's four points are all there is.
const hiddenEyes = () => $("eyes").value === "hidden";
const features = () => hiddenEyes() ? ["mouth"] : ["right_eye", "left_eye", "mouth"];
const pointCount = () => 4 * features().length;
const placed = (p) => features().reduce((n, k) => n + Object.keys(p?.[k] || {}).length, 0);
const marked = () => ({ ...facePoints, ...(hiddenEyes() ? { right_eye: {}, left_eye: {} } : {}) });
// The points alone, in one order: what is kept (face.json) and what the marker hands back are the
// same points written differently (key order, an eye left out or empty), and read as a change.
const written = (p) => JSON.stringify(["right_eye", "left_eye", "mouth"].map((k) =>
  Object.entries(p?.[k] || {}).sort(([a], [b]) => a < b ? -1 : 1).map(([n, [x, y]]) => [n, Math.round(x * 10), Math.round(y * 10)])));
async function savePoints() {
  if (placed(facePoints) < pointCount()) return false;
  if (pointsSaved === written(marked())) return true;
  await call("face_save", { points: marked() });
  pointsSaved = written(marked());
  await refreshManifest(); await refreshStatus(); app.renderAll(); showPointsState();
  return "saved";
}
function showPointsState() {
  const n = placed(facePoints), done = n === pointCount() && pointsSaved === written(marked());
  S.facePlaced = n;
  $("points-state").textContent = done ? "saved" : n === pointCount() ? "saving…" : `${n} of ${pointCount()} placed`;
  $("points-state").className = `chip ${done ? "done" : n ? "warn" : ""}`;
  app.renderNav();
}
function sendFaceImage() {
  if (markerReady && faceImage) $("marker").contentWindow.postMessage({ type: "face-image", ...faceImage, points: facePoints, features: features() }, location.origin);
  sendMouth();
}
// The page comes up: the marker measures its box and fits the picture to it. It was handed the picture
// while the page was hidden (no size to fit to), and waited for the browser to tell it of a new size,
// which a page coming out of hiding doesn't always do: an empty box, "the face patch didn't load".
// (again after the arcade look's page has slid in)
const fitMarker = () => markerReady && $("marker").contentWindow.postMessage({ type: "face-fit" }, location.origin);
onShow.face = () => { for (const ms of [0, 350, 1000]) setTimeout(fitMarker, ms); };
// The open mouth's outline on the picture, as big as the size asks for: seen here, not after a build.
function sendMouth() {
  if (!markerReady) return;
  const mouth = $("mouth-fix").checked ? { scale: +$("mouth-scale").value || 1, free: $("mouth-free").checked } : null;
  $("marker").contentWindow.postMessage({ type: "face-mouth", mouth }, location.origin);
}

const PATCH_NEEDED = "Needed once per character. It renders the head from the front and maps that picture onto the face as its own sharp texture; step 2 marks the points on it. Two Blender runs, about two minutes. The model is copied, never changed.";
// (one load counts, the last: a character picked while the one before was still loading was read
// twice at once, the two crossing each other's patch and picture)
let loading = 0;
export async function loadFace() {
  const mine = ++loading;
  faceView.clear(); faceDots.clear(); facePatch = faceImage = patchFit = null; $("face-checks").innerHTML = "";
  if (!S.manifestPath) return;
  const face = await call("face");
  if (mine !== loading) return;
  facePoints = face.points; pointsSaved = face.points ? written(face.points) : null; showPointsState();
  if (!face.model) {
    $("patch-make").textContent = "Make a face patch";
    $("patch-state").textContent = PATCH_NEEDED + " (It isn't a .glb, so it shows here once the patch is made.)";
    return;
  }
  const gltf = await new GLTFLoader().loadAsync(face.model);
  if (mine !== loading) return;
  faceView.clear();
  faceView.model = gltf.scene; faceView.scene.add(gltf.scene); gltf.scene.updateMatrixWorld(true);
  gltf.scene.traverse((o) => { if (o.isMesh && o.material?.name === "face_patch") facePatch = o; });
  const box = new THREE.Box3().setFromObject(gltf.scene), size = box.getSize(new THREE.Vector3());
  const head = new THREE.Vector3((box.min.x + box.max.x) / 2, box.max.y - size.y * 0.12, (box.min.z + box.max.z) / 2);
  faceView.orbit.target.copy(head); faceView.camera.position.set(head.x, head.y, head.z + size.y * 0.55); faceView.orbit.update();
  $("patch-make").textContent = facePatch ? "Remake the face patch" : "Make a face patch";
  if (!facePatch) { $("patch-state").textContent = PATCH_NEEDED; return; }
  $("patch-state").textContent = "Done. The head was rendered from the front and that picture mapped onto the face, so step 2 marks a sharp picture instead of the blurry texture. Remake it only if the picture below looks wrong (face cut off, off-centre).";
  const image = facePatch.material.map?.image;
  if (!image) return;
  const canvas = document.createElement("canvas");
  canvas.width = image.width; canvas.height = image.height;
  canvas.getContext("2d").drawImage(image, 0, 0);
  const blob = await new Promise((done) => canvas.toBlob(done, "image/png"));
  if (mine !== loading) return;
  faceImage = { src: URL.createObjectURL(blob), name: "face_patch.png" };
  sendFaceImage(); showFacePoints();
}

// A pixel of the patch image on the model: the patch triangle whose UVs hold it, interpolated.
function onPatch(px, py, size) {
  const geo = facePatch.geometry, pos = geo.attributes.position, uv = geo.attributes.uv, index = geo.index;
  const u = px / size[0], v = py / size[1];  // glTF UVs count down from the top, like image pixels
  const count = index ? index.count : pos.count;
  const a = new THREE.Vector2(), b = new THREE.Vector2(), c = new THREE.Vector2();
  for (let t = 0; t < count; t += 3) {
    const [i, j, k] = [0, 1, 2].map((o) => index ? index.getX(t + o) : t + o);
    a.fromBufferAttribute(uv, i); b.fromBufferAttribute(uv, j); c.fromBufferAttribute(uv, k);
    const d = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    if (Math.abs(d) < 1e-12) continue;
    const w0 = ((b.y - c.y) * (u - c.x) + (c.x - b.x) * (v - c.y)) / d;
    const w1 = ((c.y - a.y) * (u - c.x) + (a.x - c.x) * (v - c.y)) / d;
    const w2 = 1 - w0 - w1;
    if (w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6) continue;
    const p = new THREE.Vector3().fromBufferAttribute(pos, i).multiplyScalar(w0)
      .add(new THREE.Vector3().fromBufferAttribute(pos, j).multiplyScalar(w1))
      .add(new THREE.Vector3().fromBufferAttribute(pos, k).multiplyScalar(w2));
    return p.applyMatrix4(facePatch.matrixWorld);
  }
  return fromFront(u, v);
}

// Past the patch's own triangles, the way the build does it (face_points._pixel_to_world): the
// patch's UVs are a straight front mapping, so u and v are fitted as lines in x and height, and
// the point is where a ray from the front at that x and height meets the model.
function fromFront(u, v) {
  if (!patchFit) {
    const pos = facePatch.geometry.attributes.position, uv = facePatch.geometry.attributes.uv, p = new THREE.Vector3();
    const xs = [], ys = [], us = [], vs = [];
    for (let i = 0; i < pos.count; i++) {
      p.fromBufferAttribute(pos, i).applyMatrix4(facePatch.matrixWorld);
      xs.push(p.x); ys.push(p.y); us.push(uv.getX(i)); vs.push(uv.getY(i));
    }
    patchFit = { u: line(xs, us), v: line(ys, vs) };
  }
  const x = (u - patchFit.u[1]) / patchFit.u[0], y = (v - patchFit.v[1]) / patchFit.v[0];
  const box = new THREE.Box3().setFromObject(faceView.model);
  const ray = new THREE.Raycaster(new THREE.Vector3(x, y, box.max.z + 1), new THREE.Vector3(0, 0, -1));
  return ray.intersectObject(faceView.model, true)[0]?.point || null;
}
function line(xs, ys) {
  const n = xs.length, mx = xs.reduce((a, b) => a + b) / n, my = ys.reduce((a, b) => a + b) / n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
  const slope = sxy / sxx;
  return [slope, my - slope * mx];
}

function showFacePoints() {
  faceDots.clear();
  const checks = [];
  if (!facePatch || !facePoints?.size) { $("face-checks").innerHTML = facePatch ? `<p class="msg">No points yet.</p>` : ""; return; }
  const at = {}, missing = [];
  const radius = 0.004 * new THREE.Box3().setFromObject(faceView.model).getSize(new THREE.Vector3()).y / 1.6;
  for (const key of features()) {
    for (const [name, [px, py]] of Object.entries(facePoints[key] || {})) {
      const p = onPatch(px, py, facePoints.size);
      if (!p) { missing.push(`${key.replace("_", " ")} ${name}`); continue; }
      (at[key] ||= []).push(p);
      const dot = new THREE.Mesh(new THREE.SphereGeometry(radius), new THREE.MeshBasicMaterial({ color: FEATURE_COLOUR[key], depthTest: false }));
      dot.renderOrder = 3; dot.position.copy(p); faceDots.add(dot);
    }
  }
  if (missing.length) checks.push(["bad", `not on the model: ${missing.join(", ")}`]);
  const centre = (k) => at[k]?.length === 4 ? at[k].reduce((s, p) => s.add(p), new THREE.Vector3()).multiplyScalar(0.25) : null;
  const r = centre("right_eye"), l = centre("left_eye"), m = centre("mouth");
  if (r && l) {
    const eyeWidth = (at.right_eye[0].distanceTo(at.right_eye[1]) + at.left_eye[0].distanceTo(at.left_eye[1])) / 2;
    checks.push([Math.abs(r.y - l.y) < eyeWidth * 0.5 ? "done" : "bad", `eyes level to ${(Math.abs(r.y - l.y) / eyeWidth).toFixed(2)} eye widths`]);
    if (m) {
      const off = Math.abs(m.x - (r.x + l.x) / 2) / eyeWidth;
      checks.push([m.y < Math.min(r.y, l.y) ? "done" : "bad", m.y < Math.min(r.y, l.y) ? "mouth under the eyes" : "mouth above the eyes"]);
      checks.push([off < 0.5 ? "done" : "bad", `mouth ${off.toFixed(2)} eye widths off the middle`]);
    }
  }
  $("face-checks").innerHTML = `<ul class="plain">${checks.map(([k, t]) => `<li class="${k}">${t}</li>`).join("")}</ul>`;
}
// No automatic mouth-versus-nose check: how far down the face the mouth sits can't tell them
// apart (model A's nose is 0.66 eye widths under her eyes, a chibi's mouth 0.69), and a search
// for the drawn stroke took model A's nose shading or her collar. The 3D preview shows it.

$("patch-make").onclick = () => job("face_patch", {});
// A first guess from the face detector (app/face_guess.py): the marker takes the points as if
// they had been clicked, and they save as hand-placed ones do. Never instead of looking: the
// guess is shown on the picture and on the head, to be dragged where it is off.
$("points-guess").onclick = async () => {
  if (!facePatch) { say("guess-msg", "Make the face patch first (step 1).", true); return; }
  if (S.setup?.checks.some((c) => c.name === "Auto face marking" && c.state !== "ok")) {
    // said here, with the way to it, not found out as a failed run
    $("guess-msg").innerHTML = `The face detector isn't installed. <a href="#setup">Setup</a> installs it with one button; or mark the twelve points by hand below.`;
    $("guess-msg").classList.add("bad");
    return;
  }
  if (placed(facePoints) && !confirm("Replace the points already placed with the detector's guess?")) return;
  say("guess-msg", "Looking for the face…");
  // a job like a build: the detector must not run beside a Blender (memory)
  const state = await job("face_guess", {});
  if (!state?.result?.points) { say("guess-msg", state?.error ? state.error.split("\n")[0] : state?.cancelled ? "Stopped." : "", !!state?.error); return; }
  try {
    const { sure, ...kept } = state.result.points;
    facePoints = kept; sendFaceImage(); showFacePoints(); showPointsState();
    await savePoints();
    say("guess-msg", `Placed (the detector is ${Math.round(sure * 100)}% sure of its least certain point). Check each one and drag any that is off.`);
  } catch (e) { say("guess-msg", e.message, true); }
};
export function renderFace() {
  if (!S.manifest) return;
  const o = S.manifest.options;
  $("eyes").value = o.eyes ?? "";
  $("mouth-tracking").value = String(o.face_tracking !== false);
  const fix = o.mouth_fix;
  $("mouth-fix").checked = !!fix; $("mouth-free").checked = !!fix?.free; $("mouth-scale").value = fix?.scale ?? 1;
  $("mouth-free").disabled = $("mouth-scale").disabled = !fix;
  showEyeChoice(); showFeatures(); sendMouth();
}
// The choices, kept as each is made (as the Options page keeps its own). Kept only by the Save
// buttons, a choice was put back to the saved one whenever the page was drawn again (after the
// points saved themselves, after any job): "eyes are covered" kept turning back into automatic,
// which asked for eye points again and held the build.
async function saveChoices() {
  // (what else a manifest says of the mouth, its rest and lift, stays)
  const fix = $("mouth-fix").checked ? { ...S.manifest.options.mouth_fix, free: $("mouth-free").checked, scale: +$("mouth-scale").value || 1 } : null;
  const now = { eyes: $("eyes").value || null, face_tracking: $("mouth-tracking").value === "true", mouth_fix: fix };
  const changed = Object.entries(now).filter(([key, value]) => JSON.stringify(S.manifest.options[key] ?? null) !== JSON.stringify(value));
  for (const [key, value] of changed) await call("set", { key, value });
  if (changed.length) { await refreshManifest(); await refreshStatus(); app.renderAll(); }
  return changed.length;
}
const keepChoices = () => saveChoices().then((n) => { say("face-msg", ""); if (n) saved("Saved. Rebuild for the face to change."); }).catch((e) => say("face-msg", e.message, true));
// Saves the points and the choices; true when the points went in too.
async function saveFace() {
  try {
    await saveChoices();
    if (await savePoints()) { say("face-msg", ""); return true; }
    say("face-msg", `${placed(facePoints)} of ${pointCount()} points are placed. Place them all in step 2.`, true);
  } catch (e) { say("face-msg", e.message, true); }
  return false;
}
// The face points shape the head, eyes and mouth from the rig step on: a first build runs
// everything, a later one from "build".
$("face-build").onclick = async () => {
  if (!await saveFace()) return;
  const built = S.status?.steps.some((s) => s.step === "build" && ["done", "stale", "failed"].includes(s.status));
  app.go("build");
  job("run", built ? { from: "build" } : {});
};
const EYE_CHOICE = {
  "": () => `Picks for you: a moving iris if the eyes are modelled as their own pieces, sliding patches if they're painted on the face${S.manifest?.face ? "" : " (and no eye movement until the points are marked)"}.`,
  iris: () => "The coloured part of the eye (iris and pupil) moves inside the eye white. For models whose eyes are separate pieces.",
  slide: () => "The whole painted eye slides a little across the face. For eyes painted onto the face texture.",
  none: () => "The eyes stay still; blinking still works.",
  hidden: () => "For eyes behind a visor, glasses or a mask: step 2 asks for the mouth's four points only, and the avatar gets no blink and no eye movement.",
};
const showEyeChoice = () => { $("eyes-choice").textContent = EYE_CHOICE[$("eyes").value](); };
// which points step 2 asks for follows the choice at once
function showFeatures() {
  showPointsState(); showFacePoints();
  if (markerReady) $("marker").contentWindow.postMessage({ type: "face-features", features: features() }, location.origin);
}
// (the eyes' choice first, then the points it leaves to keep: with the eyes covered the mouth's four are all)
$("eyes").onchange = async () => { showEyeChoice(); await keepChoices(); showFeatures(); savePoints().catch((e) => say("face-msg", e.message, true)); };
$("mouth-fix").onchange = () => { $("mouth-free").disabled = $("mouth-scale").disabled = !$("mouth-fix").checked; sendMouth(); keepChoices(); };
for (const id of ["mouth-tracking", "mouth-free", "mouth-scale"]) $(id).onchange = () => { sendMouth(); keepChoices(); };
// (the outline follows the number as it is typed; it is saved when the field is left)
$("mouth-scale").oninput = sendMouth;
