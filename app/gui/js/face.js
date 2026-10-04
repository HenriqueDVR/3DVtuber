// Face: the face patch, the twelve marked points (in the embedded marker) and how the face moves.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { $, S, app, call, say, refreshManifest, refreshStatus } from "./core.js";
import { makeView } from "./views.js";
import { saved, toast } from "./ui.js";
import { job } from "./jobs.js";

// The marker (viewer/setup.html, embedded) marks pixels on the face patch texture. Here the same
// pixels are found on the 3D model through the patch's UVs, so a point on the wrong feature shows.
const faceView = makeView($("face-view"));
const faceDots = new THREE.Group(); faceView.scene.add(faceDots);
const FEATURE_COLOUR = { right_eye: 0x4ad7ff, left_eye: 0x7dff6b, mouth: 0xff6bb5 };
let facePatch = null, faceImage = null, facePoints = null, markerReady = false, patchFit = null;
$("marker").src = "/viewer/setup.html?embed=1";

addEventListener("message", async (e) => {
  if (e.origin !== location.origin) return;
  const { type, data } = e.data || {};
  if (type === "face-marker-ready") { markerReady = true; sendFaceImage(); }
  if (type === "face-points-draft") {
    facePoints = data; showFacePoints(); showPointsState();
    // saved as soon as all twelve are placed: a separate save inside the marker was easy to miss,
    // and "Save face options" next to it saved only the options
    clearTimeout(pointsTimer);
    if (e.data.complete) pointsTimer = setTimeout(() => savePoints().then((wrote) => wrote === "saved" && toast("Face points saved.")).catch((err) => say("face-msg", err.message, true)), 600);
  }
});
let pointsTimer = null, pointsSaved = null;
const POINT_COUNT = 12;
const placed = (p) => ["right_eye", "left_eye", "mouth"].reduce((n, k) => n + Object.keys(p?.[k] || {}).length, 0);
async function savePoints() {
  if (placed(facePoints) < POINT_COUNT) return false;
  if (pointsSaved === JSON.stringify(facePoints)) return true;
  await call("face_save", { points: facePoints });
  pointsSaved = JSON.stringify(facePoints);
  await refreshManifest(); await refreshStatus(); app.renderAll(); showPointsState();
  return "saved";
}
function showPointsState() {
  const n = placed(facePoints), done = n === POINT_COUNT && pointsSaved === JSON.stringify(facePoints);
  S.facePlaced = n;
  $("points-state").textContent = done ? "saved" : n === POINT_COUNT ? "saving…" : `${n} of ${POINT_COUNT} placed`;
  $("points-state").className = `chip ${done ? "done" : n ? "warn" : ""}`;
  app.renderNav();
}
function sendFaceImage() {
  if (markerReady && faceImage) $("marker").contentWindow.postMessage({ type: "face-image", ...faceImage, points: facePoints }, location.origin);
}

const PATCH_NEEDED = "Needed once per character. It renders the head from the front and maps that picture onto the face as its own sharp texture; step 2 marks the points on it. Two Blender runs, about two minutes. The model is copied, never changed.";
export async function loadFace() {
  faceView.clear(); faceDots.clear(); facePatch = faceImage = patchFit = null; $("face-checks").innerHTML = "";
  if (!S.manifestPath) return;
  const face = await call("face");
  facePoints = face.points; pointsSaved = face.points ? JSON.stringify(face.points) : null; showPointsState();
  if (!face.model) {
    $("patch-make").textContent = "Make a face patch";
    $("patch-state").textContent = PATCH_NEEDED + " (It isn't a .glb, so it shows here once the patch is made.)";
    return;
  }
  const gltf = await new GLTFLoader().loadAsync(face.model);
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
  for (const key of ["right_eye", "left_eye", "mouth"]) {
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
  if (placed(facePoints) && !confirm("Replace the points already placed with the detector's guess?")) return;
  $("points-guess").disabled = true; say("guess-msg", "Looking for the face…");
  try {
    const { points } = await call("face_guess", {});
    const { sure, ...kept } = points;
    facePoints = kept; sendFaceImage(); showFacePoints(); showPointsState();
    await savePoints();
    say("guess-msg", `Placed (the detector is ${Math.round(sure * 100)}% sure of its least certain point). Check each one and drag any that is off.`);
  } catch (e) { say("guess-msg", e.message, true); }
  $("points-guess").disabled = false;
};
export function renderFace() {
  if (!S.manifest) return;
  const o = S.manifest.options;
  $("eyes").value = o.eyes ?? "";
  const fix = o.mouth_fix;
  $("mouth-fix").checked = !!fix; $("mouth-free").checked = !!fix?.free; $("mouth-scale").value = fix?.scale ?? 1;
  $("mouth-free").disabled = $("mouth-scale").disabled = !fix;
  showEyeChoice();
}
// Saves the points and the choices; true when the points went in too.
async function saveFace() {
  try {
    const points = await savePoints();
    await call("set", { key: "eyes", value: $("eyes").value || null });
    await call("set", { key: "mouth_fix", value: $("mouth-fix").checked ? { free: $("mouth-free").checked, scale: +$("mouth-scale").value } : null });
    await refreshManifest(); await refreshStatus(); app.renderAll();
    if (points) { say("face-msg", ""); return true; }
    say("face-msg", `Saved the choices, but not the points: ${placed(facePoints)} of ${POINT_COUNT} are placed. Place them all in step 2.`, true);
  } catch (e) { say("face-msg", e.message, true); }
  return false;
}
$("face-apply").onclick = async () => { if (await saveFace()) saved("Saved the points and the choices. Rebuild for the face to change."); };
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
  iris: () => "The coloured part of the eye (iris and pupil) moves inside the eye white. For models whose eyes are separate pieces, like model A.",
  slide: () => "The whole painted eye slides a little across the face. For eyes painted onto the face texture, like a mascot.",
  none: () => "The eyes stay still; blinking still works.",
};
const showEyeChoice = () => { $("eyes-choice").textContent = EYE_CHOICE[$("eyes").value](); };
$("eyes").onchange = showEyeChoice;
$("mouth-fix").onchange = () => { $("mouth-free").disabled = $("mouth-scale").disabled = !$("mouth-fix").checked; };
