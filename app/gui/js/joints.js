// Joints: the skeleton's joints over a see-through model, each draggable; a move overrides detection.
import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { $, S, app, call, say, refreshManifest } from "./core.js";
import { makeView, toThree, toBlender } from "./views.js";
import { pushUndo, saved } from "./ui.js";
import { onShow } from "./nav.js";

export const jointsView = makeView($("joints-view"));
const markers = new THREE.Group(); jointsView.scene.add(markers);
const gizmo = new TransformControls(jointsView.camera, jointsView.renderer.domElement);
jointsView.scene.add(gizmo.getHelper ? gizmo.getHelper() : gizmo);
gizmo.addEventListener("dragging-changed", (e) => {
  jointsView.orbit.enabled = !e.value;
  if (!e.value && gizmo.object) saveJoint(gizmo.object.userData.key, toBlender(gizmo.object.position));
});

// the listed model, see-through
export function setGhost(scene) {
  jointsView.clear();
  if (!scene) return;
  const ghost = scene.clone(true);
  ghost.traverse((o) => { if (o.isMesh) o.material = new THREE.MeshStandardMaterial({ color: 0x8899aa, transparent: true, opacity: 0.25, depthWrite: false }); });
  jointsView.model = ghost; jointsView.scene.add(ghost);
  jointsView.frame();
}
onShow.joints = () => showJoints();

// the joint being moved: kept across saves, which rebuild the markers (the gizmo used to drop it)
let pickedJoint = null, jointsDown = null;
function pickJoint(key) {
  pickedJoint = key;
  document.querySelectorAll("#joint-rows tr").forEach((t) => t.classList.toggle("sel", t.dataset.key === key));
  const dot = key && markers.children.find((m) => m.userData.key === key);
  if (dot) gizmo.attach(dot); else gizmo.detach();
  if (key) $("joint-rows").querySelector(`tr[data-key="${key}"]`)?.scrollIntoView({ block: "nearest" });
}
jointsView.renderer.domElement.addEventListener("pointerdown", (e) => { jointsDown = { x: e.clientX, y: e.clientY }; });
jointsView.renderer.domElement.addEventListener("click", (e) => {
  if (gizmo.dragging || (jointsDown && Math.hypot(e.clientX - jointsDown.x, e.clientY - jointsDown.y) > 4)) return;
  const hit = jointsView.pick(e, markers.children);
  if (hit) pickJoint(hit.object.userData.key);
});
jointsView.renderer.domElement.addEventListener("pointermove", (e) => {
  if (gizmo.dragging) return;
  jointsView.renderer.domElement.style.cursor = jointsView.pick(e, markers.children) ? "pointer" : "";
});
addEventListener("keydown", (e) => { if (e.key === "Escape" && pickedJoint) pickJoint(null); });

const detected = () => S.listing?.joints || {};
function jointPositions() { return { ...detected(), ...(S.manifest?.joints || {}) }; }
const cm = (a, b) => 100 * Math.hypot(...a.map((v, i) => v - b[i]));

export function showJoints() {
  markers.clear(); gizmo.detach();
  const joints = jointPositions(), own = S.manifest?.joints || {}, body = $("joint-rows");
  if (!S.listing) { body.innerHTML = `<tr><td colspan="3" class="msg">List the pieces first (Parts).</td></tr>`; return; }
  $("joints-note").textContent = S.manifest?.rig ? "This character uses Tripo's rig: the build takes its joints from the rig. "
    + "Only move a joint here to overrule the rig; what's shown is the app's own detection, not the rig's." : "";
  for (const [key, p] of Object.entries(joints)) {
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.02), new THREE.MeshBasicMaterial({ color: key in own ? 0xffb347 : 0x7cc4ff, depthTest: false }));
    dot.renderOrder = 2; dot.position.copy(toThree(p)); dot.userData.key = key; markers.add(dot);
  }
  body.innerHTML = Object.entries(joints).map(([key, p]) => {
    const moved = key in own, from = detected()[key];
    return `<tr data-key="${key}"><td>${key.replaceAll("_", " ")}</td>
      <td class="${moved ? "warn" : "msg"}" title="${p.map((v) => v.toFixed(3)).join(", ")} m">${moved ? (from ? `moved ${cm(p, from).toFixed(1)} cm` : "placed by you") : "detected"}</td>
      <td>${moved ? `<button class="act alt small" data-reset>Reset</button>` : ""}</td></tr>`;
  }).join("");
  body.querySelectorAll("tr").forEach((tr) => {
    tr.onclick = (e) => { if (e.target.tagName !== "BUTTON") pickJoint(tr.dataset.key); };
    tr.querySelector("[data-reset]")?.addEventListener("click", () => saveJoint(tr.dataset.key, null));
  });
  if (pickedJoint) pickJoint(pickedJoint);
}

async function saveJoint(key, xyz, { quiet = false } = {}) {
  const name = key.replaceAll("_", " "), before = S.manifest.joints?.[key] || null;
  try {
    await call("joint", xyz ? { key, xyz } : { key, clear: true });
    await refreshManifest(); showJoints(); app.renderNav(); app.refreshStatus?.();
    if (quiet) return;
    pushUndo(`move of ${name}`, () => saveJoint(key, before, { quiet: true }));
    say("joints-msg", "");
    saved(xyz ? `Saved ${name}.` : `Reset ${name} to detection.`);
  } catch (err) { say("joints-msg", `Not saved: ${err.message}`, true); }
}
$("mirror").onclick = async () => {
  const joints = jointPositions(), own = S.manifest.joints || {}, changed = [];
  for (const key of Object.keys(own)) if (key.startsWith("left_")) {
    const right = key.replace("left_", "right_");
    changed.push([right, own[right] || null]);
    await call("joint", { key: right, xyz: [-joints[key][0], joints[key][1], joints[key][2]] });
  }
  await refreshManifest(); showJoints(); app.renderNav();
  if (!changed.length) return say("joints-msg", "No left joint is moved yet: move one, then mirror it.");
  pushUndo("mirror", async () => { for (const [key, xyz] of changed) await call("joint", xyz ? { key, xyz } : { key, clear: true }); await refreshManifest(); showJoints(); });
  saved(`Mirrored ${changed.length} joint${changed.length > 1 ? "s" : ""} to the right side.`);
};
