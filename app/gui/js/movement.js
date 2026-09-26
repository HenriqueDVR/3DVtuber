// Movement: the pose test (pieces left behind or torn) and the physics test (springs running).
import * as THREE from "three";
import { $, S, app, call, esc, hex, say, roleName, ROLE_INFO, ROLE_STAGE } from "./core.js";
import { makeView } from "./views.js";
import { job } from "./jobs.js";

const moveView = makeView($("movement-view"));
let moveMeshes = [];
const VERDICT_COLOUR = { behind: 0xff5a5a, torn: 0xffa040, fine: 0x8a8a8a };

// `gltf` is the listed model Parts loaded: cloned here, coloured by what the pose test found
export async function loadMovement(gltf) {
  moveView.clear(); moveMeshes = [];
  S.movement = S.manifestPath ? (await call("movement_result")).result : null;
  $("movement-next").textContent = "";
  if (gltf) {
    const scene = gltf.scene.clone(true);
    moveView.model = scene; moveView.scene.add(scene);
    scene.traverse((o) => { if (o.isMesh) { moveMeshes.push(o); o.geometry = o.geometry.clone(); o.material = o.material.clone(); Object.assign(o.material, { vertexColors: true, map: null, transparent: false, depthWrite: true }); } });
    moveView.frame();
  }
  renderMovement();
}
function verdictOf(label) {
  const row = S.movement?.pieces.find((r) => r.label === label);
  if (!row) return "fine";
  return row.findings.some((f) => f.kind === "stays behind") ? "behind" : "torn";
}
function renderMovement(selectedLabel) {
  for (const mesh of moveMeshes) {
    const attr = mesh.geometry.attributes._piece;
    if (!attr) continue;
    const colours = new Float32Array(attr.count * 3), c = new THREE.Color();
    for (let i = 0; i < attr.count; i++) {
      const label = Math.round(attr.getX(i));
      c.setHex(VERDICT_COLOUR[verdictOf(label)]);
      if (label === selectedLabel) c.lerp(new THREE.Color(0xffffff), 0.5);
      colours.set(c.toArray(), i * 3);
    }
    mesh.geometry.setAttribute("color", new THREE.BufferAttribute(colours, 3));
    mesh.material.needsUpdate = true;
  }
  const body = $("movement-rows"), movement = S.movement;
  if (!movement) { body.innerHTML = `<tr><td colspan="5" class="msg">Not run yet: press “Run the pose test”.${moveMeshes.length ? "" : " List the pieces (Parts) to see them here."}</td></tr>`; return; }
  if (!movement.pieces.length) { body.innerHTML = `<tr><td colspan="5" class="done">Every piece moves with the body in every pose.</td></tr>`; return; }
  body.innerHTML = movement.pieces.map((row) => {
    const verdict = verdictOf(row.label);
    // one line per kind of problem, with the poses it shows in
    const behind = row.findings.filter((f) => f.kind === "stays behind"), torn = row.findings.filter((f) => f.kind === "torn");
    const what = [behind.length ? `stays where it is while the body moves away (${behind.map((f) => f.pose).join(", ")})` : "",
      torn.length ? `stretches up to ${Math.max(...torn.map((f) => f.stretch_cm))} cm between the part that moves and the part that stays (${torn.map((f) => f.pose).join(", ")})` : ""]
      .filter(Boolean).join("<br>");
    const has = row.role && row.role === row.suggested;
    const side = row.findings.find((f) => f.bone)?.bone.startsWith("Left") ? "left" : "right";
    const fix = !row.suggested ? esc(row.why.replace(/: gather it for an agent \(Stuck\?\)/, ": if it looks wrong in the Review renders, ask for help on the Stuck? page"))
      : `<b>${roleName(row.suggested)}</b>${row.suggested === "follow_bone" ? ` (${side} upper arm)` : ""}<span class="sub">${ROLE_INFO[row.suggested]?.[1] ?? ""}</span>`;
    return `<tr data-label="${row.label}"><td><span class="swatch" style="background:${hex(VERDICT_COLOUR[verdict])}"></span></td>
      <td>${esc(row.name || `piece ${row.label}`)}<span class="sub">${row.count.toLocaleString()} vertices${row.role ? `, now: ${roleName(row.role)}` : ""}</span></td>
      <td>${what}</td><td>${fix}</td>
      <td>${row.suggested && !has ? `<button class="act alt small" data-use>Apply this fix</button>` : has ? `<span class="msg">applied: rebuild, then test again</span>` : ""}</td></tr>`;
  }).join("");
  body.querySelectorAll("tr[data-label]").forEach((tr) => {
    const row = movement.pieces.find((r) => r.label === +tr.dataset.label);
    tr.onclick = (e) => { if (e.target.tagName !== "BUTTON") renderMovement(row.label); };
    tr.querySelector("[data-use]")?.addEventListener("click", async () => {
      const side = row.findings.find((f) => f.bone)?.bone.startsWith("Left") ? "Left" : "Right";
      const patch = { role: row.suggested, ...(row.suggested === "follow_bone" ? { bone: `${side}UpperArm` } : {}) };
      try {
        const listed = S.listing?.pieces.find((r) => r.label === row.label);
        if (listed) await app.setRole(listed, patch);
        else await call("piece", { name: row.name || `${row.suggested}_${row.label}`, label: row.label, ...patch });
        row.role = row.suggested;
        await app.piecesChanged(); renderMovement(row.label);
        $("movement-next").textContent = `Applied: ${roleName(row.suggested)}. Rebuild from “${ROLE_STAGE(row.suggested)}”, then run the pose test again.`;
      } catch (err) { say("movement-msg", err.message, true); }
    });
  });
}
$("movement-run").onclick = () => job("movement", {});
$("springs-run").onclick = () => job("springs", {});
$("move-viewer-open").onclick = () => { app.go("review"); app.openViewer(); };

export async function loadSprings() {
  const r = S.manifestPath ? (await call("springs_result")).result : null;
  S.springs = r;
  if (!r) { $("springs-out").textContent = "not run yet"; return; }
  const lines = [...r.sagging.map((s) => `${esc(s.chain)} sags ${s.sag_cm} cm lower than modelled just standing still (its springs are too soft for its weight)`),
    ...r.coming_away.map((c) => `a piece of ${c.count} vertices (#${c.label}) is left ${c.gap_cm} cm behind the hair or cloth it sits on when that swings: give it a role on Parts (a small bit sitting on hair or cloth: “rides another piece”)`),
    r.clipping ? `swinging hair or cloth sinks up to ${r.clipping.deepest_cm} cm into the body` : ""].filter(Boolean);
  $("springs-out").innerHTML = lines.length ? `<ul class="plain">${lines.map((l) => `<li>${l}</li>`).join("")}</ul>` + (r.sagging.length || r.clipping ? `<span class="help">Sagging and sinking aren't adjustable in the app yet; if they look wrong in the viewer, say so on the Stuck? page.</span>` : "")
    : `<span class="done">${r.chains} spring chains: nothing droops or comes away</span>`;
}
