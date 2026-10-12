// Movement: the pose test (pieces left behind or torn) and the physics test (springs running).
import * as THREE from "three";
import { $, S, app, call, esc, hex, movesFine, say, roleName, ROLE_INFO, ROLE_STAGE } from "./core.js";
import { makeView } from "./views.js";
import { keyable, toast } from "./ui.js";
import { job } from "./jobs.js";

const moveView = makeView($("movement-view"));
let moveMeshes = [];
const VERDICT_COLOUR = { behind: 0xff5a5a, torn: 0xffa040, expected: 0x5aa0ff, fine: 0x8a8a8a };

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
  if (movesFine(row)) return "expected";
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
  $("movement-summary").innerHTML = summary(movement);
  if (!movement) { body.innerHTML = `<tr><td colspan="5" class="msg">Not run yet: press “Run the pose test”.${moveMeshes.length ? "" : " List the pieces (Parts) to see them here."}</td></tr>`; return; }
  if (!movement.pieces.length) { body.innerHTML = `<tr><td colspan="5" class="done">Every piece moves with the body in every pose.</td></tr>`; return; }
  // what needs a fix first; what was looked at and is right as it is after it
  body.innerHTML = [...movement.pieces].sort((a, b) => movesFine(a) - movesFine(b)).map((row) => {
    const verdict = verdictOf(row.label);
    if (verdict === "expected") return `<tr data-label="${row.label}"><td><span class="swatch" style="background:${hex(VERDICT_COLOUR.expected)}"></span></td>
      <td>${esc(row.name || `piece ${row.label}`)}<span class="sub">${row.count.toLocaleString()} vertices${row.role ? `, now: ${roleName(row.role)}` : ""}</span></td>
      <td>stays where it is while the arms move (${[...new Set(row.findings.map((f) => f.pose))].join(", ")})</td>
      <td><b class="done">Nothing to fix</b><span class="sub">${esc(row.why[0].toUpperCase() + row.why.slice(1))}. It is listed so you know the test looked at it.</span></td><td></td></tr>`;
    // one line per kind of problem, with the poses it shows in
    const behind = row.findings.filter((f) => f.kind === "stays behind"), torn = row.findings.filter((f) => f.kind === "torn");
    const carried = row.findings.filter((f) => f.kind === "goes with the head");
    const taken = row.findings.filter((f) => f.kind === "taken by a hand");
    const what = [behind.length ? `stays where it is while the body moves away (${behind.map((f) => f.pose).join(", ")})` : "",
      carried.length ? `turns with the head while the body under it stays (${carried.map((f) => f.pose).join(", ")})` : "",
      taken.length ? `a hand hanging against it takes a patch of it along, up to ${Math.max(...taken.map((f) => f.moved_cm))} cm (${taken.map((f) => f.pose).join(", ")})` : "",
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
    keyable(tr);
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
// What a test found, in a line: on the page above its results, and said when the run ends (the page
// only changed some small print, and a finished test read as one that had told nothing).
function summary(movement) {
  if (!movement) return "";
  const fine = movement.pieces.filter(movesFine).length, wrong = movement.pieces.length - fine;
  const pieces = (n) => `${n} piece${n === 1 ? "" : "s"}`;
  return `<b class="${wrong ? "warn" : "done"}">${wrong ? `${pieces(wrong)} to fix: each row below says what was seen and what fixed it before.` : fine ? "Nothing to fix." : "Nothing to fix: every piece moves with the body in every pose."}</b>`
    + (fine ? ` ${pieces(fine)} looked at and right as ${fine === 1 ? "it is" : "they are"} (blue).` : "");
}
function springSummary(r) {
  const found = r.sagging.length + r.coming_away.length + (r.clipping ? 1 : 0);
  return found ? `${found} thing${found === 1 ? "" : "s"} to look at: listed under the physics test.` : `Nothing wrong: ${r.chains} spring chains ran, nothing droops, comes away or sinks into the body.`;
}
const text = (html) => Object.assign(document.createElement("i"), { innerHTML: html }).textContent;
$("movement-run").onclick = async () => {
  const state = await job("movement", {});
  if (state && !state.error && !state.cancelled && S.movement) toast(`Pose test: ${text(summary(S.movement))}`, S.movement.pieces.some((row) => !movesFine(row)) ? "info" : "good", { ms: 9000 });
};
$("springs-run").onclick = async () => {
  const state = await job("springs", {});
  if (state && !state.error && !state.cancelled && S.springs) toast(`Physics test: ${springSummary(S.springs)}`, springSummary(S.springs).startsWith("Nothing") ? "good" : "info", { ms: 9000 });
};

export async function loadSprings() {
  const r = S.manifestPath ? (await call("springs_result")).result : null;
  S.springs = r;
  if (!r) { $("springs-out").innerHTML = "<b>Result:</b> not run yet"; return; }
  const lines = [...r.sagging.map((s) => `${esc(s.chain)} sags ${s.sag_cm} cm lower than modelled just standing still (its springs are too soft for its weight)`),
    ...r.coming_away.map((c) => `a piece of ${c.count} vertices (#${c.label}) is left ${c.gap_cm} cm behind the hair or cloth it sits on when that swings: give it a role on Parts (a small bit sitting on hair or cloth: “rides another piece”)`),
    r.clipping ? `swinging hair or cloth sinks up to ${r.clipping.deepest_cm} cm into the body` : ""].filter(Boolean);
  $("springs-out").innerHTML = `<b>Result:</b> ` + (lines.length ? `<ul class="plain">${lines.map((l) => `<li>${l}</li>`).join("")}</ul>` + (r.sagging.length || r.clipping ? `<span class="help">Sagging and sinking aren't adjustable in the app yet; if they look wrong in the viewer, say so on the Stuck? page.</span>` : "")
    : `<span class="done">✓ ${springSummary(r)}</span>`);
}
