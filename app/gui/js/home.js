// Characters: one card per character, and the new-character form and survey on Model.
import { $, S, app, call, esc, say } from "./core.js";
import { toast } from "./ui.js";
import { job } from "./jobs.js";

export function renderHome() {
  const cards = S.projects.map((p) => {
    const chips = [p.approved ? ["marked good", "good"] : null,
      p.built ? (p.failed?.length ? [`built without ${p.failed.join(", ").replaceAll("_", " ")}`, "warn"] : ["built", "info"])
        : p.surveyed ? ["started", ""] : ["new", ""],
      p.stale?.length ? ["changes not built", "warn"] : null].filter(Boolean);
    const when = p.finished ? new Date(p.finished).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "";
    const where = p.manifest.replaceAll("\\", "/").split("/").slice(-3, -1).join("/");
    return `<button class="char ${p.manifest === S.manifestPath ? "open" : ""}" data-manifest="${esc(p.manifest)}">
      <div class="pic" ${p.picture ? `style="background-image:url('${esc(p.picture)}')"` : ""}>${p.picture ? "" : esc(p.name[0] || "?")}</div>
      <div class="body"><span class="name">${esc(p.name)}</span>
        <div class="chips">${chips.map(([t, k]) => `<span class="chip ${k}">${esc(t)}</span>`).join("")}</div>
        <span class="when">${when ? `last run ${esc(when)} · ` : ""}${esc(where)}</span></div></button>`;
  });
  $("cards").innerHTML = `<button class="char new" data-new><div class="pic">+</div><div class="body"><span class="name">New character</span>
    <span class="when">from a Tripo model (.glb, .obj, .fbx)</span></div></button>` + cards.join("");
  $("cards").querySelectorAll("[data-manifest]").forEach((b) => b.onclick = async () => {
    await app.openProject(b.dataset.manifest);
    const p = S.projects.find((q) => q.manifest === b.dataset.manifest);
    app.go(p?.surveyed ? "build" : "model");
  });
  $("cards").querySelector("[data-new]").onclick = async () => { await app.openProject(null); app.go("model"); };
}

// ---------- Model: a new character's form, or the open one's survey and rig
$("create").onclick = async () => {
  try {
    const made = await call("init", { model: $("new-model").value.trim(), name: $("new-name").value.trim(),
      author: $("new-author").value.trim(), face: $("new-face").value.trim(), reference: $("new-reference").value.trim(),
      rig: $("new-rig").value.trim() });
    toast(`Made ${made.manifest}`);
    say("create-msg", "");
    await app.loadProjects(made.manifest);
  } catch (e) { say("create-msg", e.message, true); }
};
$("survey-run").onclick = () => job("run", { only: "survey" });
document.querySelectorAll("[data-browse]").forEach((b) => b.onclick = async () => {
  try { const { path } = await call("browse", { kind: b.dataset.browse }); if (path) $(b.dataset.for).value = path; }
  catch (e) { toast(e.message, "bad"); }
});
$("rig-save").onclick = async () => {
  try {
    await call("set_path", { key: "rig", value: $("rig-path").value.trim() });
    toast("Rig saved: rebuild from “build” for it to count.");
    await app.openProject(S.manifestPath);
  } catch (e) { say("rig-msg", e.message, true); }
};

export async function renderSurvey() {
  // one card at a time: a new character's form, or the open one's survey and rig. Both at once
  // read as two ways to start, and a rigged FBX looked like a second character to create
  const m = S.manifest;
  $("model-new").hidden = !!m; $("model-open").hidden = !m;
  $("rig-path").value = m?.rig || "";
  $("model-files").innerHTML = m ? `Model in use: <code>${esc(m.source)}</code>`
    + (m.reference ? `<br>Original, before the face patch: <code>${esc(m.reference)}</code>` : "") : "";
  const entry = S.status?.steps.find((s) => s.step === "survey");
  $("survey-state").textContent = entry ? entry.status : "";
  $("survey-state").className = `chip ${entry?.status?.split(" ")[0] || ""}`;
  $("survey").innerHTML = "";
  if (!entry?.json) return;
  const { url } = await call("url", { path: entry.json });
  if (!url) return;
  const s = await (await fetch(url)).json();
  const o = s.options || {};
  $("survey").innerHTML = `<table>
    <tr><td>Size</td><td>${s.size.map((v) => v.toFixed(2)).join(" × ")} m, ${s.triangles.toLocaleString()} triangles, ${s.pieces} pieces</td></tr>
    <tr><td>Pose</td><td>${esc(s.pose.kind)}, arms ${Math.round(s.pose.arm_drop_degrees)}° below horizontal</td></tr>
    <tr><td>Face</td><td>${s.parts.eye_inserts ? "modelled eyes" : "painted face"}${s.face_patch ? ", face patch" : ""}</td></tr>
    <tr><td>Head</td><td>${o.rigid_head ? "solid head or hat: rigid head suggested" : "hair"}</td></tr>
    <tr><td>Cloth</td><td>${Math.round(s.cloth.share * 100)}% below the hips hangs free${o.cloth_physics ? ": cloth physics suggested" : ""}</td></tr>
    </table>${notesFor(s.notes || []).length ? `<ul class="plain msg">${notesFor(s.notes || []).map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}`;
}

// The survey's notes in the app's words: with a Tripo rig the joints come from it, so notes about
// joints placed by proportion don't apply; and no command-line flags.
function notesFor(notes) {
  const rigged = !!S.manifest?.rig;
  const out = notes.filter((n) => !(rigged && /joint|placed from typical proportions|Joints page|legs are joined|legs only stand apart/i.test(n)))
    .map((n) => n.replace(/,? ?or pass --joints/g, " on the Joints page").replace(/: building with --cloth/g, ": cloth physics is on (Options)")
      .replace(/--free-mouth/g, "a free mouth (Face)").replace(/--mouth-only/g, "mouth-only face tracking"));
  if (rigged) out.unshift("the joints and body weights come from Tripo's rig");
  return out;
}
