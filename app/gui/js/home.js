// Characters: one card per character, and the new-character form and survey on Model.
import { $, S, app, call, esc, say } from "./core.js";
import { ask, toast } from "./ui.js";
import { job } from "./jobs.js";

export function renderHome() {
  // a build can't run yet: said where a new user lands, not found out minutes into the first build
  const stops = (S.setup?.checks || []).filter((c) => c.state === "bad");
  $("home-notice").hidden = !stops.length;
  if (stops.length) {
    $("home-notice").innerHTML = `<b>Not ready to build yet</b>${stops.map((c) => `<div>${esc(c.text)}: ${esc(c.fix)}</div>`).join("")}
      <button class="act alt small" id="home-setup" style="width:auto">Open Setup</button>`;
    $("home-setup").onclick = () => app.go("setup");
  }
  const cards = S.projects.map((p) => {
    const chips = [p.approved ? ["marked good", "good"] : null,
      p.built ? (p.failed?.length ? [`built without ${p.failed.join(", ").replaceAll("_", " ")}`, "warn"] : ["built", "info"])
        : p.surveyed ? ["started", ""] : ["new", ""],
      p.stale?.length ? ["changes not built", "warn"] : null,
      // built and checked, and nobody has said how it looks: SETUP.md filled with "awaiting verdict"
      p.waiting && !p.stale?.length ? ["waiting for your look", "info"] : null].filter(Boolean);
    const when = p.finished ? new Date(p.finished).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "";
    const where = p.manifest.replaceAll("\\", "/").split("/").slice(-3, -1).join("/");
    return `<button class="char ${p.manifest === S.manifestPath ? "open" : ""}" data-manifest="${esc(p.manifest)}">
      <div class="pic" ${p.picture ? `style="background-image:url('${esc(p.picture)}')"` : ""}>${p.picture ? "" : esc(p.name[0] || "?")}</div>
      <div class="body"><span class="name">${esc(p.name)}</span>
        <div class="chips">${chips.map(([t, k]) => `<span class="chip ${k}">${esc(t)}</span>`).join("")}</div>
        <span class="when">${when ? `last run ${esc(when)} · ` : ""}${esc(where)}</span></div></button>`;
  });
  $("cards").innerHTML = `<button class="char new" data-new><div class="pic">+</div><div class="body"><span class="name">New character</span>
    <span class="when">from a Tripo model (rigged .fbx best)</span></div></button>` + cards.join("");
  $("cards").querySelectorAll("[data-manifest]").forEach((b) => b.onclick = async () => {
    await app.openProject(b.dataset.manifest);
    const p = S.projects.find((q) => q.manifest === b.dataset.manifest);
    app.go(p?.surveyed ? "build" : "model");
  });
  $("cards").querySelector("[data-new]").onclick = async () => { await app.openProject(null); app.go("model"); };
}

// ---------- disk: what the runs take, and freeing what nothing reads any more
const gb = (bytes) => bytes >= 2 ** 30 ? `${(bytes / 2 ** 30).toFixed(1)} GB` : `${Math.round(bytes / 2 ** 20)} MB`;
let disk = null;
async function loadDisk() {
  say("disk-msg", "measuring…");
  try { disk = await (await fetch("/api/disk")).json(); } catch (e) { say("disk-msg", e.message, true); return; }
  if (disk.error) { say("disk-msg", disk.error, true); return; }
  say("disk-msg", "");
  const rows = disk.characters.filter((c) => c.bytes).sort((a, b) => b.bytes - a.bytes);
  const low = disk.free < 5 * 2 ** 30;
  $("disk-chip").textContent = `${gb(disk.free)} free on ${disk.drive}`;
  $("disk-chip").className = `chip ${low ? "warn" : "good"}`;
  $("disk").innerHTML = `<table><thead><tr><th>Character</th><th>Takes</th><th>Runs</th><th>Can be freed</th></tr></thead><tbody>
    ${rows.map((c) => `<tr><td>${esc(c.name)}</td><td class="num">${gb(c.bytes)}</td><td class="num">${c.runs.length}</td>
      <td class="num">${c.prunable ? `${gb(c.prunable)} <span class="msg">from ${c.prunable_runs} old run${c.prunable_runs === 1 ? "" : "s"}</span>` : `<span class="msg">nothing</span>`}</td></tr>`).join("")}
    </tbody></table>
    ${disk.other > 2 ** 20 ? `<p class="msg">Also in out\\: ${gb(disk.other)} the app doesn't manage (builds from before it, comparison runs). Delete those by hand if you no longer need them.</p>` : ""}`;
  $("disk-prune").disabled = !disk.prunable;
  $("disk-prune").textContent = disk.prunable ? `Free up ${gb(disk.prunable)}…` : "Nothing to free";
}
$("disk-check").onclick = loadDisk;
$("disk-prune").onclick = async () => {
  const runs = disk.characters.reduce((n, c) => n + c.prunable_runs, 0);
  const ok = await ask("Free up disk space", { text: `Delete the model files (.blend, .vrm) of ${runs} old run${runs === 1 ? "" : "s"}: ${gb(disk.prunable)}.\n\n`
    + "Their reports and logs stay. Each character's build in use, the build marked good and builds with a Warudo note are not touched.\n\nThis can't be undone.",
    ok: `Delete ${gb(disk.prunable)}`, danger: true });
  if (!ok) return;
  try {
    const done = await call("prune", { all: true });
    toast(`Freed ${gb(done.freed)} (${done.removed} files).`);
    await loadDisk();
  } catch (e) { toast(e.message, "bad"); }
};

// ---------- every character rebuilt and compared with its good build (python -m app regress)
$("regress-run").onclick = async () => {
  const count = S.projects.filter((p) => p.built).length;
  if (!await ask("Rebuild every character", { text: `Rebuilds ${count || "every"} built character${count === 1 ? "" : "s"} in scratch folders and compares each with its good build. `
    + "A few minutes each; nothing of theirs is changed.", ok: "Rebuild and compare" })) return;
  say("regress-msg", ""); $("regress-out").innerHTML = "";
  const state = await job("regress", {});
  if (!state?.result?.characters) { say("regress-msg", state?.error ? state.error.split("\n")[0] : state?.cancelled ? "Stopped." : "", !!state?.error); return; }
  const rows = state.result.characters;
  say("regress-msg", state.result.worse ? `${state.result.worse} thing${state.result.worse === 1 ? "" : "s"} measured worse.` : "Nothing measured worse.", !!state.result.worse);
  $("regress-out").innerHTML = `<table><thead><tr><th>Character</th><th>Against</th><th>Worse</th><th>Better</th><th>Time</th></tr></thead><tbody>
    ${rows.map((r) => `<tr><td>${esc(r.name)}</td><td class="msg">${esc(r.baseline || "")}</td>
      <td class="${r.error || r.worse.length ? "failed" : "done"}">${r.error ? esc(r.error) : r.worse.length ? r.worse.map(esc).join("<br>") : "nothing"}</td>
      <td class="done">${r.better.map(esc).join("<br>")}</td><td class="num">${r.seconds ? `${Math.round(r.seconds / 60)} min` : ""}</td></tr>`).join("")}</tbody></table>`;
};

// ---------- Model: a new character's form, or the open one's survey and rig
$("create").onclick = async () => {
  try {
    const made = await call("init", { model: $("new-model").value.trim(), name: $("new-name").value.trim(),
      author: $("new-author").value.trim(), face: $("new-face").value.trim(), reference: $("new-reference").value.trim(),
      rig: $("new-rig").value.trim() });
    toast(made.other_rig ? `Made ${made.manifest}. The model carries a ${made.other_rig} skeleton: the app places its own unless you name this file as the rig too (Model page).`
      : !made.rig || $("new-rig").value.trim() ? `Made ${made.manifest}`
      : made.rig === made.source ? `Made ${made.manifest}. Tripo's rig found in the model: it will be used.`
      : `Made ${made.manifest}. The model names Tripo's rig it was made from (${made.rig}): it will be used.`,
      "good", { ms: made.other_rig ? 12000 : 4500 });
    say("create-msg", "");
    await app.loadProjects(made.manifest);
    if ($("create-prepare").checked) job("prepare", {});
  } catch (e) { say("create-msg", e.message, true); }
};
$("duplicate").onclick = async () => {
  const name = await ask("Duplicate this character", { text: "A name for the copy: letters, digits, _ and -. Nothing is built yet in it.",
    input: true, value: `${S.manifest?.name || "copy"}_b`, ok: "Duplicate" });
  if (!name) return;
  try {
    const made = await call("duplicate", { name });
    toast(`Made ${made.name}. Change what you want to compare, then build it.`, "good", { ms: 8000 });
    await app.loadProjects(made.manifest);
    app.go("options");
  } catch (e) { say("duplicate-msg", e.message, true); }
};
$("survey-run").onclick = () => job("run", { only: "survey" });
$("prepare-run").onclick = () => job("prepare", {});
// Ready: the one thing only the user can do is next, with the way past it for a faceless model.
app.prepared = () => {
  app.go("face");
  toast("Ready. Mark the eyes and mouth on the picture, then press Save and build.", "good",
    { action: { label: "Build without a face", run: () => { app.go("build"); job("run", {}); } }, ms: 15000 });
};
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
  // a rigged FBX model is its own rig: the same path twice reads as a mistake without a word
  const ownRig = !!m?.rig && m.rig === m.source;
  $("rig-path").value = m?.rig || "";
  $("rig-state").innerHTML = !m ? "" : ownRig ? "<b>In use: Tripo's rig inside the model file</b> (clear the field and save to rig it yourself)."
    : m.rig ? "<b>In use: the rig file below.</b>" : "No rig: the app places the skeleton itself.";
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
