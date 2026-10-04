// Build: what's left, the steps with their logs, the verdict, the .vrm, and every earlier run.
import { $, S, app, call, esc, plain, served, stepName, lastVrm, refreshStatus } from "./core.js";
import { ask, toast } from "./ui.js";
import { job } from "./jobs.js";
import { nextRun } from "./nav.js";

let selectedStep = null;

export function renderSteps() {
  const body = $("step-rows");
  renderTodo(); renderVrm();
  $("ships").innerHTML = shipsHtml(S.status?.ships);
  if (!S.status) { body.innerHTML = ""; $("check").innerHTML = ""; $("verdict").innerHTML = ""; $("approved").innerHTML = ""; $("step-detail").hidden = true; return; }
  body.innerHTML = S.status.steps.map((s) => `<tr data-step="${s.step}" class="${s.step === selectedStep ? "sel" : ""}">
    <td>${stepName(s.step)}</td><td><span class="chip ${s.runs ? s.status.split(" ")[0] : "off"}">${s.runs ? esc(s.status) : "off"}</span></td>
    <td>${s.seconds ? s.seconds + " s" : ""}</td><td class="msg">${!s.runs ? esc(s.skip_reason)
      : s.status === "failed" ? `built without it: ${esc(plain(s.step, s.reason))}` : ""}</td></tr>`).join("");
  body.querySelectorAll("tr").forEach((tr) => tr.onclick = () => { selectedStep = selectedStep === tr.dataset.step ? null : tr.dataset.step; renderSteps(); });
  renderDetail();
  const v = S.status.verdict;
  const line = (cls) => ([page, text]) => `<li class="${cls}">${esc(text)} <a href="#${page}">${page}</a></li>`;
  $("verdict").innerHTML = !v ? "" : (v.ready ? `<p class="done">Ready for Warudo${v.looks.length ? ", with a few things worth a look" : ": every check passed"}. Try it pose by pose (Review).</p>` : "")
    + `<ul class="plain">${v.issues.map(line("failed")).join("")}${v.looks.map(line("warn")).join("")}</ul>`;
  // the baseline: what the user said was good, and what this build did differently
  const a = S.status.approved, since = S.status.since_approved;
  $("approved").innerHTML = (a ? `<p class="msg">Compared with the build you marked good on ${esc(a.when.replace("T", " "))}${a.note ? ` (${esc(a.note)})` : ""}:
      ${since && !since.worse.length && !since.better.length ? "no change past the noise." : ""}</p>`
      + `<ul class="plain">${(since?.better || []).map((b) => `<li class="done">better: ${esc(b)}</li>`).join("")}</ul>`
    : `<p class="msg">No build marked good yet: once one looks right in Warudo, mark it, and later builds are compared with it.</p>`)
    + (v?.ready || S.status.steps.find((s) => s.step === "vrm_check")?.status === "done"
      ? `<button class="act alt" id="approve">Mark this build as good</button>` : "");
  $("approve")?.addEventListener("click", async () => {
    const note = await ask("Mark this build as good", { text: "What did you check it with? Later builds are compared with this one.", input: true, value: "Warudo: face, arms", ok: "Mark as good" });
    if (note === null) return;
    try { await call("approve", { note }); await refreshStatus(); app.renderAll(); toast("Marked as good."); } catch (e) { toast(e.message, "bad"); }
  });
  const failed = S.status.check_failed;
  $("check").innerHTML = (failed == null ? `<p class="msg">not checked yet</p>` : failed.length ? `<ul class="plain">${failed.map((f) => `<li class="failed">${esc(f)}</li>`).join("")}</ul>` : `<p class="done">nothing over the limits</p>`)
    + spotsHtml(S.status.steps.find((s) => s.step === "check")?.report?.spots);
  // the still close-ups show where; the viewer shows the avatar going into that pose, springs running
  $("check").querySelectorAll("[data-watch]").forEach((b) => b.onclick = () => {
    app.go("review"); app.openViewer(null, null, { pose: b.dataset.watch });
  });
}

// What the finished .vrm can do, read from the file itself (engine.shipped): the checks say whether
// it bends well, this says what is in it, in the words a tracking app's user thinks in.
function shipsHtml(s) {
  if (!s) return "";
  const yes = (t) => `<span class="done">${t}</span>`, no = (t) => `<span class="warn">${t}</span>`;
  const extra = s.shapes.filter((n) => !["aa", "ih", "ou", "ee", "oh"].includes(n)).length;
  const rows = [
    ["Body", `${s.bones} bones, ${s.fingers ? "fingers included" : "no finger bones"}`],
    ["Blink", s.blink ? yes("yes") : no("no")],
    ["Talking", s.vowels.length === 5 ? yes("all five mouth shapes") : no(`${s.vowels.length} of 5 mouth shapes`)],
    ["Face tracking", extra ? yes(`${extra} shapes (jaw, mouth, blink, eyes)`) : no("none")],
    ["Eyes", s.look === "expression" && s.looks_bound ? yes("follow your gaze") : no("stay still")],
    ["Physics", s.springs ? `${s.springs} swinging chains, ${s.colliders} colliders` : "nothing swings"],
    ["Look", `${s.toon} of ${s.materials} materials toon-shaded, ${s.textures} texture${s.textures === 1 ? "" : "s"}`],
    ["File", `${s.megabytes} MB`],
  ];
  return `<div class="title" style="margin-top:var(--s3)">What's in the file</div>
    <dl class="facts">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
}

// One step up close: what it read and made, why it failed, and its log's end.
async function renderDetail() {
  const box = $("step-detail"), s = S.status?.steps.find((x) => x.step === selectedStep);
  box.hidden = !s;
  if (!s) return;
  // a failed step's log is named in its reason ("... (see <log>)"), not kept as its own field
  const fail = s.last_failure, known = s.known || fail?.known;
  const log = s.log || fail?.log || String(s.reason || fail?.reason || "").match(/see (.+?\.log)\)/)?.[1];
  const file = (p) => p ? `<code>${esc(String(p).replaceAll("\\", "/").split("/").slice(-2).join("/"))}</code>` : "";
  box.innerHTML = `<div class="title">${stepName(s.step)} <span class="chip ${s.status.split(" ")[0]}">${esc(s.runs ? s.status : "off")}</span></div>
    <dl>${s.finished ? `<dt>finished</dt><dd>${esc(s.finished.replace("T", " "))}${s.seconds ? `, ${s.seconds} s` : ""}</dd>` : ""}
      ${!s.runs ? `<dt>off</dt><dd>${esc(s.skip_reason)}</dd>` : ""}
      ${s.blend ? `<dt>made</dt><dd>${file(s.blend)}${s.vrm ? `, ${file(s.vrm)}` : ""}</dd>` : s.json ? `<dt>report</dt><dd>${file(s.json)}</dd>` : ""}
      ${s.reason ? `<dt>why</dt><dd class="failed">${esc(plain(s.step, s.reason))}</dd>` : ""}
      ${fail ? `<dt>last try</dt><dd class="failed">${esc(fail.finished.replace("T", " "))}: ${esc(plain(s.step, fail.reason))}</dd>` : ""}
      ${known ? `<dt>${esc(known.code)}</dt><dd>${esc(known.what)}. What to do: ${esc(known.fix)}</dd>` : ""}</dl>
    <div class="row"><button class="act alt small" data-job id="detail-from" ${s.runs ? "" : "disabled"}>Run from here</button>
      <button class="act alt small" data-job id="detail-only" ${s.runs ? "" : "disabled"}>Run only this</button>
      ${log ? `<button class="act alt small" id="detail-reveal">Show log in folder</button>` : ""}</div>
    ${log ? `<pre id="detail-log">loading the log…</pre>` : ""}`;
  $("detail-from").onclick = () => job("run", { from: s.step });
  $("detail-only").onclick = () => job("run", { only: s.step });
  $("detail-reveal")?.addEventListener("click", () => reveal(log));
  if (!$("runbar").hidden) box.querySelectorAll("[data-job]").forEach((b) => b.disabled = true);
  if (!log) return;
  try {
    const { url } = await call("url", { path: log });
    const text = url ? await (await fetch(url)).text() : "";
    $("detail-log").textContent = text ? text.split(/\r?\n/).slice(-60).join("\n") : "the log is gone";
    $("detail-log").scrollTop = 1e9;
  } catch (e) { $("detail-log").textContent = e.message; }
}

// the rig check's worst stretch and squash, close up: at rest, then in the pose that did it
function spotsHtml(spots) {
  if (!spots?.length) return "";
  return spots.map((s) => `<p class="msg" style="margin:8px 0 2px">Where it ${esc(s.what)} (${esc(s.pose)}), red dots on the edge's ends:
      <button class="act alt small" data-watch="${esc(s.pose)}" title="open the avatar in the viewer, in this pose">Watch it</button></p>
    <div class="spots">${["face", "side"].flatMap((v) => ["rest", "posed"].map((st) => {
      const caption = `${v === "face" ? (s.at[1] > 0 ? "back" : "front") : "side"}, ${st === "rest" ? "at rest" : s.pose}`;
      return `<figure><img src="${served(s.files[`${v}_${st}`])}" alt="${esc(caption)}"><figcaption>${esc(caption)}</figcaption></figure>`;
    })).join("")}</div>`).join("");
}

// [label, action] rerunning a failed step, first setting the options its error code names as the fix
function retry(step, known) {
  if (!known?.options) return ["Try it again", () => job("run", { from: step.step })];
  const change = Object.entries(known.options);
  return [`Set ${change.map(([k, v]) => `${k} to ${v}`).join(", ")} and rerun`, async () => {
    try { for (const [key, value] of change) await call("set", { key, value }); await refreshStatus(); }
    catch (e) { toast(e.message, "bad"); return; }
    job("run", { from: step.step });
  }];
}

// What to do next, in order, each with the button that does it: the steps table alone left the
// user reading notes like "no face points, or face tracking off" with no way on.
function renderTodo() {
  if (!S.manifest || !S.status) { $("todo").innerHTML = `<p class="help">Open or create a character first (Characters).</p>`; return; }
  const steps = Object.fromEntries(S.status.steps.map((s) => [s.step, s])), items = [];
  const add = (text, label, act, kind = "warn", isJob = false) => items.push({ text, label, act, kind, isJob });
  if (steps.survey?.status !== "done") add("Survey the model: reads how it stands and what it's made of.", "Run survey", () => job("run", { only: "survey" }), "warn", true);
  if (!S.listing) add("List the pieces, then give any hair, hat, cape or sleeve its role.", "Go to Parts", () => app.go("parts"));
  if (!S.manifest.face) add("Mark the eyes and mouth. Without them the avatar can't blink, has no face tracking and its eyes don't move: only the vowel mouth shapes ship.", "Go to Face", () => app.go("face"));
  const failed = S.status.steps.filter((s) => s.runs && s.status === "failed");
  const built = S.status.steps.some((s) => s.step !== "survey" && ["done", "stale", "failed"].includes(s.status));
  const next = nextRun(S.status);
  if (!built) add("Build the avatar: a few minutes, one step after another.", "Build everything", () => job("run", {}), "warn", true);
  else if (next) add(`${next.count} step${next.count > 1 ? "s" : ""} to (re)build, from “${stepName(next.from)}”${next.stale ? ": something before it changed" : ""}.`,
    `Run from ${stepName(next.from)}`, () => job("run", { from: next.from }), "warn", true);
  for (const f of failed) add(`“${stepName(f.step)}” failed and the avatar was built without it: ${plain(f.step, f.reason)}${f.known ? ` (${f.known.code}: ${f.known.fix})` : ""}`,
    ...retry(f, f.known), "failed", true);
  // a step that stopped the chain: its last good output stays, the failure rides beside it
  for (const f of S.status.steps.filter((s) => s.runs && s.last_failure)) {
    const k = f.last_failure.known;
    add(`“${stepName(f.step)}” failed on ${f.last_failure.finished.replace("T", " ")}${k ? `: ${k.what}. ${k.fix}` : `: ${plain(f.step, f.last_failure.reason).split("\n")[0]}`}`,
      ...retry(f, k), "failed", true);
  }
  const pageName = (page) => page[0].toUpperCase() + page.slice(1);
  for (const [page, text] of S.status.verdict?.issues || []) if (page !== "build") add(text, `Go to ${pageName(page)}`, () => app.go(page), "failed");
  for (const [page, text] of S.status.verdict?.looks || []) if (page !== "face" || S.manifest.face) add(text, `Go to ${pageName(page)}`, () => app.go(page));
  if (S.status.verdict?.ready) add("Load the .vrm in Warudo and try it pose by pose; note what you see on the Review page.", "Go to Review", () => app.go("review"), "done");
  $("todo").innerHTML = items.length ? items.map((it, i) => `<div class="todo"><span class="${it.kind}">${esc(it.text)}</span>
    <button class="act alt small" data-todo="${i}" ${it.isJob ? "data-job" : ""} ${it.isJob && !$("runbar").hidden ? "disabled" : ""}>${esc(it.label)}</button></div>`).join("") : `<p class="done">Nothing left.</p>`;
  $("todo").querySelectorAll("[data-todo]").forEach((b) => b.onclick = items[+b.dataset.todo].act);
}
$("run-all").onclick = () => job("run", {});

// ---------- the .vrm: where it is, for loading into Warudo
export async function reveal(path) {
  try { await call("reveal", { path }); } catch (e) { toast(e.message, "bad"); }
}
export async function copyPath(path) {
  try { await navigator.clipboard.writeText(path); }
  catch {
    const t = Object.assign(document.createElement("textarea"), { value: path });
    document.body.append(t); t.select(); document.execCommand("copy"); t.remove();
  }
  toast("Path copied: paste it into Warudo's file dialog.", "info");
}
// Copies the finished .vrm into Warudo's Characters folder; asks for the folder once if Steam doesn't have it.
async function sendToWarudo(folder) {
  try {
    const sent = await call("send_warudo", folder === undefined ? {} : { folder });
    toast(`${sent.already ? "Already in Warudo" : "Sent to Warudo"}: pick ${sent.name} as the character's source there.`, "info");
  } catch (e) {
    if (folder !== undefined || !/Warudo wasn't found/.test(e.message)) return toast(e.message, "bad");
    const given = await ask("Where is Warudo?", { text: "Warudo wasn't found in a Steam library. Paste the path of its Characters folder "
      + "(Warudo\\Warudo_Data\\StreamingAssets\\Characters); the app remembers it.", input: true, value: "", ok: "Send" });
    if (given) sendToWarudo(given.trim());
  }
}
function renderVrm() {
  const vrm = lastVrm();
  for (const id of ["vrm-send", "vrm-show", "vrm-copy", "review-vrm-send", "review-vrm-show", "review-vrm-copy"]) $(id).disabled = !vrm;
  $("vrm-path").innerHTML = vrm ? `<code>${esc(vrm)}</code>` : "";
}
$("vrm-send").onclick = $("review-vrm-send").onclick = () => sendToWarudo();
$("vrm-show").onclick = $("review-vrm-show").onclick = () => reveal(lastVrm());
$("vrm-copy").onclick = $("review-vrm-copy").onclick = () => copyPath(lastVrm());

// ---------- history: every run folder, to open, render or compare
let ticked = [];
export async function loadRuns() {
  S.runs = S.manifestPath ? (await call("runs")).runs : [];
  ticked = ticked.filter((r) => S.runs.some((x) => x.run === r));
  renderRuns();
}
function renderRuns() {
  const body = $("run-rows");
  $("runs-compare").disabled = ticked.length !== 2;
  $("runs-view").disabled = ticked.length !== 2 || !pair().every((r) => r.vrm_url);
  if (!S.runs.length) { body.innerHTML = `<tr><td colspan="5" class="msg">No runs yet.</td></tr>`; return; }
  body.innerHTML = S.runs.map((r) => {
    const when = r.when ? new Date(r.when).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : r.run;
    const check = r.failed == null ? `<span class="msg">—</span>` : r.failed.length ? `<span class="warn" title="${esc(r.failed.join("\n"))}">${r.failed.length} over the limits</span>` : `<span class="done">passed</span>`;
    return `<tr data-run="${esc(r.run)}"><td><input type="checkbox" aria-label="compare" ${ticked.includes(r.run) ? "checked" : ""} ${r.check ? "" : "disabled"}></td>
      <td>${esc(when)} ${r.approved ? `<span class="chip good">good</span>` : ""}${r.current ? ` <span class="chip info" title="the steps' latest files come from this run">in use</span>` : ""}</td>
      <td class="msg" title="${esc(r.steps.join(", "))}">${r.steps.length > 3 ? `${esc(stepName(r.steps[0]))} … ${esc(stepName(r.steps.at(-1)))} (${r.steps.length})` : esc(r.steps.map(stepName).join(", "))}</td>
      <td>${check}</td>
      <td class="row" style="flex-wrap:nowrap">${r.vrm_url ? `<button class="act alt small" data-view title="open this run's .vrm in the viewer (Review)">View</button>` : ""}
        <button class="act alt small" data-render data-job title="render this run's last model in the Review page's pose">Render</button>
        <button class="act alt small" data-folder title="show this run's files">Folder</button></td></tr>`;
  }).join("");
  body.querySelectorAll("tr[data-run]").forEach((tr) => {
    const r = S.runs.find((x) => x.run === tr.dataset.run);
    tr.querySelector("input").onchange = (e) => {
      ticked = e.target.checked ? [...ticked, r.run].slice(-2) : ticked.filter((x) => x !== r.run);
      renderRuns();
    };
    tr.querySelector("[data-view]")?.addEventListener("click", () => { app.go("review"); app.openViewer(r.vrm_url, `run of ${r.when?.replace("T", " ") || r.run}`); });
    tr.querySelector("[data-render]").onclick = () => { app.go("review"); job("render", { pose: $("pose").value, run: r.run }); };
    tr.querySelector("[data-folder]").onclick = () => reveal(r.folder);
    if (!$("runbar").hidden) tr.querySelector("[data-render]").disabled = true;
  });
}
// older first: the table says what got better or worse since
const pair = () => ticked.map((t) => S.runs.find((r) => r.run === t)).sort((a, b) => a.run.localeCompare(b.run));
const whenOf = (r) => r.when ? new Date(r.when).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : r.run;
$("runs-compare").onclick = async () => {
  const [before, after] = pair();
  try {
    const out = await call("compare", { before: before.check, after: after.check });
    const cell = ([a, b], way, unit) => `<td class="num ${way === "worse" ? "failed" : way === "better" ? "done" : ""}">${a ?? "–"} → ${b ?? "–"}${unit}${way ? ` (${way})` : ""}</td>`;
    $("runs-out").hidden = false;
    $("runs-out").innerHTML = `<p class="msg">${esc(whenOf(before))} → ${esc(whenOf(after))}: ${out.worse ? `<span class="failed">something got worse</span>` : "nothing got worse"}.
        Stretch is the longest an edge grows in that pose; squash the shortest one gets, as a share of its length (1 is untouched).</p>
      <table><thead><tr><th>Pose</th><th>Stretch</th><th>Squash</th><th></th></tr></thead><tbody>
      ${out.poses.map((p) => `<tr><td>${esc(p.pose)}</td>${cell(p.stretch, p.stretch_way, " cm")}${cell(p.squash, p.squash_way, "")}
        <td><button class="act alt small" data-both="${esc(p.pose)}" ${before.vrm_url && after.vrm_url ? "" : "disabled"} title="both avatars in this pose, side by side">Watch</button></td></tr>`).join("")}
      ${out.unweighted ? `<tr><td>vertices on no bone</td><td class="num">${out.unweighted[0]} → ${out.unweighted[1]}</td><td></td><td></td></tr>` : ""}</tbody></table>`;
    $("runs-out").querySelectorAll("[data-both]").forEach((b) => b.onclick = () => viewBoth(b.dataset.both));
  } catch (e) { toast(e.message, "bad"); }
};
// The two ticked runs' avatars side by side, springs running, in the same pose.
function viewBoth(pose) {
  const runs = pair();
  if (runs.length !== 2 || !runs.every((r) => r.vrm_url)) return toast("One of the two runs has no .vrm (its model files may have been freed).", "bad");
  $("runs-viewers").hidden = false;
  for (const [id, r] of [["a", runs[0]], ["b", runs[1]]]) {
    $(`runs-viewer-${id}-label`).textContent = `${whenOf(r)}${r.approved ? " · marked good" : ""}${r.current ? " · in use" : ""}`;
    $(`runs-viewer-${id}`).src = `/viewer/index.html?model=${encodeURIComponent(r.vrm_url)}${pose ? `&pose=${encodeURIComponent(pose)}` : ""}`;
  }
  $("runs-viewers").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
$("runs-view").onclick = () => viewBoth("");
