// Build: what's left, the steps with their logs, the verdict, the .vrm, and every earlier run.
import { $, S, app, call, esc, sentence, served, stepName, tell, tookLast, lastVrm, refreshStatus } from "./core.js";
import { ask, keyable, toast } from "./ui.js";
import { job } from "./jobs.js";
import { nextRun } from "./nav.js";

let selectedStep = null, checkDrawn = null;

export function renderSteps() {
  const body = $("step-rows");
  renderPreflight(); renderTodo(); renderVrm();
  $("ships").innerHTML = shipsHtml(S.status?.ships);
  if (!S.status) { checkDrawn = null; body.innerHTML = ""; $("check").innerHTML = ""; $("verdict").innerHTML = ""; $("approved").innerHTML = ""; $("step-detail").hidden = true; return; }
  body.innerHTML = S.status.steps.map((s) => `<tr data-step="${s.step}" class="${s.step === selectedStep ? "sel" : ""}">
    <td>${stepName(s.step)}</td><td><span class="chip ${s.runs ? s.status.split(" ")[0] : "off"}">${s.runs ? esc(s.status) : "off"}</span></td>
    <td>${s.seconds ? s.seconds + " s" : ""}</td><td class="msg">${!s.runs ? esc(s.skip_reason)
      : s.status === "failed" ? `built without it: ${esc(tell(s.failure))}` : ""}</td></tr>`).join("");
  body.querySelectorAll("tr").forEach((tr) => { keyable(tr); tr.onclick = () => { selectedStep = selectedStep === tr.dataset.step ? null : tr.dataset.step; renderSteps(); }; });
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
  const failed = S.status.check_failed, spots = S.status.steps.find((s) => s.step === "check")?.report?.spots;
  // drawn again only when the check is another one: every role given on the Parts page asks for
  // the status, and each time the ten close-ups were fetched afresh
  const drawn = JSON.stringify([S.manifestPath, failed, spots]);
  if (drawn === checkDrawn) return;
  checkDrawn = drawn;
  $("check").innerHTML =(failed == null ? `<p class="msg">not checked yet</p>` : failed.length ? `<ul class="plain">${failed.map((f) => `<li class="failed">${esc(f)}</li>`).join("")}</ul>` : `<p class="done">nothing over the limits</p>`)
    + spotsHtml(spots);
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
  const fail = s.last_failure, f = s.failure || fail?.failure;
  const log = s.log || fail?.log;
  const file = (p) => p ? `<code>${esc(String(p).replaceAll("\\", "/").split("/").slice(-2).join("/"))}</code>` : "";
  box.innerHTML = `<div class="title">${stepName(s.step)} <span class="chip ${s.status.split(" ")[0]}">${esc(s.runs ? s.status : "off")}</span></div>
    <dl>${s.finished ? `<dt>finished</dt><dd>${esc(s.finished.replace("T", " "))}${s.seconds ? `, ${s.seconds} s` : ""}</dd>` : ""}
      ${!s.runs ? `<dt>off</dt><dd>${esc(s.skip_reason)}</dd>` : ""}
      ${s.blend ? `<dt>made</dt><dd>${file(s.blend)}${s.vrm ? `, ${file(s.vrm)}` : ""}</dd>` : s.json ? `<dt>report</dt><dd>${file(s.json)}</dd>` : ""}
      ${s.failure ? `<dt>why</dt><dd class="failed">${esc(sentence(tell(s.failure)))}</dd>` : s.reason && s.status !== "skipped" ? `<dt>why</dt><dd>${esc(s.reason)}</dd>` : ""}
      ${fail ? `<dt>last try</dt><dd class="failed">${esc(fail.finished.replace("T", " "))}: ${esc(sentence(tell(fail.failure)))}</dd>` : ""}
      ${f ? `<dt>${esc(f.code || "what to do")}</dt><dd>${esc(sentence(f.fix))}</dd>` : ""}</dl>
    <div class="row"><button class="act alt small" data-job id="detail-from" ${s.runs ? "" : "disabled"}>Run from here</button>
      <button class="act alt small" data-job id="detail-only" ${s.runs ? "" : "disabled"}>Run only this</button>
      ${log ? `<button class="act alt small" id="detail-reveal">Show log in folder</button>` : ""}
      ${fail || s.status === "failed" ? `<button class="act alt small" id="detail-report">Report this</button>` : ""}</div>
    ${log ? `<pre id="detail-log">loading the log…</pre>` : ""}`;
  $("detail-from").onclick = () => job("run", { from: s.step });
  $("detail-only").onclick = () => job("run", { only: s.step });
  $("detail-reveal")?.addEventListener("click", () => reveal(log));
  $("detail-report")?.addEventListener("click", () => app.report(`${stepName(s.step)} failed: ${tell(f)}${f?.code ? ` [${f.code}]` : ""}`));
  if (!$("runbar").hidden) box.querySelectorAll("[data-job]").forEach((b) => b.disabled = true);
  if (!log) return;
  // (another step may have been picked while this one's log was on its way: its box is not this log's)
  const shown = () => selectedStep === s.step && $("detail-log");
  try {
    const { url } = await call("url", { path: log });
    const text = url ? await (await fetch(url)).text() : "";
    if (!shown()) return;
    $("detail-log").textContent = text ? text.split(/\r?\n/).slice(-60).join("\n") : "the log is gone";
    $("detail-log").scrollTop = 1e9;
  } catch (e) { if (shown()) $("detail-log").textContent = e.message; }
}

// the rig check's worst stretch and squash, close up: at rest, then in the pose that did it
function spotsHtml(spots) {
  if (!spots?.length) return "";
  // what to look for, said: the pictures alone were a close-up of cloth with two dots nobody found
  const look = { stretch: "the surface between the two red dots is pulled long: a thin, smeared or torn-looking strip",
    squash: "the surface between the two red dots is pressed together: a crease, a fold or a dent" };
  return spots.map((s) => `<p class="msg" style="margin:8px 0 2px"><b>The spot that ${esc(s.what)}</b> in the pose “${esc(s.pose)}”. ${s.files.where ? "In the pink ring" : "Look for"}: ${look[s.kind] || "the two red dots are the ends of the measured edge"}.
      Each pair is the same view standing still, then in the pose; click a picture to see it large.
      <button class="act alt small" data-watch="${esc(s.pose)}" title="open the avatar in the viewer, in this pose">Watch it move</button></p>
    <div class="spots">${(s.files.where ? [["where", `where on the body (${s.pose})`]] : []).concat(["face", "side"].flatMap((v) => ["rest", "posed"].map((st) =>
      [`${v}_${st}`, `${v === "face" ? (s.at[1] > 0 ? "back" : "front") : "side"}, ${st === "rest" ? "standing still" : s.pose}`])))
      .map(([file, caption]) => `<figure><img src="${served(s.files[file])}" alt="${esc(caption)}"><figcaption>${esc(caption)}</figcaption></figure>`).join("")}</div>`).join("");
}

// [label, action] rerunning a failed step, first setting the options its failure names as the fix
function retry(step, failure) {
  if (!failure?.options) return ["Try it again", () => job("run", { from: step.step })];
  const change = Object.entries(failure.options);
  return [`Set ${change.map(([k, v]) => `${k} to ${v ?? "automatic"}`).join(", ")} and rerun`, async () => {
    try { for (const [key, value] of change) await call("set", { key, value }); await refreshStatus(); }
    catch (e) { toast(e.message, "bad"); return; }
    job("run", { from: step.step });
  }];
}

// ---------- before a build: what is known to go wrong with it (app/preflight.py), each with the button that mends it
// What a line's `action` does (errors.py names them); "gone": the page was left or a job started, which
// reads the character again by itself.
const ACTIONS = {
  list_pieces: ["List the parts again", () => { job("list_pieces"); return "gone"; }],
  remove_pieces: ["Take the role off", async (line) => { for (const name of line.args?.names || []) await call("piece", { name, remove: true }); }],
  forget_eyes_fallback: ["Try the moving iris again", () => call("forget_eyes_fallback", {})],
};
const PAGE_NAMES = { home: "Characters", setup: "Setup", model: "Model", parts: "Parts", joints: "Joints", face: "Face", options: "Options",
  paint: "Paint", movement: "Movement", review: "Review" };
// [the button's words, what it does] for a line: the options it names set, else its action, else its page opened
function mend(line) {
  const set = Object.entries(line.options || {});
  if (set.length) return [`Set ${set.map(([key, value]) => `${stepName(key)} to ${value ?? "automatic"}`).join(", ")}`,
    async () => { for (const [key, value] of set) await call("set", { key, value }); }];
  if (ACTIONS[line.action]) return [line.action === "remove_pieces" && line.args?.names?.length > 1 ? "Take the roles off" : ACTIONS[line.action][0], () => ACTIONS[line.action][1](line)];
  return PAGE_NAMES[line.page] ? [`Go to ${PAGE_NAMES[line.page]}`, () => { app.go(line.page); return "gone"; }] : null;
}
function renderPreflight() {
  const lines = S.manifest ? (S.preflight?.lines || []).filter((line) => line.kind !== "ok") : [];
  $("preflight-card").hidden = !lines.length;
  $("preflight").innerHTML = lines.map((line, i) => `<div class="todo"><span class="${line.kind === "stop" ? "failed" : "warn"}" title="${esc(line.code)}">${esc(sentence(line.text))}
      <span class="msg">${esc(sentence(line.fix))}</span></span>
    ${mend(line) ? `<button class="act alt small" data-mend="${i}">${esc(mend(line)[0])}</button>` : ""}</div>`).join("");
  $("preflight").querySelectorAll("[data-mend]").forEach((b) => b.onclick = async () => {
    try { if (await mend(lines[+b.dataset.mend])[1]() !== "gone") await app.openProject(S.manifestPath); }
    catch (e) { toast(e.message, "bad"); }
  });
}

// What to do next, in order, each with the button that does it: the steps table alone left the
// user reading notes like "no face points, or face tracking off" with no way on.
function renderTodo() {
  if (!S.manifest || !S.status) { $("todo").innerHTML = `<p class="help">Open or create a character first (Characters).</p>`; return; }
  const steps = Object.fromEntries(S.status.steps.map((s) => [s.step, s])), items = [];
  const add = (text, label, act, kind = "warn", isJob = false) => items.push({ text, label, act, kind, isJob });
  const improving = !!S.status.improving;
  if (!improving && steps.survey?.status !== "done") add("Survey the model: reads how it stands and what it's made of.", "Run survey", () => job("run", { only: "survey" }), "warn", true);
  if (!improving && !S.listing) add("List the pieces, then give any hair, hat, cape or sleeve its role.", "Go to Parts", () => app.go("parts"));
  // (the face not marked is said above, with what else is known before a build: preflight's VM-NO-FACE-POINTS)
  if (!improving && !S.manifest.face && !S.preflight) add("Mark the eyes and mouth. Without them the avatar can't blink, has no face tracking and its eyes don't move: only the vowel mouth shapes ship.", "Go to Face", () => app.go("face"));
  const failed = S.status.steps.filter((s) => s.runs && s.status === "failed");
  const built = S.status.steps.some((s) => s.step !== "survey" && ["done", "stale", "failed"].includes(s.status));
  const next = nextRun(S.status);
  if (!built) add(improving ? "Take the avatar in and write it out improved: about a minute. The file you gave is not changed."
    : "Build the avatar: a few minutes, one step after another.", "Build everything", () => job("run", {}), "warn", true);
  const took = next && tookLast(S.status, next.from);
  if (built && next) add(`${next.count} step${next.count > 1 ? "s" : ""} to (re)build, from “${stepName(next.from)}”${next.stale ? ": something before it changed" : ""}${took ? ` (about ${took} the last time)` : ""}.`,
    `Run from ${stepName(next.from)}`, () => job("run", { from: next.from }), "warn", true);
  for (const f of failed) add(`“${stepName(f.step)}” failed and the avatar was built without it: ${tell(f.failure)}. ${sentence(f.failure?.fix)}${f.failure?.code ? ` [${f.failure.code}]` : ""}`,
    ...retry(f, f.failure), "failed", true);
  // a step that stopped the chain: its last good output stays, the failure rides beside it
  for (const f of S.status.steps.filter((s) => s.runs && s.last_failure)) {
    const k = f.last_failure.failure;
    // which file: a tester's good FBX was refused and nothing on the page said what path the character held
    const named = k?.code === "VM-UNSUPPORTED-FILE" ? ` The model this character names: ${S.manifest.source}` : "";
    add(`“${stepName(f.step)}” failed on ${f.last_failure.finished.replace("T", " ")}: ${tell(k)}. ${sentence(k?.fix)}${k?.code ? ` [${k.code}]` : ""}${named}`,
      ...retry(f, k), "failed", true);
  }
  const pageName = (page) => page[0].toUpperCase() + page.slice(1);
  for (const [page, text] of S.status.verdict?.issues || []) if (page !== "build") add(text, `Go to ${pageName(page)}`, () => app.go(page), "failed");
  for (const [page, text] of S.status.verdict?.looks || []) if (page !== "face" || S.manifest.face) add(text, `Go to ${pageName(page)}`, () => app.go(page));
  for (const line of S.status.gained || []) add(`Better than the file you gave: ${line}.`, "Go to Review", () => app.go("review"), "done");
  if (S.status.verdict?.ready) add("Load the .vrm in Warudo and try it pose by pose; note what you see on the Review page.", "Go to Review", () => app.go("review"), "done");
  // one way on from a failure that retrying doesn't fix: tell the developers, with what failed already said
  const broken = items.find((it) => it.kind === "failed" && it.isJob);
  if (broken) add("Can't get past this? Tell the developers: the report says what failed.", "Report", () => app.report(broken.text));
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
    if (folder !== undefined || e.known?.code !== "VM-NO-WARUDO") return toast(e.message, "bad");
    const given = await ask("Where is Warudo?", { text: "Warudo isn't in any Steam library the app knows. Paste the path of its Characters folder "
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
  // (a row's buttons wrap when the list is narrow: held on one line, the third was cut off at the list's
  // edge behind a sideways scroll bar wherever the page stands in two columns, 1280 px wide and the like)
  body.innerHTML = S.runs.map((r) => {
    const when = r.when ? new Date(r.when).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : r.run;
    const check = r.failed == null ? `<span class="msg">—</span>` : r.failed.length ? `<span class="warn" title="${esc(r.failed.join("\n"))}">${r.failed.length} over the limits</span>` : `<span class="done">passed</span>`;
    return `<tr data-run="${esc(r.run)}"><td><input type="checkbox" aria-label="compare" ${ticked.includes(r.run) ? "checked" : ""} ${r.check ? "" : "disabled"}></td>
      <td>${esc(when)} ${r.approved ? `<span class="chip good">good</span>` : ""}${r.current ? ` <span class="chip info" title="the steps' latest files come from this run">in use</span>` : ""}</td>
      <td class="msg" title="${esc(r.steps.join(", "))}">${r.steps.length > 3 ? `${esc(stepName(r.steps[0]))} … ${esc(stepName(r.steps.at(-1)))} (${r.steps.length})` : esc(r.steps.map(stepName).join(", "))}</td>
      <td>${check}</td>
      <td class="row">${r.vrm_url ? `<button class="act alt small" data-view title="open this run's .vrm in the viewer (Review)">View</button>` : ""}
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
