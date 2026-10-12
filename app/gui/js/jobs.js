// Long jobs (a build, a listing, a render): start one, follow it in the run bar, stop it, say when it ends.
import { $, S, app, call, say, sentence, stepName, tell } from "./core.js";
import { ask, toast } from "./ui.js";

let polling = false, run = null, ticker = null, waiting = [];

// Resolves when the job ends, to its last report ({result, error, cancelled}), after the open
// character has been read again; to null when it couldn't start.
export async function job(route, body = {}) {
  if (route === "run" && !(await mayBuild(body))) return null;
  try { await call(route, body); } catch (e) { toast(e.message, "bad"); $("events").textContent += `\n${e.message}`; return null; }
  $("events").textContent = "";
  // the last run's failure isn't this one's
  for (const id of new Set(Object.values(SAYS))) say(id, "");
  run = plan(route, body);
  showBar(true);
  const ended = new Promise((done) => waiting.push(done));
  if (!polling) poll(0);
  return ended;
}
app.job = job;

// Before a build: what is known to go wrong with these steps (app/preflight.py), asked afresh. A stop is
// asked about, each with what mends it; "Build anyway" is remembered for this character, so the same
// stop is a warning from then on. Every button that starts a build comes through here.
async function mayBuild(body) {
  if (!S.manifestPath) return true;
  const asked = new URLSearchParams(Object.entries({ from: body.from, to: body.to, only: body.only }).filter(([, step]) => step));
  let found;
  try { found = await call(`preflight${[...asked].length ? `?${asked}` : ""}`); } catch { return true; }
  const stops = found.lines.filter((line) => line.kind === "stop");
  if (!stops.length) return true;
  const sure = await ask(stops.length > 1 ? `${stops.length} things will go wrong with this build` : "This build will go wrong",
    { text: stops.map((line) => `${sentence(line.text)}\n${sentence(line.fix)}`).join("\n\n"), ok: "Build anyway", danger: true });
  if (!sure) { app.go("build"); return false; }
  try { await call("preflight_accept", { codes: stops.map((line) => line.code) }); } catch {}
  return true;
}

// Where a failed job says so: beside the button that started it. Every job's failure was once written
// on every page, and a face detector that found no face stood in red on Movement after a pose test.
const SAYS = { list_pieces: "pieces-msg", render: "render-msg", sheet: "sheet-msg", movement: "movement-msg", springs: "movement-msg",
  face_patch: "patch-msg", detector_install: "setup-msg", vendor: "setup-msg", install_blender: "setup-msg", install_addon: "setup-msg",
  install_package: "setup-msg", update: "setup-msg" };

// The steps a build will go through and how long each took last time: the bar's step count and ETA.
function plan(route, body) {
  const started = Date.now();
  const label = route !== "run" ? null : body.only ? stepName(body.only) : body.from ? `build from ${stepName(body.from)}` : "build";
  if (route !== "run" || !S.status) return { route, label, started, steps: [], done: 0, current: null, going: new Set(), over: new Set() };
  const names = S.status.steps.map((s) => s.step);
  const first = names.indexOf(body.only || body.from || names[0]), last = body.only ? first : names.length - 1;
  const steps = S.status.steps.slice(first, last + 1).filter((s) => s.runs).map((s) => ({ step: s.step, seconds: s.seconds || null }));
  return { route, label, started, steps, done: 0, current: null, stepStarted: started, going: new Set(), over: new Set() };
}

async function poll(since) {
  polling = true;
  // not call(): a failed job's report has an "error" field, which call() throws on, and the poll
  // then stopped with the page stuck on "working" and the error never shown
  let state;
  try { state = await (await fetch(`/api/events?since=${since}`)).json(); }
  catch { setTimeout(() => poll(since), 3000); return; }
  if (state.running && !run) run = { route: state.running, started: Date.now(), steps: [], done: 0, current: null, going: new Set(), over: new Set() };
  for (const e of state.events) {
    // what was known to go wrong before the build started, said into its progress: no step began or ended
    if (e.status === "preflight") { $("events").textContent += `before the build, ${e.kind === "stop" ? "known to go wrong" : "worth knowing"}: ${e.reason} [${e.code}]\n`; continue; }
    // "ahead": one of the build's last steps, run beside the others, is over; how it ended is said in its turn
    if (e.status !== "ahead") $("events").textContent += `${e.step} ${e.status}${e.seconds ? ` (${e.seconds} s)` : ""}${e.reason ? ": " + e.reason : ""}\n`;
    // short of memory or disk, an untested Blender: said where it is seen, not only in the log
    if (e.status === "warning") { toast(e.reason, "info", { ms: 12000 }); continue; }
    if (!run) continue;
    if (e.status === "running") { run.current = e.step; run.going.add(e.step); run.stepStarted = Date.now(); continue; }
    run.going.delete(e.step);
    // only the steps the bar counts: one that is switched off says "skipped" and is not among them, and
    // each put the bar a step ahead of the build and took a step's time off what was left. And each once:
    // a step run ahead of its turn says it is over, then how it ended
    if (run.steps.some((s) => s.step === e.step) && !run.over.has(e.step)) { run.over.add(e.step); run.done += 1; }
  }
  $("busy").textContent = state.running ? `working: ${state.running}` : "idle";
  // the running step's own words: Blender says what it is at long before the step ends
  $("run-tail").hidden = !state.tail;
  if (state.tail) { $("run-tail").textContent = state.tail.split(/\r?\n/).slice(1).join("\n").trimEnd(); $("run-tail").scrollTop = 1e9; }
  if (state.running) {
    run.label ||= state.running;
    showBar(true); paintBar();
    setTimeout(() => poll(state.next), 1500);
    return;
  }
  polling = false;
  const finished = run;
  run = null; showBar(false);
  if (!finished) return;  // nothing was running when the page opened
  const took = Math.round((Date.now() - finished.started) / 1000);
  if (state.error) {
    $("events").textContent += `\n${state.error}`;
    // what happened and what to do, as the engine hands them over (a step that ended the run, a refusal
    // with a code); else the error's own first line, which is a sentence of the app's
    const f = state.failure;
    const said = f ? `${f.step ? `“${stepName(f.step)}”: ` : ""}${sentence(tell(f))} ${sentence(f.fix)}${f.code ? ` [${f.code}]` : ""}` : state.error.split("\n")[0];
    if (SAYS[finished.route]) say(SAYS[finished.route], said, true);
    toast(`${finished.label || "The run"} failed: ${said}`, "bad", { ms: 9000 });
  } else if (state.cancelled) toast("Stopped. The step that was running keeps what it last finished; run it again when you like.", "info");
  else toast(`${cap(finished.label || "done")}: finished in ${duration(took)}.`);
  if (took > 20) ended(state.error ? "failed" : state.cancelled ? "stopped" : "finished", finished.label);
  app.jobDone?.(state);
  if (S.manifestPath) await app.openProject(S.manifestPath);
  if (finished.route === "prepare" && !state.error && !state.cancelled) app.prepared?.();
  if (finished.route === "vendor") app.vendored?.();
  for (const done of waiting.splice(0)) done(state);
}

const cap = (t) => t[0].toUpperCase() + t.slice(1);
const duration = (s) => s < 90 ? `${s} s` : `${Math.round(s / 60)} min`;

function showBar(on) {
  $("runbar").hidden = !on;
  // one run at a time: every button that starts one waits
  document.querySelectorAll("[data-job]").forEach((b) => b.disabled = on);
  // called at every poll: a ticker started anew each time never ticked evenly, and the clock jumped
  if (!on) { clearInterval(ticker); ticker = null; $("run-stop").disabled = false; $("run-stop").textContent = "Stop"; }
  else ticker ||= setInterval(paintBar, 250);
}
function paintBar() {
  if (!run) return;
  const elapsed = Math.floor((Date.now() - run.started) / 1000);
  $("run-label").textContent = cap(run.label || run.route);
  const total = run.steps.length;
  // (the build's last steps run side by side when the computer has room: all that are going are named)
  const going = run.going.size > 1 ? [...run.going].map(stepName).join(", ") : run.current ? stepName(run.current) : "";
  $("run-step").textContent = going ? `${going}${total ? ` · step ${Math.min(run.done + 1, total)} of ${total}` : ""}` : "";
  const bar = $("run-bar");
  bar.classList.toggle("indeterminate", !total);
  if (total) bar.firstElementChild.style.width = `${Math.round(100 * run.done / total)}%`;
  // time left: the remaining steps' last times, less what the running one has had
  const remaining = run.steps.slice(run.done), known = remaining.every((s) => s.seconds);
  let eta = null;
  if (total && known) eta = Math.max(0, remaining.reduce((t, s) => t + s.seconds, 0) - (Date.now() - run.stepStarted) / 1000);
  $("run-time").textContent = `${clock(elapsed)}${eta != null ? ` · about ${duration(Math.round(eta))} left` : ""}`;
}
const clock = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

$("run-stop").onclick = async () => {
  $("run-stop").disabled = true; $("run-stop").textContent = "Stopping…";
  try { await call("cancel", {}); } catch (e) { toast(e.message, "bad"); }
};
$("run-log").onclick = () => app.go("build");

// ---------- telling the user a long run ended: they have usually gone to do something else
const NOTIFY_KEY = "vtuber-maker-notify";
let notify = false;
try { notify = localStorage.getItem(NOTIFY_KEY) === "1"; } catch {}
$("notify").checked = notify;
$("notify").onchange = async () => {
  notify = $("notify").checked;
  try { localStorage.setItem(NOTIFY_KEY, notify ? "1" : "0"); } catch {}
  if (notify && "Notification" in window && Notification.permission === "default") await Notification.requestPermission();
};
function ended(how, label) {
  const title = document.title.replace(/^[✓✗■] /, "");
  document.title = `${how === "finished" ? "✓" : how === "failed" ? "✗" : "■"} ${title}`;
  addEventListener("focus", () => { document.title = title; }, { once: true });
  if (!notify) return;
  try { if (Notification.permission === "granted") new Notification("OC2VRM", { body: `${cap(label || "the run")} ${how}.` }); } catch {}
  chime(how === "finished");
}
function chime(good) {
  try {
    const audio = new AudioContext(), t = audio.currentTime;
    (good ? [660, 880] : [440, 330]).forEach((f, i) => {
      const o = audio.createOscillator(), g = audio.createGain();
      o.frequency.value = f; o.connect(g); g.connect(audio.destination);
      g.gain.setValueAtTime(0.0001, t + i * 0.16); g.gain.exponentialRampToValueAtTime(0.15, t + i * 0.16 + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + i * 0.16 + 0.3);
      o.start(t + i * 0.16); o.stop(t + i * 0.16 + 0.32);
    });
  } catch {}
}

// A run started before the page opened (a reload mid-build) shows in the bar too.
export function resume() { if (!polling) poll(0); }
