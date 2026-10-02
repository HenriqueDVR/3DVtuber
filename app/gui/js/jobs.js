// Long jobs (a build, a listing, a render): start one, follow it in the run bar, stop it, say when it ends.
import { $, S, app, call, say, stepName } from "./core.js";
import { toast } from "./ui.js";

let polling = false, run = null, ticker = null;

export async function job(route, body = {}) {
  try { await call(route, body); } catch (e) { toast(e.message, "bad"); $("events").textContent += `\n${e.message}`; return; }
  $("events").textContent = "";
  run = plan(route, body);
  showBar(true);
  if (!polling) poll(0);
}
app.job = job;

// The steps a build will go through and how long each took last time: the bar's step count and ETA.
function plan(route, body) {
  const started = Date.now();
  const label = route !== "run" ? null : body.only ? stepName(body.only) : body.from ? `build from ${stepName(body.from)}` : "build";
  if (route !== "run" || !S.status) return { route, label, started, steps: [], done: 0, current: null };
  const names = S.status.steps.map((s) => s.step);
  const first = names.indexOf(body.only || body.from || names[0]), last = body.only ? first : names.length - 1;
  const steps = S.status.steps.slice(first, last + 1).filter((s) => s.runs).map((s) => ({ step: s.step, seconds: s.seconds || null }));
  return { route, label, started, steps, done: 0, current: null, stepStarted: started };
}

async function poll(since) {
  polling = true;
  // not call(): a failed job's report has an "error" field, which call() throws on, and the poll
  // then stopped with the page stuck on "working" and the error never shown
  let state;
  try { state = await (await fetch(`/api/events?since=${since}`)).json(); }
  catch { setTimeout(() => poll(since), 3000); return; }
  if (state.running && !run) run = { route: state.running, started: Date.now(), steps: [], done: 0, current: null };
  for (const e of state.events) {
    $("events").textContent += `${e.step} ${e.status}${e.seconds ? ` (${e.seconds} s)` : ""}${e.reason ? ": " + e.reason : ""}\n`;
    // short of memory or disk, an untested Blender: said where it is seen, not only in the log
    if (e.status === "warning") { toast(e.reason, "info", { ms: 12000 }); continue; }
    if (run && e.status === "running") { run.current = e.step; run.stepStarted = Date.now(); }
    else if (run) run.done += 1;
  }
  $("busy").textContent = state.running ? `working: ${state.running}` : "idle";
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
    for (const id of ["pieces-msg", "render-msg", "movement-msg", "patch-msg"]) say(id, state.error.split("\n")[0], true);
    toast(`${finished.label || "The run"} failed: ${state.error.split("\n")[0]}`, "bad", { ms: 9000 });
  } else if (state.cancelled) toast("Stopped. The step that was running keeps what it last finished; run it again when you like.", "info");
  else toast(`${cap(finished.label || "done")}: finished in ${duration(took)}.`);
  if (took > 20) ended(state.error ? "failed" : state.cancelled ? "stopped" : "finished", finished.label);
  app.jobDone?.(state);
  if (S.manifestPath) await app.openProject(S.manifestPath);
  if (finished.route === "prepare" && !state.error && !state.cancelled) app.prepared?.();
  if (finished.route === "vendor") app.vendored?.();
}

const cap = (t) => t[0].toUpperCase() + t.slice(1);
const duration = (s) => s < 90 ? `${s} s` : `${Math.round(s / 60)} min`;

function showBar(on) {
  $("runbar").hidden = !on;
  // one run at a time: every button that starts one waits
  document.querySelectorAll("[data-job]").forEach((b) => b.disabled = on);
  clearInterval(ticker);
  if (on) ticker = setInterval(paintBar, 1000);
  $("run-stop").disabled = false; $("run-stop").textContent = "Stop";
}
function paintBar() {
  if (!run) return;
  const elapsed = Math.round((Date.now() - run.started) / 1000);
  $("run-label").textContent = cap(run.label || run.route);
  const total = run.steps.length;
  $("run-step").textContent = run.current ? `${stepName(run.current)}${total ? ` · step ${Math.min(run.done + 1, total)} of ${total}` : ""}` : "";
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
  try { if (Notification.permission === "granted") new Notification("VTuber Maker", { body: `${cap(label || "the run")} ${how}.` }); } catch {}
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
