// Setup: what the app needs on this computer (engine.setup) and which Blender runs the builds.
import { $, S, app, call, esc, say } from "./core.js";
import { toast } from "./ui.js";

const MARK = { ok: ["✓", "done"], warn: ["!", "warn"], bad: ["✗", "failed"] };
const same = (a, b) => String(a || "").toLowerCase().replaceAll("/", "\\") === String(b || "").toLowerCase().replaceAll("/", "\\");

export async function loadSetup() {
  try { S.setup = await call("setup"); } catch (e) { say("setup-msg", e.message, true); return; }
  renderSetup();
  app.renderAll();
}

function renderSetup() {
  const s = S.setup, bad = s.checks.filter((c) => c.state === "bad").length;
  $("setup-chip").textContent = bad ? `${bad} to fix` : "ready";
  $("setup-chip").className = `chip ${bad ? "bad" : "good"}`;
  $("setup-checks").innerHTML = `<dl class="facts">${s.checks.map((c) => `<dt class="${MARK[c.state][1]}">${MARK[c.state][0]} ${esc(c.name)}</dt>
    <dd>${esc(c.text)}${c.fix ? `<span class="sub msg">${esc(c.fix)}</span>` : ""}</dd>`).join("")}</dl>`;
  $("setup-blenders").innerHTML = s.blenders.length ? s.blenders.map((b, i) => `<tr>
    <td>${esc(b.version)} ${b.tested ? "" : `<span class="chip warn" title="the test suite hasn't run on this series">untested</span>`}</td>
    <td class="msg" title="${esc(b.path)}">${b.portable ? "portable copy in tools\\blender" : "installed"}</td>
    <td>${b.vrm_addon ? esc(b.vrm_addon) : `<span class="failed">not installed</span>`}</td>
    <td>${same(b.path, s.blender) ? `<span class="chip good">in use</span>`
      : `<button class="act alt small" data-use="${i}" ${b.vrm_addon ? "" : "disabled title=\"install the VRM Add-on in it first\""}>Use this one</button>`}</td></tr>`).join("")
    : `<tr><td colspan="4" class="msg">No Blender 4.2 or later found on this computer.</td></tr>`;
  $("setup-blenders").querySelectorAll("[data-use]").forEach((b) => b.onclick = () => use(s.blenders[+b.dataset.use]));
  $("setup-auto").hidden = !s.chosen;
}

async function use(blender) {
  say("setup-msg", "asking that Blender for its version…");
  try {
    S.setup = await call("use_blender", { path: blender ? blender.path : "" });
    say("setup-msg", "");
    toast(blender ? `Builds now run on Blender ${blender.version}.` : "The app picks the Blender again.");
    renderSetup(); app.renderAll();
  } catch (e) { say("setup-msg", e.message, true); }
}
$("setup-auto").onclick = () => use(null);
$("setup-refresh").onclick = loadSetup;
