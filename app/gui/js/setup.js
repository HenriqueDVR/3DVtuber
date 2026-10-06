// Setup: what the app needs on this computer (engine.setup) and which Blender runs the builds.
import { $, S, app, call, esc, say } from "./core.js";
import { ask, toast } from "./ui.js";

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
  // found by the app (installed, or a portable copy in tools\blender) or added by hand below
  const where = (b) => b.missing ? `<span class="failed">not there any more</span>`
    : b.added ? "added by you" : b.portable ? "portable copy in tools\\blender" : "installed";
  $("setup-blenders").innerHTML = s.blenders.length ? s.blenders.map((b, i) => `<tr>
    <td>${esc(b.version || "?")} ${b.tested || b.missing ? "" : `<span class="chip warn" title="the test suite hasn't run on this series">untested</span>`}</td>
    <td class="msg" title="${esc(b.path)}">${where(b)}<span class="sub">${esc(b.path)}</span></td>
    <td>${b.missing ? "" : b.vrm_addon ? esc(b.vrm_addon) : `<span class="failed">not installed</span>`}</td>
    <td class="row" style="flex-wrap:nowrap">${b.missing ? "" : same(b.path, s.blender) ? `<span class="chip good">in use</span>`
      : `<button class="act alt small" data-use="${i}" ${b.vrm_addon ? "" : "disabled title=\"install the VRM Add-on in it first\""}>Use this one</button>`}
      ${b.added ? `<button class="act alt small" data-forget="${i}" title="take it off this list (the Blender itself stays where it is)">Remove</button>` : ""}</td></tr>`).join("")
    : `<tr><td colspan="4" class="msg">No Blender 4.2 or later found on this computer: install one, or add it below.</td></tr>`;
  $("setup-blenders").querySelectorAll("[data-use]").forEach((b) => b.onclick = () => use(s.blenders[+b.dataset.use]));
  $("setup-blenders").querySelectorAll("[data-forget]").forEach((b) => b.onclick = () => forget(s.blenders[+b.dataset.forget]));
  $("setup-auto").hidden = !s.chosen;
  // offered while something the pages need still comes from the internet
  $("setup-vendor").hidden = !s.checks.some((c) => c.name === "Internet" && c.state !== "ok");
  $("setup-detector").hidden = !s.checks.some((c) => c.name === "Auto face marking" && c.state !== "ok");
}
$("setup-detector").onclick = async () => {
  if (!await ask("Install auto face marking", { text: "Downloads PyTorch and an anime face detector into the app's tools folder: about 250 MB to download, "
    + "1.5 GB on disk, a few minutes. It needs Python 3.12 or newer on this computer. The first face it marks downloads another 273 MB. "
    + "Nothing else on the computer is changed; deleting tools\\face_detector removes it.", ok: "Download and install" })) return;
  const state = await app.job("detector_install", {});
  await loadSetup();
  if (state?.result?.python) toast("Installed. \"Place them for me\" on the Face page now works.", "good", { ms: 8000 });
};
$("setup-vendor").onclick = () => app.job("vendor", {});
// the pages already open still name the internet's copy: the kept one is read when a page loads
app.vendored = async () => {
  await loadSetup();
  if (!S.setup?.checks.some((c) => c.name === "Internet" && c.state !== "ok")) toast("Kept with the app. It is used from the next time the app opens.");
};

async function use(blender) {
  say("setup-msg", "asking that Blender for its version…");
  try {
    S.setup = await call("use_blender", { path: blender ? blender.path : "" });
    say("setup-msg", "");
    toast(blender ? `Builds now run on Blender ${blender.version}.` : "The app picks the Blender again.");
    renderSetup(); app.renderAll();
  } catch (e) { say("setup-msg", e.message, true); }
}
async function forget(blender) {
  try {
    S.setup = await call("forget_blender", { path: blender.path });
    toast("Taken off the list.");
    renderSetup(); app.renderAll();
  } catch (e) { say("setup-msg", e.message, true); }
}
// A Blender the app didn't find: its blender.exe, typed or browsed for. The engine checks it is one.
$("setup-add").onclick = async () => {
  const path = $("setup-add-path").value.trim();
  if (!path) return say("setup-msg", "Type the path of a blender.exe, or press Browse.", true);
  say("setup-msg", "asking that Blender for its version…");
  try {
    S.setup = await call("use_blender", { path });
    say("setup-msg", ""); $("setup-add-path").value = "";
    toast(`Builds now run on Blender ${S.setup.version}.`);
    renderSetup(); app.renderAll();
  } catch (e) { say("setup-msg", e.message, true); }
};
$("setup-auto").onclick = () => use(null);
$("setup-refresh").onclick = loadSetup;
