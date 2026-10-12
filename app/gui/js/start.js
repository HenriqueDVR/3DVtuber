// The Setup page's checks: what the app needs, in order, each with the button that gets it. A computer
// that lacks something a build needs opens on Setup (main.js); one that has it all sees it unasked once, the first start.
import { $, S, app, call, esc } from "./core.js";
import { ask, toast } from "./ui.js";
import { loadSetup } from "./setup.js";

const MARK = { ok: ["✓", "done"], warn: ["!", "warn"], bad: ["✗", "failed"] };
// What each button downloads and changes, said before it does: [title, text, the button's word]
const CONFIRM = {
  "install:pillow": ["Install Pillow", "Downloads the Pillow picture library from PyPI (about 3 MB) into the Python that runs this app. Nothing else is changed.", "Install"],
  "install:pywebview": ["Install pywebview", "Downloads pywebview from PyPI into the Python that runs this app. From the next start the app opens in its own window instead of a browser tab.", "Install"],
  install_blender: ["Get Blender", "Downloads Blender 5.2 LTS from blender.org (about 390 MB, checked against blender.org's own checksum) and unpacks it into the app's "
    + "tools folder (1.3 GB), then has it install the VRM Add-on. A few minutes. Nothing is installed on the computer: it is a copy the app keeps for itself, "
    + "and deleting the tools/blender folder removes it. A Blender you already have is not touched.", "Download"],
  install_addon: ["Install the VRM Add-on", "Blender downloads the “VRM format” add-on from its own extensions site (extensions.blender.org) and switches it on, "
    + "as Get Extensions does inside Blender. About a minute; nothing else in your Blender is changed.", "Download and install"],
};
const LABEL = { get_blender: "Get Blender…", install_blender: "Get it for me", vendor: "Keep a copy", install_addon: "Install it for me", "install:pillow": "Install it for me", "install:pywebview": "Install it" };

// The needed things first, in the order they are got; a check that isn't in the report is fine.
const NEEDED = [["Pictures", "Python's picture library (Pillow)", "Pillow is installed"],
  ["Blender", "Blender 4.2 or later", ""], ["VRM Add-on", "The VRM Add-on in that Blender", ""]];
const EXTRAS = ["Fully local", "Own window", "Auto face marking", "File dialogs", "Space", "Speed"];

function row(check, title, i) {
  const [mark, kind] = MARK[check.state];
  const button = check.action && check.state !== "ok" ? `<button class="act ${check.state === "bad" ? "" : "alt"} small" data-start="${i}">${LABEL[check.action] || "Fix"}</button>` : "";
  return `<div class="todo"><span><b class="${kind}">${mark} ${esc(title)}</b><br>${esc(check.text)}${check.fix && check.state !== "ok" ? `<br><span class="msg">${esc(check.fix)}</span>` : ""}</span>${button}</div>`;
}

export function renderStart() {
  const s = S.setup;
  if (!s) return;
  const by = Object.fromEntries(s.checks.map((c) => [c.name, c]));
  const rows = [];
  for (const [name, title, fine] of NEEDED) {
    // the add-on can't be looked for until there is a Blender to look in
    const check = by[name] || (name === "VRM Add-on" && by.Blender?.state === "bad"
      ? { state: "warn", text: "looked for once Blender is there", fix: "" } : { state: "ok", text: fine, fix: "" });
    rows.push([check, title]);
  }
  const extras = EXTRAS.filter((name) => by[name]).map((name) => [by[name], name]);
  const all = [...rows, ...extras];
  $("start-needed").innerHTML = rows.map(([c, t]) => row(c, t, all.findIndex(([x]) => x === c))).join("");
  $("start-extras").innerHTML = extras.length ? extras.map(([c, t]) => row(c, t, all.findIndex(([x]) => x === c))).join("")
    : `<p class="done">Nothing more to add.</p>`;
  $("start-warudo").innerHTML = s.warudo ? `<b class="done">✓ Warudo found</b><br>Finished avatars can be sent straight into its character list from the Build page.<br><span class="msg">${esc(s.warudo)}</span>`
    : `<b class="warn">! Warudo not found</b><br>Not needed to build. The Build page's “Send to Warudo” asks for its Characters folder the first time; VNyan and other apps load the .vrm file from wherever you keep it.`;
  // offered once everything needed is there and no character is open yet: the way on from a first start
  $("start-go").hidden = !s.ready || !!S.manifest;
  $("start-version").textContent = `App version ${s.app || "unknown"} · Python ${s.python} · ${s.version ? `Blender ${s.version}` : "no Blender"}`;
  document.querySelectorAll("[data-start]").forEach((b) => b.onclick = () => act(all[+b.dataset.start][0]));
}

async function act(check) {
  if (check.action === "get_blender") {
    // their browser, their download: the app only shows where
    window.open("https://www.blender.org/download/", "_blank", "noopener");
    toast("Install Blender from the page that opened (the LTS release is the tested one), then press “Check again”.", "good", { ms: 15000 });
    return;
  }
  if (check.name === "Auto face marking") { $("setup-detector").click(); return; }
  if (check.action === "vendor") { await app.job("vendor", {}); await loadSetup(); return; }
  const [title, text, ok] = CONFIRM[check.action] || [];
  if (!title || !await ask(title, { text, ok })) return;
  const state = check.action === "install_addon" ? await app.job("install_addon", {})
    : check.action === "install_blender" ? await app.job("install_blender", {})
    : await app.job("install_package", { name: check.action.split(":")[1] });
  await loadSetup();
  if (state?.result?.installed || state?.result?.ready) toast("Installed.", "good");
}

// (a first start that had to get Blender ends here: Setup isn't opened unasked again)
$("start-go").onclick = () => { if (!S.setup?.welcomed) call("welcome", {}).catch(() => {}); app.go("model"); };
app.renderStart = renderStart;
