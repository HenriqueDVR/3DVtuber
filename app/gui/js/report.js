// Telling the developers: one dialog, opened from wherever something went wrong. Nothing is sent
// until Send is pressed in it; "Only make the zip" writes the same report and sends nothing.
import { $, S, app, call } from "./core.js";
import { ask, toast } from "./ui.js";

let about = "", made = null, busy = false;

// `context`: what the page was showing (an error's words). Without one, the last error the app
// gave in the last few minutes, wherever it was written.
export function reportDialog(context = "") {
  // one is on its way: closing the dialog doesn't stop it, and opening it again must not start a second
  if (busy) { $("report-dialog").showModal(); return; }
  const recent = S.lastError && Date.now() - S.lastError.at < 5 * 60 * 1000 ? S.lastError.text : "";
  about = String(context || recent || "").trim();
  made = null;
  const open = !!S.manifestPath;
  $("rd-who").textContent = open ? `About ${S.manifest?.name || "this character"}` : "No character is open: this tells them about the app and this computer.";
  $("rd-about").hidden = !about; $("rd-about").textContent = about ? `They will see: ${about.split("\n")[0].slice(0, 300)}` : "";
  $("rd-choices").hidden = !open;
  $("rd-out").hidden = true; $("rd-out").textContent = ""; $("rd-show").hidden = true;
  $("rd-send").disabled = $("rd-zip").disabled = false;
  $("report-dialog").showModal();
  $("rd-note").focus();
}
app.report = reportDialog;

const body = () => ({ note: $("rd-note").value.trim(), about,
  pictures: !!S.manifestPath && $("rd-pictures").checked, model: !!S.manifestPath && $("rd-model").checked });

function said(report, sent) {
  made = report.report;
  $("rd-out").textContent = `${sent ? "Sent to the developers. Thank you." : "Made; nothing was sent."}\n${report.megabytes} MB, ${report.files} files: ${report.holds.join(", ")}.` +
    (report.left_out.length ? `\nLeft out: ${report.left_out.join(", ")}.` : "") + `\n${sent ? "A copy is kept in" : "It is in"} ${report.report}`;
  if (sent) $("rd-note").value = "";
}

async function run(route) {
  busy = true;
  $("rd-send").disabled = $("rd-zip").disabled = true;
  $("rd-out").hidden = false; $("rd-out").textContent = route === "report_send" ? "Sending…" : "Making the zip…";
  try {
    let report;
    try { report = await call(route, body()); }
    catch (e) {
      if (route !== "report_send" || !/no address/.test(e.message)) throw e;
      // a copy from before reports could be sent: the developers give the address once
      const address = await ask("Where do reports go?", { input: true, ok: "Keep and send",
        text: "This copy has no address to send reports to. Paste the one the developers gave you (it starts with https://discord.com/api/webhooks/)." });
      if (!address) throw e;
      await call("report_address", { address });
      report = await call(route, body());
    }
    said(report, route === "report_send");
    // the dialog was closed meanwhile: the report went all the same, and they are told so
    if (!$("report-dialog").open) toast(route === "report_send" ? "Report sent to the developers. Thank you." : "The report's zip is made; nothing was sent.");
  } catch (e) {
    $("rd-out").textContent = e.message;
    if (!$("report-dialog").open) toast(e.message, "info", { ms: 12000, action: { label: "Open", run: () => $("report-dialog").showModal() } });
  }
  busy = false;
  $("rd-show").hidden = !made; $("rd-send").disabled = $("rd-zip").disabled = false;
}

$("rd-send").onclick = () => run("report_send");
$("rd-zip").onclick = () => run("report");
$("rd-show").onclick = async () => { try { await call("reveal", { path: made }); } catch (e) { $("rd-out").textContent = e.message; } };
$("rd-cancel").onclick = () => $("report-dialog").close();
