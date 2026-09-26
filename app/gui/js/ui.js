// Toasts, the one dialog, the picture viewer and undo: shared by every page.
import { $, esc } from "./core.js";

// A short note in the corner. Saved-confirmations go here; errors stay next to what failed.
export function toast(text, kind = "good", { action, ms = 4500 } = {}) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `<span>${esc(text)}</span>`;
  if (action) {
    const b = document.createElement("button");
    b.className = "act alt small"; b.textContent = action.label;
    b.onclick = () => { el.remove(); action.run(); };
    el.append(b);
  }
  $("toasts").append(el);
  setTimeout(() => el.remove(), action ? ms + 3000 : ms);
}

// A question in the page's own dialog; native prompt() and confirm() block pywebview and look
// like an error box. Resolves to the text typed (with `input`), true for OK, or null.
export function ask(title, { text = "", input = false, value = "", ok = "OK", danger = false } = {}) {
  const dialog = $("ask");
  $("ask-title").textContent = title;
  $("ask-text").textContent = text; $("ask-text").hidden = !text;
  $("ask-input").hidden = !input; $("ask-input").value = value;
  $("ask-ok").textContent = ok; $("ask-ok").classList.toggle("danger", danger);
  return new Promise((done) => {
    // answered from the buttons, not the close event: that waits for the next painted frame
    const finish = (ok) => { dialog.close(); done(ok ? (input ? $("ask-input").value.trim() : true) : null); };
    dialog.querySelector("form").onsubmit = (e) => { e.preventDefault(); finish(true); };
    $("ask-cancel").onclick = () => finish(false);
    dialog.oncancel = (e) => { e.preventDefault(); finish(false); };
    dialog.showModal();
    if (input) { $("ask-input").focus(); $("ask-input").select(); }
  });
}

// Pictures full size, with the arrow keys going through the set they came from.
let shown = [], at = 0;
export function lightbox(pictures, index = 0) {
  shown = pictures; at = index;
  const box = $("lightbox");
  box.hidden = false;
  paint();
  box.querySelector(".close").focus();
}
function paint() {
  const box = $("lightbox"), [src, caption] = shown[at];
  box.querySelector("img").src = src;
  box.querySelector(".cap span").textContent = `${caption || ""}${shown.length > 1 ? `  ·  ${at + 1} of ${shown.length}` : ""}`;
  box.querySelectorAll(".nav").forEach((b) => b.hidden = shown.length < 2);
}
const step = (d) => { at = (at + d + shown.length) % shown.length; paint(); };
$("lightbox").querySelector(".prev").onclick = () => step(-1);
$("lightbox").querySelector(".next").onclick = () => step(1);
$("lightbox").querySelector(".close").onclick = () => { $("lightbox").hidden = true; };
$("lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox") $("lightbox").hidden = true; });
addEventListener("keydown", (e) => {
  if ($("lightbox").hidden) return;
  if (e.key === "Escape") $("lightbox").hidden = true;
  if (e.key === "ArrowLeft") step(-1);
  if (e.key === "ArrowRight") step(1);
});
// Any picture marked zoom opens with its siblings (the images in the same container).
document.addEventListener("click", (e) => {
  const img = e.target.closest("img.zoom, .imgs img, .examples img, .feedback img, .spots img");
  if (!img) return;
  e.preventDefault();
  const set = [...img.closest(".imgs, .examples, .feedback, .spots, .compare, section")
    .querySelectorAll("img.zoom, .imgs img, .examples img, .feedback img, .spots img")];
  lightbox(set.map((i) => [i.src, i.alt]), set.indexOf(img));
});

// Undo: each change the user makes pushes the call that puts it back.
const undos = [];
export function pushUndo(label, run) {
  undos.push({ label, run });
  if (undos.length > 50) undos.shift();
}
export async function undo() {
  const last = undos.pop();
  if (!last) return toast("Nothing to undo", "info");
  try { await last.run(); toast(`Undone: ${last.label}`, "info"); } catch (e) { toast(`Couldn't undo: ${e.message}`, "bad"); }
}
// a toast that offers undo for the change it reports
export const saved = (text) => toast(text, "good", { action: { label: "Undo", run: undo }, ms: 6000 });
addEventListener("keydown", (e) => {
  if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "z" || e.shiftKey) return;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName) && document.activeElement.type !== "checkbox") return;
  e.preventDefault(); undo();
});
