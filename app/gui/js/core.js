// What every page shares: the open character's data, the API call, and the words for roles and bones.
export const $ = (id) => document.getElementById(id);
export const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const hex = (n) => `#${n.toString(16).padStart(6, "0")}`;
export const now = () => new Date().toLocaleTimeString();

// The open character. Pages read and replace these; nothing else holds a copy.
export const S = { manifestPath: null, manifest: null, status: null, preflight: null, listing: null, movement: null, springs: null,
  feedback: [], projects: [], runs: [], renders: [], setup: null };
// Functions one page calls on another, filled in by the module that owns them (go, job,
// openProject, ...): the pages import each other's work without importing each other.
export const app = {};

export async function call(route, body) {
  const options = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json", "X-VTuber-Maker": "1" },
    body: JSON.stringify({ manifest: S.manifestPath, ...body }) };
  const q = body === undefined && S.manifestPath ? `${route.includes("?") ? "&" : "?"}manifest=${encodeURIComponent(S.manifestPath)}` : "";
  let response;
  try { response = await fetch(`/api/${route}${q}`, options); }
  catch {
    // the browser's own words here are "Failed to fetch": the app's server isn't there to answer
    throw new Error("The app isn't answering. Its black window (OC2VRM.bat) must stay open while you use this page: "
      + "if it was closed, start OC2VRM.bat again and reload this page.");
  }
  const data = await response.json();
  // with the refusal's code when the engine named one (app/errors.py): what a page acts on, never the words
  if (data.error) throw Object.assign(new Error(data.error), { known: data.known || null });
  return data;
}
export async function refreshManifest() { S.manifest = await call("manifest"); }
// ... and, with it, what is known to go wrong with the next build (app/preflight.py): read anew whenever
// the status is, since nearly everything that changes one changes the other. Without it the page still works.
export async function refreshStatus() { [S.status, S.preflight] = await Promise.all([call("status"), call("preflight").catch(() => null)]); }
// "the arms hang at the sides" as a sentence of its own: the engine's words begin small and end open
export const sentence = (t) => { t = String(t ?? "").trim(); return t ? t[0].toUpperCase() + t.slice(1) + (/[.!?]$/.test(t) ? "" : ".") : ""; };

export function say(id, text, bad = false) {
  // an error written beside its button is remembered for a report made in the next minutes
  if (bad && text) S.lastError = { text, at: Date.now() };
  const el = $(id); if (!el) return; el.textContent = text; el.classList.toggle("bad", bad);
}

// A failure as the engine hands it over (app/errors.py `failure`: code, what, detail, fix, options), said.
// Nothing of a log is read here: the page once worked a failure's meaning out of its words, and told a
// tester who had marked the face to mark it. A failure with no code is one the app has no name for: its
// `what` says so, its `detail` is the log's last telling line, and the way on is a report.
export const tell = (f) => f ? `${f.what}${f.detail ? ` (${f.detail})` : ""}` : "";
// A piece the pose test listed and found right as it is (a cape the arms move free under): not a problem.
// (results written before the test said so itself are read by its words)
export const movesFine = (row) => row.expected ?? (!row.suggested && /as they should$/.test(row.why || ""));
export const stepName = (s) => s.replaceAll("_", " ");
// How long the steps from `from` on (all of them without it) took in this character's last build, in
// words ("1 min 40 s"); null when one of them has never run here. Their own times added up: where the
// computer has room the last steps run side by side, and a build is then quicker than this says.
export function tookLast(status, from) {
  const steps = status?.steps || [], first = from ? steps.findIndex((s) => s.step === from) : 0;
  const run = first < 0 ? [] : steps.slice(first).filter((s) => s.runs);
  if (!run.length || !run.every((s) => s.seconds)) return null;
  const total = Math.max(10, Math.round(run.reduce((sum, s) => sum + s.seconds, 0) / 10) * 10);
  return total < 60 ? `${total} s` : `${Math.floor(total / 60)} min${total % 60 ? ` ${total % 60} s` : ""}`;
}

// a file under out/ as the server hands it out
export const served = (path) => { const tail = String(path).replaceAll("\\", "/").split("/out/")[1]; return tail ? `/out/${tail}?t=${Date.now()}` : ""; };

export const ROLES = ["", "body", "hair", "hat", "swing_cloth", "appendage", "fitted_skirt", "shoulder_cloth", "follow_bone", "detail"];
// What each role means in practice, in the user's words: shown in the lists and on Parts/Movement.
export const ROLE_INFO = {
  "": ["automatic", "leave it to detection: skinned with the body, or hair/cloth if it looks like it"],
  body: ["body", "bends with the skeleton like skin; also keeps it out of hair and cloth physics"],
  hair: ["hair (swings)", "gets its own spring strands, so it sways when the head moves"],
  hat: ["hat (rigid on head)", "stuck to the head as one solid piece: turns with the head, never bends"],
  swing_cloth: ["swinging cloth", "cape, coat tail, loose skirt: hangs and swings with physics"],
  appendage: ["wing / tail", "grows from the back or hips on its own bones and sways"],
  fitted_skirt: ["fitted skirt", "stays on the hips; cut free of hands resting on it so they don't drag it"],
  shoulder_cloth: ["mantle / shoulder cloth", "drapes both shoulders: bends with the upper arms, collar and hem stay"],
  follow_bone: ["follows one bone", "moves rigidly with one bone you pick, e.g. a hanging sleeve with the upper arm, a bracelet with the hand"],
  detail: ["rides another piece", "small bit (cuff trim, button) that moves with the piece you pick"],
};
export const roleName = (r) => ROLE_INFO[r || ""]?.[0] ?? r;
export const ROLE_COLOUR = { "": 0x9a9a9a, body: 0xd8c3a5, hair: 0xb07cff, hat: 0xffb347, swing_cloth: 0x4fc3f7,
  fitted_skirt: 0x81c784, shoulder_cloth: 0x4db6ac, follow_bone: 0xff8a65, detail: 0xf06292, appendage: 0xba68c8 };
export const CORE = ["", "Hips", "Spine", "Chest", "UpperChest", "Neck", "Head"];
export const BONES = ["LeftUpperArm", "LeftLowerArm", "LeftHand", "RightUpperArm", "RightLowerArm", "RightHand",
  "Head", "Neck", "Chest", "UpperChest", "Spine", "Hips", "LeftUpperLeg", "LeftLowerLeg", "RightUpperLeg", "RightLowerLeg"];
// Where a role takes effect: rerunning from here is enough for it to count.
export const ROLE_STAGE = (role) => ["hat", "hair", "swing_cloth", "body", "appendage", ""].includes(role || "") ? "build" : "pieces";

// the manifest piece a listed row is; one named since carries the row's own point
export function samePiece(piece, row) {
  if (piece.region?.length) return false;  // part of a piece: it has its own row
  return row.manifest?.name === piece.name ||
    (piece.count === row.count && Math.hypot(...piece.point.map((v, i) => v - row.point[i])) < 1e-6);
}
export const namedFor = (row) => S.manifest?.pieces.find((p) => samePiece(p, row));

// The finished .vrm: the last step that exported one and is done.
export function lastVrm() {
  const step = [...(S.status?.steps || [])].reverse().find((s) => s.vrm && s.status === "done");
  return step ? step.vrm : null;
}
