// The app: opening a character and drawing every page from it.
import { $, S, app, call, esc, refreshStatus } from "./core.js";
import { toast } from "./ui.js";
import { go, renderNav } from "./nav.js";
import { resume } from "./jobs.js";
import { renderHome, renderSurvey } from "./home.js";
import { loadParts, renderPieces, colourPieces } from "./parts.js";
import { setGhost, showJoints } from "./joints.js";
import { loadFace, renderFace } from "./face.js";
import { renderOptions } from "./options.js";
import { renderSteps, loadRuns } from "./build.js";
import { loadMovement, loadSprings } from "./movement.js";
import { loadFeedback, loadRenders, loadSheets } from "./review.js";
import { loadSetup } from "./setup.js";
import { resetPaint } from "./paint.js";

const LAST = "vtuber-maker-project";

export function renderAll() {
  renderNav(); renderHome(); renderSurvey().catch(failed); renderPieces(); colourPieces(); renderFace(); renderOptions(); renderSteps(); showJoints();
}
app.renderAll = renderAll;
app.refreshStatus = async () => { await refreshStatus(); renderSteps(); renderNav(); };
const failed = (e) => toast(e.message, "bad");

async function refreshProjects() {
  const listed = await call("projects");
  S.projects = listed.projects; S.hidden = listed.hidden || [];
  $("project").innerHTML = `<option value="">— new character —</option>` +
    S.projects.map((p) => `<option value="${esc(p.manifest)}">${esc(p.name)}</option>`).join("");
  $("project").value = S.manifestPath || "";
  renderHome(); renderNav();
}

async function loadProjects(select) {
  await refreshProjects();
  // a refresh reopens the character that was open: it showed an empty app, as if the work were lost
  let last = null;
  try { last = localStorage.getItem(LAST); } catch {}
  const asked = new URLSearchParams(location.search).get("character");
  const named = asked && S.projects.find((p) => p.name === asked)?.manifest;
  const pick = select || named || (last && S.projects.some((p) => p.manifest === last) ? last : null);
  await openProject(pick);
}
app.loadProjects = loadProjects;
$("project").onchange = () => openProject($("project").value || null);

async function openProject(path) {
  S.manifestPath = path || null; S.listing = null; S.facePlaced = 0;
  $("project").value = S.manifestPath || "";
  try { path ? localStorage.setItem(LAST, path) : localStorage.removeItem(LAST); } catch {}
  if (!path) {
    Object.assign(S, { manifest: null, status: null, movement: null, springs: null, feedback: [], runs: [], renders: [] });
    await loadParts(null); setGhost(null); resetPaint();
    renderAll();
    await Promise.all([loadFace(), loadMovement(null), loadSprings(), loadFeedback(), loadRuns(), loadRenders(), loadSheets()]).catch(failed);
    return;
  }
  try {
    [S.manifest, S.status] = await Promise.all([call("manifest"), call("status")]);
    resetPaint();
    const pieces = await call("pieces");
    S.listing = pieces.listing;
    const gltf = await loadParts(pieces.glb);
    setGhost(gltf?.scene);
    renderAll();
    await Promise.all([loadFace(), loadMovement(gltf), loadSprings(), loadFeedback(), loadRuns(), loadRenders(), loadSheets()].map((p) => p.catch(failed)));
    renderNav();
    refreshProjects().catch(failed);
  } catch (e) { failed(e); }
}
app.openProject = openProject;

// Before/after pictures from characters already built, in out/examples/: private art, so
// they're read where they lie and never copied into the app. A picture that isn't there is left out.
const EXAMPLES = {
  follow_bone: [["/out/examples/follow_bone_before.png", "before: the sleeve tabs stay at the hips when the arms go up"],
    ["/out/examples/follow_bone_after.png", "after giving them “follows one bone” (upper arm): they go up with the arms"]],
  eyes: [["/out/examples/eyes_iris_ahead.png", "moving iris: looking ahead"], ["/out/examples/eyes_iris_side.png", "moving iris: looking to the side"],
    ["/out/examples/eyes_slide_ahead.png", "sliding patches: looking ahead"], ["/out/examples/eyes_slide_side.png", "sliding patches: looking to the side"]],
  mouth: [["/out/examples/mouth_before.png", "before: saying “a” with the model's own mouth"],
    ["/out/examples/mouth_after.png", "after: redrawn, free mouth, size × 1.3"]],
};
document.querySelectorAll("[data-examples]").forEach((host) => {
  host.innerHTML = (EXAMPLES[host.dataset.examples] || []).map(([src, text]) =>
    `<figure><img src="${src}" alt="${esc(text)}" loading="lazy" onerror="this.closest('figure').remove()"><figcaption>${esc(text)}</figcaption></figure>`).join("");
});

go(location.hash.slice(1) || "home");
loadProjects().then(resume).then(loadSetup).catch(failed);
