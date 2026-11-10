// The sidebar: pages, what each has left (badges), and the rebuild cue once something changed.
import { $, S, app, movesFine, stepName } from "./core.js";

export const ORDER = ["model", "parts", "joints", "face", "options", "build", "movement", "review"];
export const TITLES = { quick: "Quick build", model: "Model", parts: "Parts", joints: "Joints", face: "Face", options: "Options", build: "Build", movement: "Movement", review: "Review" };
// Things a page does when it shows (a view that needs the listing, a list to reload), and when it
// is left for another (a camera to let go of).
export const onShow = {}, onLeave = {};
let at = null;

export function go(name) {
  if (!document.getElementById(name)?.classList.contains("page")) name = "home";
  if (at && at !== name) onLeave[at]?.();
  at = name;
  document.querySelectorAll(".page").forEach((s) => s.classList.toggle("on", s.id === name));
  document.querySelectorAll("#steps button").forEach((b) => b.classList.toggle("on", b.dataset.go === name));
  if (location.hash.slice(1) !== name) history.replaceState(null, "", `#${name}`);
  $("main").scrollTop = 0;
  onShow[name]?.();
  app.shown?.(name);
}
app.go = go;
document.querySelectorAll("#steps button").forEach((b) => b.onclick = () => go(b.dataset.go));
addEventListener("hashchange", () => go(location.hash.slice(1)));

// The pages with nothing to do for a finished avatar (.vrm), which keeps its own pieces, joints and face.
const KEPT = ["parts", "joints", "face", "paint", "movement"];
// The page after `name`: the next in order, past the ones a finished avatar has no use for; and the one before it.
// Paint is a side page of Face's: it goes on to Options and back to Face.
const at_ = (name) => ORDER.indexOf(name === "paint" ? "face" : name);
export const skipped = (page) => S.status?.improving && KEPT.includes(page);
export const after = (name) => ORDER.slice(at_(name) + 1).find((page) => !skipped(page));
export const before = (name) => name === "paint" ? "face" : ORDER.slice(0, Math.max(at_(name), 0)).reverse().find((page) => !skipped(page));

// A step that ran and has nothing more to give: done, failed, or with nothing to do on this model
// (no ahoge on the head). The last read as pending, and "Changes not built yet" never went away.
const settled = (s) => ["done", "failed"].includes(s.status) || s.status.startsWith("nothing");

// The steps the next run should cover: the first that isn't done, when something was built already.
export function nextRun(status) {
  if (!status) return null;
  const pending = status.steps.filter((s) => s.runs && !settled(s));
  const built = status.steps.some((s) => s.step !== "survey" && ["done", "stale", "failed"].includes(s.status));
  return built && pending.length ? { from: pending[0].step, count: pending.length, stale: pending[0].status === "stale" } : null;
}

// Each page's badge and chip: what's done there and what's left, from what's already loaded.
export function renderNav() {
  const m = S.manifest, st = S.status, steps = Object.fromEntries((st?.steps || []).map((s) => [s.step, s]));
  const flagged = S.movement?.pieces?.filter((row) => !movesFine(row)).length;
  const failed = (st?.steps || []).filter((s) => s.runs && s.status === "failed").length;
  const pending = (st?.steps || []).filter((s) => s.runs && !settled(s)).length;
  const moved = Object.keys(m?.joints || {}).length;
  const facePlaced = S.facePlaced ?? 0;
  // Setup says so only when something stops builds: a missing Blender or VRM Add-on
  const setup = S.setup && !S.setup.ready ? ["to fix", "bad"] : S.update?.available ? ["update", "warn"] : ["", ""];
  const badges = !m ? { home: [S.projects.length || "", ""], setup } : {
    home: [S.projects.length, ""], setup,
    model: st?.improving ? ["a finished avatar", "info"] : steps.survey?.status === "done" ? ["surveyed", "good"] : ["survey", "warn"],
    parts: st?.improving ? ["", ""] : !S.listing ? ["not listed", "warn"] : [`${m.pieces.length} role${m.pieces.length === 1 ? "" : "s"}`, m.pieces.length ? "info" : ""],
    joints: !S.listing ? ["", ""] : moved ? [`${moved} moved`, "info"] : ["detected", ""],
    face: st?.improving ? ["", ""] : m.face ? ["marked", "good"] : facePlaced ? [`${facePlaced} placed`, "warn"] : ["to do", "warn"],
    build: st?.verdict?.ready && !pending ? ["ready", "good"] : failed ? [`${failed} failed`, "bad"]
      : pending && nextRun(st) ? [`${pending} to build`, "warn"] : steps.build?.status === "done" ? ["built", "info"] : ["", ""],
    movement: S.movement ? (flagged ? [`${flagged} flagged`, "warn"] : ["clear", "good"]) : ["", ""],
    paint: m.paint ? ["painted", "info"] : ["", ""],
    review: S.status?.approved ? ["marked good", "done"] : ["", ""],
  };
  document.querySelectorAll("[data-badge]").forEach((el) => {
    const [text, kind] = badges[el.dataset.badge] || ["", ""];
    el.textContent = text; el.className = `badge ${kind}`;
  });
  document.querySelectorAll("[data-chip]").forEach((el) => {
    const [text, kind] = badges[el.dataset.chip] || ["", ""];
    el.textContent = text; el.className = `chip ${kind}`;
  });
  const done = { model: steps.survey?.status === "done" || !!st?.improving, parts: !!S.listing, face: !!m?.face,
    build: !!st?.verdict?.ready && !pending, movement: !!S.movement && !flagged };
  document.querySelectorAll("#steps button[data-go]").forEach((b) => {
    b.querySelector(".n").classList.toggle("done", !!(m && done[b.dataset.go]));
    // a page that can't do anything yet says why, instead of opening empty
    const why = !m && !["home", "model", "help", "setup"].includes(b.dataset.go) ? "open or create a character first"
      // a finished avatar keeps its own pieces, joints, face and pictures: those pages have nothing to do
      : st?.improving && KEPT.includes(b.dataset.go) ? "a finished avatar (.vrm) keeps its own: nothing to do here"
      : b.dataset.go === "joints" && !S.listing ? "list the pieces first (Parts)"
      : b.dataset.go === "movement" && steps.pieces?.status !== "done" && steps.build?.status !== "done" ? "build first" : "";
    b.classList.toggle("locked", !!why); b.title = why;
  });
  // which Blender runs the builds, a click from changing it
  $("nav-blender").textContent = !S.setup ? "" : S.setup.version ? `Blender ${S.setup.version}` : "no Blender found";
  $("nav-blender").className = S.setup && !S.setup.version ? "failed" : "";
  const next = nextRun(st);
  $("rebuild").hidden = !next;
  if (next) {
    $("rebuild-text").textContent = `${next.count} step${next.count > 1 ? "s" : ""} to rebuild, from “${stepName(next.from)}”${next.stale ? ": something before it changed" : ""}.`;
    $("rebuild-run").textContent = `Rebuild from ${stepName(next.from)}`;
    $("rebuild-run").onclick = () => app.job("run", { from: next.from });
  }
  app.navDrawn?.();
}
app.renderNav = renderNav;
