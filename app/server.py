"""The desktop app's local server: the guided build's page and a JSON API over the engine.

    python -m app.server [--port 8322] [--no-window]

Serves app/gui/ and, read-only, the repository's model and output files, on 127.0.0.1 only:
the art is private. Opens a pywebview window when pywebview is installed, else the browser.
The API calls the same functions as the command line, so the app, the CLI and an agent all
build the same way; long runs happen on a thread and report progress through /api/events.
"""

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, unquote, urlparse

from . import cli, engine, paint, update, vendor
from .engine import REPO, Cancelled, Project, StepFailed

GUI = Path(__file__).resolve().parent / "gui"
# Only these repository folders are served, and only these kinds of file (.mjs, .wasm and .task:
# the face tracker's, when the app keeps its own copy, see vendor.py).
SERVED = ("out", "samples", "tools/work", "viewer")
KINDS = {".glb", ".vrm", ".png", ".jpg", ".jpeg", ".json", ".html", ".js", ".css", ".log", ".mjs", ".wasm", ".task", ".mp3", ".ogg", ".wav"}

mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf-binary", ".vrm")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")  # a browser only compiles wasm served as this

# The calls a GET may make: they only read. Everything else starts a run or changes a file and
# needs the POST with the page's header, which another site open in the browser can't send
# (an <img src=".../api/prune?manifest=..."> could otherwise delete a character's old runs).
READ_ONLY = {"projects", "setup", "update_check", "disk", "status", "manifest", "pieces", "renders", "sheets", "runs", "feedback_list",
             "face", "movement_result", "springs_result", "events", "paint"}
# How much of the running step's log the Progress pane shows.
LOG_TAIL = 1500


class Jobs:
    """The one long job (a build, a piece listing, a render) running at a time, and its events."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events, self.running, self.result, self.error, self.cancelled = [], None, None, None, False

    def start(self, label, work):
        with self.lock:
            if self.running:
                raise StepFailed(f"busy: {self.running} is still running")
            self.events, self.running, self.result, self.error, self.cancelled = [], label, None, None, False
            engine.CANCEL.clear()

        def go():
            try:
                self.result = work(self.events.append)
            except Cancelled:
                self.cancelled = True
            except Exception as error:  # reported to the page, not lost in a thread
                self.error = str(error) if isinstance(error, (StepFailed, ValueError)) else traceback.format_exc()
            finally:
                self.running = None
        threading.Thread(target=go, daemon=True).start()
        return {"started": label}

    def cancel(self):
        """Stops the running job: its Blender is killed within a couple of seconds, the lock freed."""
        if self.running:
            engine.CANCEL.set()
        return {"cancelling": self.running}

    def poll(self, since=0):
        return {"running": self.running, "events": self.events[since:], "next": len(self.events),
                "result": self.result, "error": self.error, "cancelled": self.cancelled, "tail": self.tail()}

    def tail(self):
        """The end of the running step's log: a three-minute step otherwise shows nothing until it ends."""
        log = next((e["log"] for e in reversed(self.events) if e.get("status") == "running" and e.get("log")), None)
        if not self.running or not log:
            return None
        try:
            return engine._log_tail(log, LOG_TAIL)
        except OSError:
            return None  # Blender hasn't opened it yet


JOBS = Jobs()


def _args(**kwargs):
    return SimpleNamespace(blender=None, **kwargs)


def api(route, query, body):
    m = body.get("manifest") or (query.get("manifest") or [None])[0]
    if route == "projects":
        return {"projects": [{"manifest": str(p), "name": _name(p), **_summary(p)} for p in _characters()],
                "hidden": [{"manifest": h, "name": _name(h)} for h in engine.hidden_characters()]}
    if route == "setup":
        return engine.setup()
    if route == "detector_install":
        return JOBS.start("installing the face detector", lambda emit: cli.install_detector(on_event=emit))
    if route == "vendor":
        return JOBS.start("keeping a copy for offline use", lambda emit: vendor.fetch(on_event=emit))
    # the first-run page's buttons: each a download the user asked for by pressing it
    if route == "install_blender":
        return JOBS.start("fetching Blender", lambda emit: engine.install_blender(on_event=emit))
    if route == "install_addon":
        return JOBS.start("installing the VRM Add-on", lambda emit: engine.install_vrm_addon(on_event=emit))
    if route == "install_package":
        engine.package_command(body.get("name"))  # refused here, not minutes into a job, when it isn't one of ours
        return JOBS.start(f"installing {body['name']}", lambda emit: engine.install_package(body["name"], on_event=emit))
    if route == "welcome":
        return engine.welcome(body.get("done", True))
    if route == "open_in":
        return engine.open_in(body.get("value"))
    # the Setup page's Update button: is a newer version published, get it, start again with it
    if route == "update_check":
        return update.check()
    if route == "update":
        def work(emit):
            try:
                return update.apply(on_event=emit)
            except update.UpdateFailed as error:
                raise StepFailed(str(error)) from None
        return JOBS.start("updating the app", work)
    if route == "restart":
        if JOBS.running:
            raise StepFailed(f"busy: {JOBS.running} is still running")
        # only a launcher starts the app again (OC2VRM.bat, oc2vrm.sh): without one the user does
        if not os.environ.get("OC2VRM_LAUNCHER"):
            return {"restarting": False}
        threading.Timer(0.5, lambda: os._exit(update.RESTART)).start()
        return {"restarting": True}
    if route in ("use_blender", "forget_blender"):
        if JOBS.running:
            raise StepFailed(f"busy: {JOBS.running} is still running")
        return engine.use_blender(body.get("path") or "") if route == "use_blender" else engine.forget_blender(body["path"])
    if route == "disk":
        return _disk()
    if route == "prune":
        if JOBS.running:
            raise StepFailed(f"busy: {JOBS.running} is still running")
        if not body.get("all") and not m:
            raise ValueError("prune: name a character, or all of them")
        chosen = _characters() if body.get("all") else [Path(m)]
        done = [Project(path).prune(dry=bool(body.get("dry"))) for path in chosen]
        return {"characters": done, "removed": sum(d["removed"] for d in done), "freed": sum(d["freed"] for d in done)}
    if route == "status":
        return cli.cmd_status(_args(manifest=Path(m)))
    if route == "manifest":
        return engine.manifest.load(m)
    if route == "pieces":
        listing = engine._read_json(Project(m).out / "pieces.json", None)
        return {"listing": listing, "glb": _url(Project(m).out / "pieces.glb")}
    if route == "init":
        if not body.get("name") or Path(body["name"]).name != body["name"]:
            raise ValueError("the name can't contain a path")
        return cli.cmd_init(_args(model=Path(body["model"]), name=body["name"], author=body.get("author") or "Unknown",
                                  manifest=None, face=Path(body["face"]) if body.get("face") else None,
                                  reference=Path(body["reference"]) if body.get("reference") else None,
                                  rig=Path(body["rig"]) if body.get("rig") else None))
    if route == "set":
        return cli.cmd_set(_args(manifest=Path(m), key=body["key"], value=json.dumps(body["value"])))
    if route == "set_path":
        if body.get("key") not in ("source", "reference", "face", "author", "rig"):
            raise ValueError("only source, reference, face, rig and author are set this way")
        data = engine.manifest.load(m)
        data[body["key"]] = body["value"] or None
        engine.manifest.save(data, m)
        return {"saved": body["key"]}
    if route == "joint":
        return cli.cmd_joint(_args(manifest=Path(m), key=body["key"], xyz=body.get("xyz") or [],
                                   clear=bool(body.get("clear"))))
    if route == "piece":
        return cli.cmd_piece(_args(manifest=Path(m), name=body["name"], role=body.get("role"), point=None, count=None,
                                   label=body.get("label"), bone=body.get("bone") or None,
                                   anchor=body.get("anchor") or None, tip=bool(body.get("tip")),
                                   remove=bool(body.get("remove")), region=body.get("region"), rename=body.get("rename")))
    if route == "list_pieces":
        return JOBS.start("listing pieces", lambda emit: cli.cmd_pieces(_args(manifest=Path(m))))
    if route == "prepare":
        return JOBS.start("getting it ready", lambda emit: cli.prepare(Path(m), on_event=emit))
    if route == "run":
        def work(emit):
            return Project(m).run(start=body.get("from"), stop=body.get("to"), only=body.get("only"), on_event=emit)
        return JOBS.start("build", work)
    if route == "regress":
        # every character with a finished build, in scratch folders: the A/B a change to the app needs
        return JOBS.start("rebuilding every character", lambda emit: cli.cmd_regress(
            _args(manifests=body.get("manifests") or [], keep=False, on_event=emit)))
    if route == "portraits":
        return JOBS.start("portraits", lambda emit: cli.cmd_portraits(_args(manifest=Path(m), on_event=emit)))
    if route == "render":
        return JOBS.start("render", lambda emit: _with_urls(cli.cmd_render(
            _args(manifest=Path(m), pose=body.get("pose", "raised"), step=body.get("step"), run=body.get("run"), size=1400))))
    if route == "renders":
        return {"renders": _renders(Project(m).out / "renders")}
    if route == "sheet":
        # a browser, not Blender, but a job all the same: it loads the whole .vrm beside whatever runs
        return JOBS.start("contact sheets", lambda emit: _sheet(Project(m), body.get("run")))
    if route == "sheets":
        return {"sheets": _sheets(Project(m).out / "renders")}
    if route == "runs":
        return {"runs": [dict(r, vrm_url=_url(r["vrm"]) if r["vrm"] else None) for r in Project(m).runs()]}
    if route == "compare":
        return cli.cmd_compare(_args(before=Path(body["before"]), after=Path(body["after"])))
    if route == "cancel":
        return JOBS.cancel()
    if route == "reveal":
        return _reveal(Path(body["path"]))
    if route == "send_warudo":
        if "folder" in body:
            engine.use_warudo(body["folder"])
        return Project(m).send_to_warudo()
    if route == "movement":
        return JOBS.start("movement check", lambda emit: cli.cmd_movement(_args(manifest=Path(m), step=body.get("step"))))
    if route == "movement_result":
        return {"result": engine._read_json(Project(m).out / "movement.json", None)}
    if route == "springs":
        return JOBS.start("spring check", lambda emit: cli.cmd_springs(_args(manifest=Path(m), step=body.get("step"))))
    if route == "springs_result":
        return {"result": engine._read_json(Project(m).out / "springs.json", None)}
    if route == "face":
        project = Project(m)
        face = engine.manifest.path_of(project.data, "face")
        source = engine.manifest.path_of(project.data, "source")
        # the page views glTF only: an OBJ or FBX source shows once the face patch (a .glb) replaces it
        viewable = source is not None and source.suffix.lower() in (".glb", ".vrm")
        return {"points": engine._read_json(face, None) if face else None,
                "model": f"/api/model?key=source&manifest={quote(str(project.manifest_path))}" if viewable else None}
    if route == "paint":
        state = cli.paint_state(Path(m))
        at = f"manifest={quote(str(Path(m).resolve()))}"
        for picture in state["pictures"]:
            picture["url"] = f"/api/overlay?{at}&name={quote(picture['overlay'])}" if picture["painted"] else None
            picture["draft_url"] = f"/api/overlay?{at}&name={quote(picture['overlay'])}&draft=1" if picture["draft"] else None
        return dict(state, model=f"/api/model?key=source&{at}" if state["paintable"] else None)
    if route == "paint_save":
        if JOBS.running:
            # a build may be reading the painted copy this would write again
            raise StepFailed(f"busy: {JOBS.running} is still running")
        return cli.save_paint(Path(m), body.get("images") or {})
    if route == "paint_draft":
        return cli.save_paint_draft(Path(m), body.get("images") or {})
    if route == "face_patch":
        return JOBS.start("making the face patch", lambda emit: cli.cmd_face_patch(_args(manifest=Path(m))))
    if route == "face_guess":
        # a job: the detector takes a GB of memory, and beside a running Blender that hung the machine
        return JOBS.start("finding the face", lambda emit: {"points": cli.guess_face(Path(m))})
    if route == "face_save":
        return cli.save_face(Path(m), body["points"])
    if route == "feedback":
        return cli.save_feedback(Path(m), body.get("note", ""), body.get("images") or [],
                                 checks=body.get("checks") or [], verdict=body.get("verdict") or None)
    if route == "feedback_list":
        return {"feedback": [dict(f, urls=[_url(Path(f["folder"]) / n) for n in f["screenshots"]]) for f in cli.list_feedback(Path(m))]}
    if route == "rename":
        return cli.rename(Path(m), (body.get("name") or "").strip())
    if route == "hide":
        return engine.hide_character(Path(body.get("path") or m), hide=not body.get("show"))
    if route == "duplicate":
        return cli.duplicate(Path(m), (body.get("name") or "").strip())
    if route == "approve":
        return cli.cmd_approve(_args(manifest=Path(m), note=body.get("note", "")))
    if route == "bundle":
        return cli.cmd_bundle(_args(manifest=Path(m), note=body.get("note", "")))
    if route == "browse":
        return {"path": _browse(body.get("kind", "model"))}
    if route == "events":
        return JOBS.poll(int((query.get("since") or ["0"])[0]))
    if route == "url":
        return {"url": _url(Path(body["path"]))}
    raise ValueError(f"no such call: {route}")


def _characters():
    return engine.characters()


def _disk():
    """Disk use for the Characters page: the drive, each character, and what pruning would free.
    `other` is what else lies in out/ (builds from before the app, comparison runs)."""
    characters = []
    for path in _characters():
        try:
            characters.append(dict(Project(path).usage(), manifest=str(path)))
        except (OSError, ValueError, KeyError, StepFailed):
            continue
    out = REPO / "out"
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) if out.is_dir() else 0
    inside = sum(c["bytes"] for c in characters if out.resolve() in Path(c["out"]).resolve().parents)
    return {"drive": out.resolve().anchor, "free": engine.free_disk(out), "characters": characters,
            "prunable": sum(c["prunable"] for c in characters), "other": max(0, total - inside)}


# Where a request's unexpected failure is written down, newest last: what to send with a report.
ERROR_LOG = REPO / "out" / "app_errors.log"


def _log_error(what, text):
    try:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(ERROR_LOG, "a", encoding="utf-8") as log:
            log.write(f"--- {datetime.now().isoformat(timespec='seconds')} {what}\n{text}\n")
    except OSError:
        pass


def _summary(manifest_path):
    """A character card's facts; a character that can't be read still lists, without them: one
    broken folder must not empty the whole Characters page."""
    try:
        summary = Project(manifest_path).summary()
        # the pictures are drawn again by every build under the same names: the address says which drawing it is
        posed = {name: dict(p, url=f"{_url(p['file'])}?t={int(Path(p['file']).stat().st_mtime)}") for name, p in summary["portraits"].items()}
        return dict(summary, stills={name: _url(path) for name, path in summary["stills"].items()}, portraits=posed,
                    **{key: _url(summary[key]) if summary[key] else None for key in ("picture", "portrait")})
    except (OSError, ValueError, KeyError):
        return {}
    except Exception:  # noqa: BLE001
        _log_error(f"character {manifest_path}", traceback.format_exc())
        return {}


# A render folder is <run>_<step's blend>_<pose> (cli.cmd_render); older ones lack the run.
RENDER_FOLDER = re.compile(r"^(?:(?P<run>\d{8}-\d{6})_)?(?P<blend>.+)_(?P<pose>[^_]+)$")


def _renders(folder):
    """Every render of a character, newest first, with its run, pose and served pictures."""
    found = []
    for path in sorted((p for p in folder.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        match = RENDER_FOLDER.match(path.name)
        if match:
            found.append({**match.groupdict(), "urls": [_url(p) for p in sorted(path.glob("*.png"))]})
    return found


def _sheet(project, run=None):
    """Draws the contact sheets of the finished .vrm, or of an earlier run's (the build marked good)."""
    if not run:
        return _with_urls(cli.cmd_sheet(_args(target=project.manifest_path, out=None, browser=None)))
    vrm = next((r["vrm"] for r in project.runs() if r["run"] == run and r["vrm"]), None)
    if vrm is None:
        raise StepFailed(f"run {run} has no .vrm (its model files may have been freed)")
    return _with_urls(cli.cmd_sheet(_args(target=Path(vrm), out=project.out / "renders" / f"sheets_{run}", browser=None)))


def _sheets(folder):
    """Every build's contact sheets (cli.cmd_sheet: renders/sheets_<run>/), newest first."""
    found = []
    for path in sorted(folder.glob("sheets_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        urls = [_url(p) for p in sorted(path.glob("sheet_*.png"))]
        if urls:
            found.append({"run": path.name[len("sheets_"):], "urls": urls,
                          "report": engine._read_json(path / "sheet.json", None)})
    return found


def _reveal(path):
    """Shows an output file in Explorer. Only files under out/: the page must not open anything else."""
    try:
        path.resolve().relative_to((REPO / "out").resolve())
    except ValueError:
        raise ValueError("only files the build made can be shown") from None
    if not path.exists():
        raise FileNotFoundError(f"{path} is gone")
    if os.name == "nt":
        subprocess.Popen(["explorer", "/select,", str(path.resolve())] if path.is_file() else ["explorer", str(path.resolve())])
    else:
        webbrowser.open(path.resolve().parent.as_uri() if path.is_file() else path.resolve().as_uri())
    return {"shown": str(path)}


# What each Browse button asks for, and where its dialog opens.
BROWSE = {"model": ("Model", "*.fbx *.glb *.gltf *.obj *.vrm"), "rig": ("Tripo rig (FBX)", "*.fbx"),
          "face": ("Face points", "*.json"), "blender": ("Blender", "blender.exe blender")}
BROWSE_FROM = {"blender": engine.WINDOWS_BLENDER.parent}


def _browse_mac(title, start):
    """macOS's own open-file dialog, through osascript. Tk may only be used from a program's first
    thread there, and the server answers on another: a Tk dialog from here takes Python down."""
    script = (f'POSIX path of (choose file with prompt "Choose the {title.lower()}" '
              f'default location POSIX file "{start}")')
    done = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return done.stdout.strip() or None  # pressing Cancel is an error to osascript: nothing chosen


def _browse(kind):
    """A native open-file dialog, from the server: a web page can't learn a file's real path."""
    if sys.platform == "darwin":
        title, _ = BROWSE.get(kind, BROWSE["model"])
        start = BROWSE_FROM.get(kind, REPO / "samples")
        return _browse_mac(title, start if start.is_dir() else Path.home())
    # a Linux Python without python3-tk, or a computer with no screen: said to the page, where the
    # path can be typed. Unsaid, the request died in the server and the button did nothing
    try:
        import tkinter
        from tkinter import filedialog
    except ImportError:
        raise ValueError("this Python has no file dialog (tkinter): type the path instead. "
                         "Linux: install python3-tk and start the app again") from None
    title, pattern = BROWSE.get(kind, BROWSE["model"])
    start = BROWSE_FROM.get(kind, REPO / "samples")
    try:
        root = tkinter.Tk()
    except tkinter.TclError as error:
        raise ValueError(f"no file dialog can open here ({error}): type the path instead") from None
    root.withdraw()
    root.attributes("-topmost", True)  # over the app's window, not behind it
    try:
        chosen = filedialog.askopenfilename(title=f"Choose the {title.lower()}", filetypes=[(title, pattern), ("All files", "*.*")],
                                            initialdir=str(start if start.is_dir() else Path.home()))
    finally:
        root.destroy()
    return str(Path(chosen)) if chosen else None


def _name(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("name") or Path(path).stem
    except (OSError, json.JSONDecodeError):
        return Path(path).stem


def _served(path):
    """The repository-relative path of a file this server may hand out, else None."""
    try:
        relative = Path(path).resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return None
    if not any(relative.startswith(folder + "/") for folder in SERVED):
        return None
    return relative if Path(path).is_file() and Path(path).suffix.lower() in KINDS else None


def _url(path):
    """A served URL for a repository file, or None when it isn't served."""
    relative = _served(path)
    return "/" + relative if relative else None


def _with_urls(result):
    return dict(result, urls=[_url(p) for p in result.get("images", [])])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _trusted(self):
        # another site open in the browser must not drive builds: a page on this origin sends
        # the header, and a DNS-rebound name fails the host check
        return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

    def do_GET(self):
        if not self._trusted():
            return self.send_error(403)
        url = urlparse(self.path)
        if url.path == "/api/model":
            return self._model(parse_qs(url.query))
        if url.path == "/api/overlay":
            return self._overlay(parse_qs(url.query))
        if url.path.startswith("/api/"):
            if url.path[5:] not in READ_ONLY:
                return self._json({"error": "this call changes things: it takes a POST from the app's own page"}, 405)
            return self._api(url.path[5:], parse_qs(url.query), {})
        path = unquote(url.path)
        if path in ("/", "/index.html"):
            return self._file(GUI / "index.html")
        if path == "/music":
            # the user's own tune for the arcade look: music.mp3 (or .ogg, .wav) beside the app. None is no error
            # (the look then plays its own), so no 404 in the page's console
            found = next((REPO / f"music{kind}" for kind in (".mp3", ".ogg", ".wav") if (REPO / f"music{kind}").is_file()), None)
            if found is None:
                self.send_response(204)
                return self.end_headers()
            return self._file(found)
        if path.startswith("/gui/"):
            return self._file(GUI / path[5:], root=GUI)
        relative = _served(REPO / path.lstrip("/"))
        if relative:
            return self._file(REPO / relative, root=REPO)
        self.send_error(404)

    def do_POST(self):
        if not self._trusted() or self.headers.get("X-VTuber-Maker") != "1":
            return self.send_error(403)
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad JSON"}, 400)
        self._api(url.path[5:], parse_qs(url.query), body)

    def _api(self, route, query, body):
        try:
            self._json(api(route, query, body))
        except (StepFailed, ValueError, FileNotFoundError, KeyError) as error:
            self._json({"error": str(error)}, 400)
        except ConnectionError:
            pass  # the page left before the answer was written (closed, or reloaded): nobody to tell
        except Exception as error:  # noqa: BLE001 -- said to the page and kept, not lost with the connection
            # anything else used to end the request with no answer: the page said "Failed to fetch",
            # listed no characters, and nothing anywhere said why
            _log_error(route, traceback.format_exc())
            self._json({"error": f"the app tripped over its own feet in \"{route}\" ({type(error).__name__}: {error}). "
                                 f"The details are in {ERROR_LOG}"}, 500)

    def _model(self, query):
        """A listed character's source or reference model, wherever it lives on disk."""
        wanted = Path((query.get("manifest") or [""])[0]).resolve()
        listed = {Path(p["manifest"]).resolve() for p in api("projects", {}, {})["projects"]}
        key = (query.get("key") or [""])[0]
        if wanted not in listed or key not in ("source", "reference"):
            return self.send_error(403)
        path = engine.manifest.path_of(engine.manifest.load(wanted), key)
        if path is None or path.suffix.lower() not in (".glb", ".vrm"):
            return self.send_error(404)
        self._file(Path(path))

    def _overlay(self, query):
        """A listed character's paint on one picture, wherever its folder lives on disk."""
        wanted = Path((query.get("manifest") or [""])[0]).resolve()
        if wanted not in {Path(p).resolve() for p in _characters()}:
            return self.send_error(403)
        if (query.get("draft") or [""])[0]:
            kept = paint.overlays(cli._draft_folder(Project(wanted)))
        else:
            kept = paint.overlays(engine.manifest.path_of(engine.manifest.load(wanted), "paint"))
        path = kept.get((query.get("name") or [""])[0])
        if path is None:
            return self.send_error(404)
        self._file(path)

    def _json(self, data, code=200):
        raw = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _file(self, path, root=None):
        path = path.resolve()
        if root is not None and root.resolve() not in path.parents:
            return self.send_error(403)
        if not path.is_file() or path.suffix.lower() not in KINDS:
            return self.send_error(404)
        raw = None
        if path.suffix.lower() == ".html":
            # the pages name three.js and the tracker by their CDN addresses: with a copy kept here, that copy
            raw = vendor.local(path.read_text(encoding="utf-8")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(raw) if raw is not None else path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if raw is not None:
            return self.wfile.write(raw)
        # a model is tens of megabytes: sent as it is read, not held whole for every viewer that opens
        with open(path, "rb") as handle:
            shutil.copyfileobj(handle, self.wfile, 1 << 20)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8322)
    parser.add_argument("--no-window", action="store_true", help="serve only; open the page yourself")
    args = parser.parse_args()
    server, port = _listen(args.port)
    address = f"http://127.0.0.1:{port}/"
    if server is None:
        # started twice: the first one holds the builds and the one Blender, so show that one
        print(f"OC2VRM is already running at {address}", flush=True)
        if not args.no_window:
            _show(address)
        return
    print(f"OC2VRM at {address}", flush=True)
    if args.no_window:
        server.serve_forever()
        return
    if not _own_window():
        # started again after an update: the page is still open in the browser and reloads itself
        if not os.environ.get("OC2VRM_RESTARTED"):
            webbrowser.open(address)
        server.serve_forever()
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _show(address)


# A port another program holds is passed over, this many times.
PORT_TRIES = 10


class Server(ThreadingHTTPServer):
    # Windows lets a second server take a port in use when this is on: two apps then answered on
    # one port, each with its own idea of what was running
    allow_reuse_address = os.name != "nt"


def _listen(port):
    """(server, port). (None, port) when the app itself is already answering there."""
    for candidate in range(port, port + PORT_TRIES):
        try:
            server = Server(("127.0.0.1", candidate), Handler)
            return server, server.server_port
        except OSError:
            if _ours(candidate):
                return None, candidate
    raise SystemExit(f"ports {port} to {port + PORT_TRIES - 1} are all taken: pass --port")


def _ours(port):
    """Whether this app is what answers on the port (its job report has these fields)."""
    from urllib.request import urlopen
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/events", timeout=2) as response:
            return {"running", "events", "next"} <= set(json.load(response))
    except (OSError, ValueError):
        return False


def _own_window():
    """Whether the app opens in its own window: pywebview is there and the Setup page's choice
    ("open the app in") isn't the browser."""
    choice = engine.open_in()
    return choice["open_in"] == "window" and choice["can_window"]


def _show(address):
    """The page in a pywebview window (back once it is closed), else in the browser."""
    if not _own_window():
        webbrowser.open(address)
        return
    import webview
    webview.create_window("OC2VRM", address, width=1400, height=900)
    webview.start()


if __name__ == "__main__":
    main()
