"""Contact sheets of an exported .vrm, drawn by three-vrm in a headless browser. Plain Python.

Blender's renders are not what an app shows: no MToon, no springs, both sides of every face.
A gap at the back of the neck, a pale piece by the lowered hand and a bent hand were all seen
in Warudo and never in a Blender render (2026-10-03). `viewer/sheet.html` loads the file as
apps do and draws it in the poses, expressions and camera angles where those were; this serves
the page to Edge or Chrome without a window and saves the sheets it sends back.
"""
import json
import mimetypes
import os
import shutil
import signal
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import vendor

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "viewer" / "sheet.html"
GUI = Path(__file__).resolve().parent / "gui"
BROWSERS = (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium")
# Loading 110 MB of .vrm and drawing 44 shots took about 15 s with a graphics card
TIMEOUT = 300

mimetypes.add_type("model/gltf-binary", ".vrm")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")


class NoBrowser(Exception):
    pass


def find_browser(given=None):
    """Edge or Chrome: `given`, else $VTUBER_MAKER_BROWSER, else where Windows installs them, else PATH."""
    for path in (given, os.environ.get("VTUBER_MAKER_BROWSER"), *BROWSERS):
        if path and Path(path).is_file():
            return str(path)
    for name in ("msedge", "microsoft-edge", "chrome", "google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if shutil.which(name):
            return shutil.which(name)
    raise NoBrowser("no Edge or Chrome found: set VTUBER_MAKER_BROWSER to one")


def launch(browser, profile, url, *flags, log=subprocess.DEVNULL):
    """The browser without a window on `url`, in a profile of its own (the user's browser, its sign-ins
    and its open windows are left alone). Software WebGL is allowed for a machine without a graphics
    card the browser will use. End it with close(), never by its own exit: see there."""
    process = subprocess.Popen([browser, "--headless=new", f"--user-data-dir={profile}", "--no-first-run", "--mute-audio",
                                "--enable-unsafe-swiftshader", *flags, url], stdout=subprocess.DEVNULL, stderr=log,
                               start_new_session=os.name != "nt")
    process.browser, process.profile = str(browser), str(profile)
    return process


def close(process):
    """Ends the browser and every process it started, and waits for it. Killing the browser alone left its
    page, network and GPU processes running on Windows for seconds: still asking the server for things,
    still holding the profile's lockfile, so the profile couldn't be removed (2026-10-08).

    On Windows the browser is found by its profile, not by the process that was started: Edge 154 hands
    over to a child and exits within half a second, so the process in hand was already gone, nothing
    was killed, and each contact sheet and each page of the test left sixteen processes running.
    """
    alive = process.poll() is None
    if os.name == "nt":
        if alive:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        quoted = lambda text: "'" + text.replace("'", "''") + "'"  # noqa: E731
        sweep = (f"Get-CimInstance Win32_Process -Filter \"Name={quoted(Path(process.browser).name)}\" | "
                 f"Where-Object {{ $_.CommandLine -and $_.CommandLine.Contains({quoted(process.profile)}) }} | "
                 "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")
        try:
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", sweep], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            pass  # no PowerShell to ask: the browser started is still ended below
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)  # the group outlives a browser that has itself gone
        except (ProcessLookupError, PermissionError):
            pass
    if alive:
        process.kill()
    process.wait()


def render(vrm, out_dir, browser=None, timeout=TIMEOUT):
    """{"images": the sheets' PNGs, "report": what the page said of the file (expressions the
    shots asked for and it lacks, spring joints)}. Raises NoBrowser, TimeoutError, RuntimeError."""
    got = browse(PAGE, vrm, out_dir, browser, timeout)
    report = dict(got["report"], model=str(Path(vrm).resolve()))
    (Path(out_dir).resolve() / "sheet.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return {"images": got["images"], "report": report}


def browse(page, vrm, out_dir, browser=None, timeout=TIMEOUT, query="", prefix="sheet_"):
    """A page of viewer/ opened on `vrm` in a browser without a window: {"images": the PNGs it sent
    (kept in `out_dir` as <prefix><name>.png), "report": what it said when it was done}.
    Raises NoBrowser, TimeoutError, RuntimeError (the page said it failed)."""
    # whole paths: Edge shows a dialog ("can't read and write to its data directory") for a relative profile
    vrm, out_dir = Path(vrm).resolve(), Path(out_dir).resolve()
    if not vrm.is_file():
        raise FileNotFoundError(vrm)
    browser = find_browser(browser)
    out_dir.mkdir(parents=True, exist_ok=True)
    got = {"images": [], "report": None, "error": None}
    finished = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == "/model.vrm":
                return self._send(vrm)
            if path == f"/viewer/{page.name}":
                # three.js by its CDN address, or the app's own copy when it keeps one (offline)
                return self._send(page, vendor.local(page.read_text(encoding="utf-8")).encode("utf-8"))
            target = (GUI / path[5:]).resolve() if path.startswith("/gui/") else None
            if target is None or GUI.resolve() not in target.parents or not target.is_file():
                return self.send_error(404)
            self._send(target)

        def do_POST(self):
            name = (parse_qs(urlparse(self.path).query).get("name") or [""])[0]
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if name in ("done", "failed"):
                said = json.loads(body or b"{}")
                got["report"], got["error"] = (said, None) if name == "done" else (None, said.get("error") or "the page failed")
                finished.set()
            elif name.isalnum():
                path = out_dir / f"{prefix}{name}.png"
                path.write_bytes(body)
                got["images"].append(str(path))
            self.send_response(204)
            self.end_headers()

        def _send(self, path, raw=None):
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(raw) if raw is not None else path.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if raw is not None:
                return self.wfile.write(raw)
            with open(path, "rb") as handle:
                shutil.copyfileobj(handle, self.wfile)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # beside the sheets, not in %TEMP%: the browser keeps large blobs (a 4096 px texture) in its
    # profile, and on a system drive with 3.7 GB free it refused the main texture: a white body
    profile = tempfile.mkdtemp(prefix=".browser_", dir=out_dir)
    url = f"http://127.0.0.1:{server.server_port}/viewer/{page.name}?model=/model.vrm&post=1{query}"
    process = launch(browser, profile, url, "--window-size=1280,900")
    try:
        if not finished.wait(timeout):
            raise TimeoutError(f"{page.name} wasn't drawn in {timeout} s ({browser})")
        if got["error"]:
            raise RuntimeError(got["error"])
    finally:
        close(process)
        server.shutdown()
        server.server_close()
        shutil.rmtree(profile, ignore_errors=True)
    return {"images": sorted(got["images"]), "report": got["report"]}
