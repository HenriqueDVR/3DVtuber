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
import time
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
# A browser asks for its page within a second or two of starting. One that hasn't after this long (s)
# is ended and started again, up to this many starts in all.
NEVER_ASKED, STARTS = 25, 3
# What a browser needs below its profile folder (characters): Edge wrote files up to 120 below it while
# it drew an avatar, and writes them the old way, inside the 260 Windows gives a path. With less room
# it drops files without a word (at 200 characters of profile, 280 of its 352), and at 250 it never
# asks for its page: a character in a deep folder waited the whole five minutes of every build for a
# look through its file that never came (2026-10-12, SETUP.md "The tester's computer").
PROFILE_ROOM, WINDOWS_PATH = 120, 259
PROFILE_NAME = len("\\.browser_12345678")
# Bytes free a drive needs for the browser to keep an avatar's textures in its profile there. Below it the
# browser refuses a texture ("a texture didn't load"): nothing drawn on E: at 3.4 and 4.0 GB free, a white body
# on C: at 3.7; drawn at 6.5 GB and more (2026-10-16, SETUP.md "The open mouth laid on the face").
BROWSER_FREE = 5 * 2**30

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


def profile_places(out_dir):
    """Where a browser's profile may go, best first: beside the sheets, not in %TEMP% (the browser keeps large
    blobs, a 4096 px texture, in its profile, and on a system drive with 3.7 GB free it refused the main texture:
    a white body); then the app's own out folder; then the temp folder. Those too deep on the disk for the
    browser to write in (PROFILE_ROOM) go last, shallowest first; of the rest, those on a drive with
    BROWSER_FREE go first: KemonoGirl's and the fox girl's files went undrawn on an E: with 4 GB free while
    C: had 9."""
    places = [Path(out_dir), REPO / "out", Path(tempfile.gettempdir())]
    deep = lambda place: os.name == "nt" and len(str(place)) + PROFILE_NAME + PROFILE_ROOM > WINDOWS_PATH  # noqa: E731
    roomy = [place for place in places if not deep(place)]
    roomy.sort(key=lambda place: free_at(place) < BROWSER_FREE)  # stable: the order above within each
    return roomy + sorted((place for place in places if deep(place)), key=lambda place: len(str(place)))


def free_at(path):
    """Bytes free on the drive `path` is, or will be, on (0 when that can't be read)."""
    path = Path(path).resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def profile_for(out_dir):
    """A new, empty folder for the browser's profile, in the first of profile_places that takes one."""
    failed = None
    for place in profile_places(out_dir):
        try:
            place.mkdir(parents=True, exist_ok=True)
            return tempfile.mkdtemp(prefix=".browser_", dir=place)
        except OSError as error:
            failed = error
    raise failed


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
    finished, asked = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == "/model.vrm":
                return self._send(vrm)
            if path == f"/viewer/{page.name}":
                asked.set()
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
    profile = profile_for(out_dir)
    url = f"http://127.0.0.1:{server.server_port}/viewer/{page.name}?model=/model.vrm&post=1{query}"
    process = launch(browser, profile, url, "--window-size=1280,900")
    started = launched = time.time()
    starts = 1
    try:
        while not finished.wait(0.5):
            if time.time() - started > timeout:
                raise TimeoutError(f"{page.name} wasn't drawn in {timeout} s ({browser}): "
                                   + ("the page was sent and nothing came back" if asked.is_set()
                                      else f"the browser never asked for the page, started {starts} times"))
            if not asked.is_set() and time.time() - launched > NEVER_ASKED and starts < STARTS:
                # a browser that hasn't come for its page by now isn't going to: one such start held a build
                # for the whole five minutes (2026-10-12), nothing drawn. Ended and started once more.
                close(process)
                shutil.rmtree(profile, ignore_errors=True)
                profile = profile_for(out_dir)
                process, launched, starts = launch(browser, profile, url, "--window-size=1280,900"), time.time(), starts + 1
        if got["error"]:
            raise RuntimeError(got["error"])
    finally:
        close(process)
        server.shutdown()
        server.server_close()
        shutil.rmtree(profile, ignore_errors=True)
    return {"images": sorted(got["images"]), "report": got["report"]}
