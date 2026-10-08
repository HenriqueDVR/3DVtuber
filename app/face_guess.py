"""Face points placed by a detector, for the user to check and nudge. Plain Python.

Marking twelve points by hand is the one step of a build nobody can skip, and a mouth drawn as
a line leaves people unsure where to click. hysts' anime-face-detector finds 28 landmarks on the
face patch's picture; twelve of them are our points. On model A and model A (T-pose) they fell 10 to 25
pixels from the hand-placed marks (an eye there is 220 pixels wide), and the mouth on the mouth
where the first model A's hand-placed marks sat on her nose (model C: 0.08 of an eye's width off on the
eyes too). Model B's face it does not read (its points 1.4 to 2.9 eye widths off, and it says so:
0.10 sure), so a guess it isn't sure of is refused. A first guess, never trusted blind:
automatic detection was dropped once for being the least reliable part of a build.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "retexture" / "guess_face_points.py"
# A Python that has PyTorch: the app's own needn't. The retexture tools' ComfyUI brings one, and
# `install` makes one for the detector alone (the Setup page's button).
COMFY_PYTHON = REPO / "tools" / "ComfyUI_windows_portable" / "python_embeded" / "python.exe"
HOME = REPO / "tools" / "face_detector"
INSTALL = ("install it from the app's Setup page (python -m app detector --install), or pip install anime-face-detector "
           "(Python 3.12+, with PyTorch), or set VTUBER_MAKER_AI_PYTHON to a Python that has it")
# The detector's pure-PyTorch release needs this Python or newer: an older one is given its old
# release by pip, which wants three more toolkits built from source.
MIN_PYTHON = (3, 12)
PACKAGE = "anime-face-detector>=0.1.0"
# PyTorch for the processor alone: a tenth of the download of the one for graphics cards, and a
# face takes it a second.
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
# Free space the install wants where it goes (PyTorch unpacked, the detector's weights later).
SPACE_NEEDED = 3 << 30
# The detector's landmarks (image left to right: the character's right eye comes first).
# Eyes: outer corner, top, inner corner, then three along the lower lid, the middle one the bottom.
EYES = {"right_eye": {"outer": 11, "top": 12, "inner": 13, "bottom": 15},
        "left_eye": {"inner": 17, "top": 18, "outer": 19, "bottom": 21}}
MOUTH = {"left": 24, "top": 25, "right": 26, "bottom": 27}
# A mouth drawn as a line: its top and bottom landmarks fall on the line. The points go above and
# below its middle by these shares of its width, as the Face page asks of a hand-placed one
# (only the corners set the mouth's size; model A (T-pose)'s hand marks: 0.23 above, 0.32 below).
LINE, LINE_UP, LINE_DOWN = 0.3, 0.25, 0.3
SURE = 0.5


class NoDetector(Exception):
    pass


class NoFace(Exception):
    pass


def own_python(home=None):
    """The Python `install` makes for the detector."""
    home = Path(home or HOME)
    return home / "Scripts" / "python.exe" if os.name == "nt" else home / "bin" / "python"


_FOUND = {}


def find_python(given=None):
    """A Python with the detector: `given`, $VTUBER_MAKER_AI_PYTHON, the one `install` made, the
    tools' ComfyUI, this one. Asked of each once (it means starting it); `forget` asks again."""
    for path in (given, os.environ.get("VTUBER_MAKER_AI_PYTHON"), own_python(), COMFY_PYTHON, sys.executable):
        if not path or not Path(path).is_file():
            continue
        path = str(path)
        if path not in _FOUND:
            _FOUND[path] = _has_detector(path)
        if _FOUND[path]:
            return path
    raise NoDetector(f"no Python with the face detector found: {INSTALL}")


def forget():
    _FOUND.clear()


def status():
    """{"python": the Python that has the detector, or None}: for the Setup page."""
    try:
        return {"python": find_python()}
    except NoDetector:
        return {"python": None}


def base_python(run=subprocess.run):
    """A Python new enough for the detector, to make its own from: this one, else what the `py`
    launcher lists (Windows), else python3.x on PATH. None without one."""
    import shutil
    if sys.version_info[:2] >= MIN_PYTHON:
        return sys.executable
    found = []
    if os.name == "nt" and shutil.which("py"):
        try:
            listed = run(["py", "-0p"], capture_output=True, text=True, errors="replace", timeout=30).stdout
        except (OSError, subprocess.TimeoutExpired):
            listed = ""
        for line in listed.splitlines():
            # " -V:3.12          C:\...\python.exe"; a star marks the default
            parts = line.replace("*", " ").split(None, 1)
            if len(parts) == 2 and parts[0].startswith("-V:"):
                try:
                    version = tuple(int(v) for v in parts[0][3:].split("-")[0].split(".")[:2])
                except ValueError:
                    continue
                found.append((version, parts[1].strip()))
    for minor in range(MIN_PYTHON[1], MIN_PYTHON[1] + 6):
        path = shutil.which(f"python3.{minor}")
        if path:
            found.append(((3, minor), path))
    # the oldest that will do: the newest Python is the one PyTorch has no wheels for yet
    good = sorted((v, p) for v, p in found if v >= MIN_PYTHON and Path(p).is_file())
    return good[0][1] if good else None


def install(on_event=None, cancelled=lambda: False, home=None, run=subprocess.Popen, free=None):
    """Make a Python of the detector's own in tools/face_detector and put PyTorch and the detector
    in it: about 250 MB of downloads, 1.5 GB on disk. Returns {"python"}. Raises NoDetector.

    Three steps, each a running event with the install log (the app shows its end). Nothing
    outside that folder is touched; removing the folder removes it all.
    """
    import shutil
    emit = on_event or (lambda event: None)
    home = Path(home or HOME)
    base = base_python()
    if base is None:
        raise NoDetector(f"the face detector needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer, and none was found: "
                         "install one from python.org, then press this again")
    home.parent.mkdir(parents=True, exist_ok=True)
    room = (free or (lambda path: shutil.disk_usage(path).free))(home.parent)
    if room < SPACE_NEEDED:
        raise NoDetector(f"the face detector needs about {SPACE_NEEDED >> 30} GB free on {home.anchor}; there is {room / 2**30:.1f} GB")
    python = own_python(home)
    log = home.parent / "face_detector_install.log"
    steps = [("making a Python of its own", [base, "-m", "venv", str(home)]),
             ("downloading PyTorch (about 250 MB)", [str(python), "-m", "pip", "install", "torch", "torchvision", "--index-url", TORCH_INDEX]),
             ("downloading the face detector", [str(python), "-m", "pip", "install", PACKAGE])]
    with open(log, "w", encoding="utf-8", errors="replace") as out:
        for what, command in steps:
            emit({"step": what, "status": "running", "log": str(log)})
            out.write(f"\n> {' '.join(command)}\n")
            out.flush()
            process = run(command, stdout=out, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if cancelled():
                    process.kill()
                    process.wait()
                    raise NoDetector("stopped: press the button again to carry on (what was downloaded is kept)")
                time.sleep(0.5)
            if process.returncode != 0:
                out.flush()
                tail = Path(log).read_text(encoding="utf-8", errors="replace")[-600:]
                raise NoDetector(f"{what} failed (see {log})\n{tail}")
            emit({"step": what, "status": "done"})
    forget()
    if not _has_detector(str(python)):
        raise NoDetector(f"installed, but the detector doesn't load in {python} (see {log})")
    return {"python": str(python), "next": "the Face page's \"Place them for me\" now works; its first use downloads 273 MB of weights"}


def _has_detector(python):
    try:
        found = subprocess.run([python, "-c", "import importlib.util, sys; sys.exit(importlib.util.find_spec('anime_face_detector') is None)"],
                               capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return found.returncode == 0


def from_landmarks(found):
    """The detector's output (see the script) as face points: {"size", "right_eye", "left_eye",
    "mouth", "sure"}. The largest face it is sure of; raises NoFace without one."""
    faces = [f for f in found.get("faces", []) if f["box"][4] >= SURE]
    if not faces:
        raise NoFace("the detector found no face in the picture: mark the points by hand")
    face = max(faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
    at = [point[:2] for point in face["points"]]
    points = {"size": list(found["size"]),
              **{eye: {name: at[index] for name, index in marks.items()} for eye, marks in EYES.items()},
              "mouth": {name: list(at[index]) for name, index in MOUTH.items()}}
    mouth = points["mouth"]
    width = abs(mouth["right"][0] - mouth["left"][0])
    if mouth["bottom"][1] - mouth["top"][1] < LINE * width:
        x = (mouth["left"][0] + mouth["right"][0]) / 2
        y = (mouth["top"][1] + mouth["bottom"][1]) / 2
        mouth["top"], mouth["bottom"] = [x, y - LINE_UP * width], [x, y + LINE_DOWN * width]
    used = [i for marks in (*EYES.values(), MOUTH) for i in marks.values()]
    points["sure"] = round(min(face["points"][i][2] for i in used), 2)
    if points["sure"] < SURE:
        # model B: a face box at full score, its landmarks at 0.10 and up to 2.9 eye widths off
        raise NoFace(f"the detector isn't sure where this face's eyes and mouth are ({points['sure']:.2f}): mark the points by hand")
    return points


def guess(picture, python=None, cache=None):
    """Face points for a picture of a face from the front. Raises NoDetector, NoFace, RuntimeError."""
    python = find_python(python)
    env = dict(os.environ)
    if cache:
        env["HF_HOME"] = str(cache)  # the weights (273 MB) beside the app's other downloads, not in the user's profile
    ran = subprocess.run([python, str(SCRIPT), str(picture)], capture_output=True, text=True, errors="replace", env=env, timeout=600)
    line = next((line for line in ran.stdout.splitlines() if line.startswith("FACE_LANDMARKS ")), None)
    if ran.returncode != 0 or line is None:
        raise RuntimeError(f"the face detector failed\n{(ran.stderr or ran.stdout)[-1500:]}")
    return from_landmarks(json.loads(line[len("FACE_LANDMARKS "):]))
