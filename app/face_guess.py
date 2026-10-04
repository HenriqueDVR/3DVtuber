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
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "retexture" / "guess_face_points.py"
# A Python that has PyTorch: the app's own needn't. The retexture tools' ComfyUI brings one.
COMFY_PYTHON = REPO / "tools" / "ComfyUI_windows_portable" / "python_embeded" / "python.exe"
INSTALL = "pip install anime-face-detector (Python 3.12+, with PyTorch), or set VTUBER_MAKER_AI_PYTHON to a Python that has it"
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


def find_python(given=None):
    """A Python with the detector: `given`, $VTUBER_MAKER_AI_PYTHON, the tools' ComfyUI, this one."""
    for path in (given, os.environ.get("VTUBER_MAKER_AI_PYTHON"), COMFY_PYTHON, sys.executable):
        if path and Path(path).is_file() and _has_detector(str(path)):
            return str(path)
    raise NoDetector(f"no Python with the face detector found: {INSTALL}")


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
    ran = subprocess.run([python, str(SCRIPT), str(picture)], capture_output=True, text=True, env=env, timeout=600)
    line = next((line for line in ran.stdout.splitlines() if line.startswith("FACE_LANDMARKS ")), None)
    if ran.returncode != 0 or line is None:
        raise RuntimeError(f"the face detector failed\n{(ran.stderr or ran.stdout)[-1500:]}")
    return from_landmarks(json.loads(line[len("FACE_LANDMARKS "):]))
