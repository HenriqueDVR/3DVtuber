"""What shows through a hole cut in the avatar's surface, counted on the exported .vrm as apps draw it. Plain Python.

The fox girl's eye white lay 3 cm down her socket (2026-10-10): right from the front and in the
contact sheets, the inside of her head beside the iris from the side in Warudo. The rig check's
rays (`check_avatar.py` `iris_depths`) read depth and can't know that the eye's frame is an opaque
sheet with a window cut by its picture's alpha, so the catgirl's white 16 mm down could not be
told from model B's eye patches standing 13 mm off her round face, and only over 20 mm failed.
`viewer/inside.html` draws the file with three-vrm from 21 cameras round the face and counts,
where the first surface on a ray is cut away by its picture (alphaMode MASK), how far behind that
surface lies what shows. A blended patch is no window: nothing is behind it but the face it lies on.

Measured on 37 files (2026-10-11, SETUP.md "Seen through a window"): the worst camera's area lying
over 12 mm behind its window is 278 to 573 mm2 on the four fox builds with the deep white and 408 on
the catgirl's two, and 0 to 12 on every other file (12: the plain boy's white 6 mm down, which no
one saw). Not judged, only measured: the hair's inside seen beside the neck (`hair_inside_mm2`).

Also judged (2026-10-16): an eye layer in the face's picture standing past the face (`past_mm2`). The miko's
white sheet and the disc's still copy stood out beyond her far eye's corner as a straight-edged strip of skin
before her side hair, seen only from the side; gaps, depths and outlines from the front read the same before
and after the fix. The page finds the layers by shape (the face patch's FRAME_GRID pieces and the copies of
the iris discs), draws the file without them to read what lies behind, and counts where a layer is seen first
showing the face's skin with hair on springs over OUT_MM behind it. Seen through the eye's opening a layer
shows the drawn white and has the head's inside behind it.
The gap the user saw under model A's back hair reads 4,911 mm2 on the copy that had it and 167 on
the one with the rim capped, but every build of long hair made here reads as the first does.
"""
import json
import struct
from pathlib import Path

from . import sheet

PAGE = sheet.REPO / "viewer" / "inside.html"
# What shows through a window counts as deep from this far behind it (mm, square to the window), and
# the build fails when the worst camera sees more than this much of it (mm2). Deep whites: 278 mm2
# (one eye of the fox girl's) to 573; good builds: 12 at most. The mended catgirl's white lies 6.5 mm
# down and reads 322 mm2 over 8 mm from 40 degrees, 7 over 10: hence 12.
DEEP_MM = 12
LIMIT_MM2 = 60
# An eye layer in the face's picture (the white sheet, the disc's still copy) standing past the face: seen first
# where it shows the face's skin, with hair on springs over OUT_MM behind it (mm2 of the worst camera). The miko's
# far-eye panel before 326d777 (a strip of skin before her side hair from 40 degrees, which no ray from the front
# told apart: SETUP.md "Checks for what Warudo caught by eye"): 75 mm2; every good build: 9 at most.
OUT_MM = 10
PAST_MM2 = 30
# loading 110 MB and 105 cameras took 11 s with a graphics card
TIMEOUT = 300


def measure(vrm, out_dir, browser=None, timeout=TIMEOUT, pictures=False):
    """The page's report of `vrm` ({"face": a row a camera, "hair": ..., "cut_out": materials with a
    window, ...}), kept as <out_dir>/inside.json. Raises sheet.NoBrowser, TimeoutError, RuntimeError."""
    out_dir = Path(out_dir).resolve()
    got = sheet.browse(PAGE, vrm, out_dir, browser, timeout, query=f"&deep={DEEP_MM}&out={OUT_MM}" + ("&pictures=1" if pictures else ""), prefix="inside_")
    report = dict(got["report"], model=str(Path(vrm).resolve()), images=got["images"])
    (out_dir / "inside.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def judge(report):
    """{"deep_mm2", "shot", "window_mm2", "through_mm2", "hair_inside_mm2", "hair_shot", "past_mm2", "past_shot",
    "failed": [lines]} of a page's report: the worst camera's count, and what of it is over the limit."""
    worst = max(report["face"], key=lambda shot: shot["deep_mm2"])
    inside = max(report.get("hair") or [{"hair_inside_mm2": 0.0, "label": ""}], key=lambda shot: shot["hair_inside_mm2"])
    # a report from before the eye layers were counted has none
    past = max(report["face"], key=lambda shot: shot.get("past_mm2", 0.0))
    found = {"deep_mm2": worst["deep_mm2"], "shot": worst["label"], "window_mm2": max(shot["window_mm2"] for shot in report["face"]),
             "through_mm2": max(shot["through_mm2"] for shot in report["face"]), "cut_out": report.get("cut_out", []),
             "hair_inside_mm2": inside["hair_inside_mm2"], "hair_shot": inside["label"],
             "past_mm2": past.get("past_mm2", 0.0), "past_shot": past["label"], "failed": []}
    if worst["deep_mm2"] > LIMIT_MM2:
        through = ", ".join(sorted(worst.get("by") or {})) or "a cut-out surface"
        found["failed"].append(f"{worst['deep_mm2']:.0f} mm2 of what shows through the opening in {through} lies over {DEEP_MM} mm "
                               f"behind it ({worst['label']}: degrees round from the front, up): the inside of the head shows from the side")
    if found["past_mm2"] > PAST_MM2:
        found["failed"].append(f"{found['past_mm2']:.0f} mm2 of an eye layer stands past the face, showing the face's skin before "
                               f"the hair ({past['label']}: degrees round from the front, up): a flat panel beside the eye from the side")
    return found


def is_vrm(path):
    """Whether the file is a glTF binary that says it is a VRM: no browser is started for anything else."""
    try:
        with open(path, "rb") as handle:
            magic, _, _, length, kind = struct.unpack("<4sIII4s", handle.read(20))
            if magic != b"glTF" or kind != b"JSON":
                return False
            extensions = json.loads(handle.read(length)).get("extensions") or {}
        return "VRMC_vrm" in extensions or "VRM" in extensions
    except (OSError, ValueError, struct.error):
        return False


def check(vrm, out_dir, browser=None, timeout=TIMEOUT):
    """`judge` of the file, or {"left": why} when it couldn't be drawn (no browser, a browser whose
    graphics can't be read back, a page that never answered). Never raises: a build is not lost to it."""
    if not is_vrm(vrm):
        return {"left": "not a VRM file"}
    try:
        return judge(measure(vrm, out_dir, browser, timeout, pictures=True))
    except sheet.NoBrowser as error:
        return {"left": str(error)}
    except (TimeoutError, RuntimeError, OSError, ValueError, KeyError) as error:
        return {"left": f"the file couldn't be drawn: {error}"}
