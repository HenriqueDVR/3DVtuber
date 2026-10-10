"""A tester's report: what a character's build did on their computer, in one zip.

    python -m app report out/MyOC/character.json [--note "..."] [--pictures] [--model] [--send]
    python -m app report --address https://discord.com/api/webhooks/...     # another channel for this copy, kept in update.json
    python -m app report --check received.zip [--unpack]                     # a report that came in: looked at before it is opened

Plain Python. The zip is written to <out>/reports/ and stays there; it leaves the computer only when
the tester sends it: by hand, or with --send (the page's "Send to the developers"), which posts it
to the developers' Discord channel through the webhook that comes with the app (app/reports.json). A webhook
can be posted to and not read from. --model adds the files the build reads; sent, their textures are
written smaller until the zip fits what Discord takes. Without --pictures it holds numbers and text only (settings, step logs, checks,
shape key numbers, timings, the computer's kind): no picture of the character, no model. Paths in
every text file are rewritten, so the Windows user name and the folders the art is kept in stay
behind (`scrub`).

`bundle` (cli.cmd_bundle) is the other one: a folder on this computer for an agent working here.
Thresholds in the pipeline rest on two or three characters each; these reports are how they come
to rest on more.
"""
import json
import os
import platform
import re
import struct
import subprocess
import textwrap
import urllib.error
import urllib.request
import uuid
import zipfile
import zlib
from datetime import datetime
from pathlib import Path

from . import engine, paint, trust, update
from .engine import REPO, Project, manifest

# Text files are read, scrubbed and written; anything else goes in only when asked for
TEXT = {".json", ".log", ".txt", ".md"}
PICTURES = {".png", ".jpg", ".jpeg"}
# The end of a log this long (bytes) is kept: a Blender that spins writes the same line for hours
LOG_MOST = 2 * 2**20
# Folders of rendered pictures taken, newest first: the last contact sheets and the last stills
RENDERS_MOST = 3
# Pictures go in no larger than this on their longer side (pixels)
PICTURE_MOST = 1400
# Where a report may be posted: Discord's webhooks and nothing else, whatever update.json says
ADDRESS = re.compile(r"https://(?:(?:canary|ptb)\.)?discord(?:app)?\.com/api/webhooks/\d+/[\w-]+")
# The most Discord takes in one message's files (bytes), with room for the message itself
SEND_MOST = 9 * 2**20
# The note is the message's text: Discord takes 2000 characters
NOTE_MOST = 1500
# A model that is sent has its textures written again as JPEG, no larger than the first of these
# (pixels on the longer side, quality) with which everything fits. Tripo's 4096 px PNGs are 80 to 95%
# of its files: model A's 29.2 MB is 6.5 at the first, 4.9 at the second, 4.0 at the third.
TEXTURES = ((4096, 85), (2048, 85), (1024, 80))
# Left free in a sent zip for the summary, written last (bytes)
SUMMARY_ROOM = 256 * 1024


# The developers' webhook, shipped with the app. The repository is private and so is this address:
# whoever has it can post to the reports channel (not read it). Before the repository is ever made
# public, delete that webhook in Discord and make another.
SHIPPED = Path(__file__).with_name("reports.json")


class NotSent(ValueError):
    """A report that couldn't be sent; the message is for the page. The zip is still on the computer."""


def make(manifest_path, note="", pictures=False, model=False, blender=None, fit=None, about=""):
    """Write the zip and say what went into it: {"report", "megabytes", "files", "holds", "left_out"}.

    `model`: also the files the build reads (the model, its rig, its reference, the paint), so the
    developers can build the character again; as they are, or with `fit` (bytes the whole zip may
    be) their textures made smaller until it is that small (`model_files`). `about`: what the page was
    showing when the report was asked for (an error's words). `manifest_path` None: no character is
    open (the app won't start a Blender, an install failed): the app and the computer alone."""
    if manifest_path is None:
        return make_app(note, about)
    project = Project(manifest_path, blender=blender)
    clean = scrubber(project)
    folder = project.out / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    stamp, n = datetime.now().strftime("%Y%m%d-%H%M%S"), 0
    while (folder / f"{project.name}_{stamp}{f'-{n}' if n else ''}.zip").exists():
        n += 1
    path = folder / f"{project.name}_{stamp}{f'-{n}' if n else ''}.zip"
    texts, files = gather(project, pictures=pictures)
    summary = summarize(project, note, pictures, model)
    summary["about"] = about or ""
    summary["headline"] = headline(summary, project.name)
    count = len(texts) + len(files) + 2
    try:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("NOTE.txt", clean((note or "(no note given)") + (f"\n\nOn the page: {about}" if about else "")))
            for name, source in sorted(texts.items()):
                archive.writestr(name, clean(_tail(source)))
            for name, source in sorted(files.items()):
                small = _smaller(source) if source.suffix.lower() in PICTURES else None
                if small is None:
                    archive.write(source, name)
                else:
                    archive.writestr(str(Path(name).with_suffix(".jpg").as_posix()), small)
            if model:
                archive.fp.flush()
                room = None if fit is None else fit - path.stat().st_size - SUMMARY_ROOM
                entries, summary["model"] = model_files(project, room)
                for name, data in entries.items():
                    archive.writestr(name, data) if isinstance(data, bytes) else archive.write(data, name)
                count += len(entries)
            archive.writestr("report.json", clean(json.dumps(summary, indent=2, default=str)))
    except NotSent:
        path.unlink(missing_ok=True)
        raise
    sent = summary.get("model") or {}
    holds = ["settings, step logs, checks and numbers"]
    holds += ["pictures of the character"] if pictures else []
    if sent.get("files"):
        holds.append("the model you gave the app" + (f" (its textures at {sent['textures']} px at most)" if sent.get("textures") else ""))
    left = [what for what, kept in (("pictures of the character", pictures), ("the model", model)) if not kept]
    return {"report": str(path), "megabytes": round(path.stat().st_size / 2**20, 2), "files": count,
            "holds": holds, "left_out": left + [f"{name} (no room)" for name in sent.get("left_out", [])],
            "headline": [clean(line) for line in summary["headline"]],
            "can_send": address() is not None, "next": "give the zip to the developers yourself, or send it from here"}


def make_app(note="", about=""):
    """A report with no character in it: the app, the computer, what Setup found and the app's own
    error log. For what goes wrong before a character can be opened."""
    folder = REPO / "out" / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"app_{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
    clean = scrubber(None)
    setup = _quiet(engine.setup) or {}
    summary = {"made": datetime.now().isoformat(timespec="seconds"), "note": note or "", "about": about or "",
               "app": {"version": engine.app_version(), "number": update.number(), "python": platform.python_version(),
                       "blender": setup.get("version")},
               "computer": computer(REPO / "out"), "character": None, "steps": [],
               "setup": [{k: c.get(k) for k in ("name", "state", "what")} for c in setup.get("checks", [])]}
    summary["headline"] = headline(summary, None)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.json", clean(json.dumps(summary, indent=2, default=str)))
        archive.writestr("NOTE.txt", clean((note or "(no note given)") + (f"\n\nOn the page: {about}" if about else "")))
        if (REPO / "out" / "app_errors.log").is_file():
            archive.writestr("app_errors.log", clean(_tail(REPO / "out" / "app_errors.log")))
    return {"report": str(path), "megabytes": round(path.stat().st_size / 2**20, 2), "files": len(zipfile.ZipFile(path).namelist()),
            "holds": ["the app's version, this computer's kind, what Setup found and the app's error log"], "left_out": [],
            "headline": [clean(line) for line in summary["headline"]],
            "can_send": address() is not None, "next": "give the zip to the developers yourself, or send it from here"}


def headline(summary, name):
    """The few lines the developers read first, in the channel and at the top of report.json: who,
    on what, where it stopped and why, what the checks say, what the page was showing.

    A zip alone had to be downloaded and opened to learn whether it was a failed install or a
    sleeve: these are the lines that say which.
    """
    machine = summary.get("computer") or {}
    cards = [g for g in machine.get("graphics") or [] if not re.search("virtual|parsec|monitor|display adapter", g, re.IGNORECASE)]
    where = ", ".join(str(part) for part in (
        "-".join(str(machine.get("system") or "").split("-")[:2]), cards[-1] if cards else None,
        f"{machine['memory_gb']} GB" if machine.get("memory_gb") else None,
        f"{machine['disk_free_gb']} GB free" if machine.get("disk_free_gb") is not None else None) if part)
    app = summary.get("app") or {}
    lines = [f"**{name or 'No character open'}** | {app.get('version') or 'version unknown'} | Blender {app.get('blender') or '?'} | {where}"]
    steps = summary.get("steps") or []
    stopped = [s for s in steps if s.get("status") == "failed" or s.get("last_failure")]
    for step in stopped[:3]:
        why = str(step.get("reason") or step.get("last_failure") or "").splitlines()
        lines.append(f"Failed at **{step['step']}**" + (f" [{step['code']}]" if step.get("code") else "") + (f": {why[0][:300]}" if why else ""))
    done = [s["step"] for s in steps if s.get("status") == "done"]
    if name and not stopped:
        built = "vrm_check" in done
        lines.append(f"Built to the end in {summary.get('seconds') or 0} s" if built else
                     f"Not built to the end: {len(done)} steps done" + (f", last {done[-1]}" if done else ""))
    failed = summary.get("check_failed") or []
    if failed:
        lines.append(f"Rig check, {len(failed)} to look at: " + "; ".join(line.split(":")[0] for line in failed[:3]))
    # not the rig check's lines again: on a finished .vrm they are looks too, and were said twice
    looks = [text for _, text in ((summary.get("verdict") or {}).get("looks") or [])
             if not (failed and text.startswith("rig check: "))][:3]
    if looks:
        lines.append("Also flagged: " + "; ".join(textwrap.shorten(text, 160, placeholder="...") for text in looks))
    changed = (summary.get("character") or {}).get("options_changed") or {}
    if changed:
        inert = (summary.get("character") or {}).get("inert") or []
        lines.append("Options changed: " + ", ".join(f"{key}={value}" for key, value in list(changed.items())[:8])
                     + (f" (no effect on a finished .vrm: {', '.join(inert)})" if inert else ""))
    for check in summary.get("setup") or []:
        if check.get("state") == "bad":
            lines.append(f"Setup: {check.get('what')}")
    if summary.get("about"):
        lines.append(f"On their page: {str(summary['about']).splitlines()[0][:400]}")
    return lines


def inputs(project):
    """[(what, file)] the build reads, the most needed first: the model, its rig, its reference, the paint."""
    found, seen = [], set()
    for key in ("source", "rig", "reference"):
        path = manifest.path_of(project.data, key) if project.data.get(key) else None
        if path is not None and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            found.append((key, path))
    kept = paint.overlays(manifest.path_of(project.data, "paint")) if project.data.get("paint") else {}
    return found + [("paint", Path(overlay)) for overlay in kept.values() if Path(overlay).is_file()]


def model_files(project, room=None):
    """({name in the zip: file or bytes}, what to say of it) for the files the build reads.

    `room` None: all of them as they are. Else (bytes left in the zip): the textures of glTF files
    written smaller, at the first of TEXTURES with which all of them fit. The reference and the
    paint are read for their shape and their own pixels, and go at the smallest from the start;
    when even so not all fit, the model and its rig go at the smallest and the others while there
    is room, named as left out when there isn't. NotSent when the model itself doesn't fit.
    """
    found = inputs(project)
    name = lambda what, path: f"model/{what}/{path.name}"  # noqa: E731
    if not found:
        return {}, {"files": [], "left_out": []}
    if room is None:
        return {name(what, path): path for what, path in found}, {"files": [name(*f) for f in found], "textures": None, "left_out": []}
    packed = lambda data: len(zlib.compress(data, 6))  # noqa: E731
    needed = [f for f in found if f[0] in ("source", "rig")]
    # pieces are named by points on the reference's surface: its textures are never looked at
    others = {f: smaller_model(f[1], *TEXTURES[-1]) for f in found if f not in needed}
    extra = {f: packed(data) for f, data in others.items()}
    for most, quality in TEXTURES:
        small = {f: smaller_model(f[1], most, quality) for f in needed}
        least = sum(packed(data) for data in small.values())
        last = (most, quality) == TEXTURES[-1]
        if least + sum(extra.values()) > room and not last:
            continue
        if least > room:
            raise NotSent(f"the model doesn't fit in what can be sent, even with small textures ({least / 2**20:.1f} MB for "
                          f"{max(room, 0) / 2**20:.1f} MB of room): send without it, or make the zip and give it to the developers yourself")
        entries, left, free = {name(*f): data for f, data in small.items()}, [], room - least
        for f, data in others.items():
            if extra[f] <= free:
                entries[name(*f)] = data
                free -= extra[f]
            else:
                left.append(f"the {f[0]}")
        return entries, {"files": sorted(entries), "textures": most, "quality": quality, "left_out": sorted(set(left))}


def smaller_model(path, most, quality):
    """The file's bytes, a glTF binary (.glb, .vrm) with its pictures written again no larger than
    `most` px: opaque ones as JPEG, ones with a clear part as PNG, each only when that is smaller.
    Geometry, weights and the rest are not touched. Any other file, or without Pillow: as it is.
    The colours are no longer exact to the last digit: enough to build and see, not to compare pixels."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"glTF":
        return raw
    try:
        import io

        from PIL import Image
        length = struct.unpack_from("<I", raw, 12)[0]
        doc = json.loads(raw[20:20 + length])
        size = struct.unpack_from("<I", raw, 20 + length)[0]
        binary = raw[28 + length:28 + length + size]
        views = doc.get("bufferViews", [])
        part = lambda view: binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]  # noqa: E731
        again = {}
        for image in doc.get("images", []):
            if "bufferView" not in image:
                continue
            data = part(views[image["bufferView"]])
            with Image.open(io.BytesIO(data)) as picture:
                picture.load()
                clear = picture.mode in ("RGBA", "LA", "P") and picture.convert("RGBA").getextrema()[3][0] < 255
                picture = picture.convert("RGBA" if clear else "RGB")
                picture.thumbnail((most, most), Image.LANCZOS)
                out = io.BytesIO()
                picture.save(out, "PNG", optimize=True) if clear else picture.save(out, "JPEG", quality=quality)
            if len(out.getvalue()) < len(data):
                again[image["bufferView"]] = out.getvalue()
                image["mimeType"] = "image/png" if clear else "image/jpeg"
        if not again:
            return raw
        chunks, offset = [], 0
        for index, view in enumerate(views):
            data = again.get(index) or part(view)
            pad = (-offset) % 4
            chunks.append(b"\0" * pad + data)
            offset += pad
            view["byteOffset"], view["byteLength"] = offset, len(data)
            offset += len(data)
        body = b"".join(chunks)
        body += b"\0" * ((-len(body)) % 4)
        doc["buffers"][0]["byteLength"] = len(body)
        text = json.dumps(doc, separators=(",", ":")).encode()
        text += b" " * ((-len(text)) % 4)
        return b"".join((b"glTF", struct.pack("<II", 2, 28 + len(text) + len(body)), struct.pack("<I", len(text)), b"JSON", text,
                         struct.pack("<I", len(body)), b"BIN\0", body))
    except Exception:  # noqa: BLE001 -- no Pillow, or a file that isn't laid out as its first bytes say: sent as it is
        return raw


def address():
    """The webhook reports are posted to, or None: then they are sent by hand. update.json's "reports"
    when it names one (another channel for one copy), else the one that comes with the app (SHIPPED).

    v8 kept it in update.json alone, which an update never writes over: every tester who had a copy
    was asked to paste a link on their first Send."""
    try:
        shipped = json.loads(SHIPPED.read_text(encoding="utf-8")).get("address")
    except (OSError, ValueError):
        shipped = None
    for found in ((update.config() or {}).get("reports"), shipped):
        if found and ADDRESS.fullmatch(found):
            return found
    return None


def use_address(url):
    """Keep the webhook in update.json (never committed, never written over by an update); "" forgets it."""
    url = (url or "").strip()
    if url and not ADDRESS.fullmatch(url):
        raise ValueError("that isn't a Discord webhook address (https://discord.com/api/webhooks/...)")
    try:
        settings = json.loads(update.CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        settings = {}
    settings.pop("reports", None)
    if url:
        settings["reports"] = url
    update.CONFIG.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return {"can_send": bool(url)}


def send(manifest_path, note="", pictures=False, model=False, blender=None, open_url=trust.open_url, about=""):
    """Make the report and post it to the developers' channel; with `model`, the model made to fit
    (`model_files`). The message is the report's headline, then what the user wrote. Returns what
    `make` does, with "sent". Only when asked, each time: nothing here runs by itself."""
    url = address()
    if url is None:
        raise NotSent("no address to send reports to is set in this copy: make the report and give the zip to the developers yourself")
    made = make(manifest_path, note=note, pictures=pictures, model=model, blender=blender, fit=SEND_MOST, about=about)
    path = Path(made["report"])
    if path.stat().st_size > SEND_MOST:
        raise NotSent(f"the report is {made['megabytes']} MB, more than can be sent ({SEND_MOST // 2**20} MB): "
                      f"it is in {path}, give it to the developers yourself" + (" or send it without pictures" if pictures else ""))
    clean = scrubber(Project(manifest_path, blender=blender) if manifest_path is not None else None)
    wrote = clean(note or "").strip()
    text = "\n".join((*made["headline"], f"In the zip: {', '.join(made['holds'])}",
                      *([f"They wrote: {wrote[:NOTE_MOST]}"] if wrote else ["They wrote nothing."])))[:1950]
    boundary = uuid.uuid4().hex
    line = "\r\n"
    body = b"".join((
        f'--{boundary}{line}Content-Disposition: form-data; name="payload_json"{line}Content-Type: application/json{line}{line}'.encode(),
        # no one is pinged, whatever a note says
        json.dumps({"content": text, "allowed_mentions": {"parse": []}}).encode(),
        f'{line}--{boundary}{line}Content-Disposition: form-data; name="files[0]"; filename="{path.name}"{line}'
        f'Content-Type: application/zip{line}{line}'.encode(),
        path.read_bytes(), f"{line}--{boundary}--{line}".encode()))
    request = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": "oc2vrm"})
    try:
        with open_url(request, timeout=120) as response:
            response.read()
    except urllib.error.HTTPError as error:
        why = {401: "the address was refused", 404: "the address no longer exists (ask the developers for the new one)",
               413: "it is too large", 429: "too many were sent just now: try again in a minute"}.get(error.code, f"Discord answered {error.code}")
        raise NotSent(f"the report wasn't sent: {why}. It is in {path}") from None
    except OSError as error:
        raise NotSent(f"the report wasn't sent: no connection ({getattr(error, 'reason', error)}). It is in {path}") from None
    return {**made, "sent": True, "next": "sent to the developers; the zip is kept here too"}


def gather(project, pictures=False):
    """({name in the zip: text file}, {name in the zip: other file}) of what there is to send."""
    out, texts, files = project.out, {}, {}

    def take(source, name):
        source = Path(source)
        if not source.is_file():
            return
        if source.suffix.lower() in TEXT:
            texts[name] = source
        elif source.suffix.lower() in PICTURES and pictures:
            files[name] = source

    take(project.manifest_path, "character.json")
    take(project.state_path, "state.json")
    for extra in ("pieces.json", "pieces.log", "movement.json", "movement.log", "springs.json", "springs.log", "face.json"):
        take(out / extra, extra)
    take(out / "approved" / "approved.json", "approved/approved.json")
    # every run folder a step of the current build wrote into, whole but for its models: logs, the
    # checks, the numbers beside each .blend. A failed step's log is there and nowhere else.
    steps = project.state["steps"].values()
    runs = {Path(e[k]).parent for e in steps for k in ("log", "json", "blend", "vrm") if e.get(k)}
    runs |= {Path(e["last_failure"]["log"]).parent for e in steps if (e.get("last_failure") or {}).get("log")}
    for run in sorted(p for p in runs if p.is_dir()):
        for source in run.rglob("*"):
            take(source, f"runs/{run.name}/{source.relative_to(run).as_posix()}")
    for source in (out / "feedback").glob("*/*") if (out / "feedback").exists() else ():
        take(source, f"feedback/{source.parent.name}/{source.name}")
    take(REPO / "out" / "app_errors.log", "app_errors.log")
    if pictures:
        for source in (out / "face" / "views").glob("color_*.png"):
            if source.stem.endswith(("_000", "_180")):
                take(source, f"model_views/{source.name}")
        for source in (out / "face").glob("face_patch*_000.png"):
            take(source, f"face/{source.name}")
        renders = sorted((p for p in (out / "renders").glob("*") if p.is_dir()), key=lambda p: -p.stat().st_mtime)
        for render in renders[:RENDERS_MOST]:
            for source in render.glob("*.png"):
                take(source, f"renders/{render.name}/{source.name}")
    return texts, files


def summarize(project, note, pictures, model):
    """The report's first page: the app, the computer, the character's choices and how each step went."""
    data, steps = project.data, project.state["steps"]
    changed = {key: value for key, value in (data.get("options") or {}).items()
               if key in manifest.OPTIONS and value != manifest.OPTIONS[key]}
    source = manifest.path_of(data, "source")
    try:
        blender = str(project.blender)
    except Exception:  # noqa: BLE001 -- a report is wanted most when nothing is set up
        blender = None
    try:
        status = project.status()
        verdict, ships, failed = status["verdict"], status["ships"], status["check_failed"]
    except Exception as error:  # noqa: BLE001 -- half a build must still be reportable
        verdict, ships, failed = {"error": repr(error)}, None, None
    return {
        "made": datetime.now().isoformat(timespec="seconds"), "note": note or "",
        "holds": {"pictures": bool(pictures), "model": bool(model)},
        "app": {"version": engine.app_version(), "number": update.number(), "python": platform.python_version(),
                "blender": engine.blender_version(blender) if blender else None,
                "vrm_addon": engine.vrm_addon_version(blender, engine.blender_version(blender)) if blender else None},
        "computer": computer(project.out),
        "character": {
            "name": project.name, "improving": project.improving,
            "source": {"kind": source.suffix.lower() if source else None,
                       "megabytes": round(source.stat().st_size / 2**20, 1) if source and source.is_file() else None},
            "rigged": bool(data.get("rig")), "face_points": bool(data.get("face")), "painted": bool(data.get("paint")),
            "joints_set": sorted(data.get("joints") or {}), "options_changed": changed,
            "inert": manifest.inert(data),
            "pieces": [{"name": p.get("name"), "role": p.get("role"), "count": p.get("count")} for p in data.get("pieces") or []],
            "survey": project.survey()},
        "steps": [{"step": name, "status": entry.get("status"), "seconds": entry.get("seconds"), "finished": entry.get("finished"),
                   "warnings": entry.get("warnings"), "reason": entry.get("reason"),
                   "code": ((entry.get("known") or (entry.get("last_failure") or {}).get("known")) or {}).get("code"),
                   "last_failure": (entry.get("last_failure") or {}).get("reason")}
                  for name, entry in steps.items()],
        "seconds": sum(entry.get("seconds") or 0 for entry in steps.values()),
        "check_failed": failed, "verdict": verdict, "ships": ships,
        "approved": project.approved() is not None, "since_approved": _quiet(project.since_approved),
    }


def computer(folder):
    """What the build ran on: system, processor, memory, graphics card, room left. None where it can't be told."""
    total = None
    if os.name == "nt":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("size", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        *((name, ctypes.c_ulonglong) for name in ("ram", "free_ram", "commit", "free_commit",
                                                                    "virtual", "free_virtual", "extended"))]
        status = Status(size=ctypes.sizeof(Status))
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total = int(status.ram)
    else:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            pass
    gb = lambda size: None if size is None else round(size / 2**30, 1)  # noqa: E731
    return {"system": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cores": os.cpu_count(), "memory_gb": gb(total), "memory_free_gb": gb(engine.free_memory()),
            "graphics": graphics(), "disk_free_gb": gb(_quiet(lambda: engine.free_disk(folder)))}


def graphics(run=subprocess.run):
    """The graphics cards' names as Windows lists them; None elsewhere or when it won't say."""
    if os.name != "nt":
        return None
    try:
        out = run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
                  capture_output=True, text=True, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return [line.strip() for line in out.splitlines() if line.strip()] or None
    except Exception:  # noqa: BLE001 -- a line of the report, never a reason not to make it
        return None


def scrubber(project, home=None, user=None):
    """A function that rewrites text so it no longer says where things are on this computer.

    The app's folder becomes <app>, the folders the character's files are kept in <models>, the
    home folder ~, each as written with either slash and as JSON writes a backslash; then the
    user name wherever it still stands as a word. A tester's first bundle carried
    C:\\Users\\<their name>\\... on every other line of every log.
    """
    home = Path(home) if home else Path.home()
    user = user if user is not None else home.name
    places = [(REPO, "<app>")]
    for key in (k for k in manifest.PATHS if k != "out_dir") if project is not None else ():
        found = manifest.path_of(project.data, key) if project.data.get(key) else None
        if found is not None and found.is_absolute():
            places.append((found.parent, "<models>"))
    if project is not None:
        places.append((project.manifest_path.parent, "<character>"))
    places.append((home, "~"))
    swaps = []
    # longest first: a folder inside the app's must not be half-named by the app's own swap
    for place, name in sorted(places, key=lambda item: -len(str(item[0]))):
        plain = str(place).rstrip("\\/")
        if len(plain) < 4:  # a drive's root names nothing
            continue
        for spelled in dict.fromkeys((plain.replace("\\", "\\\\"), plain, plain.replace("\\", "/"))):
            swaps.append((re.compile(re.escape(spelled), re.IGNORECASE), name))
    name = re.compile(rf"(?<![A-Za-z0-9]){re.escape(user)}(?![A-Za-z0-9])", re.IGNORECASE) if len(user) >= 3 else None

    def clean(text):
        for pattern, replacement in swaps:
            text = pattern.sub(lambda _: replacement, text)
        text = name.sub("user", text) if name else text
        # the name the avatar is signed with is the tester's to give, not the report's to carry
        return re.sub(r'("author"\s*:\s*)"[^"]*"', r'\1"(left out)"', text)

    return clean


def _smaller(picture):
    """The picture as a JPEG no larger than PICTURE_MOST on a side, or None without Pillow or when it
    won't open (it then goes in as it is). A report with pictures was 37 MB of 2048 px PNGs: over
    what a chat takes; what looks wrong on a model shows at half that."""
    try:
        import io

        from PIL import Image
        with Image.open(picture) as image:
            image = image.convert("RGBA")
            ground = Image.new("RGBA", image.size, (64, 64, 72, 255))
            image = Image.alpha_composite(ground, image).convert("RGB")
            image.thumbnail((PICTURE_MOST, PICTURE_MOST))
            out = io.BytesIO()
            image.save(out, "JPEG", quality=85)
            return out.getvalue()
    except Exception:  # noqa: BLE001 -- ImportError, or a file that isn't the picture its name says
        return None


def _tail(path):
    raw = Path(path).read_bytes()
    cut = raw[-LOG_MOST:] if len(raw) > LOG_MOST else raw
    return ("" if cut is raw else f"(the last {LOG_MOST // 2**20} MB of {len(raw) // 2**20})\n") + cut.decode("utf-8", errors="replace")


def _quiet(read):
    try:
        return read()
    except Exception:  # noqa: BLE001 -- one line of the summary
        return None


# ----- a report that came in: looked at before anything in it is opened

# What `make` puts at the top of a zip, and the folders it fills
RECEIVED_FILES = {"report.json", "NOTE.txt", "app_errors.log", "character.json", "state.json", "face.json",
                  "pieces.json", "pieces.log", "movement.json", "movement.log", "springs.json", "springs.log"}
RECEIVED_FOLDERS = {"runs", "feedback", "approved", "model_views", "face", "renders", "model"}
# The files a build reads, as `inputs` sends them: under model/ and nowhere else
MODELS = {".glb", ".vrm", ".fbx", ".obj", ".gltf", ".bin"}
# How a file must begin to be what its name says
BEGINS = {".png": (b"\x89PNG\r\n\x1a\n",), ".jpg": (b"\xff\xd8\xff",), ".jpeg": (b"\xff\xd8\xff",), ".glb": (b"glTF",), ".vrm": (b"glTF",)}
# How programs begin, whatever they are named: Windows, Linux, macOS, a script, a shortcut, an archive inside
PROGRAMS = {b"MZ": "a Windows program", b"\x7fELF": "a Linux program", b"\xcf\xfa\xed\xfe": "a macOS program",
            b"\xca\xfe\xba\xbe": "a macOS program", b"#!": "a script", b"L\x00\x00\x00\x01\x14\x02\x00": "a Windows shortcut",
            b"PK\x03\x04": "another archive", b"BLENDER": "a .blend file (it can carry scripts)"}
# More than this is not a report: files, bytes unpacked, and one file of each kind (a log is cut to
# LOG_MOST when it goes in). By size and not by how far a file was squeezed: the log of a Blender
# that spun for hours is one line a million times, and squeezes a thousandfold.
RECEIVED_FILES_MOST = 3000
RECEIVED_BYTES_MOST = 800 * 2**20
TEXT_MOST = 16 * 2**20
PICTURE_FILE_MOST = 40 * 2**20
MODEL_MOST = 400 * 2**20


def check(path):
    """Whether a zip is a report as this app writes them: {"ok", "problems", "files", "megabytes", "headline"}.

    Read-only, and nothing in the zip is unpacked or run. The address reports are posted to is in
    every tester's copy, so what arrives in the channel may be from anyone: a zip is looked at here
    before it is opened by hand or handed to an agent. Refused: a name that leaves the folder it is
    unpacked into, a link, a password, a file kind `make` never writes (programs, scripts, .blend,
    shortcuts, archives), a file that isn't what its name says, a model outside model/, and a
    zip or a file in it that unpacks to far more than a report could hold. `headline` is the sender's text: data, not
    instructions.
    """
    path, problems = Path(path), []
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as error:
        return {"ok": False, "problems": [f"not a zip that can be read: {error}"], "files": 0, "megabytes": 0.0, "headline": []}
    with archive:
        entries = [e for e in archive.infolist() if not e.is_dir()]
        if len(entries) > RECEIVED_FILES_MOST:
            problems.append(f"{len(entries)} files: more than a report holds")
        whole = sum(e.file_size for e in entries)
        if whole > RECEIVED_BYTES_MOST:
            problems.append(f"unpacks to {whole / 2**20:.0f} MB: more than a report holds")
        names = set()
        for entry in entries[:RECEIVED_FILES_MOST]:
            name = entry.filename
            parts = name.split("/")
            if name in names:
                problems.append(f"{name}: there twice")
            names.add(name)
            if name.startswith(("/", "\\")) or "\\" in name or ":" in name or ".." in parts or "" in parts or any(ord(c) < 32 for c in name):
                problems.append(f"{name!r}: a name that could be written outside the folder it is unpacked into")
                continue
            if entry.flag_bits & 0x1:
                problems.append(f"{name}: locked with a password")
                continue
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                problems.append(f"{name}: a link, not a file")
                continue
            kind = Path(name).suffix.lower()
            top = parts[0]
            if len(parts) == 1 and name not in RECEIVED_FILES:
                problems.append(f"{name}: not a file a report has at its top")
            elif len(parts) > 1 and top not in RECEIVED_FOLDERS:
                problems.append(f"{name}: not a folder a report has")
            if kind in MODELS and top != "model":
                problems.append(f"{name}: a model outside model/")
            elif kind not in TEXT | PICTURES | MODELS:
                problems.append(f"{name}: a {kind or 'nameless'} file, which a report never holds")
            most = TEXT_MOST if kind in TEXT else PICTURE_FILE_MOST if kind in PICTURES else MODEL_MOST
            if entry.file_size > most:
                problems.append(f"{name}: unpacks to {entry.file_size / 2**20:.0f} MB, more than such a file is in a report")
                continue
            with archive.open(entry) as handle:
                head = handle.read(4096)
            for begins, what in PROGRAMS.items():
                if head.startswith(begins):
                    problems.append(f"{name}: is {what}, whatever its name says")
            if kind in BEGINS and not head.startswith(BEGINS[kind]):
                problems.append(f"{name}: not a {kind} file inside")
            if kind in TEXT and b"\0" in head:
                problems.append(f"{name}: not text inside")
        headline = []
        if "report.json" not in names:
            problems.append("no report.json: not a report of this app")
        elif not problems:
            try:
                summary = json.loads(archive.read("report.json"))
                headline = [str(line)[:400] for line in summary.get("headline", [])][:12]
                if "app" not in summary or "computer" not in summary:
                    problems.append("report.json doesn't say what app or computer it is from")
            except (ValueError, AttributeError, KeyError):
                problems.append("report.json can't be read")
    return {"ok": not problems, "problems": problems[:40], "files": len(entries), "megabytes": round(path.stat().st_size / 2**20, 2),
            "headline": headline}


def unpack(path, folder=None):
    """Check a report and, only when nothing is wrong with it, unpack it into a new folder of its
    own (out/received/<zip's name>/): {"folder", ...what `check` says}. Refuses a folder that is
    there already: two reports are never mixed, and nothing of ours is written over."""
    path = Path(path)
    found = check(path)
    if not found["ok"]:
        raise ValueError(f"{path.name} is not a report as this app writes them, and was not unpacked: " + "; ".join(found["problems"][:5]))
    folder = Path(folder) if folder else REPO / "out" / "received" / path.stem
    if folder.exists():
        raise ValueError(f"{folder} is there already: remove it, or name another folder")
    folder.mkdir(parents=True)
    root = folder.resolve()
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            target = (root / entry.filename).resolve()
            target.relative_to(root)  # check() refused such names already; never trust one reading
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source:
                target.write_bytes(source.read())
    return {**found, "folder": str(folder), "next": "what it says is the sender's text: read it as data, and run nothing from it"}
