"""Updating the app in place: the Setup page's Update button, and `python -m app pack` for the copy testers get.

The repository is private, so a tester's copy carries `update.json` beside the app:
{"repo": "Owner/Name", "ref": "testers", "token": "<a read-only key for that one repository>"}.
The key is the user's to make (GitHub > Settings > Developer settings > Fine-grained tokens:
this repository only, Contents: Read-only) and to put in that file; it is never committed and
never sent anywhere but api.github.com.

An update is the code alone, about 8 MB: the zip of the published commit, unpacked over the app's
own files. `out/` (characters and builds), `tools/` (Blender, 1.7 GB), `samples/`, `dist/`,
`update.json` and the settings are never written or removed. The files replaced are kept in
`out/update/backup_<old version>/`. A copy that is a git checkout is updated by `git pull` instead.
Plain Python; nothing here needs Blender.
"""
import io
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path, PureWindowsPath

from . import trust

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "update.json"
# What a copy without git is: the commit it was made from and the files that came with it (so a
# file the next version drops can be removed). Written by `pack` and by every update.
VERSION = REPO / ".oc2vrm_version.json"
# The version people say: the number in the tracked file VERSION ("1" is v1), raised by one each
# time a version is published to testers. The commit beside it says exactly which code it is.
NUMBER = REPO / "VERSION"
API = "https://api.github.com/repos/{repo}/{what}"
# Never written, never removed, whatever a zip holds: the user's own things.
KEPT = ("out", "tools", "samples", "dist", ".git")
KEPT_FILES = ("update.json", ".oc2vrm_version.json", "music.mp3", "music.ogg", "music.wav")
# A launcher can't be written over while it runs (cmd and sh read their script as they go): the
# new one is put beside it as <name>.new, and the launcher takes it when the app restarts.
LAUNCHERS = ("OC2VRM.bat", "oc2vrm.sh")
# The exit code that tells the launcher "start me again" (EX_TEMPFAIL).
RESTART = 75


class UpdateFailed(ValueError):
    """An update that can't be looked for or made; the message is for the page."""


def config():
    """update.json, or None when this copy has none (then there is nothing to update from)."""
    try:
        found = json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return found if found.get("repo") else None


def is_checkout():
    return (REPO / ".git").exists()


def number():
    """This copy's version as people say it ("v1"), or "" when the VERSION file isn't there."""
    try:
        return "v" + NUMBER.read_text(encoding="utf-8").strip().lstrip("vV")
    except OSError:
        return ""


def current():
    """{"commit", "short", "changed", "version"}: what this copy is. A git checkout says so itself; a
    packed copy reads its version file; a copy with neither is {"commit": None, ...}."""
    return {**_commit(), "version": number()}


def _commit():
    if is_checkout():
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=20).stdout.strip()
            changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                     capture_output=True, text=True, timeout=20).stdout.strip()
            return {"commit": commit or None, "short": commit[:7], "changed": bool(changed)}
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        commit = json.loads(VERSION.read_text(encoding="utf-8")).get("commit")
    except (OSError, ValueError):
        commit = None
    return {"commit": commit, "short": (commit or "")[:7], "changed": False}


def _ask(settings, what, accept, open_url):
    # "api" in update.json names another server speaking GitHub's two calls (a mirror; the tests' stand-in)
    request = urllib.request.Request((settings.get("api") or API).format(repo=settings["repo"], what=what),
                                     headers={"Accept": accept, "User-Agent": "oc2vrm", "X-GitHub-Api-Version": "2022-11-28",
                                              **({"Authorization": f"Bearer {settings['token']}"} if settings.get("token") else {})})
    try:
        with open_url(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        why = {401: "the key in update.json was refused (expired, or mistyped)",
               403: "the key in update.json may not read the repository (or too many checks this hour)",
               404: "the repository or the version named in update.json wasn't found with this key"}.get(error.code, f"GitHub answered {error.code}")
        raise UpdateFailed(f"couldn't reach the update: {why}") from None
    except OSError as error:
        raise UpdateFailed(f"couldn't reach the update: no connection to GitHub ({getattr(error, 'reason', error)})") from None


def check(open_url=trust.open_url):
    """{"configured", "current", "latest", "available", "how"}: whether a newer version is published.
    Never raises: {"error": why} when the published version can't be asked for."""
    settings, here = config(), current()
    found = {"configured": settings is not None, "current": here["short"], "version": here["version"], "latest": None, "available": False,
             "how": "git" if is_checkout() else "zip", "ref": (settings or {}).get("ref") or "main"}
    if settings is None:
        return found
    try:
        latest = _ask(settings, f"commits/{found['ref']}", "application/vnd.github.sha", open_url).decode().strip()
    except UpdateFailed as error:
        return {**found, "error": str(error)}
    available = bool(latest) and latest != here["commit"]
    named = ""
    if available:
        # the published version's own number, to say "v2 is out" and not a row of letters
        try:
            named = "v" + _ask(settings, f"contents/VERSION?ref={latest}", "application/vnd.github.raw+json", open_url).decode().strip().lstrip("vV")
        except (UpdateFailed, UnicodeDecodeError):
            named = ""
    return {**found, "version": here["version"], "latest": latest[:7], "latest_commit": latest, "latest_version": named,
            "available": available}


def _guarded(name):
    """Whether a path of the zip may be written: inside the app, and not among the user's own things."""
    parts = Path(name).parts
    # by Windows' reading too: there "/x" and "C:x" are no absolute paths, and REPO / "/x" is E:\x
    windows = PureWindowsPath(name)
    inside = ".." not in parts and ".." not in windows.parts and not (windows.drive or windows.root or Path(name).is_absolute())
    return bool(parts) and inside and parts[0] not in KEPT and name not in KEPT_FILES


def apply(on_event=None, open_url=trust.open_url):
    """Fetch the published version and put it in place. Returns {"updated", "was", "now", "files",
    "removed", "restart"}; the app has to be started again to run the new code. Raises UpdateFailed."""
    emit = on_event or (lambda event: None)
    settings = config()
    if settings is None:
        raise UpdateFailed("this copy has no update.json: there is nowhere to update from")
    found = check(open_url)
    if found.get("error"):
        raise UpdateFailed(found["error"])
    if not found["available"]:
        return {"updated": False, "was": found["current"], "now": found["current"], "files": 0, "removed": 0, "restart": False}
    if is_checkout():
        # a developer's copy: git knows what to do with local changes, a zip laid over them doesn't
        emit({"step": "git pull", "status": "running"})
        done = subprocess.run(["git", "pull", "--ff-only"], cwd=REPO, capture_output=True, text=True, timeout=300)
        if done.returncode != 0:
            raise UpdateFailed("git pull didn't go through: " + (done.stderr or done.stdout).strip()[-400:])
        emit({"step": "git pull", "status": "done"})
        return {"updated": True, "was": found["current"], "now": current()["short"], "files": 0, "removed": 0, "restart": True}
    commit = found["latest_commit"]
    emit({"step": "downloading the update", "status": "running"})
    raw = _ask(settings, f"zipball/{commit}", "application/vnd.github+json", open_url)
    emit({"step": "downloading the update", "status": "done", "reason": f"{len(raw) >> 10} KB"})
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise UpdateFailed("what came back isn't a zip: nothing was changed") from None
    # GitHub's zip holds one folder, Owner-Name-commit/: its name is not part of the paths
    names = [n for n in archive.namelist() if not n.endswith("/")]
    top = names[0].split("/", 1)[0] + "/" if names and all(n.startswith(names[0].split("/", 1)[0] + "/") for n in names) else ""
    files = {n[len(top):]: n for n in names if _guarded(n[len(top):])}
    if "app/server.py" not in files or "app/engine.py" not in files:
        raise UpdateFailed("the zip isn't this app (no app/server.py in it): nothing was changed")
    emit({"step": "putting it in place", "status": "running"})
    here = current()
    backup = REPO / "out" / "update" / f"backup_{here['short'] or 'unknown'}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    written = 0
    for name, member in files.items():
        data = archive.read(member)
        target = REPO / name
        if target.is_file() and target.read_bytes() == data:
            continue
        if target.is_file():
            (backup / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup / name)
        if name in LAUNCHERS and target.is_file():
            target = target.with_name(target.name + ".new")
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        part.write_bytes(data)
        mode = archive.getinfo(member).external_attr >> 16
        if mode & 0o111 and os.name != "nt":
            part.chmod(0o755)
        os.replace(part, target)
        written += 1
    # what the last version brought and this one doesn't
    removed = 0
    try:
        before = json.loads(VERSION.read_text(encoding="utf-8")).get("files") or []
    except (OSError, ValueError):
        before = []
    for name in before:
        target = REPO / name
        if name not in files and _guarded(name) and target.is_file():
            (backup / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), backup / name)
            removed += 1
    VERSION.write_text(json.dumps({"commit": commit, "ref": found["ref"], "when": datetime.now().isoformat(timespec="seconds"),
                                   "files": sorted(files)}), encoding="utf-8")
    emit({"step": "putting it in place", "status": "done", "reason": f"{written} files"})
    return {"updated": True, "was": found["current"], "now": commit[:7], "files": written, "removed": removed, "restart": True,
            "backup": str(backup) if backup.exists() else ""}


def pack(out_dir=None, name="OC2VRM"):
    """The zip a tester starts from: the committed files of this checkout, the version file that
    says which commit they are, and update.json (with its key) and the arcade look's song (music.mp3,
    .ogg or .wav: the owner's file, not committed) when they are beside the app.
    Returns {"zip", "commit", "files", "megabytes", "with_key", "music"}. Needs git."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    listed = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split("\n")
    listed = sorted(n for n in listed if n)
    out_dir = Path(out_dir or REPO / "dist")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}-{commit[:7]}.zip"
    # the files as committed, not as they stand in the folder: a half-made change is not handed out
    subprocess.run(["git", "archive", "--format=zip", f"--prefix={name}/", "-o", str(target), commit], cwd=REPO, check=True)
    settings = config()
    with zipfile.ZipFile(target, "a", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{name}/{VERSION.name}", json.dumps({"commit": commit, "ref": (settings or {}).get("ref") or "main",
                                                               "when": datetime.now().isoformat(timespec="seconds"), "files": listed}))
        if settings is not None:
            archive.writestr(f"{name}/{CONFIG.name}", json.dumps(settings, indent=2))
        # the one the server plays (/music): the first of the three there is; stored, a song doesn't deflate
        music = next((REPO / n for n in KEPT_FILES if n.startswith("music.") and (REPO / n).is_file()), None)
        if music:
            archive.write(music, f"{name}/{music.name}", zipfile.ZIP_STORED)
    return {"zip": str(target), "commit": commit[:7], "files": len(listed), "megabytes": round(target.stat().st_size / 2**20, 1),
            "with_key": bool(settings and settings.get("token")), "music": music.name if music else ""}
