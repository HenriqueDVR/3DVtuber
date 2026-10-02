"""The app's own copies of what its pages otherwise load from the internet. Plain Python.

The 3D views are three.js and three-vrm and the viewer's webcam test is MediaPipe's face
tracker; the pages name all of them by their CDN addresses, so without a connection the views
stayed empty. The copies live in app/gui/vendor/ (17 MB, in the repository so a fresh checkout
runs offline; their licences in THIRD_PARTY.md there), and the server gives the pages those
copies (`local`). `fetch` puts back any that is missing, and is how a newer version gets in.

Every file is pinned by its SHA-256. jsDelivr serves an npm version's files unchanged for good
and publishes these hashes itself (data.jsdelivr.com); the model comes from Google's own bucket,
pinned from the copy this was tested with (its MD5 matched the bucket's). A download that differs
is refused and nothing is kept of it.
"""

import hashlib
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

FOLDER = Path(__file__).resolve().parent / "gui" / "vendor"
SERVED = "/gui/vendor/"
NPM = "https://cdn.jsdelivr.net/npm/"
# (what needs it, the address the pages use, {file under it: SHA-256}). "views" is every 3D view
# in the app; "tracker" only the viewer's webcam test. three's addons are the ones the pages
# import: tests/test_app.py checks none is missing here.
PACKAGES = [
    ("views", NPM + "three@0.180.0/", {
        "build/three.module.js": "c8211c69345d2e9949dc7a8ac969380497aa0600a5a8ac6a459c8cd02dd9cb8a",
        "build/three.core.js": "eb077d2417f61d3e6d9264c317cabc4ea35769ed6b0ab533067292a550784c20",
        "examples/jsm/controls/OrbitControls.js": "b97879c748170baadeb3fb84cea1ffdf4674e283dc06042f34e2acb95a76042c",
        "examples/jsm/controls/TransformControls.js": "621fb8894c0f748c2be2a6ff8a6bdd72c18a22b24ddfe21d506514544df5c84e",
        "examples/jsm/loaders/GLTFLoader.js": "67ac5551fdafa6e349bd80c8f8e5e39c136d6b2fb1ad647db9abb21dc86f9e4a",
        "examples/jsm/utils/BufferGeometryUtils.js": "fda7e946b8e0b5ab39b779206589e7a1079a22eb24efb89d7223e03fdfb1f751"}),
    ("views", NPM + "@pixiv/three-vrm@3.5.5/", {
        "lib/three-vrm.module.js": "bfc0f3bd49ee5b706b3055d366759e881f20dee32be344c4c8f586600f223434"}),
    ("tracker", NPM + "@mediapipe/tasks-vision@0.10.14/", {
        "vision_bundle.mjs": "e77f281f9619150d937023c355bae170e9120e3b9e43f1e23a2a7bee07197669",
        "wasm/vision_wasm_internal.js": "9440cf0cc0cea21800e31581ec32aeedcc5fbf9df4509796bbc7d3f99e52ab9c",
        "wasm/vision_wasm_internal.wasm": "f82a8e6c05e08a44cc9f9e7ec5f845935bcbb1b1500ebe8c2f4812fb4e2917dc"}),
    ("tracker", "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/", {
        "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"}),
]


def _under(address):
    """Where an address's files are kept, relative to FOLDER: its path without the host."""
    return urlparse(address).path.strip("/")


def _kept(address, files):
    return all((FOLDER / _under(address) / name).is_file() for name in files)


def status():
    """{"views", "tracker": whether each is wholly kept here, "bytes", "folder"}."""
    kept = {kind: all(_kept(address, files) for k, address, files in PACKAGES if k == kind) for kind in ("views", "tracker")}
    size = sum(p.stat().st_size for p in FOLDER.rglob("*") if p.is_file()) if FOLDER.is_dir() else 0
    return {**kept, "bytes": size, "folder": str(FOLDER)}


def local(page):
    """A page's text with each CDN address whose files are all kept here turned into the app's
    own. One package at a time: a views-only copy must leave the tracker's addresses alone."""
    for _, address, files in PACKAGES:
        if _kept(address, files):
            page = page.replace(address, SERVED + _under(address) + "/")
    return page


def fetch(on_event=None, open_url=urllib.request.urlopen):
    """Download every pinned file not kept yet; returns `status()` with what was fetched.

    Each goes to a .part file, is hashed as it arrives and only takes its name once the hash is
    the pinned one, so a cut connection or a changed file never leaves something a page would load.
    """
    emit = on_event or (lambda event: None)
    fetched = []
    for _, address, files in PACKAGES:
        for name, pinned in files.items():
            path = FOLDER / _under(address) / name
            if path.is_file():
                continue
            emit({"step": name.rsplit("/", 1)[-1], "status": "running"})
            path.parent.mkdir(parents=True, exist_ok=True)
            part, digest = path.with_name(path.name + ".part"), hashlib.sha256()
            request = urllib.request.Request(address + name, headers={"User-Agent": "vtuber-maker"})
            try:
                with open_url(request, timeout=120) as response, open(part, "wb") as out:
                    while chunk := response.read(1 << 16):
                        digest.update(chunk)
                        out.write(chunk)
                if digest.hexdigest() != pinned:
                    raise ValueError(f"{address}{name} isn't the file this app was tested with "
                                     f"(SHA-256 {digest.hexdigest()}, expected {pinned}): not kept")
                os.replace(part, path)
            finally:
                part.unlink(missing_ok=True)
            fetched.append(address + name)
            emit({"step": name.rsplit("/", 1)[-1], "status": "done"})
    return dict(status(), fetched=fetched)
