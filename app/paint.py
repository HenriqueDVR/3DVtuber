"""Paint on a character's texture, kept apart from the model. Plain Python, no Blender.

The Paint page draws on the source model's base-colour pictures. What is drawn is kept as one
transparent overlay per picture (`<paint folder>/<picture's name>.png`), never baked into the
generated texture: erasing it gives the original back, and a remade face patch keeps the paint.
A build reads `painted()`: a copy of the source .glb with each overlay laid over its picture.
Only the pictures' bytes change in that copy; geometry, UVs and vertex order are the source's,
so piece points and face points still hold.
"""
import hashlib
import io
import json
import re
import struct
from pathlib import Path

JSON_CHUNK, BIN_CHUNK = 0x4E4F534A, 0x004E4942
# One overlay may be this big (bytes): a 4096 px picture painted all over is about 20 MB of PNG.
OVERLAY_MAX = 64 << 20


class NotPaintable(ValueError):
    pass


def read_glb(path):
    """(the glTF JSON, its binary chunk) of a .glb file."""
    raw = Path(path).read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise NotPaintable(f"{Path(path).name} is not a .glb file")
    length, kind = struct.unpack_from("<II", raw, 12)
    if kind != JSON_CHUNK:
        raise NotPaintable(f"{Path(path).name}: no JSON chunk")
    gltf = json.loads(raw[20:20 + length])
    at, binary = 20 + length, b""
    if at + 8 <= len(raw):
        length, kind = struct.unpack_from("<II", raw, at)
        if kind == BIN_CHUNK:
            binary = raw[at + 8:at + 8 + length]
    return gltf, binary


def write_glb(path, gltf, binary):
    text = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    binary = bytes(binary) + b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total))
        handle.write(struct.pack("<II", len(text), JSON_CHUNK) + text)
        handle.write(struct.pack("<II", len(binary), BIN_CHUNK) + binary)
    temporary.replace(path)


def overlay_name(image, index):
    """The overlay file of a glTF image: by the picture's name, so a remade face patch (whose
    pictures keep their names, not always their numbers) finds its paint again."""
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", image.get("name") or "") or f"image_{index}"
    return f"{name}.png"


def pictures(gltf):
    """The pictures paint can go on: those a material uses as its base colour, kept inside the
    file. [{"index", "name", "overlay", "materials"}]"""
    used = {}
    for material in gltf.get("materials", []):
        texture = (material.get("pbrMetallicRoughness") or {}).get("baseColorTexture")
        if texture is None:
            continue
        index = gltf["textures"][texture["index"]].get("source")
        if index is not None and "bufferView" in gltf["images"][index]:
            used.setdefault(index, []).append(material.get("name") or "")
    return [{"index": index, "name": gltf["images"][index].get("name") or f"image {index}",
             "overlay": overlay_name(gltf["images"][index], index), "materials": names}
            for index, names in sorted(used.items())]


def replace_images(gltf, binary, new):
    """(gltf, binary) with the images in `new` ({image index: PNG bytes}) put in place of the old.
    Every buffer view is moved to make room; nothing else in the file changes."""
    if len(gltf.get("buffers", [])) != 1 or "uri" in gltf["buffers"][0]:
        raise NotPaintable("the model keeps its data outside the .glb")
    gltf = json.loads(json.dumps(gltf))
    views = gltf["bufferViews"]
    swap = {gltf["images"][index]["bufferView"]: data for index, data in new.items()}
    out = bytearray()
    # in file order: a view's bytes stay where they were among the others
    for number in sorted(range(len(views)), key=lambda n: views[n].get("byteOffset", 0)):
        view = views[number]
        start = view.get("byteOffset", 0)
        data = swap.get(number, binary[start:start + view["byteLength"]])
        out += b"\0" * (-len(out) % 4)
        view["byteOffset"], view["byteLength"] = len(out), len(data)
        out += data
    for index in new:
        gltf["images"][index]["mimeType"] = "image/png"
    gltf["buffers"][0]["byteLength"] = len(out) + (-len(out) % 4)
    return gltf, bytes(out)


def composite(picture, overlay):
    """PNG bytes of `overlay` (RGBA PNG bytes) laid over `picture` (the model's own image bytes)."""
    from PIL import Image
    base = Image.open(io.BytesIO(picture))
    alpha = base.mode in ("RGBA", "LA") or "transparency" in base.info
    base = base.convert("RGBA")
    over = Image.open(io.BytesIO(overlay)).convert("RGBA")
    if over.size != base.size:
        # painted on a picture of another size (a face patch remade larger): stretched to fit
        over = over.resize(base.size, Image.LANCZOS)
    done = Image.alpha_composite(base, over)
    out = io.BytesIO()
    (done if alpha else done.convert("RGB")).save(out, "PNG")
    return out.getvalue()


def overlays(folder):
    """{overlay file name: path} of the paint kept in a folder."""
    folder = Path(folder) if folder else None
    return {p.name: p for p in sorted(folder.glob("*.png"))} if folder and folder.is_dir() else {}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def painted(source, folder, out, name):
    """The model a build should read: `source` itself when the character has no paint, else
    `<out>/paint/<name>_painted.glb`, made again whenever the source or an overlay changed.

    Overlays whose picture the source doesn't hold are left out (and stay on disk).
    """
    source = Path(source)
    kept = overlays(folder)
    if not kept or source.suffix.lower() != ".glb":
        return source
    stat = source.stat()
    stamp = {"source": str(source.resolve()), "size": stat.st_size, "modified": stat.st_mtime_ns,
             "overlays": {n: _sha(p) for n, p in kept.items()}}
    target = Path(out) / "paint" / f"{name}_painted.glb"
    record = target.with_suffix(".json")
    try:
        if target.exists() and json.loads(record.read_text(encoding="utf-8")) == stamp:
            return target
    except (OSError, ValueError):
        pass
    gltf, binary = read_glb(source)
    new = {}
    for picture in pictures(gltf):
        if picture["overlay"] in kept:
            view = gltf["bufferViews"][gltf["images"][picture["index"]]["bufferView"]]
            start = view.get("byteOffset", 0)
            new[picture["index"]] = composite(binary[start:start + view["byteLength"]], kept[picture["overlay"]].read_bytes())
    if not new:
        return source
    target.parent.mkdir(parents=True, exist_ok=True)
    write_glb(target, *replace_images(gltf, binary, new))
    record.write_text(json.dumps(stamp, indent=1), encoding="utf-8")
    return target


def save(folder, source, images):
    """Keep what the page painted: `images` is {overlay file name: PNG data URL, or None to drop
    that picture's paint}. Only names of the source's own pictures are taken. Returns the names kept."""
    import base64
    allowed = {p["overlay"] for p in pictures(read_glb(source)[0])}
    folder = Path(folder)
    for name, url in images.items():
        if name not in allowed:
            raise ValueError(f"{name} is not one of this model's pictures")
        if url is None:
            (folder / name).unlink(missing_ok=True)
            continue
        head, _, data = str(url).partition(",")
        if head != "data:image/png;base64":
            raise ValueError("paint is saved as PNG")
        raw = base64.b64decode(data)
        if len(raw) > OVERLAY_MAX:
            raise ValueError(f"{name}: over {OVERLAY_MAX >> 20} MB of paint")
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / (name + ".tmp")
        temporary.write_bytes(raw)
        temporary.replace(folder / name)
    return sorted(overlays(folder))
