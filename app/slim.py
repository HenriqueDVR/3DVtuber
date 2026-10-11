"""The pictures of an exported .vrm made no bigger than the avatar uses. Plain Python and Pillow, no Blender.

The export step's last act (`engine.Project._run_step`), on the file Blender wrote: Blender's Python
has no picture codec. Two things, each seen by the user in Warudo beside the same file without it
(2026-10-11, the fox girl and model A: no difference):

- A picture whose triangles use a small part of it is cut to that part. The face patch is a 2048 px
  view of the whole head of which the face is about a third, and hair lying over the face reads a
  second copy of it (`face_patch_over`): 21 MB each as an app loads them, 7 or 8 after. The texels
  stay as they are; the UVs of the triangles that read the picture move to the new frame.
- A normal map of 4096 px (a GLB input carries one) is halved: 85 MB loaded, for relief that shows
  no finer.

Nothing else in the file changes, and a file this can't read as it expects is left as it was, with
the reason. Not for a .vrm taken in as it is: its pictures are its author's.
"""
import array
import io
import math
import struct
import sys
from pathlib import Path

from . import paint

# A cut picture's corners lie on multiples of ALIGN texels, PAD texels clear of its triangles: its
# first four mipmaps are then the original's own, and a texel read between two is never the edge's.
ALIGN = PAD = 16
# Cut only what leaves under three quarters of the picture: a body picture, used all over, stays.
WORTH = 0.75
# A normal map this wide or tall is halved.
NORMAL_FULL = 4096
PNG = b"\x89PNG\r\n\x1a\n"
MTOON_SLOTS = ("shadeMultiplyTexture", "shadingShiftTexture", "matcapTexture", "rimMultiplyTexture",
               "outlineWidthMultiplyTexture", "uvAnimationMaskTexture")
UV_ANIMATION = ("uvAnimationScrollXSpeedFactor", "uvAnimationScrollYSpeedFactor", "uvAnimationRotationSpeedFactor")
MIPMAPPED = (9984, 9985, 9986, 9987, None)  # sampler min filters an app makes mipmaps for (None: its default)


class Kept(Exception):
    """Why a picture stays as it is."""


def pictures(path):
    """Cut and halve the pictures of the .vrm at `path`, in place. Returns what was done:
    {"cut": [{"picture", "from", "to"}], "halved": [...], "megabytes": [before, after],
    "loaded_megabytes": [before, after]}, or {"left": why} when the file stays as Blender wrote it
    (no Pillow, nothing worth doing, or a file that isn't laid out as expected).

    Never raises: a picture that can't be read must not take the avatar away. The file is written
    once, whole, after every new picture has been read back.
    """
    try:
        from PIL import Image
    except ImportError:
        return {"left": "Pillow isn't installed (see the Setup page): the pictures stay at full size"}
    try:
        path = Path(path)
        gltf, binary = paint.read_glb(path)
        before = path.stat().st_size
        report = _slim(gltf, bytearray(binary), Image)
        if not report["cut"] and not report["halved"]:
            return {"left": "no picture is bigger than the avatar uses"}
        paint.write_glb(path, gltf, report.pop("binary"))
        report["megabytes"] = [round(before / 2**20, 1), round(path.stat().st_size / 2**20, 1)]
        return report
    except Exception as error:  # noqa: BLE001 -- whatever a picture or a layout can do wrong, the avatar stays
        return {"left": f"{type(error).__name__}: {error}"}


def _slim(gltf, binary, Image):
    images, views = gltf.get("images", []), gltf["bufferViews"]
    if len(gltf.get("buffers", [])) != 1 or any(view.get("buffer", 0) != 0 or "extensions" in view for view in views):
        raise Kept("the file's data isn't one plain buffer")
    readers = {}  # picture -> {(material, slot)}
    for m, material in enumerate(gltf.get("materials", [])):
        for slot, info in _slots(material):
            picture = gltf["textures"][info["index"]].get("source")
            if picture is not None:
                readers.setdefault(picture, set()).add((m, slot))
    loaded = lambda data: round(sum(_loaded(gltf, data, picture) for picture in readers), 1)  # noqa: E731
    fresh, report = {}, {"cut": [], "halved": [], "loaded_megabytes": [loaded(binary)]}
    held = [entry.get("bufferView") for entry in images]
    for picture in sorted(readers):
        entry = images[picture]
        if "bufferView" not in entry or held.count(entry["bufferView"]) > 1:
            continue  # (two pictures of the same bytes: what one needs, the other may not bear)
        data = _view(binary, views[entry["bufferView"]])
        try:
            if {slot for _, slot in readers[picture]} == {"normalTexture"}:
                made, size, now = _halved(Image, data)
                report["halved"].append({"picture": entry.get("name"), "from": list(size), "to": list(now)})
            else:
                (x0, y0, x1, y1), accessors = _rectangle(gltf, binary, picture, readers, data)
                whole = Image.open(io.BytesIO(data))
                made = _png_like(Image, data, whole.crop((x0, y0, x1, y1)))
                for accessor in accessors:
                    _move_uvs(gltf, binary, accessor, whole.size, (x0, y0, x1, y1))
                report["cut"].append({"picture": entry.get("name"), "from": list(whole.size), "to": [x1 - x0, y1 - y0]})
        except Kept:
            continue
        fresh[entry["bufferView"]] = made
    packed = bytearray()
    for index, view in enumerate(views):
        start = view.get("byteOffset", 0)
        data = fresh.get(index) or binary[start:start + view["byteLength"]]
        # where it lay, give or take whole words: what reads it was aligned there
        packed.extend(b"\0" * ((start - len(packed)) % 4))
        view["byteOffset"], view["byteLength"] = len(packed), len(data)
        packed.extend(data)
    packed.extend(b"\0" * (-len(packed) % 4))
    gltf["buffers"][0]["byteLength"] = len(packed)
    report["loaded_megabytes"].append(loaded(packed))
    report["binary"] = packed
    return report


def _slots(material):
    """(slot, texture info) for every picture a material reads."""
    pbr = material.get("pbrMetallicRoughness") or {}
    found = [(key, pbr[key]) for key in ("baseColorTexture", "metallicRoughnessTexture") if pbr.get(key)]
    found += [(key, material[key]) for key in ("normalTexture", "occlusionTexture", "emissiveTexture") if material.get(key)]
    mtoon = (material.get("extensions") or {}).get("VRMC_materials_mtoon") or {}
    return found + [(key, mtoon[key]) for key in MTOON_SLOTS if mtoon.get(key)]


def _view(binary, view):
    return bytes(binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]])


def _loaded(gltf, binary, picture):
    """Megabytes a picture takes once an app has loaded it: four bytes a texel, a third more with mipmaps."""
    entry = gltf["images"][picture]
    if "bufferView" not in entry:
        return 0.0
    width, height = _size(_view(binary, gltf["bufferViews"][entry["bufferView"]]))
    samplers = gltf.get("samplers", [])
    filters = {samplers[t["sampler"]].get("minFilter") if "sampler" in t else None
               for t in gltf.get("textures", []) if t.get("source") == picture}
    return width * height * 4 * (4 / 3 if filters & set(MIPMAPPED) else 1) / 2**20


def _size(data):
    """A picture's width and height from its first bytes (a PNG's header, a JPEG's frame)."""
    if data[:8] == PNG:
        return struct.unpack_from(">II", data, 16)
    at = 2
    while at + 9 < len(data) and data[at] == 0xFF:
        kind, length = data[at + 1], struct.unpack_from(">H", data, at + 2)[0]
        if 0xC0 <= kind <= 0xCF and kind not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack_from(">HH", data, at + 5)
            return width, height
        at += 2 + length
    return 0, 0


def _rectangle(gltf, binary, picture, readers, data):
    """((x0, y0, x1, y1), its UV accessors) for a picture worth cutting; Kept, with the reason, otherwise.

    Refused, whatever it would save: a picture anything but plain UV mapping reads (a matcap, a UV
    animation or an expression that moves UVs, a texture transform, a second UV set), one whose
    materials read another picture by the same UVs, UVs that leave the picture (it repeats) or that
    triangles of another material share, and what can't be cut texel for texel (a JPEG, a 16-bit PNG).
    """
    materials, accessors = gltf.get("materials", []), gltf["accessors"]
    mine = {m for m, _ in readers[picture]}
    if any(other != picture and mine & {m for m, _ in users} for other, users in readers.items()):
        raise Kept("its materials read another picture by the same UVs")
    if any(slot == "matcapTexture" for _, slot in readers[picture]):
        raise Kept("a matcap isn't mapped by UVs")
    if any(info.get("texCoord") or info.get("extensions") for m in mine for _, info in _slots(materials[m])):
        raise Kept("a second UV set or a texture transform reads it")
    expressions = ((gltf.get("extensions") or {}).get("VRMC_vrm") or {}).get("expressions") or {}
    moved = {bind.get("material") for group in expressions.values() for expression in (group or {}).values()
             for bind in expression.get("textureTransformBinds") or []}
    if mine & moved or any(((materials[m].get("extensions") or {}).get("VRMC_materials_mtoon") or {}).get(key)
                           for m in mine for key in UV_ANIMATION):
        raise Kept("its UVs are animated")
    if data[:8] != PNG or data[24] != 8:
        raise Kept("not an 8-bit PNG")
    primitives = [p for mesh in gltf.get("meshes", []) for p in mesh.get("primitives", [])]
    using = [p for p in primitives if p.get("material") in mine]
    uvs = {p.get("attributes", {}).get("TEXCOORD_0") for p in using}
    elsewhere = {a for p in primitives if p.get("material") not in mine for a in p.get("attributes", {}).values()}
    if not using or None in uvs or uvs & elsewhere:
        raise Kept("no triangle reads it, or one without UVs of its own")
    if any(key.startswith("TEXCOORD") for p in using for target in p.get("targets", []) for key in target):
        raise Kept("a morph target moves its UVs")
    for a in uvs:
        accessor = accessors[a]
        if (accessor.get("componentType") != 5126 or accessor.get("type") != "VEC2" or "sparse" in accessor
                or "bufferView" not in accessor or not accessor.get("count")
                or gltf["bufferViews"][accessor["bufferView"]].get("byteStride", 8) != 8):
            raise Kept("its UVs aren't plain floats")
    width, height = _size(data)
    us, vs = [], []
    for a in uvs:
        values = _floats(binary, *_span(gltf, a))
        us += (min(values[0::2]), max(values[0::2]))
        vs += (min(values[1::2]), max(values[1::2]))
    if min(us + vs) < 0 or max(us + vs) > 1:
        raise Kept("its UVs leave the picture")
    x0, y0 = (max(math.floor(low * size) - PAD, 0) // ALIGN * ALIGN for low, size in ((min(us), width), (min(vs), height)))
    x1, y1 = (min(-(-(math.ceil(high * size) + PAD) // ALIGN) * ALIGN, size) for high, size in ((max(us), width), (max(vs), height)))
    if (x1 - x0) * (y1 - y0) >= WORTH * width * height:
        raise Kept("its triangles use most of it")
    return (x0, y0, x1, y1), sorted(uvs)


def _span(gltf, accessor):
    """(where an accessor's numbers start in the binary chunk, how many numbers) for tightly packed floats."""
    accessor = gltf["accessors"][accessor]
    view = gltf["bufferViews"][accessor["bufferView"]]
    return view.get("byteOffset", 0) + accessor.get("byteOffset", 0), accessor["count"] * 2


def _floats(binary, start, count):
    values = array.array("f")
    values.frombytes(bytes(binary[start:start + count * 4]))
    if sys.byteorder == "big":
        values.byteswap()
    return values


def _move_uvs(gltf, binary, accessor, size, rectangle):
    """Point a UV accessor at the same texels of the cut picture (changes `binary`)."""
    (width, height), (x0, y0, x1, y1) = size, rectangle
    start, count = _span(gltf, accessor)
    values = _floats(binary, start, count)
    values[0::2] = array.array("f", [(u * width - x0) / (x1 - x0) for u in values[0::2]])
    values[1::2] = array.array("f", [(v * height - y0) / (y1 - y0) for v in values[1::2]])
    entry = gltf["accessors"][accessor]
    if "min" in entry:
        entry["min"], entry["max"] = ([low(values[0::2]), low(values[1::2])] for low in (min, max))
    if sys.byteorder == "big":
        values.byteswap()
    binary[start:start + count * 4] = values.tobytes()


def _chunks(data):
    """[(kind, the whole chunk)] of a PNG."""
    at, chunks = 8, []
    while at < len(data):
        length, kind = struct.unpack_from(">I4s", data, at)
        chunks.append((kind, data[at:at + 12 + length]))
        at += 12 + length
    return chunks


def _png_like(Image, original, picture):
    """`picture` as a PNG that is `original` in all but its size and its texels.

    Blender writes what it knows of a picture into each file (sRGB, gAMA, cHRM, eXIf, oFFs, pHYs
    chunks), and an app may read the colour space from there: only the header and the texels are
    new. Kept when it wouldn't read back as it was written.
    """
    out = io.BytesIO()
    picture.save(out, format="PNG", compress_level=9)
    made, was = _chunks(out.getvalue()), _chunks(original)
    head = next(chunk for kind, chunk in made if kind == b"IHDR")
    if head[16:18] != was[0][1][16:18]:
        raise Kept("it would be written back with another bit depth or colour type")
    chunks, placed = [original[:8], head], False
    for kind, chunk in was[1:]:
        if kind != b"IDAT":
            chunks.append(chunk)
        elif not placed:
            chunks += [texels for name, texels in made if name == b"IDAT"]
            placed = True
    data = b"".join(chunks)
    again, first = Image.open(io.BytesIO(data)), Image.open(io.BytesIO(original))
    if again.mode != picture.mode or again.size != picture.size or again.tobytes() != picture.tobytes() or again.info != first.info:
        raise Kept("its PNG doesn't read back as it was written")
    return data


def _halved(Image, data):
    """(a normal map at half size, its size before, its size now); Kept when it is small already.

    Each texel the mean of four, made a unit vector again: the picture's own first mipmap, as a
    texture tool writes one. Seen small on screen nothing changes, and close up the relief was no
    finer than this shows (model A, beside its 4096 px one in Warudo). A few seconds of plain Python
    for a 4096 px map: the app's Python has Pillow and no NumPy.
    """
    width, height = _size(data)
    if max(width, height) < NORMAL_FULL:
        raise Kept("small already")
    whole = Image.open(io.BytesIO(data))
    if whole.mode not in ("RGB", "RGBA") or (whole.format != "JPEG" and (data[:8] != PNG or data[24] != 8)):
        raise Kept("neither a JPEG nor an 8-bit PNG of three or four channels")
    # channel by channel (halved whole, a picture with an alpha channel is weighted by it), and as
    # fractions: rounded before the vector is made unit, a short mean would lose its direction
    channels = whole.split()
    means = [array.array("f", channel.convert("F").reduce(2).tobytes()) for channel in channels[:3]]
    size = (width // 2, height // 2)
    small = Image.merge("RGB", [Image.frombytes("L", size, bytes(values)) for values in _unit(*means)])
    if whole.mode == "RGBA":
        small.putalpha(channels[3].reduce(2))
    if whole.format == "JPEG":
        out = io.BytesIO()
        small.save(out, format="JPEG", quality=95, subsampling=0)
        made = out.getvalue()
        Image.open(io.BytesIO(made)).load()  # read back, as a cut PNG is
    else:
        made = _png_like(Image, data, small)
    return made, whole.size, small.size


def _unit(reds, greens, blues):
    """Three channels of a normal map (0 to 255, fractions kept) as bytes, each vector unit length."""
    middle, root = 127.5, math.sqrt
    out = bytearray(len(reds)), bytearray(len(reds)), bytearray(len(reds))
    for i, (x, y, z) in enumerate(zip(reds, greens, blues)):
        x -= middle
        y -= middle
        z -= middle
        scale = middle / (root(x * x + y * y + z * z) or 1.0)
        out[0][i], out[1][i], out[2][i] = round(x * scale + middle), round(y * scale + middle), round(z * scale + middle)
    return out
