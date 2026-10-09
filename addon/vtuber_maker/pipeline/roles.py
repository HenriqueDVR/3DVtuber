"""Manifest pieces found on the mesh, and given the roles the user chose for them.

See manifest.py for the roles. The repairs themselves live in loose.py and take masks; this
module only turns "the piece at this point" into those masks and calls them, so a character's
clothing is fixed from its manifest rather than from code written for it.
"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import loose, meshdata

# A reference must sit on the rigged mesh this closely (95th percentile, metres): a baked
# T-pose or another model is refused rather than silently mismatched.
MATCH = 1e-4
# A region picked by box or brush misses vertices hidden behind its own surface; left out, they
# keep the piece's weights and tear from it (226 seam edges, 17 cm on a head turn on model C's
# twin-tail). Gaps in the region smaller than this share of it are its own, and are filled;
# vertices with at least half their neighbours in it join it, over REGION_PASSES rounds.
REGION_HOLE = 0.25
REGION_PASSES = 3


@contextmanager
def reference(path, height=1.6):
    """The source model imported and normalized alongside the rig, removed again afterwards."""
    import bpy
    from . import normalize
    old = set(bpy.data.objects)
    normalize.import_model(Path(path))
    try:
        yield normalize.prepare(list(set(bpy.data.objects) - old), height=height)
    finally:
        for obj in list(set(bpy.data.objects) - old):
            bpy.data.objects.remove(obj, do_unlink=True)


def find(mesh_obj, pieces, reference=None):
    """{name: vertex mask} of `mesh_obj` for each manifest piece.

    Without `reference` the pieces are found on `mesh_obj` itself, which must still be the
    normalized source (make_vrm does this before anything changes the topology). With the
    source model as `reference` (an object already put through normalize.prepare) they are
    found there and carried over by position, for a rigged A-pose whose sleeve cuts and mouth
    have added vertices since. Refuses a piece whose vertex count differs from the manifest's.
    """
    labels, mapped, distance, tree = source_labels(mesh_obj, reference)
    # a piece's "region" (source vertex indices) is part of it, for a part fused into a bigger piece
    # (model C's right twin-tail is part of her body): each rig vertex is the source vertex under it
    nearest = None
    if any(piece.get("region") for piece in pieces):
        nearest = (np.arange(len(mapped)) if reference is None
                   else np.array([tree.find(p)[1] for p in meshdata.world_vertices(mesh_obj)]))
    masks = {}
    for piece in pieces:
        _, i, gap = tree.find(piece["point"])
        label = labels[i]
        count = int(np.count_nonzero(labels == label))
        if gap > 1e-3:
            raise ValueError(f"piece {piece['name']}: nothing within 1 mm of its point")
        if count != piece["count"]:
            raise ValueError(f"piece {piece['name']}: the piece there has {count} vertices, the manifest says "
                             f"{piece['count']} (a different or re-exported model?)")
        masks[piece["name"]] = mapped == label
        if piece.get("region"):
            whole = masks[piece["name"]]
            masks[piece["name"]] = close_region(mesh_obj, whole, whole & np.isin(nearest, np.asarray(piece["region"], dtype=int)))
        if not masks[piece["name"]].any() or distance[masks[piece["name"]]].max() > .01:
            raise ValueError(f"piece {piece['name']}: not found on the rig")
    return masks


def close_region(mesh_obj, piece, region):
    """`region` (a mask inside the mask `piece`) with its holes filled and ragged edge smoothed.

    Walks the surface of `piece` only, with vertices sharing a position as one, so nothing
    outside the piece is taken. A gap is a connected part of the piece outside the region that is
    smaller than REGION_HOLE of the region: the rest of the piece is far bigger than a gap.
    """
    weld, a, b = meshdata.welded_edges(mesh_obj, meshdata.world_vertices(mesh_obj))
    inside = piece[a] & piece[b] & (a != b)
    a, b = a[inside], b[inside]
    region = np.zeros(len(weld), bool) | region
    region[weld[region]] = True
    region = region[weld]
    outside = piece & ~region
    labels = np.arange(len(weld))
    ea, eb = a[outside[a] & outside[b]], b[outside[a] & outside[b]]
    while True:
        updated = labels.copy()
        np.minimum.at(updated, ea, labels[eb])
        np.minimum.at(updated, eb, labels[ea])
        updated = updated[updated]
        if np.array_equal(updated, labels):
            break
        labels = updated
    ids, sizes = np.unique(labels[outside & (weld == np.arange(len(weld)))], return_counts=True)
    gaps = ids[sizes < REGION_HOLE * max(int((region & (weld == np.arange(len(weld)))).sum()), 1)]
    region |= outside & np.isin(labels, gaps)
    degree = np.bincount(np.concatenate([a, b]), minlength=len(weld))
    for _ in range(REGION_PASSES):
        near = np.bincount(np.concatenate([a[region[b]], b[region[a]]]), minlength=len(weld))
        grown = region | (piece & (degree > 0) & (2 * near >= degree))
        if np.array_equal(grown, region):
            break
        region = grown
    return region[weld]


def source_labels(mesh_obj, reference=None):
    """(source labels, label of each rig vertex, its distance to the source, KD tree of the source).

    Labels are `meshdata.pieces` of the source, so they match `scripts/list_pieces.py`'s. Without
    `reference` the source is `mesh_obj` itself. Refuses a rig that doesn't sit on the source.
    """
    from mathutils.kdtree import KDTree
    source = reference if reference is not None else mesh_obj
    original = meshdata.world_vertices(source)
    labels = meshdata.pieces(source, original)
    tree = KDTree(len(original))
    for i, point in enumerate(original):
        tree.insert(point, i)
    tree.balance()
    if reference is None:
        return labels, labels, np.zeros(len(original)), tree
    rig = meshdata.world_vertices(mesh_obj)
    matches = [tree.find(p) for p in rig]
    gap = np.array([m[2] for m in matches])
    if np.percentile(gap, 95) > MATCH:
        raise ValueError("the rig doesn't sit on its source model: a baked T-pose, or another model")
    mapped = labels[[m[1] for m in matches]]
    # Vertices the sleeve cut added lie on the source's surface but can be over a centimetre
    # from its nearest vertex (model A's skirt: 1.4 cm), and that vertex can be another piece's:
    # they take the piece of the face they lie on.
    from mathutils.bvhtree import BVHTree
    polygons = [tuple(p.vertices) for p in source.data.polygons]
    surface = BVHTree.FromPolygons([tuple(v) for v in original], polygons)
    distance = np.zeros(len(rig))
    for i in np.flatnonzero(gap > 1e-5):
        _, _, face, distance[i] = surface.find_nearest(rig[i])
        mapped[i] = labels[polygons[face][0]]
    return labels, mapped, distance, tree


def of_role(pieces, masks, role, count):
    """Union of the masks of every piece with `role`, over `count` vertices."""
    result = np.zeros(count, dtype=bool)
    for piece in pieces:
        if piece["role"] == role:
            result |= pad(masks[piece["name"]], count)
    return result


def pad(mask, count):
    """A mask extended with False over vertices added since (the mouth interior, cut lids)."""
    mask = np.asarray(mask, dtype=bool)
    return np.concatenate([mask, np.zeros(count - len(mask), dtype=bool)]) if len(mask) < count else mask[:count]


def carry(masks, before, after, verts=None, was=None):
    """Masks moved onto a new vertex list, given each vertex's part label before and after.

    rig.skin cuts sleeves free, which adds vertices but carries every part label with them;
    a manifest piece is a whole part, so it is the same labels afterwards. A region (part of a
    piece) isn't: the vertices the cut added take the mask of the old vertex where they sit, which
    needs `verts`, the mesh's vertices after the cut. With `was`, the vertices before it, every
    vertex is found by its place: the cut renumbers them (a tester's pirate: two ears marked on his
    head came out as 199 and 121 vertices of his sleeves and coat); without it the old ones are
    taken to keep their indices.
    """
    carried = {}
    for name, mask in masks.items():
        ids = np.unique(before[mask])
        if np.array_equal(np.isin(before, ids), mask):
            carried[name] = np.isin(after, ids)
            continue
        if verts is None:
            raise ValueError(f"piece {name} isn't one whole part of the mesh")
        from mathutils.kdtree import KDTree
        old = len(before)
        tree = KDTree(old)
        for i, p in enumerate(verts[:old] if was is None else was):
            tree.insert(p, i)
        tree.balance()
        if was is not None:
            carried[name] = np.array([mask[tree.find(p)[1]] for p in verts], dtype=bool)
            continue
        grown = np.concatenate([mask, np.zeros(len(after) - old, dtype=bool)])
        for i in range(old, len(after)):
            grown[i] = mask[tree.find(verts[i])[1]]
        carried[name] = grown
    return carried


def hair_mask(hair, pieces, masks):
    """Detected hair with the manifest's say added: its hair pieces in, any other listed piece out.

    Removals first: a hair region inside a piece listed as body (model C's right twin-tail) would
    otherwise be taken out again by that piece, depending on the order they are listed in.
    """
    hair = np.asarray(hair, dtype=bool).copy()
    for piece in pieces:
        if piece["role"] != "hair":
            hair &= ~pad(masks[piece["name"]], len(hair))
    for piece in pieces:
        if piece["role"] == "hair":
            hair |= pad(masks[piece["name"]], len(hair))
    return hair


def apply(mesh_obj, armature_obj, pieces, masks):
    """Give the "pieces"-stage roles to a rigged, modelled A-pose; returns (report, origin).

    Weights first, then the skirt cut, which adds vertices: `origin` maps each vertex of the
    result to the one it came from. Hands, shape keys and UVs are never touched (callers can
    check with `hands_snapshot`); nothing is deleted.
    """
    count = len(mesh_obj.data.vertices)
    report = {}
    skirt = of_role(pieces, masks, "fitted_skirt", count)
    if skirt.any():
        report["fitted_skirt"] = loose.restore_skirt_weights(mesh_obj, armature_obj, skirt)
    shoulder = of_role(pieces, masks, "shoulder_cloth", count)
    if shoulder.any():
        report["shoulder_cloth"] = loose.follow_shoulder_cloth(mesh_obj, armature_obj, shoulder)
    for piece in pieces:
        members = np.flatnonzero(pad(masks[piece["name"]], count))
        if piece["role"] == "follow_bone":
            if piece["bone"] not in armature_obj.data.bones:
                raise ValueError(f"piece {piece['name']}: no bone {piece['bone']} in the rig")
            report[piece["name"]] = loose.follow_bone(mesh_obj, armature_obj, members, piece["bone"])
        elif piece["role"] == "detail":
            anchors = np.flatnonzero(pad(masks[piece["anchor"]], count))
            report[piece["name"]] = loose.attach_clothing_detail(mesh_obj, armature_obj, members, anchors)
    origin = np.arange(count)
    if skirt.any():
        origin = loose.separate_skirt_contact(mesh_obj, armature_obj, skirt)
        report["skirt_vertices_added"] = len(origin) - count
    return report, origin


def hands_snapshot(mesh_obj, origin=None, exclude=None):
    """Exact hand geometry/keys/weights/face-corner UVs, independent of clothing reindexing.

    A hand vertex is one mostly on hand or finger bones. `exclude` (over the vertices before any
    cut) leaves out named clothing: 108 of model A's skirt vertices were skinned to the hand, which
    is the defect a fitted_skirt repair fixes, not hand geometry it must keep.
    """
    mesh = mesh_obj.data
    if origin is None:
        origin = np.arange(len(mesh.vertices))
    skip = np.zeros(len(origin), dtype=bool) if exclude is None else np.asarray(exclude, dtype=bool)[origin]
    groups = mesh_obj.vertex_groups
    hand_groups = {g.index for g in groups if g.name.endswith("Hand") or
                   any(finger in g.name for finger in ("Thumb", "Index", "Middle", "Ring", "Little"))}
    indices = [v.index for v in mesh.vertices
               if not skip[v.index] and sum(g.weight for g in v.groups if g.group in hand_groups) > .5]
    selected = set(indices)
    keys = mesh.shape_keys.key_blocks if mesh.shape_keys else []
    vertices = {}
    for i in indices:
        label = int(origin[i])
        assert label not in vertices, "hand vertex duplicated by clothing repair"
        vertices[label] = (tuple(mesh.vertices[i].co),
                           tuple((key.name, tuple(key.data[i].co)) for key in keys),
                           tuple(sorted((groups[g.group].name, g.weight) for g in mesh.vertices[i].groups)))
    faces = []
    for face in mesh.polygons:
        if not any(i in selected for i in face.vertices):
            continue
        corners = tuple((int(origin[mesh.loops[i].vertex_index]),
                         tuple(tuple(layer.data[i].uv) for layer in mesh.uv_layers)) for i in face.loop_indices)
        faces.append((face.material_index, min(corners[k:] + corners[:k] for k in range(len(corners)))))
    return vertices, sorted(faces)
