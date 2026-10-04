"""What each build step may change in the file it is handed, checked from the numbers
scripts/_fingerprint.py writes beside every .blend. Plain Python.

A step that changes something it doesn't own has damaged another step's work: the loose step
welding the mouth patch, a T-pose step relaying an adopted rig's fingers, an import emptying the
eyes' look keys (all 2026-10-03, all found by eye in Warudo). The limits are read off seven real
runs: through steps that don't own them, a key's reach and a finger's angle came out the same to
the last digit on every good one.
"""

# Steps that add, remove or join vertices. Joining twins at one position (loose) takes up to half
# of the vertices a key moves and leaves how far it moves them alone, so there only reach is held.
GEOMETRY = {"build", "hair", "loose", "fingers", "pieces", "eyes", "mouth"}
# Steps that make or reshape shape keys
KEYS = {"face_tracking", "eyes", "mouth"}
# Steps that place or turn finger bones; none of them on an adopted rig, whose hands stay as modelled
FINGERS = {"fingers", "finger_guides", "tpose"}

# A key's furthest move may drift this much of itself through a step that doesn't own keys.
# Good runs: 0 to 0.1%. The welded mouth patch: aa 15.98 -> 11.37 mm (29%), jawOpen 16.51 -> 11.81.
REACH = 0.1
# Degrees a finger bone may turn from its hand. Good runs: 0.0. The fanned fingers: 13 to 22.
TURN = 3.0
# Share of the vertices the face shapes move that spring bones may hold, and the least the Head
# bone should. Model A (T-pose) before the rider and face-weight fixes: 328 of 1970 on Hair8/Hair11, 0.72.
SWINGING = 0.02
ON_HEAD = 0.9


def compare(step, before, after, rigged=False):
    """(damage, warnings): what `step` changed between two fingerprints that it doesn't own, and
    what the file holds that no step should leave. `before` None (the first .blend, or one of a
    build older than the fingerprints): warnings only. `after` None: nothing to judge."""
    if after is None:
        return [], []
    damage = [] if before is None else [*_keys(step, before, after), *_geometry(step, before, after),
                                        *_fingers(step, before, after, rigged), *_bones(before, after)]
    # said once, at the step that brings it: the counts in a line change at every weld after
    was = {} if before is None else _state(before)
    return damage, [line for kind, line in _state(after).items() if kind not in was]


def _keys(step, before, after):
    for mesh, keys in before.get("keys", {}).items():
        now = after.get("keys", {}).get(mesh)
        if now is None:
            if mesh in after.get("meshes", {}):
                yield f"{mesh} lost all its shape keys"
            continue
        for name, was in keys.items():
            if not was["moved"]:
                continue
            if name not in now:
                yield f"shape key {name} is gone"
            elif not now[name]["moved"]:
                yield f"shape key {name} no longer moves anything (it moved {was['moved']} vertices)"
            elif step not in KEYS:
                if abs(now[name]["max_mm"] - was["max_mm"]) > REACH * was["max_mm"]:
                    yield f"shape key {name} reaches {now[name]['max_mm']} mm, was {was['max_mm']}"
                elif step not in GEOMETRY and now[name]["moved"] != was["moved"]:
                    yield f"shape key {name} moves {now[name]['moved']} vertices, was {was['moved']}"


def _geometry(step, before, after):
    if step in GEOMETRY:
        return
    for mesh, was in before.get("meshes", {}).items():
        now = after.get("meshes", {}).get(mesh)
        if now is None:
            yield f"mesh {mesh} is gone"
        elif (now["vertices"], now["polygons"], now["materials"]) != (was["vertices"], was["polygons"], was["materials"]):
            yield (f"{mesh} has {now['vertices']} vertices and {now['polygons']} polygons, "
                   f"was {was['vertices']} and {was['polygons']}")


def _fingers(step, before, after, rigged):
    if step in FINGERS and not rigged:
        return
    turned = {name: (was, after.get("fingers", {}).get(name)) for name, was in before.get("fingers", {}).items()}
    gone = sorted(name for name, (_, now) in turned.items() if now is None)
    if gone:
        yield f"finger bones gone: {', '.join(gone[:4])}"
    far = {name: (was, now) for name, (was, now) in turned.items() if now is not None and abs(now - was) > TURN}
    if far:
        name, (was, now) = max(far.items(), key=lambda item: abs(item[1][1] - item[1][0]))
        yield f"{len(far)} finger bones turned from their hand, {name} most: {was} -> {now} degrees"


def _bones(before, after):
    for what, label in (("bones", "bones"), ("spring_bones", "spring bones")):
        if after.get(what, 0) < before.get(what, 0):
            yield f"{after.get(what, 0)} {label}, was {before[what]}"


def _state(fingerprint):
    lines = {}
    for mesh, face in fingerprint.get("face", {}).items():
        if not face["vertices"]:
            continue
        if face["on_springs"] > SWINGING * face["vertices"]:
            lines[mesh, "swinging"] = f"{face['on_springs']} of the {face['vertices']} vertices the face shapes move swing on spring bones"
        if face["head"] is not None and face["head"] < ON_HEAD:
            lines[mesh, "head"] = f"the face follows the Head bone by {face['head']:.2f} (median): patches on it will slide"
    return lines
