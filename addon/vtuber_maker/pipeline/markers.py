"""Draggable markers for the joints, eyes and mouth a rig is built from.

Detection can misplace a point on an unusual model, or not find it at all. Each
point gets a marker on the model that can be dragged into place before building.
Eye and mouth markers are ellipsoids as big as the painted eye and mouth, so
scaling one resizes what blinking and the mouth shapes move.
"""

import bpy
import numpy as np
from mathutils import Matrix, Vector

COLLECTION = "VTuber Markers"
POINT_PROPERTY = "vtuber_maker_point"
FACE_PROPERTY = "vtuber_maker_face"

# Point key -> the marker's name, shown next to it. A joint's marker sits where its bone starts.
LABELS = {
    "hips": "Hips",
    "neck": "Neck",
    "head": "Head",
    "left_upper_arm": "Left shoulder",
    "left_lower_arm": "Left elbow",
    "left_hand": "Left wrist",
    "right_upper_arm": "Right shoulder",
    "right_lower_arm": "Right elbow",
    "right_hand": "Right wrist",
    "left_upper_leg": "Left hip",
    "left_lower_leg": "Left knee",
    "left_foot": "Left ankle",
    "right_upper_leg": "Right hip",
    "right_lower_leg": "Right knee",
    "right_foot": "Right ankle",
    "left_eye": "Left eye",
    "right_eye": "Right eye",
    "mouth": "Mouth",
}
FACE_POINTS = ("left_eye", "right_eye", "mouth")
JOINTS = tuple(key for key in LABELS if key not in FACE_POINTS)
# The body's centre line runs through these, for mirroring one side onto the other.
CENTER_JOINTS = ("hips", "neck", "head")

JOINT_MARKER_RADIUS = 0.02
# Only a mouth marker's width matters, so it is drawn flattened.
MOUTH_MARKER_FLATNESS = 0.25

NO_MARKERS = "There are no markers yet; use Find Body and Face first"


class MarkersMissing(ValueError):
    """There are no markers to build from, or some were deleted."""


def marker_name(key):
    return LABELS[key]


def collection():
    """The markers' collection, or None when there are no markers."""
    return bpy.data.collections.get(COLLECTION)


def owner():
    """The mesh the markers were placed on, or None."""
    found = collection()
    if found is None:
        return None
    return next((marker.parent for marker in found.objects if marker.parent is not None), None)


def place(mesh_obj, joints, features):
    """Replace any markers with new ones at these joints and face features.

    Markers are parented to the mesh, so they move with it.
    """
    remove()
    bpy.context.view_layer.update()
    found = bpy.data.collections.new(COLLECTION)
    bpy.context.scene.collection.children.link(found)
    found[FACE_PROPERTY] = {
        "skin_color": [float(channel) for channel in features["skin_color"]],
        "hair_color": [float(channel) for channel in features["hair_color"]],
        "mouth_estimated": bool(features["mouth_estimated"]),
    }
    for key in JOINTS:
        marker = _add_marker(found, mesh_obj, key, joints[key], np.full(3, JOINT_MARKER_RADIUS))
        marker.lock_scale = (True, True, True)
    width, height = (float(size) for size in features["eye_size"])
    for key in ("left_eye", "right_eye"):
        _add_marker(found, mesh_obj, key, features[key], np.array([width, min(width, height), height]) / 2)
    half_width = float(features["mouth_width"]) / 2
    flat = MOUTH_MARKER_FLATNESS * half_width
    _add_marker(found, mesh_obj, "mouth", features["mouth"], np.array([half_width, flat, flat]))
    return found


def read():
    """Where the joints and face features are now, in world space.

    Returns (joints, features) with the keys landmarks.detect_body and
    face.detect_features give them.
    """
    found, by_key = _markers()

    def position(key):
        return np.array(by_key[key].matrix_world.translation)

    def size(key):
        return 2 * np.abs(np.array(by_key[key].matrix_world.to_scale()))

    stored = found[FACE_PROPERTY]
    joints = {key: position(key) for key in JOINTS}
    features = {
        "left_eye": position("left_eye"),
        "right_eye": position("right_eye"),
        "eye_size": (size("left_eye")[[0, 2]] + size("right_eye")[[0, 2]]) / 2,
        "mouth": position("mouth"),
        "mouth_width": float(size("mouth")[0]),
        "mouth_estimated": bool(stored["mouth_estimated"]),
        "skin_color": np.array(list(stored["skin_color"])),
        "hair_color": np.array(list(stored["hair_color"])),
    }
    return joints, features


def mirror(source):
    """Copy the markers of one side ("left" or "right") onto the other, mirrored across the body's centre line."""
    _, by_key = _markers()
    target = {"left": "right", "right": "left"}[source]
    mesh_obj = owner()
    to_world = mesh_obj.matrix_world.copy() if mesh_obj is not None else Matrix.Identity(4)
    to_mesh = to_world.inverted()
    center_x = float(np.mean([(to_mesh @ by_key[key].matrix_world.translation).x for key in CENTER_JOINTS]))
    across = (
        Matrix.Translation((center_x, 0.0, 0.0))
        @ Matrix.Scale(-1.0, 4, (1.0, 0.0, 0.0))
        @ Matrix.Translation((-center_x, 0.0, 0.0))
    )
    flip = to_world @ across @ to_mesh
    for key in LABELS:
        if key.startswith(f"{source}_"):
            marker, other = by_key[key], by_key[f"{target}_{key[len(source) + 1 :]}"]
            location = flip @ marker.matrix_world.translation
            other.matrix_world = Matrix.LocRotScale(location, None, marker.matrix_world.to_scale())


def remove():
    """Delete the markers, if there are any."""
    found = collection()
    if found is None:
        return
    for marker in list(found.objects):
        bpy.data.objects.remove(marker)
    bpy.data.collections.remove(found)


def _markers():
    found = collection()
    if found is None:
        raise MarkersMissing(NO_MARKERS)
    by_key = {marker.get(POINT_PROPERTY): marker for marker in found.objects}
    missing = [label for key, label in LABELS.items() if key not in by_key]
    if missing:
        raise MarkersMissing(f"Missing markers: {', '.join(missing)}. Use Find Body and Face to place them again")
    bpy.context.view_layer.update()
    return found, by_key


def _add_marker(found, mesh_obj, key, location, radii):
    marker = bpy.data.objects.new(LABELS[key], None)
    marker[POINT_PROPERTY] = key
    marker.empty_display_type = "SPHERE"
    marker.empty_display_size = 1.0
    marker.show_name = True
    marker.show_in_front = True
    marker.lock_rotation = (True, True, True)
    found.objects.link(marker)
    marker.parent = mesh_obj
    marker.matrix_parent_inverse = mesh_obj.matrix_world.inverted()
    marker.matrix_basis = Matrix.LocRotScale(
        Vector([float(c) for c in location]), None, Vector([float(r) for r in radii])
    )
    return marker
