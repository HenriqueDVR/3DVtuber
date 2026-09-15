"""Build a synthetic character shaped like AI image-to-3D output.

Generators such as Tripo return one fused, triangulated mesh with a single
base-color texture: eyes and mouth are painted on and the hair is glued to the
head. This fixture reproduces that and writes a truth file with the joint and
face positions used to build it, so detectors can be checked against known
answers.

Run:
    blender -b --factory-startup -P tests/fixtures/make_synthetic_character.py -- <out_dir>
"""

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "addon"))

from vtuber_maker.pipeline import preview, texture_space  # noqa: E402

TEXTURE_SIZE = 1024
VOXEL_SIZE = 0.006
DECIMATE_RATIO = 0.6
ARM_DROP = math.radians(40)

REGION_COLORS = {
    "skin": (255, 224, 206),
    "hair": (176, 92, 204),
    "top": (64, 76, 150),
    "pants": (44, 44, 56),
    "shoes": (30, 28, 30),
}

HEAD_CENTER = Vector((0.0, 0.0, 1.47))
HEAD_RADIUS = 0.13
EYE_OFFSET_X = 0.045
EYE_Z = 1.475
EYE_RADII = (0.017, 0.02)
MOUTH_Z = 1.392
MOUTH_RADII = (0.012, 0.0035)


def ellipsoid(name, region, center, radii, axis=(0.0, 0.0, 1.0)):
    return {
        "name": name,
        "region": region,
        "center": Vector(center),
        "radii": Vector(radii),
        "axis": Vector(axis).normalized(),
    }


def limb(name, region, start, end, radius):
    start, end = Vector(start), Vector(end)
    half_length = (end - start).length / 2
    return ellipsoid(name, region, (start + end) / 2, (radius, radius, half_length + radius * 0.6), end - start)


def arm_joints(side):
    shoulder = Vector((0.17 * side, 0.0, 1.28))
    direction = Vector((math.cos(ARM_DROP) * side, 0.0, -math.sin(ARM_DROP)))
    elbow = shoulder + direction * 0.26
    wrist = elbow + direction * 0.24
    return shoulder, elbow, wrist, wrist + direction * 0.09


def leg_joints(side):
    x = 0.085 * side
    return Vector((x, 0.0, 0.84)), Vector((x, 0.0, 0.46)), Vector((x, 0.0, 0.08))


def shapes():
    parts = [
        ellipsoid("head", "skin", HEAD_CENTER, (HEAD_RADIUS,) * 3),
        limb("neck", "skin", (0, 0, 1.30), (0, 0, 1.38), 0.045),
        ellipsoid("torso", "top", (0, 0, 1.08), (0.16, 0.10, 0.24)),
        ellipsoid("chest", "top", (0, 0, 1.19), (0.165, 0.105, 0.13)),
        ellipsoid("pelvis", "pants", (0, 0, 0.86), (0.15, 0.10, 0.10)),
        ellipsoid("hair_cap", "hair", (0, 0.03, 1.52), (0.145, 0.14, 0.12)),
        ellipsoid("hair_back", "hair", (0, 0.07, 1.30), (0.15, 0.06, 0.26)),
        ellipsoid("bang_mid", "hair", (0, -0.11, 1.56), (0.03, 0.02, 0.045)),
    ]
    for side in (1, -1):
        tag = "L" if side > 0 else "R"
        shoulder, elbow, wrist, hand_tip = arm_joints(side)
        hip, knee, ankle = leg_joints(side)
        parts += [
            ellipsoid(f"shoulder_{tag}", "top", shoulder, (0.05, 0.05, 0.05)),
            limb(f"upper_arm_{tag}", "skin", shoulder, elbow, 0.042),
            limb(f"forearm_{tag}", "skin", elbow, wrist, 0.036),
            limb(f"hand_{tag}", "skin", wrist, hand_tip, 0.035),
            limb(f"thigh_{tag}", "pants", hip, knee, 0.07),
            limb(f"shin_{tag}", "pants", knee, ankle, 0.05),
            ellipsoid(f"foot_{tag}", "shoes", (0.085 * side, -0.04, 0.035), (0.045, 0.10, 0.035)),
            limb(f"side_lock_{tag}", "hair", (0.115 * side, -0.08, 1.50), (0.13 * side, -0.08, 1.22), 0.028),
            ellipsoid(f"bang_{tag}", "hair", (0.04 * side, -0.105, 1.555), (0.035, 0.02, 0.05)),
        ]
    return parts


def build_mesh(parts):
    objects = []
    for part in parts:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0)
        obj = bpy.context.active_object
        obj.scale = part["radii"]
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(part["axis"])
        obj.location = part["center"]
        objects.append(obj)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.join()

    body = bpy.context.active_object
    body.name = "synthetic_character"
    body.data.remesh_voxel_size = VOXEL_SIZE
    bpy.ops.object.voxel_remesh()
    decimate = body.modifiers.new("decimate", "DECIMATE")
    decimate.ratio = DECIMATE_RATIO
    bpy.ops.object.modifier_apply(modifier=decimate.name)
    triangulate = body.modifiers.new("triangulate", "TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=triangulate.name)
    bpy.ops.object.shade_smooth()
    return body


def unwrap(obj):
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.004)
    bpy.ops.object.mode_set(mode="OBJECT")


def nearest_part(points, parts):
    """Index of the part whose surface each point lies on (smallest signed distance)."""
    best = np.full(len(points), np.inf)
    index = np.zeros(len(points), dtype=np.int32)
    for i, part in enumerate(parts):
        rotation = np.array(Vector((0.0, 0.0, 1.0)).rotation_difference(part["axis"]).to_matrix())
        local = (points - np.array(part["center"])) @ rotation
        radii = np.array(part["radii"])
        distance = (np.linalg.norm(local / radii, axis=1) - 1.0) * radii.min()
        closer = distance < best
        best[closer] = distance[closer]
        index[closer] = i
    return index


def in_ellipse(points, center_x, center_z, radius_x, radius_z):
    return ((points[:, 0] - center_x) / radius_x) ** 2 + ((points[:, 2] - center_z) / radius_z) ** 2 <= 1.0


def paint_face(colors, points, face):
    for side in (1, -1):
        x = EYE_OFFSET_X * side
        rx, rz = EYE_RADII
        lash = face & in_ellipse(points, x, EYE_Z, rx * 1.15, rz * 1.12) & (points[:, 2] > EYE_Z)
        colors[lash] = (34, 22, 36)
        colors[face & in_ellipse(points, x, EYE_Z, rx, rz)] = (250, 250, 252)
        colors[face & in_ellipse(points, x, EYE_Z - 0.003, 0.012, 0.016)] = (86, 44, 128)
        colors[face & in_ellipse(points, x, EYE_Z - 0.003, 0.005, 0.008)] = (20, 12, 28)
        colors[face & in_ellipse(points, x - 0.004 * side, EYE_Z + 0.004, 0.003, 0.003)] = (255, 255, 255)
    colors[face & in_ellipse(points, 0.0, MOUTH_Z, *MOUTH_RADII)] = (150, 44, 56)


def paint_texture(obj, parts, size):
    positions, covered = texture_space.bake_position_map(obj, size)
    points = positions[covered]
    nearest = nearest_part(points, parts)
    palette = np.array([REGION_COLORS[p["region"]] for p in parts], dtype=np.float32)
    colors = palette[nearest]
    head = [p["name"] for p in parts].index("head")
    paint_face(colors, points, (nearest == head) & (points[:, 1] < -0.06))

    rgba = np.ones((size, size, 4), dtype=np.float32)
    rgba[..., :3] = np.array(REGION_COLORS["skin"], dtype=np.float32) / 255
    rgba[covered, :3] = colors / 255
    image = bpy.data.images.new("synthetic_character_basecolor", size, size, alpha=False)
    image.pixels.foreach_set(rgba.ravel())
    return image


def assign_material(obj, image):
    material = bpy.data.materials.new("synthetic_character")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 1.0
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    material.node_tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.clear()
    obj.data.materials.append(material)


def head_surface_point(x, z):
    dx, dz = x - HEAD_CENTER.x, z - HEAD_CENTER.z
    return [x, HEAD_CENTER.y - math.sqrt(HEAD_RADIUS**2 - dx**2 - dz**2), z]


def truth():
    joints = {"hips": [0.0, 0.0, 0.86], "neck": [0.0, 0.0, 1.30], "head": [0.0, 0.0, 1.38]}
    for side, tag in ((1, "left"), (-1, "right")):
        shoulder, elbow, wrist, _ = arm_joints(side)
        hip, knee, ankle = leg_joints(side)
        joints[f"{tag}_upper_arm"] = list(shoulder)
        joints[f"{tag}_lower_arm"] = list(elbow)
        joints[f"{tag}_hand"] = list(wrist)
        joints[f"{tag}_upper_leg"] = list(hip)
        joints[f"{tag}_lower_leg"] = list(knee)
        joints[f"{tag}_foot"] = list(ankle)
    return {
        "height": 1.64,
        "joints": joints,
        "face": {
            "left_eye": head_surface_point(EYE_OFFSET_X, EYE_Z),
            "right_eye": head_surface_point(-EYE_OFFSET_X, EYE_Z),
            "eye_size": [EYE_RADII[0] * 2, EYE_RADII[1] * 2],
            "mouth": head_surface_point(0.0, MOUTH_Z),
            "mouth_width": MOUTH_RADII[0] * 2,
        },
        # Surface points near the ends of hanging hair (from the hair_back and side_lock shapes).
        "hair": {
            "back_tip": [0.0, 0.119, 1.15],
            "left_lock_tip": [0.128, -0.10, 1.26],
            "right_lock_tip": [-0.128, -0.10, 1.26],
        },
        "colors": REGION_COLORS,
    }


def main():
    out_dir = Path(sys.argv[sys.argv.index("--") + 1]) if "--" in sys.argv else REPO / "tests" / "fixtures" / "out"
    # Blender resolves relative output paths against its own working directory, not ours.
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    parts = shapes()
    body = build_mesh(parts)
    unwrap(body)
    image = paint_texture(body, parts, TEXTURE_SIZE)
    image.filepath_raw = str(out_dir / "synthetic_character_basecolor.png")
    image.file_format = "PNG"
    image.save()
    assign_material(body, image)

    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(out_dir / "synthetic_character.glb"), export_format="GLB", use_selection=True
    )
    (out_dir / "synthetic_character.truth.json").write_text(json.dumps(truth(), indent=2))

    for view in ("front", "side"):
        preview.render(str(out_dir / f"synthetic_character_{view}.png"), view)
    face_center = (HEAD_CENTER.x, HEAD_CENTER.y, HEAD_CENTER.z)
    preview.render(str(out_dir / "synthetic_character_face.png"), "front", focus=(face_center, 0.34))
    print(f"FIXTURE faces={len(body.data.polygons)} verts={len(body.data.vertices)}")


main()
