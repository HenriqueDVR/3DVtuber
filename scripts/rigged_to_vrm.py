"""Export a model someone else rigged (Tripo's auto-rig, VRoid bone names) as a VRM 1.0, as it is.

    blender -b --python-exit-code 1 -P scripts/rigged_to_vrm.py -- rigged.fbx out.vrm [--name N] [--author A] [--height 1.6]

For comparing another rig with ours in the apps: the skeleton and weights are left exactly as
they came. Only what a VRM needs is added: the humanoid bone mapping (VRoid's J_Bip names), the
avatar's name and author, MToon materials (so it shades like our exports), and a uniform scale to
`--height` (Tripo exports 1 m tall; apps size tracking for an adult). No expressions, eye
tracking, springs or repairs. Exported the way the pipeline exports: the rest pose as it is, joint
axes normalized. Needs the VRM Add-on, so don't pass --factory-startup.
"""

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import meshdata, orient, vrm_export  # noqa: E402

TORSO = {"Hips": "hips", "Spine": "spine", "Chest": "chest", "UpperChest": "upper_chest", "Neck": "neck",
         "Head": "head"}
LIMBS = {"Shoulder": "shoulder", "UpperArm": "upper_arm", "LowerArm": "lower_arm", "Hand": "hand",
         "UpperLeg": "upper_leg", "LowerLeg": "lower_leg", "Foot": "foot", "ToeBase": "toes"}
FINGERS = {"Thumb": ("metacarpal", "proximal", "distal"), "Index": ("proximal", "intermediate", "distal"),
           "Middle": ("proximal", "intermediate", "distal"), "Ring": ("proximal", "intermediate", "distal"),
           "Little": ("proximal", "intermediate", "distal")}


def vroid_mapping(bones):
    """{VRM humanoid key: bone name} for the VRoid-style names present."""
    mapping = {key: f"J_Bip_C_{name}" for name, key in TORSO.items()}
    for letter, side in (("L", "left"), ("R", "right")):
        mapping.update({f"{side}_{key}": f"J_Bip_{letter}_{name}" for name, key in LIMBS.items()})
        for finger, segments in FINGERS.items():
            for number, segment in enumerate(segments, 1):
                mapping[f"{side}_{finger.lower()}_{segment}"] = f"J_Bip_{letter}_{finger}{number}"
        mapping[f"{side}_eye"] = f"J_Adj_{letter}_FaceEye"
    return {key: name for key, name in mapping.items() if name in bones}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("rigged", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--name", default="Avatar")
    parser.add_argument("--author", default="Unknown")
    parser.add_argument("--height", type=float, default=1.6)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; not overwriting")

    vrm_export.ensure_vrm_addon()
    bpy.ops.wm.read_homefile(use_empty=True)
    if args.rigged.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(args.rigged.resolve()))
    else:
        bpy.ops.import_scene.gltf(filepath=str(args.rigged.resolve()))
    armature = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.find_armature() == armature]

    verts = np.concatenate([meshdata.world_vertices(m) for m in meshes])
    found = orient.detect(verts, np.concatenate([meshdata.triangles(m.data) + sum(len(n.data.vertices) for n in meshes[:i])
                                                 for i, m in enumerate(meshes)]))
    print(f"orientation: {found.describe()} (sure: {found.confident})")
    if not found.is_identity:
        raise SystemExit("the model doesn't stand +Z up facing -Y; turn it first (this script changes nothing but scale)")

    scale = args.height / float(np.ptp(verts[:, 2]))
    bpy.ops.object.select_all(action="DESELECT")
    for obj in [armature, *meshes]:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    armature.scale = [s * scale for s in armature.scale]
    armature.location = [v * scale for v in armature.location]
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    print(f"scaled by {scale:.3f} to {args.height} m")

    extension = armature.data.vrm_addon_extension
    extension.spec_version = "1.0"
    human_bones = extension.vrm1.humanoid.human_bones
    mapping = vroid_mapping(armature.data.bones)
    for key, name in mapping.items():
        getattr(human_bones, key).node.bone_name = name
    missing = [b.name for b in armature.data.bones if b.name not in mapping.values()]
    print(f"mapped {len(mapping)} humanoid bones; not mapped: {missing}")
    meta = extension.vrm1.meta
    meta.vrm_name = args.name
    meta.authors.clear()
    meta.authors.add().value = args.author
    for mesh in meshes:
        vrm_export.use_toon_shading(mesh)
    vrm_export.export(args.out, armature)
    print(f"wrote {args.out}")


main()
