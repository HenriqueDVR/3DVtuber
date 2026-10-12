"""Take a model that came whole in with its own rig: the first step of a build that keeps one.

    blender -b --python-exit-code 1 -P scripts/import_rigged.py -- in.fbx out.vrm [--blend out.blend]
        [--name N] [--author A] [--height 1.6] [--hair-head-only]

For an MMD model passed round as FBX (pipeline/kept.py): its skeleton and weights stay, its hair,
skirt and ribbon bone chains become VRM springs, its face shapes are bound to the VRM expressions
by name, its eye bones look, its materials go toon. The humanoid bones take the pipeline's names.
The .blend is what the later steps (T-pose, the checks, the export) read. Never overwrites an
existing file.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import kept, meshdata, vrm_export  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Take a rigged model in with its own rig.")
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--blend", type=Path, help="the scene the later steps read")
    parser.add_argument("--name", default="Avatar")
    parser.add_argument("--author", default="Unknown")
    parser.add_argument("--height", type=float, default=1.6, help="only for a model saved in other units")
    parser.add_argument("--hair-head-only", action="store_true", help="hair springs collide with the head alone")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def main():
    args = parse_args()
    for path in (args.output, args.blend):
        if path is not None and path.exists():
            sys.exit(f"{path} exists: choose a new name; files are never overwritten")
    bpy.ops.wm.read_homefile(use_empty=True)
    vrm_export.ensure_vrm_addon()
    try:
        armature, mesh = kept.load(args.model.resolve(), height=args.height)
        report = kept.name_humanoid(armature, mesh, args.name, args.author, path=args.model.name)
    except kept.NotKept as error:
        sys.exit(f"MODEL NOT TAKEN IN {error}")
    chains = kept.spring_chains(armature, mesh)
    kept.bind_springs(armature, mesh, chains, hair_hits_body=not args.hair_head_only)
    bound = kept.bind_expressions(armature, mesh)
    tracked = kept.tracking_shapes(mesh)
    eyes = kept.bind_eyes(armature)
    vrm_export.use_toon_shading(mesh)
    cut = kept.see_through(mesh)
    verts = meshdata.world_vertices(mesh)
    keys = mesh.data.shape_keys.key_blocks if mesh.data.shape_keys else []
    report.update({"height": round(float(verts[:, 2].max() - verts[:, 2].min()), 3), "kept_rig": True,
                   "vertices": len(verts), "bones": len(armature.data.bones),
                   "hair_chains": len(chains["hair"]), "cloth_chains": len(chains["cloth"]), "stiff_chains": len(chains["stiff"]),
                   "expressions": bound, "tracking_shapes": tracked, "shape_keys": max(len(keys) - 1, 0), "eye_bones": eyes, "see_through": cut,
                   "look_at": armature.data.vrm_addon_extension.vrm1.look_at.type,
                   "materials": len({s.material.name for s in mesh.material_slots if s.material})})
    for kind, found in chains.items():
        print(f"{kind}: {len(found)} chains" + (": " + ", ".join(f"{c[0]} ({len(c)})" for c in found[:40]).encode("ascii", "backslashreplace").decode() if found else ""))
    if args.blend:
        # the pictures lie beside the model: packed, the .blend holds them wherever it goes
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(args.blend.resolve()))
    vrm_export.export(args.output.resolve(), armature)
    print("VTUBER_MAKER_REPORT " + json.dumps(report))


main()
