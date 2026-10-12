"""Give a rigged avatar's clothing pieces the roles its character manifest names.

    blender -b in_apose.blend --python-exit-code 1 -P scripts/apply_pieces.py -- out.blend out.vrm --manifest character.json

Acts on the "pieces" roles (fitted_skirt, shoulder_cloth, follow_bone, detail; see
pipeline/manifest.py) on the modelled A-pose, before the T-pose bake. The pieces are found on
the manifest's `reference` model (else its `source`), which must be the unmodified model the
rig was built from. Hands (other than vertices of the named pieces), shape keys and UVs are
checked unchanged; nothing is deleted. A piece marked `found` was given its role by the movement
check, not by the user (the app's option auto_pieces): with any such piece the rig is measured
before and after (loose.worst_stretch), and a tear refuses the whole step, as the tidy-up's does.
Refuses an existing output and a posed rig. Writes a JSON report next to the output.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from vtuber_maker.pipeline import loose, manifest, roles, vrm_export  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_blend", type=Path)
    parser.add_argument("out_vrm", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.out_blend.exists() or args.out_vrm.exists():
        sys.exit("output exists: choose a new name; originals are never overwritten")
    data = manifest.load(args.manifest)
    pieces = manifest.pieces_for(data, "pieces")
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
    if any(not np.allclose(np.array(b.matrix_basis), np.eye(4), atol=1e-6) for b in arm.pose.bones):
        sys.exit("clear the pose first: pieces are found on the modelled rest pose")
    # anchors are found too, whatever their own role
    wanted = {p["name"] for p in pieces} | {p["anchor"] for p in pieces if p.get("anchor")}
    named = [p for p in data["pieces"] if p["name"] in wanted]
    source = manifest.path_of(data, "reference") or manifest.path_of(data, "source")
    with roles.reference(source, height=data["options"]["height"]) as ref:
        masks = roles.find(mesh, named, ref)
    clothing = np.any([masks[p["name"]] for p in pieces], axis=0)
    hands = roles.hands_snapshot(mesh, exclude=clothing)
    found = [p["name"] for p in pieces if p.get("found")]
    before = loose.worst_stretch([mesh], arm) if found else {}
    report, origin = roles.apply(mesh, arm, pieces, masks)
    torn = loose.tore(before, loose.worst_stretch([mesh], arm)) if found else []
    if torn:
        sys.exit("the roles found for pieces by themselves (" + ", ".join(found) + ") tore the rig: "
                 + "; ".join(f"{pose} {was * 100:.1f} -> {now * 100:.1f} cm" for pose, was, now in torn) + ": nothing saved")
    if roles.hands_snapshot(mesh, origin, exclude=clothing) != hands:
        sys.exit("the piece roles changed the hands' geometry, keys, weights or UVs: nothing saved")
    report = {"source": bpy.data.filepath, "manifest": str(args.manifest.resolve()),
              "pieces": {p["name"]: {"role": p["role"], "vertices": int(masks[p["name"]].sum())} for p in named},
              "changed": report, "hands_unchanged": True, "found": found}
    print("VTUBER_MAKER_PIECES " + json.dumps(report))
    args.out_blend.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    vrm_export.ensure_vrm_addon()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_blend.resolve()), copy=True)
    vrm_export.export(args.out_vrm.resolve(), arm)
    print("wrote", args.out_blend.name)


if __name__ == "__main__":
    main()
