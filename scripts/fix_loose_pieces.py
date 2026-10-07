"""Tidy the weights and the seams before the model is posed, then export VRM.

    blender -b in.blend -P scripts/fix_loose_pieces.py -- out.blend out.vrm [--keep-seams]

Three things, all of which only show once something moves:

* Surfaces that only touch are joined (vertices in the same spot merged). An image-to-3D
  character's sleeve and torso are usually two shells resting against each other, and raising
  the arm pulls them apart into a hole in the armpit. Joined, the skin stretches instead.
  `--keep-seams` skips this. UVs live on face corners, so the texture is unaffected.
* Stray patches of surface are handed back to the bone their neighbours use. A few vertices of
  Model A's skirt were driven by a forearm, and a 3 mm edge drew out to 41 cm in the T-pose.
* Small loose pieces (buckles, straps, panels) are put on one bone each, so a piece can't be
  torn between two limbs (see pipeline/loose.py). Big pieces are left alone.

The rig is measured in the poses apps use before and after. If the tidy-up tore it (an edge
growing by more than loose.TEARS: joining glued model A's cut sleeves back onto its skirt, a 77 cm
sheet), it is done again without joining, and failing that left out; the report says which.

Run before `add_fingers.py` and `to_tpose.py`. The original file is left alone.
Needs the VRM Add-on for Blender.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import loose, rig, vrm_export  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_blend", type=Path)
    parser.add_argument("out_vrm", type=Path)
    parser.add_argument("--keep-seams", action="store_true",
                        help="leave touching surfaces unjoined (armpits will open up when the arms rise)")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:])


def skinned():
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers)]
    if not meshes:
        sys.exit("no skinned mesh in the file")
    return armature, meshes


def main():
    args = parse_args()
    source = bpy.data.filepath
    armature, meshes = skinned()
    # measured before and after: the tidy-up must not tear what it tidies (see loose.TEARS)
    before = loose.worst_stretch(meshes, armature)
    tidy(armature, meshes, join=not args.keep_seams)
    after = loose.worst_stretch(meshes, armature)
    undone = []
    for join in ([False] if not args.keep_seams else []) + [None]:
        torn = loose.tore(before, after)
        if not torn:
            break
        what = ", ".join(f"{pose} {b * 100:.1f} -> {a * 100:.1f} cm" for pose, b, a in torn)
        undone.append(f"the tidy-up tore the rig ({what}): "
                      + ("done again without joining touching surfaces" if join is False else "left as it was"))
        print("UNDONE", undone[-1])
        bpy.ops.wm.open_mainfile(filepath=source)
        armature, meshes = skinned()
        if join is None:
            after = before
            break
        tidy(armature, meshes, join=join)
        after = loose.worst_stretch(meshes, armature)
    print("VTUBER_MAKER_REPORT " + json.dumps({"stretch_cm": {pose: round(v * 100, 1) for pose, v in after.items()},
                                               "undone": undone}))

    vrm_export.ensure_vrm_addon()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_blend.resolve()), copy=True)
    vrm_export.export(args.out_vrm.resolve(), armature)
    print("wrote", args.out_blend.name, "and", args.out_vrm.name)


def tidy(armature, meshes, join):
    for mesh in meshes:
        # first: the join below only joins copies whose leading bones agree
        shared = loose.share_strand_seams(mesh)
        if shared:
            print(f"{shared} copies of vertices along texture seams in hair and cloth given one set of weights")
    if join:
        for mesh in meshes:
            # a crack in the generated surface: its two rims under a millimetre apart, so not "touching"
            # for the join below, each with its own weights. Model D's under the right elbow (6 edges,
            # 0.8 mm) opened 6 mm with the arm bent and showed in Warudo as a tear. Closed first, the
            # join makes one surface of it
            closed = loose.close_small_holes(mesh)
            if closed:
                print(f"closed cracks in the surface: {closed} vertices brought together")
            before = len(mesh.data.vertices)
            joined = loose.join_touching_surfaces(mesh, armature)
            # a merged vertex keeps one side's weights, and just occasionally that side had none
            filled = rig.fill_unweighted(mesh)
            print(f"joined surfaces that were only touching: {before} -> {joined} vertices"
                  + (f" ({filled} left without weights, filled from around them)" if filled else ""))

    strays = []
    for mesh in meshes:
        strays += loose.fix_weight_islands(mesh, armature)
    if strays:
        print(f"{len(strays)} stray patches handed back to the surface around them")
        for stray in sorted(strays, key=lambda s: -s["vertices"])[:8]:
            print(f"    {stray['vertices']:4d} verts, {stray['was']} -> {stray['now']}")

    for mesh in meshes:
        cut = loose.split_torn_seams(mesh, armature)
        if cut:
            print(f"cut {cut} edges where the surface joined two limbs that move apart")

    changed = []
    for mesh in meshes:
        changed += loose.rigidify_props(mesh, armature)
    if not changed and not strays:
        print("every loose piece already moves as one")
    changed.sort(key=lambda piece: -piece["vertices"])
    print(f"{len(changed)} loose pieces put on one bone each")
    for piece in changed[:12]:
        print(f"    {piece['vertices']:5d} verts, {piece['size_cm']:5.1f} cm across -> {piece['bone']}"
              f" (was pulled by {', '.join(piece['was_torn_with'])})")


main()
