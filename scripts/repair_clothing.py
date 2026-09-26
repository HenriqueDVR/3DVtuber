"""Reference-guided clothing repairs for model A and model B, before the T-pose bake.

blender -b source_fingers.blend --python-exit-code 1 -P scripts/repair_clothing.py -- \
    --profile fitted_skirt --reference samples/model_matte.glb --out out/model_clothing

Use --check-only to reproduce the clothing failures without changing or saving a model.
The known reference points select connected original surfaces, NOT everything near a hand.
Existing outputs are refused. Repaired files retain the modelled pose: bake with to_tpose.py
and --keep-finger-pose afterwards. Model A separates the fused skirt/sleeve contact using
interpolated clothing-only cuts; original coordinates, textures and expression keys are retained.
Model B changes weights only: mantle onto the upper arms, and its hanging coat sleeves wholly
onto them. Neither repair selects the hands.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from vtuber_maker.pipeline import loose, meshdata, roles, vrm_export, tpose

# The two characters' pieces as manifest pieces (see pipeline/manifest.py): normalized GLB
# surface points, with component sizes to reject an unrelated reference.
PROFILES = {
    "fitted_skirt": [{"name": "skirt", "point": [-.170004845, -.085588560, .804298997], "count": 13569, "role": "fitted_skirt"},
            {"name": "cuff", "point": [-.223937437, -.059013065, .842598855], "count": 389, "role": "body"},
            {"name": "detail", "point": [-.257547587, -.076208957, .862139642], "count": 118, "role": "detail",
             "anchor": "cuff"}],
    "mantle": [{"name": "cape", "point": [-.306008816, -.073082618, .334538370], "count": 505, "role": "swing_cloth"},
              {"name": "mantle", "point": [-.239570081, -.106692746, 1.106008768], "count": 1039,
               "role": "shoulder_cloth"},
              # the coat's wide sleeves, hanging under each upper arm
              {"name": "sleeve_Left", "point": [.162188590, -.045725424, .996580362], "count": 158,
               "role": "follow_bone", "bone": "LeftUpperArm"},
              {"name": "sleeve_Right", "point": [-.162188560, -.045725465, .996580362], "count": 171,
               "role": "follow_bone", "bone": "RightUpperArm"}],
}


def reference_masks(mesh, reference, profile):
    with roles.reference(reference.resolve()) as ref:
        return roles.find(mesh, PROFILES[profile], ref)


def probes(mesh, arm, masks, profile):
    """Fixed physical checks, derived before the repair (not from its new weights)."""
    coords = meshdata.world_vertices(mesh)
    if profile == "fitted_skirt":
        core = np.zeros(len(coords), dtype=bool)
        inner = core.copy()
        # The inner forearm sleeve lies within this box too; it must follow the arm, not the skirt.
        sleeve = loose._sleeve_tube(mesh, arm, coords, masks["skirt"])
        for side, sign in (("Left", 1), ("Right", -1)):
            elbow = np.array(arm.matrix_world @ arm.data.bones[side + "LowerArm"].head_local)
            wrist = np.array(arm.matrix_world @ arm.data.bones[side + "Hand"].head_local)
            t = np.clip((elbow[2] - coords[:, 2]) / (elbow[2] - wrist[2]), 0, 1)
            axis_x = elbow[0] + t * (wrist[0] - elbow[0])
            medial = (sign * coords[:, 0] > 0) & (coords[:, 2] < elbow[2] - .05)
            core |= medial & (sign * (axis_x - coords[:, 0]) > .035)
            inner |= medial & (sign * (axis_x - coords[:, 0]) > 0)
        return {"skirt_core": np.flatnonzero(masks["skirt"] & core & ~sleeve),
                "inner_sleeve": np.flatnonzero(masks["skirt"] & inner & sleeve)}
    result = {}
    for side, sign in (("Left", 1), ("Right", -1)):
        result[side] = np.flatnonzero(masks["mantle"] & (sign * coords[:, 0] > .24)
                                      & (coords[:, 2] > 1.12) & (coords[:, 2] < 1.20))
        result["sleeve_" + side] = np.flatnonzero(masks["sleeve_" + side])
    return result


def measure(mesh, arm, masks, selected, profile):
    rest = meshdata.world_vertices(mesh)
    saved = {b.name: b.matrix_basis.copy() for b in arm.pose.bones}
    try:
        for side, sign in (("Left", 1), ("Right", -1)):
            tpose._point_arm(arm, side + "UpperArm", side + "Hand", Vector((sign, 0, 0)))
        graph = bpy.context.evaluated_depsgraph_get()
        graph.update()
        posed = meshdata.world_vertices(mesh.evaluated_get(graph))
        if profile == "fitted_skirt":
            indices = selected["skirt_core"]
            assert len(indices), "no skirt probes"
            detail, cuff = np.flatnonzero(masks["detail"]), np.flatnonzero(masks["cuff"])
            tree = KDTree(len(cuff))
            for i in cuff:
                tree.insert(rest[i], int(i))
            tree.balance()
            anchors = np.array([tree.find(rest[i])[1] for i in detail])
            growth = np.linalg.norm(posed[detail] - posed[anchors], axis=1) - np.linalg.norm(rest[detail] - rest[anchors], axis=1)
            sleeve = selected["inner_sleeve"]
            assert len(sleeve), "no inner sleeve probes"
            return {"skirt_probe_vertices": len(indices),
                    "skirt_max_arm_displacement_cm": float(np.linalg.norm(posed[indices] - rest[indices], axis=1).max() * 100),
                    "detail_max_gap_growth_cm": float(growth.max() * 100),
                    "inner_sleeve_vertices": len(sleeve),
                    "inner_sleeve_min_arm_displacement_cm": float(np.linalg.norm(posed[sleeve] - rest[sleeve], axis=1).min() * 100)}
        result = {}
        for key, indices in selected.items():
            assert len(indices), f"no {key} probes"
            bone = arm.pose.bones[key.removeprefix("sleeve_") + "UpperArm"]
            transform = np.array(arm.matrix_world @ bone.matrix @ bone.bone.matrix_local.inverted() @ arm.matrix_world.inverted())
            expected = rest[indices] @ transform[:3, :3].T + transform[:3, 3]
            error = np.linalg.norm(posed[indices] - expected, axis=1)
            result[key] = {"vertices": len(indices), "mean_movement_cm": float(np.linalg.norm(posed[indices] - rest[indices], axis=1).mean() * 100),
                           "max_upper_arm_error_cm": float(error.max() * 100)}
        return result
    finally:
        for name, value in saved.items():
            arm.pose.bones[name].matrix_basis = value
        bpy.context.view_layer.update()


def check(result, profile):
    if profile == "fitted_skirt":
        assert result["skirt_max_arm_displacement_cm"] < .1, result
        assert result["detail_max_gap_growth_cm"] < .5, result
        # nothing on the inner forearm may stay at the waist when the arms go out
        assert result["inner_sleeve_min_arm_displacement_cm"] > 10, result
    else:
        for side in ("Left", "Right"):
            assert result[side]["mean_movement_cm"] > 5, result
            assert result[side]["max_upper_arm_error_cm"] < 1.5, result
            assert result["sleeve_" + side]["max_upper_arm_error_cm"] < .1, result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output stem, no suffix")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
    if any(not np.allclose(np.array(b.matrix_basis), np.eye(4), atol=1e-6) for b in arm.pose.bones):
        raise ValueError("reset the source rig to its modelled rest pose before repair")
    if not args.check_only and any(args.out.with_suffix(ext).exists() for ext in (".blend", ".vrm")):
        raise ValueError("output exists: choose a new stem; original files are never overwritten")
    masks = reference_masks(mesh, args.reference, args.profile)
    selected = probes(mesh, arm, masks, args.profile)
    result = {"source": bpy.data.filepath, "reference": str(args.reference), "profile": args.profile,
              "before": measure(mesh, arm, masks, selected, args.profile)}
    print("BEFORE", json.dumps(result["before"]))
    if args.check_only:
        args.out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        check(result["before"], args.profile)
        return
    original_hands = roles.hands_snapshot(mesh)
    result["repairs"], origin = roles.apply(mesh, arm, PROFILES[args.profile], masks)
    masks = {name: mask[origin] for name, mask in masks.items()}
    selected = {name: np.flatnonzero(np.isin(origin, indices)) for name, indices in selected.items()}
    assert roles.hands_snapshot(mesh, origin) == original_hands, "clothing repair changed hand geometry, keys, skinning or UVs"
    result["hands_unchanged_before_bake"] = True
    result["after"] = measure(mesh, arm, masks, selected, args.profile)
    print("AFTER", json.dumps(result["after"]))
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    check(result["after"], args.profile)
    vrm_export.ensure_vrm_addon()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out.with_suffix(".blend").resolve()), copy=True)
    vrm_export.export(args.out.with_suffix(".vrm").resolve(), arm)
    print("PASS clothing regression; wrote", args.out)


if __name__ == "__main__":
    main()
