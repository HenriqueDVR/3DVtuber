"""Make a T-pose copy of an A-pose source model, to test the pipeline on T-pose input.

    blender -b rigged_apose.blend -P scripts/make_tpose_sample.py -- source.glb out.glb

The rigged A-pose (a build's `pieces` step, say) swings its arms out to the VRM T-pose; each
vertex of the source model then moves as the rig vertex on top of it did, so the copy keeps
the source's topology, UVs and texture and differs only in pose, like the same character
generated in a T-pose. Fingers are left as modelled. A real image-to-3D T-pose also models the
armpits open, which this can't: treat it as a stand-in, not a replacement.
"""

import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils.kdtree import KDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "addon"))

from vtuber_maker.pipeline import meshdata, normalize, tpose  # noqa: E402

# Source vertices further than this from the rig (m) mean the rig wasn't built from this source.
MATCH = 1e-3


def main():
    source, out = (Path(a).resolve() for a in sys.argv[sys.argv.index("--") + 1:][:2])
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = max((o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm), key=lambda o: len(o.data.vertices))
    extra = [o for o in bpy.data.objects if o.type == "MESH" and o is not mesh
             and any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)]
    before = meshdata.world_vertices(mesh)
    turned = tpose.to_t_pose(mesh, arm, extra_meshes=extra, straighten=False)
    after = meshdata.world_vertices(mesh)
    print(f"arms turned: {turned}")

    old = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(source))
    model = normalize.prepare(list(set(bpy.data.objects) - old), height=1.6)
    verts = meshdata.world_vertices(model)
    tree = KDTree(len(before))
    for i, p in enumerate(before):
        tree.insert(p, i)
    tree.balance()
    found = [tree.find(p) for p in verts]
    gap = np.array([f[2] for f in found])
    if np.percentile(gap, 95) > MATCH:
        raise SystemExit(f"the rig isn't built from {source.name}: 95th percentile gap {np.percentile(gap, 95):.4f} m")
    moved = verts + (after - before)[[f[1] for f in found]]
    inverse = np.linalg.inv(np.array(model.matrix_world))
    local = moved @ inverse[:3, :3].T + inverse[:3, 3]
    model.data.vertices.foreach_set("co", local.astype(np.float32).ravel())
    model.data.update()
    print(f"moved {len(verts)} vertices, furthest {np.linalg.norm(moved - verts, axis=1).max():.3f} m")

    bpy.ops.object.select_all(action="DESELECT")
    model.select_set(True)
    bpy.context.view_layer.objects.active = model
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", use_selection=True, export_yup=True)
    print(f"wrote {out}")


main()
