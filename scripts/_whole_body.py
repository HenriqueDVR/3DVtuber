"""Run before a check on a body made of several meshes: joins them into one, in memory only.

    blender -b in.blend --python-exit-code 1 -P scripts/_whole_body.py -P scripts/check_avatar.py -- ...

A .vrm taken in from elsewhere keeps its meshes apart (VRoid: Face, Body, Hair), and the build
never joins them: draw order and first-person settings hang on them. The checks measure one mesh,
the first or the biggest, which on such a file is the hair alone. Joined here they see all of it.
Nothing is saved. A body that is one mesh already is left as it is.
"""
import bpy

armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
meshes = sorted((o for o in bpy.data.objects if o.type == "MESH" and o.find_armature() == armature),
                key=lambda o: -len(o.data.vertices))
if armature is not None and len(meshes) > 1:
    names = [obj.name for obj in meshes[1:]]
    for obj in meshes:
        obj.hide_set(False)
    with bpy.context.temp_override(active_object=meshes[0], object=meshes[0], selected_objects=meshes,
                                   selected_editable_objects=meshes):
        assert bpy.ops.object.join() == {"FINISHED"}
    # the join leaves the others in the file, out of the scene, where nothing moves them: a check
    # taking "the first mesh on the armature" measured one of those and found nothing stretched
    for name in names:
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    print(f"joined {len(meshes)} meshes for the check: {len(meshes[0].data.vertices)} vertices")
for obj in meshes[:1]:
    if obj.parent != armature:
        world = obj.matrix_world.copy()
        obj.parent = armature
        obj.matrix_world = world
