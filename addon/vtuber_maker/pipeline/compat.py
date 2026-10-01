"""What differs between the Blender versions the pipeline runs on (4.2 LTS and later).

Everything version-dependent goes through here, so a new Blender release is one file to check.
The pipeline refuses Blender older than MIN_VERSION; later versions run, untested ones with a warning
from the app (app/engine.py TESTED).
"""

import bpy

MIN_VERSION = (4, 2, 0)

if bpy.app.version < MIN_VERSION:
    print('VTUBER_MAKER_ERROR {"code": "VM-BLENDER-OLD", "detail": "needs Blender 4.2 or later, this is %s"}'
          % bpy.app.version_string)
    raise SystemExit(1)

# EEVEE Next took the name BLENDER_EEVEE_NEXT in 4.2 and got the old name back in 5.0.
EEVEE = "BLENDER_EEVEE_NEXT" if bpy.app.version < (5, 0, 0) else "BLENDER_EEVEE"


def set_alpha(material, blend):
    """Blended or alpha-clipped transparency for a plain (non-MToon) material.

    4.2 replaced blend_method with surface_render_method ("BLENDED"/"DITHERED"); the old property is
    deprecated and may go. Clipping is kept on blend_method while it exists: EEVEE Next has no clip
    mode of its own, and the glTF/VRM exporters still read blend_method for alphaMode MASK.
    """
    if hasattr(material, "blend_method"):
        material.blend_method = "BLEND" if blend else "CLIP"
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "BLENDED" if blend else "DITHERED"


def import_fbx(path):
    """The Python FBX importer (the add-on Tripo's rigs were tested with), else 4.5's built-in one."""
    if hasattr(bpy.ops.import_scene, "fbx"):
        return bpy.ops.import_scene.fbx(filepath=str(path))
    return bpy.ops.wm.fbx_import(filepath=str(path))
