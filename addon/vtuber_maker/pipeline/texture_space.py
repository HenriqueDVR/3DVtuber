"""Map texture pixels to the 3D surface points they cover."""

import bpy
import numpy as np

# Baked positions are squeezed into [MARGIN, 1 - MARGIN] so empty texels (0) stand out.
MARGIN = 0.05


def vertex_positions(mesh):
    coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    return coords.reshape(-1, 3)


def base_color_image(obj):
    """The image feeding Base Color of the first textured material on obj."""
    for slot in obj.material_slots:
        image = material_base_color_image(slot.material)
        if image is not None:
            return image
    raise ValueError(f"{obj.name} has no base color texture")


def material_base_color_image(material):
    """The image linked into a Principled BSDF's Base Color, or None."""
    if material is None or not material.use_nodes:
        return None
    for node in material.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        links = node.inputs["Base Color"].links
        if links and links[0].from_node.type == "TEX_IMAGE" and links[0].from_node.image:
            return links[0].from_node.image
    return None


def image_rgb(image):
    """Pixels as a (height, width, 3) array, rows bottom to top."""
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(height, width, 4)[..., :3]


def vertex_colors(obj, image):
    """Average texture color under each vertex's UVs, shape (vertices, 3)."""
    mesh = obj.data
    rgb = image_rgb(image)
    height, width = rgb.shape[:2]
    uv = np.empty(len(mesh.loops) * 2)
    mesh.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2) % 1.0
    loop_vertices = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loop_vertices)
    rows = (uv[:, 1] * height).astype(int).clip(0, height - 1)
    columns = (uv[:, 0] * width).astype(int).clip(0, width - 1)
    sums = np.zeros((len(mesh.vertices), 3))
    np.add.at(sums, loop_vertices, rgb[rows, columns])
    counts = np.bincount(loop_vertices, minlength=len(mesh.vertices))
    return sums / np.maximum(counts, 1)[:, None]


def bake_position_map(obj, width, height=None):
    """Bake the object-space position behind every texel of the active UV map.

    Returns (positions, covered): positions is (height, width, 3) and covered marks
    texels inside a UV island. Rows run bottom to top, like Blender image pixels.
    """
    height = height or width
    verts = vertex_positions(obj.data)
    low = verts.min(axis=0) - 0.01
    span = verts.max(axis=0) + 0.01 - low

    image = bpy.data.images.new("position_bake", width, height, alpha=True, float_buffer=True)
    image.colorspace_settings.name = "Non-Color"

    material = bpy.data.materials.new("position_bake")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    coord = nodes.new("ShaderNodeTexCoord")
    shift = nodes.new("ShaderNodeVectorMath")
    shift.operation = "SUBTRACT"
    shift.inputs[1].default_value = low.tolist()
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "DIVIDE"
    scale.inputs[1].default_value = (span / (1 - 2 * MARGIN)).tolist()
    offset = nodes.new("ShaderNodeVectorMath")
    offset.operation = "ADD"
    offset.inputs[1].default_value = (MARGIN, MARGIN, MARGIN)
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    target = nodes.new("ShaderNodeTexImage")
    target.image = image
    links.new(coord.outputs["Object"], shift.inputs[0])
    links.new(shift.outputs["Vector"], scale.inputs[0])
    links.new(scale.outputs["Vector"], offset.inputs[0])
    links.new(offset.outputs["Vector"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    nodes.active = target

    original_materials = [slot.material for slot in obj.material_slots]
    obj.data.materials.clear()
    obj.data.materials.append(material)

    scene = bpy.context.scene
    previous_engine = scene.render.engine
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type="EMIT", margin=2, use_clear=True)
    scene.render.engine = previous_engine

    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(height, width, 4)[..., :3]
    covered = pixels.max(axis=2) > MARGIN / 2
    positions = (pixels - MARGIN) / (1 - 2 * MARGIN) * span + low

    obj.data.materials.clear()
    for original in original_materials:
        obj.data.materials.append(original)
    bpy.data.materials.remove(material)
    bpy.data.images.remove(image)
    return positions, covered
