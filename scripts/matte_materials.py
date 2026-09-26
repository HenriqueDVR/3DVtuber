"""Make a generated model's materials matte: no metallic/roughness map, metallic 0, roughness 0.9.

    blender -b -P scripts/matte_materials.py -- in.glb out.glb

Tripo ships a metallic/roughness map that makes skin and cloth look like plastic or metal in
ordinary viewers. Colour and normal maps are kept. The original file is left alone.
"""
import sys
from pathlib import Path

import bpy

ROUGHNESS = 0.9


def main():
    src, out = (Path(p).resolve() for p in sys.argv[sys.argv.index("--") + 1:])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(src))
    unused = set()
    for material in bpy.data.materials:
        if not material.use_nodes or "Principled BSDF" not in material.node_tree.nodes:
            continue
        tree = material.node_tree
        bsdf = tree.nodes["Principled BSDF"]
        for socket in ("Metallic", "Roughness"):
            for link in list(bsdf.inputs[socket].links):
                node = link.from_node
                tree.links.remove(link)
                unused.add((tree, node))
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = ROUGHNESS
        # Newer Tripo exports multiply the colour map by a grey base colour factor (a Mix node),
        # which darkens it and hides the image from scripts that expect it on Base Color.
        for link in list(bsdf.inputs["Base Color"].links):
            mix = link.from_node
            if mix.bl_idname == "ShaderNodeMix" and mix.blend_type == "MULTIPLY":
                image = next((s.links[0].from_node for s in mix.inputs
                              if s.links and s.links[0].from_node.bl_idname == "ShaderNodeTexImage"), None)
                if image is not None:
                    tree.links.new(image.outputs["Color"], bsdf.inputs["Base Color"])
                    tree.nodes.remove(mix)
    # drop the nodes that only fed metallic/roughness (the separate-colour node and its image)
    for tree, node in unused:
        if node.name not in tree.nodes or any(out.links for out in node.outputs):
            continue
        feeders = [link.from_node for socket in node.inputs for link in socket.links]
        tree.nodes.remove(node)
        for feeder in feeders:
            if not any(out.links for out in feeder.outputs):
                tree.nodes.remove(feeder)
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB")
    print("wrote", out)


main()
