from pathlib import Path
from lxml import etree
import mitsuba as mi

def prettyprint(element, **kwargs):
    xml = etree.tostring(element, pretty_print=True, **kwargs)
    print(xml.decode(), end='')

def export_trained_scene(scene: mi.Scene, scene_fp: str, mesh_indices: list[int], envmap_fp: str = None):
    path = Path(scene_fp)
    scene_directory = path.parent
    scene_name = path.stem

    parser = etree.XMLParser(remove_comments=True)
    document = etree.parse(scene_fp, parser)
    etree.tostring(document)

    # Replace the envmap, if needed
    if envmap_fp is not None:
        emitters = [elem for elem in document.findall("emitter") if elem.get("type") == "envmap"]
        if len(emitters) == 1:
            envmap = emitters[0]
            nodes = [child for child in envmap.findall("string") if child.get("name") == "filename"]
            if len(nodes) == 1:
                filename_node = nodes[0]
                filename_node.set("value", envmap_fp)

        else:
            emitters = [elem for elem in document.findall("emitter") if elem.get("type") == "constant"]
            if len(emitters) == 1:
                envmap = emitters[0]
                env_node = etree.Element("emitter", type="envmap")
                fp_node = etree.SubElement(env_node, "string", name="filename", value=envmap_fp)
                transform_node = etree.SubElement(env_node, "transform", name="to_world")
                rotate  = etree.SubElement(transform_node, "rotate", y="1", angle="90")
                trf_mat = etree.SubElement(transform_node, "matrix", value="-0.224951 -0.000001 -0.974370 0.000000 -0.974370 0.000000 0.224951 0.000000 0.000000 1.000000 -0.000001 8.870000 0.000000 0.000000 0.000000 1.000000")
                parent = envmap.getparent()
                parent.replace(envmap, env_node)

        # <transform name="to_world">
        #     <rotate y="1" angle="90" />
        #     <matrix value="-0.224951 -0.000001 -0.974370 0.000000 -0.974370 0.000000 0.224951 0.000000 0.000000 1.000000 -0.000001 8.870000 0.000000 0.000000 0.000000 1.000000" />
        # </transform>


        # Relit scene, reference
        document.write(scene_directory.joinpath(f"{scene_name}_relit_ref.xml"))


    for mesh_index in mesh_indices:
        # Find the original mesh in the XML scene
        old_shape_node = document.findall("shape")[mesh_index]
        old_shape_fp_node = [child for child in old_shape_node.findall("string") if child.get("name") == "filename"][0]
        mesh_directory = Path(old_shape_fp_node.get("value")).parent

        # Save the actual mesh to disk
        mesh_filepath = scene_directory.joinpath(f"mesh{mesh_index}_textured.ply").__str__()
        # mesh_filepath = mesh_directory.joinpath(f"mesh{mesh_index}_textured.ply").__str__()
        scene.shapes()[mesh_index].write_ply(mesh_filepath)

        # Create an XML node for the new mesh
        shape_node = etree.Element("shape", type="ply")
        shape_fp = etree.SubElement(shape_node, "string", name="filename", value=mesh_filepath)
        node_bsdf = etree.SubElement(shape_node, "bsdf", type="principled")
        bsdf_albedo = etree.SubElement(node_bsdf, "texture", type="mesh_attribute", name="base_color")
        bsdf_albedo_name = etree.SubElement(bsdf_albedo, "string", name="name", value="vertex_bsdf_base_color")
        bsdf_rough = etree.SubElement(node_bsdf, "texture", type="mesh_attribute", name="roughness")
        bsdf_rough_name = etree.SubElement(bsdf_rough, "string", name="name", value="vertex_bsdf_roughness")
        bsdf_metal = etree.SubElement(node_bsdf, "texture", type="mesh_attribute", name="metallic")
        bsdf_metal_name = etree.SubElement(bsdf_metal, "string", name="name", value="vertex_bsdf_metallic")

        # Update XML to reference the new mesh
        parent = old_shape_node.getparent()
        parent.replace(old_shape_node, shape_node)

    if envmap_fp is not None:
        scene_name_new = f"{scene_name}_relit_trained.xml"
    else:
        scene_name_new = f"{scene_name}_trained.xml"
    path_new = scene_directory.joinpath(scene_name_new)
    document.write(path_new)

    return scene_name_new, path_new




def export_init_scene(scene: mi.Scene, scene_fp: str, mesh_indices: list[int]):
    path = Path(scene_fp)
    scene_directory = path.parent
    scene_name = path.stem

    parser = etree.XMLParser(remove_comments=True)
    document = etree.parse(scene_fp, parser)
    etree.tostring(document)

    for mesh_index in mesh_indices:
        # Find the original mesh in the XML scene
        old_shape_node = document.findall("shape")[mesh_index]
        old_shape_fp_node = [child for child in old_shape_node.findall("string") if child.get("name") == "filename"][0]
        mesh_directory = Path(old_shape_fp_node.get("value")).parent

        # Save the actual mesh to disk
        mesh_filepath = scene_directory.joinpath(f"mesh{mesh_index}_textured.ply").__str__()
        # mesh_filepath = mesh_directory.joinpath(f"mesh{mesh_index}_textured.ply").__str__()
        scene.shapes()[mesh_index].write_ply(mesh_filepath)

        # Create an XML node for the new mesh
        shape_node = etree.Element("shape", type="ply", id=f"mesh{mesh_index}")
        shape_fp = etree.SubElement(shape_node, "string", name="filename", value=mesh_filepath)
        node_bsdf = etree.SubElement(shape_node, "bsdf", type="principled")
        bsdf_albedo = etree.SubElement(node_bsdf, "texture", type="mesh_attribute", name="base_color")
        bsdf_albedo_name = etree.SubElement(bsdf_albedo, "string", name="name", value="vertex_bsdf_base_color")
        bsdf_rough = etree.SubElement(node_bsdf, "texture", type="mesh_attribute", name="roughness")
        bsdf_rough_name = etree.SubElement(bsdf_rough, "string", name="name", value="vertex_bsdf_roughness")
        bsdf_metal = etree.SubElement(node_bsdf, "texture", type="mesh_attribute", name="metallic")
        bsdf_metal_name = etree.SubElement(bsdf_metal, "string", name="name", value="vertex_bsdf_metallic")

        # Update XML to reference the new mesh
        parent = old_shape_node.getparent()
        parent.replace(old_shape_node, shape_node)

    scene_name_new = f"{scene_name}_init.xml"
    path_new = scene_directory.joinpath(scene_name_new)
    document.write(path_new)

    return scene_name_new, path_new
