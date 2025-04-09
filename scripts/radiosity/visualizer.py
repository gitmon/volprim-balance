import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-gs/build/python')

import polyscope as ps
import gpytoolbox as gp
import numpy as np
import mitsuba as mi
import drjit as dr
from drjit.auto import Float

def camera_to_mi() -> mi.ScalarTransform4f:
    # return mi.ScalarTransform4f(ps.get_camera_view_matrix())
    M = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]])
    return mi.ScalarTransform4f(np.linalg.inv(ps.get_camera_view_matrix().T @ M).T).rotate([0,0,1], 180)

def plot_sphere(center: np.ndarray = np.zeros(3), radius: float = 1.0, label: str = None) -> ps.surface_mesh.SurfaceMesh:
    V, F = gp.icosphere(n=4)
    V = V * radius + center[None, :]
    return ps.register_surface_mesh(label if label is not None else "Sphere", V, F)

def plot_mesh(mesh: mi.Mesh, label: str = None) -> ps.surface_mesh.SurfaceMesh:
    V, F = mesh.vertex_positions_buffer(), mesh.faces_buffer()
    V = V.numpy().reshape((-1,3))#.T
    F = F.numpy().reshape((-1,3))#.T
    return ps.register_surface_mesh(label if label is not None else "Mesh", V, F, material='flat', edge_width=1.0)

def plot_mesh_attributes(mesh: mi.Mesh, mesh_label: str, attrib_names: list[str], is_color: bool = False):
    m = plot_mesh(mesh, mesh_label)

    for attrib_name in attrib_names:
        tmp = attrib_name.split('_')
        attrib_type = tmp[0]
        attrib_label = ''.join(tmp[1:])

        values = mesh.attribute_buffer(attrib_name).numpy()
        attrib_size = values.size // (mesh.vertex_count() if attrib_type == "vertex" else mesh.face_count())
        domain = "vertices" if attrib_type == "vertex" else "faces"

        if attrib_size == 1:
            m.add_scalar_quantity(attrib_label, values, defined_on=domain, vminmax = (0.0, 1.0), enabled=False)
        elif attrib_size == 3:
            if is_color:
                m.add_color_quantity(attrib_label, values.reshape(-1,3), defined_on=domain, enabled=True)
            else:
                m.add_vector_quantity(attrib_label, values.reshape(-1,3), defined_on=domain, enabled=False)
        else:
            for i in range(attrib_size):
                m.add_scalar_quantity(attrib_label + f"{i}", values[i::attrib_size], defined_on=domain, enabled=False)

def plot_rays(rays: mi.Ray3f, label: str = None) -> ps.point_cloud.PointCloud:
    o, d = rays.o.numpy().T, rays.d.numpy().T
    point_cloud = ps.register_point_cloud(label if label is not None else "Rays", o)
    point_cloud.add_vector_quantity("dirs", d, enabled=True)
    return point_cloud