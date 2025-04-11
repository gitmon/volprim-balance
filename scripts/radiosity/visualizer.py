import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-gs/build/python')

import polyscope as ps
import gpytoolbox as gp
import numpy as np
import mitsuba as mi
import drjit as dr
from drjit.auto import Float, UInt

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

def visualize_ellipsoids(gs_scene: mi.Scene):
    # Look for the ellipsoids mesh in the scene
    ellipsoids_found = False
    for shape in gs_scene.shapes():
        if shape.shape_type() == +mi.ShapeType.Ellipsoids:
            mesh: mi.Mesh = shape
            ellipsoids_found = True
            break

    if not(ellipsoids_found):
        raise Exception("Scene does not contain an EllipsoidMesh!")

    # `primitive_count()` returns the number of triangles used in the mesh representation of *all* the gaussians
    tri_count = mesh.primitive_count()

    pose_key = "primitives.data"
    params = mi.traverse(gs_scene)

    # 10 floats define the pose: translation (3) + rotation (4) + scale (3)
    pose_count = 10     
    pose_data = params[pose_key]
    ellipsoid_count = dr.width(pose_data) // pose_count
    tris_per_splat = tri_count // ellipsoid_count

    # Load splat positions
    indices = dr.arange(UInt, ellipsoid_count)
    x = dr.gather(dr.auto.ad.Float, pose_data, pose_count * indices + 0)
    y = dr.gather(dr.auto.ad.Float, pose_data, pose_count * indices + 1)
    z = dr.gather(dr.auto.ad.Float, pose_data, pose_count * indices + 2)
    pos = mi.Point3f(x,y,z)

    # Build per-triangle buffers of i) sh_coeffs; ii) radiant intensities
    tri_count = mesh.primitive_count()
    F = dr.unravel(mi.Point3u, mesh.faces_buffer())
    v1 = mesh.vertex_position(F.x)
    v2 = mesh.vertex_position(F.y)
    v3 = mesh.vertex_position(F.z)
    vc = (v1 + v2 + v3) / 3.0
    nc = dr.cross(v2 - v1, v3 - v1)
    tri_areas = dr.norm(nc)
    nc /= tri_areas
    ell_index = dr.repeat(dr.arange(UInt, 0, ellipsoid_count), tris_per_splat)

    si = dr.zeros(mi.SurfaceInteraction3f, tri_count)
    si.p, si.n, si.prim_index = vc, nc, ell_index

    sh_coeffs = mesh.eval_attribute_x("sh_coeffs", si)
    sh_coeffs[0] += 0.5 * dr.sqrt(dr.four_pi)
    sh_coeffs[1] += 0.5 * dr.sqrt(dr.four_pi)
    sh_coeffs[2] += 0.5 * dr.sqrt(dr.four_pi)
    sh_norm = dr.norm(sh_coeffs)
    opacities = mesh.eval_attribute_1("opacities", si)
    I = sh_norm * opacities #* tri_areas

    # Begin visualization
    ps.init()
    points = ps.register_point_cloud("GS", pos.numpy().T)

    # Gather per-ellipsoid data by picking one triangle for each ellipsoid
    tri_idxs = dr.arange(UInt, ellipsoid_count) * tris_per_splat
    dc_R = dr.gather(dr.auto.ad.Float, sh_coeffs[0], tri_idxs)
    dc_G = dr.gather(dr.auto.ad.Float, sh_coeffs[1], tri_idxs)
    dc_B = dr.gather(dr.auto.ad.Float, sh_coeffs[2], tri_idxs)
    dc_color = mi.Color3f(dc_R, dc_G, dc_B) / dr.sqrt(dr.four_pi)

    points.add_color_quantity("dc_color", dc_color.numpy().T)
    points.add_scalar_quantity("I", (dr.gather(dr.auto.ad.Float, I, tri_idxs)).numpy().ravel())
    ps.show()