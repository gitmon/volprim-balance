import drjit as dr
import mitsuba as mi
from drjit.auto import Float
from gpytoolbox import edges

class TVRegularizer:
    def __init__(self, scene: mi.Scene):
        self.rebuild(scene)

    def rebuild(self, scene: mi.Scene) -> None:
        mesh_edge_lists = []
        for mesh in scene.shapes_dr():
            if mesh.is_mesh():
                edge_list = self.build_edge_list(mesh)
            else:
                edge_list = dr.zeros(mi.Point2u)
            mesh_edge_lists.append(edge_list)

        self.mesh_edge_lists = mesh_edge_lists
        self.num_meshes = len(self.mesh_edge_lists)

    def build_edge_list(self, mesh: mi.MeshPtr) -> mi.Point2u:
        faces = dr.unravel(mi.Point3u, mesh.faces_buffer()).numpy()
        E = edges(faces.T)
        edges_dr = mi.Point2u(E[:,0], E[:,1])
        return edges_dr

    def _compute_TV_scalar(self, mesh: mi.MeshPtr, edges_dr: mi.Point2u, attrib_key: str) -> Float:
        attrib = mesh.attribute_buffer(attrib_key)
        # the reverse-mode scatter_reduce has an average contention of 6 (== vtxs valence)
        # TODO: is ReduceMode.Local or ReduceMode.Direct better?
        vtx0_val = dr.gather(type(attrib), attrib, edges_dr.x, mode = dr.ReduceMode.Direct) 
        vtx1_val = dr.gather(type(attrib), attrib, edges_dr.y, mode = dr.ReduceMode.Direct) 
        return dr.mean(dr.abs(vtx0_val - vtx1_val), axis=None)

    def _compute_TV_color(self, mesh: mi.MeshPtr, edges_dr: mi.Point2u) -> Float:
        attrib_key = "vertex_bsdf_base_color"
        attrib = dr.unravel(mi.Color3f, mesh.attribute_buffer(attrib_key))
        # the reverse-mode scatter_reduce has an average contention of 6 (== vtxs valence)
        # TODO: is ReduceMode.Local or ReduceMode.Direct better?
        vtx0_val = dr.gather(type(attrib), attrib, edges_dr.x, mode = dr.ReduceMode.Direct) 
        vtx1_val = dr.gather(type(attrib), attrib, edges_dr.y, mode = dr.ReduceMode.Direct) 
        return dr.mean(dr.abs(vtx0_val - vtx1_val), axis=None)

    def compute_loss(self, scene: mi.Scene, attrib_keys: list[str] = [])-> Float:
        loss = Float(0.0)
        for mesh, edges in zip(scene.shapes_dr(), self.mesh_edge_lists):
            if dr.width(edges) < 1:
                continue
            # Compute the TV on each of the mesh's active attributes
            mesh_loss = Float(0.0)
            for key in attrib_keys:
                mesh_loss += self._compute_TV_scalar(mesh, dr.detach(edges), key)
            mesh_loss += self._compute_TV_color(mesh, dr.detach(edges))

            # Update the global TV loss; the mesh's contribution is normalized by its edge count
            loss += mesh_loss
        return loss / self.num_meshes