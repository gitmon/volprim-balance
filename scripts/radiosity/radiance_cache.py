import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-gs/build/python')

import drjit as dr
import mitsuba as mi
from drjit.auto import Float, UInt, Bool
from scripts.radiosity.sh_fitting import get_sh_count, fit_sh_on_scene
from scripts.radiosity.surface_sampler import SceneSurfaceSampler

def balance_heuristic(pdf1: Float, pdf2: Float, n1: int = 1, n2: int = 1):
    return (n1 * pdf1) * dr.rcp(dr.fma(n1, pdf1, n2 * pdf2))

def balance_heuristic_3(pdf1: Float, pdf2: Float, pdf3: Float, n1: int = 1, n2: int = 1, n3: int = 1):
    return (n1 * pdf1) * dr.rcp(dr.fma(n1, pdf1, dr.fma(n2, pdf2, n3 * pdf3)))

def power_heuristic(pdf1: Float, pdf2: Float, n1: int = 1, n2: int = 1):
    pdf1_ = dr.square(pdf1 * n1)
    pdf2_ = dr.square(pdf2 * n2)
    return pdf1_ * dr.rcp(pdf1_ + pdf2_)

def compute_face_areas(mesh: mi.Mesh, face_idxs: mi.Point3u):
        p0 = mesh.vertex_position(face_idxs.x)
        p1 = mesh.vertex_position(face_idxs.y)
        p2 = mesh.vertex_position(face_idxs.z)
        e0, e1 = p1 - p0, p2 - p0
        return 0.5 * dr.norm(dr.cross(e0, e1))


# TODO: update for GS scene.
class EnergyPMF:
    def __init__(self, scene: mi.Scene):
        self.build_energy_pmf(scene)
        self.meshes = scene.shapes_dr()
        # keep scene reference to perform ray intersection tests
        self.scene = scene

    def build_energy_pmf(self, scene: mi.Scene) -> mi.DiscreteDistribution:
        # Populate the meshes with SH coefficient data
        print("Fitting spherical harmonics to scene...")

        sh_order = 3
        fit_sh_on_scene(scene, sh_order)

        print("Fitting complete.")

        # Compute triangle indexing data
        meshes = scene.shapes_dr()
        mesh_active = meshes.is_mesh()
        face_counts = dr.cuda.ad.UInt(dr.select(mesh_active, mi.MeshPtr(meshes).face_count(), 0))
        mesh_start_indices = dr.prefix_reduce(dr.ReduceOp.Add, face_counts, exclusive=True)
        mesh_end_indices   = dr.prefix_reduce(dr.ReduceOp.Add, face_counts, exclusive=False)

        # Compute per-triangle energies
        E_scene = dr.zeros(Float, mesh_end_indices[-1])
        for mesh_idx, mesh in enumerate(meshes):
            # Compute the outgoing energy from each mesh vertex as the sum of the squared SH coefficients
            vertex_energies = dr.zeros(Float, mesh.vertex_count())
            for sh_idx in range(get_sh_count(sh_order)):
                vertex_coeffs = dr.unravel(mi.Color3f, mesh.attribute_buffer(f"vertex_Lo_coeffs_{sh_idx}"))
                vertex_energies += dr.squared_norm(vertex_coeffs)

            E_faces = dr.zeros(Float, mesh.face_count())
            face_idxs = dr.unravel(mi.Point3u, mesh.faces_buffer())
            face_area = compute_face_areas(mesh, face_idxs)
            dtype = type(vertex_energies)
            # E_vi has units of (irradiance) ** 2 -- it is the integral of the *squared* radiance Lo over 
            # the hemisphere. 
            E_v1 = dr.gather(dtype, vertex_energies, face_idxs.x)
            E_v2 = dr.gather(dtype, vertex_energies, face_idxs.y)
            E_v3 = dr.gather(dtype, vertex_energies, face_idxs.z)
            # To compute the emitted power of the triangle, take the sqrt() of E_vi and scale by the triangle area.
            E_faces = face_area * dr.rcp(3.0) * (dr.sqrt(E_v1) + dr.sqrt(E_v2) + dr.sqrt(E_v3))

            # Store the mesh faces' energy values in the scene-level energy-per-vertex array
            start_idx = mesh_start_indices[mesh_idx]
            end_idx = mesh_end_indices[mesh_idx]
            dr.scatter_add(E_scene, E_faces, dr.arange(UInt, start_idx, end_idx))

        # Construct the probability distribution
        pmf = mi.DiscreteDistribution(E_scene)
        self.pmf = pmf
        self.mesh_start_idxs = mesh_start_indices 
        self.mesh_end_idxs = mesh_end_indices

    
    def sample(self, si: mi.SurfaceInteraction3f, sample1: Float, sample2: mi.Point2f) -> tuple[mi.Vector3f, Float, Float]:
        # Sample a triangle in the emissive scene
        # NOTE: sample_reuse_pmf() appears to be nondeterministic wrt reused_sample!!!
        global_face_idx, pdf = self.pmf.sample_pmf(sample1)

        # Find the mesh and prim index corresponding to this triangle
        dtype = type(self.mesh_end_idxs)
        mesh_count = len(self.mesh_end_idxs)
        mesh_idxs = dr.binary_search(0, mesh_count, lambda index: dr.gather(dtype, self.mesh_end_idxs, index) <= global_face_idx)
        idx_offset = dr.gather(dtype, self.mesh_start_idxs, mesh_idxs)
        local_face_idx = global_face_idx - idx_offset

        # Sample a point on the triangle
        mesh = dr.gather(mi.MeshPtr, self.meshes, mesh_idxs)
        fi = mesh.face_indices(local_face_idx)

        p0 = mesh.vertex_position(fi.x)
        p1 = mesh.vertex_position(fi.y)
        p2 = mesh.vertex_position(fi.z)
        e0 = p1 - p0
        e1 = p2 - p0
        b = mi.warp.square_to_uniform_triangle(sample2)

        # Compute surface data at sample
        pdf_area = pdf / compute_face_areas(mesh, fi)
        sample_pos = dr.fma(e0, b.x, dr.fma(e1, b.y, p0))
        sample_n = dr.select(
            mesh.has_vertex_normals(), 
            dr.fma(mesh.vertex_normal(fi.x), (1.0 - b.x - b.y),
                    dr.fma(mesh.vertex_normal(fi.y), b.x,
                           mesh.vertex_normal(fi.z) * b.y)),
            dr.cross(e0, e1))
        sample_n = dr.normalize(sample_n)

        # Handle geometry term and convert pdf to solid angle measure
        d_world = sample_pos - si.p
        dist_squared = dr.squared_norm(d_world)
        d_world *= dr.rsqrt(dist_squared)
        cos_theta_o = dr.abs(dr.dot(sample_n, -d_world))
        G = dr.select(dist_squared > 0.0, cos_theta_o * dr.rcp(dist_squared), 0.0)
        pdf_angle = dr.select(G > 0.0, pdf_area * dr.rcp(G), 0.0)

        # Handle visibility
        vis_ray = si.spawn_ray_to(sample_pos)
        occluded = self.scene.ray_test(vis_ray)

        # MC weight: vis / l_pdf_angle
        weight = dr.select(~occluded & (pdf_angle > 0.0), dr.rcp(pdf_angle), 0.0)
        d_local = si.to_local(d_world)
        return d_local, weight, pdf_angle
    
    def eval_pdf(self, si: mi.SurfaceInteraction3f, wi_local: mi.Vector3f) -> Float:
        # Launch rays along `wi` and identify the hit emissive triangle
        d_world = si.to_world(wi_local)
        ray = si.spawn_ray(d_world)
        pi = self.scene.ray_intersect_preliminary(ray)
        active = pi.is_valid()
        local_face_idx = pi.prim_index

        # Find the index of the hit triangle in the energy PMF
        mesh_idx = dr.zeros(dr.cuda.ad.UInt, dr.width(wi_local))
        for i, shape in enumerate(self.meshes):
            mesh_idx[pi.shape == shape] = i
        idx_offsets = dr.gather(type(self.mesh_start_idxs), self.mesh_start_idxs, mesh_idx)
        global_face_idx = dr.select(active, local_face_idx + idx_offsets, 0)
        
        # Evaluate the hit probability
        meshes = dr.gather(mi.MeshPtr, self.meshes, mesh_idx)
        fi = meshes.face_indices(local_face_idx)
        pdf = dr.select(active, 
                        self.pmf.eval_pmf_normalized(global_face_idx) / compute_face_areas(meshes, fi),
                        Float(0.0))
        
        # Compute geometry term for this hit point
        dist = pi.t
        si_triangle = pi.compute_surface_interaction(ray, active=active)
        # TODO: sh_frame.n or si.n?
        # cos_theta_o = dr.abs(dr.dot(si_triangle.sh_frame.n, -d_world))
        cos_theta_o = dr.abs(dr.dot(si_triangle.n, -d_world))   
        G = dr.select(active & (dist > 0.0), cos_theta_o * dr.rcp(dist * dist), 0.0)

        # Convert area pdf to solid angle measure
        pdf_angle = dr.select(G > 0.0, pdf * dr.rcp(G), 0.0)
        return pdf_angle
    
    def test(self, si: mi.SurfaceInteraction3f, sample1: Float, sample2: mi.Point2f):
        wi_local, _, pdf_ref, G_ref = self.sample(si, sample1, sample2)
        pdf, G = self.eval_pdf(si, wi_local)
        print(pdf_ref)
        print(pdf)
        print(G_ref)
        print(G)
        print("pdf close: ", dr.allclose(pdf, pdf_ref))
        print("G close: ", dr.allclose(G, G_ref))
        print((pdf - pdf_ref).numpy())


from enum import Enum
import json

class SamplingMethod(Enum):
    Emitter = 0
    Envmap = 1
    Cosine = 2
    BSDF = 3

class RadianceCache:
    def __init__(self, mi_scene: mi.Scene, pcd_path: str):
        '''
        Inputs:
            - pcd_path: str. The filepath to the GS scene, "point_cloud.ply".
            - spp_per_wo: int. The number of pathtrace samples to use per Lo ray.
            - spp_per_wi: int. The number of pathtrace samples to use per Li ray.
        '''
        scene_dict = {
            'type': 'scene',
            'primitives': {
                'type': 'ellipsoidsmesh',
                'filename': pcd_path
            },
            'integrator': {
                'type': 'volprim_rf',
                'max_depth': 128,
                'rr_depth':  128,
                'kernel_type': 'gaussian',
            }
        }

        # self.mi_scene = mi_scene
        self.gs_scene: mi.Scene = mi.load_dict(scene_dict)
        self.integrator: mi.ad.common.ADIntegrator = self.gs_scene.integrator()
        
        # TODO
        self.surface_eps = 3e-1
        # self.energy_pmf = EnergyPMF(self.gs_scene)

        # TODO
        # merged_dict = scene_dict
        # self.merged_scene: mi.Scene = mi.load_dict(merged_dict)

    def _pathtrace(self, rays: mi.Ray3f, sampler_rt: mi.Sampler, rng_state: int, active: Bool = None) -> tuple[mi.Color3f, int]:
        '''
        Inputs:
            - TODO
        Outputs:
            - L: mi.Color3f. Incident radiances along `rays`, computed via pathtracing.
            - rng_state: int. RNG seed for the next operation involving random numbers.
        '''
        if active is None:
            active = dr.full(Bool, True, dr.width(rays))
        num_rays = dr.width(rays)
        sampler_rt.seed(rng_state, num_rays); rng_state += 0x00FF_FFFF
        # L = self.integrator.sample(self.gs_scene, sampler_rt, rays, active = active)[0]
        L, active = self.integrator.sample(dr.ADMode.Primal, self.gs_scene, sampler_rt, rays, δL = None, state_in = 0.0, active = active)[:2]
        L = dr.select(active, L, dr.zeros(mi.Color3f))
        return L, rng_state
    
    def _ray_intersect_geometry(self, ray_: mi.Ray3f):
        with dr.suspend_grad():
            active = dr.ones(mi.Bool, dr.width(ray_))
            escaped = dr.zeros(mi.Bool, dr.width(ray_))
            t = dr.zeros(Float, dr.width(ray_))
            ray = mi.Ray3f(ray_)
            # Trace the ray, ignoring ellipsoids, until it either hits real geometry
            # or leaves the scene.
            while dr.any(active):
                    pi: mi.PreliminaryIntersection3f = self.merged_scene.ray_intersect_preliminary(ray, active)
                    t[active] += pi.t
                    escaped |= ~pi.is_valid()
                    active &= ~escaped & (pi.shape.shape_type() & +mi.ShapeType.Ellipsoids)
                    ray.o[active] += ray.d * (pi.t + 1e-4)

            hit_geo = ~escaped
            return t, hit_geo, pi.compute_surface_interaction(ray)

    
    def _spawn_offset_ray(self, si: mi.SurfaceInteraction3f, d: mi.Vector3f):
        # Fixed epsilon method
        ray = mi.Ray3f(si.p, d) # si.spawn_ray(d)
        ray.o += self.surface_eps * ray.d
        return ray
    
        # # TODO: `intersect_preliminary` method
        # ray = si.spawn_ray(d)
        # pi = self.mi_scene.ray_intersect_preliminary(ray)
        # surface_eps = dr.select(
        #     pi.is_valid(), 
        #     dr.maximum(0.5 * pi.t, self.surface_eps), 
        #     self.surface_eps)
        # ray.o += surface_eps * ray.d
        # return ray

        # # TODO: `intersect_preliminary` method
        # max_extent = dr.max(self.merged_scene.bbox().extents())
        # ray = si.spawn_ray(d)
        # t, hit_geo = self._ray_intersect_geometry(ray)
        # surface_offset = dr.select(
        #     hit_geo,
        #     dr.maximum(0.5 * t, self.surface_eps),
        #     0.5 * max_extent)
        # ray.o += surface_offset * ray.d
        # return ray

    def eval_Le(self, si: mi.SurfaceInteraction3f) -> mi.Color3f:
        # mesh = si.shape
        # return dr.select(mesh.is_emitter(), mesh.emitter().eval(si), dr.zeros(mi.Color3f))
        # TODO
        return dr.zeros(mi.Color3f, dr.width(si))

    def eval_Lo(self, si: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, rng_state: int = 0)  -> tuple[mi.Color3f, Bool, int]:
        '''
        Inputs:
            - sampler: Sampler. The pseudo-random number generator.
            - si: SurfaceInteraction3f. Array of surface sample points of size [#si,].
        Outputs: 
            - Lo: mi.Color3f. Array of outgoing radiances of size [#si,].
            - active: dr.Bool. Active lanes.
            - rng_state: int. RNG seed for the next operation involving random numbers.
        '''
        # Compute the outgoing radiance from `A` for a direction, `wo`
        # Note that `wo` is stored in the `si.wi` field (unintuitive, but needed for BSDF.eval() later)
        wo_local = si.wi
        wo_world = si.to_world(wo_local)
        # Lo_rays = si.spawn_ray(wo_world)
        Lo_rays = self._spawn_offset_ray(si, wo_world)
        Lo_rays.d = -Lo_rays.d

        active = True   # TODO: disabled
        # # The ray that's spawned from `si` should intersect `si.shape` again. Due to RayEpsilons and 
        # # spawn offsets, there exist edge cases where this does not occur: for example, when `si` lies
        # # on the boundary edge of a rectangular plane. If this occurs, we should omit this `si` from 
        # # the loss calculation.
        # active = (self.mi_scene.ray_intersect_preliminary(Lo_rays).shape == si.shape)

        # Pathtrace along `-wo` to get the radiance when looking at `A`. For each `Lo_ray`,
        # compute `SPP_LO` different pathtraced samples and average them to get the outgoing 
        # radiance.
        Lo, rng_state = self._pathtrace(Lo_rays, sampler_rt, rng_state, active)
        return Lo, active, rng_state, Lo_rays

    def _render_hemisphere(self, si_scalar: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, N: int, offset: float = 0.0):
        theta = dr.linspace(Float, 0.0, 0.5 * dr.pi, N)
        phi = dr.linspace(Float, 0.0, dr.two_pi, 4 * N)
        tt, pp = dr.meshgrid(theta, phi)

        # wi = mi.sph_to_dir(tt, pp)
        st, ct = dr.sincos(tt)
        sp, cp = dr.sincos(pp)
        wi = mi.Vector3f(
            st * cp,
            st * sp,
            ct)

        wi_world = si_scalar.to_world(wi)

        rays = si_scalar.spawn_ray(wi_world)
        rays.o += offset * rays.d
        L = self._pathtrace(rays, sampler_rt, rng_state=0)[0]
        image_out = mi.TensorXf(dr.ravel(L), shape=(4*N, N, 3))
        return image_out, rays


    def eval_Li_mat(self, si_wide: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, rng_state: int = 0) \
        -> tuple[mi.Color3f, mi.Vector3f, Bool, int]:
        '''
        Inputs:
            - si_wide: SurfaceInteraction3f. Widened array of surface sample points of size [#si * #wi,].
            - sampler: Sampler. The pseudo-random number generator.
            - rng_state: int. The RNG seed.
        Outputs: 
            - Li: mi.Color3f. Flattened array of incident radiances of size [#si * #wi,]. The data 
            is in contiguous order, i.e. the first #wi entries belong to si0, and so on.
            - wi_local: mi.Vector3f. Flattened array of incident directions of size [#si * #wi,].
            - active: dr.Bool. Active lanes.
            - rng_state: int. RNG seed for the next operation involving random numbers.
        '''
        sampler_rt.seed(rng_state, dr.width(si_wide)); rng_state += 0x00FF_FFFF
        uv = sampler_rt.next_2d()

        hemi_wi = mi.warp.square_to_cosine_hemisphere(uv)
        hemi_pdf = dr.maximum(0.0, mi.warp.square_to_cosine_hemisphere_pdf(hemi_wi))
        hemi_weight = dr.select(hemi_pdf > 0.0, dr.rcp(hemi_pdf), 0.0)
        wi_local, wi_pdf, wi_weight = hemi_wi, hemi_pdf, hemi_weight

        assert not(dr.any((wi_pdf == 0.0) & (wi_weight != 0.0)))

        # wi_rays = si_wide.spawn_ray(si_wide.to_world(wi_local))
        wi_rays = self._spawn_offset_ray(si_wide, si_wide.to_world(wi_local))

        active = wi_weight > 0.0

        # Compute Li for each of the incident directions. For each `Li_ray`, trace `SPP_LI` 
        # different MC samples and average them to get the outgoing radiance.
        Li, rng_state = self._pathtrace(wi_rays, sampler_rt, rng_state, active)
        Li *= wi_weight
        return Li, wi_local, active, rng_state, wi_rays

    # TODO
    # def eval_Li(self, si_wide: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, rng_state: int = 0, sampling_method: SamplingMethod = SamplingMethod.Cosine) \
    #     -> tuple[mi.Color3f, mi.Vector3f, Bool, int]:
    #     '''
    #     Inputs:
    #         - si_wide: SurfaceInteraction3f. Widened array of surface sample points of size [#si * #wi,].
    #         - sampler: Sampler. The pseudo-random number generator.
    #         - rng_state: int. The RNG seed.
    #     Outputs: 
    #         - Li: mi.Color3f. Flattened array of incident radiances of size [#si * #wi,]. The data 
    #         is in contiguous order, i.e. the first #wi entries belong to si0, and so on.
    #         - wi_local: mi.Vector3f. Flattened array of incident directions of size [#si * #wi,].
    #         - active: dr.Bool. Active lanes.
    #         - rng_state: int. RNG seed for the next operation involving random numbers.
    #     '''
    #     sampler_rt.seed(rng_state, dr.width(si_wide)); rng_state += 0x00FF_FFFF
    #     uv = sampler_rt.next_2d()

    #     if sampling_method == SamplingMethod.Emitter:
    #         # light_pdf is expressed in units of solid angle
    #         em_wi, em_weight, em_pdf = self.energy_pmf.sample(si_wide, sampler_rt.next_1d(), uv)

    #         # Evaluate material pdf and MIS weight
    #         # max() is needed because this pdf() implementation can return negative values for invalid directions!
    #         hemi_pdf = dr.maximum(0.0, mi.warp.square_to_cosine_hemisphere_pdf(em_wi))
    #         mis_weight = dr.select(hemi_pdf > 0.0, balance_heuristic(em_pdf, hemi_pdf), 1.0)
    #         em_weight *= mis_weight 
    #         wi_local, wi_pdf, wi_weight = em_wi, em_pdf, em_weight
    #     elif sampling_method == SamplingMethod.Cosine:
    #         hemi_wi = mi.warp.square_to_cosine_hemisphere(uv)
    #         hemi_pdf = dr.maximum(0.0, mi.warp.square_to_cosine_hemisphere_pdf(hemi_wi))
    #         hemi_weight = dr.select(hemi_pdf > 0.0, dr.rcp(hemi_pdf), 0.0)

    #         # Evaluate light pdf and MIS weight
    #         # light_pdf is expressed in units of solid angle
    #         em_pdf = self.energy_pmf.eval_pdf(si_wide, hemi_wi)
    #         mis_weight = dr.select(em_pdf > 0.0, balance_heuristic(hemi_pdf, em_pdf), 1.0)
    #         hemi_weight *= mis_weight
    #         wi_local, wi_pdf, wi_weight = hemi_wi, hemi_pdf, hemi_weight
    #     else:
    #         raise NotImplementedError()

    #     assert not(dr.any((wi_pdf == 0.0) & (wi_weight != 0.0)))

    #     # wi_rays = si_wide.spawn_ray(si_wide.to_world(wi_local))
    #     wi_rays = self._spawn_offset_ray(si_wide, si_wide.to_world(wi_local))

    #     active = wi_weight > 0.0

    #     # Compute Li for each of the incident directions. For each `Li_ray`, trace `SPP_LI` 
    #     # different MC samples and average them to get the outgoing radiance.
    #     Li, rng_state = self._pathtrace(wi_rays, sampler_rt, rng_state, active)
    #     Li *= wi_weight
    #     return Li, wi_local, active, rng_state


    # TODO
    # def eval_Li_envmap(self, si_wide: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, envmap: mi.Emitter, sampling_method: SamplingMethod, rng_state: int = 0) \
    #     -> tuple[mi.Color3f, mi.Vector3f, Bool, int]:
    #     sampler_rt.seed(rng_state, dr.width(si_wide)); rng_state += 0x00FF_FFFF
    #     uv = sampler_rt.next_2d()

    #     if sampling_method == SamplingMethod.Cosine:
    #         # Sample material
    #         hemi_wi = mi.warp.square_to_cosine_hemisphere(uv)
    #         # max() is needed because this pdf() implementation can return negative values for invalid directions!
    #         hemi_pdf = dr.maximum(0.0, mi.warp.square_to_cosine_hemisphere_pdf(hemi_wi))
    #         hemi_weight = dr.select(hemi_pdf > 0.0, dr.rcp(hemi_pdf), 0.0)

    #         # Evaluate light pdf; pdf is expressed in units of solid angle
    #         em_pdf = self.energy_pmf.eval_pdf(si_wide, hemi_wi)

    #         # Evaluate envmap pdf
    #         ds = dr.zeros(mi.DirectionSample3f, dr.width(si_wide)); ds.d = si_wide.to_world(hemi_wi)
    #         env_pdf = envmap.pdf_direction(dr.zeros(mi.SurfaceInteraction3f), ds)

    #         # MIS weight
    #         mis_weight = dr.select((em_pdf > 0.0) | (env_pdf > 0.0), balance_heuristic_3(hemi_pdf, em_pdf, env_pdf), 1.0)
    #         hemi_weight *= mis_weight
    #         wi_local, wi_pdf, wi_weight = hemi_wi, hemi_pdf, hemi_weight

    #     elif sampling_method == SamplingMethod.Emitter:
    #         # Sample mesh emitters
    #         em_wi, em_weight, em_pdf = self.energy_pmf.sample(si_wide, sampler_rt.next_1d(), uv)

    #         # Evaluate material pdf
    #         hemi_pdf = dr.maximum(0.0, mi.warp.square_to_cosine_hemisphere_pdf(em_wi))

    #         # Evaluate envmap pdf
    #         ds = dr.zeros(mi.DirectionSample3f, dr.width(si_wide)); ds.d = si_wide.to_world(em_wi)
    #         env_pdf = envmap.pdf_direction(dr.zeros(mi.SurfaceInteraction3f), ds)

    #         # Compute MIS weight
    #         mis_weight = dr.select((env_pdf > 0.0) | (hemi_pdf > 0.0), balance_heuristic_3(em_pdf, env_pdf, hemi_pdf), 1.0)
    #         em_weight *= mis_weight 
    #         wi_local, wi_pdf, wi_weight = em_wi, em_pdf, em_weight

    #     elif sampling_method == SamplingMethod.Envmap:
    #         # Sample envmap
    #         env_ds, env_weight = envmap.sample_direction(si_wide, uv)
    #         env_wi = si_wide.to_local(env_ds.d)
    #         env_pdf = env_ds.pdf
    #         env_weight = dr.select(env_pdf > 0.0, dr.rcp(env_pdf), 0.0)

    #         # Evaluate light pdf; pdf is expressed in units of solid angle
    #         em_pdf = self.energy_pmf.eval_pdf(si_wide, env_wi)

    #         # Evaluate material pdf
    #         hemi_pdf = dr.maximum(0.0, mi.warp.square_to_cosine_hemisphere_pdf(env_wi))

    #         # MIS weight
    #         mis_weight = dr.select((hemi_pdf > 0.0) | (em_pdf > 0.0), balance_heuristic_3(env_pdf, hemi_pdf, em_pdf), 1.0)
    #         env_weight *= mis_weight
    #         wi_local, wi_pdf, wi_weight = env_wi, env_pdf, env_weight

    #     else:
    #         raise NotImplementedError()

    #     assert not(dr.any((wi_pdf == 0.0) & (wi_weight != 0.0), axis=None))

    #     wi_rays = si_wide.spawn_ray(si_wide.to_world(wi_local))
    #     active = wi_weight > 0.0

    #     # Compute Li for each of the incident directions. For each `Li_ray`, trace `SPP_LI` 
    #     # different MC samples and average them to get the outgoing radiance.
    #     Li, rng_state = self._pathtrace(wi_rays, sampler_rt, rng_state, active)
    #     Li *= wi_weight

    #     return Li, wi_local, active, rng_state




# from visualizer import plot_rays
# from bsdf_utils import ps_visualize_textures
# import polyscope as ps
# import numpy as np

def compute_loss(
        scene_sampler: SceneSurfaceSampler, 
        radiance_cache: RadianceCache, 
        trainable_bsdf: mi.BSDF, 
        num_points: int = 1,
        num_wi: int = 256, 
        num_wo: int = 1,
        rng_state: int = 0,
        ):
    # if radiance_cache.mi_scene.environment() is not None:
    #     return _compute_loss_envmap(scene_sampler, radiance_cache, trainable_bsdf, num_points, num_wi, num_wo, rng_state)
    
    return _compute_loss_mat(scene_sampler, radiance_cache, trainable_bsdf, num_points, num_wi, num_wo, rng_state)

def _compute_loss_mat(
        scene_sampler: SceneSurfaceSampler, 
        radiance_cache: RadianceCache, 
        trainable_bsdf: mi.BSDF, 
        num_points: int,
        num_wi: int,
        num_wo: int,
        rng_state: int,
        ):
    '''
    Inputs:
        - scene_sampler: SceneSurfaceSampler. The scene sampler draws random points from the scene's surfaces.
        - radiance_cache: RadianceCache. Data structure containing the emissive surface data.
        - trainable_bsdf: mi.BSDF. 
        - num_points: int. The number of surface point samples to use.
        - num_wi: int. The number of incident directions per surface point to use to calculate the radiosity integral.
    Outputs:
        - loss: Float. The scalar loss.
    '''
    loss = Float(0.0)
    dr.enable_grad(loss)

    with dr.suspend_grad():
        sampler: mi.Sampler = mi.load_dict({'type': 'independent'})

        # Sample `NUM_POINTS` different surface points
        si, delta_emitter_sample, delta_emitter_Li, rng_state = scene_sampler.sample(num_points, sampler, rng_state)
        # dr.eval(si, rng_state)

        # Build the "wide" `si`
        #     For each surface point `si`, we should sample `num_wi` incident directions.
        # `wi` can be thought of as a 2D matrix[NUM_POINTS, num_wi] while `si` is an 
        # array[NUM_POINTS]. The latter needs to be broadcasted to match the shape of `wi`, 
        # which is done using the `gather()` (aka "widen") operation.
        #
        #     `si_wide` has the form:          v---- NUM_WI copies ---v
        # [s0, ..., s0, s1, ..., s1,    ...   sN-1,      ...,       sN-1]   (contiguous order)
        si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, num_points), num_wi), dr.ReduceMode.Local)

        # RHS: Evaluate incident directions
        Li_mat, wi_mat, active_mat, rng_state = radiance_cache.eval_Li_mat(si_wide, sampler, rng_state)[:4]

        # TODO: disabled
        # # RHS delta term: Perform ray visibility test from `si` to the delta emitter
        # vis_rays = si.spawn_ray(delta_emitter_sample.d)
        # vis_rays.maxt = delta_emitter_sample.dist
        # emitter_occluded = radiance_cache.mi_scene.ray_test(vis_rays)
        # delta_emitter_Li &= ~emitter_occluded
        # delta_emitter_wi = si.to_local(delta_emitter_sample.d)

        ctx = mi.BSDFContext(mi.TransportMode.Radiance, mi.BSDFFlags.All)
        # Loop through the outgoing directions
        for _ in range(num_wo):
            rhs = dr.zeros(mi.Color3f, num_points)

            # # RHS: compute the delta emitter term
            # with dr.resume_grad():
            #     f_emitter = trainable_bsdf.eval(ctx, si, wo = delta_emitter_wi)
            #     rhs += f_emitter * delta_emitter_Li

            # LHS: evaluate the emissive and outgoing radiances
            Le = radiance_cache.eval_Le(si)
            Lo, active_si, rng_state = radiance_cache.eval_Lo(si, sampler, rng_state)[:3]
            lhs = -Le + Lo

            # RHS: integrate over the incident directions and update the loss
            with dr.resume_grad():
                integrand = Li_mat * trainable_bsdf.eval(ctx, si = si_wide, wo = wi_mat, active = active_mat)
                rhs += dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi
                residuals = dr.select(active_si, dr.squared_norm(lhs - rhs), 0.0)
                loss += 0.5 * dr.mean(residuals) / num_wo

            # Pick new outgoing directions to sample
            sampler.seed(rng_state, num_points); rng_state += 0x00FF_FFFF
            si.wi = mi.warp.square_to_cosine_hemisphere(sampler.next_2d())

            # Update `si_wide` with the new directions
            si_wide.wi = dr.gather(mi.Vector3f, si.wi, dr.repeat(dr.arange(UInt, num_points), num_wi), dr.ReduceMode.Local)

            # if False: #plot or (loss.numpy().item() > 0.01):
            #     err = dr.squared_norm(lhs - rhs).numpy()
            #     idx = np.where(err > 0.01)[0]
            #     bad_si = dr.gather(mi.SurfaceInteraction3f, si, idx)
                
            #     # print(np.histogram(err, bins = np.logspace(-4,2, base=10, num=13)))
            #     print(f"Max error at index {idx} (err = {err[idx]}).")
            #     print(f"lhs = {lhs.numpy()[:,idx]}")
            #     print(f"rhs = {rhs.numpy()[:,idx]}")
            #     print(bad_si)
            #     print(f"RNG: {rng_state}")

            #     ps.init()
            #     ps_visualize_textures(radiance_cache.mi_scene, False)
            #     plot_rays(wi_rays, "wi")
            #     si_cloud = ps.register_point_cloud("si", si.p.numpy().T)
            #     si_cloud.add_vector_quantity("wo", si.to_world(si.wi).numpy().T)
            #     points = ps.register_point_cloud("Bad si", bad_si.p.numpy().T)
            #     points.add_vector_quantity("wo", bad_si.to_world(bad_si.wi).numpy().T)
            #     ps.show()            
    return loss

# TODO
def _compute_loss_mis(
        scene_sampler: SceneSurfaceSampler, 
        radiance_cache: RadianceCache, 
        trainable_bsdf: mi.BSDF, 
        num_points: int,
        num_wi: int,
        num_wo: int,
        rng_state: int,
        ):
    '''
    Inputs:
        - scene_sampler: SceneSurfaceSampler. The scene sampler draws random points from the scene's surfaces.
        - radiance_cache: RadianceCache. Data structure containing the emissive surface data.
        - trainable_bsdf: mi.BSDF. 
        - num_points: int. The number of surface point samples to use.
        - num_wi: int. The number of incident directions per surface point to use to calculate the radiosity integral.
    Outputs:
        - loss: Float. The scalar loss.
    '''
    loss = Float(0.0)
    dr.enable_grad(loss)

    with dr.suspend_grad():
        sampler: mi.Sampler = mi.load_dict({'type': 'independent'})

        # Sample `NUM_POINTS` different surface points
        si, delta_emitter_sample, delta_emitter_Li, rng_state = scene_sampler.sample(num_points, sampler, rng_state)

        # Build the "wide" `si`
        #     For each surface point `si`, we should sample `num_wi` incident directions.
        # `wi` can be thought of as a 2D matrix[NUM_POINTS, num_wi] while `si` is an 
        # array[NUM_POINTS]. The latter needs to be broadcasted to match the shape of `wi`, 
        # which is done using the `gather()` (aka "widen") operation.
        #
        #     `si_wide` has the form:          v---- NUM_WI copies ---v
        # [s0, ..., s0, s1, ..., s1,    ...   sN-1,      ...,       sN-1]   (contiguous order)
        si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, num_points), num_wi), dr.ReduceMode.Local)

        # RHS: Evaluate incident directions
        Li_em, wi_em, active_em, rng_state    = radiance_cache.eval_Li(si_wide, sampler, rng_state, SamplingMethod.Emitter)
        Li_mat, wi_mat, active_mat, rng_state = radiance_cache.eval_Li(si_wide, sampler, rng_state, SamplingMethod.Cosine)

        # RHS delta term: Perform ray visibility test from `si` to the delta emitter
        vis_rays = si.spawn_ray(delta_emitter_sample.d)
        vis_rays.maxt = delta_emitter_sample.dist
        emitter_occluded = radiance_cache.mi_scene.ray_test(vis_rays)
        delta_emitter_Li &= ~emitter_occluded
        delta_emitter_wi = si.to_local(delta_emitter_sample.d)

        ctx = mi.BSDFContext(mi.TransportMode.Radiance, mi.BSDFFlags.All)
        # Loop through the outgoing directions
        for _ in range(num_wo):
            rhs = dr.zeros(mi.Color3f, num_points)

            # RHS: compute the delta emitter term
            with dr.resume_grad():
                f_emitter = trainable_bsdf.eval(ctx, si, wo = delta_emitter_wi)
                rhs += f_emitter * delta_emitter_Li

            # LHS: evaluate the emissive and outgoing radiances
            Le = radiance_cache.eval_Le(si)
            Lo, active_si, rng_state = radiance_cache.eval_Lo(si, sampler, rng_state)
            lhs = -Le + Lo

            # RHS: integrate over the incident directions and update the loss
            with dr.resume_grad():
                integrand = Li_mat * trainable_bsdf.eval(ctx, si = si_wide, wo = wi_mat, active = active_mat) \
                           + Li_em * trainable_bsdf.eval(ctx, si = si_wide, wo = wi_em,  active = active_em)
                rhs += dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi
                residuals = dr.select(active_si, dr.squared_norm(lhs - rhs), 0.0)
                loss += 0.5 * dr.mean(residuals) / num_wo

            # Pick new outgoing directions to sample
            sampler.seed(rng_state, num_points); rng_state += 0x00FF_FFFF
            si.wi = mi.warp.square_to_cosine_hemisphere(sampler.next_2d())

            # Update `si_wide` with the new directions
            si_wide.wi = dr.gather(mi.Vector3f, si.wi, dr.repeat(dr.arange(UInt, num_points), num_wi), dr.ReduceMode.Local)

            # if False: #plot or (loss.numpy().item() > 0.01):
            #     err = dr.squared_norm(lhs - rhs).numpy()
            #     idx = np.where(err > 0.01)[0]
            #     bad_si = dr.gather(mi.SurfaceInteraction3f, si, idx)
                
            #     # print(np.histogram(err, bins = np.logspace(-4,2, base=10, num=13)))
            #     print(f"Max error at index {idx} (err = {err[idx]}).")
            #     print(f"lhs = {lhs.numpy()[:,idx]}")
            #     print(f"rhs = {rhs.numpy()[:,idx]}")
            #     print(bad_si)
            #     print(f"RNG: {rng_state}")

            #     ps.init()
            #     ps_visualize_textures(radiance_cache.mi_scene, False)
            #     plot_rays(wi_rays, "wi")
            #     si_cloud = ps.register_point_cloud("si", si.p.numpy().T)
            #     si_cloud.add_vector_quantity("wo", si.to_world(si.wi).numpy().T)
            #     points = ps.register_point_cloud("Bad si", bad_si.p.numpy().T)
            #     points.add_vector_quantity("wo", bad_si.to_world(bad_si.wi).numpy().T)
            #     ps.show()            
    return loss


# def _compute_loss_envmap(
#         scene_sampler: SceneSurfaceSampler, 
#         radiance_cache: RadianceCacheEM, 
#         trainable_bsdf: mi.BSDF, 
#         num_points: int,
#         num_wi: int, 
#         num_wo: int,
#         rng_state: int,
#         ):
#     '''
#     Inputs:
#         - scene_sampler: SceneSurfaceSampler. The scene sampler draws random points from the scene's surfaces.
#         - radiance_cache: RadianceCache. Data structure containing the emissive surface data.
#         - trainable_bsdf: mi.BSDF. 
#         - num_points: int. The number of surface point samples to use.
#         - num_wi: int. The number of incident directions per surface point to use to calculate the radiosity integral.
#     Outputs:
#         - loss: Float. The scalar loss.
#     '''
#     loss = Float(0.0)
#     dr.enable_grad(loss)
#     envmap = radiance_cache.mi_scene.environment()

#     with dr.suspend_grad():
#         sampler: mi.Sampler = mi.load_dict({'type': 'independent'})

#         # Sample `NUM_POINTS` different surface points
#         si, delta_emitter_sample, delta_emitter_Li, rng_state = scene_sampler.sample(num_points, sampler, rng_state)

#         # Build the "wide" `si`
#         si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, num_points), num_wi), dr.ReduceMode.Local)

#         # RHS: Evaluate incident directions
#         Li_em, wi_em, active_em, rng_state    = radiance_cache.eval_Li_envmap(si_wide, sampler, envmap, SamplingMethod.Emitter, rng_state)
#         Li_mat, wi_mat, active_mat, rng_state = radiance_cache.eval_Li_envmap(si_wide, sampler, envmap, SamplingMethod.Cosine, rng_state)
#         Li_env, wi_env, active_env, rng_state = radiance_cache.eval_Li_envmap(si_wide, sampler, envmap, SamplingMethod.Envmap, rng_state)

#         # RHS delta term: Perform ray visibility test from `si` to the delta emitter
#         vis_rays = si.spawn_ray(delta_emitter_sample.d)
#         vis_rays.maxt = delta_emitter_sample.dist
#         emitter_occluded = radiance_cache.mi_scene.ray_test(vis_rays)
#         delta_emitter_Li &= ~emitter_occluded
#         delta_emitter_wi = si.to_local(delta_emitter_sample.d)

#         ctx = mi.BSDFContext(mi.TransportMode.Radiance, mi.BSDFFlags.All)
#         # Loop through the outgoing directions
#         for _ in range(num_wo):
#             rhs = dr.zeros(mi.Color3f, num_points)

#             # RHS: compute the delta emitter term
#             with dr.resume_grad():
#                 f_emitter = trainable_bsdf.eval(ctx, si, wo = delta_emitter_wi)
#                 rhs += f_emitter * delta_emitter_Li

#             # LHS: evaluate the emissive and outgoing radiances
#             Le = radiance_cache.eval_Le(si)
#             Lo, active_si, rng_state = radiance_cache.eval_Lo(si, sampler, rng_state)
#             lhs = -Le + Lo

#             # RHS: integrate over the incident directions and update the loss
#             with dr.resume_grad():
#                 integrand = Li_mat  * trainable_bsdf.eval(ctx, si = si_wide, wo = wi_mat, active = active_mat) \
#                            + Li_em  * trainable_bsdf.eval(ctx, si = si_wide, wo = wi_em,  active = active_em) \
#                            + Li_env * trainable_bsdf.eval(ctx, si = si_wide, wo = wi_env,  active = active_env)
#                 rhs += dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi
#                 residuals = dr.select(active_si, dr.squared_norm(lhs - rhs), 0.0)
#                 loss += 0.5 * dr.mean(residuals) / num_wo

#             # Pick new outgoing directions to sample
#             sampler.seed(rng_state, num_points); rng_state += 0x00FF_FFFF
#             si.wi = mi.warp.square_to_cosine_hemisphere(sampler.next_2d())

#             # Update `si_wide` with the new directions
#             si_wide.wi = dr.gather(mi.Vector3f, si.wi, dr.repeat(dr.arange(UInt, num_points), num_wi), dr.ReduceMode.Local)

#     return loss