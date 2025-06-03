import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-gs/build/python')

import drjit as dr
import mitsuba as mi
from drjit.auto import Float, UInt, Bool
from scripts.radiosity.sh_fitting import get_sh_count, fit_sh_on_scene
from scripts.radiosity.surface_sampler import SceneSurfaceSampler
from scripts.radiosity.vertex_bsdf import VertexBSDF
from enum import Enum
from scripts.radiosity.tmp import EnergyPMFv2
from volprim.integrators.volprim_rf_pdf import EllipsoidRfPdfIntegrator

class SamplingMethod(Enum):
    Emitter = 0
    Envmap = 1
    Cosine = 2
    BSDF = 3

def sph_to_dir(theta, phi):
        st, ct = dr.sincos(theta)
        sp, cp = dr.sincos(phi)
        wi = mi.Vector3f(
            st * cp,
            st * sp,
            ct)
        return wi

def dir_to_sph(v):
    '''
    Returns (theta, phi)
    '''
    theta = dr.safe_acos(v.z)
    phi = dr.atan2(v.y, v.x)
    return mi.Point2f(theta, phi)

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
                'extent': 3.0,
                # 'shell': 'uv_sphere',
                'filename': pcd_path
            },
            'integrator': {
                'type': 'volprim_rf',
                'max_depth': -1, # TODO
                # 'rr_depth':  -1, # TODO
                'kernel_type': 'gaussian',
                # Assume that the GS scene outputs *linear* RGB radiance data
                'srgb_primitives': False,
            }
        }

        self.mi_scene = mi_scene
        self.gs_scene: mi.Scene = mi.load_dict(scene_dict)
        self.integrator: mi.ad.common.ADIntegrator = self.gs_scene.integrator()
        self.energy_pmf = EnergyPMFv2(self.gs_scene)
        self.max_extent = dr.max(self.gs_scene.bbox().extents())

        self.pdf_integrator = EllipsoidRfPdfIntegrator()
        self.pdf_integrator.set_pmf(self.energy_pmf.pmf)
        self.pdf_integrator.use_gaussian_density = self.energy_pmf.use_gaussian_density


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
        L, active = self.integrator.sample(dr.ADMode.Primal, self.gs_scene, sampler_rt, rays, δL = None, state_in = 0.0, active = active)[:2]
        L = dr.select(active, L, dr.zeros(mi.Color3f))
        return L, rng_state
    
    def _spawn_offset_ray(self, si: mi.SurfaceInteraction3f, d: mi.Vector3f):
        ray = si.spawn_ray(d)
        # `intersect_preliminary` method
        dr.eval(ray)
        # find offset using gt_geometry raytrace
        first_hit = self.mi_scene.ray_intersect_preliminary(ray)
        offset = dr.select(
            first_hit.is_valid(), 
            0.5 * first_hit.t,
            0.1 * self.max_extent)
        dr.eval(ray, offset)
        ray.o += offset * ray.d
        return ray
    
    def _get_ray_offset(self, ray_: mi.Ray3f):
        ray = mi.Ray3f(ray_)
        # `intersect_preliminary` method
        dr.eval(ray)
        # find offset using gt_geometry raytrace
        first_hit = self.mi_scene.ray_intersect_preliminary(ray)
        offset = dr.select(
            first_hit.is_valid(), 
            0.5 * first_hit.t,
            # 0.0)
            0.1 * self.max_extent)
        dr.eval(ray, offset)
        return offset
        # return dr.clamp(offset - 1e-2, 0.0, dr.inf)

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
        Lo_rays = self._spawn_offset_ray(si, wo_world)
        Lo_rays.d = -Lo_rays.d

        active = True   
        # # The ray that's spawned from `si` should intersect `si.shape` again. Due to RayEpsilons and 
        # # spawn offsets, there exist edge cases where this does not occur: for example, when `si` lies
        # # on the boundary edge of a rectangular plane. If this occurs, we should omit this `si` from 
        # # the loss calculation.
        # active = (self.mi_scene.ray_intersect_preliminary(Lo_rays).shape == si.shape)
        # ^ TODO: disabled

        # Pathtrace along `-wo` to get the radiance when looking at `A`.
        Lo, rng_state = self._pathtrace(Lo_rays, sampler_rt, rng_state, active)
        return Lo, active, rng_state, Lo_rays

    def _render_hemisphere(self, si_scalar: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, N: int, offset: float = 0.0):
        theta = dr.linspace(Float, 0.0, 0.5 * dr.pi, N)
        phi = dr.linspace(Float, 0.0, dr.two_pi, 4 * N)
        pp, tt = dr.meshgrid(phi, theta)

        wi = sph_to_dir(tt, pp)
        wi_world = si_scalar.to_world(wi)

        rays = si_scalar.spawn_ray(wi_world)
        rays.o += offset * rays.d
        L = self._pathtrace(rays, sampler_rt, rng_state=0)[0]
        image_out = mi.TensorXf(dr.ravel(L), shape=(N, 4*N, 3))
        return image_out, rays

    def _render_hemisphere_auto_offset(self, si_scalar: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, N: int):
        theta = dr.linspace(Float, 0.0, 0.5 * dr.pi, N)
        phi = dr.linspace(Float, 0.0, dr.two_pi, 4 * N)
        pp, tt = dr.meshgrid(phi, theta)

        wi = sph_to_dir(tt, pp)
        wi_world = si_scalar.to_world(wi)

        rays = self._spawn_offset_ray(si_scalar, wi_world)

        # rays = si_scalar.spawn_ray(wi_world)
        # # rays = mi.Ray3f(si_scalar.p, wi_world)

        # dr.eval(rays)
        # # find offset using gt_geometry raytrace
        # first_hit = self.mi_scene.ray_intersect_preliminary(rays)
        # offset = dr.select(
        #     first_hit.is_valid(), 
        #     0.5 * first_hit.t,
        #     0.1 * self.max_extent)
        # dr.eval(rays, offset)

        # rays.o += offset * rays.d
        L = self._pathtrace(rays, sampler_rt, rng_state=0)[0]
        image_out = mi.TensorXf(dr.ravel(L), shape=(N, 4*N, 3))

        dtheta = 0.5 * dr.pi / (N-1)
        dphi = dr.two_pi / (4*N-1)
        quad_weights = dr.sin(tt) * dtheta * dphi
        I_R = dr.sum(L.x * quad_weights, axis=None)
        I_G = dr.sum(L.y * quad_weights, axis=None)
        I_B = dr.sum(L.z * quad_weights, axis=None)
        return image_out, rays, mi.Color3f(I_R, I_G, I_B)


    def eval_Li_uniform(self, si_wide: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, rng_state: int = 0) \
        -> tuple[mi.Color3f, int]:
        '''
        Inputs:
            - bsdf: VertexBSDF. The BSDF associated with the surface mesh.
            - si_wide: SurfaceInteraction3f. Widened array of surface sample points of size [#si * #wi,].
            - sampler: Sampler. The pseudo-random number generator.
            - rng_state: int. The RNG seed.
        Outputs: 
            - Li: mi.Color3f. Flattened array of incident radiances of size [#si * #wi,]. The data 
            is in contiguous order, i.e. the first #wi entries belong to si0, and so on.
            - wi_local: mi.Vector3f. Flattened array of incident directions of size [#si * #wi,].
            - rng_state: int. RNG seed for the next operation involving random numbers.
        '''
        sampler_rt.seed(rng_state, dr.width(si_wide)); rng_state += 0x00FF_FFFF
        uv = sampler_rt.next_2d()

        hemi_wi = mi.warp.square_to_uniform_hemisphere(uv)
        hemi_pdf = mi.warp.square_to_uniform_hemisphere_pdf(hemi_wi)
        hemi_weight = dr.select(hemi_pdf > 0.0, dr.rcp(hemi_pdf), 0.0)
        wi_local, wi_pdf, wi_weight = hemi_wi, hemi_pdf, hemi_weight

        assert not(dr.any((wi_pdf == 0.0) & (wi_weight != 0.0)))

        wi_rays = self._spawn_offset_ray(si_wide, si_wide.to_world(wi_local))
        active = (wi_weight > 0.0) & (mi.Frame3f.cos_theta(wi_local) >= 0.0)

        # Compute Li for each of the incident directions
        Li, rng_state = self._pathtrace(wi_rays, sampler_rt, rng_state, active)
        Li *= wi_weight
        return Li, wi_local, active, rng_state, wi_rays


    def eval_Li_mat(self, bsdf: VertexBSDF, si_wide: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, rng_state: int = 0) \
        -> tuple[mi.Color3f, int]:
        '''
        Inputs:
            - bsdf: VertexBSDF. The BSDF associated with the surface mesh.
            - si_wide: SurfaceInteraction3f. Widened array of surface sample points of size [#si * #wi,].
            - sampler: Sampler. The pseudo-random number generator.
            - rng_state: int. The RNG seed.
        Outputs: 
            - Li: mi.Color3f. Flattened array of incident radiances of size [#si * #wi,]. The data 
            is in contiguous order, i.e. the first #wi entries belong to si0, and so on.
            - wi_local: mi.Vector3f. Flattened array of incident directions of size [#si * #wi,].
            - rng_state: int. RNG seed for the next operation involving random numbers.
        '''
        sampler_rt.seed(rng_state, dr.width(si_wide)); rng_state += 0x00FF_FFFF
        uv = sampler_rt.next_2d()
        w = sampler_rt.next_1d()

        ctx = mi.BSDFContext()
        bs, _ = bsdf.sample(ctx, si_wide, w, uv)
        hemi_wi, hemi_pdf = bs.wo, bs.pdf
        hemi_weight = dr.select(hemi_pdf > 0.0, dr.rcp(hemi_pdf), 0.0)
        wi_local, wi_pdf, wi_weight = hemi_wi, hemi_pdf, hemi_weight

        assert not(dr.any((wi_pdf == 0.0) & (wi_weight != 0.0)))

        wi_rays = self._spawn_offset_ray(si_wide, si_wide.to_world(wi_local))
        active = (wi_weight > 0.0) & (mi.Frame3f.cos_theta(wi_local) >= 0.0)

        # Compute Li for each of the incident directions
        Li, rng_state = self._pathtrace(wi_rays, sampler_rt, rng_state, active)
        Li *= wi_weight
        return Li, wi_local, active, rng_state, wi_rays


    def eval_Li(self, bsdf: VertexBSDF, si_wide: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, rng_state: int = 0, sampling_method: SamplingMethod = SamplingMethod.Cosine) \
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
        w = sampler_rt.next_1d()
        ctx = mi.BSDFContext()

        if sampling_method == SamplingMethod.Emitter:
            # light_pdf is expressed in units of solid angle
            em_wi_world, em_weight, em_pdf = self.energy_pmf.sample(si_wide, sampler_rt)
            em_wi_local = si_wide.to_local(em_wi_world)

            # Evaluate material pdf and MIS weight
            mat_pdf = bsdf.pdf(ctx, si_wide, wo = em_wi_local)
            mis_weight = dr.select((mat_pdf > 0.0) & ~dr.isinf(em_pdf), balance_heuristic(em_pdf, mat_pdf), 1.0)
            em_weight *= mis_weight
            wi_local, wi_world, wi_pdf, wi_weight = em_wi_local, em_wi_world, em_pdf, em_weight
        elif sampling_method == SamplingMethod.Cosine:
            bs = bsdf.sample(ctx, si_wide, w, uv)[0]
            mat_wi_local, mat_pdf = bs.wo, bs.pdf
            mat_wi_world = si_wide.to_world(mat_wi_local)
            mat_weight = dr.select(mat_pdf > 0.0, dr.rcp(mat_pdf), 0.0)

            # Evaluate light pdf and MIS weight
            # light_pdf is expressed in units of solid angle
            em_pdf = self.energy_pmf.eval_pdf(si_wide, mat_wi_world, sampler_rt)
            mis_weight = dr.select(em_pdf > 0.0, balance_heuristic(mat_pdf, em_pdf), 1.0)
            mat_weight *= mis_weight
            wi_local, wi_world, wi_pdf, wi_weight = mat_wi_local, mat_wi_world, mat_pdf, mat_weight
        else:
            raise NotImplementedError()

        assert not(dr.any((wi_pdf == 0.0) & (wi_weight != 0.0)))

        wi_rays = self._spawn_offset_ray(si_wide, wi_world)
        active = (wi_weight > 0.0) & (mi.Frame3f.cos_theta(wi_local) >= 0.0)

        # Compute Li for each of the incident directions. For each `Li_ray`, trace `SPP_LI` 
        # different MC samples and average them to get the outgoing radiance.
        Li, rng_state = self._pathtrace(wi_rays, sampler_rt, rng_state, active)
        Li *= wi_weight
        return Li, wi_local, active, rng_state


    def eval_Li_pdf(self, bsdf: VertexBSDF, si_wide: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, rng_state: int = 0, sampling_method: SamplingMethod = SamplingMethod.Cosine) \
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
        num_rays = dr.width(si_wide)
        sampler_rt.seed(rng_state, num_rays); rng_state += 0x00FF_FFFF
        ctx = mi.BSDFContext()

        if sampling_method == SamplingMethod.Emitter:
            # light_pdf is expressed in units of solid angle
            wi_world, active = self.energy_pmf.sample_no_pdf(si_wide, sampler_rt)
            wi_local = si_wide.to_local(wi_world)
        else:
            uv = sampler_rt.next_2d()
            w = sampler_rt.next_1d()
            bs = bsdf.sample(ctx, si_wide, w, uv)[0]
            wi_local, wi_pdf = bs.wo, bs.pdf
            wi_world = si_wide.to_world(wi_local)
            active = wi_pdf > 0.0

        active &= mi.Frame3f.cos_theta(wi_local) >= 0.0
        wi_rays = si_wide.spawn_ray(wi_world)
        t_start = self._get_ray_offset(wi_rays)

        # Compute Li for each of the incident directions. For each `Li_ray`, trace `SPP_LI` 
        # different MC samples and average them to get the outgoing radiance.
        sampler_rt.seed(rng_state, num_rays); rng_state += 0x00FF_FFFF
        Li, active, aovs = self.pdf_integrator.sample(dr.ADMode.Primal, self.gs_scene, sampler_rt, wi_rays, δL = None, state_in = 0.0, active = active, t_start = t_start)[:3]
        em_pdf = aovs[0]

        # Account for emitter pdf in integrator weight using MIS
        if sampling_method == SamplingMethod.Emitter:
            em_weight = dr.select(active & (em_pdf > 0.0), dr.rcp(em_pdf), 0.0)
            # Evaluate material pdf and MIS weight
            mat_pdf = bsdf.pdf(ctx, si_wide, wo = wi_local)
            # mis_weight = dr.select((mat_pdf > 0.0) & ~dr.isinf(em_pdf), balance_heuristic(em_pdf, mat_pdf), 1.0)
            mis_weight = dr.select(mat_pdf > 0.0, balance_heuristic(em_pdf, mat_pdf), 1.0)
            em_weight *= mis_weight
            wi_weight = em_weight
        else:
            # Evaluate light pdf and MIS weight
            mat_pdf = wi_pdf
            mat_weight = dr.select(mat_pdf > 0.0, dr.rcp(mat_pdf), 0.0)
            mis_weight = dr.select(em_pdf  > 0.0, balance_heuristic(mat_pdf, em_pdf), 1.0)
            mat_weight *= mis_weight
            wi_weight = mat_weight

        Li *= wi_weight
        return Li, wi_local, active, rng_state


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
    # return _compute_loss_mat(scene_sampler, radiance_cache, trainable_bsdf, num_points, num_wi, num_wo, rng_state)
    return _compute_loss_mis(scene_sampler, radiance_cache, trainable_bsdf, num_points, num_wi, num_wo, rng_state)

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
        si, rng_state = scene_sampler.sample(num_points, sampler, rng_state)

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
        Li_mat, wi_mat, active_mat, rng_state = radiance_cache.eval_Li_mat(trainable_bsdf, si_wide, sampler, rng_state)[:4]

        ctx = mi.BSDFContext()
        # Loop through the outgoing directions
        for _ in range(num_wo):
            rhs = dr.zeros(mi.Color3f, num_points)

            # LHS: evaluate the emissive and outgoing radiances
            Le = radiance_cache.eval_Le(si)
            Lo, active_si, rng_state = radiance_cache.eval_Lo(si, sampler, rng_state)[:3]
            lhs = -Le + Lo

            # RHS: integrate over the incident directions and update the loss
            with dr.resume_grad():
                integrand = Li_mat * trainable_bsdf.eval(ctx, si_wide, wo = wi_mat, active = active_mat)
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
        si, rng_state = scene_sampler.sample(num_points, sampler, rng_state)

        # Build the "wide" `si`
        #     For each surface point `si`, we should sample `num_wi` incident directions.
        # `wi` can be thought of as a 2D matrix[NUM_POINTS, num_wi] while `si` is an 
        # array[NUM_POINTS]. The latter needs to be broadcasted to match the shape of `wi`, 
        # which is done using the `gather()` (aka "widen") operation.
        #
        #     `si_wide` has the form:          v---- NUM_WI copies ---v
        # [s0, ..., s0, s1, ..., s1,    ...   sN-1,      ...,       sN-1]   (contiguous order)
        si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, num_points), num_wi), dr.ReduceMode.Local)

        # # RHS: Evaluate incident directions
        Li_em,  wi_em,  active_em,  rng_state = radiance_cache.eval_Li_pdf(trainable_bsdf, si_wide, sampler, rng_state, SamplingMethod.Emitter)
        Li_mat, wi_mat, active_mat, rng_state = radiance_cache.eval_Li_pdf(trainable_bsdf, si_wide, sampler, rng_state, SamplingMethod.Cosine)
        dr.eval(Li_em, wi_em, active_em, Li_mat, wi_mat, active_mat)

        ctx = mi.BSDFContext()
        # Loop through the outgoing directions
        for _ in range(num_wo):
            rhs = dr.zeros(mi.Color3f, num_points)

            # LHS: evaluate the emissive and outgoing radiances
            Le = radiance_cache.eval_Le(si)
            Lo, active_si, rng_state = radiance_cache.eval_Lo(si, sampler, rng_state)[:3]
            lhs = -Le + Lo

            # RHS: integrate over the incident directions and update the loss
            with dr.resume_grad():
                integrand = Li_mat * trainable_bsdf.eval(ctx, si_wide, wo = wi_mat, active = active_mat) \
                           + Li_em * trainable_bsdf.eval(ctx, si_wide, wo = wi_em,  active = active_em)
                rhs += dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi
                residuals = dr.select(active_si, dr.squared_norm(lhs - rhs), 0.0)
                loss += 0.5 * dr.mean(residuals) / num_wo

            # Pick new outgoing directions to sample
            # TODO: i think we can get rid of the re-seeding? the sampler should have the same width throughout this phase
            sampler.seed(rng_state, num_points); rng_state += 0x00FF_FFFF
            si.wi = mi.warp.square_to_cosine_hemisphere(sampler.next_2d())

            # Update `si_wide` with the new directions
            si_wide.wi = dr.gather(mi.Vector3f, si.wi, dr.repeat(dr.arange(UInt, num_points), num_wi), dr.ReduceMode.Local)

    return loss


def _compute_loss_uniform(
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
        si, rng_state = scene_sampler.sample(num_points, sampler, rng_state)
        si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, num_points), num_wi), dr.ReduceMode.Local)

        # RHS: Evaluate incident directions
        Li_mat, wi_mat, active_mat, rng_state = radiance_cache.eval_Li_uniform(si_wide, sampler, rng_state)[:4]

        ctx = mi.BSDFContext()
        # Loop through the outgoing directions
        for _ in range(num_wo):
            rhs = dr.zeros(mi.Color3f, num_points)
            # LHS: evaluate the emissive and outgoing radiances
            Le = radiance_cache.eval_Le(si)
            Lo, active_si, rng_state = radiance_cache.eval_Lo(si, sampler, rng_state)[:3]
            lhs = -Le + Lo

            # RHS: integrate over the incident directions and update the loss
            with dr.resume_grad():
                integrand = Li_mat * trainable_bsdf.eval(ctx, si_wide, wo = wi_mat, active = active_mat)
                rhs += dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi
                residuals = dr.select(active_si, dr.squared_norm(lhs - rhs), 0.0)
                loss += 0.5 * dr.mean(residuals) / num_wo

            # Pick new outgoing directions to sample
            sampler.seed(rng_state, num_points); rng_state += 0x00FF_FFFF
            si.wi = mi.warp.square_to_cosine_hemisphere(sampler.next_2d())

            # Update `si_wide` with the new directions
            si_wide.wi = dr.gather(mi.Vector3f, si.wi, dr.repeat(dr.arange(UInt, num_points), num_wi), dr.ReduceMode.Local)
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
#         si, rng_state = scene_sampler.sample(num_points, sampler, rng_state)

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




def render_hemisphere_rt(scene: mi.Scene, si_scalar: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, N: int, spp: int, rng_state: int = 0):
    theta = dr.linspace(Float, 0.0, 0.5 * dr.pi, N)
    phi = dr.linspace(Float, 0.0, dr.two_pi, 4 * N)
    pp, tt = dr.meshgrid(phi, theta)
    st, ct = dr.sincos(tt)
    sp, cp = dr.sincos(pp)
    wi = mi.Vector3f(
        st * cp,
        st * sp,
        ct)

    wi_world = si_scalar.to_world(wi)
    rays = si_scalar.spawn_ray(wi_world)
    rays.o += 1e-3 * rays.d

    num_rays = dr.width(rays)
    rays_flat = dr.gather(mi.Ray3f, rays, dr.repeat(dr.arange(UInt, num_rays), spp))
    sampler_rt.seed(rng_state, num_rays * spp)
    colors = scene.integrator().sample(scene, sampler_rt, rays_flat)[0]
    L = dr.block_reduce(dr.ReduceOp.Add, colors, block_size = spp) / spp
    image_out = mi.TensorXf(dr.ravel(L), shape=(N, 4*N, 3))

    dtheta = 0.5 * dr.pi / (N-1)
    dphi = dr.two_pi / (4*N-1)
    quad_weights = st * dtheta * dphi
    I_R = dr.sum(L.x * quad_weights, axis=None)
    I_G = dr.sum(L.y * quad_weights, axis=None)
    I_B = dr.sum(L.z * quad_weights, axis=None)
    return image_out, rays, mi.Color3f(I_R, I_G, I_B)