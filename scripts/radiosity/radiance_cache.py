import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-gs/build/python')

import drjit as dr
import mitsuba as mi
from drjit.auto import Float, UInt, Bool
from scripts.radiosity.sh_fitting import get_sh_count, fit_sh_on_scene
from scripts.radiosity.surface_sampler import SceneSurfaceSampler
from scripts.radiosity.vertex_bsdf import VertexBSDF
from enum import Enum
from scripts.radiosity.energy_pmf import EnergyPMF
from volprim.integrators.volprim_rf_pdf import EllipsoidRfPdfIntegrator

from scripts.restir.reservoir import MultiReservoirVector3f

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
    phi[phi<0.0] += dr.two_pi
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
                'max_depth': -1,
                # 'rr_depth':  -1,
                'kernel_type': 'gaussian',
                # Assume that the GS scene outputs *linear* RGB radiance data
                'srgb_primitives': False,
            }
        }

        self.mi_scene = mi_scene
        self.gs_scene: mi.Scene = mi.load_dict(scene_dict)
        self.integrator: mi.ad.common.ADIntegrator = self.gs_scene.integrator()
        self.energy_pmf = EnergyPMF(self.gs_scene)
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

    def _get_ray_and_offset(self, si: mi.SurfaceInteraction3f, d: mi.Vector3f) -> tuple[mi.Ray3f, mi.Float]:
        ray_ = si.spawn_ray(d)
        ray = mi.Ray3f(ray_)
        # `intersect_preliminary` method
        dr.eval(ray)
        # find offset using gt_geometry raytrace
        first_hit = self.mi_scene.ray_intersect_preliminary(ray)
        offset = dr.select(
            first_hit.is_valid(), 
            0.5 * first_hit.t,
            0.1 * self.max_extent)
        dr.eval(ray, offset)
        return ray_, offset

    def eval_Le(self, si: mi.SurfaceInteraction3f) -> mi.Color3f:
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
        # Lo, rng_state = self._pathtrace(Lo_rays, sampler_rt, rng_state, active)

        Lo, active = self.integrator.sample(dr.ADMode.Primal, self.gs_scene, sampler_rt, Lo_rays, δL = None, state_in = 0.0, active = active)[:2]

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
        L = self._pathtrace(rays, sampler_rt, rng_state=0)[0]
        image_out = mi.TensorXf(dr.ravel(L), shape=(N, 4*N, 3))

        dtheta = 0.5 * dr.pi / (N-1)
        dphi = dr.two_pi / (4*N-1)
        quad_weights = dr.sin(tt) * dtheta * dphi
        I_R = dr.sum(L.x * quad_weights, axis=None)
        I_G = dr.sum(L.y * quad_weights, axis=None)
        I_B = dr.sum(L.z * quad_weights, axis=None)
        return image_out, rays, mi.Color3f(I_R, I_G, I_B)

    def _render_hemisphere_auto_offset_inverse(self, si_scalar: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, N: int):
        theta = dr.linspace(Float, 0.0, 0.5 * dr.pi, N)
        phi = dr.linspace(Float, 0.0, dr.two_pi, 4 * N)
        pp, tt = dr.meshgrid(phi, theta)

        wi = sph_to_dir(tt, pp)
        wi_world = si_scalar.to_world(wi)

        rays = self._spawn_offset_ray(si_scalar, wi_world)
        rays.d = -rays.d

        # rays = mi.Ray3f(si_scalar.p + 0.5 * wi_world, -wi_world)
        dr.eval(rays)
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
        bs = bsdf.sample(ctx, si_wide, w, uv)[0]
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

        # Compute Li for each of the incident directions
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
            # wi_local = mi.warp.square_to_cosine_hemisphere(uv)
            # wi_pdf = mi.warp.square_to_cosine_hemisphere_pdf(wi_local)
            wi_world = si_wide.to_world(wi_local)
            active = wi_pdf > 0.0

        active &= mi.Frame3f.cos_theta(wi_local) >= 0.0
        wi_rays = si_wide.spawn_ray(wi_world)
        t_start = self._get_ray_offset(wi_rays)

        # Compute Li for each of the incident directions. For each `Li_ray`, trace `SPP_LI` 
        # different MC samples and average them to get the outgoing radiance.
        sampler_rt.seed(rng_state, num_rays); rng_state += 0x00FF_FFFF  # TODO: unnecessary!
        Li, _, aovs = self.pdf_integrator.sample(dr.ADMode.Primal, self.gs_scene, sampler_rt, wi_rays, δL = None, state_in = 0.0, active = active, t_start = t_start)[:3]
        em_pdf = aovs[0]

        # Account for emitter pdf in integrator weight using MIS
        if sampling_method == SamplingMethod.Emitter:
            em_weight = dr.select(active & (em_pdf > 0.0), dr.rcp(em_pdf), 0.0)
            # Evaluate material pdf and MIS weight
            mat_pdf = bsdf.pdf(ctx, si_wide, wo = wi_local)
            # mat_pdf = dr.select(mi.Frame3f.cos_theta(wi_local) >= 0.0, mi.warp.square_to_cosine_hemisphere_pdf(wi_local), 0.0)
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

    def eval_Li_pdf_RT(self, bsdf: VertexBSDF, si_wide: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, rng_state: int = 0, sampling_method: SamplingMethod = SamplingMethod.Cosine, spp: int = 64) \
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
            # # light_pdf is expressed in units of solid angle
            ds = self.mi_scene.sample_emitter_direction(si_wide, sampler_rt.next_2d())[0]
            wi_world = ds.d
            wi_local = si_wide.to_local(wi_world)
            em_pdf = ds.pdf
            active = mi.Frame3f.cos_theta(wi_local) >= 0.0
            em_weight = dr.select(active & (em_pdf > 0.0), dr.rcp(em_pdf), 0.0)

            # Evaluate material pdf
            mat_pdf = bsdf.pdf(ctx, si_wide, wo = wi_local, active = active)

            # Evaluate MIS weight
            mis_weight = dr.select(mat_pdf > 0.0, balance_heuristic(em_pdf, mat_pdf), 1.0)
            em_weight *= mis_weight
            wi_weight = em_weight
        else:
            uv = sampler_rt.next_2d()
            w  = sampler_rt.next_1d()
            bs = bsdf.sample(ctx, si_wide, w, uv)[0]
            wi_local = bs.wo
            mat_pdf  = bs.pdf
            wi_world = si_wide.to_world(wi_local)
            active   = (mat_pdf > 0.0) & (mi.Frame3f.cos_theta(wi_local) >= 0.0)
            mat_weight = dr.select(mat_pdf > 0.0, dr.rcp(mat_pdf), 0.0)

            # Evaluate light pdf
            em_ray = si_wide.spawn_ray(wi_world)
            em_si = self.mi_scene.ray_intersect(em_ray, active)
            active_bsdf = em_si.emitter(self.mi_scene) != None
            ds = mi.DirectionSample3f(self.mi_scene, em_si, si_wide)
            em_pdf = self.mi_scene.pdf_emitter_direction(si_wide, ds, active_bsdf)

            # Evaluate MIS weight
            mis_weight = dr.select(em_pdf  > 0.0, balance_heuristic(mat_pdf, em_pdf), 1.0)
            mat_weight *= mis_weight
            wi_weight = mat_weight

        wi_rays = si_wide.spawn_ray(wi_world)
        rays_wide = dr.gather(mi.Ray3f, wi_rays, dr.repeat(dr.arange(mi.UInt, dr.width(wi_rays)), spp))
        active_wide = dr.repeat(active, spp)
        sampler_rt.seed(rng_state, dr.width(rays_wide)); rng_state += 0x0FF_FFFF
        integrator: mi.Integrator = self.mi_scene.integrator()
        Li = dr.block_sum(
            integrator.sample(self.mi_scene, sampler_rt, rays_wide, active = active_wide)[0],
            spp) / spp
        Li *= wi_weight
        dr.eval(Li)
        return Li, wi_local, active, rng_state


    def eval_Li_pdf_RT_unweighted(self, bsdf: VertexBSDF, si_wide: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, rng_state: int = 0, sampling_method: SamplingMethod = SamplingMethod.Cosine, spp: int = 64) \
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
            # # light_pdf is expressed in units of solid angle
            ds = self.mi_scene.sample_emitter_direction(si_wide, sampler_rt.next_2d())[0]
            wi_world = ds.d
            wi_local = si_wide.to_local(wi_world)
            em_pdf = ds.pdf
            active = mi.Frame3f.cos_theta(wi_local) >= 0.0
            em_weight = dr.select(active & (em_pdf > 0.0), dr.rcp(em_pdf), 0.0)

            # Evaluate material pdf
            mat_pdf = bsdf.pdf(ctx, si_wide, wo = wi_local, active = active)

            # Evaluate MIS weight
            mis_weight = dr.select(mat_pdf > 0.0, balance_heuristic(em_pdf, mat_pdf), 1.0)
            em_weight *= mis_weight
            wi_weight = em_weight
        else:
            uv = sampler_rt.next_2d()
            w  = sampler_rt.next_1d()
            bs = bsdf.sample(ctx, si_wide, w, uv)[0]
            wi_local = bs.wo
            mat_pdf  = bs.pdf
            wi_world = si_wide.to_world(wi_local)
            active   = (mat_pdf > 0.0) & (mi.Frame3f.cos_theta(wi_local) >= 0.0)
            mat_weight = dr.select(mat_pdf > 0.0, dr.rcp(mat_pdf), 0.0)

            # Evaluate light pdf
            em_ray = si_wide.spawn_ray(wi_world)
            em_si = self.mi_scene.ray_intersect(em_ray, active)
            active_bsdf = em_si.emitter(self.mi_scene) != None
            ds = mi.DirectionSample3f(self.mi_scene, em_si, si_wide)
            em_pdf = self.mi_scene.pdf_emitter_direction(si_wide, ds, active_bsdf)

            # Evaluate MIS weight
            mis_weight = dr.select(em_pdf  > 0.0, balance_heuristic(mat_pdf, em_pdf), 1.0)
            mat_weight *= mis_weight
            wi_weight = mat_weight

        wi_rays = si_wide.spawn_ray(wi_world)
        rays_wide = dr.gather(mi.Ray3f, wi_rays, dr.repeat(dr.arange(mi.UInt, dr.width(wi_rays)), spp))
        active_wide = dr.repeat(active, spp)
        sampler_rt.seed(rng_state, dr.width(rays_wide)); rng_state += 0x0FF_FFFF
        integrator: mi.Integrator = self.mi_scene.integrator()
        Li = dr.block_sum(
            integrator.sample(self.mi_scene, sampler_rt, rays_wide, active = active_wide)[0],
            spp) / spp
        dr.eval(Li)
        return Li, wi_local, active, rng_state, wi_weight
    

    def eval_Lo_RT(self, si: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, rng_state: int = 0, spp: int = 64)  -> tuple[mi.Color3f, Bool, int]:
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
        Lo_rays = si.spawn_ray(si.to_world(si.wi))
        Lo_rays.o += 0.01 * Lo_rays.d
        Lo_rays.d = -Lo_rays.d

        active = (self.mi_scene.ray_intersect_preliminary(Lo_rays).shape == si.shape)
        rays_wide = dr.gather(mi.Ray3f, Lo_rays, dr.repeat(dr.arange(mi.UInt, dr.width(Lo_rays)), spp))
        active_wide = dr.repeat(active, spp)
        sampler_rt.seed(rng_state, dr.width(rays_wide)); rng_state += 0x0FF_FFFF
        integrator: mi.Integrator = self.mi_scene.integrator()
        Lo = dr.block_sum(
            integrator.sample(self.mi_scene, sampler_rt, rays_wide, active = active_wide)[0],
            spp) / spp
        return Lo, active, rng_state, Lo_rays



    def eval_Li_multislot(
            self,
            bsdf: VertexBSDF,
            si: mi.SurfaceInteraction3f, 
            sampler: mi.Sampler, 
            STREAM_LENGTH: int, 
            num_slots: int,
            rng_state: int = 0) -> tuple[mi.Color3f, MultiReservoirVector3f]:
        '''
        Compute the hemispheric integral of f(x) using reservoir sampling with `num_slots` 
        slots. All slots share the same input stream of proposals, of length `M`.
        '''
        NUM_STREAMS = dr.width(si)

        result = dr.zeros(mi.Color3f, NUM_STREAMS)
        rsv = MultiReservoirVector3f(NUM_STREAMS, num_slots)
        si_wide = dr.repeat(si, STREAM_LENGTH)
        sampler.seed(rng_state, NUM_STREAMS * STREAM_LENGTH); rng_state += 0x00FF_0000


        # ---------- Proposal 1: Emissive distribution ----------
        # # Draw samples from the proposal distribution and compute `p_hat`
        # d_sample, _, em_pdf = self.energy_pmf.sample(si_wide, sampler)
        # p_hat = eval_target_function(d_sample, si_wide, radiance_cache, sampler, rng_state)

        d_sample, active = self.energy_pmf.sample_no_pdf(si_wide, sampler)
        wi_sample = si_wide.to_local(d_sample)
        ray, t_start = self._get_ray_and_offset(si_wide, d_sample)
        Li, _, aovs = self.pdf_integrator.sample(dr.ADMode.Primal, self.gs_scene, sampler, ray, δL = None, state_in = 0.0, active = active, t_start = t_start)[:3]
        I = Li * bsdf.eval(mi.BSDFContext(), si_wide, wi_sample, active)
        p_hat = dr.norm(I)
        em_pdf = aovs[0]

        # Contrib. weight of drawn sample, `s.W`
        ds_W = dr.rcp(em_pdf)

        # Compute weight `w`
        mat_pdf = bsdf.pdf(mi.BSDFContext(), si_wide, wi_sample, active)
        mis_weight = em_pdf / (STREAM_LENGTH * (mat_pdf + em_pdf))
        # mis_weight = 1.0 / STREAM_LENGTH
        w = dr.select(em_pdf > 0.0, mis_weight * p_hat * ds_W, 0.0)

        # Add samples to reservoir
        rsv.add_proposals_vectorized(STREAM_LENGTH, d_sample, sampler, w)

        # ---------- Proposal 2: BSDF distribution ----------
        # # Draw samples from the proposal distribution and compute `p_hat`
        bs = bsdf.sample(mi.BSDFContext(), si_wide, sampler.next_1d(), sampler.next_2d())[0]
        wi_sample, mat_pdf = bs.wo, bs.pdf
        d_sample = si_wide.to_world(wi_sample)
        ray, t_start = self._spawn_offset_ray(si_wide, d_sample)
        Li, _, aovs = self.pdf_integrator.sample(dr.ADMode.Primal, self.gs_scene, sampler, ray, δL = None, state_in = 0.0, active = True, t_start = t_start)[:3]
        I = Li * bsdf.eval(mi.BSDFContext(), si_wide, wi_sample, active)
        p_hat = dr.norm(I)
        em_pdf = aovs[0]

        # Contrib. weight of drawn sample, `s.W`
        ds_W = dr.rcp(mat_pdf)

        # Compute weight `w`
        mis_weight = mat_pdf / (STREAM_LENGTH * (mat_pdf + em_pdf))
        # mis_weight = 1.0 / STREAM_LENGTH
        w = dr.select(em_pdf > 0.0, mis_weight * p_hat * ds_W, 0.0)

        # Add samples to reservoir
        rsv.add_proposals_vectorized(STREAM_LENGTH, d_sample, sampler, w)

        # ---------- Compute integral ----------
        ds = dr.zeros(mi.Vector3f, NUM_STREAMS * num_slots)
        ws = dr.zeros(Float, NUM_STREAMS * num_slots)
        for slot_idx in range(num_slots):
            scatter_idx = dr.arange(UInt, NUM_STREAMS) * num_slots + slot_idx
            dr.scatter(ds, rsv.sample[slot_idx], scatter_idx)
            dr.scatter(ws, rsv.w_sum[slot_idx],  scatter_idx)
        si_wide = dr.repeat(si, num_slots)
        # I = eval_target_and_integrand(ds, si_wide, radiance_cache, sampler, rng_state)
        ray = self._spawn_offset_ray(si_wide, ds)
        Li = self.integrator.sample(dr.ADMode.Primal, self.gs_scene, sampler, ray, δL = None, state_in = 0.0, active = active)[0]
        I = Li * bsdf.eval(mi.BSDFContext(), si_wide, si_wide.to_local(d_sample), active)
        p_hat = dr.norm(I)
        result = dr.block_sum(I * ws * dr.rcp(p_hat), block_size=num_slots)   # == I * contrib_weight

        result /= num_slots
        return result, rsv


def compute_loss(
        scene_sampler: SceneSurfaceSampler, 
        radiance_cache: RadianceCache, 
        trainable_bsdf: mi.BSDF, 
        num_points: int = 1,
        num_wi: int = 256, 
        num_wo: int = 1,
        rng_state: int = 0,
        ):
    return _compute_loss_mis(scene_sampler, radiance_cache, trainable_bsdf, num_points, num_wi, num_wo, rng_state)


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
        # si, num_points, rng_state = scene_sampler.sample_stratified(sampler, num_points, rng_state)
        si_wide = dr.repeat(si, num_wi)

        # # RHS: Evaluate incident directions
        Li_em,  wi_em,  active_em,  rng_state = radiance_cache.eval_Li_pdf(trainable_bsdf, si_wide, sampler, rng_state, SamplingMethod.Emitter)
        Li_mat, wi_mat, active_mat, rng_state = radiance_cache.eval_Li_pdf(trainable_bsdf, si_wide, sampler, rng_state, SamplingMethod.Cosine)
        dr.eval(Li_em, wi_em, active_em, Li_mat, wi_mat, active_mat)

        ctx = mi.BSDFContext()
        sampler.seed(rng_state, num_points); rng_state += 0x00FF_FFFF
        # Loop through the outgoing directions
        for _ in range(num_wo):
            # LHS: evaluate the emissive and outgoing radiances
            Le = radiance_cache.eval_Le(si)
            Lo, active_si, rng_state = radiance_cache.eval_Lo(si, sampler, rng_state)[:3]
            lhs = -Le + Lo

            # RHS: integrate over the incident directions and update the loss
            with dr.resume_grad():
                integrand = Li_mat * trainable_bsdf.eval(ctx, si_wide, wo = wi_mat, active = active_mat) \
                           + Li_em * trainable_bsdf.eval(ctx, si_wide, wo = wi_em,  active = active_em)
                rhs = dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi
                scale = dr.detach(dr.sqr(0.5 * (lhs + rhs)) + 1e-2)
                residuals = dr.select(active_si, dr.sqr(lhs - rhs), 0.0)
                loss += 0.5 * dr.mean(residuals / scale, axis=None) / num_wo

            # Pick new outgoing directions to sample
            sampler.seed(rng_state, num_points); rng_state += 0x00FF_FFFF
            si.wi = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())

            # Update `si_wide` with the new directions
            si_wide.wi = dr.repeat(si.wi, num_wi)
            sampler.schedule_state()
            dr.schedule(loss)

    return loss


def _compute_loss_mis_RT(
        scene_sampler: SceneSurfaceSampler, 
        radiance_cache: RadianceCache, 
        trainable_bsdf: mi.BSDF, 
        num_points: int,
        num_wi: int,
        num_wo: int,
        rng_state: int,
        spp: int = 64
        ):
    loss = Float(0.0)
    dr.enable_grad(loss)

    with dr.suspend_grad():
        sampler: mi.Sampler = mi.load_dict({'type': 'independent'})

        # Sample `NUM_POINTS` different surface points
        # si, rng_state = scene_sampler.sample(num_points, sampler, rng_state)
        si, num_points, rng_state = scene_sampler.sample_stratified(sampler, num_points, rng_state = rng_state)
        si_wide = dr.repeat(si, num_wi)

        # # RHS: Evaluate incident directions
        Li_em,  wi_em,  active_em,  rng_state = radiance_cache.eval_Li_pdf_RT(trainable_bsdf, si_wide, sampler, rng_state, SamplingMethod.Emitter, spp)
        Li_mat, wi_mat, active_mat, rng_state = radiance_cache.eval_Li_pdf_RT(trainable_bsdf, si_wide, sampler, rng_state, SamplingMethod.Cosine, spp)
        dr.eval(Li_em, wi_em, active_em)
        dr.eval(Li_mat, wi_mat, active_mat)

        ctx = mi.BSDFContext()
        sampler.seed(rng_state, num_points); rng_state += 0x00FF_FFFF
        # Loop through the outgoing directions
        for _ in range(num_wo):
            # LHS: evaluate the emissive and outgoing radiances
            Lo, active_si, rng_state = radiance_cache.eval_Lo_RT(si, sampler, rng_state, spp)[:3]
            lhs = Lo

            # RHS: integrate over the incident directions and update the loss
            with dr.resume_grad():
                bsdf_em  = trainable_bsdf.eval(ctx, si_wide, wo = wi_em,  active = active_em)
                bsdf_mat = trainable_bsdf.eval(ctx, si_wide, wo = wi_mat, active = active_mat)
                integrand = Li_mat * bsdf_mat + Li_em * bsdf_em

                rhs = dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi
                scale = dr.detach(dr.sqr(0.5 * (lhs + rhs)) + 1e-2)
                residuals = dr.select(active_si, dr.sqr(lhs - rhs), 0.0)
                loss += 0.5 * dr.mean(residuals / scale, axis=None) / num_wo

            # Pick new outgoing directions to sample
            sampler.seed(rng_state, num_points); rng_state += 0x00FF_FFFF
            si.wi = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())

            # Update `si_wide` with the new directions
            si_wide.wi = dr.repeat(si.wi, num_wi)
            sampler.schedule_state()
            dr.schedule(loss)

    return loss


def _compute_loss_mis_RT_onesample(
        scene_sampler: SceneSurfaceSampler, 
        radiance_cache: RadianceCache, 
        trainable_bsdf: mi.BSDF, 
        num_points: int,
        rng_state: int,
        spp: int = 64
        ):
    loss = Float(0.0)
    dr.enable_grad(loss)

    with dr.suspend_grad():
        sampler: mi.Sampler = mi.load_dict({'type': 'independent'})

        # Sample `NUM_POINTS` different surface points
        si, rng_state = scene_sampler.sample(num_points, sampler, rng_state)
        # si, num_points, rng_state = scene_sampler.sample_stratified(sampler, num_points, rng_state = rng_state)
        sampler.seed(rng_state, num_points); rng_state += 0x00FF_FFFF
        si.wi = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())
        wo_pdf = mi.warp.square_to_uniform_hemisphere_pdf(si.wi)

        # # RHS: Evaluate incident directions
        Li_em,  wi_em,  active_em,  rng_state, weight_em  = radiance_cache.eval_Li_pdf_RT_unweighted(trainable_bsdf, si, sampler, rng_state, SamplingMethod.Emitter, spp)
        Li_mat, wi_mat, active_mat, rng_state, weight_mat = radiance_cache.eval_Li_pdf_RT_unweighted(trainable_bsdf, si, sampler, rng_state, SamplingMethod.Cosine, spp)
        dr.eval(Li_em, wi_em, active_em, Li_mat, wi_mat, active_mat)
        # dr.eval(Li_mat, wi_mat, active_mat)

        ctx = mi.BSDFContext()
        sampler.seed(rng_state, num_points); rng_state += 0x00FF_FFFF
        # LHS: evaluate the emissive and outgoing radiances
        Lo, active_si, rng_state = radiance_cache.eval_Lo_RT(si, sampler, rng_state, spp)[:3]
        active_si &= wo_pdf > 0.0
        lhs = Lo * dr.inv_two_pi

        # RHS: integrate over the incident directions and update the loss
        with dr.resume_grad():
            res = weight_mat * dr.squared_norm(-lhs + Li_mat * trainable_bsdf.eval(ctx, si, wo = wi_mat, active = active_mat)) \
                + weight_em  * dr.squared_norm(-lhs + Li_em  * trainable_bsdf.eval(ctx, si, wo = wi_em,  active = active_em))
            # res = weight_mat  * dr.squared_norm(-lhs + Li_mat * trainable_bsdf.eval(ctx, si, wo = wi_mat, active = active_mat))
            res *= dr.rcp(wo_pdf)
            loss += dr.mean(dr.select(active_si, res, 0.0))
    return loss



def gather_rsv(rsv: MultiReservoirVector3f, prim_index: mi.UInt) -> MultiReservoirVector3f:
    rsv_sample = MultiReservoirVector3f(rsv.size(), rsv.num_slots)
    for slot_idx in range(rsv.num_slots):
        rsv_sample.sample[slot_idx] = dr.gather(type(rsv_sample.sample[slot_idx]), rsv.sample[slot_idx], prim_index)
        rsv_sample.p_hat [slot_idx] = dr.gather(type(rsv_sample.p_hat [slot_idx]), rsv.p_hat [slot_idx], prim_index)
        rsv_sample.w_sum [slot_idx] = dr.gather(type(rsv_sample.w_sum [slot_idx]), rsv.w_sum [slot_idx], prim_index)
    return rsv_sample

def scatter_rsv(rsv_target: MultiReservoirVector3f, rsv_sample: MultiReservoirVector3f, prim_index: mi.UInt) -> None:
    for slot_idx in range(rsv_target.num_slots):
        dr.scatter(rsv_target.sample[slot_idx], rsv_sample.sample[slot_idx], prim_index)
        dr.scatter(rsv_target.p_hat [slot_idx], rsv_sample.p_hat [slot_idx], prim_index)
        dr.scatter(rsv_target.w_sum [slot_idx], rsv_sample.w_sum [slot_idx], prim_index)


def _compute_loss_mis_woRIS(
        scene_sampler: SceneSurfaceSampler, 
        radiance_cache: RadianceCache, 
        trainable_bsdf: mi.BSDF, 
        num_points_target: int,
        num_wi: int,
        num_wo: int,
        rng_state: int,
        rsv_global: MultiReservoirVector3f,
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
    # TODO/NOTE: should switch from RT->GS if we want to actually use this
    num_slots = rsv_global.num_slots
    loss = Float(0.0)
    dr.enable_grad(loss)

    with dr.suspend_grad():
        sampler: mi.Sampler = mi.load_dict({'type': 'independent'})

        # Sample `NUM_POINTS` different surface points
        si, num_points, rng_state = scene_sampler.sample_stratified(sampler, num_points_target=num_points_target, rng_state=rng_state)
        si_wide = dr.repeat(si, num_wi)

        # # RHS: Evaluate incident directions
        Li_em,  wi_em,  active_em,  rng_state = radiance_cache.eval_Li_pdf_RT(trainable_bsdf, si_wide, sampler, rng_state, SamplingMethod.Emitter)
        Li_mat, wi_mat, active_mat, rng_state = radiance_cache.eval_Li_pdf_RT(trainable_bsdf, si_wide, sampler, rng_state, SamplingMethod.Cosine)
        dr.eval(Li_em, wi_em, active_em, Li_mat, wi_mat, active_mat)

        ctx = mi.BSDFContext()
        sampler.seed(rng_state, num_points * num_wo); rng_state += 0x000F_0000

        rsv_curr = MultiReservoirVector3f(num_points, num_slots)
        si_rsv = dr.repeat(si, num_wo)
        wo  = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())
        pdf = mi.warp.square_to_uniform_hemisphere_pdf(wo)
        si_rsv.wi = wo
        p_hat = dr.norm(radiance_cache.eval_Lo_RT(si_rsv, sampler, rng_state)[0])
        sampler.seed(rng_state, num_points * num_wo); rng_state += 0x000F_0000
        mis_weight = 1.0 / num_wo
        w = dr.select(pdf > 0.0, mis_weight * p_hat * dr.rcp(pdf * num_wo), 0.0)
        # assert dr.allclose(dr.squared_norm(wo), dr.ones(mi.Float, dr.width(wo)))
        rsv_curr.add_proposals_vectorized(num_wo, wo, sampler, w)
        # for _ in range(num_wo):
        #     wo = mi.warp.square_to_cosine_hemisphere(sampler.next_2d())
        #     pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)
        #     si.wi = wo
        #     p_hat = dr.norm(radiance_cache.eval_Lo(si, sampler, rng_state)[0])
        #     mis_weight = 1.0 / num_wo
        #     w = dr.select(pdf > 0.0, mis_weight * p_hat * dr.rcp(pdf * num_wo), 0.0)
        #     rsv_curr.add_proposal(wo, sampler, w)
        
        sampler_rsv = mi.load_dict({'type': 'independent'})
        sampler_rsv.seed(rng_state, num_points); rng_state += 0x000F_0000
        rsv_reuse = MultiReservoirVector3f(num_points, num_slots)
        rsv_prev = gather_rsv(rsv_global, si.prim_index)
        for slot_idx in range(num_slots):
            C_VALUE = 1
            curr_sample = rsv_curr.sample[slot_idx]
            mis_curr = balance_heuristic(C_VALUE, 20)
            rsv_reuse.add_proposal_on_slot(slot_idx, curr_sample, sampler_rsv.next_1d(), mis_curr * rsv_curr.w_sum[slot_idx])

            # Add previous iteration's reservoir
            prev_sample = rsv_prev.sample[slot_idx]
            mis_prev = balance_heuristic(20, C_VALUE)
            rsv_reuse.add_proposal_on_slot(slot_idx, prev_sample, sampler_rsv.next_1d(), mis_prev * rsv_prev.w_sum[slot_idx])

        sampler.seed(rng_state, num_points); rng_state += 0x000F_0000
        # # Loop through the outgoing directions
        for slot_idx in range(num_slots):
            # Update `si` with the new directions
            si.wi = dr.detach(rsv_reuse.sample[slot_idx])
            # Update `si_wide` with the new directions
            si_wide.wi = dr.repeat(si.wi, num_wi)

            # LHS: evaluate the emissive and outgoing radiances
            Le = radiance_cache.eval_Le(si)
            Lo, active_si, rng_state = radiance_cache.eval_Lo_RT(si, sampler, rng_state)[:3]
            lhs = -Le + Lo

            p_hat = dr.norm(Lo)

            # RHS: integrate over the incident directions and update the loss
            with dr.resume_grad():
                integrand = Li_mat * trainable_bsdf.eval(ctx, si_wide, wo = wi_mat, active = active_mat) \
                           + Li_em * trainable_bsdf.eval(ctx, si_wide, wo = wi_em,  active = active_em)
                rhs = dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi

                inv_scale = dr.rcp(dr.detach(dr.sqr(0.5 * (lhs + rhs)) + 1e-2))
                residuals = dr.select(active_si, dr.sqr(lhs - rhs) * inv_scale, 0.0)
                # residuals = dr.select(p_hat > 0.0, residuals * rsv_reuse.w_sum[slot_idx] * dr.rcp(p_hat), 0.0) / num_slots
                loss += 0.5 * dr.mean(residuals, axis=None)

                # residuals = dr.select(active_si & (p_hat > 0.0), dr.squared_norm(lhs - rhs) * rsv_reuse.w_sum[slot_idx] * dr.rcp(p_hat), 0.0) / num_slots
                # loss += 0.5 * dr.mean(residuals)

            sampler.schedule_state()
            # sampler_rsv.schedule_state()
            dr.schedule(loss)

        # NOTE: one reason this doesn't work as-is might be because of contention!!! 2 threads (`si` samples)
        # sharing and writing to the same index has nondeterministic behavior!
        # To fix this, we'd need to change the behavior of SurfaceSampler.sample() to do "stratified" sampling
        # per-triangle, i.e. ensure that each triangle contains at most one sampled `si`.
        scatter_rsv(rsv_global, rsv_reuse, si.prim_index)
    return loss #, rsv_reuse


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
        # si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, num_points), num_wi), dr.ReduceMode.Local)
        si_wide = dr.repeat(si, num_wi)

        # RHS: Evaluate incident directions
        Li, wi, active, rng_state = radiance_cache.eval_Li_uniform(si_wide, sampler, rng_state)[:4]
        dr.eval(Li, wi, active, rng_state)

        ctx = mi.BSDFContext()
        sampler.seed(rng_state, num_points); rng_state += 0x00FF_FFFF
        # Loop through the outgoing directions
        for _ in range(num_wo):
            # LHS: evaluate the emissive and outgoing radiances
            Lo, active_si, rng_state = radiance_cache.eval_Lo(si, sampler, rng_state)[:3]
            lhs = Lo

            # RHS: integrate over the incident directions and update the loss
            with dr.resume_grad():
                integrand = Li * trainable_bsdf.eval(ctx, si_wide, wo = wi, active = active)
                rhs = dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi
                # residuals = dr.select(active_si, dr.squared_norm(lhs - rhs), 0.0)
                # loss += 0.5 * dr.mean(residuals) / num_wo
                scale = dr.detach(dr.sqr(0.5 * (lhs + rhs)) + 1e-2)
                residuals = dr.select(active_si, dr.sqr(lhs - rhs), 0.0)
                loss += 0.5 * dr.mean(residuals / scale, axis=None) / num_wo

            # Pick new outgoing directions to sample
            sampler.seed(rng_state, num_points); rng_state += 0x00FF_FFFF
            si.wi = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())

            # Update `si_wide` with the new directions
            # si_wide.wi = dr.gather(mi.Vector3f, si.wi, dr.repeat(dr.arange(UInt, num_points), num_wi), dr.ReduceMode.Local)
            si_wide.wi = dr.repeat(si.wi, num_wi)
            sampler.schedule_state()
            dr.schedule(loss)
    return loss


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

def render_hemisphere_rt_inverse(scene: mi.Scene, si_scalar: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, N: int, spp: int, rng_state: int = 0):
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
    rays.d = -rays.d

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