import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-gs/build/python')
sys.path.append("..")

import drjit as dr
import mitsuba as mi
from drjit.auto.ad import Float, UInt, Bool
mi.set_variant('cuda_ad_rgb')

import volprim
from volprim.integrators.common import *
from volprim.integrators.volprim_pdf import EllipsoidPdfIntegrator
from volprim.integrators.volprim_alpha import EllipsoidAlphaIntegrator


def cube_to_std_normal(sample2: mi.Point2f, sample2_: mi.Point2f) -> mi.Point3f:
    xy = mi.warp.square_to_std_normal(sample2)
    zw = mi.warp.square_to_std_normal(sample2_)
    return mi.Point3f(xy.x, xy.y, zw.x)

def cube_to_unit_ball(sample1u: Float, sample1v: Float, sample1w: Float) -> mi.Point3f:
    z = 2.0 * sample1u - 1.0
    theta = dr.two_pi * sample1v

    sin_t, cos_t = dr.sincos(theta)
    sqrt_z = dr.sqrt(1.0 - z * z)
    r = dr.cbrt(sample1w)
    points = mi.Point3f(
        r * sqrt_z * cos_t,
        r * sqrt_z * sin_t,
        r * z)
    return points

def ball_angle_pdf(ellipsoid: Ellipsoid, ray_o, ray_d):
    S = ellipsoid.scale * ellipsoid.extent
    R = ellipsoid.rot
    o_ = (R.T @ (ray_o - ellipsoid.center)) / S # dr.select(S < 1e-4, 1e-4, S) #(S + dr.max(S) * 1e-4)
    d_ = (R.T @ ray_d) / S # dr.select(S < 1e-4, 1e-4, S) #(S + dr.max(S) * 1e-4)
    inv_dsq = dr.rcp(dr.squared_norm(d_))
    norm_osq = dr.squared_norm(o_)
    active = norm_osq >= 1.0    # only accept points outside the ellipsoid

    alpha = dr.dot(o_,d_) * inv_dsq
    beta = (1.0 - norm_osq) * inv_dsq
    discr = dr.fma(alpha, alpha, beta)
    active &= discr >= 0.0
    pdf = dr.select(active,
                    dr.inv_two_pi * dr.rcp(dr.prod(S)) * dr.sqrt(discr) * dr.fma(4.0 * alpha, alpha, beta),
                    0.0)
    return pdf, active
    
    # active_, t0, t1 = mi.math.solve_quadratic(dr.squared_norm(d_), 2.0 * dr.dot(o_,d_), dr.squared_norm(o_) - 1.0)
    # active &= active_
    # pdf_ = dr.inv_four_pi * dr.rcp(dr.prod(S)) * dr.select(active & active_, (t1 * dr.square(t1) - t0 * dr.square(t0)), 0.0)
    # # print(dr.max((pdf_ - pdf) / pdf))
    # return pdf_, active

def gaussian_angle_pdf(
        ellipsoid: Ellipsoid,
        origin: mi.Point3f,
        direction: mi.Vector3f) -> tuple[mi.Float, mi.Bool]:
    '''
    Compute the solid angle pdf for an ellipsoid along the given ray direction. Mathematically, this
    is a line integral of the kernel's volumetric probability density along the ray, which is weighted 
    by the squared distance (t ** 2) to account for the geometry factor:

        angle_pdf = integrate(t ** 2 * vol_pdf(o + t * d), t_range=[-infty, infty])
    '''
    o = (ellipsoid.rot.T @ (origin - ellipsoid.center)) / ellipsoid.scale
    d = (ellipsoid.rot.T @ direction) / ellipsoid.scale
    d_dot_d = dr.squared_norm(d)
    o_dot_o = dr.squared_norm(o)
    o_dot_d = dr.dot(o, d)
    d_norm = dr.sqrt(d_dot_d)
    ratio = dr.square(o_dot_d) * dr.rcp(d_dot_d)
    vol_pdf = dr.inv_two_pi * dr.rcp(dr.prod(ellipsoid.scale)) * \
        dr.rcp(d_dot_d * d_norm) * (1.0 + ratio) * \
        dr.exp(0.5 * (ratio - o_dot_o))
    
    active = dr.squared_norm(o) >= dr.square(ellipsoid.extent)  # only accept points outside the ellipsoid

    return dr.select(active, vol_pdf, 0.0), active


class EllipsoidSampler:
    def __init__(self, scene: mi.Scene, pmf: mi.DiscreteDistribution):
        self.scene = scene
        self.ellipsoids = dr.gather(mi.ShapePtr, scene.shapes_dr(), 0)
        self.pmf = pmf
        self.kernel: Kernel = scene.integrator().kernel
        self.pdf_eval: mi.Integrator = EllipsoidPdfIntegrator()
        self.alpha_eval: mi.Integrator = EllipsoidAlphaIntegrator()
        self.use_gaussian_density = False
        self.pdf_eval.use_gaussian_density = self.use_gaussian_density

    def sample(self, si: mi.SurfaceInteraction3f, sampler: mi.Sampler) -> tuple[mi.Point3f, Float]:
        return self.sample_gaussian(si, sampler) if self.use_gaussian_density else self.sample_ball(si, sampler)

    def sample_gaussian(self, si: mi.SurfaceInteraction3f, sampler: mi.Sampler) -> tuple[mi.Point3f, Float]:
        sample2       = sampler.next_2d()
        sample2_      = sampler.next_2d()
        sample1       = sampler.next_1d()
        ell_idx, dpdf = self.pmf.sample_pmf(sample1)
        ellipsoid     = Ellipsoid.gather(self.ellipsoids, ell_idx, dr.full(Bool, True, dr.width(ell_idx)))

        u = cube_to_std_normal(sample2, sample2_)
        sample_pos = ellipsoid.rot @ (ellipsoid.scale * u) + ellipsoid.center
        direction  = dr.normalize(sample_pos - si.p)

        pdf, active = gaussian_angle_pdf(ellipsoid, si.p, direction)
        pdf *= dpdf

        opacity = self.kernel.eval_opacity_ray(ellipsoid, sample_pos, direction, active)
        emission = self.kernel.eval_sh_emission(ellipsoid, direction, active)
        # NOTE: still need to multiply by transmittance!
        Le = opacity * emission
        return direction, pdf, Le, ell_idx, sample_pos
    
    def sample_ball(self, si: mi.SurfaceInteraction3f, sampler: mi.Sampler) -> tuple[mi.Point3f, Float]:
        sample2       = sampler.next_2d()
        sample1_      = sampler.next_1d()
        sample1       = sampler.next_1d()
        ell_idx, dpdf = self.pmf.sample_pmf(sample1)
        ellipsoid     = Ellipsoid.gather(self.ellipsoids, ell_idx, dr.full(Bool, True, dr.width(ell_idx)))

        u = cube_to_unit_ball(sample2.x, sample2.y, sample1_)
        sample_pos = ellipsoid.rot @ (ellipsoid.extent * ellipsoid.scale * u) + ellipsoid.center
        direction  = dr.normalize(sample_pos - si.p)

        pdf, active = ball_angle_pdf(ellipsoid, si.p, direction) 
        pdf *= dpdf

        opacity = self.kernel.eval_opacity_ray(ellipsoid, sample_pos, direction, active)
        emission = self.kernel.eval_sh_emission(ellipsoid, direction, active)
        # NOTE: still need to multiply by transmittance!
        Le = opacity * emission
        return direction, pdf, Le, ell_idx, sample_pos
    
    def eval_alpha(self, ray, ell_idx: UInt, sampler: mi.Sampler) -> mi.Float:
        T = self.alpha_eval.sample(
            dr.ADMode.Primal,
            self.scene,
            sampler,
            ray, None, 0.0, True, ell_idx=ell_idx)[0]
        return T.x

    def eval_pdf(self, si: mi.SurfaceInteraction3f, direction: mi.Vector3f, sampler: mi.Sampler) -> mi.Float:
        pdf = self.pdf_eval.sample(
            dr.ADMode.Primal, 
            self.scene, 
            sampler, 
            mi.Ray3f(si.p, direction), 
            0.0, 
            None, 
            True, 
            ellipsoid_pmf=self.pmf)[0]
        return pdf.x


# def run_emitter(scene, ellipsoid_pmf, si, sampler, num_rays, rng_state):
#     NUM_STREAMS = dr.width(si)
#     sampler.seed(rng_state, NUM_STREAMS * num_rays)
#     si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, NUM_STREAMS), num_rays))

#     ellipsoid_sampler = EllipsoidSampler(scene, ellipsoid_pmf)
#     d, pdf, Le, ell_idx, sample_pos = ellipsoid_sampler.sample(si_wide, sampler)#[:4]
#     active = mi.Frame3f.cos_theta(si_wide.to_local(d)) > 0.0

#     # trace a transmittance ray from `si.p` to `sample_pos` and accumulate the alpha-blending
#     ray = si_wide.spawn_ray(d)  # must use this one for `sample_gaussian()`
#     # ray = si_wide.spawn_ray_to(sample_pos)
#     T = ellipsoid_sampler.eval_alpha(ray, ell_idx, sampler)
#     L_out = dr.block_sum(dr.select(active & (pdf > 0.0), T * Le * dr.rcp(pdf), 0.0), num_rays) / num_rays
#     return L_out, d


def run_emitter(scene, ellipsoid_pmf, si, sampler, num_rays, rng_state):
    NUM_STREAMS = dr.width(si)
    sampler.seed(rng_state, NUM_STREAMS * num_rays)
    si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, NUM_STREAMS), num_rays))

    ellipsoid_sampler = EllipsoidSampler(scene, ellipsoid_pmf)
    d = ellipsoid_sampler.sample(si_wide, sampler)[0]
    active = mi.Frame3f.cos_theta(si_wide.to_local(d)) > 0.0
    pdf = dr.select(active, ellipsoid_sampler.eval_pdf(si_wide, d, sampler), 0.0)

    # trace a transmittance ray from `si.p` to `sample_pos` and accumulate the alpha-blending
    ray = si_wide.spawn_ray(d)  # must use this one for `sample_gaussian()`
    Le = scene.integrator().sample(
        dr.ADMode.Primal,
        scene,
        sampler,
        ray,
        0.0,
        None,
        True)[0]
    L_out = dr.block_sum(dr.select(active & (pdf > 0.0), Le * dr.rcp(pdf), 0.0), num_rays) / num_rays
    return L_out, d
