import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-gs/build/python')

import drjit as dr
import mitsuba as mi
from drjit.auto import Float, UInt, Bool

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


class EnergyPMF:
    def __init__(self, scene: mi.Scene):
        ellipsoids_found = False
        for shape in scene.shapes():
            if shape.shape_type() == +mi.ShapeType.Ellipsoids:
                ellipsoids = shape
                ellipsoids_found = True
                break

        if not(ellipsoids_found):
            raise Exception("Scene does not contain an EllipsoidMesh!")
            
        self.parse_ellipsoids(scene, ellipsoids)
        self.build_energy_pmf(ellipsoids)

        # self.ellipsoids = ellipsoids
        ellipsoid_shape_idx = dr.compress(scene.shapes_dr().shape_type() == +mi.ShapeType.Ellipsoids)
        self.ellipsoids = dr.gather(mi.ShapePtr, scene.shapes_dr(), ellipsoid_shape_idx)
        self.kernel: Kernel = scene.integrator().kernel
        self.pdf_eval: mi.Integrator = EllipsoidPdfIntegrator()
        self.use_gaussian_density = True    # TODO
        self.pdf_eval.use_gaussian_density = self.use_gaussian_density

    def parse_ellipsoids(self, scene: mi.Scene, ellipsoids: mi.Mesh):
        # `primitive_count()` returns the number of triangles used in the mesh representation of *all* the gaussians
        tri_count = ellipsoids.primitive_count()

        pose_key = "primitives.data"
        sh_key = "primitives.sh_coeffs"
        params_tmp = mi.traverse(scene)

        # 10 floats define the pose: translation (3) + rotation (4) + scale (3)
        pose_count = 10
        SH_DEGREE = 3
        sh_count = (3 * (SH_DEGREE + 1) ** 2)
        assert dr.width(params_tmp[sh_key]) // sh_count == dr.width(params_tmp[pose_key]) // pose_count, "SH and/or pose counts are invalid!"

        ellipsoid_count = dr.width(params_tmp[pose_key]) // pose_count
        assert ellipsoid_count * pose_count == dr.width(params_tmp[pose_key]), "Calculation of `ellipsoid_count` is invalid!"

        tris_per_splat = tri_count // ellipsoid_count
        assert tris_per_splat * ellipsoid_count == tri_count, "Calculation of `ellipsoid_count` and/or `tri_count` is invalid!"

        # self.ellipsoids: mi.Mesh = ellipsoids
        self.ellipsoid_count: int = ellipsoid_count
        self.tri_count: int = tri_count
        self.tris_per_splat: int = tris_per_splat
        # keep scene reference to perform ray intersection tests
        self.scene: mi.Scene = scene

    def build_energy_pmf(self, ellipsoids: mi.Mesh) -> mi.DiscreteDistribution:
        # Build an energy PMF over the splats. Each ellipsoid is weighted by its 
        # radiant intensity, which we compute using the L2 norm of the ellipsoid's 
        # SH coefficients.

        # First, look up the SH coeffs on each ellipsoid.
        si = dr.zeros(mi.SurfaceInteraction3f, self.ellipsoid_count)
        si.prim_index = dr.arange(UInt, self.ellipsoid_count)
        sh_coeffs = ellipsoids.eval_attribute_x("sh_coeffs", si)
        # GS adds a DC offset of +0.5 to the emitted radiance; this can be absorbed 
        # into the zeroth SH coeff
        sh_coeffs[0] += 0.5 * dr.sqrt(dr.four_pi)
        sh_coeffs[1] += 0.5 * dr.sqrt(dr.four_pi)
        sh_coeffs[2] += 0.5 * dr.sqrt(dr.four_pi)
        sh_norm = dr.norm(sh_coeffs)
        opacities = ellipsoids.eval_attribute_1("opacities", si)
        scales = ellipsoids.eval_attribute_x("scale", si)
        area_estimate = dr.sqrt(
            (scales[0] * scales[1]) ** 2 + 
            (scales[1] * scales[2]) ** 2 +
            (scales[0] * scales[2]) ** 2)

        # Compute the per-splat energy
        E_splats = sh_norm 
        # E_splats *= opacities 
        E_splats *= area_estimate
        # E_splats = dr.sqrt(E_splats)

        # Construct the probability distribution
        pmf = mi.DiscreteDistribution(E_splats)
        self.pmf = pmf

    def sample(self, si: mi.SurfaceInteraction3f, sampler: mi.Sampler) -> tuple[mi.Vector3f, Float]:
        if self.use_gaussian_density:
            d, active = self.sample_direction_gaussian(si, sampler) 
        else:
            d, active = self.sample_direction_ball(si, sampler)
        pdf = dr.select(active, self.eval_pdf(si, d, sampler), 0.0)
        weight = dr.select(active & (pdf > 0.0), dr.rcp(pdf), 0.0)
        return d, weight, pdf
    
    def sample_no_pdf(self, si: mi.SurfaceInteraction3f, sampler: mi.Sampler) -> tuple[mi.Vector3f, mi.Bool]:
        if self.use_gaussian_density:
            d, active = self.sample_direction_gaussian(si, sampler) 
        else:
            d, active = self.sample_direction_ball(si, sampler)
        return d, active
    
    def sample_direction(self, si: mi.SurfaceInteraction3f, sampler: mi.Sampler) -> tuple[mi.Vector3f, Float]:
        return self.sample(si, sampler)

    def sample_direction_gaussian(self, si: mi.SurfaceInteraction3f, sampler: mi.Sampler) -> mi.Vector3f:
        sample2   = sampler.next_2d()
        sample2_  = sampler.next_2d()
        sample1   = sampler.next_1d()
        ell_idx   = self.pmf.sample(sample1)
        ellipsoid = Ellipsoid.gather(self.ellipsoids, ell_idx, dr.full(Bool, True, dr.width(ell_idx)))

        u = cube_to_std_normal(sample2, sample2_)
        sample_pos = ellipsoid.rot @ (ellipsoid.scale * u) + ellipsoid.center
        direction  = mi.Vector3f(dr.normalize(sample_pos - si.p))

        _, active = gaussian_angle_pdf(ellipsoid, si.p, direction)
        return direction, active
    
    def sample_direction_ball(self, si: mi.SurfaceInteraction3f, sampler: mi.Sampler) -> mi.Vector3f:
        sample2   = sampler.next_2d()
        sample1_  = sampler.next_1d()
        sample1   = sampler.next_1d()
        ell_idx   = self.pmf.sample(sample1)
        ellipsoid = Ellipsoid.gather(self.ellipsoids, ell_idx, dr.full(Bool, True, dr.width(ell_idx)))

        u = cube_to_unit_ball(sample2.x, sample2.y, sample1_)
        sample_pos = ellipsoid.rot @ (ellipsoid.extent * ellipsoid.scale * u) + ellipsoid.center
        direction  = mi.Vector3f(dr.normalize(sample_pos - si.p))

        _, active = ball_angle_pdf(ellipsoid, si.p, direction)
        return direction, active
    
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
    

