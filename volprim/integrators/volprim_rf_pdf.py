# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import RBIntegrator
from .common import *

def ball_angle_pdf(ellipsoid: Ellipsoid, ray_o, ray_d):
    S = ellipsoid.scale * ellipsoid.extent
    R = ellipsoid.rot
    o_ = (R.T @ (ray_o - ellipsoid.center)) / S
    d_ = (R.T @ ray_d) / S
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


class EllipsoidRfPdfIntegrator(RBIntegrator):
    '''
    This plugin implements a simple radiance field integrator for ellipsoids shapes.

    Parameters:
        max_depth (int): Maximum path depth. A value of -1 indicates no limit.
        kernel_type (str): Name of the kernel to use for rendering the volumetric primitives, one of ['gaussian', 'epanechnikov'].
    '''
    def __init__(self, props=mi.Properties()):
        super().__init__(props)

        max_depth = int(props.get("max_depth", -1))
        if max_depth < 0 and max_depth != -1:
            raise Exception('"max_depth" must be set to -1 (infinite) or a value >= 0')
        self.max_depth = mi.UInt32(max_depth if max_depth != -1 else 0xFFFFFFFF) # Map -1 (infinity) to 2^32-1 bounces

        self.srgb_primitives = False

        # Those kernel parameters are required for the volumetric transmittance model
        props['kernel_full_range'] = True
        props['kernel_normalized'] = True
        props['kernel_type'] = 'gaussian'
        self.kernel = Kernel.factory(props)
        self.use_gaussian_density = True
        self.ellipsoid_pmf = None

    def set_pmf(self, pmf: mi.DiscreteDistribution) -> None:
        self.ellipsoid_pmf = pmf

    def traverse(self, callback):
        callback.put_parameter("max_depth",       self.max_depth,       mi.ParamFlags.NonDifferentiable)
        callback.put_parameter('srgb_primitives', self.srgb_primitives, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter('kernel_type',     self.kernel.type,     mi.ParamFlags.NonDifferentiable)
        callback.put_parameter('hide_emitters',   self.hide_emitters,   mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        if 'kernel_type' in keys:
            self.kernel = Kernel.factory({
                'kernel_type': self.kernel.type,
                'kernel_full_range': True,
                'kernel_normalized': True
            })

    def eval_pdf(self, si: mi.SurfaceInteraction3f, o: mi.Point3f, d: mi.Vector3f, active: mi.Bool):
        '''
        Evaluate the transmission model on intersected volumetric primitives
        '''
        ellipsoid = Ellipsoid.gather(si.shape, si.prim_index, active)
        if self.use_gaussian_density:
            vol_pdf, active_ = gaussian_angle_pdf(ellipsoid, o, d)
        else:
            vol_pdf, active_ = ball_angle_pdf(ellipsoid, o, d)
        active &= active_
        dpdf = self.ellipsoid_pmf.eval_pmf_normalized(si.prim_index, active)
        return dr.select(active, vol_pdf * dpdf, 0.0)

    def eval_transmission(self, si, ray, active):
        '''
        Evaluate the transmission model on intersected volumetric primitives
        '''
        ellipsoid = Ellipsoid.gather(si.shape, si.prim_index, active)
        opacity = si.shape.eval_attribute_1('opacities', si, active)

        # Gaussian splatting transmittance model
        # Find the peak location along the ray. From "3D Gaussian Ray Tracing"
        o = ellipsoid.rot.T * (ray.o - ellipsoid.center) / ellipsoid.scale
        d = ellipsoid.rot.T * ray.d / ellipsoid.scale
        t_peak = -dr.dot(o, d) / dr.dot(d, d)
        p_peak = ray(t_peak)

        density = self.kernel.eval(p_peak, ellipsoid, active)
        transmission = (1.0 - dr.minimum(opacity * density, 0.9999))
        return transmission

    def eval_sh_emission(self, si, ray, active):
        '''
        Evaluate the SH directionally emission on intersected volumetric primitives
        '''
        def eval(shape, si, ray, active):
            if shape is not None and shape.shape_type() == +mi.ShapeType.Ellipsoids:
                sh_coeffs = shape.eval_attribute_x("sh_coeffs", si, active)
                sh_degree = int(dr.sqrt((sh_coeffs.shape[0] // 3) - 1))
                sh_dir_coef = dr.sh_eval(ray.d, sh_degree)
                emission = mi.Color3f(0.0)
                for i, sh in enumerate(sh_dir_coef):
                    emission += sh * mi.Color3f(
                        [sh_coeffs[i * 3 + j] for j in range(3)]
                    )
                return dr.maximum(emission + 0.5, 0.0)
            else:
                return mi.Color3f(0.0)

        out = dr.dispatch(si.shape, eval, si, ray, active)
        return out

    @dr.syntax
    def sample(self, mode, scene, sampler, ray, δL, state_in, active, **kwargs):
        # --------------------- Configure integrator state ---------------------

        primal = mode == dr.ADMode.Primal
        ray = mi.Ray3f(dr.detach(ray))
        active = mi.Bool(active)
        depth = mi.UInt32(0)

        # This prevents `eval_pdf()` from falsely rejecting some rays due to their
        # intersections being too close to another splat's surface. DO NOT REMOVE
        origin = mi.Point3f(ray.o)

        # Skip emission and alpha blending when ray.t is less than `t_start`
        t_start: mi.Float = kwargs['t_start']

        if not primal:  # If the gradient is zero, stop early
            active &= dr.any((δL != 0))

        L  = mi.Spectrum(0.0 if primal else state_in)  # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0)  # Differential radiance
        β  = mi.Spectrum(1.0) # Path throughput weight
        total_pdf = dr.zeros(mi.Float, dr.width(ray))  # Pdf accumulator

        # ----------------------------- Main loop ------------------------------

        curr_t = dr.zeros(mi.Float, dr.width(ray))

        while dr.hint(active, label=f"Primitive splatting ({mode.name})"):

            # --------------------- Find next intersection ---------------------

            si: mi.SurfaceInteraction3f = scene.ray_intersect(
                ray,
                coherent=(depth == 0),
                ray_flags=mi.RayFlags.All | mi.RayFlags.BackfaceCulling,
                active=active,
            )

            active &= si.is_valid() & (si.shape.shape_type() == +mi.ShapeType.Ellipsoids)

            curr_t[active] += si.t

            # ----------------- Primitive emission evaluation ------------------

            Le = mi.Spectrum(0.0)

            with dr.resume_grad(when=not primal):
                emission     = self.eval_sh_emission(si, ray, active)
                transmission = self.eval_transmission(si, ray, active)
                Le[active] = β * (1.0 - transmission) * emission
                Le[~dr.isfinite(Le)] = 0.0

            # Update directional pdf
            # pdf = mi.Float(0.0)
            # pdf[active] = self.eval_pdf(si, origin, ray.d, active)
            pdf = self.eval_pdf(si, origin, ray.d, active)
            pdf[~dr.isfinite(pdf)] = 0.0

            # ------- Update loop variables based on current interaction -------

            # L[active] = (L + Le) if primal else (L - Le)
            # β[active] *= transmission
            L[active & (curr_t >= t_start)] = (L + Le) if primal else (L - Le)
            β[active & (curr_t >= t_start)] *= transmission

            # Accumulate pdf
            total_pdf[active] += pdf

            # Spawn new ray (don't use si.spawn_ray to avoid self intersections)
            ray.o[active] = si.p + ray.d * 1e-4
            curr_t[active] += 1e-4

            # -------------- Differential phase only (PRB logic) ---------------

            # with dr.resume_grad(when=not primal):
            #     if not primal:
            #         # Differentiable reflected indirect radiance for primitives
            #         Lr_ind = L * transmission / dr.detach(transmission)

            #         # Differentiable Monte Carlo estimate of all contributions
            #         Lo = Le + Lr_ind
            #         Lo = dr.select(active & dr.isfinite(Lo), Lo, 0.0)

            #         if mode == dr.ADMode.Backward:
            #             dr.backward_from(δL * Lo)
            #         else:
            #             δL += dr.forward_to(Lo)

            # ----------------------- Stopping criterion -----------------------

            active &= si.is_valid()
            depth[active] += 1

            # # Kill path if has insignificant contribution
            # β_max = dr.max(β)
            # active &= (β_max > 0.01)

            # # Perform Russian Roulette
            # sample_rr = sampler.next_1d() # Ensures the same sequence of random number is drawn for the primal and adjoint passes.
            # if primal and self.use_rr:
            #     rr_prob = dr.maximum(β_max, 0.1)
            #     rr_active = (depth >= self.rr_depth) & (β_max < 0.1)
            #     β[rr_active] *= dr.rcp(rr_prob)
            #     rr_continue = sample_rr < rr_prob
            #     active &= ~rr_active | rr_continue

            # Don't estimate next recursion if we exceeded number of bounces
            active &= depth < self.max_depth

        # Convert sRGB light transport to linear color space
        if self.srgb_primitives:
            L = mi.math.srgb_to_linear(L)

        return L if primal else δL, True, [total_pdf], L

    def to_string(self):
        return f"EllipsoidRfPdfIntegrator[]"
    
    def aov_names(self):
        return ['pdf']

mi.register_integrator("volprim_rf_pdf", lambda props: EllipsoidRfPdfIntegrator(props))
