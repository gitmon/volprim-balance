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


class EllipsoidPdfIntegrator(RBIntegrator):
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

        # Those kernel parameters are required for the volumetric transmittance model
        props['kernel_full_range'] = True
        props['kernel_normalized'] = True
        self.kernel = Kernel.factory(props)
        self.use_gaussian_density = False

    def traverse(self, callback):
        callback.put_parameter("max_depth",       self.max_depth,       mi.ParamFlags.NonDifferentiable)
        callback.put_parameter('kernel_type',     self.kernel.type,     mi.ParamFlags.NonDifferentiable)
        callback.put_parameter('hide_emitters',   self.hide_emitters,   mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        if 'kernel_type' in keys:
            self.kernel = Kernel.factory({
                'kernel_type': self.kernel.type,
                'kernel_full_range': True,
                'kernel_normalized': True
            })

    def eval_pdf(self, ellipsoid_pmf: mi.DiscreteDistribution, si, o, d, active):
        '''
        Evaluate the transmission model on intersected volumetric primitives
        '''
        ellipsoid = Ellipsoid.gather(si.shape, si.prim_index, active)
        if self.use_gaussian_density:
            vol_pdf, active_ = gaussian_angle_pdf(ellipsoid, o, d)
        else:
            vol_pdf, active_ = ball_angle_pdf(ellipsoid, o, d)
        active &= active_
        dpdf = ellipsoid_pmf.eval_pmf_normalized(si.prim_index, active)
        return dr.select(active, vol_pdf * dpdf, 0.0)

    @dr.syntax
    def sample(self, mode, scene, sampler, ray, δL, state_in, active, **kwargs):
        # --------------------- Configure integrator state ---------------------
        pmf = kwargs.get('ellipsoid_pmf')

        ray = mi.Ray3f(dr.detach(ray))
        active = mi.Bool(active)
        depth = mi.UInt32(0)

        L  = mi.Spectrum(0.0)  # Radiance (pdf) accumulator

        # This prevents `eval_pdf()` from falsely rejecting some rays due to their
        # intersections being too close to another splat's surface. DO NOT REMOVE
        origin = mi.Point3f(ray.o)

        # ----------------------------- Main loop ------------------------------

        while dr.hint(active, label=f"Primitive splatting ({mode.name})"):

            # --------------------- Find next intersection ---------------------

            si = scene.ray_intersect(
                ray,
                coherent=(depth == 0),
                ray_flags=mi.RayFlags.All | mi.RayFlags.BackfaceCulling,
                active=active,
            )

            active &= si.is_valid() & (si.shape.shape_type() == +mi.ShapeType.Ellipsoids)

            # ----------------- Primitive emission evaluation ------------------

            Le = mi.Spectrum(0.0)
            vol_pdf = self.eval_pdf(pmf, si, origin, ray.d, active)
            Le[active] = vol_pdf
            Le[~dr.isfinite(vol_pdf)] = 0.0

            # ------- Update loop variables based on current interaction -------

            L[active] = L + Le

            # Spawn new ray (don't use si.spawn_ray to avoid self intersections)
            ray.o[active] = si.p + ray.d * 1e-4

            # ----------------------- Stopping criterion -----------------------

            active &= si.is_valid()
            depth[active] += 1

            # Don't estimate next recursion if we exceeded number of bounces
            active &= depth < self.max_depth

        return L, True, [], L

    def to_string(self):
        return f"EllipsoidPdfIntegrator[]"

mi.register_integrator("volprim_pdf", lambda props: EllipsoidPdfIntegrator(props))
