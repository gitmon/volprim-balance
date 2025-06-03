# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import RBIntegrator
from .common import *

class EllipsoidAlphaIntegrator(RBIntegrator):
    '''
    This plugin implements a simple radiance field integrator for ellipsoids shapes.

    Parameters:
        max_depth (int): Maximum path depth. A value of -1 indicates no limit.
        rr_depth (int): Minimum path depth before enabling the Russian roulette path termination.
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

    @dr.syntax
    def sample(self, mode, scene, sampler, ray, δL, state_in, active, **kwargs):
        # --------------------- Configure integrator state ---------------------

        ray = mi.Ray3f(dr.detach(ray))
        active = mi.Bool(active)
        depth = mi.UInt32(0)

        L  = mi.Spectrum(0.0)   # Radiance accumulator
        β  = mi.Spectrum(1.0)   # Path throughput weight

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

            transmission = self.eval_transmission(si, ray, active)

            # ------- Update loop variables based on current interaction -------

            L[active] = β
            β[active] *= transmission

            # Spawn new ray (don't use si.spawn_ray to avoid self intersections)
            ray.o[active] = si.p + ray.d * 1e-4

            # ----------------------- Stopping criterion -----------------------

            active &= si.is_valid()
            depth[active] += 1

            # Kill path if has insignificant contribution
            # TODO: remove?
            β_max = dr.max(β)
            kill = (β_max <= 0.01)
            active &= ~kill
            L[kill] = 0.0

            # Don't estimate next recursion if we exceeded number of bounces
            active &= depth < self.max_depth

            # Stop tracing if we hit the ellipsoid selected by the sampling method
            active &= si.prim_index != kwargs['ell_idx']


        # L[depth >= 128] = 0.0

        return L, True, [], L

    def to_string(self):
        return f"EllipsoidAlphaIntegrator[]"

mi.register_integrator("volprim_alpha", lambda props: EllipsoidAlphaIntegrator(props))
