# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import RBIntegrator
from .common import *

class VolumetricPrimitiveTomographyIntegrator(RBIntegrator):
    '''
    This plugin implements a simple tomography integrator for ellipsoids shapes,
    which only account for the absorption in the volume (e.g. no scattering).

    Parameters:
        max_depth (int): Maximum path depth. A value of -1 indicates no limit.
        kernel_type (str): Name of the kernel to use for rendering the volumetric primitives, one of ['gaussian', 'epanechnikov'].
        hide_emitters (bool): Hide emitters from indirect light sampling.
    '''
    def __init__(self, props=mi.Properties()):
        super().__init__(props)

        max_depth = int(props.get("max_depth", 64))
        if max_depth < 0 and max_depth != -1:
            raise Exception('"max_depth" must be set to -1 (infinite) or a value >= 0')
        self.max_depth = mi.UInt32(max_depth if max_depth != -1 else 0xFFFFFFFF) # Map -1 (infinity) to 2^32-1 bounces

        # Those kernel parameters are required for the volumetric transmittance model
        props['kernel_full_range'] = True
        props['kernel_normalized'] = False
        self.kernel = Kernel.factory(props)

    def eval_transmission(self, si, ray, active):
        '''
        Evaluate the transmission model on intersected volumetric primitives
        '''
        ellipsoid = Ellipsoid.gather(si.shape, si.prim_index, active)
        sigmat = si.shape.eval_attribute_1('sigma_t', si, active)
        density = self.kernel.density_integral(ray, ellipsoid, None, None, active)
        return dr.exp(-density * sigmat)

    @dr.syntax
    def sample(self, mode, scene, sampler, ray, δL, state_in, active, **kwargs):
        # This integrator only supports environment emitters
        emitter = scene.environment()

        # --------------------- Configure integrator state ---------------------

        primal = mode == dr.ADMode.Primal
        ray = mi.Ray3f(dr.detach(ray))
        active = mi.Bool(active)
        depth = mi.UInt32(0)

        if not primal:  # If the gradient is zero, stop early
            active &= dr.any((δL != 0))

        L  = mi.Spectrum(0.0 if primal else state_in)  # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0)  # Differential radiance
        β  = mi.Spectrum(1.0) # Path throughput weight

        # ----------------------------- Main loop ------------------------------

        while dr.hint(active, label=f"Primitive tracing ({mode.name})"):

            # --------------------- Find next intersection ---------------------

            si = scene.ray_intersect(
                ray,
                coherent=(depth == 0),
                ray_flags=mi.RayFlags.All | mi.RayFlags.BackfaceCulling,
                active=active,
            )

            # ----------------- Primitive emission evaluation ------------------

            Le = mi.Spectrum(0.0)

            with dr.resume_grad(when=not primal):
                transmission = self.eval_transmission(si, ray, active)

            β[active] *= transmission

            # -------------- Differential phase only (PRB logic) ---------------

            with dr.resume_grad(when=not primal):
                if not primal:
                    # Differentiable reflected indirect radiance for primitives
                    Lr_ind = L * transmission / dr.detach(transmission)

                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = Le + Lr_ind
                    Lo = dr.select(active & dr.isfinite(Lo), Lo, 0.0)

                    if mode == dr.ADMode.Backward:
                        dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

            # -------------- Interaction with environment emitter --------------

            escaped_medium = active & ~si.is_valid()
            escaped_medium &= ~((depth == 0) & mi.Bool(self.hide_emitters))
            L_dir = dr.select(escaped_medium, β * emitter.eval(si, escaped_medium) , 0.0)

            # ------- Update loop variables based on current interaction -------

            L[active] = (L + L_dir) if primal else (L - L_dir)

            # Spawn new ray (don't use si.spawn_ray to avoid self intersections)
            ray.o[active] = si.p + ray.d * 1e-4

            # ----------------------- Stopping criterion -----------------------

            active &= si.is_valid()
            depth[active] += 1

            # # Kill path if has insignificant contribution
            # active &= dr.max(β) > 0.01

            # Don't estimate next recursion if we exceeded number of bounces
            active &= depth < self.max_depth

        return L if primal else δL, True, [], L

    def to_string(self):
        return f"VolumetricPrimitiveTomographyIntegrator[]"

mi.register_integrator("volprim_tomography", lambda props: VolumetricPrimitiveTomographyIntegrator(props))
