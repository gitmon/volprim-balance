import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-gs/build/python')

import drjit as dr
import mitsuba as mi
from drjit.auto import Float, UInt, Bool
# from vertex_bsdf import Principled
# from sh_fitting import get_sh_count, fit_sh_on_scene

def is_delta_emitter(emitter: mi.Emitter):
    return (emitter.m_flags & mi.EmitterFlags.Delta) | \
           (emitter.m_flags & mi.EmitterFlags.DeltaPosition) | \
           (emitter.m_flags & mi.EmitterFlags.DeltaDirection)

class SceneSurfaceSampler:
    def __init__(self, scene: mi.Scene, method="equiarea"):
        shape_ptrs = scene.shapes_dr()
        if method == "equiarea":
            # probability is proportional to mesh area
            self.distribution = mi.DiscreteDistribution(shape_ptrs.surface_area())
        elif method == "mesh-res":
            # probability is inversely proportional to avg_triangle_area
            prims_per_shape = dr.select(shape_ptrs.is_mesh(), mi.MeshPtr(shape_ptrs).face_count(), 1)
            mean_prim_area = shape_ptrs.surface_area() / prims_per_shape
            self.distribution = mi.DiscreteDistribution(dr.rcp(mean_prim_area))
        self.shape_ptrs = shape_ptrs
        self.has_delta_emitters = dr.any((scene.emitters_dr().flags() & UInt(mi.EmitterFlags.Delta)) > 0)
        self.delta_emitters = [emitter for emitter in scene.emitters() if is_delta_emitter(emitter)]

    def sample(self, num_points: int, sampler_rt: mi.Sampler, rng_state: int = 0) -> mi.SurfaceInteraction3f:
        '''
        Inputs:
            - num_points: int. Number of surface points to sample.
            - sampler: Sampler. The pseudo-random number generator.
            - rng_state: int. Seed for the PRNG.
        Outputs: 
            - si: SurfaceInteraction3f. Array of surface sample points of size [#si,].
            - em_ds: DirectionSample3f. Direction sample records from the delta emitter -> surface samples, size [#si,].
            - em_Li: mi.Color3f. Incident radiances from the delta emitter to the surface samples, size [#si,].
        '''
        # Generate `NUM_POINTS` different surface samples
        sampler_rt.seed(rng_state, num_points); rng_state += 0x00FF_FFFF
        idx = self.distribution.sample(sampler_rt.next_1d(), True)
        shape = dr.gather(mi.ShapePtr, self.shape_ptrs, idx)
        uv = sampler_rt.next_2d()
        si = shape.sample_surface_interaction(0.0, uv)

        # Generate one outgoing ray per surface sample
        uv = sampler_rt.next_2d()
        wo_local = mi.warp.square_to_cosine_hemisphere(uv)
        si.wi = wo_local

        # Compute the Li contribution from delta emitter sources
        if len(self.delta_emitters) > 0:
            point_light = self.delta_emitters[0]
            # NOTE: sample `uv` is not actually used in the case of point lights
            uv = sampler_rt.next_2d()
            return si, *point_light.sample_direction(si, uv, True), rng_state
        else:
            return si, dr.zeros(mi.DirectionSample3f), dr.zeros(mi.Color3f), rng_state
        


# class RadianceCacheMiSH:
#     def __init__(self, scene: mi.Scene, sh_order: int = 3, fit_Nquad: int = 128, fit_spp: int = 64):
#         '''
#         Inputs:
#             - scene: Scene. The Mitsuba scene.
#             - sh_order: int. Maximum spherical harmonics (SH) degree to use in the SH fit.
#             - fit_Nquad: int. Number of quadrature points to use per angle (\theta, \phi) in the SH fit.
#             - fit_spp: int. Number of samples per ray in the SH fit integral.
#         '''
#         print("Fitting spherical harmonics to scene...")
#         fit_sh_on_scene(scene, sh_order, fit_Nquad, fit_spp)
#         print("Fitting complete.")
#         self.scene = scene
#         self.order = sh_order

#     def query_cached_Lo(self, si: mi.SurfaceInteraction3f, active: Bool = True) -> mi.Color3f:
#         '''
#         Inputs:
#             - si: SurfaceInteraction3f. Array of surface sample points of size [#si,].
#         Outputs: 
#             - Lo: mi.Color3f. Array of outgoing radiances of size [#si,].
#         '''
#         num_points = dr.width(si)
#         Lo = dr.zeros(mi.Color3f, num_points)
#         for sh_idx, basis in enumerate(dr.sh_eval(si.wi, self.order)):  # `wo` is stored in `si.wi`
#             Lo += dr.select(active, 
#                             basis * si.shape.eval_attribute_3(f"vertex_Lo_coeffs_{sh_idx}", si), 
#                             dr.zeros(mi.Color3f))
#         return Lo

#     def query_cached_Le(self, si: mi.SurfaceInteraction3f, active: Bool = True) -> mi.Color3f:
#         '''
#         Inputs:
#             - si: SurfaceInteraction3f. Array of surface sample points of size [#si,].
#         Outputs: 
#             - Le: mi.Color3f. Array of emission radiances of size [#si,].
#         '''
#         num_points = dr.width(si)
#         Le = dr.zeros(mi.Color3f, num_points)
#         for sh_idx, basis in enumerate(dr.sh_eval(si.wi, self.order)):  # `wo` is stored in `si.wi`
#             Le += dr.select(si.shape.is_emitter() & active, 
#                             basis * si.shape.eval_attribute_3(f"vertex_Le_coeffs_{sh_idx}", si), 
#                             dr.zeros(mi.Color3f))
#         return Le

#     def query_cached_Li(self, si: mi.SurfaceInteraction3f, num_wi: int, sampler_rt: mi.Sampler, rng_state: int = 0) -> mi.Color3f:
#         '''
#         Inputs:
#             - sampler: Sampler. The pseudo-random number generator.
#             - si: SurfaceInteraction3f. Array of surface sample points of size [#si,].
#             - num_wi: int. Number of incident directions `wi` to use at each surface point.
#         Outputs: 
#             - Li: mi.Color3f. Flattened array of incident radiances of size [#si * #wi,]. The data 
#             is in contiguous order, i.e. the first #wi entries belong to si0, and so on.
#             - wi_local: mi.Vector3f. Flattened array of incident directions of size [#si * #wi,].
#             - si_flattened: mi.Vector3f. Flattened array of surface sample points of size [#si * #wi,].
#         '''
#         # For each surface point, we should sample `NUM_DIRS` different `wi` directions.
#         # For now, assume that the points do NOT use identical `wi` directions, i.e. we
#         # need to draw a total of `NUM_DIRS * NUM_POINTS` samples for `wi`. 
#         # In that case, `wi` is a 2D matrix[NUM_POINTS, NUM_DIRS] while `si` is an 
#         # array[NUM_POINTS]. The latter needs to be broadcasted to match the shape of 
#         # `wi`, which is done using the `gather()` (aka "flatten") operation.
#         #
#         # There is one possible simplification with unknown impact on correctness: use 
#         # the same set of local `wi` directions at every point.
#         # Code changes: sample only `NUM_DIR` vectors for `wo_local`. Subsequently,
#         # we still need to take the "outer product" of the `wo_local` array with the 
#         # `si` array to get `NUM_POINTS * NUM_DIRS` rays.
#         #
#         num_points = dr.width(si)
#         # the `flat_idxs` has the form: 
#         #                                      v---- NUM_WI copies ---v
#         # [0, ..., 0, 1, ..., 1,    ...    NUM_POINTS-1, ..., NUM_POINTS-1]   (contiguous order)
#         #
#         si_flat_idxs = dr.repeat(dr.arange(UInt, num_points), num_wi)
#         # `si_flattened` has the form:
#         # [s0, ..., s0, s1, ..., s1,    ...    sN-1, ..., sN-1]
#         #
#         # NOTE: `dr.ReduceMode` is not used for the `gather()` itself, but instead to 
#         # implement its adjoint operation (scatter) for reverse-mode AD. For our context,
#         # the choice doesn't matter since we never backprop through this part of the algorithm,
#         # but `ReduceMode.Local` is probably optimal when our arrays are laid out 
#         # contiguously, as we do here.
#         si_flattened = dr.gather(mi.SurfaceInteraction3f, si, si_flat_idxs, mode = dr.ReduceMode.Local)

#         # Compute the incident radiance on `A` for a direction, `wi`
#         NUM_RAYS = num_points * num_wi
#         sampler_rt.seed(rng_state + 2 * 0x0FFF_FFFF, NUM_RAYS)
#         uv = sampler_rt.next_2d()
#         wi_local = mi.warp.square_to_cosine_hemisphere(uv)
#         wi_pdf   = mi.warp.square_to_cosine_hemisphere_pdf(wi_local)
#         wi_world = si_flattened.to_world(wi_local)
#         wi_rays = si_flattened.spawn_ray(wi_world)

#         # Compute Li for each of the incident directions by querying the radiance cache at the
#         # traced intersection.
#         si = self.scene.ray_intersect(wi_rays)
#         Li = self.query_cached_Lo(si, si.is_valid())

#         # Account for sampling weight
#         Li = dr.select(wi_pdf > 0.0, Li * dr.rcp(wi_pdf), dr.zeros(mi.Color3f))

#         return Li, wi_local, si_flattened

# def compute_loss(
#     scene_sampler: SceneSurfaceSampler, 
#     radiance_cache: RadianceCacheMiSH, 
#     trainable_bsdf: Principled | mi.BSDF,
#     num_points: int,
#     num_wi: int, 
#     rng_state: int
#     ):
#     '''
#     Inputs:
#         - scene_sampler: SceneSurfaceSampler. The scene sampler draws random points from the scene's surfaces.
#         - radiance_cache: RadianceCache. Data structure containing the emissive surface data.
#         - trainable_bsdf: Principled. 
#         - num_points: int. The number of surface point samples to use.
#         - num_wi: int. The number of incident directions per surface point to use to calculate the radiosity integral.
#     Outputs:
#         - loss: Float. The scalar loss.
#     '''
#     with dr.suspend_grad():
#         # Temp workaround. TODO: avoid initializing a new sampler at each iteration
#         sampler_rt: mi.Sampler = mi.load_dict({'type': 'independent'})
#         ctx = mi.BSDFContext(mi.TransportMode.Radiance, mi.BSDFFlags.All)

#         # Sample `NUM_POINTS` different surface points
#         si, delta_emitter_sample, delta_emitter_Li = scene_sampler.sample(num_points, sampler_rt, rng_state)

#         # perform a ray visibility test from `si` to the delta emitter
#         vis_rays = si.spawn_ray(delta_emitter_sample.d)
#         vis_rays.maxt = delta_emitter_sample.dist
#         emitter_occluded = radiance_cache.scene.ray_test(vis_rays)
#         delta_emitter_Li &= ~emitter_occluded

#         # Evaluate RHS scene emitter contribution
#         with dr.resume_grad():
#             f_emitter = trainable_bsdf.eval(ctx, si, wo = si.to_local(delta_emitter_sample.d), active = True)
#             rhs = f_emitter * delta_emitter_Li

#         # Evaluate LHS of balance equation
#         lhs = radiance_cache.query_cached_Lo(si) - radiance_cache.query_cached_Le(si)

#         # Evaluate RHS integral
#         Li, wi_local, si_flattened = radiance_cache.query_cached_Li(si, num_wi, sampler_rt, rng_state + 0x0FFF_FFFF)

#         with dr.resume_grad():
#             f_io = trainable_bsdf.eval(ctx, si = si_flattened, wo = wi_local, active = True)
#             integrand = f_io * Li
#             rhs += dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi

#             # # # DEBUG
#             # print(f"Li:{rhs}")
#             # # print(f"Light contribution: {point_Li}")
#             # # print(f"Light contribution an: {albedo * dr.inv_pi * I / (r * r)}")
#             # print(f"Lo:{lhs}")
#             # # albedo = mi.Color3f([0.2, 0.25, 0.7])
#             # # # L_an = albedo * dr.rcp(1.0 - albedo) * intensity / (dr.pi * r ** 2)
#             # # L_an = delta_emitter_Li * dr.rcp(1.0 - albedo)
#             # # print(f"Lo_an:{L_an}")
#             # # print(L_an/lhs)

#             return 0.5 * dr.mean(dr.squared_norm(lhs - rhs))