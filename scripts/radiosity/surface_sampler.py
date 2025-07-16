import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-gs/build/python')

import drjit as dr
import mitsuba as mi
from drjit.auto import Float, UInt, Bool

def is_delta_emitter(emitter: mi.Emitter):
    return (emitter.m_flags & mi.EmitterFlags.Delta) | \
           (emitter.m_flags & mi.EmitterFlags.DeltaPosition) | \
           (emitter.m_flags & mi.EmitterFlags.DeltaDirection)

class SceneSurfaceSampler:
    def __init__(self, scene: mi.Scene, method="equiarea", mesh_indexes: list[int] = None):
        shape_ptrs = scene.shapes_dr()
        if mesh_indexes is not None:
            # pmfs = dr.arange(Float, dr.width(shape_ptrs)) == Float(mesh_index)
            pmfs = dr.zeros(Float, dr.width(shape_ptrs))
            for idx in mesh_indexes:
                pmfs[idx] = 1.0
        elif method == "equiarea":
            # probability is proportional to mesh area
            pmfs = shape_ptrs.surface_area()
        elif method == "mesh-res":
            # probability is inversely proportional to avg_triangle_area
            prims_per_shape = dr.select(shape_ptrs.is_mesh(), mi.MeshPtr(shape_ptrs).face_count(), 1)
            mean_prim_area = shape_ptrs.surface_area() / prims_per_shape
            pmfs = dr.rcp(mean_prim_area)

        self.distribution = mi.DiscreteDistribution(pmfs)
        self.shape_ptrs = shape_ptrs
        # self.has_delta_emitters = dr.any((scene.emitters_dr().flags() & UInt(mi.EmitterFlags.Delta)) > 0)
        # self.delta_emitters = [emitter for emitter in scene.emitters() if is_delta_emitter(emitter)]

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

        return si, rng_state

        # # Compute the Li contribution from delta emitter sources
        # if len(self.delta_emitters) > 0:
        #     point_light = self.delta_emitters[0]
        #     # NOTE: sample `uv` is not actually used in the case of point lights
        #     uv = sampler_rt.next_2d()
        #     return si, *point_light.sample_direction(si, uv, True), rng_state
        # else:
        #     return si, dr.zeros(mi.DirectionSample3f), dr.zeros(mi.Color3f), rng_state

    def sample_stratified(self, sampler: mi.Sampler, num_points_target: int = None, fraction: float = 1.0, rng_state: int = 0) -> tuple[mi.SurfaceInteraction3f, int]:
        sampler.seed(rng_state, 1); rng_state += 0x00FF_FFFF
        idx = self.distribution.sample(sampler.next_1d(), True)

        mesh = dr.gather(mi.ShapePtr, self.shape_ptrs, idx)[0]
        if not(mesh.is_mesh()):
            raise AssertionError("Shape to sample from is not a triangle mesh!")

        F = dr.unravel(mi.Point3u, mesh.faces_buffer())
        sampler.seed(rng_state, mesh.face_count()); rng_state += 0x00FF_FFFF
        bary = mi.warp.square_to_uniform_triangle(sampler.next_2d())

        p0 = mesh.vertex_position(F.x)
        p1 = mesh.vertex_position(F.y)
        p2 = mesh.vertex_position(F.z)
        e0 = p1 - p0
        e1 = p2 - p0
        sample_pos = dr.fma(e0, bary.x, dr.fma(e1, bary.y, p0))
        if mesh.has_vertex_normals():
            n0 = mesh.vertex_normal(F.x)
            n1 = mesh.vertex_normal(F.y)
            n2 = mesh.vertex_normal(F.z)
            sample_n = dr.fma(n0, 1.0 - bary.x - bary.y, 
                              dr.fma(n1, bary.x, n2 * bary.y))
        else:
            sample_n = dr.cross(e0, e1)
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.p = sample_pos
        si.n = dr.normalize(sample_n)
        si.sh_frame = mi.Frame3f(si.n)
        si.prim_index = dr.arange(mi.UInt, mesh.face_count())
        si.shape = dr.gather(mi.ShapePtr, self.shape_ptrs, dr.repeat(mi.UInt(idx[0]), mesh.face_count()))

        uv = sampler.next_2d()
        wo_local = mi.warp.square_to_cosine_hemisphere(uv)
        si.wi = wo_local

        if num_points_target is not None:
            fraction = min(1.0, num_points_target / mesh.face_count())

        if fraction < 1.0:
            retained_indices = dr.compress(sampler.next_1d() < fraction)
            si = dr.gather(mi.SurfaceInteraction3f, si, retained_indices)

        num_points = dr.width(si)
        return si, num_points, rng_state
