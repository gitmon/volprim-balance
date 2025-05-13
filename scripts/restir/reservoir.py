import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-gs/build/python')

import drjit as dr
import mitsuba as mi
from drjit.auto import Float, UInt, Bool
mi.set_variant('cuda_ad_rgb')

from collections.abc import Callable

TEMPORAL_C_CAP = 500.0

class ReservoirVector3f:
    DRJIT_STRUCT = {
        'sample': mi.Vector3f,
        'valid': Bool,
        'w_sum': Float,
        'p_hat': Float,
        'c': Float,
    }

    def __init__(self, num_shading_points: int):
        self.sample = dr.zeros(mi.Vector3f, num_shading_points)
        self.valid = dr.zeros(Bool,  num_shading_points)
        self.w_sum = dr.zeros(Float, num_shading_points)
        self.p_hat = dr.zeros(Float, num_shading_points)
        self.c     = dr.zeros(Float, num_shading_points)

    def add_proposal(self, direction: mi.Vector3f, rand: Float, weight: Float, confidence: float = 1.0):
        self.w_sum += weight
        cond = (self.w_sum > 0.0) & (rand < weight * dr.rcp(self.w_sum))
        self.sample = dr.select(cond, direction, self.sample)
        self.valid |= weight > 0.0
        self.c = dr.minimum(self.c + confidence, TEMPORAL_C_CAP)
    
    def add_proposals_vectorized(self, stream_length: int, directions: mi.Vector3f, sampler: mi.Sampler, weights: Float, confidence: float = 1.0):
        # Vectorized version of `add_proposal()` that avoids unrolling the for-loop over the 
        # stream elements, at the cost of temporarily materializing the full stream.

        # Assume that the data (direction samples + their corresponding weights & uniform RNGs)
        # are laid out in block-wise order: [0, STREAM_LENGTH] is the range for the first pixel/
        # shading point/etc. , [STREAM_LENGTH + 1, 2*STREAM_LENGTH] is the next pixel, etc., for
        # a total of STREAM_LENGTH * NUM_STREAMS elements:
        #
        #   weights = dr.ones(Float, STREAM_LENGTH * NUM_STREAMS)
        #   directions = mi.Vector3f(sampler.next_1d(), sampler.next_1d(), sampler.next_1d())
        #   rand = sampler.next_1d()

        STREAM_LENGTH = stream_length
        NUM_STREAMS = self.size()

        # Compute the acceptance probabilities w_i / w_sum
        # caps = weights * dr.rcp(dr.cumsum(weights))
        cumsum_weights = dr.block_prefix_sum(weights, block_size=STREAM_LENGTH, exclusive=False)
        caps = weights * dr.rcp(cumsum_weights)
        block_end_idx = (1 + dr.arange(UInt, NUM_STREAMS)) * STREAM_LENGTH - 1
        w_sums = dr.gather(type(weights), cumsum_weights, block_end_idx)

        # From the probabilities, find the acceptance decisions over the full stream. We cast to
        # `Int` because bitwise prefix reduction ops don't support `Bool`
        rand = sampler.next_1d()
        accept_int  = dr.select(caps > rand, 1, 0)
        # print(accept_int)

        # We want to keep only the final accepted sample; its index can be found using a search. 
        # First, build a monotonic `accept_last` array that transitions from True->False at the 
        # index of the final accepted sample, `i_last`. 
        # (arr[i_last : -1] == False and arr[0 : i_last-1 (inclusive)] == True)
        accept_last = dr.block_prefix_reduce(dr.ReduceOp.Or, accept_int, block_size=STREAM_LENGTH, exclusive=True, reverse=True)
        # print(accept_last)

        # i_last = dr.zeros(dr.cuda.ad.Int, NUM_STREAMS)
        # for stream_id in range(NUM_STREAMS):
        #     start_idx = stream_id * STREAM_LENGTH
        #     end_idx = start_idx + STREAM_LENGTH - 1
        #     i = dr.binary_search(start_idx, end_idx, 
        #                             lambda i: dr.gather(type(accept_last), accept_last, i) == 1)
        #     dr.scatter(i_last, i, UInt(stream_id))

        # Finally, the per-stream selection index `i_last` is simply the blockwise sum of the elements 
        # in the `accept_last` array.
        i_last_local = dr.block_sum(accept_last, block_size=STREAM_LENGTH)
        i_last_global = i_last_local + dr.arange(UInt, NUM_STREAMS) * STREAM_LENGTH
        accepted_sample = dr.gather(mi.Vector3f, directions, i_last_global)

        # Update the reservoir members
        self.w_sum = w_sums
        self.sample = accepted_sample
        self.valid = w_sums > 0.0
        self.c = dr.minimum(self.c + confidence * stream_length, TEMPORAL_C_CAP)
    
    def size(self):
        return dr.width(self.sample)
    
    def eval_target(self, target_func: Callable[[mi.Vector3f], Float]):
        self.p_hat = dr.select(self.valid, target_func(self.sample), 0.0)

    def get_sample(self, target_func: Callable[[mi.Vector3f], Float]):
        self.eval_target(target_func)
        W = dr.select(self.p_hat > 0.0, self.w_sum * dr.rcp(self.p_hat), 0.0)
        return self.sample, W
    
    def reinitialize(self, mis_weight):
        # # update sample; no-op
        # self.sample = self.sample

        # update `w_sum`
        self.w_sum = dr.select(self.p_hat > 0.0, mis_weight * self.w_sum, 0.0)

        # update `valid`
        self.valid = self.w_sum > 0.0


def test_add_proposals(si: mi.SurfaceInteraction3f, M: int = 5, seed: int = 0):
    def proposal_direction(sample2: mi.Point2f):
        d = mi.warp.square_to_uniform_hemisphere(sample2)
        pdf = mi.warp.square_to_uniform_hemisphere_pdf(d)
        return d, pdf

    def eval_target_function(d: mi.Vector3f, si: mi.SurfaceInteraction3f, sampler: mi.Sampler):
        return si.sh_frame.cos_theta(si.to_local(d))

    STREAM_LENGTH = M
    NUM_STREAMS = dr.width(si)
    sampler = mi.load_dict({'type':'independent'})
    sampler.seed(seed, NUM_STREAMS * STREAM_LENGTH)
    sample2 = sampler.next_2d()
    sample1 = sampler.next_1d()

    # Populate reservoir using default `addSample()`
    rsv_ref = ReservoirVector3f(dr.width(si))
    for m in range(M):
        # Draw a sample from the proposal distribution
        sample2_ = dr.gather(type(sample2), sample2, dr.arange(UInt, NUM_STREAMS) * STREAM_LENGTH + m)
        sample1_ = dr.gather(type(sample1), sample1, dr.arange(UInt, NUM_STREAMS) * STREAM_LENGTH + m)
        d_sample, pdf = proposal_direction(sample2_)
        # Contrib. weight of drawn sample, `s.W`
        ds_W = dr.rcp(pdf)
        # Compute `p_hat`
        p_hat = eval_target_function(d_sample, si, sampler)
        # Compute weight `w`
        mis_weight = 1.0 / M
        w = dr.select(pdf > 0.0, mis_weight * p_hat * ds_W, 0.0)
        # Add sample to reservoir
        rsv_ref.add_proposal(d_sample, sample1_, w, 1.0)

    # Populate reservoir using vectorized `addSample()`
    rsv = ReservoirVector3f(dr.width(si))
    si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, NUM_STREAMS), STREAM_LENGTH))
    d_sample, pdf = proposal_direction(sample2)
    ds_W = dr.rcp(pdf)
    p_hat = eval_target_function(d_sample, si_wide, sampler)
    mis_weight = 1.0 / M
    w = dr.select(pdf > 0.0, mis_weight * p_hat * ds_W, 0.0)
    rsv.add_proposals_vectorized(STREAM_LENGTH, d_sample, sample1, w)


    eval_target = lambda d: eval_target_function(d, si, sampler)
    d1, W1 = rsv_ref.get_sample(eval_target)
    d2, W2 = rsv    .get_sample(eval_target)

    assert dr.allclose(d1, d2) and dr.allclose(W1, W2), "Vectorized implementation is wrong!"




class MultiReservoirVector3f:
    DRJIT_STRUCT = {
        'sample': list[mi.Vector3f],
        'p_hat': list[Float],
        'valid': bool,
        'w_sum': Float,
        'c': Float,
    }

    def __init__(self, num_shading_points: int, num_slots: int):
        self.sample = [dr.zeros(mi.Vector3f, num_shading_points) for _ in range(num_slots)]
        self.p_hat  = [dr.zeros(Float, num_shading_points) for _ in range(num_slots)]
        self.valid  =  dr.zeros(Bool,  num_shading_points)
        self.w_sum  =  dr.zeros(Float, num_shading_points)
        self.c      =  dr.zeros(Float, num_shading_points)
        self.num_slots = num_slots

    def add_proposal(self, direction: mi.Vector3f, sampler: mi.Sampler, weight: Float, confidence: float = 1.0):
        self.w_sum += weight
        self.valid |= weight > 0.0
        self.c = dr.minimum(self.c + confidence, TEMPORAL_C_CAP)
        for slot_idx in range(self.num_slots):
            rand = sampler.next_1d()
            cond = (self.w_sum > 0.0) & (rand < weight * dr.rcp(self.w_sum))
            self.sample[slot_idx] = dr.select(cond, direction, self.sample[slot_idx])
    
    def add_proposal_on_slot(self, slot_idx: int, direction: mi.Vector3f, rand: Float, weight: Float, confidence: float = 1.0):
        self.w_sum += (weight / self.num_slots)
        self.valid |= weight > 0.0
        self.c = dr.minimum(self.c + confidence, TEMPORAL_C_CAP)
        cond = (self.w_sum > 0.0) & (rand < weight * dr.rcp(self.w_sum))
        self.sample[slot_idx] = dr.select(cond, direction, self.sample[slot_idx])
    
    def add_proposals_vectorized(self, stream_length: int, directions: mi.Vector3f, sampler: mi.Sampler, weights: Float, confidence: float = 1.0):
        # Vectorized version of `add_proposal()` that processes the full stream of proposals in 
        # one step. It avoids unrolling the for-loop over the stream elements at the cost of 
        # temporarily materializing the full stream.

        # The input data (direction samples + their corresponding weights & uniform RNGs) should
        # be laid out in block-wise order: [0, STREAM_LENGTH] is the stream for the first pixel/
        # shading point/etc., [STREAM_LENGTH + 1, 2*STREAM_LENGTH] is the next pixel, etc., for
        # a total of STREAM_LENGTH * NUM_STREAMS elements:
        #
        #   weights = dr.ones(Float, STREAM_LENGTH * NUM_STREAMS)
        #   directions = mi.Vector3f(sampler.next_1d(), sampler.next_1d(), sampler.next_1d())
        #   rand = sampler.next_1d()

        STREAM_LENGTH = stream_length
        NUM_STREAMS = self.size()

        # Compute the acceptance probabilities w_i / w_sum -- 
        cumsum_weights = dr.block_prefix_sum(weights, block_size=STREAM_LENGTH, exclusive=False)
        caps = weights * dr.rcp(cumsum_weights)
        block_end_idx = (1 + dr.arange(UInt, NUM_STREAMS)) * STREAM_LENGTH - 1
        w_sums = dr.gather(type(weights), cumsum_weights, block_end_idx)

        for slot_idx in range(self.num_slots):
            rand = sampler.next_1d()
            # From the probabilities, find the acceptance decisions over the full stream. We cast to
            # `Int` because bitwise prefix reduction ops don't support `Bool`
            accept_int: UInt = dr.select(caps > rand, 1, 0)

            # We want to keep only the final accepted sample; its index can be found using a search. 
            # First, build a monotonic `accept_last` array that transitions from True->False at the 
            # index of the final accepted sample, `i_last`. 
            # (arr[i_last : -1] == False and arr[0 : i_last-1 (inclusive)] == True)
            accept_last = dr.block_prefix_reduce(dr.ReduceOp.Or, accept_int, block_size=STREAM_LENGTH, exclusive=True, reverse=True)

            # Finally, the per-stream selection index `i_last` is simply the blockwise sum of the elements 
            # in the `accept_last` array.
            i_last_local = dr.block_sum(accept_last, block_size=STREAM_LENGTH)
            i_last_global = i_last_local + dr.arange(UInt, NUM_STREAMS) * STREAM_LENGTH
            accepted_sample = dr.gather(mi.Vector3f, directions, i_last_global)
            self.sample[slot_idx] = accepted_sample

        # Update the reservoir members
        self.w_sum = w_sums
        self.valid = w_sums > 0.0
        self.c = dr.minimum(self.c + confidence * stream_length, TEMPORAL_C_CAP)
    
    def size(self):
        return dr.width(self.sample)

    def eval_target(self, target_func: Callable[[mi.Vector3f], Float], slot_idx: int):
        self.p_hat[slot_idx] = dr.select(self.valid, target_func(self.sample[slot_idx]), 0.0)

    def get_sample(self, target_func: Callable[[mi.Vector3f], Float], slot_idx: int):
        sample: mi.Vector3f = self.sample[slot_idx]
        self.eval_target(target_func, slot_idx)
        p_hat = self.p_hat[slot_idx]
        W = dr.select(p_hat > 0.0, self.w_sum * dr.rcp(p_hat), 0.0)
        return sample, W

    def reinitialize(self, mis_weight):
        # # update sample; no-op
        # self.sample = self.sample

        # update `w_sum`
        # self.w_sum = dr.select(self.p_hat > 0.0, mis_weight * self.w_sum, 0.0)
        self.w_sum = mis_weight * self.w_sum    # TODO: is this correct?

        # update `valid`
        self.valid = self.w_sum > 0.0


class MultiReservoirVector3f__:
    DRJIT_STRUCT = {
        'sample': list[mi.Vector3f],
        'p_hat': list[Float],
        'w_sum': list[Float],
    }

    def __init__(self, num_shading_points: int, num_slots: int):
        self.sample = [dr.zeros(mi.Vector3f, num_shading_points) for _ in range(num_slots)]
        self.p_hat  = [dr.zeros(Float, num_shading_points) for _ in range(num_slots)]
        self.w_sum  = [dr.zeros(Float, num_shading_points) for _ in range(num_slots)]
        self.num_slots = num_slots

    def add_proposal(self, direction: mi.Vector3f, sampler: mi.Sampler, weight: Float):
        for slot in range(self.num_slots):
            self.add_proposal_on_slot(slot, direction, sampler.next_1d(), weight)
    
    def add_proposal_on_slot(self, slot_idx: int, direction: mi.Vector3f, rand: Float, weight: Float):
        self.w_sum[slot_idx] += weight
        cond = (self.w_sum[slot_idx] > 0.0) & (rand < weight * dr.rcp(self.w_sum[slot_idx]))
        self.sample[slot_idx] = dr.select(cond, direction, self.sample[slot_idx])
    
    def add_proposals_vectorized(self, stream_length: int, directions: mi.Vector3f, sampler: mi.Sampler, weights: Float):
        # Vectorized version of `add_proposal()` that processes the full stream of proposals in 
        # one step. It avoids unrolling the for-loop over the stream elements at the cost of 
        # temporarily materializing the full stream.

        # The input data (direction samples + their corresponding weights & uniform RNGs) should
        # be laid out in block-wise order: [0, STREAM_LENGTH] is the stream for the first pixel/
        # shading point/etc., [STREAM_LENGTH + 1, 2*STREAM_LENGTH] is the next pixel, etc., for
        # a total of STREAM_LENGTH * NUM_STREAMS elements:
        #
        #   weights = dr.ones(Float, STREAM_LENGTH * NUM_STREAMS)
        #   directions = mi.Vector3f(sampler.next_1d(), sampler.next_1d(), sampler.next_1d())
        #   rand = sampler.next_1d()

        STREAM_LENGTH = stream_length
        NUM_STREAMS = self.size()

        # Compute the acceptance probabilities w_i / w_sum -- 
        cumsum_weights = dr.block_prefix_sum(weights, block_size=STREAM_LENGTH, exclusive=False)
        caps = weights * dr.rcp(cumsum_weights)
        block_end_idx = (1 + dr.arange(UInt, NUM_STREAMS)) * STREAM_LENGTH - 1
        w_sums = dr.gather(type(weights), cumsum_weights, block_end_idx)

        for slot in range(self.num_slots):
            rand = sampler.next_1d()
            # From the probabilities, find the acceptance decisions over the full stream. We cast to
            # `Int` because bitwise prefix reduction ops don't support `Bool`
            accept_int: UInt = dr.select(caps > rand, 1, 0)

            # We want to keep only the final accepted sample; its index can be found using a search. 
            # First, build a monotonic `accept_last` array that transitions from True->False at the 
            # index of the final accepted sample, `i_last`. 
            # (arr[i_last : -1] == False and arr[0 : i_last-1 (inclusive)] == True)
            accept_last = dr.block_prefix_reduce(dr.ReduceOp.Or, accept_int, block_size=STREAM_LENGTH, exclusive=True, reverse=True)

            # Finally, the per-stream selection index `i_last` is simply the blockwise sum of the elements 
            # in the `accept_last` array.
            i_last_local = dr.block_sum(accept_last, block_size=STREAM_LENGTH)
            i_last_global = i_last_local + dr.arange(UInt, NUM_STREAMS) * STREAM_LENGTH
            accepted_sample = dr.gather(mi.Vector3f, directions, i_last_global)
            self.sample[slot] = accepted_sample

        # Update the reservoir members
        self.w_sum = [w_sums for _ in range(self.num_slots)]
    
    def size(self):
        return dr.width(self.sample)

    def eval_target(self, target_func: Callable[[mi.Vector3f], Float], slot_idx: int):
        self.p_hat[slot_idx] = dr.select(self.w_sum[slot_idx] > 0.0, target_func(self.sample[slot_idx]), 0.0)

    def get_sample(self, target_func: Callable[[mi.Vector3f], Float], slot_idx: int):
        sample: mi.Vector3f = self.sample[slot_idx]
        self.eval_target(target_func, slot_idx)
        p_hat = self.p_hat[slot_idx]
        W = dr.select(p_hat > 0.0, self.w_sum[slot_idx] * dr.rcp(p_hat), 0.0)
        return sample, W

    def reinitialize(self, mis_weight):
        # # update sample; no-op
        # self.sample = self.sample

        # update `w_sum`
        # self.w_sum = dr.select(self.p_hat > 0.0, mis_weight * self.w_sum, 0.0)
        for slot in range(self.num_slots):
            self.w_sum[slot] *= mis_weight