import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-gs/build/python')

import drjit as dr
import mitsuba as mi
# from mitsuba.ad.integrators.common import mis_weight as power_heuristic
from drjit.auto import Float, UInt, Bool
mi.set_variant('cuda_ad_rgb')

import numpy as np
from scripts.restir.reservoir import ReservoirVector3f, MultiReservoirVector3f, MultiReservoirVector3f__

def balance_heuristic(pdf_a: Float, pdf_b: Float) -> Float:
     return pdf_a * dr.rcp(pdf_a + pdf_b)

def eval_integrand(d: mi.Vector3f, si: mi.SurfaceInteraction3f) -> mi.Color3f:
        '''
        Define the integrand, `f(x)`. The integral of this choice of `f(x)` over the
        hemisphere admits an analytic solution: [0.5 * pi ** 2, pi, 0.0].
        '''
        d_local = si.to_local(d)
        # Le = mi.Color3f(
        #       mi.Frame3f.sin_theta(d_local), 
        #       mi.Frame3f.cos_theta(d_local), 
        #       0.0)
        Le = mi.Color3f(
              mi.Frame3f.sin_theta(d_local), 
              mi.Frame3f.sin_theta(d_local) + 0.01 * mi.Frame3f.cos_theta(d_local), 
              0.0)
        return Le

def eval_target_function(d: mi.Vector3f, si: mi.SurfaceInteraction3f) -> Float:
        '''
        Define the target density `p_hat` to be directly proportional to the integrand.
        Note that there will still be some mismatch between `p_hat(x)` and `f(x)` since 
        we use the *luminance* (== norm) of the color.
        '''
        Le = eval_integrand(d, si)
        p_hat = dr.norm(Le)
        return p_hat

def proposal_direction(si: mi.SurfaceInteraction3f, sample2: mi.Point2f) -> tuple[mi.Vector3f, Float]:
     '''
     The proposal distribution is chosen to be the uniform hemisphere.
     '''
     d_local = mi.warp.square_to_uniform_hemisphere(sample2)
     pdf     = mi.warp.square_to_uniform_hemisphere_pdf(d_local)
     d_world = si.to_world(d_local)
     return d_world, pdf


def run_singleslot(
          si: mi.SurfaceInteraction3f, 
          sampler: mi.Sampler, 
          M: int, 
          rng_state: int = 0) -> tuple[mi.Color3f, ReservoirVector3f]:
    '''
    Compute the hemispheric integral of f(x) using reservoir sampling with one slot.
    '''
    STREAM_LENGTH = M
    NUM_STREAMS = dr.width(si)

    rsv = ReservoirVector3f(NUM_STREAMS)
    si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, NUM_STREAMS), STREAM_LENGTH))
    sampler.seed(rng_state, NUM_STREAMS * STREAM_LENGTH)

    # Draw samples from the proposal distribution
    d_sample, pdf = proposal_direction(si, sampler.next_2d())

    # Contrib. weight of drawn sample, `s.W`
    ds_W = dr.rcp(pdf)

    # Compute `p_hat`
    p_hat = eval_target_function(d_sample, si_wide)

    # Compute weight `w`
    mis_weight = 1.0 / M
    w = dr.select(pdf > 0.0, mis_weight * p_hat * ds_W, 0.0)

    # Add samples to reservoir
    rsv.add_proposals_vectorized(STREAM_LENGTH, d_sample, sampler, w)

    # Evaluate integrand and compute integral
    target_eval = lambda ds: eval_target_function(ds, si)
    d, contrib_weight = rsv.get_sample(target_eval)
    result = eval_integrand(d, si) * contrib_weight
    return result, rsv


def run_singleslot_temporal(
          si: mi.SurfaceInteraction3f, 
          sampler: mi.Sampler, 
          M: int, 
          num_frames: int, 
          rng_state: int = 0) -> mi.Color3f:
    '''
    Compute the hemispheric integral of f(x) using reservoir sampling with one slot.
    '''
    STREAM_LENGTH = M
    NUM_STREAMS = dr.width(si)

    result, rsv_prev = run_singleslot(si, sampler, M, rng_state); rng_state += 0x00FF_0000

    target_func = lambda ds: eval_target_function(ds, si)
    si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, NUM_STREAMS), STREAM_LENGTH))

    sampler.seed(rng_state, NUM_STREAMS * STREAM_LENGTH)
    sampler_ = mi.load_dict({'type': 'independent'})
    sampler_.seed(rng_state + 0x00FF_0000, NUM_STREAMS)
    for _ in range(num_frames - 1):
        rsv_seed = ReservoirVector3f(NUM_STREAMS)

        # Draw samples from the proposal distribution
        d_sample, pdf = proposal_direction(si, sampler.next_2d())

        # Contrib. weight of drawn sample, `s.W`
        ds_W = dr.rcp(pdf)

        # Compute `p_hat`
        p_hat = eval_target_function(d_sample, si_wide)

        # Compute weight `w`
        mis_weight = 1.0 / M
        w = dr.select(pdf > 0.0, mis_weight * p_hat * ds_W, 0.0)

        # Add samples to reservoir
        rsv_seed.add_proposals_vectorized(STREAM_LENGTH, d_sample, sampler, w)

        # # Temporal reuse
        rsv_reuse = ReservoirVector3f(NUM_STREAMS)
        target_curr, target_prev = target_func, target_func

        # Add current iteration's reservoir
        curr_sample, curr_W = rsv_seed.get_sample(target_curr)
        p_hat_curr = target_curr(curr_sample)
        p_hat_prev = target_prev(curr_sample)
        mis_curr = balance_heuristic(
             p_hat_curr * 1, 
             p_hat_prev * 20)
        # mis_curr = balance_heuristic(
        #      p_hat_curr * rsv_seed.c, 
        #      p_hat_prev * rsv_prev.c)
        # print(mis_curr)
        rsv_reuse.add_proposal(curr_sample, sampler_.next_1d(), mis_curr * p_hat_curr * curr_W, rsv_seed.c)

        # Add previous iteration's reservoir
        # TODO optimization: insert the new proposals directly into `rsv_prev` instead of relying on `rsv_seed` and `rsv_reuse`?
        prev_sample, prev_W = rsv_prev.get_sample(target_prev)
        p_hat_curr = target_curr(prev_sample)
        p_hat_prev = target_prev(prev_sample)
        mis_prev = balance_heuristic(
             p_hat_prev * 20, 
             p_hat_curr * 1)
        # mis_prev = balance_heuristic(
        #      p_hat_prev * rsv_prev.c, 
        #      p_hat_curr * rsv_seed.c)
        rsv_reuse.add_proposal(prev_sample, sampler_.next_1d(), mis_prev * p_hat_prev * prev_W, rsv_prev.c)

        # Evaluate integrand and compute integral
        d, contrib_weight = rsv_reuse.get_sample(target_func)
        result = eval_integrand(d, si) * contrib_weight

        # Update reservoir buffer
        rsv_prev = rsv_reuse
        dr.eval(rsv_prev)

    return result


def run_multislot_temporal(
          si: mi.SurfaceInteraction3f, 
          sampler: mi.Sampler, 
          M: int, 
          num_slots: int,
          num_frames: int, 
          rng_state: int = 0) -> mi.Color3f:
    '''
    Compute the hemispheric integral of f(x) using reservoir sampling with one slot.
    '''
    STREAM_LENGTH = M
    NUM_STREAMS = dr.width(si)

    result, rsv_prev = run_multislot__(si, sampler, M, num_slots, rng_state); rng_state += 0x00FF_0000

    target_func = lambda ds: eval_target_function(ds, si)
    si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, NUM_STREAMS), STREAM_LENGTH))

    sampler.seed(rng_state, NUM_STREAMS * STREAM_LENGTH)
    sampler_ = mi.load_dict({'type': 'independent'})
    sampler_.seed(rng_state + 0x00FF_0000, NUM_STREAMS)
    for frame_i in range(num_frames - 1):
        # print(f"=================== Frame {frame_i+1} ===================")
        # print(f"Contents of rsv_prev: {rsv_prev.sample}")
        result = dr.zeros(mi.Color3f, NUM_STREAMS)
        rsv_seed = MultiReservoirVector3f__(NUM_STREAMS, num_slots)

        # Draw samples from the proposal distribution
        d_sample, pdf = proposal_direction(si, sampler.next_2d())

        # Contrib. weight of drawn sample, `s.W`
        ds_W = dr.rcp(pdf)

        # Compute `p_hat`
        p_hat = eval_target_function(d_sample, si_wide)

        # Compute weight `w`
        mis_weight = 1.0 / M
        w = dr.select(pdf > 0.0, mis_weight * p_hat * ds_W, 0.0)

        # Add samples to reservoir
        rsv_seed.add_proposals_vectorized(STREAM_LENGTH, d_sample, sampler, w)

        # # Temporal reuse
        rsv_reuse = MultiReservoirVector3f__(NUM_STREAMS, num_slots)
        target_curr, target_prev = target_func, target_func

        for slot_idx in range(num_slots):
            # print(f"\n-------- {slot_idx=} --------")
            # Add current iteration's reservoir
            curr_sample, curr_W = rsv_seed.get_sample(target_curr, slot_idx)
            p_hat_curr = target_curr(curr_sample)
            p_hat_prev = target_prev(curr_sample)
            mis_curr = balance_heuristic(
                p_hat_curr * 1, 
                p_hat_prev * 20)
            rsv_reuse.add_proposal_on_slot(slot_idx, curr_sample, sampler_.next_1d(), mis_curr * p_hat_curr * curr_W)
            # print(f"REUSE -- CURR_RSV -- adding {curr_sample=}, {curr_W=}")

            # Add previous iteration's reservoir
            prev_sample, prev_W = rsv_prev.get_sample(target_prev, slot_idx)
            p_hat_curr = target_curr(prev_sample)
            p_hat_prev = target_prev(prev_sample)
            mis_prev = balance_heuristic(
                p_hat_prev * 20, 
                p_hat_curr * 1)
            # print(f"REUSE -- PREV_RSV -- adding {prev_sample=}, {prev_W=}")
            # assert dr.none(dr.isnan(mis_prev))
            rsv_reuse.add_proposal_on_slot(slot_idx, prev_sample, sampler_.next_1d(), mis_prev * p_hat_prev * prev_W)

            # Evaluate integrand and compute integral
            d, contrib_weight = rsv_reuse.get_sample(target_func, slot_idx)
            result += eval_integrand(d, si) * contrib_weight

            # print(f"CONTENTS OF RSV_REUSE: {rsv_reuse.sample}, {rsv_reuse.w_sum}")

        result /= num_slots
        # Update reservoir buffer
        rsv_prev = rsv_reuse
        # print("\nSaving reservoir for next iteration ...")
        # print(f"{rsv_prev.sample}, {rsv_prev.w_sum}")
        dr.eval(result, rsv_prev)

    return result


def run_singleslot_temporal_opt(
          si: mi.SurfaceInteraction3f, 
          sampler: mi.Sampler, 
          M: int, 
          num_frames: int, 
          rng_state: int = 0) -> mi.Color3f:
    '''
    Compute the hemispheric integral of f(x) using reservoir sampling with one slot.
    '''
    STREAM_LENGTH = M
    NUM_STREAMS = dr.width(si)

    result, rsv_prev = run_singleslot(si, sampler, M, rng_state); rng_state += 0x00FF_0000

    target_func = lambda ds: eval_target_function(ds, si)
    si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, NUM_STREAMS), STREAM_LENGTH))

    sampler.seed(rng_state, NUM_STREAMS * STREAM_LENGTH)
    sampler_ = mi.load_dict({'type': 'independent'})
    sampler_.seed(rng_state + 0x00FF_0000, NUM_STREAMS)
    for _ in range(num_frames - 1):
        rsv_curr = ReservoirVector3f(NUM_STREAMS)

        # Draw samples from the proposal distribution
        d_sample, pdf = proposal_direction(si, sampler.next_2d())

        # Contrib. weight of drawn sample, `s.W`
        ds_W = dr.rcp(pdf)

        # Compute `p_hat`
        p_hat = eval_target_function(d_sample, si_wide)

        # Compute weight `w`
        mis_weight = 1.0 / M
        w = dr.select(pdf > 0.0, mis_weight * p_hat * ds_W, 0.0)

        # Add samples to reservoir
        rsv_curr.add_proposals_vectorized(STREAM_LENGTH, d_sample, sampler, w)

        # # Temporal reuse
        target_curr, target_prev = target_func, target_func

        # Add current iteration's reservoir
        rsv_curr.eval_target(target_curr)
        p_hat_curr = rsv_curr.p_hat
        p_hat_prev = target_prev(rsv_curr.sample)
        mis_curr = balance_heuristic(
             p_hat_curr * 1, 
             p_hat_prev * 20)
        rsv_curr.reinitialize(mis_curr)

        # Add previous iteration's reservoir
        prev_sample, prev_W = rsv_prev.get_sample(target_prev)
        p_hat_curr = target_curr(prev_sample)
        p_hat_prev = target_prev(prev_sample)
        mis_prev = balance_heuristic(
             p_hat_prev * 20, 
             p_hat_curr * 1)
        rsv_curr.add_proposal(prev_sample, sampler_.next_1d(), mis_prev * p_hat_prev * prev_W, rsv_prev.c)

        # Evaluate integrand and compute integral
        d, contrib_weight = rsv_curr.get_sample(target_func)
        result = eval_integrand(d, si) * contrib_weight

        # Update reservoir buffer
        rsv_prev = rsv_curr
        dr.eval(rsv_prev)

    return result


def run_multislot(
          si: mi.SurfaceInteraction3f, 
          sampler: mi.Sampler, 
          M: int, 
          num_slots: int, 
          rng_state: int = 0) -> tuple[mi.Color3f, MultiReservoirVector3f]:
    '''
    Compute the hemispheric integral of f(x) using reservoir sampling with `num_slots` 
    slots. All slots share the same input stream of proposals, of length `M`.
    '''
    STREAM_LENGTH = M
    NUM_STREAMS = dr.width(si)

    result = dr.zeros(mi.Color3f, NUM_STREAMS)
    rsv = MultiReservoirVector3f(NUM_STREAMS, num_slots)
    si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, NUM_STREAMS), STREAM_LENGTH))
    sampler.seed(rng_state, NUM_STREAMS * STREAM_LENGTH)

    # Draw samples from the proposal distribution
    d_sample, pdf = proposal_direction(si, sampler.next_2d())

    # Contrib. weight of drawn sample, `s.W`
    ds_W = dr.rcp(pdf)

    # Compute `p_hat`
    p_hat = eval_target_function(d_sample, si_wide)

    # Compute weight `w`
    mis_weight = 1.0 / M
    w = dr.select(pdf > 0.0, mis_weight * p_hat * ds_W, 0.0)

    # Add samples to reservoir
    rsv.add_proposals_vectorized(STREAM_LENGTH, d_sample, sampler, w)

    # Evaluate integrand and compute integral
    target_eval = lambda ds: eval_target_function(ds, si)
    for slot_idx in range(num_slots):
        d, contrib_weight = rsv.get_sample(target_eval, slot_idx)
        result += eval_integrand(d, si) * contrib_weight
    result /= num_slots
    return result, rsv


def run_multislot__(
          si: mi.SurfaceInteraction3f, 
          sampler: mi.Sampler, 
          M: int, 
          num_slots: int, 
          rng_state: int = 0) -> tuple[mi.Color3f, MultiReservoirVector3f__]:
    '''
    Compute the hemispheric integral of f(x) using reservoir sampling with `num_slots` 
    slots. All slots share the same input stream of proposals, of length `M`.
    '''
    STREAM_LENGTH = M
    NUM_STREAMS = dr.width(si)

    result = dr.zeros(mi.Color3f, NUM_STREAMS)
    rsv = MultiReservoirVector3f__(NUM_STREAMS, num_slots)
    si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, NUM_STREAMS), STREAM_LENGTH))
    sampler.seed(rng_state, NUM_STREAMS * STREAM_LENGTH)

    # Draw samples from the proposal distribution
    d_sample, pdf = proposal_direction(si, sampler.next_2d())

    # Contrib. weight of drawn sample, `s.W`
    ds_W = dr.rcp(pdf)

    # Compute `p_hat`
    p_hat = eval_target_function(d_sample, si_wide)

    # Compute weight `w`
    mis_weight = 1.0 / M
    w = dr.select(pdf > 0.0, mis_weight * p_hat * ds_W, 0.0)

    # Add samples to reservoir
    rsv.add_proposals_vectorized(STREAM_LENGTH, d_sample, sampler, w)

    # Evaluate integrand and compute integral
    target_eval = lambda ds: eval_target_function(ds, si)
    for slot_idx in range(num_slots):
        d, contrib_weight = rsv.get_sample(target_eval, slot_idx)
        result += eval_integrand(d, si) * contrib_weight
    result /= num_slots
    return result, rsv


def run_naive(si, sampler, M, rng_state: int = 0):
    '''
    Compute the hemispheric integral of f(x) using the naive method of importance 
    sampling the proposal distribution.
    '''
    STREAM_LENGTH = M
    NUM_STREAMS = dr.width(si)

    sampler.seed(rng_state, NUM_STREAMS * STREAM_LENGTH)
    d, pdf = proposal_direction(si, sampler.next_2d())
    integrand = eval_integrand(d, si)
    result = dr.mean(integrand * dr.rcp(pdf), axis=1)
    return result

def test_reservoir():
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.n = mi.Vector3f(0.0, 0.0, 1.0)
    si.sh_frame = mi.Frame3f(si.n)

    sampler = mi.load_dict({'type': 'independent'})
    res_exact = np.array([dr.pi * dr.pi * 0.5, dr.pi, 0.0])

    M = 64
    # Run naive integrator
    I_naive = []
    for rng_state in range(50):
         I = run_naive(si, sampler, M, rng_state)
         I_naive.append(I.numpy().T)
    I_naive = np.array(I_naive).squeeze()
    mean_naive, std_naive = np.mean(I_naive, axis=0), np.std(I_naive, axis=0)
    
    # Run single-slot RIS integrator
    I_single = []
    for rng_state in range(50):
         I = run_singleslot(si, sampler, M, rng_state)[0]
         I_single.append(I.numpy().T)
    I_single = np.array(I_single).squeeze()
    mean_single, std_single = np.mean(I_single, axis=0), np.std(I_single, axis=0)
    
    # Run multi-slot RIS integrator
    NUM_SLOTS = 4      # TODO
    I_multi = []
    for rng_state in range(50):
         I = run_multislot(si, sampler, M, NUM_SLOTS, rng_state)[0]
         I_multi.append(I.numpy().T)
    I_multi = np.array(I_multi).squeeze()
    mean_multi, std_multi = np.mean(I_multi, axis=0), np.std(I_multi, axis=0)
    
    # Run single-slot RIS integrator with temporal accumulation
    NUM_FRAMES = 64     # TODO
    I_temporal1 = []
    for rng_state in range(50):
         I = run_singleslot_temporal(si, sampler, M, NUM_FRAMES, rng_state)
         I_temporal1.append(I.numpy().T)
    I_temporal1 = np.array(I_temporal1).squeeze()
    mean_temporal1, std_temporal1 = np.mean(I_temporal1, axis=0), np.std(I_temporal1, axis=0)

    # Run multi-slot RIS integrator with temporal accumulation
    I_temporal2 = []
    for rng_state in range(10):
         I = run_multislot_temporal(si, sampler, M, NUM_SLOTS, NUM_FRAMES, rng_state)
         I_temporal2.append(I.numpy().T)
    I_temporal2 = np.array(I_temporal2).squeeze()
    mean_temporal2, std_temporal2 = np.mean(I_temporal2, axis=0), np.std(I_temporal2, axis=0)

    with np.printoptions(precision=3, suppress=True):
        print(res_exact)
        print(f"Naive:\t\t{mean_naive} +- {std_naive}")
        print(f"Single:\t\t{mean_single} +- {std_single}")
        print(f"Multi:\t\t{mean_multi} +- {std_multi}")
        print(f"Single+T:\t{mean_temporal1} +- {std_temporal1}")
        print(f"Multi+T:\t{mean_temporal2} +- {std_temporal2}")