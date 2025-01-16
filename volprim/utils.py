# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
Miscellaneous utility functions
'''
from __future__ import annotations # Delayed parsing of type annotations
import time
from contextlib import contextmanager
import drjit as dr
import mitsuba as mi

def concatenate_tensors(images: List[mi.TensorXf]) -> mi.TensorXf:
    '''
    Concatenate a list of tensors on the X axis
    '''
    shape = images[0].shape
    h = shape[0]
    w = shape[1]
    if len(shape) == 2:
        concatenated = dr.zeros(mi.TensorXf, shape=(h, len(images) * w))
        for i, img in enumerate(images):
            concatenated[:, (w*i):(w*(i+1))] = img.array
        concatenated = concatenated[:, :, None]
    else:
        concatenated = dr.zeros(mi.TensorXf, shape=(h, len(images) * w, shape[2]))
        for i, img in enumerate(images):
            concatenated[:, (w*i):(w*(i+1)), :] = img.array
    dr.eval(concatenated) # TODO necessary?
    return concatenated

@contextmanager
def time_operation(label):
    '''
    Context manager to time some Mitsuba / Dr.Jit operations.

    Make sure so use `dr.schedule(var)` to trigger the computation of a specific
    variable in the next kernel launch, and therefore include it in the timing.
    '''
    print(f'{label} ...')
    start = time.time()
    yield
    dr.eval()
    dr.sync_thread()
    print(f'{label} → done in {(time.time() - start)} sec')
