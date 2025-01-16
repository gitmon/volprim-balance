# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
Benchmarking utilities for measuring function performance with Dr.Jit. Includes
a decorator and context manager for timing, logging, and storing results in
dataframes.
'''

import time, gc, os
from contextlib import contextmanager
from typing import Callable
from functools import wraps as _wraps
import drjit as _dr

def wrap_function(label: str,
                  dataframes: list=None,
                  nb_runs: int=4,
                  nb_dry_runs: int=0,
                  log_level: int=2,
                  clear_cache: bool=True,
                  no_async: bool=False):
    '''
    This function is used to wrap another function and measure its performance.
    It takes the following parameters:

    Parameters:
        label (str): A label for the function being wrapped.
        dataframes (list, optional): A list of dataframes to store the benchmark results in. (default: None)
        nb_runs (int, optional): The number of times to run the function being wrapped. (default: 4)
        nb_dry_runs (int, optional): The number of dry runs to perform before starting the benchmark. (default: 0)
        log_level (int, optional): Define the amount of log produced by the benchmarking. (default: 2)
            0 -> turns off all logs
            1 -> display progress bar for runs
            2 -> display benchmark results
        clear_cache (bool, optional): whether to clear the kernel cache between every run (necessary to compute the backend times)
        no_async (bool, optional): whether to measure the kernel synchronization benefits (default: False)

    Results:
        Total time (async): overall execution time.
        Total time (sync): overall execution time with kernel synchronization enforced.
                           This is useful to understand the impact of asynchronous launches.
        Jitting time: time spent by Dr.Jit to parse the Python/C++ code and assemble the computation graph
        Codegen time: time spent by Dr.Jit to assemble the LLVM/PTX kernels from the computation graph
        Backend time: time spent by the backend compiler (NVCC or LLVM) to compile the kernels
        Execution time: time spent executing the kernels

    A call to the wrapped function can take define a 'label' keyword argument,
    which will be appended to the label defined in the wrapper, and not passed
    to the function.

    Example:

        @volprim.benchmark.wrap_function(label='Render', nb_runs=8, dataframes=dataframes, log_level=2)
        def my_render_function(spp):
            return mi.render(scene, spp=spp, seed=0)

        img = my_render_function(spp=32, label='with 32 spp!')
        img = my_render_function(spp=16, label='another suffix')
    '''
    def wrapper(func: Callable):
        @_wraps(func)
        def f(*args, **kwargs):
            # If 'label' keyword argument exist, append it to the main label
            if 'label' in kwargs.keys():
                suffix = f" [{kwargs['label']}]"
                del kwargs['label']
            else:
                suffix = ''

            # Execute the function a number of times before measuring (optional)
            for i in range(nb_dry_runs):
                ret = func(*args, **kwargs)
                _dr.eval(ret)
                _dr.sync_thread()

            # Helper function to time a single run
            def single_run():
                clean_and_reset_drjit(clear_cache)
                start_time = time.time_ns() / 1e6
                ret = func(*args, **kwargs)
                _dr.eval(ret)
                _dr.sync_thread()
                total_time = float(time.time_ns() / 1e6 - start_time)
                return ret, total_time

            if log_level > 0:
                print(f'Benchmarking: \"{label}{suffix}\" ...')

            with _dr.scoped_set_flag(_dr.JitFlag.KernelHistory, True):
                # Use the Launch Blocking feature of Dr.Jit to ensure synchronization after every kernel.
                # This ensures accurate timing measurements for compilation times.
                with _dr.scoped_set_flag(_dr.JitFlag.LaunchBlocking, True):
                    codegen_times    = []
                    backend_times    = []
                    execution_times  = []
                    jitting_times    = []
                    sync_total_times = []

                    # Execute the different runs
                    for i in range(nb_runs):
                        ret, sync_total_time = single_run()

                        # Look at kernel history for measured timings
                        history = _dr.kernel_history([_dr.KernelType.JIT])
                        codegen_time   = sum([k['codegen_time'] for k in history])
                        backend_time   = sum([k['backend_time'] for k in history])
                        execution_time = sum([k['execution_time'] for k in history])
                        jitting_time = sync_total_time - (codegen_time + backend_time + execution_time)

                        codegen_times.append(codegen_time)
                        backend_times.append(backend_time)
                        execution_times.append(execution_time)
                        jitting_times.append(jitting_time)
                        sync_total_times.append(sync_total_time)
                        print(f'-- Run {i+1}/{nb_runs}', end='\r')

                if not no_async:
                    # Perform another set of runs with asynchronous kernel
                    # launches, and then measure the impact of the asynchronicity
                    with _dr.scoped_set_flag(_dr.JitFlag.LaunchBlocking, False):
                        async_total_times = []
                        for i in range(nb_runs):
                            ret, total_time_async = single_run()
                            async_total_times.append(total_time_async)
                            print(f'-- Run {i+1}/{nb_runs} (async)', end='\r')

                print(f'', end='\n')

                def mean(x):
                    return sum(x) / float(nb_runs)

                def std(x):
                    return _dr.sqrt(mean([v**2 for v in x]) - mean(x)**2)

                # Compute average of the different measures over the runs
                if not no_async:
                    async_total_time = mean(async_total_times)
                sync_total_time  = mean(sync_total_times)
                jitting_time     = mean(jitting_times)
                codegen_time     = mean(codegen_times)
                backend_time     = mean(backend_times)
                execution_time   = mean(execution_times)

                if not no_async:
                    async_total_time_std = std(async_total_times)
                sync_total_time_std  = std(sync_total_times)
                jitting_time_std     = std(jitting_times)
                codegen_time_std     = std(codegen_times)
                backend_time_std     = std(backend_times)
                execution_time_std   = std(execution_times)

                # Log results
                if log_level > 1:
                    print(f'-- Results (averaged over {nb_runs} runs):')
                    if not no_async:
                        print(f'        - Total time (async): {async_total_time:.2f} ms (± {async_total_time_std:.2f})')
                        print(f'        - Total time (sync):  {sync_total_time:.2f} ms (± {sync_total_time_std:.2f}) -> (async perf. gain: {sync_total_time - async_total_time:.2f} ms)')
                    else:
                        print(f'        - Total time:         {sync_total_time:.2f} ms (± {sync_total_time_std:.2f})')
                    print(f'        - Jitting time:       {jitting_time:.2f} ms (± {jitting_time_std:.2f})')
                    print(f'        - Codegen time:       {codegen_time:.2f} ms (± {codegen_time_std:.2f})')
                    print(f'        - Backend time:       {backend_time:.2f} ms (± {backend_time_std:.2f})')
                    print(f'        - Execution time:     {execution_time:.2f} ms (± {execution_time_std:.2f})')

                # Record dataframe
                if dataframes is not None:
                    assert isinstance(dataframes, list)
                    df = {
                        'label': label,
                        'suffix': suffix,
                        'nb_runs': nb_runs,
                        'sync_total_time': sync_total_time,
                        'codegen_time': codegen_time,
                        'backend_time': backend_time,
                        'execution_time': execution_time,
                        'jitting_time': jitting_time,
                        'sync_total_time_std': sync_total_time_std,
                        'codegen_time_std': codegen_time_std,
                        'backend_time_std': backend_time_std,
                        'execution_time_std': execution_time_std,
                        'jitting_time_std': jitting_time_std,
                    }

                    if not no_async:
                        df['async_total_time'] = async_total_time
                        df['async_total_time_std']= async_total_time_std

                    if len(args):
                        df['args'] = args
                    if len(kwargs):
                        df['kwargs'] = kwargs
                    dataframes.append(df)

            return ret

        return f

    return wrapper

@contextmanager
def single_run(label: str,
               dataframes: list=None,
               log_level=1,
               clear_cache: bool=True):
    '''
    Context manager to time some Mitsuba / Dr.Jit operations.

    Make sure so use `dr.schedule(var)` to trigger the computation of a specific
    variable in the next kernel launch, and therefore include it in the timing.

    Parameters:
        label (str): A label for the function being wrapped.
        dataframes (list, optional): A list of dataframes to store the benchmark results in. (default: None)
        log_level (int, optional): Define the amount of log produced by the benchmarking. (default: 1)
            0 -> turns off all logs
            1 -> display benchmark results
        clear_cache (bool, optional): whether to clear the kernel cache between every run (necessary to compute the backend times)

    Results:
        Benchmarking: "label"
            Total time (sync): overall execution time with kernel synchronization enforced.
            Jitting time: time spent by Dr.Jit to parse the Python/C++ code and assemble the computation graph
            Codegen time: time spent by Dr.Jit to assemble the LLVM/PTX kernels from the computation graph
            Backend time: time spent by the backend compiler (NVCC or LLVM) to compile the kernels
            Execution time: time spent executing the kernels

    Example:

        with benchmark.single_run(label='Rendering'):
            img = mi.render(scene, spp=512, seed=0)
    '''
    with _dr.scoped_set_flag(_dr.JitFlag.KernelHistory, True):
        with _dr.scoped_set_flag(_dr.JitFlag.LaunchBlocking, True):
            print(f'Benchmarking: \"{label}\" ...')
            clean_and_reset_drjit(clear_cache)
            start_time = time.time_ns() / 1e6
            yield
            _dr.eval()
            _dr.sync_thread()
            total_time = float(time.time_ns() / 1e6 - start_time)

            # Look at kernel history for measured timings
            history = _dr.kernel_history([_dr.KernelType.JIT])

            codegen_time   = sum([k['codegen_time'] for k in history])
            backend_time   = sum([k['backend_time'] for k in history])
            execution_time = sum([k['execution_time'] for k in history])
            jitting_time = total_time - (codegen_time + backend_time + execution_time)

            if log_level > 0:
                print(f'{label} benchmark results (single run):')
                print(f'    - Total time (sync):  {(total_time):.2f} ms')
                print(f'    - Jitting time:       {(jitting_time):.2f} ms')
                print(f'    - Codegen time:       {(codegen_time):.2f} ms')
                print(f'    - Backend time:       {(backend_time):.2f} ms')
                print(f'    - Execution time:     {(execution_time):.2f} ms')

            if dataframes:
                assert dataframes is list
                dataframes.append({
                    'label': label,
                    'nb_runs': nb_runs,
                    'total_time': total_time,
                    'codegen_time': codegen_time,
                    'backend_time': backend_time,
                    'execution_time': execution_time,
                    'jitting_time': jitting_time,
                })

def clear_cache_folders(clear_drjit=True, clear_nvdia=True) -> None:
    '''
    Remove temporary folders where compiled kernels may be stored on the system.

    This method is designed to prevent the renderer from using cached kernels
    from a previous run. It can be used to clear both the Dr.Jit cache folder
    and the temporary folder maintained by the Nvidia driver.
    '''
    import shutil, time

    if os.name != 'nt':
        def clear(path):
            if os.path.exists(path):
                time.sleep(0.1)
                shutil.rmtree(path, ignore_errors=True)
                while os.path.exists(path):
                    time.sleep(0.1)
                os.mkdir(path)
                time.sleep(0.1)
        if clear_drjit:
            clear(os.path.join(os.path.expanduser('~'), '.drjit/'))
        if clear_nvdia:
            clear(os.path.join(os.path.expanduser('~'), '.nv/'))
    else:
        folder = os.path.join(os.path.expanduser('~'), 'AppData/Local/Temp/drjit/')
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path, ignore_errors=True)
            except Exception as e:
                pass

def clean_and_reset_drjit(clear_cache: bool=True) -> None:
    '''
    Clean all internal data-structure of the JIT and wipe out the cache folders.
    '''
    _dr.kernel_history_clear()
    gc.collect()
    gc.collect()
    _dr.flush_malloc_cache()
    _dr.malloc_clear_statistics()
    if clear_cache:
        clear_cache_folders()
        _dr.flush_kernel_cache()
