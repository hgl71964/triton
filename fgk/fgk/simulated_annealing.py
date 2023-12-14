import os
import math
import random
import tempfile
import time
from copy import deepcopy
from multiprocessing import Process, Queue, set_start_method

import numpy as np
import torch

# asm (add CuAsm to PYTHONPATH!)
from CuAsm.CubinFile import CubinFile

# mutation
from fgk.mutator import MutationEngine
from fgk.sample import Sample
from fgk.utils.logger import get_logger
from fgk.utils.record import save_data

# a custom compilation pipeline to be executed in another process
from fgk.compiler import compile, CompiledKernel

logger = get_logger(__name__)


class SimulatedSample(Sample):

    def apply(self, index, action):
        lineno = self.candidates[index]
        if action == -1:
            self.kernel_section[lineno - 1], self.kernel_section[
                lineno] = self.kernel_section[lineno], self.kernel_section[
                    lineno - 1]
            self.candidates[index] -= 1
        elif action == 1:
            self.kernel_section[lineno], self.kernel_section[
                lineno +
                1] = self.kernel_section[lineno +
                                         1], self.kernel_section[lineno]
            self.candidates[index] += 1
        elif action == 0:
            pass
        else:
            assert False, f'invalid action: {action}'


def generate_neighbor(sample: SimulatedSample, n_choices, policy):
    mutable = sample.get_mutable()
    if policy == 'single':
        index = random.randint(0, len(mutable) - 1)
        action = random.choice([-1, 1])
        indexes = [index]
        actions = [action]
    elif policy == 'all':
        n = len(mutable)
        indexes = [i for i in range(n)]
        actions = [
            random.choice(range(-n_choices, n_choices)) for _ in range(n)
        ]
    else:
        raise RuntimeError(f'invalid policy: {policy}')

    neighbor = SimulatedSample(sample.kernel_section, sample.engine)
    neighbor.candidates = deepcopy(mutable)
    neighbor.dims = sample.dims
    neighbor.apply_all(indexes, actions)
    return neighbor


def acceptance_probability(old_fitness, new_fitness, temperature,
                           noise_factor):
    noise = random.uniform(-noise_factor, noise_factor)
    adjusted_difference = new_fitness - old_fitness + noise

    if adjusted_difference > 0:
        return 1.0

    return math.exp(adjusted_difference / temperature)


def simulated_annealing(
    initial_solution: SimulatedSample,
    init_fitness,
    n_choices,
    max_iterations,
    temperature,
    cooling_rate,
    policy,
    noise_factor,
    eng: MutationEngine,
) -> SimulatedSample:
    current_solution = initial_solution
    current_fitness = init_fitness
    best_solution = current_solution
    best_fitness = current_fitness
    cnt = 1

    while temperature > 0.05 and cnt < max_iterations:
        new_solution = generate_neighbor(current_solution, n_choices, policy)
        new_fitness = eng.get_perf(new_solution)

        logger.info(
            f'iter: {cnt}, current_fitness: {current_fitness:.2f}, new_fitness: {new_fitness:.2f}, best_fitness: {best_fitness:.2f}; temperature: {temperature:.2f}'
        )
        if acceptance_probability(current_fitness, new_fitness, temperature,
                                  noise_factor) > random.random():
            current_solution = new_solution
            current_fitness = new_fitness

        temperature *= 1 - cooling_rate
        cnt += 1

        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = current_solution

        # early stop
        if new_fitness < 0:
            # once illegal memory access, subsequent call may fail
            # so we early stop here
            break

    return best_solution, best_fitness


def run_simulated_annealing(
    # kernel
    bin,
    args,
    sig_key,
    non_constexpr_arg_values,
    grid_0,
    grid_1,
    grid_2,
    stream,
    launch_enter_hook,
    launch_exit_hook,

    # sa config
    n_choices=1,
    max_iterations=5000,
    temperature=1.0,
    cooling_rate=0.003,
    policy='single',
    noise_factor=0.0,

    # other config
    seed=0,
    test_sample=10,
    total_flops=None,
    save_suffix='',
    save_dir=None,
    warmup=100,
    rep=100,
):
    logger.info(
        'run simulated annealing with n_choices %d; max_iterations %d; temperature %f; cooling_rate %f; policy %s; noise_factor %f ',
        n_choices, max_iterations, temperature, cooling_rate, policy,
        noise_factor)

    # print(f'bin id {id(bin)}')

    # get initial cubin and asm (the initial have to file IO)
    with tempfile.NamedTemporaryFile(mode='wb', delete=True) as temp_file:

        cubin = bin.asm['cubin']
        temp_file.write(cubin)
        # Ensure data is written to the file before reading it
        temp_file.flush()
        temp_file.seek(0)

        time.sleep(1)
        cf = CubinFile(temp_file.name)

    # ===== config =====
    config = {
        'atol': 1e-2,
        "total_flops": total_flops,
        'warmup': warmup,
        'rep': rep,
    }

    eng = MutationEngine(
        bin,
        cf,
        config,
        grid_0,
        grid_1,
        grid_2,
        stream,
        launch_enter_hook,
        launch_exit_hook,
        non_constexpr_arg_values,
    )

    # ===== start =====
    initial_solution = SimulatedSample(eng.kernel_section, eng)
    init_perf = eng.get_perf(initial_solution)
    logger.info(f'init perf: {init_perf:.2f}')
    if init_perf < 0:
        raise RuntimeError(f'init perf {init_perf} < 0; not valid cubin')

    _t1 = time.perf_counter()
    best_solution, best_perf = simulated_annealing(
        initial_solution,
        init_perf,
        n_choices,
        max_iterations,
        temperature,
        cooling_rate,
        policy,
        noise_factor,
        eng,
    )

    _t2 = time.perf_counter()
    hours = (_t2 - _t1) / 3600

    final_perf = best_perf
    _ = eng.assemble(
        best_solution
    )  # if illegal memory access, this gives error, but cubin is valid

    logger.info(
        f'Performance: {final_perf:.2f}; init perf: {init_perf:.2f}; Search time: {hours:.2f}h'
    )
    logger.info(
        f'improvement: {(final_perf - init_perf) / init_perf * 100:.2f}%')

    # ===== test =====
    # TODO

    # ===== save =====
    save_data(
        bin,
        final_perf,
        init_perf,
        hours,
        args,
        sig_key,
        non_constexpr_arg_values,
        seed,
        save_suffix,
        save_dir,
        algo='sa',
    )
    # print(f'final bin id {id(bin)}')
    return bin


def run(
    fn,
    signature,
    device,
    constants,
    num_warps,
    num_ctas,
    num_stages,
    enable_warp_specialization,
    enable_fp_fusion,
    extern_libs,
    configs,
    debug,
    device_type,
    queue,

    # args
    args,
    sig_key,
    non_constexpr_arg_values,
    grid_0,
    grid_1,
    grid_2,
    stream,
    max_iterations,
    temperature,
    cooling_rate,
    policy,
    noise_factor,
    seed,
    test_sample,
    total_flops,
    save_suffix,
    warmup,
    rep,
):
    bin = compile(
        # TODO
        fn,

        # ``
        signature=signature,
        device=device,
        constants=constants,
        num_warps=num_warps,
        num_ctas=num_ctas,
        num_stages=num_stages,
        enable_warp_specialization=enable_warp_specialization,
        enable_fp_fusion=enable_fp_fusion,
        extern_libs=extern_libs,
        configs=configs,
        debug=debug,
        device_type=device_type,
    )
    bin = run_simulated_annealing(
        bin,
        args,
        sig_key,
        non_constexpr_arg_values,
        grid_0,
        grid_1,
        grid_2,
        stream,
        CompiledKernel.launch_enter_hook,
        CompiledKernel.launch_exit_hook,
        # algo
        1,
        max_iterations,
        temperature,
        cooling_rate,
        policy,
        noise_factor,
        seed,
        test_sample,
        total_flops,
        save_suffix,
        warmup,
        rep,
    )

    queue.put('ok')
    return bin


def launch(
    # compile args
    fn,
    signature,
    device,
    constants,
    num_warps,
    num_ctas,
    num_stages,
    enable_warp_specialization,
    enable_fp_fusion,
    extern_libs,
    configs,
    debug,
    device_type,

    # args
    args,
    sig_key,
    non_constexpr_arg_values,
    grid_0,
    grid_1,
    grid_2,
    stream,
    max_iterations,
    temperature,
    cooling_rate,
    policy,
    noise_factor,
    seed,
    test_sample,
    total_flops,
    save_suffix,
    warmup,
    rep,
):
    set_start_method('spawn')
    queue = Queue()
    process = Process(
        target=run,
        args=(
            fn,  # TODO cannot pickle fn
            signature,
            device,
            constants,
            num_warps,
            num_ctas,
            num_stages,
            enable_warp_specialization,
            enable_fp_fusion,
            extern_libs,
            configs,
            debug,
            device_type,
            queue,

            # args
            args,
            sig_key,
            non_constexpr_arg_values,
            grid_0,
            grid_1,
            grid_2,
            stream,
            max_iterations,
            temperature,
            cooling_rate,
            policy,
            noise_factor,
            seed,
            test_sample,
            total_flops,
            save_suffix,
            warmup,
            rep,
        ))
    process.start()
    process.join()

    result = queue.get()
    # print("Result from the process:", result)
    return result
