import os
import math
import random
import tempfile
import time
from copy import deepcopy

import numpy as np
import torch

# asm (add CuAsm to PYTHONPATH!)
from CuAsm.CubinFile import CubinFile

# mutation
from mutator import MutationEngine
from fgk.sample import Sample
from utils.logger import get_logger

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
    n_choices,
    max_iterations,
    temperature,
    cooling_rate,
    policy,
    noise_factor,
    eng: MutationEngine,
) -> SimulatedSample:
    current_solution = initial_solution
    current_fitness = eng.get_perf(current_solution)
    best_solution = current_solution
    best_fitness = current_fitness
    cnt = 1

    while temperature > 0.05 and cnt < max_iterations:
        new_solution = generate_neighbor(current_solution, n_choices, policy)
        new_fitness = eng.get_perf(new_solution)

        logger.info(
            f'iter: {cnt}, current_fitness: {current_fitness:.2f}, new_fitness: {new_fitness:.2f}, best_fitness: {best_fitness:.2f}'
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

    return best_solution, best_fitness


def run_simulated_annealing(
    # kernel
    bin,
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
    noise_factor=0.1,

    # other config
    seed=0,
    test_sample=10,
    total_flops=None,
    warmup=100,
    rep=100,
):
    # ===== seed =====
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # get initial cubin and asm (the initial have to file IO)
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file:

        cubin = bin.asm['cubin']
        temp_file.write(cubin)
        # Ensure data is written to the file before reading it
        temp_file.flush()
        temp_file.seek(0)

        time.sleep(1)
        cf = CubinFile(temp_file.name)

    # ===== config =====
    fn = lambda: bin.c_wrapper(
        grid_0,
        grid_1,
        grid_2,
        bin.num_warps,
        bin.num_ctas,
        bin.clusterDims[0],
        bin.clusterDims[1],
        bin.clusterDims[2],
        bin.shared,
        stream,
        bin.cu_function,
        launch_enter_hook,
        launch_exit_hook,
        bin,
        *bin.assemble_tensormap_to_arg(non_constexpr_arg_values),
    )

    config = {
        'atol': 1e-2,
        "total_flops": total_flops,
        'warmup': warmup,
        'rep': rep,
    }
    eng = MutationEngine(
        cf,
        fn,
        bin.hack_cubin,
        config,
    )

    # ===== start =====
    initial_solution = SimulatedSample(eng.kernel_section, eng)

    _t1 = time.perf_counter()

    # Run simulated annealing
    best_solution, best_fitness = simulated_annealing(
        initial_solution,
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
    logger.info(f'Performance: {best_fitness:.2f}; Search time: {hours:.2f}h')

    # ===== test =====
    # TODO
