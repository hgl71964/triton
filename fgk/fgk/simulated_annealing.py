import os
import math
import random
import tempfile
import time
from copy import deepcopy

import multiprocessing

import numpy as np
import torch

# asm (add CuAsm to PYTHONPATH!)
from CuAsm.CubinFile import CubinFile

# mutation
from fgk.mutator import MutationEngine
from fgk.sample import Sample
from fgk.utils.logger import get_logger
from fgk.utils.record import save_data, read_data

from fgk.compiler import CompiledKernel as fgk_CompiledKernel

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
    _ = initial_solution.get_mutable()
    init_perf = max([eng.get_init_perf() for _ in range(5)])
    logger.info(f'init perf: {init_perf:.2f}; dims: {initial_solution.dims}')
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

    # ===== save =====
    path = save_data(
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
    return path


def gen_test_samples(
    bin,
    non_constexpr_arg_values,
    grid_0,
    grid_1,
    grid_2,
    stream,
    launch_enter_hook,
    launch_exit_hook,
    n_test_samples,
    ret_ptr,
) -> list[dict]:
    test_samples = []
    for t in range(n_test_samples):
        # generate test sample
        test_list = []
        for i, inp in enumerate(non_constexpr_arg_values):
            arg = None

            if isinstance(inp, torch.Tensor):
                if i == ret_ptr:
                    arg = torch.empty_like(inp)
                else:
                    arg = torch.randn_like(inp).uniform_(0, 1)
            else:
                arg = deepcopy(inp)

            test_list.append(arg)

        # call
        bin.c_wrapper(
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
            *bin.assemble_tensormap_to_arg(test_list),
        )
        test_samples.append(test_list)

    return test_samples


def e2e_test(
    bin,
    grid_0,
    grid_1,
    grid_2,
    stream,
    launch_enter_hook,
    launch_exit_hook,
    ret_ptr,
    test_samples,
) -> list[bool]:
    oks = []
    for t, test_sample in enumerate(test_samples):

        test_list = []
        ref = None
        for idx, inp in enumerate(test_sample):
            if isinstance(inp, torch.Tensor) and idx == ret_ptr:
                ref = inp
                arg = torch.empty_like(ref)
                out_buffer = arg
            else:
                arg = inp

            test_list.append(arg)

        bin.c_wrapper(
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
            *bin.assemble_tensormap_to_arg(test_list),
        )

        # TODO: atm we only consider one output from kernel
        if torch.allclose(ref, out_buffer, atol=1e-2, rtol=0):
            oks.append(True)
        else:
            oks.append(False)

    passes = sum(oks)
    total = len(oks)
    if np.all(oks):
        logger.info(f"✅ kernel verified for {total} test samples")
    else:
        logger.error(f"❌ kernel fail; only {passes}/{total} passes")

    return oks


# YAPF: disable

def target_func(
    # compile args
    so_path, metadata, asm,

    # args
    args, sig_key, non_constexpr_arg_values,

    # kernel args
    grid_0, grid_1, grid_2, stream, # 

    enter_hook, exit_hook,

    # algo
    n_choice,
    max_iterations,
    temperature,
    cooling_rate,
    policy,
    noise_factor,

    # utils
    seed, total_flops, save_suffix, save_dir,
    warmup, rep,

    # mp
    queue,
):
    # print('asm: ', id(asm))
    bin = fgk_CompiledKernel(so_path, metadata, asm)
    path = run_simulated_annealing(
        bin,
        args,
        sig_key,
        non_constexpr_arg_values,

        grid_0, grid_1, grid_2, stream,
        enter_hook, exit_hook,

        # algo
        n_choice,
        max_iterations,
        temperature,
        cooling_rate,
        policy,
        noise_factor,
        seed,
        total_flops,
        save_suffix,
        save_dir,
        warmup,
        rep,
    )
    queue.put(path)

def test_func(
    # compile args
    so_path, metadata, asm,

    # args
    args, sig_key, non_constexpr_arg_values,
    ret_ptr, test_inputs, test_outputs,

    # kernel args
    grid_0, grid_1, grid_2, stream, # 

    enter_hook, exit_hook,

    path,   # path to get cubin
    n_test_samples,

    # mp
    queue,
):
    bin = fgk_CompiledKernel(so_path, metadata, asm)

    # use hint to generate test cases
    if ret_ptr is not None:
        test_samples = gen_test_samples(bin,
            non_constexpr_arg_values,
            grid_0,
            grid_1,
            grid_2,
            stream,
            enter_hook,
            exit_hook,
            n_test_samples,
            ret_ptr,
        )
        data = read_data(path)
        opt_asm = {
            'cubin': data['cubin'],
        }
        opt_bin = fgk_CompiledKernel(so_path, metadata, opt_asm)

        oks = e2e_test(
            opt_bin,
            grid_0,
            grid_1,
            grid_2,
            stream,
            enter_hook,
            exit_hook,
            ret_ptr,
            test_samples,
        )
        all_ok = np.all(oks)
        time.sleep(0.1)
        if not all_ok:  # we don't save unverified kernel
            os.remove(path)
        time.sleep(0.5)
        queue.put(all_ok)

    else:
        raise NotImplementedError('impl custom test verifier')


def launch_simulated_annealing(
    # compile args
    so_path, metadata, asm,

    # args
    args,
    sig_key,
    non_constexpr_arg_values,
    ret_ptr, test_inputs, test_outputs,

    # kernel args
    grid_0, grid_1, grid_2, stream, # 

    enter_hook, exit_hook,

    # algo
    sa_runs,
    n_choice,
    max_iterations,
    temperature,
    cooling_rate,
    policy,
    noise_factor,

    # test
    n_test_samples,

    # utils
    seed, total_flops, save_suffix, save_dir,
    warmup, rep,
):
    # we only need cubin
    dels = []
    for k, _ in asm.items():
        if k != 'cubin':
            dels.append(k)
    for d in dels:
        asm.pop(d)

    # NOTE: the driver code must wrap within if __name__ == '__main__'
    # To use CUDA with multiprocessing, you must use the 'spawn' start method
    mp_context = multiprocessing.get_context('spawn')
    rankings = {}

    for run in range(sa_runs):
        queue = mp_context.Queue()
        process = mp_context.Process(
            target=target_func,
            args=(
                so_path, metadata, asm,

                # args
                args, sig_key, non_constexpr_arg_values,

                # kernel args
                grid_0, grid_1, grid_2, stream,

                # enter_hook, exit_hook,
                None, None,  # TODO only use None for now

                # algo
                n_choice,
                max_iterations,
                temperature,
                cooling_rate,
                policy,
                noise_factor,

                # utils
                seed, total_flops, save_suffix, save_dir,
                warmup, rep,  #

                # mp
                queue,
            ))
        process.start()
        process.join()
        path = queue.get()
        logger.debug(f'cubin path: {path}')

        # test_func(
        #     so_path, metadata, asm,

        #     # args
        #     args, sig_key, non_constexpr_arg_values,
        #     ret_ptr, test_inputs, test_outputs,

        #     # kernel args
        #     grid_0, grid_1, grid_2, stream,

        #     # enter_hook, exit_hook,
        #     None, None,

        #     path, n_test_samples,

        #     # mp
        #     queue,
        # )
        queue = mp_context.Queue()
        process = mp_context.Process(
            target=test_func,
            args=(
                so_path, metadata, asm,

                # args
                args, sig_key, non_constexpr_arg_values,
                ret_ptr, test_inputs, test_outputs,

                # kernel args
                grid_0, grid_1, grid_2, stream,

                # enter_hook, exit_hook,
                None, None,

                path, n_test_samples,

                # mp
                queue,
            ))
        process.start()
        process.join()
        ok = queue.get()

        if ok:
            data = read_data(path)
            rankings[run] = (data['final_perf'], path)

    # find best perf
    best_run = -1
    best_perf = -1
    for runs, (perf, _) in rankings.items():
        if perf > best_perf:
            best_run = runs
            best_perf = perf

    logger.info(f'best run: {best_run}, perf: {best_perf:.2f}')
    if best_run == -1:
        # all search fail
        opt_bin = fgk_CompiledKernel(so_path, metadata, asm)
    else:
        _, path = rankings[best_run]
        data = read_data(path)
        opt_asm = {
            'cubin': data['cubin'],
        }
        opt_bin = fgk_CompiledKernel(so_path, metadata, opt_asm)
    return opt_bin
