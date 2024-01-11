import os
import random
import tempfile
import time
from copy import deepcopy
from typing import Union

import numpy as np
import torch

# asm (add CuAsm to PYTHONPATH!)
from CuAsm.CubinFile import CubinFile

# mutation
from fgk.mutator import MutationEngine
from fgk.sample import Sample
from fgk.utils.logger import get_logger
from fgk.utils.record import save_data

logger = get_logger(__name__)


class CtrlSample(Sample):
    """ only Mutate the control code """

    def get_mutable(self) -> list[int]:
        lines = []
        self.actions = []
        for i, line in enumerate(self.kernel_section):
            line = line.strip()
            # skip headers
            if len(line) > 0 and line[0] == '[':
                out = self.engine.decode(line)
                ctrl_code, _, _, _, _, _ = out

                out = self.engine.decode_ctrl_code(ctrl_code)
                _, _, _, yield_flag, stall_count = out
                if yield_flag is None or stall_count is None:
                    continue

                # 1. existing yield flag
                # if yield_flag == 'Y':
                #     self.candidates.append(i)
                #     lines.append(line)
                # elif int(stall_count[-2]) > 3:
                #     self.candidates.append(i)
                #     lines.append(line)

                # 2. consider all flag
                # TODO: can filter out e.g. NOP?
                self.candidates.append(i)
                lines.append(line)
                if yield_flag == 'Y':
                    self.actions.append(1)
                elif yield_flag == '-':
                    self.actions.append(0)
                else:
                    raise RuntimeError(f'invalid yield flag: {yield_flag}')

        # dimension of the optimization problem
        self.dims = len(self.candidates)
        return self.candidates

    def apply(self, index, action):
        lineno = self.candidates[index]
        line = self.kernel_section[lineno]
        self.kernel_section[lineno] = self._set_yield_for_line(line, action)

    def _set_yield_for_line(self, line: str, action: int):
        # NOTE: the line is guranteed to be a valid asm line
        index = -1
        for i, char in enumerate(line):
            if char == 'S':
                index = i
                break

        if index == -1:
            raise RuntimeError(f'invalid line: {line}')

        # print(f'action: {action}')
        # print(f'before action: {line}')
        if action == 1:
            line = line[:index - 2] + 'Y' + line[index - 1:]
        elif action == 0:
            line = line[:index - 2] + '-' + line[index - 1:]
        else:
            raise RuntimeError(f'invalid action: {action}')
        # print(f"after action: {line}")

        return line


def create_population(
    pop_size: int,
    init_sample: CtrlSample,
) -> CtrlSample:
    population = []
    for _ in range(pop_size):
        sample = CtrlSample(init_sample.kernel_section, init_sample.engine)

        ## 1. random init
        # mutable = sample.get_mutable()
        # n = len(mutable)
        # indexes = [i for i in range(n)]
        # actions = [random.choice([0, 1]) for _ in range(n)]
        # sample.apply_all(indexes, actions)

        ## 2. init from default
        sample.get_mutable()
        population.append(sample)
    return population


def tournament_selection(
    population: list[CtrlSample],
    tournament_size,
    eng: MutationEngine,
) -> list[CtrlSample]:
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        best_individual = max(tournament, key=eng.get_perf)
        selected.append(best_individual)

    # sanity check
    find = False
    for sample in selected:
        if sample.perf > 0:
            find = True
            break
    if not find:
        raise RuntimeError(f'all selected are invalid')

    return selected


def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]  # Flip 0 to 1 or 1 to 0
    return individual


def build_child(
    parent: CtrlSample,
    mutated_action: list[int],
    eng: MutationEngine,
):
    child = CtrlSample(parent.kernel_section, eng)
    child.get_mutable()
    child.apply_all([i for i in range(len(mutated_action))], mutated_action)
    return child


def crossover(
    parent1: CtrlSample,
    parent2: CtrlSample,
    mutation_rate: float,
    eng: MutationEngine,
):
    n = parent1.dims
    crossover_point = random.randint(0, n - 1)
    crossover_action1 = parent1.actions[:crossover_point] + parent2.actions[
        crossover_point:]
    crossover_action2 = parent2.actions[:crossover_point] + parent1.actions[
        crossover_point:]

    mutated_action1 = mutation(crossover_action1, mutation_rate)
    mutated_action2 = mutation(crossover_action2, mutation_rate)

    child1 = build_child(parent1, mutated_action1, eng)
    child2 = build_child(parent2, mutated_action2, eng)
    return child1, child2


def genetic_algorithm(
    init_sample,
    init_perf,
    population_size,
    generations,
    tournament_size,
    mutation_rate,
    eng: MutationEngine,
):
    cmp = lambda x: x if x is not None else float("-inf")
    population = create_population(population_size, init_sample)

    for generation in range(generations):
        selected_population = tournament_selection(
            population,
            tournament_size,
            eng,
        )

        perfs = [x.perf for x in selected_population]
        logger.info(
            f'generation {generation}: best perf: {max(perfs, key=cmp):.2f}; init perf: {init_perf:.2f}'
        )

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover(
                parent1,
                parent2,
                mutation_rate,
                eng,
            )
            new_population.extend([child1, child2])

        population = new_population

    best_individual = max(population, key=eng.get_perf)
    return best_individual


def run_genetic_algorithm(
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
    population_size=100,
    generations=500,
    tournament_size=5,
    mutation_rate=0.1,

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
        'run genetic algorithm with population_size %d; generations %d; tournament_size %d; mutation_rate %f ',
        population_size, generations, tournament_size, mutation_rate)

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
    sample = CtrlSample(eng.kernel_section, eng)
    sample.get_mutable()
    init_perf = max([eng.get_init_perf() for _ in range(3)])
    logger.info(f'init perf: {init_perf:.2f}; dims: {sample.dims}')
    if init_perf < 0:
        raise RuntimeError(f'init perf {init_perf} < 0; not valid cubin')

    _t1 = time.perf_counter()
    best_sample = genetic_algorithm(
        sample,
        init_perf,
        population_size,
        generations,
        tournament_size,
        mutation_rate,
        eng,
    )
    _t2 = time.perf_counter()
    hours = (_t2 - _t1) / 3600

    # if init perf is > 0, it is unlikely this gives invalid cubin
    final_perf = eng.assemble(best_sample)

    # mutation fails
    if init_perf > final_perf:
        best_sample = sample
        final_perf = eng.assemble(best_sample)

    logger.info(
        f'Performance: {final_perf:.2f}; init perf: {init_perf:.2f}; search time: {hours:.2f}h'
    )
    logger.info(
        f'improvement: {(final_perf - init_perf) / init_perf * 100:.2f}%')

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
        algo='ga',
    )
