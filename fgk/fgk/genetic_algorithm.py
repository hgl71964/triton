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
from fgk.sample import Sample, CtrlSample
from fgk.utils.logger import get_logger
from fgk.utils.record import save_data

logger = get_logger(__name__)


class GeneticSample(Sample):

    def apply(self, index, action):
        lineno = self.candidates[index]
        if action == -1:
            try:
                self.kernel_section[lineno - 1], self.kernel_section[
                    lineno] = self.kernel_section[lineno], self.kernel_section[
                        lineno - 1]
                self.candidates[index] -= 1
            except IndexError as e:
                # it is possible that the index is out of range during mutation
                print(f'IndexError: {lineno}; {len(self.kernel_section)};')
        elif action == 1:
            try:
                self.kernel_section[lineno], self.kernel_section[
                    lineno +
                    1] = self.kernel_section[lineno +
                                             1], self.kernel_section[lineno]
                self.candidates[index] += 1
            except IndexError as e:
                print(f'IndexError: {lineno}; {len(self.kernel_section)};')
        elif action == 0:
            pass
        else:
            assert False, f'invalid action: {action}'


def create_population(
    pop_size: int,
    init_sample: Sample,
    sample_ctr: Union[CtrlSample, GeneticSample],
) -> Union[CtrlSample, GeneticSample]:
    population = []
    for _ in range(pop_size):
        sample = sample_ctr(init_sample.kernel_section, init_sample.engine)
        mutable = sample.get_mutable()
        n = len(mutable)
        indexes = [i for i in range(n)]
        # actions = [random.randint(-1, 1) for _ in range(n)]
        actions = [random.choice([0, 1]) for _ in range(n)]
        sample.apply_all(indexes, actions)
        population.append(sample)
    return population


def tournament_selection(population: list[Sample], tournament_size,
                         eng: MutationEngine) -> list[Sample]:
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        best_individual = max(tournament, key=eng.get_perf)
        selected.append(best_individual)

    # TODO what if all selected are invalid
    find = False
    for sample in selected:
        if sample.perf > 0:
            find = True
            break

    if not find:
        print('all selected are invalid')

    return selected


def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]  # Flip 0 to 1 or 1 to 0
    return individual


def build_child(
    parent: GeneticSample,
    mutated_action: list[int],
    sample_ctr: Union[CtrlSample, GeneticSample],
    eng: MutationEngine,
):
    child = sample_ctr(parent.kernel_section, eng)
    child.candidates = deepcopy(parent.candidates)
    child.dims = parent.dims
    child.apply_all([i for i in range(len(mutated_action))], mutated_action)
    return child


def crossover(
    parent1: GeneticSample,
    parent2: GeneticSample,
    mutation_rate: float,
    sample_ctr: Union[CtrlSample, GeneticSample],
    eng: MutationEngine,
):

    # crossover_point = random.randint(0, len(parent1) - 1)
    # child1 = parent1[:crossover_point] + parent2[crossover_point:]
    # child2 = parent2[:crossover_point] + parent1[crossover_point:]
    # return child1, child2

    n = parent1.dims
    crossover_point = random.randint(0, n - 1)
    crossover_action1 = parent1.actions[:crossover_point] + parent2.actions[
        crossover_point:]
    crossover_action2 = parent2.actions[:crossover_point] + parent1.actions[
        crossover_point:]

    mutated_action1 = mutation(crossover_action1, mutation_rate)
    mutated_action2 = mutation(crossover_action2, mutation_rate)

    child1 = build_child(parent1, mutated_action1, sample_ctr, eng)
    child2 = build_child(parent2, mutated_action2, sample_ctr, eng)
    return child1, child2


def genetic_algorithm(
    init_sample,
    population_size,
    generations,
    tournament_size,
    mutation_rate,
    sample_ctr: Union[CtrlSample, GeneticSample],
    eng: MutationEngine,
):
    population = create_population(population_size, init_sample, sample_ctr)

    for generation in range(generations):
        selected_population = tournament_selection(population, tournament_size,
                                                   eng)

        cmp = lambda x: x if x is not None else float("-inf")
        perfs = [x.perf for x in selected_population]
        logger.info(
            f'generation {generation}: best perf: {max(perfs, key=cmp):.2f}')

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover(
                parent1,
                parent2,
                mutation_rate,
                sample_ctr,
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
    warmup=100,
    rep=100,
):
    logger.info(
        'run genetic algorithm with population_size %d; generations %d; tournament_size %d; mutation_rate %f ',
        population_size, generations, tournament_size, mutation_rate)

    print(f'bin id {id(bin)}')

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
    # sample = GeneticSample(eng.kernel_section, eng)
    sample = CtrlSample(eng.kernel_section, eng)
    init_perf = eng.get_perf(sample)
    logger.info(f'init perf: {init_perf:.2f}')
    if init_perf < 0:
        raise RuntimeError(f'init perf {init_perf} < 0; not valid cubin')

    _t1 = time.perf_counter()
    best_sample = genetic_algorithm(
        sample,
        population_size,
        generations,
        tournament_size,
        mutation_rate,
        CtrlSample,
        eng,
    )
    _t2 = time.perf_counter()
    hours = (_t2 - _t1) / 3600

    final_perf = eng.get_perf(best_sample)
    logger.info(
        f'Performance: {final_perf:.2f}; init perf: {init_perf:.2f}; search time: {hours:.2f}h'
    )
    logger.info(
        f'improvement: {(final_perf - init_perf) / init_perf * 100:.2f}%')

    # mutation fails
    if init_perf > final_perf:
        best_sample = sample

    # ===== test =====
    # TODO

    # ===== save =====
    eng.assemble(best_sample)
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
        algo='ga',
    )
    return bin
