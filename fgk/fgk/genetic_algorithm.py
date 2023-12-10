import os
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
from sample import Sample
from fgk.utils.logger import get_logger

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


def create_population(pop_size: int,
                      init_sample: GeneticSample) -> list[GeneticSample]:
    population = []
    for _ in range(pop_size):
        sample = GeneticSample(init_sample.kernel_section, init_sample.engine)
        mutable = sample.get_mutable()
        n = len(mutable)
        indexes = [i for i in range(n)]
        actions = [random.randint(-1, 1) for _ in range(len(mutable))]
        sample.apply_all(indexes, actions)
        population.append(sample)
    return population


def tournament_selection(
        population: list[GeneticSample], tournament_size,
        eng: MutationEngine) -> list[GeneticSample]:
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
            individual[i] = random.randint(-1, 1)  # Assign a new random value
    return individual


def build_child(
    parent: GeneticSample,
    mutated_action: list[int],
    eng: MutationEngine,
):
    child = GeneticSample(parent.kernel_section, eng)
    child.candidates = deepcopy(parent.candidates)
    child.dims = parent.dims
    child.apply_all([i for i in range(len(mutated_action))], mutated_action)
    return child


def crossover(
    parent1: GeneticSample,
    parent2: GeneticSample,
    mutation_rate: float,
    eng: MutationEngine,
):

    # crossover_point = random.randint(0, len(parent1) - 1)
    # child1 = parent1[:crossover_point] + parent2[crossover_point:]
    # child2 = parent2[:crossover_point] + parent1[crossover_point:]
    # return child1, child2

    # TODO mutation should base on a common set of kernel_section

    n = parent1.dims
    crossover_point = random.randint(0, n - 1)
    crossover_action1 = parent1.actions[:crossover_point] + parent2.actions[
        crossover_point:]
    crossover_action2 = parent2.actions[:crossover_point] + parent1.actions[
        crossover_point:]

    mutated_action1 = mutation(crossover_action1, mutation_rate)
    mutated_action2 = mutation(crossover_action2, mutation_rate)

    child1 = build_child(parent1, mutated_action1, eng)
    child2 = build_child(parent1, mutated_action2, eng)
    child3 = build_child(parent2, mutated_action1, eng)
    child4 = build_child(parent2, mutated_action2, eng)
    return child1, child2, child3, child4


def genetic_algorithm(
    init_sample,
    population_size,
    generations,
    tournament_size,
    mutation_rate,
    eng: MutationEngine,
):
    population = create_population(population_size, init_sample)

    _t1 = time.perf_counter()
    for generation in range(generations):
        selected_population = tournament_selection(
            population, tournament_size, eng)

        cmp = lambda x: x if x is not None else float("-inf")
        perfs = [x.perf for x in selected_population]
        logger.info(
            f'generation {generation}: best perf: {max(perfs, key=cmp):.2f}')

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2, child3, child4 = crossover(
                parent1, parent2, mutation_rate, eng)
            new_population.extend([child1, child2, child3, child4])

        population = new_population

    best_individual = max(population, key=eng.get_perf)
    _t2 = time.perf_counter()
    hours = (_t2 - _t1) / 3600
    logger.info(
        f'Performance: {eng.get_perf(best_individual)}; search time: {hours:.2f}h'
    )
    return best_individual


def run_genetic_algorithm(
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
    population_size=100,
    generations=500,
    tournament_size=5,
    mutation_rate=0.1,

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
    sample = GeneticSample(eng.kernel_section, eng)
    init_perf = eng.get_perf(sample)
    logger.info(f'init perf: {init_perf:.2f}')

    best_sample = genetic_algorithm(
        sample,
        population_size,
        generations,
        tournament_size,
        mutation_rate,
        eng,
    )
    # ===== test =====
    # TODO
