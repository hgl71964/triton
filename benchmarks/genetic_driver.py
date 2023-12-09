import os
import random
import tempfile
import time
from copy import deepcopy

import numpy as np
import torch

from CuAsm.CubinFile import CubinFile

# utils
from search_attn import _attn_fwd, get_cubin, set_cubin, attn_forward
from mutator import MutationEngine
from sample import Sample

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# kernel
flags.DEFINE_string("default_out_path", "data", "output dir")
flags.DEFINE_integer("dump", 0, "whether to dump")
flags.DEFINE_integer("hack", 0, "whether to hack")
flags.DEFINE_string("fn", None, "cubin name to load")
flags.DEFINE_integer("flash", 0, "whether to use flash attention")
flags.DEFINE_integer("seed", 1337, "")
flags.DEFINE_integer("test_sample", 10, "")
# genetic
flags.DEFINE_integer("population_size", 100, "")
flags.DEFINE_integer("generations", 50, "")
flags.DEFINE_float("mutation_rate", 0.1, "")
flags.DEFINE_integer("tournament_size", 5, "")


class GA_Sample(Sample):

    def __init__(self, kernel_section: list[str], engine):
        self.kernel_section = deepcopy(kernel_section)
        self.engine = engine

        self.candidates = []  # list of index mutable
        self.dims = None
        self._perf = None
        self.actions = []

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

    def apply_all(self, indexes, actions):
        self.actions = actions
        for index, action in zip(indexes, actions):
            self.apply(index, action)


def create_population(pop_size: int, eng: MutationEngine) -> list[GA_Sample]:
    population = []
    for _ in range(pop_size):
        sample = GA_Sample(eng.kernel_section, eng)
        mutable = sample.get_mutable()
        n = len(mutable)
        indexes = [i for i in range(n)]
        actions = [random.randint(-1, 1) for _ in range(len(mutable))]
        sample.apply_all(indexes, actions)
        population.append(sample)
    return population


def tournament_selection(
        population: list[GA_Sample], tournament_size,
        eng: MutationEngine) -> list[GA_Sample]:
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        best_individual = max(tournament, key=eng.get_perf)
        selected.append(best_individual)

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
    parent: GA_Sample,
    mutated_action: list[int],
    eng: MutationEngine,
):
    child = GA_Sample(parent.kernel_section, eng)
    child.candidates = deepcopy(parent.candidates)
    child.dims = parent.dims
    child.apply_all([i for i in range(len(mutated_action))], mutated_action)
    return child


def crossover(
    parent1: GA_Sample,
    parent2: GA_Sample,
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


def main(_):
    # ===== seed =====
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True

    # ===== workload =====
    ## kernel = _attn_fwd
    kernel = attn_forward
    fn = kernel.__name__
    sm_scale = 0.5
    causal = True
    Z, H, N_CTX, D_HEAD = 1, 32, 4096, 64
    dtype = torch.float16

    test_args = []
    ref_outs = []

    ## reference
    for _ in range(FLAGS.test_sample):
        q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype,
                        device="cuda").normal_(
                            mean=0.0, std=0.5)
        k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype,
                        device="cuda").normal_(
                            mean=0.0, std=0.5)
        v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype,
                        device="cuda").normal_(
                            mean=0.0, std=0.5)

        M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        if causal:
            p[:, :, M == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).half()
        ref_out = torch.matmul(p, v)

        test_args.append((q, k, v, causal, sm_scale))
        ref_outs.append(ref_out)

    bench_args = test_args[0]

    # ===== config =====
    warmup = 100
    rep = 100
    flops_per_matmul = 2.0 * Z * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    config = {
        'atol': 1e-2,
        "total_flops": total_flops,
        'warmup': warmup,
        'rep': rep,
    }

    # population_size = FLAGS.population_size
    # generations = FLAGS.generations
    population_size = 5
    generations = 2
    mutation_rate = FLAGS.mutation_rate
    tournament_size = FLAGS.tournament_size

    # get cubin and asm (the initial have to file IO)
    cubin = get_cubin(q, k, v, causal, sm_scale)
    folder = f'{FLAGS.default_out_path}'
    file_path = f'{fn}_{0}.cubin'
    prefix = os.path.join(folder, f"tmp_{FLAGS.seed}")
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    file_path = os.path.join(prefix, file_path)
    with open(file_path, 'wb') as file:
        file.write(cubin)

    # disassemble
    time.sleep(1)
    binname = file_path
    cf = CubinFile(binname)  # read back

    eng = MutationEngine(
        cf,
        kernel,
        set_cubin,
        bench_args,
        test_args,
        ref_outs,
        config,
    )

    # ===== start =====
    _t1 = time.perf_counter()
    population = create_population(population_size, eng)

    for generation in range(generations):
        # Select individuals for reproduction
        selected_population = tournament_selection(
            population, tournament_size, eng)

        # Create next generation through crossover and mutation
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            # child1, child2 = crossover(parent1, parent2)
            # new_population.extend([child1, child2])
            child1, child2, child3, child4 = crossover(
                parent1, parent2, mutation_rate, eng)
            new_population.extend([child1, child2, child3, child4])

        # Replace old population with new population
        population = new_population

    # Find the best individual after all generations
    best_individual = max(population, key=eng.get_perf)

    _t2 = time.perf_counter()
    hours = (_t2 - _t1) / 3600
    print(
        f'Performance: {eng.get_perf(best_individual)}; search time: {hours:.2f}h'
    )


if __name__ == "__main__":
    app.run(main)
