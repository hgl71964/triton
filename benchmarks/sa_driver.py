import os
import math
import random
import tempfile
import time
from copy import deepcopy

import numpy as np
import torch

# asm
from CuAsm.CubinFile import CubinFile

# mutation
from mutator import MutationEngine
from sample import Sample
from logger import get_logger

# kernel
from search_attn import _attn_fwd, get_cubin, set_cubin, attn_forward

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
flags.DEFINE_integer("n_choices", 1, "+-n choices")
# sa
flags.DEFINE_integer("max_iterations", 1000, "")
flags.DEFINE_float("temperature", 1.0, "")
flags.DEFINE_float("cooling_rate", 0.003, "")
flags.DEFINE_float("noise_factor", 0.1, "")
flags.DEFINE_string("policy", "single", "mutation policy; single or all")

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


def acceptance_probability(old_fitness, new_fitness, temperature):
    noise = random.uniform(-FLAGS.noise_factor, FLAGS.noise_factor)
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
        if acceptance_probability(current_fitness, new_fitness,
                                  temperature) > random.random():
            current_solution = new_solution
            current_fitness = new_fitness

        temperature *= 1 - cooling_rate
        cnt += 1

        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = current_solution

    return best_solution, best_fitness


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
                        device="cuda").normal_(mean=0.0, std=0.5)
        k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype,
                        device="cuda").normal_(mean=0.0, std=0.5)
        v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype,
                        device="cuda").normal_(mean=0.0, std=0.5)

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

    # get initial cubin and asm (the initial have to file IO)
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
    cf = CubinFile(binname)
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
    initial_solution = SimulatedSample(eng.kernel_section, eng)

    _t1 = time.perf_counter()

    # Run simulated annealing
    best_solution, best_fitness = simulated_annealing(
        initial_solution,
        FLAGS.n_choices,
        FLAGS.max_iterations,
        FLAGS.temperature,
        FLAGS.cooling_rate,
        FLAGS.policy,
        eng,
    )

    _t2 = time.perf_counter()
    hours = (_t2 - _t1) / 3600
    logger.info(f'Performance: {best_fitness:.2f}; Search time: {hours:.2f}h')


if __name__ == "__main__":
    app.run(main)
