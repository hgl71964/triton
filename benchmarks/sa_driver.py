import os
import math
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
flags.DEFINE_integer("max_iterations", 10, "")
flags.DEFINE_float("temperature", 1.0, "")
flags.DEFINE_float("cooling_rate", 0.003, "")
flags.DEFINE_float("noise_factor", 0.3, "")


class Sample:
    def __init__(self, kernel_section: list[str], engine):
        self.kernel_section = deepcopy(kernel_section)
        self.engine = engine

        self.candidates = []  # list of index mutable
        self.dims = None
        self._perf = None

    def __eq__(self, other):
        if not isinstance(other, Sample):
            return False
        if not len(self.kernel_section) == len(other.kernel_section):
            return False

        # an optimization for approximate equality
        for i in range(len(self.kernel_section)):
        # for i in range(1000):  
            if i > len(self.kernel_section):
                break
            if not self.kernel_section[i] == other.kernel_section[i]:
                return False
        return True 

    def __hash__(self):
        # approximate hash
        # concatenated_string = ''.join(self.kernel_section[:1000])
        concatenated_string = ''.join(self.kernel_section)
        return hash(concatenated_string)

    def __len__(self):
        assert self.dims is not None, f'no dims'
        return self.dims

    @property
    def perf(self):
        return self._perf

    @perf.setter
    def perf(self, value):
        self._perf = value
    
    def get_kernel_section(self):
        return self.kernel_section
    
    def get_mutable(self) -> list[int]:
        if self.dims is not None:
            return self.candidates
        
        # determine which lines are possible to mutate
        # e.g. LDG, STG, and they should not cross the boundary of a label or 
        # LDGDEPBAR or BAR.SYNC or rw dependencies
        for i, line in enumerate(self.kernel_section):
            line = line.strip()
            # skip headers
            if len(line) > 0 and line[0]=='[':  
                _, _, opcode, _, _ = self.engine.decode(line)
                if opcode in ['LDG', 'STG', 'LDS', 'LDSM']:
                    self.candidates.append(i)
        
        # dimension of the optimization problem
        self.dims = len(self.candidates)
        return self.candidates
    
    def apply(self, index, action):
        if action == -1:
            self.kernel_section[index-1], self.kernel_section[index] = self.kernel_section[index], self.kernel_section[index-1]
            self.candidates[index]-=1
        elif action == 1:
            self.kernel_section[index], self.kernel_section[index+1] = self.kernel_section[index+1], self.kernel_section[index]
            self.candidates[index]+=1
        else:
            assert False, f'invalid action: {action}'
        
        


def generate_neighbor(sample: Sample, n_choices):
    neighbor = Sample(sample.kernel_section, sample.engine)
    mutable = sample.get_mutable()
    index = random.randint(0, len(mutable) - 1)
    action = random.choice([-1, 1])

    # action = random.randint(-n_choices, n_choices)
    # mutable[index] = random.randint(-n_choices, n_choices)
    # neighbor[index] = random.randint(0, n_choices - 1)

    neighbor.apply(index, action)
    return neighbor

def acceptance_probability(old_fitness, new_fitness, temperature):
    # if new_fitness > old_fitness:
    #     return 1.0
    # return math.exp((new_fitness - old_fitness) / temperature)

    noise = random.uniform(-FLAGS.noise_factor, FLAGS.noise_factor)
    adjusted_difference = new_fitness - old_fitness + noise
    
    if adjusted_difference > 0:
        return 1.0
    
    return math.exp(adjusted_difference / temperature)

def simulated_annealing(initial_solution: Sample,
                       n_choices,
                       temperature,
                       cooling_rate,
                       eng: MutationEngine,
                    ) -> Sample:
    current_solution = initial_solution
    current_fitness = eng.get_perf(current_solution)
    
    while temperature > 0.1:
        new_solution = generate_neighbor(current_solution, n_choices)
        new_fitness = eng.get_perf(new_solution)
        
        if acceptance_probability(current_fitness, new_fitness, temperature) > random.random():
            current_solution = new_solution
            current_fitness = new_fitness
        
        temperature *= 1 - cooling_rate
    
    return current_solution, current_fitness


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
    dtype=torch.float16

    test_args = []
    ref_outs = []

    ## reference
    for _ in range(FLAGS.test_sample):
        q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
        k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
        v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)


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
    eng = MutationEngine(cf,
                    kernel,
                    set_cubin,
                    bench_args,
                    test_args,
                    ref_outs,
                    config,
                )

    # ===== start =====
    max_iterations = FLAGS.max_iterations
    temperature = FLAGS.temperature
    cooling_rate = FLAGS.cooling_rate
    n_choices = FLAGS.n_choices

    # Initialize a random solution
    initial_solution = Sample(eng.kernel_section, eng)

    _t1 = time.perf_counter()

    # Run simulated annealing
    best_solution, best_fitness = simulated_annealing(initial_solution,
                                                      n_choices, 
                                                      max_iterations, 
                                                      temperature, 
                                                      cooling_rate, 
                                                      eng,
                                                      )

    _t2 = time.perf_counter()
    hours = int((_t2 - _t1) / 3600)
    print(f'Performance: {best_fitness:.2f}; Search time: {hours:.2f}h')


if __name__ == "__main__":
    app.run(main)