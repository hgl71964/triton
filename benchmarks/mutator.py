import tempfile
from functools import lru_cache

from copy import deepcopy
from io import StringIO, BytesIO

import torch

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction

from CuAsm.CuAsmParser import CuAsmParser
from CuAsm.CubinFile import CubinFile

class Sample:
    def __init__(self, kernel_section: list[str]):
        self.kernel_section = kernel_section

    def __eq__(self, other):
        if not isinstance(other, Sample):
            return False
        if not len(self.kernel_section) == len(other.kernel_section):
            return False

        # an optimization for approximate equality
        # for i in range(len(self.kernel_section)):
        for i in range(1000):  
            if i > len(self.kernel_section):
                break
            if not self.kernel_section[i] == other.kernel_section[i]:
                return False
        return True 

    def __hash__(self):
        # approximate hash
        concatenated_string = ''.join(self.kernel_section[:1000])
        return hash(concatenated_string)
    
    def get_mutable(self) -> list[int]:
        # determine which lines are possible to mutate
        # e.g. LDG, STG, and they should not cross the boundary of a label or 
        # LDGDEPBAR or BAR.SYNC or rw dependencies
        self.candidates = []  # list of index mutable
        for i, line in enumerate(self.kernel_section):
            line = line.strip()
            # skip headers
            if len(line) > 0 and line[0]=='[':  
                _, _, opcode, _, _ = self.decode(line)
                if opcode in ['LDG', 'STG', 'LDS', 'LDSM']:
                    self.candidates.append(i)
        
        # dimension of the optimization problem
        self.dims = len(self.candidates)
        return self.candidates
    
    def mutate(self, mutations):
        for lineno, action in zip(self.candidates, mutations):
            if action == -1:
                self.kernel_section[lineno-1], self.kernel_section[lineno] = self.kernel_section[lineno], self.kernel_section[lineno-1]
            elif action == 0:
                pass
            elif action == 1:
                self.kernel_section[lineno], self.kernel_section[lineno+1] = self.kernel_section[lineno+1], self.kernel_section[lineno]
            else:
                assert False, f'invalid action: {action}'
    
    def decode(self, line: str):
        line = line.split(' ')
        n = len(line)

        ctrl_code = line[0]
        comment = None
        opcode = None
        dest = None
        src = []

        idx = -1
        for i in range(1, n):
            if line[i] != '':
                idx=i
                comment = line[i]
                break
        assert idx > 0, f'no comment: {line}'

        for i in range(idx+1, n):
            if line[i] != '':
                opcode = line[i]
                idx=i
                break

        for i in range(idx+1, n):
            if line[i] != '':
                dest = line[i].strip(',')
                idx=i
                break

        for i in range(idx+1, n):
            if line[i]==';':
                break

            if line[i] != '':
                src.append(line[i])

        return ctrl_code, comment, opcode, dest, src
    
    def decode_ctrl_code(self, ctrl_code: str):
        ctrl_code = ctrl_code.split(':')
        assert len(ctrl_code) == 5, f'invalid ctrl code: {ctrl_code}'

        barr = ctrl_code[0]
        read = ctrl_code[1]
        write = ctrl_code[2]
        yield_flag = ctrl_code[3]
        stall_count = ctrl_code[4]
        return barr, read, write, yield_flag, stall_count



class MutationEngine:
    def __init__(self, 
                cf,
                kernel_func: callable,
                updater: callable,
                bench_args: tuple,
                test_args: list[tuple],
                ref_outs: list,
                config: dict,
            ):
        # get sass
        text_buffer_1, text_buffer_2 = cf.dump_sass()
        sass = text_buffer_1.getvalue().split('\n')
        kernel_section = text_buffer_2.getvalue().split('\n')

        # in-memory sass text TODO assme only one kernel
        kernel_label = None
        kernel_start_line = None
        for i, line in enumerate(kernel_section):
            if '.text.' in line:
                kernel_label = line
                kernel_start_line = i
        assert kernel_label is not None, f'Could not find kernel label'

        start_line = None
        for i, line in enumerate(sass):
            if kernel_label == line:
                start_line = i 
                break
        assert start_line is not None, f'Could not find start line'

        end_line = None
        line = start_line
        k_line = kernel_start_line
        while line < len(sass) and k_line < len(kernel_section):
            if kernel_section[k_line] != sass[line]:
                end_line = line
                break
            k_line+=1
            line+=1

        if end_line is None:
            assert sass[line-1] == kernel_section[k_line-1], f'{sass[end_line]} vs {kernel_section[k_line-1]}'
            end_line = line-1

        self.start_line = start_line
        self.end_line = end_line
        self.sass = sass
        self.kernel_section = kernel_section

        self.cf = cf
        self.kernel_func = kernel_func

        self.bench_args = bench_args
        assert len(test_args) == len(ref_outs), f'{len(test_args)} vs {len(ref_outs)}'
        self.test_args = test_args
        self.ref_outs = ref_outs
        self.config = config

        self.updater = updater
    
    def create_sample(self):
        return Sample(self.kernel_section)

    @lru_cache(maxsize=1000)
    def get_perf(self, sample: Sample):
        mutated_kernel = deepcopy(sample.kernel_section)
        mutated_sass = deepcopy(self.sass)
        mutated_sass[self.start_line:self.end_line+1] = mutated_kernel

        # assemble and update
        cap = CuAsmParser()
        cap.parse_from_buffer(mutated_sass)
        cubin: BytesIO = cap.dump_cubin()
        self.updater(*self.bench_args, cubin)

        ## XXX NOT test here to allow possible intermediate incorrect results

        # BENCH
        try:
            warmup = self.config['warmup']
            rep = self.config['rep']
            fn = lambda: self.kernel_func(*self.bench_args)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        except Exception as e:
            ms = -1
            raise e  # which should catch?

        total_flops = self.config.get('total_flops', None)

        if total_flops is not None:
            tflops = total_flops / ms * 1e-9
            print(f'total_flops: {total_flops:.0f}; ms: {ms:.3f}; tflops: {tflops:.3f};')
            return tflops
        

        raise 
        print(f'ms: {ms:.3f};')
        return -ms

    def verify(self, candidate):
        raise

        ## TEST 
        atol=self.config.get('atol', 1e-5)
        for test_args, ref_out in zip(self.test_args, self.ref_outs):
            tri_out = self.kernel_func(*test_args)
            assert torch.allclose(ref_out, tri_out, atol=atol, rtol=0)