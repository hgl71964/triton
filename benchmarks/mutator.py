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


class MutationEngine:

    def __init__(
        self,
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
            k_line += 1
            line += 1

        if end_line is None:
            assert sass[line - 1] == kernel_section[
                k_line - 1], f'{sass[end_line]} vs {kernel_section[k_line-1]}'
            end_line = line - 1

        self.start_line = start_line
        self.kernel_start_line = kernel_start_line
        self.end_line = end_line
        self.sass = sass
        self.kernel_section = kernel_section

        self.cf = cf
        self.kernel_func = kernel_func

        self.bench_args = bench_args
        assert len(test_args) == len(
            ref_outs), f'{len(test_args)} vs {len(ref_outs)}'
        self.test_args = test_args
        self.ref_outs = ref_outs
        self.config = config

        self.updater = updater

    def decode(self, line: str):
        line = line.strip('\n')
        line = line.split(' ')
        n = len(line)

        ctrl_code = None
        predicate = None
        comment = None
        opcode = None
        dest = None
        src = []

        idx = -1
        for i in range(0, n):
            if line[i] != '':
                idx = i
                ctrl_code = line[i]
                break
        assert idx > -1, f'no ctrl: {line}'

        for i in range(idx + 1, n):
            if line[i] != '':
                idx = i
                comment = line[i]
                break

        for i in range(idx + 1, n):
            if line[i] != '':

                if line[i][0] == '@':
                    predicate = line[i]
                else:
                    opcode = line[i]

                idx = i
                break

        if opcode is None:
            for i in range(idx + 1, n):
                if line[i] != '':
                    opcode = line[i]
                    idx = i
                    break

        for i in range(idx + 1, n):
            if line[i] != '':
                dest = line[i].strip(',')
                idx = i
                break

        for i in range(idx + 1, n):
            if line[i] == ';':
                break

            if line[i] != '':
                src.append(line[i])

        return ctrl_code, comment, predicate, opcode, dest, src

    def decode_ctrl_code(self, ctrl_code: str):
        ctrl_code = ctrl_code.split(':')
        assert len(ctrl_code) == 5, f'invalid ctrl code: {ctrl_code}'

        barr = ctrl_code[0]
        read = ctrl_code[1]
        write = ctrl_code[2]
        yield_flag = ctrl_code[3]
        stall_count = ctrl_code[4]
        return barr, read, write, yield_flag, stall_count

    @lru_cache(maxsize=1000)
    def get_perf(self, sample):
        mutated_kernel = sample.kernel_section[self.kernel_start_line:]
        mutated_sass = deepcopy(self.sass)
        mutated_sass[self.start_line:self.end_line + 1] = mutated_kernel

        # buffer IO
        cap = CuAsmParser()
        cap.parse_from_buffer(mutated_sass)
        cubin = cap.dump_cubin()
        self.updater(*self.bench_args, cubin)

        # file IO
        # with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file:
        #     content_to_write = '\n'.join(mutated_sass)
        #     temp_file.write(content_to_write)
        #     # Ensure data is written to the file before reading it
        #     temp_file.flush()
        #     temp_file.seek(0)
        #     # file_content = temp_file.read()
        #     sass_path = temp_file.name
        #     cap = CuAsmParser()
        #     cap.parse(sass_path)
        #     cubin_stream = BytesIO()
        #     cap.saveAsCubin(cubin_stream)
        #     cubin = cubin_stream.getvalue()
        #     self.updater(*self.bench_args, cubin)

        ## XXX NOT test here to allow possible intermediate incorrect results

        # BENCH
        try:
            warmup = self.config['warmup']
            rep = self.config['rep']
            fn = lambda: self.kernel_func(*self.bench_args)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        except ZeroDivisionError as zero_err:
            # Catch a specific exception (ZeroDivisionError in this case)
            print(f"Caught a ZeroDivisionError: {zero_err}")
            ms = -1
        except Exception as e:
            print(f'Run error: {e}')
            ms = -1

        total_flops = self.config.get('total_flops', None)

        if total_flops is not None:
            tflops = total_flops / ms * 1e-9
            sample.perf = tflops
            # print(f'total_flops: {total_flops:.0f}; ms: {ms:.3f}; tflops: {tflops:.3f};')
            print(f'ms: {ms:.3f}; tflops: {tflops:.3f};')
            return tflops

        raise RuntimeError(f'no total_flops')

        print(f'ms: {ms:.3f};')
        return ms

    def verify(self, candidate):
        raise

        ## TEST
        atol = self.config.get('atol', 1e-5)
        for test_args, ref_out in zip(self.test_args, self.ref_outs):
            tri_out = self.kernel_func(*test_args)
            assert torch.allclose(ref_out, tri_out, atol=atol, rtol=0)


def main():

    def decode(line: str):
        line = line.strip('\n')
        line = line.split(' ')
        n = len(line)

        ctrl_code = None
        predicate = None
        comment = None
        opcode = None
        dest = None
        src = []

        idx = -1
        for i in range(0, n):
            if line[i] != '':
                idx = i
                ctrl_code = line[i]
                break
        assert idx > 0, f'no ctrl: {line}'

        for i in range(idx + 1, n):
            if line[i] != '':
                idx = i
                comment = line[i]
                break

        for i in range(idx + 1, n):
            if line[i] != '':

                if line[i][0] == '@':
                    predicate = line[i]
                else:
                    opcode = line[i]

                idx = i
                break

        if opcode is None:  # if exists predicate
            for i in range(idx + 1, n):
                if line[i] != '':
                    opcode = line[i]
                    idx = i
                    break

        for i in range(idx + 1, n):
            if line[i] != '':
                dest = line[i].strip(',')
                idx = i
                break

        for i in range(idx + 1, n):
            if line[i] == ';':
                break

            if line[i] != '':
                src.append(line[i])

        return ctrl_code, comment, predicate, opcode, dest, src

    # l = ".CUASM_OFFSET_LABEL._attn_fwd_0d1d2d34d5d6de7de8de9c10de11de12de13c14de15de16de17c18de19de20de21c22c23de.EIATTR_COOP_GROUP_INSTR_OFFSETS.#:"
    # l = " .L_x_2:"   # for label line, label is in ctrl_code
    # l = '      [B------:R-:W0:-:S01]         /*1080*/                   LDGDEPBAR ;\n'   # LDGDEPBAR is opcode and dest is ;
    # out = decode(l)

    with open('data/test.cuasm', 'r') as f:
        ls = f.readlines()

    for l in ls:
        out = decode(l)
        ctrl_code, comment, predicate, opcode, dest, src = out


if __name__ == '__main__':
    main()
