import os
import time
import pickle
import tempfile

import torch

from CuAsm.CubinFile import CubinFile
from CuAsm.CuAsmParser import CuAsmParser

from fgk.compiler import CompiledKernel as fgk_CompiledKernel
from fgk.verify import test_via_cubin
from fgk.utils.logger import get_logger
from fgk.mutator import MutationEngine

logger = get_logger(__name__)


# YAPF: disable
def run_selection(
    so_path, metadata, asm,

    # args
    args, sig_key, non_constexpr_arg_values,
    ret_ptr, test_inputs, test_outputs,

    # kernel args
    grid_0, grid_1, grid_2, stream,

    enter_hook, exit_hook,

    cubin_dir_path, n_test_samples,
):

    rankings = {}
    if cubin_dir_path.endswith('.cubin'):

        # NOTE: we want the kernel section substituted by the hacked cubin;
        # and everything else from the standard cubin
        with open(cubin_dir_path, 'rb') as f:
            hacked_cubin = f.read()

        with tempfile.NamedTemporaryFile(mode='wb', delete=True) as temp_file:
            temp_file.write(hacked_cubin)
            temp_file.flush()
            temp_file.seek(0)
            time.sleep(1)
            hacked_cf = CubinFile(temp_file.name)

        with tempfile.NamedTemporaryFile(mode='wb', delete=True) as temp_file:
            standard_cubin = asm['cubin']
            temp_file.write(standard_cubin)
            temp_file.flush()
            temp_file.seek(0)
            time.sleep(1)
            standard_cf = CubinFile(temp_file.name)

        hacked_eng = MutationEngine(
            None, hacked_cf, None,
            None, None, None, None, None, None, None,
        )
        standard_eng = MutationEngine(
            None, standard_cf, None,
            None, None, None, None, None, None, None,
        )

        mutated_kernel = hacked_eng.kernel_section[hacked_eng.kernel_start_line:]
        standard_sass = standard_eng.sass
        standard_sass[standard_eng.start_line:standard_eng.end_line + 1] = mutated_kernel

        # assemble
        cap = CuAsmParser()
        cap.parse_from_buffer(standard_sass)
        cubin = cap.dump_cubin()

    else:
        for i, fn in enumerate(os.listdir(cubin_dir_path)):
            if fn == 'cache_config.pkl':
                continue
            if not fn.endswith('.pkl'):
                continue

            with open(os.path.join(cubin_dir_path, fn), 'rb') as f:
                data = pickle.load(f)
            rankings[i] = data

        if len(rankings) == 0:
            raise RuntimeError('no valid cubin found')

        # for run, data in sorted(rankings.items(),
        #                         key=lambda x: x[1]['final_perf'],
        #                         reverse=True):
        #     print(f'run {run}; perf: {data["final_perf"]:.2f}:{data["init_perf"]:.2f}')

        # start verify from the best
        for run, data in sorted(rankings.items(),
                                key=lambda x: x[1]['final_perf'],
                                reverse=True):
            cubin = data['cubin']
            try:
                ok = test_via_cubin(
                    so_path,
                    metadata,
                    asm,

                    # args
                    args,
                    sig_key,
                    non_constexpr_arg_values,
                    ret_ptr,
                    test_inputs,
                    test_outputs,

                    # kernel args
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,

                    # enter_hook, exit_hook,
                    None,
                    None,
                    cubin,
                    n_test_samples,
                )
            except Exception as e:
                logger.warning(f'run {run} verify failed: {e}')
                ok = False

            torch.cuda.empty_cache()  # free test memory
            final_perf = data['final_perf']
            init_perf = data['init_perf']
            improvement = (final_perf - init_perf) / init_perf
            if ok:
                logger.info(f'run {run} verified ok; final perf: {final_perf:.2f}; init perf: {init_perf:.2f}; improvement: {improvement*100:.2f}%')
                break
            else:
                logger.warning(f'run {run} verified failed; final perf: {final_perf:.2f}; init perf: {init_perf:.2f}; improvement: {improvement*100:.2f}%')


    opt_asm = {
        'cubin': cubin,
    }
    opt_bin = fgk_CompiledKernel(so_path, metadata, opt_asm)
    return opt_bin
