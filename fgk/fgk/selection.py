import os
import pickle

from fgk.compiler import CompiledKernel as fgk_CompiledKernel
from fgk.verify import test_via_cubin
from fgk.utils.logger import get_logger

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
    for i, fn in enumerate(os.listdir(cubin_dir_path)):
        if not fn.endswith('.pkl'):
            continue

        with open(os.path.join(cubin_dir_path, fn), 'rb') as f:
            data = pickle.load(f)
        rankings[i] = data

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
