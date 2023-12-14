import os
import torch
import pickle

from fgk.utils.gpu_utils import get_gpu_name


def save_data(
    bin,
    final_perf,
    init_perf,
    search_time,
    args,
    sig_key,
    non_constexpr_arg_values,
    seed,
    save_suffix,
    save_dir,
    algo,
):
    gpu_name = get_gpu_name()
    gpu_name = gpu_name.replace(' ', '_')

    data = {}
    data['cubin'] = bin.asm['cubin']  # binary
    data['final_perf'] = final_perf
    data['init_perf'] = init_perf
    data['search_time'] = search_time
    data['sig_key'] = sig_key
    data['args'] = {}  # map tensor name to shape
    for i in range(len(sig_key)):
        if isinstance(sig_key[i], torch.dtype):
            data['args'][args[i].name] = list(
                non_constexpr_arg_values[i].shape)

    dir_path = f'data/{gpu_name}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if save_dir is not None:
        dir_path = f'data/{gpu_name}/save_dir'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    kernel_name = bin.metadata['name'][:20]
    file_name = f'{kernel_name}_{algo}_{seed}'
    if not isinstance(save_suffix, str):
        save_suffix = str(save_suffix)
    file_name += '_'
    file_name += save_suffix
    file_name += '.pkl'

    with open(f'{dir_path}/{file_name}', 'wb') as f:
        pickle.dump(data, f)
