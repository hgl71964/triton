import time
import builtins
from typing import Dict, Union
from copy import deepcopy

import torch

from triton.testing import do_bench
from triton.runtime.autotuner import Autotuner as TritonAutotuner
from triton.runtime.autotuner import OutOfResources

from fgk.jit import asm_JITFunction
from fgk.utils.logger import get_logger

logger = get_logger(__name__)


class Autotuner(TritonAutotuner):
    # 1. we use Triton's Autotuner to search for good kernel configurations
    # 2. use search to find good assembly schedule

    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        prune_configs_by,
        warmup,
        rep,

        # gh512
        ret_ptr,
        test_samples,
    ):
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            prune_configs_by,
            warmup,
            rep,
        )

        self.ret_ptr = ret_ptr
        self.test_samples = test_samples

    def _bench(self, *args, config, **meta):

        # gh512
        assert isinstance(self.fn, asm_JITFunction)

        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols.")
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.kwargs)
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(args)

            # gh512
            self.fn.triton_run(
                *args,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                num_ctas=config.num_ctas,
                enable_warp_specialization=config.enable_warp_specialization,
                # enable_persistent=False,
                **current,
            )
            self.post_hook(args)

        try:
            # this populates data to the ret_ptr
            return do_bench(kernel_call,
                            warmup=self.warmup,
                            rep=self.rep,
                            quantiles=(0.5, 0.2, 0.8))
        except OutOfResources:
            return [float("inf"), float("inf"), float("inf")]

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))

        # gh512 check data at ret_ptr
        if isinstance(self.ret_ptr, int):
            idx = self.ret_ptr
            self.ret_ptr = self.arg_names[idx]
        elif isinstance(self.ret_ptr, str):
            pass
        else:
            raise TypeError(
                f"Unsupported type {type(self.ret_ptr)} for {self.ret_ptr}")

        # if the group gemm example, addr of tensor is passed, so it is
        testable = True
        if not torch.all_close(self.nargs[self.ret_ptr], torch.Tensor([0.0])):
            testable = False
            logger.critical(f'cannot generate test example for {self.ret_ptr}')

        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = []
            for name in self.arg_names:
                if name in all_args:
                    _args.append(all_args[name])
            key = [_args[i] for i in self.key_idx]
            for arg in _args:
                if hasattr(arg, "dtype"):
                    key.append(str(arg.dtype))
            key = tuple(key)
            if key not in self.cache:
                # prune configs
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()
                timings = {
                    config: self._bench(*args, config=config, **kwargs)
                    for config in pruned_configs
                }
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                self.pre_hook(args, reset_only=True)
                self.configs_timings = timings
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        full_nargs = {**self.nargs, **kwargs, **self.best_config.kwargs}
        if config.pre_hook is not None:
            config.pre_hook(full_nargs)

        # gh512: run search and test
        if testable:
            test_samples = self.gen_test_samples()

        # TODO
        ret = self.fn.run(
            *args,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
            num_ctas=config.num_ctas,
            enable_warp_specialization=config.enable_warp_specialization,
            **kwargs,
            **config.kwargs,
        )
        self.nargs = None

        # TODO add test
        if testable:
            self.e2e_test()

        return ret

    def gen_test_samples(self) -> list[dict]:
        test_samples = []
        for t in range(self.test_samples):
            test = {}
            for name, inp in self.nargs.items():
                if isinstance(inp, torch.Tensor):
                    if name == self.ret_ptr:
                        test[name] = torch.empty_like(inp)
                    else:
                        test[name] = torch.randn_like(inp).uniform_(0, 1)
                else:
                    test[name] = deepcopy(inp)

            test_samples.append(test)

        return test_samples

    def e2e_test(self, test_samples):
        return


def autotune(
        configs,
        key,

        # the index to the ret_ptr
        ret_ptr: Union[int, str],
        test_samples: int = 10,
        prune_configs_by=None,
        reset_to_zero=None,
        restore_value=None,
        warmup=100,
        rep=100):
    def decorator(fn):
        return Autotuner(fn, fn.arg_names, configs, key, reset_to_zero,
                         restore_value, prune_configs_by, warmup, rep, ret_ptr,
                         test_samples)

    return decorator
