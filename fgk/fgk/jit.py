from triton.runtime.jit import T, JITFunction, KernelArg, get_current_device, set_current_device, get_cuda_stream
from triton.compiler.compiler import CompiledKernel, compile, get_arch_default_num_stages, get_arch_default_num_warps
from triton.common.backend import get_backend, get_cuda_version_key

from fgk.simulated_annealing import run_simulated_annealing, launch_simulated_annealing
from fgk.genetic_algorithm import run_genetic_algorithm

from fgk.compiler import compile as fgk_compile
from fgk.compiler import CompiledKernel as fgk_CompiledKernel


def jit(
    fn,
    *,
    version=None,
    do_not_specialize=None,
    debug=None,
    noinline=None,
):

    def decorator(fn: T) -> JITFunction[T]:
        assert callable(fn)
        return ASMJITFunction(
            fn,
            version=None,
            do_not_specialize=None,
            debug=None,
            noinline=None,
        )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator


class ASMJITFunction(JITFunction):

    def run(self, *args, **kwargs):

        # Get a compiler-flags arg like `num_warps` and remove it from kwargs.
        def get_special_arg(name: str, default=None):
            if name not in kwargs:
                return default
            ret = kwargs[name]
            del kwargs[name]
            return ret

        grid = get_special_arg("grid")
        num_warps = get_special_arg("num_warps")
        num_ctas = get_special_arg("num_ctas", 1)
        num_stages = get_special_arg("num_stages")
        enable_warp_specialization = get_special_arg(
            "enable_warp_specialization", False)
        enable_fp_fusion = get_special_arg("enable_fp_fusion", True)
        extern_libs = get_special_arg("extern_libs")
        stream = get_special_arg("stream")
        warmup = get_special_arg("warmup", False)
        device = get_special_arg("device")
        device_type = get_special_arg("device_type")

        # gh512
        ret_ptr = get_special_arg("ret_ptr")
        test_inputs = get_special_arg("test_inputs")
        test_outputs = get_special_arg("test_outputs")

        # Bind the remaining arguments to `fn`.
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        assert len(bound_args.arguments) == len(self.params)
        args = [
            KernelArg(arg_value, param)
            for (_, arg_value
                 ), param in zip(bound_args.arguments.items(), self.params)
        ]

        non_constexpr_arg_values = [
            arg.value for arg in args if not arg.param.is_constexpr
        ]

        sig_key = tuple(arg.signature_key() for arg in args
                        if not arg.param.is_constexpr)
        spec_key = tuple(arg.specialization_key() for arg in args
                         if not arg.param.do_not_specialize)
        constexpr_key = tuple(arg.value for arg in args
                              if arg.param.is_constexpr)

        assert num_ctas > 0
        assert grid is not None
        if callable(grid):
            # Arguments are passed as a dict to `grid`, by contract.
            # TODO(jlebar): In the new launch API, pass the compiler flags as a
            # second parameter to `grid`.
            grid = grid(dict(bound_args.arguments))
        grid_size = len(grid)
        grid_0 = grid[0]
        grid_1 = grid[1] if grid_size > 1 else 1
        grid_2 = grid[2] if grid_size > 2 else 1
        if device_type is None:
            device_types = [
                self._device_of(arg) for arg in non_constexpr_arg_values
            ]
            device_types = [
                _device_type for _device_type in device_types
                if _device_type != ""
            ]
            device_type = self._conclude_device_type(device_types, [
                self._pinned_memory_of(arg) for arg in non_constexpr_arg_values
            ])

        device_backend = None
        if device_type not in ["cuda"]:
            device_backend = get_backend(device_type)
            if device_backend is None:
                raise ValueError("Cannot find backend for " + device_type)

        if device is None:
            if device_type in ["cuda"]:
                device = get_current_device()
                set_current_device(device)
            else:
                device = device_backend.get_current_device()
                device_backend.set_current_device(device)
        if stream is None and not warmup:
            if device_type in ["cuda"]:
                stream = get_cuda_stream(device)
            else:
                stream = device_backend.get_stream()

        if num_warps is None:
            num_warps = get_arch_default_num_warps(device_type)
        if num_stages is None:
            num_stages = get_arch_default_num_stages(device_type)

        if device_type in ["cuda"]:
            version_key = get_cuda_version_key()
        else:
            version_key = device_backend.get_version_key()
        key = (
            version_key,
            sig_key,
            constexpr_key,
            spec_key,
            num_warps,
            num_ctas,
            num_stages,
            enable_warp_specialization,
            enable_fp_fusion,
            self.debug,
        )
        if extern_libs is not None:
            key = (key, tuple(extern_libs.items()))

        # Kernel is not cached; we have to compile.
        if key not in self.cache[device]:
            configs = (self._get_config(*[arg.value for arg in args]), )
            constants = {
                arg.param.num: arg.value
                for arg in args if arg.param.is_constexpr
                or arg.param.num in configs[0].equal_to_1 or arg.value is None
            }
            for i, arg in constants.items():
                if callable(arg):
                    raise TypeError(
                        f"Callable constexpr at index {i} is not supported")

            # Build kernel signature -- doesn't include constexpr arguments.
            signature = {
                arg.param.num: self._type_of(self._key_of(arg.value))
                for arg in args if not arg.param.is_constexpr
            }

            if self._call_hook(
                    key,
                    signature,
                    device,
                    constants,
                    num_warps,
                    num_ctas,
                    num_stages,
                    enable_warp_specialization,
                    enable_fp_fusion,
                    extern_libs,
                    configs,
            ):
                return None

            so_path, metadata, asm = fgk_compile(
                self,
                signature=signature,
                device=device,
                constants=constants,
                num_warps=num_warps,
                num_ctas=num_ctas,
                num_stages=num_stages,
                enable_warp_specialization=enable_warp_specialization,
                enable_fp_fusion=enable_fp_fusion,
                extern_libs=extern_libs,
                configs=configs,
                debug=self.debug,
                device_type=device_type,
            )
            bin = launch_simulated_annealing(
                so_path,
                metadata,
                asm,
                args,
                sig_key,
                non_constexpr_arg_values,
                ret_ptr,
                test_inputs,
                test_outputs,
                grid_0,
                grid_1,
                grid_2,
                stream,  #
                CompiledKernel.launch_enter_hook,
                CompiledKernel.launch_exit_hook,  # 
                # algo
                1,
                self.max_iterations,
                self.temperature,
                self.cooling_rate,
                self.policy,
                self.noise_factor,
                n_test_samples=self.n_test_samples,
                seed=self.seed,
                total_flops=self.total_flops,
                save_suffix=self.save_suffix,
                save_dir=self.save_dir,
                warmup=100,
                rep=100,
            )
            bin = fgk_CompiledKernel(so_path, metadata, asm)
            self.cache[device][key] = bin

        bin = self.cache[device][key]
        if not warmup:
            bin.c_wrapper(
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
                CompiledKernel.launch_enter_hook,
                CompiledKernel.launch_exit_hook,
                bin,
                *bin.assemble_tensormap_to_arg(non_constexpr_arg_values),
            )
        return bin

    # execute triton's default run
    def triton_run(self, *args, **kwargs):
        return super().run(*args, **kwargs)


def search(
    # workload config
    total_flops,
    seed=0,
    save_suffix="",
    save_dir=None,

    # sa
    max_iterations=1000,
    temperature=0.4,
    cooling_rate=0.003,
    noise_factor=0.0,
    policy="single",
    # ga
    population_size=100,
    generations=50,
    mutation_rate=0.1,
    tournament_size=5,
):

    def wrapper(fn):

        def decorator(fn: T):
            assert callable(fn)
            return asm_JITFunction(
                fn,
                # sa
                max_iterations=max_iterations,
                temperature=temperature,
                cooling_rate=cooling_rate,
                noise_factor=noise_factor,
                policy=policy,
                # ga
                population_size=population_size,
                generations=generations,
                mutation_rate=mutation_rate,
                tournament_size=tournament_size,

                # workload config
                seed=seed,
                total_flops=total_flops,
                save_suffix=save_suffix,
                save_dir=save_dir,

                # other
                version=None,
                do_not_specialize=None,
                debug=None,
                noinline=None,
            )

        return decorator(fn)

    return wrapper


class asm_JITFunction(JITFunction):

    def __init__(
            self,
            fn,

            # sa
            max_iterations,
            temperature,
            cooling_rate,
            noise_factor,
            policy,
            # ga
            population_size,
            generations,
            mutation_rate,
            tournament_size,

            # other config
            seed,
            total_flops,
            save_suffix,
            save_dir,
            version=None,
            do_not_specialize=None,
            debug=None,
            noinline=None):
        super(asm_JITFunction, self).__init__(
            fn,
            version=version,
            do_not_specialize=do_not_specialize,
            debug=debug,
            noinline=noinline,
        )
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.noise_factor = noise_factor
        self.policy = policy

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

        self.seed = seed
        self.total_flops = total_flops
        self.save_suffix = save_suffix
        self.save_dir = save_dir

    def run(self, *args, **kwargs):

        # Get a compiler-flags arg like `num_warps` and remove it from kwargs.
        def get_special_arg(name: str, default=None):
            if name not in kwargs:
                return default
            ret = kwargs[name]
            del kwargs[name]
            return ret

        grid = get_special_arg("grid")
        num_warps = get_special_arg("num_warps")
        num_ctas = get_special_arg("num_ctas", 1)
        num_stages = get_special_arg("num_stages")
        enable_warp_specialization = get_special_arg(
            "enable_warp_specialization", False)
        enable_fp_fusion = get_special_arg("enable_fp_fusion", True)
        extern_libs = get_special_arg("extern_libs")
        stream = get_special_arg("stream")
        warmup = get_special_arg("warmup", False)
        device = get_special_arg("device")
        device_type = get_special_arg("device_type")

        # Bind the remaining arguments to `fn`.
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        assert len(bound_args.arguments) == len(self.params)
        args = [
            KernelArg(arg_value, param)
            for (_, arg_value
                 ), param in zip(bound_args.arguments.items(), self.params)
        ]

        non_constexpr_arg_values = [
            arg.value for arg in args if not arg.param.is_constexpr
        ]

        sig_key = tuple(arg.signature_key() for arg in args
                        if not arg.param.is_constexpr)
        spec_key = tuple(arg.specialization_key() for arg in args
                         if not arg.param.do_not_specialize)
        constexpr_key = tuple(arg.value for arg in args
                              if arg.param.is_constexpr)

        assert num_ctas > 0
        assert grid is not None
        if callable(grid):
            # Arguments are passed as a dict to `grid`, by contract.
            # TODO(jlebar): In the new launch API, pass the compiler flags as a
            # second parameter to `grid`.
            grid = grid(dict(bound_args.arguments))
        grid_size = len(grid)
        grid_0 = grid[0]
        grid_1 = grid[1] if grid_size > 1 else 1
        grid_2 = grid[2] if grid_size > 2 else 1
        if device_type is None:
            device_types = [
                self._device_of(arg) for arg in non_constexpr_arg_values
            ]
            device_types = [
                _device_type for _device_type in device_types
                if _device_type != ""
            ]
            device_type = self._conclude_device_type(device_types, [
                self._pinned_memory_of(arg) for arg in non_constexpr_arg_values
            ])

        device_backend = None
        if device_type not in ["cuda"]:
            device_backend = get_backend(device_type)
            if device_backend is None:
                raise ValueError("Cannot find backend for " + device_type)

        if device is None:
            if device_type in ["cuda"]:
                device = get_current_device()
                set_current_device(device)
            else:
                device = device_backend.get_current_device()
                device_backend.set_current_device(device)
        if stream is None and not warmup:
            if device_type in ["cuda"]:
                stream = get_cuda_stream(device)
            else:
                stream = device_backend.get_stream()

        if num_warps is None:
            num_warps = get_arch_default_num_warps(device_type)
        if num_stages is None:
            num_stages = get_arch_default_num_stages(device_type)

        if device_type in ["cuda"]:
            version_key = get_cuda_version_key()
        else:
            version_key = device_backend.get_version_key()
        key = (
            version_key,
            sig_key,
            constexpr_key,
            spec_key,
            num_warps,
            num_ctas,
            num_stages,
            enable_warp_specialization,
            enable_fp_fusion,
            self.debug,
        )
        if extern_libs is not None:
            key = (key, tuple(extern_libs.items()))

        # Kernel is not cached; we have to compile.
        if key not in self.cache[device]:
            configs = (self._get_config(*[arg.value for arg in args]), )
            constants = {
                arg.param.num: arg.value
                for arg in args if arg.param.is_constexpr
                or arg.param.num in configs[0].equal_to_1 or arg.value is None
            }
            for i, arg in constants.items():
                if callable(arg):
                    raise TypeError(
                        f"Callable constexpr at index {i} is not supported")

            # Build kernel signature -- doesn't include constexpr arguments.
            signature = {
                arg.param.num: self._type_of(self._key_of(arg.value))
                for arg in args if not arg.param.is_constexpr
            }

            if self._call_hook(
                    key,
                    signature,
                    device,
                    constants,
                    num_warps,
                    num_ctas,
                    num_stages,
                    enable_warp_specialization,
                    enable_fp_fusion,
                    extern_libs,
                    configs,
            ):
                return None

            # triton's compilation pipeline
            bin = compile(
                self,
                signature=signature,
                device=device,
                constants=constants,
                num_warps=num_warps,
                num_ctas=num_ctas,
                num_stages=num_stages,
                enable_warp_specialization=enable_warp_specialization,
                enable_fp_fusion=enable_fp_fusion,
                extern_libs=extern_libs,
                configs=configs,
                debug=self.debug,
                device_type=device_type,
            )
            bin = run_simulated_annealing(
                bin,
                args,
                sig_key,
                non_constexpr_arg_values,
                grid_0,
                grid_1,
                grid_2,
                stream,
                CompiledKernel.launch_enter_hook,
                CompiledKernel.launch_exit_hook,
                # algo
                1,
                self.max_iterations,
                self.temperature,
                self.cooling_rate,
                self.policy,
                self.noise_factor,
                seed=self.seed,
                total_flops=self.total_flops,
                save_suffix=self.save_suffix,
                save_dir=self.save_dir,
                warmup=100,
                rep=100,
            )
            # bin = run_genetic_algorithm(
            #     bin,
            #     args,
            #     sig_key,
            #     non_constexpr_arg_values,
            #     grid_0,
            #     grid_1,
            #     grid_2,
            #     stream,
            #     CompiledKernel.launch_enter_hook,
            #     CompiledKernel.launch_exit_hook,
            #     # algo
            #     self.population_size,
            #     self.generations,
            #     self.tournament_size,
            #     self.mutation_rate,
            #     seed=self.seed,
            #     total_flops=self.total_flops,
            #     save_suffix=self.save_suffix,
            #     save_dir=self.save_dir,
            #     warmup=100,
            #     rep=100,
            # )
            self.cache[device][key] = bin

        bin = self.cache[device][key]
        if not warmup:
            bin.c_wrapper(
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
                CompiledKernel.launch_enter_hook,
                CompiledKernel.launch_exit_hook,
                bin,
                *bin.assemble_tensormap_to_arg(non_constexpr_arg_values),
            )
        return bin

    # execute triton's default run
    def triton_run(self, *args, **kwargs):
        return super().run(*args, **kwargs)
