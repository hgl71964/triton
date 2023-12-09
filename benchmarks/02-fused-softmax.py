"""
Fused Softmax
=============

In this tutorial, you will write a fused softmax operation that is significantly faster
than PyTorch's native op for a particular class of matrices: those whose rows can fit in
the GPU's SRAM.

In doing so, you will learn about:

* The benefits of kernel fusion for bandwidth-bound operations.

* Reduction operators in Triton.

"""

# %%
# Motivations
# -----------
#
# Custom GPU kernels for elementwise additions are educationally valuable but won't get you very far in practice.
# Let us consider instead the case of a simple (numerically stabilized) softmax operation:
import os

import torch

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction

from CuAsm.CubinFile import CubinFile

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("default_out_path", "data", "output dir")
flags.DEFINE_integer("dump", 0, "whether to dump")
flags.DEFINE_integer("hack", 0, "whether to hack")
flags.DEFINE_string("fn", None, "cubin name to load")


def softmax_kernel(
        output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
        BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def softmax(x, kernel):
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # f the input matrix
    kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


def get_cubin(x, kernel: JITFunction):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y = torch.empty_like(x)
    asm = kernel.only_compile(
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    out = asm['cubin']
    return out


def set_cubin(x, kernel: JITFunction, cubin):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y = torch.empty_like(x)
    kernel.hack_cubin(
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
        cubin=cubin,
    )


def main(_):
    kernel = triton.jit(softmax_kernel)
    fn = kernel.__name__
    dummy = torch.randn(1823, 781, device='cuda')

    # dump
    if bool(FLAGS.dump):
        out = get_cubin(dummy, kernel)
        folder = f'{FLAGS.default_out_path}'
        file_path = f'{fn}.cubin'
        file_path = os.path.join(folder, file_path)
        with open(file_path, 'wb') as file:
            file.write(out)

        # disassemble
        binname = file_path
        cf = CubinFile(binname)
        asmname = binname.replace('.cubin', '.cuasm')
        cf.saveAsCuAsm(asmname)
        return

    if bool(FLAGS.hack):
        # set
        assert FLAGS.fn is not None, 'cubin name to load'
        file_path = os.path.join(FLAGS.default_out_path, FLAGS.fn)
        with open(file_path, 'rb') as file:
            cubin = file.read()
        _ = get_cubin(
            dummy, kernel)  # this populate the cache with the same key
        set_cubin(dummy, kernel, cubin)

    ## TEST
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cuda')
    y_triton = softmax(x, kernel)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

    # Benchmark
    # @triton.testing.perf_report(
    #     triton.testing.Benchmark(
    #         x_names=['N'],  # argument names to use as an x-axis for the plot
    #         x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
    #         line_arg='provider',  # argument name whose value corresponds to a different line in the plot
    #         line_vals=[
    #             'triton',
    #             'torch-native',
    #             'torch-jit',
    #         ],  # possible values for `line_arg``
    #         line_names=[
    #             "Triton",
    #             "Torch (native)",
    #             "Torch (jit)",
    #         ],  # label name for the lines
    #         styles=[('blue', '-'), ('green', '-'), ('green', '--')],  # line styles
    #         ylabel="GB/s",  # label name for the y-axis
    #         plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
    #         args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    #     ))
    # def benchmark(M, N, provider):
    #     x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    #     quantiles = [0.5, 0.2, 0.8]
    #     if provider == 'torch-native':
    #         ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    #     if provider == 'triton':
    #         ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    #     if provider == 'torch-jit':
    #         ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    #     gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    #     return gbps(ms), gbps(max_ms), gbps(min_ms)
    #
    #
    # benchmark.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    app.run(main)
