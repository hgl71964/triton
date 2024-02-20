import glob
import os
import subprocess
import sys
import tempfile
from argparse import ArgumentParser

import numpy as np

import triton
from triton.common import cuda_include_dir, libcuda_dirs

import fgk

test_utils_src = """
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "kernel.h"

static void write_buffer_to_csv(char *filename, int32_t *buffer, int size) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    for (int i = 0; i < size; i++) {
        fprintf(file, "%d", buffer[i]);
        if (i < size - 1) {
            fprintf(file, ",");
        }
    }
    fclose(file);
}

static void read_csv_to_buffer(char *filename, int16_t *buffer, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    int index = 0;
    while (fscanf(file, "%hd,", &buffer[index]) != EOF && index < size) {
        index++;
    }
    fclose(file);
}"""


def gen_test_bin(dir, M, N, K, exe="test", algo_id=0):
    test_src = f"""
int main(int argc, char **argv) {{
  int M = {M}, N = {N}, K = {K};

  // initialize CUDA handles
  CUdevice dev;
  CUcontext ctx;
  CUstream stream;
  CUdeviceptr A, B, C;
  CUresult err = 0;
  cuInit(0);
  cuDeviceGet(&dev, 0);
  cuCtxCreate(&ctx, 0, dev);
  cuMemAlloc(&A, M * K * 2);
  cuMemAlloc(&B, K * N * 2);
  cuMemAlloc(&C, M * N * 4);
  cuStreamCreate(&stream, 0);
  load_matmul_fp16();

  // initialize input data
  int16_t hA[M*K];
  int16_t hB[K*N];
  memset(hA, 0, M*K*2);
  memset(hB, 0, K*N*2);
  read_csv_to_buffer(argv[1], hA, M*K);
  read_csv_to_buffer(argv[2], hB, K*N);
  cuMemcpyHtoD(A, hA, M*K*2);
  cuMemcpyHtoD(B, hB, K*N*2);

  // launch kernel
  cuStreamSynchronize(stream);
  CUresult ret;
  int algo_id = {algo_id};
  if (algo_id == 0) {{
    ret = matmul_fp16_default(stream, C, A, B, M, N, K, N, 1, K, 1, N, 1);
  }} else {{
    ret = matmul_fp16(stream, C, A, B, M, N, K, N, 1, K, 1, N, 1, {algo_id});
  }}
  if (ret != 0) fprintf(stderr, "kernel launch failed\\n");
  assert(ret == 0);

  cuStreamSynchronize(stream);

  // read data
  int32_t hC[M*N];
  memset(hC, 0, M*N*4);
  cuMemcpyDtoH(hC, C, M*N*4);
  write_buffer_to_csv(argv[3], hC, M*N);

  // free cuda handles
  unload_matmul_fp16();
  cuMemFree(A);
  cuMemFree(B);
  cuMemFree(C);
  cuCtxDestroy(ctx);
}}
"""
    src = test_utils_src + test_src
    with open(os.path.join(dir, "test.c"), "w") as file:
        file.write(src)
    subprocess.run(
        ["gcc"] + [
            "test.c",
            "-I",
            cuda_include_dir(),
            "-L",
            libcuda_dirs()[0],
            "-l",
            "cuda",
            "-L",
            dir,
            "-l",
            "kernel",
            "-o",
            exe,
        ],
        check=True,
        cwd=dir,
    )


def gen_kernel_library(dir, libname):
    c_files = glob.glob(os.path.join(dir, "*.c"))
    subprocess.run(
        ["gcc"] + c_files + ["-I", cuda_include_dir(), "-c", "-fPIC"],
        check=True,
        cwd=dir,
    )
    o_files = glob.glob(os.path.join(dir, "*.o"))
    subprocess.run(
        ["gcc"] + o_files +
        ["-shared", "-o", libname, "-L",
         libcuda_dirs()[0]],
        check=True,
        cwd=dir,
    )


def compile_aot_kernel(kernel_path, sip_path, kernel_name, signatures):
    compiler_path = os.path.join(fgk.__path__, 'tools', "compile.py")
    subprocess.run(
        [
            sys.executable,
            compiler_path,
            "-n",
            kernel_name,
            "--signature",
            signature,
            "--out-name",
            out_name,
            "-o",
            out_path,
            "-w",
            str(num_warps),
            "-g",
            grid,
            kernel_path,
        ],
        check=True,
        cwd=dir,
    )


def main(args):
    np.random.seed(3)

    tmp_dir = '/home/gh512/workspace/triton/tmp'
    # with tempfile.TemporaryDirectory() as tmp_dir:
    dtype = "fp16"
    BM, BN, BK = 16, 16, 16

    # compile
    kernel_path = args.path
    compile_aot_kernel(tmp_dir, kernel_path, dtype, BM, BN, BK)

    # link
    link_aot_kernels(tmp_dir)

    # compile test case
    M, N, K = 16, 16, 16
    gen_kernel_library(tmp_dir, "libkernel.so")
    gen_test_bin(tmp_dir, M, N, K)

    # initialize test data
    a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

    # run test case
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = tmp_dir
    subprocess.run(["./test", a_path, b_path, c_path],
                   env=env,
                   check=True,
                   cwd=tmp_dir)

    # read data and compare against reference
    c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
    c_tri = c.reshape((M, N)).view(np.float32)
    c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
    np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.0)


if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument(
        "path",
        help=
        "Path to Python source containing desired kernel in its scope. File will be executed."
    )
    parser.add_argument("sip_path",
                        help="Path to directory containing SIP search file.")

    # TODO

    parser.add_argument("--kernel-name",
                        "-n",
                        type=str,
                        default="",
                        help="Name of the kernel to compile",
                        required=True)
    parser.add_argument("--num-warps",
                        "-w",
                        type=int,
                        default=1,
                        help="Number of warps to launch the kernel")
    parser.add_argument("--num-stages",
                        "-ns",
                        type=int,
                        default=3,
                        help="Number of stages (meta-parameter of the kernel)")
    parser.add_argument("--out-name",
                        "-on",
                        type=str,
                        default=None,
                        help="Out name for the compiled kernel")
    parser.add_argument("--out-path",
                        "-o",
                        type=Path,
                        default=None,
                        help="Out filename")
    parser.add_argument("--signature",
                        "-s",
                        type=str,
                        help="Signature of the kernel",
                        required=True)
    parser.add_argument("--grid",
                        "-g",
                        type=str,
                        help="Launch grid of the kernel",
                        required=True)
    args = parser.parse_args()
    main(args)
