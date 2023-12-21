import random
import numpy as np
import torch

import triton
import triton.language as tl

from fgk.jit import jit

from absl import app
from absl import flags

# YAPF: disable
FLAGS = flags.FLAGS

# kernel
flags.DEFINE_string("default_out_path", "data", "output dir")
flags.DEFINE_integer("dump", 0, "whether to dump")
flags.DEFINE_integer("hack", 0, "whether to hack")
flags.DEFINE_integer("flash", 0, "whether to use flash attention")
flags.DEFINE_integer("seed", 1337, "")
flags.DEFINE_integer("test_sample", 10, "")
flags.DEFINE_integer("n_choices", 1, "+-n choices")
# sa
flags.DEFINE_integer("max_iterations", 1000, "")
flags.DEFINE_float("temperature", 0.4, "")
flags.DEFINE_float("cooling_rate", 0.003, "")
flags.DEFINE_float("noise_factor", 0.1, "")
flags.DEFINE_string("policy", "single", "mutation policy; single or all")
# genetic
flags.DEFINE_integer("population_size", 100, "")
flags.DEFINE_integer("generations", 50, "")
flags.DEFINE_float("mutation_rate", 0.1, "")
flags.DEFINE_integer("tournament_size", 5, "")
#workload
flags.DEFINE_integer("wl", 0, "")
# YAPF: enable


def forward(
        kernel,
        x,
        w,
        bias,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        transposed=False,
        output_padding=(0, 0),
        groups=1,
):
    if groups != 1:
        raise RuntimeError("groups must be 1")
    if transposed:
        raise RuntimeError("transposed must be False")

    # Q: should we check x, w, bias dtypes?
    device = x.device
    # input shapes
    shape_x = x.shape
    shape_w = w.shape
    shape_bias = bias.shape if bias is not None else None

    # indicies for the layeout
    xn, xc, xh, xw = 0, 1, 2, 3
    yn, yc, yh, yw = 0, 1, 2, 3
    wn, wc, wh, ww = 0, 1, 2, 3

    # out_channel, in_channel, kernel_height, kernel_width
    kernel_size = [shape_w[wh], shape_w[ww]]
    input_size = [shape_x[xh], shape_x[xw]]
    assert (not shape_bias or shape_bias[0] == shape_w[wn]
            ), f"bias shape did not match{shape_bias} != {shape_w[wn]}"
    in_channel = shape_w[wc] * groups

    assert shape_x[xc] % groups == 0, "in_channels must be divisible by groups"
    assert shape_w[wn] % groups == 0, "out_channels must be divisible by groups"
    assert (shape_x[xc] == in_channel
            ), f"in_channel did not match {shape_x[xc]} != {in_channel}"

    assert (len(stride) == len(padding) == len(dilation) == len(output_padding)
            == len(kernel_size) == len(input_size))

    # output shape
    shape_y = [0] * 4
    shape_y[yn] = shape_x[xn]
    shape_y[yc] = shape_w[wn]
    shape_y[yh] = (input_size[0] + 2 * padding[0] - dilation[0] *
                   (kernel_size[0] - 1) - 1 +
                   stride[0]) // stride[0] + 2 * output_padding[0]
    shape_y[yw] = (input_size[1] + 2 * padding[1] - dilation[1] *
                   (kernel_size[1] - 1) - 1 +
                   stride[1]) // stride[1] + 2 * output_padding[1]

    BATCH = shape_x[xn]
    IN_C = shape_x[xc]
    IN_H = shape_x[xh]
    IN_W = shape_x[xw]
    KERNEL_N = shape_w[wn]
    KERNEL_H = shape_w[wh]
    KERNEL_W = shape_w[ww]
    OUT_H = shape_y[yh]
    OUT_W = shape_y[yw]

    # allocate output
    y = torch.empty(shape_y, device=device, dtype=x.dtype)

    # get strides for tensors
    stride_x = x.stride()
    stride_w = w.stride()
    stride_bias = bias.stride() if shape_bias else None
    stride_biasn = stride_bias[0] if stride_bias else None

    # output layout should be the same as x
    if stride_x[xc] < stride_x[xh] and stride_x[xc] < stride_x[xw]:
        y = y.to(memory_format=torch.channels_last)
    stride_y = y.stride()

    # allocate tmp
    # WINDOW_SIZE = KERNEL_H * KERNEL_W * IN_C
    # tmp_x = torch.empty((BATCH * OUT_H * OUT_W, WINDOW_SIZE), device=device, dtype=x.dtype)
    # tmp_w = torch.empty((WINDOW_SIZE, KERNEL_N), device=device, dtype=w.dtype)
    # accumulator types
    ACC_TYPE = (tl.float32 if x.dtype in [
        torch.float16, torch.bfloat16, torch.float32
    ] else tl.int32)
    # if stride_x[xc] == 1 and stride_x > 1 and stride_y > 1:
    CONV1X1_NHWC = False
    if stride_x[xc] == 1 and KERNEL_H == 1 and KERNEL_W == 1:
        CONV1X1_NHWC = True
    #  do we need delta x ptr for h, w, c dimension each or not
    DELTA_X_PTR_HWC = (False if ((padding[0] == 0 and padding[1] == 0) or
                                 (KERNEL_H == 1 and KERNEL_W == 1)) else True)
    if not CONV1X1_NHWC:
        if DELTA_X_PTR_HWC:
            delta_xh, delta_xw, delta_xc = _conv._delta_x_ptr_hwc(
                IN_C,
                KERNEL_H,
                KERNEL_W,
                dilation[0],
                dilation[1],
                stride_w[wc],
                stride_w[wh],
                stride_w[ww],
                stride_x[xc],
                stride_x[xh],
                stride_x[xw],
                device,
            )
        else:
            delta_x = _conv._delta_x_ptr(
                IN_C,
                KERNEL_H,
                KERNEL_W,
                dilation[0],
                dilation[1],
                stride_w[wc],
                stride_w[wh],
                stride_w[ww],
                stride_x[xc],
                stride_x[xh],
                stride_x[xw],
                device,
            )
    else:
        delta_x = None
        delta_xh, delta_xw, delta_xc = None, None, None

    # launch kernel, 2-dim, batch*h*w, kernel
    def grid(META):
        return (
            triton.cdiv(BATCH * OUT_H * OUT_W, META["BLOCK_M"]),
            triton.cdiv(KERNEL_N, META["BLOCK_N"]),
        )

    # YAPF: disable
    # conv1x1 or padding==0
    if CONV1X1_NHWC or not DELTA_X_PTR_HWC:
        _kernel_delta_x[grid](
            x, w, y, #
            # stride nchw for x,w,y tensor
            stride_x[xn], stride_x[xc], stride_x[xh], stride_x[xw], stride_w[wn], stride_w[wc], stride_w[wh], stride_w[ww], stride_y[yn], stride_y[yc], stride_y[yh], stride_y[yw], stride_biasn,  #
            # pointer inc for x
            delta_x,
            # Tensor dimensions
            BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, OUT_H, OUT_W,  #
            # conv parameters
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], output_padding[0], output_padding[1], groups,  #
            # Metaparameters
            ACC_TYPE=ACC_TYPE,
            CONV1X1_NHWC=CONV1X1_NHWC,
            # BLOCK_M=128,
            # BLOCK_N=32,
            # BLOCK_K=32,
            GROUP_H=1,
        )
    # need to know ptr update for each dimension to check if
    # the sliding window is out of bounds
    else:
        # kernel = _kernel_delta_x_hwc
        _kernel_delta_x_hwc[grid](
            x, w, y,  # 
            # stride nchw for x,w,y tensor
            stride_x[xn], stride_x[xc], stride_x[xh], stride_x[xw], stride_w[wn], stride_w[wc], stride_w[wh], stride_w[ww], stride_y[yn], stride_y[yc], stride_y[yh], stride_y[yw], stride_biasn,  # 
            # pointer inc for x
            delta_xh, delta_xw, delta_xc, #
            # Tensor dimensions
            BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, OUT_H, OUT_W,  # 
            # conv parameters
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], output_padding[0], output_padding[1], groups, # 
            # Metaparameters
            ACC_TYPE=ACC_TYPE,
            CONV1X1_NHWC=CONV1X1_NHWC,
            # BLOCK_M=128,
            # BLOCK_N=32,
            # BLOCK_K=32,
            GROUP_H=1,
        )
    # YAPF: enable

    if bias is not None:
        if len(bias.shape) == 1:
            bias = bias.reshape([1, bias.shape[0], 1, 1])
        y += bias
    return y


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True

    resnet50_layers = (
        # IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding
        (224, 224, 3, 7, 7, 64, (2, 2), (0, 0)),
        # conv2_x
        (56, 56, 64, 1, 1, 64, (1, 1), (0, 0)),
        (56, 56, 64, 3, 3, 64, (1, 1), (0, 0)),
        (56, 56, 64, 1, 1, 256, (1, 1), (0, 0)),
        # conv3_x
        (56, 56, 256, 1, 1, 128, (2, 2), (0, 0)),
        (28, 28, 128, 3, 3, 128, (1, 1), (0, 0)),
        (28, 28, 128, 1, 1, 512, (1, 1), (0, 0)),
        # conv4_x
        (28, 28, 512, 1, 1, 256, (2, 2), (0, 0)),
        (14, 14, 256, 3, 3, 256, (1, 1), (0, 0)),
        (14, 14, 256, 1, 1, 1024, (1, 1), (0, 0)),
        # conv5_x
        (14, 14, 1024, 1, 1, 512, (2, 2), (0, 0)),
        (7, 7, 512, 3, 3, 512, (1, 1), (0, 0)),
        (7, 7, 512, 1, 1, 2048, (1, 1), (0, 0)),
    )

    # workload
    BATCH = 32
    wl = FLAGS.wl
    assert wl < len(resnet50_layers), f"wl {wl} > {len(resnet50_layers)}"
    dtype = torch.float32
    IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding = resnet50_layers[
        wl]
    layout = "nhwc"
    dilation = (1, 1),
    groups = 1,

    # allocate inputs, nchw
    x = torch.randn((BATCH, IN_C, IN_H, IN_W), dtype=dtype, device="cuda")
    w = torch.randn((KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W),
                    dtype=dtype,
                    device="cuda")
    bias = torch.randn((KERNEL_N), dtype=dtype, device="cuda")
    if layout == "nhwc":
        x = x.to(memory_format=torch.channels_last)
        w = w.to(memory_format=torch.channels_last)
    OUT_H = (IN_H + 2 * padding[0] - dilation[0] *
             (KERNEL_H - 1) - 1 + stride[0]) // stride[0]
    OUT_W = (IN_W + 2 * padding[1] - dilation[1] *
             (KERNEL_W - 1) - 1 + stride[1]) // stride[1]

    flops = 2.0 * BATCH * OUT_H * OUT_W * IN_C * KERNEL_H * KERNEL_W * KERNEL_N

    # YAPF: disable
    # @triton.jit
    # @jit(
    #     total_flops=flops,
    #     seed=FLAGS.seed,
    #     # save_suffix=str(N_CTX)+"_"+str(dtype).replace('.', '_'),
    #     save_suffix='',
    #     save_dir='conv',
    # )
    @triton.jit
    def _kernel_delta_x_hwc(
        x, w, y, # stride of tensor
        stride_xn, stride_xc, stride_xh, stride_xw, stride_wn, stride_wc, stride_wh, stride_ww, stride_yn, stride_yc, stride_yh, stride_yw, stride_biasn, # pointer inc for x
        delta_xh_ptr, delta_xw_ptr, delta_xc_ptr, # Tensor dimensions
        BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, OUT_H, OUT_W, # parameters of conv
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, output_padding_h, output_padding_w, groups, # 
        # Metaparameters
        ACC_TYPE: tl.constexpr,
        CONV1X1_NHWC: tl.constexpr,
        # blocks in different dimension
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        # reduction tiling parameter for matmul
        BLOCK_K: tl.constexpr,
        # Super-blocking for better L2 peformance
        GROUP_H: tl.constexpr,
    ):
        """
        each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] block of y
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of y it should compute.
        pid_nhw = tl.program_id(0)
        pid_k = tl.program_id(1)

        # offset for output y
        off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
        off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
        off_y_n = off_y_nhw // (OUT_H * OUT_W)
        off_y_hw = off_y_nhw % (OUT_H * OUT_W)
        off_y_h = off_y_hw // OUT_W + output_padding_h
        off_y_w = off_y_hw % OUT_W + output_padding_w

        # offset for the initial ptr for x
        off_x_n = off_y_n
        off_x_h = off_y_h * stride_h - padding_h
        off_x_w = off_y_w * stride_w - padding_w
        off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
        off_x_crs = tl.arange(0, BLOCK_K)

        CRS = IN_C * KERNEL_H * KERNEL_W
        # load inc ptr of x, upade x_ptrs
        if not CONV1X1_NHWC:
            delta_xh_ptrs = delta_xh_ptr + off_x_crs
            delta_xw_ptrs = delta_xw_ptr + off_x_crs
            delta_xc_ptrs = delta_xc_ptr + off_x_crs
            delta_xh = tl.load(delta_xh_ptrs, mask=off_x_crs < CRS, other=0)
            delta_xw = tl.load(delta_xw_ptrs, mask=off_x_crs < CRS, other=0)
            delta_xc = tl.load(delta_xc_ptrs, mask=off_x_crs < CRS, other=0)
            off_x_crs_unpacked = (
                delta_xh * stride_xh + delta_xw * stride_xw + delta_xc * stride_xc
            )
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
        else:
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs[None, :]
            delta_xh = 0
            delta_xw = 0

        mask_x = (
            (off_x_n < BATCH)[:, None]
            & (off_x_crs < CRS)[None, :]
            & (off_x_h[:, None] + delta_xh[None, :] >= 0)
            & (off_x_h[:, None] + delta_xh[None, :] < IN_H)
            & (off_x_w[:, None] + delta_xw[None, :] >= 0)
            & (off_x_w[:, None] + delta_xw[None, :] < IN_W)
        )

        # offset for the inital ptr for w
        off_w_crs = tl.arange(0, BLOCK_K)
        off_w_k = off_y_k
        w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
        mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]

        # ------ load x ------
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        # ------ load w ------
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # -----------------------------------------------------------
        # allocate accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        for crs in range(0, CRS, BLOCK_K):

            # ------ matrix multiplication ------
            acc += tl.dot(matrix_x, matrix_w)
            # ------ update ptrs ------
            w_ptrs += BLOCK_K
            # load inc ptr of x, upade x_ptrs
            off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
            if not CONV1X1_NHWC:
                delta_xh_ptrs += BLOCK_K
                delta_xw_ptrs += BLOCK_K
                delta_xc_ptrs += BLOCK_K
                delta_xh = tl.load(delta_xh_ptrs, mask=off_x_crs < CRS, other=0)
                delta_xw = tl.load(delta_xw_ptrs, mask=off_x_crs < CRS, other=0)
                delta_xc = tl.load(delta_xc_ptrs, mask=off_x_crs < CRS, other=0)
                off_x_crs_unpacked = (
                    delta_xh * stride_xh + delta_xw * stride_xw + delta_xc * stride_xc
                )
                x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
            else:
                x_ptrs += BLOCK_K

            mask_x = (
                (off_x_n < BATCH)[:, None]
                & (off_x_crs < CRS)[None, :]
                & (off_x_h[:, None] + delta_xh[None, :] >= 0)
                & (off_x_h[:, None] + delta_xh[None, :] < IN_H)
                & (off_x_w[:, None] + delta_xw[None, :] >= 0)
                & (off_x_w[:, None] + delta_xw[None, :] < IN_W)
            )
            mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
            # ------ prefetch ------
            # ------ load x ------
            matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
            # ------ load w ------
            matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = acc.to(y.dtype.element_ty)

        # rematerialize -- this saves some registers
        # offset for output y
        off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
        off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
        off_y_n = off_y_nhw // (OUT_H * OUT_W)
        off_y_hw = off_y_nhw % (OUT_H * OUT_W)
        # consider output padding
        off_y_h = off_y_hw // OUT_W + output_padding_h
        off_y_w = off_y_hw % OUT_W + output_padding_w

        # y ptrs in the block of [BLOCK_M, BLOCK_N]
        y_ptrs = (
            y
            + off_y_n[:, None] * stride_yn
            + off_y_h[:, None] * stride_yh
            + off_y_w[:, None] * stride_yw
            + off_y_k[None, :] * stride_yc
        )

        # out-of-bounds check
        mask_y = (
            (off_y_n < BATCH)[:, None]
            & (off_y_h < OUT_H + output_padding_h)[:, None]
            & (off_y_w < OUT_W + output_padding_w)[:, None]
            & (off_y_k < KERNEL_N)[None, :]
        )

        tl.store(y_ptrs, acc, mask=mask_y)
    # YAPF: enable

    ref_func = torch.conv2d  # (x, w, bias, stride, padding, dilation, groups)

    tri_out = forward(
        _kernel_delta_x_hwc,
        x,
        w,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        transposed=False,
        output_padding=(0, 0),
        groups=1,
    )
    ref_out = ref_func(x, w, bias, stride, padding, dilation, groups=1)

    assert torch.allclose(tri_out, ref_out, atol=1e-2, rtol=0)


if __name__ == "__main__":
    app.run(main)