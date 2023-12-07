import os
import random
from time import sleep

import numpy as np
import tempfile

import torch

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# flags.DEFINE_string("default_out_path", "data", "output dir")
# flags.DEFINE_integer("dump", 0, "whether to dump")
# flags.DEFINE_integer("hack", 0, "whether to hack")
# flags.DEFINE_string("fn", None, "cubin name to load")
# flags.DEFINE_integer("flash", 0, "whether to use flash attention")
# flags.DEFINE_integer("epoch", 1000, "mutation iteration")
# flags.DEFINE_integer("seed", 1337, "")


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(tl.float16), v)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H,  #
              N_CTX: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))



def attn_forward(q, k, v, causal, sm_scale):
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)
    BLOCK_M = 128
    BLOCK_N = 64 if Lk <= 64 else 32
    num_stages = 4 if Lk <= 64 else 3
    num_warps = 4
    stage = 3 if causal else 1
    # Tuning for H100
    if torch.cuda.get_device_capability()[0] == 9:
        num_warps = 8
        num_stages = 7 if Lk >= 64 else 3

    # q, k, v: (Z, H, N_CTX, D_HEAD)
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    _attn_fwd[grid](
        q, k, v, sm_scale, M, o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        q.shape[0], q.shape[1],  #
        N_CTX=q.shape[2],  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_N=BLOCK_N,  #
        BLOCK_DMODEL=Lk,  #
        STAGE=stage,  #
        num_warps=num_warps,  #
        num_stages=num_stages  #
    )
    return o.half()

def get_cubin(q, k, v, causal, sm_scale):
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)
    BLOCK_M = 128
    BLOCK_N = 64 if Lk <= 64 else 32
    num_stages = 4 if Lk <= 64 else 3
    num_warps = 4
    stage = 3 if causal else 1
    # Tuning for H100
    if torch.cuda.get_device_capability()[0] == 9:
        num_warps = 8
        num_stages = 7 if Lk >= 64 else 3

    # q, k, v: (Z, H, N_CTX, D_HEAD)
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    asm = _attn_fwd.only_compile(
        q, k, v, sm_scale, M, o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        q.shape[0], q.shape[1],  #
        N_CTX=q.shape[2],  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_N=BLOCK_N,  #
        BLOCK_DMODEL=Lk,  #
        STAGE=stage,  #
        num_warps=num_warps,  #
        num_stages=num_stages  #
    )
    out = asm['cubin']
    return out

def set_cubin(q, k, v, causal, sm_scale, cubin):
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)
    BLOCK_M = 128
    BLOCK_N = 64 if Lk <= 64 else 32
    num_stages = 4 if Lk <= 64 else 3
    num_warps = 4
    stage = 3 if causal else 1
    # Tuning for H100
    if torch.cuda.get_device_capability()[0] == 9:
        num_warps = 8
        num_stages = 7 if Lk >= 64 else 3

    # q, k, v: (Z, H, N_CTX, D_HEAD)
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    _attn_fwd.hack_cubin(
        q, k, v, sm_scale, M, o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        q.shape[0], q.shape[1],  #
        N_CTX=q.shape[2],  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_N=BLOCK_N,  #
        BLOCK_DMODEL=Lk,  #
        STAGE=stage,  #
        num_warps=num_warps,  #
        num_stages=num_stages,  #
        cubin = cubin,
    )


def main(_):
    # ===== seed =====
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True

    Z, H, N_CTX, D_HEAD = 1, 32, 4096, 64
    dtype=torch.float16
    q = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())

    kernel = _attn_fwd
    fn = kernel.__name__
    sm_scale = 0.5
    causal = True

    # get cubin and asm
    out = get_cubin(q, k, v, causal, sm_scale)
    folder = f'{FLAGS.default_out_path}'
    file_path = f'{fn}_{0}.cubin'
    prefix = os.path.join(folder, f"tmp_{FLAGS.seed}")
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    file_path = os.path.join(prefix, file_path)
    with open(file_path, 'wb') as file:
        file.write(out)

    for i in range(FLAGS.epoch):

        # load
        file_path = f'{fn}_{i}.cuasm'
        full_path = os.path.join(prefix, file_path)
        with open(full_path, 'rb') as file:
            cuasm = file.read()

        # assemble
        full_path = os.path.join(prefix, file_path)
        with open(full_path, 'wb') as file:
            file.write(out)
        cubin = ...

        # execute
        set_cubin(q, k, v, causal, sm_scale, cubin)  



if __name__ == "__main__":
    app.run(main)
