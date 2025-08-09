import pytest

import torch

import triton
import triton.language as tl
from triton._internal_testing import is_hip_cdna4

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("M, N, K", [(1024, 512, 256), (128, 256, 256), (128, 128, 128)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K, rhs_scale, mxfp_type, normal_type",
                         [(32, 32, 128, rhs_scale, mxfp_type, normal_type)
                          for rhs_scale in [True, False]
                          for mxfp_type in ["e2m1"]
                          for normal_type in ["e4m3", "e5m2"]])
def test_amd_mxfp8_mxfp4_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, rhs_scale, mxfp_type, normal_type):
    device = 'cuda'

    @triton.jit
    def triton_kernel(  #
            a_ptr, b_ptr, output_ptr,  #
            a_scale_ptr, b_scale_ptr,  #
            M, N, K,  #
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,  #
            stride_scale: tl.constexpr,  #
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, #
            a_type: tl.constexpr, b_type: tl.constexpr):
        DIV_FACTOR_A: tl.constexpr = 2 if a_type == "e2m1" else 1
        DIV_FACTOR_B: tl.constexpr = 2 if b_type == "e2m1" else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B
        SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m

        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
        offs_ak = tl.arange(0, PACKED_BLOCK_K_A)
        offs_bk = tl.arange(0, PACKED_BLOCK_K_B)
        offs_scale_k = tl.arange(0, SCALE_BLOCK_K)

        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
        if a_scale_ptr is not None:
            a_scale_ptrs = a_scale_ptr + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
        else:
            a_scale = None

        b_ptrs = b_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        if b_scale_ptr is not None:
            b_scale_ptrs = b_scale_ptr + offs_bn[:, None] * stride_scale + offs_scale_k[None, :]
        else:
            b_scale = None

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

            if a_scale_ptr is not None:
                a_scale = tl.load(a_scale_ptrs)
            if b_scale_ptr is not None:
                b_scale = tl.load(b_scale_ptrs)

            accumulator = tl.dot_scaled(a, a_scale, a_type, b, b_scale, b_type, accumulator)

            a_ptrs += PACKED_BLOCK_K_A * stride_ak
            b_ptrs += PACKED_BLOCK_K_B * stride_bk

            if a_scale_ptr is not None:
                a_scale_ptrs += SCALE_BLOCK_K
            if b_scale_ptr is not None:
                b_scale_ptrs += SCALE_BLOCK_K

        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(output_ptrs, accumulator, mask=c_mask)


    @gluon.jit
    def gluon_kernel(  #
            a_ptr, b_ptr, output_ptr,  #
            a_scale_ptr, b_scale_ptr,  #
            M, N, K,  #
            stride_scale: tl.constexpr,  #
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,  #
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
            type_a: tl.constexpr, type_b: tl.constexpr):
        DIV_FACTOR_A: tl.constexpr = 2 if type_a == "e2m1" else 1
        DIV_FACTOR_B: tl.constexpr = 2 if type_b == "e2m1" else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B
        SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32

        a_unpacked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
        a_packed_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
        a_layout: ttgl.constexpr = a_packed_layout if type_a == "e2m1" else a_unpacked_layout

        a_scale_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
            reg_bases=[], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp_bases=[[0, 0], [16, 0]],
            block_bases=[], shape=[32, 4])

        b_unpacked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 16], [32, 2], [4, 1], [1, 0])
        b_packed_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0])
        b_layout: ttgl.constexpr = b_packed_layout if type_b == "e2m1" else b_unpacked_layout

        b_scale_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
            reg_bases=[], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp_bases=[[16, 0], [0, 0]],
            block_bases=[], shape=[32, 4])

        mfma_layout: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=4, warps_per_cta=[2, 2], tiles_per_warp=[1, 1],
                                                             instr_shape=[16, 16], transposed=True)

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m

        a_offsets_m = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, a_layout))
        a_offsets_k = ttgl.arange(0, PACKED_BLOCK_K_A, layout=ttgl.SliceLayout(0, a_layout))
        a_offsets = a_offsets_m[:, None] * stride_am + a_offsets_k[None, :] * stride_ak
        if a_scale_ptr is not None:
            a_scale_offsets_m = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, a_scale_layout))
            a_scale_offsets_k = ttgl.arange(0, SCALE_BLOCK_K, layout=ttgl.SliceLayout(0, a_scale_layout))
            a_scale_offsets = a_scale_offsets_m[:, None] * stride_scale + a_scale_offsets_k[None, :]
        else:
            a_scale = ttgl.full((BLOCK_M, SCALE_BLOCK_K), 0x7F, dtype=ttgl.int8, layout=a_scale_layout)

        b_offsets_k = ttgl.arange(0, PACKED_BLOCK_K_B, layout=ttgl.SliceLayout(1, b_layout))
        b_offsets_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, b_layout))
        b_offsets = b_offsets_k[:, None] * stride_bk + b_offsets_n[None, :] * stride_bn
        if b_scale_ptr is not None:
            b_scale_offsets_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, b_scale_layout))
            b_scale_offsets_k = ttgl.arange(0, SCALE_BLOCK_K, layout=ttgl.SliceLayout(0, b_scale_layout))
            b_scale_offsets = b_scale_offsets_n[:, None] * stride_scale + b_scale_offsets_k[None, :]
        else:
            b_scale = ttgl.full((BLOCK_N, SCALE_BLOCK_K), 0x7F, dtype=ttgl.int8, layout=b_scale_layout)

        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty, layout=mfma_layout)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = ttgl.amd.cdna4.buffer_load(a_ptr, a_offsets)
            a = ttgl.convert_layout(a, ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=16))

            b = ttgl.amd.cdna4.buffer_load(b_ptr, b_offsets)
            b = ttgl.convert_layout(b, ttgl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=16))

            if a_scale_ptr is not None:
                a_scale = ttgl.amd.cdna4.buffer_load(a_scale_ptr, a_scale_offsets)
            if b_scale_ptr is not None:
                b_scale = ttgl.amd.cdna4.buffer_load(b_scale_ptr, b_scale_offsets)

            accumulator = ttgl.amd.cdna4.mfma_scaled(a, a_scale, type_a, b, b_scale, type_b, accumulator)

            a_offsets += PACKED_BLOCK_K_A * stride_ak
            b_offsets += PACKED_BLOCK_K_B * stride_bk

            if a_scale_ptr is not None:
                a_scale_offsets += SCALE_BLOCK_K
            if b_scale_ptr is not None:
                b_scale_offsets += SCALE_BLOCK_K

        c_offsets_m = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, mfma_layout))
        c_offsets_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, mfma_layout))
        output_offsets = c_offsets_m[:, None] * stride_cm + c_offsets_n[None, :] * stride_cn
        output_mask = (c_offsets_m[:, None] < M) & (c_offsets_n[None, :] < N)
        ttgl.amd.cdna4.buffer_store(accumulator, output_ptr, output_offsets, mask=output_mask)


    torch.manual_seed(0)

    type_a = normal_type if rhs_scale else mxfp_type
    type_b = mxfp_type if rhs_scale else normal_type

    DIV_FACTOR_A = 2 if type_a == "e2m1" else 1
    DIV_FACTOR_B = 2 if type_b == "e2m1" else 1
    x = torch.randint(20, 40, (M, K // DIV_FACTOR_A), dtype=torch.uint8, device=device)
    y = torch.randint(20, 40, (K // DIV_FACTOR_B, N), dtype=torch.uint8, device=device)

    min_scale, max_scale = (0, 142)
    x_scale = torch.randint(min_scale, max_scale + 1, (M, K // 32), dtype=torch.uint8, device=device)
    y_scale = torch.randint(min_scale, max_scale + 1, (N, K // 32), dtype=torch.uint8, device=device)
    if rhs_scale:
        x_scale = None
        stride_scale = y_scale.stride(0)
    else:
        y_scale = None
        stride_scale = x_scale.stride(0)

    def make_finite(x, dtype):
        if dtype not in ("e5m2", "e4m3"):
            return x
        mask = 0x7C if dtype == "e5m2" else 0x7F
        finite = torch.arange(x.numel(), device=device, dtype=torch.uint8).reshape_as(x) % mask
        x_finite = torch.where(x & mask == mask, finite | (0x80 & x), x)
        x.copy_(x_finite)
        return x

    x = make_finite(x, type_a)
    y = make_finite(y, type_b)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)

    z = x.new_empty((M, N), dtype=torch.float32, device=device)
    gluon_kernel[grid](
        x, y, z,
        x_scale, y_scale,
        M, N, K,
        stride_scale,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        z.stride(0), z.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        type_a, type_b)

    z_ref = x.new_empty((M, N), dtype=torch.float32, device=device)
    triton_kernel[grid](
        x, y, z_ref,
        x_scale, y_scale,
        M, N, K,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        z_ref.stride(0), z_ref.stride(1),
        stride_scale,
        BLOCK_M, BLOCK_N, BLOCK_K,
        type_a, type_b)

    assert torch.allclose(z, z_ref, atol=1e-5, rtol=1e-5)
